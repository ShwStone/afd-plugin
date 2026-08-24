# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Open-loop replay client for the Mooncake-derived workload (plan 6.5/9.3).

Replays a requests file against a frozen scaled arrival plan: every request
is submitted at its absolute planned offset, requests sharing a timestamp
are submitted concurrently, and a slow request never delays later sends
(open loop). The client records planned vs actual send time per request;
a run is INVALID when the send-deviation p99 exceeds 50 ms or the maximum
exceeds 250 ms (client failed to realize the planned load).

Payloads are JSON-serialized once before the clock starts so large prompts
(63K tokens ~ 400 KB JSON) do not serialize on the event loop mid-burst.

Optional ``--dp-rank`` pins all requests to one Attention data replica via
the ``X-data-parallel-rank`` header (mechanism batches / acceptance only;
capacity runs must use the system's normal distribution).

Runs inside the experiment container (needs aiohttp, which ships with vLLM).
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import threading
import time
from pathlib import Path

import aiohttp

REQUEST_TIMEOUT_S = 1800.0
# Send-deviation validity gate. The plan's p99 <= 50 ms was not realizable:
# measuring from inside the same pod means the client competes with the
# server's 32-卡 prefill for CPU, and whole Mooncake bursts land uniformly
# late by 13-57 ms (2026-08-20 baseline screening) — a fixed scheduler
# latency, not gross client failure. Revised (user decision, 2026-08-20):
# p99 <= 100 ms. This is still far below the service-side noise (solo
# prefill of a >30K-token request alone takes 20-50 s), so the gate keeps
# its purpose (catch a client that stopped realizing the planned load)
# without mislabeling healthy runs. The max gate stays 250 ms (single
# outlier, e.g. a scheduler stall, still flags).
DEVIATION_P99_GATE_MS = 100.0
DEVIATION_MAX_GATE_MS = 250.0
# Default SLO for the capacity gate. The plan's 10 s was authored from
# 10-layer reduced-model measurements; on the 61-layer model the longest
# (63,778-token) request alone needs ~28-32 s, and Mooncake same-timestamp
# bursts (preserved by the frozen arrival plans) set a rate-independent
# TTFT p99 floor of ~40 s (A2) / ~50 s (baseline). Revised to 50 s with the
# user on 2026-08-20 so both systems sit in a load-sensitive regime.
TTFT_P99_SLO_S = 50.0


def _sha256_file(file_path: Path) -> str:
    digest = hashlib.sha256()
    with file_path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    rank = (len(sorted_values) - 1) * pct / 100.0
    low = int(rank)
    high = min(low + 1, len(sorted_values) - 1)
    frac = rank - low
    return sorted_values[low] * (1 - frac) + sorted_values[high] * frac


def load_requests(
    requests_path: Path, request_ids: set[str] | None
) -> list[dict[str, object]]:
    requests: list[dict[str, object]] = []
    with requests_path.open(encoding="utf-8") as dataset_file:
        for line in dataset_file:
            record = json.loads(line)
            if request_ids is not None and record["request_id"] not in request_ids:
                continue
            if record["input_length"] != len(record["prompt_token_ids"]):
                raise ValueError(
                    f"length mismatch in {record['request_id']}"
                )
            requests.append(record)
    if not requests:
        raise ValueError(f"no requests selected from {requests_path}")
    return requests


def load_plan_offsets(plan_path: Path) -> dict[str, int]:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    return {
        arrival["request_id"]: int(arrival["scaled_arrival_offset_ns"])
        for arrival in plan["arrivals"]
    }


async def send_one(
    session: aiohttp.ClientSession,
    api_url: str,
    prepared: dict[str, object],
    start_monotonic: float,
    semaphore: asyncio.Semaphore | None,
) -> dict[str, object]:
    due = start_monotonic + prepared["offset_ns"] / 1e9
    delay = due - time.monotonic()
    if delay > 0:
        await asyncio.sleep(delay)
    if semaphore is not None:
        await semaphore.acquire()
    actual_send = time.monotonic()
    headers = {"Content-Type": "application/json"}
    if prepared["dp_rank"] is not None:
        headers["X-data-parallel-rank"] = str(prepared["dp_rank"])
    ttft_s = 0.0
    error = ""
    prompt_tokens_reported = None
    try:
        async with session.post(
            api_url, data=prepared["body"], headers=headers
        ) as response:
            if response.status != 200:
                text = await response.text()
                raise RuntimeError(f"HTTP {response.status}: {text[:500]}")
            buffer = b""
            async for chunk in response.content.iter_any():
                if ttft_s == 0.0 and chunk.strip():
                    ttft_s = time.monotonic() - actual_send
                buffer += chunk
            # usage arrives in the final SSE chunk (include_usage requested).
            for line in buffer.decode("utf-8", "replace").splitlines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                try:
                    event = json.loads(line[6:])
                except json.JSONDecodeError:
                    continue
                usage = event.get("usage") if isinstance(event, dict) else None
                if usage:
                    prompt_tokens_reported = usage.get("prompt_tokens")
    except Exception as exc:  # noqa: BLE001 - record per-request failure
        error = f"{type(exc).__name__}: {exc}"
    finally:
        if semaphore is not None:
            semaphore.release()
    end = time.monotonic()
    return {
        "request_id": prepared["request_id"],
        "input_length": prepared["input_length"],
        "planned_offset_ns": prepared["offset_ns"],
        "send_deviation_ms": (actual_send - due) * 1000.0,
        "actual_send_s": actual_send - start_monotonic,
        "ttft_s": ttft_s,
        "e2el_s": end - actual_send,
        "completed_s": end - start_monotonic,
        "prompt_tokens_reported": prompt_tokens_reported,
        "prompt_tokens_match": prompt_tokens_reported == prepared["input_length"],
        "success": not error and ttft_s > 0,
        "error": error,
    }


async def _run_slice(
    api_url: str,
    slice_prepared: list[dict[str, object]],
    start_monotonic: float,
    args: argparse.Namespace,
    results: dict[str, dict[str, object]],
) -> None:
    """One sender thread's event loop: replay its share of the plan."""
    timeout = aiohttp.ClientTimeout(total=args.request_timeout)
    # force_close: never reuse a pooled connection — idle gaps between
    # Mooncake bursts exceed uvicorn's keepalive timeout and stale-connection
    # reuse races the server close (instant ServerDisconnectedError), which
    # the zero-failure formal gate cannot tolerate.
    connector = aiohttp.TCPConnector(
        limit=max(4, args.max_connections // args.send_threads),
        force_close=True,
    )
    async with aiohttp.ClientSession(
        timeout=timeout, connector=connector
    ) as session:
        tasks = await asyncio.gather(
            *(
                send_one(session, api_url, p, start_monotonic, None)
                for p in slice_prepared
            )
        )
    for result in tasks:
        results[result["request_id"]] = result


def _thread_main(
    api_url: str,
    slice_prepared: list[dict[str, object]],
    start_monotonic: float,
    args: argparse.Namespace,
    results: dict[str, dict[str, object]],
) -> None:
    asyncio.run(_run_slice(api_url, slice_prepared, start_monotonic, args, results))


async def async_main(args: argparse.Namespace) -> dict[str, object]:
    request_ids = None
    if args.request_ids:
        request_ids = set(args.request_ids.split(","))
    requests = load_requests(args.requests, request_ids)

    if args.plan:
        offsets = load_plan_offsets(args.plan)
        missing = [r["request_id"] for r in requests if r["request_id"] not in offsets]
        if missing:
            raise ValueError(f"plan misses {len(missing)} requests, e.g. {missing[:3]}")
        plan_sha256 = _sha256_file(args.plan)
        plan_doc = json.loads(args.plan.read_text(encoding="utf-8"))
    else:
        offsets = {r["request_id"]: 0 for r in requests}
        plan_sha256 = None
        plan_doc = None

    api_url = args.base_url.rstrip("/") + args.endpoint
    prepared: list[dict[str, object]] = []
    for record in requests:
        body = json.dumps(
            {
                "model": args.model,
                "prompt": record["prompt_token_ids"],
                "max_tokens": 1,
                "temperature": 0,
                "stream": True,
                "stream_options": {"include_usage": True},
                "ignore_eos": True,
                "add_special_tokens": False,
            }
        ).encode("utf-8")
        prepared.append(
            {
                "request_id": record["request_id"],
                "input_length": record["input_length"],
                "offset_ns": offsets[record["request_id"]],
                "dp_rank": record.get("dp_rank", args.dp_rank),
                "body": body,
            }
        )
    prepared.sort(key=lambda p: p["offset_ns"])

    # Sender fan-out: a single asyncio loop on the server pod loses 13-57 ms
    # per burst to loop-scheduling + body writes competing with the server's
    # own CPU load (measured: whole bursts uniformly late, 2026-08-20). One
    # loop per thread, requests partitioned round-robin after sorting by
    # offset, so same-timestamp bursts split across threads.
    start_monotonic = time.monotonic()
    if args.send_threads <= 1:
        timeout = aiohttp.ClientTimeout(total=args.request_timeout)
        connector = aiohttp.TCPConnector(
            limit=args.max_connections, force_close=True
        )
        semaphore = (
            asyncio.Semaphore(args.max_concurrency)
            if args.max_concurrency
            else None
        )
        async with aiohttp.ClientSession(
            timeout=timeout, connector=connector
        ) as session:
            results_list = await asyncio.gather(
                *(
                    send_one(session, api_url, p, start_monotonic, semaphore)
                    for p in prepared
                )
            )
    else:
        if args.max_concurrency:
            raise SystemExit("--max-concurrency requires --send-threads 1")
        results_by_id: dict[str, dict[str, object]] = {}
        slices: list[list[dict[str, object]]] = [
            prepared[i:: args.send_threads] for i in range(args.send_threads)
        ]
        threads = [
            threading.Thread(
                target=_thread_main,
                args=(api_url, slice_prepared, start_monotonic, args, results_by_id),
                daemon=True,
            )
            for slice_prepared in slices
            if slice_prepared
        ]
        for thread in threads:
            thread.start()
        while any(thread.is_alive() for thread in threads):
            await asyncio.sleep(1.0)
        results_list = [
            results_by_id[p["request_id"]] for p in prepared
        ]
    wall_s = time.monotonic() - start_monotonic
    results = results_list

    deviations = sorted(r["send_deviation_ms"] for r in results)
    ttfts = sorted(r["ttft_s"] for r in results if r["success"])
    total_tokens = sum(r["input_length"] for r in results)
    success_tokens = sum(r["input_length"] for r in results if r["success"])
    actual_send_span = max(r["actual_send_s"] for r in results) - min(
        r["actual_send_s"] for r in results
    )
    last_send = max(r["actual_send_s"] for r in results)
    last_completion = max(r["completed_s"] for r in results)

    dev_p99 = _percentile(deviations, 99)
    dev_max = deviations[-1] if deviations else 0.0
    ttft_p99 = _percentile(ttfts, 99)
    failed = [r for r in results if not r["success"]]
    deviation_ok = dev_p99 <= DEVIATION_P99_GATE_MS and dev_max <= DEVIATION_MAX_GATE_MS
    summary = {
        "request_count": len(results),
        "successful": len(results) - len(failed),
        "failed": len(failed),
        "wall_s": wall_s,
        "send_deviation_ms": {
            "p50": _percentile(deviations, 50),
            "p95": _percentile(deviations, 95),
            "p99": dev_p99,
            "max": dev_max,
        },
        "send_deviation_ok": deviation_ok,
        "ttft_s": {
            "p50": _percentile(ttfts, 50),
            "p95": _percentile(ttfts, 95),
            "p99": ttft_p99,
            "max": ttfts[-1] if ttfts else 0.0,
        },
        "ttft_slo_s": args.ttft_slo_s,
        "ttft_p99_slo_ok": ttft_p99 <= args.ttft_slo_s,
        "total_input_tokens": total_tokens,
        "prompt_tokens_mismatched": sum(
            1 for r in results if r["success"] and not r["prompt_tokens_match"]
        ),
        "actual_send_token_rate": (
            total_tokens / actual_send_span if actual_send_span > 0 else 0.0
        ),
        "completion_token_throughput": (
            success_tokens / (last_completion - min(r["actual_send_s"] for r in results))
            if last_completion > 0
            else 0.0
        ),
        "queue_drain_s": last_completion - last_send,
        "target_input_tokens_per_second": (
            plan_doc["target_input_tokens_per_second"] if plan_doc else None
        ),
    }
    return {
        "schema_version": 1,
        "base_url": args.base_url,
        "endpoint": args.endpoint,
        "model": args.model,
        "requests_path": str(args.requests),
        "requests_sha256": _sha256_file(args.requests),
        "plan_path": str(args.plan) if args.plan else None,
        "plan_sha256": plan_sha256,
        "dp_rank": args.dp_rank,
        "summary": summary,
        "requests": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--endpoint", default="/v1/completions")
    parser.add_argument("--model", default="deepseek_v3_2")
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--plan", type=Path, default=None)
    parser.add_argument(
        "--request-ids",
        default=None,
        help="comma-separated subset filter (acceptance checks only)",
    )
    parser.add_argument("--dp-rank", type=int, default=None)
    parser.add_argument("--request-timeout", type=float, default=REQUEST_TIMEOUT_S)
    parser.add_argument("--ttft-slo-s", type=float, default=TTFT_P99_SLO_S)
    parser.add_argument(
        "--send-threads",
        type=int,
        default=4,
        help="sender thread pool size (one event loop each); 1 = single loop",
    )
    parser.add_argument("--max-connections", type=int, default=1024)
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=0,
        help="acceptance-only throttle; 0 = unlimited (required for formal runs)",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.max_concurrency and args.plan:
        raise SystemExit("--max-concurrency breaks open-loop semantics; burst only")

    result = asyncio.run(async_main(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result) + "\n", encoding="utf-8")
    summary = result["summary"]
    print(
        f"REPLAY_OK ok={summary['successful']}/{summary['request_count']} "
        f"ttft_p99={summary['ttft_s']['p99']:.2f}s "
        f"dev_p99={summary['send_deviation_ms']['p99']:.1f}ms "
        f"dev_max={summary['send_deviation_ms']['max']:.1f}ms "
        f"wall={summary['wall_s']:.1f}s"
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
