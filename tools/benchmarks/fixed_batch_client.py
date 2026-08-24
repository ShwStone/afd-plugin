# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Fixed-batch client for the mechanism experiments (plan section 6.1).

Sends one frozen fixed batch as a concurrent burst, waits for all requests,
and repeats the burst. All requests are pinned to a single Attention data
replica with the ``X-data-parallel-rank`` header (vLLM 0.26.0 native support,
see ``vllm/entrypoints/generate/base/serving.py``) so the 4-replica baseline
and the 2-replica AFD deployment form the same local batch instead of
auto-sharding across replicas.

Runs inside the experiment container (needs aiohttp, which ships with vLLM).
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import time
from pathlib import Path

import aiohttp

REQUEST_TIMEOUT_S = 900.0
INTER_REPEAT_GAP_S = 5.0


def _sha256_file(file_path: Path) -> str:
    digest = hashlib.sha256()
    with file_path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_batch(dataset_path: Path) -> list[dict[str, object]]:
    requests: list[dict[str, object]] = []
    with dataset_path.open(encoding="utf-8") as dataset_file:
        for line in dataset_file:
            record = json.loads(line)
            requests.append(
                {
                    "request_id": str(record.get("request_id", len(requests))),
                    "prompt": record["prompt"],
                }
            )
    if not requests:
        raise ValueError(f"Empty fixed batch: {dataset_path}")
    return requests


async def send_one(
    session: aiohttp.ClientSession,
    api_url: str,
    model: str,
    request: dict[str, object],
    dp_rank: int | None,
) -> dict[str, object]:
    headers = {"Content-Type": "application/json"}
    if dp_rank is not None:
        headers["X-data-parallel-rank"] = str(dp_rank)
    payload = {
        "model": model,
        "prompt": request["prompt"],
        "max_tokens": 1,
        "temperature": 0,
        "stream": True,
        "ignore_eos": True,
        "add_special_tokens": False,
    }
    start = time.perf_counter()
    ttft_s = 0.0
    error = ""
    try:
        async with session.post(api_url, json=payload, headers=headers) as response:
            if response.status != 200:
                body = await response.text()
                raise RuntimeError(f"HTTP {response.status}: {body[:500]}")
            async for chunk in response.content.iter_any():
                if chunk.strip():
                    ttft_s = time.perf_counter() - start
                    break
            # Drain the rest of the stream so the request completes.
            async for _ in response.content.iter_any():
                pass
    except Exception as exc:  # noqa: BLE001 - record per-request failure
        error = f"{type(exc).__name__}: {exc}"
    end = time.perf_counter()
    return {
        "request_id": request["request_id"],
        "prompt_len": len(request["prompt"]),
        "ttft_s": ttft_s,
        "e2el_s": end - start,
        "success": not error and ttft_s > 0,
        "error": error,
    }


async def run_burst(
    session: aiohttp.ClientSession,
    api_url: str,
    model: str,
    batch: list[dict[str, object]],
    dp_rank: int | None,
) -> dict[str, object]:
    burst_start = time.perf_counter()
    results = await asyncio.gather(
        *(
            send_one(session, api_url, model, request, dp_rank)
            for request in batch
        )
    )
    wall_s = time.perf_counter() - burst_start
    return {
        "wall_s": wall_s,
        "successful": sum(1 for result in results if result["success"]),
        "failed": sum(1 for result in results if not result["success"]),
        "requests": results,
    }


async def async_main(args: argparse.Namespace) -> dict[str, object]:
    batch = load_batch(args.dataset)
    api_url = args.base_url.rstrip("/") + args.endpoint
    timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_S)
    # force_close: never reuse a connection. The 5 s inter-repeat gap sits
    # exactly at uvicorn's keepalive timeout, and reusing a stale pooled
    # connection races the server's close -> instant ServerDisconnectedError
    # (observed on both systems, ~1 burst in 11).
    connector = aiohttp.TCPConnector(limit=len(batch) + 4, force_close=True)
    dp_rank = None if args.dp_rank < 0 else args.dp_rank
    async with aiohttp.ClientSession(
        timeout=timeout,
        connector=connector,
    ) as session:
        warmup_bursts = []
        for _ in range(args.warmups):
            warmup_bursts.append(
                await run_burst(session, api_url, args.model, batch, dp_rank)
            )
            await asyncio.sleep(INTER_REPEAT_GAP_S)
        repeats = []
        for repeat_index in range(args.repeats):
            burst = await run_burst(
                session, api_url, args.model, batch, dp_rank
            )
            burst["repeat"] = repeat_index + 1
            repeats.append(burst)
            print(
                f"repeat {repeat_index + 1}/{args.repeats}: "
                f"wall {burst['wall_s']:.3f}s, "
                f"ok {burst['successful']}/{len(batch)}",
                flush=True,
            )
            await asyncio.sleep(INTER_REPEAT_GAP_S)
    wall_times = [burst["wall_s"] for burst in repeats]
    return {
        "schema_version": 1,
        "base_url": args.base_url,
        "endpoint": args.endpoint,
        "model": args.model,
        "dp_rank": args.dp_rank,
        "dataset_path": str(args.dataset),
        "dataset_sha256": _sha256_file(args.dataset),
        "batch_requests": len(batch),
        "batch_prompt_tokens": sum(len(request["prompt"]) for request in batch),
        "warmups": warmup_bursts,
        "repeats": repeats,
        "summary": {
            "wall_s_min": min(wall_times),
            "wall_s_median": sorted(wall_times)[len(wall_times) // 2],
            "wall_s_max": max(wall_times),
            "all_successful": all(
                burst["failed"] == 0 for burst in repeats
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--endpoint", default="/v1/completions")
    parser.add_argument("--model", default="deepseek_v3_2")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--dp-rank", type=int, default=0,
                        help="DP replica to pin via X-data-parallel-rank; "
                        "-1 = unpinned (router spreads across all DPs)")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = asyncio.run(async_main(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
