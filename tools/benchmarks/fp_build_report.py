# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Build the full-prefill 32-card experiment report data + charts.

Reads the result tree (00_accept / 01_fixed_batch / 02_profiles_32 /
03_capacity_32) plus the merged rank0 trace with device flows, crunches
everything into report/stats.json, and renders report/charts.html
(Chart.js 4, same style family as bench_results/prefill reports).

Usage: python3 tools/benchmarks/fp_build_report.py \
    --results bench_results/full_prefill_performance \
    --merged-trace merged_rank0_with_device_flows.json
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

SLO_S = 50.0


def _pct(sorted_vals: list[float], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    rank = (len(sorted_vals) - 1) * pct / 100.0
    low = int(rank)
    high = min(low + 1, len(sorted_vals) - 1)
    frac = rank - low
    return sorted_vals[low] * (1 - frac) + sorted_vals[high] * frac


def _series(vals: list[float]) -> dict[str, float]:
    s = sorted(vals)
    return {
        "n": len(vals),
        "min": s[0] if s else 0.0,
        "p50": _pct(s, 50),
        "p90": _pct(s, 90),
        "p95": _pct(s, 95),
        "p99": _pct(s, 99),
        "max": s[-1] if s else 0.0,
        "mean": sum(s) / len(s) if s else 0.0,
    }


def _load(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------- capacity
def analyze_capacity(results: Path) -> dict:
    cap: dict[str, dict] = {}
    root = results / "03_capacity_32/screening"
    for system_dir in sorted(root.iterdir()):
        system = system_dir.name
        points = {}
        for point_dir in sorted(system_dir.iterdir()):
            replay = point_dir / "replay.json"
            if not replay.exists():
                continue
            doc = _load(replay)
            reqs = [r for r in doc["requests"] if r.get("success")]
            ttfts = [r["ttft_s"] for r in reqs]
            lens = [r["input_length"] for r in reqs]
            offsets = [r["actual_send_s"] for r in reqs]
            devs = [r["send_deviation_ms"] for r in reqs]
            s = doc["summary"]
            ttft_stats = _series(ttfts)
            points[point_dir.name] = {
                "target_tps": float(s["target_input_tokens_per_second"]),
                "actual_tps": float(s["actual_send_token_rate"]),
                "n_ok": s["successful"],
                "n_total": s["request_count"],
                "ttft": ttft_stats,
                # recompute SLO against the revised 50 s gate — stored flag
                # may predate the revision (early runs used 10 s).
                "slo_ok": ttft_stats["p99"] <= SLO_S,
                "slo_attain_rate": sum(1 for t in ttfts if t <= SLO_S)
                / max(1, len(ttfts)),
                "dev_ok": s["send_deviation_ok"],
                "dev_ms": s["send_deviation_ms"],
                "queue_drain_s": s["queue_drain_s"],
                "wall_s": s["wall_s"],
                "total_input_tokens": s["total_input_tokens"],
                "goodput_tps": s["total_input_tokens"] / s["wall_s"],
                # per-request series for time/length scatter charts
                "per_request": [
                    {
                        "len": le,
                        "offset_s": round(off, 3),
                        "ttft_s": round(t, 3),
                    }
                    for le, off, t in sorted(
                        zip(lens, offsets, ttfts), key=lambda x: x[1]
                    )
                ],
            }
        cap[system] = points
    return cap


# ------------------------------------------------------------- fixed batch
def analyze_fixed(results: Path) -> dict:
    out: dict[str, dict] = {}
    root = results / "01_fixed_batch"
    for system_dir in sorted(root.iterdir()):
        system = system_dir.name
        for f in sorted(system_dir.glob("*.fixed_batch.json")):
            doc = _load(f)
            batch = f.name.replace(".fixed_batch.json", "")
            walls = [r["wall_s"] for r in doc["repeats"] if r["successful"]]
            n_failed = sum(1 for r in doc["repeats"] if not r["successful"])
            mean_wall = sum(walls) / len(walls) if walls else None
            # per-request ttft within bursts, split by prompt length thirds
            per_len: list[tuple[int, float]] = []
            for rep in doc["repeats"]:
                for req in rep["requests"]:
                    if req.get("success"):
                        per_len.append((req["prompt_len"], req["ttft_s"]))
            lens = sorted({pl for pl, _ in per_len})
            short_cut = lens[len(lens) // 3] if len(lens) >= 3 else 0
            long_cut = lens[2 * len(lens) // 3] if len(lens) >= 3 else 0
            short_ttft = [t for pl, t in per_len if pl <= short_cut]
            long_ttft = [t for pl, t in per_len if pl >= long_cut]
            out.setdefault(system, {})[batch] = {
                "batch_prompt_tokens": doc["batch_prompt_tokens"],
                "batch_requests": doc["batch_requests"],
                "n_repeats": len(doc["repeats"]),
                "n_failed_repeats": n_failed,
                "wall": _series(walls),
                "throughput_tps": (
                    doc["batch_prompt_tokens"] / _series(walls)["p50"]
                    if walls
                    else None
                ),
                "short_req_ttft": _series(short_ttft),
                "long_req_ttft": _series(long_ttft),
                "short_len_max": short_cut,
                "long_len_min": long_cut,
            }
    return out


# ----------------------------------------------------------------- accept
def analyze_accept(results: Path) -> dict:
    out: dict[str, dict] = {}
    root = results / "00_accept"
    for system_dir in sorted(root.iterdir()):
        system = system_dir.name
        singles = []
        for f in sorted(system_dir.glob("accept_single_*.json")):
            doc = _load(f)
            for r in doc["requests"]:
                if r.get("success"):
                    singles.append({"len": r["input_length"], "ttft_s": r["ttft_s"]})
        entry: dict = {"singles": singles}
        warm = system_dir / "accept_warmup32.json"
        if warm.exists():
            w = _load(warm)["summary"]
            entry["warmup32"] = {
                "wall_s": w["wall_s"],
                "ttft": w["ttft_s"],
                "total_input_tokens": w["total_input_tokens"],
            }
        quad = system_dir / "accept_4x52k.json"
        if quad.exists():
            q = _load(quad)
            entry["quad_52k"] = {
                "wall_s": q["summary"]["wall_s"],
                "ttfts": [
                    {"len": r["input_length"], "ttft_s": r["ttft_s"]}
                    for r in q["requests"]
                    if r.get("success")
                ],
            }
        out[system] = entry
    return out


# ---------------------------------------------------------------- profile
def analyze_profiles(results: Path) -> dict:
    out: dict[str, dict] = {}
    root = results / "02_profiles_32"
    for path in sorted(root.rglob("*rank0_summary.json")):
        doc = _load(path)
        # key must include the file stem: attention/ffn summaries share a dir
        rel = f"{path.parent.relative_to(root)}/{path.stem.replace('_rank0_summary', '')}"
        busy = doc["busy_occupancy"]
        cats = {
            k: {"union_ms": v["global_union_ms"], "count": v["event_count"]}
            for k, v in doc["categories"].items()
        }
        out[rel] = {
            "busy_ratio": busy["busy_ratio"],
            "bubble_ratio": busy["bubble_ratio"],
            "cam_wait_ratio": busy["cam_wait_ratio"],
            "span_ms": doc["trace_span_ms"],
            "categories": cats,
            "top_events": [
                {"name": e["name"], "union_ms": e.get("global_union_ms", 0)}
                for e in doc.get("top_events", [])[:12]
            ],
        }
    return out


# -------------------------------------------------------- merged flows
_MSTX_NAME_PREFIX = "afd.cam."
_EVENT_TO_OP = {
    "afd.cam.dispatch_send": "CamMoeDistributeDispatchSend",
    "afd.cam.combine_recv": "CamMoeDistributeCombineRecv",
    "afd.cam.dispatch_recv": "CamMoeDistributeDispatchRecv",
    "afd.cam.combine_send": "CamMoeDistributeCombineSend",
}


def analyze_flows(merged_path: Path) -> dict:
    """Same-clock mstx marker -> device op queue delay (FIFO by ts, the
    exact 1:1 pairing from link_afd_device_flows), plus device op durations.

    Both endpoints live in the torch_npu trace clock domain, so the delay
    is a true on-host queueing measure — unlike corr-marker->device, which
    crosses the sidecar clock domain and picks up alignment residual.
    """
    with open(merged_path, encoding="utf-8") as f:
        payload = json.load(f)
    events = payload["traceEvents"] if isinstance(payload, dict) else payload

    mstx: dict[str, list[float]] = defaultdict(list)
    for e in events:
        name = str(e.get("name", ""))
        if name.startswith(_MSTX_NAME_PREFIX) and " flow_id=" in name:
            ev = name.split(" flow_id=")[0]
            if ev in _EVENT_TO_OP and "ts" in e:
                mstx[ev].append(float(e["ts"]))

    op_ts: dict[str, list[float]] = defaultdict(list)
    op_durs: dict[str, list[float]] = defaultdict(list)
    for e in events:
        if e.get("ph") == "X" and str(e.get("name", "")).startswith("CamMoeDistribute"):
            op_ts[str(e["name"])].append(float(e["ts"]))
            op_durs[str(e["name"])].append(float(e["dur"]) / 1000.0)

    queue_delays: dict[str, list[float]] = {}
    for ev, op in _EVENT_TO_OP.items():
        marks = sorted(mstx.get(ev, []))
        ops = sorted(op_ts.get(op, []))
        delays = [(o - m) / 1000.0 for m, o in zip(marks, ops)]
        queue_delays[ev] = delays

    return {
        "queue_delay_ms": {k: _series(v) for k, v in sorted(queue_delays.items())},
        "device_op_dur_ms": {k: _series(v) for k, v in sorted(op_durs.items())},
    }


# ----------------------------------------------------------------- report
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--merged-trace", type=Path, default=None)
    args = parser.parse_args()

    stats = {
        "capacity": analyze_capacity(args.results),
        "fixed_batch": analyze_fixed(args.results),
        "accept": analyze_accept(args.results),
        "profiles": analyze_profiles(args.results),
    }
    if args.merged_trace and args.merged_trace.exists():
        stats["flows"] = analyze_flows(args.merged_trace)

    out_dir = args.results / "report"
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=1)
    print(f"wrote {out_dir / 'stats.json'}")

    # ---- console digest -------------------------------------------------
    print("\n== capacity ==")
    for system, points in stats["capacity"].items():
        for name, p in sorted(points.items(), key=lambda kv: kv[1]["target_tps"]):
            print(
                f"{system:10s} {p['target_tps']:>9.2f} tps  "
                f"p50={p['ttft']['p50']:6.2f}s p95={p['ttft']['p95']:6.2f}s "
                f"p99={p['ttft']['p99']:6.2f}s  slo={'PASS' if p['slo_ok'] else 'FAIL'} "
                f"dev_p99={p['dev_ms']['p99']:5.1f}ms drain={p['queue_drain_s']:5.1f}s "
                f"goodput={p['goodput_tps']:8.1f}tps"
            )
    print("\n== fixed batch ==")
    for system, batches in stats["fixed_batch"].items():
        for batch, b in sorted(batches.items()):
            tps = f"{b['throughput_tps']:8.1f}" if b["throughput_tps"] else "   crash"
            print(
                f"{system:8s} {batch:24s} med={b['wall']['p50']:7.3f}s "
                f"[{b['wall']['min']:7.3f},{b['wall']['max']:7.3f}] "
                f"tps={tps} failed={b['n_failed_repeats']}/{b['n_repeats']} "
                f"short_ttft_med={b['short_req_ttft']['p50']:6.2f}s "
                f"long_ttft_med={b['long_req_ttft']['p50']:6.2f}s"
            )
    print("\n== accept singles ==")
    for system, entry in stats["accept"].items():
        for sgl in sorted(entry["singles"], key=lambda x: x["len"]):
            tps = sgl["len"] / sgl["ttft_s"]
            print(
                f"{system:8s} len={sgl['len']:6d} ttft={sgl['ttft_s']:6.2f}s "
                f"solo_tps={tps:7.0f}"
            )
    print("\n== profiles ==")
    for rel, p in stats["profiles"].items():
        print(
            f"{rel:28s} busy={p['busy_ratio']*100:5.1f}% "
            f"bubble={p['bubble_ratio']*100:5.1f}% "
            f"cam_wait={p['cam_wait_ratio']*100:5.1f}%"
        )
    if "flows" in stats:
        print("\n== device flow queue delay (ms) ==")
        for ev, s in stats["flows"]["queue_delay_ms"].items():
            print(
                f"{ev:16s} n={s['n']:4d} p50={s['p50']:8.2f} "
                f"p99={s['p99']:8.2f} max={s['max']:9.2f}"
            )
        print("\n== device op duration (ms) ==")
        for ev, s in stats["flows"]["device_op_dur_ms"].items():
            print(
                f"{ev:36s} n={s['n']:4d} p50={s['p50']:7.2f} "
                f"p99={s['p99']:8.2f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
