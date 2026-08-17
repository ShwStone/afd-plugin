# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Aggregate Stage-3 btsweep L0 results and build the occupancy/throughput/
bubble attribution table.

L0 sweep: reads ``*.verified.json`` under ``01_sweep`` and produces one row
per (system, rps, mbt): TTFT mean/p99, tokens/s, req/s, SLO, repeat spread.

L2 attribution (optional, when profile data exists): merges
``profile_trace`` busy/bubble summaries and ``sample_npu_smi`` AICore% curves
under ``02_profile`` to attribute the TTFT-vs-mbt trend to occupancy ↑ /
throughput ↑ / bubble ↓.

Usage:
  python3 -m tools.benchmarks.analyze_btsweep \
    [--sweep-dir bench_results/prefill_stage3/01_sweep] \
    [--profile-dir bench_results/prefill_stage3/02_profile] \
    [--output-dir bench_results/prefill_stage3/03_reports]
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

from tools.benchmarks.profile_trace import summarize_trace

MILLISECONDS_PER_SECOND = 1_000.0
TRACE_GLOBS = ("*.pt.trace.json", "*.pt.trace.json.gz", "*.json.gz")


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _flatten_runs(sweep_dir: Path) -> list[dict[str, object]]:
    runs: list[dict[str, object]] = []
    for result_path in sorted(sweep_dir.rglob("*.verified.json")):
        result = _read_json(result_path)
        verification = result.get("afd_verification")
        if not isinstance(verification, dict):
            continue
        ttft = verification.get("successful_ttft_ms")
        if not isinstance(ttft, dict):
            continue
        runs.append(
            {
                "system": str(result.get("afd_system")),
                "max_num_batched_tokens": int(result.get("max_num_batched_tokens")),
                "request_rate": float(result.get("request_rate")),
                "prefix_ratio": str(result.get("prefix_ratio")),
                "repeat": int(result.get("repeat")),
                "ttft_mean_ms": ttft.get("mean"),
                "ttft_p99_ms": ttft.get("p99"),
                "tokens_per_s": result.get("total_token_throughput"),
                "requests_per_s": result.get("request_throughput"),
                "slo_attainment_all": verification.get(
                    "slo_attainment_all_requests"
                ),
                "issued_requests": verification.get("issued_requests"),
            }
        )
    return runs


def summarize_sweep(runs: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, float, int], list[dict[str, object]]] = defaultdict(
        list
    )
    for run in runs:
        grouped[(run["system"], run["request_rate"], run["max_num_batched_tokens"])].append(
            run
        )
    rows: list[dict[str, object]] = []
    for (system, rps, mbt), cell_runs in sorted(grouped.items()):
        ttft_means = [r["ttft_mean_ms"] for r in cell_runs if r["ttft_mean_ms"] is not None]
        ttft_p99s = [r["ttft_p99_ms"] for r in cell_runs if r["ttft_p99_ms"] is not None]
        token_thru = [r["tokens_per_s"] for r in cell_runs if r["tokens_per_s"] is not None]
        req_thru = [r["requests_per_s"] for r in cell_runs if r["requests_per_s"] is not None]
        slos = [r["slo_attainment_all"] for r in cell_runs if r["slo_attainment_all"] is not None]
        rows.append(
            {
                "system": system,
                "rps": rps,
                "max_num_batched_tokens": mbt,
                "repeats": len(cell_runs),
                "ttft_mean_ms": statistics.fmean(ttft_means) if ttft_means else None,
                "ttft_p99_ms": statistics.fmean(ttft_p99s) if ttft_p99s else None,
                "ttft_mean_repeat_stdev_ms": (
                    statistics.stdev(ttft_means) if len(ttft_means) > 1 else None
                ),
                "tokens_per_s": statistics.fmean(token_thru) if token_thru else None,
                "requests_per_s": statistics.fmean(req_thru) if req_thru else None,
                "slo_attainment_all": statistics.fmean(slos) if slos else None,
            }
        )
    return rows


def _npu_smi_occupancy(profile_dir: Path) -> dict[tuple[str, int, float], float]:
    """Mean AICore% per (system, mbt, rps) from the continuous sampler CSVs."""
    occupancy: dict[tuple[str, int, float], list[float]] = defaultdict(list)
    for csv_path in sorted(profile_dir.rglob("telemetry/npu_smi_node*.csv")):
        try:
            lines = csv_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        if not lines or not lines[0].startswith("timestamp"):
            continue
        for line in lines[1:]:
            parts = line.split(",")
            if len(parts) < 4:
                continue
            try:
                aicore = float(parts[2])
            except ValueError:
                continue
            # Map the CSV to its (system, mbt, rps) via the parent path
            # .../02_profile/rps<R>/mbt<M>/<system>/telemetry/...
            rel = csv_path.relative_to(profile_dir)
            parts_rel = rel.parts
            try:
                system = parts_rel[2]
                mbt = int(parts_rel[1].replace("mbt", ""))
                rps = float(parts_rel[0].replace("rps", ""))
            except (IndexError, ValueError):
                continue
            occupancy[(system, mbt, rps)].append(aicore)
    return {
        key: statistics.fmean(values) if values else None
        for key, values in occupancy.items()
    }


def _trace_metrics(profile_dir: Path) -> dict[tuple[str, int, float], dict[str, object]]:
    """Aggregate busy/bubble/overlap from torch_npu trace summaries."""
    metrics: dict[tuple[str, int, float], list[dict[str, object]]] = defaultdict(list)
    for trace_path in sorted(profile_dir.rglob("traces/*")):
        if not any(trace_path.name.endswith(glob) for glob in TRACE_GLOBS):
            continue
        if "rank" not in trace_path.name:
            continue
        try:
            summary = summarize_trace(trace_path)
        except (ValueError, OSError):
            continue
        occupancy = summary.get("busy_occupancy")
        if not isinstance(occupancy, dict):
            continue
        rel = trace_path.relative_to(profile_dir)
        parts_rel = rel.parts
        # .../02_profile/rps<R>/mbt<M>/<system>/traces/rank<k>...
        try:
            system = parts_rel[2]
            mbt = int(parts_rel[1].replace("mbt", ""))
            rps = float(parts_rel[0].replace("rps", ""))
        except (IndexError, ValueError):
            continue
        metrics[(system, mbt, rps)].append(
            {
                "busy_ratio": occupancy.get("busy_ratio"),
                "bubble_ratio": occupancy.get("bubble_ratio"),
                "attention_ffn_overlap_ratio": occupancy.get(
                    "attention_ffn_overlap_ratio"
                ),
            }
        )
    aggregated: dict[tuple[str, int, float], dict[str, object]] = {}
    for key, values in metrics.items():
        def _mean(field: str) -> float | None:
            vals = [v[field] for v in values if v.get(field) is not None]
            return statistics.fmean(vals) if vals else None

        aggregated[key] = {
            "traces_summarized": len(values),
            "busy_ratio": _mean("busy_ratio"),
            "bubble_ratio": _mean("bubble_ratio"),
            "attention_ffn_overlap_ratio": _mean("attention_ffn_overlap_ratio"),
        }
    return aggregated


def build_attribution(
    sweep_rows: list[dict[str, object]],
    profile_dir: Path,
) -> list[dict[str, object]]:
    occupancy = _npu_smi_occupancy(profile_dir)
    trace_metrics = _trace_metrics(profile_dir)
    rows: list[dict[str, object]] = []
    for row in sweep_rows:
        key = (str(row["system"]), int(row["max_num_batched_tokens"]), float(row["rps"]))
        rows.append(
            {
                **row,
                "occupancy_npu_smi": occupancy.get(key),
                **(
                    trace_metrics.get(key, {})
                    if trace_metrics.get(key)
                    else {
                        "traces_summarized": None,
                        "busy_ratio": None,
                        "bubble_ratio": None,
                        "attention_ffn_overlap_ratio": None,
                    }
                ),
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-dir", type=Path,
                        default=Path("bench_results/prefill_stage3/01_sweep"))
    parser.add_argument("--profile-dir", type=Path,
                        default=Path("bench_results/prefill_stage3/02_profile"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("bench_results/prefill_stage3/03_reports"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_argument_parser().parse_args(argv)
    runs = _flatten_runs(args.sweep_dir)
    if not runs:
        print("No verified runs under", args.sweep_dir)
        return 1
    sweep_rows = summarize_sweep(runs)
    attribution = build_attribution(sweep_rows, args.profile_dir)
    report = {
        "schema_version": 1,
        "sweep_dir": str(args.sweep_dir),
        "profile_dir": str(args.profile_dir),
        "rows": attribution,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "btsweep_attribution.json"
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_csv(args.output_dir / "btsweep_attribution.csv", attribution)
    print(f"Wrote {len(attribution)} attribution rows -> {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
