# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Aggregate verified prefill results and compare AFD with the baseline."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

MILLISECONDS_PER_SECOND = 1_000.0
PERCENTILES = (50, 90, 95, 99)
REPORT_SCHEMA_VERSION = 1
DEFAULT_EXPECTED_REPEATS = 3


@dataclass(frozen=True, order=True)
class ResultGroupKey:
    """Dimensions shared by repeat results."""

    system: str
    batch_tokens: int
    request_rate: float
    prefix_ratio: str


@dataclass(frozen=True)
class VerifiedRun:
    """Fields required from one verified result."""

    key: ResultGroupKey
    repeat: int
    issued_requests: int
    slo_met_requests: int
    successful_ttfts_ms: tuple[float, ...]


def _percentile(sorted_values: Sequence[float], percentile: int) -> float | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = percentile / 100 * (len(sorted_values) - 1)
    lower_index = int(rank)
    upper_index = min(lower_index + 1, len(sorted_values) - 1)
    fraction = rank - lower_index
    return (
        sorted_values[lower_index] * (1 - fraction)
        + sorted_values[upper_index] * fraction
    )


def _required_integer(mapping: dict[str, object], key: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool):
        raise ValueError(f"Verified result field {key!r} must be an integer.")
    try:
        converted_value = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Verified result field {key!r} must be an integer."
        ) from error
    return converted_value


def _required_float(mapping: dict[str, object], key: str) -> float:
    value = mapping.get(key)
    if isinstance(value, bool):
        raise ValueError(f"Verified result field {key!r} must be numeric.")
    try:
        return float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Verified result field {key!r} must be numeric.") from error


def _required_string(mapping: dict[str, object], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Verified result field {key!r} must be a string.")
    return value


def load_verified_run(result_path: Path) -> VerifiedRun:
    """Load one `.verified.json` artifact."""
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError(f"{result_path} must contain a JSON object.")
    verification = result.get("afd_verification")
    if not isinstance(verification, dict):
        raise ValueError(f"{result_path} does not contain afd_verification.")
    successes = result.get("successes")
    ttfts = result.get("ttfts")
    if (
        not isinstance(successes, list)
        or not isinstance(ttfts, list)
        or len(successes) != len(ttfts)
    ):
        raise ValueError(f"{result_path} has invalid detailed request arrays.")

    successful_ttfts_ms: list[float] = []
    for success, ttft_seconds in zip(successes, ttfts, strict=True):
        if not isinstance(success, bool):
            raise ValueError(f"{result_path} has a non-boolean success value.")
        if isinstance(ttft_seconds, bool) or not isinstance(
            ttft_seconds,
            (int, float),
        ):
            raise ValueError(f"{result_path} has a non-numeric TTFT.")
        if success:
            successful_ttfts_ms.append(float(ttft_seconds) * MILLISECONDS_PER_SECOND)

    return VerifiedRun(
        key=ResultGroupKey(
            system=_required_string(result, "afd_system"),
            batch_tokens=_required_integer(result, "max_num_batched_tokens"),
            request_rate=_required_float(result, "request_rate"),
            prefix_ratio=_required_string(result, "prefix_ratio"),
        ),
        repeat=_required_integer(result, "repeat"),
        issued_requests=_required_integer(verification, "issued_requests"),
        slo_met_requests=_required_integer(verification, "slo_met_requests"),
        successful_ttfts_ms=tuple(successful_ttfts_ms),
    )


def aggregate_runs(
    runs: Sequence[VerifiedRun],
    *,
    expected_repeats: int = DEFAULT_EXPECTED_REPEATS,
) -> list[dict[str, object]]:
    """Aggregate requests across repeats for each experiment cell."""
    if expected_repeats <= 0:
        raise ValueError("expected_repeats must be positive.")
    grouped_runs: dict[ResultGroupKey, list[VerifiedRun]] = {}
    for run in runs:
        grouped_runs.setdefault(run.key, []).append(run)

    summaries: list[dict[str, object]] = []
    for key, group_runs in sorted(grouped_runs.items()):
        repeats = sorted({run.repeat for run in group_runs})
        if len(repeats) != len(group_runs):
            raise ValueError(f"Duplicate repeats for result group {key}.")
        issued_requests = sum(run.issued_requests for run in group_runs)
        slo_met_requests = sum(run.slo_met_requests for run in group_runs)
        successful_ttfts_ms = sorted(
            ttft_ms for run in group_runs for ttft_ms in run.successful_ttfts_ms
        )
        successful_requests = len(successful_ttfts_ms)
        repeat_mean_ttfts_ms = [
            statistics.fmean(run.successful_ttfts_ms)
            for run in group_runs
            if run.successful_ttfts_ms
        ]
        repeat_slo_attainment = [
            run.slo_met_requests / run.issued_requests
            for run in group_runs
            if run.issued_requests
        ]
        summaries.append(
            {
                "system": key.system,
                "max_num_batched_tokens": key.batch_tokens,
                "request_rate": key.request_rate,
                "prefix_ratio": key.prefix_ratio,
                "repeats": repeats,
                "repeat_count": len(repeats),
                "expected_repeats": expected_repeats,
                "complete": len(repeats) == expected_repeats,
                "issued_requests": issued_requests,
                "successful_requests": successful_requests,
                "success_rate": (
                    successful_requests / issued_requests if issued_requests else None
                ),
                "slo_met_requests": slo_met_requests,
                "slo_attainment_all_requests": (
                    slo_met_requests / issued_requests if issued_requests else None
                ),
                "successful_ttft_ms": {
                    "mean": (
                        statistics.fmean(successful_ttfts_ms)
                        if successful_ttfts_ms
                        else None
                    ),
                    **{
                        f"p{percentile}": _percentile(
                            successful_ttfts_ms,
                            percentile,
                        )
                        for percentile in PERCENTILES
                    },
                },
                "repeat_variation": {
                    "mean_ttft_ms_by_repeat": repeat_mean_ttfts_ms,
                    "mean_ttft_ms_standard_deviation": (
                        statistics.stdev(repeat_mean_ttfts_ms)
                        if len(repeat_mean_ttfts_ms) > 1
                        else None
                    ),
                    "slo_attainment_by_repeat": repeat_slo_attainment,
                    "slo_attainment_standard_deviation": (
                        statistics.stdev(repeat_slo_attainment)
                        if len(repeat_slo_attainment) > 1
                        else None
                    ),
                },
            }
        )
    return summaries


def compare_systems(
    summaries: Sequence[dict[str, object]],
    *,
    baseline_system: str,
    candidate_system: str,
) -> list[dict[str, object]]:
    """Pair cell summaries and compute candidate deltas."""
    by_dimensions: dict[
        tuple[str, int, float, str],
        dict[str, object],
    ] = {}
    for summary in summaries:
        system = _required_string(summary, "system")
        dimensions = (
            system,
            _required_integer(summary, "max_num_batched_tokens"),
            _required_float(summary, "request_rate"),
            _required_string(summary, "prefix_ratio"),
        )
        by_dimensions[dimensions] = summary

    comparisons: list[dict[str, object]] = []
    candidate_dimensions = sorted(
        dimensions for dimensions in by_dimensions if dimensions[0] == candidate_system
    )
    for _, batch_tokens, request_rate, prefix_ratio in candidate_dimensions:
        baseline = by_dimensions.get(
            (baseline_system, batch_tokens, request_rate, prefix_ratio)
        )
        candidate = by_dimensions[
            (candidate_system, batch_tokens, request_rate, prefix_ratio)
        ]
        if baseline is None:
            continue
        baseline_ttft = baseline.get("successful_ttft_ms")
        candidate_ttft = candidate.get("successful_ttft_ms")
        if not isinstance(baseline_ttft, dict) or not isinstance(
            candidate_ttft,
            dict,
        ):
            continue
        baseline_mean = baseline_ttft.get("mean")
        candidate_mean = candidate_ttft.get("mean")
        baseline_p99 = baseline_ttft.get("p99")
        candidate_p99 = candidate_ttft.get("p99")
        baseline_slo = baseline.get("slo_attainment_all_requests")
        candidate_slo = candidate.get("slo_attainment_all_requests")
        if not all(
            isinstance(value, (int, float))
            for value in (
                baseline_mean,
                candidate_mean,
                baseline_p99,
                candidate_p99,
                baseline_slo,
                candidate_slo,
            )
        ):
            continue
        comparisons.append(
            {
                "max_num_batched_tokens": batch_tokens,
                "request_rate": request_rate,
                "prefix_ratio": prefix_ratio,
                "baseline_system": baseline_system,
                "candidate_system": candidate_system,
                "mean_ttft_reduction_percent": (
                    (baseline_mean - candidate_mean) / baseline_mean * 100
                    if baseline_mean
                    else None
                ),
                "p99_ttft_reduction_percent": (
                    (baseline_p99 - candidate_p99) / baseline_p99 * 100
                    if baseline_p99
                    else None
                ),
                "slo_attainment_delta_percentage_points": (candidate_slo - baseline_slo)
                * 100,
                "baseline_complete": baseline.get("complete"),
                "candidate_complete": candidate.get("complete"),
            }
        )
    return comparisons


def _flatten_summary(summary: dict[str, object]) -> dict[str, object]:
    ttft_summary = summary.get("successful_ttft_ms")
    if not isinstance(ttft_summary, dict):
        ttft_summary = {}
    return {
        key: value
        for key, value in summary.items()
        if key not in {"successful_ttft_ms", "repeat_variation"}
    } | {f"ttft_{key}_ms": value for key, value in ttft_summary.items()}


def _write_csv(csv_path: Path, summaries: Sequence[dict[str, object]]) -> None:
    flattened_summaries = [_flatten_summary(summary) for summary in summaries]
    if not flattened_summaries:
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(flattened_summaries[0]))
        writer.writeheader()
        writer.writerows(flattened_summaries)


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument(
        "--expected-repeats",
        type=int,
        default=DEFAULT_EXPECTED_REPEATS,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""
    args = _build_argument_parser().parse_args(argv)
    result_paths = sorted(args.result_dir.glob("*.verified.json"))
    if not result_paths:
        raise ValueError(f"No verified results found under {args.result_dir}.")
    runs = [load_verified_run(result_path) for result_path in result_paths]
    summaries = aggregate_runs(runs, expected_repeats=args.expected_repeats)
    report: dict[str, object] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "result_directory": str(args.result_dir),
        "cells": summaries,
        "comparisons": compare_systems(
            summaries,
            baseline_system=args.baseline,
            candidate_system=args.candidate,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.csv_output:
        _write_csv(args.csv_output, summaries)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
