# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Validate and enrich detailed prefill-only vLLM benchmark results."""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Sequence
from pathlib import Path

from tools.benchmarks.prefill_dataset import INDEX_SUFFIX

DEFAULT_TTFT_SLO_MS = 10_000.0
MILLISECONDS_PER_SECOND = 1_000.0
PROMPT_LENGTH_BUCKETS = (8_192, 16_384, 32_768, 49_152)
PERCENTILES = (50, 90, 95, 99)
VERIFIED_RESULT_SCHEMA_VERSION = 1


def _load_json_object(json_path: Path) -> dict[str, object]:
    value = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{json_path} must contain one JSON object.")
    return value


def _load_request_metadata(
    dataset_path: Path,
    request_count: int,
) -> tuple[list[str], list[int], list[int]]:
    request_ids: list[str] = []
    source_rows: list[int] = []
    prompt_lengths: list[int] = []
    index_path = dataset_path.with_name(dataset_path.name + INDEX_SUFFIX)
    metadata_path = index_path if index_path.is_file() else dataset_path
    with metadata_path.open(encoding="utf-8") as dataset_file:
        for line_number, line in enumerate(dataset_file, start=1):
            if len(request_ids) >= request_count:
                break
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(
                    f"{metadata_path}:{line_number} must contain a JSON object."
                )
            request_id = record.get("request_id")
            source_row = record.get("source_row")
            prompt_length = record.get("prompt_len")
            if (
                not isinstance(request_id, str)
                or isinstance(source_row, bool)
                or not isinstance(source_row, int)
                or isinstance(prompt_length, bool)
                or not isinstance(prompt_length, int)
            ):
                raise ValueError(
                    f"{metadata_path}:{line_number} has invalid request metadata."
                )
            request_ids.append(request_id)
            source_rows.append(source_row)
            prompt_lengths.append(prompt_length)
    if len(request_ids) != request_count:
        raise ValueError(
            f"{dataset_path} contains fewer than {request_count} requests."
        )
    return request_ids, source_rows, prompt_lengths


def _require_list(
    result: dict[str, object],
    field_name: str,
    request_count: int,
) -> list[object]:
    value = result.get(field_name)
    if not isinstance(value, list) or len(value) != request_count:
        raise ValueError(
            f"Detailed result field {field_name!r} must contain "
            f"{request_count} entries."
        )
    return value


def _numeric_values(values: Sequence[object], field_name: str) -> list[float]:
    numeric_values: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Result field {field_name!r} must be numeric.")
        numeric_values.append(float(value))
    return numeric_values


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


def _bucket_label(prompt_length: int) -> str:
    lower_bound = 0
    for upper_bound in PROMPT_LENGTH_BUCKETS:
        if prompt_length <= upper_bound:
            return f"{lower_bound + 1}-{upper_bound}"
        lower_bound = upper_bound
    return f"{PROMPT_LENGTH_BUCKETS[-1] + 1}+"


def _build_length_bucket_summary(
    prompt_lengths: Sequence[int],
    ttfts_seconds: Sequence[float],
    successes: Sequence[bool],
    slo_met: Sequence[bool],
) -> dict[str, dict[str, object]]:
    buckets: dict[str, list[int]] = {}
    for request_index, prompt_length in enumerate(prompt_lengths):
        buckets.setdefault(_bucket_label(prompt_length), []).append(request_index)

    summary: dict[str, dict[str, object]] = {}
    for bucket_name, request_indices in buckets.items():
        successful_ttfts_ms = sorted(
            ttfts_seconds[index] * MILLISECONDS_PER_SECOND
            for index in request_indices
            if successes[index]
        )
        summary[bucket_name] = {
            "issued": len(request_indices),
            "successful": sum(successes[index] for index in request_indices),
            "slo_met": sum(slo_met[index] for index in request_indices),
            "slo_attainment_all_requests": (
                sum(slo_met[index] for index in request_indices) / len(request_indices)
            ),
            "successful_ttft_ms": {
                f"p{percentile}": _percentile(successful_ttfts_ms, percentile)
                for percentile in PERCENTILES
            },
        }
    return summary


def _verify_arrival_plan(
    result: dict[str, object],
    request_count: int,
    arrival_plan_path: Path | None,
    *,
    max_arrival_deviation_s: float,
) -> dict[str, object] | None:
    """Compare recorded send times against the frozen arrival plan."""
    if arrival_plan_path is None:
        return None
    plan = _load_json_object(arrival_plan_path)
    planned_offsets = plan.get("planned_send_offsets")
    if not isinstance(planned_offsets, list) or len(planned_offsets) != request_count:
        raise ValueError(
            "Arrival plan request count does not match benchmark result."
        )
    start_times = _numeric_values(
        _require_list(result, "start_times", request_count),
        "start_times",
    )
    origin_actual = start_times[0]
    origin_planned = float(planned_offsets[0])
    deviations = [
        (start_time - origin_actual) - (float(planned) - origin_planned)
        for start_time, planned in zip(start_times, planned_offsets, strict=True)
    ]
    max_abs_deviation = max(abs(deviation) for deviation in deviations)
    if max_abs_deviation > max_arrival_deviation_s:
        raise ValueError(
            "Benchmark send times deviate from the frozen arrival plan by "
            f"{max_abs_deviation:.3f}s (limit {max_arrival_deviation_s:.3f}s)."
        )
    manifest_path = arrival_plan_path.with_name(
        arrival_plan_path.name + ".manifest.json"
    )
    plan_sha256: object = None
    if manifest_path.is_file():
        plan_sha256 = _load_json_object(manifest_path).get("arrival_plan_sha256")
    return {
        "arrival_plan_path": str(arrival_plan_path),
        "arrival_plan_sha256": plan_sha256,
        "seed": plan.get("seed"),
        "request_rate": plan.get("request_rate"),
        "planned_span_s": float(planned_offsets[-1]) - origin_planned,
        "actual_span_s": start_times[-1] - origin_actual,
        "max_abs_deviation_s": max_abs_deviation,
        "mean_abs_deviation_s": statistics.fmean(
            abs(deviation) for deviation in deviations
        ),
    }


def verify_and_enrich_result(
    result_path: Path,
    dataset_path: Path,
    output_path: Path,
    *,
    ttft_slo_ms: float = DEFAULT_TTFT_SLO_MS,
    arrival_plan_path: Path | None = None,
    max_arrival_deviation_s: float = 2.0,
) -> dict[str, object]:
    """Cross-check detailed arrays and compute all-issued-request SLO metrics.

    When ``arrival_plan_path`` is given, the recorded per-request
    ``start_times`` are compared against the frozen Poisson arrival plan
    (plan section 6.3): the run is rejected if send times deviate from the
    plan by more than ``max_arrival_deviation_s``.
    """
    if ttft_slo_ms <= 0:
        raise ValueError("ttft_slo_ms must be positive.")

    result = _load_json_object(result_path)
    request_count_value = result.get("num_prompts")
    if (
        isinstance(request_count_value, bool)
        or not isinstance(request_count_value, int)
        or request_count_value <= 0
    ):
        raise ValueError("Benchmark result must contain a positive num_prompts.")
    request_count = request_count_value

    request_ids, source_rows, dataset_prompt_lengths = _load_request_metadata(
        dataset_path,
        request_count,
    )
    result_prompt_lengths = _numeric_values(
        _require_list(result, "input_lens", request_count),
        "input_lens",
    )
    if result_prompt_lengths != dataset_prompt_lengths:
        raise ValueError(
            "Benchmark input_lens do not match the unshuffled custom dataset."
        )

    errors = _require_list(result, "errors", request_count)
    if any(not isinstance(error, str) for error in errors):
        raise ValueError("Result field 'errors' must contain strings.")
    ttfts_seconds = _numeric_values(
        _require_list(result, "ttfts", request_count),
        "ttfts",
    )
    itls = _require_list(result, "itls", request_count)
    if any(not isinstance(request_itls, list) for request_itls in itls):
        raise ValueError("Result field 'itls' must contain lists.")

    successes = [
        error == "" and ttft_seconds > 0
        for error, ttft_seconds in zip(errors, ttfts_seconds, strict=True)
    ]
    slo_seconds = ttft_slo_ms / MILLISECONDS_PER_SECOND
    slo_met = [
        success and ttft_seconds <= slo_seconds
        for success, ttft_seconds in zip(successes, ttfts_seconds, strict=True)
    ]
    latencies_seconds = [
        ttft_seconds + sum(_numeric_values(request_itls, "itls"))
        for ttft_seconds, request_itls in zip(ttfts_seconds, itls, strict=True)
    ]

    completed_value = result.get("completed")
    failed_value = result.get("failed")
    if completed_value != sum(successes):
        raise ValueError(
            "Benchmark completed count does not match detailed success records."
        )
    if failed_value != request_count - sum(successes):
        raise ValueError(
            "Benchmark failed count does not match detailed failure records."
        )

    successful_count = sum(successes)
    successful_ttfts_ms = sorted(
        ttft_seconds * MILLISECONDS_PER_SECOND
        for ttft_seconds, success in zip(ttfts_seconds, successes, strict=True)
        if success
    )
    successful_latencies_ms = sorted(
        latency_seconds * MILLISECONDS_PER_SECOND
        for latency_seconds, success in zip(
            latencies_seconds,
            successes,
            strict=True,
        )
        if success
    )
    result["afd_verification"] = {
        "schema_version": VERIFIED_RESULT_SCHEMA_VERSION,
        "dataset_path": str(dataset_path),
        "ttft_slo_ms": ttft_slo_ms,
        "issued_requests": request_count,
        "successful_requests": successful_count,
        "failed_requests": request_count - successful_count,
        "slo_met_requests": sum(slo_met),
        "slo_attainment_all_requests": sum(slo_met) / request_count,
        "slo_attainment_successful_requests": (
            sum(slo_met) / successful_count if successful_count else None
        ),
        "successful_ttft_ms": {
            "mean": (
                statistics.fmean(successful_ttfts_ms) if successful_ttfts_ms else None
            ),
            **{
                f"p{percentile}": _percentile(
                    successful_ttfts_ms,
                    percentile,
                )
                for percentile in PERCENTILES
            },
        },
        "successful_e2el_ms": {
            "mean": (
                statistics.fmean(successful_latencies_ms)
                if successful_latencies_ms
                else None
            ),
            **{
                f"p{percentile}": _percentile(
                    successful_latencies_ms,
                    percentile,
                )
                for percentile in PERCENTILES
            },
        },
        "length_buckets": _build_length_bucket_summary(
            dataset_prompt_lengths,
            ttfts_seconds,
            successes,
            slo_met,
        ),
    }
    arrival_summary = _verify_arrival_plan(
        result,
        request_count,
        arrival_plan_path,
        max_arrival_deviation_s=max_arrival_deviation_s,
    )
    if arrival_summary is not None:
        result["afd_verification"]["arrival"] = arrival_summary
    result["request_ids"] = request_ids
    result["source_rows"] = source_rows
    result["successes"] = successes
    result["slo_met"] = slo_met
    result["latencies"] = latencies_seconds

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return result


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ttft-slo-ms", type=float, default=DEFAULT_TTFT_SLO_MS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""
    args = _build_argument_parser().parse_args(argv)
    result = verify_and_enrich_result(
        args.result,
        args.dataset,
        args.output,
        ttft_slo_ms=args.ttft_slo_ms,
    )
    print(json.dumps(result["afd_verification"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
