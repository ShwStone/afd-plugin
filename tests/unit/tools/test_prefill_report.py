# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from __future__ import annotations

import pytest

from tools.benchmarks.prefill_report import (
    ResultGroupKey,
    VerifiedRun,
    aggregate_runs,
    compare_systems,
)


def _run(
    system: str,
    repeat: int,
    ttfts_ms: tuple[float, ...],
    slo_met_requests: int,
) -> VerifiedRun:
    return VerifiedRun(
        key=ResultGroupKey(
            system=system,
            batch_tokens=32_768,
            request_rate=8.0,
            prefix_ratio="0",
        ),
        repeat=repeat,
        issued_requests=3,
        slo_met_requests=slo_met_requests,
        successful_ttfts_ms=ttfts_ms,
    )


def test_report_aggregates_repeats_and_compares_systems() -> None:
    summaries = aggregate_runs(
        [
            _run("baseline", 1, (10_000.0, 20_000.0), 1),
            _run("baseline", 2, (12_000.0, 18_000.0), 1),
            _run("candidate", 1, (8_000.0, 10_000.0, 12_000.0), 2),
            _run("candidate", 2, (8_000.0, 9_000.0, 10_000.0), 3),
        ],
        expected_repeats=2,
    )

    baseline = next(summary for summary in summaries if summary["system"] == "baseline")
    assert baseline["repeat_count"] == 2
    assert baseline["successful_requests"] == 4
    assert baseline["slo_attainment_all_requests"] == pytest.approx(2 / 6)
    assert baseline["repeat_variation"]["mean_ttft_ms_standard_deviation"] is not None

    comparison = compare_systems(
        summaries,
        baseline_system="baseline",
        candidate_system="candidate",
    )[0]
    assert comparison["mean_ttft_reduction_percent"] > 0
    assert comparison["slo_attainment_delta_percentage_points"] == pytest.approx(50)
