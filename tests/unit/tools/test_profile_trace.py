# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.benchmarks.profile_trace import (
    compare_trace_summaries,
    summarize_trace,
)


def _write_trace(trace_path: Path, duration_multiplier: float = 1.0) -> None:
    trace_path.write_text(
        json.dumps(
            {
                "traceEvents": [
                    {
                        "ph": "X",
                        "name": "HCCL_AllReduce",
                        "cat": "kernel",
                        "pid": 1,
                        "tid": 1,
                        "ts": 0,
                        "dur": 10 * duration_multiplier,
                    },
                    {
                        "ph": "X",
                        "name": "GroupedMatMul",
                        "cat": "kernel",
                        "pid": 1,
                        "tid": 2,
                        "ts": 5,
                        "dur": 10 * duration_multiplier,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )


def test_trace_summary_reports_communication_compute_overlap(
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "trace.json"
    _write_trace(trace_path)

    summary = summarize_trace(trace_path)

    overlap = summary["communication_compute_overlap"]
    assert overlap["overlap_ms"] == pytest.approx(0.005)
    assert overlap["overlap_ratio_of_communication"] == pytest.approx(0.5)
    assert summary["categories"]["communication"]["event_count"] == 1
    assert summary["categories"]["ffn_moe"]["event_count"] == 1


def test_trace_comparison_reports_candidate_delta(tmp_path: Path) -> None:
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    _write_trace(baseline_path)
    _write_trace(candidate_path, duration_multiplier=0.5)

    comparison = compare_trace_summaries(
        summarize_trace(baseline_path),
        summarize_trace(candidate_path),
    )

    assert comparison["trace_span_ms"]["delta"] < 0
