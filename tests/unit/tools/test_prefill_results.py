# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from __future__ import annotations

import json
from pathlib import Path

from tools.benchmarks.prefill_results import verify_and_enrich_result


def test_verify_result_counts_failures_in_all_request_slo(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "request_id": "request-1",
                        "source_row": 1,
                        "prompt_len": 4,
                    }
                ),
                json.dumps(
                    {
                        "request_id": "request-2",
                        "source_row": 2,
                        "prompt_len": 8,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    result_path = tmp_path / "result.json"
    result_path.write_text(
        json.dumps(
            {
                "num_prompts": 2,
                "completed": 1,
                "failed": 1,
                "input_lens": [4, 8],
                "errors": ["", "server error"],
                "ttfts": [0.5, 0.0],
                "itls": [[], []],
            }
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "verified.json"

    verified = verify_and_enrich_result(
        result_path,
        dataset_path,
        output_path,
        ttft_slo_ms=1_000,
    )

    summary = verified["afd_verification"]
    assert summary["successful_requests"] == 1
    assert summary["slo_met_requests"] == 1
    assert summary["slo_attainment_all_requests"] == 0.5
    assert verified["successes"] == [True, False]
    assert verified["request_ids"] == ["request-1", "request-2"]
    assert output_path.is_file()


def _write_basic_run(tmp_path: Path) -> tuple[Path, Path, Path]:
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text(
        "\n".join(
            json.dumps(
                {
                    "request_id": f"request-{index}",
                    "source_row": index,
                    "prompt_len": 4,
                }
            )
            for index in range(2)
        )
        + "\n",
        encoding="utf-8",
    )
    result_path = tmp_path / "result.json"
    result_path.write_text(
        json.dumps(
            {
                "num_prompts": 2,
                "completed": 2,
                "failed": 0,
                "input_lens": [4, 4],
                "errors": ["", ""],
                "ttfts": [0.5, 0.6],
                "itls": [[], []],
                "start_times": [100.0, 100.5],
            }
        ),
        encoding="utf-8",
    )
    return dataset_path, result_path, tmp_path / "verified.json"


def test_verify_result_checks_arrival_plan_within_tolerance(tmp_path: Path) -> None:
    dataset_path, result_path, output_path = _write_basic_run(tmp_path)
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "seed": 20260817,
                "request_rate": 2.0,
                "planned_send_offsets": [0.0, 0.55],
            }
        ),
        encoding="utf-8",
    )

    verified = verify_and_enrich_result(
        result_path,
        dataset_path,
        output_path,
        ttft_slo_ms=1_000,
        arrival_plan_path=plan_path,
    )

    arrival = verified["afd_verification"]["arrival"]
    assert arrival["seed"] == 20260817
    assert arrival["max_abs_deviation_s"] == 0.050000000000000044


def test_verify_result_rejects_arrival_plan_violation(tmp_path: Path) -> None:
    dataset_path, result_path, output_path = _write_basic_run(tmp_path)
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "seed": 20260817,
                "request_rate": 2.0,
                "planned_send_offsets": [0.0, 3.5],
            }
        ),
        encoding="utf-8",
    )

    import pytest

    with pytest.raises(ValueError, match="arrival plan"):
        verify_and_enrich_result(
            result_path,
            dataset_path,
            output_path,
            ttft_slo_ms=1_000,
            arrival_plan_path=plan_path,
        )
