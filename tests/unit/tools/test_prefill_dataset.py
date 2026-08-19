# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.benchmarks.prefill_dataset import (
    INDEX_SUFFIX,
    generate_dataset,
    read_source_requests,
    validate_dataset,
)


def _write_model_config(config_path: Path) -> None:
    config_path.write_text(
        json.dumps(
            {
                "vocab_size": 32,
                "bos_token_id": 0,
                "eos_token_id": [1, 2],
                "pad_token_id": 3,
            }
        ),
        encoding="utf-8",
    )
    (config_path.parent / "tokenizer_config.json").write_text(
        json.dumps(
            {"added_tokens_decoder": {"4": {"content": "<special>", "special": True}}}
        ),
        encoding="utf-8",
    )


def test_generate_dataset_is_deterministic_and_preserves_nonzero_rows(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "lengths.csv"
    csv_path.write_text("8\n0\n6\n4\n", encoding="utf-8")
    config_path = tmp_path / "config.json"
    _write_model_config(config_path)
    first_output = tmp_path / "first.jsonl"
    second_output = tmp_path / "second.jsonl"

    first_manifest = generate_dataset(csv_path, config_path, first_output)
    second_manifest = generate_dataset(csv_path, config_path, second_output)

    assert first_output.read_bytes() == second_output.read_bytes()
    first_index = first_output.with_name(first_output.name + INDEX_SUFFIX)
    second_index = second_output.with_name(second_output.name + INDEX_SUFFIX)
    assert first_index.read_bytes() == second_index.read_bytes()
    assert first_manifest["request_count"] == 3
    assert first_manifest["total_prompt_tokens"] == 18
    assert first_manifest["prompt_length_percentiles"]["p50"] == 6
    assert first_manifest["dataset_sha256"] == second_manifest["dataset_sha256"]
    output_lines = first_output.read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in output_lines]
    assert [record["source_row"] for record in records] == [1, 3, 4]
    assert [record["prompt_len"] for record in records] == [8, 6, 4]
    assert all(
        token_id not in {0, 1, 2, 3, 4}
        for record in records
        for token_id in record["prompt"]
    )
    assert (
        validate_dataset(
            first_output,
            csv_path=csv_path,
            model_config_path=config_path,
        )["request_count"]
        == 3
    )


def test_prefix_dataset_shares_aligned_prefix_inside_group(tmp_path: Path) -> None:
    csv_path = tmp_path / "lengths.csv"
    csv_path.write_text("8\n6\n8\n", encoding="utf-8")
    config_path = tmp_path / "config.json"
    _write_model_config(config_path)
    output_path = tmp_path / "prefix.jsonl"

    manifest = generate_dataset(
        csv_path,
        config_path,
        output_path,
        prefix_ratio=0.5,
        prefix_block_size=2,
        prefix_group_size=2,
    )

    output_lines = output_path.read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in output_lines]
    shared_length = min(
        records[0]["shared_prefix_len"],
        records[1]["shared_prefix_len"],
    )
    assert shared_length > 0
    assert records[0]["prompt"][:shared_length] == records[1]["prompt"][:shared_length]
    assert records[2]["prefix_group"] == 1
    assert manifest["estimated_sequential_reusable_prefix_token_ratio"] > 0


def test_csv_rejects_more_than_one_nonzero_length_per_row(tmp_path: Path) -> None:
    csv_path = tmp_path / "invalid.csv"
    csv_path.write_text("1,2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="exactly one non-zero"):
        read_source_requests(csv_path)
