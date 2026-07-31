# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from __future__ import annotations

import pytest

from tools.benchmarks.prefill_bench import (
    REQUIRED_SWITCHES,
    build_vllm_arguments,
)


def test_bench_wrapper_adds_prefill_invariants() -> None:
    arguments = build_vllm_arguments(["--dataset-path", "dataset.jsonl"])

    assert arguments[:2] == ["bench", "serve"]
    assert arguments[arguments.index("--dataset-name") + 1] == "custom"
    assert arguments[arguments.index("--custom-output-len") + 1] == "1"
    assert (
        arguments[arguments.index("--extra-body") + 1] == '{"add_special_tokens":false}'
    )
    assert all(switch in arguments for switch in REQUIRED_SWITCHES)


def test_bench_wrapper_rejects_non_custom_dataset() -> None:
    with pytest.raises(ValueError, match="dataset-name"):
        build_vllm_arguments(
            [
                "--dataset-name",
                "sharegpt",
                "--dataset-path",
                "dataset.jsonl",
            ]
        )


def test_bench_wrapper_merges_extra_body_and_rejects_special_tokens() -> None:
    arguments = build_vllm_arguments(
        [
            "--dataset-path",
            "dataset.jsonl",
            "--extra-body",
            '{"priority":1}',
        ]
    )

    assert (
        arguments[arguments.index("--extra-body") + 1]
        == '{"priority":1,"add_special_tokens":false}'
    )
    with pytest.raises(ValueError, match="add_special_tokens"):
        build_vllm_arguments(
            [
                "--dataset-path",
                "dataset.jsonl",
                "--extra-body",
                '{"add_special_tokens":true}',
            ]
        )
