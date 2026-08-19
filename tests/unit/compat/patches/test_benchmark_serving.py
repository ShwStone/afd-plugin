# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from types import ModuleType, SimpleNamespace

import pytest

from afd_plugin.compat.patches.benchmark_serving import (
    custom_dataset_load_data,
    custom_dataset_sample,
)


@dataclass
class _FakeSampleRequest:
    prompt: list[int]
    prompt_len: int
    expected_output_len: int
    request_id: str


class _FakeCustomDataset:
    def __init__(self, data: list[dict[str, object]]) -> None:
        self.data = data
        self.num_available_samples = 0
        self.oversample_arguments: tuple[object, ...] | None = None

    def maybe_oversample_requests(self, *arguments: object) -> None:
        self.oversample_arguments = arguments


@pytest.fixture
def fake_vllm_dataset_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    vllm_module = ModuleType("vllm")
    benchmarks_module = ModuleType("vllm.benchmarks")
    datasets_module = ModuleType("vllm.benchmarks.datasets")
    datasets_module.SampleRequest = _FakeSampleRequest
    datasets_module.logger = SimpleNamespace(info=lambda *args: None)
    monkeypatch.setitem(sys.modules, "vllm", vllm_module)
    monkeypatch.setitem(sys.modules, "vllm.benchmarks", benchmarks_module)
    monkeypatch.setitem(sys.modules, "vllm.benchmarks.datasets", datasets_module)
    return datasets_module


def test_custom_dataset_patch_preserves_token_ids_and_true_length(
    fake_vllm_dataset_module: ModuleType,
) -> None:
    dataset = _FakeCustomDataset(
        [
            {
                "request_id": "cp8sp50k-000001",
                "prompt": [10, 11, 12],
                "prompt_len": 3,
                "output_tokens": 1,
            }
        ]
    )

    requests = custom_dataset_sample(
        dataset,
        tokenizer=None,
        num_requests=1,
        output_len=None,
        skip_chat_template=True,
        no_oversample=True,
    )

    assert requests == [
        _FakeSampleRequest(
            prompt=[10, 11, 12],
            prompt_len=3,
            expected_output_len=1,
            request_id="cp8sp50k-000001",
        )
    ]


def test_custom_dataset_loader_streams_jsonl_without_type_coercion(
    tmp_path,
) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text(
        json.dumps({"request_id": "request-1", "prompt": [10, 11]}) + "\n",
        encoding="utf-8",
    )
    dataset = SimpleNamespace(
        dataset_path=str(dataset_path),
        data=None,
        disable_shuffle=True,
        random_seed=1,
    )

    custom_dataset_load_data(dataset)

    assert dataset.data == [{"request_id": "request-1", "prompt": [10, 11]}]


def test_custom_dataset_patch_rejects_wrong_declared_length(
    fake_vllm_dataset_module: ModuleType,
) -> None:
    dataset = _FakeCustomDataset(
        [{"prompt": [10, 11], "prompt_len": 3, "output_tokens": 1}]
    )

    with pytest.raises(ValueError, match="prompt_len"):
        custom_dataset_sample(
            dataset,
            tokenizer=None,
            num_requests=1,
            skip_chat_template=True,
        )
