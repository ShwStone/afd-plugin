# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from __future__ import annotations

import json
from pathlib import Path

import tools.benchmarks.prefill_preflight as preflight_module
from afd_plugin.compat.patches.benchmark_serving import TARGET_VLLM_VERSION
from tools.benchmarks.prefill_dataset import generate_dataset


def test_preflight_validates_generated_artifacts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    csv_path = tmp_path / "lengths.csv"
    csv_path.write_text("4\n0\n6\n", encoding="utf-8")
    model_config_path = tmp_path / "config.json"
    model_config_path.write_text(
        json.dumps({"vocab_size": 32, "eos_token_id": 1}),
        encoding="utf-8",
    )
    dataset_path = tmp_path / "dataset.jsonl"
    generate_dataset(csv_path, model_config_path, dataset_path)
    experiment_config_path = tmp_path / "experiment.json"
    experiment_config_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model": "/models/reduced",
                "served_model_name": "reduced",
                "result_directory": str(tmp_path / "results"),
                "datasets": {"0": str(dataset_path)},
                "systems": {
                    "baseline": {
                        "base_url": "http://127.0.0.1:8000",
                        "endpoint": "/v1/completions",
                        "server_command_template": "launch {max_num_batched_tokens}",
                    }
                },
                "batch_tokens": [8192],
                "request_rates": [4],
                "repeats": 1,
                "num_prompts": 2,
                "num_warmups": 0,
                "ttft_slo_ms": 10000,
                "burstiness": 1,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        preflight_module,
        "_package_versions",
        lambda: {
            "vllm": TARGET_VLLM_VERSION,
            "vllm-ascend": "test",
            "vllm-afd-plugin": "test",
            "torch": "test",
            "torch-npu": "test",
        },
    )
    monkeypatch.setattr(
        preflight_module,
        "_run_capture",
        lambda command: {
            "command": list(command),
            "available": True,
            "returncode": 0,
            "output": "test",
        },
    )

    report, failures = preflight_module.collect_preflight_report(
        dataset_paths=[],
        experiment_config_path=experiment_config_path,
        model_config_path=model_config_path,
        require_npu=False,
    )

    assert failures == []
    assert report["datasets"][0]["index_exists"] is True


def test_dataset_report_identifies_git_lfs_pointer(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text(
        "version https://git-lfs.github.com/spec/v1\noid sha256:0123456789\nsize 100\n",
        encoding="utf-8",
    )

    report = preflight_module._dataset_report(dataset_path)

    assert report["is_git_lfs_pointer"] is True
