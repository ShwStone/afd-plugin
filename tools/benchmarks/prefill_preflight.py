# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Collect a reproducibility report and fail on missing benchmark prerequisites."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
from collections.abc import Sequence
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from afd_plugin.compat.patches.benchmark_serving import (
    TARGET_VLLM_VERSION,
    _base_vllm_version,
)
from tools.benchmarks.prefill_dataset import (
    DATASET_SCHEMA_VERSION,
    INDEX_SUFFIX,
    MANIFEST_SUFFIX,
    TOKENIZER_CONFIG_FILENAME,
)
from tools.benchmarks.prefill_experiment import (
    GIT_LFS_POINTER_PREFIX,
    load_experiment_config,
)

COMMAND_TIMEOUT_SECONDS = 30
HASH_READ_CHUNK_BYTES = 1024 * 1024
PREFLIGHT_SCHEMA_VERSION = 1
VERSION_PACKAGES = ("vllm", "vllm-ascend", "vllm-afd-plugin", "torch", "torch-npu")
NPU_COMMANDS = (
    ("npu-smi", "info"),
    # Some CANN builds ship an msprof without --version; --help is supported
    # by every build and still proves the tool exists.
    ("msprof", "--help"),
)


def _sha256_file(file_path: Path) -> str:
    digest = hashlib.sha256()
    with file_path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(HASH_READ_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_versions() -> dict[str, str | None]:
    package_versions: dict[str, str | None] = {}
    for package_name in VERSION_PACKAGES:
        try:
            package_versions[package_name] = version(package_name)
        except PackageNotFoundError:
            package_versions[package_name] = None
    return package_versions


def _run_capture(command: Sequence[str]) -> dict[str, object]:
    executable = shutil.which(command[0])
    if executable is None:
        return {
            "command": list(command),
            "available": False,
            "returncode": None,
            "output": "",
        }
    completed_process = subprocess.run(
        [executable, *command[1:]],
        check=False,
        capture_output=True,
        text=True,
        timeout=COMMAND_TIMEOUT_SECONDS,
    )
    output = (completed_process.stdout + completed_process.stderr).strip()
    return {
        "command": list(command),
        "available": True,
        "returncode": completed_process.returncode,
        "output": output,
    }


def _dataset_report(dataset_path: Path) -> dict[str, object]:
    report: dict[str, object] = {
        "path": str(dataset_path),
        "exists": dataset_path.is_file(),
    }
    if not dataset_path.is_file():
        return report
    with dataset_path.open("rb") as dataset_file:
        first_line = dataset_file.readline().decode(errors="replace").strip()
    report["is_git_lfs_pointer"] = first_line == GIT_LFS_POINTER_PREFIX
    report["size_bytes"] = dataset_path.stat().st_size
    if first_line != GIT_LFS_POINTER_PREFIX:
        report["sha256"] = _sha256_file(dataset_path)

    manifest_path = dataset_path.with_name(dataset_path.name + MANIFEST_SUFFIX)
    report["manifest_path"] = str(manifest_path)
    report["manifest_exists"] = manifest_path.is_file()
    if manifest_path.is_file():
        report["manifest"] = json.loads(manifest_path.read_text(encoding="utf-8"))
    index_path = dataset_path.with_name(dataset_path.name + INDEX_SUFFIX)
    report["index_path"] = str(index_path)
    report["index_exists"] = index_path.is_file()
    if index_path.is_file():
        report["index_sha256"] = _sha256_file(index_path)
    return report


def collect_preflight_report(
    *,
    dataset_paths: Sequence[Path],
    experiment_config_path: Path | None,
    model_config_path: Path | None,
    require_npu: bool,
) -> tuple[dict[str, object], list[str]]:
    """Collect environment evidence and return hard validation failures."""
    package_versions = _package_versions()
    failures: list[str] = []
    if _base_vllm_version(package_versions["vllm"]) != TARGET_VLLM_VERSION:
        failures.append(
            f"vLLM {TARGET_VLLM_VERSION} is required; "
            f"found {package_versions['vllm']!r}."
        )

    experiment_config_report: dict[str, object] | None = None
    configured_dataset_paths: tuple[Path, ...] = ()
    if experiment_config_path is not None:
        experiment_config = load_experiment_config(experiment_config_path)
        configured_dataset_paths = tuple(experiment_config.datasets.values())
        experiment_config_report = {
            "path": str(experiment_config_path),
            "sha256": _sha256_file(experiment_config_path),
            "systems": sorted(experiment_config.systems),
            "batch_tokens": list(experiment_config.batch_tokens),
            "request_rates": list(experiment_config.request_rates),
            "repeats": experiment_config.repeats,
            "num_prompts": experiment_config.num_prompts,
            "ttft_slo_ms": experiment_config.ttft_slo_ms,
        }

    effective_dataset_paths = tuple(
        dict.fromkeys((*dataset_paths, *configured_dataset_paths))
    )
    datasets = [
        _dataset_report(dataset_path) for dataset_path in effective_dataset_paths
    ]
    for dataset in datasets:
        dataset_path = dataset["path"]
        if not dataset["exists"]:
            failures.append(f"Dataset does not exist: {dataset_path}")
        elif dataset.get("is_git_lfs_pointer"):
            failures.append(f"Dataset is a Git LFS pointer: {dataset_path}")
        elif not dataset.get("manifest_exists"):
            failures.append(f"Dataset manifest is missing: {dataset_path}")
        elif not dataset.get("index_exists"):
            failures.append(f"Dataset index is missing: {dataset_path}")
        else:
            manifest = dataset.get("manifest")
            if not isinstance(manifest, dict):
                failures.append(f"Dataset manifest is invalid: {dataset_path}")
            elif manifest.get("schema_version") != DATASET_SCHEMA_VERSION:
                failures.append(
                    f"Dataset manifest schema is unsupported: {dataset_path}"
                )
            elif manifest.get("dataset_sha256") != dataset.get("sha256"):
                failures.append(f"Dataset hash does not match manifest: {dataset_path}")
            elif manifest.get("index_sha256") != dataset.get("index_sha256"):
                failures.append(f"Dataset index hash mismatch: {dataset_path}")
            elif (
                experiment_config_report is not None
                and manifest.get("request_count")
                != experiment_config_report["num_prompts"]
            ):
                failures.append(
                    f"Dataset request count does not match experiment: {dataset_path}"
                )

    model_config_report: dict[str, object] | None = None
    if model_config_path is not None:
        resolved_config_path = (
            model_config_path / "config.json"
            if model_config_path.is_dir()
            else model_config_path
        )
        model_config_report = {
            "path": str(resolved_config_path),
            "exists": resolved_config_path.is_file(),
        }
        if resolved_config_path.is_file():
            model_config_report["sha256"] = _sha256_file(resolved_config_path)
            tokenizer_config_path = (
                resolved_config_path.parent / TOKENIZER_CONFIG_FILENAME
            )
            model_config_report["tokenizer_config_path"] = str(tokenizer_config_path)
            model_config_report["tokenizer_config_exists"] = (
                tokenizer_config_path.is_file()
            )
            model_config_report["tokenizer_config_sha256"] = (
                _sha256_file(tokenizer_config_path)
                if tokenizer_config_path.is_file()
                else None
            )
        else:
            failures.append(f"Model config does not exist: {resolved_config_path}")
        for dataset in datasets:
            manifest = dataset.get("manifest")
            if isinstance(manifest, dict) and manifest.get(
                "model_config_sha256"
            ) != model_config_report.get("sha256"):
                failures.append(
                    "Dataset model config hash does not match the supplied model "
                    f"config: {dataset['path']}"
                )
            if isinstance(manifest, dict) and manifest.get(
                "tokenizer_config_sha256"
            ) != model_config_report.get("tokenizer_config_sha256"):
                failures.append(
                    "Dataset tokenizer config hash does not match the supplied "
                    f"model directory: {dataset['path']}"
                )

    npu_commands = [_run_capture(command) for command in NPU_COMMANDS]
    if require_npu:
        for command_report in npu_commands:
            if not command_report["available"]:
                failures.append(
                    f"Required NPU command is unavailable: "
                    f"{command_report['command'][0]}"
                )
            elif command_report["returncode"] != 0:
                failures.append(f"NPU command failed: {command_report['command'][0]}")

    git_revision = _run_capture(("git", "rev-parse", "HEAD"))
    git_status = _run_capture(("git", "status", "--short"))
    report: dict[str, object] = {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
        },
        "packages": package_versions,
        "git": {
            "revision": git_revision,
            "status": git_status,
        },
        "datasets": datasets,
        "model_config": model_config_report,
        "experiment_config": experiment_config_report,
        "npu_commands": npu_commands,
        "require_npu": require_npu,
        "failures": failures,
    }
    return report, failures


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, action="append", default=[])
    parser.add_argument("--experiment-config", type=Path)
    parser.add_argument("--model-config", type=Path)
    parser.add_argument("--require-npu", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""
    args = _build_argument_parser().parse_args(argv)
    report, failures = collect_preflight_report(
        dataset_paths=args.dataset,
        experiment_config_path=args.experiment_config,
        model_config_path=args.model_config,
        require_npu=args.require_npu,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
