# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Plan or run the AFD versus DP+TP+SP prefill benchmark matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from tools.benchmarks.prefill_dataset import (
    DATASET_SCHEMA_VERSION,
    INDEX_SUFFIX,
    MANIFEST_SUFFIX,
)
from tools.benchmarks.prefill_results import verify_and_enrich_result

DEFAULT_CONFIG_PATH = Path("tools/benchmarks/prefill_experiment.example.json")
GIT_LFS_POINTER_PREFIX = "version https://git-lfs.github.com/spec/v1"
EXPERIMENT_SCHEMA_VERSION = 1
HASH_READ_CHUNK_BYTES = 1024 * 1024
SERVER_READY_TIMEOUT_SECONDS = 600.0
SERVER_READY_POLL_SECONDS = 2.0
HTTP_RESPONSE_TIMEOUT_SECONDS = 10.0
OPENAI_API_KEY_ENV = "OPENAI_API_KEY"


@dataclass(frozen=True)
class SystemConfig:
    """One benchmarked serving layout."""

    name: str
    base_url: str
    endpoint: str
    server_command_template: str


@dataclass(frozen=True)
class ExperimentConfig:
    """Validated experiment configuration."""

    model: str
    served_model_name: str
    result_directory: Path
    datasets: dict[str, Path]
    systems: dict[str, SystemConfig]
    batch_tokens: tuple[int, ...]
    request_rates: tuple[float, ...]
    repeats: int
    num_prompts: int
    num_warmups: int
    ttft_slo_ms: float
    burstiness: float
    arrival_seed_base: int = 20260817
    arrival_plan_directory: Path | None = None


@dataclass(frozen=True)
class BenchmarkRun:
    """One cell and repeat in the experiment matrix."""

    system: SystemConfig
    batch_tokens: int
    request_rate: float
    repeat: int
    prefix_ratio_key: str
    dataset_path: Path
    result_path: Path
    verified_result_path: Path


def _require_string(mapping: dict[str, object], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Configuration field {key!r} must be a non-empty string.")
    return value


def _require_integer(mapping: dict[str, object], key: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Configuration field {key!r} must be an integer.")
    return value


def _require_number(mapping: dict[str, object], key: str) -> float:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Configuration field {key!r} must be numeric.")
    return float(value)


def _require_number_tuple(
    mapping: dict[str, object],
    key: str,
    *,
    integer: bool,
) -> tuple[int, ...] | tuple[float, ...]:
    values = mapping.get(key)
    if not isinstance(values, list) or not values:
        raise ValueError(f"Configuration field {key!r} must be a non-empty list.")
    if integer:
        if any(
            isinstance(value, bool) or not isinstance(value, int) for value in values
        ):
            raise ValueError(f"Configuration field {key!r} must contain integers.")
        return tuple(values)
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        for value in values
    ):
        raise ValueError(f"Configuration field {key!r} must contain numbers.")
    return tuple(float(value) for value in values)


def load_experiment_config(config_path: Path) -> ExperimentConfig:
    """Load and validate an experiment JSON file."""
    raw_config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw_config, dict):
        raise ValueError(f"{config_path} must contain a JSON object.")
    if raw_config.get("schema_version") != EXPERIMENT_SCHEMA_VERSION:
        raise ValueError(
            f"{config_path} must use schema_version {EXPERIMENT_SCHEMA_VERSION}."
        )

    raw_datasets = raw_config.get("datasets")
    if not isinstance(raw_datasets, dict) or not raw_datasets:
        raise ValueError("Configuration field 'datasets' must be an object.")
    datasets: dict[str, Path] = {}
    for ratio_key, dataset_path in raw_datasets.items():
        if not isinstance(ratio_key, str) or not isinstance(dataset_path, str):
            raise ValueError("Dataset ratio keys and paths must be strings.")
        datasets[ratio_key] = Path(dataset_path)

    raw_systems = raw_config.get("systems")
    if not isinstance(raw_systems, dict) or not raw_systems:
        raise ValueError("Configuration field 'systems' must be an object.")
    systems: dict[str, SystemConfig] = {}
    for system_name, raw_system in raw_systems.items():
        if not isinstance(system_name, str) or not isinstance(raw_system, dict):
            raise ValueError("System names must map to configuration objects.")
        systems[system_name] = SystemConfig(
            name=system_name,
            base_url=_require_string(raw_system, "base_url"),
            endpoint=_require_string(raw_system, "endpoint"),
            server_command_template=_require_string(
                raw_system,
                "server_command_template",
            ),
        )

    batch_tokens = _require_number_tuple(
        raw_config,
        "batch_tokens",
        integer=True,
    )
    request_rates = _require_number_tuple(
        raw_config,
        "request_rates",
        integer=False,
    )
    repeats = _require_integer(raw_config, "repeats")
    num_prompts = _require_integer(raw_config, "num_prompts")
    num_warmups = _require_integer(raw_config, "num_warmups")
    ttft_slo_ms = _require_number(raw_config, "ttft_slo_ms")
    burstiness = _require_number(raw_config, "burstiness")
    if repeats <= 0 or num_prompts <= 0 or num_warmups < 0:
        raise ValueError(
            "repeats/num_prompts must be positive; num_warmups non-negative."
        )
    if ttft_slo_ms <= 0 or burstiness <= 0:
        raise ValueError("ttft_slo_ms and burstiness must be positive.")
    if any(value <= 0 for value in batch_tokens):
        raise ValueError("All batch token limits must be positive.")
    if any(value <= 0 for value in request_rates):
        raise ValueError("All request rates must be positive.")

    arrival_seed_base = raw_config.get("arrival_seed_base", 20260817)
    if isinstance(arrival_seed_base, bool) or not isinstance(arrival_seed_base, int):
        raise ValueError("Configuration field 'arrival_seed_base' must be an integer.")
    raw_plan_directory = raw_config.get("arrival_plan_directory")
    arrival_plan_directory: Path | None = None
    if raw_plan_directory is not None:
        if not isinstance(raw_plan_directory, str) or not raw_plan_directory:
            raise ValueError(
                "Configuration field 'arrival_plan_directory' must be a string."
            )
        arrival_plan_directory = Path(raw_plan_directory)

    return ExperimentConfig(
        model=_require_string(raw_config, "model"),
        served_model_name=_require_string(raw_config, "served_model_name"),
        result_directory=Path(_require_string(raw_config, "result_directory")),
        datasets=datasets,
        systems=systems,
        batch_tokens=batch_tokens,
        request_rates=request_rates,
        repeats=repeats,
        num_prompts=num_prompts,
        num_warmups=num_warmups,
        ttft_slo_ms=ttft_slo_ms,
        burstiness=burstiness,
        arrival_seed_base=arrival_seed_base,
        arrival_plan_directory=arrival_plan_directory,
    )


def _rate_filename_component(request_rate: float) -> str:
    return str(request_rate).replace(".", "p")


def arrival_seed(config: ExperimentConfig, repeat: int) -> int:
    """Deterministic per-repeat Poisson seed (plan section 6.3).

    Repeat N uses the same seed for every system, so traditional and AFD
    deployments replay the identical arrival plan for a paired comparison.
    """
    return config.arrival_seed_base + repeat - 1


def arrival_plan_filename(
    config: ExperimentConfig,
    request_rate: float,
    repeat: int,
) -> str:
    """File name shared with tools.benchmarks.make_arrival_plan outputs."""
    seed = arrival_seed(config, repeat)
    return (
        f"arrival_seed{seed}_rps{_rate_filename_component(request_rate)}"
        f"_n{config.num_prompts}.json"
    )


def build_runs(
    config: ExperimentConfig,
    *,
    system_names: Sequence[str] | None = None,
    batch_tokens: Sequence[int] | None = None,
    prefix_ratio_key: str = "0",
) -> list[BenchmarkRun]:
    """Expand selected configuration dimensions into concrete run records."""
    if prefix_ratio_key not in config.datasets:
        raise ValueError(
            f"Prefix ratio {prefix_ratio_key!r} is not configured in datasets."
        )
    selected_system_names = (
        tuple(system_names) if system_names else tuple(config.systems)
    )
    unknown_systems = set(selected_system_names) - set(config.systems)
    if unknown_systems:
        raise ValueError(f"Unknown systems: {sorted(unknown_systems)}")
    selected_batch_tokens = tuple(batch_tokens) if batch_tokens else config.batch_tokens
    unknown_batch_tokens = set(selected_batch_tokens) - set(config.batch_tokens)
    if unknown_batch_tokens:
        raise ValueError(
            f"Unconfigured batch token limits: {sorted(unknown_batch_tokens)}"
        )

    runs: list[BenchmarkRun] = []
    for system_name in selected_system_names:
        system = config.systems[system_name]
        for maximum_batch_tokens in selected_batch_tokens:
            for request_rate in config.request_rates:
                for repeat in range(1, config.repeats + 1):
                    stem = (
                        f"{system_name}-mbt{maximum_batch_tokens}"
                        f"-rps{_rate_filename_component(request_rate)}"
                        f"-prefix{prefix_ratio_key.replace('.', 'p')}"
                        f"-repeat{repeat}"
                    )
                    runs.append(
                        BenchmarkRun(
                            system=system,
                            batch_tokens=maximum_batch_tokens,
                            request_rate=request_rate,
                            repeat=repeat,
                            prefix_ratio_key=prefix_ratio_key,
                            dataset_path=config.datasets[prefix_ratio_key],
                            result_path=config.result_directory / f"{stem}.json",
                            verified_result_path=(
                                config.result_directory / f"{stem}.verified.json"
                            ),
                        )
                    )
    return runs


def build_benchmark_command(
    config: ExperimentConfig,
    run: BenchmarkRun,
    *,
    plot_timeline: bool = False,
) -> list[str]:
    """Build the exact patched vLLM bench serve command for one run."""
    built_in_warmups = config.num_warmups if run.prefix_ratio_key == "0" else 0
    seed = arrival_seed(config, run.repeat)
    command = [
        sys.executable,
        "-m",
        "tools.benchmarks.prefill_bench",
        "--backend",
        "vllm",
        "--base-url",
        run.system.base_url,
        "--endpoint",
        run.system.endpoint,
        "--model",
        config.model,
        "--served-model-name",
        config.served_model_name,
        "--dataset-name",
        "custom",
        "--dataset-path",
        str(run.dataset_path),
        "--num-prompts",
        str(config.num_prompts),
        "--custom-output-len",
        "1",
        "--request-rate",
        str(run.request_rate),
        "--burstiness",
        str(config.burstiness),
        "--seed",
        str(seed),
        "--num-warmups",
        str(built_in_warmups),
        "--goodput",
        f"ttft:{config.ttft_slo_ms}",
        "--percentile-metrics",
        "ttft,e2el",
        "--metric-percentiles",
        "50,90,95,99",
        "--ignore-eos",
        "--temperature",
        "0",
        "--save-result",
        "--result-filename",
        str(run.result_path),
        "--metadata",
        f"afd_system={run.system.name}",
        f"max_num_batched_tokens={run.batch_tokens}",
        f"repeat={run.repeat}",
        f"prefix_ratio={run.prefix_ratio_key}",
        f"arrival_seed={seed}",
    ]
    if plot_timeline:
        command.append("--plot-timeline")
    return command


def _sha256_file(file_path: Path) -> str:
    digest = hashlib.sha256()
    with file_path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(HASH_READ_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_dataset_artifact(
    dataset_path: Path,
    expected_request_count: int,
) -> None:
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset does not exist: {dataset_path}")
    with dataset_path.open("rb") as dataset_file:
        first_line = dataset_file.readline().decode(errors="replace").strip()
    if first_line == GIT_LFS_POINTER_PREFIX:
        raise RuntimeError(
            f"{dataset_path} is a Git LFS pointer. Fetch or regenerate the dataset."
        )
    manifest_path = dataset_path.with_name(dataset_path.name + MANIFEST_SUFFIX)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Dataset manifest does not exist: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"{manifest_path} must contain a JSON object.")
    if manifest.get("schema_version") != DATASET_SCHEMA_VERSION:
        raise ValueError(f"Dataset manifest schema is unsupported: {manifest_path}")
    if manifest.get("dataset_sha256") != _sha256_file(dataset_path):
        raise ValueError(f"Dataset hash does not match manifest: {dataset_path}")
    index_path = dataset_path.with_name(dataset_path.name + INDEX_SUFFIX)
    if not index_path.is_file():
        raise FileNotFoundError(f"Dataset index does not exist: {index_path}")
    if manifest.get("index_sha256") != _sha256_file(index_path):
        raise ValueError(f"Dataset index hash does not match manifest: {index_path}")
    if manifest.get("request_count") != expected_request_count:
        raise ValueError(
            f"Dataset request count does not match num_prompts: {dataset_path}"
        )


def format_server_command(run: BenchmarkRun) -> str:
    """Render the operator-managed server command for a run group."""
    try:
        return run.system.server_command_template.format(
            max_num_batched_tokens=run.batch_tokens,
            system=run.system.name,
        )
    except KeyError as error:
        raise ValueError(
            f"Unknown server command template placeholder: {error.args[0]}"
        ) from error


def wait_for_server(
    system: SystemConfig,
    served_model_name: str,
    *,
    timeout_seconds: float = SERVER_READY_TIMEOUT_SECONDS,
) -> None:
    """Wait for the model-list endpoint and verify the expected served model."""
    models_url = system.base_url.rstrip("/") + "/v1/models"
    api_key = os.environ.get(OPENAI_API_KEY_ENV)
    request_headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    models_request = urllib.request.Request(
        models_url,
        headers=request_headers,
    )
    deadline = time.monotonic() + timeout_seconds
    last_error = "no response"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(
                models_request,
                timeout=HTTP_RESPONSE_TIMEOUT_SECONDS,
            ) as response:
                payload = json.load(response)
            if not isinstance(payload, dict) or not isinstance(
                payload.get("data"),
                list,
            ):
                last_error = "invalid /v1/models response"
            else:
                model_ids = {
                    model["id"]
                    for model in payload["data"]
                    if isinstance(model, dict) and isinstance(model.get("id"), str)
                }
                if served_model_name in model_ids:
                    return
                last_error = (
                    f"served model {served_model_name!r} not found in "
                    f"{sorted(model_ids)}"
                )
        except (
            OSError,
            urllib.error.URLError,
            json.JSONDecodeError,
        ) as error:
            last_error = str(error)
        time.sleep(SERVER_READY_POLL_SECONDS)
    raise TimeoutError(
        f"Server {models_url} was not ready within {timeout_seconds}s: {last_error}"
    )


def run_benchmarks(
    config: ExperimentConfig,
    runs: Sequence[BenchmarkRun],
    *,
    resume: bool,
    plot_timeline: bool,
) -> None:
    """Execute selected client runs against an operator-configured server."""
    for dataset_path in {run.dataset_path for run in runs}:
        _validate_dataset_artifact(dataset_path, config.num_prompts)
    if not runs:
        return
    wait_for_server(runs[0].system, config.served_model_name)
    for run in runs:
        run.result_path.parent.mkdir(parents=True, exist_ok=True)
        if resume and run.verified_result_path.is_file():
            print(f"Skipping verified result: {run.verified_result_path}")
            continue
        if resume and run.result_path.is_file():
            resume_plan_path: Path | None = None
            if config.arrival_plan_directory is not None:
                candidate = config.arrival_plan_directory / arrival_plan_filename(
                    config,
                    run.request_rate,
                    run.repeat,
                )
                if candidate.is_file():
                    resume_plan_path = candidate
            verify_and_enrich_result(
                run.result_path,
                run.dataset_path,
                run.verified_result_path,
                ttft_slo_ms=config.ttft_slo_ms,
                arrival_plan_path=resume_plan_path,
            )
            continue

        command = build_benchmark_command(
            config,
            run,
            plot_timeline=plot_timeline,
        )
        print(shlex.join(command), flush=True)
        subprocess.run(command, check=True)
        arrival_plan_path: Path | None = None
        if config.arrival_plan_directory is not None:
            candidate = config.arrival_plan_directory / arrival_plan_filename(
                config,
                run.request_rate,
                run.repeat,
            )
            if candidate.is_file():
                arrival_plan_path = candidate
        verify_and_enrich_result(
            run.result_path,
            run.dataset_path,
            run.verified_result_path,
            ttft_slo_ms=config.ttft_slo_ms,
            arrival_plan_path=arrival_plan_path,
        )


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command_name in ("plan", "run"):
        command_parser = subparsers.add_parser(command_name)
        command_parser.add_argument("--system", action="append")
        command_parser.add_argument("--batch-tokens", type=int, action="append")
        command_parser.add_argument("--prefix-ratio", default="0")
        command_parser.add_argument("--plot-timeline", action="store_true")
        if command_name == "run":
            command_parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""
    args = _build_argument_parser().parse_args(argv)
    config = load_experiment_config(args.config)
    runs = build_runs(
        config,
        system_names=args.system,
        batch_tokens=args.batch_tokens,
        prefix_ratio_key=args.prefix_ratio,
    )
    if args.command == "plan":
        previous_group: tuple[str, int] | None = None
        for run in runs:
            run_group = (run.system.name, run.batch_tokens)
            if run_group != previous_group:
                print(
                    "Start/restart server:",
                    format_server_command(run),
                )
                previous_group = run_group
            print(
                "Run client:",
                shlex.join(
                    build_benchmark_command(
                        config,
                        run,
                        plot_timeline=args.plot_timeline,
                    )
                ),
            )
    else:
        selected_groups = {(run.system.name, run.batch_tokens) for run in runs}
        if len(selected_groups) != 1:
            raise ValueError(
                "The run command requires exactly one --system and one "
                "--batch-tokens value so it cannot target a misconfigured server."
            )
        run_benchmarks(
            config,
            runs,
            resume=args.resume,
            plot_timeline=args.plot_timeline,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
