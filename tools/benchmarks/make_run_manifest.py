# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Write a per-run `run_manifest.json` for a Stage-2 result (doc §12).

Assembles the manifest schema from the experiment config, the raw/verified
result artifacts, and the archived server logs. Every artifact is hashed;
the manifest never relies on directory names for provenance.

Usage:
  python3 -m tools.benchmarks.make_run_manifest \
    --config <stage2_config.json> \
    --result <...repeat-N.json> \
    --verified <...repeat-N.verified.json> \
    --collection-level L0 \
    --run-id <stable-id> \
    [--log-dir bench_results/logs] \
    [--extra-artifact <path>:<type>]...
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections.abc import Sequence
from pathlib import Path

from tools.benchmarks.prefill_experiment import load_experiment_config

MANIFEST_SCHEMA_VERSION = 1
HASH_CHUNK_BYTES = 1024 * 1024


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head(repo: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return completed.stdout.strip() or None


def _dataset_hashes(dataset_path: Path) -> tuple[str, str]:
    """Return (dataset_sha256, request_ids_sha256) for a JSONL dataset."""
    dataset_sha = _sha256_file(dataset_path)
    request_digest = hashlib.sha256()
    with dataset_path.open("r", encoding="utf-8") as dataset_file:
        for line in dataset_file:
            request_digest.update(json.loads(line)["request_id"].encode() + b"\n")
    return dataset_sha, request_digest.hexdigest()


def _artifact(
    path: Path,
    artifact_type: str,
    *,
    repo: Path,
) -> dict[str, object]:
    return {
        "path": str(path.relative_to(repo)),
        "type": artifact_type,
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def build_manifest(
    *,
    config_path: Path,
    result_path: Path,
    verified_path: Path,
    collection_level: str,
    run_id: str,
    log_dir: Path,
    extra_artifacts: Sequence[tuple[str, str]],
    repo: Path,
) -> dict[str, object]:
    config = load_experiment_config(config_path)
    stage2 = {
        key: value
        for key, value in json.loads(config_path.read_text(encoding="utf-8")).get(
            "stage2", {}
        ).items()
    }
    dataset_path = next(iter(config.datasets.values()))
    dataset_sha, request_ids_sha = _dataset_hashes(repo / dataset_path)

    purpose = str(stage2.get("purpose") or "e2e")
    cache_state = str(stage2.get("cache_state") or "off")
    variant = stage2.get("variant")
    system_name = next(iter(config.systems))
    system_label = "afd" if "afd" in system_name else "baseline"

    artifacts: list[dict[str, object]] = []
    if result_path.is_file():
        artifacts.append(
            _artifact(result_path, "result-raw", repo=repo)
        )
    if verified_path.is_file():
        artifacts.append(
            _artifact(verified_path, "result-verified", repo=repo)
        )
    for raw_spec in extra_artifacts:
        raw_path, artifact_type = raw_spec.split(":", 1)
        artifact_path = Path(raw_path)
        if artifact_path.is_file():
            artifacts.append(_artifact(artifact_path, artifact_type, repo=repo))
    if log_dir.is_dir():
        for log_file in sorted(log_dir.glob("*.log")):
            artifacts.append(_artifact(log_file, "server-log", repo=repo))

    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "run_id": run_id,
        "collection_level": collection_level,
        "purpose": purpose,
        "system": system_name,
        "variant": variant if variant is not None else system_label,
        "software": {
            "afd_commit": _git_head(repo),
            "vllm": "0.19.1",
            "vllm_ascend": None,
            "container_digest": None,
        },
        "workload": {
            "dataset_sha256": dataset_sha,
            "request_ids_sha256": request_ids_sha,
            # §5.2: no exported arrival schedule yet; degraded to same-seed.
            "arrival_schedule_sha256": None,
            "arrival_caveat": "same-seed (--disable-shuffle + dataset order)",
            "num_prompts": config.num_prompts,
            "batch_tokens": config.batch_tokens[0],
            "rps": config.request_rates[0],
            "prefix_ratio_requested": str(next(iter(config.datasets))),
            "cache_state": cache_state,
        },
        "profile": {
            "enabled": False,
            "tool": None,
            "active_window_sha256": None,
        },
        "artifacts": artifacts,
    }
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--verified", type=Path, required=True)
    parser.add_argument("--collection-level", default="L0")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--log-dir", type=Path, default=Path("bench_results/logs"))
    parser.add_argument("--extra-artifact", action="append", default=[])
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)

    manifest = build_manifest(
        config_path=args.config,
        result_path=args.result,
        verified_path=args.verified,
        collection_level=args.collection_level,
        run_id=args.run_id,
        log_dir=args.log_dir,
        extra_artifacts=tuple(args.extra_artifact),
        repo=args.repo,
    )
    # The result directory is flat (repeat encoded in the filename), so the
    # manifest is named per run to avoid three repeats overwriting each other.
    output_path = args.verified.parent / f"run_manifest_{args.result.stem}.json"
    output_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
