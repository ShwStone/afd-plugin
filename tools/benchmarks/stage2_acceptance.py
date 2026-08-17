# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Stage-2 acceptance gate (doc §13) with reject routing.

Discovers every Stage-2 config under ``bench_results/prefill_stage2``, checks
the collected runs against the §13 acceptance criteria, and reports per-group
pass/fail. With ``--move-rejects``, failing run files are moved to
``bench_results/prefill_stage2/rejected/`` so a re-run starts clean.

Criteria checked per run (L0):
  * verified artifact exists and has afd_verification;
  * issued_requests == num_prompts from the config;
  * successful + failed == issued (no silent drops);
  * repeat is one of 1..3;
  * prefix steady runs have a precondition dataset present.

Usage:
  python3 -m tools.benchmarks.stage2_acceptance \
    --stage2-dir bench_results/prefill_stage2 \
    --expected-repeats 3 \
    [--move-rejects]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path

from tools.benchmarks.prefill_experiment import load_experiment_config

ACCEPTANCE_SCHEMA_VERSION = 1


def discover_configs(stage2_dir: Path) -> list[Path]:
    """Find the canonical Stage-2 configs (same globs as the orchestrator)."""
    configs: list[Path] = []
    configs += sorted(
        stage2_dir.glob("02_e2e/*/*/stage2_e2e_*.json")
    )
    configs += sorted(
        stage2_dir.glob("03_ablation/*/*/stage2_ablation_*.json")
    )
    configs += sorted(
        stage2_dir.glob("04_prefix/*/*/*/stage2_prefix_*.json")
    )
    return configs


def _check_run(
    verified_path: Path,
    config,
    *,
    precondition_dirs: Sequence[Path],
) -> list[str]:
    """Return a list of failure reasons (empty means pass)."""
    failures: list[str] = []
    verified = json.loads(verified_path.read_text(encoding="utf-8"))
    verification = verified.get("afd_verification")
    if not isinstance(verification, dict):
        return ["missing afd_verification"]

    issued = verification.get("issued_requests")
    if issued != config.num_prompts:
        failures.append(
            f"issued_requests={issued} != num_prompts={config.num_prompts}"
        )
    successful = verification.get("successful_requests")
    failed = verification.get("failed_requests")
    if not all(isinstance(v, int) for v in (successful, failed)):
        failures.append("successful/failed not integers")
    elif successful + failed != issued:
        failures.append(
            f"successful({successful}) + failed({failed}) != issued({issued})"
        )
    repeat = verified.get("repeat")
    if not isinstance(repeat, int) or not 1 <= repeat <= 3:
        failures.append(f"invalid repeat={repeat}")

    stage2 = verified.get("stage2") or {}
    cache_state = stage2.get("cache_state")
    if cache_state == "steady" and not precondition_dirs:
        failures.append("steady run but no precondition dataset found")
    if cache_state not in ("cold", "steady"):
        failures.append(f"unknown cache_state={cache_state}")
    return failures


def run_acceptance(
    stage2_dir: Path,
    *,
    expected_repeats: int,
    move_rejects: bool,
) -> dict[str, object]:
    precondition_dirs = sorted(stage2_dir.glob("04_prefix/*/*/steady"))
    precondition_manifests = [
        path
        for directory in precondition_dirs
        for path in directory.glob("prefix_precondition_*.json")
    ]

    groups: list[dict[str, object]] = []
    total_runs = 0
    failed_runs = 0
    for config_path in discover_configs(stage2_dir):
        config = load_experiment_config(config_path)
        stage2 = json.loads(config_path.read_text(encoding="utf-8")).get(
            "stage2", {}
        )
        result_dir = Path(config.result_directory)
        verified_paths = sorted(result_dir.glob("*.verified.json"))
        repeats: set[int] = set()
        run_checks: list[dict[str, object]] = []
        for verified_path in verified_paths:
            failures = _check_run(verified_path, config, precondition_dirs=precondition_manifests)
            total_runs += 1
            if failures:
                failed_runs += 1
                if move_rejects:
                    _move_to_rejected(stage2_dir, verified_path)
            else:
                try:
                    repeats.add(int(json.loads(verified_path.read_text(encoding="utf-8")).get("repeat")))
                except (ValueError, json.JSONDecodeError):
                    repeats.add(-1)
            run_checks.append(
                {
                    "file": str(verified_path.relative_to(stage2_dir)),
                    "repeat": json.loads(verified_path.read_text(encoding="utf-8")).get("repeat"),
                    "failures": failures,
                }
            )
        complete = len(repeats) == expected_repeats and repeats == {
            repeat for repeat in range(1, expected_repeats + 1)
        }
        groups.append(
            {
                "config": str(config_path.relative_to(stage2_dir)),
                "system": next(iter(config.systems)),
                "case_id": stage2.get("case_id"),
                "purpose": stage2.get("purpose"),
                "variant": stage2.get("variant"),
                "cache_state": stage2.get("cache_state"),
                "result_directory": str(result_dir),
                "repeats_found": sorted(repeats),
                "complete": complete,
                "run_checks": run_checks,
            }
        )

    report = {
        "schema_version": ACCEPTANCE_SCHEMA_VERSION,
        "stage2_dir": str(stage2_dir),
        "expected_repeats": expected_repeats,
        "group_count": len(groups),
        "run_count": total_runs,
        "failed_run_count": failed_runs,
        "groups": groups,
    }
    return report


def _move_to_rejected(stage2_dir: Path, verified_path: Path) -> None:
    rejected = stage2_dir / "rejected" / verified_path.parent.relative_to(stage2_dir)
    rejected.mkdir(parents=True, exist_ok=True)
    for artifact in (verified_path, verified_path.with_suffix(".json")):
        if artifact.is_file():
            shutil.move(str(artifact), str(rejected / artifact.name))
    print(f"[reject] moved {verified_path.name} -> {rejected}")


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage2-dir", type=Path, required=True)
    parser.add_argument("--expected-repeats", type=int, default=3)
    parser.add_argument("--move-rejects", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    report = run_acceptance(
        args.stage2_dir,
        expected_repeats=args.expected_repeats,
        move_rejects=args.move_rejects,
    )
    output = args.output or args.stage2_dir / "06_reports" / "acceptance.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"{report['group_count']} groups, {report['run_count']} runs, "
        f"{report['failed_run_count']} failed"
    )
    for group in report["groups"]:
        status = "OK " if group["complete"] and not any(
            run["failures"] for run in group["run_checks"]
        ) else "FAIL"
        print(
            f"  [{status}] {group['case_id']} {group['system']} "
            f"variant={group['variant']} cache={group['cache_state']} "
            f"repeats={group['repeats_found']}"
        )
    return 1 if report["failed_run_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
