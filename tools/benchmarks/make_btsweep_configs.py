# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Generate Stage-3 btsweep cell configs (L0 wide sweep + L2 profile replays).

Stage-3 deep-dives the Stage-1 finding that AFD's TTFT *improves* as
``max_num_batched_tokens`` grows at fixed RPS (analysis_all_angles.md §5 A4).
This generator writes one frozen config per (system, mbt, rps):

* ``01_sweep/<rps>/<mbt>/<system>/`` — L0 formal cells, 3 repeats, prefix=0
  cold, no instrumentation (profiler off).
* ``02_profile/<rps>/<mbt>/<system>/`` — L2 profiler replays, 1 repeat
  (torch_npu / npu-smi / AFD layout log enabled by the orchestrator).

It also writes ``00_plan/sweep_grid.json`` with the cell paths in server-reuse
order (group by system, then mbt, then rps) so ``run_stage2_l0.sh PHASE=btsweep``
can build the CELLS list and reuse servers for (system, mbt).

Usage:
  python3 -m tools.benchmarks.make_btsweep_configs [--mbs 4096 8192 ...]
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STAGE3 = ROOT / "bench_results" / "prefill_stage3"
STAGE3_REL = Path("bench_results") / "prefill_stage3"

MODEL = "/a3_inference/itask/workdir/tq02357756/shwstone/deepseek-v3.2-reduced"
SERVED_MODEL_NAME = "deepseek_v3_2"
BASE_URL = "http://127.0.0.1:8000"
ENDPOINT = "/v1/completions"
NUM_PROMPTS = 875
TTFT_SLO_MS = 10000.0
BURSTINESS = 1.0
ARRIVAL_SEED = 20260805
DATASET_DIR = "tools/datasets"
PREFIX0_DATASET = "cp8sp50k_token_ids.jsonl"
SCHEMA_VERSION = 1

SYSTEMS = {
    "baseline": {"name": "dp4_tp8_sp", "launch": "prefill_launch_baseline_dp4tp8.sh"},
    "afd": {"name": "afd_dp3_tp8_ep8", "launch": "prefill_launch_afd_attention.sh"},
}

# Stage-1 sweep used {8192,16384,32768,49152,65536}; Stage-3 extends both ends.
DEFAULT_MBS = [4096, 8192, 16384, 32768, 49152, 65536, 98304, 131072]
RPS_MAIN = 6.0
RPS_SUB = 8.0
SUB_MBS = {8192, 32768, 65536, 131072}  # rps8 subset (saturation boundary)

# L2 profiler replays: rps6 x {small, mid, large} x both systems, plus the
# rps8 saturation boundary for AFD.
PROFILE_CELLS = [
    (6.0, 8192, "baseline"),
    (6.0, 8192, "afd"),
    (6.0, 32768, "baseline"),
    (6.0, 32768, "afd"),
    (6.0, 131072, "baseline"),
    (6.0, 131072, "afd"),
    (8.0, 32768, "afd"),
]


def _server_template(system_key: str, mbt: int) -> str:
    launch = SYSTEMS[system_key]["launch"]
    if system_key == "baseline":
        return (
            f"MAX_NUM_BATCHED_TOKENS={mbt} DP_START_RANK=0 "
            f"bash tools/benchmarks/{launch}  # node1: DP_START_RANK=2"
        )
    return (
        f"MAX_NUM_BATCHED_TOKENS={mbt} DP_START_RANK=0 "
        f"bash tools/benchmarks/{launch}  "
        "# node1: DP_START_RANK=2 ATTN_DEVICES=0-7 attention + FFN EP8 devices 8-15"
    )


def _cell_config(
    *,
    system_key: str,
    result_directory: Path,
    mbt: int,
    rps: float,
    repeats: int,
) -> dict[str, object]:
    system_name = SYSTEMS[system_key]["name"]
    return {
        "schema_version": SCHEMA_VERSION,
        "model": MODEL,
        "served_model_name": SERVED_MODEL_NAME,
        "result_directory": str(result_directory),
        "datasets": {"0": f"{DATASET_DIR}/{PREFIX0_DATASET}"},
        "systems": {
            system_name: {
                "base_url": BASE_URL,
                "endpoint": ENDPOINT,
                "server_command_template": _server_template(system_key, mbt),
            }
        },
        "batch_tokens": [mbt],
        "request_rates": [rps],
        "repeats": repeats,
        "num_prompts": NUM_PROMPTS,
        "num_warmups": 32,
        "ttft_slo_ms": TTFT_SLO_MS,
        "burstiness": BURSTINESS,
        "stage2": {
            "case_id": f"btsweep-rps{rps:g}-mbt{mbt}",
            "purpose": "btsweep",
            "variant": None,
            "workload": "btsweep",
            "cache_state": "cold",
            "arrival_seed": ARRIVAL_SEED,
            "arrival_schedule": None,
        },
    }


def _write_config(path: Path, config: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def generate_all(mbs: Sequence[int]) -> dict[str, object]:
    rps_for_mbt: dict[int, list[float]] = {
        mbt: [RPS_MAIN] + ([RPS_SUB] if mbt in SUB_MBS else [])
        for mbt in mbs
    }

    # Order for server reuse: system -> mbt ascending -> rps (6 then 8).
    cells: list[str] = []
    for system_key, system in SYSTEMS.items():
        for mbt in mbs:
            for rps in rps_for_mbt[mbt]:
                result_dir_rel = (
                    STAGE3_REL / "01_sweep" / f"rps{rps:g}" / f"mbt{mbt}" / system["name"]
                )
                config = _cell_config(
                    system_key=system_key,
                    result_directory=result_dir_rel,
                    mbt=mbt,
                    rps=rps,
                    repeats=3,
                )
                config_path = (
                    STAGE3
                    / "01_sweep"
                    / f"rps{rps:g}"
                    / f"mbt{mbt}"
                    / system["name"]
                    / f"stage3_btsweep_rps{rps:g}_mbt{mbt}_{system['name']}.json"
                )
                _write_config(config_path, config)
                cells.append(str(config_path.relative_to(ROOT)))

    profile_cells: list[str] = []
    for rps, mbt, system_key in PROFILE_CELLS:
        system = SYSTEMS[system_key]
        result_dir_rel = (
            STAGE3_REL / "02_profile" / f"rps{rps:g}" / f"mbt{mbt}" / system["name"]
        )
        config = _cell_config(
            system_key=system_key,
            result_directory=result_dir_rel,
            mbt=mbt,
            rps=rps,
            repeats=1,
        )
        config_path = (
            STAGE3
            / "02_profile"
            / f"rps{rps:g}"
            / f"mbt{mbt}"
            / system["name"]
            / f"stage3_profile_rps{rps:g}_mbt{mbt}_{system['name']}.json"
        )
        _write_config(config_path, config)
        profile_cells.append(str(config_path.relative_to(ROOT)))

    grid: dict[str, object] = {
        "mbs": list(mbs),
        "rps_main": RPS_MAIN,
        "rps_sub": RPS_SUB,
        "sub_mbs": sorted(SUB_MBS),
        "cells": cells,
        "profile_cells": profile_cells,
        "note": "cells are in server-reuse order (system, mbt, rps).",
    }
    grid_path = STAGE3 / "00_plan" / "sweep_grid.json"
    grid_path.parent.mkdir(parents=True, exist_ok=True)
    grid_path.write_text(
        json.dumps(grid, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return grid


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mbs", type=int, nargs="*", default=DEFAULT_MBS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""
    args = _build_argument_parser().parse_args(argv)
    grid = generate_all(args.mbs)
    print(f"Wrote {len(grid['cells'])} L0 configs + "
          f"{len(grid['profile_cells'])} profile configs under {STAGE3}.")
    print(f"L0 grid: {grid['mbs']} mbt x rps6-full + rps8-subset, 2 systems, 3 repeats")
    print("sweep_grid.json written with server-reuse-ordered cells list.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
