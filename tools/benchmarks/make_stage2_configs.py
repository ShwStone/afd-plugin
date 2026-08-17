# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Regenerate Stage-2 per-cell configs and fill the selection hashes.

Reads ``bench_results/prefill_stage2/00_selection/selected_cells.json`` and
writes one frozen config per (case, system) and (ablation variant) and
(prefix cache-state). Fixes two bugs from the hand-built configs:

* ``03_ablation/<workload>/<variant>/`` result_directory (was pointing into
  ``02_e2e/real-knee-...``).
* ``04_prefix/<case>/<system>/<cache-state>/`` result_directory so cold and
  steady runs for the same prefix cell do not collide (doc §5.3).

Also fills ``dataset_sha256`` / ``request_ids_sha256`` in selected_cells.json.

Usage:
  python3 -m tools.benchmarks.make_stage2_configs
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STAGE2 = ROOT / "bench_results" / "prefill_stage2"
# result_directory is written into configs as a REPO-RELATIVE path so the
# container (which runs prefill_experiment from $REPO) writes results under
# $REPO/bench_results/... rather than an absolute local path.
STAGE2_REL = Path("bench_results") / "prefill_stage2"
SELECTION = STAGE2 / "00_selection" / "selected_cells.json"

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
LONG_SHORT_DATASET = "cp8sp50k_longshort_token_ids.jsonl"
SCHEMA_VERSION = 1
HASH_CHUNK_BYTES = 1024 * 1024

E2E_PURPOSES = {"low", "knee", "high", "regression-guard", "regression-healthy"}

SYSTEMS = {
    "baseline": {
        "name": "dp4_tp8_sp",
        "launch": "prefill_launch_baseline_dp4tp8.sh",
    },
    "afd": {
        "name": "afd_dp3_tp8_ep8",
        "launch": "prefill_launch_afd_attention.sh",
    },
}

# Stage-2 ablation variants applied via STAGE2_VARIANT on both AFD launch
# scripts (A0=ubatch off, A1=split request, A2=split token).
ABLATION_VARIANTS = {
    "real-knee": ("A0", "A1", "A2"),
    "long-short": ("A1", "A2"),
}

VARIANT_CONFIG = {
    "A0": {"ubatch": False, "split": "request"},
    "A1": {"ubatch": True, "split": "request"},
    "A2": {"ubatch": True, "split": "token"},
}

PREFIX_ANCHOR = {
    "prefix-50-mbt32768-rps10": ("0.5", "cp8sp50k_token_ids_prefix50.jsonl"),
    "prefix-90-mbt32768-rps10": ("0.9", "cp8sp50k_token_ids_prefix90.jsonl"),
    "prefix-99-mbt32768-rps10": ("0.99", "cp8sp50k_token_ids_prefix99.jsonl"),
}


@dataclass(frozen=True)
class Cell:
    """One selected cell from the Stage-1 sweep."""

    case_id: str
    purpose: str
    batch_tokens: int
    rps: float
    prefix_ratio: str

    @classmethod
    def from_json(cls, raw: dict[str, object]) -> "Cell":
        return cls(
            case_id=str(raw["case_id"]),
            purpose=str(raw["purpose"]),
            batch_tokens=int(raw["batch_tokens"]),
            rps=float(raw["rps"]),
            prefix_ratio=str(raw["prefix_ratio"]),
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _request_ids_sha256(dataset_path: Path) -> str:
    digest = hashlib.sha256()
    with dataset_path.open("r", encoding="utf-8") as dataset_file:
        for line in dataset_file:
            record = json.loads(line)
            digest.update(str(record["request_id"]).encode() + b"\n")
    return digest.hexdigest()


def _server_template(system_key: str, cell: Cell, variant: str | None = None) -> str:
    launch = SYSTEMS[system_key]["launch"]
    if system_key == "baseline":
        return (
            f"MAX_NUM_BATCHED_TOKENS={cell.batch_tokens} DP_START_RANK=0 "
            f"bash tools/benchmarks/{launch}  # node1: DP_START_RANK=2"
        )
    variant_prefix = f"STAGE2_VARIANT={variant} " if variant else ""
    return (
        f"{variant_prefix}MAX_NUM_BATCHED_TOKENS={cell.batch_tokens} DP_START_RANK=0 "
        f"bash tools/benchmarks/{launch}  "
        "# node1: DP_START_RANK=2 ATTN_DEVICES=0-7 attention + FFN EP8 devices 8-15"
    )


def _base_config(
    cell: Cell,
    system_key: str,
    *,
    result_directory: Path,
    dataset_key: str,
    dataset_path: str,
    num_warmups: int,
    cache_state: str,
    variant: str | None = None,
    workload: str | None = None,
    variant_config: dict[str, object] | None = None,
) -> dict[str, object]:
    stage2: dict[str, object] = {
        "case_id": cell.case_id,
        "purpose": cell.purpose,
        "variant": variant,
        "workload": workload,
        "cache_state": cache_state,
        "arrival_seed": ARRIVAL_SEED,
        # §5.2: client cannot yet export/replay an arrival schedule, so we
        # degrade to same dataset + --disable-shuffle + same seed. The
        # manifest records this "same-seed" caveat.
        "arrival_schedule": None,
    }
    if variant_config is not None:
        stage2["variant_config"] = variant_config
    return {
        "schema_version": SCHEMA_VERSION,
        "model": MODEL,
        "served_model_name": SERVED_MODEL_NAME,
        "result_directory": str(result_directory),
        "datasets": {dataset_key: dataset_path},
        "systems": {
            SYSTEMS[system_key]["name"]: {
                "base_url": BASE_URL,
                "endpoint": ENDPOINT,
                "server_command_template": _server_template(
                    system_key, cell, variant
                ),
            }
        },
        "batch_tokens": [cell.batch_tokens],
        "request_rates": [cell.rps],
        "repeats": 3,
        "num_prompts": NUM_PROMPTS,
        "num_warmups": num_warmups,
        "ttft_slo_ms": TTFT_SLO_MS,
        "burstiness": BURSTINESS,
        "stage2": stage2,
    }


def _write_config(path: Path, config: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _dataset_for(cell: Cell) -> tuple[str, str]:
    """Return (datasets-key, relative-path) for a cell's dataset."""
    if cell.prefix_ratio == "0":
        return "0", f"{DATASET_DIR}/{PREFIX0_DATASET}"
    key, filename = PREFIX_ANCHOR[cell.case_id]
    return key, f"{DATASET_DIR}/{filename}"


def _long_short_cell() -> Cell:
    return Cell(
        case_id="long-short-mbt32768-rps10",
        purpose="ablation",
        batch_tokens=32768,
        rps=10.0,
        prefix_ratio="0",
    )


def generate_all() -> list[Path]:
    selection = json.loads(SELECTION.read_text(encoding="utf-8"))
    cells = [Cell.from_json(raw) for raw in selection["cells"]]
    generated: list[Path] = []

    for cell in cells:
        dataset_key, dataset_path = _dataset_for(cell)
        for system_key in SYSTEMS:
            system_name = SYSTEMS[system_key]["name"]
            if cell.purpose in E2E_PURPOSES:
                result_dir_rel = STAGE2_REL / "02_e2e" / cell.case_id / system_name
                config = _base_config(
                    cell,
                    system_key,
                    result_directory=result_dir_rel,
                    dataset_key=dataset_key,
                    dataset_path=dataset_path,
                    num_warmups=32 if cell.prefix_ratio == "0" else 0,
                    cache_state="cold",
                )
                config_path = (
                    STAGE2 / "02_e2e" / cell.case_id / system_name
                    / f"stage2_e2e_{cell.case_id}_{system_name}.json"
                )
                _write_config(config_path, config)
                generated.append(config_path)
            else:  # prefix purpose
                for cache_state in ("cold", "steady"):
                    result_dir_rel = (
                        STAGE2_REL
                        / "04_prefix"
                        / cell.case_id
                        / system_name
                        / cache_state
                    )
                    config = _base_config(
                        cell,
                        system_key,
                        result_directory=result_dir_rel,
                        dataset_key=dataset_key,
                        dataset_path=dataset_path,
                        num_warmups=0,
                        cache_state=cache_state,
                    )
                    config_path = (
                        STAGE2
                        / "04_prefix"
                        / cell.case_id
                        / system_name
                        / cache_state
                        / f"stage2_prefix_{cell.case_id}_{cache_state}_{system_name}.json"
                    )
                    _write_config(config_path, config)
                    generated.append(config_path)

    # Ablation configs (AFD only).
    for workload, variants in ABLATION_VARIANTS.items():
        cell = (
            _long_short_cell()
            if workload == "long-short"
            else next(c for c in cells if c.case_id == "real-knee-p0-mbt32768-rps10")
        )
        dataset_key, dataset_path = (
            _dataset_for(cell)
            if workload == "real-knee"
            else ("0", f"{DATASET_DIR}/{LONG_SHORT_DATASET}")
        )
        for variant in variants:
            result_dir_rel = STAGE2_REL / "03_ablation" / workload / variant
            config = _base_config(
                cell,
                "afd",
                result_directory=result_dir_rel,
                dataset_key=dataset_key,
                dataset_path=dataset_path,
                num_warmups=32,
                cache_state="cold",
                variant=variant,
                workload=workload,
                variant_config=VARIANT_CONFIG[variant],
            )
            config_path = (
                STAGE2 / "03_ablation" / workload / variant
                / f"stage2_ablation_{workload}_{variant}.json"
            )
            _write_config(config_path, config)
            generated.append(config_path)

    # Fill selection hashes.
    reason_by_case = {
        str(raw["case_id"]): str(raw["selection_reason"])
        for raw in selection["cells"]
    }
    selection["cells"] = []
    for cell in cells:
        dataset_key, dataset_path = _dataset_for(cell)
        dataset_file = ROOT / dataset_path
        selection["cells"].append(
            {
                "case_id": cell.case_id,
                "purpose": cell.purpose,
                "batch_tokens": cell.batch_tokens,
                "rps": cell.rps,
                "prefix_ratio": cell.prefix_ratio,
                "dataset_sha256": _sha256_file(dataset_file),
                "request_ids_sha256": _request_ids_sha256(dataset_file),
                "selection_reason": reason_by_case[cell.case_id],
            }
        )
    SELECTION.write_text(
        json.dumps(selection, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return generated


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""
    generated = generate_all()
    print(f"Wrote {len(generated)} configs under {STAGE2}.")
    print("Filled dataset_sha256/request_ids_sha256 in selected_cells.json.")
    for path in sorted(generated):
        print(f"  {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
