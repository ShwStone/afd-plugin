# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Generate the long-short mixed dataset for the Stage-2 A1/A2 split ablation.

The long-short workload stresses the async-MoE ubatch split modes: a bimodal
mix of short (64-1024 token) and long (8192-49152 token) prompts. The token IDs
are generated deterministically by :mod:`prefill_dataset` from the model config,
so this script must run inside the pod container where the model lives.

Usage (on NODE0):
  python3 -m tools.benchmarks.make_long_short_dataset \
    --model-config /a3_inference/itask/workdir/tq02357756/shwstone/deepseek-v3.2-reduced \
    --output tools/datasets/cp8sp50k_longshort_token_ids.jsonl \
    --seed 20260805
"""

from __future__ import annotations

import argparse
import csv
import json
import tempfile
from collections.abc import Sequence
from pathlib import Path

from tools.benchmarks.prefill_dataset import generate_dataset

REQUEST_COUNT = 875
SEED = 20260805
SHORT_RANGE = (64, 1024)
LONG_RANGE = (8192, 49152)
DEFAULT_MODEL_CONFIG = (
    "/a3_inference/itask/workdir/tq02357756/shwstone/deepseek-v3.2-reduced"
)
DEFAULT_OUTPUT = "tools/datasets/cp8sp50k_longshort_token_ids.jsonl"


def build_lengths(seed: int = SEED) -> list[int]:
    """Deterministic bimodal prompt lengths: half short, half long."""
    import random

    generator = random.Random(seed)
    short_count = REQUEST_COUNT // 2
    lengths = [
        generator.randint(*SHORT_RANGE) for _ in range(short_count)
    ] + [
        generator.randint(*LONG_RANGE) for _ in range(REQUEST_COUNT - short_count)
    ]
    # Interleave so short/long requests are evenly spread through the run.
    lengths.sort()
    interleaved: list[int] = []
    mid = len(lengths) // 2
    left, right = lengths[:mid], lengths[mid:]
    for index in range(mid):
        interleaved.append(left[index])
        interleaved.append(right[index])
    if len(right) > mid:  # odd count
        interleaved.append(right[-1])
    return interleaved


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", type=Path, default=Path(DEFAULT_MODEL_CONFIG))
    parser.add_argument("--output", type=Path, default=Path(DEFAULT_OUTPUT))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--source-csv", type=Path)
    args = parser.parse_args(argv)

    lengths = build_lengths(args.seed)
    csv_path = args.source_csv
    if csv_path is None:
        temporary_csv = Path(tempfile.mkstemp(suffix=".csv")[1])
        with temporary_csv.open("w", newline="", encoding="utf-8") as csv_file:
            csv.writer(csv_file).writerows((length,) for length in lengths)
        csv_path = temporary_csv

    manifest = generate_dataset(
        csv_path,
        args.model_config,
        args.output,
        random_seed=args.seed,
        prefix_ratio=0.0,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
