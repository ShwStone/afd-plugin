# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Build the phase-zero data-acceptance request sets (plan 8.2).

Deterministic selections from the frozen bundle:

- ``accept_8_warmup.jsonl``: 8 warmup requests covering short/mid/long
  inputs (evenly spaced by length rank) for the token-ID service check.
- ``accept_long_singles.jsonl``: requests closest to 32K and 52K plus the
  bundle-longest 63,778-token request, for single-request long-input replay.
- ``accept_4x52k.jsonl``: 4 requests closest to 52K, one per baseline DP
  replica (``dp_rank`` 0..3), for the concurrent EP32 workspace check.

Selections may overlap between sets (acceptance only); mechanism batches and
capacity windows are unaffected because these files are separate artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _load(dataset_dir: Path, window: str) -> list[dict[str, object]]:
    path = dataset_dir / "workloads" / f"{window}_requests.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _emit(record: dict[str, object], dp_rank: int | None = None) -> dict[str, object]:
    out: dict[str, object] = {
        "request_id": record["request_id"],
        "input_length": record["input_length"],
        "prompt": record["prompt_token_ids"],
        "prompt_token_ids": record["prompt_token_ids"],
        "prompt_token_ids_sha256": record["prompt_token_ids_sha256"],
    }
    if dp_rank is not None:
        out["dp_rank"] = dp_rank
    return out


def _write(path: Path, records: list[dict[str, object]]) -> str:
    body = (
        "\n".join(
            json.dumps(r, sort_keys=True, separators=(",", ":")) for r in records
        )
        + "\n"
    ).encode("utf-8")
    path.write_bytes(body)
    return hashlib.sha256(body).hexdigest()


def _closest(pool: list[dict[str, object]], target: int, used: set[str]) -> dict[str, object]:
    candidates = sorted(
        (r for r in pool if r["request_id"] not in used),
        key=lambda r: (abs(r["input_length"] - target), r["request_id"]),
    )
    chosen = candidates[0]
    used.add(chosen["request_id"])
    return chosen


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("tools/datasets/moonconv-wildchat-prefill"),
    )
    args = parser.parse_args(argv)
    out_dir = args.dataset_dir / "workloads"

    warmup = _load(args.dataset_dir, "warmup")
    formal1 = _load(args.dataset_dir, "formal_1")
    screening = _load(args.dataset_dir, "screening")

    # 8 warmup requests evenly spaced by length rank (short -> long).
    by_length = sorted(warmup, key=lambda r: (r["input_length"], r["request_id"]))
    indices = [round(i * (len(by_length) - 1) / 7) for i in range(8)]
    warmup8 = [by_length[i] for i in indices]

    # Long singles: closest to 32K and 52K anywhere in the bundle, plus the
    # bundle-longest request (63,778 tokens, in formal_1).
    bundle = warmup + screening + formal1
    used: set[str] = set()
    long_32k = _closest(bundle, 32768, used)
    long_52k = _closest(bundle, 52000, used)
    longest = max(bundle, key=lambda r: r["input_length"])
    singles = [long_32k, long_52k, longest]

    # 4 x ~52K for the baseline EP32 concurrent check, one per DP rank.
    pool52 = sorted(
        bundle,
        key=lambda r: (abs(r["input_length"] - 52000), r["request_id"]),
    )
    four_52k = pool52[:4]

    outputs = {
        "accept_8_warmup.jsonl": [_emit(r) for r in warmup8],
        "accept_long_singles.jsonl": [_emit(r) for r in singles],
        "accept_4x52k.jsonl": [
            _emit(r, dp_rank=rank) for rank, r in enumerate(four_52k)
        ],
    }
    for name, records in outputs.items():
        digest = _write(out_dir / name, records)
        lengths = [r["input_length"] for r in records]
        print(f"{name}: ids={[r['request_id'] for r in records]}")
        print(f"  lengths={lengths} sha256={digest[:16]}...")


if __name__ == "__main__":
    main()
