# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Select the frozen mechanism batches from the screening window (plan 6.3).

Selection rule (deterministic): for each target length, sort candidate
requests by absolute length difference, break ties by ``request_id``, and
take the first N. Every request is used at most once across all batches, in
the fixed processing order: 8K balanced, 32K balanced, 32K long-short.

Output: ``mechanism_batches.json`` with one entry per batch (name, target,
request IDs, per-request lengths, total tokens) plus the SHA-256 of each
per-batch JSONL request file that is also written for the burst client.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

BATCH_SPECS = [
    ("fixed_8k_balanced", [(1024, 8)]),
    ("fixed_32k_balanced", [(4096, 8)]),
    ("fixed_32k_long_short", [(24576, 1), (1170, 7)]),
]


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(file_path: Path) -> str:
    digest = hashlib.sha256()
    with file_path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_requests(dataset_dir: Path, window: str) -> list[dict[str, object]]:
    requests: list[dict[str, object]] = []
    path = dataset_dir / "workloads" / f"{window}_requests.jsonl"
    with path.open(encoding="utf-8") as dataset_file:
        for line in dataset_file:
            record = json.loads(line)
            if record["input_length"] != len(record["prompt_token_ids"]):
                raise ValueError(
                    f"length mismatch in {record['request_id']}: "
                    f"{record['input_length']} != "
                    f"{len(record['prompt_token_ids'])}"
                )
            requests.append(record)
    return requests


def select_batch(
    candidates: list[dict[str, object]],
    used: set[str],
    parts: list[tuple[int, int]],
) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    for target_length, count in parts:
        pool = [r for r in candidates if r["request_id"] not in used]
        pool.sort(
            key=lambda r: (
                abs(r["input_length"] - target_length),
                r["request_id"],
            )
        )
        if len(pool) < count:
            raise ValueError(
                f"not enough unused candidates for target {target_length}: "
                f"need {count}, have {len(pool)}"
            )
        for record in pool[:count]:
            used.add(record["request_id"])
            selected.append(record)
    return selected


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("tools/datasets/moonconv-wildchat-prefill"),
    )
    parser.add_argument("--window", default="screening")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tools/datasets/moonconv-wildchat-prefill/workloads"),
    )
    args = parser.parse_args(argv)

    candidates = load_requests(args.dataset_dir, args.window)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    used: set[str] = set()
    manifest: dict[str, object] = {
        "schema_version": 1,
        "source_window": args.window,
        "selection_rule": (
            "sort by abs(input_length - target), tie-break by request_id; "
            "each request used at most once; batch order: "
            "fixed_8k_balanced, fixed_32k_balanced, fixed_32k_long_short"
        ),
        "batches": [],
    }
    for batch_name, parts in BATCH_SPECS:
        records = select_batch(candidates, used, parts)
        lines = []
        for record in records:
            lines.append(
                json.dumps(
                    {
                        "request_id": record["request_id"],
                        "input_length": record["input_length"],
                        "prompt": record["prompt_token_ids"],
                        "prompt_token_ids_sha256": record[
                            "prompt_token_ids_sha256"
                        ],
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
        body = ("\n".join(lines) + "\n").encode("utf-8")
        batch_path = args.output_dir / f"{batch_name}.jsonl"
        batch_path.write_bytes(body)
        manifest["batches"].append(
            {
                "name": batch_name,
                "parts": [
                    {"target_length": t, "count": c} for t, c in parts
                ],
                "request_ids": [r["request_id"] for r in records],
                "input_lengths": [r["input_length"] for r in records],
                "total_tokens": sum(r["input_length"] for r in records),
                "file": batch_path.name,
                "sha256": _sha256_bytes(body),
            }
        )

    manifest_path = args.output_dir / "mechanism_batches.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {manifest_path}")
    for batch in manifest["batches"]:
        print(
            f"  {batch['name']}: {len(batch['request_ids'])} requests, "
            f"{batch['total_tokens']} tokens, sha256={batch['sha256'][:16]}..."
        )
    print(f"manifest sha256={_sha256_file(manifest_path)}")


if __name__ == "__main__":
    main()
