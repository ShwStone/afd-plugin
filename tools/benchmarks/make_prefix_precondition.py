# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Build prefix-cache precondition datasets + manifests for the Stage-2 prefix runs.

For each formal prefix dataset (cp8sp50k_token_ids_prefix{p}.jsonl) the
prefix groups (12 requests each) share a block-128-aligned token prefix. This
script extracts the per-group shared prefix and emits one precondition request
per group. Sending these requests to a warm server fills the prefix cache so the
formal run (which reuses the same group prefixes) hits cached blocks — the
"steady" cache state in §9.3. We send only the shared prefix, never the full
prompt, so no tautological full-workload warmup.

Runs locally (reads the existing datasets, no model config required).

Usage:
  python3 -m tools.benchmarks.make_prefix_precondition \
    --prefix 50 90 99
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = ROOT / "tools" / "datasets"
FORMAL_TEMPLATE = "cp8sp50k_token_ids_prefix{p}.jsonl"
OUTPUT_TEMPLATE = "prefix_precondition_prefix{p}.jsonl"
MANIFEST_SUFFIX = ".manifest.json"
PREFIX_BLOCK_SIZE = 128
PREFIX_GROUP_SIZE = 12
SCHEMA_VERSION = 1
HASH_CHUNK_BYTES = 1024 * 1024


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _request_ids_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("r", encoding="utf-8") as input_file:
        for line in input_file:
            digest.update(json.loads(line)["request_id"].encode() + b"\n")
    return digest.hexdigest()


def build_precondition(prefix_ratio: int, output_path: Path) -> dict[str, object]:
    formal_path = DATASET_DIR / FORMAL_TEMPLATE.format(p=prefix_ratio)
    requests_by_group: dict[int, list[dict[str, object]]] = {}
    with formal_path.open("r", encoding="utf-8") as formal_file:
        for line in formal_file:
            record = json.loads(line)
            group = int(record["prefix_group"])
            requests_by_group.setdefault(group, []).append(record)

    precondition_records: list[dict[str, object]] = []
    for group in sorted(requests_by_group):
        group_requests = requests_by_group[group]
        representative = max(
            group_requests,
            key=lambda record: int(record["shared_prefix_len"]),
        )
        shared_length = int(representative["shared_prefix_len"])
        if shared_length <= 0:
            raise ValueError(
                f"Group {group} has no shared prefix (prefix ratio {prefix_ratio})."
            )
        group_prefix = list(representative["prompt"][:shared_length])
        precondition_records.append(
            {
                "request_id": f"precond-{prefix_ratio}-{group:03d}",
                "prefix_group": group,
                "prompt": group_prefix,
                "prompt_len": len(group_prefix),
                "output_tokens": 1,
                "shared_prefix_len": len(group_prefix),
                "source_request_ids": [
                    str(record["request_id"]) for record in group_requests
                ],
            }
        )

    precondition_records.sort(key=lambda record: int(record["prefix_group"]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.name + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as output_file:
        for record in precondition_records:
            output_file.write(json.dumps(record, separators=(",", ":")) + "\n")
    temporary_path.replace(output_path)

    total_shared_prefix_tokens = sum(
        int(record["shared_prefix_len"]) for record in precondition_records
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset_path": str(output_path),
        "dataset_sha256": _sha256_file(output_path),
        "request_ids_sha256": _request_ids_sha256(output_path),
        "source_dataset": str(formal_path),
        "source_dataset_sha256": _sha256_file(formal_path),
        "prefix_ratio": prefix_ratio / 100.0,
        "prefix_block_size": PREFIX_BLOCK_SIZE,
        "prefix_group_size": PREFIX_GROUP_SIZE,
        "group_count": len(precondition_records),
        "request_count": len(precondition_records),
        "total_shared_prefix_tokens": total_shared_prefix_tokens,
        "send_order": "groups ascending, one request per group, sequential",
        # §9.3: cached/computed tokens are best-effort; this is a constructed
        # shared-prefix precondition (not a full-prompt warmup).
        "caveat": "constructed-prefix",
    }
    manifest_path = output_path.with_name(output_path.name + MANIFEST_SUFFIX)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", type=int, nargs="+", required=True)
    args = parser.parse_args(argv)

    manifests: dict[str, object] = {}
    for prefix_ratio in args.prefix:
        output_path = DATASET_DIR / OUTPUT_TEMPLATE.format(p=prefix_ratio)
        manifest = build_precondition(prefix_ratio, output_path)
        manifests[str(prefix_ratio)] = manifest
        print(
            f"Wrote {manifest['request_count']} precondition requests "
            f"({manifest['total_shared_prefix_tokens']} shared-prefix tokens) "
            f"to {output_path}"
        )
    print(json.dumps(manifests, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
