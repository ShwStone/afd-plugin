# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Generate a scaled arrival plan for one experiment point (plan 6.5 / 9.2).

Reads a frozen base arrival file (``<window>_arrivals.jsonl``) and a target
input-token rate, computes the uniform time dilation factor with Decimal
arithmetic, converts scaled offsets once to integer nanoseconds (nearest),
and writes one plan JSON. Both systems under test must read the same plan
file; clients never redo the float scaling.

Plan file content (canonical JSON, sorted keys, no extra spaces):
  schema_version, window, target_input_tokens_per_second (fixed-point
  string), dilation_factor (fixed-point string), base_arrivals_sha256,
  requests_sha256, base_duration_s, scaled_duration_s, target_rps,
  request_count, total_input_tokens, arrivals ([{request_id,
  sequence_index, scaled_arrival_offset_ns}]).

The plan file SHA-256 printed at the end is the scaled-plan checksum that
run manifests must record.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

NS_PER_MS = Decimal(1_000_000)


def _sha256_file(file_path: Path) -> str:
    digest = hashlib.sha256()
    with file_path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("tools/datasets/moonconv-wildchat-prefill"),
    )
    parser.add_argument("--window", required=True)
    parser.add_argument(
        "--target-tokens-per-second",
        required=True,
        help="target input token arrival rate, decimal string",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    workloads = args.dataset_dir / "workloads"
    arrivals_path = workloads / f"{args.window}_arrivals.jsonl"
    requests_path = workloads / f"{args.window}_requests.jsonl"

    base_arrivals = [
        json.loads(line)
        for line in arrivals_path.read_text(encoding="utf-8").splitlines()
    ]
    requests = [
        json.loads(line)
        for line in requests_path.read_text(encoding="utf-8").splitlines()
    ]
    if [a["request_id"] for a in base_arrivals] != [
        r["request_id"] for r in requests
    ]:
        raise ValueError(f"request/arrival order mismatch in {args.window}")

    total_tokens = sum(r["input_length"] for r in requests)
    base_duration_ms = max(a["base_arrival_offset_ms"] for a in base_arrivals)
    if base_duration_ms <= 0:
        raise ValueError("base arrival plan has zero duration")

    target_rate = Decimal(args.target_tokens_per_second)
    base_rate = Decimal(total_tokens) / Decimal(base_duration_ms) * 1000
    dilation = base_rate / target_rate

    scaled = []
    for arrival in base_arrivals:
        offset_ns = int(
            (
                Decimal(arrival["base_arrival_offset_ms"])
                * dilation
                * NS_PER_MS
            ).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
        )
        scaled.append(
            {
                "request_id": arrival["request_id"],
                "sequence_index": arrival["sequence_index"],
                "scaled_arrival_offset_ns": offset_ns,
            }
        )

    scaled_duration_s = Decimal(scaled[-1]["scaled_arrival_offset_ns"]) / Decimal(
        1_000_000_000
    )
    plan = {
        "schema_version": 1,
        "window": args.window,
        "target_input_tokens_per_second": str(target_rate),
        "dilation_factor": str(dilation),
        "base_arrivals_sha256": _sha256_file(arrivals_path),
        "requests_sha256": _sha256_file(requests_path),
        "request_count": len(scaled),
        "total_input_tokens": total_tokens,
        "base_duration_s": str(Decimal(base_duration_ms) / 1000),
        "scaled_duration_s": str(scaled_duration_s),
        "target_rps": str(
            Decimal(len(scaled)) / scaled_duration_s
        )
        if scaled_duration_s > 0
        else "0",
        "arrivals": scaled,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    body = (json.dumps(plan, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )
    args.output.write_bytes(body)
    print(f"wrote {args.output}")
    print(f"  dilation_factor={dilation}")
    print(f"  scaled_duration_s={scaled_duration_s}")
    print(f"  target_rps={plan['target_rps']}")
    print(f"  plan sha256={hashlib.sha256(body).hexdigest()}")


if __name__ == "__main__":
    main()
