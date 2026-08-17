# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Continuous npu-smi AICore%/HBM sampler for Stage-3 occupancy attribution.

Runs ON the pod (npu-smi is node-local) and appends one CSV row per NPU per
interval: ``timestamp,npu,aicore_pct,hbm_used_mb,hbm_total_mb``. The Stage-3
orchestrator starts one instance on node0 and node1 around each measured run
and kills it afterwards; ``--duration 0`` (default) samples until killed.

Parsing handles the ``npu-smi info`` two-line-per-NPU block: the AICore(%)
and HBM-Usage(used/total) live on the second row (Chip/Phy-ID).

Usage:
  python3 -m tools.benchmarks.sample_npu_smi \
    --output telemetry/npu_smi_node0.csv --interval 1 [--duration 0]
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import time
from pathlib import Path

# Matches the utilization row of an npu-smi info block:
#   | <npu> <phy> | <bus-id> | <aicore> <mem_used>/<mem_total> <hbm_used>/<hbm_total> |
_ROW_PATTERN = re.compile(
    r"^\|\s*(\d+)\s+\d+\s+\|\s*[^|]+\|\s*(\d+)\s+"
    r"(\d+)\s*/\s*(\d+)\s+(\d+)\s*/\s*(\d+)"
)
# columns: npu, aicore, mem_used, mem_total, hbm_used, hbm_total


def _parse_npu_smi(output: str) -> list[tuple[int, int, int]]:
    """Return [(npu, aicore_pct, hbm_used_mb)] for each NPU row parsed."""
    rows: list[tuple[int, int, int]] = []
    for line in output.splitlines():
        match = _ROW_PATTERN.search(line)
        if not match:
            continue
        npu = int(match.group(1))
        aicore = int(match.group(2))
        hbm_used = int(match.group(5))
        rows.append((npu, aicore, hbm_used))
    return rows


def _sample_once(output_csv: csv.writer) -> None:
    try:
        result = subprocess.run(
            ["npu-smi", "info"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as error:
        print(f"[sample_npu_smi] npu-smi unavailable: {error}", flush=True)
        return
    timestamp = time.time()
    for npu, aicore, hbm_used in _parse_npu_smi(result.stdout or ""):
        output_csv.writerow([f"{timestamp:.3f}", npu, aicore, hbm_used])
    if not (result.stdout or ""):
        print("[sample_npu_smi] npu-smi returned no parseable rows", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--duration", type=float, default=0.0,
                        help="0 = sample until killed")
    args = parser.parse_args(argv)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + args.duration if args.duration > 0 else None
    with args.output.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["timestamp_s", "npu", "aicore_pct", "hbm_used_mb"])
        while True:
            _sample_once(writer)
            output_file.flush()
            if deadline is not None and time.monotonic() >= deadline:
                break
            time.sleep(args.interval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
