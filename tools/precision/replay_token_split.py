#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Replay the Async CAM token-split dispatch structure from capture records.

Reads the token-mode ``layer_input`` capture records and prints, per MoE layer,
the stage-0 / stage-1 token slices so the token-split dispatch process is
visible: which global token range went to which stage, with the actual / padded
token counts per stage.

Usage::

    python3 tools/precision/replay_token_split.py <capture-dir>
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

_BOUNDS_FIELDS = (
    "request_slice",
    "global_token_start",
    "global_token_stop",
    "actual_tokens",
    "input_tokens",
    "pad_size",
)


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: replay_token_split.py <capture-dir>", file=sys.stderr)
        return 2
    capture_dir = Path(sys.argv[1])
    if not capture_dir.is_dir():
        print(f"not a dir: {capture_dir}", file=sys.stderr)
        return 2

    # Group layer_input records by (layer_idx, stage_idx).
    stages: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for path in capture_dir.glob("attention_*layer_input.json"):
        with open(path, encoding="utf-8") as fh:
            rec = json.load(fh)
        if rec.get("boundary") != "layer_input":
            continue
        key = (int(rec.get("layer_idx", -1)), int(rec.get("stage_idx", 0)))
        stages[key].append(rec)

    if not stages:
        print("no layer_input records found", file=sys.stderr)
        return 1

    print(f"{'layer':>5} {'stage':>5} {'g_start':>8} {'g_stop':>8} "
          f"{'actual':>7} {'input':>7} {'pad':>4}  request_slice")
    print("-" * 80)
    for (layer, stage) in sorted(stages):
        recs = stages[(layer, stage)]
        rec = next((r for r in recs if r.get("tp_rank") == 0), recs[0])
        start = rec.get("global_token_start")
        stop = rec.get("global_token_stop")
        actual = rec.get("actual_tokens")
        inp = rec.get("input_tokens")
        pad = rec.get("pad_size")
        req_slice = rec.get("request_slice")
        print(f"{layer:>5} {stage:>5} {str(start):>8} {str(stop):>8} "
              f"{str(actual):>7} {str(inp):>7} {str(pad):>4}  {req_slice}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
