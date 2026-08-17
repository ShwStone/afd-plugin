#!/usr/bin/env python3
"""Backfill global_token_start/stop onto per-stage capture records.

`capture_attention_output` (and gate_topk / restored_ffn) historically wrote
`bounds=None`, so staged (token-split) records lack global_token_start/stop and
the offline compare maps every stage onto the same token range (garbage rel_l2).

The stage bounds are recoverable from the `layer_input` records of the same run
(which DO carry global_token_start/stop via `_stage_bounds`). This script copies
those bounds onto every staged record (attn_output / gate_topk / restored_ffn)
that is missing them, so the existing captures can be compared correctly
without re-running.

Usage::

    python3 tools/precision/fix_stage_bounds.py <capture-dir>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

STAGED_BOUNDARIES = {"attn_output", "gate_topk", "restored_ffn"}


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: fix_stage_bounds.py <capture-dir>", file=sys.stderr)
        return 2
    cap = Path(sys.argv[1])
    if not cap.is_dir():
        print(f"not a dir: {cap}", file=sys.stderr)
        return 2

    # stage_idx -> (global_token_start, global_token_stop, actual_tokens)
    # from layer_input records (which carry the real stage bounds).
    stage_bounds: dict[int, tuple[int, int, int]] = {}
    for path in cap.glob("attention_*layer_input.json"):
        with open(path, encoding="utf-8") as fh:
            rec = json.load(fh)
        start = rec.get("global_token_start")
        stop = rec.get("global_token_stop")
        stage = int(rec.get("stage_idx", 0))
        actual = rec.get("actual_tokens")
        if start is not None and stop is not None and actual is not None:
            stage_bounds.setdefault(stage, (int(start), int(stop), int(actual)))

    if not stage_bounds:
        print("warning: no layer_input stage bounds found; only fixing actual_tokens", file=sys.stderr)

    # Layers whose layer_input is a full-batch capture (actual >= 100) are
    # dense: their attn_output/gate_topk/restored_ffn must be treated as
    # full-batch too (no stage bounds), even if a capture bug tagged them with
    # stage bounds (the dense prefix wrongly inherits ubatch_idx=0).
    dense_layers: set[int] = set()
    for path in cap.glob("attention_*layer_input.json"):
        with open(path, encoding="utf-8") as fh:
            rec = json.load(fh)
        actual = rec.get("actual_tokens")
        layer = rec.get("layer_idx")
        if actual is not None and int(actual) >= 100 and layer is not None:
            dense_layers.add(int(layer))

    # Optional: real (unpadded) batch token count. Full-batch records that
    # fell back to _bounds_from_fc have actual_tokens == the TP-padded count
    # (e.g. 112 for 105 real); fix them so the compare excludes padding rows.
    real_tokens = None
    for arg in sys.argv[2:]:
        if arg.startswith("--real-tokens="):
            real_tokens = int(arg.split("=", 1)[1])
    fixed = reverted = actual_fixed = 0
    for path in cap.glob("attention_*.json"):
        with open(path, encoding="utf-8") as fh:
            rec = json.load(fh)
        changed = False
        # Fix full-batch actual_tokens (TP-padded) -> real count.
        if real_tokens is not None:
            actual = rec.get("actual_tokens")
            if actual is not None and int(actual) >= 100 and int(actual) != real_tokens:
                rec["actual_tokens"] = real_tokens
                inp = rec.get("input_tokens")
                if inp is not None:
                    rec["pad_size"] = int(inp) - real_tokens
                changed = True
                actual_fixed += 1
        if path.name.endswith("layer_input.json"):
            if changed:
                with open(path, "w", encoding="utf-8") as fh:
                    json.dump(rec, fh, sort_keys=True)
            continue
        boundary = next(
            (b for b in STAGED_BOUNDARIES if path.name.endswith(f"_{b}.json")),
            None,
        )
        if boundary is None:
            if changed:
                with open(path, "w", encoding="utf-8") as fh:
                    json.dump(rec, fh, sort_keys=True)
            continue
        stage = int(rec.get("stage_idx", 0))
        actual = rec.get("actual_tokens")
        layer = int(rec.get("layer_idx", -1))
        bounds_changed = False
        if layer in dense_layers:
            # dense full-batch capture: must have no stage bounds (even if the
            # capture wrongly inherited ubatch_idx=0), the real token count,
            # and the TP-padded input_tokens recovered from the tensor shard.
            if "global_token_start" in rec or "global_token_stop" in rec:
                rec.pop("global_token_start", None)
                rec.pop("global_token_stop", None)
                bounds_changed = True
            tp_size = int(rec.get("tp_size", 1) or 1)
            shape0 = None
            for tmeta in (rec.get("tensors") or {}).values():
                if isinstance(tmeta, dict) and tmeta.get("shape"):
                    shape0 = tmeta["shape"][0]
                    break
            if shape0 is not None and tp_size > 1:
                padded = int(shape0) * tp_size
                if rec.get("input_tokens") != padded:
                    rec["input_tokens"] = padded
                    bounds_changed = True
            if real_tokens is not None:
                a = rec.get("actual_tokens")
                if a is not None and int(a) != real_tokens:
                    rec["actual_tokens"] = real_tokens
                    inp = rec.get("input_tokens")
                    if inp is not None:
                        rec["pad_size"] = int(inp) - real_tokens
                    bounds_changed = True
                    actual_fixed += 1
            reverted += 1 if bounds_changed else 0
        elif stage in stage_bounds and actual is not None and int(actual) == stage_bounds[stage][2]:
            # genuine per-stage capture: (re)apply the stage's global range
            start, stop, _ = stage_bounds[stage]
            if rec.get("global_token_start") != start or rec.get("global_token_stop") != stop:
                rec["global_token_start"] = start
                rec["global_token_stop"] = stop
                bounds_changed = True
                fixed += 1
        else:
            # full-batch (final / unknown) capture: must have no stage bounds,
            # otherwise the compare caps it at the wrong range
            if "global_token_start" in rec or "global_token_stop" in rec:
                rec.pop("global_token_start", None)
                rec.pop("global_token_stop", None)
                bounds_changed = True
                reverted += 1
        if changed or bounds_changed:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(rec, fh, sort_keys=True)

    print(f"stage bounds: {stage_bounds}")
    print(f"actual_fixed={actual_fixed}, staged={fixed}, reverted={reverted} in {cap}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
