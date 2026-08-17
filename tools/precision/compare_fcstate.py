#!/usr/bin/env python3
"""Compare the full forward-context state + metadata candidates between two
capture modes (no_ubatch vs token) to find the mode divergence source.

Reads the ``layer_input`` records of both runs and diffs:

- every ``attn_metadata`` candidate (now captured for ALL values: the
  ``AscendSFAMetadata`` objects, plus the ``indexer.k_cache`` entries whose
  value may be None / str / bytes),
- the ``forward_context_state`` block (slot_mapping dict, additional_kwargs,
  num_tokens / pad_size / ubatch_idx / num_ubatches / dbo_enabled / ...).

Only fields that DIFFER between the two modes are printed, so the first
mode-divergent input stands out immediately.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_layer_input(cap_dir: Path, mode: str, layer: int) -> dict:
    glob = cap_dir.glob(f"*/{mode}/attention_*_l{layer}_s0_layer_input.json")
    files = list(glob)
    if not files:
        # also accept a directly-passed run dir
        files = list(cap_dir.glob(f"*_{mode}_l{layer}_s0_layer_input.json"))
    if not files:
        raise FileNotFoundError(f"no layer_input for mode={mode} layer={layer} in {cap_dir}")
    # prefer the t2 (latest txn) record
    files.sort(key=lambda p: (p.name.count("t2"), str(p)))
    return json.load(open(files[-1], encoding="utf-8"))


def _walk(value, prefix=""):
    """Yield (path, value) leaves for any JSON-safe structure."""
    if isinstance(value, dict):
        for k, v in value.items():
            yield from _walk(v, f"{prefix}.{k}")
    elif isinstance(value, list):
        for i, v in enumerate(value):
            yield from _walk(v, f"{prefix}[{i}]")
    else:
        yield prefix, value


def _diff_dicts(a: dict, b: dict, label: str) -> list[str]:
    out = []
    paths = sorted(set(_walk(a)).union(p for p, _ in _walk(b)))
    a_leaves = dict(_walk(a))
    b_leaves = dict(_walk(b))
    for p in paths:
        va, vb = a_leaves.get(p, "<MISSING>"), b_leaves.get(p, "<MISSING>")
        if va != vb:
            out.append(f"  {label}{p}: no_ubatch={va!r} token={vb!r}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_noub", type=Path)
    ap.add_argument("run_token", type=Path)
    ap.add_argument("--layers", default="0,1,2")
    args = ap.parse_args()

    any_diff = False
    for layer in (int(x) for x in args.layers.split(",")):
        try:
            noub = _load_layer_input(args.run_noub, "no_ubatch", layer)
            tok = _load_layer_input(args.run_token, "token", layer)
        except FileNotFoundError as exc:
            print(f"layer {layer}: {exc}")
            continue
        print(f"==== layer {layer} layer_input ====")
        # --- attn_metadata candidates ---
        nm = noub.get("structural", {}).get("attn_metadata") or {}
        tm = tok.get("structural", {}).get("attn_metadata") or {}
        nkeys = set(nm.get("_metadata_keys") or [])
        tkeys = set(tm.get("_metadata_keys") or [])
        if nkeys != tkeys:
            any_diff = True
            print("  metadata_keys differ:")
            print("    only no_ubatch:", sorted(nkeys - tkeys))
            print("    only token:    ", sorted(tkeys - nkeys))
        ncand = nm.get("candidates") or {}
        tcand = tm.get("candidates") or {}
        for key in sorted(set(ncand) | set(tcand)):
            nc, tc = ncand.get(key), tcand.get(key)
            if nc is None and tc is None:
                continue
            diffs = _diff_dicts(nc or {}, tc or {}, f"  cand[{key.split(chr(46))[-2]}]")
            if diffs:
                any_diff = True
                print(f"  candidate {key}:")
                for d in diffs:
                    print(d)
        # --- forward_context_state ---
        nf = noub.get("structural", {}).get("forward_context_state") or {}
        tf = tok.get("structural", {}).get("forward_context_state") or {}
        for p in sorted(set(nf) | set(tf)):
            if nf.get(p) != tf.get(p):
                any_diff = True
                print(f"  fcstate.{p}: no_ubatch={nf.get(p)!r} token={tf.get(p)!r}")
        # --- record-level scalar fields ---
        for field in ("actual_tokens", "input_tokens", "num_tokens", "pad_size",
                      "global_token_start", "global_token_stop", "tp_rank", "tp_size"):
            if noub.get(field) != tok.get(field):
                any_diff = True
                print(f"  rec.{field}: no_ubatch={noub.get(field)!r} token={tok.get(field)!r}")
    print("RESULT:", "DIFFERENT" if any_diff else "ALL IDENTICAL")
    return 0 if any_diff else 1


if __name__ == "__main__":
    raise SystemExit(main())
