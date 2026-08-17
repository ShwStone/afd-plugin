#!/usr/bin/env python3
"""Compare attn_output divergence by token position (front vs back half).

Loads the layer-0 attn_output captures from two modes (no_ubatch / token),
maps every global token to its vector via the same SP alignment as
compare_async_moe_captures, then reports divergence stats split into token
windows so we can see whether the first or last tokens diverge more.

Usage::

    python3 tools/precision/analyze_token_position.py \\
        <no_ubatch_dir> <token_dir> [--window 50]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from tools.benchmarks.compare_async_moe_captures import (  # noqa: E402
    _global_token_vectors,
    _load_records,
)


def _vecs_by_token(mode_dir: Path, layer: int, boundary: str, tensor: str) -> dict[int, torch.Tensor]:
    """Global token -> row vector for a given layer/boundary/tensor."""
    out: dict[int, torch.Tensor] = {}
    for path, record in _load_records(mode_dir):
        if int(record.get("layer_idx", -1)) != layer:
            continue
        if record.get("boundary") != boundary:
            continue
        mapping = _global_token_vectors(mode_dir, record, tensor)
        for gtoken, vec in mapping.items():
            out[gtoken] = vec
    return out


def _rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = a.norm() + b.norm()
    if denom == 0:
        return 0.0
    return float((a - b).norm() / denom)


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_a", type=Path, help="no_ubatch capture dir")
    ap.add_argument("run_b", type=Path, help="token capture dir")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--boundary", default="attn_output")
    ap.add_argument("--tensor", default="attn_output")
    ap.add_argument("--window", type=int, default=50)
    args = ap.parse_args()

    a = _vecs_by_token(args.run_a, args.layer, args.boundary, args.tensor)
    b = _vecs_by_token(args.run_b, args.layer, args.boundary, args.tensor)
    if not a or not b:
        print("no vectors found; check layer/boundary/tensor names")
        return 2
    tokens = sorted(set(a) & set(b))
    if not tokens:
        print("no overlapping tokens")
        return 2
    n = len(tokens)
    print(f"layer={args.layer} boundary={args.boundary} tensor={args.tensor} "
          f"tokens={n} (min={tokens[0]} max={tokens[-1]})")
    # windows: front, back, middle (if enough tokens), full
    front = tokens[: args.window]
    back = tokens[-args.window:]
    mid = tokens[n // 4 : 3 * n // 4] if n >= 8 else []
    for label, sel in (("front", front), ("back", back), ("middle", mid), ("FULL", tokens)):
        if not sel:
            continue
        rel = [_rel_l2(a[t], b[t]) for t in sel]
        mx = [_max_abs(a[t], b[t]) for t in sel]
        mean_rel = sum(rel) / len(rel)
        mean_abs = sum(mx) / len(mx)
        max_abs = max(mx)
        print(f"  {label:6s} n={len(sel):4d} rel_l2_mean={mean_rel:8.4f} "
              f"max_abs_mean={mean_abs:8.5f} max_abs_max={max_abs:8.5f}")
    # per-token divergence for a quick histogram head
    print("  per-token rel_l2 (first 12):")
    for t in tokens[:12]:
        print(f"    token {t:3d} rel_l2={_rel_l2(a[t], b[t]):8.4f}")
    print("  per-token rel_l2 (last 12):")
    for t in tokens[-12:]:
        print(f"    token {t:3d} rel_l2={_rel_l2(a[t], b[t]):8.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
