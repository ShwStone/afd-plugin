#!/usr/bin/env python3
"""FFN load imbalance analysis for one AFD transaction.

Reads correlation sidecars (afd-trace-<session>-ffn-rankN-*.jsonl[.tmp]) and
measures how evenly the routed tokens of a single transaction are spread
across FFN EP ranks, using `afd.ffn.compute` begin events' `num_tokens`
(the tokens actually routed to that rank for that layer).

Usage:
  python3 tools/benchmarks/analyze_ffn_balance.py --corr-dir /tmp/a2v2_sidecars \
      [--transaction-id afd-npu-2] [--json out.json]

Default transaction: the one with the most ffn.compute events.
Metrics: per-rank total tokens + share, max/mean, CV, and per-layer max/mean.
"""
import argparse
import glob
import json
import os
import re
import statistics
import sys

FNAME_RE = re.compile(r"afd-trace-(?P<session>[0-9a-f]+)-(?P<role>ffn)-rank(?P<rank>\d+)-")


def iter_compute_begins(corr_dir):
    """Yield (rank, transaction_id, layer_idx, stage_idx, num_tokens)."""
    for path in sorted(glob.glob(os.path.join(corr_dir, "afd-trace-*-ffn-rank*-*.jsonl*"))):
        m = FNAME_RE.search(os.path.basename(path))
        if not m:
            continue
        rank = int(m.group("rank"))
        with open(path) as f:
            for line in f:
                if '"afd.ffn.compute"' not in line or '"begin"' not in line:
                    continue
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue  # tolerate truncated tail of .tmp files
                if e.get("record_type") != "event" or e.get("event") != "afd.ffn.compute":
                    continue
                if e.get("phase") != "begin":
                    continue
                yield (rank, e.get("transaction_id"), e.get("layer_idx"),
                       e.get("stage_idx"), e.get("num_tokens") or 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corr-dir", required=True)
    ap.add_argument("--transaction-id", default=None,
                    help="default: transaction with most ffn.compute events")
    ap.add_argument("--json", default=None, help="optional JSON output path")
    args = ap.parse_args()

    # per transaction: per rank: per layer: summed tokens (over stages)
    tx_rank_layer = {}
    for rank, tx, layer, stage, n in iter_compute_begins(args.corr_dir):
        if tx is None:
            continue
        tx_rank_layer.setdefault(tx, {}).setdefault(rank, {}).setdefault(layer, 0)
        tx_rank_layer[tx][rank][layer] += n

    if not tx_rank_layer:
        sys.exit("no afd.ffn.compute events found in " + args.corr_dir)

    tx = args.transaction_id
    if tx is None:
        tx = max(tx_rank_layer,
                 key=lambda t: sum(sum(l.values()) for l in tx_rank_layer[t].values()))
    if tx not in tx_rank_layer:
        sys.exit(f"transaction {tx!r} not found; have: {sorted(tx_rank_layer)[:8]}...")

    rank_layer = tx_rank_layer[tx]
    n_ranks = max(rank_layer) + 1
    rank_total = {r: sum(layers.values()) for r, layers in rank_layer.items()}
    # ranks with zero events still count as zero load
    for r in range(n_ranks):
        rank_total.setdefault(r, 0)
    layers = sorted({l for layers in rank_layer.values() for l in layers})

    totals = [rank_total[r] for r in sorted(rank_total)]
    grand = sum(totals)
    mean = grand / len(totals) if totals else 0.0
    cv = statistics.pstdev(totals) / mean if mean else 0.0
    mx, mn = max(totals), min(totals)

    # per-layer imbalance across ranks
    per_layer = []
    for l in layers:
        row = [rank_layer.get(r, {}).get(l, 0) for r in range(n_ranks)]
        m = sum(row) / len(row)
        per_layer.append({
            "layer_idx": l,
            "total": sum(row),
            "max": max(row),
            "min": min(row),
            "max_over_mean": (max(row) / m) if m else 0.0,
        })

    print(f"transaction: {tx}")
    print(f"ffn ranks: {n_ranks}, layers seen: {len(layers)}, total routed tokens: {grand}")
    print()
    print(f"{'rank':>4} {'tokens':>10} {'share%':>7}")
    for r in sorted(rank_total):
        t = rank_total[r]
        print(f"{r:>4} {t:>10} {100*t/grand if grand else 0:>6.2f}")
    print()
    print(f"max/mean = {mx/mean if mean else 0:.3f}   min/mean = {mn/mean if mean else 0:.3f}"
          f"   CV = {cv:.3f}   ideal share = {100/n_ranks:.2f}%")
    worst = max(per_layer, key=lambda p: p["max_over_mean"])
    print(f"per-layer max/mean: median = "
          f"{statistics.median(p['max_over_mean'] for p in per_layer):.3f}, "
          f"worst layer {worst['layer_idx']}: {worst['max_over_mean']:.3f}")

    if args.json:
        out = {
            "transaction_id": tx,
            "n_ranks": n_ranks,
            "total_tokens": grand,
            "rank_tokens": {str(r): rank_total[r] for r in sorted(rank_total)},
            "max_over_mean": mx / mean if mean else 0.0,
            "min_over_mean": mn / mean if mean else 0.0,
            "cv": cv,
            "per_layer": per_layer,
        }
        with open(args.json, "w") as f:
            json.dump(out, f, indent=1)
        print("wrote", args.json)


if __name__ == "__main__":
    main()
