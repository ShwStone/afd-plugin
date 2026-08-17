#!/usr/bin/env python3
"""Export per-cell SLO attainment at multiple thresholds + TTFT stats.

Reads verified.json (has per-request ttfts + successes), computes:
  - TTFT mean / p99 (already in afd_verification)
  - SLO attainment at thresholds [2, 5, 10, 20] seconds
Writes compact CSV for local plotting.
"""
import json
import sys
from pathlib import Path

RESULT_DIR = Path("bench_results/prefill")
SLO_THRESHOLDS = (2.0, 5.0, 10.0, 20.0)
OUT = sys.argv[1] if len(sys.argv) > 1 else "bench_results/prefill/slo_summary.csv"


def parse(f: Path):
    d = json.loads(f.read_text())
    v = d.get("afd_verification", {})
    stem = f.stem.replace(".verified", "")
    parts = stem.split("-")
    system = parts[0]
    bt = int(parts[1].replace("mbt", ""))
    rps = float(parts[2].replace("rps", "").replace("p0", ""))
    pr_raw = parts[3].replace("prefix", "")
    pr = 0.0 if pr_raw == "0" else float(pr_raw.replace("p", "."))
    ttfts = d.get("ttfts", [])
    successes = d.get("successes", [])
    # SLO attainment at each threshold (all issued requests)
    slo_vals = {}
    total = len(ttfts)
    for th in SLO_THRESHOLDS:
        met = sum(1 for t, s in zip(ttfts, successes) if s and t <= th)
        slo_vals[th] = (met / total * 100) if total else 0.0
    t = v.get("successful_ttft_ms", {}) or {}
    return {
        "system": system, "bt": bt, "rps": rps, "prefix": pr,
        "ttft_mean_ms": t.get("mean"),
        "ttft_p99_ms": t.get("p99"),
        **{f"slo_{int(th)}s": slo_vals[th] for th in SLO_THRESHOLDS},
    }


def main():
    rows = []
    for sysname in ("dp4_tp8_sp", "afd_dp3_tp8_ep8"):
        for f in sorted(RESULT_DIR.glob(f"{sysname}-*.verified.json")):
            rows.append(parse(f))
    hdr = ("system,bt,prefix,rps,ttft_mean_ms,ttft_p99_ms,"
           + ",".join(f"slo_{int(th)}s" for th in SLO_THRESHOLDS))
    with open(OUT, "w") as fh:
        fh.write(hdr + "\n")
        for r in rows:
            fh.write(
                f"{r['system']},{r['bt']},{r['prefix']},{r['rps']},"
                f"{r['ttft_mean_ms']},{r['ttft_p99_ms']},"
                + ",".join(f"{r[f'slo_{int(th)}s']:.2f}" for th in SLO_THRESHOLDS)
                + "\n"
            )
    print(f"Wrote {len(rows)} rows to {OUT}")


if __name__ == "__main__":
    main()
