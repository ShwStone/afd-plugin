#!/usr/bin/env python3
"""Generate comprehensive prefill experiment analysis + markdown report."""
import json
import collections
import statistics
from pathlib import Path

RESULT_DIR = Path("bench_results/prefill")


def parse_verified(f: Path):
    d = json.loads(f.read_text())
    v = d.get("afd_verification", {})
    stem = f.stem.replace(".verified", "")
    parts = stem.split("-")
    system = parts[0]
    bt = int(parts[1].replace("mbt", ""))
    rps = float(parts[2].replace("rps", "").replace("p0", ""))
    pr_raw = parts[3].replace("prefix", "")
    pr = 0.0 if pr_raw == "0" else float(pr_raw.replace("p", "."))
    t = v.get("successful_ttft_ms", {}) or {}
    return {
        "system": system, "bt": bt, "rps": rps, "prefix": pr,
        "stem": stem,
        "slo": v.get("slo_attainment_all_requests", 0) * 100,
        "ttft_mean": t.get("mean"),
        "ttft_p50": t.get("p50"),
        "ttft_p90": t.get("p90"),
        "ttft_p95": t.get("p95"),
        "ttft_p99": t.get("p99"),
        "success": v.get("successful_requests", 0),
        "failed": v.get("failed_requests", 0),
        "issued": v.get("issued_requests", 0),
    }


def load_all():
    rows = []
    for sysname in ("dp4_tp8_sp", "afd_dp3_tp8_ep8"):
        for f in sorted(RESULT_DIR.glob(f"{sysname}-*.verified.json")):
            rows.append(parse_verified(f))
    return rows


def main():
    rows = load_all()
    print(f"Loaded {len(rows)} verified rows")

    by_cell = collections.defaultdict(list)
    for r in rows:
        by_cell[(r["system"], r["bt"], r["prefix"])].append(r)

    # Summary table
    print("\n=== Per-cell summary (mean TTFT in s / SLO%) ===")
    header = f"{'system':<18} {'bt':>7} {'prefix':>6} | " + " | ".join(f"rps={rps}" for rps in (4, 6, 8, 10, 12))
    print(header)
    print("-" * len(header))
    for (sysname, bt, pr), cells in sorted(by_cell.items()):
        d = {c["rps"]: c for c in cells}
        cells_by_rps = [d.get(rps) for rps in (4, 6, 8, 10, 12)]
        vals = []
        for c in cells_by_rps:
            if c is None:
                vals.append("  MISSING ")
            elif c["ttft_mean"] is None:
                vals.append(f"  FAIL {c['failed']}")
            else:
                vals.append(f"{c['ttft_mean']:.2f}/{c['slo']:.0f}%")
        print(f"{sysname:<18} {bt:>7} {pr:>6} | " + " | ".join(f"{v:>8}" for v in vals))


if __name__ == "__main__":
    main()
