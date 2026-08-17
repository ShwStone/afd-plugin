#!/usr/bin/env python3
"""Analyze verified prefill results, detect anomalies, print summary."""
import json
import collections
import sys
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
    mean = t.get("mean")
    p99 = t.get("p99")
    return {
        "system": system, "bt": bt, "rps": rps, "prefix": pr,
        "slo": v.get("slo_attainment_all_requests", 0) * 100,
        "ttft_mean": (mean / 1000) if mean is not None else None,
        "ttft_p99": (p99 / 1000) if p99 is not None else None,
        "success": v.get("successful_requests", 0),
        "failed": v.get("failed_requests", 0),
    }


def main():
    systems = ["dp4_tp8_sp", "afd_dp3_tp8_ep8"]
    rows = []
    for sysname in systems:
        files = sorted(RESULT_DIR.glob(f"{sysname}-*.verified.json"))
        for f in files:
            rows.append(parse_verified(f))
    print(f"Total verified rows: {len(rows)}")

    by_cell = collections.defaultdict(list)
    for r in rows:
        by_cell[(r["system"], r["bt"], r["prefix"])].append(r)

    print(f"Total cells: {len(by_cell)}")
    for (sysname, bt, pr), cells in sorted(by_cell.items()):
        n_rps = len(cells)
        expected = 5
        status = "OK" if n_rps == expected else f"MISSING {expected-n_rps} RPS"
        print(f"  {sysname} bt={bt} prefix={pr}: {n_rps} RPS {status}")

    # Anomaly detection: within a cell, TTFT should generally rise with RPS.
    # Flag >3x jump in mean TTFT, or SLO collapsing from ~100% to <50%.
    anomalies = []
    for (sysname, bt, pr), cells in sorted(by_cell.items()):
        cells_sorted = sorted(cells, key=lambda x: x["rps"])
        for i in range(1, len(cells_sorted)):
            prev, cur = cells_sorted[i-1], cells_sorted[i]
            if prev["ttft_mean"] and cur["ttft_mean"] and prev["ttft_mean"] > 0:
                ratio = cur["ttft_mean"] / prev["ttft_mean"]
                if ratio > 3:
                    anomalies.append(
                        f"  {sysname} bt={bt} prefix={pr} rps={cur['rps']}: "
                        f"ttft_mean {prev['ttft_mean']:.2f}->{cur['ttft_mean']:.2f} ({ratio:.1f}x)"
                    )
        # SLO collapse
        slos = [c["slo"] for c in cells_sorted]
        if len(slos) >= 2 and slos[0] >= 95 and slos[-1] < 50:
            anomalies.append(
                f"  {sysname} bt={bt} prefix={pr}: SLO collapse {slos[0]:.0f}% -> {slos[-1]:.0f}%"
            )

    print(f"\n=== Anomalies ({len(anomalies)}) ===")
    for a in anomalies:
        print(a)


if __name__ == "__main__":
    main()
