#!/usr/bin/env python3
"""Find anomalous cells: TTFT=0, unusually low TTFT, or failed requests."""
import json
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
    return {
        "system": system, "bt": bt, "rps": rps, "prefix": pr,
        "stem": f.stem.replace(".verified", ""),
        "slo": v.get("slo_attainment_all_requests", 0) * 100,
        "ttft_mean": t.get("mean"),
        "ttft_p99": t.get("p99"),
        "success": v.get("successful_requests", 0),
        "failed": v.get("failed_requests", 0),
        "issued": v.get("issued_requests", 0),
    }


def main():
    files = sorted(RESULT_DIR.glob("*.verified.json"))
    rows = [parse_verified(f) for f in files]
    print(f"Total verified files: {len(rows)}")

    print("\n=== Cells with TTFT mean == 0 or None ===")
    zero = [r for r in rows if r["ttft_mean"] in (0, None)]
    for r in zero:
        print(f"  {r['system']} bt={r['bt']} prefix={r['prefix']} rps={r['rps']} "
              f"mean={r['ttft_mean']} p99={r['ttft_p99']} succ={r['success']} failed={r['failed']}")

    print(f"\n=== Cells with failed requests > 0 ===")
    failed = [r for r in rows if r["failed"] > 0]
    for r in failed:
        print(f"  {r['system']} bt={r['bt']} prefix={r['prefix']} rps={r['rps']} "
              f"succ={r['success']}/{r['issued']} failed={r['failed']}")

    print("\n=== Abnormally low TTFT (suspicious network drops) ===")
    # Group by system+bt+prefix, check rps trend. A cell whose TTFT collapses to
    # near-zero at high RPS while lower RPS had substantial TTFT = likely failed run.
    import collections
    by_cell = collections.defaultdict(list)
    for r in rows:
        by_cell[(r["system"], r["bt"], r["prefix"])].append(r)
    for (sysname, bt, pr), cells in sorted(by_cell.items()):
        cells_sorted = sorted(cells, key=lambda x: x["rps"])
        for r in cells_sorted:
            if r["ttft_mean"] is not None and r["ttft_mean"] < 0.05:
                print(f"  {sysname} bt={bt} prefix={pr} rps={r['rps']} "
                      f"mean={r['ttft_mean']*1000:.1f}ms p99={r['ttft_p99']}ms")

    # Summary of any cell missing from expected matrix
    print("\n=== Expected cells present ===")
    expected = set()
    for sysname in ("dp4_tp8_sp", "afd_dp3_tp8_ep8"):
        for bt in (8192, 16384, 32768, 49152, 65536):
            for pr in (0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99):
                expected.add((sysname, bt, pr))
    present = set((r["system"], r["bt"], r["prefix"]) for r in rows)
    missing = expected - present
    print(f"  Missing {len(missing)} of {len(expected)} expected cells:")
    for m in sorted(missing):
        print(f"    {m[0]} bt={m[1]} prefix={m[2]}")


if __name__ == "__main__":
    main()
