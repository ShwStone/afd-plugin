#!/usr/bin/env python3
"""Export per-cell summary to compact CSV for local plotting."""
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
    return {
        "system": system, "bt": bt, "rps": rps, "prefix": pr,
        "slo": v.get("slo_attainment_all_requests", 0) * 100,
        "ttft_mean_ms": t.get("mean"),
        "ttft_p99_ms": t.get("p99"),
        "success": v.get("successful_requests", 0),
        "failed": v.get("failed_requests", 0),
    }


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "bench_results/prefill/summary.csv"
    rows = []
    for sysname in ("dp4_tp8_sp", "afd_dp3_tp8_ep8"):
        for f in sorted(RESULT_DIR.glob(f"{sysname}-*.verified.json")):
            rows.append(parse_verified(f))
    with open(out, "w") as fh:
        fh.write("system,bt,prefix,rps,slo,ttft_mean_ms,ttft_p99_ms,success,failed\n")
        for r in rows:
            fh.write(
                f"{r['system']},{r['bt']},{r['prefix']},{r['rps']},{r['slo']:.1f},"
                f"{r['ttft_mean_ms']},{r['ttft_p99_ms']},{r['success']},{r['failed']}\n"
            )
    print(f"Wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
