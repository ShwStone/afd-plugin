#!/usr/bin/env python3
"""Render comprehensive prefix-aware charts (all dimensions).
Reads summary.csv (TTFT) + slo_summary.csv (multi-threshold SLO).
"""
import csv
import json
from pathlib import Path

SUM = Path("bench_results/prefill/summary.csv")
SLO = Path("bench_results/prefill/slo_summary.csv")
OUT = Path("bench_results/prefill/all_charts.html")

BATCH = (8192, 16384, 32768, 49152, 65536)
RPS = (4, 6, 8, 10, 12)
PREFIX = (0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)
SA, SB = "dp4_tp8_sp", "afd_dp3_tp8_ep8"
LA, LB = "Baseline DP4xTP8", "AFD DP3xTP8+EP8"
SYS_COLOR = {SA: "#2c7fb8", SB: "#d95f0e"}
RPS_COLOR = {4: "#1f77b4", 6: "#ff7f0e", 8: "#2ca02c", 10: "#d62728", 12: "#9467bd"}
PREFIX_LABEL = {0.0: "p0", 0.25: "p25", 0.5: "p50", 0.75: "p75", 0.9: "p90", 0.95: "p95", 0.99: "p99"}


def load_sum():
    rows = {}
    with open(SUM) as fh:
        for line in fh:
            if line.startswith("system,"):
                continue
            p = [x.strip() for x in line.strip().split(",")]
            key = (p[0], int(p[1]), float(p[2]), float(p[3]))
            rows[key] = {
                "mean": None if p[5] == "None" else float(p[5]),
                "p99": None if p[6] == "None" else float(p[6]),
                "slo10": float(p[4]),
            }
    return rows


def load_slo():
    rows = {}
    with open(SLO) as fh:
        for line in fh:
            if line.startswith("system,"):
                continue
            p = [x.strip() for x in line.strip().split(",")]
            key = (p[0], int(p[1]), float(p[2]), float(p[3]))
            rows[key] = {"slo2": float(p[6]), "slo5": float(p[7]), "slo10": float(p[8]), "slo20": float(p[9])}
    return rows


def ds(label, data, color):
    return "{" + f"label:'{label}',data:{json.dumps(data)},borderColor:'{color}',fill:false" + "}"


def main():
    sum_rows = load_sum()
    slo_rows = load_slo()
    charts = ""

    # ===== 1. TTFT (mean & p99) vs batch, per system, 5 RPS curves, prefix=0 =====
    for sysname, syslabel in [(SA, LA), (SB, LB)]:
        for metric, mlabel in [("mean", "Mean TTFT"), ("p99", "P99 TTFT")]:
            cid = f"ttft_batch_{sysname}_{metric}"
            dss = []
            for rps in RPS:
                data = [sum_rows.get((sysname, bt, 0.0, rps), {}).get(metric) for bt in BATCH]
                data = [None if v is None else round(v, 1) for v in data]
                dss.append(ds(f"rps={rps}", data, RPS_COLOR[rps]))
            charts += f"""
<div class="card"><h3>{syslabel} — {mlabel} vs Max Batch (prefix=0, cold cache)</h3>
<canvas id="{cid}" height="70"></canvas>
<script>new Chart(document.getElementById('{cid}'), {{type:'line',
data:{{labels:{json.dumps(list(BATCH))},datasets:[{','.join(dss)}]}},
options:{{responsive:true,scales:{{x:{{title:{{display:true,text:'Max Batch Tokens'}}}},y:{{title:{{display:true,text:'{mlabel} (ms)'}}}}}}}}}});</script></div>
"""

    # ===== 2. TTFT (mean) vs RPS, per batch, 2 system lines, prefix=0 =====
    for metric, mlabel in [("mean", "Mean TTFT"), ("p99", "P99 TTFT")]:
        for bt in BATCH:
            cid = f"ttft_rps_{metric}_{bt}"
            a = [sum_rows.get((SA, bt, 0.0, rps), {}).get(metric) for rps in RPS]
            b = [sum_rows.get((SB, bt, 0.0, rps), {}).get(metric) for rps in RPS]
            a = [None if v is None else round(v, 1) for v in a]
            b = [None if v is None else round(v, 1) for v in b]
            charts += f"""
<div class="card"><h3>{mlabel} vs RPS — bt={bt} (prefix=0)</h3>
<canvas id="{cid}" height="60"></canvas>
<script>new Chart(document.getElementById('{cid}'), {{type:'line',
data:{{labels:{json.dumps(list(RPS))},datasets:[
 {ds(LA, a, SYS_COLOR[SA])},{ds(LB, b, SYS_COLOR[SB])}]}},
options:{{responsive:true,scales:{{x:{{title:{{display:true,text:'RPS'}}}},y:{{title:{{display:true,text:'{mlabel} (ms)'}}}}}}}}}});</script></div>
"""

    # ===== 3. Prefix sensitivity: TTFT mean vs prefix, per (bt, rps) =====
    for bt in BATCH:
        for rps in (6, 8, 10):
            cid2 = f"prefix_{bt}_rps{rps}"
            a = [sum_rows.get((SA, bt, pr, rps), {}).get("mean") for pr in PREFIX]
            b = [sum_rows.get((SB, bt, pr, rps), {}).get("mean") for pr in PREFIX]
            a = [None if v is None else round(v, 1) for v in a]
            b = [None if v is None else round(v, 1) for v in b]
            charts += f"""
<div class="card"><h3>Prefix Sensitivity — Mean TTFT vs prefix ratio, bt={bt}, RPS={rps}</h3>
<canvas id="{cid2}" height="60"></canvas>
<script>new Chart(document.getElementById('{cid2}'), {{type:'line',
data:{{labels:{json.dumps([PREFIX_LABEL[pr] for pr in PREFIX])},datasets:[
 {ds(LA, a, SYS_COLOR[SA])},{ds(LB, b, SYS_COLOR[SB])}]}},
options:{{responsive:true,scales:{{x:{{title:{{display:true,text:'Prefix ratio'}}}},y:{{title:{{display:true,text:'Mean TTFT (ms)'}}}}}}}}}});</script></div>
"""

    # ===== 4. SLO attainment vs batch, per (rps, slo-threshold) =====
    slo_panels = [(8, "slo2", "RPS=8, SLO=2s"), (10, "slo5", "RPS=10, SLO=5s"),
                  (12, "slo10", "RPS=12, SLO=10s"), (12, "slo20", "RPS=12, SLO=20s")]
    for rps, slokey, title in slo_panels:
        cid = f"slo_{rps}_{slokey}"
        a = [slo_rows.get((SA, bt, 0.0, rps), {}).get(slokey, 0) for bt in BATCH]
        b = [slo_rows.get((SB, bt, 0.0, rps), {}).get(slokey, 0) for bt in BATCH]
        charts += f"""
<div class="card"><h3>SLO Attainment vs Max Batch — {title} (prefix=0)</h3>
<canvas id="{cid}" height="60"></canvas>
<script>new Chart(document.getElementById('{cid}'), {{type:'line',
data:{{labels:{json.dumps(list(BATCH))},datasets:[
 {ds(LA, a, SYS_COLOR[SA])},{ds(LB, b, SYS_COLOR[SB])}]}},
options:{{responsive:true,scales:{{x:{{title:{{display:true,text:'Max Batch Tokens'}}}},y:{{title:{{display:true,text:'SLO attainment (%)'}},min:0,max:100}}}}}}}}}});</script></div>
"""

    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>Prefill Full Comparison</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>body{{font-family:sans-serif;margin:20px;background:#fafafa;}}
h1{{color:#222;}}h2{{color:#444;border-bottom:1px solid #ddd;padding-bottom:6px;}}
h3{{color:#555;margin:12px 0 4px;}}
.card{{background:#fff;border-radius:8px;padding:16px;margin:12px 0;box-shadow:0 1px 3px rgba(0,0,0,.1);}}
</style></head><body>
<h1>Prefill Performance: Baseline DP4xTP8 vs AFD DP3xTP8+EP8</h1>
<p>cp8sp50k, 875 requests, 1 output token. Full matrix: 5 batch × 7 prefix × 5 RPS.</p>
<h2>1. TTFT vs Max Batch (prefix=0)</h2>
{charts}
</body></html>"""
    OUT.write_text(html)
    print(f"Wrote {OUT} ({len(html)} bytes)")


if __name__ == "__main__":
    main()
