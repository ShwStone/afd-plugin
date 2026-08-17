#!/usr/bin/env python3
"""Render paper-style charts:
  Group A: TTFT vs batch, {baseline,afd} x {mean,p99}, 5 RPS curves each
  Group B: SLO attainment vs batch, {(rps8,slo2),(rps10,slo5),(rps12,slo10),(rps12,slo20)},
           afd + baseline lines each
Data: prefix=0 only (cold cache), from slo_summary.csv
"""
import csv
import json
from pathlib import Path

CSV = Path("bench_results/prefill/slo_summary.csv")
OUT = Path("bench_results/prefill/paper_charts.html")

BATCH = (8192, 16384, 32768, 49152, 65536)
RPS = (4, 6, 8, 10, 12)
SA, SB = "dp4_tp8_sp", "afd_dp3_tp8_ep8"
LA, LB = "Baseline DP4xTP8", "AFD DP3xTP8+EP8"


def load():
    rows = []
    with open(CSV) as fh:
        for line in fh:
            if line.startswith("system,"):
                continue
            p = [x.strip() for x in line.strip().split(",")]
            if float(p[2]) != 0.0:
                continue
            rows.append({
                "system": p[0], "bt": int(p[1]), "rps": float(p[3]),
                "mean": None if p[4] == "None" else float(p[4]),
                "p99": None if p[5] == "None" else float(p[5]),
                "slo2": float(p[6]), "slo5": float(p[7]),
                "slo10": float(p[8]), "slo20": float(p[9]),
            })
    return rows


def get(rows, sysname, bt, rps, key):
    for r in rows:
        if r["system"] == sysname and r["bt"] == bt and r["rps"] == rps:
            return r[key]
    return None


def main():
    rows = load()
    rps_labels = json.dumps(list(BATCH))
    colors = {4: "#1f77b4", 6: "#ff7f0e", 8: "#2ca02c", 10: "#d62728", 12: "#9467bd"}

    charts = ""

    # ---- Group A: TTFT vs batch ----
    for sysname, syslabel, colorbase in [(SA, LA, "#2c7fb8"), (SB, LB, "#d95f0e")]:
        for metric, metlabel in [("mean", "Mean TTFT"), ("p99", "P99 TTFT")]:
            cid = f"ttft_{sysname}_{metric}"
            datasets = []
            for rps in RPS:
                data = [get(rows, sysname, bt, rps, metric) for bt in BATCH]
                datasets.append(
                    "{" + f"label:'rps={rps}',data:{json.dumps([None if x is None else round(x,1) for x in data])},"
                    f"borderColor:'{colors[rps]}',fill:false" + "}"
                )
            charts += f"""
<div class="card"><h3>{syslabel} — {metlabel} vs Max Batch Tokens (prefix=0)</h3>
<canvas id="{cid}" height="80"></canvas>
<script>new Chart(document.getElementById('{cid}'), {{type:'line',
data:{{labels:{rps_labels},datasets:[{','.join(datasets)}]}},
options:{{responsive:true,scales:{{x:{{title:{{display:true,text:'Max Batch Tokens'}}}},y:{{title:{{display:true,text:'{metlabel} (ms)'}}}}}}}}}});</script></div>
"""

    # ---- Group B: SLO attainment vs batch ----
    slo_panels = [
        (8, "slo2", "RPS=8, SLO=2s"),
        (10, "slo5", "RPS=10, SLO=5s"),
        (12, "slo10", "RPS=12, SLO=10s"),
        (12, "slo20", "RPS=12, SLO=20s"),
    ]
    for rps, slokey, title in slo_panels:
        cid = f"slo_{rps}_{slokey}"
        a_data = [get(rows, SA, bt, rps, slokey) for bt in BATCH]
        b_data = [get(rows, SB, bt, rps, slokey) for bt in BATCH]
        charts += f"""
<div class="card"><h3>{title} — SLO Attainment vs Max Batch Tokens</h3>
<canvas id="{cid}" height="80"></canvas>
<script>new Chart(document.getElementById('{cid}'), {{type:'line',
data:{{labels:{rps_labels},datasets:[
 {{label:'{LA}',data:{json.dumps(a_data)},borderColor:'#2c7fb8',fill:false}},
 {{label:'{LB}',data:{json.dumps(b_data)},borderColor:'#d95f0e',fill:false}}]}},
options:{{responsive:true,scales:{{x:{{title:{{display:true,text:'Max Batch Tokens'}}}},y:{{title:{{display:true,text:'SLO attainment (%)'}},min:0,max:100}}}}}}}});</script></div>
"""

    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>Prefill Paper Charts</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>body{{font-family:sans-serif;margin:20px;background:#fafafa;}}
h1{{color:#222;}}h3{{color:#555;margin:12px 0 4px;}}
.card{{background:#fff;border-radius:8px;padding:16px;margin:12px 0;box-shadow:0 1px 3px rgba(0,0,0,.1);}}
</style></head><body>
<h1>Prefill Performance — prefix cache DISABLED (cold cache)</h1>
<p>cp8sp50k, 875 requests, 1 output token.</p>
<h2>Group A: TTFT vs Max Batch Tokens</h2>
{charts}
</body></html>"""
    OUT.write_text(html)
    print(f"Wrote {OUT} ({len(html)} bytes)")


if __name__ == "__main__":
    main()
