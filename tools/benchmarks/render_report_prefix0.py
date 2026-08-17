#!/usr/bin/env python3
"""Render prefix=0-only performance report (pure architecture comparison)."""
import csv
import json
from pathlib import Path

CSV = Path("bench_results/prefill/summary.csv")
OUT = Path("bench_results/prefill/report_prefix0.html")

BATCH_TOKENS = (8192, 16384, 32768, 49152, 65536)
RPS = (4, 6, 8, 10, 12)
SYS_A, SYS_B = "dp4_tp8_sp", "afd_dp3_tp8_ep8"
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
                "slo": float(p[4]),
                "mean": None if p[5] == "None" else float(p[5]),
                "p99": None if p[6] == "None" else float(p[6]),
            })
    return rows


def main():
    rows = load()
    charts = ""
    for bt in BATCH_TOKENS:
        a_mean = [next((r["mean"] for r in rows if r["system"] == SYS_A and r["bt"] == bt and r["rps"] == rp), None) for rp in RPS]
        b_mean = [next((r["mean"] for r in rows if r["system"] == SYS_B and r["bt"] == bt and r["rps"] == rp), None) for rp in RPS]
        a_p99 = [next((r["p99"] for r in rows if r["system"] == SYS_A and r["bt"] == bt and r["rps"] == rp), None) for rp in RPS]
        b_p99 = [next((r["p99"] for r in rows if r["system"] == SYS_B and r["bt"] == bt and r["rps"] == rp), None) for rp in RPS]
        a_slo = [next((r["slo"] for r in rows if r["system"] == SYS_A and r["bt"] == bt and r["rps"] == rp), None) for rp in RPS]
        b_slo = [next((r["slo"] for r in rows if r["system"] == SYS_B and r["bt"] == bt and r["rps"] == rp), None) for rp in RPS]
        fmt = lambda xs: json.dumps([None if x is None else round(x, 1) for x in xs])
        charts += f"""
<div class="card"><h3>bt={bt}</h3>
<canvas id="m_{bt}" height="70"></canvas>
<canvas id="p_{bt}" height="70"></canvas>
<canvas id="s_{bt}" height="70"></canvas>
<script>
new Chart(document.getElementById('m_{bt}'), {{type:'line',data:{{labels:{json.dumps(list(RPS))},datasets:[
 {{label:'{LA}',data:{fmt(a_mean)},borderColor:'#2c7fb8',fill:false}},
 {{label:'{LB}',data:{fmt(b_mean)},borderColor:'#d95f0e',fill:false}}]}},
 options:{{responsive:true,scales:{{y:{{title:{{display:true,text:'Mean TTFT (ms)'}}}}}}}}}});
new Chart(document.getElementById('p_{bt}'), {{type:'line',data:{{labels:{json.dumps(list(RPS))},datasets:[
 {{label:'{LA}',data:{fmt(a_p99)},borderColor:'#2c7fb8',fill:false}},
 {{label:'{LB}',data:{fmt(b_p99)},borderColor:'#d95f0e',fill:false}}]}},
 options:{{responsive:true,scales:{{y:{{title:{{display:true,text:'P99 TTFT (ms)'}}}}}}}}}});
new Chart(document.getElementById('s_{bt}'), {{type:'line',data:{{labels:{json.dumps(list(RPS))},datasets:[
 {{label:'{LA}',data:{json.dumps(a_slo)},borderColor:'#2c7fb8',fill:false}},
 {{label:'{LB}',data:{json.dumps(b_slo)},borderColor:'#d95f0e',fill:false}}]}},
 options:{{responsive:true,scales:{{y:{{title:{{display:true,text:'SLO attainment (%)'}},min:0,max:100}}}}}}}});
</script></div>
"""
    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>Prefill prefix=0 comparison</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>body{{font-family:sans-serif;margin:20px;background:#fafafa;}}
h1{{color:#222;}}h3{{color:#555;margin:12px 0 4px;}}
.card{{background:#fff;border-radius:8px;padding:16px;margin:12px 0;box-shadow:0 1px 3px rgba(0,0,0,.1);}}
</style></head><body>
<h1>Prefill Performance — prefix cache DISABLED (cold cache)</h1>
<p>Baseline DP4xTP8 vs AFD DP3xTP8+EP8. cp8sp50k, 875 req, 1 output token, TTFT SLO=10s. Pure architecture comparison.</p>
{charts}
</body></html>"""
    OUT.write_text(html)
    print(f"Wrote {OUT} ({len(html)} bytes)")


if __name__ == "__main__":
    main()
