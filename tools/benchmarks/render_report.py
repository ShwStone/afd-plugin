#!/usr/bin/env python3
"""Render an interactive HTML performance report from summary.csv.

Charts:
  1. Mean TTFT vs RPS, one panel per batch_tokens, baseline vs AFD lines
  2. P99 TTFT vs RPS, per batch_tokens
  3. SLO attainment vs RPS, per batch_tokens
  4. Prefix sensitivity: TTFT vs prefix ratio, per batch_tokens, at fixed RPS
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

CSV = Path("bench_results/prefill/summary.csv")
OUT = Path("bench_results/prefill/report.html")

BATCH_TOKENS = (8192, 16384, 32768, 49152, 65536)
RPS_LEVELS = (4.0, 6.0, 8.0, 10.0, 12.0)
PREFIXES = (0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)
SYS_A, SYS_B = "dp4_tp8_sp", "afd_dp3_tp8_ep8"
SYS_LABEL = {SYS_A: "Baseline DP4xTP8", SYS_B: "AFD DP3xTP8+EP8"}


def load():
    rows = []
    with open(CSV) as fh:
        for line in fh:
            if line.startswith("system,"):
                continue
            p = [x.strip() for x in line.strip().split(",")]
            rows.append({
                "system": p[0], "bt": int(p[1]), "prefix": float(p[2]),
                "rps": float(p[3]), "slo": float(p[4]),
                "ttft_mean": None if p[5] == "None" else float(p[5]),
                "ttft_p99": None if p[6] == "None" else float(p[6]),
                "success": int(p[7]), "failed": int(p[8]),
            })
    return rows


def lookup(rows, sysname, bt, prefix, rps):
    for r in rows:
        if (r["system"], r["bt"], r["prefix"], r["rps"]) == (sysname, bt, prefix, rps):
            return r
    # Missing cell: return a stub so plots degrade gracefully
    return {"ttft_mean": None, "ttft_p99": None, "slo": None, "success": 0, "failed": 0}


def js_num(v):
    if v is None:
        return "null"
    return f"{v:.1f}"


def main():
    rows = load()
    by_bt_prefix = defaultdict(list)
    for r in rows:
        by_bt_prefix[(r["system"], r["bt"], r["prefix"])].append(r)

    # ---- Build per-bt panels ----
    bt_panels = []
    for bt in BATCH_TOKENS:
        # RPS curves at prefix=0 (no cache) and prefix=0.99 (high cache)
        for label, prefix in [("prefix0", 0.0), ("prefix0p99", 0.99)]:
            a_mean = [lookup(rows, SYS_A, bt, prefix, rps)["ttft_mean"] for rps in RPS_LEVELS]
            b_mean = [lookup(rows, SYS_B, bt, prefix, rps)["ttft_mean"] for rps in RPS_LEVELS]
            a_p99 = [lookup(rows, SYS_A, bt, prefix, rps)["ttft_p99"] for rps in RPS_LEVELS]
            b_p99 = [lookup(rows, SYS_B, bt, prefix, rps)["ttft_p99"] for rps in RPS_LEVELS]
            a_slo = [lookup(rows, SYS_A, bt, prefix, rps)["slo"] for rps in RPS_LEVELS]
            b_slo = [lookup(rows, SYS_B, bt, prefix, rps)["slo"] for rps in RPS_LEVELS]
            a_slo = [x if x is not None else "null" for x in a_slo]
            b_slo = [x if x is not None else "null" for x in b_slo]
            bt_panels.append({
                "bt": bt, "label": label, "prefix": prefix,
                "a_mean": a_mean, "b_mean": b_mean,
                "a_p99": a_p99, "b_p99": b_p99,
                "a_slo": a_slo, "b_slo": b_slo,
            })

    # ---- Prefix sensitivity at fixed RPS=8 ----
    prefix_panels = []
    for bt in BATCH_TOKENS:
        a = [lookup(rows, SYS_A, bt, pr, 8.0)["ttft_mean"] for pr in PREFIXES]
        b = [lookup(rows, SYS_B, bt, pr, 8.0)["ttft_mean"] for pr in PREFIXES]
        prefix_panels.append({"bt": bt, "a": a, "b": b})

    # ---- Emit HTML ----
    rps_labels = json.dumps([str(int(r)) for r in RPS_LEVELS])
    prefix_labels = json.dumps([str(p) for p in PREFIXES])

    # RPS panels: each bt has 2 sub-charts (prefix0, prefix0p99)
    rps_charts = ""
    for p in bt_panels:
        cid = f"rps_{p['bt']}_{p['label']}"
        rps_charts += f"""
<h3>bt={p['bt']} ({p['label']})</h3>
<canvas id="{cid}_mean" height="70"></canvas>
<canvas id="{cid}_p99" height="70"></canvas>
<canvas id="{cid}_slo" height="70"></canvas>
<script>
new Chart(document.getElementById('{cid}_mean'), {{type:'line',
  data:{{labels:{rps_labels},datasets:[
    {{label:'{SYS_LABEL[SYS_A]}',data:{json.dumps([js_num(x) for x in p['a_mean']])},borderColor:'#2c7fb8',fill:false}},
    {{label:'{SYS_LABEL[SYS_B]}',data:{json.dumps([js_num(x) for x in p['b_mean']])},borderColor:'#d95f0e',fill:false}}
  ]}},
  options:{{responsive:true,scales:{{y:{{title:{{display:true,text:'Mean TTFT (ms)'}}}}}}}}}});
new Chart(document.getElementById('{cid}_p99'), {{type:'line',
  data:{{labels:{rps_labels},datasets:[
    {{label:'{SYS_LABEL[SYS_A]}',data:{json.dumps([js_num(x) for x in p['a_p99']])},borderColor:'#2c7fb8',fill:false}},
    {{label:'{SYS_LABEL[SYS_B]}',data:{json.dumps([js_num(x) for x in p['b_p99']])},borderColor:'#d95f0e',fill:false}}
  ]}},
  options:{{responsive:true,scales:{{y:{{title:{{display:true,text:'P99 TTFT (ms)'}}}}}}}}}});
new Chart(document.getElementById('{cid}_slo'), {{type:'line',
  data:{{labels:{rps_labels},datasets:[
    {{label:'{SYS_LABEL[SYS_A]}',data:{json.dumps(p['a_slo'])},borderColor:'#2c7fb8',fill:false}},
    {{label:'{SYS_LABEL[SYS_B]}',data:{json.dumps(p['b_slo'])},borderColor:'#d95f0e',fill:false}}
  ]}},
  options:{{responsive:true,scales:{{y:{{title:{{display:true,text:'SLO attainment (%)'}},min:0,max:100}}}}}}}});
</script>
"""
    # Prefix panels
    prefix_charts = ""
    for p in prefix_panels:
        cid = f"prefix_{p['bt']}"
        prefix_charts += f"""
<canvas id="{cid}" height="70"></canvas>
<script>
new Chart(document.getElementById('{cid}'), {{type:'line',
  data:{{labels:{prefix_labels},datasets:[
    {{label:'{SYS_LABEL[SYS_A]}',data:{json.dumps([js_num(x) for x in p['a']])},borderColor:'#2c7fb8',fill:false}},
    {{label:'{SYS_LABEL[SYS_B]}',data:{json.dumps([js_num(x) for x in p['b']])},borderColor:'#d95f0e',fill:false}}
  ]}},
  options:{{responsive:true,scales:{{y:{{title:{{display:true,text:'Mean TTFT (ms) @ RPS 8'}}}}}}}}}});
</script>
"""
    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Prefill Performance Report</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>body{{font-family:sans-serif;margin:20px;background:#fafafa;}}
h1{{color:#222;}}h2{{color:#444;border-bottom:1px solid #ddd;padding-bottom:6px;}}
h3{{color:#555;margin:16px 0 4px;}}.card{{background:#fff;border-radius:8px;padding:16px;margin:12px 0;box-shadow:0 1px 3px rgba(0,0,0,.1);}}
</style></head><body>
<h1>Prefill Performance: Baseline DP4xTP8 vs AFD DP3xTP8+EP8</h1>
<p>Dataset: cp8sp50k (875 requests, 18.18M input tokens, 1 output token). TTFT SLO = 10s.</p>

<h2>1. Mean / P99 TTFT &amp; SLO vs Request Rate</h2>
<div class="card">{rps_charts}</div>

<h2>2. Prefix Cache Sensitivity (Mean TTFT @ RPS 8 vs prefix ratio)</h2>
<div class="card">{prefix_charts}</div>
</body></html>"""
    OUT.write_text(html)
    print(f"Report written to {OUT} ({len(html)} bytes)")


if __name__ == "__main__":
    main()
