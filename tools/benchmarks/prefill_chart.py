#!/usr/bin/env python3
"""Chart prefill experiment results: TTFT and SLO vs RPS."""
import json, sys, os
from pathlib import Path

RESULTS_DIR = Path("bench_results/prefill")

def load_verified(f):
    """Load a .verified.json and return (bt, rps, system, prefix, data)."""
    d = json.loads(f.read_text())
    v = d.get("afd_verification", {})
    stem = f.stem.replace(".verified", "")
    parts = stem.split("-")
    # stem format: system-mbt{rps}-rps{rps}p0-prefix{pr}-repeat1
    system = parts[0]
    bt = int(parts[1].replace("mbt", ""))
    rps = float(parts[2].replace("rps", "").replace("p0", ""))
    pr_raw = parts[3].replace("prefix", "")
    pr = float(pr_raw) if pr_raw != "0" else 0.0
    return bt, rps, system, pr, v

def main():
    files = sorted(RESULTS_DIR.glob("*.verified.json"))
    if not files:
        print("No verified results found in", RESULTS_DIR)
        return

    rows = []
    for f in files:
        bt, rps, system, pr, v = load_verified(f)
        t = v.get("successful_ttft_ms", {})
        rows.append({
            "bt": bt, "rps": rps, "system": system, "prefix": pr,
            "slo": v.get("slo_attainment_all_requests", 0) * 100,
            "ttft_mean": t.get("mean", 0) / 1000,
            "ttft_p50": t.get("p50", 0) / 1000,
            "ttft_p90": t.get("p90", 0) / 1000,
            "ttft_p95": t.get("p95", 0) / 1000,
            "ttft_p99": t.get("p99", 0) / 1000,
        })

    # Generate HTML chart
    html = """<!DOCTYPE html><html><head><meta charset="utf-8">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
body { font-family: sans-serif; margin: 20px; background: #f8f9fa; }
h1 { color: #333; }
.chart-container { background: white; border-radius: 8px; padding: 20px; margin: 20px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
table { border-collapse: collapse; width: 100%; }
th, td { border: 1px solid #ddd; padding: 8px; text-align: right; font-size: 13px; }
th { background: #4a90d9; color: white; }
tr:nth-child(even) { background: #f2f2f2; }
</style></head><body>
<h1>Prefill Experiment Results — dp4_tp8_sp (Baseline)</h1>
"""

    baselines = [r for r in rows if r["system"] == "dp4_tp8_sp" and r["prefix"] == 0]
    bts = sorted(set(r["bt"] for r in baselines))
    colors = ["#4a90d9", "#e67e22", "#2ecc71", "#e74c3c", "#9b59b6"]

    # TTFT chart
    html += '<div class="chart-container"><h2>Mean TTFT vs Request Rate</h2><canvas id="ttftChart" height="80"></canvas></div>'
    html += '<div class="chart-container"><h2>P99 TTFT vs Request Rate</h2><canvas id="p99Chart" height="80"></canvas></div>'
    html += '<div class="chart-container"><h2>SLO Attainment (10s) vs Request Rate</h2><canvas id="sloChart" height="80"></canvas></div>'

    # Data table
    html += '<div class="chart-container"><h2>Data Table</h2><table><tr><th>Batch Tokens</th><th>RPS</th><th>Mean TTFT (s)</th><th>P50 (s)</th><th>P90 (s)</th><th>P95 (s)</th><th>P99 (s)</th><th>SLO%</th></tr>'
    for bt in bts:
        for r in sorted([x for x in baselines if x["bt"] == bt], key=lambda x: x["rps"]):
            html += f"<tr><td>{r['bt']}</td><td>{r['rps']}</td><td>{r['ttft_mean']:.2f}</td><td>{r['ttft_p50']:.2f}</td><td>{r['ttft_p90']:.2f}</td><td>{r['ttft_p95']:.2f}</td><td>{r['ttft_p99']:.2f}</td><td>{r['slo']:.1f}</td></tr>"
    html += "</table></div>"

    # Build datasets for Chart.js
    labels = sorted(set(r["rps"] for r in baselines))
    def make_dataset(label, key, color):
        data = []
        for rps in labels:
            vals = [r[key] for r in baselines if r["rps"] == rps and r["bt"] == bt]
            data.append(vals[0] if vals else None)
        return f"{{label:'{label}',data:{json.dumps(data)},borderColor:'{color}',fill:false}}"

    # TTFT per bt
    ttft_ds = []
    for i, bt in enumerate(bts):
        data = []
        for rps in labels:
            vals = [r["ttft_mean"] for r in baselines if r["rps"] == rps and r["bt"] == bt]
            data.append(round(vals[0], 3) if vals else None)
        ttft_ds.append(f"{{label:'bt={bt}',data:{json.dumps(data)},borderColor:'{colors[i%len(colors)]}',fill:false}}")

    p99_ds = []
    for i, bt in enumerate(bts):
        data = []
        for rps in labels:
            vals = [r["ttft_p99"] for r in baselines if r["rps"] == rps and r["bt"] == bt]
            data.append(round(vals[0], 3) if vals else None)
        p99_ds.append(f"{{label:'bt={bt}',data:{json.dumps(data)},borderColor:'{colors[i%len(colors)]}',fill:false}}")

    slo_ds = []
    for i, bt in enumerate(bts):
        data = []
        for rps in labels:
            vals = [r["slo"] for r in baselines if r["rps"] == rps and r["bt"] == bt]
            data.append(round(vals[0], 1) if vals else None)
        slo_ds.append(f"{{label:'bt={bt}',data:{json.dumps(data)},borderColor:'{colors[i%len(colors)]}',fill:false}}")

    html += f"""
<script>
const labels = {json.dumps([f'{rps} RPS' for rps in labels])};
new Chart(document.getElementById('ttftChart'), {{type:'line',data:{{labels,datasets:[{','.join(ttft_ds)}]}},options:{{responsive:true,scales:{{y:{{title:{{display:true,text:'Mean TTFT (s)'}}}}}}}}}});
new Chart(document.getElementById('p99Chart'), {{type:'line',data:{{labels,datasets:[{','.join(p99_ds)}]}},options:{{responsive:true,scales:{{y:{{title:{{display:true,text:'P99 TTFT (s)'}}}}}}}}}});
new Chart(document.getElementById('sloChart'), {{type:'line',data:{{labels,datasets:[{','.join(slo_ds)}]}},options:{{responsive:true,scales:{{y:{{title:{{display:true,text:'SLO %'}},min:0,max:100}}}}}}}});
</script></body></html>"""

    out_path = RESULTS_DIR / "report.html"
    out_path.write_text(html)
    print(f"Report written to {out_path}")

if __name__ == "__main__":
    main()