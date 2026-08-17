# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Render Stage-3 btsweep charts (self-contained HTML via Chart.js CDN).

Reads ``btsweep_attribution.json`` (from ``analyze_btsweep``) and renders:

1. TTFT (mean/p99) vs max_num_batched_tokens — per system, per RPS (log-x).
2. tokens/s vs max_num_batched_tokens — throughput trend.
3. Occupancy/bubble attribution (busy_ratio / bubble_ratio / npu-smi AICore%)
   vs max_num_batched_tokens, only for profile cells that carry the data.

Usage:
  python3 -m tools.benchmarks.render_btsweep_charts \
    --input bench_results/prefill_stage3/03_reports/btsweep_attribution.json \
    --output bench_results/prefill_stage3/03_reports/btsweep_charts.html
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SA = "dp4_tp8_sp"
SB = "afd_dp3_tp8_ep8"
LA = "Baseline DP4xTP8"
LB = "AFD DP3xTP8+EP8"
SYS_COLOR = {SA: "#2c7fb8", SB: "#d95f0e"}
RPS_COLOR = {6.0: "#2ca02c", 8.0: "#d62728"}
RPS_ORDER = [6.0, 8.0]

# Default to the 10 first colors of a categorical palette (no hard deps).
PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
    "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


def _json_data(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _rows_by(data: dict[str, object]) -> dict[tuple[str, float], list[dict[str, object]]]:
    rows = data["rows"]
    grouped: dict[tuple[str, float], list[dict[str, object]]] = {}
    for row in rows:
        key = (str(row["system"]), float(row["rps"]))
        grouped.setdefault(key, []).append(row)
    for cell_rows in grouped.values():
        cell_rows.sort(key=lambda r: int(r["max_num_batched_tokens"]))
    return grouped


def _ds(label: str, data: list[object], color: str) -> str:
    return "{" + (
        f"label:'{label}',data:{json.dumps(data)},"
        f"borderColor:'{color}',backgroundColor:'{color}',fill:false,"
        f"pointStyle:'rectRounded'"
    ) + "}"


def _ttft_chart(grouped) -> str:
    panels = []
    for rps in RPS_ORDER:
        for metric in ("ttft_mean_ms", "ttft_p99_ms"):
            mlabel = "Mean TTFT" if metric == "ttft_mean_ms" else "P99 TTFT"
            for system in (SA, SB):
                rows = grouped.get((system, rps), [])
                if not rows:
                    continue
                labels = [int(r["max_num_batched_tokens"]) for r in rows]
                values = [round(float(r[metric]), 1) if r.get(metric) is not None else None
                          for r in rows]
                cid = f"ttft_{rps}_{metric}_{system.replace('.', '_')}"
                panels.append(f"""
<div class="card"><h3>{'Baseline' if system == SA else 'AFD'} — {mlabel} vs Max Batch (RPS={rps:g}, prefix=0 cold)</h3>
<canvas id="{cid}" height="70"></canvas>
<script>new Chart(document.getElementById('{cid}'), {{type:'line',
data:{{labels:{json.dumps(labels)},datasets:[{_ds('', values, SYS_COLOR[system])}]}},
options:{{responsive:true,scales:{{x:{{type:'logarithmic',title:{{display:true,text:'Max Batch Tokens'}}}},
y:{{title:{{display:true,text:'{mlabel} (ms)'}}}}}}}}}});</script></div>
""")
    return "\n".join(panels)


def _throughput_chart(grouped) -> str:
    panels = []
    for rps in RPS_ORDER:
        cid = f"thr_{rps}"
        labels = []
        datasets = []
        for system in (SA, SB):
            rows = grouped.get((system, rps), [])
            if not rows:
                continue
            labels = [int(r["max_num_batched_tokens"]) for r in rows]
            values = [round(float(r["tokens_per_s"]), 1) if r.get("tokens_per_s") is not None else None
                      for r in rows]
            datasets.append(_ds('Baseline' if system == SA else 'AFD', values, SYS_COLOR[system]))
        if not datasets:
            continue
        panels.append(f"""
<div class="card"><h3>Input tokens/s vs Max Batch (RPS={rps:g}, prefix=0 cold)</h3>
<canvas id="{cid}" height="70"></canvas>
<script>new Chart(document.getElementById('{cid}'), {{type:'line',
data:{{labels:{json.dumps(labels)},datasets:[{','.join(datasets)}]}},
options:{{responsive:true,scales:{{x:{{type:'logarithmic',title:{{display:true,text:'Max Batch Tokens'}}}},
y:{{title:{{display:true,text:'tokens/s'}}}}}}}}}});</script></div>
""")
    return "\n".join(panels)


def _attribution_chart(grouped) -> str:
    """Busy/bubble/occupancy vs mbt for profile cells that carry the data."""
    panels = []
    for rps in RPS_ORDER:
        for system in (SA, SB):
            rows = grouped.get((system, rps), [])
            profiled = [r for r in rows if r.get("busy_ratio") is not None]
            if not profiled:
                continue
            labels = [int(r["max_num_batched_tokens"]) for r in profiled]
            cid = f"attr_{rps}_{system.replace('.', '_')}"
            ttft = [round(float(r["ttft_mean_ms"]), 1) if r.get("ttft_mean_ms") else None
                    for r in profiled]
            busy = [round(float(r["busy_ratio"]) * 100, 1) for r in profiled]
            bubble = [round(float(r["bubble_ratio"]) * 100, 1) for r in profiled]
            occupancy = [round(float(r["occupancy_npu_smi"]), 1)
                         for r in profiled if r.get("occupancy_npu_smi") is not None]
            panels.append(f"""
<div class="card"><h3>{'Baseline' if system == SA else 'AFD'} — Attribution vs Max Batch (RPS={rps:g}, L2 profile)</h3>
<canvas id="{cid}" height="90"></canvas>
<script>
var ctx = document.getElementById('{cid}');
new Chart(ctx, {{type:'line',
data:{{labels:{json.dumps(labels)},datasets:[
 {{label:'Mean TTFT (ms)',data:{json.dumps(ttft)},yAxisID:'y1',borderColor:'#000000',fill:false,pointStyle:'rectRounded'}},
 {{label:'busy_ratio (%)',data:{json.dumps(busy)},yAxisID:'y',borderColor:'{SYS_COLOR[system]}',fill:false}},
 {{label:'bubble_ratio (%)',data:{json.dumps(bubble)},yAxisID:'y',borderColor:'#e377c2',borderDash:[6,4],fill:false}}
]}},
options:{{responsive:true,scales:{{x:{{type:'logarithmic',title:{{display:true,text:'Max Batch Tokens'}}}},
y:{{position:'left',title:{{display:true,text:'ratio (%)'}},min:0,max:100}},
y1:{{position:'right',title:{{display:true,text:'Mean TTFT (ms)'}}}}}}}}}});</script></div>
""")
            if occupancy:
                cid2 = cid + "_smi"
                panels.append(f"""
<div class="card"><h3>{'Baseline' if system == SA else 'AFD'} — npu-smi AICore% vs Max Batch (RPS={rps:g})</h3>
<canvas id="{cid2}" height="60"></canvas>
<script>new Chart(document.getElementById('{cid2}'), {{type:'bar',
data:{{labels:{json.dumps(labels)},datasets:[{_ds('AICore%', occupancy, SYS_COLOR[system])}]}},
options:{{responsive:true,scales:{{x:{{type:'logarithmic'}},y:{{title:{{display:true,text:'AICore %'}},min:0,max:100}}}}}}}}}});</script></div>
""")
    return "\n".join(panels)


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_argument_parser().parse_args(argv)
    data = _json_data(args.input)
    grouped = _rows_by(data)
    charts = (
        _ttft_chart(grouped)
        + _throughput_chart(grouped)
        + _attribution_chart(grouped)
    )
    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Stage-3 btsweep — Batch-Token Scaling Deep-Dive</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>body{{font-family:sans-serif;margin:20px;background:#fafafa;}}
.card{{background:#fff;border:1px solid #ddd;border-radius:6px;padding:14px;margin:14px 0;}}
h3{{margin:0 0 10px;font-size:15px;}}</style></head><body>
<h2>Stage-3 btsweep — Batch-Token Scaling Deep-Dive (prefix=0, cold)</h2>
{charts}
</body></html>"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
