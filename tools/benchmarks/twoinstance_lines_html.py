#!/usr/bin/env python3
"""Rate-sweep line charts: 2x independent single-node AFD instances (DP4TP2
token-split behind least_load_router) vs the 32-card dual-node dp6tp4 AFD and
the dual-instance DP4TP4 baseline — 1x / 1.5x / 2x, formal_1.

Color = deployment family, dash = config within family:
  blue   = DP4TP2+EP8 16c instances via router (solid)
  orange = dp6tp4+DP8EP8 dual-node AFD (solid = token-split, dashed = request)
  green  = 2xDPR4TP4EP16 baseline2x (solid = mbt8192, dashed = mbt32768)

Usage: python3 twoinstance_lines_html.py
Output: bench_results/dsv4_afd_flash_xnode32/line_charts_2instances_vs_dp6tp4.html
"""
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent.parent / "bench_results" / "dsv4_afd_flash_xnode32"

OFFERED = [35071, 52606, 70142]
LABELS = ["1x", "1.5x", "2x"]
SERIES = {
    "inst2_tok": [f"singlenode2x/singlenode2x_dp4tp2tok_{t}.json" for t in ["1x", "fast1p5x", "fast2x"]],
    "dp6tp4_token": [f"async_sched/xnode32_token_as_mbt65536_{t}.json" for t in ["1x", "fast1p5x", "fast2x"]],
    "dp6tp4_request": [f"async_sched/xnode32_request_as_mbt65536_{t}.json" for t in ["1x", "fast1p5x", "fast2x"]],
    "base2x_8k": [f"async_sched/base2x_as_mbt8192_{t}.json" for t in ["1x", "fast1p5x", "fast2x"]],
    "base2x_32k": [f"async_sched/base2x_as_mbt32768_{t}.json" for t in ["1x", "fast1p5x", "fast2x"]],
}
LABELS_MAP = {
    "inst2_tok": "2×DP4TP2+EP8 实例+router（token）",
    "dp6tp4_token": "DP6TP4 双机 token",
    "dp6tp4_request": "DP6TP4 双机 request",
    "base2x_8k": "baseline2x mbt=8192",
    "base2x_32k": "baseline2x mbt=32768",
}
FAMILY = {"inst2_tok": "blue",
          "dp6tp4_token": "orange", "dp6tp4_request": "orange",
          "base2x_8k": "green", "base2x_32k": "green"}
DASH = {"inst2_tok": [],
        "dp6tp4_token": [], "dp6tp4_request": [7, 5],
        "base2x_8k": [], "base2x_32k": [7, 5]}
COLORS = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73"}


def metrics(path: Path) -> dict:
    d = json.load(open(path))
    s = d["summary"]
    reqs = [r for r in d["requests"] if r["success"]]
    buckets = {}
    for r in reqs:
        b = int(r["completed_s"] // 15)
        buckets[b] = buckets.get(b, 0) + int(r.get("prompt_tokens_reported") or r["input_length"])
    return {
        "p50": s["ttft_s"]["p50"],
        "p99": s["ttft_s"]["p99"],
        "eff": s["completion_token_throughput"],
        "peak": (max(buckets.values()) / 15) if buckets else 0,
        "slo10": sum(1 for r in reqs if r["ttft_s"] <= 10.0) / len(reqs) * 100,
    }


def main() -> None:
    out = {
        "offered": OFFERED,
        "labels": LABELS,
        "series": {k: [metrics(BASE / f) if f else None for f in files]
                   for k, files in SERIES.items()},
        "colors": COLORS,
        "labels_map": LABELS_MAP,
        "family": FAMILY,
        "dash": DASH,
    }
    html = TEMPLATE.replace("__PAYLOAD__", json.dumps(out, separators=(",", ":")))
    dst = BASE / "line_charts_2instances_vs_dp6tp4.html"
    dst.write_text(html)
    print(f"saved {dst} ({dst.stat().st_size // 1024} KiB)")


TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>2×单机实例 vs 双机 DP6TP4 vs 双机 baseline（2026-09-04）</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
  body { font: 13px/1.45 -apple-system, "Segoe UI", sans-serif; margin: 18px;
         background: #fafafa; color: #111; }
  h1 { font-size: 17px; margin: 0 0 4px; }
  .sub { color: #555; font-size: 12px; margin-bottom: 14px; }
  .panel { background: #fff; border: 1px solid #ddd; border-radius: 8px;
           padding: 14px 16px 10px; margin-bottom: 16px; }
  .panel h2 { font-size: 14px; margin: 0 0 8px; }
  .chartbox { position: relative; height: 340px; }
</style>
</head>
<body>
<h1>实例级扩展对比（formal_1,32 卡总预算）：2×单机 AFD 实例 vs 双机 DP6TP4 vs 双机 baseline</h1>
<div class="sub">async-sched ON，CWS，util 0.80。颜色 = 部署形态
（<b style="color:#0072B2">蓝 = DP4TP2+EP8 单机实例</b> /
<b style="color:#B7791F">橙 = DP6TP4+EP8 双机单实例</b> /
<b style="color:#1e7d64">绿 = 2×DP4TP4EP16 baseline</b>），
线型 = 配置（实线/虚线见 图例）。横轴为供给速率（tok/s）。
router 分流实测 766:772。</div>
<div id="charts"></div>
<script>
const P = __PAYLOAD__;
const KEYS = ["inst2_tok", "dp6tp4_token", "dp6tp4_request", "base2x_8k", "base2x_32k"];
const CHARTS = [
  { id: "p50",  title: "TTFT p50（s）",             unit: "s",    yfmt: v => v.toFixed(1) },
  { id: "p99",  title: "TTFT p99（s）",             unit: "s",    yfmt: v => v.toFixed(0) },
  { id: "eff",  title: "有效吞吐（tok/s）",          unit: "tok/s", yfmt: v => (v/1000).toFixed(0) + "K" },
  { id: "peak", title: "峰值 15s 服务速率（tok/s）",  unit: "tok/s", yfmt: v => (v/1000).toFixed(0) + "K" },
  { id: "slo10",title: "10s TTFT SLO 达成率（%）",    unit: "%",    yfmt: v => v.toFixed(0) },
];
const wrap = document.getElementById("charts");
for (const c of CHARTS) {
  const panel = document.createElement("div");
  panel.className = "panel";
  panel.innerHTML = `<h2>${c.title}</h2><div class="chartbox"><canvas id="cv_${c.id}"></canvas></div>`;
  wrap.appendChild(panel);
  new Chart(document.getElementById(`cv_${c.id}`), {
    type: "line",
    data: {
      labels: P.offered,
      datasets: KEYS.map(k => ({
        label: P.labels_map[k],
        data: P.series[k].map(m => m ? m[c.id] : null),
        borderColor: P.colors[P.family[k]],
        backgroundColor: P.colors[P.family[k]],
        borderDash: P.dash[k],
        borderWidth: 2,
        pointRadius: 4,
        pointHoverRadius: 6,
        tension: 0.15,
        spanGaps: false,
      })),
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "nearest", intersect: false },
      scales: {
        x: {
          title: { display: true, text: "供给速率（tok/s）" },
          ticks: { callback: function(v, i) { return P.labels[i] + "\n" + (P.offered[i]/1000) + "K"; } },
          grid: { color: "#eee" },
        },
        y: { title: { display: true, text: c.unit }, grid: { color: "#eee" } },
      },
      plugins: {
        legend: { position: "top", labels: { usePointStyle: true, pointStyle: "line", font: { size: 11 } } },
        tooltip: {
          callbacks: {
            title: items => `供给 ${P.labels[items[0].dataIndex]}（${P.offered[items[0].dataIndex].toLocaleString()} tok/s）`,
            label: item => (item.parsed.y == null ? null : ` ${item.dataset.label}: ${c.yfmt(item.parsed.y)} ${c.unit}`),
          },
        },
      },
    },
  });
}
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
