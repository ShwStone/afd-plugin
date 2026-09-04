#!/usr/bin/env python3
"""Rate-sweep line charts for the 2026-09-04 single-node comparison.

Five charts (p50 TTFT / p99 TTFT / eff throughput / peak 15s throughput /
10s TTFT SLO rate), X = offered request rate (tok/s), 4 lines each:
  color   = family (blue=AFD, orange=baseline)
  dash    = config (solid = family best: AFD token-split & baseline-32k;
            dashed = AFD request-split & baseline-8k)

Usage: python3 singlenode_rate_lines_html.py <result_dir>
Output: <result_dir>/line_charts_20260904.html
"""
import json
import sys
from pathlib import Path

SW = Path(__file__).resolve().parent
BASE = SW.parent.parent / "bench_results" / "dsv4_afd_flash_xnode32"

OFFERED = [17535, 26303, 35071, 43839, 52606]
RATE_LABELS = ["0.5x", "0.75x", "1x", "1.25x", "1.5x"]

SERIES = {  # key -> (files relative to BASE)
    "afd_token": ["dp4tp2_single_sweeps/dp4tp2tok_mbt65536_0p5x.json",
                  "dp4tp2_single_sweeps/dp4tp2tok_mbt65536_0p75x.json",
                  "dp4tp2_single_sweeps/dp4tp2tok_mbt65536_1x.json",
                  "dp4tp2_single_sweeps/dp4tp2tok_mbt65536_1p25x.json",
                  "dp4tp2_single_sweeps/dp4tp2tok_mbt65536_1p5x.json"],
    "afd_request": ["dp4tp2_singlenode/singlenode_dp4tp2_mbt65536_slow2x.json",
                    "dp4tp2_singlenode/singlenode_dp4tp2_mbt65536_slow1p33x.json",
                    "dp4tp2_singlenode/singlenode_dp4tp2_mbt65536_1x.json",
                    "dp4tp2_single_sweeps/dp4tp2req_mbt65536_1p25x.json",
                    "dp4tp2_single_sweeps/dp4tp2req_mbt65536_1p5x.json"],
    "base_32k": [f"dp4tp2_single_sweeps/base1xas_mbt32768_{t}.json"
                 for t in ["0p5x", "0p75x", "1x", "1p25x", "1p5x"]],
    "base_8k": [f"dp4tp2_single_sweeps/base1xas_mbt8192_{t}.json"
                for t in ["0p5x", "0p75x", "1x", "1p25x", "1p5x"]],
}
COLORS = {"AFD": "#0072B2", "baseline": "#E69F00"}  # blue vs orange, CVD-safe
LABELS = {
    "afd_token": "AFD token-split 65536",
    "afd_request": "AFD request-split 65536",
    "base_32k": "baseline mbt=32768",
    "base_8k": "baseline mbt=8192",
}
FAMILY = {"afd_token": "AFD", "afd_request": "AFD", "base_32k": "baseline", "base_8k": "baseline"}
DASH = {"afd_token": [], "afd_request": [7, 5], "base_32k": [], "base_8k": [7, 5]}  # [] = solid


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
        "labels": RATE_LABELS,
        "series": {k: [metrics(BASE / f) for f in files] for k, files in SERIES.items()},
        "colors": COLORS,
        "labels_map": LABELS,
        "family": FAMILY,
        "dash": DASH,
    }
    html = TEMPLATE.replace("__PAYLOAD__", json.dumps(out, separators=(",", ":")))
    dst = BASE / "line_charts_20260904.html"
    dst.write_text(html)
    print(f"saved {dst} ({dst.stat().st_size // 1024} KiB)")


TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>单机 16 卡速率扫描 — AFD vs baseline（2026-09-04）</title>
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
  .legend-note { color: #555; font-size: 12px; margin-top: 2px; }
</style>
</head>
<body>
<h1>单机 16 卡速率扫描：AFD (DP4TP2+EP8, mbt=65536) vs baseline (DP4TP4EP16)</h1>
<div class="sub">formal_1（512 请求，长 prefill），async-sched ON，util 0.80，2026-09-04。
颜色 = 系统（<b style="color:#0072B2">AFD 蓝</b> / <b style="color:#B7791F">baseline 橙</b>），
线型 = 配置（实线 = 各自最优：AFD token-split / baseline-32k；虚线 = AFD request-split / baseline-8k）。
横轴为供给速率（tok/s）。DP8TP2 不含（32k OOM 缺格）。</div>
<div id="charts"></div>
<script>
const P = __PAYLOAD__;
const KEYS = ["afd_token", "afd_request", "base_32k", "base_8k"];
const CHARTS = [
  { id: "p50",  title: "TTFT p50（s）",            unit: "s",    yfmt: v => v.toFixed(1) },
  { id: "p99",  title: "TTFT p99（s）",            unit: "s",    yfmt: v => v.toFixed(0) },
  { id: "eff",  title: "有效吞吐（tok/s）",         unit: "tok/s", yfmt: v => (v/1000).toFixed(0) + "K" },
  { id: "peak", title: "峰值 15s 服务速率（tok/s）", unit: "tok/s", yfmt: v => (v/1000).toFixed(0) + "K" },
  { id: "slo10",title: "10s TTFT SLO 达成率（%）",   unit: "%",    yfmt: v => v.toFixed(0) },
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
        data: P.series[k].map(m => m[c.id]),
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
        y: {
          title: { display: true, text: c.unit },
          grid: { color: "#eee" },
        },
      },
      plugins: {
        legend: { position: "top", labels: { usePointStyle: true, pointStyle: "line", font: { size: 12 } } },
        tooltip: {
          callbacks: {
            title: items => `供给 ${P.labels[items[0].dataIndex]}（${P.offered[items[0].dataIndex].toLocaleString()} tok/s）`,
            label: item => ` ${item.dataset.label}: ${c.yfmt(item.parsed.y)} ${c.unit}`,
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
