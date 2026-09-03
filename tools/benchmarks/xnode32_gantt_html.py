#!/usr/bin/env python3
"""Build a single-file interactive Gantt HTML for per-request comparison.

X = time in window (s), Y = request (arrival order). Each request is a
horizontal segment: send -> TTFT (light) -> done (dark). Systems overlaid
with alpha; hover a row to see that request's TTFT/E2E under every config.

Navigation (Perfetto-style): W/S zoom in/out, A/D pan left/right, R reset,
mouse wheel zooms around the cursor.

Usage: python3 xnode32_gantt_html.py <result_dir>
Output: <result_dir>/gantt_interactive.html
"""
import json
import re
import sys
from pathlib import Path

SYSTEMS = {  # key -> (label, color) — high-contrast categorical set
    "afd_req":      ("AFD 24+8 request-split mbt=65536", "#0072B2"),
    "afd_tok":      ("AFD 24+8 token-split mbt=65536",   "#D55E00"),
    "afd_off":      ("AFD 24+8 split-off mbt=65536",     "#009E73"),
    "base2x_8k":    ("baseline 2×DP4TP4EP16 mbt=8192",   "#E69F00"),
    "base2x_32k":   ("baseline 2×DP4TP4EP16 mbt=32768",  "#CC79A7"),
    # Virtual: measured 1x TTFT held fixed, arrivals compressed 2x (what-if).
    "b2x8k_dbl":    ("baseline 2×…8k @1x→2x到达(TTFT不变)",  "#E7298A"),
    "b2x32k_dbl":   ("baseline 2×…32k @1x→2x到达(TTFT不变)", "#A6761D"),
    # 2026-09-03 async-scheduler redo (all FLASHCOMM1+CWS+async-sched ON)
    "afd_req_as":    ("AFD request-split async",          "#88CCEE"),
    "afd_tok_as":    ("AFD token-split async",            "#DDCC77"),
    "afd_toklb_as":  ("AFD token-split+tokDPLB async",    "#AA4499"),
    "afd_off_as":    ("AFD split-off async",              "#44AA99"),
    "base2x_8k_as":  ("baseline 2×…8k async",             "#999933"),
    "base2x_32k_as": ("baseline 2×…32k async",            "#CC6677"),
}
DEFAULT_ON = ["afd_req_as", "afd_tok_as", "afd_toklb_as", "afd_off_as"]

# Virtual doubled-arrival systems: key -> (source system, source load).
# send times halved, TTFT/E2E durations unchanged; rendered in the 2x tab.
DOUBLED = {
    "b2x8k_dbl":  ("base2x_8k", "1x"),
    "b2x32k_dbl": ("base2x_32k", "1x"),
}
DOUBLE_WINDOW_S = 75.0  # formal_1 at 2x arrival speed

RUNS = {  # (system, load) -> relative path
    ("afd_req", "1x"):   "xnode32_mbt65536.json",
    ("afd_req", "1.5x"): "xnode32_mbt65536_fast1p5x.json",
    ("afd_req", "2x"):   "xnode32_mbt65536_fast2x.json",
    ("afd_tok", "1x"):   "split_sweep/xnode32_token_mbt65536_1x.json",
    ("afd_tok", "1.5x"): "split_sweep/xnode32_token_mbt65536_fast1p5x.json",
    ("afd_tok", "2x"):   "split_sweep/xnode32_token_mbt65536_fast2x.json",
    ("afd_off", "1x"):   "split_sweep/xnode32_off_mbt65536_1x.json",
    ("afd_off", "1.5x"): "split_sweep/xnode32_off_mbt65536_fast1p5x.json",
    ("afd_off", "2x"):   "split_sweep/xnode32_off_mbt65536_fast2x.json",
    ("base2x_8k", "1x"):   "baseline2x/base2x_mbt8192_1x.json",
    ("base2x_8k", "1.5x"): "baseline2x/base2x_mbt8192_fast1p5x.json",
    ("base2x_8k", "2x"):   "baseline2x/base2x_mbt8192_fast2x.json",
    ("base2x_32k", "1x"):   "baseline2x/base2x_mbt32768_1x.json",
    ("base2x_32k", "1.5x"): "baseline2x/base2x_mbt32768_fast1p5x.json",
    ("base2x_32k", "2x"):   "baseline2x/base2x_mbt32768_fast2x.json",
    ("afd_req_as", "1x"):   "async_sched/xnode32_request_as_mbt65536_1x.json",
    ("afd_req_as", "1.5x"): "async_sched/xnode32_request_as_mbt65536_fast1p5x.json",
    ("afd_req_as", "2x"):   "async_sched/xnode32_request_as_mbt65536_fast2x.json",
    ("afd_tok_as", "1x"):   "async_sched/xnode32_token_as_mbt65536_1x.json",
    ("afd_tok_as", "1.5x"): "async_sched/xnode32_token_as_mbt65536_fast1p5x.json",
    ("afd_tok_as", "2x"):   "async_sched/xnode32_token_as_mbt65536_fast2x.json",
    ("afd_toklb_as", "1x"):   "async_sched/toklb_as_1x.json",
    ("afd_toklb_as", "1.5x"): "async_sched/toklb_as_fast1p5x.json",
    ("afd_toklb_as", "2x"):   "async_sched/toklb_as_fast2x.json",
    ("afd_off_as", "1x"):   "async_sched/xnode32_off_as_mbt65536_1x.json",
    ("afd_off_as", "1.5x"): "async_sched/xnode32_off_as_mbt65536_fast1p5x.json",
    ("afd_off_as", "2x"):   "async_sched/xnode32_off_as_mbt65536_fast2x.json",
    ("base2x_8k_as", "1x"):   "async_sched/base2x_as_mbt8192_1x.json",
    ("base2x_8k_as", "1.5x"): "async_sched/base2x_as_mbt8192_fast1p5x.json",
    ("base2x_8k_as", "2x"):   "async_sched/base2x_as_mbt8192_fast2x.json",
    ("base2x_32k_as", "1x"):   "async_sched/base2x_as_mbt32768_1x.json",
    ("base2x_32k_as", "1.5x"): "async_sched/base2x_as_mbt32768_fast1p5x.json",
    ("base2x_32k_as", "2x"):   "async_sched/base2x_as_mbt32768_fast2x.json",
}

LOADS = ["1x", "1.5x", "2x"]


def rid_num(rid: str) -> int:
    m = re.search(r"(\d+)$", rid)
    return int(m.group(1))


def main():
    result_dir = Path(sys.argv[1])
    # data[load][syskey] = {rid_num: [send, ttft, e2e, input_length]}
    data = {l: {} for l in LOADS}
    summaries = {l: {} for l in LOADS}
    for (syskey, load), rel in RUNS.items():
        d = json.load(open(result_dir / rel))
        s = d["summary"]
        summaries[load][syskey] = {
            "eff": round(s["total_input_tokens"] / s["wall_s"]),
            "drain": round(s["queue_drain_s"], 1),
            "p50": round(s["ttft_s"]["p50"], 2),
            "p99": round(s["ttft_s"]["p99"], 2),
        }
        recs = {}
        for r in d["requests"]:
            if not r["success"]:
                continue
            recs[rid_num(r["request_id"])] = [
                round(r["actual_send_s"], 3),
                round(r["ttft_s"], 3),
                round(r["e2el_s"], 3),
                r["input_length"],
            ]
        data[load][syskey] = recs

    # Build virtual doubled-arrival systems into the "2x" tab.
    for vkey, (srckey, srcload) in DOUBLED.items():
        src = data[srcload][srckey]
        ssrc = summaries[srcload][srckey]
        recs = {rid: [round(r[0] / 2, 3), r[1], r[2], r[3]]
                for rid, r in src.items()}
        wall = max(r[0] + r[2] for r in recs.values())
        total_tokens = sum(r[3] for r in recs.values())
        data["2x"][vkey] = recs
        summaries["2x"][vkey] = {
            "eff": round(total_tokens / wall),
            "drain": round(wall - DOUBLE_WINDOW_S, 1),
            "p50": ssrc["p50"],
            "p99": ssrc["p99"],
        }

    payload = {
        "systems": {k: {"label": v[0], "color": v[1]} for k, v in SYSTEMS.items()},
        "defaultOn": DEFAULT_ON,
        "loads": LOADS,
        "data": data,
        "summaries": summaries,
    }

    html = TEMPLATE.replace("__PAYLOAD__", json.dumps(payload, separators=(",", ":")))
    out = result_dir / "gantt_interactive.html"
    out.write_text(html)
    print(f"saved {out} ({out.stat().st_size//1024} KiB)")


TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>per-request Gantt — AFD vs baseline</title>
<style>
  body { font: 13px/1.4 -apple-system, "Segoe UI", sans-serif; margin: 16px;
         background: #fafafa; color: #111; }
  #bar { display: flex; gap: 18px; align-items: center; flex-wrap: wrap;
         margin-bottom: 8px; }
  .tab { padding: 3px 12px; border: 1px solid #888; border-radius: 4px;
         cursor: pointer; background: #fff; }
  .tab.active { background: #111; color: #fff; border-color: #111; }
  .sys { cursor: pointer; user-select: none; }
  .sys input { vertical-align: -2px; }
  .swatch { display: inline-block; width: 10px; height: 10px;
            border-radius: 2px; margin: 0 3px 0 6px; }
  #wrap { position: relative; }
  canvas { background: #fff; border: 1px solid #bbb; display: block; }
  #tip { position: absolute; display: none; pointer-events: none;
         background: rgba(20,20,20,.94); color: #eee; padding: 8px 10px;
         border-radius: 6px; font-size: 12px; z-index: 5; max-width: 520px; }
  #tip table { border-collapse: collapse; margin-top: 4px; }
  #tip td, #tip th { padding: 1px 7px; text-align: right; }
  #tip td:first-child, #tip th:first-child { text-align: left; }
  #sum { margin-top: 10px; border-collapse: collapse; background: #fff; }
  #sum td, #sum th { border: 1px solid #ccc; padding: 3px 10px;
                     text-align: right; }
  #sum td:first-child, #sum th:first-child { text-align: left; }
  .note { color: #555; font-size: 12px; margin-top: 6px; }
  #viewinfo { font-family: monospace; color: #333; }
</style>
</head>
<body>
<div id="bar">
  <span>负载:</span>
  <span id="tabs"></span>
  <span style="margin-left:24px">配置:</span>
  <span id="sysboxes"></span>
  <span id="viewinfo" style="margin-left:24px"></span>
</div>
<div id="wrap">
  <canvas id="cv" tabindex="0"></canvas>
  <div id="tip"></div>
</div>
<table id="sum"></table>
<div class="note">线段 = 每个请求：浅色段 send→TTFT（排队），深色段 TTFT→完成（服务）。
横轴秒，纵轴请求按到达顺序。悬停某行看该请求各配置明细。<br>
<b>导航（Perfetto 式）：W 放大 / S 缩小 / A 左移 / D 右移 / R 重置；滚轮以鼠标为中心缩放。</b><br>
@1x→2x到达(TTFT不变) = 虚拟配置：该 baseline 1x 实测 TTFT/E2E 保持不变、到达间隔压缩一半，
放进 2x 页签与真实 2x 数据同坐标对比（假设性对标：若 baseline 在翻倍负载下能守住延迟会怎样）。</div>
<script>
const P = __PAYLOAD__;
const cv = document.getElementById("cv"), ctx = cv.getContext("2d");
const tip = document.getElementById("tip");
const viewinfo = document.getElementById("viewinfo");
const W = 1860, H = 660, PADL = 46, PADR = 12, PADT = 10, PADB = 30;
const dpr = window.devicePixelRatio || 1;
cv.width = W * dpr; cv.height = H * dpr;
cv.style.width = W + "px"; cv.style.height = H + "px";
ctx.scale(dpr, dpr);

let curLoad = "1x";
const visible = {};
for (const k in P.systems) visible[k] = P.defaultOn.includes(k);
let view = { t0: 0, t1: 1 };   // seconds window; reset on load switch

function loadXmax(load) {
  let xmax = 0;
  for (const k in P.data[load])
    for (const id in P.data[load][k]) {
      const r = P.data[load][k][id];
      xmax = Math.max(xmax, r[0] + r[2]);
    }
  return Math.ceil(xmax / 10) * 10;
}
const XMAX = {};
for (const l of P.loads) XMAX[l] = Math.max(loadXmax(l), 1);

function ids(load) {
  const any = Object.values(P.data[load])[0];
  return Object.keys(any).map(Number).sort((a, b) => a - b);
}

function niceStep(span) {
  const target = span / 12;
  const steps = [0.2, 0.5, 1, 2, 5, 10, 20, 30, 60, 120];
  for (const s of steps) if (s >= target) return s;
  return 300;
}

function draw() {
  ctx.clearRect(0, 0, W, H);
  const xmax = XMAX[curLoad];
  view.t0 = Math.max(0, Math.min(view.t0, xmax - 0.1));
  view.t1 = Math.min(xmax, Math.max(view.t1, view.t0 + 0.1));
  const span = view.t1 - view.t0;
  const IDS = ids(curLoad);
  const n = IDS.length;
  const rowH = (H - PADT - PADB) / n;
  const xs = t => PADL + ((t - view.t0) / span) * (W - PADL - PADR);
  const ys = i => PADT + (i + 0.5) * rowH;

  // grid + x labels
  const step = niceStep(span);
  ctx.strokeStyle = "#ddd"; ctx.fillStyle = "#333";
  ctx.font = "11px sans-serif"; ctx.textAlign = "center";
  const tStart = Math.ceil(view.t0 / step) * step;
  for (let t = tStart; t <= view.t1 + 1e-9; t += step) {
    ctx.beginPath(); ctx.moveTo(xs(t), PADT); ctx.lineTo(xs(t), H - PADB);
    ctx.stroke();
    ctx.fillText(+t.toFixed(1), xs(t), H - PADB + 16);
  }
  ctx.fillText("time in window (s)", PADL + (W - PADL) / 2, H - 4);

  // segments (clipped to plot area)
  ctx.save();
  ctx.beginPath(); ctx.rect(PADL, PADT, W - PADL - PADR, H - PADT - PADB);
  ctx.clip();
  const lw = Math.max(1, rowH * 0.55);
  ctx.lineWidth = lw; ctx.lineCap = "butt";
  for (const k in P.data[curLoad]) {
    if (!visible[k]) continue;
    const color = P.systems[k].color;
    const recs = P.data[curLoad][k];
    IDS.forEach((id, i) => {
      const r = recs[id]; if (!r) return;
      if (r[0] + r[2] < view.t0 || r[0] > view.t1) return;
      const y = ys(i);
      ctx.globalAlpha = 0.32; ctx.strokeStyle = color;
      ctx.beginPath(); ctx.moveTo(xs(r[0]), y); ctx.lineTo(xs(r[0] + r[1]), y);
      ctx.stroke();
      ctx.globalAlpha = 0.88;
      ctx.beginPath(); ctx.moveTo(xs(r[0] + r[1]), y);
      ctx.lineTo(xs(r[0] + r[2]), y); ctx.stroke();
    });
  }
  ctx.restore();
  ctx.globalAlpha = 1;
  cv._geom = { IDS, rowH, xs, ys };
  viewinfo.textContent = `[${view.t0.toFixed(1)}s – ${view.t1.toFixed(1)}s] / ${xmax}s`;
  drawSummary();
}

function drawSummary() {
  const el = document.getElementById("sum");
  let h = "<tr><th>config</th><th>eff tok/s</th><th>drain s</th>" +
          "<th>TTFT p50</th><th>TTFT p99</th></tr>";
  for (const k in P.summaries[curLoad]) {
    if (!visible[k]) continue;
    const s = P.summaries[curLoad][k];
    h += `<tr><td><span class="swatch" style="background:${P.systems[k].color}"></span>` +
         `${P.systems[k].label}</td><td>${s.eff}</td><td>${s.drain}</td>` +
         `<td>${s.p50}</td><td>${s.p99}</td></tr>`;
  }
  el.innerHTML = h;
}

function zoom(factor, centerT) {
  const c = centerT !== undefined ? centerT : (view.t0 + view.t1) / 2;
  let span = (view.t1 - view.t0) * factor;
  span = Math.max(0.5, Math.min(span, XMAX[curLoad]));
  let t0 = c - (c - view.t0) * (span / (view.t1 - view.t0));
  view.t0 = Math.max(0, Math.min(t0, XMAX[curLoad] - span));
  view.t1 = view.t0 + span;
  draw();
}
function pan(frac) {
  const span = view.t1 - view.t0;
  let t0 = view.t0 + span * frac;
  t0 = Math.max(0, Math.min(t0, XMAX[curLoad] - span));
  view.t0 = t0; view.t1 = t0 + span;
  draw();
}

cv.addEventListener("keydown", e => {
  const k = e.key.toLowerCase();
  if (k === "w") zoom(1 / 1.6);
  else if (k === "s") zoom(1.6);
  else if (k === "a") pan(-0.3);
  else if (k === "d") pan(0.3);
  else if (k === "r") { view = { t0: 0, t1: XMAX[curLoad] }; draw(); }
  else return;
  e.preventDefault();
});
cv.addEventListener("wheel", e => {
  const rect = cv.getBoundingClientRect();
  const frac = (e.clientX - rect.left - PADL) / (W - PADL - PADR);
  const centerT = view.t0 + Math.max(0, Math.min(1, frac)) * (view.t1 - view.t0);
  zoom(e.deltaY < 0 ? 1 / 1.25 : 1.25, centerT);
  e.preventDefault();
}, { passive: false });

cv.addEventListener("mousemove", e => {
  const g = cv._geom; if (!g) return;
  const rect = cv.getBoundingClientRect();
  const mx = e.clientX - rect.left, my = e.clientY - rect.top;
  const i = Math.floor((my - PADT) / g.rowH);
  if (i < 0 || i >= g.IDS.length) { tip.style.display = "none"; return; }
  const id = g.IDS[i];
  let len = null, rows = "";
  for (const k in P.data[curLoad]) {
    if (!visible[k]) continue;
    const r = P.data[curLoad][k][id]; if (!r) continue;
    len = r[3];
    rows += `<tr><td><span class="swatch" style="background:${P.systems[k].color}"></span>` +
            `${P.systems[k].label}</td><td>${r[0].toFixed(1)}s</td>` +
            `<td>${r[1].toFixed(2)}s</td><td>${r[2].toFixed(2)}s</td></tr>`;
  }
  if (!rows) { tip.style.display = "none"; return; }
  tip.innerHTML = `<b>#${id}</b> &nbsp; input ${len.toLocaleString()} tok` +
    `<table><tr><th>config</th><th>send</th><th>TTFT</th><th>E2E</th></tr>${rows}</table>`;
  tip.style.display = "block";
  const tw = tip.offsetWidth;
  tip.style.left = Math.min(mx + 14, W - tw - 6) + "px";
  tip.style.top = Math.max(my - 20, 4) + "px";
});
cv.addEventListener("mouseleave", () => tip.style.display = "none");

// tabs
const tabs = document.getElementById("tabs");
for (const l of P.loads) {
  if (!Object.keys(P.data[l]).length) continue;
  const b = document.createElement("span");
  b.className = "tab" + (l === curLoad ? " active" : "");
  b.textContent = l;
  b.onclick = () => {
    curLoad = l;
    view = { t0: 0, t1: XMAX[l] };
    document.querySelectorAll(".tab").forEach(x => x.classList.remove("active"));
    b.classList.add("active");
    draw();
  };
  tabs.appendChild(b);
}
// system checkboxes
const sb = document.getElementById("sysboxes");
for (const k in P.systems) {
  const lab = document.createElement("label");
  lab.className = "sys";
  lab.innerHTML = `<input type="checkbox" ${visible[k] ? "checked" : ""}>` +
    `<span class="swatch" style="background:${P.systems[k].color}"></span>` +
    `${P.systems[k].label}`;
  lab.querySelector("input").onchange = ev => { visible[k] = ev.target.checked; draw(); };
  sb.appendChild(lab);
}
view = { t0: 0, t1: XMAX[curLoad] };
cv.focus();
draw();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
