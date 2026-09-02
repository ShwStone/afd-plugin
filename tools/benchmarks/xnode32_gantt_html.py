#!/usr/bin/env python3
"""Build a single-file interactive Gantt HTML for per-request comparison.

X = time in window (s), Y = request (arrival order). Each request is a
horizontal segment: send -> TTFT (light) -> done (dark). Systems overlaid
with alpha; hover a row to see that request's TTFT/E2E under every config.

Usage: python3 xnode32_gantt_html.py <result_dir>
Output: <result_dir>/gantt_interactive.html
"""
import json
import re
import sys
from pathlib import Path

SYSTEMS = {  # key -> (label, color)
    "afd_mbt65536":  ("AFD attnDP6TP4+ffnDP8EP8 mbt=65536", "#0072B2"),
    "base_mbt8192":  ("baseline 2xDP4TP4EP16 mbt=8192",     "#E69F00"),
    "base_mbt32768": ("baseline 2xDP4TP4EP16 mbt=32768",    "#009E73"),
    # 1x-only, from the AFD mbt sweep (single logical server)
    "afd_mbt8192":   ("AFD mbt=8192 (1x only)",             "#7570B3"),
    "afd_mbt16384":  ("AFD mbt=16384 (1x only)",            "#66A61E"),
    "afd_mbt32768":  ("AFD mbt=32768 (1x only)",            "#E7298A"),
}
DEFAULT_ON = ["afd_mbt65536", "base_mbt8192", "base_mbt32768"]

RUNS = {  # (system, load) -> relative path
    ("afd_mbt65536", "1x"):   "xnode32_mbt65536.json",
    ("afd_mbt65536", "1.5x"): "xnode32_mbt65536_fast1p5x.json",
    ("afd_mbt65536", "2x"):   "xnode32_mbt65536_fast2x.json",
    ("afd_mbt8192", "1x"):    "xnode32_mbt8192.json",
    ("afd_mbt16384", "1x"):   "xnode32_mbt16384.json",
    ("afd_mbt32768", "1x"):   "xnode32_mbt32768.json",
    ("base_mbt8192", "1x"):   "baseline2x/base2x_mbt8192_1x.json",
    ("base_mbt8192", "1.5x"): "baseline2x/base2x_mbt8192_fast1p5x.json",
    ("base_mbt8192", "2x"):   "baseline2x/base2x_mbt8192_fast2x.json",
    ("base_mbt32768", "1x"):  "baseline2x/base2x_mbt32768_1x.json",
    ("base_mbt32768", "1.5x"): "baseline2x/base2x_mbt32768_fast1p5x.json",
    ("base_mbt32768", "2x"):  "baseline2x/base2x_mbt32768_fast2x.json",
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
            "eff": round(s["completion_token_throughput"]),
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
         background: #fafafa; color: #222; }
  #bar { display: flex; gap: 18px; align-items: center; flex-wrap: wrap;
         margin-bottom: 8px; }
  .tab { padding: 3px 12px; border: 1px solid #bbb; border-radius: 4px;
         cursor: pointer; background: #fff; }
  .tab.active { background: #222; color: #fff; border-color: #222; }
  .sys { cursor: pointer; user-select: none; }
  .sys input { vertical-align: -2px; }
  .swatch { display: inline-block; width: 10px; height: 10px;
            border-radius: 2px; margin: 0 3px 0 6px; }
  #wrap { position: relative; }
  canvas { background: #fff; border: 1px solid #ddd; display: block; }
  #tip { position: absolute; display: none; pointer-events: none;
         background: rgba(20,20,20,.92); color: #eee; padding: 8px 10px;
         border-radius: 6px; font-size: 12px; z-index: 5; max-width: 460px; }
  #tip table { border-collapse: collapse; margin-top: 4px; }
  #tip td, #tip th { padding: 1px 7px; text-align: right; }
  #tip td:first-child, #tip th:first-child { text-align: left; }
  #sum { margin-top: 10px; border-collapse: collapse; background: #fff; }
  #sum td, #sum th { border: 1px solid #ddd; padding: 3px 10px;
                     text-align: right; }
  #sum td:first-child, #sum th:first-child { text-align: left; }
  .note { color: #777; font-size: 12px; margin-top: 6px; }
</style>
</head>
<body>
<div id="bar">
  <span>负载:</span>
  <span id="tabs"></span>
  <span style="margin-left:24px">配置:</span>
  <span id="sysboxes"></span>
</div>
<div id="wrap">
  <canvas id="cv"></canvas>
  <div id="tip"></div>
</div>
<table id="sum"></table>
<div class="note">线段 = 每个请求：浅色段 send→TTFT（排队），深色段 TTFT→完成（服务）。
横轴秒，纵轴请求按到达顺序。鼠标悬停某一行查看该请求在各配置下的明细。</div>
<script>
const P = __PAYLOAD__;
const cv = document.getElementById("cv"), ctx = cv.getContext("2d");
const tip = document.getElementById("tip");
const W = 1860, H = 660, PADL = 46, PADR = 12, PADT = 10, PADB = 30;
const dpr = window.devicePixelRatio || 1;
cv.width = W * dpr; cv.height = H * dpr;
cv.style.width = W + "px"; cv.style.height = H + "px";
ctx.scale(dpr, dpr);

let curLoad = "1x";
const visible = {};
for (const k in P.systems) visible[k] = P.defaultOn.includes(k);

// order: by request id (arrival order is identical across runs)
function ids(load) {
  const any = Object.values(P.data[load])[0];
  return Object.keys(any).map(Number).sort((a, b) => a - b);
}

function draw() {
  ctx.clearRect(0, 0, W, H);
  const IDS = ids(curLoad);
  const n = IDS.length;
  const rowH = (H - PADT - PADB) / n;
  let xmax = 0;
  for (const k in P.data[curLoad])
    if (visible[k])
      for (const id of IDS) {
        const r = P.data[curLoad][k][id];
        if (r) xmax = Math.max(xmax, r[0] + r[2]);
      }
  xmax = Math.ceil(xmax / 10) * 10;
  const xs = t => PADL + (t / xmax) * (W - PADL - PADR);
  const ys = i => PADT + (i + 0.5) * rowH;

  // grid + x labels
  ctx.strokeStyle = "#e4e4e4"; ctx.fillStyle = "#666";
  ctx.font = "11px sans-serif"; ctx.textAlign = "center";
  for (let t = 0; t <= xmax; t += 10) {
    ctx.beginPath(); ctx.moveTo(xs(t), PADT); ctx.lineTo(xs(t), H - PADB);
    ctx.stroke();
    ctx.fillText(t, xs(t), H - PADB + 16);
  }
  ctx.fillText("time in window (s)", PADL + (W - PADL) / 2, H - 4);

  // segments
  const lw = Math.max(1, rowH * 0.55);
  ctx.lineWidth = lw; ctx.lineCap = "butt";
  for (const k in P.data[curLoad]) {
    if (!visible[k]) continue;
    const color = P.systems[k].color;
    const recs = P.data[curLoad][k];
    IDS.forEach((id, i) => {
      const r = recs[id]; if (!r) return;
      const y = ys(i);
      ctx.globalAlpha = 0.22; ctx.strokeStyle = color;
      ctx.beginPath(); ctx.moveTo(xs(r[0]), y); ctx.lineTo(xs(r[0] + r[1]), y);
      ctx.stroke();
      ctx.globalAlpha = 0.65;
      ctx.beginPath(); ctx.moveTo(xs(r[0] + r[1]), y);
      ctx.lineTo(xs(r[0] + r[2]), y); ctx.stroke();
    });
  }
  ctx.globalAlpha = 1;
  cv._geom = { IDS, rowH, xmax, xs, ys };
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
  const b = document.createElement("span");
  b.className = "tab" + (l === curLoad ? " active" : "");
  b.textContent = l;
  b.onclick = () => {
    curLoad = l;
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
draw();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
