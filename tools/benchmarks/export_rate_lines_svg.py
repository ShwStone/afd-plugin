#!/usr/bin/env python3
"""Export the rate-sweep line charts as standalone SVGs (16:9 canvas).

Imports build_payload() from the two HTML generators (single source of truth
for data + gap config) and renders one SVG per metric:

  line_charts_20260904.html        -> svg_20260904/singlenode_<metric>.svg   (5)
  line_charts_2instances_vs_...    -> svg_20260904/twoinst_<metric>.svg      (5)

Canvas 1280x720 (16:9).  Same encodings as the HTML: color = deployment
family, dash = config, vertical dashed gap annotation from the AFD token point
to the per-point best baseline with abs + relative % (SLO: abs pp).

Usage: python3 export_rate_lines_svg.py
Output: bench_results/dsv4_afd_flash_xnode32/svg_20260904/*.svg
"""
import importlib.util
import math
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
BASE = TOOLS.parent.parent / "bench_results" / "dsv4_afd_flash_xnode32"
OUT = BASE / "svg_20260904"

W, H = 1280, 720
PADL, PADR, PADT, PADB = 94, 30, 112, 70

CHARTS = [
    ("p50", "TTFT p50（s）", "s"),
    ("p99", "TTFT p99（s）", "s"),
    ("eff", "有效吞吐（tok/s）", "tok/s"),
    ("peak", "峰值 15s 服务速率（tok/s）", "tok/s"),
    ("slo10", "10s TTFT SLO 达成率（%）", "%"),
]
PAYLOADS = [  # (module name, file stem prefix, series key order)
    ("singlenode_rate_lines_html", "singlenode",
     ["afd_token", "afd_request", "base_32k", "base_8k"]),
    ("twoinstance_lines_html", "twoinst",
     ["inst2_tok", "dp6tp4_token", "dp6tp4_request", "base2x_8k", "base2x_32k"]),
]


def load_module(name: str):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def yfmt_axis(metric: str, v: float) -> str:
    if metric in ("eff", "peak"):
        return f"{v/1000:.0f}K"
    if metric == "slo10":
        return f"{v:.0f}"
    return f"{v:g}"


def fmt_abs(metric: str, d: float) -> str:
    ad = abs(d)
    if metric == "slo10":
        return f"{ad:.1f}pp"
    if metric in ("p50", "p99"):
        return f"{ad:.2f}s" if ad < 10 else f"{ad:.1f}s"
    return f"{ad/1000:.2f}K"


def nice_max(v: float) -> float:
    if v <= 0:
        return 1.0
    exp = 10 ** math.floor(math.log10(v))
    for m in (1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10):
        if m * exp >= v:
            return m * exp
    return 10 * exp


def render_chart(payload: dict, keys: list, metric: str, title: str, unit: str) -> str:
    offered, labels = payload["offered"], payload["labels"]
    series, gap = payload["series"], payload["gap"]
    n = len(offered)
    xs = [PADL + i * (W - PADL - PADR) / (n - 1) for i in range(n)]

    vals = [c[metric] for k in keys for c in series[k] if c]
    if metric == "slo10":
        ymax = 100.0
    else:
        ymax = nice_max(max(vals) * 1.08)
    def Y(v):
        return PADT + (1 - v / ymax) * (H - PADT - PADB)

    # y ticks (5-6 divisions)
    raw = ymax / 5
    exp = 10 ** math.floor(math.log10(raw)) if raw > 0 else 1
    step = min(m * exp for m in (1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10) if m * exp >= raw)
    ticks = [i * step for i in range(int(ymax / step) + 1)]

    s = [f'<?xml version="1.0" encoding="UTF-8"?>',
         f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
         f'viewBox="0 0 {W} {H}" font-family="-apple-system,\'Segoe UI\',sans-serif">',
         f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
         f'<text x="16" y="30" font-size="20" font-weight="600" fill="#111">{esc(title)}</text>']

    # legend (wrap if narrow)
    lx, ly = 16, 54
    for k in keys:
        label = payload["labels_map"][k]
        color = payload["colors"][payload["family"][k]]
        dash = ' stroke-dasharray="7 5"' if payload["dash"][k] else ""
        est = 54 + len(label) * 8.8
        if lx + est > W - 20:
            lx, ly = 16, ly + 24
        s.append(f'<line x1="{lx}" y1="{ly}" x2="{lx+32}" y2="{ly}" stroke="{color}" '
                 f'stroke-width="2.5"{dash}/>')
        s.append(f'<circle cx="{lx+16}" cy="{ly}" r="4.2" fill="{color}"/>')
        s.append(f'<text x="{lx+40}" y="{ly+5}" font-size="15" fill="#111">{esc(label)}</text>')
        lx += est
    top = PADT if ly < 60 else PADT + 20

    # grid + y tick labels
    for tv in ticks:
        y = Y(tv)
        s.append(f'<line x1="{PADL}" y1="{y:.1f}" x2="{W-PADR}" y2="{y:.1f}" stroke="#eee"/>')
        s.append(f'<text x="{PADL-9}" y="{y+5:.1f}" font-size="14" fill="#555" '
                 f'text-anchor="end">{yfmt_axis(metric, tv)}</text>')
    # axes
    s.append(f'<line x1="{PADL}" y1="{top}" x2="{PADL}" y2="{H-PADB}" stroke="#999"/>')
    s.append(f'<line x1="{PADL}" y1="{H-PADB}" x2="{W-PADR}" y2="{H-PADB}" stroke="#999"/>')
    # x ticks (two-line label)
    for i, (lab, off) in enumerate(zip(labels, offered)):
        s.append(f'<line x1="{xs[i]:.1f}" y1="{H-PADB}" x2="{xs[i]:.1f}" y2="{H-PADB+6}" stroke="#999"/>')
        s.append(f'<text x="{xs[i]:.1f}" y="{H-PADB+24}" font-size="14" fill="#333" text-anchor="middle">'
                 f'{esc(lab)}<tspan x="{xs[i]:.1f}" dy="16">{off/1000:.1f}K</tspan></text>')
    s.append(f'<text x="{(PADL+W-PADR)/2:.0f}" y="{H-8}" font-size="14" fill="#555" text-anchor="middle">'
             f'供给速率（tok/s）</text>')
    s.append(f'<text x="16" y="{(top+H-PADB)/2:.0f}" font-size="14" fill="#555" '
             f'text-anchor="middle" transform="rotate(-90 16 {(top+H-PADB)/2:.0f})">{esc(unit)}</text>')

    # series lines + points
    for k in keys:
        color = payload["colors"][payload["family"][k]]
        dash = ' stroke-dasharray="7 5"' if payload["dash"][k] else ""
        pts = " ".join(f"{xs[i]:.1f},{Y(c[metric]):.1f}" for i, c in enumerate(series[k]) if c)
        s.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2.5"{dash}/>')
        for i, c in enumerate(series[k]):
            if c:
                s.append(f'<circle cx="{xs[i]:.1f}" cy="{Y(c[metric]):.1f}" r="4.2" fill="{color}"/>')

    # gap annotation: AFD token vs per-point best baseline
    akey = gap["afdKey"]
    ai = keys.index(akey)
    for i in range(n):
        a = series[akey][i]
        if not a:
            continue
        base = None
        for bk in gap["baseKeys"]:
            b = series[bk][i]
            if not b:
                continue
            if not base or (gap["better"][metric] == "high" and b[metric] > base[metric]) or \
               (gap["better"][metric] == "low" and b[metric] < base[metric]):
                base = b
        if not base:
            continue
        diff = a[metric] - base[metric]
        rel = diff / base[metric] * 100
        ya, yb = Y(a[metric]), Y(base[metric])
        s.append(f'<line x1="{xs[i]:.1f}" y1="{ya:.1f}" x2="{xs[i]:.1f}" y2="{yb:.1f}" '
                 f'stroke="#666" stroke-width="1.2" stroke-dasharray="4 4"/>')
        txt = f"Δ {'+' if diff >= 0 else '−'}{fmt_abs(metric, diff)}"
        if metric not in gap["noRel"]:
            txt += f" ({'+' if rel >= 0 else '−'}{abs(rel):.1f}%)"
        w_est = len(txt) * 7.8 + 12
        bx = xs[i] + 8 if xs[i] + 8 + w_est < W - PADR else xs[i] - 8 - w_est
        my = min(max((ya + yb) / 2, top + 10), H - PADB - 10)
        s.append(f'<rect x="{bx:.1f}" y="{my-11:.1f}" width="{w_est:.0f}" height="22" rx="4" '
                 f'fill="#ffffff" fill-opacity="0.9" stroke="#bbb"/>')
        s.append(f'<text x="{bx+6:.1f}" y="{my+5:.1f}" font-size="13" fill="#111">{esc(txt)}</text>')

    s.append("</svg>")
    return "\n".join(s)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    for mod_name, prefix, keys in PAYLOADS:
        payload = load_module(mod_name).build_payload()
        for metric, title, unit in CHARTS:
            svg = render_chart(payload, keys, metric, title, unit)
            dst = OUT / f"{prefix}_{metric}.svg"
            dst.write_text(svg, encoding="utf-8")
            print(f"saved {dst} ({dst.stat().st_size // 1024} KiB)")


if __name__ == "__main__":
    main()
