#!/usr/bin/env python3
"""Gantt overlay: one horizontal segment per request (send -> completed),
x = time (s), y = request id, systems overlaid with alpha.

Queue portion (send -> first token) drawn lighter; service portion
(first token -> completed) drawn darker, same hue.

Usage: python3 xnode32_gantt_overlay.py <result_dir>
"""
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def rid_num(rid) -> int:
    if isinstance(rid, int):
        return rid
    return int(re.search(r"(\d+)$", rid).group(1))

SYSTEMS = [  # (key, label, color)
    ("afd_mbt65536",  "AFD (attn DP6TP4 mbt=65536 + FFN DP8EP8)", "#0072B2"),
    ("base_mbt8192",  "baseline 2xDP4TP4EP16 mbt=8192",           "#E69F00"),
    ("base_mbt32768", "baseline 2xDP4TP4EP16 mbt=32768",          "#009E73"),
]
LOADS = [("1x", "1x"), ("1.5x", "fast1p5x"), ("2x", "fast2x")]


def load_run(result_dir: Path, system: str, load_key: str):
    if system.startswith("afd"):
        name = {"1x": "xnode32_mbt65536.json",
                "1.5x": "xnode32_mbt65536_fast1p5x.json",
                "2x": "xnode32_mbt65536_fast2x.json"}[load_key]
        path = result_dir / name
    else:
        mbt = system.split("mbt")[1]
        suffix = {"1x": "1x", "1.5x": "fast1p5x", "2x": "fast2x"}[load_key]
        path = result_dir / "baseline2x" / f"base2x_mbt{mbt}_{suffix}.json"
    d = json.load(open(path))
    return [r for r in d["requests"] if r["success"]]


def main():
    result_dir = Path(sys.argv[1])
    fig, axes = plt.subplots(1, 3, figsize=(19, 7), sharey=False)
    for ax, (load_label, _lk) in zip(axes, LOADS):
        for sys_key, sys_label, color in SYSTEMS:
            reqs = load_run(result_dir, sys_key, load_label)
            for r in reqs:
                y = rid_num(r["request_id"])
                t0 = r["actual_send_s"]
                t1 = t0 + r["ttft_s"]
                t2 = r["completed_s"]
                ax.plot([t0, t1], [y, y], color=color, alpha=0.18, lw=1.4,
                        solid_capstyle="butt")
                ax.plot([t1, t2], [y, y], color=color, alpha=0.55, lw=1.4,
                        solid_capstyle="butt")
        ax.set_title(f"load = {load_label}", fontsize=12)
        ax.set_xlabel("time in window (s)")
        ax.grid(axis="x", alpha=0.25)
        ax.set_xlim(left=0)
        ax.set_ylim(0, 520)
    axes[0].set_ylabel("request id (arrival order)")
    handles = [Line2D([], [], color=c, lw=2.5, label=l)
               for _k, l, c in SYSTEMS]
    handles += [Line2D([], [], color="gray", lw=2.5, alpha=0.18,
                       label="queue wait (send -> TTFT)"),
                Line2D([], [], color="gray", lw=2.5, alpha=0.7,
                       label="serving (TTFT -> done)")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Per-request Gantt overlay: send -> TTFT (light) -> done (dark)",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    out = result_dir / "gantt_overlay.png"
    fig.savefig(out, dpi=140)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
