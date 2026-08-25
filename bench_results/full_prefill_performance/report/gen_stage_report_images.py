#!/usr/bin/env python3
"""Render static PNG charts for the GitHub stage report.

Reads report/stats.json (built by fp_build_report.py) and writes PNGs to
report/images/. Labels are English (no CJK font on the render host).
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(HERE, "images")
os.makedirs(IMG, exist_ok=True)

with open(os.path.join(HERE, "stats.json")) as f:
    S = json.load(f)

C_BASE = "#d62728"
C_A2 = "#1f77b4"
C_A1 = "#2ca02c"
GREY = "#888888"

plt.rcParams.update({
    "figure.dpi": 130,
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def cap_points(sysname):
    d = S["capacity"][sysname]
    pts = [v for v in d.values()]
    pts.sort(key=lambda p: p["target_tps"])
    return pts


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, name))
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------- 1. p99 curve
fig, ax = plt.subplots(figsize=(7, 4.2))
for sysname, c, lbl in (("baseline", C_BASE, "Baseline DP4xTP8"), ("afd_a2", C_A2, "AFD A2 (token split)")):
    pts = cap_points(sysname)
    x = [p["target_tps"] for p in pts]
    p50 = [p["ttft"]["p50"] for p in pts]
    p99 = [p["ttft"]["p99"] for p in pts]
    ok = [p["slo_ok"] for p in pts]
    ax.plot(x, p99, "-o", color=c, label=f"{lbl} p99", lw=1.8)
    ax.plot(x, p50, "--s", color=c, alpha=0.55, label=f"{lbl} p50", lw=1.2)
    for xi, yi, o in zip(x, p99, ok):
        if not o:
            ax.annotate("FAIL", (xi, yi), textcoords="offset points",
                        xytext=(6, 6), fontsize=8, color=c)
ax.axhline(50, color="k", ls=":", lw=1.2)
ax.text(2300, 51, "SLO: TTFT p99 = 50s", fontsize=8)
ax.axvspan(2599, 4371, color=C_A2, alpha=0.06)
ax.set_xlabel("Offered load (input tokens/s)")
ax.set_ylabel("TTFT (s)")
ax.set_title("Capacity screening: TTFT vs offered load (128 req window)")
ax.legend(fontsize=8)
save(fig, "capacity_ttft.png")

# ---------------------------------------------------------------- 2. CDF
fig, ax = plt.subplots(figsize=(7, 4.2))
for sysname, c, lbl in (("baseline", C_BASE, "Baseline"), ("afd_a2", C_A2, "AFD A2")):
    pts = {p["target_tps"]: p for p in cap_points(sysname)}
    for load, ls in ((2185.5, "-"), (4371.0, "--")):
        if load not in pts:
            continue
        t = sorted(r["ttft_s"] for r in pts[load]["per_request"])
        y = np.arange(1, len(t) + 1) / len(t)
        ax.plot(t, y, ls, color=c, lw=1.6,
                label=f"{lbl} @ {load:g} tok/s")
ax.axvline(50, color="k", ls=":", lw=1.2)
ax.text(50.5, 0.05, "SLO 50s", fontsize=8)
ax.set_xlabel("TTFT (s)")
ax.set_ylabel("CDF")
ax.set_title("TTFT distribution at matched loads")
ax.legend(fontsize=8)
save(fig, "capacity_cdf.png")

# ------------------------------------------------------- 3. window dynamics
fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.4), sharex=False)
for ax, sysname, c, lbl in ((axes[0], "baseline", C_BASE, "Baseline"),
                            (axes[1], "afd_a2", C_A2, "AFD A2")):
    pts = {p["target_tps"]: p for p in cap_points(sysname)}
    load = 4371.0
    p = pts[load]
    off = [r["offset_s"] for r in p["per_request"]]
    tt = [r["ttft_s"] for r in p["per_request"]]
    ln = [r["len"] for r in p["per_request"]]
    sc = ax.scatter(off, tt, s=np.array(ln) / 1200, c=c, alpha=0.55, edgecolors="none")
    ax.axhline(50, color="k", ls=":", lw=1)
    ax.set_ylabel("TTFT (s)")
    verdict = "PASS" if p["slo_ok"] else "FAIL"
    ax.set_title(f"{lbl} @ {load:g} tok/s  (p99={p['ttft']['p99']:.1f}s, {verdict}; "
                 f"bubble size = prompt len)", fontsize=9)
axes[1].set_xlabel("Request offset in window (s)")
save(fig, "window_scatter.png")

# ---------------------------------------------------------------- 4. goodput
fig, ax = plt.subplots(figsize=(6.4, 4.2))
lim = [0, 9500]
ax.plot(lim, lim, ":", color=GREY, lw=1, label="ideal (offered = done)")
for sysname, c, lbl in (("baseline", C_BASE, "Baseline"), ("afd_a2", C_A2, "AFD A2")):
    pts = cap_points(sysname)
    x = [p["target_tps"] for p in pts]
    y = [p["goodput_tps"] for p in pts]
    m = ["o" if p["slo_ok"] else "x" for p in pts]
    ax.plot(x, y, "-", color=c, lw=1.4)
    for xi, yi, mi in zip(x, y, m):
        ax.plot(xi, yi, mi, color=c, ms=7, mew=2)
ax.plot([], [], "o", color="k", label="SLO PASS")
ax.plot([], [], "x", color="k", label="SLO FAIL")
ax.set_xlabel("Offered load (input tokens/s)")
ax.set_ylabel("Goodput (tokens/s)")
ax.set_title("Goodput: both systems finish the work; SLO differs")
ax.legend(fontsize=8)
save(fig, "goodput.png")

# ---------------------------------------------------------- 5. queue share
def service_model(sysname):
    """Piecewise-linear empty-load service time from accept singles."""
    singles = sorted(S["accept"][sysname]["singles"], key=lambda r: r["len"])
    xs = np.array([0] + [r["len"] for r in singles], dtype=float)
    ys = np.array([0] + [r["ttft_s"] for r in singles], dtype=float)
    return lambda n: np.interp(n, xs, ys)

fig, ax = plt.subplots(figsize=(7, 4.2))
share_rows = {}
for sysname, c, lbl in (("baseline", C_BASE, "Baseline"), ("afd_a2", C_A2, "AFD A2")):
    svc = service_model(sysname)
    pts = cap_points(sysname)
    x, share = [], []
    for p in pts:
        ratios = []
        for r in p["per_request"]:
            s_est = float(svc(r["len"]))
            q = max(r["ttft_s"] - s_est, 0.0)
            ratios.append(q / r["ttft_s"])
        x.append(p["target_tps"])
        share.append(100 * float(np.mean(ratios)))
        share_rows[(sysname, p["target_tps"])] = share[-1]
    ax.plot(x, share, "-o", color=c, label=lbl, lw=1.6)
ax.set_xlabel("Offered load (input tokens/s)")
ax.set_ylabel("Queueing share of TTFT (%)")
ax.set_title("Queueing dominates TTFT even at the lowest load\n"
             "(service time modelled from empty-load singles)")
ax.legend(fontsize=8)
ax.set_ylim(0, 100)
save(fig, "queue_share.png")
print("queue_share:", {f"{k[0]}@{k[1]}": round(v, 1) for k, v in share_rows.items()})

# ------------------------------------------------------------ 6. fixed batch
fb = S["fixed_batch"]
batches = ["fixed_8k_balanced", "fixed_32k_balanced", "fixed_32k_long_short"]
btitles = ["8K balanced\n(7,743 tok)", "32K balanced\n(30,857 tok)", "32K long+short\n(33,704 tok)"]
systems = [("baseline", "Baseline", C_BASE), ("afd_a1", "A1 (req split)", C_A1),
           ("afd_a2", "A2 (token split)", C_A2)]
fig, axes = plt.subplots(1, 2, figsize=(10, 4.0))
width = 0.25
for j, (bt, btitle) in enumerate(zip(batches, btitles)):
    for i, (sysk, sysl, c) in enumerate(systems):
        d = fb.get(sysk, {}).get(bt)
        if not d or not d.get("wall"):
            continue
        x = j + (i - 1) * width
        axes[0].bar(x, d["wall"]["p50"], width * 0.9, color=c,
                    label=sysl if j == 0 else None)
        axes[1].bar(x, d["throughput_tps"], width * 0.9, color=c)
axes[0].set_xticks(range(len(batches)))
axes[0].set_xticklabels(btitles, fontsize=8)
axes[0].set_ylabel("Batch wall time p50 (s)")
axes[0].set_title("Fixed-batch TTFT (closed loop, pinned DP0)")
axes[0].legend(fontsize=8)
axes[1].set_xticks(range(len(batches)))
axes[1].set_xticklabels(btitles, fontsize=8)
axes[1].set_ylabel("Throughput (tokens/s)")
axes[1].set_title("Fixed-batch throughput")
# A0 crashed - annotate
axes[0].annotate("A0 (no pipeline): crashed 507015\n(known SP+async race)",
                 xy=(0.02, 0.95), xycoords="axes fraction", fontsize=8,
                 color=GREY, va="top")
save(fig, "fixed_batch.png")

# ----------------------------------------------------------- 7. solo scaling
fig, ax = plt.subplots(figsize=(7, 4.2))
for sysname, c, lbl in (("baseline", C_BASE, "Baseline"), ("afd_a2", C_A2, "AFD A2")):
    singles = sorted(S["accept"][sysname]["singles"], key=lambda r: r["len"])
    x = [r["len"] for r in singles]
    y = [r["ttft_s"] for r in singles]
    ax.plot(x, y, "-o", color=c, label=lbl, lw=1.6)
    for xi, yi in zip(x, y):
        ax.annotate(f"{yi:.1f}s", (xi, yi), textcoords="offset points",
                    xytext=(0, 7), fontsize=8, color=c, ha="center")
bs = {r["len"]: r["ttft_s"] for r in S["accept"]["baseline"]["singles"]}
a2 = {r["len"]: r["ttft_s"] for r in S["accept"]["afd_a2"]["singles"]}
for ln in bs:
    if ln in a2:
        pct = 100 * (a2[ln] - bs[ln]) / bs[ln]
        ax.annotate(f"{pct:+.0f}%", ((ln), (bs[ln] + a2[ln]) / 2),
                    textcoords="offset points", xytext=(12, -4), fontsize=8,
                    color="k", style="italic")
ax.set_xlabel("Prompt length (tokens)")
ax.set_ylabel("Solo TTFT (s)")
ax.set_title("Empty-load single-request TTFT (acceptance)")
ax.legend(fontsize=8)
save(fig, "solo_scaling.png")

# -------------------------------------------------------- 8. device occupancy
prof = S["profiles"]
order = [("baseline/rank0_summary", "Baseline\nrank0"),
         ("afd_a1/attention", "A1 attn\nrank0"), ("afd_a1/ffn", "A1 FFN\nrank0"),
         ("afd_a2/attention", "A2 attn\nrank0"), ("afd_a2/ffn", "A2 FFN\nrank0")]
fig, ax = plt.subplots(figsize=(7.6, 4.2))
xs = np.arange(len(order))
busy = [prof[k]["busy_ratio"] * 100 for k, _ in order]
camw = [prof[k]["cam_wait_ratio"] * 100 for k, _ in order]
bub = [prof[k]["bubble_ratio"] * 100 for k, _ in order]
# cam_wait is a subset of busy (wait kernels on device); show busy total and overlay cam part
ax.bar(xs, busy, 0.55, color="#7fb3d5", label="device busy (union)")
ax.bar(xs, camw, 0.55, color="#1a5276", label="  of which cam_wait (blocked on AFD comms)")
ax.bar(xs, bub, 0.55, bottom=busy, color="#d5dbdb", label="bubble (idle)")
ax.set_xticks(xs)
ax.set_xticklabels([t for _, t in order], fontsize=8)
ax.set_ylabel("% of profiled window")
ax.set_title("Device time composition (whole-window union, rank0, directional only)")
ax.legend(fontsize=8, loc="lower right")
save(fig, "device_occupancy.png")

# ------------------------------------------------------------ 9. flow queue
fq = S["flows"]["queue_delay_ms"]
stages = [("afd.cam.dispatch_send", "attn\ndispatch_send"),
          ("afd.cam.dispatch_recv", "FFN\ndispatch_recv"),
          ("afd.cam.combine_send", "FFN\ncombine_send"),
          ("afd.cam.combine_recv", "attn\ncombine_recv")]
fig, ax = plt.subplots(figsize=(7, 4.2))
xs = np.arange(len(stages))
p50 = [fq[k]["p50"] for k, _ in stages]
p99 = [fq[k]["p99"] for k, _ in stages]
ax.bar(xs - 0.17, p50, 0.32, color="#2874a6", label="p50")
ax.bar(xs + 0.17, p99, 0.32, color="#ca6f1e", label="p99")
for x, v in zip(xs, p50):
    ax.annotate(f"{v:,.0f}" if v > 10 else f"{v:.1f}", (x - 0.17, v),
                textcoords="offset points", xytext=(0, 4), ha="center", fontsize=8)
for x, v in zip(xs, p99):
    ax.annotate(f"{v:,.0f}", (x + 0.17, v), textcoords="offset points",
                xytext=(0, 4), ha="center", fontsize=8)
ax.set_yscale("log")
ax.set_xticks(xs)
ax.set_xticklabels([t for _, t in stages], fontsize=9)
ax.set_ylabel("marker -> device enqueue delay (ms, log)")
ax.set_title("Device queue backlog: attention side piles up to ~5.7s, FFN side ~1.5ms\n"
             "(same-clock FIFO pairing, 1,856 flows, A2 rank0 stressed replay)")
ax.legend(fontsize=8)
save(fig, "flow_queue.png")

# ------------------------------------------------------- 10. CAM op durations
dop = S["flows"]["device_op_dur_ms"]
ops = [k for k in ("CamMoeDistributeDispatchSend", "CamMoeDistributeDispatchRecv",
                   "CamMoeDistributeCombineSend", "CamMoeDistributeCombineRecv") if k in dop]
fig, ax = plt.subplots(figsize=(7, 4.0))
xs = np.arange(len(ops))
p50 = [dop[k]["p50"] for k in ops]
p99 = [dop[k]["p99"] for k in ops]
ax.bar(xs - 0.17, p50, 0.32, color="#2874a6", label="p50")
ax.bar(xs + 0.17, p99, 0.32, color="#ca6f1e", label="p99")
for x, v in zip(xs, p50):
    ax.annotate(f"{v:.2f}", (x - 0.17, v), textcoords="offset points",
                xytext=(0, 4), ha="center", fontsize=8)
for x, v in zip(xs, p99):
    ax.annotate(f"{v:.1f}", (x + 0.17, v), textcoords="offset points",
                xytext=(0, 4), ha="center", fontsize=8)
ax.set_yscale("log")
ax.set_xticks(xs)
ax.set_xticklabels([o.replace("CamMoeDistribute", "") for o in ops], fontsize=9)
ax.set_ylabel("device op duration (ms, log)")
ax.set_title("CAM device op durations (rank0) - DispatchRecv blocks waiting for data;\n"
             "transfer itself is sub-ms to ms -> comms is not the bottleneck")
ax.legend(fontsize=8)
save(fig, "cam_op_dur.png")

# ------------------------------------------------------- 11. all-rank figure stats
ar = {}
for name in ("baseline", "afd_a0", "afd_a2"):
    with open(os.path.join(HERE, "..", "02_profiles_32", "allrank",
                           f"{name}_allranks.report.json")) as f:
        ar[name] = json.load(f)
fig, ax = plt.subplots(figsize=(6.8, 3.6))
names = ["baseline", "afd_a0", "afd_a2"]
lbls = ["Baseline\n(32 ranks)", "A0 no-pipeline\n(32r x 3 lanes)", "A2 token-split\n(32r x 3 lanes)"]
evm = [ar[n]["event_count"] / 1e6 for n in names]
spans = [ar[n]["span_s"] for n in names]
xs = np.arange(3)
ax.bar(xs - 0.17, evm, 0.32, color="#5d6d7e", label="events (M)")
ax2 = ax.twinx()
ax2.bar(xs + 0.17, spans, 0.32, color="#af7ac5", label="span (s)")
ax2.grid(False)
for x, v in zip(xs, evm):
    ax.annotate(f"{v:.1f}M", (x - 0.17, v), textcoords="offset points",
                xytext=(0, 4), ha="center", fontsize=8)
for x, v in zip(xs, spans):
    ax2.annotate(f"{v:.0f}s", (x + 0.17, v), textcoords="offset points",
                 xytext=(0, 4), ha="center", fontsize=8, color="#7d3c98")
ax.set_xticks(xs)
ax.set_xticklabels(lbls, fontsize=9)
ax.set_ylabel("trace events (millions)")
ax2.set_ylabel("captured span (s)")
ax.set_title("All-rank stacked traces (one figure per system)")
save(fig, "allrank_stats.png")

print("all done")
