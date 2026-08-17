#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Stage-1 350-cell 全矩阵深挖分析（多视角、数据支撑、可复现）。

输入：bench_results/prefill/summary.csv + slo_summary.csv（Stage-1 单次 sweep 聚合）。
输出（bench_results/prefill/stage1_deep/）：
  report.md             — 8 个视角的详细分析报告
  charts.html           — Chart.js 交互图
  token_flow_gantt.svg  — 数据推导的步级 token 流动图（非 L2 trace，聚合数据推导）

用法：
  python3 -m tools.benchmarks.analyze_stage1_deep
"""
from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

BASE = Path("bench_results/prefill")
OUT = BASE / "stage1_deep"
SA = "dp4_tp8_sp"
SB = "afd_dp3_tp8_ep8"
LA, LB = "Baseline DP4xTP8", "AFD DP3xTP8+EP8"
BT = [8192, 16384, 32768, 49152, 65536]
PRE = [0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
RPS = [4.0, 6.0, 8.0, 10.0, 12.0]
MEDIAN_LEN = 16936  # cp8sp50k median input tokens

C_BLUE = "#2c7fb8"
C_ORANGE = "#d95f0e"


# ---------------- data ----------------
def load():
    rows = []
    with open(BASE / "summary.csv") as f:
        for r in csv.DictReader(f):
            rows.append(dict(system=r["system"], bt=int(r["bt"]),
                             prefix=float(r["prefix"]), rps=float(r["rps"]),
                             ttft=float(r["ttft_mean_ms"]), p99=float(r["ttft_p99_ms"]),
                             success=int(r["success"]), failed=int(r["failed"])))
    slo = {}
    with open(BASE / "slo_summary.csv") as f:
        for r in csv.DictReader(f):
            slo[(r["system"], int(r["bt"]), float(r["prefix"]), float(r["rps"]))] = (
                float(r["slo_2s"]), float(r["slo_5s"]), float(r["slo_10s"]), float(r["slo_20s"]))
    return rows, slo


def get(rows, system, bt=None, prefix=None, rps=None):
    out = []
    for r in rows:
        if r["system"] != system:
            continue
        if bt is not None and r["bt"] != bt:
            continue
        if prefix is not None and r["prefix"] != prefix:
            continue
        if rps is not None and r["rps"] != rps:
            continue
        out.append(r)
    return out


def ols_loglog(data):
    X, y = [], []
    for r in data:
        X.append([1.0, math.log(r["bt"]), r["rps"], r["prefix"]])
        y.append(math.log(r["ttft"]))
    n = len(y)
    Xt = [[X[j][i] for j in range(n)] for i in range(4)]
    XtX = [[sum(a * b for a, b in zip(ra, col)) for col in zip(*X)] for ra in Xt]
    Xty = [sum(Xt[i][j] * y[j] for j in range(n)) for i in range(4)]
    aug = [XtX[i] + [Xty[i]] for i in range(4)]
    for col in range(4):
        piv = max(range(col, 4), key=lambda r: abs(aug[r][col]))
        aug[col], aug[piv] = aug[piv], aug[col]
        pv = aug[col][col]
        aug[col] = [v / pv for v in aug[col]]
        for r in range(4):
            if r != col:
                f = aug[r][col]
                aug[r] = [a - f * b for a, b in zip(aug[r], aug[col])]
    beta = [aug[i][4] for i in range(4)]
    ybar = sum(y) / n
    sst = sum((v - ybar) ** 2 for v in y)
    sse = 0.0
    for r, yi in zip(X, y):
        pred = sum(b * x for b, x in zip(beta, r))
        sse += (yi - pred) ** 2
    return beta, 1 - sse / sst


def per_step(r4cells):
    out = {}
    for sys_name, label in ((SA, LA), (SB, LB)):
        out[label] = {}
        for bt in BT:
            steps = max(1, math.ceil(MEDIAN_LEN / bt))
            cells = [r for r in r4cells if r["system"] == sys_name and r["bt"] == bt
                     and r["prefix"] == 0.0 and r["rps"] == 4.0]
            t = cells[0]["ttft"] if cells else None
            out[label][bt] = {"steps": steps, "ttft": t,
                              "per_step": round(t / steps, 1) if t else None,
                              "tok_per_ms": round(MEDIAN_LEN / t, 2) if t else None}
    return out


def prefix_marginal(rows):
    out = {}
    for sys_name, label in ((SA, LA), (SB, LB)):
        out[label] = {}
        for bt in BT:
            cells = {r["prefix"]: r for r in get(rows, sys_name, bt=bt, rps=8.0)}
            seq = sorted(cells)
            out[label][bt] = {}
            for i in range(1, len(seq)):
                p, c = seq[i - 1], seq[i]
                out[label][bt][f"{p:g}->{c:g}"] = round((1 - cells[c]["ttft"] / cells[p]["ttft"]) * 100, 1)
    return out


def degrade(rows):
    out = {}
    for sys_name, label in ((SA, LA), (SB, LB)):
        out[label] = {}
        for bt in BT:
            lo = get(rows, sys_name, bt=bt, prefix=0.0, rps=4.0)
            hi = get(rows, sys_name, bt=bt, prefix=0.0, rps=12.0)
            if lo and hi:
                out[label][bt] = round(hi[0]["ttft"] / lo[0]["ttft"], 2)
    return out


def tail(rows):
    out = {}
    for sys_name, label in ((SA, LA), (SB, LB)):
        by_rps = defaultdict(list)
        n_tail = 0
        for r in get(rows, sys_name):
            ratio = r["p99"] / r["ttft"] if r["ttft"] else 0
            by_rps[r["rps"]].append(ratio)
            if ratio > 2.0:
                n_tail += 1
        out[label] = {"overall": round(statistics.fmean(
            r["p99"] / r["ttft"] for r in get(rows, sys_name) if r["ttft"]), 2),
            "cells": n_tail,
            "by_rps": {str(k): round(statistics.fmean(v), 2) for k, v in sorted(by_rps.items())}}
    return out


def slo_frontier(slo):
    out = {}
    for sys_name, label in ((SA, LA), (SB, LB)):
        good = [(bt, pre, rps, s2) for (s, bt, pre, rps), (s2, *_) in slo.items()
                if s == sys_name and s2 >= 90]
        out[label] = {"count": len(good),
                      "best": sorted(good, key=lambda x: -x[3])[:6]}
    return out


def crossover(rows):
    by = {}
    for r in rows:
        by[(r["system"], r["bt"], r["prefix"], r["rps"])] = r
    out = []
    for pre in PRE:
        row = {"prefix": pre, "wins": {}}
        for rps in RPS:
            wins = [bt for bt in BT
                    if (SA, bt, pre, rps) in by and (SB, bt, pre, rps) in by
                    and by[(SB, bt, pre, rps)]["ttft"] <= by[(SA, bt, pre, rps)]["ttft"]]
            row["wins"][str(rps)] = wins
        out.append(row)
    return out


# ---------------- markdown ----------------
def md_report(rows, slo, ela, ps, pm, deg, tl, sf, cr):
    L = []
    A = L.append
    A("# Stage-1 数据深挖报告 — 350-cell 全矩阵的多视角分析")
    A("")
    A(f"**Baseline {LA}（EP32） vs AFD {LB}（EP8）** · 数据：`bench_results/prefill/summary.csv` "
      "（2 sys × 5 bt × 7 prefix × 5 rps = 350 cell，TTFT mean/p99）+ `slo_summary.csv`。"
      f"失败请求共 {sum(r['failed'] for r in rows)} 条。本报告用 8 个相互独立的视角，对 Stage-1 `FINAL_REPORT.md` 既有印证也有修正。")
    A("")
    A("> ⚠️ 口径：本报告基于 Stage-1 单次 sweep 的**聚合数据**（无逐请求长度/到达序）。")
    A("> 流水图（§9）是**由聚合数据推导的步级示意图**，非 L2 trace；真实 trace 流水图在 Stage-3 P3（需 pod 重启）。")
    A("")
    A("---")
    A("")
    A("## 1. 弹性系数：AFD 与 baseline 的杠杆结构根本不同")
    A("")
    A("拟合 `ln(TTFT) = b0 + b_bt·ln(bt) + b_rps·rps + b_pre·prefix`（各自 175 cell）：")
    A("")
    A("| 系统 | b_bt（bt 弹性） | b_rps（/rps） | b_pre（/prefix） | R² |")
    A("|---|---|---|---|---|")
    for label, (beta, r2, _) in ela.items():
        A(f"| {label} | {beta[1]:+.3f} | {beta[2]:+.3f} | {beta[3]:+.3f} | {r2:.3f} |")
    A("")
    A("- **bt 弹性符号相反**：baseline bt 翻倍 → TTFT ×2^(+0.312)≈**×1.24（变差 24%）**；"
      "AFD bt 翻倍 → ×2^(-0.127)≈**×0.92（变好 8%）**。Stage-3 深挖的反直觉现象在回归层面成立且可量化。")
    A(f"- **AFD 对负载的敏感性是 baseline 的 6 倍**（{ela[LB][0][2]:+.3f} vs {ela[LA][0][2]:+.3f} / rps）："
      "AFD 的甜区深但窄，是用负载敏感性换来的——负载每高 1 rps，AFD 的 log-TTFT 涨幅是 baseline 的 ~6×。")
    A(f"- **baseline 从 prefix 命中获益更大**（b_pre {ela[LA][0][3]:.3f} vs {ela[LB][0][3]:.3f}）："
      "命中率是两系统共同最强杠杆，但 baseline 更依赖它。")
    A(f"- 三因子解释了两系统 TTFT 变异约 3/4（R²={ela[LA][1]:.2f}/{ela[LB][1]:.2f}）。")
    A("")
    A("---")
    A("")
    A("## 2. 每步有效时间：'bt 越大 AFD 越快'的最干净量化（中位请求 16936 token, prefix=0, rps4）")
    A("")
    A(f"一个 16936-token 请求按 `steps=⌈16936/bt⌉` 切分；每步有效时间 = TTFT/steps。")
    A("")
    A("| bt | steps | baseline 每步(ms) | AFD 每步(ms) | baseline 单请求 tok/ms | AFD tok/ms |")
    A("|---|---|---|---|---|---|")
    for bt in BT:
        b, a = ps[LA][bt], ps[LB][bt]
        A(f"| {bt} | {b['steps']} | {b['per_step']} | {a['per_step']} | {b['tok_per_ms']} | {a['tok_per_ms']} |")
    A("")
    b_r = ps[LB][32768]["per_step"] / ps[LB][8192]["per_step"]
    A(f"- **baseline 每步成本超线性增长**：batch 从 8192→32768（×4），每步时间 {ps[LA][8192]['per_step']}→{ps[LA][32768]['per_step']}ms = **×4.9**（比线性还糟，内存/调度压力主导）。")
    A(f"- **AFD 每步成本次线性增长**：同样 ×4 batch，每步 {ps[LB][8192]['per_step']}→{ps[LB][32768]['per_step']}ms = **×{b_r:.2f}**——"
      "attention 与 FFN 异步并发，边际 token 近乎免费。")
    A(f"- 单请求端到端吞吐：baseline **26→11 tok/ms 单调下降**；AFD **28→37 tok/ms 上升后平台**。"
      "步数减少直接兑换为总延迟下降（AFD bt8192 597ms → bt32768 460ms；baseline 655→1064ms 反向恶化）。")
    A("")
    A("---")
    A("")
    A("## 3. 交叉点边界：AFD 何时反超，与负载严格单调")
    A("")
    A("prefix=0 下，AFD TTFT ≤ baseline 所需的最小 bt（即第一个反超点）：")
    A("")
    A("| rps | 4 | 6 | 8 | 10 | 12 |")
    A("|---|---|---|---|---|---|")
    base = next(c for c in cr if c["prefix"] == 0.0)
    row = "| prefix=0 最小反超 bt | " + " | ".join(
        (f"bt{base['wins'][str(rps)][0]}" if base["wins"][str(rps)] else "—") for rps in RPS) + " |"
    A(row)
    A("")
    A("- 交叉 bt 随负载**单调上移**：rps4 全区间赢 → rps6/8 从 16k 起赢 → rps10 需 49k → rps12 需 64k。"
      "判据可压缩成一句：**AFD 要赢，必须 bt ≥ f(rps)**。")
    A("- **修正 Stage-1 报告**：`FINAL_REPORT` 称'小 batch(bt8192) 任何 RPS 都落后'——**rps4 下 AFD 其实赢（597 vs 655ms）**。"
      "小 batch 落后成立的范围是 rps≥6。")
    A("")
    A("---")
    A("")
    A("## 4. Prefix 敏感谱：两系统形状相反")
    A("")
    A("rps8 下相邻 prefix 档的 TTFT 降幅（%），按 bt 取范围：")
    A("")
    A("| prefix 档 | baseline 降幅 % | AFD 降幅 % |")
    A("|---|---|---|")
    pmb, pma = pm[LA], pm[LB]
    keys = list(pmb[BT[0]])
    for k in keys:
        bmin, bmax = min(pmb[bt][k] for bt in BT), max(pmb[bt][k] for bt in BT)
        amin, amax = min(pma[bt][k] for bt in BT), max(pma[bt][k] for bt in BT)
        A(f"| {k} | {bmin}-{bmax} | {amin}-{amax} |")
    A("")
    A("- **baseline 中段最强**：0.5→0.75（36-66%）、0.75→0.9（44-67%）；低段（0→0.25）仅 12-24%。")
    A(f"- **AFD 第一档最强**：0→0.25 降 29-61%（bt8192 高达 60.5%）；中段衰减；尾段 0.95→0.99 再加速（bt32768 48%）。")
    A("- **含义**：AFD 的冷缓存惩罚高度集中在 prefix=0 那一点（async 固定开销叠加在全量计算上）；"
      "baseline 的惩罚铺满整个中段。**给 AFD 配一点点命中（0.25）的边际收益远大于给 baseline**——"
      "ops 上'先喂饱 AFD 的缓存命中'是最划算的杠杆。")
    A("")
    A("---")
    A("")
    A("## 5. 尾延迟签名：AFD 的均值优势带系统性长尾惩罚")
    A("")
    A("P99/Mean 相对尾比（>2 视为尾粗）：")
    A("")
    A("| 系统 | 整体 p99/mean | 尾比>2 的 cell 数 | by rps |")
    A("|---|---|---|---|")
    for label, d in tl.items():
        A(f"| {label} | {d['overall']} | {d['cells']}/175 | {d['by_rps']} |")
    A("")
    A("- **AFD 几乎每个 cell 尾比都 >2（174/175），baseline 为 138/175**——长尾是 AFD 的系统性签名。")
    A("- 甜区例（bt49152 rps4 prefix0）：baseline 1386/3026（2.18）；AFD 469/1574（**3.36**）——"
      "AFD 相对尾更长，但**绝对 p99 仍只有 baseline 一半**。")
    A("- 反转例（bt16384 prefix0.99 rps8）：AFD **均值赢**（149 vs 152ms）但 **p99 输**（753 vs 622ms）。"
      "在均值咬紧的 cell，AFD 会赢均值输 p99。")
    A("- rps≥10 时 AFD 尾比反而收窄到 2.8-2.94——饱和态不再放大尾部，与排队波主导（B2 结论）吻合。")
    A("")
    A("---")
    A("")
    A("## 6. 负载退化因子：因子 vs 绝对值要分开看")
    A("")
    A("prefix=0 下 rps12/rps4 的 TTFT 放大倍数：")
    A("")
    A("| bt | baseline × | AFD × |")
    A("|---|---|---|")
    for bt in BT:
        A(f"| {bt} | {deg[LA][bt]} | {deg[LB][bt]} |")
    A("")
    A("- AFD 退化因子整体比 baseline 差 2-5×，即便大 batch 仍 ~19×。**但因子会误导**："
      "AFD 大 batch 起点 466ms，19×≈8.9s；baseline 起点 1597ms，6.4×≈10.2s——绝对量级仍接近甚至更好。")
    A("- 结论：**用绝对 TTFT（或 SLO 达成）做决策，不要用退化因子**。因子高≠更差，只说明起点低。")
    A("")
    A("---")
    A("")
    A("## 7. 严格 SLO(2s) 前沿：AFD 有更多达标 cell，且甜区峰在 bt16k")
    A("")
    A("| 系统 | slo_2s≥90% 的 cell | 最优 cell（bt, prefix, rps, slo2%） |")
    A("|---|---|---|")
    for label, d in sf.items():
        best = ", ".join(f"bt{g[0]} pre{g[1]:g} rps{g[2]:g} {g[3]:.0f}%" for g in d["best"][:3])
        A(f"| {label} | {d['count']}/175 | {best} |")
    A("")
    A(f"- AFD 达标 cell 更多（{sf[LB]['count']} vs {sf[LA]['count']}），但两系统的**最优 cell 都是 bt16384**（不是最大 batch）"
      "+ prefix≥0.75 + 高压 rps10/12 → 100%。")
    A("- **'bt 越大越好'在严格 SLO 下有极值：16k 是甜区峰，49k+ 反而滑出前沿**——大 batch 均值好，但尾延迟把严格 SLO 拉下来。")
    A("")
    A("---")
    A("")
    A("## 8. 高 prefix 死区 / 收敛带")
    A("")
    A("- prefix=0.9-0.95：除 rps4 外 **AFD 基本全输**——残留计算太小，dispatch/combine 固定开销主导。")
    A("- prefix=0.99：两系统收敛到 150-350ms，胜负落在噪声内（如 bt16384 rps8: 149 vs 152ms；bt32768: 166 vs 164ms）。")
    A("- **修正 Stage-1 报告**：`p90+ AFD 反超'应表述为'收敛带 + 噪声内胜负'，无系统性 AFD 优势。")
    A("")
    A("---")
    A("")
    A("## 9. 流水图（数据推导的步级 token 流动）")
    A("")
    A("`token_flow_gantt.svg`：同一中位请求（16936 token, prefix=0, rps4）在 baseline（串行整步）vs AFD（A2 token-split 两 lane 异步）"
      "下，bt8192（3 步）与 bt32768（1 步）的时间线。每步宽度 = 实测每步有效时间。")
    A("")
    A("**这是聚合数据推导的示意图**（每步时间 = TTFT/步数，AFD 双 lane 为架构示意），"
      "非 L2 trace 逐事件时间线。真实 trace 流水图待 Stage-3 P3（pod 重启 + `PHASE=btsweep-profile` 后）。")
    A("")
    A("---")
    A("")
    A("## 10. 运营判据汇总（一页决策表）")
    A("")
    A("| 场景 | 结论 | 数据 |")
    A("|---|---|---|")
    A("| **用 AFD** | prefix≤0.75 且 rps≤8 且 bt≥16384 | 甜区 TTFT 降 31-71%，SLO2 达成 88-92%（Stage-1 报告） |")
    A("| 用 AFD（修正） | rps4 下连 bt8192 也赢 | 597 vs 655ms（§3） |")
    A("| **别用 AFD** | prefix 0.9-0.95，或 bt8192 且 rps≥6 | §8 / §3 |")
    A("| **指标提醒** | 场景按 p99 计 → AFD 甜区收窄 | §5：AFD 尾比 3.4 vs baseline 2.2 |")
    A("| **严格 SLO(2s)** | 两系统最佳 bt 都是 16k，prefix 命中是充要杠杆 | §7 |")
    A("")
    A("## 局限")
    A("")
    A("- 聚合数据：无逐请求长度/到达序/TTFT 分布；Stage-2 的 81 verified 与 L2 trace 在已停 pod 上，未纳入本报告。")
    A("- 每步模型假设步内均匀；真实逐事件验证需 Stage-3 P2（L2 profiler）与 P3（token-flow Gantt）。")
    A(f"- {sum(r['failed'] for r in rows)} 条 failed 请求未展开归因。")
    A("")
    A("---")
    A("*Generated by `tools/benchmarks/analyze_stage1_deep.py` · 数据可直接从 summary.csv 复现。*")
    return "\n".join(L)


# ---------------- charts html ----------------
def _ds(label, data, color):
    return "{" + f"label:'{label}',data:{json.dumps(data)},borderColor:'{color}',backgroundColor:'{color}',fill:false" + "}"


def charts_html(rows, ela, ps, pm, deg, tl, cr):
    panels = []

    # 1 elasticity
    panels.append("""<div class="card"><h3>1. 弹性系数（ln TTFT ~ ln bt + rps + prefix）</h3>
<canvas id="ela" height="60"></canvas>
<script>new Chart(document.getElementById('ela'),{type:'bar',data:{labels:['b_bt (bt 弹性)','b_rps (每 rps)','b_pre (每 prefix)'],
datasets:[{label:'Baseline',data:%s,backgroundColor:'#2c7fb8'},{label:'AFD',data:%s,backgroundColor:'#d95f0e'}]},
options:{responsive:true,scales:{y:{title:{display:true,text:'回归系数'}}}}});</script></div>""" % (
        json.dumps([round(ela[LA][0][i], 3) for i in (1, 2, 3)]),
        json.dumps([round(ela[LB][0][i], 3) for i in (1, 2, 3)])))

    # 2 TTFT vs bt, prefix=0, rps4 + rps10
    by = {(r["system"], r["bt"], r["prefix"], r["rps"]): r for r in rows}
    for rps in (4.0, 10.0):
        labels = BT
        da, db = [], []
        for bt in BT:
            da.append(round(by[(SA, bt, 0.0, rps)]["ttft"], 0))
            db.append(round(by[(SB, bt, 0.0, rps)]["ttft"], 0))
        panels.append(f"""<div class="card"><h3>2. TTFT vs max batch（prefix=0, rps={rps:g}）— 交叉点可见</h3>
<canvas id="ttft{rps}" height="60"></canvas>
<script>new Chart(document.getElementById('ttft{rps}'),{{type:'line',
data:{{labels:{json.dumps(labels)},datasets:[{_ds('Baseline', da, C_BLUE)},{_ds('AFD', db, C_ORANGE)}]}},
options:{{responsive:true,scales:{{x:{{type:'logarithmic',title:{{display:true,text:'Max Batch Tokens'}}}},
y:{{title:{{display:true,text:'Mean TTFT (ms)'}}}}}}}}}});</script></div>""")

    # 3 per-step
    labels = [str(bt) for bt in BT]
    panels.append(f"""<div class="card"><h3>3. 每步有效时间 vs bt（中位 16936-token 请求, prefix=0 rps4）</h3>
<canvas id="step" height="60"></canvas>
<script>new Chart(document.getElementById('step'),{{type:'line',
data:{{labels:{json.dumps(labels)},datasets:[{_ds('Baseline 每步 ms', [ps[LA][bt]['per_step'] for bt in BT], C_BLUE)},{_ds('AFD 每步 ms', [ps[LB][bt]['per_step'] for bt in BT], C_ORANGE)}]}},
options:{{responsive:true,scales:{{y:{{title:{{display:true,text:'ms / step'}}}}}}}}}});</script></div>""")

    # 4 single-req throughput
    panels.append(f"""<div class="card"><h3>4. 单请求端到端吞吐 vs bt（16936 tokens / TTFT）</h3>
<canvas id="thr" height="60"></canvas>
<script>new Chart(document.getElementById('thr'),{{type:'line',
data:{{labels:{json.dumps(labels)},datasets:[{_ds('Baseline tok/ms', [ps[LA][bt]['tok_per_ms'] for bt in BT], C_BLUE)},{_ds('AFD tok/ms', [ps[LB][bt]['tok_per_ms'] for bt in BT], C_ORANGE)}]}},
options:{{responsive:true,scales:{{y:{{title:{{display:true,text:'tokens / ms'}}}}}}}}}});</script></div>""")

    # 5 prefix marginal (bt32768)
    k = list(pm[LA][32768])
    panels.append(f"""<div class="card"><h3>5. Prefix 边际降幅 %（rps8, bt32768）— 两系统敏感谱相反</h3>
<canvas id="pm" height="60"></canvas>
<script>new Chart(document.getElementById('pm'),{{type:'bar',
data:{{labels:{json.dumps(k)},datasets:[{{label:'Baseline',data:{json.dumps([pm[LA][32768][x] for x in k])},backgroundColor:'#2c7fb8'}},
{{label:'AFD',data:{json.dumps([pm[LB][32768][x] for x in k])},backgroundColor:'#d95f0e'}}]}},
options:{{responsive:true,scales:{{y:{{title:{{display:true,text:'TTFT 降幅 %'}}}}}}}}}});</script></div>""")

    # 6 tail ratio by rps
    panels.append(f"""<div class="card"><h3>6. P99/Mean 尾比 by rps</h3>
<canvas id="tail" height="60"></canvas>
<script>new Chart(document.getElementById('tail'),{{type:'line',
data:{{labels:{json.dumps(list(tl[LA]['by_rps'].keys()))},datasets:[{_ds('Baseline', list(tl[LA]['by_rps'].values()), C_BLUE)},{_ds('AFD', list(tl[LB]['by_rps'].values()), C_ORANGE)}]}},
options:{{responsive:true,scales:{{y:{{title:{{display:true,text:'p99/mean'}}}}}}}}}});</script></div>""")

    # 7 degrade factor
    panels.append(f"""<div class="card"><h3>7. 负载退化因子（rps12/rps4 @prefix=0）</h3>
<canvas id="deg" height="60"></canvas>
<script>new Chart(document.getElementById('deg'),{{type:'bar',
data:{{labels:{json.dumps(labels)},datasets:[{{label:'Baseline',data:{json.dumps([deg[LA][bt] for bt in BT])},backgroundColor:'#2c7fb8'}},
{{label:'AFD',data:{json.dumps([deg[LB][bt] for bt in BT])},backgroundColor:'#d95f0e'}}]}},
options:{{responsive:true,scales:{{y:{{title:{{display:true,text:'TTFT ×'}}}}}}}}}});</script></div>""")

    # 8 crossover boundary (prefix=0)
    c0 = next(c for c in cr if c["prefix"] == 0.0)
    xs, ys = [], []
    for rps in RPS:
        wins = c0["wins"][str(rps)]
        xs.append(rps)
        ys.append(wins[0] if wins else None)
    panels.append(f"""<div class="card"><h3>8. 交叉边界：prefix=0 下 AFD 最小反超 bt vs rps</h3>
<canvas id="cr" height="60"></canvas>
<script>new Chart(document.getElementById('cr'),{{type:'line',
data:{{labels:{json.dumps(xs)},datasets:[{{label:'最小反超 bt',data:{json.dumps(ys)},borderColor:'#111111',backgroundColor:'#111111',fill:false}}]}},
options:{{responsive:true,scales:{{y:{{type:'logarithmic',title:{{display:true,text:'bt'}}}}}}}}}});</script></div>""")

    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Stage-1 深挖 — 多视角分析</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>body{{font-family:sans-serif;margin:20px;background:#fafafa;}}
.card{{background:#fff;border:1px solid #ddd;border-radius:6px;padding:14px;margin:14px 0;}}
h3{{margin:0 0 10px;font-size:15px;}}</style></head><body>
<h2>Stage-1 350-cell 深挖 — 多视角分析</h2>
{''.join(panels)}
</body></html>"""
    return html


# ---------------- svg gantt ----------------
def svg_gantt(ps):
    """数据推导的步级 token 流动图（非 L2 trace）。"""
    W, H = 980, 470
    X0, X1 = 120, W - 40          # timeline axis range (px)
    TMAX = 1100.0                 # ms full scale
    Y = {"b8": 40, "b32": 130, "a8": 220, "a32": 320}
    LANE_H = 22

    def x(ms):
        return X0 + (ms / TMAX) * (X1 - X0)

    def bar(ms, y, h, color, label):
        return (f'<rect x="{x(0):.0f}" y="{y}" width="{x(ms)-x(0):.0f}" height="{h}" '
                f'rx="3" fill="{color}"/>'
                f'<text x="{x(ms)+6:.0f}" y="{y+h/2+4}" font-size="12" fill="#333">{label}</text>')

    def step_group(ms_vals, y, color, label):
        out = []
        cx = x(0)
        for i, m in enumerate(ms_vals):
            out.append(f'<rect x="{cx:.0f}" y="{y}" width="{x(m)-x(0):.0f}" height="{LANE_H}" rx="3" fill="{color}" opacity="0.85"/>')
            out.append(f'<text x="{cx+6:.0f}" y="{y+LANE_H/2+4}" font-size="10" fill="#fff">S{i+1} {m}ms</text>')
            cx += x(m) - x(0)
        out.append(f'<text x="{cx+8:.0f}" y="{y+LANE_H/2+4}" font-size="11" fill="#111" font-weight="bold">{label}</text>')
        return "".join(out)

    def afd_group(ms_vals, y, label):
        out = []
        cx = x(0)
        for i, m in enumerate(ms_vals):
            wpx = x(m) - x(0)
            out.append(f'<rect x="{cx:.0f}" y="{y}" width="{wpx}" height="{LANE_H}" rx="3" fill="{C_ORANGE}" opacity="0.7"/>')
            out.append(f'<rect x="{cx:.0f}" y="{y+LANE_H+2}" width="{wpx}" height="{LANE_H}" rx="3" fill="{C_ORANGE}" opacity="0.35"/>')
            out.append(f'<text x="{cx+6:.0f}" y="{y+LANE_H/2+4}" font-size="10" fill="#fff">A{i+1} {m}ms</text>')
            cx += wpx
        out.append(f'<text x="{cx+8:.0f}" y="{y+LANE_H/2+4}" font-size="11" fill="#111" font-weight="bold">{label}</text>')
        out.append(f'<text x="{X0+6}" y="{y+LANE_H*2+16}" font-size="9" fill="#888">上 lane=attention · 下 lane=FFN（A2 token-split 异步重叠，架构示意）</text>')
        return "".join(out)

    b8 = ps[LA][8192]; a8 = ps[LB][8192]
    b32 = ps[LA][32768]; a32 = ps[LB][32768]
    b8m = [b8["per_step"]] * 2 + [round(b8["ttft"] - 2 * b8["per_step"], 0)]
    a8m = [a8["per_step"]] * 3

    rows_svg = ""
    rows_svg += f'<text x="{X0}" y="{Y["b8"]-12}" font-size="13" fill="{C_BLUE}" font-weight="bold">Baseline bt=8192 · 3 步串行 · 总 {b8["ttft"]}ms</text>'
    rows_svg += step_group(b8m, Y["b8"], C_BLUE, f"= {b8['ttft']}ms")
    rows_svg += f'<text x="{X0}" y="{Y["b32"]-12}" font-size="13" fill="{C_BLUE}" font-weight="bold">Baseline bt=32768 · 1 步 · 总 {b32["ttft"]}ms</text>'
    rows_svg += step_group([b32["per_step"]], Y["b32"], C_BLUE, f"= {b32['ttft']}ms")
    rows_svg += f'<text x="{X0}" y="{Y["a8"]-12}" font-size="13" fill="{C_ORANGE}" font-weight="bold">AFD bt=8192 · 3 步 · 总 {a8["ttft"]}ms</text>'
    rows_svg += afd_group(a8m, Y["a8"], f"= {a8['ttft']}ms")
    rows_svg += f'<text x="{X0}" y="{Y["a32"]-12}" font-size="13" fill="{C_ORANGE}" font-weight="bold">AFD bt=32768 · 1 步 · 总 {a32["ttft"]}ms</text>'
    rows_svg += afd_group([a32["per_step"]], Y["a32"], f"= {a32['ttft']}ms")

    # time gridlines
    grid = ""
    for ms in (0, 200, 400, 600, 800, 1000):
        grid += (f'<line x1="{x(ms):.0f}" y1="20" x2="{x(ms):.0f}" y2="{H-20}" stroke="#ddd" stroke-dasharray="3,4"/>'
                 f'<text x="{x(ms)-4:.0f}" y="14" font-size="10" fill="#999">{ms}</text>')
    grid += '<text x="%d" y="%d" font-size="10" fill="#999">ms</text>' % (X1 - 14, 14)

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="sans-serif">
<rect width="{W}" height="{H}" fill="#fff"/>
<title>Token-flow step diagram (aggregate-derived, not L2 trace)</title>
<text x="16" y="30" font-size="16" font-weight="bold" fill="#111">同一中位请求（16936 token, prefix=0, rps4）的步级 token 流动（数据推导，非 trace）</text>
{grid}
{rows_svg}
</svg>"""
    return svg


# ---------------- main ----------------
def main() -> int:
    rows, slo = load()
    OUT.mkdir(parents=True, exist_ok=True)

    ela = {label: (b, r2, len(get(rows, sys_name))) for sys_name, label in ((SA, LA), (SB, LB))
           for (b, r2) in [ols_loglog(get(rows, sys_name))]}
    ps = per_step(get(rows, SA) + get(rows, SB))
    pm = prefix_marginal(rows)
    deg = degrade(rows)
    tl = tail(rows)
    sf = slo_frontier(slo)
    cr = crossover(rows)

    (OUT / "report.md").write_text(md_report(rows, slo, ela, ps, pm, deg, tl, sf, cr), encoding="utf-8")
    (OUT / "charts.html").write_text(charts_html(rows, ela, ps, pm, deg, tl, cr), encoding="utf-8")
    (OUT / "token_flow_gantt.svg").write_text(svg_gantt(ps), encoding="utf-8")
    print(f"Wrote {OUT / 'report.md'}")
    print(f"Wrote {OUT / 'charts.html'}")
    print(f"Wrote {OUT / 'token_flow_gantt.svg'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
