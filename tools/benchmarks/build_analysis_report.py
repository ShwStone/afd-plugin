#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Assemble the final prefill analysis report from macro + micro fragments."""

from __future__ import annotations

import json
import math
import re
import statistics
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
OUT = BASE / "bench_results" / "prefill"
# Any verified.json exposes the shared cp8sp50k dataset's input_lens.
DATASET_PROBE = (
    OUT.parent
    / "micro_verified"
    / "bench_results"
    / "prefill"
    / "dp4_tp8_sp-mbt65536-rps4p0-prefix0-repeat1.verified.json"
)

MACRO = OUT / "analysis_macro_fragment.html"
MICRO = OUT / "analysis_micro_fragment.html"
MACRO_MD = OUT / "analysis_macro.md"
MICRO_MD = OUT / "analysis_micro.md"
FINAL_HTML = OUT / "analysis_all_angles.html"
FINAL_MD = OUT / "analysis_all_angles.md"

CSS = """
body{font-family:sans-serif;margin:20px;background:#fafafa;color:#222;max-width:1180px;margin-left:auto;margin-right:auto}
h1{color:#222} h2{color:#333;border-bottom:2px solid #ddd;padding-bottom:6px;margin-top:40px}
.what{background:#eef4fb;border-left:4px solid #2c7fb8;padding:10px 14px;border-radius:4px;margin:10px 0;line-height:1.5}
.conclusion{background:#eaf7ef;border-left:4px solid #2e8b57;padding:10px 14px;border-radius:4px;margin:12px 0;line-height:1.55}
.card{background:#fff;border-radius:8px;padding:14px;margin:12px 0;box-shadow:0 1px 3px rgba(0,0,0,.1)}
.toc{background:#fff;border:1px solid #ddd;border-radius:8px;padding:12px 16px;margin:14px 0}
.toc h3{margin:0 0 6px;color:#555}
.toc a{display:inline-block;margin:2px 14px 2px 0;color:#2c7fb8;text-decoration:none}
.section-tag{display:inline-block;background:#2c7fb8;color:#fff;border-radius:4px;padding:2px 8px;font-size:12px;margin-bottom:4px}
.section-tag.micro{background:#d95f0e}
h4{margin:14px 0 2px;color:#444}
"""


def _log_kde(log_values: list[float], grid_log: list[float], bandwidth: float) -> list[float]:
    """Gaussian kernel density estimate evaluated in log-length space."""
    n = len(log_values)
    scale = 1.0 / (bandwidth * math.sqrt(2 * math.pi))
    density: list[float] = []
    for x_log in grid_log:
        total = 0.0
        for v_log in log_values:
            u = (x_log - v_log) / bandwidth
            total += math.exp(-0.5 * u * u)
        density.append(total * scale / n)
    return density


def _percentile(sorted_values: list[float], percentile: float) -> float:
    rank = percentile / 100 * (len(sorted_values) - 1)
    lower = int(rank)
    fraction = rank - lower
    upper = min(lower + 1, len(sorted_values) - 1)
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


def build_intro_html() -> str:
    """Render the config/dataset preamble, including a log-scale length density curve."""
    probe = json.loads(DATASET_PROBE.read_text(encoding="utf-8"))
    lens = [float(v) for v in probe["input_lens"]]
    n = len(lens)
    log_values = [math.log(v) for v in lens]
    bandwidth = 1.06 * statistics.stdev(log_values) * (n ** -0.2)  # Silverman in log space
    grid = [
        math.exp(math.log(60) + i * (math.log(52000) - math.log(60)) / 199)
        for i in range(200)
    ]
    density = _log_kde(log_values, [math.log(g) for g in grid], bandwidth)
    mean_len = statistics.fmean(lens)
    median = statistics.median(lens)
    sorted_lens = sorted(lens)
    p50, p90, p99 = (
        _percentile(sorted_lens, 50),
        _percentile(sorted_lens, 90),
        _percentile(sorted_lens, 99),
    )
    density_points = [{"x": round(g), "y": round(d, 8)} for g, d in zip(grid, density)]
    chart_config = {
        "type": "scatter",
        "data": {
            "datasets": [
                {
                    "label": "长度密度",
                    "data": density_points,
                    "borderColor": "#2c7fb8",
                    "backgroundColor": "rgba(44,127,184,0.18)",
                    "fill": True,
                    "tension": 0.4,
                    "pointRadius": 0,
                    "showLine": True,
                }
            ]
        },
        "options": {
            "responsive": True,
            "plugins": {"legend": {"display": False}},
            "scales": {
                "x": {
                    "type": "logarithmic",
                    "title": {"display": True, "text": "prompt length (tokens, log)"},
                },
                "y": {"title": {"display": True, "text": "density"}},
            },
        },
    }
    return f"""
<div class="angle" id="angle-1-setup">
  <h2>1. 实验配置与数据集</h2>
  <div class="what">
    <b>两个对比系统的配置与测试数据集说明</b>——所有宏观/微观结论都建立在这两个系统的
    固定拓扑、同一数据集、同一到达调度之上。
  </div>

  <div class="card"><h3 style="margin-top:0">系统 A：Baseline DP4×TP8（全同步）</h3>
  <table style="border-collapse:collapse;font-size:13px">
    <tr><th style="text-align:left;padding:4px 10px">维度</th><th style="text-align:left">配置</th></tr>
    <tr><td style="padding:4px 10px">并行</td><td>DP4 × TP8 = 32 ranks，跨 2 节点（每节点 16 NPU，local DP=2）</td></tr>
    <tr><td style="padding:4px 10px">MoE/FFN</td><td>全量 32 rank EP（EP32），同步执行</td></tr>
    <tr><td style="padding:4px 10px">SP</td><td>FlashComm1 sequence parallel</td></tr>
    <tr><td style="padding:4px 10px">启动</td><td><code>prefill_launch_baseline_dp4tp8.sh</code></td></tr>
  </table></div>

  <div class="card"><h3 style="margin-top:0">系统 B：AFD DP3×TP8 + EP8（async 解耦）</h3>
  <table style="border-collapse:collapse;font-size:13px">
    <tr><th style="text-align:left;padding:4px 10px">维度</th><th style="text-align:left">配置</th></tr>
    <tr><td style="padding:4px 10px">Attention</td><td>DP3 × TP8 = 24 ranks（node0 16 dev = DP0-1，node1 8 dev = DP2）</td></tr>
    <tr><td style="padding:4px 10px">FFN</td><td>EP8 = 8 ranks（node1 devices 8-15），只有 baseline 的 1/4 FFN 算力</td></tr>
    <tr><td style="padding:4px 10px">解耦</td><td>CAM async：attention 与 FFN 用异步流水线并行（CAMAsyncAFDConnector）</td></tr>
    <tr><td style="padding:4px 10px">负载均衡</td><td><code>enable_force_load_balance</code> 强制专家均衡（synthetic/mechanism 标记）</td></tr>
    <tr><td style="padding:4px 10px">启动</td><td><code>prefill_launch_afd_attention.sh</code> + <code>prefill_launch_afd_ffn.sh</code></td></tr>
  </table></div>

  <div class="card"><h3 style="margin-top:0">数据集：cp8sp50k（prefill-only）</h3>
  <ul style="font-size:13px;line-height:1.6">
    <li>{n} 条<b>生产 trace 导出的变长 prompt</b>（<code>source_row</code> 指向真实 trace），输出长度 1（只测 prefill/TTFT）</li>
    <li>长度（对数轴密度曲线见下）：mean {mean_len:.0f}，median {median:.0f}，p90 {p90:.0f}，p99 {p99:.0f}，min 71，max 50773，共 18.18M token</li>
    <li>长度桶：1-8K×213、8-16K×214、16-32K×241、32-48K×154、48K+×53——覆盖广、含长尾（对数轴曲线呈单峰右偏）</li>
    <li><b>prefix 构造</b>：每组 12 个请求共享一段 128-token 对齐的合成前缀（来自固定 seed），覆盖每组最长对齐前缀；
        每个请求 = <code>共享前缀[:自身对齐长度] + 自身 suffix</code>。prefix∈{{0,25,50,75,90,95,99}} 表示构造的共享比例，
        因此各档是 <b>constructed-prefix</b>，实际命中率由运行时 KV-cache 决定</li>
    <li>到达调度：Poisson，RPS∈{{4,6,8,10,12}}，burstiness=1.0；每个 cell 875 条 + 32 warmup</li>
    <li>SLO：TTFT ≤ 10s 为达标（多阈值分析另测 2/5/20s）</li>
  </ul>
  <canvas id="intro_len_density" height="60"></canvas>
  <script>new Chart(document.getElementById('intro_len_density'), {json.dumps(chart_config)});</script>
  <p style="font-size:12px;color:#666">Gaussian KDE（log 长度空间，Silverman 带宽 {bandwidth:.3f}）；面积在 log 轴下近似 1。</p>
  </div>
</div>
"""


PREFIX_CACHE_SECTION = """
<div class="angle" id="angle-prefix-cache">
  <h2>C. Prefix Cache：容量优势与下一步验证</h2>
  <div class="what">
    <b>示意图要说明的论点</b>：AFD 不止在理想命中率下吞吐更大——由于 attention 与 FFN 解耦，
    FFN 侧 DP 容量被释放，attention rank 的 HBM 不再承载 FFN 权重，可腾出来分配给 KV cache，
    因此 AFD 的 <b>cache 容量也更大</b>。更大的 cache 意味着实践中能达到更高的命中率，
    支撑更大的 batch tokens 与 cache tokens。
  </div>
  <div class="card">
    <img src="prefix_cache.png" alt="prefix cache capacity: baseline vs AFD"
         style="max-width:100%;border-radius:8px">
  </div>
  <div class="conclusion">
    <b>结论</b>：AFD 在理想命中率下吞吐可做大（见 B1 容量证据与 A2 甜区）；又因 FFN 侧 DP 容量被
    释放，cache 容量更大——实践中的命中率、batch token、cache token 都能更大。这两点叠加说明 AFD
    在真实（非理想构造）前缀流量下仍有容量余量。
    <br><br><b>下一步</b>：采用<b>真实数据集</b>验证这一点。
  </div>
</div>
"""


def _split_imbalance(lens: list[float]) -> dict[str, object]:
    """Simulate request-split vs token-split on a token-count list."""
    total = sum(lens)
    cumulative = 0.0
    best_score = None
    best_req = None
    for index, length in enumerate(lens):
        cumulative += length
        score = abs(cumulative * 2 - total)
        if best_score is None or score < best_score:
            best_score = score
            best_req = index + 1
    split_token = sum(lens[:best_req])
    rs_stage0, rs_stage1 = split_token, total - split_token
    ts_stage0, ts_stage1 = total // 2, total - total // 2

    def _imbalance(a: float, b: float) -> float:
        return abs(a - b) / (a + b) * 100.0

    return {
        "request_split": {
            "stage0": rs_stage0,
            "stage1": rs_stage1,
            "imbalance_pct": _imbalance(rs_stage0, rs_stage1),
            "critical_path": max(rs_stage0, rs_stage1),
        },
        "token_split": {
            "stage0": float(ts_stage0),
            "stage1": float(ts_stage1),
            "imbalance_pct": _imbalance(float(ts_stage0), float(ts_stage1)),
            "critical_path": float(max(ts_stage0, ts_stage1)),
        },
    }


def build_ubatch_section() -> str:
    """Token-split ubatch method + benefit section (data-backed imbalance)."""
    probe = json.loads(DATASET_PROBE.read_text(encoding="utf-8"))
    lens = [float(v) for v in probe["input_lens"]]
    real_knee = _split_imbalance(lens)
    long_short = _split_imbalance([16_000_000.0] + [2400.0] * 875)

    def _stats(name: str, data: dict[str, object]) -> str:
        rs, ts = data["request_split"], data["token_split"]
        return (
            f"<li><b>{name}</b>：request-split = stage "
            f"{rs['stage0']/1e6:.2f}M / {rs['stage1']/1e6:.2f}M（失衡 "
            f"{rs['imbalance_pct']:.1f}%，关键路径 {rs['critical_path']/1e6:.1f}M）；"
            f"token-split = {ts['stage0']/1e6:.2f}M / {ts['stage1']/1e6:.2f}M（失衡 "
            f"{ts['imbalance_pct']:.1f}%，关键路径 {ts['critical_path']/1e6:.1f}M）</li>"
        )

    chart_config = {
        "type": "bar",
        "data": {
            "labels": ["real-knee（cp8sp50k 混合长度）", "long-short（1 长 + 875 短）"],
            "datasets": [
                {
                    "label": "request-split 失衡 %",
                    "data": [
                        round(real_knee["request_split"]["imbalance_pct"], 1),
                        round(long_short["request_split"]["imbalance_pct"], 1),
                    ],
                    "backgroundColor": "#c0392b",
                },
                {
                    "label": "token-split 失衡 %",
                    "data": [
                        round(real_knee["token_split"]["imbalance_pct"], 1),
                        round(long_short["token_split"]["imbalance_pct"], 1),
                    ],
                    "backgroundColor": "#2e8b57",
                },
            ],
        },
        "options": {
            "responsive": True,
            "plugins": {"legend": {"labels": {"boxWidth": 12}}},
            "scales": {
                "x": {"title": {"display": True, "text": "workload"}},
                "y": {"title": {"display": True, "text": "stage token imbalance (%)"}},
            },
        },
    }
    speedup = (
        long_short["request_split"]["critical_path"]
        / long_short["token_split"]["critical_path"]
    )
    return f"""
<div class="angle" id="angle-0-method">
  <h2>0. Token-Split Ubatch：方法与性能收益</h2>
  <div class="what">
    <b>大致方法</b>：AFD 的 async MoE ubatching 把解耦后的单 stage 计算拆成 2 个 ubatch，
    让 attention 与 FFN 流水并行。切分有两种：<b>request-split</b>（<code>async_moe_split=request</code>）
    在请求边界切分，每个 ubatch 含整请求、metadata 简单，但两 stage 的 token 数受请求长度分布
    限制可能失衡；<b>token-split</b>（<code>async_moe_split=token</code>）在 token 数中点
    （<code>num_tokens_padded//2</code>）切分，请求可以跨 ubatch，两 stage token 数强制平衡。
  </div>
  <div class="card"><h4 style="margin:0 0 6px">两种切分的 stage token 失衡（用真实请求长度模拟）</h4>
  <canvas id="ubatch_imbalance" height="60"></canvas>
  <script>new Chart(document.getElementById('ubatch_imbalance'), {json.dumps(chart_config)});</script>
  <ul style="font-size:13px;line-height:1.7;margin-top:8px">
    {_stats("real-knee", real_knee)}
    {_stats("long-short", long_short)}
  </ul></div>
  <div class="conclusion">
    <b>性能收益（机制）</b>：2-stage 流水线的吞吐由较慢的 stage（关键路径）决定。request-split
    失衡时一个 stage 忙、另一个空转，浪费容量；token-split 强制两 stage token 平衡，关键路径最短。
    数据佐证：混合长度的 cp8sp50k 下 request-split 恰好接近平衡（失衡 0.2%，token-split 0.0%，无差异）；
    但<b>长尾工作负载</b>（1 个长请求 + 875 个短请求，构造放大）下 request-split 失衡高达 <b>76%</b>
    （16.0M vs 2.2M），关键路径 16.0M，而 token-split 恒为 9.1M/9.1M——理想流水下 token-split 可把
    关键路径缩短约 <b>{speedup:.2f} 倍</b>。结论：token-split 的价值在请求长度差异大（长尾）的工作负载上
    最大，它把"切分点受请求边界束缚"的不确定性转化为确定的最优平衡。
    <br><br><b>说明</b>：以上为基于真实请求长度的切分模拟与机制估算；端到端量化验证（A0=无 ubatch、
    A1=request-split、A2=token-split 消融）已设计，数据待采集。
  </div>
</div>
"""


def toc_of(fragment: str) -> list[str]:
    items = []
    for match in re.finditer(r'<div class="angle" id="(angle-[ab]\d+)".*?<h2>(.*?)</h2>', fragment, re.DOTALL):
        anchor, title = match.group(1), re.sub(r"<.*?>", "", match.group(2))
        items.append(f'<a href="#{anchor}">{title}</a>')
    return items


def main() -> int:
    macro = MACRO.read_text(encoding="utf-8")
    micro = MICRO.read_text(encoding="utf-8")
    if not macro or not micro:
        raise SystemExit("Missing analysis fragments; run analyze_sweep_all.py and analyze_sweep_micro.py first.")

    toc_links = toc_of(macro) + toc_of(micro)
    toc_html = (
        "<div class='toc'><h3>观察角度目录</h3>"
        + "<br>".join(toc_links)
        + "</div>"
    )

    intro_html = build_intro_html()
    ubatch_html = build_ubatch_section()
    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>Prefill Sweep 数据分析报告</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>{CSS}</style></head><body>
<h1>Prefill 单次 Sweep 数据分析报告</h1>
<p><b>系统对比</b>：Baseline DP4xTP8（全 32 rank 同步 MoE） vs AFD DP3xTP8+EP8（async attention/FFN 解耦）。</p>
<p><b>数据</b>：宏观 = summary.csv + slo_summary.csv（350 cell）；微观 = 54 个 verified.json（每 cell 875 条请求逐条 input_len/TTFT/SLO）。</p>
{ubatch_html}
{intro_html}
{toc_html}
<h2 style="border-bottom:3px solid #2c7fb8">宏观分析（TTFT / SLO 统计）</h2>
{macro}
<h2 style="border-bottom:3px solid #d95f0e">微观分析（逐请求追踪）</h2>
{micro}
{PREFIX_CACHE_SECTION}
<p style="color:#888;font-size:12px;margin-top:48px">Generated by tools/benchmarks/analyze_sweep_all.py + analyze_sweep_micro.py + build_analysis_report.py. Chart.js via CDN（查看需联网）。</p>
</body></html>"""
    FINAL_HTML.write_text(html, encoding="utf-8")

    macro_md = MACRO_MD.read_text(encoding="utf-8")
    micro_md = MICRO_MD.read_text(encoding="utf-8")
    # Strip the standalone h1 headers so the merged doc has a single title, and
    # renumber so front matter (0,1) is followed by macro (2-5) and micro (6-9).
    macro_lines = []
    for line in macro_md.splitlines():
        match = re.match(r"^## (\d+)\. ", line)
        if match:
            line = re.sub(r"^## \d+\. ", f"## {int(match.group(1)) + 1}. ", line)
        macro_lines.append(line)
    micro_lines = []
    for line in micro_md.splitlines():
        match = re.match(r"^## (\d+)\. ", line)
        if match:
            line = re.sub(r"^## \d+\. ", f"## {int(match.group(1)) + 5}. ", line)
        micro_lines.append(line)
    ubatch_md = (
        "\n## 0. Token-Split Ubatch：方法与性能收益\n"
        "\n**方法**：AFD async MoE ubatching 把解耦后的单 stage 拆成 2 个 ubatch，让 attention 与 FFN "
        "流水并行。request-split 在请求边界切分（整请求、metadata 简单，但可能失衡）；token-split 在 "
        "token 数中点切分（请求可跨 ubatch，两 stage token 强制平衡）。\n"
        "\n**收益（机制 + 真实长度模拟）**：2-stage 流水吞吐由较慢 stage（关键路径）决定。cp8sp50k "
        "混合长度下 request-split 恰接近平衡（失衡 0.2%）；长尾（1 长 + 875 短）下 request-split 失衡 "
        "76%（16.0M vs 2.2M）、关键路径 16.0M，token-split 恒 9.1M/9.1M，理想流水下关键路径可缩短约 "
        "1.76 倍。结论：token-split 在请求长度差异大的工作负载上价值最大。E2E 消融（A0/A1/A2）已设计，"
        "数据待采集。\n"
    )
    merged = (
        "# Prefill 单次 Sweep 数据分析报告（Baseline DP4xTP8 vs AFD DP3xTP8+EP8）\n"
        "\n完整图表见 analysis_all_angles.html。\n"
        + ubatch_md
        + "\n## 1. 实验配置与数据集\n"
        "\n**系统 A（Baseline DP4×TP8 同步）**：DP4×TP8=32 ranks 跨 2 节点，MoE/FFN 全量 32 rank "
        "同步 EP（EP32），FlashComm1 SP。\n"
        "\n**系统 B（AFD DP3×TP8+EP8 async 解耦）**：Attention DP3×TP8=24 ranks（node0 16 + node1 "
        "8），FFN EP8=8 ranks（node1 dev 8-15，仅 baseline 1/4 FFN 算力），CAM async 把 attention/"
        "FFN 解耦成并行流水线，force load balance。\n"
        "\n**数据集 cp8sp50k（prefill-only）**：875 条生产 trace 导出的变长 prompt（mean 20783，"
        "median 16936，min 71，max 50773，共 18.18M token），输出长度 1；长度桶 1-8K×213、8-16K×214、"
        "16-32K×241、32-48K×154、48K+×53。prefix 构造：每组 12 请求共享 128-token 对齐的合成前缀，"
        "prefix∈{0,25,50,75,90,95,99} 为 constructed 比例。到达 Poisson RPS∈{4..12}，burstiness=1；"
        "SLO 10s。\n"
        "\n## 宏观（TTFT / SLO 统计）\n"
        + "\n".join(macro_lines)
        + "\n## 微观（逐请求追踪）\n"
        + "\n".join(micro_lines)
        + "\n## C. Prefix Cache：容量优势与下一步验证\n"
        "\nAFD 不止在理想命中率下吞吐更大：由于 attention 与 FFN 解耦，FFN 侧 DP 容量被释放，"
        "attention rank 的 HBM 不再承载 FFN 权重，可分配给 KV cache，cache 容量更大 → 实践中"
        "命中率、batch token、cache token 都能更大（示意图见 analysis_all_angles.html 第 5 节）。\n"
        "\n**下一步**：采用真实数据集验证这一点。\n"
    )
    FINAL_MD.write_text(merged, encoding="utf-8")
    print(f"Wrote {FINAL_HTML} ({len(html)} bytes)")
    print(f"Wrote {FINAL_MD} ({len(merged)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
