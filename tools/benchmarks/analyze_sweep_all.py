#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Macro analysis of the prefill single-pass sweep (angles A1-A4).

Reads the aggregated summaries produced by ``export_summary_csv.py`` and
``export_slo_csv.py`` and renders a self-contained HTML report (Chart.js via
CDN) plus a markdown conclusion file. No third-party Python packages required.

Angles covered
--------------
A1  Load-latency curves: TTFT (mean/p99) vs RPS for each batch limit.
A2  AFD/baseline mean-TTFT speedup heatmap over (batch_tokens x RPS).
A3  Prefix sensitivity at the anchor cell (bt=32768, rps=10).
A4  Batch scaling: TTFT vs batch_tokens for each RPS.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections.abc import Sequence
from pathlib import Path

SA = "dp4_tp8_sp"          # baseline system name
SB = "afd_dp3_tp8_ep8"     # AFD candidate system name
LA = "Baseline DP4xTP8"
LB = "AFD DP3xTP8+EP8"
BATCH = (8192, 16384, 32768, 49152, 65536)
RPS = (4, 6, 8, 10, 12)
PREFIX = (0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)
PREFIX_LABEL = {0.0: "p0", 0.25: "p25", 0.5: "p50", 0.75: "p75", 0.9: "p90", 0.95: "p95", 0.99: "p99"}
COL_A = "#2c7fb8"
COL_B = "#d95f0e"
TTFT_SLO_MS = 10_000.0

ReportSection = tuple[str, str, str, str]  # (title, what-it-shows, html, conclusion)


def _rate_str(rps: float) -> str:
    return str(rps).replace(".", "p")


def load_summary(summary_path: Path) -> dict[tuple[str, int, float, float], dict[str, float]]:
    """Load summary.csv -> {(system, bt, prefix, rps): {mean_ms, p99_ms, slo10}}."""
    rows: dict[tuple[str, int, float, float], dict[str, float]] = {}
    with summary_path.open(encoding="utf-8") as summary_file:
        reader = csv.DictReader(summary_file)
        for record in reader:
            key = (
                record["system"],
                int(record["bt"]),
                float(record["prefix"]),
                float(record["rps"]),
            )
            rows[key] = {
                "mean_ms": float(record["ttft_mean_ms"]),
                "p99_ms": float(record["ttft_p99_ms"]),
                "slo10": float(record["slo"]),
            }
    return rows


def load_slo(slo_path: Path) -> dict[tuple[str, int, float, float], dict[str, float]]:
    """Load slo_summary.csv -> {(system, bt, prefix, rps): {slo2s..slo20s}}."""
    rows: dict[tuple[str, int, float, float], dict[str, float]] = {}
    with slo_path.open(encoding="utf-8") as slo_file:
        reader = csv.DictReader(slo_file)
        for record in reader:
            key = (
                record["system"],
                int(record["bt"]),
                float(record["prefix"]),
                float(record["rps"]),
            )
            rows[key] = {
                "slo_2s": float(record["slo_2s"]),
                "slo_5s": float(record["slo_5s"]),
                "slo_10s": float(record["slo_10s"]),
                "slo_20s": float(record["slo_20s"]),
            }
    return rows


def _ds(label: str, data: list[float | None], color: str, dashed: bool = False) -> dict[str, object]:
    clean = [None if v is None else round(float(v), 1) for v in data]
    dataset: dict[str, object] = {
        "label": label,
        "data": clean,
        "borderColor": color,
        "backgroundColor": color,
        "fill": False,
        "tension": 0.2,
        "pointRadius": 3,
    }
    if dashed:
        dataset["borderDash"] = [5, 3]
    return dataset


def line_chart(
    canvas_id: str,
    labels: list[str],
    datasets: list[dict[str, object]],
    x_label: str,
    y_label: str,
    *,
    y_min: float | None = None,
    y_max: float | None = None,
) -> str:
    y_scale: dict[str, object] = {"title": {"display": True, "text": y_label}}
    if y_min is not None:
        y_scale["min"] = y_min
    if y_max is not None:
        y_scale["max"] = y_max
    config = {
        "type": "line",
        "data": {"labels": labels, "datasets": datasets},
        "options": {
            "responsive": True,
            "plugins": {"legend": {"labels": {"boxWidth": 12}}},
            "scales": {
                "x": {"title": {"display": True, "text": x_label}},
                "y": y_scale,
            },
        },
    }
    return (
        f'<div class="card"><canvas id="{canvas_id}" height="72"></canvas>'
        f"<script>new Chart(document.getElementById('{canvas_id}'), "
        f"{json.dumps(config)});</script></div>"
    )


def heatmap_table(
    canvas_id: str,
    row_labels: list[str],
    col_labels: list[str],
    values: list[list[float]],
    *,
    cell_text: list[list[str]] | None = None,
) -> str:
    """CSS-grid heatmap. value>1 green (AFD faster), <1 red (slower)."""
    cells_html: list[str] = []
    for r_index, row in enumerate(values):
        for c_index, value in enumerate(row):
            ratio = value
            # green scale for >1, red scale for <1 (log spacing)
            if ratio >= 1:
                intensity = min(1.0, (ratio - 1.0) / 3.0)
                color = "rgba(46,139,87,{:.2f})".format(0.15 + 0.8 * intensity)
            else:
                intensity = min(1.0, (1.0 - ratio) / 3.0)
                color = "rgba(178,34,34,{:.2f})".format(0.15 + 0.8 * intensity)
            label = cell_text[r_index][c_index] if cell_text else f"{ratio:.2f}"
            cells_html.append(
                f'<div style="background:{color};padding:8px;text-align:center;'
                f'border:1px solid #ddd;font-size:13px;color:#111">{label}</div>'
            )
    header = "".join(f'<div style="font-weight:bold;padding:6px;text-align:center">{c}</div>' for c in ["bt \\ rps", *col_labels])
    grid_rows = [header]
    for r_index, row_label in enumerate(row_labels):
        row_start = r_index * len(col_labels)
        row_cells = "".join(cells_html[row_start:row_start + len(col_labels)])
        grid_rows.append(
            f'<div style="font-weight:bold;padding:6px;text-align:center;'
            f'display:flex;align-items:center">{row_label}</div>{row_cells}'
        )
    return f"""
<div class="card"><h4 style="margin:0 0 8px">Mean TTFT ratio (baseline / AFD) — <b>&gt;1 green = AFD faster, &lt;1 red = AFD slower</b></h4>
<div style="display:grid;grid-template-columns:60px repeat({len(col_labels)},1fr);gap:2px;max-width:760px">
{''.join(grid_rows)}
</div>
<p style="font-size:12px;color:#666">prefix=0, cold cache. Cell value = baseline mean TTFT / AFD mean TTFT.</p>
</div>
"""


def _section(title: str, what: str, html: str, conclusion: str) -> ReportSection:
    return (title, what, html, conclusion)


# ---------------------------------------------------------------------------
# Angle builders
# ---------------------------------------------------------------------------

def angle_a1(summary: dict) -> ReportSection:
    """Load-latency curves: TTFT vs RPS per batch limit."""
    charts = []
    for bt in BATCH:
        labels = [f"r{r}" for r in RPS]
        b_mean = [summary.get((SA, bt, 0.0, r), {}).get("mean_ms") for r in RPS]
        a_mean = [summary.get((SB, bt, 0.0, r), {}).get("mean_ms") for r in RPS]
        b_p99 = [summary.get((SA, bt, 0.0, r), {}).get("p99_ms") for r in RPS]
        a_p99 = [summary.get((SB, bt, 0.0, r), {}).get("p99_ms") for r in RPS]
        charts.append(
            line_chart(
                f"a1_mean_{bt}",
                labels,
                [_ds(LA, b_mean, COL_A), _ds(LB, a_mean, COL_B)],
                "request rate (RPS)",
                f"Mean TTFT (ms)  bt={bt}",
            )
            + line_chart(
                f"a1_p99_{bt}",
                labels,
                [_ds(LA, b_p99, COL_A, dashed=True), _ds(LB, a_p99, COL_B, dashed=True)],
                "request rate (RPS)",
                f"P99 TTFT (ms)  bt={bt}",
            )
        )
    what = (
        "固定 batch 上限、固定 prefix=0（冷缓存），把每个系统的 Mean/P99 TTFT 画成 RPS 的"
        "负载-延迟曲线。曲线的弯曲点即系统在该配置下的饱和‘拐点’：拐点越靠左，能支撑的"
        "稳定负载越低。对比两条曲线的相对位置，就能看出 AFD 在哪些 batch/RPS 区间占优、"
        "哪些区间回退。"
    )
    conclusion = (
        "冷缓存下 AFD 的负载曲线呈明显的双 regime 结构：大 batch（bt≥32768）在低-中 RPS "
        "(4-8) 全面低于 baseline（均值低 44-76%，如 bt65536 rps6 快 4.2 倍、bt49152 rps6 快 "
        "3.6 倍），说明 async 解耦让 attention/FFN 并行、FFN 无需等整批 attention 完成；"
        "小 batch（bt=8192）在 rps8-10 落后 2.6-4 倍（EP8 只有 baseline 1/4 的 FFN 算力是硬"
        "瓶颈），rps4-6 则与 baseline 接近。P99 与 Mean 走势一致，说明这不是个别长尾请求"
        "造成，而是整体分布平移。"
    )
    return _section(
        "A1. 负载-延迟曲线（TTFT vs RPS，每 batch）",
        what,
        "".join(charts),
        conclusion,
    )


def angle_a2(summary: dict) -> ReportSection:
    """Speedup heatmap over (batch_tokens x RPS)."""
    values: list[list[float]] = []
    cell_text: list[list[str]] = []
    for bt in BATCH:
        row: list[float] = []
        text: list[str] = []
        for r in RPS:
            b = summary.get((SA, bt, 0.0, r), {}).get("mean_ms")
            a = summary.get((SB, bt, 0.0, r), {}).get("mean_ms")
            if b and a and b > 0:
                ratio = b / a
                row.append(ratio)
                text.append(f"{ratio:.2f}x\n\n({a:.0f}ms vs {b:.0f}ms)")
            else:
                row.append(1.0)
                text.append("--")
        values.append(row)
        cell_text.append(text)
    what = (
        "把整张冷缓存对比压缩成一张 (batch_tokens × RPS) 热图：每个格子 = baseline 与 AFD "
        "同 cell 的 Mean TTFT 比值。>1 表示 AFD 更快（绿），<1 表示 AFD 更慢（红）。"
        "这张图回答‘AFD 到底在哪些配置下该用、哪些不该用’。"
    )
    html = heatmap_table("a2_heatmap", [str(b) for b in BATCH], [f"r{r}" for r in RPS], values, cell_text=cell_text)
    conclusion = (
        "甜区（绿）集中在左下：bt∈{32768,49152,65536} × RPS∈{4,6,8}，加速 1.8-4.2 倍，"
        "峰值在 bt65536 rps6（4.19x，baseline 2431ms → AFD 580ms）。回退区（红）集中在 "
        "小 batch + 中高 RPS：bt8192 rps10 慢 4.0 倍（3083→12350ms）、bt16384 rps10 慢 3.7 倍。"
        "当 RPS=12 高压时两者拉平（0.49-1.16x），说明高负载下排队成为主导，AFD 的解耦收益"
        "被饱和抵消。结论：AFD 的甜区是大 batch（bt≥32768）× 低-中负载（rps≤8）的冷缓存"
        "prefill；小 batch 场景应保持全同步。"
    )
    return _section("A2. AFD/Baseline 加速比热力图", what, html, conclusion)


def angle_a4(summary: dict) -> ReportSection:
    """Prefix sensitivity at the anchor cell, at two RPS (sweet vs knee)."""
    bt = 32768
    labels = [PREFIX_LABEL[p] for p in PREFIX]

    def _panel(rps: float) -> str:
        b_mean = [summary.get((SA, bt, p, rps), {}).get("mean_ms") for p in PREFIX]
        a_mean = [summary.get((SB, bt, p, rps), {}).get("mean_ms") for p in PREFIX]
        b_p99 = [summary.get((SA, bt, p, rps), {}).get("p99_ms") for p in PREFIX]
        a_p99 = [summary.get((SB, bt, p, rps), {}).get("p99_ms") for p in PREFIX]
        ratio = [
            round(b / a, 2) if b and a and a > 0 else None
            for b, a in zip(b_mean, a_mean, strict=True)
        ]
        tag = "rps=6（AFD 甜区）" if rps == 6.0 else "rps=10（拐点 cell）"
        return (
            line_chart(
                f"a4_mean_{int(rps)}",
                labels,
                [_ds(LA, b_mean, COL_A), _ds(LB, a_mean, COL_B)],
                "prefix hit ratio",
                f"Mean TTFT (ms)  anchor bt={bt} {tag}",
            )
            + line_chart(
                f"a4_p99_{int(rps)}",
                labels,
                [_ds(LA, b_p99, COL_A, dashed=True), _ds(LB, a_p99, COL_B, dashed=True)],
                "prefix hit ratio",
                f"P99 TTFT (ms)  anchor bt={bt} {tag}",
            )
            + line_chart(
                f"a4_ratio_{int(rps)}",
                labels,
                [_ds("baseline/AFD mean-TTFT", ratio, "#6a329f")],
                "prefix hit ratio",
                f"AFD speedup ratio (baseline/AFD)  {tag}",
            )
        )

    what = (
        "在同一个 anchor cell（bt=32768）上扫描 prefix 命中率 {0..99}，分别在 AFD 甜区负载"
        "（rps6）和拐点负载（rps10）各画一组。横轴是请求前缀可被 KV-cache 命中的比例，纵轴"
        "是 Mean/P99 TTFT，第三行是 AFD/baseline 加速比。这组曲线量化‘cache 命中’这一杠杆对"
        "两系统各自有多强，并回答：AFD 的优势（冷缓存低负载）是否随命中率上升而保持、翻转？"
    )
    charts = _panel(6.0) + _panel(10.0)
    conclusion = (
        "prefix 命中是两系统共同的单点杠杆：rps10 下 baseline 从 p0 的 2.58s 降到 p99 的 0.15s "
        "（-94%），AFD 从 2.97s 降到 0.21s（-93%）。但 AFD 相对 baseline 的优势并非随命中率单调"
        "变化，而是出现在一个‘中段命中带’：p25/p50 时 AFD 在 rps8 快 2.4-2.6 倍、rps10 快 "
        "1.8-1.9 倍；p90+ 高命中下 AFD 反而慢 1.4-1.7 倍（rps10 p90：398ms vs 239ms）。冷缓存 p0 "
        "端则依赖负载：rps6-8 AFD 快 1.8-2.7 倍，rps10（baseline 拐点附近）AFD 反慢 15%。机制："
        "AFD 的收益来自 async 解耦把真实 compute 并行化——当命中率把待计算量压到缓存查找量级"
        "时收益归零，EP8 的转发/吞吐开销反而暴露；当负载逼近饱和时排队又吞掉并行收益。结论："
        "AFD 的价值是‘中段命中 × 非饱和负载’的窄带优势，冷缓存低负载或中段命中时显著，高命中"
        "生产流量下应关闭或改用全同步。"
    )
    return _section("A3. Prefix 敏感性（cache 命中杠杆）", what, charts, conclusion)


def angle_a5(summary: dict) -> ReportSection:
    """Batch scaling: TTFT vs batch_tokens per RPS."""
    charts = []
    for r in RPS:
        labels = [str(b) for b in BATCH]
        b_mean = [summary.get((SA, b, 0.0, r), {}).get("mean_ms") for b in BATCH]
        a_mean = [summary.get((SB, b, 0.0, r), {}).get("mean_ms") for b in BATCH]
        charts.append(
            line_chart(
                f"a5_{int(r)}",
                labels,
                [_ds(LA, b_mean, COL_A), _ds(LB, a_mean, COL_B)],
                "max batch tokens",
                f"Mean TTFT (ms)  RPS={r}",
            )
        )
    what = (
        "固定 RPS、prefix=0，把 TTFT 画成 max batch tokens 的函数（A1 的转置视角）。"
        "它回答两个问题：(1) 扩大 batch 上限是否带来更长的排队/延迟？(2) AFD 和 baseline "
        "对 batch 扩容的响应方向是否一致？"
    )
    conclusion = (
        "baseline 在低-中负载下 TTFT 随 batch 单调上涨（rps4: 从 bt8192 的 655ms 涨到 bt65536 "
        "的 1.60s），是典型的‘更大 batch = 更长的同步 prefill 链’。AFD 则相反：TTFT 从 bt8192 "
        "一路降到 bt32768，之后进入平台（rps4-6 下 bt32768→65536 稳定在 0.46-0.58s，rps6 三档 "
        "为 550/577/580ms，几乎水平）。机制：async 解耦后 batch 越大，attention 阶段越满、FFN "
        "并行利用率越高，固定转发开销被摊薄——这解释了甜区为什么在大 batch。注意小 batch 端 "
        "（bt8192 rps6）AFD 仍略慢于 baseline（849 vs 810ms）。rps12 高压下两条曲线趋于收敛，"
        "AFD 的解耦收益被饱和排队吞掉。"
    )
    return _section("A4. Batch 伸缩性（TTFT vs batch tokens）", what, "".join(charts), conclusion)


def md_inline_to_html(text: str) -> str:
    """Convert the small subset of markdown used in prose to HTML."""
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    return text


def render_fragment(sections: Sequence[ReportSection]) -> tuple[str, str]:
    """Render the angle sections to (html fragment, markdown conclusions)."""
    body_parts: list[str] = []
    md_parts: list[str] = []
    for index, (title, what, html, conclusion) in enumerate(sections, start=1):
        anchor = f"angle-a{index}"
        body_parts.append(f"""
<div class="angle" id="{anchor}">
  <h2>{index}. {title}</h2>
  <div class="what"><b>这一组数据要说明什么：</b>{md_inline_to_html(what)}</div>
  {html}
  <div class="conclusion"><b>数据支持的结论：</b>{md_inline_to_html(conclusion)}</div>
</div>""")
        md_parts.append(f"\n## {index}. {title}\n\n**这一组数据要说明什么：** {what}\n\n**数据支持的结论：** {conclusion}\n")
    return "\n".join(body_parts), "\n".join(md_parts)


def _write_report(
    out_dir: Path,
    sections: Sequence[ReportSection],
    summary: dict,
    slo_rows: dict,
) -> None:
    body_parts, md_parts = render_fragment(sections)

    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>Prefill 单次 Sweep 宏观分析</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
body{{font-family:sans-serif;margin:20px;background:#fafafa;color:#222}}
h1{{color:#222}} h2{{color:#333;border-bottom:2px solid #ddd;padding-bottom:6px;margin-top:36px}}
.what{{background:#eef4fb;border-left:4px solid #2c7fb8;padding:10px 14px;border-radius:4px;margin:10px 0}}
.conclusion{{background:#eaf7ef;border-left:4px solid #2e8b57;padding:10px 14px;border-radius:4px;margin:12px 0;line-height:1.55}}
.card{{background:#fff;border-radius:8px;padding:14px;margin:12px 0;box-shadow:0 1px 3px rgba(0,0,0,.1)}}
.toc a{{display:inline-block;margin:2px 10px 2px 0;color:#2c7fb8;text-decoration:none}}
</style></head><body>
<h1>Prefill 单次 Sweep 宏观分析（Baseline DP4xTP8 vs AFD DP3xTP8+EP8）</h1>
<p>数据源：<code>summary.csv</code>（350 cell）+ <code>slo_summary.csv</code>（2/5/10/20s SLO）。</p>
<div class="toc">{''.join(f'<a href="#angle-a{i}">{i}. {t.split(".")[0]}</a>' for i, (t, *_rest) in enumerate(sections, start=1))}</div>
{body_parts}
<p style="color:#888;font-size:12px;margin-top:40px">Generated by tools/benchmarks/analyze_sweep_all.py. Charts use Chart.js CDN (requires network when viewing).</p>
</body></html>"""

    out_dir.mkdir(parents=True, exist_ok=True)
    html_path = out_dir / "analysis_macro.html"
    md_path = out_dir / "analysis_macro.md"
    fragment_path = out_dir / "analysis_macro_fragment.html"
    html_path.write_text(html, encoding="utf-8")
    fragment_path.write_text(body_parts, encoding="utf-8")
    md_text = f"# Prefill 单次 Sweep 宏观分析（Baseline DP4xTP8 vs AFD DP3xTP8+EP8）\n\n数据源：summary.csv + slo_summary.csv。完整图表见 analysis_macro.html。{md_parts}\n"
    md_path.write_text(md_text, encoding="utf-8")
    print(f"Wrote {html_path} ({len(html)} bytes)")
    print(f"Wrote {md_path} ({len(md_text)} bytes)")


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=Path("bench_results/prefill/summary.csv"))
    parser.add_argument("--slo", type=Path, default=Path("bench_results/prefill/slo_summary.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("bench_results/prefill"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Render all macro angles."""
    args = _build_argument_parser().parse_args(argv)
    summary = load_summary(args.summary)
    slo_rows = load_slo(args.slo)
    expected = 350
    if len(summary) != expected:
        print(f"WARNING: summary.csv has {len(summary)} rows, expected {expected}")
    sections = [
        angle_a1(summary),
        angle_a2(summary),
        angle_a4(summary),
        angle_a5(summary),
    ]
    _write_report(args.output_dir, sections, summary, slo_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
