# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Render report/charts.html (Chart.js 4) from report/stats.json.

Same chart family as bench_results/prefill/all_charts.html (jsdelivr CDN).
Pure stdlib; all data baked into the HTML.

Usage: python3 tools/benchmarks/fp_render_report.py \
    --results bench_results/full_prefill_performance
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SLO_S = 50.0

C_BASE = "#4e79a7"
C_A2 = "#e15759"
C_A1 = "#59a14f"
C_A0 = "#9c755f"
C_GREY = "#bab0ac"

HEADER = """<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>DeepSeek-V3.2 全模型 Prefill 32 卡实验图表</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
body{font-family:-apple-system,'Segoe UI','PingFang SC',sans-serif;max-width:1180px;margin:24px auto;padding:0 16px;color:#222}
h1{font-size:22px} h2{font-size:17px;margin-top:42px;border-left:4px solid #4e79a7;padding-left:8px}
.note{color:#666;font-size:13px;margin:4px 0 8px}
.chart{position:relative;height:420px;margin-bottom:12px}
table{border-collapse:collapse;font-size:13px;margin:8px 0}
td,th{border:1px solid #ccc;padding:3px 10px;text-align:right}
th{background:#f2f2f2}
</style></head><body>
<h1>DeepSeek-V3.2 全模型（61 层 W8A8）Prefill 32 卡实验图表</h1>
<p class="note">baseline = DP4xTP8 EP32（两节点各 16 卡）；AFD = 16 Attention + 16 FFN 解耦。
SLO = TTFT p99 &le; 50s。数据源：00_accept / 01_fixed_batch / 02_profiles_32 / 03_capacity_32 + rank0 合并 trace。
生成：fp_render_report.py</p>
"""


def _chart(cid: str, config: dict, title: str, note: str = "") -> str:
    cfg = json.dumps(config)
    return (
        f"<h2>{title}</h2>\n"
        + (f'<p class="note">{note}</p>\n' if note else "")
        + f'<div class="chart"><canvas id="{cid}"></canvas></div>\n'
        + f"<script>new Chart(document.getElementById('{cid}'), {cfg});</script>\n"
    )


def _cap_series(points: dict) -> list[dict]:
    return sorted(points.values(), key=lambda p: p["target_tps"])


def render(stats: dict) -> str:
    html = [HEADER]
    cap = stats["capacity"]
    base_pts = _cap_series(cap["baseline"])
    a2_pts = _cap_series(cap["afd_a2"])

    # ---------------------------------------------------------- 1 capacity curve
    def _line_pts(pts, key):
        return [
            {"x": p["target_tps"], "y": round(p["ttft"][key], 2)} for p in pts
        ]

    html.append(
        _chart(
            "capacity",
            {
                "type": "line",
                "data": {
                    "datasets": [
                        {"label": "baseline p50", "data": _line_pts(base_pts, "p50"),
                         "borderColor": C_BASE, "borderDash": [6, 4], "fill": False},
                        {"label": "baseline p99", "data": _line_pts(base_pts, "p99"),
                         "borderColor": C_BASE, "backgroundColor": C_BASE, "fill": False},
                        {"label": "AFD A2 p50", "data": _line_pts(a2_pts, "p50"),
                         "borderColor": C_A2, "borderDash": [6, 4], "fill": False},
                        {"label": "AFD A2 p99", "data": _line_pts(a2_pts, "p99"),
                         "borderColor": C_A2, "backgroundColor": C_A2, "fill": False},
                        {"label": "SLO 50s", "data": [
                            {"x": 1500, "y": SLO_S}, {"x": 9500, "y": SLO_S}],
                         "borderColor": C_GREY, "borderDash": [2, 4],
                         "pointRadius": 0, "fill": False},
                    ]
                },
                "options": {
                    "scales": {
                        "x": {"type": "logarithmic",
                              "title": {"display": True, "text": "offered input tok/s (log)"}},
                        "y": {"title": {"display": True, "text": "TTFT (s)"}},
                    }
                },
            },
            "1. 容量曲线：TTFT vs 提供负载（screening 窗口 128 请求 / 1.34M tok）",
            "p99 越过 50s SLO 即 FAIL。baseline 拐点 ≈2600 tok/s，A2 ≈4400 tok/s（1.68×）。"
            "注意 baseline@4371(52.3s) 比 @3091(74.4s) 失败更轻——Mooncake 同刻突发的采样噪声，正式窗口裁决。",
        )
    )

    # ---------------------------------------------------------- 2 SLO attainment
    html.append(
        _chart(
            "slo",
            {
                "type": "line",
                "data": {
                    "datasets": [
                        {"label": "baseline", "data": [
                            {"x": p["target_tps"], "y": round(100 * p["slo_attain_rate"], 1)}
                            for p in base_pts], "borderColor": C_BASE,
                         "backgroundColor": C_BASE, "fill": False},
                        {"label": "AFD A2", "data": [
                            {"x": p["target_tps"], "y": round(100 * p["slo_attain_rate"], 1)}
                            for p in a2_pts], "borderColor": C_A2,
                         "backgroundColor": C_A2, "fill": False},
                    ]
                },
                "options": {
                    "scales": {
                        "x": {"type": "logarithmic",
                              "title": {"display": True, "text": "offered input tok/s (log)"}},
                        "y": {"min": 0, "max": 105,
                              "title": {"display": True, "text": "TTFT ≤ 50s 请求占比 (%)"}},
                    }
                },
            },
            "2. 逐请求 SLO 达成率 vs 负载",
            "p99 判定只看尾部 1%；本图显示整体请求面。A2 在 4371 tok/s 仍有 >98% 请求达标，"
            "baseline 在 3091 tok/s 已有 ~1/4 请求超时。",
        )
    )

    # ---------------------------------------------------------- 3 goodput
    html.append(
        _chart(
            "goodput",
            {
                "type": "line",
                "data": {
                    "datasets": [
                        {"label": "baseline goodput", "data": [
                            {"x": p["target_tps"], "y": round(p["goodput_tps"], 0)}
                            for p in base_pts], "borderColor": C_BASE,
                         "backgroundColor": C_BASE, "fill": False},
                        {"label": "AFD A2 goodput", "data": [
                            {"x": p["target_tps"], "y": round(p["goodput_tps"], 0)}
                            for p in a2_pts], "borderColor": C_A2,
                         "backgroundColor": C_A2, "fill": False},
                        {"label": "理想 (offered=goodput)", "data": [
                            {"x": 1500, "y": 1500}, {"x": 9500, "y": 9500}],
                         "borderColor": C_GREY, "borderDash": [2, 4],
                         "pointRadius": 0, "fill": False},
                    ]
                },
                "options": {
                    "scales": {
                        "x": {"type": "logarithmic",
                              "title": {"display": True, "text": "offered input tok/s (log)"}},
                        "y": {"title": {"display": True, "text": "完成 goodput tok/s"}},
                    }
                },
            },
            "3. Goodput（完成 token/s）vs 提供负载",
            "两者都能“做完”远超拐点的负载（排队而非失败），差别在延迟。"
            "A2 峰值 goodput ~5.4k tok/s，baseline ~4.2k；低于理想线 = 窗尾排空时间。",
        )
    )

    # ---------------------------------------------------------- 4 CDF @ common points
    def _cdf(pts_dict, name):
        pr = sorted(r["ttft_s"] for r in pts_dict[name]["per_request"])
        n = len(pr)
        return [
            {"x": round(t, 2), "y": round(100.0 * (i + 1) / n, 1)}
            for i, t in enumerate(pr)
        ]

    html.append(
        _chart(
            "cdf4371",
            {
                "type": "line",
                "data": {
                    "datasets": [
                        {"label": "baseline @4371", "data": _cdf(cap["baseline"], "4371tps"),
                         "borderColor": C_BASE, "pointRadius": 0, "fill": False, "borderWidth": 2},
                        {"label": "AFD A2 @4371", "data": _cdf(cap["afd_a2"], "4371tps"),
                         "borderColor": C_A2, "pointRadius": 0, "fill": False, "borderWidth": 2},
                        {"label": "baseline @2185.5", "data": _cdf(cap["baseline"], "2185p5tps"),
                         "borderColor": C_BASE, "borderDash": [6, 4], "pointRadius": 0, "fill": False},
                        {"label": "AFD A2 @2185.5", "data": _cdf(cap["afd_a2"], "2185p5tps"),
                         "borderColor": C_A2, "borderDash": [6, 4], "pointRadius": 0, "fill": False},
                    ]
                },
                "options": {
                    "scales": {
                        "x": {"type": "logarithmic",
                              "title": {"display": True, "text": "TTFT (s, log)"}},
                        "y": {"title": {"display": True, "text": "CDF (%)"}, "max": 100},
                    }
                },
            },
            "4. TTFT 经验 CDF（共同负载点）",
            "4371 tok/s 下 baseline p50 已高于 A2（24.0 vs 18.1s），尾部更是越线；"
            "2185.5 下两系统 p50 接近，A2 尾部明显更薄（40.2 vs 46.8s）。",
        )
    )

    # ---------------------------------------------------------- 5 TTFT vs length scatter @4371
    def _scatter(pts_dict, name):
        return [
            {"x": r["len"], "y": round(r["ttft_s"], 2)}
            for r in pts_dict[name]["per_request"]
        ]

    html.append(
        _chart(
            "lens4371",
            {
                "type": "scatter",
                "data": {
                    "datasets": [
                        {"label": "baseline @4371", "data": _scatter(cap["baseline"], "4371tps"),
                         "backgroundColor": C_BASE + "99"},
                        {"label": "AFD A2 @4371", "data": _scatter(cap["afd_a2"], "4371tps"),
                         "backgroundColor": C_A2 + "99"},
                    ]
                },
                "options": {
                    "scales": {
                        "x": {"title": {"display": True, "text": "输入长度 (tokens)"}},
                        "y": {"title": {"display": True, "text": "TTFT (s)"}},
                    }
                },
            },
            "5. TTFT vs 输入长度 @4371 tok/s",
            "下包络 = 服务时间（随长度增长），上方的点 = 排队抬升。"
            "两系统服务时间斜率接近，A2 的排队抬升更小且更均匀。",
        )
    )

    # ---------------------------------------------------------- 6 in-window dynamics
    def _ts(pts_dict, name):
        return [
            {"x": round(r["offset_s"], 1), "y": round(r["ttft_s"], 2)}
            for r in pts_dict[name]["per_request"]
        ]

    html.append(
        _chart(
            "dynamics",
            {
                "type": "scatter",
                "data": {
                    "datasets": [
                        {"label": "baseline @3090.76 (FAIL 74.4s)",
                         "data": _ts(cap["baseline"], "3090p76tps"),
                         "backgroundColor": C_BASE + "99"},
                        {"label": "A2 @8742 (FAIL 120.5s)",
                         "data": _ts(cap["afd_a2"], "8742tps"),
                         "backgroundColor": C_A2 + "99"},
                        {"label": "A2 @4371 (PASS 44.1s)",
                         "data": _ts(cap["afd_a2"], "4371tps"),
                         "backgroundColor": C_A1 + "99"},
                    ]
                },
                "options": {
                    "scales": {
                        "x": {"title": {"display": True, "text": "请求到达偏移 (s, 窗内)"}},
                        "y": {"title": {"display": True, "text": "TTFT (s)"}},
                    }
                },
            },
            "6. 窗内排队动态（到达时刻 → TTFT）",
            "过载点可见到达越早排队越深（队列单调积压），达标点则平坦。"
            "突发簇（同刻多点垂直堆叠）来自 Mooncake 原始到达的零间隔保留。",
        )
    )

    # ---------------------------------------------------------- 7 fixed batch bars
    fb = stats["fixed_batch"]
    batches = ["fixed_8k_balanced", "fixed_32k_balanced", "fixed_32k_long_short"]
    systems = [("baseline", C_BASE), ("afd_a1", C_A1), ("afd_a2", C_A2)]
    ds = []
    for system, color in systems:
        ds.append({
            "label": system,
            "data": [
                round(fb.get(system, {}).get(b, {}).get("wall", {}).get("p50", 0), 3)
                for b in batches
            ],
            "backgroundColor": color,
        })
    html.append(
        _chart(
            "fixedbatch",
            {
                "type": "bar",
                "data": {"labels": ["8K 均衡 (7,743 tok)", "32K 均衡 (30,857 tok)",
                                    "32K 长短混合 (33,704 tok)"],
                         "datasets": ds},
                "options": {
                    "scales": {"y": {"title": {"display": True,
                                               "text": "批次 wall 中位数 (s, 10 重复)"}}}
                },
            },
            "7. 固定批次完成时间（无排队，钉 DP rank 0）",
            "32K 均衡：A2 7.06s / A1 7.21s / baseline 10.01s → AFD 快 ~30%。"
            "A0（无流水）两次在 4 个 burst 内 507015 崩（SP+async 竞态），不可得。"
            "A2 的 32K 均衡有 1 个 71.7s 离群 burst（其余 6.9-7.6s），原因待查。",
        )
    )

    # ---------------------------------------------------------- 8 fairness in long-short
    ds2 = []
    for system, color in systems:
        cell = fb.get(system, {}).get("fixed_32k_long_short", {})
        ds2.append({
            "label": system,
            "data": [
                round(cell.get("short_req_ttft", {}).get("p50", 0), 2),
                round(cell.get("long_req_ttft", {}).get("p50", 0), 2),
            ],
            "backgroundColor": color,
        })
    html.append(
        _chart(
            "fairness",
            {
                "type": "bar",
                "data": {"labels": ["批内短请求 TTFT 中位", "批内长请求 TTFT 中位"],
                         "datasets": ds2},
                "options": {
                    "scales": {"y": {"title": {"display": True, "text": "TTFT (s)"}}}
                },
            },
            "8. 长短混合批次内的个请求公平性（32K long-short）",
            "三系统都把短请求压到和长请求一起等批次尾巴（同步批处理宿命）；"
            "AFD 整体低 ~24-30%。token 均分（A2）相对按请求拆分（A1）在此负载无额外收益（9.69 vs 9.72s）。",
        )
    )

    # ---------------------------------------------------------- 9 solo scaling
    acc = stats["accept"]
    def _solo(system):
        return [
            {"x": s["len"], "y": round(s["ttft_s"], 2)}
            for s in sorted(acc[system]["singles"], key=lambda x: x["len"])
        ]
    html.append(
        _chart(
            "solo",
            {
                "type": "line",
                "data": {
                    "datasets": [
                        {"label": "baseline", "data": _solo("baseline"),
                         "borderColor": C_BASE, "backgroundColor": C_BASE, "fill": False},
                        {"label": "AFD A2", "data": _solo("afd_a2"),
                         "borderColor": C_A2, "backgroundColor": C_A2, "fill": False},
                    ]
                },
                "options": {
                    "scales": {
                        "x": {"title": {"display": True, "text": "输入长度 (tokens)"}},
                        "y": {"title": {"display": True, "text": "空载单请求 TTFT (s)"}},
                    }
                },
            },
            "9. 空载单请求延迟扩展律（验收 §8.2）",
            "63,778 tok：A2 27.9s vs baseline 31.8s（-12%）；32,850：-14%。"
            "baseline 另有 4×~52K 并发钉副本验证（4/4 成功，43.9s）。",
        )
    )

    # ---------------------------------------------------------- 10 occupancy
    prof = stats["profiles"]
    rows = [
        ("baseline rank0", prof.get("baseline/rank0_summary", {})),
        ("A2 attention rank0", prof.get("afd_a2/attention", {})),
        ("A2 ffn rank0", prof.get("afd_a2/ffn", {})),
        ("A1 attention rank0", prof.get("afd_a1/attention", {})),
        ("A1 ffn rank0", prof.get("afd_a1/ffn", {})),
    ]
    html.append(
        _chart(
            "occupancy",
            {
                "type": "bar",
                "data": {
                    "labels": [r[0] for r in rows],
                    "datasets": [
                        {"label": "CAM 等待 (Notify_Wait)",
                         "data": [round(100 * r[1].get("cam_wait_ratio", 0), 1) for r in rows],
                         "backgroundColor": C_A0},
                        {"label": "其余忙碌",
                         "data": [round(100 * (r[1].get("busy_ratio", 0)
                                               - min(r[1].get("busy_ratio", 0),
                                                     r[1].get("cam_wait_ratio", 0))), 1)
                                  for r in rows],
                         "backgroundColor": C_A1},
                        {"label": "空泡",
                         "data": [round(100 * r[1].get("bubble_ratio", 0), 1) for r in rows],
                         "backgroundColor": C_GREY},
                    ],
                },
                "options": {
                    "indexAxis": "y",
                    "scales": {
                        "x": {"stacked": True, "max": 100,
                              "title": {"display": True, "text": "采集窗口占比 (%)"}},
                        "y": {"stacked": True},
                    },
                },
            },
            "10. Device 时间构成（rank0 全采集窗口并集，含首尾空泡）",
            "注意：全窗口并集未做逐 step 切片，仅供参考。FFN 的 98%“忙碌”里含大量"
            " dispatch_recv 阻塞等待（op 中位 42ms）——真实等待需看 flow 图。",
        )
    )

    # ---------------------------------------------------------- 11 flow queue delay
    fl = stats.get("flows", {})
    qd = fl.get("queue_delay_ms", {})
    order = ["afd.cam.dispatch_send", "afd.cam.dispatch_recv",
             "afd.cam.combine_send", "afd.cam.combine_recv"]
    html.append(
        _chart(
            "flowdelay",
            {
                "type": "bar",
                "data": {
                    "labels": ["attn dispatch_send", "ffn dispatch_recv",
                               "ffn combine_send", "attn combine_recv"],
                    "datasets": [
                        {"label": "p50",
                         "data": [round(qd.get(k, {}).get("p50", 0), 1) for k in order],
                         "backgroundColor": C_BASE},
                        {"label": "p99",
                         "data": [round(qd.get(k, {}).get("p99", 0), 1) for k in order],
                         "backgroundColor": C_A2},
                    ],
                },
                "options": {
                    "scales": {"y": {"type": "logarithmic",
                                     "title": {"display": True,
                                               "text": "标记→设备算子入队延迟 (ms, log)"}}}
                },
            },
            "11. CAM 标记→设备算子的排队延迟（profiled 32K 重放，同钟 FIFO 配对）",
            "attention 侧 dispatch_send 从首个 48.6ms 单调积压到 5.8s（队列失衡）；"
            "FFN 侧 recv/send 始终 ~1.5ms——FFN 设备队列恒空，瓶颈在 attention 侧算力。",
        )
    )

    html.append("</body></html>\n")
    return "\n".join(html)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    stats = json.loads((args.results / "report/stats.json").read_text())
    html = render(stats)
    out = args.results / "report/charts.html"
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out} ({len(html)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
