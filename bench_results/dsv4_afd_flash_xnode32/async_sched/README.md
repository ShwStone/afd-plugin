# async scheduler 重做矩阵（2026-09-03，双机 32 卡）

背景：用户指令 2026-09-03 —— 之后所有实验开 vLLM async scheduler（启动器默认已翻转为 ON，
commit 3279e3f/8231001），并把此前全部双机端到端实验在该条件下重做。本目录即重做结果。

统一负载 = formal_1（512 请求 / 5,260,666 input tokens / output=1 / 原速 35,070 tok/s=150s 发完；
1.5x=52.6K、2x=70.1K）。口径同前：eff=总 tokens/wall，峰值15s桶=单窗口完成请求 input tokens/15，
排空=最后请求完成距窗口结束秒数。全格 512/512 零失败、prompt 全匹配、tokens_mismatch=0。

栈：vllm 568afb3 + vllm-ascend 80d8c194f+2×CWS补丁 + afd_plugin test/dsv4-afd-flash-dplb-toklb
(db69b02=44501c2+prefill-token DPLB cherry-pick)。**全部配置 FLASHCOMM1=1、CWS 开、util 0.80、
async-scheduling ON**、chunked prefill、无前缀缓存。
pod：v4f-base-2（33.182.142.213）+ v4f-xnode-2（33.182.141.223，删除重建后的新节点）。

## 1. AFD DP6TP4 + FFN DP8EP8（24+8），mbt=65536 — split × DPLB × 速率

eff tok/s / 峰值15s桶 / TTFT p50 / TTFT p99 / 排空：

| 速率 | request-split | token-split | token-split + **prefill_token_sum DPLB** | split-off |
|---|---|---|---|---|
| 1x | 33,684 / 54.0K / 2.45 / 8.21 / 5.7s | 33,469 / 53.2K / 3.17 / 8.05 / 6.3s | 33,471 / 53.1K / 3.20 / **26.22** / 6.8s | 33,684 / 53.8K / 1.94 / 6.62 / 5.8s |
| 1.5x | 49,104 / 63.2K / 3.35 / 11.96 / 6.2s | 48,650 / 61.2K / 3.98 / 10.64 / 7.5s | 48,650 / 64.7K / 3.96 / 9.47 / 7.9s | 49,566 / 67.0K / 3.35 / 10.19 / 6.0s |
| 2x | 58,377 / 75.4K / 6.77 / 13.95 / 14.4s | 61,091 / 74.9K / 6.31 / 14.27 / 10.4s | 61,093 / 73.0K / 6.66 / 15.25 / 10.3s | 59,030 / 74.7K / 8.02 / 14.78 / 13.3s |

读数：
- **prefill_token_sum DPLB vs request_count（同 token-split）吞吐完全打平**（三档 eff 差 ≤2 tok/s，
  即 <0.01%）。formal_1 是均匀长 prefill 负载，token 债与请求数排序高度相关，两种计分选中的
  引擎序列几乎一致 → 该负载下 token-DPLB 无收益也无损失（1x p99=26.22 为单运行离群，
  p95 仅 7.55 与对照相当，需复跑确认）。
- split 模式在 async 下依旧无显著差异（2x 档 token 61.1K ≥ off 59.0K ≈ request 58.4K，≤4%）。
- async vs 旧 no-async（同 token-split）：1x 持平，1.5x +3.7%（48.7K vs 46.9K），2x +4.7%
  （61.1K vs 58.4K）；request-split 2x 略降（58.4K vs 60.4K）。整体 async 对 AFD 中性偏正。

## 2. 对照：baseline 双实例 2×DP4TP4EP16 + least_load_router（async ON）

| 速率 | mbt=8192 | mbt=32768 |
|---|---|---|
| 1x | 32,841 / 51.6K / 2.69 / 10.21 / 10.0s | 31,277 / 51.3K / 9.92 / 19.42 / 17.6s |
| 1.5x | 46,912 / 59.9K / 4.17 / 13.52 / 11.8s | 41,373 / 62.6K / 11.48 / 25.05 / 26.7s |
| 2x | 57,108 / 72.5K / 5.97 / 16.16 / 16.2s | 53,611 / 87.0K / 12.44 / 22.87 / 22.2s |

读数：
- async 下 mbt=32768 启动 KV estimate 一次通过（此前担心 async 翻倍 in-flight 预算会卡，未复现）。
- baseline 32k 的 mbt 权衡在 async 下加剧：32k 吞吐峰值桶更高（87.0K）但延迟显著劣化
  （2x p50 12.44 vs 8k 的 5.97）；8k 仍是 baseline 综合最优。
- async vs 旧 no-async baseline2x 8k：1x 32.8K vs 32.2K（+2%）、2x 57.1K vs 58.4K（-2%）——噪声带内。

## 3. 同卡数对照（32 卡，async ON）：AFD 24+8 vs baseline 2×16

| 指标 | AFD 最优（token/toklb） | baseline2x 最优 | AFD 优势 |
|---|---|---|---|
| 2x eff | 61,091 | 57,108（8k） | **+7.0%** |
| 2x TTFT p99 | 14.27s | 16.16s（8k） | -12% |
| 2x TTFT p50 | 6.31s | 5.97s（8k） | 持平（-6%） |
| 排空（全档） | 5.7-14.4s | 10.0-26.7s | 短 40%+ |
| 轻载 p50（1x） | 2.45（request） | 2.69（8k） | 优 |

async 条件下 AFD 对 baseline 的吞吐优势比 no-async 时（+3.5%）扩大到 **+7%**。

## 4. 文件

- `toklb_as_{1x,fast1p5x,fast2x}.json`：token-split + prefill_token_sum DPLB
- `xnode32_{request,token,off}_as_mbt65536_{1x,fast1p5x,fast2x}.json`
- `base2x_as_mbt{8192,32768}_{1x,fast1p5x,fast2x}.json`
- NAS：`shwstone/{xnode32_toklb_as,xnode32_split_as,xnode32_baseline2x_as}/`（含三侧日志、router decisions）
- 驱动：`tools/itask/{xnode32_toklb_runs.sh,xnode32_split_runs.sh,xnode32_baseline2x_runs.sh}`
  （split 驱动新增 SPLIT=request 模式；三者 TAG/NASDIR 均带 `_as` 后缀）
