# DeepSeek-V4-Flash W8A8 端到端性能数据总汇总（2026-09-03）

> **2026-09-03 async scheduler 重做**：用户指令后所有实验开 vLLM async scheduler。
> 双机 32 卡全部配置（AFD 三分裂 + token-split&prefill_token_sum DPLB + baseline2x ×2mbt）
> 已在 async ON 下重做，结果见 §0 与 `async_sched/README.md`（含 DPLB 对照结论：
> formal_1 均匀长 prefill 下 token 债路由与请求数路由吞吐完全打平 ≤0.01%）。
> 下文 §1-§5 为 no-async 旧数据，保留作对照。

## 0. async scheduler 重做矩阵（2026-09-03，双机 32 卡，全配置 async ON）

AFD DP6TP4+DP8EP8 mbt=65536（eff / 峰值15s桶 / TTFT p50 / p99 / 排空）：

| 速率 | request | token | token+tokDPLB | off |
|---|---|---|---|---|
| 1x | 33,684 / 54.0K / 2.45 / 8.21 / 5.7s | 33,469 / 53.2K / 3.17 / 8.05 / 6.3s | 33,471 / 53.1K / 3.20 / 26.22* / 6.8s | 33,684 / 53.8K / 1.94 / 6.62 / 5.8s |
| 1.5x | 49,104 / 63.2K / 3.35 / 11.96 / 6.2s | 48,650 / 61.2K / 3.98 / 10.64 / 7.5s | 48,650 / 64.7K / 3.96 / 9.47 / 7.9s | 49,566 / 67.0K / 3.35 / 10.19 / 6.0s |
| 2x | 58,377 / 75.4K / 6.77 / 13.95 / 14.4s | 61,091 / 74.9K / 6.31 / 14.27 / 10.4s | 61,093 / 73.0K / 6.66 / 15.25 / 10.3s | 59,030 / 74.7K / 8.02 / 14.78 / 13.3s |

\* toklb 1x p99=26.22 为单运行离群（p95 仅 7.55），待复跑确认。

baseline2x（async ON）：8k = 32,841/46,912/57,108（1x/1.5x/2x eff）；32k = 31,277/41,373/53,611。
**同卡数 2x：AFD 61.1K vs baseline 57.1K → AFD +7.0%**（no-async 时为 +3.5%）。

详见 `async_sched/README.md`；数据 `async_sched/*.json`（15 格，全 512/512 零失败）。

## 以下为 no-async 旧数据（2026-09-01/02）

统一负载 = **formal_1**（512 请求 / 5,260,666 input tokens / output=1 / 长度 p50=6,628 p99=48,015 max=63,778 / 突发保留）。
速率档位基于原速 35,070 tok/s（150s 发完）：0.5x=17.5K、0.75x=26.3K、1x=35.1K、1.5x=52.6K、2x=70.1K。
指标：eff = 有效吞吐（总 tokens / wall），峰值15s桶 = 单 15s 窗口内完成请求的 input tokens / 15（服务率上限读数），TTFT 单位秒，排空 = 最后请求完成距窗口结束的秒数。
全部结果 `prompt_tokens` 全匹配；除标注外均 512/512 零失败。分析工具 `tools/itask/bucket15_summary.py`（口径统一可复算）。

栈：vllm 568afb3 + vllm-ascend e19e14da7（CWS compressor workspace）+ afd_plugin test/dsv4-afd-flash(-dplb)。
**所有配置 FLASHCOMM1=1（FFN 角色 TP1 物理不可开，除外）**、CWS 开、util 0.80、async-sched OFF、enforce-eager、chunked prefill、无前缀缓存。

## 1. 双机 32 卡：AFD DP6TP4 + FFN DP8EP8（24+8）

### 1a. mbt 扫描（request-split，1x 速率，09-01）

| mbt | eff | 峰值15s桶 | TTFT p50/p99 | 排空 |
|---|---|---|---|---|
| 8192 | 25,143 | 33.5K | 29.8 / 59.4 | 58.6s（1.39×超载跑不动） |
| 16384 | 33,256 | 43.2K | 3.19 / 14.7 | 8.2s |
| 32768 | 33,257 | 50.7K | 2.46 / 10.5 | 7.8s |
| 65536 | 33,258 | 55.3K | 2.22 / 7.29 | 7.5s |

**mbt 单调越大越好，65536 最优**（CWS 解锁 KV 容量）。

### 1b. split 模式 × 速率（mbt=65536，09-01/02）

| 速率 | request-split eff / 峰值桶 / p50 / p99 / 排空 | token-split eff / 峰值桶 / p50 / p99 / 排空 | split-off eff / 峰值桶 / p50 / p99 / 排空 |
|---|---|---|---|
| 1x | 33,258 / 55.3K / 2.22 / 7.29 / 7.5s | 33,685 / 47.2K / 4.04 / 8.91 / 6.1s | 33,048 / 57.6K / 2.46 / 8.49 / 8.4s |
| 1.5x | 47,336 / 66.6K / 4.45 / 11.37 / 11.1s | 46,912 / 71.5K / 4.98 / 10.52 / 11.4s | 46,088 / 62.7K / 3.55 / 13.35 / 13.7s |
| 2x | 60,396 / 77.2K / 5.44 / 13.91 / 12.0s | 58,374 / **76.8K** / 6.51 / 16.44 / 14.3s | 58,377 / 73.9K / 6.56 / 17.81 / 14.7s |

- **split 模式对吞吐无显著影响（≤2%，噪声带内）**——32 卡 mbt65536 拓扑下 MoE 串行段不是瓶颈，ubatch 流水无容量贡献；request-split 的 TTFT 尾部略优
- **容量下界 ≥70K tok/s，三模式均未触膝**（2x 排空仅 12-14.7s）

### 1c. 对照：baseline 双实例 2×DP4TP4EP16 + least_load_router（09-02）

| 速率 | baseline2x mbt8192 eff / 峰值桶 / p50 / p99 / 排空 | baseline2x mbt32768 eff / 峰值桶 / p50 / p99 / 排空 |
|---|---|---|
| 1x | 32,236 / 50.8K / 2.84 / 11.31 / 13.2s | 31,655 / 56.1K / 6.87 / 16.28 / 15.5s |
| 1.5x | 44,154 / 66.5K / 3.42 / 12.94 / 18.4s | 47,334 / 61.0K / 7.89 / 15.26 / 11.1s |
| 2x | 58,363 / 67.9K / 5.77 / 16.63 / 14.3s | 57,735 / 81.1K / 8.08 / 16.82 / 15.2s |

## 2. 同卡数对照结论（32 卡：AFD 24+8 vs baseline 2×16）

| 指标 | AFD（request-split mbt65536） | baseline2x 最优 | AFD 优势 |
|---|---|---|---|
| 2x 有效吞吐 | 60,396 | 58,363（8k） | +3.5% |
| 2x 峰值 15s 桶 | 77.2K | 67.9K（8k）/ 81.1K（32k 单峰，持续桶 69-73K） | **+13.7%（持续桶口径）** |
| 2x TTFT p99 | 13.91s | 16.63s | -16% |
| 排空（全档） | 7.5-12.0s | 13.2-18.4s | 短 16-43% |
| 轻载 p50（1x） | 2.22s | 2.84s（8k） | 优 |

- baseline 的 mbt 二选一权衡（8k 延迟 / 32k 吞吐，65536 OOM）被 AFD 解耦消除：大 chunk 服务率 + 低延迟兼得

## 3. 假设性对标：baseline 1x TTFT 不变 × 到达速率翻倍 vs 真实 2x

把 baseline 1x 的实测 TTFT/E2E 冻结、到达间隔压缩一半（等效 2x 到达），与真实 2x 负载的各配置同坐标比较
（甘特图 2x 页签中 `…@1x→2x到达(TTFT不变)` 两个虚拟配置）：

| 配置 | 虚拟 eff（2×窗口） | TTFT p50 | TTFT p99 | 排空 |
|---|---|---|---|---|
| baseline 2×8k @翻倍（虚拟） | 59,652 | **2.84** | **11.31** | 13.2s |
| baseline 2×32k @翻倍（虚拟） | 57,082 | 6.87 | 16.28 | 17.2s |
| **AFD req-split 真实 2x** | 60,396 | 5.44 | 13.91 | 12.0s |
| baseline 2×8k 真实 2x | 58,363 | 5.77 | 16.63 | 14.3s |

**读数**：虚拟化后 baseline 2×8k 的 p50/p99（2.84/11.31）优于 AFD 真实 2x（5.44/13.91）——说明 baseline 的 1x 轻载延迟底子好；
但真实翻倍后 baseline p50 退化到 5.77-8.08（排队吃掉优势），AFD 真实 2x 反而守住 5.44。
虚拟 vs 真实的差值就是排队代价：baseline p50 劣化 +2.9s（8k 双实例），AFD 从 1x 到 2x 只劣化 +3.2s 但起点更低（2.22→5.44）。

## 4. Profile 产物（单机 DP2TP4+EP8，2×64k 请求）

| 配置 | TTFT | 合并质量 | 文件 |
|---|---|---|---|
| request-split mbt10240（chunked 7 块） | 9.7 / 10.4s | 817/817 flows 完整 | merged_dsv4prof-64k-01.json.gz |
| token-split mbt65536（单 chunk） | 9.7 / **6.7s** | 430/432（2 条 layer37/stage1/rank2 缺 attention 侧，存证） | merged_dsv4prof-64k-ts-01.json.gz |

## 5. 归档

- 本地：`bench_results/dsv4_afd_flash_formal1/`（单机 AFD 矩阵）、`bench_results/dsv4_afd_flash_xnode32/`（xnode32 mbt/speed、`baseline2x/`、`baseline1x/`、`split_sweep/`、profile 产物、`gantt_interactive.html`）
- NAS：`shwstone/{formal1_results,xnode32_results,xnode32_baseline2x,baseline1x_results,xnode32_split,dsv4_profile_64k_ts}/`
