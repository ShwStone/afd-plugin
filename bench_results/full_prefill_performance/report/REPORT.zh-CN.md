# DeepSeek-V3.2 全模型 Prefill 32 卡资源对等实验报告

**Baseline DP4×TP8（EP32） vs AFD 16 Attention + 16 FFN 解耦**

- 模型：DeepSeek-V3.2 完整 61 层，W8A8（mtp-quarot），最长输入 63,778 token
- 硬件：2 × A3 节点，共 32 × 910B（同超节点 33.182.140.0/22），两系统等资源
- 数据集：Mooncake-WildChat prefill（`ShwStone/moonconv-wildchat-prefill` @d7d54c16，bundle fe4f751b6dab），到达模式为生产 trace 时间膨胀缩放（含原生的同刻突发）
- 计划：`docs/npu/DEEPSEEK_V3_2_FULL_PREFILL_PERFORMANCE_PLAN.zh-CN.md` @edd21e6
- 本报告覆盖：阶段零验收（§8）、阶段一机制实验（§9）、阶段二容量筛选（§10.2）
- 未覆盖：§10.3 正式三窗口测量、§11 64 卡实验（待进行）
- 交互图表：同目录 `charts.html`（11 张，Chart.js）
- 数据与口径：`report/stats.json`（由 `fp_build_report.py` / `fp_render_report.py` 生成）

## 有效性口径（先读）

- **SLO：TTFT p99 ≤ 50s**（从 10s 修订，用户批准）：61 层下 63,778 token 单请求空载就需 28–32s；Mooncake 同刻突发（最大 27 请求/时刻，零间隔保留）造成 ~40–50s 的负载无关 p99 底噪。10s 在任何负载下都不可达。
- **发送偏差门槛：p99 ≤ 100ms / max ≤ 250ms**（从 50ms 修订，用户批准）：客户端与服务端同 pod，CPU 竞争使整组突发均匀晚 ~13–57ms；对 20–50s 量级的排队测量不敏感。本报告所有数据点均通过该门槛（实际最差 p99 56.0ms）。
- 所有容量点：128/128 请求成功、usage 校验 prompt_tokens 零不匹配、两系统重放**同一**计划文件（SHA-256 一致）。

---

## 1. 核心发现总览

### 1.1 容量：AFD A2 约为 baseline 的 1.68×（主要结论）

TTFT p99 ≤ 50s 约束下的 screening 容量（§10.2 停规：界比 ≤1.20）：

| 系统 | 最后通过点 | 首个失败点 | 容量估计 |
|---|---:|---:|---:|
| baseline | 2599.01 tok/s（p99 49.36s，擦线） | 3090.76（p99 74.40s） | **≈2,600 tok/s** |
| AFD A2 | 4371.00 tok/s（p99 44.08s） | 5198.03（p99 70.74s） | **≈4,400 tok/s** |

**A2 / baseline ≈ 1.68×**。换算每卡：baseline 81 tok/s/卡，A2 137 tok/s/卡。

注意数据非单调（baseline 4371 的失败 52.3s 反而比 3091 的 74.4s 轻；A2 6182 比 5198 轻）——这是 Mooncake 突发的采样方差，单个 screening 点只做选点用，最终容量结论应以 §10.3 的 formal_0/1/2 三窗口为准（尚未运行）。

### 1.2 无排队固定批次：AFD 快 ~30%，A1≈A2

中位 10 次重复，钉 DP rank 0，无到达排队：

| 批次 | tokens | baseline | A1（按请求拆） | A2（按 token 拆） | A2 优势 |
|---|---:|---:|---:|---:|---:|
| 8K 均衡 | 7,743 | 2.095s | — | 1.501s | **-28%** |
| 32K 均衡 | 30,857 | 10.010s | 7.208s | 7.058s | **-29%** |
| 32K 长短混合 | 33,704 | 12.735s | 9.718s | 9.691s | **-24%** |

- **A1≈A2（差 <2%）**：按请求拆分已拿到几乎全部流水收益，token 均分在这两个批次上没有额外价值（与计划 §4.4 假设四的预期方向一致但未观察到正收益）。
- **A0（无流水）不可用**：两个批次均在最开始几个 burst 内 507015（DDR MTE 越界，attention 侧）崩溃（8K 1/10 成功，32K 0/10）——已知 SP+async+no-ubatch 竞态。流水开/关的归因 cell 在当前栈上不可得，记录为不稳定性发现。
- A2 32K 均衡有 1 个 71.7s 离群 burst（其余 6.9–7.6s），单次事件，原因待查。

### 1.3 瓶颈在 attention 侧算力，FFN 饥饿（trace flow 直证）

rank0 合并 trace（correlation marker + mstx + CANN 算子，同钟 FIFO 配对，1,856 条 flow）：

| 环节 | 标记→设备算子入队延迟 p50 | p99 |
|---|---:|---:|
| attention dispatch_send | **1,327 ms**（首个仅 48.6ms，随负载单调积压） | 5,662 ms |
| FFN dispatch_recv | 1.57 ms | 3.9 ms |
| FFN combine_send | 1.51 ms | 2.8 ms |
| attention combine_recv | **1,333 ms** | 5,700 ms |

attention 侧设备队列在压测重放下持续积压（min 48.6ms → max 5.8s），FFN 侧队列恒空（~1.5ms）。FFN 的"等待"发生在 dispatch_recv 算子内部（阻塞等数据，中位 42.2ms），而不是设备队列里。**这直接证实计划 §4.1 假设一：16A16F 的瓶颈是 attention 资源不足，FFN 有富余。**

### 1.4 排队才是 TTFT 的大头，且低载时就占 ~3/4

用验收单请求拟合空载服务时间（分段线性），把每请求 TTFT 分解为 服务 + 排队：

| 负载 (tok/s) | baseline 排队占比 | A2 排队占比 |
|---:|---:|---:|
| 2185.5（最低） | 77.8% | 73.5% |
| 4371 | 82.4% | 79.2% |
| 拐点附近 | 87.0% @3091 | 86.2% @5198 |
| 过载 | — | 93.0% @8742 |

**即使最低负载，排队也占 TTFT 约 3/4**——原因是 Mooncake 生产 trace 的同刻突发：2185.5 tok/s 点内就有 20 请求/253,833 token 的单时刻突发（该突发内 TTFT 跨 5.1–28.3s）。p99 底噪由此而来，不是服务慢，是突发内互相排队。

### 1.5 空载差距远小于容量差距（证实 §4.5 假设五）

空载单请求（验收 §8.2）：

| 输入长度 | baseline | A2 | 差距 |
|---:|---:|---:|---:|
| 32,850 | 12.75s | 10.97s | -14% |
| 51,976 | 26.33s | 20.68s | -21% |
| 63,778 | 31.75s | 27.93s | -12% |

空载只快 12–21%，但容量差 68%——**AFD 的收益主要体现在满足延迟目标时的容量，而非空载延迟**（假设五成立）。baseline 另有 4×~52K 并发钉副本验收（4/4 成功，43.9s，峰值 ~51.5GB/卡，最热 56.6GB）。

### 1.6 Goodput：两系统都能"做完"超拐点的负载，差别全在延迟

| 提供负载 | baseline goodput | A2 goodput |
|---:|---:|---:|
| 4371 tok/s | 4,227 tok/s（FAIL） | 4,252 tok/s（PASS） |
| 6181.5 | — | 5,445 tok/s（FAIL） |
| 8742 | — | 5,172 tok/s（FAIL，排空 105.7s） |

关键点：**同样在 4371 tok/s，两边 goodput 几乎一样（4227 vs 4252），但 baseline p99 52.3s  FAIL、A2 44.1s PASS**——A2 不是"做得更多"，是同样的吞吐下排队分布更好（请求面更均匀）。窗尾排空时间（drain）在各点 8.6–11.7s 之间，过载点升至 21.6–105.7s，是 p99 恶化的直接来源。

---

## 2. 分维度细节

### 2.1 容量筛选全点表（screening 窗口：128 请求 / 1,340,915 tok / 基窗 33.0s）

| target tok/s | 系统 | TTFT p50 | p95 | p99 | 判定 | 发送偏差 p99 | 排空 |
|---:|---|---:|---:|---:|---|---:|---:|
| 2185.5 | baseline | 17.85 | 46.73 | 46.78 | PASS | 56.0ms | 11.7s |
| 2185.5 | A2 | 12.50 | 33.57 | 40.21 | PASS | 46.8ms | 8.7s |
| 2599.01 | baseline | 15.22 | 46.06 | 49.36 | PASS（擦线） | 46.4ms | 11.1s |
| 3090.76 | baseline | 34.79 | 60.64 | 74.40 | FAIL | 40.2ms | 11.4s |
| 4371 | baseline | 24.03 | 42.55 | 52.30 | FAIL | 28.3ms | 10.4s |
| 4371 | A2 | 18.10 | 34.96 | 44.08 | PASS | 24.3ms | 8.6s |
| 5198.03 | A2 | 22.35 | 58.68 | 70.74 | FAIL | 24.8ms | 21.6s |
| 6181.53 | A2 | 37.19 | 58.31 | 62.60 | FAIL | 21.2ms | 28.9s |
| 8742 | A2 | 44.32 | 114.02 | 120.54 | FAIL | 15.9ms | 105.7s |

读法：A2 在 2185.5 的 p50 比 baseline 低 30%（12.5 vs 17.9s）；同负载 4371 下 A2 全分位更低。baseline 的 p50/p99 在 3091 处的剧烈跳变（15→35/49→74）显示其拐点比 A2 更"脆"。

### 2.2 Device 时间构成（rank0 全窗口并集，仅供方向参考）

| 采集 | busy | cam_wait | bubble |
|---|---:|---:|---:|
| baseline rank0 | 90.9% | 57.7% | 9.1% |
| A2 attention rank0 | 55.7% | 55.7% | 44.3% |
| A2 FFN rank0 | 98.2% | ~0% | 1.8% |
| A1 attention rank0 | 90.3% | 90.2% | 9.7% |
| A1 FFN rank0 | 98.2% | ~0% | 1.8% |

⚠️ 全窗口并集未做逐 step 切片（§9.4 正式指标提取待离线完成）：FFN 的 98.2% busy 含大量 dispatch_recv 阻塞等待（不是真算力占用）；A2 attention 的 55.7% busy 几乎全是 cam_wait。方向性读法：A2 下 attention 等 FFN 返回的时间占比大，符合 token 拆分流水的预期形态；§9.5 的 T_attention/T_ffn 正式比例留待切片后计算。

### 2.3 CAM 设备算子时长（rank0，profiled 重放）

| 算子 | n | p50 | p99 |
|---|---:|---:|---:|
| CamMoeDistributeDispatchSend | 348 | 0.37ms | 1.22ms |
| CamMoeDistributeDispatchRecv | 580 | 42.20ms（阻塞等数据） | 112.92ms |
| CamMoeDistributeCombineSend | 580 | 0.92ms | 1.56ms |
| CamMoeDistributeCombineRecv | 348 | 0.13ms | 3.26ms |

通信本身极快（亚毫秒–毫秒级），**CAM 不是瓶颈**；瓶颈是 attention 侧喂不出数据（§1.3）。

### 2.4 发送端健康度

所有点发送偏差 p99 15.9–56.0ms、max ≤ 64ms，全部在修订后的门檾内；偏差随系统负载升高反而下降（服务排队摊薄了同 pod CPU 竞争的相对影响）。验收阶段 8/8 token 校验 + 32/32 warmup 两系统均通过。

---

## 3. 计划假设的验证状态（§4）

| 假设 | 结论 | 证据 |
|---|---|---|
| 一：16A16F 瓶颈是 attention 资源不足 | **证实** | flow 图 attention 队列积压至 5.8s，FFN 队列恒空（§1.3） |
| 二：EP16 通信局部性/粒度优于 EP32 | 间接支持 | 固定批次 AFD 快 24–29%（§1.2）；CAM 算子本身亚毫秒（§2.3） |
| 三：A2 能隐藏 FFN 和通信时间 | 部分支持 | A2 attention busy 仅 55.7% 且几乎全 cam_wait——重叠发生了，但代价是 attention 大量等待（§2.2） |
| 四：token 均分只在长度偏斜时有额外价值 | 未观察到额外价值 | 长短混合批次 A2≈A1（9.69 vs 9.72s）（§1.2） |
| 五：收益在容量而非空载延迟 | **证实** | 空载 -12~21% vs 容量 +68%（§1.5） |

## 4. 异常与无效记录（完整披露）

1. **A0 507015 崩溃**（§1.2）：SP+async 竞态，已知问题，流水归因 cell 缺失。
2. **A2 32K 均衡 71.7s 离群 burst**：10 次中 1 次，未查明，正式阶段若复现需追查。
3. **SLO 与偏差门槛修订**（§"有效性口径"）：两次阈值修订均经用户批准；早期 run 的存储判定字段已按新口径重算（stats.json 为准）。
4. **baseline 2185.5 曾判无效后回溯有效**：dev 56ms 在 100ms 门檾下有效（p99 42.80s PASS）。
5. **2026-08-24 节点 wedge**：afd-perf-1 迁移后的物理节点上，16 个 worker 在 TSD 设备打开的文件锁上卡死（node1 正常）；delete+create 重建后恢复。与性能数据无关，仅影响当天进度。
6. **数据边界**：profile 指标是全窗口并集，§9.4/9.5 的逐 step 阶段时间比（T_attention/T_ffn）尚未提取；容量数字来自 screening（128 请求），正式容量以 §10.3 三窗口为准。

## 5. 下一步

1. §10.3 正式负载点：低载 ~1300（双系统 formal_0）、共同近拐点 ~2599（各 formal_0/1/2）、高容量点 ~4371（A2 ×3；baseline formal_0 起）。
2. §9.4/9.5 离线切片分析（本地 trace 即可做，不需 pod），给出 T_attention/T_ffn 正式比例，决定 64 卡拓扑（32A16F vs 48A16F）。
3. §11 64 卡周末实验（资源申请待批）。

## 6. 产物清单

- `report/stats.json`：全部汇总数字（含逐请求序列）
- `report/charts.html`：11 张交互图（容量曲线 / SLO 达成率 / goodput / CDF / 长度散点 / 窗内动态 / 固定批次 / 批内公平性 / solo 扩展律 / 占用构成 / flow 排队）
- 生成脚本：`tools/benchmarks/fp_build_report.py`、`tools/benchmarks/fp_render_report.py`
- 原始数据：`00_accept/`、`01_fixed_batch/`、`02_profiles_32/`（摘要；110G trace 在 NAS）、`03_capacity_32/`
- 合并 trace（含 flow 箭头）：仓库根目录 `merged_rank0_with_device_flows.json[.gz]`（Perfetto 打开）
