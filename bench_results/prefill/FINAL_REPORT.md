# Prefill 性能对比最终报告

**Baseline DP4xTP8 vs AFD DP3xTP8+EP8**

- 模型：DeepSeek-V3.2-reduced（10 层，256 experts，W8A8）
- 硬件：2 × A3 节点，32 NPUs（同超节点）
- 数据集：cp8sp50k（875 请求，18.18M 输入 token，1 输出 token）
- 完整矩阵：5 batch × 7 prefix × 5 RPS = 350 cell，全部完成

## 实验配置

| | Baseline DP4xTP8 | AFD DP3xTP8+EP8 |
|---|---|---|
| Attention | DP4 × TP8 = 32 ranks | DP3 × TP8 = 24 ranks |
| FFN/MoE | 全 32 rank EP | EP8（8 ranks） |
| 并行 | FlashComm1 SP | CAM async 解耦 |
| prefix cache | 关闭/开启 | 关闭/开启 |

## 1. 核心发现总览

### 1.1 Prefix 命中率是最大性能杠杆（两系统都受益 74-95%）
RPS=8 下 TTFT 从 prefix=0 到 prefix=0.99：
- baseline：降低 87-95%
- AFD：降低 74-94%
高 prefix 命中下两者都降到 150-260ms。

### 1.2 AFD 优势：大 batch + 冷缓存（低-中负载）
| bt | baseline rps4 | AFD rps4 | AFD 优势 |
|----|--------------|---------|---------|
| 16384 | 713ms | 493ms | **31%** |
| 32768 | 1064ms | 459ms | **57%** |
| 49152 | 1386ms | 469ms | **66%** |
| 65536 | 1597ms | 466ms | **71%** |

bt≥16384 时 AFD TTFT 降 31-71%。原因：async 解耦让 attention 和 FFN 并行，FFN 无需等 attention 全部完成。

### 1.3 AFD 劣势：小 batch + 冷缓存（bt=8192）
| rps | baseline | AFD | 差距 |
|-----|---------|-----|------|
| 8 | 1131ms | 2903ms | AFD 慢 2.6x |
| 10 | 3083ms | 12350ms | AFD 慢 4x |
| 12 | 9950ms | 18395ms | AFD 慢 1.8x |

原因：EP8 吞吐上限只有 baseline 的 1/4 + async 固定转发开销在小 batch 占比高。

### 1.4 Baseline 更均衡
- 小 batch 强（32 rank 分摊 MoE）
- 大 batch 高负载吞吐更稳（全同步，无转发开销）
- 但大 batch 冷缓存下 TTFT 长（async 缺失导致 attention 串行）

## 2. 分维度数据

### 2.1 Prefix 敏感性（Mean TTFT @ RPS8，ms）
| bt | baseline p0→p99 | AFD p0→p99 |
|----|----------------|-----------|
| 8192 | 1131→151 (-87%) | 2903→180 (-94%) |
| 16384 | 1221→152 (-88%) | 1122→149 (-87%) |
| 32768 | 1936→164 (-92%) | 1073→166 (-85%) |
| 49152 | 2705→168 (-94%) | 1003→199 (-80%) |
| 65536 | 3220→164 (-95%) | 975→256 (-74%) |

### 2.2 Batch 敏感性（Mean TTFT @ RPS8，ms）
| system | prefix | bt8192→bt65536 |
|--------|--------|---------------|
| baseline | 0 | 1131→3220 |
| baseline | 0.99 | 151→164 |
| AFD | 0 | 2903→**975**（反而下降） |
| AFD | 0.99 | 180→256 |

**AFD 冷缓存下 bt 增大反而更快**：async 解耦在大 batch 下让 FFN 并行效率更高。这是 AFD 最显著的特征。

### 2.3 SLO 达成率（prefix=0）
| 场景 | baseline | AFD |
|------|---------|-----|
| rps8 slo2s @bt65536 | 3% | **92%** |
| rps10 slo5s @bt49152 | 91% | **98%** |
| rps12 slo10s @bt8192 | 51% | 25% |
| rps12 slo20s @bt8192 | 98% | 55% |

严格 SLO（2s）下 AFD 在大 batch 优势巨大（92% vs 3%）；宽松 SLO（20s）下小 batch baseline 反超。

## 3. 结论

**AFD（async FFN 解耦 + EP8）** 的定位：
- ✅ 甜区：大 batch prefill（bt≥16384）+ 冷缓存低-中负载，TTFT 降 31-71%，严格 SLO 达成率 88-92%
- ✅ 大 batch 冷缓存下 TTFT 随 bt 增大反而下降（async 并行）
- ❌ 短板：小 batch（bt8192）任何负载下都落后，EP8 吞吐上限是瓶颈
- ❌ 高负载（rps≥12）下 EP8 饱和，SLO 崩溃早于 baseline

**Baseline（全 32 rank 同步 MoE）**：
- ✅ 均衡，小 batch 强，高负载吞吐稳
- ❌ 大 batch 冷缓存 TTFT 长（无 async）

**架构本质**：AFD 用「更少 FFN 资源 + async 解耦」换大 batch 低延迟，适合大 batch、严格 TTFT 的 prefill 场景；baseline 更适合小 batch、高吞吐混合负载。

## 4. 产物清单
- `all_charts.html`：全维度交互图（TTFT vs batch / RPS / prefix，SLO vs batch）
- `paper_charts.html`：论文版图（prefix=0）
- `report_prefix0.html`：冷缓存专用图
- `summary.csv` / `slo_summary.csv`：完整数据
- `prefix0_ANALYSIS.md`：prefix=0 专项分析
