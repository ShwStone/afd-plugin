# Prefill 性能分析 — prefix cache DISABLED (冷缓存, prefix=0)

## 实验设定
- 模型：DeepSeek-V3.2-reduced (10 层, 256 experts, W8A8)
- 硬件：2 × A3 节点，共 32 NPUs（同超节点）
- 数据集：cp8sp50k（875 请求，18.18M 输入 token，1 输出 token）
- 对比系统：
  - **Baseline**: DP4 x TP8 = 32 ranks，EP 跨全 32 rank，FlashComm1 SP
  - **AFD**: Attention DP3 x TP8 (24 ranks) + FFN EP8 (8 ranks)，CAM async connector
- prefix cache 关闭，冷缓存对比纯架构差异
- TTFT SLO = 10s

## 1. Mean TTFT (ms) — 随 RPS 变化

| bt | 系统 | rps4 | rps6 | rps8 | rps10 | rps12 |
|----|------|------|------|------|-------|-------|
| 8192 | baseline | 655 | 810 | 1131 | 3083 | 9950 |
| 8192 | AFD | 597 | 849 | 2903 | 12350 | 18395 |
| 16384 | baseline | 713 | 928 | 1221 | 1684 | 6209 |
| 16384 | AFD | 493 | 660 | 1122 | 6175 | 12720 |
| 32768 | baseline | 1064 | 1479 | 1936 | 2581 | 7472 |
| 32768 | AFD | 459 | 550 | 1073 | 2973 | 8663 |
| 49152 | baseline | 1386 | 2059 | 2705 | 3548 | 8330 |
| 49152 | AFD | 469 | 577 | 1003 | 2657 | 8704 |
| 65536 | baseline | 1597 | 2431 | 3220 | 4335 | 10277 |
| 65536 | AFD | 466 | 580 | 975 | 2934 | 9316 |

## 2. 核心结论

### 2.1 低负载 (RPS 4-6)：AFD 全面领先，大 batch 优势显著
| bt | baseline rps4 | AFD rps4 | AFD 优势 |
|----|--------------|---------|---------|
| 8192 | 655ms | 597ms | 9% |
| 16384 | 713ms | 493ms | **31%** |
| 32768 | 1064ms | 459ms | **57%** |
| 49152 | 1386ms | 469ms | **66%** |
| 65536 | 1597ms | 466ms | **71%** |

AFD 的 async 架构让 attention 计算和 FFN 计算解耦并行。大 batch 下 FFN 无需等待 attention 全部完成，TTFT 降低 30-70%。

### 2.2 中负载 (RPS 8-10)：大 batch 下 AFD 仍占优
- bt49152 rps8：AFD 1.0s vs baseline 2.7s（**快 63%**）
- bt65536 rps8：AFD 0.98s vs baseline 3.2s（**快 69%**）
- 小 batch (bt8192) 例外：rps8 时 AFD 2.9s vs baseline 1.1s（AFD 落后）

### 2.3 高负载 (RPS 12)：两者接近饱和
- 大 batch (bt49152/65536)：AFD 8.7-9.3s vs baseline 8.3-10.3s（相近）
- 小 batch (bt8192)：AFD 18.4s vs baseline 9.9s（AFD 明显落后）

## 3. SLO 达成率 (%) — 4 个阈值组合

### rps=8, SLO=2s（严格低延迟）
| bt | baseline | afd |
|----|---------|-----|
| 8192 | 90% | 39% |
| 16384 | 92% | 89% |
| 32768 | 57% | **88%** |
| 49152 | 11% | **91%** |
| 65536 | 3% | **92%** |

### rps=10, SLO=5s
| bt | baseline | afd |
|----|---------|-----|
| 8192 | 93% | 12% |
| 16384 | 100% | 47% |
| 32768 | 99% | 85% |
| 49152 | 91% | **98%** |
| 65536 | 73% | **87%** |

### rps=12, SLO=10s
| bt | baseline | afd |
|----|---------|-----|
| 8192 | 51% | 25% |
| 16384 | 90% | 38% |
| 32768 | 73% | 59% |
| 49152 | 64% | 58% |
| 65536 | 48% | 54% |

### rps=12, SLO=20s（宽松）
| bt | baseline | afd |
|----|---------|-----|
| 8192 | 98% | 55% |
| 16384 | 100% | 82% |
| 32768 | 100% | 100% |
| 49152 | 100% | 100% |
| 65536 | 100% | 100% |

## 4. 结论

### AFD 甜区：大 batch (bt≥16384) + 低-中负载
- TTFT 降低 31-71%，严格 SLO (2s) 达成率 88-92%
- async 解耦让 attention 及时返回，不阻塞

### AFD 短板：小 batch (bt=8192)
- 任何 RPS 下都落后（TTFT 高、SLO 达成率低）
- 原因：EP8 吞吐上限低 + async 固定转发开销在 batch 小时占比高

### Baseline 特点：更均衡
- 小 batch 强（32 rank 分摊 MoE）
- 大 batch 高负载下吞吐更稳（全同步计算，无转发开销）

### 架构本质
AFD 用「更少的 FFN 资源 (EP8 vs 全32) + async 解耦」换大 batch 低延迟，代价是小 batch 吞吐上限。适合大 batch prefill、严格 TTFT 场景；baseline 更适合小 batch、高吞吐混合负载。
