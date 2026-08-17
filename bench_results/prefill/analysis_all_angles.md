# Prefill 单次 Sweep 数据分析报告（Baseline DP4xTP8 vs AFD DP3xTP8+EP8）

完整图表见 analysis_all_angles.html。

## 0. Token-Split Ubatch：方法与性能收益

**方法**：AFD async MoE ubatching 把解耦后的单 stage 拆成 2 个 ubatch，让 attention 与 FFN 流水并行。request-split 在请求边界切分（整请求、metadata 简单，但可能失衡）；token-split 在 token 数中点切分（请求可跨 ubatch，两 stage token 强制平衡）。

**收益（机制 + 真实长度模拟）**：2-stage 流水吞吐由较慢 stage（关键路径）决定。cp8sp50k 混合长度下 request-split 恰接近平衡（失衡 0.2%）；长尾（1 长 + 875 短）下 request-split 失衡 76%（16.0M vs 2.2M）、关键路径 16.0M，token-split 恒 9.1M/9.1M，理想流水下关键路径可缩短约 1.76 倍。结论：token-split 在请求长度差异大的工作负载上价值最大。E2E 消融（A0/A1/A2）已设计，数据待采集。

## 1. 实验配置与数据集

**系统 A（Baseline DP4×TP8 同步）**：DP4×TP8=32 ranks 跨 2 节点，MoE/FFN 全量 32 rank 同步 EP（EP32），FlashComm1 SP。

**系统 B（AFD DP3×TP8+EP8 async 解耦）**：Attention DP3×TP8=24 ranks（node0 16 + node1 8），FFN EP8=8 ranks（node1 dev 8-15，仅 baseline 1/4 FFN 算力），CAM async 把 attention/FFN 解耦成并行流水线，force load balance。

**数据集 cp8sp50k（prefill-only）**：875 条生产 trace 导出的变长 prompt（mean 20783，median 16936，min 71，max 50773，共 18.18M token），输出长度 1；长度桶 1-8K×213、8-16K×214、16-32K×241、32-48K×154、48K+×53。prefix 构造：每组 12 请求共享 128-token 对齐的合成前缀，prefix∈{0,25,50,75,90,95,99} 为 constructed 比例。到达 Poisson RPS∈{4..12}，burstiness=1；SLO 10s。

## 宏观（TTFT / SLO 统计）
# Prefill 单次 Sweep 宏观分析（Baseline DP4xTP8 vs AFD DP3xTP8+EP8）

数据源：summary.csv + slo_summary.csv。完整图表见 analysis_macro.html。
## 2. A1. 负载-延迟曲线（TTFT vs RPS，每 batch）

**这一组数据要说明什么：** 固定 batch 上限、固定 prefix=0（冷缓存），把每个系统的 Mean/P99 TTFT 画成 RPS 的负载-延迟曲线。曲线的弯曲点即系统在该配置下的饱和‘拐点’：拐点越靠左，能支撑的稳定负载越低。对比两条曲线的相对位置，就能看出 AFD 在哪些 batch/RPS 区间占优、哪些区间回退。

**数据支持的结论：** 冷缓存下 AFD 的负载曲线呈明显的双 regime 结构：大 batch（bt≥32768）在低-中 RPS (4-8) 全面低于 baseline（均值低 44-76%，如 bt65536 rps6 快 4.2 倍、bt49152 rps6 快 3.6 倍），说明 async 解耦让 attention/FFN 并行、FFN 无需等整批 attention 完成；小 batch（bt=8192）在 rps8-10 落后 2.6-4 倍（EP8 只有 baseline 1/4 的 FFN 算力是硬瓶颈），rps4-6 则与 baseline 接近。P99 与 Mean 走势一致，说明这不是个别长尾请求造成，而是整体分布平移。


## 3. A2. AFD/Baseline 加速比热力图

**这一组数据要说明什么：** 把整张冷缓存对比压缩成一张 (batch_tokens × RPS) 热图：每个格子 = baseline 与 AFD 同 cell 的 Mean TTFT 比值。>1 表示 AFD 更快（绿），<1 表示 AFD 更慢（红）。这张图回答‘AFD 到底在哪些配置下该用、哪些不该用’。

**数据支持的结论：** 甜区（绿）集中在左下：bt∈{32768,49152,65536} × RPS∈{4,6,8}，加速 1.8-4.2 倍，峰值在 bt65536 rps6（4.19x，baseline 2431ms → AFD 580ms）。回退区（红）集中在 小 batch + 中高 RPS：bt8192 rps10 慢 4.0 倍（3083→12350ms）、bt16384 rps10 慢 3.7 倍。当 RPS=12 高压时两者拉平（0.49-1.16x），说明高负载下排队成为主导，AFD 的解耦收益被饱和抵消。结论：AFD 的甜区是大 batch（bt≥32768）× 低-中负载（rps≤8）的冷缓存prefill；小 batch 场景应保持全同步。


## 4. A3. Prefix 敏感性（cache 命中杠杆）

**这一组数据要说明什么：** 在同一个 anchor cell（bt=32768）上扫描 prefix 命中率 {0..99}，分别在 AFD 甜区负载（rps6）和拐点负载（rps10）各画一组。横轴是请求前缀可被 KV-cache 命中的比例，纵轴是 Mean/P99 TTFT，第三行是 AFD/baseline 加速比。这组曲线量化‘cache 命中’这一杠杆对两系统各自有多强，并回答：AFD 的优势（冷缓存低负载）是否随命中率上升而保持、翻转？

**数据支持的结论：** prefix 命中是两系统共同的单点杠杆：rps10 下 baseline 从 p0 的 2.58s 降到 p99 的 0.15s （-94%），AFD 从 2.97s 降到 0.21s（-93%）。但 AFD 相对 baseline 的优势并非随命中率单调变化，而是出现在一个‘中段命中带’：p25/p50 时 AFD 在 rps8 快 2.4-2.6 倍、rps10 快 1.8-1.9 倍；p90+ 高命中下 AFD 反而慢 1.4-1.7 倍（rps10 p90：398ms vs 239ms）。冷缓存 p0 端则依赖负载：rps6-8 AFD 快 1.8-2.7 倍，rps10（baseline 拐点附近）AFD 反慢 15%。机制：AFD 的收益来自 async 解耦把真实 compute 并行化——当命中率把待计算量压到缓存查找量级时收益归零，EP8 的转发/吞吐开销反而暴露；当负载逼近饱和时排队又吞掉并行收益。结论：AFD 的价值是‘中段命中 × 非饱和负载’的窄带优势，冷缓存低负载或中段命中时显著，高命中生产流量下应关闭或改用全同步。


## 5. A4. Batch 伸缩性（TTFT vs batch tokens）

**这一组数据要说明什么：** 固定 RPS、prefix=0，把 TTFT 画成 max batch tokens 的函数（A1 的转置视角）。它回答两个问题：(1) 扩大 batch 上限是否带来更长的排队/延迟？(2) AFD 和 baseline 对 batch 扩容的响应方向是否一致？

**数据支持的结论：** baseline 在低-中负载下 TTFT 随 batch 单调上涨（rps4: 从 bt8192 的 655ms 涨到 bt65536 的 1.60s），是典型的‘更大 batch = 更长的同步 prefill 链’。AFD 则相反：TTFT 从 bt8192 一路降到 bt32768，之后进入平台（rps4-6 下 bt32768→65536 稳定在 0.46-0.58s，rps6 三档 为 550/577/580ms，几乎水平）。机制：async 解耦后 batch 越大，attention 阶段越满、FFN 并行利用率越高，固定转发开销被摊薄——这解释了甜区为什么在大 batch。注意小 batch 端 （bt8192 rps6）AFD 仍略慢于 baseline（849 vs 810ms）。rps12 高压下两条曲线趋于收敛，AFD 的解耦收益被饱和排队吞掉。

## 微观（逐请求追踪）

## 6. B1. 负载健康度：TTFT × 长度散点（bt=65536，RPS 4-12）

**这一组数据要说明什么：** 固定 batch=65536，把 RPS 从 4 依次摆到 12，每个负载下把 875 个请求的 (input_len, TTFT) 逐条画成散点（线性轴）。判定规则：**如果系统负载健康，TTFT 与输入长度呈明显正比**（散点向上倾斜、相关系数 r 高）——请求按自身计算量完成；**一旦饱和，队列等待主导，TTFT 几乎不随 token 数量变化**（散点变平、r 趋近 0）。横向比较两系统在哪个 RPS 还保持正比、哪个先塌平，就能读出各自的容量边界。

**数据支持的结论：** bt=65536 的负载健康度对比：**AFD 在 RPS 4-6 呈明显正比（r=0.73/0.51，mean 466-580ms）**，RPS8 开始塌陷（r=0.29，975ms），RPS10-12 完全饱和（r=0.07-0.14，2.9-9.3s）——从健康到不健康的转变清晰可见，且发生在更晚的负载点。**baseline 由于同步大 batch 预填充的延迟地板，在所有 RPS 都是平坦带（r 仅 0.09-0.25）**——即便 rps4 低负载，短请求也要等整批调度/完成，TTFT 与自身长度几乎无关（1.6-3.2s）。以 r≈0.3 作为进入不健康区的判据，AFD 撑到 RPS8（975ms），baseline 在 RPS4 就已 r≈0.23 且延迟 1.6s。结论：AFD 能承受约 **2 倍**负载才进入与 baseline 相同的‘不健康’状态，全程延迟更低（RPS8: 975ms vs 3220ms）——容量更大，适合更高的 prefill 到达率。


## 7. B2. 到达序 TTFT 时序（排队波）

**这一组数据要说明什么：** 按客户端发出的顺序（0-874，Poisson 到达）把每个请求的 TTFT 画成序列。它回答：TTFT 是否随时间出现‘排队波’（晚到的请求因队列积压而变慢）？AFD 的慢请求是均匀散布（每请求系统性变慢）还是集中在某段（某波到达/队列堆积）？

**数据支持的结论：** 回退 cell（bt8192 rps10）的时序证据是决定性的：AFD 的 mean TTFT 随到达序单调暴涨——前 1/3 5.2s → 中段 12.3s → 后 1/3 19.5s（3.7 倍），baseline 只从 2.2s 缓升到 3.9s。这是**吞吐不达标的排队崩溃**：AFD 每单位时间完成的请求数 < 到达率 10 req/s，队列持续累积，后到的请求等待越来越久——不是单次突发、也不是每请求固定开销。甜区（bt65536 rps6）AFD 三段恒定 581/579/579ms，无任何积压。结论：AFD 的甜区/回退本质是吞吐容量差异（EP8 只有 baseline 1/4 FFN 算力），小 batch 高压下容量不足导致排队崩溃，大 batch 低负载下容量富余且 async 并行生效。


## 8. B3. 长度桶分解（收益归因 + 严格 SLO 达成）

**这一组数据要说明什么：** 把每个 cell 的请求按输入长度分桶（≤8K/8-16K/16-32K/32-48K/48K+）。上排是两系统各桶的均值 TTFT，下排是各桶在严格 SLO（2s/5s）下的达成率。它把‘整 cell 赢/输’归因到具体长度 regime，并回答运营问题：在贴近生产的严格 SLO 下，哪个长度桶是失守点？（这里不用 10s——prefill 的严格目标更接近 2-5s。）

**数据支持的结论：** 甜区（bt65536 rps6）AFD 在**每个长度桶**都更优（357-949ms vs 2299-2781ms），且优势对短请求最大：≤8K 桶快 6.4 倍、48K+ 桶快 2.9 倍——async 并行收益随请求变长相对缩水（长请求 FFN 计算量更大，EP8 瓶颈占比上升）。按 2s 严格 SLO，甜区所有桶 AFD 都 100%达标，而 baseline 只有 1-8K 桶达标、长桶掉到 0-50%——‘长请求在 baseline 大 batch 冷缓存下必然超 2s’是 B4 给出的可操作结论。回退区（bt8192 rps10）AFD 每个桶都更差但比值均匀（4.4x→3.7x，无长度单调性）——该 cell 已排队饱和，等待时间主导、长度无关。


## 9. B4. Prefix 命中深挖（逐请求）

**这一组数据要说明什么：** 在同一个 anchor cell（bt=32768, rps=10）上，用 prefix 0/50/90/99 四档逐请求数据做三组分析：(1) 每个系统在各 prefix 下的 TTFT CDF——量化‘cache 命中’如何整体压扁分布；(2) 长度 × prefix 交互——命中是否消除长请求的延迟惩罚；(3) AFD/baseline 比值随 prefix 的变化（mean/p50/p90/p99 四档）——定位 AFD 的收益区间并验证其机制。

**数据支持的结论：** 三组图给出 AFD 收益的完整机制解释。**CDF**：从 p0 到 p99，baseline 中位 TTFT 从 2.50s 压到 ~0.15s，AFD 从 2.85s 压到 ~0.17s——命中率是两系统共同的最强杠杆（-94%）。**长度×prefix**：p0 下两系统都有 ~1.3 倍的长度惩罚（baseline 2.3s→3.1s、AFD 2.9s→3.7s），且该 cell（anchor rps10 高压）AFD 在每个桶都更高；p99 下两系统都塌到 124-421ms，但 AFD 保留了更大的相对长度惩罚（48K+ vs 1-8K：3.0x vs baseline 1.9x）——即使 1% 残余计算，长请求在 EP8 上也更吃力。**比值曲线**：AFD 的优势是**中段命中窄带**——p50 快 1.8 倍（685ms vs 1256ms），而 p0（冷缓存但已近饱和负载）与 p90+（计算已被缓存压到量级）两端都反超（AFD 慢 1.15-1.7 倍）。**机制**：AFD 用 24-rank attention + 8-rank FFN（baseline 的 1/4 FFN 算力），靠 async CAM 把 attention/FFN 解耦成两条并行流水线，代价是固定的 dispatch/combine 转发开销。当**计算密集（大 batch、冷缓存）且负载不饱和**时，并行收益盖过开销 → 大幅占优（见 A2/A5 的甜区）；当负载逼近饱和（rps10 p0）或计算被命中压没（p90+），EP8 的吞吐短板与固定开销成为主导 → 反超。因此 AFD 的价值是‘计算密集 × 负载不饱和’的窄带，适用前缀命中低、prefill 计算重的低-中负载场景。
## C. Prefix Cache：容量优势与下一步验证

AFD 不止在理想命中率下吞吐更大：由于 attention 与 FFN 解耦，FFN 侧 DP 容量被释放，attention rank 的 HBM 不再承载 FFN 权重，可分配给 KV cache，cache 容量更大 → 实践中命中率、batch token、cache token 都能更大（示意图见 analysis_all_angles.html 第 5 节）。

**下一步**：采用真实数据集验证这一点。
