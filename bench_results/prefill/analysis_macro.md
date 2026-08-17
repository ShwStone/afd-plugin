# Prefill 单次 Sweep 宏观分析（Baseline DP4xTP8 vs AFD DP3xTP8+EP8）

数据源：summary.csv + slo_summary.csv。完整图表见 analysis_macro.html。
## 1. A1. 负载-延迟曲线（TTFT vs RPS，每 batch）

**这一组数据要说明什么：** 固定 batch 上限、固定 prefix=0（冷缓存），把每个系统的 Mean/P99 TTFT 画成 RPS 的负载-延迟曲线。曲线的弯曲点即系统在该配置下的饱和‘拐点’：拐点越靠左，能支撑的稳定负载越低。对比两条曲线的相对位置，就能看出 AFD 在哪些 batch/RPS 区间占优、哪些区间回退。

**数据支持的结论：** 冷缓存下 AFD 的负载曲线呈明显的双 regime 结构：大 batch（bt≥32768）在低-中 RPS (4-8) 全面低于 baseline（均值低 44-76%，如 bt65536 rps6 快 4.2 倍、bt49152 rps6 快 3.6 倍），说明 async 解耦让 attention/FFN 并行、FFN 无需等整批 attention 完成；小 batch（bt=8192）在 rps8-10 落后 2.6-4 倍（EP8 只有 baseline 1/4 的 FFN 算力是硬瓶颈），rps4-6 则与 baseline 接近。P99 与 Mean 走势一致，说明这不是个别长尾请求造成，而是整体分布平移。


## 2. A2. AFD/Baseline 加速比热力图

**这一组数据要说明什么：** 把整张冷缓存对比压缩成一张 (batch_tokens × RPS) 热图：每个格子 = baseline 与 AFD 同 cell 的 Mean TTFT 比值。>1 表示 AFD 更快（绿），<1 表示 AFD 更慢（红）。这张图回答‘AFD 到底在哪些配置下该用、哪些不该用’。

**数据支持的结论：** 甜区（绿）集中在左下：bt∈{32768,49152,65536} × RPS∈{4,6,8}，加速 1.8-4.2 倍，峰值在 bt65536 rps6（4.19x，baseline 2431ms → AFD 580ms）。回退区（红）集中在 小 batch + 中高 RPS：bt8192 rps10 慢 4.0 倍（3083→12350ms）、bt16384 rps10 慢 3.7 倍。当 RPS=12 高压时两者拉平（0.49-1.16x），说明高负载下排队成为主导，AFD 的解耦收益被饱和抵消。结论：AFD 的甜区是大 batch（bt≥32768）× 低-中负载（rps≤8）的冷缓存prefill；小 batch 场景应保持全同步。


## 3. A3. Prefix 敏感性（cache 命中杠杆）

**这一组数据要说明什么：** 在同一个 anchor cell（bt=32768）上扫描 prefix 命中率 {0..99}，分别在 AFD 甜区负载（rps6）和拐点负载（rps10）各画一组。横轴是请求前缀可被 KV-cache 命中的比例，纵轴是 Mean/P99 TTFT，第三行是 AFD/baseline 加速比。这组曲线量化‘cache 命中’这一杠杆对两系统各自有多强，并回答：AFD 的优势（冷缓存低负载）是否随命中率上升而保持、翻转？

**数据支持的结论：** prefix 命中是两系统共同的单点杠杆：rps10 下 baseline 从 p0 的 2.58s 降到 p99 的 0.15s （-94%），AFD 从 2.97s 降到 0.21s（-93%）。但 AFD 相对 baseline 的优势并非随命中率单调变化，而是出现在一个‘中段命中带’：p25/p50 时 AFD 在 rps8 快 2.4-2.6 倍、rps10 快 1.8-1.9 倍；p90+ 高命中下 AFD 反而慢 1.4-1.7 倍（rps10 p90：398ms vs 239ms）。冷缓存 p0 端则依赖负载：rps6-8 AFD 快 1.8-2.7 倍，rps10（baseline 拐点附近）AFD 反慢 15%。机制：AFD 的收益来自 async 解耦把真实 compute 并行化——当命中率把待计算量压到缓存查找量级时收益归零，EP8 的转发/吞吐开销反而暴露；当负载逼近饱和时排队又吞掉并行收益。结论：AFD 的价值是‘中段命中 × 非饱和负载’的窄带优势，冷缓存低负载或中段命中时显著，高命中生产流量下应关闭或改用全同步。


## 4. A4. Batch 伸缩性（TTFT vs batch tokens）

**这一组数据要说明什么：** 固定 RPS、prefix=0，把 TTFT 画成 max batch tokens 的函数（A1 的转置视角）。它回答两个问题：(1) 扩大 batch 上限是否带来更长的排队/延迟？(2) AFD 和 baseline 对 batch 扩容的响应方向是否一致？

**数据支持的结论：** baseline 在低-中负载下 TTFT 随 batch 单调上涨（rps4: 从 bt8192 的 655ms 涨到 bt65536 的 1.60s），是典型的‘更大 batch = 更长的同步 prefill 链’。AFD 则相反：TTFT 从 bt8192 一路降到 bt32768，之后进入平台（rps4-6 下 bt32768→65536 稳定在 0.46-0.58s，rps6 三档 为 550/577/580ms，几乎水平）。机制：async 解耦后 batch 越大，attention 阶段越满、FFN 并行利用率越高，固定转发开销被摊薄——这解释了甜区为什么在大 batch。注意小 batch 端 （bt8192 rps6）AFD 仍略慢于 baseline（849 vs 810ms）。rps12 高压下两条曲线趋于收敛，AFD 的解耦收益被饱和排队吞掉。

