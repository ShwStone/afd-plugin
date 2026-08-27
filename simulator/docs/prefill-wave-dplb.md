# Prefill-only 下的 Wave Sum / Square-Sum DP 调度

本文介绍 simulator 中实验性的 Wave DPLB（Data Parallel Load
Balancing，数据并行负载均衡）策略。它只面向 **prefill-only** 服务：请求输入
一段 Prompt，系统完成 Prefill 后即视为请求完成。本文讨论的目标是降低 TTFT、
改善 P99，并提高 Prefill 请求吞吐和输入 token 吞吐；不讨论 Decode、逐 token
生成、KV Cache 容量压力或 Prefill/Decode 混部公平性。

> **状态说明：** 本文描述的是 `ShwStone/afd-plugin` 的 simulator 实验分支中
> 使用的离散事件模拟策略，不代表 `vllm-project/afd-plugin` 已经实现或启用了
> DP-wave，也不代表硬件实测结论。模拟收益必须经过真实服务和硬件验证。

对应的四个配置名是：

- AFD：`afd_wave_token_sum`、`afd_wave_token_square_sum`；
- merged：`merged_wave_token_sum`、`merged_wave_token_square_sum`。

## 一句话理解

普通 DP 调度经常只问“哪个 DP 的请求数最少”或“哪个 DP 剩余 token 最少”。
Wave 调度问得更具体：

> 如果把新请求追加到这个 DP 的 FIFO 队尾，按照真实的 batch/chunk 规则，
> 它会落入第几个 wave，又会在什么时候完成？

调度器把新请求分别试放到每个 DP，模拟尚未运行的 FIFO waves，计算每种放置的
分数，最后选择分数最小的 DP。这里的“试放”只用于估算，不会执行请求，也不会
改变已经排队请求的顺序。

```text
                       ┌─ 追加到 DP0 → 重建未来 waves → score0
新到请求 R ──逐个试放─┼─ 追加到 DP1 → 重建未来 waves → score1
                       └─ 追加到 DP2 → 重建未来 waves → score2

                         选择最小 score；同分时轮转
```

## 什么是 wave

这里的 wave 可以理解为一个 DP 下一次实际会发出的 Prefill scheduler batch。
它不是简单地按请求数切分，而是复用运行时相同的 FIFO 装箱约束：

- `max_num_batched_tokens`：一个 wave 最多容纳多少未缓存 query tokens；
- `chunk_size`：单个请求在一个 wave 中最多贡献多少 query tokens；
- `max_num_seqs`：一个 wave 最多容纳多少请求或请求 chunk；
- FIFO：已有请求不能被后来的请求越过或重新排序。

开启 chunked prefill 后，一个长请求可以跨多个 wave，但在单个 wave 中最多贡献
一个 chunk。如果这个 chunk 没填满 wave，调度器继续按 FIFO 放入后续请求。
未完成请求会保持原来的相对顺序，进入下一 wave。

例如 token 预算为 16、chunk 上限为 8，FIFO 请求长度为 `A=12, B=6,
C=5`：

```text
wave 0: A(8) + B(6) + C(2) = 16
wave 1: A(4) + C(3)        = 7
```

所以“队列里有三个请求”并不能说明实际还有几个 batch，也不能说明新请求会在哪个
batch 完成。Wave 调度的第一步，就是把这个信息恢复出来。

## Sum 和 Square-Sum 在估算什么

设一个 uBatch 或 DP wave 内的 query-token 片段长度为
`t1, t2, ..., tn`：

```text
token sum 工作量        = Σ ti
token square-sum 工作量 = Σ ti²
```

请注意，square-sum 是 **各请求/chunk 片段的平方和**，不是
`(Σ ti)²`，也不是把整个原始 Prompt 长度直接平方。

两组 batch 都有 16 个 token：

| 组成 | Sum | Square-Sum |
| --- | ---: | ---: |
| `8 + 8` | 16 | 128 |
| `15 + 1` | 16 | 226 |

Sum 认为它们一样重；Square-Sum 认为包含 15-token 长片段的 batch 更重。因此：

- Sum 更接近线性 token 成本，通常更平滑，也更偏向总工作量；
- Square-Sum 会更强烈地分散长请求，适合 Prompt 长度长尾明显、长片段对 Attention
  延迟影响更大的场景；
- Square-Sum 只是启发式成本模型，不代表算子耗时严格等于 token 长度平方。

AFD 会先按照 `ubatch_split` 把一个 wave 切成 token uBatches，再对每个 uBatch
中的片段应用上述公式。若 token-split 把一个请求 chunk 切成两段，Square-Sum
也是对切分后的两段分别平方。merged 不做这一步，而是直接估计每个 DP wave。

## AFD：选择预计最先完成请求的异步 DP

AFD 的 Attention DPs 可以异步推进，因此每个候选 DP 只需要回答“新请求在我这里
何时完成”。对候选 DP `d`，实现大致计算：

```text
AFD_score(d)
  = 当前运行 batch 尚未完成的 Attention 工作
  + 从本地 FIFO 队首到候选请求完成 wave 为止的未来 Attention 工作
```

具体步骤如下：

1. 把候选请求追加到 DP `d` 的本地 FIFO 队尾；
2. 用 AFD 自己的 token 预算、chunk 上限和 `max_num_seqs` 重建未来 waves；
3. 找到候选请求最后一个 chunk 所在的 wave；
4. 只累加到这个完成 wave，不计候选后面的排空时间；
5. 加上 DP 当前运行状态中尚未执行的 Attention ops；
6. 对正在执行的 Attention op，按到达时刻已经过去的时间扣除进度；
7. 选择标量分数最小的 DP。

每个 wave 的工作量还会乘以模型层数。这样得到的不是绝对毫秒数，而是用于比较
DP 的 Attention 工作单位；所有 DP 使用同一种单位，因此可以排序。

当前 AFD 评分刻意不把 FFN、CAM 和流水 bubble 作为加法项。它适用于 Attention
主导，或者共享 FFN 仍有余量、能被 token uBatch 流水隐藏的 Prefill 区域。如果
FFN 已成为瓶颈或处在 Attention/FFN 临界点，仅看 Attention 可能选错 DP；这时
需要把共享 FFN 队列和完整流水 makespan 纳入后续模型。

## merged：先避免新增全局 wave

merged 的各 DP 在每层通过同步点共同推进。某个 DP 的本地 batch 很轻，也可能要
等待同一全局 wave 中最重的 DP。因此它不能照搬 AFD 的“本地最先完成”分数。

对每一种候选放置，实现会重建所有 DP 的未来本地 waves。全局第 `w` 个 wave 的
工作量取该序号中最重的 DP：

```text
global_work(w) = max(local_work(dp, w))
```

每个候选 DP 得到一个三级字典序分数：

```text
(
  加入后的全局 wave 数,
  所有未来全局 waves 的总 drain 工作量,
  从队首到候选请求完成 wave 的工作量
)
```

“字典序”表示第一项优先级最高；只有第一项相等才比较第二项，前两项都相等才比较
第三项。这对应以下原则：

1. **尽量不开新全局 wave**；
2. wave 数相同时，尽量缩短整个同步队列的排空时间；
3. 前两者相同时，让当前候选请求更早完成。

假设 DP0 已有两个满 wave，DP1 只有一个。把新请求放入 DP0 会产生第三个全局
wave；放入 DP1 的第二个本地 wave 则可与 DP0 已存在的第二个 wave 对齐。即使
“新建本地 batch”发生在 DP1，第二种放置也没有新建 **全局** wave，所以应该选
DP1。这正是简单队列长度或本地 batch 数评分容易遗漏的同步语义。

已经运行或完成的 merged wave 不会重排。由于 merged 只在全局 wave 边界开始
新的调度，评分针对的是尚未运行的 FIFO waves。

## 核心伪代码

```text
on_request_arrival(request):
    scores = []
    for candidate_dp in all_dps:
        trial_queues = queues with request appended to candidate_dp
        waves = rebuild_fifo_waves(trial_queues)

        if architecture == AFD:
            score = remaining_running_attention(candidate_dp)
                  + work_until_request_completion(waves[candidate_dp])
        else:  # merged
            score = (
                global_wave_count(waves),
                total_global_drain_work(waves),
                global_work_until_request_completion(waves),
            )
        scores.append(score)

    chosen_dp = argmin(scores)
    append request to queues[chosen_dp]
```

多个 DP 分数完全相同时，调度器会轮转同分扫描起点，避免长期偏向 DP0。

## 它与简单 Token Greedy 有什么区别

| 策略 | 观察的状态 | 是否理解 batch/chunk | 是否看候选完成 wave | 是否理解 merged 同步 |
| --- | --- | --- | --- | --- |
| 请求数 DPLB | waiting/running 数量 | 否 | 否 | 否 |
| `prefill_token_greedy` | 每个 DP 剩余 token 总数 | 否 | 否 | 否 |
| `prefill_token_square_greedy` | 每个 DP 各请求剩余 token 的平方和 | 否 | 否 | 否 |
| Wave Sum/Square-Sum | 按真实约束重建的未来 FIFO waves | 是 | 是 | 是，使用专用分数 |

Token Greedy 是便宜的即时快照。Wave 调度则会回答：现有 token 究竟会装成几个
batch、候选请求在哪个 batch 完成，以及 merged 的本地 batch 是否真的增加全局
同步 wave。

另一个容易混淆的区别是：`prefill_token_square_greedy` 平方的是每个未完成请求的
全部 remaining tokens；Wave Square-Sum 平方的是规划后某个 wave/uBatch 内的
请求或 chunk 片段。

## 可能带来的好处

在 Prefill-only、长度分布不均匀的持续负载中，Wave 调度通常有以下潜在收益：

- **更准确地均衡真实工作**：一个 32K Prompt 不再和一个 512-token Prompt 都只
  算作“一个请求”；
- **理解 maxbatch 和 chunk**：不会把 token 总数相似、实际 wave 数不同的队列误判
  为等价；
- **改善 TTFT 尾延迟**：直接优化候选请求的完成 wave，Square-Sum 还会更主动地
  分散长片段；
- **减少 merged 同步浪费**：优先复用已经存在的全局 wave，降低 barrier drain 和
  空等；
- **适应 AFD 异步进度**：会扣除正在执行的 Attention 工作进度，而不是让运行中
  batch 在完成前始终保持满负载；
- **保持 FIFO 可解释性**：不需要任意重排未运行请求，行为容易复现和对照实现；
- **Sum/Square-Sum 可切换**：可以根据 profile 曲率和长度长尾程度选择较温和或较
  激进的成本估计。

这些是机制上的潜在收益，不是对所有负载的性能保证。请求很短且同质、QPS 很低
时，各策略通常接近；Square-Sum 也可能过度惩罚长片段，牺牲某些吞吐或公平性，
必须用目标数据集和 SLO 验证。

## 实现位置

主要代码位于 `simulator/engine.py`：

| 函数 | 作用 |
| --- | --- |
| `RequestDispatcher._select_dp` | 比较各 DP 分数并处理同分轮转 |
| `_plan_fifo_wave` | 按运行时约束规划一个 FIFO wave |
| `_fifo_waves` | 重建一个 DP 的全部未来 waves |
| `_fifo_completion_wave_index` | 找到候选请求最终完成的 wave |
| `_query_token_work` | 选择 `Σt` 或 `Σt²` 工作量 |
| `_afd_wave_completion_scores` | 计算 AFD 异步候选完成分数 |
| `_afd_remaining_attention_work` | 估计 AFD 运行中 Attention 剩余工作 |
| `_afd_wave_attention_work` | 应用 uBatch 切分和层数 |
| `_merged_wave_completion_scores` | 计算 merged 同步三级分数 |

配置示例：

```json
{
  "mode": "continuous",
  "arrival": {
    "kind": "poisson",
    "qps": 4,
    "duration_s": 128
  },
  "scheduler": {
    "chunked_prefill": true,
    "max_num_seqs": 64,
    "afd_policy": "afd_wave_token_sum",
    "merged_policy": "merged_wave_token_square_sum",
    "afd": {
      "max_num_batched_tokens": 32768,
      "chunk_size": 32768
    },
    "merged": {
      "max_num_batched_tokens": 32768,
      "chunk_size": 8192
    }
  }
}
```

AFD 和 merged 可以独立选择 Sum 或 Square-Sum，因为两侧的执行和同步语义不同，
不应通过一个共享策略字段隐式绑定。

## 当前边界与后续方向

理解实验结果时应牢记：

- 这是在线贪心策略，只知道当前状态，不知道未来请求，因此不是离线全局最优；
- 工作量分数只使用未缓存 query tokens；实际仿真执行仍使用包含 prefix/query
  形状的 profile，所以“路由估计”与“最终耗时模型”并不完全相同；
- AFD 当前只以 Attention 为路由瓶颈，不建模共享 FFN 饱和与瓶颈翻转；
- merged 的全局 wave 工作量以最重 DP 的 token 工作估算，不等于直接调用完整
  profile 预测每一种候选放置；
- 逐个候选重建 waves 有额外计算开销。AFD 每次到达约需规划所有 DP；merged 还要
  对每个候选重新检查所有 DP，生产实现需要增量缓存或更便宜的近似；
- 本策略只针对 Prefill-only。加入 Decode 后，还必须考虑 KV Cache、decode batch、
  token-by-token 服务时间以及 Prefill/Decode 之间的资源竞争。

因此，Wave Sum/Square-Sum 最适合作为一个“更懂 Prefill batch 形状的 DPLB
基线”：先用它定位 AFD 或 merged 的潜在收益场景，再决定哪些评分项值得进入真实
运行时。
