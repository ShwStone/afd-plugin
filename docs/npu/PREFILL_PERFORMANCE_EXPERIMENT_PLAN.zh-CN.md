# DeepSeek V3.2 减层模型 Prefill 性能实验完整计划

## 1. 实验目标

在固定 32 张 Ascend NPU 的条件下，对以下两种 DeepSeek V3.2 减层模型
prefill-only 服务方案进行公平、可复现的比较：

- 传统方案：`DP4 × TP8`，开启 SP；
- AFD 方案：Attention `DP3 × TP8`，开启 SP、关闭 EP，FFN 使用 `EP8`，
  通过 `CAMAsyncAFDConnector` 进行异步 Attention/FFN 解耦。

实验需要回答两个问题：

1. 在真实长短请求混合负载下，AFD 对 TTFT、吞吐拐点和 TTFT SLO 达成率的
   影响是什么；
2. 如果存在稳定收益，收益来自计算资源划分、DP/SP 负载均衡、CAM 通信、
   Attention/FFN 流水重叠、MoE ubatching，还是其他调度行为。

本实验只研究 prefill：每个请求固定生成 1 个 token。TPOT 不作为主要指标。

## 2. 实验原则

- 主对比只改变系统方案和 `max_num_batched_tokens`，其余条件保持一致；
- 主实验关闭 prefix cache，prefix cache 作为独立敏感性实验；
- 正式端到端数据不得开启 profiler；profile 必须使用独立重放；
- 失败请求按 SLO miss 计入全部已发请求的 SLO 达成率；
- 每个性能结论必须同时得到端到端重复实验和 profile/消融证据支持；
- 不使用 profile 后的单次结果、成功请求子集或不同硬件布局直接证明因果关系；
- 所有配置、数据集、模型配置、结果和 trace 必须记录哈希与环境信息。

## 3. 固定主实验配置

| 项目 | 配置 |
| --- | --- |
| 模型 | DeepSeek V3.2 减层模型 |
| 工作负载 | prefill only，output length = 1 |
| 硬件 | 32 张 NPU |
| Baseline | DP4 × TP8，SP on |
| AFD | Attention DP3 × TP8，SP on、EP off；FFN EP8 |
| Batch tokens | 8K、16K、32K、48K、64K |
| RPS | 4、6、8、10、12 |
| 请求到达 | `burstiness=1`，Poisson 到达 |
| 重复次数 | 每个 cell 3 次 |
| TTFT SLO | 10 秒 |
| Prefix cache | 主实验关闭 |
| 数据顺序 | 固定 CSV 非零行顺序，不 shuffle、不过采样 |
| Sampling | `temperature=0`、ignore EOS、输出 1 token |

主矩阵规模：

```text
2 systems × 5 batch-token limits × 5 RPS × 3 repeats = 150 runs
```

原始数据 `cp8sp50k.csv` 包含 1,380 行，其中 875 个非零输入长度，共
18,184,995 个输入 token；长度范围为 71～50,773 token。

## 4. 第一阶段：本地构建与迁移前验证

第一阶段不依赖 NPU 服务，目标是保证迁移到 NPU 服务器后，数据、客户端、
矩阵调度、结果校验和 profile 分析工具能够直接运行。

### 4.1 构建确定性 token-ID 数据集

使用 `tools.benchmarks.prefill_dataset`：

1. 按 CSV 原始顺序读取每个非零行；
2. 从目标模型 `config.json` 读取 `vocab_size` 和 special token ID；
3. 如果存在 `tokenizer_config.json`，排除其中标记为 special 的 added token；
4. 使用固定 seed 和 request-specific seed 生成确定性伪随机 token ID；
5. 每个请求保存 `request_id`、`source_row`、`prompt_len` 和
   `output_tokens=1`；
6. 同时输出 dataset manifest 和轻量 request index；
7. manifest 保存 CSV、模型配置、tokenizer 配置、dataset 和 index 哈希。

生成主数据集：

```bash
python -m tools.benchmarks.prefill_dataset generate \
  --csv cp8sp50k.csv \
  --model-config /models/DeepSeek-V3.2-reduced \
  --output tools/datasets/cp8sp50k_token_ids.jsonl
```

独立校验：

```bash
python -m tools.benchmarks.prefill_dataset validate \
  --dataset tools/datasets/cp8sp50k_token_ids.jsonl \
  --csv cp8sp50k.csv \
  --model-config /models/DeepSeek-V3.2-reduced
```

校验必须覆盖：请求数、顺序、输入长度、token 范围、special token 排除、输出
长度、dataset/index/manifest 哈希。Git LFS pointer 必须直接拒绝。

### 4.2 构建 prefix-cache 敏感性数据集

额外生成 25%、50%、75% 三档共享前缀数据：

```bash
for specification in 0.25:25 0.5:50 0.75:75; do
  ratio=${specification%%:*}
  suffix=${specification##*:}
  python -m tools.benchmarks.prefill_dataset generate \
    --csv cp8sp50k.csv \
    --model-config /models/DeepSeek-V3.2-reduced \
    --output "tools/datasets/cp8sp50k_token_ids_prefix${suffix}.jsonl" \
    --prefix-ratio "${ratio}" \
    --prefix-block-size 128 \
    --prefix-group-size 12
done
```

每 12 个请求构成一组，使分组对 DP3 和 DP4 都中性。组内共享 block-aligned
prefix，suffix 保持请求唯一。manifest 同时记录配置共享比例和按请求顺序估算的
可复用比例。

实际 cache hit 仍可能受到并发到达、DP-local cache、路由、驱逐和 chunked
prefill 影响，因此必须从服务端指标获得实际命中率。

非零 prefix ratio 不能使用 vLLM 内置 warmup，因为它会复用同一批请求，导致
完整 prompt 被提前缓存。应使用独立 warmup 数据或对两个系统执行完全相同的
预热和重启流程。

### 4.3 `vllm bench serve` 能力核对与补丁

基于仓库固定的 vLLM 0.19.1，原生能力和缺口如下：

| 能力 | 原生支持 | 本实验处理 |
| --- | --- | --- |
| 固定 RPS、Poisson/gamma 到达 | 支持 | 直接使用 |
| custom JSONL 文本 prompt | 支持 | 保持原行为 |
| token-ID prompt | API 支持，但 custom dataset 采样不支持精确长度 | 显式补丁 |
| 大 JSONL | pandas DataFrame 会产生额外复制 | 流式 JSONL loader |
| skip tokenizer 后的 prompt length | 原生记录为 1 | 使用 token list 的真实长度 |
| special token 插入 | completions 默认允许 | 强制 `add_special_tokens=false` |
| UTF-8 SSE 跨 chunk | 单 chunk 解码可能失败 | incremental UTF-8 decoder |
| stable request ID | 原生生成 index ID | 保留 dataset request ID |
| detailed result | 支持 | 强制开启 |
| TTFT goodput | 只对成功请求计算 | 另算 all-issued SLO rate |

补丁位于：

```text
afd_plugin/compat/patches/benchmark_serving.py
```

补丁只由 `tools.benchmarks.prefill_bench` 显式启用，不影响正常 AFD server
启动。补丁签名必须与 vLLM 0.19.1 上游完全一致；上游支持相同能力后应删除。

### 4.4 本地 mock smoke test

启动 mock OpenAI completions 服务：

```bash
python -m tools.benchmarks.prefill_mock_server --port 18000
```

mock 服务必须验证：

- prompt 是整数 token ID 列表；
- `max_tokens=1`；
- SSE 返回可被客户端解析；
- 中文字符故意跨 UTF-8 chunk 后仍能成功解码。

运行测试：

```bash
uv run pytest -q \
  tests/unit/tools \
  tests/unit/compat/patches/test_benchmark_serving.py
```

### 4.5 准备矩阵配置并 dry-run

复制并编辑：

```text
tools/benchmarks/prefill_experiment.example.json
```

必须替换模型路径、served model name、dataset 路径、结果目录、baseline 和 AFD
server launch template。

打印完整计划，不执行服务或请求：

```bash
python -m tools.benchmarks.prefill_experiment \
  --config /path/to/prefill_experiment.json \
  plan
```

第一阶段验收条件：

- 数据生成完全确定，重复生成 hash 相同；
- dataset、index、manifest 三者校验通过；
- token-ID、UTF-8 SSE 和 mock smoke test 通过；
- 完整主矩阵展开为 150 个 run；
- prefix sensitivity 不使用重叠 warmup；
- lint、format 和所有 unit tests 通过；
- NPU preflight 能明确报告缺失的硬件、软件或数据条件。

## 5. 第二阶段：NPU 服务器端到端实验

### 5.1 迁移内容

迁移以下内容到 NPU 环境：

- 当前 Git commit；
- `cp8sp50k.csv`；
- 主数据集及三个 prefix 数据集；
- 每个数据集的 manifest 和 index；
- 实际实验 JSON 配置；
- reduced-model `config.json` 和 `tokenizer_config.json`；
- baseline/AFD server launch scripts；
- 本地测试结果和依赖版本记录。

### 5.2 执行 preflight

```bash
python -m tools.benchmarks.prefill_preflight \
  --require-npu \
  --model-config /models/DeepSeek-V3.2-reduced \
  --experiment-config /path/to/prefill_experiment.json \
  --dataset tools/datasets/cp8sp50k_token_ids.jsonl \
  --output bench_results/prefill/preflight.json
```

preflight 应记录并检查：

- Git commit 和 dirty status；
- Python、vLLM、vLLM Ascend、torch、torch-npu、AFD plugin 版本；
- dataset/index/model 配置哈希；
- `npu-smi info` 和 `msprof --version`；
- 目标系统、batch tokens、RPS、repeat、SLO；
- 是否存在 Git LFS pointer、错误的 vLLM 版本或缺失文件。

另行归档：物理 NPU ID、节点 IP、NUMA 绑定、rank table、HCCL/CAM 版本、
server 完整命令和环境变量。

### 5.3 Server 配置约束

- baseline 和 AFD 使用相同模型、sampling、block size、chunked-prefill 和日志
  设置；
- AFD Attention 和 FFN 的 `max_num_batched_tokens` 必须一致；
- AFD Attention 不得传 `--enable-expert-parallel`；FlashComm1/SP 由
  `VLLM_ASCEND_ENABLE_FLASHCOMM1=1` 独立开启，只有 FFN 保留 EP8；
- AFD rank mapping、Attention rank 数、FFN rank 数和 `attn_ranks_per_dp`
  必须匹配 24 Attention + 8 FFN 的 32 卡布局；
- 主实验关闭 prefix cache；
- 输入最长 50,773 token，因此 `max_model_len` 至少为 50,774；
- 8K/16K batch-token cell 必须正确开启并验证 chunked prefill；
- 服务日志必须证明长请求被 admission/chunk，而不是 truncate 或 reject；
- 正式 E2E run 不得开启 profiler。

### 5.4 执行主矩阵

每次只启动一个 system 和一个 batch-token 配置，然后执行该组全部 RPS 与
repeat。例如：

```bash
python -m tools.benchmarks.prefill_experiment \
  --config /path/to/prefill_experiment.json \
  run \
  --system dp4_tp8_sp \
  --batch-tokens 32768 \
  --prefix-ratio 0 \
  --resume
```

执行器会：

1. 校验 dataset/index/manifest hash 和 request count；
2. 轮询 `/v1/models` 并核对 served model；
3. 执行固定的 patched `vllm bench serve` 命令；
4. 保存 raw detailed JSON；
5. 生成 `.verified.json`；
6. 记录每个请求的 ID、source row、成功状态、TTFT、E2EL 和 SLO 状态；
7. 将失败请求计入 SLO miss；
8. 按 `<=8K`、`8～16K`、`16～32K`、`32～48K`、`>48K` 分桶。

`--resume` 只能在确认当前 server 仍属于相同 system/batch-token group 后使用。

### 5.5 汇总端到端结果

```bash
python -m tools.benchmarks.prefill_report \
  --result-dir bench_results/prefill \
  --baseline dp4_tp8_sp \
  --candidate afd_dp3_tp8_ep8 \
  --expected-repeats 3 \
  --output bench_results/prefill/report.json \
  --csv-output bench_results/prefill/report.csv
```

每个 cell 至少输出：

- issued/success/failed request 数；
- all-issued SLO attainment；
- successful-only mean/P50/P90/P95/P99 TTFT；
- repeat-level mean TTFT 和 SLO；
- repeat 标准差和完整性；
- AFD 相对 baseline 的 mean/P99 TTFT 降幅；
- SLO attainment 的百分点变化。

若收益小于 repeat 波动，不应表述为稳定收益。

## 6. 敏感性实验

单次全量 cell sweep 完成后的正式重复、固定 workload 消融、prefix cold/steady、
运行时事件和流水 trace 采集，按
[`PREFILL_PERFORMANCE_STAGE2_DATA_COLLECTION.zh-CN.md`](PREFILL_PERFORMANCE_STAGE2_DATA_COLLECTION.zh-CN.md)
执行。该方案将单次 sweep 定义为 exploratory screening，并只对进入正式结论的
选中 cell 补齐至少 3 次无 profiler 重复。

主矩阵完成后，逐个改变以下因素，禁止同时改变多个因素：

1. prefix cache：off、25%、50%、75%；
2. `max_num_seqs`；
3. chunked-prefill 策略；
4. arrival burstiness，在固定平均 RPS 下改变突发程度；
5. 输入顺序或有记录的 alternate permutation；
6. 只包含短请求、长请求或长度分层的子数据集；
7. CAM `dynamicQuant`；
8. async MoE ubatching on/off；
9. request split 与 token split；
10. graph/eager，仅在两侧均为合法配置时比较；
11. rank placement、NUMA 和网络拓扑。

敏感性实验优先选择主矩阵中的低负载点、吞吐拐点和高负载点，不默认复制全部
150-cell 矩阵。

## 7. Profile 与收益归因计划

### 7.1 选择 profile 重放点

从无 profiler 的主实验中选择：

- 一个两个系统均满足 SLO 的低负载点；
- 一个开始出现排队的 knee point；
- 一个收益稳定的高负载点；
- 至少一个 32K 或 64K batch-token 点。

profile baseline 和 AFD 时必须使用相同请求 ID、请求顺序、到达计划、模型、
batch-token limit 和物理 rank placement。每个 replay 只运行一次固定 workload，
不纳入正式 E2E 统计。

### 7.2 采集内容

每个 replay 归档：

- bench detailed result 和 client timeline HTML；
- server scheduler 日志和 request/chunk 信息；
- Attention 与 FFN 各 rank 的 torch_npu/TensorBoard trace；
- msprof 时间线、算子统计、通信统计和内存数据；
- rank mapping、物理设备、节点时钟和 trace 文件 hash；
- profile schedule、WAIT/WARMUP/ACTIVE/SKIP_FIRST 参数。

AFD Attention 和 FFN profiler 独立配置。不要只采 rank 0；DP/SP 不均衡或
straggling expert rank 可能只出现在其他 rank。

复杂/分块 prefill 中，一个请求可能跨多个 model step。应先做短 calibration，
再设置足够宽的 active window，并根据 request/server 事件裁剪 matched window。

### 7.3 Trace 自动分析

单 trace 汇总：

```bash
python -m tools.benchmarks.profile_trace summarize \
  --trace /profiles/afd/attention/rank0.pt.trace.json \
  --output /profiles/afd/attention/rank0.summary.json
```

matched trace 对比：

```bash
python -m tools.benchmarks.profile_trace compare \
  --baseline /profiles/baseline/rank0.pt.trace.json \
  --candidate /profiles/afd/attention/rank0.pt.trace.json \
  --output /profiles/comparison/rank0.json
```

事件分类包括：

- communication：HCCL、all-reduce/all-to-all、dispatch/combine、send/recv；
- attention：Attention、MLA、FlashAttention、RoPE、KV cache；
- FFN/MoE：routing、expert、grouped matmul、SwiGLU；
- memory：memcpy、DMA、transpose/permute；
- host/scheduler；
- other compute 和 unclassified。

输出 event count、event time、global interval union、per-lane busy union、top
events 和 communication/compute overlap。event time 会重复计算嵌套或并行事件，
因此不能单独用作关键路径耗时。

### 7.4 归因假设

重点验证以下潜在收益来源：

1. DP 请求长度不均衡造成的同步等待是否减少；
2. SP/TP collective 是否缩短或与 FFN 计算重叠；
3. CAM dispatch/combine 是否替代了高成本的同步 all-to-all；
4. Attention 和 FFN 是否形成稳定流水，而非只是转移等待位置；
5. MoE request/token split 是否改善两个 ubatch 的 token 平衡；
6. AFD DP3 + EP8 的资源划分是否改变 Attention/FFN bottleneck；
7. host scheduler、Python、HTTP 或队列等待是否成为新瓶颈；
8. 长请求 chunking、padding、cache block 和内存搬运是否解释收益或回退；
9. expert/rank straggler 是否决定 P99；
10. prefix cache 是否改变 DP 路由和实际计算 token 数。

### 7.5 消融实验

在拓扑和运行时允许的条件下，执行最小消融：

- AFD async on/off 或同步 connector 对照；
- async MoE ubatching on/off；
- request split 与 token split；
- overlap 路径 rollback/barrier 对照；
- `dynamicQuant` on/off；
- 相同 AFD 拓扑下固定其他因素，只回退被怀疑的优化；
- 必要时按长度桶或单请求重放，分离 scheduler 与 kernel 收益。

DP4 baseline 与 DP3+EP8 AFD 同时改变执行模式和资源划分，因此主 A/B 不能单独
证明 async overlap 是收益来源。无法构造合法中间态时，必须明确把剩余部分报告为
“拓扑/资源划分综合收益”，而不是 async 因果收益。

### 7.6 性能结论四项证据门槛

每个“收益来源”结论必须同时满足：

1. **Timeline**：预测的等待、同步、通信或 kernel 序列发生可见变化；
2. **Quantification**：matched window 中能量化减少或重叠的毫秒数和比例；
3. **Ablation**：回退该行为后，相应成本和端到端方向恢复；
4. **End-to-end**：无 profiler 的多次实验中 TTFT/SLO 同方向变化且超过噪声。

不满足四项门槛时，只能写成观察或假设。

## 8. 结果报告结构

最终报告至少包含：

1. 环境、模型、拓扑、版本、rank mapping 和数据哈希；
2. 主矩阵 mean/P99 TTFT heatmap；
3. all-issued TTFT SLO attainment heatmap；
4. 每个 batch-token limit 的 RPS 曲线和饱和拐点；
5. 长度分桶结果；
6. 失败数量、错误类型和缺失 cell；
7. prefix-cache 敏感性结果及实际 cache-hit 指标；
8. repeat 波动和置信边界；
9. matched baseline/AFD pipeline timeline；
10. communication/compute overlap 和各阶段关键耗时；
11. 消融结果；
12. 已证实的收益来源、未证实假设、限制和后续工作。

推荐结论格式：

```text
在 [batch tokens / RPS / length bucket] 条件下，AFD 的 mean/P99 TTFT 从
[baseline] 降至 [candidate]，all-issued SLO 从 [x%] 变为 [y%]。
Trace 显示 [阶段] 的 [等待/通信] 减少 [ms/%]，与 [计算] 的重叠增加 [ms/%]；
[消融] 后该变化和端到端收益同时回退，因此将该部分归因于 [具体机制]。
```

## 9. 交付物

- 主/prefix JSONL 数据集；
- 每个数据集的 manifest 和 index；
- 实际实验 JSON 配置；
- preflight JSON；
- 150 个主实验 raw/verified result；
- 汇总 JSON 和 CSV；
- client timeline HTML；
- baseline/AFD 各 rank trace 与 trace summary；
- 消融实验结果；
- 最终实验报告和可复现命令清单。

## 10. 最终验收条件

实验完成需要同时满足：

- 150 个主实验 cell 均有 3 次有效、无 profiler 的重复结果；
- 所有失败请求计入 all-issued SLO；
- 主实验 prefix cache 确认关闭；
- 数据、模型、环境、配置和结果均可通过 hash 追溯；
- profile replay 与主结果隔离；
- 所有 paired trace 使用匹配请求和配置；
- 任何因果收益结论满足四项证据门槛；
- 无法分离的收益明确标记为拓扑/资源划分综合效应；
- 最终报告能从归档的命令、配置和产物重新生成。
