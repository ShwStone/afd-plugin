# DeepSeek V4 Flash W8A8 端到端预填充基线性能计划

> 状态：草案。目标是先取得低成本、可复现的单窗口端到端基线；不做同配置
> 重复运行，也不把单次观测表述为精确容量上界。

## 1. 目标

本实验使用 Mooncake-derived WildChat 预填充工作负载，测量 DeepSeek V4
Flash W8A8 在 Ascend NPU 上的端到端性能，回答：

1. `max_num_batched_tokens` 为 8K、16K、32K、64K 时，TTFT、SLO 和
   goodput 如何变化；
2. 相同 8 张物理卡下，`DP4TP4EP16` 与 `DP2TP8EP16` 的差异；
3. 从 8 张扩展到 16 张物理卡后，`DP4TP8EP32` 的性能和单卡效率变化；
4. 如何把 Mooncake-derived WildChat 固化为后续模型和同事都能复用的
   端到端预填充测试协议。

本轮不测长输出解码、prefix cache、EPLB、MTP、speculative decoding、DBO、
profiler 或算子级归因，也不比较 AFD 解耦部署。输出固定为 1 token，测得的
TTFT 包含客户端发送、服务端排队、完整预填充和首 token 返回。

## 2. 冻结的验证单元

### 2.1 模型和运行时

- 模型：DeepSeek V4 Flash W8A8；
- backend：Ascend NPU；
- checkpoint、vLLM、vLLM-Ascend、CANN、容器和插件版本使用 NPU 环境中由
  当前仓库要求固定的版本；
- 文档不硬编码内网 checkpoint 路径，但每次运行清单必须记录路径或稳定模型
  ID、配置文件 SHA-256、权重索引 SHA-256 和实际运行时版本；
- 执行模式：eager；
- 自然专家路由，EPLB 关闭；
- prefix cache 关闭；
- MTP、speculative decoding 和 DBO 关闭；
- 输出长度固定为 1，temperature 为 0，直接发送 `prompt_token_ids`。

任何一项改变都形成新的验证单元，不能和本轮结果合并为同一组重试。

### 2.2 拓扑

本文的“卡”指双 die 物理卡；一个物理卡对应两个可见 NPU rank。

| 拓扑 | NPU world size | 物理卡 | 默认放置 |
|---|---:|---:|---|
| `DP4TP4EP16` | 16 | 8 | 单节点 |
| `DP2TP8EP16` | 16 | 8 | 单节点 |
| `DP4TP8EP32` | 32 | 16 | 双节点、同一高速互联域 |

DP、TP 和 EP 的实际 group、rank 到节点/物理卡的映射必须写入运行清单，并从
服务日志核验，不能只记录启动命令中的期望值。

### 2.3 调度配置

正式矩阵使用以下精确值：

```text
max_num_batched_tokens = 8192, 16384, 32768, 65536
```

`max_num_seqs` 不是实验变量，必须足够大到不会先于 token 上限约束调度。V4
派生 bundle 将保持现有工作负载的输入长度序列，其中最短输入为 891 token。
如果一个调度步骤只包含完整 prompt，64K token 上限最多容纳：

```text
floor(65536 / 891) = 73 requests
```

本轮统一使用 `max_num_seqs=128`，等于整个 `screening` 窗口的请求数，因此
不会把任何已到达请求挡在一个更小的 sequence 上限之外。运行日志仍须确认
没有请求因 `max_num_seqs` 等待；若出现该现象，则该运行无效，而不是把结果
解释为 batch-token 上限的影响。

`max_model_len` 必须覆盖数据集中最长输入和 1 个输出 token，并在全部 12 个
配置中保持相同。内存利用率、chunked prefill 和其他内存相关配置也在第一次
有效运行前冻结，不作为调优变量。

## 3. 标准化数据集

### 3.1 Tokenizer 调研结论

现有 bundle 使用 DeepSeek-V3.2 tokenizer revision
`a7e62ac04ecb2c0a54d736dc46601c5606cf10a6`。截至 2026-08-25，对 DeepSeek
官方公开仓库的检查结果为：

| 项目 | DeepSeek V4 Flash | DeepSeek V3.2 |
|---|---|---|
| `tokenizer.json` 字节数 | 6,367,146 | 7,847,502 |
| HTTP ETag | `628e3364caad11bdf9e67cea06eae7878122811d` | `9b4d31974a7e15e519ca5425ff2245a889779cf8` |
| tokenizer class | `PreTrainedTokenizerFast` | `LlamaTokenizerFast` |
| model max length | 1,048,576 | 131,072 |
| vocab size | 129,280 | 129,280 |

两边复用了部分 BOS/EOS 和消息 special token，但 tokenizer 文件、配置和官方
消息 encoding 实现并不相同。因此本计划把它们视为不兼容 tokenizer；不能把
V3.2 bundle 的 token ID 直接作为 V4 正式输入。

官方参考：

- [DeepSeek V4 Flash tokenizer/config](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/tree/main)
- [DeepSeek V4 encoding](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/main/encoding/README.md)
- [DeepSeek V3.2 tokenizer](https://huggingface.co/deepseek-ai/DeepSeek-V3.2/blob/main/tokenizer.json)
- [DeepSeek V3.2 encoding](https://huggingface.co/deepseek-ai/DeepSeek-V3.2/blob/main/encoding/encoding_dsv32.py)

实际 NPU checkpoint 仍须在开跑前对本地 tokenizer 文件重新计算 SHA-256；
公开仓库的 ETag 不能代替内网 checkpoint 的文件校验。

### 3.2 V4 派生 bundle 契约

按
[Mooncake–WildChat 工作负载生成方案](MOONCAKE_WILDCHAT_WORKLOAD_GENERATION_PLAN.zh-CN.md)
生成新的 V4 专用 bundle。新 bundle 必须：

- 使用 V4 checkpoint 自带 tokenizer 和官方 V4 encoding；
- 生成新的 bundle ID，不覆盖 V3.2 bundle；
- 保持相同 Mooncake 窗口、trace index、request ID、请求顺序和基础到达偏移；
- 保持逐请求 `input_length` 序列完全相同，输出长度仍为 1；
- 重新选择或物化符合 V4 tokenizer 的 WildChat 内容，不复用 V3.2 token ID；
- 保存 tokenizer、encoding 实现、request 文件和 arrival 文件的 SHA-256；
- 直接发送物化后的 `prompt_token_ids`，运行时不 decode、重新套模板或
  tokenize。

新旧 bundle 的以下派生校验值必须一致：

```text
request_id_sequence_sha256
input_length_sequence_sha256
base_arrival_sequence_sha256
```

内容 token 的校验值应不同，并分别归属各自 bundle。若逐请求长度无法完全
一致，停止并更新本计划，不能用 padding、截断或重复内容强行对齐。

## 4. 低成本运行设计

### 4.1 数据窗口和证据级别

本轮只使用 128 请求的 `screening` 窗口，证据级别记为：

```text
baseline_single_window
```

每个配置只运行一次，不运行 `formal_0/1/2` 三窗口重复，不计算置信区间，也
不声称得到精确最大容量。后续可以在不改变数据和结果 schema 的前提下升级为
`formal_three_window`。

### 4.2 共同负载校准

参考配置为：

```text
DP2TP8EP16 + max_num_batched_tokens=32768
```

在 `screening` 窗口上最多运行两个不同的目标输入 token/s，目标是让参考配置
的 TTFT p99 落在 20～40 秒区间。若两个点无法完成夹逼，选择 p99 最接近该
区间且运行有效的点。

选定后冻结一份 scaled arrival plan。全部 12 个配置读取同一份 request 文件和
同一份 scaled arrival 文件，以同一绝对 offered token/s 运行。参考配置在最终
负载上的校准结果直接计入矩阵，不重复运行。

本轮得到的是共同负载下的延迟和 goodput，不是每个配置各自调优后的最大容量。
这是有限资源下保证横向可比性的取舍。

### 4.3 正式单次矩阵

| 拓扑 | 8192 | 16384 | 32768 | 65536 |
|---|---:|---:|---:|---:|
| `DP4TP4EP16` | 1 次 | 1 次 | 1 次 | 1 次 |
| `DP2TP8EP16` | 1 次 | 1 次 | 1 次 | 1 次 |
| `DP4TP8EP32` | 1 次 | 1 次 | 1 次 | 1 次 |

共 12 个计量运行，加至多 1 个不进入矩阵的校准点。每个配置需要独立启动，
先完成不计时 warmup，再运行一次 `screening`。不从多个结果中挑选最好值。

## 5. 指标

保存每条请求的原始发送时刻、首 token 时刻、完成时刻、输入 token 数、成功
状态和错误。SLO 不写死成单一布尔值，当前报告同时离线计算 20 秒和 40 秒两套
口径，以后可以从原始请求数据重算其他阈值而无需重跑。

每次运行报告：

- TTFT p50、p90、p95、p99 和最大值；
- 20s/40s 请求级 SLO attainment；
- 20s/40s token 加权 SLO attainment；
- 20s/40s SLO-goodput；
- offered、实际发送和完成输入 token/s；
- 请求成功、失败、超时和 prompt-token 数不匹配计数；
- 发送偏差 p50/p95/p99/max 和队列排空时间；
- 每个 DP rank 实际接收的请求数和输入 token 数；
- 总 goodput 和每物理卡 goodput。

对阈值 `T`：

```text
SLO-goodput(T)
  = sum(input_length_i for successful request i with TTFT_i <= T)
    / (last_completion_time - first_actual_send_time)
```

一次运行在某个 SLO 下的通过条件为：零失败、零超时、prompt token 数全部匹配、
TTFT p99 不超过该 SLO、发送端偏差通过门槛且队列最终排空。20s 和 40s 分别
给出 PASS/FAIL，不合并成一个结果。

## 6. 执行门槛

### 6.1 开跑前

1. 校验 V4 bundle ID、tokenizer/encoding SHA-256 和所有文件校验值；
2. 验证 request ID、输入长度和 arrival 序列与冻结契约一致；
3. 使用参考配置完成短、中、长共 8 请求接入检查；
4. 完成 32 请求 warmup，并单独验证最长请求；
5. 从服务日志确认模型架构、W8A8、eager、DP/TP/EP、prefix cache、EPLB、
   `max_num_batched_tokens`、`max_num_seqs` 和 `max_model_len` 的实际值；
6. 确认客户端按绝对时间开放环发送，同 timestamp 请求并发提交。

原生服务无法完成加载、健康检查或首个正确请求时，不进入性能矩阵。

### 6.2 无效运行

下列运行保留原始证据，但不进入比较：

- 数据、模型、运行时或实际生效配置与清单不符；
- warmup 未完成，或服务进程在计量中重启；
- 请求/token/顺序与冻结文件不一致；
- profiler、prefix cache、EPLB、MTP、DBO 或 speculative decoding 被意外开启；
- 服务日志显示请求因 `max_num_seqs` 等待；
- 客户端不能实现冻结到达计划；
- 缺少逐请求 TTFT，无法离线重算 SLO。

有效运行不重复。无效运行默认标记为 `INVALID`；是否补跑由资源负责人单独
决定，不能静默替换原始结果。

## 7. 产物和报告

建议结果目录：

```text
bench_results/deepseek_v4_flash_w8a8_baseline/
├── 00_plan/       # 数据、环境、拓扑和共同 arrival plan 清单
├── 01_accept/     # 8 请求、warmup 和最长请求验收
├── 02_matrix/     # 3 topology × 4 batch caps 的逐请求结果
└── 03_report/     # 汇总表、图和结论
```

每个运行清单至少保存：仓库 commit/dirty 状态、模型和 tokenizer/config hash、
运行时版本、设备和 rank 映射、完整启动命令、实际解析配置、bundle/request/
arrival SHA-256、目标负载、调度参数、逐请求结果和有效性判定。

最终报告至少包含：

1. 每个拓扑下 TTFT p50/p95/p99 随 batch-token 上限的变化；
2. 20s/40s SLO attainment 和 SLO-goodput；
3. 两个 8 卡拓扑的直接比较；
4. 8 卡到 16 卡的总 goodput 与单卡 goodput；
5. 所有失败和无效运行清单；
6. 明确声明结果为单窗口单次观测，未测容量上界和运行间波动。
