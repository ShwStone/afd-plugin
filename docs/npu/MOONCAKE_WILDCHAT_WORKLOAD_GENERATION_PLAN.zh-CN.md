# Mooncake–WildChat 真实预填充工作负载设计与生成记录

> 状态：数据已生成并发布。本文件保留生成设计、验收依据和可复现步骤；性能
> 实验的当前执行口径以
> [DeepSeek-V3.2 完整模型预填充性能实验计划](DEEPSEEK_V3_2_FULL_PREFILL_PERFORMANCE_PLAN.zh-CN.md)
> 为准。

- 数据集：[`ShwStone/moonconv-wildchat-prefill`](https://huggingface.co/datasets/ShwStone/moonconv-wildchat-prefill)
- 固定 revision：`d7d54c16db9d4b2348729b4170a9612b8a47b64a`
- bundle ID：`fe4f751b6dab`
- 数据状态：`data_valid`
- 产物：1,696 条请求、17,475,111 个输入 token；其中三个正式窗口各 512 条
- 尚未完成：数据发布过程没有运行 8 请求 NPU 服务重放，性能实验阶段零必须补齐

实际产物的字段、文件校验值和来源 revision 以数据集内的
`bundle_manifest.json`、`source_manifest.json` 和 `reports/quality_report.json`
为准。下文中的未来时态和检查表描述的是生成时的门槛，不表示数据仍待生成。

本文档记录如何为完整 DeepSeek-V3.2 性能实验生成一套可复现、有真实文本
语义、同时保留真实到达形状的预填充工作负载。

本文档只描述数据生成和验收，不修改模型、服务端或已经验证通过的功能。
生成的数据用于补充
[DeepSeek-V3.2 完整模型预填充性能实验计划](DEEPSEEK_V3_2_FULL_PREFILL_PERFORMANCE_PLAN.zh-CN.md)
中的正式工作负载。性能计划已经切换到本数据集；旧 `cp8sp50k` 和 Poisson
到达不再进入完整模型正式结果。

## 1. 要解决的问题

关闭 Expert Parallelism Load Balancer（以下简称 EPLB）后，模型不会在运行
期间重新调整专家放置。每个专家收到多少 token，取决于模型路由器对实际输入
内容的判断。因此，只复现输入长度，或者使用随机 token，不能代表真实的专家
负载。

Mooncake FAST'25 trace 提供真实请求的到达时间、输入长度、输出长度和匿名
前缀块标识，但是出于隐私原因不提供原始文本或 token。它适合描述流量形状，
不能单独产生 DeepSeek-V3.2 的真实专家路由。

本方案把工作负载拆成两个来源：

1. [Mooncake FAST'25 Conversation trace](https://github.com/kvcache-ai/Mooncake/blob/main/FAST25-release/README.md)
   提供请求顺序、到达间隔、输入长度分布和突发形状；
2. [WildChat-4.8M](https://huggingface.co/datasets/allenai/WildChat-4.8M)
   提供真实用户与模型之间的完整多轮对话内容。

最终产物是“Mooncake-derived Conversation prefill workload”，即 Mooncake
提供流量外壳，WildChat 提供语义内容。它不是 Mooncake 原始生产请求的恢复版，
也不是完整的在线生成负载。

### 1.1 本文中的数据术语

| 术语 | 本文含义 |
|---|---|
| 来源版本（revision） | 数据仓库中固定的一次提交。固定它以后，同名数据集后续更新不会改变本实验输入。 |
| 数据分片（shard） | 为了便于下载和并行处理，把大数据集拆成的多个文件。 |
| 候选请求 | 从真实 WildChat 对话的某个用户轮次构造出的完整输入，尚未与 Mooncake 请求匹配。 |
| 候选索引 | 只保存候选来源、token 数和筛选状态的表，不保存正文和完整 token ID。 |
| 物化 | 根据已经冻结的映射，再次读取少量被选正文，生成客户端能够发送的 token ID 文件。 |
| 检查点（checkpoint） | 一个数据分片处理完成后的状态记录，用于失败后继续，而不是从头处理。 |
| 清单（manifest） | 记录来源、配置、文件校验值和验收状态的机器可读文件。 |
| 工作负载包（bundle） | 一次完整生成的请求、到达计划、映射、报告和清单的集合。 |
| 数据格式版本（schema version） | 说明 JSON 字段定义的版本号；字段含义改变时必须增加版本。 |
| 虚构测试样本（fixture） | 专门为单元测试编写、不来自真实用户的数据。 |
| 同分候选排序（tie-break） | 多个候选长度误差相同时，用固定规则决定选择顺序。 |
| CPU / NPU | CPU 执行下载、分词和筛选；NPU 只用于最后的服务接入与模型运行检查。 |

## 2. 目标和非目标

### 2.1 目标

生成的数据必须满足以下要求：

- 输入来自真实用户对话，而不是随机 token 或自动生成的占位文本；
- 每条输入是某个真实多轮会话在用户发出请求时能够看到的完整历史；
- 使用正式 DeepSeek-V3.2 服务相同的分词器和对话模板；
- 输入长度尽量匹配 Mooncake 请求的输入长度；
- 请求顺序和基础到达间隔来自 Mooncake 的真实 Conversation trace；
- 传统部署和 Attention–FFN 解耦部署读取完全相同的 token、顺序和到达计划；
- 所有选择规则在查看系统性能结果之前冻结；
- 原始来源、筛选过程、排除原因和最终文件都有可核验的摘要和校验值；
- 数据生成不需要占用 NPU，只有最终的小规模服务接入检查需要 NPU。

### 2.2 非目标

本方案不尝试完成以下工作：

- 恢复 Mooncake 原始用户的文本或 token；
- 根据 Mooncake 的匿名 `hash_ids` 伪造能够命中前缀缓存的 token；
- 复现 Mooncake Tool and Agent trace 的工具调用语义；
- 测量长输出解码性能；
- 开启前缀缓存后的收益；
- 让数据天然产生均衡或不均衡的专家负载；
- 根据传统部署或 Attention–FFN 解耦部署的性能反向选择数据；
- 把 WildChat 的访问时间戳描述成 Mooncake 生产服务的到达时间。

## 3. 冻结的上游来源

### 3.1 WildChat 内容来源

首选数据集为公开、非 gated 的 `allenai/WildChat-4.8M`：

| 项目 | 固定值 |
|---|---|
| Hugging Face 数据集 | `allenai/WildChat-4.8M` |
| 配置和切分 | `default/train` |
| 固定 revision | `c827c6df8fcf008219ffaffa4d1dd77491099367` |
| 数据集许可证 | ODC-BY |
| 公开版本规模 | 3,199,860 段对话 |
| 下载大小 | 约 15.3 GB |
| 使用内容 | 消息角色和消息正文 |

WildChat 论文说明这些对话来自真实用户与 ChatGPT 服务的交互，并获得了用户
同意。输入历史同时包含真实用户消息和当时实际返回的模型回复。数据集公开版本
已经过滤被 moderation 工具标记为 toxic 的内容，并进行了脱敏和密钥扫描。
本实验仍然遵循最小化原则，不把地理位置、请求头或散列用户标识带入工作负载。

固定 revision 是必须条件。只记录数据集名称而不记录 revision，会导致后来
重新生成时使用不同内容。

### 3.2 Mooncake 到达来源

| 项目 | 固定值 |
|---|---|
| GitHub 仓库 | `kvcache-ai/Mooncake` |
| 固定提交 | `e94a0b86ba067455d8b0524eb2cbb5fbac2db024` |
| trace 文件 | `FAST25-release/traces/conversation_trace.jsonl` |
| 使用字段 | `timestamp`、`input_length`、原始行号 |
| 不用于正式映射的字段 | `output_length`、`hash_ids` |

使用 FAST'25 更新版 trace，不使用历史文件
`arxiv-trace/mooncake_trace.jsonl`。

`hash_ids` 只描述匿名的 512-token 前缀块关系。由于没有对应原始 token，
正式数据不使用该字段制造内容或缓存命中。基础报告可以统计原 trace 的前缀
复用特征，但不能声称最终 WildChat token 保留了这些关系。

### 3.3 分词器和对话模板

不能用通用 DeepSeek-V3 分词器代替实际部署的 DeepSeek-V3.2 分词器。生成时
必须从正式模型目录或与正式模型完全相同的 tokenizer revision 加载：

- tokenizer 文件；
- tokenizer 配置；
- special token 定义；
- chat template；
- 是否自动添加开头 token 的行为。

生成清单必须保存这些文件的 SHA-256 校验值，以及最终渲染出的 chat template
文本校验值。如果正式模型的 tokenizer 校验值变化，旧数据不能继续作为同一
版本工作负载使用。

## 4. 数据定义

### 4.1 什么是一条候选请求

WildChat 的一段对话可以包含多个用户轮次。对于每一个有效用户消息，取从
对话开始到该用户消息为止的全部消息：

```text
user_1
assistant_1
user_2
assistant_2
...
user_n
<等待 assistant_n>
```

这段历史就是用户第 `n` 次请求时的真实输入。原来对应的 `assistant_n`
回复不放入输入；正式性能实验仍然只生成一个输出 token。

例如，一段含三个用户轮次的对话最多产生三个自然候选请求：

| 候选 | 输入消息范围 |
|---|---|
| 第 1 个候选 | `user_1` |
| 第 2 个候选 | `user_1` 至 `user_2` |
| 第 3 个候选 | `user_1` 至 `user_3` |

不得把最终 assistant 回复放进同一轮输入，也不得把不相关的两段对话拼接成长
输入。

### 4.2 有效对话规则

候选生成前按以下规则检查每段对话：

1. 只接受 `user` 和 `assistant` 角色；出现未知角色时排除整段对话，不猜测
   角色含义；
2. 第一条有效消息必须来自用户；
3. 用户和 assistant 消息必须按对话顺序出现；连续重复角色、角色缺失或结构
   损坏时排除整段对话；
4. 当前用户消息去除空白后不能为空；
5. 消息正文必须是字符串，并且能够编码为 UTF-8；
6. 不使用公开数据中的 IP 散列、国家、地区、浏览器请求头和其他用户元数据；
7. 不尝试恢复已经脱敏的内容；
8. 以对话内容散列去重；相同内容只保留 `(source_shard, source_row_index)`
   按字典序最小的一条。

这里的“去重”只消除内容完全相同的重复记录，不做语义近似去重。语义去重会
引入模型和阈值，可能改变真实内容分布。

### 4.3 候选标识

每条候选的稳定标识由以下内容计算：

```text
source dataset revision
+ source shard repository-relative path
+ row index within that shard
+ conversation content hash
+ user turn ordinal
```

把这些字段写成 key 排序、无额外空格的规范 JSON，再对其 UTF-8 字节计算
SHA-256，得到 `candidate_id`。这样不会因为字符串连接边界不清产生歧义。
正式产物使用 `candidate_id`，不使用 IP 散列或其他用户标识。

原始 shard 按仓库相对路径排序，每个 shard 内使用 Parquet 物理行号。这样并行
worker 的完成顺序不会改变来源位置。固定 revision 后，这些字段和用户轮次
不会变化；同一份输入重新生成时必须得到相同的 `candidate_id`。

### 4.4 精确 token 化

对候选消息调用正式 tokenizer 的 chat template：

```python
tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
)
```

具体实现必须服从正式 tokenizer 的接口，不能先拼字符串再调用另一个通用
分词器。禁止以下操作：

- 随机 token 补齐；
- 重复最后一句补齐；
- 从无关文档复制 token 补齐；
- 为了命中目标长度从消息正文中间切断 token；
- 对不同系统应用不同 chat template；
- tokenize 后再次添加一套 special token。

一条候选的 `input_length` 是最终送入服务端的 token ID 数量，不是 WildChat
原论文用其他 tokenizer 统计的长度。

## 5. 两阶段生成，避免保存整个数据集的 token

WildChat 有数百万段对话。没有必要把所有候选的 token ID 长期保存。生成过程
分成索引阶段和物化阶段。

### 5.1 第一阶段：候选索引

逐 shard 读取固定 revision 的 WildChat 数据：

1. 校验对话结构；
2. 对每个用户轮次构造自然会话前缀；
3. 用正式 tokenizer 计算准确 token 数；
4. 写入不含正文和 token ID 的候选索引；
5. 记录所有排除原因的计数；
6. 每个 shard 完成后写入 checkpoint 和输出校验值。

候选索引建议使用 Parquet，至少包含：

| 字段 | 含义 |
|---|---|
| `candidate_id` | 稳定候选标识 |
| `source_row_index` | 当前 source shard 内的原始行号 |
| `conversation_id` | 对话内容散列，不包含用户身份 |
| `user_turn_ordinal` | 当前用户轮次，从 0 开始 |
| `message_count` | 输入中包含的消息数 |
| `input_length` | DeepSeek-V3.2 精确 token 数 |
| `language` | WildChat 已有语言标签；缺失时为 null |
| `source_shard` | 原始 shard 名称 |
| `eligible` | 是否满足基本输入上限 |
| `exclusion_reason` | 不满足时的单一规范原因 |

候选索引不保存消息正文、IP 散列、位置、请求头或完整 token ID。

### 5.2 第二阶段：最终物化

完成 Mooncake 窗口选择和长度匹配后，只重新读取被选中的 WildChat source
rows。对这些候选再次执行相同 token 化，然后写出最终 token ID。

第二次 token 化的结果必须与候选索引中的 `input_length` 一致。任何一条不一致
都说明来源、tokenizer 或 chat template 发生变化，应停止整个物化过程。

这种两阶段设计有三个好处：

- 不需要保存数百万条可逆的 token 序列；
- 最终工作负载体积只与被选中的请求数有关；
- 可以在不重新读取正文的情况下审查长度覆盖和选择结果。

## 6. Mooncake 请求范围

### 6.1 共同输入上限

正式工作负载只接受：

```text
1 <= Mooncake input_length <= 65,536
```

输出长度固定为 1 token。超过 65,536 token 的 Mooncake 请求不截断，记录为
`over_common_input_limit` 并从本轮正式范围排除。

Mooncake 的长度来自其原服务使用的 tokenizer，不等于 DeepSeek-V3.2 对同一
文本的长度。由于原文本不可用，本方案把 Mooncake 数字当作目标计算规模：最终
选择的是经过 DeepSeek-V3.2 tokenizer 后具有相近 token 数的另一条真实对话
输入，不声称两者文本相同。

在当前 FAST'25 Conversation trace 中，约 97.89% 的请求不超过该上限。这个
比例必须由生成工具重新计算并写入报告，不能只复制本文数字。

### 6.2 保留被排除请求的时间影响

选择窗口时，跳过超限请求，但不能压缩它们原本占据的时间。例如两个被选请求
之间夹有一条超限请求，最终两条请求仍使用原始 timestamp 的差值。这样可以
避免因为删除请求而把到达间隔人为缩短。

每条正式请求保留原始 `mooncake_trace_index`。质量报告列出每个窗口跨越的
原始行号范围、其中超限请求数和被保留请求数。

## 7. Mooncake 窗口选择

### 7.1 需要生成的窗口

第一版数据生成以下固定集合：

| 集合 | 数量 | 请求数 | 用途 |
|---|---:|---:|---|
| 正式窗口 | 3 | 每个 512 | 三组成对正式重复 |
| 筛选窗口 | 1 | 128 | 寻找容量区间，不提供最终数字 |
| 预热集合 | 1 | 32 | 服务预热，不计入指标 |

正式窗口共 1,536 条请求。筛选和预热使用与正式窗口不同的 WildChat
conversation，不得复用正式输入。

### 7.2 正式窗口必须连续

窗口不是从整个 trace 随机抽取 512 条请求，而是从某个起点开始，按原 trace
顺序收集后续 512 条符合输入上限的请求。中间超限请求可以跳过，但其时间间隔
保留。

连续窗口能够保留下列特征：

- 同时到达的请求；
- 短时间突发；
- 长短请求随时间出现的关系；
- 原始请求顺序。

随机分层抽样会破坏这些特征，因此只用于候选内容索引，不用于选择 Mooncake
请求。

### 7.3 三个时间区域

把一小时 Conversation trace 按原始 timestamp 分为三个等长时间区域。每个
区域选择一个 512 请求窗口，避免三个正式重复都来自同一个短时间片。

在每个区域内枚举所有能够得到 512 条有效请求的起点。每个候选窗口计算：

- 输入长度的平均值；
- 输入长度第 50、90、95、99 百分位；
- 相邻请求时间差的平均值和第 95、99 百分位；
- 时间差为零的请求比例；
- 任意滚动 1 秒内的最大请求数；
- 任意滚动 10 秒内的最大请求数。

以完整、符合输入上限的 Conversation trace 为参照。每项误差先除以该项在
所有候选窗口中的四分位距，消除 token、毫秒和请求数的单位差异，再取各项
绝对误差的平均值作为窗口分数。四分位距为零的项不参与评分。

每个时间区域选择分数最低的窗口。分数相同时选择原始起点更早的窗口。选择
过程不读取任何系统性能结果。

### 7.4 筛选窗口

筛选窗口按相同方法选择 128 条有效请求，但必须满足：

- 不与三个正式窗口的原始行号范围重叠；
- 长度分布和突发指标相对完整 trace 的误差最小；
- 在正式实验结束前不因筛选结果而更换内容。

### 7.5 预热集合

预热集合不需要重放真实到达间隔。它从未被筛选或正式集合使用的目标长度中
选取 32 条，覆盖短、中、长输入。预热后必须按实验计划清理请求状态；前缀
缓存本身保持关闭。

## 8. WildChat 与 Mooncake 的长度匹配

### 8.1 先建立覆盖表

根据所有目标请求的 Mooncake `input_length`，划分以下审查区间：

| 区间 | token 范围 |
|---|---:|
| 极短 | 1～1,023 |
| 短 | 1,024～4,095 |
| 中短 | 4,096～8,191 |
| 中长 | 8,192～16,383 |
| 长 | 16,384～32,767 |
| 超长 | 32,768～65,536 |

区间只用于覆盖审查，不用于把长度取整。实际匹配始终使用每条请求的精确长度。

对每个区间，计算：

- 目标 Mooncake 请求数；
- 满足基本过滤的 WildChat 候选数；
- 能够在该区间提供至少一个候选的唯一 conversation 数；
- 唯一 conversation 数与目标数的比例。

每个非空目标区间的可用唯一 conversation 数至少是目标请求数的 5 倍。
不能用同一段长对话产生的多个用户轮次把这个比例虚高。低于 5 倍时停止匹配，
先处理内容覆盖问题，不能通过重复提示或随机填充补足。

### 8.2 单条请求误差

对目标长度 `target_length` 和候选长度 `actual_length` 定义：

```text
absolute_error = abs(actual_length - target_length)
relative_error = absolute_error / target_length
```

单条请求的硬门槛为：

```text
absolute_error <= max(64, target_length × 10%)
```

使用绝对误差下限是为了避免几十 token 的短请求因少量 chat template token
产生很大的相对误差。

### 8.3 唯一性规则

第一版正式数据采用保守规则：

- 同一个 WildChat conversation 在所有正式、筛选和预热集合中最多出现一次；
- 同一个 `candidate_id` 只能使用一次；
- 内容散列相同的重复对话只能使用一次；
- 不允许两个独立 Mooncake 请求引用同一份 token ID。

每条候选本身仍可以包含真实的多轮历史。这里限制的是“同一段对话不能在运行
中被重复选中”，目的是避免在无法恢复 Mooncake 原会话关系时人为制造重复
主题或前缀共享。

因此，第一版数据保留 Mooncake 的到达和长度特征，但不声称保留 Mooncake 的
会话关系或缓存局部性。

### 8.4 确定性匹配顺序

匹配过程按以下固定顺序执行：

1. 先计算每个目标在硬门槛内有多少可用候选；
2. 候选最少的目标先匹配，避免超长请求最后无内容可选；
3. 对同一目标，优先选择绝对 token 误差最小的候选；
4. 误差相同时，用固定 `selection_seed`、`request_id` 和 `candidate_id`
   计算 SHA-256，散列值最小者优先；
5. 选中候选后，立即移除同一 conversation 的其他候选；
6. 所有匹配结束后，按原 Mooncake 顺序恢复正式请求顺序。

第一版固定：

```text
selection_seed = 20260818
```

固定 seed 只用于同误差候选的无偏排序，不生成内容、长度或到达间隔。

### 8.5 语言分布检查

Mooncake 不公开原请求语言，因此不能对齐 Mooncake 的语言比例。正式数据默认
不限制 WildChat 语言，只记录最终语言分布。

为了避免同长度候选的文件顺序让某一种语言被意外放大，不使用语言作为过滤条件
或 tie-break；同误差候选只使用前述固定散列排序。生成报告同时给出：

- 各长度区间可用唯一 conversation 的语言分布；
- 最终选中请求的语言分布；
- 选中分布相对候选池变化最大的十种语言。

如果任一主要语言的占比变化超过 5 个百分点，应检查是否由长度覆盖造成。
不自动强制调整语言配额；需要调整时先更新本计划并生成新的数据版本。

## 9. 到达计划

### 9.1 基础时间

每个窗口第一条被选请求的到达偏移设为 0：

```text
arrival_offset_ms =
    original_timestamp_ms - first_selected_timestamp_ms
```

同一 timestamp 的请求必须具有相同 `arrival_offset_ms`。不能为避免客户端并发
而添加随机抖动。

### 9.2 负载缩放

Mooncake 原始平均负载可能高于测试系统容量。容量搜索只允许使用统一的时间
膨胀系数：

```text
scaled_arrival_offset_ms =
    base_arrival_offset_ms × dilation_factor
```

`dilation_factor` 大于 1 时降低负载，小于 1 时提高负载。所有零间隔仍然保持
为零。派生计划把结果一次性换算为整数纳秒并取最近整数，避免小于 1 的系数
把非零毫秒间隔过早压成零。客户端直接读取派生后的整数纳秒，不再自行做浮点
计算。

基础到达计划随数据一起冻结。每个具体实验点的缩放计划单独生成并保存：

- `dilation_factor` 的十进制定点字符串；
- 缩放前基础计划的 SHA-256；
- 缩放后计划的 SHA-256；
- 计划总时长、平均请求速率和平均输入 token 到达速率。

传统部署和 Attention–FFN 解耦部署在同一实验点必须引用同一个缩放计划文件，
不能各自在客户端重新计算浮点时间。

### 9.3 客户端要求

现有只接受固定 RPS 或随机到达的客户端不能直接承担正式重放。正式运行前需要
确认客户端能够：

- 按每条请求的绝对偏移时间发送；
- 在相同 timestamp 并发提交多条请求；
- 记录计划时间、实际发送时间和二者差值；
- 不因前一请求变慢而推迟后续计划，即使用开放环发送；
- 在发送端饱和时报告偏差，而不是静默变成闭环负载。

客户端接入不属于本数据生成脚本，但数据验收必须包含一个小规模重放检查。

## 10. 最终产物

### 10.1 逻辑目录

真实数据不提交到 Git。建议在测试服务器持久存储中生成如下 bundle：

```text
moonconv_wildchat_v1/
├── source_manifest.json
├── build_config.json
├── candidate_index/
│   ├── part-00000.parquet
│   ├── ...
│   └── index_manifest.json
├── selection/
│   ├── mooncake_exclusions.jsonl
│   ├── window_candidates.parquet
│   ├── selected_windows.json
│   └── mapping.jsonl
├── workloads/
│   ├── formal_0_requests.jsonl
│   ├── formal_0_arrivals.jsonl
│   ├── formal_1_requests.jsonl
│   ├── formal_1_arrivals.jsonl
│   ├── formal_2_requests.jsonl
│   ├── formal_2_arrivals.jsonl
│   ├── screening_requests.jsonl
│   ├── screening_arrivals.jsonl
│   └── warmup_requests.jsonl
├── reports/
│   ├── quality_report.json
│   └── quality_report.md
└── bundle_manifest.json
```

Git 仓库只保存生成工具、单元测试、小型虚构 fixture、配置模板，以及不含真实
token 的质量摘要和 bundle 校验值。

### 10.2 请求文件格式

每行一条请求。下面只展示字段，不包含真实 token：

```json
{
  "schema_version": 1,
  "request_id": "formal-0-000000",
  "candidate_id": "<sha256>",
  "input_length": 3,
  "output_length": 1,
  "prompt_token_ids": [1, 2, 3],
  "prompt_token_ids_sha256": "<sha256>"
}
```

`prompt_token_ids_sha256` 对无空格的 JSON token 数组计算。它用于确认物化后
内容没有变化，不代替整个文件校验值。

### 10.3 到达文件格式

```json
{
  "schema_version": 1,
  "request_id": "formal-0-000000",
  "sequence_index": 0,
  "mooncake_trace_index": 1234,
  "target_input_length": 3,
  "actual_input_length": 3,
  "base_arrival_offset_ms": 0
}
```

请求文件和到达文件通过 `request_id` 一一对应。任何缺失、重复或顺序不同都
判为无效。

### 10.4 映射文件格式

映射文件不包含正文和 token ID，至少记录：

- window 和 sequence index；
- Mooncake 原始行号、timestamp 和目标长度；
- WildChat `candidate_id`、source shard、shard 内行号和用户轮次；
- 实际长度、绝对误差和相对误差；
- source language；
- 选择时的候选数量；
- 是否通过单条误差门槛。

映射文件是审查数据选择是否合理的主要依据。

### 10.5 bundle 清单

`bundle_manifest.json` 至少包含：

- bundle schema version 和 bundle ID；
- WildChat 数据集、revision、配置和切分；
- Mooncake 仓库提交和 trace 路径；
- tokenizer 文件及 chat template 校验值；
- 生成工具的 Git commit；
- Python、tokenizer 库、数据读取库和 Parquet 库的版本；
- 完整生成配置；
- 每个文件的字节数和 SHA-256；
- 请求 ID 顺序的 SHA-256；
- 每个窗口的请求数和总输入 token 数；
- 生成开始、结束时间和主机环境；
- 所有质量门槛的通过状态。

bundle ID 使用规范化 `build_config.json` 的 SHA-256 前 12 位。已有 bundle
不得原地覆盖；配置或来源变化时生成新的 bundle ID。

## 11. 规范化和校验值

为了让不同机器得到相同校验值，JSON 和 JSONL 采用：

- UTF-8；
- Unix 换行；
- 每行一个完整 JSON 对象；
- key 按字典序排列；
- 分隔符不包含额外空格；
- 文件末尾保留一个换行；
- 整数不写成浮点数；
- 比例和时间缩放参数在配置中保存为十进制定点字符串。

SHA-256 在文件关闭并完成原子重命名后计算。临时文件、未完成 shard 和日志不
写入正式 bundle 清单。

## 12. 质量验收

### 12.1 来源验收

- WildChat dataset ID、revision、config 和 split 与冻结值完全一致；
- Mooncake 仓库提交和 trace 文件校验值一致；
- tokenizer 与正式模型校验值一致；
- 数据许可证和引用信息已经写入 source manifest；
- 原始数据没有被提交到 Git。

### 12.2 内容验收

- 所有输入都来自通过结构检查的自然会话前缀；
- 没有随机 token、padding、内容重复或跨对话拼接；
- 当前用户消息不为空；
- 没有 source conversation 被重复使用；
- 请求文件不包含 IP 散列、位置或请求头；
- 第二次 token 化长度与候选索引完全一致。

### 12.3 长度验收

- 每条请求满足单条硬误差门槛；
- 全部请求相对误差中位数不超过 5%；
- 全部请求相对误差第 95 百分位不超过 10%；
- 实际长度的第 50、90、95 百分位与目标长度对应分位点偏差不超过 5%；
- 实际长度第 99 百分位与目标偏差不超过 10%；
- 每个非空长度区间在选择前至少有 5 倍唯一 conversation 覆盖；
- 没有输入超过 65,536 token。

### 12.4 到达验收

- 正式窗口来自三个不同时间区域；
- 每个正式窗口恰好有 512 条请求；
- 筛选窗口恰好有 128 条请求且不与正式窗口重叠；
- 同 timestamp 请求仍然同 timestamp；
- 基础 offset 单调不减；
- 第一个 offset 为 0；
- 原始 trace 中被排除请求造成的时间间隔没有被压缩；
- 两个系统引用同一份请求和到达文件校验值。

### 12.5 可复现性验收

在清空生成目录后，使用同一来源和配置完整重跑一次。以下项目必须逐字节相同：

- selected windows；
- mapping；
- requests；
- arrivals；
- quality report 中除生成时间和主机信息外的统计字段。

如果 Parquet 元数据导致文件级校验值不同，必须同时比较排序后的规范化记录
校验值。正式 JSONL 产物仍要求文件级校验值一致。

## 13. 失败条件和处理顺序

出现以下任一情况时，不生成“通过”状态的 bundle：

- 固定 source revision 无法取得；
- tokenizer 或 chat template 与正式模型不一致；
- WildChat 长输入的唯一 conversation 不足 5 倍覆盖；
- 存在无法在硬误差内匹配的目标请求；
- 需要随机 padding、任意 token 截断或跨对话拼接才能匹配；
- source conversation 在正式集合中重复；
- 三个正式窗口无法满足连续性和互斥要求；
- 第二次 token 化与候选索引不一致；
- 重新生成不能得到相同正式文件；
- 客户端无法按基础到达计划重放同时请求。

内容覆盖不足时按以下顺序处理：

1. 检查是否错误过滤了合法 WildChat 候选；
2. 检查 tokenizer 和 chat template 是否加载正确；
3. 报告具体不足的长度区间和请求数；
4. 评估增加另一套有明确来源的真实长上下文数据集；
5. 修改本计划和 bundle 版本后重新生成。

不得为了让流程继续而重复提示、随机补 token、选择更容易匹配但不具代表性的
Mooncake 窗口，或放宽门槛后沿用同一 bundle 名称。

## 14. 实现计划

### 14.1 生成工具

建议新增一个入口：

```text
tools/benchmarks/build_real_prefill_workload.py
```

使用子命令区分阶段：

```text
index        构造候选索引
select       选择 Mooncake 窗口并做长度匹配
materialize  重新读取被选内容并生成 token 文件
validate     独立校验已有 bundle
summarize    重新生成不含正文的质量报告
```

保持一个入口可以共享 schema 和规范化逻辑，避免多个简单脚本各自解释长度、
哈希和来源。只有当单文件复杂度明显过高时，再按候选索引、选择和 schema
拆分模块。

所有阈值、seed、来源 revision 和窗口大小放入显式配置对象，不在函数中使用
魔法数字。依赖直接通过参数传递，不增加可变全局状态。

### 14.2 配置模板

仓库可以提交一个不包含服务器路径的 JSON 配置模板，至少包含：

```json
{
  "schema_version": 1,
  "wildchat": {
    "dataset": "allenai/WildChat-4.8M",
    "revision": "c827c6df8fcf008219ffaffa4d1dd77491099367",
    "config": "default",
    "split": "train"
  },
  "mooncake": {
    "commit": "e94a0b86ba067455d8b0524eb2cbb5fbac2db024",
    "trace": "FAST25-release/traces/conversation_trace.jsonl"
  },
  "selection_seed": 20260818,
  "max_input_tokens": 65536,
  "output_tokens": 1,
  "formal_windows": 3,
  "formal_requests_per_window": 512,
  "screening_requests": 128,
  "warmup_requests": 32,
  "minimum_unique_conversation_multiplier": 5,
  "max_relative_length_error": "0.10",
  "min_absolute_length_tolerance": 64
}
```

tokenizer 路径和输出目录由执行者显式传入，不写入通用模板。

### 14.3 测试

新增单元测试应使用自行编写的虚构对话 fixture，不复制 WildChat 正文。至少
覆盖：

- 单轮和多轮候选构造；
- 最后一条 assistant 回复不会进入当前请求；
- 空消息、未知角色和错误交替被排除；
- 相同内容对话去重；
- 超长目标被记录而不是截断；
- 长度硬门槛；
- 稀缺目标优先匹配；
- conversation 唯一性；
- tie-break 在不同输入顺序下结果一致；
- 同 timestamp 到达保持一致；
- 跳过超限请求后时间差不被压缩；
- JSONL 规范化和校验值稳定；
- 物化长度与索引不一致时失败。

集成测试使用少量公开数据行和 Mooncake trace 片段，只生成临时产物，不提交
真实 token fixture。

## 15. 执行阶段

### P0：冻结配置和来源

1. 获取 WildChat 和 Mooncake 固定 revision；
2. 保存原始文件或 shard 清单及校验值；
3. 确认正式模型 tokenizer；
4. 填写并审查 build config；
5. 在任何系统性能运行之前提交配置和生成工具。

完成标准：来源、阈值、窗口数、seed 和 tokenizer 都已冻结。

### P1：小样本试运行

1. 读取一个 WildChat shard 的一小部分；
2. 构造候选索引；
3. 检查多轮前缀是否正确；
4. 测量每秒处理对话数、CPU 利用率和峰值内存；
5. 按样本结果估算完整索引耗时和索引大小。

如果预计完整处理超过可接受时间，优先按原始 Parquet shard 使用多进程并行，
但最终合并必须按稳定候选标识排序。不能用近似 token 长度替代正式精确长度。

### P2：完整候选索引

1. 流式处理全部 shard；
2. 每个 shard 独立 checkpoint；
3. 合并并去重；
4. 输出长度和语言覆盖统计；
5. 检查 1～65,536 token 区间是否足够覆盖目标。

完成标准：candidate index manifest 通过，所有排除原因都有计数。

### P3：选择 Mooncake 窗口

1. 读取固定 Conversation trace；
2. 标记超限请求；
3. 枚举三个时间区域内的候选窗口；
4. 计算代表性分数；
5. 冻结三个正式窗口、一个筛选窗口和预热目标；
6. 输出选择前的所有候选窗口摘要，防止只保留最终答案。

完成标准：selected windows 文件只依赖 trace 和配置，不依赖 WildChat 匹配或
系统性能。

### P4：长度匹配

1. 生成各长度区间覆盖表；
2. 检查 5 倍唯一 conversation 门槛；
3. 按稀缺度和长度误差进行确定性匹配；
4. 检查 conversation 唯一性；
5. 生成 mapping 和长度误差报告。

完成标准：所有请求满足单条和总体长度门槛。

### P5：物化和 bundle

1. 重新读取被选中的 source rows；
2. 再次应用正式 chat template；
3. 校验 token 数与 candidate index 一致；
4. 写请求和基础到达文件；
5. 生成 bundle manifest 和质量报告；
6. 使用独立 `validate` 子命令检查整个 bundle。

完成标准：bundle 状态为 `valid`，所有正式文件都有 SHA-256。

### P6：确定性复跑

在新的输出目录完整重跑 P3～P5，比较正式产物。候选索引可以复用，但其
manifest 校验值必须一致。

完成标准：两次生成的 selected windows、mapping、requests 和 arrivals
逐字节相同。

### P7：服务接入检查

只使用 8 条请求进行最小检查：

1. 客户端能够读取 token ID；
2. 服务端接受所有请求；
3. 服务端实际输入 token 数与数据文件一致；
4. 输出长度为 1；
5. prefix cache 关闭；
6. EPLB 关闭；
7. 同 timestamp 请求能够并发发出；
8. 实际发送时间误差被记录。

接入检查只验证格式和重放能力，不提供性能结论。

## 16. 资源和时间预算

数据生成主要消耗网络、CPU 和磁盘，不占正式 NPU 资源。

建议预留：

- 至少 20 GB 用于 WildChat 下载文件；
- 除下载文件外至少 100 GB 可用临时空间，容纳下载缓存、候选索引和中间
  shard，实际占用在 P1 后重新估算；
- 约 200 MB 以上用于最终 token JSONL，实际大小以选中请求总 token 数为准；
- 一个 1% 小样本试运行，用于估算完整 token 化时间；
- NPU 时间只用于 8 请求接入检查和后续独立的自然路由 profile。

不预先写死完整索引耗时。P1 用同一台计划执行 P2 的机器实测吞吐，然后在
quality report 中记录估算值和实际值。

## 17. 隐私、许可证和日志

- 最终报告引用 WildChat 数据集、论文和 ODC-BY 许可证；
- 原始 WildChat 数据、token ID 和 source row 映射只保存在受控测试服务器；
- 不把原始正文或可逆 token ID 提交到 GitHub；
- 不导出 IP 散列、地区、国家、请求头或用户级关联信息；
- 运行日志只打印 request ID、候选 ID、长度和错误类型，不打印正文；
- 调试时禁止批量 decode token；确需查看单条格式时使用虚构 fixture；
- 数据过期或不再需要时，按测试服务器的数据管理规则删除，不在脚本中加入
  未经确认的自动递归删除。

## 18. 与性能实验的衔接

bundle 发布后，完整性能计划已经完成以下更新：

1. 正式工作负载改为固定数据集 revision 中的 `formal_0`、`formal_1`、
   `formal_2` 三个 512 请求窗口；
2. Poisson 到达改为 Mooncake 基础时间，并按共同目标 token/s 为每个窗口
   生成确定性的 dilation plan；
3. 256 请求分层筛选改为独立 128 请求连续筛选窗口；
4. 最大输入范围明确为不超过 65,536 token；
5. `cp8sp50k` 只保留为历史资料，不进入当前正式实验；
6. 正式运行明确 EPLB 关闭、自然路由、prefix cache 关闭；
7. 每次运行清单增加 bundle ID、请求文件校验值和到达文件校验值；
8. 结论表述改为 Mooncake-derived Conversation prefill workload，不写成
   Mooncake 原始内容或完整生产流量。

自然专家路由的 token 分布在后续独立 profile replay 中采集。该 profile
使用相同 bundle，但不改变数据选择，也不能用 profile 结果重新挑选输入。

## 19. 发布验收结果

### 来源和配置

- [x] WildChat revision 已固定并校验。
- [x] Mooncake commit 和 trace 文件已固定并校验。
- [x] 正式 DeepSeek-V3.2 tokenizer 和 renderer 已固定。
- [x] build config 已纳入数据集。
- [x] WildChat 原始 shard 和原始正文未进入数据集仓库。

### 候选索引

- [x] 配置指定的 22 个 source shard 均已处理。
- [x] 对话结构过滤和排除计数完整。
- [x] 候选 token 数来自固定 tokenizer 和 renderer。
- [x] 重复对话已确定性去除。
- [x] 每个目标长度区间达到 5 倍唯一 conversation 覆盖。

### 窗口和匹配

- [x] 三个正式窗口来自三个不同时间区域。
- [x] 筛选窗口与正式窗口不重叠。
- [x] 超限请求没有被截断，原时间差得到保留。
- [x] 每条请求满足长度硬门槛。
- [x] 总体长度分位点满足验收标准。
- [x] 所有 source conversation 唯一。
- [x] 最终语言分布已报告。

### 物化和复现

- [x] 每条物化长度与候选索引一致。
- [x] 请求、到达、映射和清单格式通过校验。
- [x] 每个正式文件都有 SHA-256。
- [x] 独立目录复跑得到相同核心产物。
- [ ] 8 请求服务接入检查通过。

### 进入正式实验前

- [x] 现有完整性能计划已经按第 18 节更新。
- [ ] 两个系统引用相同 bundle ID 和校验值。
- [ ] prefix cache 和 EPLB 都确认关闭。
- [ ] 客户端能够开放环重放绝对到达偏移。
- [x] 数据选择未使用任何待比较系统的性能结果。
