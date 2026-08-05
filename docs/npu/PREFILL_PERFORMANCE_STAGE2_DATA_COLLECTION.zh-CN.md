# Prefill 单次 Cell Sweep 后第二阶段 NPU 数据采集执行方案

## 1. 目的与适用范围

本文定义 DeepSeek V3.2 减层模型完成第一遍单次 cell sweep 后，在 32 张
Ascend NPU 上执行的第二阶段数据采集方案。重点不是继续扩大全量笛卡尔积，而是：

1. 从单次 sweep 中选择有解释价值的 cell；
2. 对选中 cell 补齐可用于正式结论的无 profiler 重复；
3. 在固定 workload 上完成 AFD 消融实验；
4. 独立采集可构建 Attention/CAM/FFN 流水图的 trace 和运行时事件；
5. 对 prefix cache 区分 cold-cache、steady-state 和实际命中 token；
6. 形成可校验、可关联、可重新分析的数据归档。

当前第一遍 sweep 包含：

```text
system       = {baseline, AFD}
batch_tokens = {8192, 16384, 32768, 49152, 65536}
RPS          = {4, 6, 8, 10, 12}
prefix       = {0, 25, 50, 75, 90, 95, 99}
repeat       = 1
```

这些单次结果只能用于选点和发现异常，不能单独支撑“稳定收益”或因果归因。
任何进入正式图表或结论的 cell 必须至少有 3 次有效、无 profiler 的重复，并报告
离散程度。

本方案以 reduced-layer 模型的机制分析为主。若当前启动命令启用了 forced expert
balancing，结果必须标记为 synthetic/mechanism diagnostic。Full-layer + natural
routing 属于后续 production-oriented 验收，不阻塞本阶段采集。

## 2. 固定系统与采集分层

### 2.1 固定系统配置

- baseline：`DP4 × TP8`，SP on；
- AFD Attention：`DP3 × TP8`，SP on、EP off；
- AFD FFN：`EP8`；
- 总硬件预算：32 张 NPU；
- workload：prefill only，output length = 1；
- TTFT SLO：10 秒；
- dataset：875 个生产 trace-derived mixed-length prompt；
- client：仓库固定的 patched `vllm bench serve`；
- 正式端到端结果不得开启 torch_npu profiler 或 msprof。

### 2.2 三层采集

| 层级 | 用途 | Profiler | 重复要求 | 主要产物 |
| --- | --- | --- | --- | --- |
| L0 | 正式 E2E 性能和消融 | off | 每个正式 cell/variant 至少 3 次 | raw/verified result、TTFT、SLO、throughput、遥测 |
| L1 | 固定 workload 轻量运行时事件 | off | 每个选中 variant 至少 1 次独立 replay | step/token/cache/ubatch/expert 事件、server log |
| L2 | 流水和算子归因 | on，独立运行 | 每个 variant 至少 1 次有效 capture | torch_npu trace、msprof、pipeline manifest |

L1/L2 只解释 L0 的结果，不参与 headline TTFT、SLO 或 throughput 统计。L0 除正常
服务日志和系统遥测外，不开启新增的逐层/逐 ubatch diagnostic 事件。

## 3. 第二阶段总流程

```text
单次 sweep 汇总
  -> 冻结候选 cell 与 workload
  -> L0: 选中 cell 无 profiler 重复
  -> L0: 固定 workload AFD 消融
  -> L0: prefix cold/steady 正式重复
  -> L1: 匹配的轻量事件 replay
  -> L2: calibration
  -> L2: torch_npu 与 msprof 独立重放
  -> trace 关联和流水图
  -> 完整性检查、哈希和归档
```

任何模型、checkpoint、容器、驱动、CANN、vLLM/vLLM-Ascend、AFD commit、
物理 rank placement 或数据集发生变化，都必须创建新的采集批次，不得与旧批次直接
合并计算 repeat 统计。

### 3.1 建议的最小采集矩阵

以下数量按一次 client workload 记作一个 run；server restart、calibration 和失败
重跑不计入：

| 数据集 | 选择 | 无 profiler runs | Profile replays |
| --- | --- | ---: | ---: |
| 主对比 | low/knee/high/regression × 2 systems × 3 repeats | 24 | 选中 case 另计 |
| `real-knee` 消融 | A0/A1/A2 × 3 repeats | 9；另加 3 次 L1 event replay | 3 variants × 2 profiler tools |
| `long-short` 消融 | A1/A2 × 3 repeats | 6；另加 2 次 L1 event replay | 2 variants × 2 profiler tools |
| Prefix | 50/90/99 × cold/steady × 2 systems × 3 repeats | 36 | 1 个高 prefix paired case × 2 tools |
| Pipeline baseline | real-knee baseline | 已包含在主对比 | 1 variant × 2 tools |

Prefix=0 使用主对比的 off/cold 结果，不重复构造 steady-state。若 low、high、
regression 指向同一个 cell，应去重；若第一阶段的 run 满足第 7.1 节条件，也可以计为
对应 cell 的 repeat 1。

该最小矩阵的目标是获得可归因证据，不是覆盖所有可能交互。只有发现 prefix、chunk
或 expert skew 明显改变结论时，再增加交互 case。

## 4. Step 0：汇总单次 Sweep 并冻结候选 Cell

### 4.1 生成单次结果汇总

对第一遍结果使用 `expected-repeats=1`，避免把单次 screening 误报为缺少结果：

```bash
python -m tools.benchmarks.prefill_report \
  --result-dir bench_results/prefill_sweep \
  --baseline dp4_tp8_sp \
  --candidate afd_dp3_tp8_ep8 \
  --expected-repeats 1 \
  --output bench_results/prefill_stage2/00_selection/sweep_report.json \
  --csv-output bench_results/prefill_stage2/00_selection/sweep_report.csv
```

保留第一阶段所有 raw、verified 和错误日志，不删除失败 cell。失败请求继续按
all-issued SLO miss 统计。

### 4.2 选点规则

Prefix=0 的主对比至少选择以下四类 cell：

1. **Low**：baseline 和 AFD 均无失败，all-issued SLO 接近满额，排队尚不明显；
2. **Knee**：TTFT 曲线开始明显变陡，或 SLO 首次出现显著下降的最低 RPS；
3. **High**：knee 之后仍能完成请求、且系统差异明显的高压点；
4. **Regression guard**：单次 sweep 中 AFD 相对 baseline 回退最大的可复现候选点。

同时满足以下覆盖条件：

- 至少包含一个 `batch_tokens=32768` 或 `65536` 的 cell；
- 至少包含一个长请求占主导、或 `>48K` 长度桶差异明显的 cell；
- baseline 和 AFD 必须使用完全相同的 dataset、RPS、batch tokens 和 prefix 状态；
- 不能因为某个系统失败而只保留成功请求子集。

Prefix 敏感性从 `{0, 50, 90, 99}` 中选择同一个 batch-token/RPS anchor。优先使用
主对比的 knee cell；若该点在高 prefix 下完全失去压力，则选择能够让两侧保持稳定
排队的共同 RPS。

### 4.3 候选清单

创建 `00_selection/selected_cells.json`，至少保存：

```json
{
  "source_sweep_report_sha256": "<sha256>",
  "selection_version": 1,
  "cells": [
    {
      "case_id": "real-knee-p0-mbt32768-rps10",
      "purpose": "knee",
      "batch_tokens": 32768,
      "rps": 10,
      "prefix_ratio": "0",
      "dataset_sha256": "<sha256>",
      "request_ids_sha256": "<sha256>",
      "selection_reason": "<measured evidence from the single sweep>"
    }
  ]
}
```

`selection_reason` 必须引用实际 TTFT/SLO/失败数据，不得只写“效果最好”。

## 5. Step 1：为第二阶段冻结配置与目录

### 5.1 不复用第一阶段全矩阵配置

从第一阶段配置复制出独立文件：

```text
stage2_e2e_<case-id>.json
stage2_ablation_<workload-id>_<variant>.json
stage2_profile_<workload-id>_<variant>.json
stage2_prefix_<case-id>_<cache-state>.json
```

当前 `prefill_experiment` 的 `run` 命令会对指定的
`system + batch_tokens + prefix_ratio` 执行配置文件中的全部 RPS 和 repeat，因此
每个 Stage 2 配置只保留目标 RPS。不要在同一个运行中的配置文件上原地修改 RPS、
repeat 或 dataset。

### 5.2 固定 arrival schedule

正式 paired comparison 必须固定：

- request IDs、顺序和 token IDs；
- Poisson/gamma 生成 seed；
- 每个请求的目标发送偏移；
- warmup 请求和正式请求的边界；
- prefix precondition 请求及其顺序。

仅记录 `RPS` 和 `burstiness` 不足以证明两个系统接收了相同 workload。应将实际生成
的相对发送时间保存为 `arrival_schedule.jsonl`，并由 baseline、AFD 和所有消融
variant 共同使用。

如果当前 client 版本不能导出并重用 arrival schedule，应在正式 L0/L1 采集前补齐
该能力；在此之前只能使用显式相同 seed，并把这一限制写入结果。

### 5.3 目录结构

建议使用以下目录：

```text
bench_results/prefill_stage2/
  00_selection/
    sweep_report.json
    sweep_report.csv
    selected_cells.json
  01_environment/
    preflight.json
    stack.json
    topology.json
    time_sync/
  02_e2e/
    <case-id>/<system>/repeat-<n>/
      run_manifest.json
      client/
      server/
      telemetry/
  03_ablation/
    <workload-id>/<variant>/repeat-<n>/
  04_prefix/
    <case-id>/<cold-or-steady>/<system>/repeat-<n>/
  05_profile/
    <workload-id>/<variant>/<torch-npu-or-msprof>/
      run_manifest.json
      client/
      server/
      traces/
      summaries/
  06_reports/
  SHA256SUMS
```

推荐 run ID：

```text
<date>-<workload>-<system>-mbt<tokens>-rps<rate>-p<prefix>-<variant>-r<repeat>
```

## 6. Step 2：环境与会话级数据采集

每个连续实验批次开始前运行 preflight：

```bash
python -m tools.benchmarks.prefill_preflight \
  --require-npu \
  --model-config /models/DeepSeek-V3.2-reduced \
  --experiment-config /path/to/stage2_e2e_<case-id>.json \
  --dataset tools/datasets/cp8sp50k_token_ids.jsonl \
  --output bench_results/prefill_stage2/01_environment/preflight.json
```

除 preflight 已采集的 Git、Python/package、dataset/model hash、`npu-smi` 和
`msprof` 版本外，还必须归档：

- 容器镜像 digest；
- driver、firmware、CANN、HCCL、CAM 和算子包版本；
- 节点名、IP、NPU 型号、物理 device ID、PCIe/网络拓扑；
- rank table、global/local rank、DP/TP/SP/EP 映射；
- NUMA、CPU affinity、网卡绑定；
- baseline、AFD Attention 和 AFD FFN 的完整命令行；
- 相关环境变量，敏感值脱敏；
- 模型层数、quantization、expert-routing 模式；
- prefix cache、chunked prefill、block size、`max_num_seqs`、eager/graph；
- 节点时钟同步状态和偏移。

至少在每个 run 前后保存一次 `npu-smi info`。如果环境支持连续采样，再以固定频率
采集：

- NPU utilization、HBM、power、temperature；
- host CPU、RSS、load、I/O；
- 网络接口 byte/packet/error/drop counter；
- 服务进程存活状态和重启次数。

连续采样命令依赖目标镜像中的 `npu-smi`/系统工具版本，必须先记录实际命令和
采样周期，不能在不同系统间使用不同采样方式后直接比较。

Profile 开始前还要记录 `df -h` 和目标目录可用空间。全 rank torch_npu/msprof
trace 可能很大；原始 trace 在完成解析和 SHA256 校验前不得删除或只保留导出图。

## 7. Step 3：L0 正式 E2E 重复

### 7.1 重复策略

每个选中 cell 对 baseline 和 AFD 各采集至少 3 次无 profiler 结果。第一阶段的
单次结果只有在以下条件全部满足时才可以计为 repeat 1：

- Git、模型、容器、runtime 和配置 hash 完全一致；
- 物理 rank placement 一致；
- dataset 和 arrival seed/schedule 一致；
- raw、verified、server log 和环境记录完整；
- run 期间未开启 profiler；
- 无 NPU reset、HCCL 异常或服务重启。

否则 Stage 2 从 repeat 1 重新采集。

为了降低运行顺序和热状态偏差，按 round 交错系统顺序：

```text
round 1: baseline -> AFD
round 2: AFD -> baseline
round 3: baseline -> AFD
```

每次切换 system 或 batch-token limit 都重启服务。每个 run 的 server ready、
warmup、cache state 和监控启动顺序必须一致。

### 7.2 执行

先打印命令并归档：

```bash
python -m tools.benchmarks.prefill_experiment \
  --config /path/to/stage2_e2e_<case-id>.json \
  plan \
  --system dp4_tp8_sp \
  --batch-tokens 32768 \
  --prefix-ratio 0
```

由实验人员按 plan 中的 server command 启动对应服务，确认所有 rank ready 后执行：

```bash
python -m tools.benchmarks.prefill_experiment \
  --config /path/to/stage2_e2e_<case-id>.json \
  run \
  --system dp4_tp8_sp \
  --batch-tokens 32768 \
  --prefix-ratio 0 \
  --resume
```

AFD 组使用相同流程，只替换 `--system`。`--resume` 前必须人工确认当前运行的服务
确实属于该 system/batch-token group。

### 7.3 每个 Run 必须采集

Client：

- raw detailed JSON；
- verified JSON；
- request ID、source row、input length、success/error、TTFT、E2EL、SLO；
- 请求目标发送时间和实际发送时间；
- input tokens/s、requests/s；
- client start/end monotonic 和 wall-clock timestamp。

Server：

- 所有节点完整 stdout/stderr；
- scheduler admission、queue、scheduled token、chunk 信息；
- 服务命令、PID、环境、rank mapping；
- 服务 ready、首请求、末请求和 drain 完成时间；
- OOM、HCCL timeout、NPU error、request reject/truncate 记录。

Telemetry：

- run 前后 NPU 快照；
- 覆盖整个 measured window 的 NPU/host/network 时序；
- telemetry 的采样间隔、开始/结束时间、缺失区间。

当前 `prefill_report.py` 主要聚合 TTFT/SLO，还未正式汇总 input tokens/s 和
requests/s。正式报告前应补齐聚合，或从 raw result 中生成独立 throughput 表，并
保存生成脚本版本。

## 8. Step 4：L1 固定 Workload 消融

### 8.1 Workload

至少冻结两个 workload：

1. `real-knee`：从生产 trace-derived dataset 选择的 knee cell，保持 875 个请求；
2. `long-short`：固定总 logical tokens，包含一个长请求和多个短请求，主动放大
   request-boundary split 的 token 不均衡。

`long-short` 必须记录每个请求的长度、顺序、总 token、length CV、max/min，以及
request-boundary split 在无运行时干扰时的预期两阶段 token 数。

### 8.2 AFD Variant

在同一 AFD 拓扑上执行：

| Variant | 配置 | 目的 |
| --- | --- | --- |
| A0 | `async_moe_ubatching=false` | 判断两阶段 pipeline 本身的贡献 |
| A1 | ubatching on，`async_moe_split=request` | 保留流水，允许 request-level 不均衡 |
| A2 | ubatching on，`async_moe_split=token` | 验证 token balancing |
| A3，可选 | token split + 独立关闭 overlap | 仅在存在安全、单一控制开关时隔离 overlap |

除表中目标开关外，variant 之间的配置 diff 必须为空。至少固定：模型、硬件、AFD
rank mapping、batch tokens、RPS、prefix、dynamicQuant、expert routing、CAM/HCCL、
block size、chunk policy、arrival schedule。

每个 A0/A1/A2 正式 E2E variant 至少运行 3 次，profiler 和新增的逐层 diagnostic
事件均关闭。随后分别做一次匹配的 L1 轻量事件 replay，以及独立的 L2 profile
replay。若 A3 需要临时代码或不同 commit，必须归档 patch/commit，且不得与
A0/A1/A2 混为同一软件栈的直接比较。

### 8.3 L1 轻量事件

每个 model step/layer/ubatch 最少记录：

```text
run_id
engine_step_id
role
global_rank / dp_rank / tp_rank / ep_rank
layer_id
ubatch_id
request_ids or request/chunk digest
scheduled_real_tokens
parent_padded_tokens
stage_real_tokens
stage_padded_tokens
num_computed_tokens
num_cached_tokens
connector_sequence_id
dispatch/combine begin/end
attention/ffn begin/end
```

推荐同时派生：

```text
token_imbalance = abs(stage0_real - stage1_real) /
                  (stage0_real + stage1_real)
padding_ratio   = total_stage_padding / total_stage_physical_tokens
```

轻量事件不得调用会造成 device synchronization 的取值方法。若字段只能通过同步
获得，应放入独立 diagnostic replay，不得打开在正式 E2E run 中。

当前仓库已有 stage/ubatch 和部分布局日志，但尚未提供上述完整、结构化、可跨进程
关联的事件文件。它是正式流水采集前的实现缺口。

## 9. Step 5：Prefix Cache 数据采集

### 9.1 Anchor

对 `{0, 50, 90, 99}` 使用同一个 batch-token/RPS anchor，baseline 和 AFD 使用
相同 dataset、arrival schedule、precondition policy 和采集窗口。

### 9.2 Cold Cache

每个 cold run：

1. 重启服务并确认 KV/prefix cache 为空；
2. 不使用 vLLM 内置的重复请求 warmup；
3. 直接发送正式 dataset；
4. 记录每请求和每 step 的 cached/computed token；
5. 结束后销毁服务，下一次 repeat 重新冷启动。

### 9.3 Steady State

不要用完整正式 prompt 预热，否则会把 intended partial hit 变为整请求命中。生成
独立的 `prefix_precondition.jsonl`，只包含每组 block-aligned shared prefix。

DP3 和 DP4 的共同中性组大小为 12。precondition manifest 必须记录每组发送次数、
顺序和 token hash，并在 baseline/AFD 使用完全相同的策略。预热完成后等待请求
drain，再发送正式固定 workload。

Steady-state 至少采集：

- requested prefix ratio；
- dataset manifest 中的 actual shared ratio；
- runtime cached tokens；
- runtime newly computed tokens；
- cache lookup time；
- eviction/block 使用量；
- 每个 DP rank 的 cache-hit 分布；
- cold/steady TTFT、SLO 和 throughput。

如果当前 runtime 不能输出 cached/computed token，则高 prefix 数据只能标记为
“constructed-prefix experiment”，不能宣称达到了 90% 或 99% 实际命中率。

## 10. Step 6：L2 流水 Trace 采集

### 10.1 Profile Case

至少采集：

1. `real-knee`：baseline、A0、A1、A2；
2. `long-short`：A1、A2；
3. 高 prefix anchor：baseline、A2，使用 steady-state；
4. 一个 full-rank 高压样本，用于发现 DP/SP/expert straggler。

Profile workload 使用与 L1 相同的 request IDs 和 arrival schedule，但可以使用
显式记录的短 capture 子集。推荐结构：稳定 warmup、固定 active 请求窗口、drain。
active window 的请求列表和 hash 必须写入 profile manifest。

### 10.2 Calibration

先执行短 calibration，确认：

- 一个复杂请求可能跨多少 model step；
- profiler 的 WAIT/WARMUP/ACTIVE/SKIP_FIRST 是否覆盖完整 matched window；
- trace 文件大小和落盘时间；
- Attention 与 FFN profiler 是否在对应窗口均 active；
- profile run 完成后服务是否正常 drain。

Calibration 数据不用于正式 trace 对比。

### 10.3 Torch NPU Profile

AFD Attention 与 FFN 分别配置目录：

```bash
export AFD_NPU_ATTENTION_PROFILER_ENABLE=true
export AFD_NPU_ATTENTION_PROFILER_WAIT=0
export AFD_NPU_ATTENTION_PROFILER_WARMUP=1
export AFD_NPU_ATTENTION_PROFILER_ACTIVE=<calibrated-steps>
export AFD_NPU_ATTENTION_PROFILER_SKIP_FIRST=0
export AFD_NPU_ATTENTION_PROFILER_DIR=<run-dir>/traces/attention

export AFD_NPU_FFN_PROFILER_ENABLE=true
export AFD_NPU_FFN_PROFILER_WAIT=0
export AFD_NPU_FFN_PROFILER_WARMUP=1
export AFD_NPU_FFN_PROFILER_ACTIVE=<calibrated-steps>
export AFD_NPU_FFN_PROFILER_SKIP_FIRST=0
export AFD_NPU_FFN_PROFILER_DIR=<run-dir>/traces/ffn
```

至少有一次高压 case 采集全 rank。其他 case 可以先用 L1 事件识别每个 DP 组的代表
rank 和最慢 rank，再采集详细 trace，但必须记录 rank 选择规则，不能只看 rank 0。

### 10.4 msprof

torch_npu profiler 和 msprof 使用两个独立 replay，避免叠加开销和改变流水。msprof
至少采集：

- NPU timeline 与 stream；
- kernel/operator statistics；
- HCCL collective time/volume；
- CAM dispatch/combine 相关算子；
- memory copy/transpose；
- device utilization 和 memory；
- host/device launch gap。

保存完整 msprof 命令、版本、采集范围和导出参数。

### 10.5 跨进程关联字段

为了绘制 Attention/CAM/FFN 流水，trace event 或 sidecar 至少携带：

```text
run_id
engine_step_id
request_id or chunk_id
layer_id
ubatch_id
role
rank
connector_sequence_id
real_tokens
padded_tokens
```

跨节点关联优先使用：

```text
engine_step_id + layer_id + ubatch_id + connector_sequence_id
```

不能只依赖不同节点 trace 的绝对时间。每次会话归档 PTP/chrony/系统时钟状态和偏移，
校准后时间戳用于绘制全局位置，sequence ID 用于确定因果顺序。

当前 `tools.benchmarks.profile_trace` 可以汇总单 trace 并比较分类耗时，但还不能合并
多 rank、恢复 connector 因果箭头或自动生成关键路径。这些功能在流水图生成前需要
补齐，或通过独立分析脚本实现并归档其 commit。

## 11. Step 7：流水图与派生指标

每个正式 profile workload 至少输出四类图：

1. **Client timeline**：arrival、client wait、server processing、TTFT；
2. **Model-step waterfall**：scheduler、Attention、TP/SP communication、CAM
   dispatch、FFN、CAM combine、output；
3. **Layer/ubatch pipeline**：ubatch 0/1 在各层 Attention 与 FFN 间的交错；
4. **Rank heatmap**：每 rank 的 compute、communication、idle、tokens 和
   straggler 状态。

AFD waterfall 推荐 lane：

```text
Client
Scheduler / Engine
Attention DP/TP ranks
TP/SP collective
CAM dispatch
FFN EP ranks / experts
CAM combine
Attention continuation
Output
```

Baseline 使用相同 client/scheduler 时间轴，并将本地 Attention、MoE/FFN、TP/SP/EP
collective 分开，避免把整个 model step 画成一个不可解释的矩形。

至少输出以下数值：

- model-step critical path；
- Attention、FFN、dispatch、combine、TP/SP、EP 的 union time；
- communication 与 compute overlap；
- exposed communication；
- Attention/FFN idle bubble；
- ubatch token imbalance 和 padding ratio；
- per-rank busy/idle；
- expert/rank load max/mean、CV 和 straggler duration。

不得把跨 rank 的 event duration 直接相加当作关键路径。

## 12. 每个 Run 的 Manifest

每个 L0/L1/L2 run 保存独立 `run_manifest.json`：

```json
{
  "run_id": "<stable-id>",
  "collection_level": "L0|L1|L2",
  "purpose": "e2e|ablation|prefix|profile",
  "system": "baseline|afd",
  "variant": "baseline|A0|A1|A2|A3",
  "software": {
    "afd_commit": "<sha>",
    "vllm": "0.19.1",
    "vllm_ascend": "<version>",
    "container_digest": "<digest>"
  },
  "workload": {
    "dataset_sha256": "<sha256>",
    "request_ids_sha256": "<sha256>",
    "arrival_schedule_sha256": "<sha256>",
    "num_prompts": 875,
    "batch_tokens": 32768,
    "rps": 10,
    "prefix_ratio_requested": "0",
    "cache_state": "cold|steady|off"
  },
  "profile": {
    "enabled": false,
    "tool": null,
    "active_window_sha256": null
  },
  "artifacts": []
}
```

`artifacts` 记录相对路径、类型、字节数和 SHA256。不得依赖目录名代替 manifest。

## 13. 单 Run 验收与拒绝条件

### 13.1 L0/L1 验收

- `/v1/models` 返回预期 served model；
- issued request 数与 dataset 一致；
- raw 和 verified result 均存在且 hash 已记录；
- `success + failed = issued`；
- 输入长度与 dataset index 完全一致；
- 失败请求计入 all-issued SLO miss；
- telemetry 覆盖完整 measured window；
- server log、命令、环境和 rank mapping 完整；
- 正式 E2E profiler off；
- 正式 L0 未开启新增的逐层/逐 ubatch diagnostic 事件；
- 无 truncate、静默丢请求、NPU reset 或服务重启。

### 13.2 Prefix 验收

- 明确 cold 或 steady，不允许未知 cache state；
- precondition manifest 存在；
- requested/shared/runtime ratio 分开记录；
- 缺少 runtime hit token 时不得把 constructed ratio 写成 observed hit ratio。

### 13.3 L2 验收

- baseline/variant 使用相同 active request window；
- Attention 和 FFN capture 覆盖同一批 engine step；
- trace 能关联到 run ID、rank 和 role；
- active window 未被 warmup/drain 截断；
- profiler overhead 单独记录；
- torch_npu 与 msprof 不在同一次 run 叠加；
- trace hash、大小和解析状态已记录。

不满足验收条件的 run 移入 `rejected/`，保留原因和原始产物，不覆盖重跑。

## 14. 当前工具覆盖与实现缺口

### 14.1 已覆盖

- 确定性 token-ID dataset、manifest、index 和 hash；
- `vllm bench serve` raw detailed result；
- request ID、input length、错误、TTFT/E2EL、all-issued SLO；
- length bucket；
- Git/package/model/dataset/NPU preflight；
- 单 trace 分类、time union 和 communication/compute overlap 汇总；
- AFD Attention/FFN 独立 torch_npu profiler 配置。

### 14.2 正式第二阶段前需要补齐或人工采集

1. arrival seed 显式入配置、实际 arrival schedule 导出与重用；
2. input tokens/s 和 requests/s 的 repeat-level 聚合；
3. server 命令、环境、rank mapping 和 telemetry 自动归档；
4. runtime cached/computed tokens 与 cache lookup/eviction；
5. per-step scheduled/chunk tokens；
6. per-ubatch real/padded token 和 imbalance；
7. per-layer/per-expert/per-rank routed tokens；
8. connector sequence ID 与跨角色 trace marker；
9. 多 rank trace 合并、关键路径和流水图生成；
10. profiler overhead 对照和 artifact manifest/SHA256 自动生成。

缺口 1、4、5、6、8 是 paired replay 和流水因果分析的优先项。缺少它们时仍可采集
原始 trace，但只能做算子级观察，不能完成可靠的 request/step/ubatch 级归因。

## 15. 第二阶段完成标准

- 每个正式选中 cell 至少 3 次有效无 profiler 重复；
- single-pass sweep 与正式重复在报告中明确分层；
- baseline/AFD paired case 使用相同固定 workload 和 arrival schedule；
- A0/A1/A2 消融除目标开关外配置一致；
- prefix cold/steady 分离并有运行时实际 hit/computed token；
- 至少完成 `real-knee` 和 `long-short` 的 matched pipeline 数据；
- 至少一个高压 case 具有全 rank trace/轻量事件；
- 所有 run 有 manifest、环境、日志、遥测和 artifact hash；
- 所有因果结论同时满足无 profiler E2E、固定 workload 消融、timeline 和量化证据；
- 无法分离的部分标记为拓扑/资源划分综合效应；
- 所有 summary 和图可从归档原始数据重新生成。
