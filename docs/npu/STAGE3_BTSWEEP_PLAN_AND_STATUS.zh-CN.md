# Stage-3 btsweep — 实验计划与进度存档

> **创建：** 2026-08-07
> **用途：** Stage-3「Batch-Token 伸缩性深挖」实验的**单一权威存档**——完整计划 + 当前进度 + 恢复指引，跨会话/跨 agent 可接续。
> **原始 approved 计划：** `/home/shwstone/.codefuse/engine/cc/plans/lovely-strolling-crane.md`（2026-08-06 批准）。
> **互补文档：** 仓库根 `SESSION_STATUS.md`（全实验总状态）；Stage-2 存档见 `bench_results/prefill/` 报告。

---

## 1. 背景与目标（4 条验收标准）

Stage-1 单次 sweep 发现反直觉现象（`bench_results/prefill/analysis_all_angles.md` §5 A4）：
**AFD 在固定 RPS 下，`max_num_batched_tokens` 越大、TTFT 反而越低**（bt8192 → bt32768 一路下降，之后平台化），
而 baseline 单调变差（rps4: 655ms@8192 → 1.60s@65536）。Stage-2 采集（P3-P6）已全部完成（81 verified），
本实验在其之后**深挖这个现象**。

1. **解释**：用实验数据清晰解释「为什么 batch token 变大 AFD 性能变好」。
2. **token 流动**：对一段实际请求，采集 token 在 baseline vs AFD 中的具体流动，形象化说明。
3. **归因**：正确归因性能提升是 **AFD 忙时占用率更高 / 吞吐量更大 / 气泡更小**，且有数据支撑。
4. **宽趋势**：固定 RPS 下把 batch token 从最小扩到最大（先估算两系统各自能承受的上限），看更广趋势。

**已确认决策：** RPS = rps6 主 + rps8 次；插桩 = 仅 L2 profiler（不改 forward path）；仅 A2（生产默认，不加 A0 控制臂）。

---

## 2. 现象机制假设（待实验验证）

```
max_num_batched_tokens ↑
  → 每请求跨 step 数 ↓（如 50k 请求：bt8192=7 steps → bt65536=1 step）
  → 每 step 固定开销摊薄（CAM dispatch/combine、kernel launch、pipeline fill/drain 都是 per-step 固定成本）
  → A2 token-split 两 stage 更满，attention 阶段变长，FFN 有更多时间并行 → overlap 效率 ↑
  → FFN/attention 忙时占用率 ↑、气泡（idle/fill-drain）↓
  → 有效吞吐 tokens/s ↑（逼近 EP8 上限）
  → 队列积压 ↓ → TTFT ↓（直到平台/饱和）
```

baseline（同步 DP4TP8 EP32）没有这个机制：更大 batch → 整批同步计算更长 + 排队更久 → TTFT 单调上升。

**归因三因子如何区分（验收 3）：**
- **吞吐 tokens/s**：`verified.json` 自带 `total_token_throughput` / `request_throughput`（免费，所有 L0 都有）。
- **忙时占用率**：L2 torch_npu trace 每 rank busy-ratio（`profile_trace.py` 扩展）；npu-smi 连续采样 AICore% 曲线（独立交叉验证）。
- **气泡**：L2 trace 中 attention/FFN 之间的 idle gap、dispatch/combine 暴露时间、pipeline fill/drain；
  `AFD_ASYNC_MOE_LAYOUT_LOG` 的 per-stage real/padded token 不平衡；请求级排队代理 B1(r)/B2(arrival-order 波)。
- 因果链：气泡↓ → 占用率↑ → 吞吐↑ → TTFT↓，用数据分别展示 AFD 随 mbt 的变化方向并量化。

---

## 3. 实验设计

### 3.1 拓扑 / 配置（沿用 Stage-2，不改）
- baseline `dp4_tp8_sp`（DP4×TP8 SP，EP32，max-num-seqs 8，gpu-util 0.90，block-size 128）
- AFD `afd_dp3_tp8_ep8`（A2：ubatching on, token split；attn max-num-seqs 32, gpu-util 0.80；FFN EP8，开 EPLB）
- `--max-model-len 70000`、`--max-num-batched-tokens $MAX_NUM_BATCHED_TOKENS`（已有 env 通道）
- 数据集 `tools/datasets/cp8sp50k_token_ids.jsonl`（875 prompts，prefix=0 cold，num_warmups=32，SLO 10s）
- **max-num-seqs / max-model-len 保持不变**；mbt 超出 KV 容量只会让调度填不满、不会 OOM——真正约束是 activation 内存与性能饱和（已由 P0 校准）

### 3.2 扫描网格
- **L0 宽扫描（24 configs）**：`mbt ∈ {4096, 8192, 16384, 32768, 49152, 65536, 98304, 131072}` × 2 systems × 3 repeats；
  `rps6 全区间`（8 个 mbt）+ `rps8 子集 {8192, 32768, 65536, 131072}`（4 个 mbt）
- 服务器复用：`run_stage2_l0.sh` group key `(system, bt)`，同组内 rps6/rps8 间 reset_prefix_cache、repeat 间 reset
- 每 cell 采集：TTFT mean/p99、SLO、`total_token_throughput`、`request_throughput`（自动写入 verified.json）

---

## 4. 阶段划分 P0-P4

### P0 估算 + 校准（完成 ✅）
- `estimate_batch_capacity.py`：解析模型 config（hidden 7168, 10 layers, kv_lora 512, qk_rope 64 →
  **每 token KV ≈ (512+64)×2B×10 = 11520 B**）、npu-smi HBM（64GiB/NPU）、gpu-util、weights 占用
  → 每系统 KV token 容量 + 候选 mbt 下 activation 内存边际 → 理论上限 + 推荐扫描区间
- 校准 smoke（`--smoke`）：baseline @131072 + AFD @131072 均 boot 干净、healthy run（verified repeat1 已写）
- **交付：** `bench_results/prefill_stage3/00_plan/capacity_estimate.json`
  - baseline 上限 ≈ **131072**（activation-bound；196608 有 OOM 风险）
  - AFD 内存安全到 **262144**，但 EP8 吞吐是软顶（compute-bound）

### P1 L0 宽扫描（进行中 ⏸ 22/24）
- 网格见 §3.2；**无插桩**的正式数值（3 repeats）
- 产出：TTFT vs mbt（双 rps）、吞吐 vs mbt 曲线——AFD 下降→平台→(可能)回升、baseline 单调上升、交叉点

### P2 归因 replay（L2 profiler + npu-smi + env 日志，未开始）
- 选中 cell：`rps6 × mbt ∈ {8192, 32768, 131072} × {baseline, AFD}` = 6 个 L2 replay
  （+ rps8 mbt32768 AFD 1 个，展示饱和边界）= **7 profile cells**
- 每 replay：torch_npu profiler（AFD：Attention/FFN 服务分别通过 `--profiler-config` 注册控制端点，并按
  FFN start → Attention start → replay → Attention stop → FFN stop 调用；baseline：`VLLM_TORCH_PROFILER_DIR`）
  + npu-smi 连续采样（node0+node1）+ AFD `AFD_ASYNC_MOE_LAYOUT_LOG=1`
  与 `AFD_CAM_OP_IO_LOG=1`
- 产出：每 rank busy-ratio、attention/FFN idle bubble、dispatch/combine 时间、comm/compute overlap、
  每 NPU AICore% 占用、stage token 不平衡 → 归因表

### P3 token 流动图（未开始）
- 选 1 个代表请求（~20k token，near median 16936）
- baseline：L2 trace + `ceil(len/mbt)` 步数推导 → 每 step 串行 attention→FFN 时间线
- AFD：L2 trace + layout log + op IO log → 2-stage token-split 后 attention/FFN overlap Gantt、dispatch/combine 点
- 跨进程按 (layer, ubatch, dispatch-pair) 因果对齐（绝对时间戳跨节点仅 best-effort）

### P4 报告 / 图表（未开始）
- 图表：① TTFT vs mbt 宽趋势；② tokens/s vs mbt；③ 占用率/气泡 vs mbt（归因）；④ token 流动 Gantt
- 对照 4 条验收标准逐条核对

---

## 5. 当前进度快照（2026-08-07 09:00 CST）

| 项 | 状态 |
|---|---|
| P0 估算 + 校准 | ✅ 完成（capacity_estimate.json） |
| P1 L0 宽扫描 | ⏸ **22/24 组 [DONE]**，1 [FAIL]，0 [SKIP] |
| P2 L2 profile（7 cell） | 未开始 |
| P3 token-flow Gantt | 未开始 |
| P4 报告 | 未开始 |
| Pod 对 | ❌ **均 Stopped**（约 08:00 到期停止） |
| cron 监控 | ❌ **已删除**（30bc09d7，2026-08-07 用户要求关闭；session-only cron 一夜未触发是主因） |

**P1 已完成的 22 组：** baseline 全 12（rps6 八档 + rps8 四档）+ AFD rps6 前 7 档（4096…98304）+ AFD rps8 三档（8192/32768/65536）。
**剩余 2 组：** `AFD mbt131072 rps6`（= [FAIL] server not ready，pod 到期导致）+ `AFD mbt131072 rps8`（未开始）。

**数据保全：** 22 组 verified.json 在已停止 pod 的保留存储中（`itask stop` 保留存储），重启 pod 即恢复，`--resume` 无缝续跑。

---

## 6. 关键文件 / 工具

### 结果目录 `bench_results/prefill_stage3/`
- `00_plan/` — `capacity_estimate.json`、`sweep_grid.json`（server-reuse 有序 cell 列表 + profile_cells）
- `01_sweep/` — L0 配置（`rps6/`、`rps8/`，verified.json 实际在容器侧 result_directory）
- `02_profile/` — P2 profile 配置 + traces/ + telemetry/
- `03_reports/` — analyze/render 输出（`btsweep_attribution.json/csv`、`btsweep_charts.html`）
- `stage3_btsweep.log`、`stage3_calib_{base,afd}_131072.log`

### 工具 `tools/benchmarks/`（新建/修改）
- 新建：`estimate_batch_capacity.py`、`make_btsweep_configs.py`、`sample_npu_smi.py`、`analyze_btsweep.py`、`render_btsweep_charts.py`
- 修改：`run_stage2_l0.sh`（+`PHASE=btsweep|btsweep-profile`、`PROFILE=1` 模式：torch_npu profiler + layout log + npu-smi 采样器）、
  `profile_trace.py`（+`busy_occupancy`：busy_ratio/bubble/overlap）、`monitor_stage2.sh`（glob 覆盖 `stage3_*.log`）
- 不改 `afd_plugin/` 源（仅 L2 决策，全部是已有 env-gated 能力）

---

## 7. 恢复指引（Resume）

### S0 拉 pod（两个都 Stopped，需先拉起）
```bash
itask start afd-exp-2    # node0, 33.215.117.99
itask start afd-s2-2     # node1, 33.215.116.107
```
- 确认同 /23（33.215.116.0/23）✅；容器时钟是 UTC（比 CST 慢 8h）；
  `itask list` 的 "up" 是陈旧值，用容器内 `date` / `ps -o lstart= -p 1` 验证新鲜度
- 若 IP 变了，更新下面命令里的 `NODE0_IP`

### S1 续跑 P1（补 2 组）
```bash
nohup env NODE0=afd-exp-2 NODE1=afd-s2-2 NODE0_IP=33.215.117.99 PHASE=btsweep \
  stdbuf -oL -eL bash tools/benchmarks/run_stage2_l0.sh --resume \
  > bench_results/prefill_stage3/stage3_btsweep.log 2>&1 &
```
约 40 min（2 组 × 3 repeats）。

### S2 P2 L2 profile replay（7 cell）
```bash
nohup env NODE0=afd-exp-2 NODE1=afd-s2-2 NODE0_IP=33.215.117.99 PHASE=btsweep-profile \
  stdbuf -oL -eL bash tools/benchmarks/run_stage2_l0.sh --resume \
  > bench_results/prefill_stage3/stage3_btsweep_profile.log 2>&1 &
```
> profiler 由脚本对每个 replay 显式调用 `/start_profile`、`/stop_profile`，不再使用固定 step 窗口。
> 精确跨节点合并前，必须用 `afd_trace_clock_sync.py` 为同一 session 采集四时间戳校准文件。

### S3 分析 + 渲染
```bash
python3 -m tools.benchmarks.analyze_btsweep   # → 03_reports/btsweep_attribution.json
python3 -m tools.benchmarks.render_btsweep_charts \
  --input bench_results/prefill_stage3/03_reports/btsweep_attribution.json \
  --output bench_results/prefill_stage3/03_reports/btsweep_charts.html
```

### S4 P3/P4
- P3：从 P2 选代表请求画 token-flow Gantt；P4：出报告 + 对照 4 条验收标准。

---

## 8. 风险 / 备注

- **cron 教训**：session-only cron（`durable:false`）绑定会话进程，退出/非空闲即失效，**过夜监控必须用 durable cron 或保持会话**。
  当前无监控，进度靠手动查询。
- mbt 超 KV 容量不会 OOM（调度填不满），真正上限是 activation 内存 + 性能饱和 → P0 校准已覆盖。
- 数据量：多 rank L2 trace 很大，归档时 `df -h` + SHA256 校验，不删原始 trace。
- A0 控制臂不在范围内（用户决策）；async 解耦 vs uBATCH 流水贡献不单独量化，L2 trace 只**观察** overlap 是否实际发生。
- 可选扩展：AFD 扫到 mbt=196608（校准已验证内存安全）若趋势需要。
- Stage-2 早期 3-repeat 数据（2026-08-05）已污染，勿用；当前干净数据为准。
