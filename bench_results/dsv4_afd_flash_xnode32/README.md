# DSV4 Flash AFD 双机 32 卡 mbt 扫描（2026-09-01）

## 配置

- **拓扑**：双机 32 卡。Attention DP6TP4 = 24 ranks（node1 `v4f-base-2` 33.182.141.199 16 卡 = DP0-3，node2 `v4f-xnode-2` 33.182.140.47 卡 0-7 = DP4-5，headless，coordinator 在 node1）；FFN DP8EP8 = 8 ranks（node2 卡 8-15）。两 pod 同 /22 超节点（33.182.140.0/22）
- **代码**：插件 `test/dsv4-afd-flash-dplb`（706958a，含 PR#289 v2 DPLB + #293 未配置故 inert + 本次新增 SP hash 路由修复）；vllm-ascend `e19e14da7`（CWS 分支）；vllm 568afb3 镜像原生
- **参数**：CWS 全程开（`enable_dsv4_shared_compressor_workspace` + `multistream_dsv4_dsa_overlap:false`）、FLASHCOMM1=1（仅 attention；FFN TP1 强制 0）、HCCL_BUFFSIZE=2048（65536 格 4096）、util 0.80、max-model-len 70000、max-num-seqs 128、async-sched OFF、chunked prefill ON
- **负载**：formal_1 原速（512 条、5,260,666 tok、150s 窗口、实测发送速率 35,070 tok/s 全格一致、send_deviation_ok=True）
- 启动器 `tools/itask/launch_dsv4_afd_xnode32.sh`；cell 驱动 `tools/itask/xnode32_run_cell.sh`；清理脚本 pod 上 `/tmp/xnode32_cleanup.sh`

## 结果（每格单次，全格 512/512 成功、零失败）

| mbt | wall (s) | 排空 (s) | TTFT p50 | p95 | p99 | max | 有效吞吐 (tok/s) |
|---|---|---|---|---|---|---|---|
| 8192 | 209.2 | 58.6 | 29.84 | 54.35 | 59.44 | 66.29 | 25,143 |
| 16384 | 158.2 | 8.2 | 3.19 | 10.63 | 14.68 | 20.83 | 33,256 |
| 32768 | 158.2 | 7.8 | 2.46 | 6.66 | 10.45 | 12.04 | 33,257 |
| 65536 | 158.2 | 7.5 | 2.22 | 5.72 | 7.29 | 8.35 | 33,258 |

## 读数

- **mbt=8192 是明显瓶颈格**：有效吞吐 25.1K < 供给 35K（1.39× 超载），TTFT 被排队主导（p50 29.8s、排空 58.6s）。chunk 太小导致 prefill 效率不足
- **16384 起进入可服务区**（33.3K ≈ 供给的 95%，drain 8.2s），32768/65536 完全消化 35K 供给，差距在尾部延迟：p99 14.7 → 10.5 → 7.3s，max 20.8 → 12.0 → 8.4s
- 单调收益、无回退：**该拓扑下 mbt 越大越好，65536 是最优格**（CWS 使 65536 的 KV 容量不再受限，见 v4-cws-compressor-workspace-mbt）
- 对照单机 8+8 卡（DP2TP4+DP8EP8, mbt=10240）容量 17.8K tok/s：32 卡 mbt≥16384 后容量 ≥35K（供给受限），mbt=8192 格 25.1K 为 1.41×

## 本次新踩的坑（已修）

1. **SP hash 路由修复（代码改动）**：FLASHCOMM1=1 下 attention SP 把 token 按 TP4 切块（router_logits 32 = 128/4），但 `deepseek_v4_attention_gate.py` 直接拿 forward_context.input_ids（全量 128）做校验 → profile_run 崩 `input_ids/token count mismatch`。修复 = 对齐原生 experts_selector 的 flash_comm_v1 分支：pad 到 router_tokens×tp_size 后 `tensor_split` 取本 rank 连续块
2. **FlashComm1 只能开 attention**：FFN TP1 直接断言 `Flash Comm v1 is only supported when tp_size > 1` → 启动器按角色分流
3. **FFN recv buffer 随 attention rank 数膨胀**：24 attn ranks × mbt=65536 时 FFN `CamMoeDistributeDispatchRecv do tiling failed ret=-1`（buffer 不足签名）在 2048MB 仍炸，4096 通过（单机 8 ranks 时 1024 即够）
4. **跨代残留**：FFN worker 进程名是 `VLLM::Worker_DP*`（不是 VLLMWorker 也不含 EngineCore 于旧 pkill 模式），且 npu-smi 进程表里 pid 在第 3 列不是第 2 列 —— 两代 FFN 残留各占 42GB 导致后续启动 "Free memory 19.68/61.28 GiB less than desired"
5. **pkill 自匹配 2.0**：同一条 exec 命令里 awk 的正则文本（含 "EngineCore"）会被 pkill -f 匹配到自身 → 清理脚本必须落成 pod 上的文件再执行
6. **驱动脚本**：itask exec CRLF 污染要让所有命令替换过 `tr -d '\r'` 再比较；远程 bash -c 双引号里 `$2` 会被远程侧提前展开（改用 `cat | grep -c`）
7. CAM connector 端口 1239 会被上一代栈残留占用（EADDRINUSE）→ 清理含 /proc/net/tcp 04E7 端口等待

## 快放容量扫描（2026-09-02，固定 mbt=65536 + HCCL_BUFFSIZE=4096，同一 server 连跑）

| 负载 | 供给 tok/s | wall (s) | 排空 (s) | TTFT p50 | p95 | p99 | max | 判定 |
|---|---|---|---|---|---|---|---|---|
| 1.0x | 35,070 | 158.2 | 7.5 | 2.22 | 5.72 | 7.29 | 8.35 | 轻松 |
| 1.5x | 52,606 | 111.1 | 11.1 | 4.45 | 9.03 | 11.37 | 14.06 | 跟上 |
| 2.0x | 70,142 | 87.1 | 12.0 | 5.44 | 12.07 | 13.91 | 15.06 | 跟上 |

- 全档 512/512 零失败；两档快放排空仅 11-12s → **容量下界 ≥70K tok/s，knee 未触达**（2x 里瞬时 93K 突发桶只产生短暂排队：30-45s 桶供给 93.2K/完成 62.7K，随后两桶 75-77K 追平）
- TTFT 随负载平缓退化（p50 2.2→4.5→5.4s，p99 7.3→11.4→13.9s），无崩溃拐点
- 注：pod 重启后 vllm-ascend 需恢复（master 因 GitHub 不通改用 NAS format-patch 2 个 commit 落到 80d8c194f 上，HEAD=ce2c8d96d 内容等同 e19e14da7；xnode-2 直接 fetch 成功）；新增坑：node2 上 attention-headless 与 FFN 同时启动会在 stateless_init_dp_group 的 get_open_port 竞态 EADDRINUSE → FFN DP 组起不来整栈挂死，对策=FFN 错后 20s 启动

## 归档

- 本地：`bench_results/dsv4_afd_flash_xnode32/xnode32_mbt*.json`（4 格 mbt 扫描 + fast1p5x/fast2x，含逐请求 TTFT）、`formal_1_arrivals.png`（数据集到达形态图）
- NAS：`shwstone/xnode32_results/`（全部 JSON + 三侧日志 + README）
- 快放 plan：`tools/datasets/moonconv-wildchat-v4-flash-prefill/workloads/formal_1_fast{1p5x,2x}_plan.json`（52.6K/70.1K tok/s，窗口 100s/75s）；驱动 `tools/itask/xnode32_speed_runs.sh`

## 对照实验：双实例 stock baseline（2026-09-02）

**配置**：每节点一个独立 DP4TP4EP16 实例（16 卡×2=32 卡总量与 AFD 对齐），mbt=8192、FLASHCOMM1=1、CWS 开（与 AFD 格同代码 e19e14da7）、util 0.80、async-sched OFF、不设 HCCL_BUFFSIZE。实例间用 `tools/benchmarks/least_load_router.py` 做请求级路由：score = waiting + running + 路由侧 inflight（对齐 vLLM 内部 DP LB 语义；per-engine gauge 求和到实例级，0.5s 轮询 + inflight 补偿吸收突发）。驱动 `tools/itask/xnode32_baseline2x_runs.sh`。

路由均衡性（决策日志）：1x 266/248、1.5x 257/255、2x 267/245（约 52/48），零失败。

### 同口径对比（同数据集、同 32 卡、同指标定义）

| 指标 | baseline 2×DP4TP4EP16 | AFD DP6TP4+DP8EP8 (mbt65536) | AFD 优势 |
|---|---|---|---|
| 1x 有效吞吐 | 32,236 | 33,400 | +3.6% |
| 1x 排空 | 13.2s | 7.5s | −43% |
| 1x TTFT p50/p99/max | 2.84/11.31/15.15 | 2.22/7.29/8.35 | −22%/−36%/−45% |
| 1.5x 有效吞吐 | 44,444 | 47,368 | +6.6% |
| 1.5x 排空 | 18.4s | 11.1s | −40% |
| 1.5x TTFT p50/p99 | 3.42/12.94 | 4.45/11.37 | p50 −30%（baseline 优）/ p99 −12%（AFD 优） |
| 2x 有效吞吐 | 58,886 | 60,486 | +2.7% |
| 2x 排空 | 14.3s | 12.0s | −16% |
| 2x TTFT p50/p99/max | 5.77/16.63/19.66 | 5.44/13.91/15.06 | −6%/−16%/−23% |
| 2x 峰值 15s 桶服务率 | 67,888 | 77,193 | **+13.7%** |

### 读数

- 两侧全部负载 512/512 零失败、prompt 全匹配；两侧 knee 均未触达（2x 排空仅 12-14s）
- **AFD 的核心优势在尾部与突发吸收**：2x 档 45-75s 高压期 AFD 持续服务率 75-77K vs baseline 67-68K（+13.7%）；排空时间全档短 16-43%；TTFT 尾部全档更优
- baseline 内部 mbt=8192 → 63K 长请求 ≥8 步 chunk，突发桶内排队更深（2x 档 30-45s 桶供给 93.2K，baseline 完成 65.7K 且下一桶仍只有 67.9K；AFD 62.7K 后迅速以 77.2K 追平）
- 1.5x 档 baseline p50 反而更好（3.42 vs 4.45）——轻载时小 chunk 起步更快，但尾部仍 AFD 优
- 注意口径：baseline 按用户指定跑 mbt=8192；AFD 取其扫描最优格 65536（AFD@8192 只有 25.1K，更差）

### 本次新坑（已修）

1. **exec 嵌套引号 + 花括号展开**：`bash -c "bash -c '... ADDITIONAL_CONFIG={json} ...'"` 里 JSON 的花括号在远程解析层被 brace expansion 拆碎（VllmConfig ValidationError: input_value='multistream_dsv4_dsa_overlap:false'）。修法 = 启动器内置具名 preset（`ADDITIONAL_CONFIG=cws`），裸 JSON 不过 exec 边界
2. **崩溃签名误报**：baseline 正常启动日志含 "Free memory X/61 GiB"（KV 预算 INFO 行），崩溃 grep 里的 `Free memory` 模式误杀。改用精确模式 `less than desired`
3. 路由自测脚本（mock 双后端）必须先验证：选路正确性 + SSE 透传首 chunk <1ms（不缓冲）——本次抓到 rel_url 拼接 bug（yarl URL 不能直接 + str）

### 归档

- 本地：`bench_results/dsv4_afd_flash_xnode32/baseline2x/base2x_mbt8192_{1x,fast1p5x,fast2x}.json`
- NAS：`shwstone/xnode32_baseline2x/`（3 JSON + 两侧实例日志 + router 日志 + 路由决策 jsonl）
