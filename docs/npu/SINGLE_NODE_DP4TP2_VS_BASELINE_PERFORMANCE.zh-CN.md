# 单机 16 卡 DP4TP2+EP8 AFD vs Baseline 性能对比报告（2026-09-04）

> 双 pod 并行 4-run 矩阵：baseline(async ON, mbt 8k/32k) × AFD(dp4tp2, mbt 65536,
> token/request 两 split)，formal_1 负载 0.5x–1.5x 步进 0.25x，共 17 格全部
> 512/512 零失败。**核心结论：AFD token-split mbt65536 单机全档最优，容量
> ≈38–40K tok/s vs baseline ≈37K；async 调度混杂已排除。**

## 1. 实验设置

| 项 | 值 |
|---|---|
| 硬件 | 2×A3 超节点内同 /22 pod（v4f-base-2=33.215.116.167、v4f-xnode-2=33.215.116.209），各 16 卡，当日同镜像同配方重建（复用 NAS 编译产物，见 §7） |
| 分线 | **base-2 跑 baseline 线，xnode-2 跑 AFD 线**（并行、互不干扰） |
| 栈 | vllm 568afb3 + vllm-ascend e19e14da7（CWS）+ afd_plugin `test/dsv4-afd-flash-dplb` @ d8085e7 |
| 负载 | formal_1：512 请求 / 5,260,666 input tokens / output=1 / 长度 p50=6,628 p99=48,015 max=63,778 |
| 速率 | 0.5x=17.5K、0.75x=26.3K、1x=35.1K、1.25x=43.8K、1.5x=52.6K tok/s（1x 窗口 150s） |
| 统一开关 | async-sched ON、CWS 开、FLASHCOMM1=1（AFD 侧）、util 0.80、mml 70000、enforce-eager、chunked prefill、无前缀缓存 |
| 系列 | baseline = DP4TP4EP16 原生（`v4_launch_baseline.sh`，mbt∈{8192,32768}）；AFD = DP4TP2 attn(NPU0-7) + DP8EP8 FFN(NPU8-15)，mbt=65536，`attn_ranks_per_dp=2`，HCCL_BUFFSIZE=4096 |
| 变量控制 | 每系列内仅 mbt（baseline）或 split 模式（AFD）变化；其余超参全部一致 |

指标口径：eff = 总 input tokens / wall；峰值15s桶 = 15s 窗口完成请求 input tokens / 15；
排空 = 最后完成距窗口结束秒数。工具 `tools/itask/bucket15_summary.py` 可复算。

## 2. 总表（eff / 峰值15s桶 / TTFT p50 / p99 / 排空）

| 速率 | AFD token 65536 | AFD request 65536 | base-async 32768 | base-async 8192 |
|---|---|---|---|---|
| 0.5x | **17,075** / 30.3K / 2.95 / 8.97 / 8.1s | 16,853 / 33.8K / 2.48 / 9.94 / 12.2s | 16,796 / 31.1K / 8.78 / 18.79 / 13.2s | 16,735 / 27.4K / 3.06 / 11.26 / 14.4s |
| 0.75x | **25,230** / 40.7K / 3.25 / 10.71 / 8.5s | 24,984 / 40.7K / 3.14 / 10.39 / 10.6s | 23,928 / 42.7K / 11.01 / 20.51 / 19.8s | 24,425 / 36.4K / 4.23 / 13.18 / 15.4s |
| 1x | **32,830** / 44.7K / 4.65 / 15.57 / 10.2s | 32,742 / 42.1K / 5.52 / 15.82 / 10.7s | 31,340 / 45.3K / 11.20 / 21.80 / 17.9s | 31,058 / 38.6K / 7.43 / 19.98 / 19.4s |
| 1.25x | **37,321** / 47.2K / 10.68 / 25.63 / 21.0s | 36,856 / 43.8K / 12.44 / 27.06 / 22.7s | 35,022 / 45.8K / 15.63 / 26.37 / 30.2s | 32,645 / 38.5K / 15.13 / 35.72 / 41.1s |
| 1.5x | **40,096** / 46.6K / 16.41 / 34.32 / 31.2s | 36,837 / 44.3K / 17.92 / 36.00 / 42.8s | 36,925 / 45.4K / 18.41 / 36.60 / 42.5s | 30,917 / 38.3K / 23.30 / 59.18 / 70.2s |

AFD request 0.5x–1x 三格来自 `dp4tp2_singlenode/`（同栈同参），1.25x/1.5x 为本轮补测。

### 2.1 baseline 拓扑对照系列：DP8TP2（eff / 峰值桶 / p50 / p99 / 排空）

| 速率 | base-DP8TP2-8192 | base-DP8TP2-32768 |
|---|---|---|
| 0.5x | 16,368 / 27.1K / 4.47 / 15.81 / 21.4s | ✗ 运行时 NPU OOM |
| 0.75x | 24,062 / 36.6K / 5.96 / 23.04 / 18.6s | ✗ |
| 1x | 29,127 / 41.0K / 8.68 / 30.49 / 30.6s | ✗ |
| 1.25x | 33,888 / 49.8K / 14.04 / 35.17 / 35.2s | ✗ |
| 1.5x | 32,401 / 40.9K / 20.84 / 56.38 / 62.4s | ✗ |

HCCL_BUFFSIZE=512（EP 域拓扑强制，见 §5）；32768 在 util 0.80 + async ON 下运行时
OOM，按用户决定记为容量结论、缺格。对照分析见 §3.5。

## 3. 结论

### 3.1 AFD vs baseline（同为 async ON，公平对照）

- **AFD token-split 65536 全档最优**。vs baseline-32768：次满载档吞吐跟供给走
  （96% vs 96%），**1x +4.7%、1.25x +6.6%、1.5x +8.6%**；1x TTFT p99 **-28%**
  （15.6 vs 21.8s）、p50 减半（4.65 vs 11.20s，65536 单 chunk prefill）。
- **持续容量（膝点）**：AFD ≈ **38–40K tok/s**（token-split 1.5x 仍爬升至 40.1K，
  排空 31s）；baseline ≈ **37K**（1.25x→1.5x 仅 +5%）。16 卡上 AFD 容量更高且
  过膝后退化更慢。
- **峰值 15s 桶平手**（AFD 44–47K vs baseline 43–46K）：AFD 赢在持续速率与
  TTFT 尾，不在瞬时爆发。
- baseline-8192 对本负载是错误工作点：1.25x 起 dev_p99 抬到 7.6ms、1.5x 崩到
  59% 供给 + p99 59.2s，全档被 32768 压制。

### 3.2 token-split vs request-split（AFD 同栈同 mbt）

- **token-split 全档 ≥ request-split**：eff +0.3%（1x）～ +8.9%（1.5x）；
  1.5x 排空 31.2s vs 42.8s。
- 机制上 request-split 1.5x 平台化（36.86K→36.84K），token-split 仍 +8.9%
  ——超载域 token 均衡切分让 4 个 attention 引擎负载更均匀。

### 3.3 async-sched 混杂消除（对比旧数据的 caveat 关闭）

| 对照（baseline-32768, 1x） | no-async（09-02 旧数据） | async ON（本轮） | Δ |
|---|---|---|---|
| eff | 31,276 | 31,340 | **+0.2%** |
| TTFT p99 | 22.76s | 21.80s | -4% |

async ON/OFF 对 baseline 32768 的 1x 吞吐影响仅 +0.2%——**此前 AFD 相对
baseline 的 +4.7% 不能归于调度器，是解耦架构本身的收益**。（8192 上 async
帮助更大：1x +4.3%、p99 24.0→20.0s，但改变不了其最早饱和的命运。）

### 3.4 双机 32 卡布局对照（同日补充，DP3TP8）

同栈仅换 attention 布局：DP3TP8 vs DP6TP4（async ON，mbt 65536）：

| 速率 | DP3TP8 eff | DP6TP4 eff | Δ |
|---|---|---|---|
| 1x | 33,555 | 33,796 | -0.7%（均供给受限） |
| 1.5x | 44,079 | 49,516 | -11.0% |
| 2x | 45,644 | 58,868 | **-22.5%**（DP3TP8 drain 40.3s vs 14.4s） |

TP8 更宽的 attention 补不回引擎数减半：DP 值大的布局（更多引擎、更轻排队）
严格占优，dp6tp4 保持默认。

### 3.5 baseline 内部拓扑对照：DP8TP2 vs DP4TP4（同 8k、async ON）

| 速率 | DP8TP2 eff (Δ vs DP4TP4) | DP8TP2 TTFT p99 | DP4TP4 TTFT p99 |
|---|---|---|---|
| 0.5x | 16,368 (-2.2%) | 15.81s | 11.26s |
| 0.75x | 24,062 (-1.5%) | 23.04s | 13.18s |
| 1x | 29,127 (-6.2%) | 30.49s | 19.98s |
| 1.25x | 33,888 (+3.8%) | 35.17s | 35.72s |
| 1.5x | 32,401 (+4.8%) | 56.38s | 59.18s |

- **次满载域 DP8TP2 全面更差**（eff -1.5~-6%,TTFT 尾接近翻倍）：TP2 把每引擎
  attention 宽度砍半,单条 prefill 变慢,排队立刻显形。
- 仅深超载（1.25x/1.5x）反超 +4~5%（8 个引擎排空队列更快）,但两边 TTFT 都已
  35–56s,无实用意义。
- **DP8TP2-32768 不可行**：util 0.80 + async ON 下运行时 NPU OOM（TP2 每卡权重
  翻倍 + HCCL buffer 占用,KV 余量不足;启动估算通过、跑到 79/512 时爆）,
  按用户决定记为容量结论、缺格。
- 16 卡单机最终排序：**AFD token-65536 > AFD request-65536 ≈ baseline DP4TP4-32k
  > baseline 8k（任一拓扑）**,DP8TP2 垫底。

## 4. 数据与复算

- 结果 JSON：`bench_results/dsv4_afd_flash_xnode32/dp4tp2_single_sweeps/`（17 格，
  `base1xas_*` / `dp4tp2tok_*` / `dp4tp2req_*`）+ `dp4tp2_singlenode/`（AFD request
  0.5–1x）+ `baseline1x/`（no-async 旧数据）；NAS 归档 `shwstone/{singlenode_dp4tp2,
  singlenode_dp4tp2_token,baseline1x_async}/`
- 驱动：`tools/itask/{singlenode_dp4tp2_run,singlenode_dp4tp2_token_runs,
  singlenode_dp4tp2_request_over,baseline1x_async_runs}.sh`；1.25x plan 生成法
  = offsets÷1.25 + rate×1.25（`formal_1_fast1p25x_plan.json`）
- 复算：`python3 tools/itask/bucket15_summary.py <json...>`

## 5. 运维记录

- 两个 pod 当日 delete+create 重建（colocate + hostnet），`tools/itask/
  restore_node_env.sh` 一键恢复 CAM vendor/umdk/config.ini/vllm-ascend e19e14da7
  artifacts/插件入口点，~3min/pod，全程无重编译。
- **HCCL_BUFFSIZE 教训**：单机 DP4TP2+FLASHCOMM1+mbt65536 用旧 DP2TP4 配方的
  1024 直接 EngineDeadError，**4096 才行**（已录 playbook 报错签名表）。
  baseline DP8TP2 反向踩坑：EP 域（epWorldSize=16）需要 **279MB**，默认 200MB
  启动即挂（"HCCL_BUFFSIZE_EP is too SMALL ... 279MB"），**512 才行**——
  buffer 需求随拓扑走，别抄任何一组的旧值。
- 双 pod 并行实验纪律：按 pod 分线（base-2=baseline、xnode-2=AFD），单节点
  驱动只清本 pod；换 AFD 拓扑栈前必须清掉对端残留 connector（会回连
  host:1239 污染新栈 rank 握手）。
- 一次 request-split 1.5x 尝试把久跑的服务打挂（EngineDeadError）；新栈复测
  1.5x 正常完成，判为偶发，未复现。
