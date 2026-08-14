# CAM Async 算子调用参数（DeepSeek-V2-Lite e2e 实测）

2026-08-14，afd-v2lite-e2e pod，`afd-eager-async-cam` 场景，`AFD_CAM_OP_IO_LOG=1` + `ASCEND_LAUNCH_BLOCKING=1` 实测抓取。FFN 在首次 `async_dispatch_recv` 崩 507015（DDR OOB），本包为崩溃现场的完整入参。

## 0. 有没有"全局初始化调用"？——没有

`torch.ops.umdk_cam_op_lib` 命名空间注册的**全部**算子（torch schema 实测）：

```
async_dispatch_send / async_dispatch_recv / async_combine_send / async_combine_recv   ← async 连接器用的 4 个
moe_dispatch_prefill / moe_combine_prefill / get_dispatch_layout / fused_deep_moe     ← DeepEP 式同步算子，async 连接器未使用
```

**没有任何 init/comm_init 类算子**。我们的"初始化"= HCCL 建组：

1. `init_afd_process_group(backend="hccl", init_method="tcp://127.0.0.1:6453", world_size=4, rank=<world_rank>, group_name="afd_async_cam", timeout=30min, pg_options=ProcessGroupHCCL.Options{hccl_config={"hccl_buffer_size": N_MB}})`——rendezvous 后 torch_npu 的 ProcessGroupHCCL 创建时内部做 `hcclCommInitRootInfo`，**buffer 池在这一步按 hccl_buffer_size 分配**
2. `backend.get_hccl_comm_name(world_rank)` 拿到 comm 名字符串 → 作为 `group_name` 传给每次算子调用
3. `comm_args = torch.empty((1,), fp16, npu)` 占位 tensor + `comm_id = 0`，逐次调用传递

算子本身无状态（每次调用带全量拓扑参数），算子库内部若有 per-comm workspace，应是按 group_name/comm_id 惰性建立。

## 1. 拓扑与初始化参数

| 项 | 值 |
|---|---|
| 模型 | DeepSeek-V2-Lite（hidden 2048，64 routed experts，topk 6，2 shared，27 层） |
| Attention | DP1 TP2（NPU 0,1），`attn_ranks_per_dp=2` |
| FFN | DP2 TP1（NPU 2,3），每 rank 32 experts |
| Process group | backend=`hccl`，init_method=`tcp://127.0.0.1:6453`，world_size=**4**，group_name(PG)=`afd_async_cam`，timeout=30min |
| world rank 映射 | attn r0→world0(NPU0)，attn r1→world1(NPU1)，ffn r0→world2(NPU2)，ffn r1→world3(NPU3) |
| pg_options | `ProcessGroupHCCL.Options`，`hccl_config={"hccl_buffer_size": 825}`（MB；本次实验值，原推导 104、旧 env 行为 4096） |
| comm_args | `torch.empty((1,), dtype=float16, device=npu:local)`（占位 tensor） |
| comm_id | `0`（CAM_COMM_ID 常量） |
| group_name（算子入参） | 运行时 `backend.get_hccl_comm_name(world_rank)` 返回的 HCCL comm 名字符串（日志中被跳过未打） |
| 相关环境变量 | `HCCL_BUFFSIZE=4096`、`HCCL_OP_EXPANSION_MODE=AIV`、`ASCEND_LAUNCH_BLOCKING=1`、FlashComm1=1、ContextParallel=1 |

## 2. 四个算子接口签名与实测入参

官方 torch schema（`torch._C._jit_get_all_schemas()` 实测，参数名以 schema 为准）：

```
async_dispatch_send(Tensor x, Tensor expert_ids, Tensor comm_args, int comm_id, int max_seq_len, int batch_size, int hidden_size, int top_k, int expert_rank_size, int attention_rank_size, int expert_per_rank, int rank, int world_size, int layer_index, int tp_size, int dynamic_quant, str group_name) -> Tensor
async_dispatch_recv(Tensor x, Tensor comm_args, int comm_id, int batch_size, int hidden_size, int top_k, int expert_rank_size, int attention_rank_size, int expert_per_rank, int rank, int world_size, int tp_size, int dynamic_quant, str group_name) -> Tensor[]
async_combine_send(Tensor expand_x, Tensor expand_x_shared, Tensor comm_args, Tensor expert_token_nums, int comm_id, int batch_size, int hidden_size, int top_k, int expert_rank_size, int attention_rank_size, int expert_per_rank, int rank, int world_size, int tp_size, str group_name) -> Tensor
async_combine_recv(Tensor expand_x, Tensor expert_ids, Tensor expert_scales, Tensor comm_args, int comm_id, int batch_size, int hidden_size, int top_k, int expert_rank_size, int attention_rank_size, int expert_per_rank, int rank, int world_size, str group_name) -> Tensor
```

（我们代码里的 `ffn_size`=schema 的 `expert_rank_size`，`attn_size`=`attention_rank_size`，`num_npus_per_dp_group`=`tp_size`。）

### async_dispatch_send（attention 侧，每层 MoE 调用）

```
(hidden_states, topk_ids, comm_args, comm_id, max_seq_len, batch_size,
 hidden_size, topk, ffn_size, attn_size, expert_per_rank, rank, world_size,
 layer_idx, num_npus_per_dp_group, dynamic_quant, group_name)
```

实测（layer_idx=1，prefill 首请求）：

| rank | hidden_states | topk_ids | max_seq_len | batch_size | 其余 |
|---|---|---|---|---|---|
| world0 (attn TP0) | bf16 (153, 2048) | int32 (153, 6) | 8000 | **153** | hidden 2048, topk 6, ffn 2, attn 2, epr 32, world 4, npus/dp 2, dq 0 |
| world1 (attn TP1) | bf16 (153, 2048) | int32 (153, 6) | 8000 | **153** | 同上 |

⚠️ 注意：两个 TP rank 的 batch_size 都是 153（非对半 shard）。

### async_dispatch_recv（FFN 侧，**崩溃发生在这个调用里**）

```
(placeholder, comm_args, comm_id, batch_size, hidden_size, topk,
 ffn_size, attn_size, expert_per_rank, rank, world_size,
 num_npus_per_dp_group, dynamic_quant, group_name)
```

实测（两个 FFN rank 相同，除 rank）：

| 参数 | 值 |
|---|---|
| placeholder | bf16 (1,) |
| batch_size | **8000**（=max_seq_len，按容量 post） |
| hidden_size | 2048 |
| topk | 6 |
| ffn_size / attn_size | 2 / 2 |
| expert_per_rank | 32 |
| rank | 2（ffn r0）/ 3（ffn r1） |
| world_size | 4 |
| num_npus_per_dp_group | 2 |
| dynamic_quant | 0 |

预期输出（7 元组，崩溃未返回）：`hidden_states, expand_x_shared, dynamic_scales, dynamic_scales_shared, token_nums_rankid_layeridx, expert_token_nums, expert_token_nums_shared`

### async_combine_recv（attention 侧）

```
(placeholder, topk_ids, topk_weights, comm_args, comm_id, batch_size,
 hidden_size, topk, ffn_size, attn_size, expert_per_rank, rank, world_size,
 group_name)
```

实测：placeholder bf16 (1,)，topk_ids int32 (153,6)，topk_weights fp32 (153,6)，batch_size=153，其余同上（无 layer_idx / num_npus_per_dp_group / dynamic_quant）。

### async_combine_send（FFN 侧，本次未到达）

```
(ffn_output, expand_x_shared, comm_args, token_nums_rankid_layeridx,
 comm_id, batch_size, hidden_size, topk, ffn_size, attn_size,
 expert_per_rank, rank, world_size, num_npus_per_dp_group, group_name)
```

## 3. 崩溃现场时间线

```
08:03:24  FFN world2/world3 post async_dispatch_recv（batch_size=8000，挂起等待）
08:03:39  首请求到达，attention world0/world1 async_dispatch_send（batch_size=153, layer_idx=1）
08:03:39  attention post async_combine_recv
~08:03:4x FFN world3（另一次为 world2）崩 507015：DDR address of the MTE instruction is out of range
```

## 4. 已做的 buffer 大小二分（hccl_buffer_size，MB）

| 104 | 207 | 512 | 768 | **825** | 1024 | 4096 |
|---|---|---|---|---|---|---|
| 崩 | 崩 | 崩 | 崩 | 崩 | **过** | 过 |

"过" = completion 返回正确答案。真实门槛在 (825, 1024] MB 之间，远超数据量公式推导值（104MB），疑似算子按 max_seq_len=8000 容量做固定布局编址。
