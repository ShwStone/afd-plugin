#!/usr/bin/env bash
# AFD CAMAsyncAFDConnector DeepSeek-V3.2 recipe launcher
# Topology: DP3PCP8 Attention + EP8 FFN, 2 nodes
# Aligned with recipe/npu/CAMAsyncAFDConnector/deepseek_v3_2/README.md
set -euo pipefail

# ==================== 用户只需要改这里 ====================
MASTER_IP="33.182.142.7"                     # 拥有 Attention rank 0 的节点 IP
MODEL_PATH="/a3_inference/itask/workdir/tq02357756/shwstone/deepseek-v3.2-reduced"
REPO="/a3_inference/itask/workdir/tq02357756/shwstone/code/afd-plugin"

DP_RPC_PORT=29550
AFD_PORT=1239
MAX_BATCHED_TOKENS=140000

# 角色：node0-attn / node1-attn / node1-ffn
ROLE="${1:?Usage: $0 <node0-attn|node1-attn|node1-ffn>}"
# =========================================================

# ----- CANN / NNAL / ATB 环境 -----
# vendor set_env.sh may reference unset variables (e.g. ZSH_VERSION)
set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/cann-8.5.1/share/info/ascendnpu-ir/bin/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
set -u

# ----- 网络与性能环境 -----
export LD_PRELOAD=/usr/lib64/libjemalloc.so.2:
export GLOO_SOCKET_IFNAME=eth0
export TP_SOCKET_IFNAME=eth0
export HCCL_SOCKET_IFNAME=eth0
LOCAL_IP=$(ifconfig "${HCCL_SOCKET_IFNAME}" 2>/dev/null | awk '/inet / {print $2}')
export HCCL_IF_IP="${LOCAL_IP:?Could not determine HCCL_IF_IP from ${HCCL_SOCKET_IFNAME}}"

export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ASCEND_AGGREGATE_ENABLE=1
export ASCEND_TRANSPORT_PRINT=1
export ACL_OP_INIT_MODE=1
export ASCEND_A3_ENABLE=1
export VLLM_ASCEND_ENABLE_CONTEXT_PARALLEL=1
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_BUFFSIZE=4096
export VLLM_ASCEND_ENABLE_MLAPO=1
export AFD_FORCE_BALANCED_TOPK_IDS=1

# ----- AFD 插件加载 -----
export VLLM_PLUGINS=ascend,afd
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export LD_LIBRARY_PATH=/usr/local/Ascend/cann-8.5.1/opp/vendors/CAM/op_api/lib:${LD_LIBRARY_PATH:-}

# ----- 多角色联合启动需要更长的 ready 等待 -----
export VLLM_ENGINE_READY_TIMEOUT_S=1800

# 避免 prometheus 临时文件撑满 /tmp
export PROMETHEUS_MULTIPROC_DIR="/a3_inference/itask/workdir/tq02357756/shwstone/prometheus_tmp"
mkdir -p "$PROMETHEUS_MULTIPROC_DIR"

cd "$REPO"

# 检查模型目录是否存在基本文件
if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
  echo "ERROR: ${MODEL_PATH}/config.json not found" >&2
  exit 1
fi

# ----- 生成 AFD 配置 -----
# 注意：Attention 的 afd_role_rank 不是 markdown 里写的 16，而是 0
make_afd_config() {
  local role="$1" rank="$2"
  python3 - <<PY
import json
role = "${role}"
rank = ${rank}
host = "${MASTER_IP}"
port = ${AFD_PORT}
cfg = {
  "afd": {
    "connector": "CAMAsyncAFDConnector",
    "async": True,
    "role": role,
    "host": host,
    "port": port,
    "num_attention_ranks": 24,
    "num_ffn_ranks": 8,
    "afd_role_rank": rank,
    "compute_gate_on_attention": True,
    "connector_extra_config": {
      "dynamicQuant": 1,
      "async_moe_ubatching": True,
      "async_moe_num_ubatches": 2,
      "async_moe_split": "request",
      "attn_ranks_per_dp": 8
    }
  }
}
print(json.dumps(cfg, separators=(",", ":")))
PY
}

# ----- 公共基础参数 -----
BASE_ARGS=(
  --served-model-name deepseek_v3_2
  --quantization ascend
  --max-model-len 70000
  --enforce-eager
  --trust-remote-code
)

# ----- 按角色启动 -----
case "$ROLE" in
  node0-attn)
    AFD_CONFIG=$(make_afd_config attention 0)
    LOG=/tmp/afd_node0_attention.log
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
    setsid vllm serve "$MODEL_PATH" \
      --host 0.0.0.0 \
      --port 8000 \
      "${BASE_ARGS[@]}" \
      --max-num-seqs 32 \
      --max-num-batched-tokens "$MAX_BATCHED_TOKENS" \
      --data-parallel-size 3 \
      --data-parallel-size-local 2 \
      --data-parallel-start-rank 0 \
      --data-parallel-address "$MASTER_IP" \
      --data-parallel-rpc-port "$DP_RPC_PORT" \
      --prefill-context-parallel-size 8 \
      --no-enable-chunked-prefill \
      --enable-expert-parallel \
      --no-enable-prefix-caching \
      --gpu-memory-utilization 0.8 \
      --additional-config "$AFD_CONFIG" \
      > "$LOG" 2>&1 < /dev/null &
    echo "Node0 Attention started, pid $!, log: $LOG"
    ;;

  node1-attn)
    AFD_CONFIG=$(make_afd_config attention 0)   # <-- 修正点：不是 16
    LOG=/tmp/afd_node1_attention.log
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    setsid vllm serve "$MODEL_PATH" \
      --host 0.0.0.0 \
      --port 8000 \
      "${BASE_ARGS[@]}" \
      --max-num-seqs 32 \
      --max-num-batched-tokens "$MAX_BATCHED_TOKENS" \
      --data-parallel-size 3 \
      --data-parallel-size-local 1 \
      --data-parallel-start-rank 2 \
      --data-parallel-address "$MASTER_IP" \
      --data-parallel-rpc-port "$DP_RPC_PORT" \
      --prefill-context-parallel-size 8 \
      --no-enable-chunked-prefill \
      --enable-expert-parallel \
      --no-enable-prefix-caching \
      --gpu-memory-utilization 0.8 \
      --headless \
      --additional-config "$AFD_CONFIG" \
      > "$LOG" 2>&1 < /dev/null &
    echo "Node1 Attention started, pid $!, log: $LOG"
    ;;

  node1-ffn)
    AFD_CONFIG=$(make_afd_config ffn 0)
    LOG=/tmp/afd_node1_ffn.log
    export ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15
    setsid vllm serve "$MODEL_PATH" \
      --port 8001 \
      "${BASE_ARGS[@]}" \
      --max-num-seqs 2 \
      --max-num-batched-tokens "$MAX_BATCHED_TOKENS" \
      --data-parallel-size 8 \
      --data-parallel-size-local 8 \
      --enable-expert-parallel \
      --additional-config "$AFD_CONFIG" \
      > "$LOG" 2>&1 < /dev/null &
    echo "Node1 FFN started, pid $!, log: $LOG"
    ;;

  *)
    echo "Unknown role: $ROLE" >&2
    exit 1
    ;;
esac
