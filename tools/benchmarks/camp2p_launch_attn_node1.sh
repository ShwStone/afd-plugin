#!/usr/bin/env bash
# Multi-node CAMP2p: Node1 Attention (DP8-15, devices 0-7, headless).
# Topology: 16 attention ranks (8 per node) + 8 FFN ranks on node1.
set -euo pipefail

REPO=/a3_inference/itask/workdir/tq02357756/shwstone/code/afd-plugin
MODEL_PATH=/a3_inference/itask/workdir/tq02357756/shwstone/deepseek-v3.2-reduced
LOCAL_IP=33.182.140.225      # this node (afd-exp-2)
DP_ADDRESS=33.182.141.136    # attention node hosting DP rank 0
AFD_HOST=33.182.140.225      # FFN node (afd-exp-2) hosting FFN role rank 0
AFD_PORT=29666
DP_RPC_PORT=29550

set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/cann-8.5.1/share/info/ascendnpu-ir/bin/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
set -u
export LD_PRELOAD=/usr/lib64/libjemalloc.so.2:

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HCCL_BUFFSIZE=512
export HCCL_OP_EXPANSION_MODE=AIV
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export VLLM_ASCEND_ENABLE_MLAPO=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ASCEND_GLOBAL_LOG_LEVEL=3
export VLLM_PLUGINS=ascend,afd
export HCCL_IF_IP="$LOCAL_IP"
export GLOO_SOCKET_IFNAME=eth0
export TP_SOCKET_IFNAME=eth0
export HCCL_SOCKET_IFNAME=eth0
export VLLM_ENGINE_READY_TIMEOUT_S=2400
export PROMETHEUS_MULTIPROC_DIR="/a3_inference/itask/workdir/tq02357756/shwstone/prometheus_tmp"
mkdir -p "$PROMETHEUS_MULTIPROC_DIR"

cd "$REPO"
exec env VLLM_USE_V1=1 vllm serve "$MODEL_PATH" \
  --host 0.0.0.0 \
  --port 8000 \
  --data-parallel-size 16 \
  --data-parallel-size-local 8 \
  --data-parallel-start-rank 8 \
  --data-parallel-address "$DP_ADDRESS" \
  --data-parallel-rpc-port "$DP_RPC_PORT" \
  --tensor-parallel-size 1 \
  --enable-expert-parallel \
  --max-num-batched-tokens 16 \
  --max-num-seqs 16 \
  --seed 1024 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.9 \
  --headless \
  --served-model-name deepseek_v3_2 \
  --trust-remote-code \
  --no-enable-prefix-caching \
  --quantization ascend \
  --tokenizer-mode deepseek_v32 \
  --reasoning-parser deepseek_v3 \
  --additional-config '{
    "enable_force_load_balance": true,
    "force_load_balance_topn_per_rank": 4,
    "afd": {
      "role": "attention",
      "connector": "CAMP2pAFDConnector",
      "host": "'"$AFD_HOST"'",
      "port": '"$AFD_PORT"',
      "num_attention_ranks": 16,
      "num_ffn_ranks": 8,
      "connector_extra_config": {
        "attn_core_num": 8,
        "quant_mode": 0
      }
    }
  }'
