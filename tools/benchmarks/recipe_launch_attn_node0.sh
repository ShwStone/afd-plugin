#!/usr/bin/env bash
# Recipe DP3PCP8 + EP8: Node0 Attention (DP0-DP1, 16 devices, non-headless).
# Mirrors recipe/npu/CAMAsyncAFDConnector/deepseek_v3_2/README.md.
# Uses the reduced DeepSeek-V3.2 model; no afd_role_rank (removed in cb372c3).
set -euo pipefail

REPO=/a3_inference/itask/workdir/tq02357756/shwstone/code/afd-plugin
MODEL_PATH=/a3_inference/itask/workdir/tq02357756/shwstone/deepseek-v3.2-reduced
MASTER_IP="${MASTER_IP:-33.182.141.136}"   # node0 = afd-exp-1
AFD_PORT=1239
DP_RPC_PORT=29550

set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/cann-8.5.1/share/info/ascendnpu-ir/bin/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
set -u
export LD_PRELOAD=/usr/lib64/libjemalloc.so.2:

export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ASCEND_AGGREGATE_ENABLE=1
export ASCEND_TRANSPORT_PRINT=1
export ACL_OP_INIT_MODE=1
export ASCEND_A3_ENABLE=1
export VLLM_ASCEND_ENABLE_CONTEXT_PARALLEL=1
export HCCL_OP_EXPANSION_MODE=AIV
export LD_LIBRARY_PATH=/usr/local/Ascend/cann-8.5.1/opp/vendors/CAM/op_api/lib:${LD_LIBRARY_PATH:-}
export HCCL_BUFFSIZE=4096
export AFD_FORCE_BALANCED_TOPK_IDS=1
export VLLM_PLUGINS=ascend,afd
export GLOO_SOCKET_IFNAME=eth0
export TP_SOCKET_IFNAME=eth0
export HCCL_SOCKET_IFNAME=eth0
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export VLLM_ENGINE_READY_TIMEOUT_S=2400
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15

export PROMETHEUS_MULTIPROC_DIR="/a3_inference/itask/workdir/tq02357756/shwstone/prometheus_tmp"
mkdir -p "$PROMETHEUS_MULTIPROC_DIR"

cd "$REPO"
exec vllm serve "$MODEL_PATH" \
  --host 0.0.0.0 \
  --port 8000 \
  --max-num-seqs 32 \
  --enforce-eager \
  --served-model-name deepseek_v3_2 \
  --quantization ascend \
  --max-model-len 70000 \
  --max-num-batched-tokens 140000 \
  --tensor-parallel-size 1 \
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
  --trust-remote-code \
  --additional-config "{
    \"afd\": {
      \"connector\": \"CAMAsyncAFDConnector\",
      \"async\": true,
      \"role\": \"attention\",
      \"host\": \"$MASTER_IP\",
      \"port\": $AFD_PORT,
      \"num_attention_ranks\": 24,
      \"num_ffn_ranks\": 8,
      \"compute_gate_on_attention\": true,
      \"connector_extra_config\": {
        \"dynamicQuant\": 1,
        \"async_moe_ubatching\": true,
        \"async_moe_num_ubatches\": 2,
        \"async_moe_split\": \"request\",
        \"attn_ranks_per_dp\": 8
      }
    }
  }"
