#!/usr/bin/env bash
# Recipe DP3PCP8 + EP8: Node1 FFN (DP8=EP8, devices 8-15).
# Mirrors recipe/npu/CAMAsyncAFDConnector/deepseek_v3_2/README.md.
set -euo pipefail

REPO=/a3_inference/itask/workdir/tq02357756/shwstone/code/afd-plugin
MODEL_PATH=/a3_inference/itask/workdir/tq02357756/shwstone/deepseek-v3.2-reduced
MASTER_IP="${MASTER_IP:-33.182.141.136}"   # node0 = afd-exp-1
AFD_PORT=1239

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
export VLLM_ASCEND_ENABLE_FLASHCOMM1=0
export AFD_FORCE_BALANCED_TOPK_IDS=1
export VLLM_PLUGINS=ascend,afd
export GLOO_SOCKET_IFNAME=eth0
export TP_SOCKET_IFNAME=eth0
export HCCL_SOCKET_IFNAME=eth0
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export VLLM_ENGINE_READY_TIMEOUT_S=2400
export ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15

export PROMETHEUS_MULTIPROC_DIR="/a3_inference/itask/workdir/tq02357756/shwstone/prometheus_tmp"
mkdir -p "$PROMETHEUS_MULTIPROC_DIR"

cd "$REPO"
exec vllm serve "$MODEL_PATH" \
  --port 8001 \
  --max-num-seqs 2 \
  --enforce-eager \
  --quantization ascend \
  --max-num-batched-tokens 140000 \
  --served-model-name deepseek_v3_2 \
  --tensor-parallel-size 1 \
  --data-parallel-size 8 \
  --enable-expert-parallel \
  --no-enable-prefix-caching \
  --trust-remote-code \
  --additional-config "{
    \"enable_force_load_balance\": true,
    \"afd\": {
      \"connector\": \"CAMAsyncAFDConnector\",
      \"async\": true,
      \"role\": \"ffn\",
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
