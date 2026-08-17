#!/usr/bin/env bash
# Nightly v0.25.1rc DP4xTP8 baseline test.
# Topology: DP4 x TP8 = 32 ranks, expert parallel across all 32 ranks.
#
# Required env:
#   DP_START_RANK  0 on node0, 2 on node1
set -euo pipefail

: "${DP_START_RANK:?Set DP_START_RANK to 0 (node0) or 2 (node1)}"

REPO=/a3_inference/itask/workdir/tq02357756/afd-plugin/code/afd-plugin
MODEL_PATH=/a3_inference/itask/workdir/tq02357756/shwstone/deepseek-v3.2-reduced
DP_ADDRESS=33.182.142.132   # node0 IP
DP_RPC_PORT=29550

set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# v0.25.1 may have different CANN paths - skip if not found
source /usr/local/Ascend/cann-8.5.1/share/info/ascendnpu-ir/bin/set_env.sh 2>/dev/null || true
source /usr/local/Ascend/nnal/atb/set_env.sh 2>/dev/null || true
set -u
export LD_PRELOAD=/usr/lib64/libjemalloc.so.2:

export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ASCEND_AGGREGATE_ENABLE=1
export ASCEND_TRANSPORT_PRINT=1
export ACL_OP_INIT_MODE=1
export ASCEND_A3_ENABLE=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_BUFFSIZE=200
export VLLM_ASCEND_ENABLE_MLAPO=1
export VLLM_PLUGINS=ascend,afd
export GLOO_SOCKET_IFNAME=eth0
export TP_SOCKET_IFNAME=eth0
export HCCL_SOCKET_IFNAME=eth0
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

HEADLESS_ARGS=()
if [[ "$DP_START_RANK" != "0" ]]; then
  HEADLESS_ARGS=(--headless)
fi

cd "$REPO"
exec vllm serve "$MODEL_PATH" \
  --host 0.0.0.0 \
  --port 8000 \
  --enforce-eager \
  --data-parallel-size 4 \
  --data-parallel-size-local 2 \
  --data-parallel-start-rank "$DP_START_RANK" \
  --data-parallel-address "$DP_ADDRESS" \
  --data-parallel-rpc-port "$DP_RPC_PORT" \
  --tensor-parallel-size 8 \
  --block-size 128 \
  --quantization ascend \
  --seed 1024 \
  --served-model-name deepseek_v3_2 \
  --max-num-seqs 8 \
  --max-model-len 70000 \
  --max-num-batched-tokens 32000 \
  --trust-remote-code \
  --gpu-memory-utilization 0.90 \
  --no-enable-prefix-caching \
  --enable-expert-parallel \
  --additional-config '{
    "enable_force_load_balance": true
  }' \
  "${HEADLESS_ARGS[@]}"