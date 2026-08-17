#!/usr/bin/env bash
# Baseline DP4xTP8 (FlashComm1/SP) EP32 server for the prefill experiment.
# Docs: docs/npu/PREFILL_PERFORMANCE_EXPERIMENT.md
# Topology: DP4 x TP8 = 32 ranks, expert parallel across all 32 ranks.
#
# Required env:
#   MAX_NUM_BATCHED_TOKENS  one of 8192/16384/32768/49152/65536
#   DP_START_RANK           0 on afd-tp (node0), 2 on afd-tp-2 (node1)
set -euo pipefail

: "${MAX_NUM_BATCHED_TOKENS:?Set MAX_NUM_BATCHED_TOKENS to the group batch-token limit}"
: "${DP_START_RANK:?Set DP_START_RANK to 0 (node0) or 2 (node1)}"

REPO=/a3_inference/itask/workdir/tq02357756/shwstone/code/afd-plugin
MODEL_PATH=/a3_inference/itask/workdir/tq02357756/shwstone/deepseek-v3.2-reduced
DP_ADDRESS="${DP_ADDRESS:-33.182.141.136}"
DP_RPC_PORT=29550

# Load the Ascend runtime environment (non-interactive shells skip .bashrc).
# atb set_env.sh references ZSH_VERSION unbound, so relax -u while sourcing.
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
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_BUFFSIZE=200
export VLLM_ASCEND_ENABLE_MLAPO=1
export VLLM_PLUGINS=ascend,afd
# Mount the /reset_prefix_cache endpoint (and other debug routes) so Stage-2
# repeats can clear the prefix cache without a full server restart.
export VLLM_SERVER_DEV_MODE=1
export GLOO_SOCKET_IFNAME=eth0
export TP_SOCKET_IFNAME=eth0
export HCCL_SOCKET_IFNAME=eth0
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

PREFIX_ARGS=()
if [[ "${VLLM_ENABLE_PREFIX_CACHING:-0}" == "1" ]]; then
  PREFIX_ARGS=(--enable-prefix-caching)
else
  PREFIX_ARGS=(--no-enable-prefix-caching)
fi

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
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --trust-remote-code \
  --gpu-memory-utilization 0.90 \
  "${PREFIX_ARGS[@]}" \
  --enable-expert-parallel \
  --additional-config '{
    "enable_force_load_balance": true
  }' \
  "${HEADLESS_ARGS[@]}"
