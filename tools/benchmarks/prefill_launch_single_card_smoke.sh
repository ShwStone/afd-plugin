#!/usr/bin/env bash
# Single-card smoke test for the reduced DeepSeek-V3.2 model.
# Use this to verify that one NPU can load the model and serve /v1/models
# without the multi-node DP/HCCL complexity.
#
# Required env:
#   MAX_NUM_BATCHED_TOKENS  e.g. 8192
set -euo pipefail

: "${MAX_NUM_BATCHED_TOKENS:?Set MAX_NUM_BATCHED_TOKENS to the batch-token limit}"

REPO=/a3_inference/itask/workdir/tq02357756/shwstone/code/afd-plugin
MODEL_PATH=/a3_inference/itask/workdir/tq02357756/shwstone/deepseek-v3.2-reduced

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
# Single-card smoke: turn off FlashComm1/SP and MLAPO to keep it minimal.
export VLLM_ASCEND_ENABLE_FLASHCOMM1=0
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_BUFFSIZE=200
export VLLM_ASCEND_ENABLE_MLAPO=0
export VLLM_PLUGINS=ascend,afd
export GLOO_SOCKET_IFNAME=eth0
export TP_SOCKET_IFNAME=eth0
export HCCL_SOCKET_IFNAME=eth0
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
# Pin to the first NPU only.
export ASCEND_RT_VISIBLE_DEVICES=0

cd "$REPO"
exec vllm serve "$MODEL_PATH" \
  --host 0.0.0.0 \
  --port 8000 \
  --enforce-eager \
  --tensor-parallel-size 1 \
  --data-parallel-size 1 \
  --block-size 128 \
  --quantization ascend \
  --seed 1024 \
  --served-model-name deepseek_v3_2 \
  --max-num-seqs 1 \
  --max-model-len 70000 \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --trust-remote-code \
  --gpu-memory-utilization 0.85 \
  --no-enable-prefix-caching
