#!/usr/bin/env bash
# AFD CAM async FFN server (DP8xTP1, EP8) for the prefill experiment.
# Runs on afd-tp-2 (node1) with devices 8-15 alongside the node1 attention
# server which owns devices 0-7.
# Docs: docs/npu/PREFILL_PERFORMANCE_EXPERIMENT.md
#
# Required env:
#   MAX_NUM_BATCHED_TOKENS  must equal the attention side value
set -euo pipefail

: "${MAX_NUM_BATCHED_TOKENS:?Set MAX_NUM_BATCHED_TOKENS to the group batch-token limit}"

REPO=/a3_inference/itask/workdir/tq02357756/shwstone/code/afd-plugin
MODEL_PATH=/a3_inference/itask/workdir/tq02357756/shwstone/deepseek-v3.2-reduced
AFD_HOST="${AFD_HOST:-33.182.141.136}"
AFD_PORT=1239

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
# FFN never participates in sequence parallelism (FlashComm1 is Attention-local).
export VLLM_ASCEND_ENABLE_FLASHCOMM1=0
export HCCL_OP_EXPANSION_MODE=AIV
export LD_LIBRARY_PATH=/usr/local/Ascend/cann-8.5.1/opp/vendors/CAM/op_api/lib:${LD_LIBRARY_PATH:-}
export HCCL_BUFFSIZE=4096
export AFD_FORCE_BALANCED_TOPK_IDS=1
export VLLM_PLUGINS=ascend,afd
# Mount the /reset_prefix_cache endpoint (and other debug routes) so Stage-2
# repeats can clear the prefix cache without a full server restart.
export VLLM_SERVER_DEV_MODE=1
export GLOO_SOCKET_IFNAME=eth0
export TP_SOCKET_IFNAME=eth0
export HCCL_SOCKET_IFNAME=eth0
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15

PREFIX_ARGS=()
if [[ "${VLLM_ENABLE_PREFIX_CACHING:-0}" == "1" ]]; then
  PREFIX_ARGS=(--enable-prefix-caching)
else
  PREFIX_ARGS=(--no-enable-prefix-caching)
fi

# Avoid filling up container /tmp with prometheus multiprocess files.
export PROMETHEUS_MULTIPROC_DIR="/a3_inference/itask/workdir/tq02357756/shwstone/prometheus_tmp"
mkdir -p "$PROMETHEUS_MULTIPROC_DIR"

# Stage-2 ablation variant applied to async MoE ubatching on BOTH sides:
#   A0 = ubatching off            (async_moe_ubatching=false)
#   A1 = ubatching on, split=request
#   A2 = ubatching on, split=token (default / production)
STAGE2_VARIANT="${STAGE2_VARIANT:-A2}"
case "$STAGE2_VARIANT" in
  A0) MOE_UBATCH=false; MOE_SPLIT=request; MOE_NUBATCH=1 ;;
  A1) MOE_UBATCH=true;  MOE_SPLIT=request; MOE_NUBATCH=2 ;;
  A2) MOE_UBATCH=true;  MOE_SPLIT=token;   MOE_NUBATCH=2 ;;
  *) echo "ERROR: unknown STAGE2_VARIANT=$STAGE2_VARIANT (expect A0/A1/A2)" >&2; exit 1 ;;
esac
echo "[stage2] variant=$STAGE2_VARIANT ubatch=$MOE_UBATCH split=$MOE_SPLIT"

cd "$REPO"
exec vllm serve "$MODEL_PATH" \
  --port 8001 \
  --max-num-seqs 2 \
  --enforce-eager \
  --quantization ascend \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --served-model-name deepseek_v3_2 \
  --tensor-parallel-size 1 \
  --data-parallel-size 8 \
  --enable-expert-parallel \
  "${PREFIX_ARGS[@]}" \
  --trust-remote-code \
  --seed 1024 \
  --additional-config "{
    \"enable_force_load_balance\": true,
    \"afd\": {
      \"connector\": \"CAMAsyncAFDConnector\",
      \"async\": true,
      \"role\": \"ffn\",
      \"host\": \"$AFD_HOST\",
      \"port\": $AFD_PORT,
      \"num_attention_ranks\": 24,
      \"num_ffn_ranks\": 8,
      \"afd_role_rank\": 0,
      \"compute_gate_on_attention\": true,
      \"connector_extra_config\": {
        \"dynamicQuant\": 1,
        \"async_moe_ubatching\": $MOE_UBATCH,
        \"async_moe_num_ubatches\": $MOE_NUBATCH,
        \"async_moe_split\": \"$MOE_SPLIT\",
        \"attn_ranks_per_dp\": 8
      }
    }
  }"
