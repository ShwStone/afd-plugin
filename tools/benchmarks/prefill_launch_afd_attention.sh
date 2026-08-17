#!/usr/bin/env bash
# AFD CAM async attention server (DP3xTP8, FlashComm1/SP) for the prefill
# experiment. Docs: docs/npu/PREFILL_PERFORMANCE_EXPERIMENT.md
#
# Required env:
#   MAX_NUM_BATCHED_TOKENS  must equal the FFN side value
#   DP_START_RANK           0 on afd-tp (node0, 16 devices), 2 on afd-tp-2 (node1, 8 devices)
# Optional env:
#   ATTN_DEVICES            ASCEND_RT_VISIBLE_DEVICES (default all 16; use 0-7 on node1)
set -euo pipefail

: "${MAX_NUM_BATCHED_TOKENS:?Set MAX_NUM_BATCHED_TOKENS to the group batch-token limit}"
: "${DP_START_RANK:?Set DP_START_RANK to 0 (node0) or 2 (node1)}"

REPO=/a3_inference/itask/workdir/tq02357756/shwstone/code/afd-plugin
MODEL_PATH=/a3_inference/itask/workdir/tq02357756/shwstone/deepseek-v3.2-reduced
DP_ADDRESS="${DP_ADDRESS:-33.182.141.136}"
DP_RPC_PORT=29550
AFD_HOST="${AFD_HOST:-33.182.141.136}"
AFD_PORT=1239
ATTN_DEVICES="${ATTN_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"

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
export LD_LIBRARY_PATH=/usr/local/Ascend/cann-8.5.1/opp/vendors/CAM/op_api/lib:${LD_LIBRARY_PATH:-}
export HCCL_BUFFSIZE=4096
export VLLM_ASCEND_ENABLE_MLAPO=1
export AFD_FORCE_BALANCED_TOPK_IDS=1
export VLLM_PLUGINS=ascend,afd
# Mount the /reset_prefix_cache endpoint (and other debug routes) so Stage-2
# repeats can clear the prefix cache without a full server restart.
export VLLM_SERVER_DEV_MODE=1
export GLOO_SOCKET_IFNAME=eth0
export TP_SOCKET_IFNAME=eth0
export HCCL_SOCKET_IFNAME=eth0
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export ASCEND_RT_VISIBLE_DEVICES="$ATTN_DEVICES"

# Avoid filling up container /tmp with prometheus multiprocess files;
# previous crashes left large core dumps there.
export PROMETHEUS_MULTIPROC_DIR="/a3_inference/itask/workdir/tq02357756/shwstone/prometheus_tmp"
mkdir -p "$PROMETHEUS_MULTIPROC_DIR"

PREFIX_ARGS=()
if [[ "${VLLM_ENABLE_PREFIX_CACHING:-0}" == "1" ]]; then
  PREFIX_ARGS=(--enable-prefix-caching)
else
  PREFIX_ARGS=(--no-enable-prefix-caching)
fi

# afd_role_rank is a BASE offset only. The real role rank is derived in
# afd_plugin/v1/worker/attention_model_runner.py:_with_dp_derived_afd_rank:
#   role_rank = afd_role_rank + ((dp_rank * pcp_size + pcp_rank) * tp_size + tp_rank)
# So base=0 on every node. Node1 (dp_rank=2, tp8) already lands on roles 16..23;
# adding another 16 would push it to 32..39 and trip the role_size=24 check.
HEADLESS_ARGS=()
LOCAL_DP=2
if [[ "$DP_START_RANK" != "0" ]]; then
  HEADLESS_ARGS=(--headless)
  LOCAL_DP=1
fi
ROLE_RANK=0

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
  --host 0.0.0.0 \
  --port 8000 \
  --max-num-seqs 32 \
  --enforce-eager \
  --served-model-name deepseek_v3_2 \
  --quantization ascend \
  --max-model-len 70000 \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --tensor-parallel-size 8 \
  --data-parallel-size 3 \
  --data-parallel-size-local "$LOCAL_DP" \
  --data-parallel-start-rank "$DP_START_RANK" \
  --data-parallel-address "$DP_ADDRESS" \
  --data-parallel-rpc-port "$DP_RPC_PORT" \
  "${PREFIX_ARGS[@]}" \
  --gpu-memory-utilization 0.8 \
  --trust-remote-code \
  --seed 1024 \
  --additional-config "{
    \"afd\": {
      \"connector\": \"CAMAsyncAFDConnector\",
      \"async\": true,
      \"role\": \"attention\",
      \"host\": \"$AFD_HOST\",
      \"port\": $AFD_PORT,
      \"num_attention_ranks\": 24,
      \"num_ffn_ranks\": 8,
      \"afd_role_rank\": $ROLE_RANK,
      \"compute_gate_on_attention\": true,
      \"connector_extra_config\": {
        \"dynamicQuant\": 1,
        \"async_moe_ubatching\": $MOE_UBATCH,
        \"async_moe_num_ubatches\": $MOE_NUBATCH,
        \"async_moe_split\": \"$MOE_SPLIT\",
        \"attn_ranks_per_dp\": 8
      }
    }
  }" \
  "${HEADLESS_ARGS[@]}"
