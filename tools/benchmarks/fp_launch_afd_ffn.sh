#!/usr/bin/env bash
# Full-model (61-layer DeepSeek-V3.2 W8A8) AFD FFN side for the full-prefill
# performance experiment.
# Docs: docs/npu/DEEPSEEK_V3_2_FULL_PREFILL_PERFORMANCE_PLAN.zh-CN.md
# Topology: 16 FFN ranks = DP16 x TP1 EP16 on one node (16 NPUs),
# FlashComm1 off, CAMAsyncAFDConnector, eager, prefix caching off.
#
# Required env:
#   AFD_HOST                attention node IP used for CAM rendezvous
# Optional env:
#   AFD_VARIANT             A0 | A1 | A2 (default A2, token split)
#   FFN_MAX_NUM_BATCHED_TOKENS
#                           default 65536 — MUST match the attention common
#                           batch cap (CAM connector buffer sizing); do not
#                           lower (rendezvous hang otherwise)
#   FFN_MAX_MODEL_LEN       default 8192 (recipe value; FFN has no real
#                           sequences — work is connector-driven)
#   gpu-memory-utilization  0.85: 61-layer expert weights (~43 GB/rank) plus
#                           a 64K-token profile do not fit 0.8; 0.85 leaves
#                           ~9 GB outside the pool for the HCCL workspace
#   LOCAL_IP                default: auto-detected from NIC_NAME
#   NIC_NAME                default eth0
#   MAX_NUM_SEQS            default 8
set -euo pipefail

: "${AFD_HOST:?Set AFD_HOST to the attention node IP used for CAM rendezvous}"

REPO=/a3_inference/itask/workdir/tq02357756/shwstone/code/afd-plugin
MODEL_PATH="${MODEL_PATH:-/home/admin/model-csi/model}"
AFD_PORT="${AFD_PORT:-1239}"
FFN_PORT="${FFN_PORT:-8001}"
NIC_NAME="${NIC_NAME:-eth0}"
LOCAL_IP="${LOCAL_IP:-$(ifconfig "$NIC_NAME" | grep -oP 'inet \K[0-9.]+')}"
MAX_MODEL_LEN="${FFN_MAX_MODEL_LEN:-8192}"
# The CAM connector sizes its operator buffers from
# scheduler_config.max_num_batched_tokens (async_cam.py), so the FFN value
# must match the attention side's common batch cap even though the FFN
# scheduler itself is vestigial. Do NOT lower this to save memory — a
# mismatch with the attention dispatch stage size hangs the rendezvous.
MAX_NUM_BATCHED_TOKENS="${FFN_MAX_NUM_BATCHED_TOKENS:-65536}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
AFD_VARIANT="${AFD_VARIANT:-A2}"

# A0: no async two-stage pipeline. A1: split at request boundary.
# A2: split by token count (production candidate).
case "$AFD_VARIANT" in
  A0) ASYNC_MOE_UBATCHING=false ; ASYNC_MOE_SPLIT="request" ; ASYNC_MOE_NUM_UBATCHES=1 ;;
  A1) ASYNC_MOE_UBATCHING=true  ; ASYNC_MOE_SPLIT="request" ; ASYNC_MOE_NUM_UBATCHES=2 ;;
  A2) ASYNC_MOE_UBATCHING=true  ; ASYNC_MOE_SPLIT="token"   ; ASYNC_MOE_NUM_UBATCHES=2 ;;
  *) echo "Unknown AFD_VARIANT: $AFD_VARIANT (expected A0|A1|A2)" >&2; exit 2 ;;
esac

set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/cann-9.0.1/share/info/ascendnpu-ir/bin/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
set -u
export LD_PRELOAD=/usr/lib64/libjemalloc.so.2:
export LD_LIBRARY_PATH=/usr/local/Ascend/cann-9.0.1/opp/vendors/CAM/op_api/lib:${LD_LIBRARY_PATH:-}
export ASCEND_CUSTOM_OPP_PATH=/usr/local/Ascend/cann-9.0.1/opp/vendors/CAM:${ASCEND_CUSTOM_OPP_PATH:-}

export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ASCEND_ENABLE_FLASHCOMM1=0
export VLLM_ASCEND_ENABLE_MLAPO=1
export HCCL_OP_EXPANSION_MODE=AIV
# 2048 (not the recipe's 4096): the 4096 HCCL pool (~8.6 GB/device) plus
# ~43.4 GB expert weights plus MoE workspace OOMs the FFN at util 0.85.
# CAM stages are ~235 MB, far below a 2 GB pool, so collectives are
# unaffected. Frozen for the whole experiment.
export HCCL_BUFFSIZE=2048
export HCCL_CONNECT_TIMEOUT=3600
# Engine-core boot can exceed the 600s default while the 646 GiB
# checkpoint loads from cold cache; give it room instead of dying
# mid-boot (observed 2026-08-21 A2).
export VLLM_ENGINE_READY_TIMEOUT_S=1800
export VLLM_PLUGINS=ascend,afd
export VLLM_SERVER_DEV_MODE=1
export AFD_FORCE_BALANCED_TOPK_IDS=0
export HCCL_IF_IP="$LOCAL_IP"
export GLOO_SOCKET_IFNAME="$NIC_NAME"
export TP_SOCKET_IFNAME="$NIC_NAME"
export HCCL_SOCKET_IFNAME="$NIC_NAME"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

ADDITIONAL_CONFIG="$(
  printf '%s' "{
    \"enable_force_load_balance\": false,
    \"enable_prefill_mc2\": false,
    \"afd\": {
      \"role\": \"ffn\",
      \"connector\": \"CAMAsyncAFDConnector\",
      \"async\": true,
      \"host\": \"$AFD_HOST\",
      \"port\": $AFD_PORT,
      \"num_attention_ranks\": 16,
      \"num_ffn_ranks\": 16,
      \"compute_gate_on_attention\": true,
      \"connector_extra_config\": {
        \"dynamicQuant\": 1,
        \"attn_ranks_per_dp\": 8,
        \"async_moe_ubatching\": $ASYNC_MOE_UBATCHING,
        \"async_moe_num_ubatches\": $ASYNC_MOE_NUM_UBATCHES,
        \"async_moe_split\": \"$ASYNC_MOE_SPLIT\"
      }
    }
  }"
)"

cd "$REPO"
exec env VLLM_USE_V1=1 vllm serve "$MODEL_PATH" \
  --host 0.0.0.0 \
  --port "$FFN_PORT" \
  --served-model-name deepseek_v3_2 \
  --data-parallel-size 16 \
  --tensor-parallel-size 1 \
  --enable-expert-parallel \
  --enforce-eager \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --seed 1024 \
  --gpu-memory-utilization 0.85 \
  --trust-remote-code \
  --no-enable-prefix-caching \
  --quantization ascend \
  --tokenizer-mode deepseek_v32 \
  --additional-config "$ADDITIONAL_CONFIG"
