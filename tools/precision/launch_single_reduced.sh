#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
# Launch BOTH Async CAM roles on ONE node using the reduced-layer model,
# optionally with precision capture enabled for the layer-comparison experiment.
#
#   Attention: DP1 x TP8  (cards 0-7, FlashComm1/SP on), API port 8000
#   FFN:       DP8 x EP8  (cards 8-15, FlashComm1 off),   API port 8001
#   AFD rendezvous on 127.0.0.1:1239 (single node => no cross-node HCCL).
#
# Usage (run on the node):
#   MODE=no_ubatch|token|request RUN_ID=<shared-id> \
#   bash tools/precision/launch_single_reduced.sh
#
# Capture (CAPTURE=1 by default) writes per-mode tensors to
#   bench_results/precision/<RUN_ID>/<MODE>/
# so two modes can be compared with compare_async_moe_captures.
set -u

REPO="${REPO:-/a3_inference/itask/workdir/tq02357756/shwstone/code/afd-plugin}"
MODEL="${MODEL:-/a3_inference/itask/workdir/tq02357756/shwstone/deepseek-v3.2-reduced}"
AFD_HOST="${AFD_HOST:-127.0.0.1}"
AFD_PORT="${AFD_PORT:-1239}"
MODE="${MODE:-token}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
CAPTURE="${CAPTURE:-1}"
RUN_ID="${RUN_ID:-single-precision}"
LAYERS="${LAYERS:-3,4,5}"
FULL_TENSORS="${FULL_TENSORS:-1}"
SYNC="${SYNC:-0}"

case "$MODE" in
  no_ubatch) ASYNC_MOE_UBATCHING=false; ASYNC_MOE_SPLIT=request ;;
  request)   ASYNC_MOE_UBATCHING=true;  ASYNC_MOE_SPLIT=request ;;
  token)     ASYNC_MOE_UBATCHING=true;  ASYNC_MOE_SPLIT=token ;;
  *) echo "unknown MODE=$MODE (use no_ubatch|request|token)" >&2; exit 2 ;;
esac

cd "$REPO" || exit 2
CAPTURE_DIR="$REPO/bench_results/precision"
mkdir -p "$CAPTURE_DIR/$RUN_ID/$MODE"

# CAM vendor: register custom OPP + shadow libopapi.so (recipe scripts do this too;
# without it the CAM ops load a CANN-default libopapi.so lacking the async symbols).
CAM_VENDOR="/usr/local/Ascend/cann-9.0.1/opp/vendors/CAM"
if [ -d "$CAM_VENDOR" ]; then
  export ASCEND_CUSTOM_OPP_PATH="$CAM_VENDOR:${ASCEND_CUSTOM_OPP_PATH:-}"
  export LD_LIBRARY_PATH="$CAM_VENDOR/op_api/lib:$CAM_VENDOR/op_api:${LD_LIBRARY_PATH:-}"
fi

# Stop any running servers / orphan workers on this node.
pkill -f '[v]llm serve' 2>/dev/null || true
ps -eo pid,comm | awk '$2 ~ /^VLLM::/ {print $1}' | xargs -r kill -9 2>/dev/null || true
sleep 6
rm -f /dev/shm/vllm_* 2>/dev/null || true

# Precision capture env (inherited by both roles).
if [ "$CAPTURE" = "1" ]; then
  export AFD_ASYNC_MOE_PRECISION_DEBUG=1
  export AFD_ASYNC_MOE_PRECISION_DEBUG_DIR="$CAPTURE_DIR"
  export AFD_ASYNC_MOE_PRECISION_DEBUG_RUN_ID="$RUN_ID"
  export AFD_ASYNC_MOE_PRECISION_DEBUG_FIXTURE_ID="${FIXTURE_ID:-fixture-single}"
  export AFD_ASYNC_MOE_PRECISION_DEBUG_MODE="$MODE"
  export AFD_ASYNC_MOE_PRECISION_DEBUG_LAYERS="$LAYERS"
  export AFD_ASYNC_MOE_PRECISION_DEBUG_FULL_TENSORS="$FULL_TENSORS"
  export AFD_ASYNC_MOE_PRECISION_DEBUG_SYNC="$SYNC"
  export AFD_ASYNC_MOE_PRECISION_DEBUG_NO_OVERLAP="${NO_OVERLAP:-0}"
else
  export AFD_ASYNC_MOE_PRECISION_DEBUG=0
fi

EXTRA_CONFIG="{\"enable_force_load_balance\": false, \"afd\": {\"role\": \"ROLE\", \"connector\": \"CAMAsyncAFDConnector\", \"async\": true, \"host\": \"$AFD_HOST\", \"port\": $AFD_PORT, \"num_attention_ranks\": 8, \"num_ffn_ranks\": 8, \"compute_gate_on_attention\": true, \"connector_extra_config\": {\"dynamicQuant\": 1, \"attn_ranks_per_dp\": 8, \"async_moe_ubatching\": $ASYNC_MOE_UBATCHING, \"async_moe_num_ubatches\": 2, \"async_moe_split\": \"$ASYNC_MOE_SPLIT\"}}}"

COMMON=(
  --host 0.0.0.0 --served-model-name deepseek_v3_2
  --enable-expert-parallel --enforce-eager
  --max-model-len 8192 --max-num-batched-tokens 8192 --max-num-seqs 8
  --seed 1024 --gpu-memory-utilization "$GPU_MEM_UTIL" --trust-remote-code
  --no-enable-prefix-caching --quantization ascend --tokenizer-mode deepseek_v32
)

# --- FFN first (DP8 x EP8, cards 8-15) ---
env ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15 \
  VLLM_PLUGINS=ascend,afd PYTHONUNBUFFERED=1 VLLM_ENGINE_READY_TIMEOUT_S=3600 \
  VLLM_ASCEND_ENABLE_FLASHCOMM1=0 \
  nohup vllm serve "$MODEL" "${COMMON[@]}" --port 8001 \
  --data-parallel-size 8 --tensor-parallel-size 1 \
  --additional-config "${EXTRA_CONFIG/ROLE/ffn}" \
  > "$CAPTURE_DIR/single_ffn.log" 2>&1 < /dev/null &
echo "FFN launched pid=$! mode=$MODE (cards 8-15, DP8EP8) log=$CAPTURE_DIR/single_ffn.log"
sleep 10

# --- Attention (DP1 x TP8, cards 0-7) ---
env ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  VLLM_PLUGINS=ascend,afd PYTHONUNBUFFERED=1 VLLM_ENGINE_READY_TIMEOUT_S=3600 \
  VLLM_ASCEND_ENABLE_FLASHCOMM1=1 \
  nohup vllm serve "$MODEL" "${COMMON[@]}" --port 8000 \
  --data-parallel-size 1 --tensor-parallel-size 8 \
  --additional-config "${EXTRA_CONFIG/ROLE/attention}" \
  > "$CAPTURE_DIR/single_attn.log" 2>&1 < /dev/null &
echo "ATTENTION launched pid=$! mode=$MODE (cards 0-7, DP1TP8) log=$CAPTURE_DIR/single_attn.log"
echo "captures -> $CAPTURE_DIR/$RUN_ID/$MODE  (LAYERS=$LAYERS FULL_TENSORS=$FULL_TENSORS SYNC=$SYNC)"
