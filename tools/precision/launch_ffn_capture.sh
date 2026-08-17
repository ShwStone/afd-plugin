#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
# Launch the v0.26 FFN (DP16 × EP16, FlashComm1 off) Async CAM recipe with
# precision capture for one execution mode. Run on the FFN node; the Attention
# twin is launch_attn_capture.sh.
#
# Multi-instance CAM must be brought up together (memory afd-npu-testing-
# findings): start BOTH roles with the same MODE and RUN_ID at roughly the same
# time. To switch mode, relaunch both with the new MODE (this script stops any
# running server and orphan VLLM workers on the node first).
#
#   MODE=no_ubatch|token RUN_ID=<shared-id> \
#   MODEL_PATH=/home/admin/model-csi/model AFD_HOST=<attention-ip> \
#   bash tools/precision/launch_ffn_capture.sh
#
# Defaults tailored to a prefill-only layer 3,4,5 comparison:
#   LAYERS=3,4,5  FULL_TENSORS=1  SYNC=1
# (override each via env). Captures land in $CAPTURE_DIR/$RUN_ID/$MODE; the
# server log in $CAPTURE_DIR/ffn.log.
set -u

REPO="${REPO:-/a3_inference/itask/workdir/tq02357756/shwstone/code/afd-plugin}"
RECIPE="recipe/npu/CAMAsyncAFDConnector/deepseek_v3_2/v0_26_accuracy"
MODEL_PATH="${MODEL_PATH:?MODEL_PATH required}"
AFD_HOST="${AFD_HOST:?AFD_HOST (attention rendezvous IP) required}"
MODE="${MODE:-token}"
CAPTURE_DIR="${CAPTURE_DIR:-$REPO/bench_results/precision}"
RUN_ID="${RUN_ID:-precision-run}"
LAYERS="${LAYERS:-3,4,5}"
FULL_TENSORS="${FULL_TENSORS:-1}"
SYNC="${SYNC:-1}"
AFD_PORT="${AFD_PORT:-1239}"

case "$MODE" in
  no_ubatch) ASYNC_MOE_UBATCHING=false; ASYNC_MOE_SPLIT=request ;;
  token)     ASYNC_MOE_UBATCHING=true;  ASYNC_MOE_SPLIT=token ;;
  *) echo "unknown MODE=$MODE (use no_ubatch|token)" >&2; exit 2 ;;
esac
export ASYNC_MOE_UBATCHING ASYNC_MOE_SPLIT

export AFD_ASYNC_MOE_PRECISION_DEBUG=1
export AFD_ASYNC_MOE_PRECISION_DEBUG_DIR="$CAPTURE_DIR"
export AFD_ASYNC_MOE_PRECISION_DEBUG_RUN_ID="$RUN_ID"
export AFD_ASYNC_MOE_PRECISION_DEBUG_FIXTURE_ID="${FIXTURE_ID:-fixture-v026}"
export AFD_ASYNC_MOE_PRECISION_DEBUG_MODE="$MODE"
export AFD_ASYNC_MOE_PRECISION_DEBUG_LAYERS="$LAYERS"
export AFD_ASYNC_MOE_PRECISION_DEBUG_FULL_TENSORS="$FULL_TENSORS"
export AFD_ASYNC_MOE_PRECISION_DEBUG_SYNC="$SYNC"
export AFD_ASYNC_MOE_PRECISION_DEBUG_NO_OVERLAP="${NO_OVERLAP:-0}"

cd "$REPO" || exit 2
mkdir -p "$CAPTURE_DIR/$RUN_ID/$MODE"

# Stop any running server and orphan VLLM workers on this node.
pkill -f '[v]llm serve' 2>/dev/null || true
ps -eo pid,comm | awk '$2 ~ /^VLLM::/ {print $1}' | xargs -r kill -9 2>/dev/null || true
for pid in $(npu-smi info 2>/dev/null | grep 'VLLMWorker' | awk -F'|' '{print $3}' | tr -d ' '); do
  kill -9 "$pid" 2>/dev/null || true
done
sleep 8
rm -f /dev/shm/vllm_* 2>/dev/null || true

LOG="$CAPTURE_DIR/ffn.log"
env MODEL_PATH="$MODEL_PATH" AFD_HOST="$AFD_HOST" AFD_PORT="$AFD_PORT" \
  GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.80}" \
  nohup bash "$RECIPE/ffn_ep16.sh" > "$LOG" 2>&1 < /dev/null &
echo "launched ffn mode=$MODE pid=$! log=$LOG"
echo "captures -> $CAPTURE_DIR/$RUN_ID/$MODE  (RUN_ID=$RUN_ID, LAYERS=$LAYERS, FULL_TENSORS=$FULL_TENSORS)"
