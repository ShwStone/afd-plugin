#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
# Launch one node's v0.26 accuracy recipe server with precision debug capture.
#
# Written as a synced file (not inline itask escaping) per the operator lesson
# that nested quotes through `itask exec` break. Run it on each node after an
# `itask sync`:
#
#   ROLE=attention|ffn MODE=no_ubatch|request|token \
#   MODEL_PATH=/home/admin/model-csi/model AFD_HOST=<attn-ip> \
#   CAPTURE_DIR=<shared-nas-dir> RUN_ID=<id> \
#   bash tools/precision/precision_launch_node.sh
#
# It stops any running server and orphan VLLM workers on the node, then spawns
# the matching recipe server with the precision debug env exported. Logs land
# in $CAPTURE_DIR/<role>.log; captures in $CAPTURE_DIR/$RUN_ID/<mode>.
set -u

REPO="${REPO:-/a3_inference/itask/workdir/tq02357756/shwstone/code/afd-plugin}"
RECIPE="recipe/npu/CAMAsyncAFDConnector/deepseek_v3_2/v0_26_accuracy"
MODEL_PATH="${MODEL_PATH:?MODEL_PATH required}"
AFD_HOST="${AFD_HOST:?AFD_HOST required}"
ROLE="${ROLE:?ROLE=attention|ffn}"
MODE="${MODE:?MODE=no_ubatch|request|token}"
CAPTURE_DIR="${CAPTURE_DIR:?CAPTURE_DIR required}"
RUN_ID="${RUN_ID:?RUN_ID required}"
AFD_PORT="${AFD_PORT:-1239}"

cd "$REPO" || exit 2
case "$MODE" in
  no_ubatch) ASYNC_MOE_UBATCHING=false; ASYNC_MOE_SPLIT=request ;;
  request)   ASYNC_MOE_UBATCHING=true;  ASYNC_MOE_SPLIT=request ;;
  token)     ASYNC_MOE_UBATCHING=true;  ASYNC_MOE_SPLIT=token ;;
  *) echo "unknown MODE=$MODE" >&2; exit 2 ;;
esac
export ASYNC_MOE_UBATCHING ASYNC_MOE_SPLIT

export AFD_ASYNC_MOE_PRECISION_DEBUG=1
export AFD_ASYNC_MOE_PRECISION_DEBUG_DIR="$CAPTURE_DIR"
export AFD_ASYNC_MOE_PRECISION_DEBUG_RUN_ID="$RUN_ID"
export AFD_ASYNC_MOE_PRECISION_DEBUG_FIXTURE_ID="${FIXTURE_ID:-fixture-v026}"
export AFD_ASYNC_MOE_PRECISION_DEBUG_MODE="$MODE"
export AFD_ASYNC_MOE_PRECISION_DEBUG_FULL_TENSORS="${AFD_ASYNC_MOE_PRECISION_DEBUG_FULL_TENSORS:-0}"
export AFD_ASYNC_MOE_PRECISION_DEBUG_SYNC="${AFD_ASYNC_MOE_PRECISION_DEBUG_SYNC:-0}"
export AFD_ASYNC_MOE_PRECISION_DEBUG_NO_OVERLAP="${AFD_ASYNC_MOE_PRECISION_DEBUG_NO_OVERLAP:-0}"

mkdir -p "$CAPTURE_DIR" "$CAPTURE_DIR/$RUN_ID/$MODE"

# Stop any running servers and orphan VLLM workers on this node. Worker
# subprocesses hold NPU context independently and must be killed via their
# comm name and via npu-smi (see memory npu-worker-cleanup).
pkill -f '[v]llm serve' 2>/dev/null || true
ps -eo pid,comm | awk '$2 ~ /^VLLM::/ {print $1}' | xargs -r kill -9 2>/dev/null || true
for pid in $(npu-smi info 2>/dev/null | grep 'VLLMWorker' | awk -F'|' '{print $3}' | tr -d ' '); do
  kill -9 "$pid" 2>/dev/null || true
done
sleep 8
rm -f /dev/shm/vllm_* 2>/dev/null || true

LOG="$CAPTURE_DIR/${ROLE}.log"
if [ "$ROLE" = "attention" ]; then
  env MODEL_PATH="$MODEL_PATH" AFD_HOST="$AFD_HOST" AFD_PORT="$AFD_PORT" \
    GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.80}" \
    nohup bash "$RECIPE/attention_dp2tp8.sh" > "$LOG" 2>&1 < /dev/null &
else
  env MODEL_PATH="$MODEL_PATH" AFD_HOST="$AFD_HOST" AFD_PORT="$AFD_PORT" \
    GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.80}" \
    nohup bash "$RECIPE/ffn_ep16.sh" > "$LOG" 2>&1 < /dev/null &
fi
echo "launched role=$ROLE mode=$MODE pid=$! log=$LOG"
