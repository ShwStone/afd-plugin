#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
# Run the Async CAM precision fixture across the three execution modes on the
# v0.26 production topology (DP2TP8 attention + DP16EP16 FFN, two nodes).
#
# For each mode it restarts both recipe servers with the matching
# ASYNC_MOE_UBATCHING / ASYNC_MOE_SPLIT and the precision debug env, waits for
# readiness, sends the fixed fixture requests (pinned to attention DP rank 0),
# waits for captures to flush, and merges captures into a shared capture dir.
#
# Required env:
#   NODE0        itask pod name for the Attention node (runs attention_dp2tp8.sh)
#   NODE1        itask pod name for the FFN node (runs ffn_ep16.sh)
#   REPO         repo path on both nodes (e.g. /a3_inference/.../code/afd-plugin)
#   MODEL_PATH   full DeepSeek-V3.2 W8A8 model dir on both nodes
#   AFD_HOST     Attention node rendezvous IP
#   CAPTURE_DIR  shared capture root (NAS) for AFD_ASYNC_MOE_PRECISION_DEBUG_DIR
# Optional:
#   MODES        comma list to run, default "no_ubatch,request,token"
#   CLIENT_ARGS  extra args for precision_fixture_client.py
#   API_PORT / FFN_PORT / AFD_PORT  ports (default 8000 / 8001 / 6453)
#
# It is driven from the itask control host (where `itask` is available). Each
# recipe server is launched in the background inside its pod; logs land in
# $CAPTURE_DIR/<mode>/<node>.log. Manual per-mode use is also supported by
# setting MODE=<one> and sourcing the mode env block below.

set -u

: "${NODE0:?NODE0 (attention itask pod) is required}"
: "${NODE1:?NODE1 (ffn itask pod) is required}"
: "${REPO:?REPO path on nodes is required}"
: "${MODEL_PATH:?MODEL_PATH is required}"
: "${AFD_HOST:?AFD_HOST (attention rendezvous IP) is required}"
: "${CAPTURE_DIR:?CAPTURE_DIR (shared capture root) is required}"

MODES="${MODES:-no_ubatch,request,token}"
API_PORT="${API_PORT:-8000}"
FFN_PORT="${FFN_PORT:-8001}"
AFD_PORT="${AFD_PORT:-6453}"
CLIENT_ARGS="${CLIENT_ARGS:-}"

RECIPE="recipe/npu/CAMAsyncAFDConnector/deepseek_v3_2/v0_26_accuracy"
RUN_ID="${AFD_ASYNC_MOE_PRECISION_DEBUG_RUN_ID:-fixture-$(date +%Y%m%d_%H%M%S)}"
FIXTURE_ID="${AFD_ASYNC_MOE_PRECISION_DEBUG_FIXTURE_ID:-fixture-v026}"
CLIENT_OUT="${CAPTURE_DIR}/${RUN_ID}/client"

itask_exec() { # retry-wrapped itask exec running a shell
  local pod="$1"; shift
  for attempt in 1 2 3; do
    if itask exec "${pod}" --tty=false -- bash -c "$*"; then
      return 0
    fi
    echo "[itask] attempt ${attempt} failed for ${pod}" >&2
    sleep 10
  done
  return 1
}

mode_env() { # echo per-mode ASYNC_MOE_* settings
  case "$1" in
    no_ubatch) echo "ASYNC_MOE_UBATCHING=false ASYNC_MOE_SPLIT=request" ;;
    request)   echo "ASYNC_MOE_UBATCHING=true ASYNC_MOE_SPLIT=request" ;;
    token)     echo "ASYNC_MOE_UBATCHING=true ASYNC_MOE_SPLIT=token" ;;
    *) echo "unknown mode $1" >&2; exit 2 ;;
  esac
}

wait_ready() {
  local url="http://127.0.0.1:${API_PORT}/v1/models"
  local deadline=$((SECONDS + 1800))
  while [ "${SECONDS}" -lt "${deadline}" ]; do
    if itask exec "${NODE0}" --tty=false -- bash -c \
        "curl -sf ${url} >/dev/null 2>&1"; then
      return 0
    fi
    sleep 20
  done
  echo "[wait] ${NODE0} server not ready after 30m" >&2
  return 1
}

kill_servers() {
  for node in "${NODE0}" "${NODE1}"; do
    itask_exec "${node}" \
      "pgrep -af '[v]llm serve' | awk '{print \$1}' | xargs -r kill -9 2>/dev/null; sleep 3; true"
  done
}

run_mode() {
  local mode="$1"
  local mode_extra
  mode_extra="$(mode_env "${mode}")"
  local node_capture="${CAPTURE_DIR}/${RUN_ID}/${mode}"
  mkdir -p "${node_capture}"

  echo "=== [${mode}] restarting servers ==="
  kill_servers
  # FFN first (headless), then Attention (master).
  itask_exec "${NODE1}" \
    "cd ${REPO} && mkdir -p ${node_capture} && \
     env MODEL_PATH=${MODEL_PATH} AFD_HOST=${AFD_HOST} LOCAL_IP=${AFD_HOST} \
         ${mode_extra} \
         AFD_ASYNC_MOE_PRECISION_DEBUG=1 \
         AFD_ASYNC_MOE_PRECISION_DEBUG_DIR=${node_capture} \
         AFD_ASYNC_MOE_PRECISION_DEBUG_RUN_ID=${RUN_ID} \
         AFD_ASYNC_MOE_PRECISION_DEBUG_FIXTURE_ID=${FIXTURE_ID} \
         AFD_ASYNC_MOE_PRECISION_DEBUG_MODE=${mode} \
         AFD_ASYNC_MOE_PRECISION_DEBUG_FULL_TENSORS=${AFD_ASYNC_MOE_PRECISION_DEBUG_FULL_TENSORS:-0} \
         AFD_ASYNC_MOE_PRECISION_DEBUG_SYNC=${AFD_ASYNC_MOE_PRECISION_DEBUG_SYNC:-0} \
         AFD_ASYNC_MOE_PRECISION_DEBUG_NO_OVERLAP=${AFD_ASYNC_MOE_PRECISION_DEBUG_NO_OVERLAP:-0} \
         bash ${RECIPE}/ffn_ep16.sh > ${node_capture}/ffn.log 2>&1 < /dev/null &"
  sleep 15
  itask_exec "${NODE0}" \
    "cd ${REPO} && mkdir -p ${node_capture} && \
     env MODEL_PATH=${MODEL_PATH} AFD_HOST=${AFD_HOST} LOCAL_IP=${AFD_HOST} \
         ${mode_extra} \
         AFD_ASYNC_MOE_PRECISION_DEBUG=1 \
         AFD_ASYNC_MOE_PRECISION_DEBUG_DIR=${node_capture} \
         AFD_ASYNC_MOE_PRECISION_DEBUG_RUN_ID=${RUN_ID} \
         AFD_ASYNC_MOE_PRECISION_DEBUG_FIXTURE_ID=${FIXTURE_ID} \
         AFD_ASYNC_MOE_PRECISION_DEBUG_MODE=${mode} \
         AFD_ASYNC_MOE_PRECISION_DEBUG_FULL_TENSORS=${AFD_ASYNC_MOE_PRECISION_DEBUG_FULL_TENSORS:-0} \
         AFD_ASYNC_MOE_PRECISION_DEBUG_SYNC=${AFD_ASYNC_MOE_PRECISION_DEBUG_SYNC:-0} \
         AFD_ASYNC_MOE_PRECISION_DEBUG_NO_OVERLAP=${AFD_ASYNC_MOE_PRECISION_DEBUG_NO_OVERLAP:-0} \
         bash ${RECIPE}/attention_dp2tp8.sh > ${node_capture}/attention.log 2>&1 < /dev/null &"

  echo "=== [${mode}] waiting for server ==="
  wait_ready || return 1

  echo "=== [${mode}] sending fixtures ==="
  mkdir -p "${CLIENT_OUT}"
  python3 -m tools.precision.precision_fixture_client \
    --base-url "http://127.0.0.1:${API_PORT}" \
    --model deepseek_v3_2 \
    --output-dir "${CLIENT_OUT}/${mode}" \
    --dp-rank 0 --two-request --single-long ${CLIENT_ARGS} || return 1

  echo "=== [${mode}] waiting for captures to flush ==="
  sleep 30
  echo "=== [${mode}] done (captures in ${node_capture}) ==="
}

main() {
  IFS=',' read -r -a mode_list <<< "${MODES}"
  for mode in "${mode_list[@]}"; do
    run_mode "${mode}" || { echo "[${mode}] failed" >&2; exit 1; }
  done
  echo "all modes finished; run_id=${RUN_ID}"
  echo "compare: python -m tools.benchmarks.compare_async_moe_captures"
  echo "  --run-a ${CAPTURE_DIR}/${RUN_ID}/no_ubatch --run-b ${CAPTURE_DIR}/${RUN_ID}/token"
}

main "$@"
