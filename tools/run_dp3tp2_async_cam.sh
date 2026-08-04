#!/usr/bin/env bash
# Manual launcher for CAMAsyncAFDConnector on Ascend NPU:
#   attention: DP3 x TP2 (6 ranks, SP optional) + FFN: TP2 (2 ranks, EP2)
#
# Mirrors tests/e2e/accuracy/test_gsm8k_npu_async_cam.py DP3TP2+EP2 topology
# and the enable_sp handling from tests/e2e/runner.py.
#
# Usage:
#   ./tools/run_dp3tp2_async_cam.sh /path/to/deepseek-v2-lite
#
# Knobs (env vars):
#   ATTN_DEVICES       default "0,1,2,3,4,5"   6 attention NPUs
#   FFN_DEVICES        default "6,7"           2 FFN NPUs
#   API_PORT_BASE      default 19080           attention API port; FFN uses +1
#   AFD_PORT           default 6453            CAM HCCL rendezvous port
#   SPLIT              default "token"         async_moe_split: token|request
#                                             (main only accepts "request")
#   UBATCH             default 1               0 disables async_moe_ubatching
#   ENABLE_SP          default 1               1: VLLM_ASCEND_ENABLE_FLASHCOMM1=1
#                                             on attention only (vllm-ascend's
#                                             enable_sp() reads this env var);
#                                             0: SP off everywhere
#   MAX_BATCHED_TOKENS default 8000            --max-num-batched-tokens (both roles)
#   MAX_NUM_SEQS       default 8               --max-num-seqs (both roles)
#   VLLM_BIN           default "vllm"
#   LOG_DIR            default "./logs_dp3tp2"
#
# Notes:
#   - Attention intentionally omits --enable-expert-parallel. FlashComm1/SP is
#     enabled independently by VLLM_ASCEND_ENABLE_FLASHCOMM1; only FFN owns EP.
#   - DP3TP2 with SP OFF dispatches duplicated full-stage tokens per TP rank
#     and overflows the FFN CAM dispatch-recv buffer at warmup
#     (CamMoeDistributeDispatchRecv MTE write address out of range).
#   - "token" split requires the token-split feature branch; main rejects it.
#   - Stop with Ctrl-C; both process groups are torn down.

set -euo pipefail

MODEL="${1:-${MODEL:-}}"
if [[ -z "${MODEL}" ]]; then
    echo "usage: $0 <model-path-or-id>" >&2
    exit 1
fi

ATTN_DEVICES="${ATTN_DEVICES:-0,1,2,3,4,5}"
FFN_DEVICES="${FFN_DEVICES:-6,7}"
API_PORT_BASE="${API_PORT_BASE:-19080}"
AFD_PORT="${AFD_PORT:-6453}"
AFD_HOST="${AFD_HOST:-127.0.0.1}"
API_HOST="${API_HOST:-127.0.0.1}"
SPLIT="${SPLIT:-token}"
UBATCH="${UBATCH:-1}"
ENABLE_SP="${ENABLE_SP:-1}"
MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-8000}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
VLLM_BIN="${VLLM_BIN:-vllm}"
LOG_DIR="${LOG_DIR:-./logs_dp3tp2}"
SERVED_PREFIX="deepseek-v2-lite-afd"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "${LOG_DIR}"

if [[ "${UBATCH}" == "1" ]]; then
    UBATCH_JSON='"async_moe_ubatching":true,"async_moe_num_ubatches":2,'
else
    UBATCH_JSON='"async_moe_ubatching":false,'
fi

EXTRA_CONFIG="{\"dynamicQuant\":0,${UBATCH_JSON}\"async_moe_split\":\"${SPLIT}\",\"attn_ranks_per_dp\":2}"

afd_config() {
    local role="$1"
    cat <<EOF
{"afd":{"role":"${role}","connector":"CAMAsyncAFDConnector","host":"${AFD_HOST}","port":${AFD_PORT},"num_attention_ranks":6,"num_ffn_ranks":2,"async":true,"compute_gate_on_attention":true,"connector_extra_config":${EXTRA_CONFIG}}}
EOF
}

COMMON_ARGS=(
    --trust-remote-code
    --max-num-seqs "${MAX_NUM_SEQS}"
    --max-num-batched-tokens "${MAX_BATCHED_TOKENS}"
    --no-enable-prefix-caching
)

base_env() {
    export VLLM_PLUGINS="ascend,afd"
    export PYTHONUNBUFFERED=1
    export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
    export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
    unset AFD_PLUGIN_EARLY_ENGINE_PATCH || true
}

FFN_CMD=(
    "${VLLM_BIN}" serve "${MODEL}"
    --served-model-name "${SERVED_PREFIX}-ffn"
    --data-parallel-size 1
    --tensor-parallel-size 2
    --enable-expert-parallel
    --additional-config "$(afd_config ffn)"
    --enforce-eager
    --host "${API_HOST}" --port "$((API_PORT_BASE + 1))"
    "${COMMON_ARGS[@]}"
)

# Sequence parallelism in vllm-ascend v0.19.1rc1 is driven by the
# VLLM_ASCEND_ENABLE_FLASHCOMM1 env var (see vllm_ascend/utils.py enable_sp);
# --compilation-config pass_config.enable_sp only feeds the graph-mode
# enable_sp_by_pass path and is a no-op under --enforce-eager.
ATTN_SP_ENV=()
if [[ "${ENABLE_SP}" == "1" ]]; then
    ATTN_SP_ENV=("VLLM_ASCEND_ENABLE_FLASHCOMM1=1")
fi

ATTN_CMD=(
    "${VLLM_BIN}" serve "${MODEL}"
    --served-model-name "${SERVED_PREFIX}-attention"
    --data-parallel-size 3
    --tensor-parallel-size 2
    --additional-config "$(afd_config attention)"
    --enforce-eager
    --host "${API_HOST}" --port "${API_PORT_BASE}"
    "${COMMON_ARGS[@]}"
)

echo "== FFN (devices ${FFN_DEVICES}) =="
printf '%q ' "${FFN_CMD[@]}"; echo
echo "== Attention (devices ${ATTN_DEVICES}) =="
printf '%q ' "${ATTN_SP_ENV[@]+"${ATTN_SP_ENV[@]}" }" "${ATTN_CMD[@]}"; echo

FFN_PGID=""
ATTN_PGID=""
cleanup() {
    echo "shutting down..."
    [[ -n "${ATTN_PGID}" ]] && kill -- -"${ATTN_PGID}" 2>/dev/null || true
    [[ -n "${FFN_PGID}" ]] && kill -- -"${FFN_PGID}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

base_env

# FFN first (matches tests/e2e/conftest.py); both block at the CAM HCCL
# rendezvous until all 8 ranks connect. FFN keeps SP off: its CAM-received
# tokens are expert-routed per rank, not a replicated batch to shard.
env -u VLLM_ASCEND_ENABLE_FLASHCOMM1 -u VLLM_ASCEND_ENABLE_FLASHCOMM \
    ASCEND_RT_VISIBLE_DEVICES="${FFN_DEVICES}" \
    setsid "${FFN_CMD[@]}" >"${LOG_DIR}/ffn.log" 2>&1 &
FFN_PGID=$!

sleep 2
if ! kill -0 "${FFN_PGID}" 2>/dev/null; then
    echo "FFN exited during startup; see ${LOG_DIR}/ffn.log" >&2
    exit 1
fi

env "${ATTN_SP_ENV[@]}" \
    ASCEND_RT_VISIBLE_DEVICES="${ATTN_DEVICES}" \
    setsid "${ATTN_CMD[@]}" >"${LOG_DIR}/attention.log" 2>&1 &
ATTN_PGID=$!

echo "logs: ${LOG_DIR}/ffn.log ${LOG_DIR}/attention.log"
echo "waiting for attention API at http://${API_HOST}:${API_PORT_BASE}/v1/models ..."

deadline=$((SECONDS + ${STARTUP_TIMEOUT:-900}))
while (( SECONDS < deadline )); do
    if ! kill -0 "${FFN_PGID}" 2>/dev/null; then
        echo "FFN exited; see ${LOG_DIR}/ffn.log" >&2
        exit 1
    fi
    if ! kill -0 "${ATTN_PGID}" 2>/dev/null; then
        echo "attention exited; see ${LOG_DIR}/attention.log" >&2
        exit 1
    fi
    if curl -sf "http://${API_HOST}:${API_PORT_BASE}/v1/models" >/dev/null 2>&1; then
        echo "attention API is ready."
        echo "test request:"
        echo "  curl -s http://${API_HOST}:${API_PORT_BASE}/v1/completions -H 'Content-Type: application/json' -d '{\"model\":\"${SERVED_PREFIX}-attention\",\"prompt\":\"San Francisco is a\",\"max_tokens\":16,\"temperature\":0}'"
        wait
        exit 0
    fi
    sleep 2
done

echo "timed out waiting for attention API" >&2
exit 1
