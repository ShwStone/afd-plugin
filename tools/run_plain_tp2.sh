#!/usr/bin/env bash
# Plain vllm-ascend TP2 control run (NO AFD) for DeepSeek-V2-Lite.
#
# Purpose: discriminating experiment for the MLA prefill rope_single
# (npu_interleave_rope 561002) failure seen with FlashComm1 enabled.
# If this plain setup fails the same way, the bug is an upstream
# FlashComm1 + non-VL-model (DeepSeek) layout issue, not AFD code.
#
# Usage:
#   ./tools/run_plain_tp2.sh /path/to/deepseek-v2-lite
#
# Knobs (env vars):
#   DEVICES            default "0,1"     2 NPUs for TP2
#   API_PORT           default 19070
#   ENABLE_SP          default 1         1: VLLM_ASCEND_ENABLE_FLASHCOMM1=1
#                                       0: SP off (FlashComm1 disabled)
#   MAX_BATCHED_TOKENS default 8000
#   MAX_NUM_SEQS       default 8
#   PROMPT_TOKENS      default 512       approx. prefill prompt length
#   VLLM_BIN           default "vllm"
#   LOG_DIR            default "./logs_plain_tp2"
#
# Stop with Ctrl-C; the process group is torn down.

set -euo pipefail

MODEL="${1:-${MODEL:-}}"
if [[ -z "${MODEL}" ]]; then
    echo "usage: $0 <model-path-or-id>" >&2
    exit 1
fi

DEVICES="${DEVICES:-0,1}"
API_PORT="${API_PORT:-19070}"
API_HOST="${API_HOST:-127.0.0.1}"
ENABLE_SP="${ENABLE_SP:-1}"
MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-8000}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
PROMPT_TOKENS="${PROMPT_TOKENS:-512}"
VLLM_BIN="${VLLM_BIN:-vllm}"
LOG_DIR="${LOG_DIR:-./logs_plain_tp2}"
SERVED_NAME="deepseek-v2-lite-plain"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "${LOG_DIR}"

SP_ENV=()
if [[ "${ENABLE_SP}" == "1" ]]; then
    # vllm-ascend v0.19.1rc1 enable_sp() reads this env var (FlashComm1).
    SP_ENV=("VLLM_ASCEND_ENABLE_FLASHCOMM1=1")
fi

CMD=(
    "${VLLM_BIN}" serve "${MODEL}"
    --served-model-name "${SERVED_NAME}"
    --tensor-parallel-size 2
    --enable-expert-parallel
    --enforce-eager
    --trust-remote-code
    --max-num-seqs "${MAX_NUM_SEQS}"
    --max-num-batched-tokens "${MAX_BATCHED_TOKENS}"
    --no-enable-prefix-caching
    --host "${API_HOST}" --port "${API_PORT}"
)

echo "== Plain vllm-ascend TP2 (devices ${DEVICES}, ENABLE_SP=${ENABLE_SP}) =="
printf '%q ' "${SP_ENV[@]+"${SP_ENV[@]}" }" "${CMD[@]}"; echo

PGID=""
cleanup() {
    echo "shutting down..."
    [[ -n "${PGID}" ]] && kill -- -"${PGID}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

export VLLM_PLUGINS="ascend"
export PYTHONUNBUFFERED=1
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

env "${SP_ENV[@]}" \
    ASCEND_RT_VISIBLE_DEVICES="${DEVICES}" \
    setsid "${CMD[@]}" >"${LOG_DIR}/server.log" 2>&1 &
PGID=$!

echo "log: ${LOG_DIR}/server.log"
echo "waiting for API at http://${API_HOST}:${API_PORT}/v1/models ..."

deadline=$((SECONDS + ${STARTUP_TIMEOUT:-900}))
ready=0
while (( SECONDS < deadline )); do
    if ! kill -0 "${PGID}" 2>/dev/null; then
        echo "server exited during startup; see ${LOG_DIR}/server.log" >&2
        exit 1
    fi
    if curl -sf "http://${API_HOST}:${API_PORT}/v1/models" >/dev/null 2>&1; then
        ready=1
        break
    fi
    sleep 2
done

if [[ "${ready}" != "1" ]]; then
    echo "timed out waiting for API" >&2
    exit 1
fi
echo "API is ready."

# Send a prefill request: a long-ish prompt forces the mla_preprocess_prefill
# path that fails in the AFD setup under FlashComm1.
echo "sending prefill request (~${PROMPT_TOKENS} tokens prompt)..."
python3 - "${API_HOST}" "${API_PORT}" "${SERVED_NAME}" "${PROMPT_TOKENS}" <<'PYEOF'
import json, sys, urllib.request

host, port, model, prompt_tokens = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4])
# Rough heuristic: ~1 token per 4 chars of English text.
words = "The quick brown fox jumps over the lazy dog. " * (prompt_tokens // 10 + 1)
prompt = f"Question: {words}\nWhat is 17 + 25?\nAnswer:"
payload = {
    "model": model,
    "prompt": prompt,
    "max_tokens": 16,
    "temperature": 0,
}
req = urllib.request.Request(
    f"http://{host}:{port}/v1/completions",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
    method="POST",
)
try:
    with urllib.request.urlopen(req, timeout=180) as resp:
        body = json.loads(resp.read().decode())
    text = body.get("choices", [{}])[0].get("text", "")
    print("PREFILL OK:", text[:200])
except Exception as exc:  # noqa: BLE001
    print("PREFILL FAILED:", exc)
    sys.exit(1)
PYEOF

echo "done. server still running (Ctrl-C to stop)."
wait
