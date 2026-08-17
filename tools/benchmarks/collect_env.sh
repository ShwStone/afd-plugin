#!/usr/bin/env bash
# Stage-2 environment snapshot (doc §6). Runs inside the pod container.
#
# Usage:
#   bash tools/benchmarks/collect_env.sh [stage2-config.json]
#
# Writes into bench_results/prefill_stage2/01_environment/:
#   preflight.json, stack.json, topology.json, npu_smi_{before,after}.txt,
#   env.json, time_sync/...
#
# NOTE: this script runs inside the container (needs npu-smi, git, the repo).
set -euo pipefail

REPO=${REPO:-/a3_inference/itask/workdir/tq02357756/shwstone/code/afd-plugin}
MODEL=${MODEL:-/a3_inference/itask/workdir/tq02357756/shwstone/deepseek-v3.2-reduced}
CONFIG=${1:-}
OUT="$REPO/bench_results/prefill_stage2/01_environment"
mkdir -p "$OUT/time_sync"

echo "[env] output dir: $OUT"

# 1. Preflight via the existing tool (packages, npu-smi, dataset/model hashes, git).
#    The experiment config is optional; pass one if available so model+dataset
#    hashes are pinned to the actual Stage-2 cell.
if [ -n "$CONFIG" ]; then
  python3 -m tools.benchmarks.prefill_preflight \
    --require-npu \
    --model-config "$MODEL" \
    --experiment-config "$CONFIG" \
    --dataset tools/datasets/cp8sp50k_token_ids.jsonl \
    --output "$OUT/preflight.json" || echo "[env] preflight reported failures (see preflight.json)"
else
  python3 -m tools.benchmarks.prefill_preflight \
    --require-npu \
    --model-config "$MODEL" \
    --dataset tools/datasets/cp8sp50k_token_ids.jsonl \
    --output "$OUT/preflight.json" || echo "[env] preflight reported failures (see preflight.json)"
fi

# 2. Stack + tool versions.
{
  echo "date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "hostname: $(hostname)"
  echo "uname: $(uname -a)"
  echo "--- python packages ---"
  python3 -m pip list 2>/dev/null | grep -iE "vllm|torch|npu|ascend|transformers|cam" || true
  echo "--- msprof ---"
  msprof --help 2>&1 | head -5 || true
} > "$OUT/stack.json"
echo "[env] stack.json written"

# 3. Topology + NPU inventory.
{
  echo "=== interfaces ==="
  ip addr show 2>/dev/null | grep -E "^[0-9]+:|inet " || true
  echo "=== /etc/hosts ==="
  cat /etc/hosts 2>/dev/null || true
  echo "=== npu-smi board ==="
  npu-smi info -t board -i 0 2>&1 | head -40 || true
  echo "=== npu-smi p2p ==="
  npu-smi info -t p2p -i 0 2>&1 | head -20 || true
  echo "=== HCCL/RANK env hints ==="
  env | grep -iE "HCCL|RANK|NPU|ASCEND_RT" | sed 's/=.*/=<masked>/' || true
} > "$OUT/topology.json"
echo "[env] topology.json written"

# 4. npu-smi before/after (doc: at least once around each run).
npu-smi info > "$OUT/npu_smi_before.txt" 2>&1 || true
npu-smi info > "$OUT/npu_smi_after.txt" 2>&1 || true

# 5. Environment variables (masked secrets).
env | grep -E "VLLM|AFD|HCCL|ASCEND|DP_|TP_|MAX_NUM|GLOO|OMP|ACL|PYTHONPATH" \
  | sed -E 's/=(.*)/=<masked>/' | sort > "$OUT/env.json" || true
echo "[env] env.json written"

# 6. Time synchronization.
{
  echo "date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "epoch_us: $(date +%s%6N)"
  chronyc tracking 2>&1 | head -20 || true
  ntpq -p 2>&1 | head -10 || true
} > "$OUT/time_sync/node0.txt" 2>/dev/null || true

echo "[env] collect_env.sh done"
