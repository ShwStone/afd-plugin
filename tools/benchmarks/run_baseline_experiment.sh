#!/usr/bin/env bash
# Fork-ready runner for the BASELINE (dp4_tp8_sp) system only.
# Each fork agent points NODE0/NODE1 at its own pod pair.
#
# Usage:
#   NODE0=afd-exp-20260804-2 NODE1=afd-exp-20260804-3 \
#     bash tools/benchmarks/run_baseline_experiment.sh [--resume] [--smoke]
set -euo pipefail

NODE0=${NODE0:-afd-exp-20260804-2}
NODE1=${NODE1:-afd-exp-20260804-3}
REPO=${REPO:-/a3_inference/itask/workdir/tq02357756/shwstone/code/afd-plugin}
CONFIG=tools/benchmarks/prefill_experiment.json
RESULT_DIR=bench_results/prefill
SYSTEM=dp4_tp8_sp
MAX_RETRIES=3
RESUME=false

# ---- Helpers ----

itask_exec_retry() {
  local node=$1; shift
  local attempt=0 rc=1
  while [ $attempt -lt $MAX_RETRIES ]; do
    if itask exec "$node" --tty=false -- bash -c "$*" 2>&1; then
      rc=0; break
    fi
    attempt=$((attempt + 1))
    echo "[RETRY $attempt/$MAX_RETRIES] itask exec $node failed, retrying in 10s..."
    sleep 10
  done
  return $rc
}

kill_all() {
  echo "[KILL] Cleaning up $NODE0 $NODE1..."
  for node in $NODE0 $NODE1; do
    itask exec "$node" --tty=false -- bash -c '
      for pid in $(ps aux | grep -v grep | grep -E "[/]vllm serve|[Vv][Ll][Ll][Mm]::|multiproc_executor" | awk "{print \$2}"); do
        kill $pid 2>/dev/null || true
      done
      sleep 3
      remaining=$(ps aux | grep -v grep | grep -E "[/]vllm serve|[Vv][Ll][Ll][Mm]::|multiproc_executor" | wc -l)
      if [ "$remaining" -gt 0 ]; then echo "WARN: $remaining still alive"; false; else echo "OK: clean"; fi
    ' 2>/dev/null || true
  done
  sleep 5
}

wait_server() {
  local timeout=$1
  echo "[WAIT] Polling /v1/models (timeout=${timeout}s)..."
  itask_exec_retry "$NODE0" "
    deadline=\$((\$(date +%s) + $timeout))
    while [[ \$(date +%s) -lt \$deadline ]]; do
      if curl -sf http://127.0.0.1:8000/v1/models > /dev/null 2>&1; then
        echo '[READY]'; exit 0
      fi
      sleep 10
    done
    echo '[TIMEOUT]'; exit 1
  "
}

cell_complete() {
  local bt=$1 pr=$2
  # prefill_experiment.py:240 encodes '.' as 'p' in result filenames
  # (e.g. prefix_ratio 0.25 -> "prefix0p25"), so mirror that here.
  local pr_enc="${pr//./p}"
  for rps in 4 6 8 10 12; do
    local fname="${SYSTEM}-mbt${bt}-rps${rps}p0-prefix${pr_enc}-repeat1.verified.json"
    # test -f exits 0 if file exists, 1 if not — NOT a transient error,
    # so use bare itask exec without retry wrapper. itask exec runs no shell,
    # so the `test -f` must be wrapped in bash -c.
    if ! itask exec "$NODE0" --tty=false -- bash -c "test -f \"$REPO/$RESULT_DIR/$fname\"" 2>/dev/null; then
      return 1
    fi
  done
  return 0
}

run_benchmark() {
  local bt=$1 pr=$2 label=$3
  local attempt=0 rc=1
  while [ $attempt -lt $MAX_RETRIES ]; do
    echo "[BENCH] attempt=$((attempt+1)) system=$SYSTEM bt=$bt prefix=$pr"
    if itask_exec_retry "$NODE0" "
      cd \"$REPO\"
      python3 -m tools.benchmarks.prefill_experiment \
        --config $CONFIG run \
        --system $SYSTEM \
        --batch-tokens $bt \
        --prefix-ratio $pr \
        $( [ "$RESUME" = true ] && echo '--resume' )
    "; then
      rc=0; break
    fi
    attempt=$((attempt + 1))
    itask_exec_retry "$NODE0" "curl -sf http://127.0.0.1:8000/v1/models > /dev/null 2>&1" 2>/dev/null || {
      echo "[WARN] Server died, need full restart"; return 1
    }
    sleep 15
  done
  return $rc
}

run_group() {
  local bt=$1 pr=$2 pc=$3 label=$4
  if [ "$RESUME" = true ] && cell_complete "$bt" "$pr"; then
    echo "[SKIP] $label - all cells already verified"
    return 0
  fi

  echo ""
  echo "=============================================="
  echo "  [$label] system=$SYSTEM bt=$bt prefix=$pr pc=$pc"
  echo "=============================================="

  kill_all

  # Node1 headless first, then Node0 master
  itask exec "$NODE1" --tty=false -- bash -c "
    cd $REPO && mkdir -p bench_results/logs
    export PROMETHEUS_MULTIPROC_DIR=/a3_inference/itask/workdir/tq02357756/prometheus_tmp
    mkdir -p \$PROMETHEUS_MULTIPROC_DIR
    setsid env MAX_NUM_BATCHED_TOKENS=$bt DP_START_RANK=2 VLLM_ENABLE_PREFIX_CACHING=$pc \
      bash tools/benchmarks/prefill_launch_baseline_dp4tp8.sh \
      > bench_results/logs/${label}_node1.log 2>&1 < /dev/null &
  " 2>/dev/null || true
  sleep 10
  itask exec "$NODE0" --tty=false -- bash -c "
    cd $REPO && mkdir -p bench_results/logs
    export PROMETHEUS_MULTIPROC_DIR=/a3_inference/itask/workdir/tq02357756/prometheus_tmp
    mkdir -p \$PROMETHEUS_MULTIPROC_DIR
    setsid env MAX_NUM_BATCHED_TOKENS=$bt DP_START_RANK=0 VLLM_ENABLE_PREFIX_CACHING=$pc \
      bash tools/benchmarks/prefill_launch_baseline_dp4tp8.sh \
      > bench_results/logs/${label}_node0.log 2>&1 < /dev/null &
  " 2>/dev/null || true

  if ! wait_server 900; then
    echo "[FAIL] Server not ready for $label"
    itask exec "$NODE0" --tty=false -- tail -10 "$REPO/bench_results/logs/${label}_node0.log" 2>/dev/null || true
    kill_all
    return 1
  fi

  if ! run_benchmark "$bt" "$pr" "$label"; then
    echo "[FAIL] Benchmark failed for $label"
    kill_all
    return 1
  fi

  echo "[DONE] $label"
}

# ---- Arg Parsing ----
SMOKE=false
for arg in "$@"; do
  case "$arg" in
    --smoke) SMOKE=true ;;
    --resume) RESUME=true ;;
  esac
done

if [ "$SMOKE" = true ]; then
  run_group 8192 "0" "0" "smoke_baseline"
  exit $?
fi

# ---- Full baseline matrix ----
echo ""
echo "##############################################"
echo "# Baseline $SYSTEM on $NODE0 + $NODE1        #"
echo "##############################################"

# Phase A: prefix=0 (no prefix cache)
for bt in 8192 16384 32768 49152 65536; do
  run_group "$bt" "0" "0" "full_baseline_bt${bt}_prefix0"
done

# Phase B: prefix ratios (prefix cache enabled)
for bt in 8192 16384 32768 49152 65536; do
  for ratio in 0.25 0.5 0.75 0.9 0.95 0.99; do
    suffix="${ratio//./}"
    run_group "$bt" "$ratio" "1" "full_baseline_bt${bt}_prefix${suffix}"
  done
done

echo ""
echo "=============================================="
echo "  Baseline experiment complete!"
echo "=============================================="