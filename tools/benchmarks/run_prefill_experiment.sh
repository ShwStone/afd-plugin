#!/usr/bin/env bash
# Full automation for prefill performance experiment.
# Manages server lifecycle and client benchmarks on two A3 nodes.
set -euo pipefail

NODE0=afd-test-1
NODE1=afd-test-2
REPO=/a3_inference/itask/workdir/tq02357756/shwstone/code/afd-plugin
CONFIG=tools/benchmarks/prefill_experiment.json
RESULT_DIR=bench_results/prefill
RESUME=false
MAX_RETRIES=3

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
  echo "[KILL] Cleaning up..."
  for node in $NODE0 $NODE1; do
    itask exec "$node" --tty=false -- bash -c '
      for pid in $(ps aux | grep -v grep | grep -E "[/]vllm serve|[Vv][Ll][Ll][Mm]::|multiproc_executor" | awk "{print \$2}"); do
        kill $pid 2>/dev/null || true
      done
      sleep 3
      remaining=$(ps aux | grep -v grep | grep -E "[/]vllm serve|[Vv][Ll][Ll][Mm]::|multiproc_executor" | wc -l)
      if [ "$remaining" -gt 0 ]; then
        echo "WARN: $remaining processes still alive"
        false
      else
        echo "OK: clean"
      fi
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
        echo '[READY]'
        exit 0
      fi
      sleep 10
    done
    echo '[TIMEOUT]'
    exit 1
  "
}

# Check if a cell (system, bt, prefix) already has all 5 RPS × 1 repeat verified
cell_complete() {
  local system=$1 bt=$2 pr=$3
  local complete=0
  for rps in 4 6 8 10 12; do
    local fname="${system}-mbt${bt}-rps${rps}p0-prefix${pr}-repeat1.verified.json"
    if ! itask_exec_retry "$NODE0" "test -f \"$REPO/$RESULT_DIR/$fname\"" 2>/dev/null; then
      return 1
    fi
  done
  return 0
}

run_benchmark() {
  local system=$1 bt=$2 pr=$3 label=$4
  local attempt=0 rc=1

  while [ $attempt -lt $MAX_RETRIES ]; do
    echo "[BENCH] attempt=$((attempt+1)) system=$system bt=$bt prefix=$pr"
    if itask_exec_retry "$NODE0" "
      cd \"$REPO\"
      python3 -m tools.benchmarks.prefill_experiment \
        --config $CONFIG run \
        --system $system \
        --batch-tokens $bt \
        --prefix-ratio $pr \
        $( [ \"$RESUME\" = true ] && echo '--resume' )
    "; then
      rc=0; break
    fi
    attempt=$((attempt + 1))
    echo "[WARN] Benchmark attempt $attempt failed, checking server..."
    # Check server still alive, if not, break (needs full restart)
    itask_exec_retry "$NODE0" "curl -sf http://127.0.0.1:8000/v1/models > /dev/null 2>&1" 2>/dev/null || {
      echo "[WARN] Server died, need full restart"
      return 1
    }
    sleep 15
  done
  return $rc
}

run_group() {
  local system=$1 bt=$2 pr=$3 pc=$4 label=$5

  # Skip if already complete and --resume
  if [ "$RESUME" = true ] && cell_complete "$system" "$bt" "$pr"; then
    echo "[SKIP] $label - all cells already verified"
    return 0
  fi

  echo ""
  echo "=============================================="
  echo "  [$label] system=$system bt=$bt prefix=$pr pc=$pc"
  echo "=============================================="

  kill_all

  local rc=0
  if [ "$system" = "dp4_tp8_sp" ]; then
    itask_exec_retry "$NODE1" "
      cd $REPO && mkdir -p bench_results/logs
      export PROMETHEUS_MULTIPROC_DIR=/a3_inference/itask/workdir/tq02357756/prometheus_tmp
      mkdir -p \$PROMETHEUS_MULTIPROC_DIR
      setsid env MAX_NUM_BATCHED_TOKENS=$bt DP_START_RANK=2 VLLM_ENABLE_PREFIX_CACHING=$pc \
        bash tools/benchmarks/prefill_launch_baseline_dp4tp8.sh \
        > bench_results/logs/${label}_node1.log 2>&1 < /dev/null &
    " 2>/dev/null || rc=1
    sleep 10
    itask_exec_retry "$NODE0" "
      cd $REPO && mkdir -p bench_results/logs
      export PROMETHEUS_MULTIPROC_DIR=/a3_inference/itask/workdir/tq02357756/prometheus_tmp
      mkdir -p \$PROMETHEUS_MULTIPROC_DIR
      setsid env MAX_NUM_BATCHED_TOKENS=$bt DP_START_RANK=0 VLLM_ENABLE_PREFIX_CACHING=$pc \
        bash tools/benchmarks/prefill_launch_baseline_dp4tp8.sh \
        > bench_results/logs/${label}_node0.log 2>&1 < /dev/null &
    " 2>/dev/null || rc=1

  elif [ "$system" = "afd_dp3_tp8_ep8" ]; then
    itask_exec_retry "$NODE0" "
      cd $REPO && mkdir -p bench_results/logs
      export PROMETHEUS_MULTIPROC_DIR=/a3_inference/itask/workdir/tq02357756/prometheus_tmp
      mkdir -p \$PROMETHEUS_MULTIPROC_DIR
      setsid env MAX_NUM_BATCHED_TOKENS=$bt DP_START_RANK=0 VLLM_ENABLE_PREFIX_CACHING=$pc \
        bash tools/benchmarks/prefill_launch_afd_attention.sh \
        > bench_results/logs/${label}_node0.log 2>&1 < /dev/null &
    " 2>/dev/null || rc=1
    sleep 10
    itask_exec_retry "$NODE1" "
      cd $REPO && mkdir -p bench_results/logs
      export PROMETHEUS_MULTIPROC_DIR=/a3_inference/itask/workdir/tq02357756/prometheus_tmp
      mkdir -p \$PROMETHEUS_MULTIPROC_DIR
      setsid env MAX_NUM_BATCHED_TOKENS=$bt DP_START_RANK=2 ATTN_DEVICES=0,1,2,3,4,5,6,7 VLLM_ENABLE_PREFIX_CACHING=$pc \
        bash tools/benchmarks/prefill_launch_afd_attention.sh \
        > bench_results/logs/${label}_node1_attn.log 2>&1 < /dev/null &
      sleep 3
      setsid env MAX_NUM_BATCHED_TOKENS=$bt VLLM_ENABLE_PREFIX_CACHING=$pc \
        bash tools/benchmarks/prefill_launch_afd_ffn.sh \
        > bench_results/logs/${label}_node1_ffn.log 2>&1 < /dev/null &
    " 2>/dev/null || rc=1
  fi

  if [ $rc -ne 0 ]; then
    echo "[WARN] Server start had partial failures, continuing anyway..."
  fi

  if ! wait_server 900; then
    echo "[FAIL] Server did not become ready for $label"
    echo "[INFO] Node0 log tail:"
    itask_exec_retry "$NODE0" "tail -10 \"$REPO/bench_results/logs/${label}_node0.log\"" 2>/dev/null || true
    kill_all
    return 1
  fi

  run_benchmark "$system" "$bt" "$pr" "$label"
  local run_rc=$?

  if [ $run_rc -ne 0 ]; then
    echo "[FAIL] Benchmark failed for $label (exit=$run_rc)"
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
  run_group "dp4_tp8_sp" 8192 "0" "0" "smoke_baseline"
  exit $?
fi

# ====== Full Experiment ======
echo ""
echo "##############################################"
echo "# Phase A: Primary comparison (prefix=0)     #"
echo "##############################################"
for system in dp4_tp8_sp afd_dp3_tp8_ep8; do
  for bt in 8192 16384 32768 49152 65536; do
    label="full_${system}_bt${bt}_prefix0"
    run_group "$system" "$bt" "0" "0" "$label"
  done
done

echo ""
echo "##############################################"
echo "# Phase B: Prefix sensitivity (0.25-0.99)    #"
echo "##############################################"
for system in dp4_tp8_sp afd_dp3_tp8_ep8; do
  for bt in 8192 16384 32768 49152 65536; do
    for ratio in 0.25 0.5 0.75 0.9 0.95 0.99; do
      suffix="${ratio//./}"
      label="full_${system}_bt${bt}_prefix${suffix}"
      run_group "$system" "$bt" "$ratio" "1" "$label"
    done
  done
done

echo ""
echo "=============================================="
echo "  Experiment complete!"
echo "=============================================="