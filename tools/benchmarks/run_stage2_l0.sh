#!/usr/bin/env bash
# Stage-2 L0 orchestrator: 3-repeat E2E / ablation / prefix collection.
#
# Key mechanics (see docs/npu/PREFILL_PERFORMANCE_STAGE2_DATA_COLLECTION.zh-CN.md):
#   * Same (system, bt, prefix, variant, cache_state) group reuses ONE server;
#     the 3 repeats run consecutively and repeat isolation is done with
#     POST /reset_prefix_cache (cold prefix=0) or a re-precondition (steady).
#   * Switching (system, bt, variant) restarts the server (bt is a launch arg,
#     variant changes the AFD launch config).
#   * prefix>0 cold restarts the server for EVERY repeat (retained prefix-cache
#     blocks make block_pool.reset_prefix_cache() fail).
#   * /reset_prefix_cache returns HTTP 200 unconditionally, so success is
#     verified from the attention-master log: "Successfully reset prefix cache".
#
# Usage:
#   CELLS="<config1> <config2> ..." \
#   NODE0=afd-exp-1 NODE1=afd-exp-2 NODE0_IP=33.182.141.136 \
#   bash tools/benchmarks/run_stage2_l0.sh [--resume] [--smoke]
#
#   PHASE=main|ablation-knee|ablation-longshort|prefix-cold|prefix-steady \
#     builds a default CELLS list (see _build_cells).
set -euo pipefail

NODE0=${NODE0:-afd-exp-1}
NODE1=${NODE1:-afd-exp-2}
NODE0_IP=${NODE0_IP:-33.182.141.136}
REPO=${REPO:-/a3_inference/itask/workdir/tq02357756/shwstone/code/afd-plugin}
PHASE=${PHASE:-}
CELLS=${CELLS:-}
RESUME=false
SMOKE=false
MAX_RETRIES=3
PROM_DIR=/a3_inference/itask/workdir/tq02357756/shwstone/prometheus_tmp
LOG_DIR=bench_results/logs
# Stage-3 btsweep extensions (defaults keep Stage-2 behaviour unchanged).
PROFILE=${PROFILE:-0}
CORRELATION=${CORRELATION:-0}
RUN_REPEATS=${RUN_REPEATS:-3}
SAMPLER_INTERVAL=${SAMPLER_INTERVAL:-1}
PROFILE_DIR=""

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

# cfg_get <config> <key>              -> top-level JSON value
# cfg_systems_key <config>            -> the single systems dict key
# cfg_stage2 <config> <key>           -> stage2.<key> (empty string when null)
cfg_get()      { python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])' "$1" "$2"; }
cfg_systems_key() { python3 -c 'import json,sys; print(next(iter(json.load(open(sys.argv[1]))["systems"])))' "$1"; }
cfg_stage2()   { python3 -c 'import json,sys; v=json.load(open(sys.argv[1])).get("stage2",{}).get(sys.argv[2]); print("" if v is None else v)' "$1" "$2"; }

# result_stems <config> -> "<system_name> <bt> <rps> <prefix> <result_dir>"
result_stems() {
  python3 - "$1" <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))
print(next(iter(c["systems"])), c["batch_tokens"][0], c["request_rates"][0],
      next(iter(c["datasets"])), c["result_directory"])
PY
}

kill_all() {
  SERVER_LABEL=""
  echo "[KILL] Cleaning up $NODE0 $NODE1..."
  for node in $NODE0 $NODE1; do
    itask exec "$node" --tty=false -- bash -c '
      for pid in $(ps aux | grep -v grep | grep -E "[/]vllm serve|[Vv][Ll][Ll][Mm]::|multiproc_executor" | awk "{print \$2}"); do
        kill $pid 2>/dev/null || true
      done
      sleep 3
      remaining=$(ps aux | grep -v grep | grep -E "[/]vllm serve|[Vv][Ll][Ll][Mm]::|multiproc_executor" | wc -l)
      if [ "$remaining" -gt 0 ]; then echo "WARN: $remaining still alive"; else echo "OK: clean"; fi
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

# reset_prefix_cache — POST reset + verify via the RUNNING attention-master log.
# Uses $SERVER_LABEL (the live server), not the current cell's label, so reuse
# across groups greps the correct log. Returns 0=ok, 1=failed, 2=no log found.
reset_prefix_cache() {
  local label="${SERVER_LABEL:-}"
  if [ -z "$label" ]; then
    echo "[RESET] WARN: no running server to reset"; return 2
  fi
  # NOTE: itask exec does NOT run a shell, so every multi-word command must be
  # wrapped in `bash -c` (bare "itask exec pod -- cmd arg" execs cmd as a binary).
  #
  # The reset is called immediately after a benchmark drains; vLLM frees the
  # last blocks asynchronously, so the first POST can hit "Failed to reset
  # prefix cache because some blocks (N) are not freed yet". Retry a few times.
  local attempt last_line
  for attempt in 1 2 3; do
    echo "[RESET] attempt $attempt: POST /reset_prefix_cache (log=${label}_node0.log)"
    if ! itask exec "$NODE0" --tty=false -- bash -c \
        "curl -s -o /dev/null -w 'reset_http=%{http_code}\n' -X POST 'http://127.0.0.1:8000/reset_prefix_cache?reset_running_requests=true&reset_external=true'" \
        > /dev/null 2>&1; then
      echo "[RESET] attempt $attempt: itask exec curl failed"
      sleep 5
      continue
    fi
    local deadline=$(( $(date +%s) + 15 ))
    local saw_failure=false
    last_line=""
    while [ "$(date +%s)" -lt "$deadline" ]; do
      last_line=$(itask exec "$NODE0" --tty=false -- bash -c \
        "grep -E 'reset prefix cache' \"$REPO/$LOG_DIR/${label}_node0.log\" | tail -1" 2>/dev/null || true)
      case "$last_line" in
        *"Successfully reset prefix cache"*) echo "[RESET] OK: $last_line"; return 0 ;;
        *"Failed to reset prefix cache"*)    saw_failure=true; break ;;
      esac
      sleep 2
    done
    if [ "$saw_failure" = true ]; then
      echo "[RESET] attempt $attempt: blocks retained, retrying after drain"
    else
      echo "[RESET] attempt $attempt: no confirmed reset line yet"
    fi
    sleep 5
  done
  echo "[RESET] FAILED: no confirmed reset after 3 attempts"
  return 2
}

send_precondition() {
  local pr=$1
  local dataset
  dataset=$(python3 -c "
import re, pathlib
p = int(float('$pr') * 100)
print(pathlib.Path('tools/datasets') / f'prefix_precondition_prefix{p}.jsonl')
")
  echo "[PRECOND] Sending $dataset to warm prefix cache..."
  itask_exec_retry "$NODE0" "
    cd \"$REPO\"
    python3 -m tools.benchmarks.send_precondition \
      --base-url http://127.0.0.1:8000 \
      --dataset $dataset \
      --served-model-name deepseek_v3_2
  "
}

start_server() {
  local system_label=$1 bt=$2 pc=$3 variant=$4 label=$5
  local varg=""
  [ -n "$variant" ] && varg="STAGE2_VARIANT=$variant"
  # PROFILE registers vLLM's request-controlled profiler. CORRELATION can be
  # enabled independently and records sidecars for the full server lifetime.
  local baseline_prof_env="" attn_prof_env="" ffn_prof_env=""
  if [ "$PROFILE" = "1" ]; then
    baseline_prof_env="VLLM_TORCH_PROFILER_DIR=$PROFILE_DIR/baseline"
    if [ "$system_label" = "afd" ]; then
      attn_prof_env="VLLM_TORCH_PROFILER_DIR=$PROFILE_DIR/attention AFD_ASYNC_MOE_LAYOUT_LOG=1 AFD_CAM_OP_IO_LOG=1"
      ffn_prof_env="VLLM_TORCH_PROFILER_DIR=$PROFILE_DIR/ffn AFD_ASYNC_MOE_LAYOUT_LOG=1 AFD_CAM_OP_IO_LOG=1"
    fi
  fi
  if [ "$system_label" = "afd" ] && { [ "$PROFILE" = "1" ] || [ "$CORRELATION" = "1" ]; }; then
    local trace_session_id="${label}_$(date +%s%N)"
    local correlation_env="AFD_TRACE_ENABLE=1 AFD_TRACE_SESSION_ID=$trace_session_id AFD_TRACE_DIR=$PROFILE_DIR/correlation"
    attn_prof_env="$attn_prof_env $correlation_env"
    ffn_prof_env="$ffn_prof_env $correlation_env"
  fi
  if [ "$system_label" = "baseline" ]; then
    itask exec "$NODE1" --tty=false -- bash -c "
      cd $REPO && mkdir -p $LOG_DIR
      export PROMETHEUS_MULTIPROC_DIR=$PROM_DIR; mkdir -p \$PROMETHEUS_MULTIPROC_DIR
      setsid env PYTHONUNBUFFERED=1 $baseline_prof_env MAX_NUM_BATCHED_TOKENS=$bt DP_START_RANK=2 VLLM_ENABLE_PREFIX_CACHING=$pc DP_ADDRESS=$NODE0_IP \
        bash tools/benchmarks/prefill_launch_baseline_dp4tp8.sh \
        > $LOG_DIR/${label}_node1.log 2>&1 < /dev/null &
    " 2>/dev/null || true
    sleep 10
    itask exec "$NODE0" --tty=false -- bash -c "
      cd $REPO && mkdir -p $LOG_DIR
      export PROMETHEUS_MULTIPROC_DIR=$PROM_DIR; mkdir -p \$PROMETHEUS_MULTIPROC_DIR
      setsid env PYTHONUNBUFFERED=1 $baseline_prof_env MAX_NUM_BATCHED_TOKENS=$bt DP_START_RANK=0 VLLM_ENABLE_PREFIX_CACHING=$pc DP_ADDRESS=$NODE0_IP \
        bash tools/benchmarks/prefill_launch_baseline_dp4tp8.sh \
        > $LOG_DIR/${label}_node0.log 2>&1 < /dev/null &
    " 2>/dev/null || true
    SERVER_LABEL="$label"
  else
    itask exec "$NODE0" --tty=false -- bash -c "
      cd $REPO && mkdir -p $LOG_DIR
      export PROMETHEUS_MULTIPROC_DIR=$PROM_DIR; mkdir -p \$PROMETHEUS_MULTIPROC_DIR
      setsid env PYTHONUNBUFFERED=1 $attn_prof_env $varg MAX_NUM_BATCHED_TOKENS=$bt DP_START_RANK=0 VLLM_ENABLE_PREFIX_CACHING=$pc DP_ADDRESS=$NODE0_IP AFD_HOST=$NODE0_IP \
        bash tools/benchmarks/prefill_launch_afd_attention.sh \
        > $LOG_DIR/${label}_node0.log 2>&1 < /dev/null &
    " 2>/dev/null || true
    sleep 10
    itask exec "$NODE1" --tty=false -- bash -c "
      cd $REPO && mkdir -p $LOG_DIR
      export PROMETHEUS_MULTIPROC_DIR=$PROM_DIR; mkdir -p \$PROMETHEUS_MULTIPROC_DIR
      setsid env PYTHONUNBUFFERED=1 $attn_prof_env $varg MAX_NUM_BATCHED_TOKENS=$bt DP_START_RANK=2 ATTN_DEVICES=0,1,2,3,4,5,6,7 VLLM_ENABLE_PREFIX_CACHING=$pc DP_ADDRESS=$NODE0_IP AFD_HOST=$NODE0_IP \
        bash tools/benchmarks/prefill_launch_afd_attention.sh \
        > $LOG_DIR/${label}_node1_attn.log 2>&1 < /dev/null &
      sleep 3
      setsid env PYTHONUNBUFFERED=1 $ffn_prof_env $varg MAX_NUM_BATCHED_TOKENS=$bt VLLM_ENABLE_PREFIX_CACHING=$pc DP_ADDRESS=$NODE0_IP AFD_HOST=$NODE0_IP \
        bash tools/benchmarks/prefill_launch_afd_ffn.sh \
        > $LOG_DIR/${label}_node1_ffn.log 2>&1 < /dev/null &
    " 2>/dev/null || true
    SERVER_LABEL="$label"
  fi
}

cell_complete() {
  local sys_name=$1 bt=$2 rps=$3 prefix_ratio=$4 rd=$5
  local rps_enc="${rps//./p}" pr_enc="${prefix_ratio//./p}"
  local repeat
  for repeat in $(seq 1 "$RUN_REPEATS"); do
    local fname="${sys_name}-mbt${bt}-rps${rps_enc}-prefix${pr_enc}-repeat${repeat}.verified.json"
    # bare itask exec runs no shell; wrap the `test -f` in bash -c.
    if ! itask exec "$NODE0" --tty=false -- bash -c "test -f \"$REPO/$rd/$fname\"" 2>/dev/null; then
      return 1
    fi
  done
  return 0
}

# start_sampler <node> <csv> — background continuous npu-smi sampler on a pod.
start_sampler() {
  local node=$1 csv=$2
  echo "[SMI] starting npu-smi sampler on $node -> $(basename "$csv")"
  itask exec "$node" --tty=false -- bash -c "
    cd $REPO && mkdir -p \"\$(dirname $csv)\"
    setsid python3 -m tools.benchmarks.sample_npu_smi \
      --output $csv --interval $SAMPLER_INTERVAL --duration 0 \
      > $LOG_DIR/smi_$(basename "$csv" .csv).log 2>&1 < /dev/null &
  " 2>/dev/null || true
}

stop_sampler() {
  local node=$1
  echo "[SMI] stopping npu-smi sampler on $node"
  itask exec "$node" --tty=false -- bash -c 'pkill -f sample_npu_smi 2>/dev/null || true' \
    2>/dev/null || true
}

run_benchmark() {
  local config=$1 sys_name=$2 bt=$3 prefix_ratio=$4 repeat=$5
  local attempt=0 rc=1
  while [ $attempt -lt 2 ]; do
    echo "[BENCH] repeat=$repeat system=$sys_name bt=$bt prefix=$prefix_ratio config=$config"
    if itask_exec_retry "$NODE0" "
      cd \"$REPO\"
      python3 -m tools.benchmarks.prefill_experiment \
        --config $config run \
        --system $sys_name \
        --batch-tokens $bt \
        --prefix-ratio $prefix_ratio \
        --repeat $repeat \
        $( [ "$RESUME" = true ] && echo '--resume' )
    "; then
      rc=0; break
    fi
    attempt=$((attempt + 1))
    itask_exec_retry "$NODE0" "curl -sf http://127.0.0.1:8000/v1/models > /dev/null 2>&1" 2>/dev/null || {
      echo "[WARN] Server died, need restart"; return 1
    }
    sleep 15
  done
  return $rc
}

profile_request() {
  local node=$1 port=$2 action=$3
  itask_exec_retry "$node" \
    "curl --fail --silent --show-error --max-time 1800 -X POST http://127.0.0.1:$port/${action}_profile"
}

start_profile_window() {
  local system_label=$1
  if [ "$system_label" = "baseline" ]; then
    profile_request "$NODE0" 8000 start
    return
  fi
  if ! profile_request "$NODE1" 8001 start ||
     ! profile_request "$NODE0" 8000 start; then
    profile_request "$NODE0" 8000 stop || true
    profile_request "$NODE1" 8001 stop || true
    return 1
  fi
}

stop_profile_window() {
  local system_label=$1 rc=0
  if [ "$system_label" = "baseline" ]; then
    profile_request "$NODE0" 8000 stop
    return
  fi
  profile_request "$NODE0" 8000 stop || rc=1
  profile_request "$NODE1" 8001 stop || rc=1
  return $rc
}

# run_group <config> <label>
run_group() {
  local config=$1 label=$2
  local sys_name bt rps prefix_ratio rd
  read -r sys_name bt rps prefix_ratio rd <<< "$(result_stems "$config")"
  case "$sys_name" in *afd*) system_label=afd ;; *) system_label=baseline ;; esac
  local cache_state variant
  cache_state=$(cfg_stage2 "$config" cache_state)
  variant=$(cfg_stage2 "$config" variant)
  local pc=0; [ "$prefix_ratio" != "0" ] && pc=1

  # Profiler and correlation artifacts live under the run's result dir.
  if [ "$PROFILE" = "1" ] || [ "$CORRELATION" = "1" ]; then
    PROFILE_DIR="$REPO/$rd/traces"
  fi
  if [ "$PROFILE" = "1" ]; then
    echo "[PROFILE] request-controlled traces under $PROFILE_DIR"
  elif [ "$CORRELATION" = "1" ]; then
    echo "[CORRELATION] sidecars under $PROFILE_DIR/correlation"
  fi

  if [ "$RESUME" = true ] && cell_complete "$sys_name" "$bt" "$rps" "$prefix_ratio" "$rd"; then
    echo "[SKIP] $label (all repeats verified)"
    return 0
  fi

  echo ""
  echo "=============================================="
  echo "  [$label] system=$system_label($sys_name) bt=$bt rps=$rps pr=$prefix_ratio cache=$cache_state variant=${variant:-none}"
  echo "=============================================="

  # Server lifecycle: reuse when the key matches the previous group.
  local key="$system_label|$bt|$prefix_ratio|$variant|$cache_state"
  if [ "$PROFILE" = "1" ] || [ "$CORRELATION" = "1" ]; then
    key="$key|$label"
  fi
  if [ "$key" = "$PREV_KEY" ]; then
    echo "[REUSE] same group key — reset cache between cells"
    if ! reset_prefix_cache; then
      echo "[RESET-FAIL] restarting server for group"
      kill_all
      start_server "$system_label" "$bt" "$pc" "$variant" "$label"
      wait_server 900 || { echo "[FAIL] server not ready"; kill_all; PREV_KEY=""; return 1; }
    fi
  else
    kill_all
    start_server "$system_label" "$bt" "$pc" "$variant" "$label"
    wait_server 900 || { echo "[FAIL] server not ready"; kill_all; PREV_KEY=""; return 1; }
  fi
  PREV_KEY="$key"

  if [ "$PROFILE" = "1" ]; then
    start_sampler "$NODE0" "$REPO/$rd/telemetry/npu_smi_node0.csv"
    start_sampler "$NODE1" "$REPO/$rd/telemetry/npu_smi_node1.csv"
  fi

  local repeat
  for repeat in $(seq 1 "$RUN_REPEATS"); do
    if [ "$RESUME" = true ] && cell_complete "$sys_name" "$bt" "$rps" "$prefix_ratio" "$rd"; then
      echo "[SKIP] $label repeat $repeat (all verified)"
      continue
    fi
    if [ "$cache_state" = "steady" ]; then
      send_precondition "$prefix_ratio" || { echo "[FAIL] precondition"; kill_all; PREV_KEY=""; return 1; }
    elif [ "$cache_state" = "cold" ] && [ "$repeat" -gt 1 ]; then
      if [ "$prefix_ratio" != "0" ]; then
        # prefix>0 cold: prefix-cache blocks are retained, reset fails -> restart.
        echo "[COLD-RESTART] prefix>0 cold repeat $repeat"
        kill_all
        start_server "$system_label" "$bt" "$pc" "$variant" "$label"
        wait_server 900 || { echo "[FAIL] server not ready"; kill_all; PREV_KEY=""; return 1; }
      else
        reset_prefix_cache || {
          echo "[RESET-FAIL] restarting server for repeat $repeat"
          kill_all
          start_server "$system_label" "$bt" "$pc" "$variant" "$label"
          wait_server 900 || { echo "[FAIL] server not ready"; kill_all; PREV_KEY=""; return 1; }
        }
      fi
    fi
    if [ "$PROFILE" = "1" ] && ! start_profile_window "$system_label"; then
      echo "[FAIL] profiler start for repeat $repeat"
      kill_all
      PREV_KEY=""
      return 1
    fi
    if ! run_benchmark "$config" "$sys_name" "$bt" "$prefix_ratio" "$repeat"; then
      [ "$PROFILE" = "1" ] && stop_profile_window "$system_label" || true
      echo "[FAIL] benchmark repeat $repeat"
      kill_all
      PREV_KEY=""
      return 1
    fi
    if [ "$PROFILE" = "1" ] && ! stop_profile_window "$system_label"; then
      echo "[FAIL] profiler stop for repeat $repeat"
      kill_all
      PREV_KEY=""
      return 1
    fi
  done

  if [ "$PROFILE" = "1" ]; then
    stop_sampler "$NODE0"
    stop_sampler "$NODE1"
  fi
  echo "[DONE] $label"
}

# ---- Build default CELLS from PHASE ----
_build_cells() {
  local s
  case "$PHASE" in
    main)
      # Order so same-(system,bt) cells are adjacent for server reuse.
      for case in real-low-p0-mbt65536-rps4 real-high-p0-mbt65536-rps10 \
                  real-knee-p0-mbt32768-rps10 \
                  real-regress-p0-mbt8192-rps10 real-regress-healthy-p0-mbt8192-rps8; do
        for sys in dp4_tp8_sp afd_dp3_tp8_ep8; do
          s=$(find "bench_results/prefill_stage2/02_e2e/$case/$sys" \
                 -maxdepth 1 -name "stage2_e2e_*_${sys}.json" 2>/dev/null | head -1)
          [ -n "$s" ] && echo "$s"
        done
      done
      ;;
    ablation-knee)
      # Canonical layout: 03_ablation/<workload>/<variant>/stage2_ablation_*.json
      find bench_results/prefill_stage2/03_ablation/real-knee \
        -mindepth 2 -name 'stage2_ablation_*.json' | sort
      ;;
    ablation-longshort)
      find bench_results/prefill_stage2/03_ablation/long-short \
        -mindepth 2 -name 'stage2_ablation_*.json' | sort
      ;;
    prefix-cold)
      # Canonical layout: 04_prefix/<case>/<system>/<cold|steady>/stage2_prefix_*.json
      find bench_results/prefill_stage2/04_prefix \
        -mindepth 3 -path '*/cold/*' -name 'stage2_prefix_*.json' | sort
      ;;
    prefix-steady)
      find bench_results/prefill_stage2/04_prefix \
        -mindepth 3 -path '*/steady/*' -name 'stage2_prefix_*.json' | sort
      ;;
    btsweep)
      # Stage-3 L0 wide sweep, server-reuse-ordered by the generator.
      python3 -c 'import json; print(" ".join(json.load(open("bench_results/prefill_stage3/00_plan/sweep_grid.json"))["cells"]))'
      ;;
    btsweep-profile)
      # Stage-3 L2 profiler replays (PROFILE=1 forced in the arg parse below).
      python3 -c 'import json; print(" ".join(json.load(open("bench_results/prefill_stage3/00_plan/sweep_grid.json"))["profile_cells"]))'
      ;;
    *)
      echo "Unknown PHASE=$PHASE" >&2
      exit 2
      ;;
  esac
}

# ---- Arg Parsing ----
for arg in "$@"; do
  case "$arg" in
    --resume) RESUME=true ;;
    --smoke) SMOKE=true ;;
  esac
done

if [ -z "$CELLS" ] && [ -n "$PHASE" ]; then
  CELLS="$(_build_cells)"
fi
# btsweep-profile is the L2 profiler replay phase.
if [ "$PHASE" = "btsweep-profile" ]; then
  PROFILE=1
fi
if [ -z "$CELLS" ]; then
  echo "Set CELLS (space-separated config paths) or PHASE." >&2
  exit 2
fi

if [ "$SMOKE" = true ]; then
  CELLS="$(echo "$CELLS" | tr ' ' '\n' | head -1)"
fi

PREV_KEY=""
SERVER_LABEL=""
COUNT=0
for config in $CELLS; do
  [ -f "$config" ] || { echo "[WARN] missing config $config"; continue; }
  COUNT=$((COUNT + 1))
  label="s2_${PHASE:-manual}_$(basename "$(dirname "$config")")_$(basename "$config" .json)"
  run_group "$config" "$label"
done

kill_all
echo "=============================================="
echo "  Phase ${PHASE:-manual} complete: $COUNT groups processed"
echo "=============================================="
