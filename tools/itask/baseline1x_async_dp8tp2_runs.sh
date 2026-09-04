#!/usr/bin/env bash
# Single-instance stock baseline, topology DP8TP2EP16 (16 cards, one pod),
# async-sched ON, rate sweep 0.5x -> 1.5x (0.25x steps) at mbt {8192, 32768}.
# Identical to baseline1x_async_runs.sh (same rates, same knobs, same mbt
# grid) except the topology: DP_SIZE=8 TP_SIZE=2 instead of DP4TP4 — the
# DP8TP2-vs-DP4TP4 comparison isolates the DP/TP split at equal card count.
#
# Usage: POD=v4f-base-2 baseline1x_async_dp8tp2_runs.sh
set -uo pipefail

POD=${POD:?set POD}
CODE=/a3_inference/itask/workdir/tq02357756/shwstone/code/afd-plugin
BASE_LAUNCHER=$CODE/tools/benchmarks/v4_launch_baseline.sh
PY=/usr/local/python3.12.13/bin/python3
WL=$CODE/tools/datasets/moonconv-wildchat-v4-flash-prefill/workloads
NASDIR=/a3_inference/itask/workdir/tq02357756/shwstone/baseline1x_async_dp8tp2

X="timeout --signal=KILL 120 itask exec"
S="timeout --signal=KILL 45 itask exec"
say() { echo "[$(date +%H:%M:%S)] $*"; }

run_plan() { # $1=mbt $2=plan file $3=output tag
  local MBT=$1 TAG=base1xasdp8tp2_mbt${MBT}_$3
  say "== replay mbt=$MBT $3 =="
  $S "$POD" -- bash -c "setsid bash -c 'cd $CODE && nohup $PY tools/benchmarks/mw_replay_client.py --base-url http://127.0.0.1:8000 --model deepseek_v4_flash --requests $WL/formal_1_requests.jsonl --plan $WL/$2 --output /tmp/${TAG}.json > /tmp/${TAG}.client.log 2>&1 &'; echo launched" 2>&1 | tr -d '\r' | tail -1
  local DONE=0
  for i in $(seq 1 120); do
    sleep 30
    local OUT
    OUT=$($S "$POD" -- bash -c "test -f /tmp/${TAG}.json && echo EXISTS || echo NO" 2>/dev/null | tr -d '\r' | tail -1)
    if [[ "$OUT" == *EXISTS* ]]; then DONE=1; break; fi
  done
  if [[ $DONE != 1 ]]; then
    say "!! $TAG client did not finish in 60min"
    $S "$POD" -- bash -c "tail -20 /tmp/${TAG}.client.log" 2>&1 | tr -d '\r' | tail -20
    return 1
  fi
  say "== $TAG done: $($S "$POD" -- bash -c "grep -o 'REPLAY_OK[^{]*' /tmp/${TAG}.client.log | head -1" 2>/dev/null | tr -d '\r' | tail -1)"
}

for MBT in 8192 32768; do
  TAG=base1xasdp8tp2_mbt${MBT}
  say "== mbt=$MBT: cleanup + start baseline DP8TP2 (ASYNC_SCHEDULING=1) =="
  $S "$POD" -- bash /tmp/xnode32_cleanup.sh 2>&1 | tr -d '\r' | tail -2
  $S "$POD" -- bash -c "setsid bash -c 'env DP_SIZE=8 TP_SIZE=2 MAX_NUM_BATCHED_TOKENS=$MBT PORT=8000 ADDITIONAL_CONFIG=cws ASYNC_SCHEDULING=1 nohup bash $BASE_LAUNCHER > /tmp/${TAG}.log 2>&1 &' ; echo started" 2>&1 | tr -d '\r' | tail -1

  say "== wait readiness =="
  READY=0
  for i in $(seq 1 80); do
    sleep 30
    OUT=$($S "$POD" -- bash -c 'curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8000/v1/models' 2>/dev/null | tr -d '\r' | tail -1)
    if [[ "$OUT" == *200* ]]; then READY=1; break; fi
    C=$($S "$POD" -- bash -c "grep -cE 'do tiling failed|507015|507057|Traceback|ValidationError|less than desired' /tmp/${TAG}.log 2>/dev/null || true" 2>/dev/null | tr -d '\r' | tail -1)
    if [[ "${C:-0}" != "0" ]]; then
      say "!! crash signature (count=$C)"
      $S "$POD" -- bash -c "grep -E 'ERROR|Error|Traceback' /tmp/${TAG}.log | tail -8" 2>&1 | tr -d '\r' | tail -8
      exit 1
    fi
  done
  [[ $READY == 1 ]] || { say "!! never ready mbt=$MBT"; exit 1; }
  say "ready; sanity checks:"
  $S "$POD" -- bash -c "grep -c 'Enabled DeepSeek-V4 persistent compressor tails' /tmp/${TAG}.log || true; grep -o 'data_parallel_size=[0-9]*' /tmp/${TAG}.log | head -1; grep -o 'tensor_parallel_size=[0-9]*' /tmp/${TAG}.log | head -1" 2>&1 | tr -d '\r' | tail -3

  say "== smoke =="
  $X "$POD" -- bash -c 'curl -s -m 120 http://127.0.0.1:8000/v1/completions -H "Content-Type: application/json" -d "{\"model\":\"deepseek_v4_flash\",\"prompt\":[1,2,3,4,5,6,7,8],\"max_tokens\":1}" -o /dev/null -w "smoke=%{http_code}\n"' 2>&1 | tr -d '\r' | tail -1

  run_plan $MBT formal_1_slow2x_plan.json 0p5x
  sleep 30
  run_plan $MBT formal_1_slow1p33x_plan.json 0p75x
  sleep 30
  run_plan $MBT formal_1_plan.json 1x
  sleep 30
  run_plan $MBT formal_1_fast1p25x_plan.json 1p25x
  sleep 30
  run_plan $MBT formal_1_fast1p5x_plan.json 1p5x

  say "== archive mbt=$MBT =="
  $S "$POD" -- bash -c "
    mkdir -p $NASDIR
    cp /tmp/${TAG}_0p5x.json /tmp/${TAG}_0p75x.json /tmp/${TAG}_1x.json /tmp/${TAG}_1p25x.json /tmp/${TAG}_1p5x.json /tmp/${TAG}.log $NASDIR/ 2>/dev/null
    echo archived" 2>&1 | tr -d '\r' | tail -1
done
say "== ALL_COMPLETE baseline dp8tp2 async sweep =="
