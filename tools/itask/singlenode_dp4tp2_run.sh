#!/usr/bin/env bash
# Single-node 16-card DSV4 AFD e2e perf, attention layout DP4TP2 (+ DP8EP8 FFN).
# One server per invocation on ONE pod; replays formal_1 at 0.5x / 0.75x / 1x
# (slow2x / slow1p33x / formal_1 plans, offered 17.5K / 26.3K / 35.1K tok/s).
#
# Usage: POD=v4f-base-2 NODE1_IP=<pod ip> singlenode_dp4tp2_run.sh
# Env: MBT=65536 HCCL_BUFFSIZE=1024 FLASHCOMM1=1
# Note: HCCL_BUFFSIZE=1024 is the validated minimum for 8 attention ranks at
# mbt=65536 (512 fails CAM dispatch tiling); 2048 also works but 1024 is the
# measured configuration.
set -uo pipefail

POD=${POD:?set POD to the task name (e.g. v4f-base-2)}
NODE1_IP=${NODE1_IP:?set NODE1_IP to the pod IP}
CODE=/a3_inference/itask/workdir/tq02357756/shwstone/code/afd-plugin
LAUNCHER=$CODE/tools/itask/launch_dsv4_afd_single_node.sh
WL=$CODE/tools/datasets/moonconv-wildchat-v4-flash-prefill/workloads
NASDIR=/a3_inference/itask/workdir/tq02357756/shwstone/singlenode_dp4tp2
MBT=${MBT:-65536}
TAG=singlenode_dp4tp2_mbt${MBT}

X="timeout --signal=KILL 120 itask exec"
S="timeout --signal=KILL 45 itask exec"
say() { echo "[$(date +%H:%M:%S)] $*"; }

say "== cleanup $POD (pods stay up; only vllm processes die) =="
$S "$POD" -- bash /tmp/xnode32_cleanup.sh 2>&1 | tr -d '\r' | tail -2

ENV1="NODE1_IP=$NODE1_IP NODE2_IP=$NODE1_IP ATTN_DP_SIZE=4 ATTN_TP_SIZE=2 MODEL_PATH=/home/admin/model-csi/model MAX_NUM_BATCHED_TOKENS=$MBT HCCL_BUFFSIZE=${HCCL_BUFFSIZE:-1024} FLASHCOMM1=${FLASHCOMM1:-1} DSV4_SHARED_COMPRESSOR_WORKSPACE=1 ASYNC_MOE_UBATCHING=true ASYNC_MOE_SPLIT=request ${EXTRA_ENV:-}"
say "== start stack (dp4tp2+ep8, mbt=$MBT, HCCL_BUFFSIZE=${HCCL_BUFFSIZE:-1024}, FLASHCOMM1=1, async-sched ON) =="
# FFN first, attention 20s later: two vllm instances on one pod race
# get_open_port() for their DP store when started together (observed 2026-09-02);
# this stagger is the validated single-node recipe.
$S "$POD" -- bash -c "setsid bash -c 'env $ENV1 NODE_IP=$NODE1_IP ROLE=ffn nohup bash $LAUNCHER > /tmp/${TAG}_ffn.log 2>&1 &' ; echo started" 2>&1 | tr -d '\r' | tail -1
sleep 20
$S "$POD" -- bash -c "setsid bash -c 'env $ENV1 NODE_IP=$NODE1_IP ROLE=attention nohup bash $LAUNCHER > /tmp/${TAG}_attn.log 2>&1 &' ; echo started" 2>&1 | tr -d '\r' | tail -1

say "== wait readiness =="
READY=0
for i in $(seq 1 80); do
  sleep 30
  OUT=$($S "$POD" -- bash -c 'curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8900/v1/models' 2>/dev/null | tr -d '\r' | tail -1)
  if [[ "$OUT" == *200* ]]; then READY=1; break; fi
  C1=$($S "$POD" -- bash -c "grep -cE 'do tiling failed|507015|507057|Traceback|ValidationError|ValueError' /tmp/${TAG}_attn.log 2>/dev/null || true" 2>/dev/null | tr -d '\r' | tail -1)
  C2=$($S "$POD" -- bash -c "grep -cE 'do tiling failed|507015|507057|Traceback|ValidationError|ValueError' /tmp/${TAG}_ffn.log 2>/dev/null || true" 2>/dev/null | tr -d '\r' | tail -1)
  CRASH=$(( ${C1:-0} + ${C2:-0} ))
  if [[ "$CRASH" != "0" ]]; then
    say "!! crash signature (count=$CRASH)"
    $S "$POD" -- bash -c "tail -20 /tmp/${TAG}_attn.log; echo ----; grep -E 'ERROR|Error' /tmp/${TAG}_ffn.log | tail -8" 2>&1 | tr -d '\r' | tail -28
    exit 1
  fi
done
[[ $READY == 1 ]] || { say "!! never ready"; exit 1; }
say "ready; sanity checks:"
$S "$POD" -- bash -c "grep -c 'Enabled DeepSeek-V4 persistent compressor tails' /tmp/${TAG}_attn.log || true; grep -o '\"attn_ranks_per_dp\":[0-9]*' /tmp/${TAG}_attn.log | head -1; grep -o 'tensor_parallel_size=[0-9]*' /tmp/${TAG}_attn.log | head -1" 2>&1 | tr -d '\r' | tail -3
$S "$POD" -- bash -c "grep -c 'AFD FFN EngineCore started' /tmp/${TAG}_ffn.log || true" 2>&1 | tr -d '\r' | tail -1

say "== smoke 2 requests =="
$X "$POD" -- bash -c '
  for i in 1 2; do
    curl -s -m 120 http://127.0.0.1:8900/v1/completions -H "Content-Type: application/json" \
      -d "{\"model\":\"dsv4-afd-attention\",\"prompt\":[1,2,3,4,5,6,7,8],\"max_tokens\":1}" -o /dev/null -w "smoke$i=%{http_code} "
  done; echo' 2>&1 | tr -d '\r' | tail -1

run_plan() { # $1=plan file, $2=output tag
  say "== replay $2 =="
  $S "$POD" -- bash -c "setsid bash -c 'cd $CODE && nohup python3 tools/benchmarks/mw_replay_client.py --base-url http://127.0.0.1:8900 --model dsv4-afd-attention --requests $WL/formal_1_requests.jsonl --plan $WL/$1 --output /tmp/${TAG}_$2.json > /tmp/${TAG}_$2.client.log 2>&1 &'; echo launched" 2>&1 | tr -d '\r' | tail -1
  local DONE=0
  for i in $(seq 1 90); do
    sleep 30
    local OUT
    OUT=$($S "$POD" -- bash -c "test -f /tmp/${TAG}_$2.json && echo EXISTS || echo NO" 2>/dev/null | tr -d '\r' | tail -1)
    if [[ "$OUT" == *EXISTS* ]]; then DONE=1; break; fi
  done
  if [[ $DONE != 1 ]]; then
    say "!! $2 client did not finish in 45min"
    $S "$POD" -- bash -c "tail -20 /tmp/${TAG}_$2.client.log" 2>&1 | tr -d '\r' | tail -20
    exit 1
  fi
  say "== $2 done: $($S "$POD" -- bash -c "grep -o 'REPLAY_OK[^{]*' /tmp/${TAG}_$2.client.log | head -1" 2>/dev/null | tr -d '\r' | tail -1)"
}

run_plan formal_1_slow2x_plan.json slow2x
sleep 30
run_plan formal_1_slow1p33x_plan.json slow1p33x
sleep 30
run_plan formal_1_plan.json 1x

say "== archive =="
$S "$POD" -- bash -c "
  mkdir -p $NASDIR
  cp /tmp/${TAG}_slow2x.json /tmp/${TAG}_slow1p33x.json /tmp/${TAG}_1x.json /tmp/${TAG}_attn.log /tmp/${TAG}_ffn.log $NASDIR/
  echo archived" 2>&1 | tr -d '\r' | tail -1
say "== ALL_COMPLETE dp4tp2 mbt=$MBT =="
