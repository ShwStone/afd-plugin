#!/usr/bin/env bash
# One server (mbt=65536, HCCL_BUFFSIZE=4096), two fast-forward replays.
# Usage: xnode32_speed_runs.sh
set -uo pipefail

MASTER=${MASTER:-v4f-base-2}
SECOND=${SECOND:-v4f-xnode-2}
NODE1_IP=${NODE1_IP:-33.182.141.199}
NODE2_IP=${NODE2_IP:-33.182.140.47}
CODE=/a3_inference/itask/workdir/tq02357756/shwstone/code/afd-plugin
LAUNCHER=$CODE/tools/itask/launch_dsv4_afd_xnode32.sh
WL=$CODE/tools/datasets/moonconv-wildchat-v4-flash-prefill/workloads
TAG=xnode32_mbt65536

X="timeout --signal=KILL 120 itask exec"
S="timeout --signal=KILL 45 itask exec"
say() { echo "[$(date +%H:%M:%S)] $*"; }

say "== cleanup both pods =="
$S "$MASTER" -- bash /tmp/xnode32_cleanup.sh 2>&1 | tr -d '\r' | tail -2
$S "$SECOND" -- bash /tmp/xnode32_cleanup.sh 2>&1 | tr -d '\r' | tail -2

ENV1="NODE1_IP=$NODE1_IP NODE2_IP=$NODE2_IP MAX_NUM_BATCHED_TOKENS=65536 HCCL_BUFFSIZE=4096"
say "== start stack (mbt=65536, HCCL_BUFFSIZE=4096) =="
$S "$MASTER" -- bash -c "setsid bash -c 'env $ENV1 NODE_IP=$NODE1_IP ROLE=attention ATTENTION_NODE_ID=1 nohup bash $LAUNCHER > /tmp/${TAG}_attn1.log 2>&1 &' ; echo started" 2>&1 | tr -d '\r' | tail -1
$S "$SECOND" -- bash -c "setsid bash -c 'env $ENV1 NODE_IP=$NODE2_IP ROLE=attention ATTENTION_NODE_ID=2 nohup bash $LAUNCHER > /tmp/${TAG}_attn2.log 2>&1 &' ; echo started" 2>&1 | tr -d '\r' | tail -1
# Stagger FFN behind headless attention on node2: both vllm instances race
# get_open_port() for their DP store; simultaneous starts can EADDRINUSE and
# hang the FFN DP group mid-init (observed 2026-09-02).
sleep 20
$S "$SECOND" -- bash -c "setsid bash -c 'env $ENV1 NODE_IP=$NODE2_IP ROLE=ffn nohup bash $LAUNCHER > /tmp/${TAG}_ffn.log 2>&1 &' ; echo started" 2>&1 | tr -d '\r' | tail -1

say "== wait readiness =="
READY=0
for i in $(seq 1 80); do
  sleep 30
  OUT=$($S "$MASTER" -- bash -c 'curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8900/v1/models' 2>/dev/null | tr -d '\r' | tail -1)
  if [[ "$OUT" == *200* ]]; then READY=1; break; fi
  C1=$($S "$MASTER" -- bash -c "grep -cE 'do tiling failed|507015|507057|Traceback|ValidationError' /tmp/${TAG}_attn1.log 2>/dev/null || true" 2>/dev/null | tr -d '\r' | tail -1)
  C2=$($S "$SECOND" -- bash -c "cat /tmp/${TAG}_attn2.log /tmp/${TAG}_ffn.log 2>/dev/null | grep -cE 'do tiling failed|507015|507057|Traceback|ValidationError' || true" 2>/dev/null | tr -d '\r' | tail -1)
  CRASH=$(( ${C1:-0} + ${C2:-0} ))
  if [[ "$CRASH" != "0" ]]; then
    say "!! crash signature (count=$CRASH)"
    $S "$SECOND" -- bash -c "grep -E 'ERROR|Error' /tmp/${TAG}_ffn.log | tail -5" 2>&1 | tr -d '\r' | tail -5
    exit 1
  fi
done
[[ $READY == 1 ]] || { say "!! never ready"; exit 1; }
say "ready"

say "== smoke 2 requests =="
$X "$MASTER" -- bash -c '
  for i in 1 2; do
    curl -s -m 120 http://127.0.0.1:8900/v1/completions -H "Content-Type: application/json" \
      -d "{\"model\":\"dsv4-afd-attention\",\"prompt\":[1,2,3,4,5,6,7,8],\"max_tokens\":1}" -o /dev/null -w "smoke$i=%{http_code} "
  done; echo' 2>&1 | tr -d '\r' | tail -1

run_plan() { # $1=plan file, $2=output tag
  say "== replay $2 =="
  $S "$MASTER" -- bash -c "setsid bash -c 'cd $CODE && nohup python3 tools/benchmarks/mw_replay_client.py --base-url http://127.0.0.1:8900 --model dsv4-afd-attention --requests $WL/formal_1_requests.jsonl --plan $WL/$1 --output /tmp/${TAG}_$2.json > /tmp/${TAG}_$2.client.log 2>&1 &'; echo launched" 2>&1 | tr -d '\r' | tail -1
  local DONE=0
  for i in $(seq 1 60); do
    sleep 30
    OUT=$($S "$MASTER" -- bash -c "test -f /tmp/${TAG}_$2.json && echo EXISTS || echo NO" 2>/dev/null | tr -d '\r' | tail -1)
    if [[ "$OUT" == *EXISTS* ]]; then DONE=1; break; fi
  done
  if [[ $DONE != 1 ]]; then
    say "!! $2 client did not finish in 30min"
    $S "$MASTER" -- bash -c "tail -20 /tmp/${TAG}_$2.client.log" 2>&1 | tr -d '\r' | tail -20
    exit 1
  fi
  say "== $2 done =="
}

run_plan formal_1_fast1p5x_plan.json fast1p5x
sleep 30   # let the server settle between runs
run_plan formal_1_fast2x_plan.json fast2x

say "== archive =="
$S "$MASTER" -- bash -c '
  D=/a3_inference/itask/workdir/tq02357756/shwstone/xnode32_results
  cp /tmp/xnode32_mbt65536_fast1p5x.json /tmp/xnode32_mbt65536_fast2x.json $D/
  echo archived' 2>&1 | tr -d '\r' | tail -1
$S "$SECOND" -- bash -c '
  D=/a3_inference/itask/workdir/tq02357756/shwstone/xnode32_results
  cp /tmp/xnode32_mbt65536_attn2.log /tmp/xnode32_mbt65536_ffn.log $D/ 2>/dev/null
  echo archived' 2>&1 | tr -d '\r' | tail -1
$S "$MASTER" -- bash -c 'cp /tmp/xnode32_mbt65536_attn1.log /a3_inference/itask/workdir/tq02357756/shwstone/xnode32_results/; echo archived' 2>&1 | tr -d '\r' | tail -1
say "== ALL_COMPLETE =="
