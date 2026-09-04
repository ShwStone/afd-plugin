#!/usr/bin/env bash
# Single-node AFD DP4TP2+EP8, token-split ubatching, rate sweep 0.5x -> 1.5x
# (0.25x steps) at mbt {8192, 32768}.  Companion to baseline1x_async_runs.sh:
# same rates, same mbt grid, async-sched ON, CWS on, FLASHCOMM1 on — only the
# serving architecture differs (and mbt within a rate, as on the baseline side).
# ASYNC_MOE_SPLIT=token; all other knobs match the dp4tp2_mbt65536 run.
#
# Usage: POD=v4f-base-2 NODE1_IP=<pod ip> singlenode_dp4tp2_token_runs.sh
set -uo pipefail

POD=${POD:?set POD}
NODE1_IP=${NODE1_IP:?set NODE1_IP to the pod IP}
CODE=/a3_inference/itask/workdir/tq02357756/shwstone/code/afd-plugin
LAUNCHER=$CODE/tools/itask/launch_dsv4_afd_single_node.sh
WL=$CODE/tools/datasets/moonconv-wildchat-v4-flash-prefill/workloads
NASDIR=/a3_inference/itask/workdir/tq02357756/shwstone/singlenode_dp4tp2_token

X="timeout --signal=KILL 120 itask exec"
S="timeout --signal=KILL 45 itask exec"
say() { echo "[$(date +%H:%M:%S)] $*"; }

run_plan() { # $1=mbt $2=plan file $3=output tag
  local MBT=$1 TAG=dp4tp2tok_mbt${MBT}_$3
  say "== replay mbt=$MBT $3 =="
  $S "$POD" -- bash -c "setsid bash -c 'cd $CODE && nohup python3 tools/benchmarks/mw_replay_client.py --base-url http://127.0.0.1:8900 --model dsv4-afd-attention --requests $WL/formal_1_requests.jsonl --plan $WL/$2 --output /tmp/${TAG}.json > /tmp/${TAG}.client.log 2>&1 &'; echo launched" 2>&1 | tr -d '\r' | tail -1
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
  TAG=dp4tp2tok_mbt${MBT}
  say "== mbt=$MBT: cleanup + start AFD dp4tp2 token-split =="
  $S "$POD" -- bash /tmp/xnode32_cleanup.sh 2>&1 | tr -d '\r' | tail -2
  ENV1="NODE1_IP=$NODE1_IP NODE2_IP=$NODE1_IP ATTN_DP_SIZE=4 ATTN_TP_SIZE=2 MODEL_PATH=/home/admin/model-csi/model MAX_NUM_BATCHED_TOKENS=$MBT HCCL_BUFFSIZE=4096 FLASHCOMM1=1 DSV4_SHARED_COMPRESSOR_WORKSPACE=1 ASYNC_MOE_UBATCHING=true ASYNC_MOE_SPLIT=token ASYNC_SCHEDULING=1"
  # FFN first, attention 20s later (single-pod DP store port race).
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
  [[ $READY == 1 ]] || { say "!! never ready mbt=$MBT"; exit 1; }
  say "ready; sanity checks:"
  $S "$POD" -- bash -c "grep -c 'Enabled DeepSeek-V4 persistent compressor tails' /tmp/${TAG}_attn.log || true; grep -o \"async_moe_split[^,}]*\" /tmp/${TAG}_attn.log | head -1" 2>&1 | tr -d '\r' | tail -2
  $S "$POD" -- bash -c "grep -c 'AFD FFN EngineCore started' /tmp/${TAG}_ffn.log || true" 2>&1 | tr -d '\r' | tail -1

  say "== smoke 2 requests =="
  $X "$POD" -- bash -c '
    for i in 1 2; do
      curl -s -m 120 http://127.0.0.1:8900/v1/completions -H "Content-Type: application/json" \
        -d "{\"model\":\"dsv4-afd-attention\",\"prompt\":[1,2,3,4,5,6,7,8],\"max_tokens\":1}" -o /dev/null -w "smoke$i=%{http_code} "
    done; echo' 2>&1 | tr -d '\r' | tail -1

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
    cp /tmp/${TAG}_0p5x.json /tmp/${TAG}_0p75x.json /tmp/${TAG}_1x.json /tmp/${TAG}_1p25x.json /tmp/${TAG}_1p5x.json /tmp/${TAG}_attn.log /tmp/${TAG}_ffn.log $NASDIR/ 2>/dev/null
    echo archived" 2>&1 | tr -d '\r' | tail -1
done
say "== ALL_COMPLETE dp4tp2 token sweep =="
