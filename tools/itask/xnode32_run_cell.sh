#!/usr/bin/env bash
# Drive one mbt cell of the dual-node 32-card DSV4 AFD sweep.
# Usage: xnode32_run_cell.sh <MBT>   (e.g. 8192 16384 32768 65536)
# Env: MASTER=v4f-base-2 SECOND=v4f-xnode-2 NODE1_IP=.. NODE2_IP=.. (defaults below)
set -uo pipefail

MBT=${1:?usage: xnode32_run_cell.sh <mbt>}
MASTER=${MASTER:-v4f-base-2}
SECOND=${SECOND:-v4f-xnode-2}
NODE1_IP=${NODE1_IP:-33.182.141.199}
NODE2_IP=${NODE2_IP:?set NODE2_IP to the second pod IP}
CODE=/a3_inference/itask/workdir/tq02357756/shwstone/code/afd-plugin
LAUNCHER=$CODE/tools/itask/launch_dsv4_afd_xnode32.sh
RESULT=/tmp/xnode32_mbt${MBT}.json
TAG=xnode32_mbt${MBT}

X="timeout --signal=KILL 120 itask exec"
S="timeout --signal=KILL 45 itask exec"

say() { echo "[$(date +%H:%M:%S)] $*"; }

cleanup_pod() { # $1=pod — cleanup script lives on the pod so the exec
  # cmdline carries no vllm-ish strings for pkill to self-match (exit 137).
  $S "$1" -- bash /tmp/xnode32_cleanup.sh 2>&1 | tr -d '\r' | tail -3
}

say "== cell mbt=$MBT: cleanup both pods =="
cleanup_pod "$MASTER"
cleanup_pod "$SECOND"

ENV1="NODE1_IP=$NODE1_IP NODE2_IP=$NODE2_IP MAX_NUM_BATCHED_TOKENS=$MBT HCCL_BUFFSIZE=${HCCL_BUFFSIZE:-2048}"
say "== start attention node1 (master) =="
$S "$MASTER" -- bash -c "setsid bash -c 'env $ENV1 NODE_IP=$NODE1_IP ROLE=attention ATTENTION_NODE_ID=1 nohup bash $LAUNCHER > /tmp/${TAG}_attn1.log 2>&1 &' ; echo started" 2>&1 | tail -1
say "== start attention node2 (headless) + ffn =="
$S "$SECOND" -- bash -c "setsid bash -c 'env $ENV1 NODE_IP=$NODE2_IP ROLE=attention ATTENTION_NODE_ID=2 nohup bash $LAUNCHER > /tmp/${TAG}_attn2.log 2>&1 &' ; echo started" 2>&1 | tail -1
$S "$SECOND" -- bash -c "setsid bash -c 'env $ENV1 NODE_IP=$NODE2_IP ROLE=ffn nohup bash $LAUNCHER > /tmp/${TAG}_ffn.log 2>&1 &' ; echo started" 2>&1 | tail -1

say "== wait for readiness (poll /v1/models, up to 40min) =="
READY=0
for i in $(seq 1 80); do
  sleep 30
  OUT=$($S "$MASTER" -- bash -c 'curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8900/v1/models' 2>/dev/null | tr -d '\r' | tail -1)
  if [[ "$OUT" == *200* ]]; then READY=1; break; fi
  # early-fail on crash signatures
  CRASH=$($S "$MASTER" -- bash -c "grep -cE 'do tiling failed|507015|507057|Traceback|ValidationError' /tmp/${TAG}_attn1.log 2>/dev/null || true" 2>/dev/null | tr -d '\r' | tail -1)
  CRASH2=$($S "$SECOND" -- bash -c "cat /tmp/${TAG}_attn2.log /tmp/${TAG}_ffn.log 2>/dev/null | grep -cE 'do tiling failed|507015|507057|Traceback|ValidationError' || true" 2>/dev/null | tr -d '\r' | tail -1)
  CRASH=$(( ${CRASH:-0} + ${CRASH2:-0} ))
  if [[ "${CRASH:-0}" != "0" && -n "${CRASH:-}" ]]; then
    say "!! crash signature in attn1 log (count=$CRASH), last lines:"
    $S "$MASTER" -- bash -c "tail -30 /tmp/${TAG}_attn1.log" 2>&1 | tail -30
    exit 1
  fi
done
if [[ $READY != 1 ]]; then say "!! never became ready"; exit 1; fi
say "ready. CWS check:"
$S "$MASTER" -- bash -c "grep -c 'Enabled DeepSeek-V4 persistent compressor tails' /tmp/${TAG}_attn1.log; grep -c 'AFD FFN EngineCore started' /tmp/${TAG}_ffn.log 2>/dev/null" 2>&1 | tail -3

say "== smoke: 2 short requests (collective warmup) =="
$X "$MASTER" -- bash -c '
  for i in 1 2; do
    curl -s -m 120 http://127.0.0.1:8900/v1/completions -H "Content-Type: application/json" \
      -d "{\"model\":\"dsv4-afd-attention\",\"prompt\":[1,2,3,4,5,6,7,8],\"max_tokens\":1}" -o /dev/null -w "smoke$i=%{http_code} "
  done; echo' 2>&1 | tail -1

say "== replay formal_1 original-speed (150s window) =="
$S "$MASTER" -- bash -c "setsid bash -c 'cd $CODE && nohup python3 tools/benchmarks/mw_replay_client.py --base-url http://127.0.0.1:8900 --model dsv4-afd-attention --requests tools/datasets/moonconv-wildchat-v4-flash-prefill/workloads/formal_1_requests.jsonl --plan tools/datasets/moonconv-wildchat-v4-flash-prefill/workloads/formal_1_plan.json --output $RESULT > /tmp/${TAG}_client.log 2>&1 &'; echo launched" 2>&1 | tail -1

DONE=0
for i in $(seq 1 60); do
  sleep 30
  OUT=$($S "$MASTER" -- bash -c "test -f $RESULT && echo EXISTS || echo NO" 2>/dev/null | tr -d '\r' | tail -1)
  if [[ "$OUT" == *EXISTS* ]]; then DONE=1; break; fi
done
if [[ $DONE != 1 ]]; then
  say "!! client did not finish in 30min; client log tail:"
  $S "$MASTER" -- bash -c "tail -20 /tmp/${TAG}_client.log" 2>&1 | tail -20
  exit 1
fi
say "== cell done; archiving =="
$S "$MASTER" -- bash -c "
  D=/a3_inference/itask/workdir/tq02357756/shwstone/xnode32_results
  mkdir -p \$D
  cp $RESULT \$D/ 2>/dev/null
  for f in /tmp/${TAG}_attn1.log; do cp \$f \$D/ 2>/dev/null; done
  echo archived" 2>&1 | tail -1
$S "$SECOND" -- bash -c "
  D=/a3_inference/itask/workdir/tq02357756/shwstone/xnode32_results
  mkdir -p \$D
  cp /tmp/${TAG}_attn2.log /tmp/${TAG}_ffn.log /tmp/${TAG}_client.log \$D/ 2>/dev/null
  echo archived" 2>&1 | tail -1
say "== mbt=$MBT CELL_COMPLETE =="
