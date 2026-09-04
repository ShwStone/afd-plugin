#!/usr/bin/env bash
# Two independent AFD instances (one per 16-card pod), each DP4TP2+EP8
# token-split mbt=65536, behind the least-load router; formal_1 at 1x/1.5x/2x.
# Instance-level balance: the router polls each instance's /metrics
# (num_requests_waiting/running summed across DP engines) and routes to the
# least-loaded — same pattern as xnode32_baseline2x_runs.sh.
#
# Usage: NODE1_IP=<ip> NODE2_IP=<ip> singlenode2x_dp4tp2_tok_runs.sh
# Env: MASTER=v4f-base-2 SECOND=v4f-xnode-2
set -uo pipefail

MASTER=${MASTER:-v4f-base-2}
SECOND=${SECOND:-v4f-xnode-2}
NODE1_IP=${NODE1_IP:?set NODE1_IP}
NODE2_IP=${NODE2_IP:?set NODE2_IP}
CODE=/a3_inference/itask/workdir/tq02357756/shwstone/code/afd-plugin
LAUNCHER=$CODE/tools/itask/launch_dsv4_afd_single_node.sh
ROUTER=$CODE/tools/benchmarks/least_load_router.py
PY=/usr/local/python3.12.13/bin/python3
WL=$CODE/tools/datasets/moonconv-wildchat-v4-flash-prefill/workloads
NASDIR=/a3_inference/itask/workdir/tq02357756/shwstone/singlenode2x_dp4tp2_tok
TAG=singlenode2x_dp4tp2tok
ROUTER_PORT=8800

X="timeout --signal=KILL 120 itask exec"
S="timeout --signal=KILL 45 itask exec"
say() { echo "[$(date +%H:%M:%S)] $*"; }

ENV1="NODE1_IP=$NODE1_IP NODE2_IP=$NODE1_IP ATTN_DP_SIZE=4 ATTN_TP_SIZE=2 MODEL_PATH=/home/admin/model-csi/model MAX_NUM_BATCHED_TOKENS=65536 HCCL_BUFFSIZE=4096 FLASHCOMM1=1 DSV4_SHARED_COMPRESSOR_WORKSPACE=1 ASYNC_MOE_UBATCHING=true ASYNC_MOE_SPLIT=token ASYNC_SCHEDULING=1"

say "== cleanup both pods =="
$S "$MASTER" -- bash /tmp/xnode32_cleanup.sh 2>&1 | tr -d '\r' | tail -2
$S "$SECOND" -- bash /tmp/xnode32_cleanup.sh 2>&1 | tr -d '\r' | tail -2

start_stack() { # $1=pod $2=node_ip $3=suffix
  $S "$1" -- bash -c "setsid bash -c 'env $ENV1 NODE_IP=$2 ROLE=ffn nohup bash $LAUNCHER > /tmp/${TAG}_$3_ffn.log 2>&1 &' ; echo started" 2>&1 | tr -d '\r' | tail -1
  sleep 20
  $S "$1" -- bash -c "setsid bash -c 'env $ENV1 NODE_IP=$2 ROLE=attention nohup bash $LAUNCHER > /tmp/${TAG}_$3_attn.log 2>&1 &' ; echo started" 2>&1 | tr -d '\r' | tail -1
}

wait_ready() { # $1=pod $2=suffix
  local POD=$1 SUF=$2
  for i in $(seq 1 80); do
    sleep 30
    local OUT
    OUT=$($S "$POD" -- bash -c 'curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8900/v1/models' 2>/dev/null | tr -d '\r' | tail -1)
    if [[ "$OUT" == *200* ]]; then return 0; fi
    local C
    C=$($S "$POD" -- bash -c "cat /tmp/${TAG}_${SUF}_attn.log /tmp/${TAG}_${SUF}_ffn.log 2>/dev/null | grep -cE 'do tiling failed|507015|507057|Traceback|ValidationError|ValueError' || true" 2>/dev/null | tr -d '\r' | tail -1)
    if [[ "${C:-0}" != "0" ]]; then
      say "!! crash signature on $POD/$SUF (count=$C)"
      $S "$POD" -- bash -c "grep -E 'ERROR|Error|Traceback' /tmp/${TAG}_${SUF}_attn.log | tail -6" 2>&1 | tr -d '\r' | tail -6
      return 1
    fi
  done
  return 1
}

say "== start stacks (token-split, mbt=65536, async ON) =="
start_stack "$MASTER" "$NODE1_IP" n1
start_stack "$SECOND" "$NODE2_IP" n2

say "== wait readiness =="
wait_ready "$MASTER" n1 || { say "!! node1 never ready"; exit 1; }
say "node1 ready"
wait_ready "$SECOND" n2 || { say "!! node2 never ready"; exit 1; }
say "node2 ready; sanity checks:"
$S "$MASTER" -- bash -c "grep -o \"async_moe_split[^,}]*\" /tmp/${TAG}_n1_attn.log | head -1; grep -c 'Enabled DeepSeek-V4 persistent compressor tails' /tmp/${TAG}_n1_attn.log || true" 2>&1 | tr -d '\r' | tail -2
$S "$SECOND" -- bash -c "grep -o \"async_moe_split[^,}]*\" /tmp/${TAG}_n2_attn.log | head -1; grep -c 'AFD FFN EngineCore started' /tmp/${TAG}_n2_ffn.log || true" 2>&1 | tr -d '\r' | tail -2

say "== smoke each backend directly =="
for PR in "$MASTER n1" "$SECOND n2"; do
  set -- $PR
  $S "$1" -- bash -c 'curl -s -m 120 http://127.0.0.1:8900/v1/completions -H "Content-Type: application/json" -d "{\"model\":\"dsv4-afd-attention\",\"prompt\":[1,2,3,4,5,6,7,8],\"max_tokens\":1}" -o /dev/null -w "smoke=%{http_code}\n"' 2>&1 | tr -d '\r' | tail -1
done

say "== start router on master :$ROUTER_PORT =="
$S "$MASTER" -- bash -c 'pkill -f "[l]east_load_router"; sleep 1; echo killed' 2>&1 | tr -d '\r' | tail -1
$S "$MASTER" -- bash -c "setsid bash -c 'nohup $PY $ROUTER --port $ROUTER_PORT --backend http://$NODE1_IP:8900 --backend http://$NODE2_IP:8900 --poll-interval 0.5 --log-file /tmp/${TAG}_router_decisions.jsonl > /tmp/${TAG}_router.log 2>&1 &' ; echo started" 2>&1 | tr -d '\r' | tail -1
sleep 3
$S "$MASTER" -- bash -c "curl -s -m 5 http://127.0.0.1:$ROUTER_PORT/healthz" 2>&1 | tr -d '\r' | tail -1

say "== smoke 2 requests via router =="
$X "$MASTER" -- bash -c "
  for i in 1 2; do
    curl -s -m 120 http://127.0.0.1:$ROUTER_PORT/v1/completions -H 'Content-Type: application/json' \
      -d '{\"model\":\"dsv4-afd-attention\",\"prompt\":[1,2,3,4,5,6,7,8],\"max_tokens\":1}' -o /dev/null -w \"smoke\$i=%{http_code} \"
  done; echo" 2>&1 | tr -d '\r' | tail -1

run_plan() { # $1=plan file, $2=output tag
  say "== replay $2 =="
  $S "$MASTER" -- bash -c "setsid bash -c 'cd $CODE && nohup $PY tools/benchmarks/mw_replay_client.py --base-url http://127.0.0.1:$ROUTER_PORT --model dsv4-afd-attention --requests $WL/formal_1_requests.jsonl --plan $WL/$1 --output /tmp/${TAG}_$2.json > /tmp/${TAG}_$2.client.log 2>&1 &'; echo launched" 2>&1 | tr -d '\r' | tail -1
  local DONE=0
  for i in $(seq 1 90); do
    sleep 30
    local OUT
    OUT=$($S "$MASTER" -- bash -c "test -f /tmp/${TAG}_$2.json && echo EXISTS || echo NO" 2>/dev/null | tr -d '\r' | tail -1)
    if [[ "$OUT" == *EXISTS* ]]; then DONE=1; break; fi
  done
  if [[ $DONE != 1 ]]; then
    say "!! $2 client did not finish in 45min"
    $S "$MASTER" -- bash -c "tail -20 /tmp/${TAG}_$2.client.log" 2>&1 | tr -d '\r' | tail -20
    exit 1
  fi
  say "== $2 done: $($S "$MASTER" -- bash -c "grep -o 'REPLAY_OK[^{]*' /tmp/${TAG}_$2.client.log | head -1" 2>/dev/null | tr -d '\r' | tail -1)"
}

run_plan formal_1_plan.json 1x
sleep 30
run_plan formal_1_fast1p5x_plan.json fast1p5x
sleep 30
run_plan formal_1_fast2x_plan.json fast2x

say "== router balance check =="
$S "$MASTER" -- bash -c "grep -o '\"backend\": \"[^\"]*\"' /tmp/${TAG}_router_decisions.jsonl | sort | uniq -c" 2>&1 | tr -d '\r' | tail -4

say "== archive =="
$S "$MASTER" -- bash -c "
  mkdir -p $NASDIR
  cp /tmp/${TAG}_1x.json /tmp/${TAG}_fast1p5x.json /tmp/${TAG}_fast2x.json /tmp/${TAG}_n1_attn.log /tmp/${TAG}_n1_ffn.log /tmp/${TAG}_router.log /tmp/${TAG}_router_decisions.jsonl $NASDIR/ 2>/dev/null
  echo archived" 2>&1 | tr -d '\r' | tail -1
$S "$SECOND" -- bash -c "
  mkdir -p $NASDIR
  cp /tmp/${TAG}_n2_attn.log /tmp/${TAG}_n2_ffn.log $NASDIR/ 2>/dev/null
  echo archived" 2>&1 | tr -d '\r' | tail -1
say "== ALL_COMPLETE singlenode2x dp4tp2 token =="
