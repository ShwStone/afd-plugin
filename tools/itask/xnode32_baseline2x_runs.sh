#!/usr/bin/env bash
# Two-instance stock baseline (DP4TP4EP16 per 16-card node) behind the
# least-load router, replaying formal_1 at 1x / 1.5x / 2x.
#
# Usage: xnode32_baseline2x_runs.sh
#
# Config parity with the AFD xnode32 runs: same vllm-ascend (e19e14da7) with
# CWS on, FLASHCOMM1=1, util 0.80, no async-scheduling, no HCCL_BUFFSIZE.
# Baseline-specific (per user): mbt=8192, stock serving (no AFD plugin).
set -uo pipefail

MASTER=${MASTER:-v4f-base-2}
SECOND=${SECOND:-v4f-xnode-2}
NODE1_IP=${NODE1_IP:-33.182.141.199}
NODE2_IP=${NODE2_IP:-33.182.140.47}
CODE=/a3_inference/itask/workdir/tq02357756/shwstone/code/afd-plugin
BASE_LAUNCHER=$CODE/tools/benchmarks/v4_launch_baseline.sh
ROUTER=$CODE/tools/benchmarks/least_load_router.py
PY=/usr/local/python3.12.13/bin/python3
WL=$CODE/tools/datasets/moonconv-wildchat-v4-flash-prefill/workloads
NASDIR=/a3_inference/itask/workdir/tq02357756/shwstone/xnode32_baseline2x
TAG=base2x_mbt8192
ROUTER_PORT=8800

X="timeout --signal=KILL 120 itask exec"
S="timeout --signal=KILL 45 itask exec"
say() { echo "[$(date +%H:%M:%S)] $*"; }

CWS_CFG='{"enable_force_load_balance":false,"enable_prefill_mc2":false,"enable_dsv4_shared_compressor_workspace":true,"multistream_dsv4_dsa_overlap":false}'
# NOTE: ADDITIONAL_CONFIG must stay single-quoted at the remote bash -c layer;
# unquoted {...a,b...} JSON is mangled by remote-side brace expansion.
BENV="DP_SIZE=4 TP_SIZE=4 MAX_NUM_BATCHED_TOKENS=8192 PORT=8000 ADDITIONAL_CONFIG='$CWS_CFG'"

say "== preflight: cleanup script + plans + launcher =="
for P in "$MASTER" "$SECOND"; do
  $S "$P" -- bash -c "test -f /tmp/xnode32_cleanup.sh && test -f $BASE_LAUNCHER && echo OK" 2>&1 | tr -d '\r' | tail -1
done
$S "$MASTER" -- bash -c "ls $WL/formal_1_plan.json $WL/formal_1_fast1p5x_plan.json $WL/formal_1_fast2x_plan.json $WL/formal_1_requests.jsonl >/dev/null && echo PLANS_OK" 2>&1 | tr -d '\r' | tail -1

say "== cleanup both pods (AFD stack down) =="
$S "$MASTER" -- bash /tmp/xnode32_cleanup.sh 2>&1 | tr -d '\r' | tail -2
$S "$SECOND" -- bash /tmp/xnode32_cleanup.sh 2>&1 | tr -d '\r' | tail -2

say "== start baseline instances (DP4TP4EP16, mbt=8192, CWS on) =="
$S "$MASTER" -- bash -c "setsid bash -c 'env $BENV nohup bash $BASE_LAUNCHER > /tmp/${TAG}_n1.log 2>&1 &' ; echo started" 2>&1 | tr -d '\r' | tail -1
$S "$SECOND" -- bash -c "setsid bash -c 'env $BENV nohup bash $BASE_LAUNCHER > /tmp/${TAG}_n2.log 2>&1 &' ; echo started" 2>&1 | tr -d '\r' | tail -1

wait_ready() { # $1=pod $2=logfile
  local POD=$1 LOG=$2
  for i in $(seq 1 80); do
    sleep 30
    local OUT
    OUT=$($S "$POD" -- bash -c 'curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8000/v1/models' 2>/dev/null | tr -d '\r' | tail -1)
    if [[ "$OUT" == *200* ]]; then return 0; fi
    local C
    C=$($S "$POD" -- bash -c "grep -cE 'do tiling failed|507015|507057|Traceback|ValidationError|Free memory' $LOG 2>/dev/null || true" 2>/dev/null | tr -d '\r' | tail -1)
    if [[ "${C:-0}" != "0" ]]; then
      say "!! crash signature on $POD (count=$C)"
      $S "$POD" -- bash -c "grep -E 'ERROR|Error|Traceback' $LOG | tail -8" 2>&1 | tr -d '\r' | tail -8
      return 1
    fi
  done
  return 1
}

say "== wait readiness (both nodes) =="
wait_ready "$MASTER" /tmp/${TAG}_n1.log || exit 1
say "node1 ready"
wait_ready "$SECOND" /tmp/${TAG}_n2.log || exit 1
say "node2 ready"

say "== smoke each backend directly =="
for P in "$MASTER" "$SECOND"; do
  $S "$P" -- bash -c 'curl -s -m 120 http://127.0.0.1:8000/v1/completions -H "Content-Type: application/json" -d "{\"model\":\"deepseek_v4_flash\",\"prompt\":[1,2,3,4,5,6,7,8],\"max_tokens\":1}" -o /dev/null -w "%{http_code}\n"' 2>&1 | tr -d '\r' | tail -1
done

say "== start router on master :$ROUTER_PORT =="
$S "$MASTER" -- bash -c "setsid bash -c 'nohup $PY $ROUTER --port $ROUTER_PORT --backend http://$NODE1_IP:8000 --backend http://$NODE2_IP:8000 --poll-interval 0.5 --log-file /tmp/${TAG}_router_decisions.jsonl > /tmp/${TAG}_router.log 2>&1 &' ; echo started" 2>&1 | tr -d '\r' | tail -1
sleep 3
$S "$MASTER" -- bash -c "curl -s -m 5 http://127.0.0.1:$ROUTER_PORT/healthz" 2>&1 | tr -d '\r' | tail -1

say "== smoke 2 requests via router =="
$X "$MASTER" -- bash -c "
  for i in 1 2; do
    curl -s -m 120 http://127.0.0.1:$ROUTER_PORT/v1/completions -H 'Content-Type: application/json' \
      -d '{\"model\":\"deepseek_v4_flash\",\"prompt\":[1,2,3,4,5,6,7,8],\"max_tokens\":1}' -o /dev/null -w \"smoke\$i=%{http_code} \"
  done; echo" 2>&1 | tr -d '\r' | tail -1

run_plan() { # $1=plan file, $2=output tag
  say "== replay $2 =="
  $S "$MASTER" -- bash -c "setsid bash -c 'cd $CODE && nohup $PY tools/benchmarks/mw_replay_client.py --base-url http://127.0.0.1:$ROUTER_PORT --model deepseek_v4_flash --requests $WL/formal_1_requests.jsonl --plan $WL/$1 --output /tmp/${TAG}_$2.json > /tmp/${TAG}_$2.client.log 2>&1 &'; echo launched" 2>&1 | tr -d '\r' | tail -1
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
  say "== $2 done =="
  $S "$MASTER" -- bash -c "curl -s -m 5 http://127.0.0.1:$ROUTER_PORT/lbstats" 2>&1 | tr -d '\r' | tail -1
}

run_plan formal_1_plan.json 1x
sleep 30
run_plan formal_1_fast1p5x_plan.json fast1p5x
sleep 30
run_plan formal_1_fast2x_plan.json fast2x

say "== archive =="
$S "$MASTER" -- bash -c "
  mkdir -p $NASDIR
  cp /tmp/${TAG}_1x.json /tmp/${TAG}_fast1p5x.json /tmp/${TAG}_fast2x.json \
     /tmp/${TAG}_n1.log /tmp/${TAG}_router.log /tmp/${TAG}_router_decisions.jsonl $NASDIR/
  echo archived" 2>&1 | tr -d '\r' | tail -1
$S "$SECOND" -- bash -c "cp /tmp/${TAG}_n2.log $NASDIR/; echo archived" 2>&1 | tr -d '\r' | tail -1
say "== ALL_COMPLETE =="
