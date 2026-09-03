#!/usr/bin/env bash
# Dual-node 32-card DSV4 AFD full-load profiler capture.
# One server per invocation; replays formal_1 at fast2x and captures a
# request-controlled profiler window (default: start T+10s, length 10s).
# Raw MSPROF output dirs are tarred per pod and uploaded via obsshare.
#
# Usage: SPLIT=token NODE1_IP=<ip> NODE2_IP=<ip> xnode32_profile_capture.sh
#   SPLIT=token  -> async_moe_ubatching=true,  async_moe_split=token
#   SPLIT=off    -> async_moe_ubatching=false
# Env: MASTER=v4f-base-2 SECOND=v4f-xnode-2 NODE1_IP=.. NODE2_IP=.. (required)
#      WIN_DELAY=10 WIN_LEN=10 MBT=65536
set -uo pipefail

SPLIT=${SPLIT:?SPLIT must be token or off}
MASTER=${MASTER:-v4f-base-2}
SECOND=${SECOND:-v4f-xnode-2}
NODE1_IP=${NODE1_IP:?set NODE1_IP to the master pod IP}
NODE2_IP=${NODE2_IP:?set NODE2_IP to the second pod IP}
CODE=/a3_inference/itask/workdir/tq02357756/shwstone/code/afd-plugin
LAUNCHER=$CODE/tools/itask/launch_dsv4_afd_xnode32.sh
WL=$CODE/tools/datasets/moonconv-wildchat-v4-flash-prefill/workloads
NASDIR=/a3_inference/itask/workdir/tq02357756/shwstone/xnode32_prof_as
MBT=${MBT:-65536}
WIN_DELAY=${WIN_DELAY:-10}
WIN_LEN=${WIN_LEN:-10}
PROFTAG=dsv4_prof_${SPLIT}
PROFROOT=/tmp/$PROFTAG
TAG=xnode32_prof_${SPLIT}_as_mbt${MBT}

X="timeout --signal=KILL 120 itask exec"
S="timeout --signal=KILL 45 itask exec"
say() { echo "[$(date +%H:%M:%S)] $*"; }

if [[ "$SPLIT" == token ]]; then
  SPLENV="ASYNC_MOE_UBATCHING=true ASYNC_MOE_SPLIT=token"
elif [[ "$SPLIT" == off ]]; then
  SPLENV="ASYNC_MOE_UBATCHING=false"
else
  echo "SPLIT must be token or off" >&2; exit 2
fi

say "== cleanup both pods =="
$S "$MASTER" -- bash /tmp/xnode32_cleanup.sh 2>&1 | tr -d '\r' | tail -2
$S "$SECOND" -- bash /tmp/xnode32_cleanup.sh 2>&1 | tr -d '\r' | tail -2
$S "$MASTER" -- bash -c "mv $PROFROOT /tmp/.trash_${PROFTAG}_\$(date +%s) 2>/dev/null; echo wiped" 2>&1 | tr -d '\r' | tail -1
$S "$SECOND" -- bash -c "mv $PROFROOT /tmp/.trash_${PROFTAG}_\$(date +%s) 2>/dev/null; echo wiped" 2>&1 | tr -d '\r' | tail -1

ENV1="NODE1_IP=$NODE1_IP NODE2_IP=$NODE2_IP MAX_NUM_BATCHED_TOKENS=$MBT HCCL_BUFFSIZE=${HCCL_BUFFSIZE:-4096} FLASHCOMM1=1 $SPLENV PROFILE_VARIANT=ops PROFILE_ROOT=$PROFROOT ${EXTRA_ENV:-}"
say "== start stack (mbt=$MBT, $SPLENV, PROFILE_VARIANT=ops, async-sched ON) =="
$S "$MASTER" -- bash -c "setsid bash -c 'env $ENV1 NODE_IP=$NODE1_IP ROLE=attention ATTENTION_NODE_ID=1 nohup bash $LAUNCHER > /tmp/${TAG}_attn1.log 2>&1 &' ; echo started" 2>&1 | tr -d '\r' | tail -1
$S "$SECOND" -- bash -c "setsid bash -c 'env $ENV1 NODE_IP=$NODE2_IP ROLE=attention ATTENTION_NODE_ID=2 nohup bash $LAUNCHER > /tmp/${TAG}_attn2.log 2>&1 &' ; echo started" 2>&1 | tr -d '\r' | tail -1
# Stagger FFN behind headless attention on node2 (DP store port race).
sleep 20
$S "$SECOND" -- bash -c "setsid bash -c 'env $ENV1 NODE_IP=$NODE2_IP ROLE=ffn nohup bash $LAUNCHER > /tmp/${TAG}_ffn.log 2>&1 &' ; echo started" 2>&1 | tr -d '\r' | tail -1

say "== wait readiness =="
READY=0
for i in $(seq 1 80); do
  sleep 30
  OUT=$($S "$MASTER" -- bash -c 'curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8900/v1/models' 2>/dev/null | tr -d '\r' | tail -1)
  if [[ "$OUT" == *200* ]]; then READY=1; break; fi
  C1=$($S "$MASTER" -- bash -c "grep -cE 'do tiling failed|507015|507057|Traceback|ValidationError|ValueError' /tmp/${TAG}_attn1.log 2>/dev/null || true" 2>/dev/null | tr -d '\r' | tail -1)
  C2=$($S "$SECOND" -- bash -c "cat /tmp/${TAG}_attn2.log /tmp/${TAG}_ffn.log 2>/dev/null | grep -cE 'do tiling failed|507015|507057|Traceback|ValidationError|ValueError' || true" 2>/dev/null | tr -d '\r' | tail -1)
  CRASH=$(( ${C1:-0} + ${C2:-0} ))
  if [[ "$CRASH" != "0" ]]; then
    say "!! crash signature (count=$CRASH)"
    $S "$MASTER" -- bash -c "tail -20 /tmp/${TAG}_attn1.log" 2>&1 | tr -d '\r' | tail -20
    $S "$SECOND" -- bash -c "cat /tmp/${TAG}_attn2.log /tmp/${TAG}_ffn.log | grep -E 'ERROR|Error' | tail -8" 2>&1 | tr -d '\r' | tail -8
    exit 1
  fi
done
[[ $READY == 1 ]] || { say "!! never ready"; exit 1; }
say "ready; sanity checks:"
$S "$MASTER" -- bash -c "grep -o \"async_moe_ubatching[^,}]*\" /tmp/${TAG}_attn1.log | head -1; grep -o \"async_moe_split[^,}]*\" /tmp/${TAG}_attn1.log | head -1; grep -c 'torch_profiler_dir' /tmp/${TAG}_attn1.log || true" 2>&1 | tr -d '\r' | tail -3
$S "$SECOND" -- bash -c "grep -c 'AFD FFN EngineCore started' /tmp/${TAG}_ffn.log || true" 2>&1 | tr -d '\r' | tail -1

say "== smoke 2 requests =="
$X "$MASTER" -- bash -c '
  for i in 1 2; do
    curl -s -m 120 http://127.0.0.1:8900/v1/completions -H "Content-Type: application/json" \
      -d "{\"model\":\"dsv4-afd-attention\",\"prompt\":[1,2,3,4,5,6,7,8],\"max_tokens\":1}" -o /dev/null -w "smoke$i=%{http_code} "
  done; echo' 2>&1 | tr -d '\r' | tail -1

say "== replay fast2x (window: start T+${WIN_DELAY}s, len ${WIN_LEN}s) =="
$S "$MASTER" -- bash -c "setsid bash -c 'cd $CODE && nohup python3 tools/benchmarks/mw_replay_client.py --base-url http://127.0.0.1:8900 --model dsv4-afd-attention --requests $WL/formal_1_requests.jsonl --plan $WL/formal_1_fast2x_plan.json --output /tmp/${TAG}_fast2x.json > /tmp/${TAG}_fast2x.client.log 2>&1 &'; echo launched" 2>&1 | tr -d '\r' | tail -1

sleep "$WIN_DELAY"
# Start order: FFN first, then attention. Starts return quickly.
say "== start_profile FFN -> attention =="
$S "$SECOND" -- bash -c 'curl -s -m 60 -X POST http://127.0.0.1:8901/start_profile -o /dev/null -w "ffn_start=%{http_code}"' 2>&1 | tr -d '\r' | tail -1
$S "$MASTER" -- bash -c 'curl -s -m 60 -X POST http://127.0.0.1:8900/start_profile -o /dev/null -w "attn_start=%{http_code}"' 2>&1 | tr -d '\r' | tail -1

sleep "$WIN_LEN"
# Stop order: attention first, then FFN. Stops block on export, so background
# the curls and poll for the trace dirs instead.
say "== stop_profile attention -> FFN (backgrounded; export is slow) =="
$S "$MASTER" -- bash -c "setsid bash -c 'nohup curl -s -m 1800 -X POST http://127.0.0.1:8900/stop_profile -o /dev/null -w \"attn_stop=%{http_code}\" > /tmp/${TAG}_stop_attn.out 2>&1 &' ; echo stop_launched" 2>&1 | tr -d '\r' | tail -1
$S "$SECOND" -- bash -c "setsid bash -c 'nohup curl -s -m 1800 -X POST http://127.0.0.1:8901/stop_profile -o /dev/null -w \"ffn_stop=%{http_code}\" > /tmp/${TAG}_stop_ffn.out 2>&1 &' ; echo stop_launched" 2>&1 | tr -d '\r' | tail -1

say "== wait trace export (expect 16 attn node1 / 8 attn node2 / 8 ffn) =="
EXPORT_OK=0
for i in $(seq 1 40); do
  sleep 30
  N1=$($S "$MASTER" -- bash -c "ls -d $PROFROOT/afd_attention_node1_ops/*_ascend_pt 2>/dev/null | wc -l" 2>/dev/null | tr -d '\r' | tail -1)
  N2=$($S "$SECOND" -- bash -c "ls -d $PROFROOT/afd_attention_node2_ops/*_ascend_pt 2>/dev/null | wc -l; ls -d $PROFROOT/afd_ffn_node1_ops/*_ascend_pt 2>/dev/null | wc -l" 2>/dev/null | tr -d '\r' | grep -o '[0-9]*' | tr '\n' ' ')
  A2=$(echo "$N2" | awk '{print $1}'); F2=$(echo "$N2" | awk '{print $2}')
  say "  poll $i: attn1=${N1:-0}/16 attn2=${A2:-0}/8 ffn=${F2:-0}/8"
  if [[ "${N1:-0}" == 16 && "${A2:-0}" == 8 && "${F2:-0}" == 8 ]]; then
    STOPS=$($S "$MASTER" -- bash -c "cat /tmp/${TAG}_stop_attn.out 2>/dev/null" 2>/dev/null | tr -d '\r' | tail -1)
    STOPS2=$($S "$SECOND" -- bash -c "cat /tmp/${TAG}_stop_ffn.out 2>/dev/null" 2>/dev/null | tr -d '\r' | tail -1)
    if [[ "$STOPS" == *200* && "$STOPS2" == *200* ]]; then EXPORT_OK=1; break; fi
  fi
done
if [[ $EXPORT_OK != 1 ]]; then
  say "!! export incomplete; stop outputs:"
  $S "$MASTER" -- bash -c "cat /tmp/${TAG}_stop_attn.out 2>/dev/null; tail -30 /tmp/${TAG}_attn1.log | grep -iE 'profil|error' | tail -5" 2>&1 | tr -d '\r' | tail -6
  $S "$SECOND" -- bash -c "cat /tmp/${TAG}_stop_ffn.out 2>/dev/null; cat /tmp/${TAG}_attn2.log /tmp/${TAG}_ffn.log | grep -iE 'profil|error' | tail -5" 2>&1 | tr -d '\r' | tail -6
  say "!! continuing to archive whatever exists"
fi

say "== wait replay done =="
DONE=0
for i in $(seq 1 90); do
  sleep 30
  OUT=$($S "$MASTER" -- bash -c "test -f /tmp/${TAG}_fast2x.json && echo EXISTS || echo NO" 2>/dev/null | tr -d '\r' | tail -1)
  if [[ "$OUT" == *EXISTS* ]]; then DONE=1; break; fi
done
if [[ $DONE == 1 ]]; then
  say "== fast2x done: $($S "$MASTER" -- bash -c "grep -o 'REPLAY_OK[^{]*' /tmp/${TAG}_fast2x.client.log | head -1" 2>/dev/null | tr -d '\r' | tail -1)"
else
  say "!! replay did not finish in 45min (profile data still usable)"
fi

say "== tar profile dirs (backgrounded) =="
$S "$MASTER" -- bash -c "setsid bash -c 'cd /tmp && nohup bash -c \"tar czf /tmp/${TAG}_node1.tar.gz $PROFTAG && echo TAR_DONE > /tmp/${TAG}_node1.tardone\" > /dev/null 2>&1 &' ; echo tar_launched" 2>&1 | tr -d '\r' | tail -1
$S "$SECOND" -- bash -c "setsid bash -c 'cd /tmp && nohup bash -c \"tar czf /tmp/${TAG}_node2.tar.gz $PROFTAG && echo TAR_DONE > /tmp/${TAG}_node2.tardone\" > /dev/null 2>&1 &' ; echo tar_launched" 2>&1 | tr -d '\r' | tail -1
for i in $(seq 1 40); do
  sleep 30
  T1=$($S "$MASTER" -- bash -c "cat /tmp/${TAG}_node1.tardone 2>/dev/null || echo NO" 2>/dev/null | tr -d '\r' | tail -1)
  T2=$($S "$SECOND" -- bash -c "cat /tmp/${TAG}_node2.tardone 2>/dev/null || echo NO" 2>/dev/null | tr -d '\r' | tail -1)
  if [[ "$T1" == *TAR_DONE* && "$T2" == *TAR_DONE* ]]; then break; fi
done
say "tar status: node1=$T1 node2=$T2"
$S "$MASTER" -- bash -c "ls -la /tmp/${TAG}_node1.tar.gz; md5sum /tmp/${TAG}_node1.tar.gz" 2>&1 | tr -d '\r' | tail -2
$S "$SECOND" -- bash -c "ls -la /tmp/${TAG}_node2.tar.gz; md5sum /tmp/${TAG}_node2.tar.gz" 2>&1 | tr -d '\r' | tail -2

say "== obsshare upload =="
URL1=$($X "$MASTER" -- bash -c "export PATH=/a3_inference/itask/workdir/shared/jcz:\$PATH; bash /a3_inference/itask/workdir/shared/jcz/obsshare.sh /tmp/${TAG}_node1.tar.gz" 2>/dev/null | tr -d '\r' | grep -o "wget --no-check-certificate.*" | tail -1)
URL2=$($X "$SECOND" -- bash -c "export PATH=/a3_inference/itask/workdir/shared/jcz:\$PATH; bash /a3_inference/itask/workdir/shared/jcz/obsshare.sh /tmp/${TAG}_node2.tar.gz" 2>/dev/null | tr -d '\r' | grep -o "wget --no-check-certificate.*" | tail -1)
echo "NODE1_DL: $URL1"
echo "NODE2_DL: $URL2"

say "== archive logs + replay json to NAS =="
$S "$MASTER" -- bash -c "
  mkdir -p $NASDIR
  cp /tmp/${TAG}_fast2x.json /tmp/${TAG}_attn1.log $NASDIR/ 2>/dev/null
  echo archived" 2>&1 | tr -d '\r' | tail -1
$S "$SECOND" -- bash -c "
  mkdir -p $NASDIR
  cp /tmp/${TAG}_attn2.log /tmp/${TAG}_ffn.log $NASDIR/ 2>/dev/null
  echo archived" 2>&1 | tr -d '\r' | tail -1
say "== ALL_COMPLETE profile split=$SPLIT =="
