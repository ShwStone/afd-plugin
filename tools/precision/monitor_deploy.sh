#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
# Monitor the single-node AFD deployment health (attention API + FFN liveness).
#
# The FFN connector loop intermittently dies with `async_dispatch_recv` socket
# / ACL errors even while idle (the whole reason the precision runs needed
# prompt fixture sends). This watches for it so a crashed FFN is detected
# instead of discovered after a failed request.
#
#   bash tools/precision/monitor_deploy.sh                       # one-shot status
#   WATCH=1 INTERVAL=15 bash tools/precision/monitor_deploy.sh   # loop (watchdog)
#
# RESTART=1 (with WATCH=1) relaunches the deployment (kill -> launch -> wait
# for attention API) whenever the FFN dies, so a long-running experiment is not
# left half-dead.
set -u

REPO="${REPO:-/a3_inference/itask/workdir/tq02357756/shwstone/code/afd-plugin}"
FFN_LOG="$REPO/bench_results/precision/single_ffn.log"
ATTN_LOG="$REPO/bench_results/precision/single_attn.log"
MODE="${MODE:-token}"
RUN_ID="${RUN_ID:-v026-safe-l05}"

prev_crashes=0

status() {
  local api alive crashes
  api=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8000/v1/models 2>/dev/null)
  alive=$(pgrep -af '[v]llm serve' | wc -l)
  crashes=$(grep -c 'died unexpectedly' "$FFN_LOG" 2>/dev/null || echo 0)
  echo "$(date +%H:%M:%S) api=${api:-down} alive=${alive:-0} ffn_crashes=${crashes:-0}"
}

relaunch() {
  echo "$(date +%H:%M:%S) [monitor] FFN down -> relaunching (mode=$MODE run=$RUN_ID)"
  cd "$REPO" || return 1
  bash tools/precision/clean_thorough.sh >/dev/null 2>&1
  CAPTURE=1 MODE="$MODE" RUN_ID="$RUN_ID" bash tools/precision/launch_single_reduced.sh \
    >/dev/null 2>&1
  for _ in $(seq 1 40); do
    code=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8000/v1/models 2>/dev/null)
    [ "$code" = "200" ] && { echo "$(date +%H:%M:%S) [monitor] relaunched OK"; return 0; }
    sleep 15
  done
  echo "$(date +%H:%M:%S) [monitor] relaunch timeout" >&2
  return 1
}

if [ "${WATCH:-0}" = "1" ]; then
  while true; do
    status
    crashes=$(grep -c 'died unexpectedly' "$FFN_LOG" 2>/dev/null || echo 0)
    alive=$(pgrep -af '[v]llm serve' | wc -l)
    if [ "${RESTART:-0}" = "1" ] && { [ "$alive" -lt 2 ] || [ "$crashes" -gt "$prev_crashes" ]; }; then
      relaunch
      prev_crashes=$(grep -c 'died unexpectedly' "$FFN_LOG" 2>/dev/null || echo 0)
    else
      prev_crashes="$crashes"
    fi
    sleep "${INTERVAL:-15}"
  done
else
  status
fi
