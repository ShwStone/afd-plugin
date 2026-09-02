#!/bin/bash
# Detached launcher: pure-main baseline e2e (control run for the PR 234 crash),
# logging to /tmp/main_baseline_e2e.log.
setsid nohup /bin/bash /a3_inference/itask/workdir/tq02357756/shwstone/code/afd-plugin/tests/e2e/accuracy/gsm8k_data/_run_pr244_async_cam.sh > /tmp/main_baseline_e2e.log 2>&1 < /dev/null &
# itask exec kills background children on immediate session teardown;
# holding the session for a few seconds lets the setsid'd child escape.
sleep 5
echo "launched, log: /tmp/main_baseline_e2e.log"
