#!/bin/bash
# Detached launcher: starts the PR244 e2e runner fully detached from the
# itask exec session, logging to /tmp/pr244_e2e.log.
setsid nohup /bin/bash /a3_inference/itask/workdir/tq02357756/shwstone/code/afd-plugin/tests/e2e/accuracy/gsm8k_data/_run_pr244_async_cam.sh > /tmp/pr244_e2e.log 2>&1 < /dev/null &
# itask exec kills background children on immediate session teardown;
# holding the session for a few seconds lets the setsid'd child escape.
sleep 5
echo "launched, log: /tmp/pr244_e2e.log"
