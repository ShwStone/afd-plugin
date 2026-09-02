#!/bin/bash
# Detached launcher: PR 234 e2e (same afd-eager-async-cam scenario as PR 244),
# logging to /tmp/pr234_e2e.log.
setsid nohup /bin/bash /a3_inference/itask/workdir/tq02357756/shwstone/code/afd-plugin/tests/e2e/accuracy/gsm8k_data/_run_pr244_async_cam.sh > /tmp/pr234_e2e.log 2>&1 < /dev/null &
# itask exec kills background children on immediate session teardown;
# holding the session for a few seconds lets the setsid'd child escape.
sleep 5
echo "launched, log: /tmp/pr234_e2e.log"
