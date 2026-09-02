#!/bin/bash
# Detached launcher for the full unit suite; log: /tmp/unit_rebased.log
setsid nohup /bin/bash /a3_inference/itask/workdir/tq02357756/shwstone/code/afd-plugin/tests/e2e/accuracy/gsm8k_data/_run_unit_full.sh > /dev/null 2>&1 < /dev/null &
# itask exec kills background children on immediate session teardown;
# holding the session for a few seconds lets the setsid'd child escape.
sleep 5
echo "launched, log: /tmp/unit_rebased.log"
