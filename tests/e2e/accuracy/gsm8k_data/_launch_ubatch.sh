#!/bin/bash
# Detached launcher: afd-async-ubatch (A2F1) cold-start single-request gsm8k,
# the deterministic repro of the DP2 first-request deadlock, now with the
# upstream fix (f23e074). Log: /tmp/ubatch_fix.log
setsid nohup /bin/bash /a3_inference/itask/workdir/tq02357756/shwstone/code/afd-plugin/tests/e2e/accuracy/gsm8k_data/_run_ubatch_exp.sh > /tmp/ubatch_fix.log 2>&1 < /dev/null &
# itask exec kills background children on immediate session teardown;
# holding the session for a few seconds lets the setsid'd child escape.
sleep 5
echo "launched, log: /tmp/ubatch_fix.log"
