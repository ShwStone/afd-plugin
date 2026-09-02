#!/bin/bash
# Full unit suite on the rebased agent/npu-async-connector-tests-docs branch.
cd /a3_inference/itask/workdir/tq02357756/shwstone/code/afd-plugin
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python -m pytest -q tests/unit > /tmp/unit_rebased.log 2>&1
echo "UNIT_EXIT=$?" >> /tmp/unit_rebased.log
