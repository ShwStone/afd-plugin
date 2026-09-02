#!/bin/bash
# Unit subset for the shutdown-sentinel fix.
cd /a3_inference/itask/workdir/tq02357756/shwstone/code/afd-plugin
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python -m pytest -q \
  tests/unit/v1/worker/test_npu_runtime.py \
  tests/unit/v1/worker/test_attention_model_runner.py \
  tests/unit/connectors/test_async_cam_connector.py \
  > /tmp/unit_sentinel.log 2>&1
echo "UNIT_EXIT=$?" >> /tmp/unit_sentinel.log
