#!/bin/bash
# PR 234 unit test subset: the files touched by the HCCL buffer derivation.
cd /a3_inference/itask/workdir/tq02357756/shwstone/code/afd-plugin
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python -m pytest -q \
  tests/unit/distributed/test_cam_hccl_buffer.py \
  tests/unit/connectors/test_async_cam_connector.py \
  tests/unit/connectors/test_camp2p_connector.py \
  tests/unit/v1/worker/test_npu_device_contract.py \
  tests/unit/test_e2e_runner.py
