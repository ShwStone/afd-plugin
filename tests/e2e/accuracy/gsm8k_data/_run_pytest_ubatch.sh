#!/bin/bash
# Manual pytest run of the afd-async-ubatch e2e node (gsm8k) on NPU.
# Needs the local HF mirror on 9999 for the GSM8K dataset.
set -e
cd /a3_inference/itask/workdir/tq02357756/shwstone/code/afd-plugin
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export ASCEND_CUSTOM_OPP_PATH=/usr/local/Ascend/cann-9.0.1/opp/vendors/CAM:${ASCEND_CUSTOM_OPP_PATH:-}
export LD_LIBRARY_PATH=/usr/local/Ascend/cann-9.0.1/opp/vendors/CAM/op_api:/usr/local/Ascend/cann-9.0.1/opp/vendors/CAM/op_api/lib:${LD_LIBRARY_PATH:-}
export HCCL_BUFFSIZE=4096
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_CONNECT_TIMEOUT=3600
export AFD_E2E_BACKEND=npu
export AFD_NPU_E2E_MODEL=/home/admin/model-csi/model
export AFD_E2E_DEVICES=0,1,2
export HF_ENDPOINT=http://127.0.0.1:9999
export HF_HOME=/root/.cache/huggingface

python -m pytest -svv tests/e2e/models/deepseek_v2_lite/test_async_cam_npu.py -k async_ubatch 2>&1 | tee /tmp/pytest_ubatch.log
