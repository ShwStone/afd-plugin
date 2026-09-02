#!/bin/bash
set -e
cd /a3_inference/itask/workdir/tq02357756/shwstone/code/afd-plugin
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_CUSTOM_OPP_PATH=/usr/local/Ascend/cann-9.0.1/opp/vendors/CAM:${ASCEND_CUSTOM_OPP_PATH:-}
export LD_LIBRARY_PATH=/usr/local/Ascend/cann-9.0.1/opp/vendors/CAM/op_api/lib:/usr/local/Ascend/cann-9.0.1/opp/vendors/CAM/op_api:${LD_LIBRARY_PATH:-}
export HCCL_CONNECT_TIMEOUT=3600
export ASCEND_LAUNCH_BLOCKING=1
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800
export HCCL_BUFFSIZE=4096
export HCCL_OP_EXPANSION_MODE=AIV
export HF_ENDPOINT=http://127.0.0.1:9999
export HF_HOME=/root/.cache/huggingface
export AFD_E2E_BACKEND=npu
export AFD_E2E_ATTENTION_DEVICES=0
export AFD_E2E_FFN_DEVICES=2
export AFD_NPU_E2E_MODEL=/home/admin/model-csi/model
python -m pytest -q -s tests/e2e/models/deepseek_v2_lite/test_deepseek_v2_lite.py -k "async-cam-eager"
