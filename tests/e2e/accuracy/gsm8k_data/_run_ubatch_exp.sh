#!/bin/bash
set -e
cd /a3_inference/itask/workdir/tq02357756/shwstone/code/afd-plugin
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export ASCEND_CUSTOM_OPP_PATH=/usr/local/Ascend/cann-9.0.1/opp/vendors/CAM:${ASCEND_CUSTOM_OPP_PATH:-}
export LD_LIBRARY_PATH=/usr/local/Ascend/cann-9.0.1/opp/vendors/CAM/op_api/lib:/usr/local/Ascend/cann-9.0.1/opp/vendors/CAM/op_api:${LD_LIBRARY_PATH:-}
export HCCL_BUFFSIZE=4096
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_CONNECT_TIMEOUT=3600
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800
export AFD_GSM8K_LIMIT=7
export HF_ENDPOINT=http://127.0.0.1:9999
export HF_HOME=/root/.cache/huggingface
mkdir -p /tmp/ubatch_out
python -m tests.e2e.runner \
  --model /home/admin/model-csi/model \
  --vllm-bin vllm \
  --device-backend npu \
  --attention-devices 0,1 \
  --ffn-devices 2 \
  --scenario afd-async-ubatch \
  --gsm8k-output-path /tmp/ubatch_out
