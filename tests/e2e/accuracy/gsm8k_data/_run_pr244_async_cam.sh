#!/bin/bash
# PR 244 (parallel attention/FFN model loading) e2e: upstream afd-eager-async-cam scenario.
# Topology: attention DP1TP2 (NPU 0,1) + FFN DP2 (NPU 2,3), CAMAsyncAFDConnector.
# Env mirrors tests/e2e/models/deepseek_v2_lite/test_async_cam_npu.py plus
# HCCL_OP_EXPANSION_MODE/HCCL_CONNECT_TIMEOUT from the manual-serve recipe.
set -e
cd /a3_inference/itask/workdir/tq02357756/shwstone/code/afd-plugin
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export ASCEND_CUSTOM_OPP_PATH=/usr/local/Ascend/cann-9.0.1/opp/vendors/CAM:${ASCEND_CUSTOM_OPP_PATH:-}
export LD_LIBRARY_PATH=/usr/local/Ascend/cann-9.0.1/opp/vendors/CAM/op_api:/usr/local/Ascend/cann-9.0.1/opp/vendors/CAM/op_api/lib:${LD_LIBRARY_PATH:-}
export HCCL_BUFFSIZE=4096
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_CONNECT_TIMEOUT=3600
export VLLM_USE_V1=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ASCEND_LAUNCH_BLOCKING=1
export AFD_CAM_OP_IO_LOG=1
export VLLM_ASCEND_ENABLE_CONTEXT_PARALLEL=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800
export PYTHONUNBUFFERED=1

python -m tests.e2e.runner \
  --model /home/admin/model-csi/model \
  --vllm-bin vllm \
  --device-backend npu \
  --attention-devices 0,1 \
  --ffn-devices 2,3 \
  --scenario afd-eager-async-cam \
  --served-model-name-prefix cam-async \
  --afd-connector-extra-config '{"dynamicQuant":0}' \
  --api-port-base 19080 \
  --afd-port 6453 \
  --startup-timeout 900 \
  --common-vllm-arg=--trust-remote-code \
  --common-vllm-arg=--max-num-seqs --common-vllm-arg=8 \
  --common-vllm-arg=--max-num-batched-tokens --common-vllm-arg=8000 \
  --common-vllm-arg=--gpu-memory-utilization --common-vllm-arg=0.75 \
  --common-vllm-arg=--no-enable-prefix-caching
