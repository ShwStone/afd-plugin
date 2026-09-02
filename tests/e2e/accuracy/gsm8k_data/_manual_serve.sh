#!/bin/bash
# Manual two-serve launch for the DP2 CAM deadlock experiment.
# blocking=1 + AFD_CAM_OP_IO_LOG=1, logs to /tmp/manual_attn.log + /tmp/manual_ffn.log
set -e
cd /a3_inference/itask/workdir/tq02357756/shwstone/code/afd-plugin
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_CUSTOM_OPP_PATH=/usr/local/Ascend/cann-9.0.1/opp/vendors/CAM:${ASCEND_CUSTOM_OPP_PATH:-}
export LD_LIBRARY_PATH=/usr/local/Ascend/cann-9.0.1/opp/vendors/CAM/op_api/lib:/usr/local/Ascend/cann-9.0.1/opp/vendors/CAM/op_api:${LD_LIBRARY_PATH:-}
export HCCL_BUFFSIZE=4096
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_CONNECT_TIMEOUT=3600
export ASCEND_LAUNCH_BLOCKING=1
export AFD_CAM_OP_IO_LOG=1
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800
export VLLM_PLUGINS=ascend,afd
export PYTHONUNBUFFERED=1

ASCEND_RT_VISIBLE_DEVICES=0,1 setsid vllm serve /home/admin/model-csi/model \
  --shutdown-timeout 10 --served-model-name deepseek-v2-lite-afd-attention \
  --data-parallel-size 2 --tensor-parallel-size 1 --enable-expert-parallel \
  --additional-config '{"afd":{"role":"attention","connector":"CAMAsyncAFDConnector","host":"127.0.0.1","port":1239,"num_attention_ranks":2,"num_ffn_ranks":1,"async":true,"compute_gate_on_attention":true}}' \
  --enforce-eager --host 127.0.0.1 --port 8000 --gpu-memory-utilization 0.8 \
  > /tmp/manual_attn.log 2>&1 < /dev/null &

ASCEND_RT_VISIBLE_DEVICES=2 setsid vllm serve /home/admin/model-csi/model \
  --shutdown-timeout 10 --served-model-name deepseek-v2-lite-afd-ffn \
  --data-parallel-size 1 --tensor-parallel-size 1 --enable-expert-parallel \
  --additional-config '{"afd":{"role":"ffn","connector":"CAMAsyncAFDConnector","host":"127.0.0.1","port":1239,"num_attention_ranks":2,"num_ffn_ranks":1,"async":true,"compute_gate_on_attention":true}}' \
  --enforce-eager --host 127.0.0.1 --port 8001 --gpu-memory-utilization 0.8 \
  > /tmp/manual_ffn.log 2>&1 < /dev/null &

echo "launched"
