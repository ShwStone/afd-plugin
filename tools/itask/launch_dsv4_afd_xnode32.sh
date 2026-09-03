#!/usr/bin/env bash
set -euo pipefail

# Two-task 32-card DSV4 async CAM deployment for the mbt sweep
# (DP6TP4 attention + DP8EP8 FFN).
#
# Attention is one global DP6TP4 vLLM process group, started once on each
# node: node1 (master) owns DP0-3 (NPUs 0-15) and hosts the API; node2 owns
# DP4-5 (NPUs 0-7) and joins headlessly.  Both Attention invocations use
# node1 as the vLLM DP coordinator.  All 24 Attention ranks and 8 FFN ranks
# join one CAM communicator through node1; node2 also hosts the FFN DP8/EP8
# ranks on NPUs 8-15 (started as a second invocation with ROLE=ffn).
: "${ROLE:?ROLE must be attention or ffn}"
: "${NODE_IP:?NODE_IP is required (this pod IP)}"
: "${ATTENTION_NODE_ID:=1}"  # 1 = node1 master (DP0-3), 2 = node2 (DP4-5)
: "${ATTN_API_PORT:=8900}"
: "${FFN_API_PORT:=8901}"
if [[ "$ROLE" == attention ]]; then API_PORT=$ATTN_API_PORT; else API_PORT=$FFN_API_PORT; fi
: "${DP_RPC_PORT:=29550}"
: "${NIC_NAME:=eth0}"
: "${PROFILE_VARIANT:=none}"
: "${PROFILE_ROOT:=/tmp/dsv4_afd_profiles}"
: "${GPU_MEMORY_UTILIZATION:=0.80}"
# 2048 MB covers the mbt=65536 CAM dispatch tiling requirement with headroom
# (single-node needed 1024; 512 failed).  "none" leaves the env var unset.
: "${HCCL_BUFFSIZE:=2048}"
: "${MAX_NUM_BATCHED_TOKENS:=8192}"
: "${MAX_NUM_SEQS:=128}"
: "${MAX_MODEL_LEN:=70000}"
: "${ASYNC_SCHEDULING:=0}"
# Shared compressor workspace (CWS, cache_mode=2 CYCLE) is required for
# mbt=65536 to pass the startup KV estimate and is kept on for all sweep
# cells so mbt is the only varying knob.  vllm-ascend must be e19e14da7.
: "${DSV4_SHARED_COMPRESSOR_WORKSPACE:=1}"
: "${FLASHCOMM1:=1}"
: "${MODEL_PATH:=/home/admin/model-csi/model}"

case "$ROLE" in
  attention|ffn) ;;
  *) echo "ROLE must be attention or ffn" >&2; exit 2 ;;
esac

if [[ "$ROLE" == attention ]]; then
  case "$ATTENTION_NODE_ID" in
    1)
      ATTN_DP_SIZE_LOCAL=4
      ATTN_DP_START_RANK=0
      ATTN_HEADLESS_ARGS=()
      ATTN_DEVICES="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"
      API_SERVER_ARGS=(--api-server-count 1)
      ;;
    2)
      ATTN_DP_SIZE_LOCAL=2
      ATTN_DP_START_RANK=4
      ATTN_HEADLESS_ARGS=(--headless)
      ATTN_DEVICES="0,1,2,3,4,5,6,7"
      API_SERVER_ARGS=()
      ;;
    *)
      echo "ATTENTION_NODE_ID must be 1 or 2 for ROLE=attention" >&2
      exit 2
      ;;
  esac
fi

case "$PROFILE_VARIANT" in
  none)
    PROFILER_ARGS=()
    ;;
  full)
    PROFILE_DIR="$PROFILE_ROOT/afd_${ROLE}_node${ATTENTION_NODE_ID}_full"
    PROFILER_ARGS=(--profiler-config "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"$PROFILE_DIR\",\"torch_profiler_with_stack\":true,\"torch_profiler_record_shapes\":true,\"torch_profiler_with_memory\":false,\"torch_profiler_use_gzip\":false,\"ignore_frontend\":true,\"delay_iterations\":10,\"max_iterations\":9,\"warmup_iterations\":0,\"active_iterations\":10,\"wait_iterations\":0}")
    ;;
  ops)
    PROFILE_DIR="$PROFILE_ROOT/afd_${ROLE}_node${ATTENTION_NODE_ID}_ops"
    PROFILER_ARGS=(--profiler-config "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"$PROFILE_DIR\",\"torch_profiler_with_stack\":false,\"torch_profiler_record_shapes\":false,\"torch_profiler_with_memory\":false,\"torch_profiler_use_gzip\":false,\"ignore_frontend\":true,\"delay_iterations\":10,\"max_iterations\":9,\"warmup_iterations\":0,\"active_iterations\":10,\"wait_iterations\":0}")
    ;;
  *)
    echo "PROFILE_VARIANT must be none, full, or ops" >&2
    exit 2
    ;;
esac

# Resolve the plugin from this launcher instead of a task-specific workdir.
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PLUGIN_ROOT=${PLUGIN_ROOT:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}
: "${NODE1_IP:?NODE1_IP must be the master attention pod IP}"
: "${NODE2_IP:?NODE2_IP must be the second pod IP}"

source /usr/local/Ascend/cann-9.0.1/set_env.sh
export PYTHONPATH="$PLUGIN_ROOT:/vllm-workspace/vllm-ascend:/vllm-workspace/vllm:${PYTHONPATH:-}"
export VLLM_PLUGINS=ascend,afd
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export AFD_FORCE_SPAWN_MULTIPROCESSING=1
CAM_VENDOR=/usr/local/Ascend/cann-9.0.1/opp/vendors/CAM
CAM_OPAPI_DIR="$CAM_VENDOR/op_api/lib"
# CAM 209.x ships the vendor op-api as libcust_opapi.so; CANN's op-api lookup
# and afd_plugin resolve the vendor library by the libopapi.so name.
if [[ ! -f "$CAM_OPAPI_DIR/libopapi.so" ]]; then
  if [[ -f "$CAM_OPAPI_DIR/libcust_opapi.so" ]]; then
    ln -sfn "$CAM_OPAPI_DIR/libcust_opapi.so" "$CAM_OPAPI_DIR/libopapi.so"
  else
    echo "CAM vendor op-api not found under $CAM_OPAPI_DIR; install the CAM operator package first" >&2
    exit 1
  fi
fi
CAM_OPAPI="$CAM_OPAPI_DIR/libopapi.so"
TORCH_NPU_LIB=/usr/local/python3.12.13/lib/python3.12/site-packages/torch_npu/lib
TORCH_LIB=/usr/local/python3.12.13/lib/python3.12/site-packages/torch/lib
export ASCEND_CUSTOM_OPP_PATH="$CAM_VENDOR:${ASCEND_CUSTOM_OPP_PATH:-}"
export LD_LIBRARY_PATH="$TORCH_LIB":"$TORCH_NPU_LIB":/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64:"$CAM_OPAPI_DIR":"$CAM_VENDOR/op_api":/usr/local/Ascend/cann-9.0.1/aarch64-linux/lib64:/usr/local/Ascend/cann-9.0.1/runtime/lib64:${LD_LIBRARY_PATH:-}
export CAM_CUST_OPAPI_LIB_PATH="$CAM_OPAPI"
export LD_PRELOAD="$CAM_OPAPI${LD_PRELOAD:+:$LD_PRELOAD}"
export HCCL_IF_IP="$NODE_IP"
export HCCL_SOCKET_IFNAME="$NIC_NAME"
export GLOO_SOCKET_IFNAME="$NIC_NAME"
export TP_SOCKET_IFNAME="$NIC_NAME"
if [[ "$HCCL_BUFFSIZE" != "none" ]]; then export HCCL_BUFFSIZE; else unset HCCL_BUFFSIZE; fi
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_CONNECT_TIMEOUT=3600 HCCL_EXEC_TIMEOUT=3600
export OMP_PROC_BIND=false OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
# FlashComm1 (SP) is attention-only: FFN runs TP1 where Flash Comm v1 is
# rejected by config validation.  User-facing FLASHCOMM1 knob gates attention.
if [[ "$ROLE" == attention ]]; then
  export VLLM_ASCEND_ENABLE_FLASHCOMM1="$FLASHCOMM1"
else
  export VLLM_ASCEND_ENABLE_FLASHCOMM1=0
fi

if [[ "$ROLE" == attention ]]; then
  export ASCEND_RT_VISIBLE_DEVICES="$ATTN_DEVICES"
  # One global DP6TP4 group: node1 DP0-3 + node2 DP4-5, coordinator on node1.
  PARALLEL_ARGS=(
    --data-parallel-size 6
    --data-parallel-size-local "$ATTN_DP_SIZE_LOCAL"
    --data-parallel-start-rank "$ATTN_DP_START_RANK"
    --data-parallel-address "$NODE1_IP"
    --data-parallel-rpc-port "$DP_RPC_PORT"
    --tensor-parallel-size 4
    "${ATTN_HEADLESS_ARGS[@]}"
  )
  WORKER=afd_plugin.v1.worker.npu.AFDNPUAttentionWorker
  MODEL_NAME=dsv4-afd-attention
else
  export ASCEND_RT_VISIBLE_DEVICES="8,9,10,11,12,13,14,15"
  PARALLEL_ARGS=(--data-parallel-size 8 --tensor-parallel-size 1)
  API_SERVER_ARGS=(--api-server-count 1)
  WORKER=afd_plugin.v1.worker.npu.AFDNPUFFNWorker
  MODEL_NAME=dsv4-afd-ffn
fi

CONNECTOR_EXTRA='"dynamicQuant":1,"attn_ranks_per_dp":4,"async_moe_ubatching":'"${ASYNC_MOE_UBATCHING:-true}"',"async_moe_split":"'"${ASYNC_MOE_SPLIT:-request}"'"'
CWS_PREFIX='"enable_force_load_balance":false'
if [[ "$DSV4_SHARED_COMPRESSOR_WORKSPACE" == "1" ]]; then
  CWS_PREFIX='"enable_force_load_balance":false,"multistream_dsv4_dsa_overlap":false,"enable_dsv4_shared_compressor_workspace":true'
fi
# Opt-in async Attention DPLB policy (request_count|prefill_token_sum).
# Attention-only: the plugin validator rejects this key on the FFN role.
DPLB_KV=''
if [[ "$ROLE" == "attention" && -n "${ATTN_DPLB_POLICY:-}" ]]; then
  DPLB_KV=',"attention_dplb_policy":"'"$ATTN_DPLB_POLICY"'"'
fi
ADDITIONAL_CONFIG="{$CWS_PREFIX,\"afd\":{\"role\":\"$ROLE\",\"connector\":\"CAMAsyncAFDConnector\",\"async\":true,\"host\":\"$NODE1_IP\",\"port\":1239,\"num_attention_ranks\":24,\"num_ffn_ranks\":8,\"compute_gate_on_attention\":true${DPLB_KV},\"connector_extra_config\":{$CONNECTOR_EXTRA}}}"

SCHEDULING_ARGS=(--enable-chunked-prefill)
if [[ "$ASYNC_SCHEDULING" != "1" ]]; then
  SCHEDULING_ARGS+=(--no-async-scheduling)
fi

exec env VLLM_USE_V1=1 /usr/local/python3.12.13/bin/vllm serve "$MODEL_PATH" \
  --host 0.0.0.0 --port "$API_PORT" "${API_SERVER_ARGS[@]}" --served-model-name "$MODEL_NAME" \
  --worker-cls "$WORKER" "${PARALLEL_ARGS[@]}" --enable-expert-parallel \
  --enforce-eager --quantization ascend --tokenizer-mode deepseek_v4 \
  --block-size 128 --max-model-len "$MAX_MODEL_LEN" --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" --max-num-seqs "$MAX_NUM_SEQS" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 16}' \
  --trust-remote-code --no-enable-prefix-caching "${SCHEDULING_ARGS[@]}" \
  --additional-config "$ADDITIONAL_CONFIG" "${PROFILER_ARGS[@]}"
