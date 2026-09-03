#!/usr/bin/env bash
set -euo pipefail

# Single-task DSV4 async CAM deployment (smoke / small-batch shape).
#
# One A3 task hosts both roles on its 16 NPUs: NPUs 0-7 run the Attention
# DP1TP8 process group (hosts the API); NPUs 8-15 run the FFN DP8/EP8
# process group.  All 16 ranks join one CAM communicator through this node.
# For the two-task 24+8 deployment use launch_dsv4_afd_cross_node.sh.
: "${ROLE:?ROLE must be attention or ffn}"
: "${NODE_IP:?NODE_IP is required (this pod IP)}"
# Both roles used to share API_PORT in the two-task deployment, where each
# role lived on its own node.  On a single node the two API servers would
# collide, so attention (the serving entry) and FFN get separate defaults.
: "${ATTN_API_PORT:=8900}"
: "${FFN_API_PORT:=8901}"
if [[ "$ROLE" == attention ]]; then API_PORT=$ATTN_API_PORT; else API_PORT=$FFN_API_PORT; fi
: "${NIC_NAME:=eth0}"
: "${ATTENTION_DEVICES:=0,1,2,3,4,5,6,7}"
: "${FFN_DEVICES:=8,9,10,11,12,13,14,15}"
: "${PROFILE_VARIANT:=none}"
: "${PROFILE_ROOT:=/tmp/dsv4_afd_profiles}"
: "${GPU_MEMORY_UTILIZATION:=0.80}"
# HCCL_BUFFSIZE default 512 preserves all prior AFD measurements; set to
# "none" to leave the env var unset entirely (e.g. when the connector-scoped
# hccl_buffer_size knob below should be the only buffer override).
: "${HCCL_BUFFSIZE:=512}"
# Connector-scoped CAM HCCL buffer in MB (PR#293). Empty = no per-domain
# override; the CAM domain then falls back to env HCCL_BUFFSIZE / built-in.
: "${CONNECTOR_HCCL_BUFFER_SIZE_MB:=}"
: "${MAX_NUM_BATCHED_TOKENS:=10240}"
: "${MAX_NUM_SEQS:=128}"
: "${MAX_MODEL_LEN:=70000}"
: "${ASYNC_SCHEDULING:=0}"
# Opt into the DeepSeek-V4 shared compressor workspace (cache_mode=2 CYCLE).
# The upstream gate also requires multistream_dsv4_dsa_overlap=false, eager
# mode, prefix caching off, no kv_transfer_config, no speculative, CP=1.
: "${DSV4_SHARED_COMPRESSOR_WORKSPACE:=0}"
# Async MoE ubatch split mode: "request" (request boundaries) or "token"
# (token-balanced stages). ASYNC_MOE_UBATCHING=false disables ubatching.
: "${ASYNC_MOE_UBATCHING:=true}"
: "${ASYNC_MOE_SPLIT:=request}"
# FlashComm1 (SP) is attention-only: FFN runs TP1 where Flash Comm v1 is
# rejected by config validation.
: "${FLASHCOMM1:=0}"
# DSV4-Flash-w8a8-mtp; override when the task mounts the weights elsewhere.
: "${MODEL_PATH:=/mnt/sfs_turbo/models/DeepSeek-V4-Flash-w8a8-mtp}"

case "$ROLE" in
  attention|ffn) ;;
  *) echo "ROLE must be attention or ffn" >&2; exit 2 ;;
esac

# Profiling is request-controlled (vLLM /start_profile + /stop_profile);
# the plugin rejects iteration schedules (delay/max/warmup/active_iterations).
case "$PROFILE_VARIANT" in
  none)
    PROFILER_ARGS=()
    ;;
  full)
    PROFILE_DIR="$PROFILE_ROOT/afd_${ROLE}_single_full"
    PROFILER_ARGS=(--profiler-config "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"$PROFILE_DIR\",\"torch_profiler_with_stack\":true,\"torch_profiler_record_shapes\":true,\"torch_profiler_with_memory\":false,\"torch_profiler_use_gzip\":false,\"ignore_frontend\":true}")
    ;;
  ops)
    PROFILE_DIR="$PROFILE_ROOT/afd_${ROLE}_single_ops"
    PROFILER_ARGS=(--profiler-config "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"$PROFILE_DIR\",\"torch_profiler_with_stack\":false,\"torch_profiler_record_shapes\":false,\"torch_profiler_with_memory\":false,\"torch_profiler_use_gzip\":false,\"ignore_frontend\":true}")
    ;;
  *)
    echo "PROFILE_VARIANT must be none, full, or ops" >&2
    exit 2
    ;;
esac

# Resolve the plugin from this launcher instead of a task-specific workdir.
# ``itask sync`` may assign each task a different workspace name.
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PLUGIN_ROOT=${PLUGIN_ROOT:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}

source /usr/local/Ascend/cann-9.0.1/set_env.sh
export PYTHONPATH="$PLUGIN_ROOT:/vllm-workspace/vllm-ascend:/vllm-workspace/vllm:${PYTHONPATH:-}"
export VLLM_PLUGINS=ascend,afd
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export AFD_FORCE_SPAWN_MULTIPROCESSING=1
CAM_VENDOR=/usr/local/Ascend/cann-9.0.1/opp/vendors/CAM
CAM_OPAPI_DIR="$CAM_VENDOR/op_api/lib"
# CAM 209.x ships the vendor op-api as libcust_opapi.so; older packs named it
# libopapi.so.  CANN's op-api lookup and afd_plugin both resolve the vendor
# library by the libopapi.so name, so alias it in place when only the new name
# is present.
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
# umdk_cam_op_lib resolves libopapi.so by name on first request.  Put CAM's
# vendor implementation before CANN's stock libopapi.so, which lacks CAM ops.
export LD_LIBRARY_PATH="$TORCH_LIB":"$TORCH_NPU_LIB":/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64:"$CAM_OPAPI_DIR":"$CAM_VENDOR/op_api":/usr/local/Ascend/cann-9.0.1/aarch64-linux/lib64:/usr/local/Ascend/cann-9.0.1/runtime/lib64:${LD_LIBRARY_PATH:-}
export CAM_CUST_OPAPI_LIB_PATH="$CAM_OPAPI"
export LD_PRELOAD="$CAM_OPAPI${LD_PRELOAD:+:$LD_PRELOAD}"
export HCCL_IF_IP="$NODE_IP"
export HCCL_SOCKET_IFNAME="$NIC_NAME"
export GLOO_SOCKET_IFNAME="$NIC_NAME"
export TP_SOCKET_IFNAME="$NIC_NAME"
if [[ "$HCCL_BUFFSIZE" != "none" ]]; then export HCCL_BUFFSIZE; else unset HCCL_BUFFSIZE; fi
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_CONNECT_TIMEOUT=1800 HCCL_EXEC_TIMEOUT=1800
export OMP_PROC_BIND=false OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
if [[ "$ROLE" == attention ]]; then
  export VLLM_ASCEND_ENABLE_FLASHCOMM1="$FLASHCOMM1"
else
  export VLLM_ASCEND_ENABLE_FLASHCOMM1=0
fi

if [[ "$ROLE" == attention ]]; then
  export ASCEND_RT_VISIBLE_DEVICES="$ATTENTION_DEVICES"
  # One vLLM DP2TP4 process group on NPUs 0-7 (two replicas share the API
  # server); hosts the API server.
  PARALLEL_ARGS=(--data-parallel-size 2 --tensor-parallel-size 4)
  WORKER=afd_plugin.v1.worker.npu.AFDNPUAttentionWorker
  MODEL_NAME=dsv4-afd-attention
  API_SERVER_ARGS=(--api-server-count 1)
else
  export ASCEND_RT_VISIBLE_DEVICES="$FFN_DEVICES"
  PARALLEL_ARGS=(--data-parallel-size 8 --tensor-parallel-size 1)
  WORKER=afd_plugin.v1.worker.npu.AFDNPUFFNWorker
  MODEL_NAME=dsv4-afd-ffn
  API_SERVER_ARGS=(--api-server-count 1)
fi

CONNECTOR_EXTRA='"dynamicQuant":1,"attn_ranks_per_dp":4,"async_moe_ubatching":'"$ASYNC_MOE_UBATCHING"',"async_moe_split":"'"$ASYNC_MOE_SPLIT"'"'
if [[ -n "$CONNECTOR_HCCL_BUFFER_SIZE_MB" ]]; then
  CONNECTOR_EXTRA="$CONNECTOR_EXTRA,\"hccl_buffer_size\":$CONNECTOR_HCCL_BUFFER_SIZE_MB"
fi
ADDITIONAL_CONFIG="{\"enable_force_load_balance\":false,\"afd\":{\"role\":\"$ROLE\",\"connector\":\"CAMAsyncAFDConnector\",\"async\":true,\"host\":\"$NODE_IP\",\"port\":1239,\"num_attention_ranks\":8,\"num_ffn_ranks\":8,\"compute_gate_on_attention\":true,\"connector_extra_config\":{$CONNECTOR_EXTRA}}}"
if [[ "$DSV4_SHARED_COMPRESSOR_WORKSPACE" == "1" ]]; then
  ADDITIONAL_CONFIG="{\"enable_force_load_balance\":false,\"multistream_dsv4_dsa_overlap\":false,\"enable_dsv4_shared_compressor_workspace\":true,\"afd\":{\"role\":\"$ROLE\",\"connector\":\"CAMAsyncAFDConnector\",\"async\":true,\"host\":\"$NODE_IP\",\"port\":1239,\"num_attention_ranks\":8,\"num_ffn_ranks\":8,\"compute_gate_on_attention\":true,\"connector_extra_config\":{$CONNECTOR_EXTRA}}}"
fi

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
