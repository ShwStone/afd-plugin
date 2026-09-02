#!/usr/bin/env bash
# DeepSeek V4 Flash W8A8 stock (traditional) baseline launcher.
# Docs: docs/npu/DEEPSEEK_V4_FLASH_W8A8_BASELINE_PERFORMANCE_PLAN.zh-CN.md
#
# One script covers all three plan topologies; EP spans DP x TP ranks:
#   DP4TP4EP16  single node:  DP_SIZE=4 TP_SIZE=4                      (16 ranks)
#   DP2TP8EP16  single node:  DP_SIZE=2 TP_SIZE=8                      (16 ranks)
#   DP4TP8EP32  two nodes:    DP_SIZE=4 TP_SIZE=8 DP_SIZE_LOCAL=2
#                             node0: DP_START_RANK=0 DP_ADDRESS=<node0 ip>
#                             node1: DP_START_RANK=2 DP_ADDRESS=<node0 ip>
#
# Required env:
#   DP_SIZE                 data-parallel size (total, across nodes)
#   TP_SIZE                 tensor-parallel size
# Optional env:
#   DP_START_RANK           multi-node only; !=0 boots --headless
#   DP_ADDRESS              multi-node only; node0 IP for DP coordination
#   DP_RPC_PORT             default 29550
#   DP_SIZE_LOCAL           default DP_SIZE (single node); 2 for the 32-rank pair
#   MAX_NUM_BATCHED_TOKENS  default 32768 (plan matrix: 8192/16384/32768/65536)
#   MAX_NUM_SEQS            default 128 (plan-frozen: never gates the 128-req window)
#   MAX_MODEL_LEN           default 70000 (covers the 63,778 max input + 1)
#   GPU_MEM_UTIL            default 0.80 (user-frozen for this baseline)
#   ASYNC_SCHEDULING        default 0. Async scheduling doubles
#                           max_in_flight_tokens (2 batches in flight), which
#                           doubles the SWA/compressor transient KV budget and
#                           blocks long requests at util 0.80 (vllm#51041
#                           fallout). Off = in_flight = 1 x max_num_batched_tokens.
#   LOCAL_IP                default: auto-detected from NIC_NAME
#   NIC_NAME                default eth0
#   PORT                    default 8000
#   ADDITIONAL_CONFIG       default: force_load_balance/prefill_mc2 off.
#                           Override to opt into stage features, e.g.
#                           '{"enable_force_load_balance":false,"enable_prefill_mc2":false,"enable_dsv4_shared_compressor_workspace":true,"multistream_dsv4_dsa_overlap":false}'
#                           (compressor-tail CYCLE workspace; its gate also
#                           requires multistream_dsv4_dsa_overlap=false)
set -euo pipefail

: "${DP_SIZE:?Set DP_SIZE (4 for DP4TP4/DP4TP8, 2 for DP2TP8)}"
: "${TP_SIZE:?Set TP_SIZE (4 or 8)}"

REPO=/a3_inference/itask/workdir/tq02357756/shwstone/code/afd-plugin
MODEL_PATH="${MODEL_PATH:-/home/admin/model-csi/model}"
DP_RPC_PORT="${DP_RPC_PORT:-29550}"
DP_SIZE_LOCAL="${DP_SIZE_LOCAL:-$DP_SIZE}"
NIC_NAME="${NIC_NAME:-eth0}"
LOCAL_IP="${LOCAL_IP:-$(ifconfig "$NIC_NAME" | grep -oP 'inet \K[0-9.]+')}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-70000}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-32768}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-128}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.80}"
ASYNC_SCHEDULING="${ASYNC_SCHEDULING:-0}"
PORT="${PORT:-8000}"

# Load the Ascend runtime environment (non-interactive shells skip .bashrc).
# atb set_env.sh references ZSH_VERSION unbound, so relax -u while sourcing.
set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/cann-9.0.1/share/info/ascendnpu-ir/bin/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
set -u
export LD_PRELOAD=/usr/lib64/libjemalloc.so.2:

# Prometheus multiproc dir must pre-exist or the engine core dies at startup
# with FileNotFoundError on counter_*.db.
if [[ -z "${PROMETHEUS_MULTIPROC_DIR:-}" ]]; then
  # A shared dir across boots pollutes multiprocess aggregation with dead
  # engines' counter dbs (gauges read ~0 in live windows). Isolate per boot.
  PROMETHEUS_MULTIPROC_DIR="/tmp/prom_v4_$(date +%s)_$$"
  export PROMETHEUS_MULTIPROC_DIR
fi
mkdir -p "$PROMETHEUS_MULTIPROC_DIR"

export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export VLLM_ASCEND_ENABLE_MLAPO=1
export HCCL_OP_EXPANSION_MODE=AIV
# Do NOT set HCCL_BUFFSIZE (user instruction, same as the V3.2 baseline): the
# default ~200 MB pool is enough; a large pool wastes device memory.
export HCCL_CONNECT_TIMEOUT=3600
# Engine-core boot can exceed the 600s default while the checkpoint loads
# from cold cache; give it room instead of dying mid-boot.
export VLLM_ENGINE_READY_TIMEOUT_S=1800
# Stock baseline runs WITHOUT the afd plugin. vllm's plugin allowlist is an
# EXACT name match: plain "ascend" only activates the platform plugin and
# silently skips vllm_ascend's four general plugins carrying the global
# patches (stock MoE then dies with "apply() got an unexpected keyword
# argument 'topk_weights'"). All five names must be listed.
export VLLM_PLUGINS="${VLLM_PLUGINS:-ascend,ascend_kv_connector,ascend_model_loader,ascend_service_profiling,ascend_model}"
# Mount /reset_prefix_cache so runs can clear state without a restart
# (prefix caching itself stays disabled below).
export VLLM_SERVER_DEV_MODE=1
export AFD_FORCE_BALANCED_TOPK_IDS=0
export HCCL_IF_IP="$LOCAL_IP"
export GLOO_SOCKET_IFNAME="$NIC_NAME"
export TP_SOCKET_IFNAME="$NIC_NAME"
export HCCL_SOCKET_IFNAME="$NIC_NAME"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

DP_ARGS=(--data-parallel-size "$DP_SIZE" --data-parallel-size-local "$DP_SIZE_LOCAL")
if [[ -n "${DP_START_RANK:-}" ]]; then
  DP_ARGS+=(--data-parallel-start-rank "$DP_START_RANK"
            --data-parallel-address "$DP_ADDRESS"
            --data-parallel-rpc-port "$DP_RPC_PORT")
  if [[ "$DP_START_RANK" != "0" ]]; then
    DP_ARGS+=(--headless)
  fi
fi

# Profiling is opt-in and NOT used in this plan's matrix; keep the translation
# so ad-hoc boots can still take a profiler dir via the old env name.
PROFILER_ARGS=()
if [[ -n "${VLLM_TORCH_PROFILER_DIR:-}" ]]; then
  PROFILER_ARGS=(--profiler-config "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"${VLLM_TORCH_PROFILER_DIR}\",\"torch_profiler_with_stack\":false}")
fi

SCHED_ARGS=()
if [[ "$ASYNC_SCHEDULING" == "0" ]]; then
  SCHED_ARGS=(--no-async-scheduling)
fi

cd "$REPO"
if [[ -z "${ADDITIONAL_CONFIG:-}" ]]; then
  ADDITIONAL_CONFIG='{"enable_force_load_balance": false, "enable_prefill_mc2": false}'
fi
# Named presets: passing raw JSON through nested `itask exec bash -c "bash -c
# '...'"` layers gets mangled by brace expansion; resolve presets here instead.
if [[ "$ADDITIONAL_CONFIG" == "cws" ]]; then
  ADDITIONAL_CONFIG='{"enable_force_load_balance":false,"enable_prefill_mc2":false,"enable_dsv4_shared_compressor_workspace":true,"multistream_dsv4_dsa_overlap":false}'
fi
# Plan-frozen serving switches: eager, prefix caching off, EPLB off (default;
# no enable-eplb flags passed), natural routing, MTP/speculative off (the
# checkpoint ships MTP weights but no --speculative-config is passed), DBO
# off (default). Clients send materialized prompt_token_ids, so no
# --tokenizer-mode override: the default auto mode loads the V4
# PreTrainedTokenizerFast from the checkpoint.
exec vllm serve "$MODEL_PATH" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --served-model-name deepseek_v4_flash \
  --tensor-parallel-size "$TP_SIZE" \
  --enable-expert-parallel \
  --enforce-eager \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --block-size 128 \
  --quantization ascend \
  --seed 1024 \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --trust-remote-code \
  --no-enable-prefix-caching \
  --additional-config "$ADDITIONAL_CONFIG" \
  "${DP_ARGS[@]}" \
  "${PROFILER_ARGS[@]}" \
  "${SCHED_ARGS[@]}"
