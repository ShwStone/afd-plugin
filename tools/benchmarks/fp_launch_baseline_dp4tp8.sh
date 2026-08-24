#!/usr/bin/env bash
# Full-model (61-layer DeepSeek-V3.2 W8A8) traditional baseline for the
# full-prefill performance experiment.
# Docs: docs/npu/DEEPSEEK_V3_2_FULL_PREFILL_PERFORMANCE_PLAN.zh-CN.md
# Topology: DP4 x TP8 = 32 ranks (2 nodes x 16 NPUs), EP across all 32 ranks,
# FlashComm1 SP, eager, prefix caching off.
#
# Required env:
#   DP_START_RANK           0 on node0 (master, API server), 2 on node1 (headless)
#   DP_ADDRESS              node0 IP used for DP coordination
# Optional env:
#   MAX_NUM_BATCHED_TOKENS  default 65536 (plan-mandated common batch cap)
#   LOCAL_IP                default: auto-detected from NIC_NAME
#   NIC_NAME                default eth0
#   MAX_MODEL_LEN           default 70000
#   MAX_NUM_SEQS            default 8
set -euo pipefail

: "${DP_START_RANK:?Set DP_START_RANK to 0 (node0) or 2 (node1)}"
: "${DP_ADDRESS:?Set DP_ADDRESS to the node0 IP}"

REPO=/a3_inference/itask/workdir/tq02357756/shwstone/code/afd-plugin
MODEL_PATH="${MODEL_PATH:-/home/admin/model-csi/model}"
DP_RPC_PORT="${DP_RPC_PORT:-29550}"
NIC_NAME="${NIC_NAME:-eth0}"
LOCAL_IP="${LOCAL_IP:-$(ifconfig "$NIC_NAME" | grep -oP 'inet \K[0-9.]+')}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-70000}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-65536}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"

# Load the Ascend runtime environment (non-interactive shells skip .bashrc).
# atb set_env.sh references ZSH_VERSION unbound, so relax -u while sourcing.
set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/cann-9.0.1/share/info/ascendnpu-ir/bin/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
set -u
export LD_PRELOAD=/usr/lib64/libjemalloc.so.2:

export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export VLLM_ASCEND_ENABLE_MLAPO=1
export HCCL_OP_EXPANSION_MODE=AIV
# Do NOT set HCCL_BUFFSIZE: the default (~200 MB) pool is enough for the
# baseline's collectives, and a large pool wastes device memory — with 4096
# the ~8.6 GB/device pool left <6.3 GiB outside PyTorch and a full-length
# (50,773 token) prefill step OOMed needing 6.24 GiB of workspace.
export HCCL_CONNECT_TIMEOUT=3600
# Engine-core boot can exceed the 600s default while the 646 GiB
# checkpoint loads from cold cache; give it room instead of dying
# mid-boot (observed 2026-08-21 A2).
export VLLM_ENGINE_READY_TIMEOUT_S=1800
# Baseline is the stock deployment: run WITHOUT the afd plugin. The
# vllm_ascend GLOBAL patches (incl. the AscendMoERunner FusedMoE factory in
# patch_fused_moe.py) are applied by its "general plugins", whose entry-point
# names are ascend_kv_connector / ascend_model_loader /
# ascend_service_profiling / ascend_model — vllm's allowlist is an EXACT name
# match, so plain "ascend" only activates the platform plugin and silently
# skips the global patches (stock MoE then dies with
# "AscendFusedMoEMethod.apply() got an unexpected keyword argument
# 'topk_weights'"). All four must be named explicitly.
export VLLM_PLUGINS="${VLLM_PLUGINS:-ascend,ascend_kv_connector,ascend_model_loader,ascend_service_profiling,ascend_model}"
# Mount /reset_prefix_cache so repeats can clear state without a restart.
export VLLM_SERVER_DEV_MODE=1
export AFD_FORCE_BALANCED_TOPK_IDS=0
export HCCL_IF_IP="$LOCAL_IP"
export GLOO_SOCKET_IFNAME="$NIC_NAME"
export TP_SOCKET_IFNAME="$NIC_NAME"
export HCCL_SOCKET_IFNAME="$NIC_NAME"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

HEADLESS_ARGS=()
if [[ "$DP_START_RANK" != "0" ]]; then
  HEADLESS_ARGS=(--headless)
fi

# Profiling is opt-in (phase=profile boots only): vllm 0.26 registers the
# /start_profile API only when --profiler-config names a profiler, and the
# old VLLM_TORCH_PROFILER_DIR env var no longer exists ("Unknown vLLM
# environment variable"). The orchestrator still passes the directory via
# that env name; translate it here. with_stack=false keeps 32-rank traces
# small enough to parse.
PROFILER_ARGS=()
if [[ -n "${VLLM_TORCH_PROFILER_DIR:-}" ]]; then
  PROFILER_ARGS=(--profiler-config "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"${VLLM_TORCH_PROFILER_DIR}\",\"torch_profiler_with_stack\":false}")
fi

cd "$REPO"
exec vllm serve "$MODEL_PATH" \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name deepseek_v3_2 \
  --data-parallel-size 4 \
  --data-parallel-size-local 2 \
  --data-parallel-start-rank "$DP_START_RANK" \
  --data-parallel-address "$DP_ADDRESS" \
  --data-parallel-rpc-port "$DP_RPC_PORT" \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --enforce-eager \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --block-size 128 \
  --quantization ascend \
  --tokenizer-mode deepseek_v32 \
  --seed 1024 \
  --gpu-memory-utilization 0.70 \
  --trust-remote-code \
  --no-enable-prefix-caching \
  --additional-config '{
    "enable_force_load_balance": false,
    "enable_prefill_mc2": false
  }' \
  "${PROFILER_ARGS[@]}" \
  "${HEADLESS_ARGS[@]}"
