#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
#
# Capture and compare one no-ubatch and one token-ubatch DeepSeek V2 run using:
#   Attention: DP1/TP2 with FlashComm1
#   FFN:       DP2/TP1/EP2 without FlashComm1
#
# Usage:
#   MODEL_PATH=/path/to/DeepSeek-V2-Lite \
#     tools/run_async_cam_tensor_dump_comparison.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

MODEL_PATH=${MODEL_PATH:?set MODEL_PATH to the DeepSeek V2 model directory}
DEVICE_IDS=${DEVICE_IDS:-0,1,2,3}
DUMP_LAYERS=${DUMP_LAYERS:-1,2,3}
DUMP_TOKEN_INDICES=${DUMP_TOKEN_INDICES:-0,1,15,31,63}
DUMP_POINTS=${DUMP_POINTS:-attn_dispatch_hidden,attn_topk_ids,attn_topk_weights,attn_ffn_output}
DUMP_FULL_TENSORS=${DUMP_FULL_TENSORS:-0}
DUMP_SYNC=${DUMP_SYNC:-0}
DUMP_OUTPUT_PARENT=${DUMP_OUTPUT_PARENT:-/tmp}
BASELINE_API_PORT=${BASELINE_API_PORT:-19080}
BASELINE_AFD_PORT=${BASELINE_AFD_PORT:-6453}
TOKEN_API_PORT=${TOKEN_API_PORT:-19180}
TOKEN_AFD_PORT=${TOKEN_AFD_PORT:-6553}

mkdir -p "$DUMP_OUTPUT_PARENT"
RUN_OUTPUT_DIR=$(mktemp -d "${DUMP_OUTPUT_PARENT%/}/afd-async-cam-dump.XXXXXX")
BASELINE_DUMP_DIR="${RUN_OUTPUT_DIR}/no-ubatch"
TOKEN_DUMP_DIR="${RUN_OUTPUT_DIR}/token-ubatch"
mkdir -p "$BASELINE_DUMP_DIR" "$TOKEN_DUMP_DIR"

export AFD_NPU_ASYNC_CAM_E2E_MODEL="$MODEL_PATH"
export AFD_NPU_ASYNC_CAM_E2E_DEVICES="$DEVICE_IDS"
export AFD_NPU_E2E_STARTUP_TIMEOUT=${AFD_NPU_E2E_STARTUP_TIMEOUT:-900}

export AFD_ASYNC_MOE_PRECISION_DEBUG=1
export AFD_ASYNC_MOE_PRECISION_DEBUG_LAYERS="$DUMP_LAYERS"
export AFD_ASYNC_MOE_PRECISION_DEBUG_POINTS="$DUMP_POINTS"
export AFD_ASYNC_MOE_PRECISION_DEBUG_TOKEN_INDICES="$DUMP_TOKEN_INDICES"
export AFD_ASYNC_MOE_PRECISION_DEBUG_FULL_TENSORS="$DUMP_FULL_TENSORS"
export AFD_ASYNC_MOE_PRECISION_DEBUG_SYNC="$DUMP_SYNC"

# This value is inherited by the test runner. The runner retains it for the
# Attention TP2 workers and removes it from the FFN TP1 worker environments.
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1

TEST_TARGET="tests/e2e/models/deepseek_v2_lite/test_async_cam_npu.py::test_deepseek_v2_lite_async_cam_attn_dp1tp2_ffn_dp2ep2_smoke"

echo "Experiment output: $RUN_OUTPUT_DIR"
echo "Topology: Attention DP1/TP2 FlashComm1=1; FFN DP2/TP1/EP2 FlashComm1=0"
echo "Layers: $DUMP_LAYERS"
echo "Global token indices: $DUMP_TOKEN_INDICES"

echo "Running no-ubatch baseline"
export AFD_ASYNC_MOE_PRECISION_DEBUG_DIR="$BASELINE_DUMP_DIR"
export AFD_NPU_ASYNC_CAM_E2E_UBATCHING=0
unset AFD_NPU_ASYNC_CAM_E2E_SPLIT
export AFD_NPU_ASYNC_CAM_E2E_API_PORT="$BASELINE_API_PORT"
export AFD_NPU_ASYNC_CAM_E2E_AFD_PORT="$BASELINE_AFD_PORT"
python3 -m pytest -s -vv "$TEST_TARGET"

echo "Running token-ubatch candidate"
export AFD_ASYNC_MOE_PRECISION_DEBUG_DIR="$TOKEN_DUMP_DIR"
export AFD_NPU_ASYNC_CAM_E2E_UBATCHING=1
export AFD_NPU_ASYNC_CAM_E2E_SPLIT=token
export AFD_NPU_ASYNC_CAM_E2E_API_PORT="$TOKEN_API_PORT"
export AFD_NPU_ASYNC_CAM_E2E_AFD_PORT="$TOKEN_AFD_PORT"
python3 -m pytest -s -vv "$TEST_TARGET"

echo "Comparing globally indexed Attention tensors"
python3 tools/compare_async_cam_tensor_dumps.py \
    "$BASELINE_DUMP_DIR" \
    "$TOKEN_DUMP_DIR" \
    | tee "${RUN_OUTPUT_DIR}/comparison.txt"

echo "Baseline dumps: $BASELINE_DUMP_DIR"
echo "Token dumps:    $TOKEN_DUMP_DIR"
echo "Comparison:     ${RUN_OUTPUT_DIR}/comparison.txt"
