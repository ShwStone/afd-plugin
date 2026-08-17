#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
# Tear down vLLM servers and orphan VLLM workers on whichever node this runs on.
# Run on BOTH nodes (attention + FFN) after a hang / before relaunching:
#
#   bash tools/precision/kill_capture_node.sh
#
# Same cleanup the launch scripts do at start (memory npu-worker-cleanup:
# killing `vllm serve` leaves VLLMWorker_DP holding HBM, so they must be killed
# via comm name and npu-smi rows).
set -u

echo "[kill] stopping vllm serve on this node"
pkill -f '[v]llm serve' 2>/dev/null || true

echo "[kill] killing orphan VLLM:: workers (comm name)"
ps -eo pid,comm | awk '$2 ~ /^VLLM::/ {print $1}' | xargs -r kill -9 2>/dev/null || true

echo "[kill] killing VLLMWorker rows from npu-smi (HBM holders)"
for pid in $(npu-smi info 2>/dev/null | grep 'VLLMWorker' | awk -F'|' '{print $3}' | tr -d ' '); do
  kill -9 "$pid" 2>/dev/null || true
done

sleep 8

echo "[kill] clearing stale shared-memory vllm blocks"
rm -f /dev/shm/vllm_* 2>/dev/null || true

echo "[kill] remaining vllm/VLLM processes (should be empty):"
pgrep -af 'vllm serve|VLLMWorker' || echo "  none"
