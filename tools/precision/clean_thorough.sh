#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
# Thorough cleanup of ALL vLLM/AFD processes on this node + /dev/shm blocks,
# with multi-level verification. Run as a FILE (not inline) so the patterns
# below never match this script's own command line.
set -u

echo "=== before: vllm-ish procs"
ps -ef | grep -iE '[v]llm|EngineCore|Worker_DP|umdk|VLLMWorker' | grep -v grep | wc -l

echo "=== kill vllm serve"
pkill -9 -f '[v]llm serve' 2>/dev/null || true
echo "=== kill VLLM:: comm procs"
ps -eo pid,comm | awk '$2 ~ /^VLLM::/ {print $1}' | xargs -r kill -9 2>/dev/null || true
echo "=== kill /vllm-workspace procs"
ps -ef | grep '[/]vllm-workspace' | grep -v grep | awk '{print $2}' | xargs -r kill -9 2>/dev/null || true
echo "=== kill vllm engine/worker procs by path"
ps -ef | grep -E '[v]llm/v1/engine|[v]llm serve' | grep -v grep | awk '{print $2}' | xargs -r kill -9 2>/dev/null || true

sleep 10
rm -f /dev/shm/vllm_* 2>/dev/null || true

echo "=== after: remaining vllm-ish procs"
ps -ef | grep -iE '[v]llm|EngineCore|Worker_DP|umdk|VLLMWorker' | grep -v grep | wc -l
echo "=== npu-smi VLLMWorker count"
npu-smi info 2>/dev/null | grep -c VLLMWorker
echo "=== HBM used per card (should be ~3GB / 65536)"
npu-smi info 2>/dev/null | grep -oE '[0-9]+ / 65536'gi
