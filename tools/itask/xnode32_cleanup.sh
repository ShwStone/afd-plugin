#!/usr/bin/env bash
# Kill all vllm/AFD processes on this pod and free NPU HBM.
# Uses [bracket] patterns so pkill never matches its own command line.
set -u
pkill -f '[v]llm serve' 2>/dev/null
pkill -f '[v]llm.entrypoints' 2>/dev/null
pkill -f '[E]ngineCore' 2>/dev/null
pkill -f '[V]LLM' 2>/dev/null
pkill -f '[m]w_replay_client' 2>/dev/null
pkill -f '[f]p_export' 2>/dev/null
sleep 8
pkill -9 -f '[v]llm serve' 2>/dev/null
pkill -9 -f '[v]llm.entrypoints' 2>/dev/null
pkill -9 -f '[E]ngineCore' 2>/dev/null
pkill -9 -f '[V]LLM' 2>/dev/null
sleep 3
# Last resort: kill anything still holding NPU HBM per npu-smi table.
for pid in $(npu-smi info 2>/dev/null | awk -F'|' '{for(i=1;i<=NF;i++){gsub(/ /,"",$i); if($i ~ /^[0-9]{3,}$/ && $(i+1) ~ /python|VLLM|vllm/) print $i}}' | sort -u); do
  kill -9 "$pid" 2>/dev/null
done
sleep 2
BUSY=$(npu-smi info 2>/dev/null | grep -c "No running processes" || true)
echo "CLEANUP_DONE idle_npu_rows=$BUSY"
