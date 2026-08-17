#!/usr/bin/env bash
# Kill leftover vLLM / worker processes without self-matching.
pkill -9 -f '/bin/vllm serve'
pkill -9 -f 'multiproc_executor'
pkill -9 -f 'vllm.v1.engine'
pkill -9 -f 'VLLM::'
sleep 2
pgrep -af '/bin/vllm|multiproc_executor|vllm.v1.engine|VLLM::' || echo CLEAN
