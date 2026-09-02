#!/bin/bash
# Start the local GSM8K HF mirror on port 9999, detached.
cd /a3_inference/itask/workdir/tq02357756/shwstone/code/afd-plugin/tests/e2e/accuracy/gsm8k_data
HF_MIRROR_PORT=9999 setsid nohup python3 _hf_mirror.py > /tmp/hf_mirror.log 2>&1 < /dev/null &
sleep 5
curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:9999/api/datasets/openai/gsm8k || true
echo " mirror launched"
