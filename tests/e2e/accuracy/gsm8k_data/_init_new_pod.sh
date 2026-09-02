#!/bin/bash
# New-pod environment init for afd-v2lite-e2e2 (per memory:
# v026-accuracy-pod-migration + cam-async-dp2-coldstart-deadlock).
# 1) wait for model CSI  2) vllm_ascend checkout v0.26 baseline
# 3) restore node-local (umdk + CAM vendor + editable finder) from NAS backup
# 4) verify imports
set -x

# --- 1. wait for model CSI (async download) ---
for i in $(seq 1 60); do
  if [ -f /home/admin/model-csi/model/config.json ]; then
    echo "MODEL_READY"
    break
  fi
  sleep 10
done
grep -E '"model_type"|"num_hidden_layers"|"num_experts_per_tok"' /home/admin/model-csi/model/config.json | head -3

# --- 2. vllm_ascend checkout v0.26 baseline (editable, no reinstall) ---
cd /vllm-workspace/vllm-ascend && git checkout -B v026-baseline 80d8c194f && git log --oneline -1

# --- 3. restore node-local from NAS backup ---
BK=/a3_inference/itask/workdir/tq02357756/shwstone/v026_migrate_backup
ls -la "$BK/v026_node_local.tar.gz"
tar -xzf "$BK/v026_node_local.tar.gz" -C /
ls /usr/local/Ascend/cann-9.0.1/opp/vendors/CAM/op_api/lib/ | head -3

# --- 4. verify imports ---
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python3 -c "import afd_plugin; print('AFD_PLUGIN_OK', afd_plugin.__file__)"
python3 -c "import torch, torch_npu; import umdk_cam_op_lib; print('UMDK_OK')"
python3 -c "from afd_plugin.compat.npu import ensure_afd_ascend_ops_loaded; ensure_afd_ascend_ops_loaded(); print('COMPAT_OPS_OK')"
echo INIT_DONE
