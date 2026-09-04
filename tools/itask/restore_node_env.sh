#!/bin/bash
# Restore node-local env on a freshly created DSV4 pod (rootfs is image-pristine).
# Reuses NAS artifacts: CAM vendor + umdk from v026_migrate_backup, vllm-ascend
# CWS patches + prebuilt artifacts from vllm_ascend_cws (no recompile).
#
# Push to the pod and run once per pod after itask create (validated 2026-09-04,
# both pods, ~3min):
#   b64=$(base64 -w0 tools/itask/restore_node_env.sh)
#   itask exec <pod> -- bash -c "echo '$b64' | base64 -d > /tmp/restore_node.sh"
#   itask exec <pod> -- bash /tmp/restore_node.sh
set -e
BK=/a3_inference/itask/workdir/tq02357756/shwstone

echo "== 1. CAM vendor + umdk from NAS backup =="
tar -xzf $BK/v026_migrate_backup/perf_node_local_full_20260824.tar.gz -C / \
  usr/local/Ascend/cann-9.0.1/opp/vendors/CAM \
  usr/local/python3.12.13/lib/python3.12/site-packages/umdk_cam_op_lib.cpython-312-aarch64-linux-gnu.so \
  usr/local/python3.12.13/lib/python3.12/site-packages/umdk_cam_op_lib-209.0.0b1.dist-info

echo "== 2. config.ini load_priority =="
sed -i 's/^load_priority=.*/load_priority=CAM,batch_invariant/' /usr/local/Ascend/cann-9.0.1/opp/vendors/config.ini
cat /usr/local/Ascend/cann-9.0.1/opp/vendors/config.ini

echo "== 3. vllm-ascend -> e19e14da7 (patches + prebuilt artifacts, no recompile) =="
cd /vllm-workspace/vllm-ascend
git config --unset http.proxy 2>/dev/null || true
git config --unset https.proxy 2>/dev/null || true
git checkout -B cws-compressor-tail 80d8c194f
git am $BK/vllm_ascend_cws/patches/0001-*.patch $BK/vllm_ascend_cws/patches/0002-*.patch
tar -xzf $BK/vllm_ascend_cws/artifacts_e19e14da7.tar.gz -C /vllm-workspace/vllm-ascend
git log --oneline -1

echo "== 4. afd plugin entry point (editable, no ops build) =="
cd $BK/code/afd-plugin
source /usr/local/Ascend/cann-9.0.1/set_env.sh
AFD_BUILD_ASCEND_OPS=0 pip install -e . --no-deps --no-build-isolation -q

echo "== 5. verify =="
python3 -c "import torch, torch_npu; import umdk_cam_op_lib; print('UMDK_OK', hasattr(torch.ops.umdk_cam_op_lib,'async_dispatch_send'))"
python3 -c "from importlib.metadata import entry_points; eps=entry_points(group='vllm.general_plugins'); print('EP_OK', sorted(e.name for e in eps))"
ls /usr/local/Ascend/cann-9.0.1/opp/vendors/CAM/op_api/lib/ | head -3
echo RESTORE_DONE
