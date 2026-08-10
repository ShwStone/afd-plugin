# Async CAM partial tensor dump

This opt-in diagnostic captures selected DeepSeek V2 tensors for a small
`Attention DP1/TP2 + FFN DP2/EP2` comparison. It is disabled by default and
does not alter the normal connector data path.

The most useful comparison points are on the Attention side:

- `attn_dispatch_hidden`, `attn_topk_ids`, and `attn_topk_weights` immediately
  before CAM dispatch;
- `attn_ffn_output` immediately after CAM combine and before the model restores
  its TP/SP layout.

These files carry flattened global token indices, so the comparison tool can
match a token even when token ubatching moves it to another stage or TP rank.
FFN-side `ffn_routed_input`, `ffn_routed_output`, and `ffn_group_list` use
rank-local row indices because CAM has already reordered those rows by expert.

## Run the reduced comparison

The wrapper script runs both cases and the comparison:

```bash
MODEL_PATH=/path/to/DeepSeek-V2-Lite \
  tools/run_async_cam_tensor_dump_comparison.sh
```

The script exports `VLLM_ASCEND_ENABLE_FLASHCOMM1=1` only as input to the E2E
runner. The runner keeps it for Attention TP2 and removes it from the FFN TP1
worker environments. The effective topology is therefore Attention
FlashComm1/SP on and FFN FlashComm1/SP off.

Override the sampled layers or tokens when needed:

```bash
MODEL_PATH=/path/to/DeepSeek-V2-Lite \
DUMP_LAYERS=1,2,3,4 \
DUMP_TOKEN_INDICES=0,1,127,255 \
  tools/run_async_cam_tensor_dump_comparison.sh
```

The equivalent manual procedure follows.

Use fresh directories because each process intentionally preserves only the
first observation of a layer/stage/point:

```bash
export AFD_NPU_ASYNC_CAM_E2E_MODEL=/path/to/DeepSeek-V2-Lite
export AFD_NPU_ASYNC_CAM_E2E_DEVICES=0,1,2,3
export AFD_ASYNC_MOE_PRECISION_DEBUG=1
export AFD_ASYNC_MOE_PRECISION_DEBUG_LAYERS=1,2,3
export AFD_ASYNC_MOE_PRECISION_DEBUG_TOKEN_INDICES=0,1,15,31,63
export AFD_ASYNC_MOE_PRECISION_DEBUG_FULL_TENSORS=0

BASELINE_DUMP_DIR=$(mktemp -d /tmp/afd-no-ubatch.XXXXXX)
export AFD_ASYNC_MOE_PRECISION_DEBUG_DIR=$BASELINE_DUMP_DIR
export AFD_NPU_ASYNC_CAM_E2E_UBATCHING=0
pytest -s -vv \
  tests/e2e/models/deepseek_v2_lite/test_async_cam_npu.py::test_deepseek_v2_lite_async_cam_attn_dp1tp2_ffn_dp2ep2_smoke

TOKEN_DUMP_DIR=$(mktemp -d /tmp/afd-token-ubatch.XXXXXX)
export AFD_ASYNC_MOE_PRECISION_DEBUG_DIR=$TOKEN_DUMP_DIR
export AFD_NPU_ASYNC_CAM_E2E_UBATCHING=1
export AFD_NPU_ASYNC_CAM_E2E_SPLIT=token
pytest -s -vv \
  tests/e2e/models/deepseek_v2_lite/test_async_cam_npu.py::test_deepseek_v2_lite_async_cam_attn_dp1tp2_ffn_dp2ep2_smoke

python3 tools/compare_async_cam_tensor_dumps.py \
  "$BASELINE_DUMP_DIR" "$TOKEN_DUMP_DIR"
```

Choose token indices that exist in the fixed prompt. For broad capture, set
`AFD_ASYNC_MOE_PRECISION_DEBUG_FULL_TENSORS=1`; this can consume substantial
host memory and disk space. Without explicit token indices, the dumper keeps
the first and last two valid rows of each local tensor. Override that count
with `AFD_ASYNC_MOE_PRECISION_DEBUG_EDGE_ROWS`.

Optional controls:

| Variable | Meaning |
| --- | --- |
| `AFD_ASYNC_MOE_PRECISION_DEBUG_POINTS` | Comma-separated dump point names; defaults to the primary Attention and FFN boundaries. |
| `AFD_ASYNC_MOE_PRECISION_DEBUG_SYNC=1` | Synchronize the NPU stream before capture for fault localization. |
| `AFD_ASYNC_MOE_PRECISION_DEBUG_FULL_TENSORS=1` | Save every valid row instead of partial samples. |

Each `.pt` file includes shape, dtype, coordinate system, selected indices,
SHA-256, numeric summary, and the sampled CPU tensor. The diagnostic performs
device-to-host copies and must not be used for performance measurements.
