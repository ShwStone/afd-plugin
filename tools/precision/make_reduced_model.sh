#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
# Create a reduced-layer copy of the DeepSeek-V3.2 W8A8 model on the shared NAS.
#
# The full 61-layer model costs 6-8 min to start and, with TP8 captures on
# every layer, dominates the diagnostic runtime. A truncated config pointing at
# the same weight shards (via symlinks, no large copies) loads only the first
# N layers, which is enough to exercise the Async CAM ubatch mechanism while
# keeping restarts fast.
#
#   SRC=/home/admin/model-csi/model \
#   DST=<shared-nas>/bench_results/model_l8 \
#   LAYERS=8 \
#   bash tools/precision/make_reduced_model.sh
set -eu

SRC="${SRC:?SRC model dir required}"
DST="${DST:?DST reduced-model dir required}"
LAYERS="${LAYERS:-8}"

mkdir -p "$DST"
python3 - "$SRC" "$DST" "$LAYERS" <<'PY'
import json
import sys

src, dst, layers = sys.argv[1], sys.argv[2], int(sys.argv[3])
with open(f"{src}/config.json", encoding="utf-8") as fh:
    config = json.load(fh)
config["num_hidden_layers"] = layers
with open(f"{dst}/config.json", "w", encoding="utf-8") as fh:
    json.dump(config, fh, indent=2)
print(f"config.json num_hidden_layers -> {layers}")
PY

for name in \
    configuration.json \
    generation_config.json \
    quant_model_description.json \
    tokenizer.json \
    tokenizer_config.json; do
  cp "$SRC/$name" "$DST/$name"
done

ln -sf "$SRC"/quant_model_weights-*.safetensors "$DST/" 2>/dev/null || true
ln -sf "$SRC/quant_model_weights.safetensors.index.json" "$DST/" 2>/dev/null || true

echo "created reduced model at $DST (layers=$LAYERS)"
ls "$DST" | wc -l
