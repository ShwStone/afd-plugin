#!/usr/bin/env bash
# Parse one torch_npu PROF directory into a Chrome timeline JSON and a
# category summary (plan sections 8.2/9.4).
#
# Usage (inside the experiment container):
#   bash tools/benchmarks/fp_parse_trace.sh <PROF_XXX_dir> <summary_output.json>
#
# Steps: msprof --parse (sqlite) -> msprof.py export timeline (Chrome JSON)
# -> tools.benchmarks.profile_trace summarize. The timeline JSON is written
# next to the PROF dir under mindstudio_profiler_output/.
set -euo pipefail

PROF_DIR="${1:?usage: fp_parse_trace.sh <PROF_XXX_dir> <summary_output.json>}"
SUMMARY_OUT="${2:?usage: fp_parse_trace.sh <PROF_XXX_dir> <summary_output.json>}"

set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh
set -u

msprof --parse=on --output="$PROF_DIR"
python3 /usr/local/Ascend/ascend-toolkit/latest/tools/profiler/profiler_tool/analysis/msprof/msprof.py \
  export timeline -dir "$PROF_DIR"

TIMELINE_JSON="$(ls -t "$PROF_DIR"/mindstudio_profiler_output/msprof_*.json | head -1)"
REPO=/a3_inference/itask/workdir/tq02357756/shwstone/code/afd-plugin
cd "$REPO"
python3 -m tools.benchmarks.profile_trace summarize \
  --trace "$TIMELINE_JSON" --output "$SUMMARY_OUT"
echo "summary written: $SUMMARY_OUT (timeline: $TIMELINE_JSON)"
