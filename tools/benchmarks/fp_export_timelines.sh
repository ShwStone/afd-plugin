#!/usr/bin/env bash
# Batch-export torch_npu PROF dirs to Chrome timeline JSONs (msprof parse +
# export timeline), parallel with xargs. Idempotent: skips dirs that already
# have mindstudio_profiler_output/msprof_*.json.
#
# Usage (inside the experiment container):
#   bash tools/benchmarks/fp_export_timelines.sh <trace_dir_with_rank_dirs> [parallel]
# Example:
#   bash tools/benchmarks/fp_export_timelines.sh .../traces/baseline 8
set -uo pipefail

ROOT="${1:?usage: fp_export_timelines.sh <trace_dir> [parallel]}"
PAR="${2:-8}"

set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh
set -u

export MSPY=/usr/local/Ascend/ascend-toolkit/latest/tools/profiler/profiler_tool/analysis/msprof/msprof.py

export_one() {
    dir="$1"
    # timeline exports land in <rank_dir>/PROF_*/mindstudio_profiler_output/
    if ls "$dir"/PROF_*/mindstudio_profiler_output/msprof_*.json >/dev/null 2>&1; then
        echo "SKIP $dir"
        return 0
    fi
    msprof --parse=on --output="$dir" >/dev/null 2>&1
    python3 "$MSPY" export timeline -dir "$dir" >/dev/null 2>&1
    if ls "$dir"/PROF_*/mindstudio_profiler_output/msprof_*.json >/dev/null 2>&1; then
        echo "OK   $dir"
    else
        echo "FAIL $dir"
    fi
}
export -f export_one
export MSPY

# vllm's profiler names dirs dp{d}_.._rank{g}_.._ascend_pt; the AFD plugin
# profiler names them <host>_<pid>_<ts>_ascend_pt — accept both.
find "$ROOT" -maxdepth 1 -type d -name '*_ascend_pt' -print0 |
    xargs -0 -P "$PAR" -I{} bash -c 'export_one "$@"' _ {}
