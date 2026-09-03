#!/usr/bin/env python3
"""Offline-parse torch_npu profiler rank dirs via torch_npu's analyse API.

torch_npu workers are daemon processes, so the in-process Text/Db export is
skipped at stop time ("profiling data cannot be parsed during the daemon
process") and only raw FRAMEWORK/ + PROF_* data lands on disk.  This script
runs the recommended offline `analyse()` per rank dir, generating
ASCEND_PROFILER_OUTPUT with text CSVs / trace_view.json and/or the
ascend_pytorch_profiler_{rank}.db that MindStudio Insight imports.

Usage (inside the experiment container):
    python3 tools/benchmarks/fp_offline_analyse.py <root_with_rank_dirs> [parallel] [export_types]

    export_types: comma-separated subset of db,text (default: db,text)

Idempotent: a rank dir whose requested outputs already exist is skipped.
"""
from __future__ import annotations

import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed


def find_rank_dirs(root: str) -> list[str]:
    return sorted(
        os.path.join(root, d)
        for d in os.listdir(root)
        if d.endswith("_ascend_pt") and os.path.isdir(os.path.join(root, d))
    )


def has_output(rank_dir: str, export_types: list[str]) -> bool:
    out = os.path.join(rank_dir, "ASCEND_PROFILER_OUTPUT")
    if not os.path.isdir(out):
        return False
    want_db = "db" in export_types
    want_text = "text" in export_types
    if want_db and not any(f.endswith(".db") for f in os.listdir(out)):
        return False
    if want_text and not any(f.endswith(".csv") or f == "trace_view.json" for f in os.listdir(out)):
        return False
    return True


def analyse_one(rank_dir: str, export_types: list[str]) -> str:
    if has_output(rank_dir, export_types):
        return f"SKIP {rank_dir}"
    try:
        from torch_npu.profiler.profiler import analyse

        analyse(rank_dir, max_process_number=4, export_type=list(export_types))
    except Exception as exc:  # noqa: BLE001 - report and continue with others
        return f"FAIL {rank_dir} {exc!r}"
    if has_output(rank_dir, export_types):
        return f"OK   {rank_dir}"
    return f"FAIL {rank_dir} analyse returned but ASCEND_PROFILER_OUTPUT incomplete"


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    root = sys.argv[1]
    parallel = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    export_types = (sys.argv[3].split(",") if len(sys.argv) > 3 else ["db", "text"])
    dirs = find_rank_dirs(root)
    print(f"analysing {len(dirs)} rank dirs under {root} (parallel={parallel}, types={export_types})")
    rc = 0
    with ProcessPoolExecutor(max_workers=parallel) as pool:
        futs = {pool.submit(analyse_one, d, export_types): d for d in dirs}
        for fut in as_completed(futs):
            line = fut.result()
            print(line, flush=True)
            if line.startswith("FAIL"):
                rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
