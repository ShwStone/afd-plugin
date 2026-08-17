#!/usr/bin/env bash
# Summarize Stage-2 L0 collection progress. Runs locally (reads the nohup log
# and counts verified results on the container). Used by the cron monitor.
set -u

NODE0=${NODE0:-afd-exp-2}
REPO=${REPO:-/a3_inference/itask/workdir/tq02357756/shwstone/code/afd-plugin}
# Auto-detect the active phase log (newest stage2_*.log / stage3_*.log that a
# runner is using, falling back to the most recently modified one).
if [ -n "${LOG:-}" ]; then :; else
  LOG=$(ls -t bench_results/prefill_stage2/stage2_*.log \
           bench_results/prefill_stage3/stage3_*.log 2>/dev/null | head -1)
fi
LOG=${LOG:-bench_results/prefill_stage2/stage2_l0_main.log}

echo "=== $(date '+%Y-%m-%d %H:%M:%S') ==="
if [ ! -f "$LOG" ]; then echo "no log yet: $LOG"; exit 0; fi

echo "--- runner alive? ---"
pgrep -af "run_stage2_l0[.]sh" >/dev/null 2>&1 && echo "RUNNING" || echo "NOT-RUNNING"
echo "--- log tail ---"
tail -4 "$LOG" | tr '\r' '\n' | grep -v '^\s*$' | tail -4
echo "--- groups [DONE]/[SKIP] ---"
grep -ac "\[DONE\]\|\[SKIP\]" "$LOG" || true
echo "--- [FAIL] / reset failures ---"
grep -aE "\[FAIL\]|RESET-FAIL" "$LOG" | tail -3 || true

echo "--- verified results on container ---"
itask exec "$NODE0" --tty=false -- bash -c '
  echo "total (stage2+stage3):"
  find /a3_inference/itask/workdir/tq02357756/shwstone/code/afd-plugin/bench_results/prefill_stage2 \
       /a3_inference/itask/workdir/tq02357756/shwstone/code/afd-plugin/bench_results/prefill_stage3 \
       -name "*.verified.json" 2>/dev/null | wc -l
  echo "stage2 by group (02_e2e):"
  find /a3_inference/itask/workdir/tq02357756/shwstone/code/afd-plugin/bench_results/prefill_stage2/02_e2e -name "*.verified.json" 2>/dev/null \
    | sed -E "s#.*/02_e2e/([^/]+)/[^/]+/.*#\1#" | sort | uniq -c
  echo "stage3 by group (01_sweep):"
  find /a3_inference/itask/workdir/tq02357756/shwstone/code/afd-plugin/bench_results/prefill_stage3/01_sweep -name "*.verified.json" 2>/dev/null \
    | sed -E "s#.*/01_sweep/[^/]+/([^/]+)/.*#\1#" | sort | uniq -c
' 2>&1 || echo "(itask exec failed)"
