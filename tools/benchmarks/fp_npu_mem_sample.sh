#!/usr/bin/env bash
# Sample per-NPU HBM usage into a TSV while a long-input replay runs
# (plan section 8.2.3 peak-memory record). Usage:
#   fp_npu_mem_sample.sh <output.tsv> <duration_s> [interval_s]
# Columns: epoch_s <tab> phy_id <tab> hbm_used_mb
set -u
OUT="${1:?output tsv}"
DUR="${2:?duration s}"
INT="${3:-2}"
END=$((SECONDS + DUR))
: > "$OUT"
while [ "$SECONDS" -lt "$END" ]; do
  npu-smi info | awk -F'|' -v now="$(date +%s)" '
    # Chip rows carry the Bus-Id in field 3; field 4 packs
    # "AICore(%)  MemUsed / MemTotal  HBMUsed / HBMTotal" — HBM is the
    # LAST num/num pair. Field 2 is "<chip> <phy-id>".
    $3 ~ /[0-9A-Fa-f]{4}:/ {
      f2 = $2
      gsub(/^[[:space:]]+/, "", f2)
      split(f2, ids, /[[:space:]]+/)
      line = $4
      if (match(line, /[0-9]+[[:space:]]*\/[[:space:]]*[0-9]+[[:space:]]*$/)) {
        pair = substr(line, RSTART, RLENGTH)
        split(pair, ab, "/")
        used = ab[1]
        gsub(/[^0-9]/, "", used)
        if (used != "") print now "\t" ids[2] "\t" used
      }
    }' >> "$OUT"
  sleep "$INT"
done
