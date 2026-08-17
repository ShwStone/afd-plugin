#!/usr/bin/env bash
# Trash helper: move files to a timestamped recycle bin instead of rm.
# Usage: bash tools/benchmarks/trash.sh <file-or-glob>...
#   - Files are moved (not deleted) into bench_results/trash/<timestamp>/
#   - If a file does not exist, it's skipped silently (globs may match nothing)
set -u

TRASH_ROOT="${TRASH_ROOT:-bench_results/trash}"
TS="$(date +%Y%m%d_%H%M%S)"
DEST="${TRASH_ROOT}/${TS}"
mkdir -p "$DEST"

moved=0
for src in "$@"; do
  # Expand globs manually (bash already expands unquoted args; quoted patterns stay literal)
  if [ -e "$src" ]; then
    mv "$src" "$DEST/" 2>/dev/null && { echo "[trash] $src -> $DEST/"; moved=$((moved+1)); }
  elif [ -e "$TRASH_ROOT" ]; then
    # try glob expansion for quoted patterns
    for f in $src; do
      [ -e "$f" ] && mv "$f" "$DEST/" 2>/dev/null && { echo "[trash] $f -> $DEST/"; moved=$((moved+1)); }
    done
  fi
done

echo "[trash] moved $moved file(s) to $DEST"
