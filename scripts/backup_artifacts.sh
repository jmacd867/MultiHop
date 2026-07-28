#!/usr/bin/env bash
# Pull the small, irreplaceable run artifacts off the GB10 into this repo.
#
# Usage: bash scripts/backup_artifacts.sh   (safe to run repeatedly; idempotent)
#
# Every result this experiment produces currently exists in exactly one place:
# a shared, borrowed box that is scheduled to be wiped, and that has already
# gone unreachable once mid-run (2026-07-28 22:55Z, requiring a physical
# power-on). Losing it would cost ~29h of GPU time that cannot be re-bought
# before the deadline.
#
# What is copied is deliberately only the small stuff -- eval grids, logs,
# verification output. Together they are a few MB and they are the actual
# deliverable: the Degradation Grids, the learning curves behind them, and the
# evidence that each run was healthy. `scripts/compare_grids.py` runs off these
# files alone, so a full comparison can be regenerated locally even if the box
# never comes back.
#
# Checkpoints (~90GB across three Variants) are NOT copied. They are needed to
# resume a run or to do ADR 0005's attention capture, but not to produce or
# read the grids, and pulling them over ssh would take longer than re-running
# some of the training. If they are wanted before the box is wiped, that is a
# deliberate, separate decision -- see the note at the end of this script.

set -u
cd "$(dirname "$0")/.." || exit 1
REMOTE=kalin@10.70.124.244
REMOTE_DIR=/home/kalin/coding/multihop-kda-experiment
DEST=docs/runs/experiment_20260728

if ! timeout 20 ssh -o ConnectTimeout=12 -o BatchMode=yes "$REMOTE" 'exit 0' 2>/dev/null; then
  echo "GB10 unreachable -- nothing pulled (previous backup, if any, is untouched)"
  exit 1
fi

mkdir -p "$DEST"
echo "pulling run artifacts -> $DEST"

# --update so a stale local copy never overwrites a newer remote one, and so
# repeated runs are cheap. No --delete: a partial remote state must never
# remove something already banked locally.
rsync -az --update -e ssh \
  --include='*/' \
  --include='*_grids.json' --include='*.log' --include='*.txt' --include='FAILED_VARIANTS' \
  --exclude='*' \
  "$REMOTE:$REMOTE_DIR/runs/" "$REMOTE:$REMOTE_DIR/run_logs/" "$DEST/" 2>/dev/null

echo "--- banked locally ---"
ls -la "$DEST" 2>/dev/null | tail -n +2
echo
for f in "$DEST"/*_grids.json; do
  [ -e "$f" ] || continue
  n=$(python3 -c "import json,sys; print(len(json.load(open(sys.argv[1]))['evals']))" "$f" 2>/dev/null)
  echo "$(basename "$f"): ${n:-?} eval grids"
done

echo
echo "NOT copied: checkpoints/ (~90GB). Needed only to resume training or for"
echo "ADR 0005 attention capture -- not to read or regenerate the grids."
