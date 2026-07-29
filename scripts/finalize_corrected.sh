#!/usr/bin/env bash
# Runs on the GB10 once all three corrected Variants finish: verifies every
# final checkpoint's tensors directly, then produces the comparison.
#
# Exists so the verification-and-compare step cannot be forgotten or done
# selectively after seeing partial results. It gates on the *last* Variant
# reaching its final step, so it fires exactly once.
set -u
cd "$(dirname "$0")/.." || exit 1

# Single-instance guard. Two watchers would both verify and both write
# comparison_corrected.txt, racing on the output -- and it is easy to launch
# twice, since an ssh whose command backgrounds a process can time out while
# the process starts fine.
exec 201>/tmp/multihop_finalize.lock
if ! flock -n 201; then
  echo "[$(date -u +%H:%M:%SZ)] FINALIZE: another watcher holds the lock; exiting"
  exit 0
fi
export PATH="$HOME/.local/bin:$PATH"
TOTAL=12000
log(){ echo "[$(date -u +%H:%M:%SZ)] FINALIZE: $*"; }

for v in baseline hybrid kda; do
  until grep -q "^step ${TOTAL}: loss=" "run_logs/${v}_fixed_run.log" 2>/dev/null; do sleep 120; done
  log "$v reached step $TOTAL"
done

log "all three complete -- verifying final checkpoints"
fail=0
for v in baseline hybrid kda; do
  if uv run python scripts/verify_checkpoint.py "checkpoints/${v}_fixed_run/step_${TOTAL}" \
       >> "run_logs/${v}_fixed_verify.log" 2>&1; then
    log "$v: final checkpoint all tensors finite"
  else
    log "$v: *** CHECKPOINT VERIFICATION FAILED -- grid not trustworthy ***"
    echo "$v" >> run_logs/FIXED_FAILED_VARIANTS; fail=1
  fi
done

log "generating comparison against the measured traversal-free ceiling"
uv run python scripts/compare_grids.py runs > run_logs/comparison_corrected.txt 2>&1
[ "$fail" = 1 ] && log "!!! one or more Variants failed verification -- see FIXED_FAILED_VARIANTS"
log "DONE -- run_logs/comparison_corrected.txt"
