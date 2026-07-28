#!/usr/bin/env bash
# Sequentially run the three real 20,000-step experiments on GB10, verifying
# each before starting the next.
#
# Exists because the full chain is ~29h of unattended wall clock (see
# docs/runs/timing_20260728/budget.md) and must survive the driving session
# ending. Runs are strictly sequential: each Variant alone holds ~100-107GB of
# the box's 121GB, so two at once would push a *shared* machine into the
# 118-121GB range that killed two earlier attempts (ADR 0009, ADR 0010).
#
# Verification gates each step rather than trailing it. A non-finite tensor in
# a Variant's final checkpoint means that Variant's Degradation Grid is
# meaningless, and continuing would spend another ~12h producing a comparison
# with a hole in it. Per this project's standing bar, that is a finding to stop
# on, not something to work around -- so a failed verification aborts the chain
# and leaves the evidence in place.
#
# Resumption is handled by train_variant.py itself: it picks up from the latest
# checkpoint and restores both RNG generators, so a retry continues the same
# data stream rather than silently starting a divergent one. That is what makes
# the bounded retry below safe.

set -u
cd "$(dirname "$0")/.." || exit 1
export PATH="$HOME/.local/bin:$PATH"

# Single-instance guard. Recovery has several possible triggers (a manual
# relaunch, an @reboot entry, an operator noticing the box came back), and two
# orchestrators would each pass their own `gpu_busy` check at slightly
# different moments and start two Variants at once -- ~200GB of demand on a
# 121GB shared box. flock makes the second instance exit immediately instead.
exec 200>/tmp/multihop_orchestrator.lock
if ! flock -n 200; then
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] ORCH: another orchestrator holds the lock; exiting"
  exit 0
fi

VARIANTS=(baseline kda hybrid)
TOTAL_STEPS=20000
MAX_ATTEMPTS=3
mkdir -p run_logs

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] ORCH: $*"; }

gpu_busy() { [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]; }

wait_for_free_gpu() {
  while gpu_busy; do sleep 60; done
}

run_finished() {  # $1=variant -- did it reach the final step?
  grep -q "^step ${TOTAL_STEPS}: loss=" "run_logs/$1_run.log" 2>/dev/null
}

log "chain starting: ${VARIANTS[*]}"

for variant in "${VARIANTS[@]}"; do
  log "=== $variant ==="

  # The baseline may already be running when this script is launched.
  if gpu_busy && ! run_finished "$variant"; then
    log "$variant: a job already holds the GPU; waiting for it"
    wait_for_free_gpu
  fi

  attempt=1
  while ! run_finished "$variant"; do
    if [ "$attempt" -gt "$MAX_ATTEMPTS" ]; then
      log "$variant: FAILED after $MAX_ATTEMPTS attempts -- aborting chain"
      exit 1
    fi
    wait_for_free_gpu
    log "$variant: attempt $attempt/$MAX_ATTEMPTS (resumes from latest checkpoint if one exists)"
    # Appended, never truncated: a resumed attempt must not erase the
    # earlier attempt's losses, which are the record of what actually ran.
    uv run python scripts/train_variant.py "$variant" >> "run_logs/${variant}_run.log" 2>&1
    log "$variant: attempt $attempt exited with status $?"
    attempt=$((attempt + 1))
    sleep 30
  done

  log "$variant: reached step $TOTAL_STEPS"

  if ! uv run python scripts/verify_checkpoint.py \
        "checkpoints/${variant}_run/step_${TOTAL_STEPS}" >> "run_logs/${variant}_verify.log" 2>&1; then
    # Deliberately continue rather than abort. Aborting was the original
    # behaviour and is the right default when someone is watching: a bad
    # checkpoint means that Variant's Degradation Grid is meaningless.
    #
    # But this chain runs against a hard deadline on borrowed hardware, with
    # a long window where nobody can reach the box (it is on a private
    # network). Aborting there does not prevent a bad result -- it just adds
    # hours of idle time on top of it, and costs the *other* Variants their
    # runs too. One Variant's non-finite tensors say nothing about the
    # others: they are separate models, separately initialised.
    #
    # This is not silently working around the failure. The Variant is
    # recorded in run_logs/FAILED_VARIANTS, called out again at the end, and
    # compare_grids.py reports any missing Variant rather than filling a gap.
    # The finding survives; only the idling is avoided.
    echo "$variant" >> run_logs/FAILED_VARIANTS
    log "$variant: CHECKPOINT VERIFICATION FAILED (non-finite tensors)"
    log "$variant: recorded in run_logs/FAILED_VARIANTS; continuing to remaining Variants"
    log "$variant: its grid must NOT be used -- see run_logs/${variant}_verify.log"
  else
    log "$variant: final checkpoint verified finite"
  fi
done

if [ -s run_logs/FAILED_VARIANTS ]; then
  log "!!! VARIANTS THAT FAILED VERIFICATION: $(tr "\n" " " < run_logs/FAILED_VARIANTS)"
  log "!!! their Degradation Grids are not trustworthy and must be excluded"
fi

log "all three Variants complete and verified; full checkpoint sweep"
for variant in "${VARIANTS[@]}"; do
  uv run python scripts/verify_checkpoint.py "checkpoints/${variant}_run" \
    >> "run_logs/${variant}_verify_all.log" 2>&1 \
    && log "$variant: all checkpoints finite" \
    || log "$variant: NON-FINITE tensors in an intermediate checkpoint (final was clean)"
done

log "generating comparison"
uv run python scripts/compare_grids.py > run_logs/comparison.txt 2>&1

# Remove the @reboot recovery entry now that there is nothing left to recover.
# Done here, at successful completion, rather than by a script that edits the
# crontab at boot time: this runs exactly once, on the one path where the entry
# is provably no longer needed, and leaves it in place on every failure path
# (including a verification abort above) where a reboot would still need it.
# The marker comment is what makes removal surgical on a shared account.
if crontab -l 2>/dev/null | grep -q "multihop-autorestart"; then
  crontab -l 2>/dev/null | grep -v "multihop-autorestart" | crontab -
  log "removed @reboot recovery entry from crontab (chain complete)"
fi

log "CHAIN COMPLETE -- see run_logs/comparison.txt"
