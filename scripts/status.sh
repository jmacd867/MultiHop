#!/usr/bin/env bash
# One-screen status of the three-Variant experiment chain.
#
# Usage (from anywhere with network access to the box):
#   ssh kalin@10.70.124.244 'cd ~/coding/multihop-kda-experiment && bash scripts/status.sh'
#
# Answers, in order, the only questions worth asking remotely: is it alive,
# how far along is it, is anything broken, and when will it finish. Written as
# a script on the box rather than a long ssh one-liner so it is testable and
# survives being retyped on a phone.

cd "$(dirname "$0")/.." || exit 1
# Each run's log is the source of truth for its own step count and generator,
# because the corrected re-runs (ADR 0015) use a different total and a
# different task. Hardcoding one TOTAL silently mislabelled them.
RUNS=(baseline_run kda_run hybrid_run baseline_fixed_run kda_fixed_run hybrid_fixed_run)

# Measured steady-state seconds/step (docs/runs/timing_20260728/budget.md).
rate_for() { case "$1" in baseline) echo 0.9338;; kda) echo 1.9977;; hybrid) echo 1.7471;; esac; }

echo "===== multihop experiment status ====="
date "+time:  %Y-%m-%d %H:%M %Z"
uptime -p | sed 's/^/box:   /'

# One line per matching process. `head -1` used to pick the chain launcher,
# whose command line contains `train_variant.py "$v"` and names every variant,
# so it matched all of them and none of them correctly.
trainer_cmds=$(for p in $(pgrep -f "scripts/train_variant.py" 2>/dev/null); do
                 tr "\0" " " < /proc/"$p"/cmdline 2>/dev/null; echo; done)
trainer=$(printf %s "$trainer_cmds" | grep -c "train_variant.py [a-z]")
orch=$(pgrep -f "bash scripts/run_all_variants.sh" | head -1)
echo "procs: ${trainer:-0} trainer proc(s), orchestrator=${orch:-NONE}"
free -g | awk '/^Mem:/{printf "mem:   %sGB/%sGB used (normal 100-107; alarm >116)\n",$3,$2}'
echo

printf "%-18s %-8s %14s %10s %11s %s\n" RUN STATE STEPS LAST_LOSS NON-FINITE TASK
printf -- "----------------------------------------------------------------------------------\n"
remaining_h=0
for r in "${RUNS[@]}"; do
  log="run_logs/${r}.log"
  [ -s "$log" ] || continue        # skip empty logs (created but never run)
  v="${r%%_*}"
  case "$r" in *_fixed_run) want_fixed=1;; *) want_fixed=0;; esac

  # Read the run's own total and generator from the header it printed.
  total=$(grep -m1 -oP 'total_steps=\K[0-9]+' "$log" 2>/dev/null); total=${total:-20000}
  fixed=$(grep -m1 -oP 'randomize_gaps=\K(True|False)' "$log" 2>/dev/null)
  case "$fixed" in True) task="CORRECTED (ADR 0015)";; False) task="original (shortcut)";; *) task="original (shortcut)";; esac

  steps=$(grep -c "^step .*loss=" "$log" 2>/dev/null); steps=${steps:-0}
  last=$(grep "^step .*loss=" "$log" | tail -1 | grep -oP 'loss=\K[0-9.naif-]+')
  nonfinite=$(grep -cE "loss=(nan|inf|-inf)" "$log" 2>/dev/null); nonfinite=${nonfinite:-0}
  pct=$(awk -v s="$steps" -v t="$total" 'BEGIN{printf "%.0f", 100*s/t}')

  if grep -q "^step ${total}: loss=" "$log" 2>/dev/null; then
    state="DONE"
  elif printf %s "$trainer_cmds" | grep "train_variant.py $v " |
       { if [ "$want_fixed" = 1 ]; then grep -q -- "--fixed"; else grep -qv -- "--fixed"; fi; }; then
    state="RUNNING"; remaining_h=$(awk -v r="$remaining_h" -v s="$steps" -v t="$total" -v p="$(rate_for "$v")" 'BEGIN{print r+(t-s)*p/3600}')
  else
    state="stopped"
  fi

  printf "%-18s %-8s %6s/%-7s(%3s%%) %10s %11s %s\n" \
    "$r" "$state" "$steps" "$total" "$pct" "${last:-?}" "$nonfinite" "$task"

  # Latest three eval grids -- the actual experimental signal.
  grep "eval mean accuracy" "$log" 2>/dev/null | tail -3 | sed 's/^/    /'
done

echo
if [ -s run_logs/FAILED_VARIANTS ]; then
  echo "!!! FAILED VERIFICATION -- these grids must NOT be used:"
  sed 's/^/    /' run_logs/FAILED_VARIANTS
fi

if grep -q "CHAIN COMPLETE" run_logs/orchestrator.log 2>/dev/null; then
  echo "CHAIN COMPLETE -- see run_logs/comparison.txt"
elif [ "${trainer:-0}" = "0" ] && [ -z "$orch" ]; then
  echo "*** CHAIN STOPPED and not complete. Restart with:"
  echo "    setsid nohup bash scripts/run_all_variants.sh >> run_logs/orchestrator.log 2>&1 < /dev/null &"
else
  awk -v h="$remaining_h" 'BEGIN{printf "est. remaining: %.1fh\n", h}'
  mins=$(awk -v h="$remaining_h" 'BEGIN{printf "%d", h*60}')
  date -d "+${mins} minutes" "+est. finish:    %a %d %b %H:%M %Z" 2>/dev/null
fi

echo
echo "--- last 3 orchestrator lines ---"
tail -3 run_logs/orchestrator.log 2>/dev/null
