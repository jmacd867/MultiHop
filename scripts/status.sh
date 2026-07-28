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
VARIANTS=(baseline kda hybrid)
TOTAL=20000

# Measured steady-state seconds/step (docs/runs/timing_20260728/budget.md).
rate_for() { case "$1" in baseline) echo 0.9338;; kda) echo 1.9977;; hybrid) echo 1.7471;; esac; }

echo "===== multihop experiment status ====="
date "+time:  %Y-%m-%d %H:%M %Z"
uptime -p | sed 's/^/box:   /'

trainer=$(pgrep -f "scripts/train_variant.py" | head -1)
orch=$(pgrep -f "bash scripts/run_all_variants.sh" | head -1)
echo "procs: trainer=${trainer:-NONE} orchestrator=${orch:-NONE}"
free -g | awk '/^Mem:/{printf "mem:   %sGB/%sGB used (normal 100-107; alarm >116)\n",$3,$2}'
echo

remaining_h=0
for v in "${VARIANTS[@]}"; do
  log="run_logs/${v}_run.log"
  if [ ! -f "$log" ]; then
    printf "%-9s not started\n" "$v"
    remaining_h=$(awk -v r="$remaining_h" -v t="$TOTAL" -v s="$(rate_for "$v")" 'BEGIN{print r + t*s/3600}')
    continue
  fi

  steps=$(grep -c "^step .*loss=" "$log" 2>/dev/null); steps=${steps:-0}
  last=$(grep "^step .*loss=" "$log" | tail -1 | grep -oP 'loss=\K[0-9.naif-]+')
  nonfinite=$(grep -cE "loss=(nan|inf|-inf)" "$log" 2>/dev/null); nonfinite=${nonfinite:-0}
  pct=$(awk -v s="$steps" -v t="$TOTAL" 'BEGIN{printf "%.1f", 100*s/t}')

  if grep -q "^step ${TOTAL}: loss=" "$log" 2>/dev/null; then
    state="DONE"
  elif [ "$trainer" != "" ] && [ "$(tr -d ' ' < /proc/"$trainer"/cmdline 2>/dev/null | grep -c "$v")" != "0" ]; then
    state="RUNNING"
  else
    state="stopped"
  fi

  printf "%-9s %-8s %6s/%d (%s%%)  last_loss=%s  non-finite=%s\n" \
    "$v" "$state" "$steps" "$TOTAL" "$pct" "${last:-?}" "$nonfinite"

  # Latest three eval grids -- the actual experimental signal.
  grep "eval mean accuracy" "$log" 2>/dev/null | tail -3 | sed 's/^/            /'

  if [ "$state" != "DONE" ]; then
    remaining_h=$(awk -v r="$remaining_h" -v s="$steps" -v t="$TOTAL" -v p="$(rate_for "$v")" \
      'BEGIN{print r + (t-s)*p/3600}')
  fi
done

echo
if [ -s run_logs/FAILED_VARIANTS ]; then
  echo "!!! FAILED VERIFICATION -- these grids must NOT be used:"
  sed 's/^/    /' run_logs/FAILED_VARIANTS
fi

if grep -q "CHAIN COMPLETE" run_logs/orchestrator.log 2>/dev/null; then
  echo "CHAIN COMPLETE -- see run_logs/comparison.txt"
elif [ -z "$trainer" ] && [ -z "$orch" ]; then
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
