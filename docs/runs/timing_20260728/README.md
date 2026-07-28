# Wall-clock budget for the three real 20,000-step runs, GB10, 2026-07-28

Measured budgets for all three Variants at their real settings, produced by
the rewritten `scripts/timing_run.py` before committing any GB10 time to the
actual experimental runs. Supersedes
`docs/runs/baseline_run_20260727/timing_run.csv` as the current budget source.

## Why the earlier baseline figure was re-measured rather than reused

The prior measurement reported **5.99h** for the baseline. That number is
biased, for two independent reasons, and both are fixed here.

1. **Frequency-weighted, not uniform.** It averaged the steady-state steps of
   a 150-step *random* window. But `train()` samples the 25 Grid Cells
   uniformly, and per-step time varies ~18x across them (0.19s at seq_len=5
   to 3.42s at seq_len=287). The random window hit individual cells between 1
   and 10 times, so its raw mean reflects which cells happened to come up.
   Re-weighting that same committed CSV's per-cell means equally gives
   **1.0231s/step -> 5.68h**, not 1.0781s -> 5.99h.
2. **Measured at the pre-ADR-0008 vocab.** It predates the 16,003 -> 8,003
   `vocab_size` shrink. The re-measurement puts the baseline at **0.9338s**
   uniform-weighted, ~9% faster than the same calculation on the old CSV,
   which is the direction and rough magnitude a smaller tied embedding table
   predicts.

The new sweep is deterministic -- every cell hit `STEPS_PER_CELL=4` times --
so no cell rests on n=1, and the resulting number is the uniform-weighted
quantity directly rather than a correction applied afterward.

## Two things no prior budget in this project included

**`evaluate_grid` was never a line item.** The real runs call it every
`eval_every=500` steps: 40 grids x 25 cells x 512 examples. Measured here for
the first time. The cost turns out to be dominated by one-time XLA op
compilation, not by the eval work itself -- for the baseline, the first grid
costs 382s and every subsequent grid costs **25s**, a 15x drop (the worst
cell, seq_len=287 at 512 examples, goes 68.2s -> 4.58s). Over a full run that
is 0.38h, small enough that `eval_every=500` needs no adjustment.

**The 512-example eval path had never been run for the KDA-containing
Variants.** Both sanity runs used 32 examples/cell, so a 512-example forward
at seq_len=287 through KDA layers was untested memory territory that would
have first executed at step 500 of a 20,000-step run. Clearing it here cost
minutes instead.

## Correction: the script's memory column was measuring the wrong thing

`timing_run.py` originally reported peak process RSS from
`/proc/self/status`. On GB10 that is close to meaningless: measured
mid-baseline-run, `VmHWM` read **5.4GB while the box was 100GB deep**, of
which `nvidia-smi` attributed **93,684 MiB to that PID's own preallocated
JAX arena**. Unified memory means the device arena never appears in the
process's RSS, so the column understated actual load by ~17x -- it would have
rendered the 118-121GB kill zone of ADR 0009 and ADR 0010 as idle.

The metric is now system-wide `MemTotal - MemAvailable`, which is what those
two ADRs were actually tracking, and which on a shared box correctly counts
other users' load too. **The `baseline` log/CSV in this directory still
carries the old RSS-based column** (it ran before the fix); its `peak_host`
values are process RSS and should be ignored. The `kda` and `hybrid` logs
carry the corrected `sys_mem` column.

Steady-state memory for a running Variant is ~100GB/121GB, of which ~94GB is
JAX's fixed startup arena. That is normal operating level, not growth, and
should not be mistaken for the runaway climb that killed the two earlier
attempts.

## Results

See **`budget.md`** for the per-Variant table and the analysis of how the
cost ratio varies with sequence length. Raw data: `timing_run_<variant>.csv`,
`timing_<variant>.log`.

Headline: baseline **5.65h**, pure-KDA **12.31h**, hybrid **10.74h**;
~28.7h sequential for all three.

## What these numbers do and do not establish

They are wall-clock and memory measurements on an **untrained, freshly
initialized** model. They establish what the runs will cost and that they fit
in memory at their real settings. They establish **nothing** about task
performance: the accuracy column in the eval phase is at initialization and
is chance by construction, and the losses in the sweep phase are first-touch
losses on 4 steps per cell, not a learning curve.
