# Measured 20,000-step budget, all three Variants, GB10, 2026-07-28

All figures measured, not extrapolated: a deterministic sweep of all 25 Grid
Cells at `STEPS_PER_CELL=4`, at the real settings every Variant runs at
(`batch_size=256`, `grad_accum_steps=16`, `micro_batch_size=16`), plus two
timed `evaluate_grid` passes at the real `EVAL_EXAMPLES_PER_CELL=512`.

| Variant | params | per-step | train | compile (17 shapes) | eval (40 grids) | **total** |
|---|---:|---:|---:|---:|---:|---:|
| baseline (full attention) | 119,411,712 | 0.9338s | 5.19h | 0.09h (312s) | 0.38h (25s/grid) | **5.65h** |
| pure-KDA | 134,877,072 | 1.9977s | 11.10h | 0.13h (470s) | 1.08h (89s/grid) | **12.31h** |
| hybrid 3:1 | 131,010,732 | 1.7471s | 9.71h | 0.14h (500s) | 0.90h (71s/grid) | **10.74h** |

**Sequential total: ~28.7h.** Run one at a time -- the box is shared, and each
Variant alone holds ~100-107GB of its 121GB.

All three param counts match the figures independently recorded in the earlier
sanity-run READMEs, confirming the models built here are the ones those runs
validated.

## Cost ratio vs. the baseline is chunk-quantized, not smooth

Per-cell ratios are not a constant multiplier, and the structure is worth
recording because it explains the shape rather than averaging it away:

| seq_len | 5-29 | 35-47 | 59-143 | 191-287 |
|---|---:|---:|---:|---:|
| kda / baseline | 1.58-1.84 | 2.45-2.72 | 2.11-2.49 | 2.00-2.11 |
| hybrid / baseline | 1.42-1.78 | 2.11-2.31 | 1.84-2.15 | 1.76-1.84 |

The jump between seq_len=29 and seq_len=35 is `kda_chunked_scan`'s
`chunk_size=32` (ADR 0009): every cell up to 32 tokens is a **single** chunk,
and 35 tokens is the first to need **two**, doubling the scanned work while
the baseline's cost grows smoothly. From there the ratio *declines* with
length, because full attention is O(n²) in sequence length while the KDA
recurrence is O(n) -- so the baseline progressively closes the gap, reaching
2.00x for KDA and 1.76x for hybrid at the longest cell (seq_len=287).

This matters for anyone tempted to extrapolate: a two-point measurement taken
below and above 32 tokens would have implied a cost ratio that keeps climbing
with sequence length. The opposite is true past the chunk boundary. The
deterministic all-25-cell sweep is what makes this visible.

The hybrid sits consistently *below* pure-KDA (10.74h vs 12.31h, ~87%),
which is the direction ADR 0011's 9-KDA/3-full-attention schedule predicts.

## Eval is real but not worth restructuring for

`evaluate_grid` had never appeared in any budget in this project. It is
dominated by one-time XLA op compilation rather than by the eval work: the
first grid costs 382-464s and every subsequent grid costs 25s (baseline),
71s (hybrid), or 89s (KDA).

Raising `eval_every` from 500 was considered and **rejected**. It cannot
reduce the one-time first-grid cost, only the per-grid remainder, so
quartering the cadence would save ~0.8h of KDA's 12.31h (~6%) -- while
removing three quarters of the opportunities to notice a NaN or a collapse
early in a 12-hour unattended run. The monitoring is worth more than the
6%.

## Memory: measured, and the previously untested path is now cleared

Steady-state is ~100-107GB of 121GB for every Variant, of which ~94GB is
JAX's fixed startup arena (confirmed via `nvidia-smi` against the running
PID). That is a constant, not growth, and is not comparable to the 118-121GB
runaway climbs that killed the two earlier attempts (ADR 0009, ADR 0010).

Peak system memory: baseline not comparable (its column was the pre-fix RSS
metric -- see README.md), KDA 105.0GB, hybrid 106.9GB.

**The 512-example eval path is no longer untested for the KDA-containing
Variants.** Both sanity runs used 32 examples/cell, so a 512-example forward
at seq_len=287 through KDA layers had never executed and would first have run
at step 500 of a 20,000-step run. It completes: 70.1s for KDA and 65.7s for
hybrid on that worst cell, with no OOM.

## What these numbers do not establish

Nothing about task performance. Every accuracy figure in these sweeps is from
an untrained, freshly initialized model and is chance by construction; the
losses are first-touch values over 4 steps per cell, not a learning curve.
These runs establish cost and memory feasibility only.
