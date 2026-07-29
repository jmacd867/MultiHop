# The Degradation Grid saturates at 1.0000; acquisition rate and converged loss are what carry signal

The experiment's intended output is a Degradation Grid per Variant -- accuracy
in each of 25 (hop_count, distance) cells -- compared across full attention,
pure KDA, and the 3:1 hybrid. The first two runs are complete enough to say
what that grid actually shows:

```
baseline (step 20000)          pure-KDA (step 18500)
all 25 cells = 1.0000          all 25 cells = 1.0000

kda minus baseline: +0.00pp in every cell, mean +0.00pp
```

Both Variants answer **every query in every cell correctly**. The Degradation
Grid has no degradation in it. Comparing the two grids is comparing two
matrices of ones.

This is a real result, not a bug: at ~125M parameters and 20,000 steps, both
full attention and pure delta-rule linear attention completely solve 1-5 hop
chained retrieval at distances up to 45 on this task. But it is a **ceiling
result**, and a ceiling tells you the task was too easy, not where the
architectures differ. Read alone it would support the (unjustified) claim
that state compression costs nothing for multi-hop retrieval.

## Two metrics survive saturation, and both are properly controlled

**1. Steps to criterion** -- the first eval step at which a cell reaches 90%
of the headroom above its ADR 0012 shortcut floor. Measured on the same 40
eval grids per Variant that the accuracy view uses, so it costs nothing extra.

```
              overall    by distance (d0..d45)        by hop (h1..h5)
baseline         2660    2600 3100 2500 2500 2600     2800 2600 2700 2600 2600
pure-KDA         4420    3900 3800 4700 5000 4700     4500 4500 4800 4000 4300
```

KDA needs ~1.66x more steps overall, and the *shape* differs: KDA slows with
distance (+29% from d<=3 to d>=9) while the baseline is flat. Hop count is
flat for both.

The obvious objection -- that this is really ADR 0013's distractor-density
confound rather than a distance effect -- does not hold, and the reason is
worth stating. ADR 0013 establishes d=3 as the *highest*-interference column
(100% distractor packing at (1,3)), and the baseline duly is slowest there
(3100). **KDA is fastest at d=3 and slowest at d=21.** Its distance trend runs
*opposite* to the density confound, so density cannot explain it. What remains
is sequence length, which is what state compression would predict.

**2. Converged per-cell training loss** -- higher resolution and a genuinely
controlled comparison. `train_variant.py` drives every Variant from
`np.random.default_rng(0)`, and `sample_training_microbatches` consumes it
deterministically, so **at any given step every Variant trained on the same
cell and the same examples**. A step-matched difference is an architecture
difference with the data held fixed.

The step logs record loss but not the cell; the cell sequence is a pure
function of the seed, so `scripts/loss_by_cell.py` replays it (~2.4 min, CPU,
no GPU contention) and attributes each step. That gives ~800 samples per cell
against 40 eval grids total.

```
kda minus baseline, converged (step >= 10000), positive = KDA fits worse
hop \ dist      0         3         9        21        45
     1    -0.0647   +0.2009   +0.0069   -0.0075   -0.0044
     2    +0.0510   +0.0360   +0.0041   -0.0003   -0.0049
     3    +0.0492   +0.0007   -0.0224   -0.0059   -0.0090
     4    +0.0236   -0.0085   -0.0061   -0.0067   -0.0090
     5    +0.0093   -0.0027   -0.0056   -0.0042   -0.0078
                                                mean +0.0085
```

At convergence KDA matches the baseline almost everywhere -- most cells are
slightly *negative* -- with one clear outlier: **(1,3) at +0.201**, four times
larger than any other cell. That is precisely ADR 0013's fully packed cell,
the only one in the grid where Distractors occupy 100% of available gap space.

The loss trajectory shows the gap is in acquisition, not final quality:

| step window | baseline | KDA | delta | KDA worse on |
|---|---:|---:|---:|---:|
| 1-2000 | 11.71 | 12.38 | +0.66 | 88.3% of steps |
| 2000-5000 | 2.72 | 6.51 | +3.79 | 98.7% |
| 5000-10000 | 1.75 | 1.92 | +0.17 | 82.8% |
| 10000-15000 | 1.704 | 1.712 | +0.007 | 35.8% |
| 15000-18000 | 1.646 | 1.648 | +0.002 | 25.8% |

Note the last row: KDA is *lower* on ~75% of matched steps yet slightly higher
on average, i.e. usually marginally better and occasionally much worse. The
per-cell view localises that minority to (1,3).

## Decision

**Report all three views, with the Degradation Grid stated plainly as
saturated.** The grid remains the headline artifact the experiment set out to
produce and must be shown -- reporting only the metrics that happen to show a
difference would be selecting the analysis after seeing the data. Acquisition
rate and converged loss are reported as what the comparison rests on, with
their weaker epistemic status stated: neither is the capability measure the
experiment was designed around.

**Do not report loss differences as accuracy differences.** Both Variants
answer every cell correctly. A positive loss delta means a looser fit, not
more wrong answers, and the writeup must not blur those.

**Resolution limits are reported alongside.** Steps-to-criterion is quantised
to the 500-step eval cadence; the baseline's cells collapse into only three
distinct values, so its internal ordering is suggestive at best.
`compare_grids.py` prints this warning itself rather than leaving it to a
reader. The cadence cannot be raised retroactively without changing how many
times the shared eval generator is drawn, which would change which examples
every grid is scored on and break ADR 0008's cross-variant comparability.

## The follow-up this makes necessary

Saturation is fixable without retraining. Evaluation examples are generated on
the fly, so the trained checkpoints can be scored on harder configurations --
most usefully a higher `distractor_count`, which attacks saturation and the
ADR 0012 shortcut floor at once (10 Distractors drops the floor from 33.3% to
9.1%; per-cell capacity allows up to 90 at (5,45), for a 1.1% floor).

That is a stress test, **not** an extension of this grid: ADR 0003 draws the
line between measuring capacity on trained-on configurations and extrapolating
to unseen ones, and models trained at `distractor_count=2` scored at 10 are
out of distribution. It must be reported separately and labelled as such. The
distance=0 column can never be hardened this way -- capacity there is 0 -- so
those five cells stay degenerate under any variation.

## Status

Written with the baseline complete (20,000 steps, verified finite) and pure-KDA
at step 18,500 of 20,000. KDA's grids have read 1.0000 continuously since step
16,500 and its per-cell loss has been flat since ~step 10,000, so the figures
above are not expected to move; they are to be confirmed against the completed
run, and the hybrid's column added, before the final writeup.
