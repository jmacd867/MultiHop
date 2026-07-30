# The corrected experiment: three 20k-equivalent runs on a task that requires traversal

**Status: COMPLETE.** All three Variants ran 12,000 steps on the corrected
generator, were verified at tensor level, and are compared below.

The methodology sections were written and committed *before* any result
existed -- the 0.4803 traversal-free bar, the 16-cell valid surface, and what
the experiment can and cannot answer. That ordering is deliberate: this
generator produced five separate shortcuts over the project's life, every one
invisible in the accuracy number, so the reading had to be fixed in advance to
avoid being shaped by the result.

**Headline: the 3:1 hybrid matches full attention (+1.28pp, 0.6 SE). Pure-KDA
does not (-11.90pp, 5.4 SE).**

## Why this exists at all

The first experiment (`docs/runs/experiment_20260728/`) produced 1.0000 in all
25 cells for both completed Variants and was invalidated: `probe_mechanism.py`
measured both models solving the task by **copying a fixed token position**,
never reading the Query (ADR 0015, ADR 0016). A second attempt, with gap
lengths randomised, was abandoned at 4,745 steps when a **Fact-ordering**
shortcut worth 45-69% was found still live (ADR 0018).

This run uses both fixes together: `randomize_gaps` **and**
`randomize_fact_order`.

## Configuration

| | |
|---|---|
| steps | 12,000 per Variant (not 20,000 -- the corrected task is harder and the budget is fixed) |
| eval cadence | every 250 steps (finer than the original 500, for resolution near the ceiling) |
| batch | effective 256 via `grad_accum_steps=16` (ADR 0006, ADR 0010) -- unchanged |
| order | baseline -> hybrid -> pure-KDA, sequential, identical budget each |
| generator | `randomize_gaps=True`, `randomize_fact_order=True` |

Run order puts **hybrid second, before pure-KDA**, because hybrid-vs-transformer
is the comparison this project was commissioned to make; if the deadline forces
a cut, the priority pair completes.

## The bar: 0.4803, not chance and not 0.333

`scripts/shortcut_ceiling.py` computes the best score reachable **without
traversing the Chain**, on this exact generator, before any result is read.
This is ADR 0018's stated lesson: after fixing a shortcut, the question is not
"is the old shortcut gone" but "what is the best score achievable without the
capability I am trying to measure".

Six traversal-free strategies were scored per cell. The strongest is **"the
object that never appears as a subject, and whose own subject does appear as an
object"** -- pure role statistics, no Chain following. It filters Distractors
hanging off the Query entity, whose subject is never an object:

| hop_count | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|
| traversal-free ceiling | 0.60 | 0.47 | 0.45 | 0.41 |

**Mean over the valid surface: 0.4803.** Per-cell values are cached in
`docs/runs/shortcut_ceiling_corrected.json` and `compare_grids.py` reports
against them, warning when fewer than 75% of cells clear their own ceiling.

Note the ceiling *declines* with hop_count, because Distractors attach to a
uniformly random Chain entity so fewer are filtered at greater depth. **A
result whose excess over the ceiling grows with hop_count is therefore the
opposite of what a surface heuristic produces**, and is the signature to look
for.

## The valid surface is 16 cells, not 25

Two exclusions, both structural and both established before this run:

- **distance=0 (5 cells)** -- no Distractors fit, so the appears-once rule
  (ADR 0012) scores 1.0. Degenerate; cannot discriminate between Variants.
- **hop_count=1 (5 cells)** -- every Distractor's subject is the Query entity
  itself (`rng.integers(0, 1)` is always 0), and with no second hop the true
  target never reappears as a subject. Content cannot identify the answer
  (ADR 0017), and what remains is an ordering artifact (ADR 0018).

The remaining **hop_count >= 2, distance >= 3** cells have a uniform 1.0
ceiling and a known traversal-free floor. That is the only surface on which
Variants are compared.

## What this experiment can and cannot answer

**Can:** whether the 3:1 hybrid matches full attention at matched hop_count and
distance, on a task where the answer is not recoverable by position, ordering,
or occurrence counting.

**Cannot:** how performance degrades with chain length. The hop axis is
confounded -- the traversal-free ceiling itself varies with hop_count
(0.60 -> 0.41), so raw accuracy across that axis mixes capability with a moving
bar. Comparisons are valid *within* a hop_count across Variants, not *along*
the axis.

**Cannot:** anything about production KDA. This is core-mechanism KDA without
ShortConv (ADR 0009).

## Results

### baseline (full attention) -- COMPLETE

12,000 steps in 3.71h. Checkpoint verified: 110 model + 223 optimizer tensors,
all finite.

```
headline (all 25 cells, not to be quoted)   0.8019
VALID surface (hop>=2, distance>=3)         0.7676
traversal-free ceiling                      0.4803
excess                                      +0.2873

hop\dist       3       9      21      45    mean   ceiling   excess
    2      0.826   0.756   0.764   0.719   0.766     0.60    +0.166
    3      0.863   0.770   0.770   0.773   0.794     0.47    +0.324
    4      0.795   0.764   0.768   0.734   0.765     0.45    +0.315
    5      0.746   0.764   0.727   0.744   0.745     0.41    +0.335

cells above their own ceiling: 16/16
```

**Full attention genuinely traverses the Chain.** All 16 valid cells clear
their own traversal-free ceiling, by +0.2873 on average against a per-cell
standard error of ~0.022 -- roughly 13 SE, not a marginal call. This is the
first valid capability measurement this project has produced; every earlier
grid was a shortcut score.

**The excess grows with hop_count** (+0.166 at hop=2 to +0.335 at hop=5),
which is the discriminator registered in the methodology above. The
traversal-free ceiling *declines* with depth, so a model exploiting a surface
heuristic would show a *shrinking* excess. It shows the opposite.

Two calibration checks behave exactly as the design predicts, which is
independent evidence the grid is measuring what it claims:

- **distance=0 column: 1.000** -- degenerate by ADR 0012 (no Distractors fit,
  so appears-once solves it outright). The model has clearly mastered the
  surface rules; the 0.7676 is not a training failure.
- **hop_count=1 row: 0.753** -- above its 33.3% content ceiling (ADR 0017)
  because ordering still leaks there (ADR 0018), as recorded.

**Not saturated.** 0.7676 against a 1.0 ceiling leaves genuine headroom in
both directions, which is what makes the Variant comparison able to show a
difference at all -- the failure mode of the first experiment was two grids of
ones.

**On the learning curve.** It was called plateaued twice during the run and
resumed climbing both times (at ~0.66 around step 3,500, and again near 0.79
around step 10,500) as the cosine schedule decayed. Four consecutive flat eval
grids are not sufficient evidence of convergence on this task -- recorded
because the same judgement will be tempting for the remaining two Variants.

### hybrid (3:1) -- COMPLETE

12,000 steps in 4.86h, with one interruption at step 4,067: the run was killed
and resumed from `step_4000` to apply the memory cap (see Operational record).
Exact resume, so this is not a confound -- both RNG generators were restored
and the 16 earlier eval grids preserved.

```
                  headline    VALID surface (16 cells)
baseline            0.8019          0.7676
hybrid              0.8128          0.7804

hybrid - baseline:  +0.0128  (+1.28pp)  =  0.6 SE
cells clearing the traversal-free ceiling: 16/16 (both Variants)
```

**The 3:1 hybrid matches full attention.** +1.28pp against a ~2.2pp per-cell
standard error is 0.6 SE -- not a difference. At 125M parameters and matched
budget, replacing 9 of 12 attention layers with KDA costs nothing measurable in
multi-hop retrieval accuracy on this task.

Per-cell difference (positive = hybrid better):

```
hop\dist       3       9      21      45   hop mean
    2     -0.010  -0.033  -0.021  -0.008   -0.018
    3     -0.023  +0.021  +0.023  +0.008   +0.007
    4     +0.006  +0.031  +0.029  +0.057   +0.031
    5     +0.018  +0.023  +0.064  +0.020   +0.031
```

11/16 cells favour the hybrid and 8/16 exceed the 2.2pp noise floor, but they
point in **both** directions. The structure is the opposite of a capability
cost: the hybrid is marginally *worse* at hop=2 and marginally *better* at
hop=4-5. If the 3:1 ratio were eroding chaining, deep chains -- where a
compressed state has the most to lose -- are exactly where it should show, and
it does not.

**The real difference is acquisition, not capability.** The hybrid sat at
chance through step 2,500 (the baseline was already at 0.54), then climbed the
same curve shifted ~4,000 steps right and closed the gap entirely by step
11,500:

| step | 2500 | 5000 | 7500 | 10000 | 12000 |
|---|---:|---:|---:|---:|---:|
| baseline | 0.538 | 0.690 | 0.724 | 0.787 | 0.802 |
| hybrid | 0.0005 | 0.131 | 0.654 | 0.772 | 0.813 |

This reproduces the one finding that survived the invalidated experiment --
KDA-family Variants learn more slowly -- but on a valid task it now means
something different: **slower to learn genuine traversal, equally capable once
learned.**

A judgement worth recording: at step 2,500 the hybrid's loss floor had moved
only 0.22 nats across 2,000 steps and this was flagged as a possible failure to
learn. It was a long pre-takeoff plateau. That is the second time a plateau
call on this task was premature, and the reason the run was left to finish
rather than cut short.

### pure-KDA -- COMPLETE

12,000 steps in 8.36h, the slowest of the three (baseline 3.71h, hybrid 4.86h).
Uninterrupted.

```
headline (all 25 cells, not to be quoted)   0.7252
VALID surface (hop>=2, distance>=3)         0.6486
traversal-free ceiling                      0.4803
excess                                      +0.1683
cells above their own ceiling               16/16
```

**Pure-KDA also genuinely traverses** -- all 16 cells clear the traversal-free
ceiling, so this is not a Variant that failed to learn the task. It learned it
less well.

Slowest to acquire as well as lowest at convergence: still at chance through
step 3,250 (the baseline was at 0.66 by then), taking off around step 3,500 and
never closing the gap.

### Comparison

```
variant      steps   wall   headline   VALID(16)   vs ceiling   cells clear
baseline     12000  3.71h     0.8019      0.7676      +0.2873        16/16
hybrid 3:1   12000  4.86h     0.8128      0.7804      +0.3001        16/16
pure-KDA     12000  8.36h     0.7252      0.6486      +0.1683        16/16
```

Differences on the valid surface, against a per-cell standard error of ~0.022:

| comparison | difference | significance |
|---|---:|---:|
| hybrid − baseline | **+1.28pp** | 0.6 SE -- **no difference** |
| pure-KDA − baseline | **−11.90pp** | 5.4 SE -- **real degradation** |
| hybrid − pure-KDA | **+13.18pp** | 6.0 SE -- what the 3 attention layers buy |

**The 3:1 hybrid preserves full-attention multi-hop performance; pure linear
attention does not.** This is the question the project was commissioned to
answer, and the answer is clean in both directions: replacing 9 of 12 attention
layers with KDA costs nothing measurable, and replacing all 12 costs 11.9pp.

The three retained full-attention layers do not merely narrow the gap -- they
**recover all of it**. Removing them costs 11.90pp; the hybrid is +1.28pp above
the baseline, i.e. indistinguishable from it. Whatever the full-attention
layers contribute, 3 of 12 at the ADR 0011 schedule (indices 3, 7, 11) is
sufficient.

#### The degradation is flat in hop_count, not accumulating

```
  hop   baseline   hybrid   pure-KDA   kda − baseline
    2      0.766    0.748      0.655           −0.111
    3      0.794    0.801      0.665           −0.129
    4      0.765    0.796      0.642           −0.123
    5      0.745    0.776      0.633           −0.112
```

**This contradicts the project's motivating hypothesis.** CONTEXT.md frames the
study as asking whether state compression produces a *hop-distance* degradation
pattern -- the expectation being that a compressed state loses more with each
additional hop, so the deficit should widen with depth. It does not. Pure-KDA's
deficit is 11-13pp at every hop count, with no trend.

What that looks like is a **fixed capability tax** rather than accumulating
error: the delta-rule state costs something constant, and that something is not
"ability to chain one more hop".

The honest caveat is that this grid cannot measure degradation *along* the hop
axis at all (ADR 0017) -- the traversal-free ceiling itself varies with
hop_count, so the axis mixes capability with a moving bar. The comparison above
is *within* each hop_count across Variants, which is valid. A claim about how
any single Variant degrades with depth is not available from this data.

#### Acquisition order matches capability order

| step | 2500 | 5000 | 7500 | 10000 | 12000 |
|---|---:|---:|---:|---:|---:|
| baseline | 0.538 | 0.690 | 0.724 | 0.787 | 0.802 |
| hybrid | 0.0005 | 0.131 | 0.654 | 0.772 | 0.813 |
| pure-KDA | 0.0000 | 0.297 | 0.541 | 0.685 | 0.725 |

The hybrid was *slower than pure-KDA* through the middle of training and still
finished 13pp ahead, so acquisition speed and final capability are not the same
axis. Wall-clock cost orders the same way as neither: 3.71h / 4.86h / 8.36h.

#### Verification

All three verified on three independent axes before these numbers were
reported:

- **Runs completed**: each reached step 12,000 with **0** non-finite losses.
- **Checkpoint tensors read directly** (the ADR 0006 bar -- a finite printed
  loss is not evidence): baseline 110 model + 223 optimizer, hybrid 191 + 385,
  pure-KDA 218 + 439, **all finite**. The counts also confirm the intended
  architectures ran, ordering full-attention < hybrid < pure-KDA as the
  parameter counts require.
- **Independently re-derived** by a separate copy of the tooling running on the
  GB10, matching the local computation to four decimals.

#### What this does and does not establish

**Establishes**, on a task where the answer is provably not recoverable by
position, ordering, or occurrence counting:

- The 3:1 hybrid matches full attention at matched hop_count and distance.
- Pure-KDA is measurably worse, by 11.9pp, at every hop_count.
- All three genuinely traverse Chains (16/16 cells above the traversal-free
  ceiling each).

**Does not establish:**

- **How performance degrades with chain length.** ADR 0017: the axis is
  confounded by a ceiling that moves with hop_count.
- **That the hybrid equals the baseline.** +1.28pp at 0.6 SE is a null result
  from one seed per Variant, not a demonstration of equality.
- **Anything about production KDA.** ADR 0009: this is core-mechanism KDA
  without ShortConv, not Kimi Linear as deployed.
- **Ceiling behaviour.** None of the three reaches 1.0; all sit at partial
  competence, so this compares them part-way up a curve, not at saturation.
- **Whether the ADR 0011 schedule matters.** Full attention sits at layers
  3/7/11. The opposite phase (0/4/8) is untested and remains the ablation
  ADR 0011 names.

## Operational record

**The memory cap was bypassed, and I caused it.** The cap
(`XLA_PYTHON_CLIENT_MEM_FRACTION=0.65`) lived in
`scripts/run_all_variants.sh`. A second, simpler launcher was written for these
corrected runs and the cap did not come with it, so the hybrid ran uncapped to
**116GB of the shared box's 121GB** -- inside the range ADR 0009 and ADR 0010
both had to kill runs in -- with pure-KDA, the heaviest Variant, queued behind
it. Capping dropped usage to 89GB.

Cost: **67 steps.** The run was killed and resumed from `step_4000` with both
RNG generators restored and its 16 earlier eval grids preserved, so it remains
exactly comparable to the other two. The exact-resume work (ADR-less, in
`train.py`) paid for itself here.

The cap now lives in `train_variant.py` via `os.environ.setdefault`, so any
invocation gets it and an explicit caller value still wins. The general lesson:
**a safeguard that lives in one launcher is not a safeguard for a task that can
be started another way.**

With the cap, pure-KDA ran at 102GB with 19GB headroom (arena 81,916 MiB rather
than the ~92GB it preallocates uncapped) -- the case the cap existed for.

**Plateau calls were premature twice.** The baseline was read as converged at
~0.66 (step 3,500) and again near 0.79 (step 10,500); it finished at 0.80. The
hybrid's loss floor moved only 0.22 nats across 2,000 steps around step 2,500
and was flagged as a possible failure to learn; it was a long pre-takeoff
plateau and it went on to match the baseline. **Four consecutive flat eval
grids are not evidence of convergence on this task.** Both runs were left to
finish because of this, which was the right call both times.

## Files

- `../experiment_20260728/{baseline,hybrid,kda}_fixed_grids.json` -- every eval
  grid, 48 per run at cadence 250.
- `..../{baseline,hybrid,kda}_fixed_run.log` -- per-step loss, seconds, elapsed.
- `..../baseline_gapsonly_ABANDONED*` -- the 4,745-step attempt with only gap
  randomisation, retained as the record of why `randomize_fact_order` exists.
- `../shortcut_ceiling_corrected.json` -- per-cell traversal-free ceilings.
- Final model-only checkpoints archived outside the repo in
  `checkpoints_final/` (gitignored).

## The follow-up this makes possible

The obvious next experiment is the one ADR 0011 already names: the hybrid's
full-attention layers sit at indices 3/7/11 by a phase choice, and the opposite
phase (0/4/8) is untested. Given that 3 of 12 attention layers recover the
entire 11.9pp gap, *where* they sit is now a substantive question rather than a
tidiness one.

Second: pure-KDA's deficit is flat in hop_count, which contradicts the
compression-accumulates-error intuition the project was built on. A
distractor-count sweep at fixed hop_count would test whether the tax is about
interference rather than depth.
