# The corrected experiment: three 20k-equivalent runs on a task that requires traversal

**Status: in progress.** Methodology below is final and was written before the
results existed, deliberately -- every section that fixes how a number must be
read is settled in advance so it cannot be shaped by what the numbers turn out
to be. Results are filled in at the bottom as runs complete.

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

*(filled in as runs complete)*

### baseline (full attention) -- COMPLETE

12,000 steps in 3.71h. Final checkpoint written; tensor verification runs in
the finalize pass once all three Variants are done.

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

### hybrid (3:1)

### pure-KDA

### Comparison
