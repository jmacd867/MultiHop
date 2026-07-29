# The generator places the answer at a fixed offset from the Query, making the task 100% solvable without traversal

Written before the mechanism probes returned, deliberately. This is a property
of `generator.py`, provable from the source and measurable without any model,
so it is true regardless of what the trained Variants turn out to do with it.
Recording it first is what stops the finding being shaded by the results.

## The invariant

```
answer_token_index == query_position - (distance + 3)
```

Always. At every hop_count, every distance, and every distractor_count.

**Why it holds.** `emit_gap` computes
`filler_count = config.distance - 3 * len(distractor_indices)` and then emits
that many Filler tokens plus a 3-token block per Distractor. **Every gap is
therefore exactly `distance` tokens long no matter what is packed into it** --
Distractors displace Filler one-for-one and never change a gap's length. Facts
are always exactly 3 tokens (`subject, object, SEP`). Every position in the
sequence is consequently a deterministic function of (hop_count, distance)
alone, and the arithmetic collapses to the offset above.

**Measured**, not merely derived:

| condition | examples | violations |
|---|---:|---:|
| reference count (k=2, k=0 at distance=0) | 5,000 | **0** |
| each cell's *maximum* distractor capacity (up to k=90 at (5,45)) | 2,500 | **0** |

Observed offsets exactly match `distance + 3`: d=0 -> 3, d=3 -> 6, d=9 -> 12,
d=21 -> 24, d=45 -> 48.

## Consequence: a 100% shortcut that no eval-time parameter can remove

Copying the token at `query_position - (distance + 3)` scores **1.0000 in
every cell of the Degradation Grid**, with no traversal, no binding, and no
attention to the Query at all. It is strictly stronger than ADR 0012's
appears-exactly-once shortcut, which tops out at 33.3% where Distractors exist.

Critically, `filler_count` is defined so that **distractor count is orthogonal
to the invariant** -- it holds at k=90 exactly as at k=2. This kills the
distractor-stress eval that ADR 0012 and ADR 0014 both proposed as the
follow-up: its outcome is predictable from the source as 1.0000 in every cell,
and running it would establish nothing. ADR 0012's own suggestion of raising
`distractor_count` to lower the shortcut floor is void: it lowers the
*appears-once* floor while leaving the positional floor at 1.0.

A related premise in that plan was also wrong. `max_distractor_capacity(1,3)`
is 2, which is `_REFERENCE_DISTRACTOR_COUNT` -- so a *uniform* distractor count
across the grid is capped at exactly the value already in use, making the
"uniform" stress eval a bit-for-bit rerun of the grid already collected.

## What this does to the results already recorded

**ADR 0012 is superseded on the floor, not on its reasoning.** Its analysis of
the appears-once shortcut is correct and its per-cell arithmetic stands. But
its claim that 33.3%/100% "is the floor every result must be read against" is
now false: the reachable floor without reasoning is **1.0 in every cell**.
CONTEXT.md's `Shortcut Floor` entry is corrected accordingly, since that is the
sentence a future reader would most reasonably trust.

**ADR 0014's surviving result is placed in doubt, not refuted.** Its cleanest
positive finding was that KDA's acquisition penalty scales with distance
(+29%, baseline flat) and that ADR 0013's density confound cannot explain it,
because the trend runs opposite to density. That argument remains sound as far
as it goes. But "learn a fixed lookback of length `distance + 3`" also predicts
a penalty growing with distance and flat in hop_count -- and predicts it more
directly for a NoPE recurrent state (ADR 0001) than for RoPE'd full attention.
The finding survives the confound that was checked and has an untested
alternative explanation in the one that was not.

The distinction matters and must not be blurred: **the existence of a shortcut
does not establish that the models use it.** That is a separate empirical
claim, tested by `scripts/probe_mechanism.py`, and its results are recorded
separately.

**ADR 0013's (1,3) anomaly is predicted by this account, not contradicted by
it.** KDA's converged loss at (1,3) is +0.201 nats against the baseline, four
times any other cell, which a uniform positional rule appears not to explain.
But the *cost of learning to execute* the rule is not uniform. `(1,3)` is the
**only cell in the entire grid with zero Filler tokens** at the reference count
(verified across all 25 cells), and at hop_count=1 `rng.integers(0, hop_count)`
is always 0, so every Distractor shares subject `c_0`. The example is three
structurally identical `subject, object, SEP` blocks back to back with no
Filler landmarks anywhere:

```
688 1897 SEP | 688 6494 SEP | 688 1453 SEP | ? 688      answer 6494, index 4 = qpos-6
```

Everywhere else Filler runs give a trivial positional landmark (12 of 23 tokens
at (1,9), 264 of 287 at (5,45)). Full attention carries explicit relative
position via RoPE; pure-KDA is NoPE by ADR 0001 and carries position only in
gating dynamics, which is exactly where a landmark-free offset across identical
blocks is hardest to learn. A KDA-specific anomaly at exactly one cell is what
this account predicts, and (1,3) is that cell.

## Why this was not caught earlier

The same reason as ADR 0012, one level deeper. ADR 0004 reasoned carefully
about Distractor *density* against capacity, and ADR 0013 measured the
consequence. Both treated the gap as a place where interference lives. Neither
asked what `filler_count`'s definition implies about *position*, which is that
the gap's length is invariant to its contents by construction -- the very
property that makes Distractors and Distance independently sweepable is also
what makes every Fact's index deterministic.

The sanity runs could not have surfaced it (all Variants scored ~0.0000 at 20
steps), and the full runs' 1.0000 was read as saturation rather than as a
signature. It became visible only when a fresh reader went to the generator
source with the question "what could produce a perfect score" rather than
"why is the score perfect".

This is the **third** instance of this project's own standing lesson, after
ADR 0006's silent NaN and ADR 0008's tied-embedding memorization: check what a
number *could* be produced by before reporting what it appears to show.

## Consequence for any future run of this task

A generator fix must destroy **both** offsets, not one. Randomising only the
final gap breaks the end-relative offset `query_position - (distance + 3)` but
leaves every Fact's *absolute* index untouched -- measured: the answer sits at
a single fixed index (142 at (3,45), 238 at (5,45)) while only
`query_position - answer_index` varies. A start-relative rule survives that
completely.

The safe specification is to **randomly partition the total gap budget
`(hop_count + 1) * distance` across all gaps**, subject to each gap still
holding its assigned Distractors. That preserves total sequence length exactly
-- so cell structure and the one-shape-per-step invariant
`sample_training_microbatches` depends on both survive -- while destroying the
start-relative and end-relative offsets together. `scripts/probe_mechanism.py`
implements it as the `jitter_all` condition, verified to preserve length in
every cell and to yield many distinct answer indices.

One cell resists it: `(1,3)` is fully packed, so there is no slack to
redistribute and `jitter_all` there is identical to `control`. That cell cannot
be de-confounded by gap randomisation at the reference Distractor count.
