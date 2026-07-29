# Measured: both Variants execute positional rules, never read the Query, and use *different* rules

ADR 0015 established that the generator admits a fixed-offset positional
solution. That a shortcut exists does not establish that a trained model uses
it. `scripts/probe_mechanism.py` tested the mechanism directly against the
archived `step_20000` checkpoints of both completed Variants. It does use it,
and the detail matters more than the headline.

Five conditions, 512 examples per cell, all 25 cells, no retraining. Each cell
reports the model's accuracy beside the accuracy the fixed end-relative rule
would achieve on the identical examples.

## Result 1: neither Variant reads the Query at all

`query_ablate` replaces the Query entity with an entity that appears nowhere
in the sequence. Every position is preserved; exactly one token changes. A
model doing content-based retrieval has nothing to retrieve and should
collapse.

| condition | hop=1 | hop=2 | hop=3 | hop=4 | hop=5 |
|---|---|---|---|---|---|
| baseline | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| pure-KDA | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

Both Variants emit the original answer perfectly when asked about something
that is not in the text. **The Query is not an input to either model's
computation.** This is the cleanest result in the set: it is one token from
the training distribution, it is model-agnostic (unlike ADR 0005 attention
capture, which cannot speak for pure-KDA at all), and it alone falsifies any
claim that these models perform retrieval.

## Result 2: the shuffle profile matches the positional rule to three decimals

`shuffle` emits the same Facts in random order, leaving gaps, Distractors and
the Query untouched. The end-relative rule then lands on the answer only when
the answer's Fact happens to be emitted last -- `1/hop_count`.

| | hop=1 | hop=2 | hop=3 | hop=4 | hop=5 |
|---|---|---|---|---|---|
| baseline model | 1.000 | 0.525 | 0.326 | 0.247 | 0.188 |
| pure-KDA model | 1.000 | 0.515 | 0.329 | 0.245 | 0.190 |
| **the rule** | 1.000 | 0.515 | 0.329 | 0.245 | 0.189 |

Not merely the same shape -- the same numbers, and several individual cells
agree exactly (`0.492/0.492`, `0.164/0.164`). Read as a profile, per the
pre-registered procedure, this is the positional rung on every one of the five
rungs for both Variants.

The profile, not a scalar, is what identifies it. At hop_count=3 the
positional rule gives 0.333, numerically identical to ADR 0012's appears-once
floor at the reference Distractor count; only the decline from 1.000 to 0.190
distinguishes them.

## Result 3: the two Variants use *different* positional rules

This is the operationally important one, and it was nearly missed.

`jitter_end` randomises only the final gap. It breaks the **end**-relative
offset `query_position - (distance + 3)` while leaving every Fact at a fixed
**absolute** index (measured: the answer sits at index 142 at (3,45) and 238 at
(5,45), invariant across examples). `jitter_all` randomly partitions the whole
gap budget, destroying both.

| condition | Variant | hop=1 | hop=2 | hop=3 | hop=4 | hop=5 |
|---|---|---|---|---|---|---|
| jitter_end | baseline | 0.583 | 0.601 | 0.549 | 0.554 | 0.551 |
| jitter_end | **pure-KDA** | **0.999** | **1.000** | **0.999** | **0.998** | **0.910** |
| jitter_all | baseline | 0.352 | 0.178 | 0.094 | 0.087 | 0.062 |
| jitter_all | pure-KDA | 0.415 | 0.428 | 0.313 | 0.400 | 0.368 |

**Pure-KDA is essentially unharmed by `jitter_end`** and collapses only under
`jitter_all`. It is executing a **start-relative** rule -- counting forward
from the sequence start -- not counting back from the Query. The baseline sits
between the two conditions (0.55 under `jitter_end`, 0.06-0.35 under
`jitter_all`), consistent with a mixture rather than one clean rule.

That divergence is plausible from ADR 0001: the baseline has RoPE and explicit
relative position; pure-KDA is NoPE and carries position only through the
recurrence's step count, which is inherently start-anchored.

## Consequence: the generator fix had to randomise *all* gaps

This is what the probe bought. The obvious repair -- randomise the final gap so
the answer is no longer a fixed distance before the Query -- **would have left
pure-KDA solving the "corrected" task at ~1.0**, and it would have looked like
a clean re-run. The re-run would have measured the same shortcut under a new
name, and the second experiment would have failed the same way as the first.

`GeneratorConfig.randomize_gaps` therefore partitions the entire gap budget
`(hop_count + 1) * distance` across all gaps, preserving total sequence length
(so `required_sequence_length` and the one-shape-per-step invariant survive)
while destroying both offsets. `tests/data/test_generator.py` asserts both the
absolute and the relative offset vary, precisely so a future "simplification"
back to final-gap-only cannot pass.

## What this does to the recorded results

**ADR 0014's acquisition finding is now explained, not merely doubted.** Its
positive result was that KDA's steps-to-criterion scales with distance (+29%)
while the baseline is flat, and that ADR 0013's density confound runs the wrong
way to explain it. Both Variants are now measured executing positional rules,
so what that curve tracks is the cost of learning a *longer positional offset*,
not multi-hop retrieval. ADR 0014 flagged this as an untested alternative; it
is now the tested one.

**The Degradation Grids of `docs/runs/experiment_20260728/` measure a
positional copy.** They are retained as the record of what was run and are not
to be read as evidence about multi-hop capability, for either architecture.

## Scope

These conditions are outside the training distribution, and ADR 0003 forbids
reading OOD performance as a *capacity* measurement. This is not one: it is
falsification of a mechanism claim, and the two carry different evidentiary
weight. A model that collapses under `shuffle` has not been shown incapable of
traversal -- only shown not to have learned it here.

`query_ablate` is the exception and is why it carries the most weight: it
changes one token, preserves every position, and a retrieval-based model would
fail it regardless of distribution. Both Variants pass it perfectly, which is
not consistent with any account in which the Query is read.
