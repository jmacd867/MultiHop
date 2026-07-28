# The task has a traversal-free "appears exactly once" shortcut; it sets the grid's floor, not chance

Discovered mid-run, from the full-attention baseline's step-2000 Degradation
Grid. Two things in it did not fit the task's intended structure: accuracy was
**flat across hop_count** (per-hop means 0.281, 0.309, 0.320, 0.312, 0.307 for
hop_count 1-5), and it was **non-monotonic in distance** (d=3 worse than d=45).
Multi-hop retrieval should get harder as hop_count rises. A pattern that
ignores hop_count entirely is the signature of a strategy that never traverses
the Chain.

## The shortcut

In every example, the Chain's terminal entity is **the only entity that is
never used as a subject**. Chain A->B->C->D emits Facts (A,B), (B,C), (C,D) and
a Query naming A; counting occurrences gives A=2 (Fact 1 plus the Query), B=2,
C=2, and **D=1**. The answer is therefore always recoverable by "output the
entity that appears exactly once", with no traversal, no ordering, and no use
of hop_count.

Distractors are the only thing standing in the way, and per ADR 0004 there are
exactly two of them (a decoy Fact shares one endpoint with a real Chain Fact,
so its far endpoint also appears exactly once).

Measured directly over 256 freshly generated examples in each of the 25 Grid
Cells:

| | entities appearing exactly once | shortcut accuracy | answer in that set |
|---|---:|---:|---:|
| distance = 0 | **1.00** | **100%** | 100% of examples |
| distance >= 3 | **3.00** | **33.3%** | 100% of examples |

Both figures are **identical at every hop_count from 1 to 5**. The `1.00` at
distance=0 is not an accident of sampling: `max_distractor_capacity` is 0
there, so ADR 0004 forces `distractor_count=0`, leaving the answer as the sole
once-appearing entity.

## Consequences for reading a Degradation Grid

**The floor is not chance.** Every prior discussion in this project (the sanity
run READMEs, ADR 0008's diagnosis) benchmarked accuracy against ~0.000125,
chance over the 8,003-token vocabulary. That is the right floor for *predicting
an arbitrary token* but the wrong one for *this task*: the reachable floor
without any reasoning is **33.3% at distance>=3 and 100% at distance=0**.

**The five distance=0 cells are degenerate.** They are saturable with no
traversal whatever, so they cannot discriminate between the three Variants --
all three should approach 100% there. A Variant *failing* at distance=0 is
still informative (it indicates something badly broken), but a Variant
succeeding there says nothing about multi-hop capability. ADR 0004 introduced
distance=0 as a deliberate "distractor-free floor cell"; this ADR records that
the floor it establishes is higher and less informative than intended.

**Above 33.3% at distance>=3 is genuine evidence of traversal.** Choosing the
true terminal over the two distractor terminals requires knowing which leaf the
Chain from the Query entity actually reaches, and that requires following the
Chain. So the grid retains real discriminating power over the range
33.3% -> 100%; it simply has a much higher floor and a narrower dynamic range
than a naive reading assumes.

**Flat-across-hop_count near 33% means "learned the shortcut only".** This is
the specific misreading to guard against: a final grid sitting at ~33% for
every hop_count would look like "multi-hop works equally well at all depths"
and would actually mean the opposite -- that no traversal is happening at all.

## Decision

**Accept the shortcut, document it, and report every cell against its shortcut
baseline rather than against chance.** `scripts/compare_grids.py` renders the
shortcut floor per cell and marks the distance=0 column degenerate, so the
headline quantity is accuracy *above the floor*.

**The cross-variant comparison is unaffected and remains the point of the
experiment.** All three Variants are trained and evaluated on identical task
structure (identical cells, identical examples per cell under the shared
`EVAL_SEED` of ADR 0008), so the shortcut is available equally to all of them.
Differences between their grids remain attributable to the sequence-mixing
mechanism, which is what CONTEXT.md defines a Variant by.

**Rejected: re-running with a higher `distractor_count`.** Raising it from 2 to
~8-10 would push the distance>=3 floor down to ~10%, which is genuinely better.
It was rejected on cost and incompleteness: it discards ~28h of committed runs
against ~45h of remaining time on borrowed hardware, leaving no margin, and it
**cannot fix the distance=0 column at all** -- capacity there is 0, so that
column stays degenerate regardless. Recorded as the obvious follow-up
experiment rather than a fix applied under deadline.

## Why this was not caught earlier

ADR 0004 chose `distractor_count=2` and explicitly reasoned about distractor
*density* relative to capacity, recording as a known trade-off that density
shrinks at large distance. It did not consider the *absolute* count as setting
a guessing baseline over once-appearing entities, which is the quantity that
actually matters here: 2 distractors means 3 candidates means a 33.3% floor,
independent of distance. The density framing and the floor framing point at the
same parameter for different reasons.

The sanity runs could not have surfaced this: at 20 optimizer steps every
Variant scored ~0.0000, far below the shortcut floor, so there was nothing to
notice. It became visible only once a real run trained long enough for the
model to find the shortcut -- step 2000 of 20,000, roughly 40 minutes in.

This is the third time in this project that a plausible-looking result turned
out to be produced by a mechanism other than the intended one (ADR 0006's
silent NaN, ADR 0008's tied-embedding memorization, and now this). The standing
lesson holds: check what a number *could* be produced by before reporting what
it appears to show.
