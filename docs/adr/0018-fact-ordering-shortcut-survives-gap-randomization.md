# A Fact-ordering shortcut survives gap randomisation; both must be destroyed together

ADR 0015 established that uniform gaps put the answer at
`query_position - (distance + 3)`, and `randomize_gaps` was added to destroy
that. It does. It is not sufficient.

Chain Facts are emitted **between** gaps; Distractors are emitted **inside**
them. Randomising gap *lengths* therefore moves every Fact's token offset while
leaving the answer-bearing Fact's **rank** among the Fact-shaped blocks
concentrated. "Always answer the object of the k-th block" is a complete
strategy requiring no traversal, no binding, and no reading of the Query:

| cell | blocks | best fixed-rank guess | chance |
|---|---:|---:|---:|
| (1,3) | 3 | **1.000** | 0.333 |
| (2,9) | 4 | 0.476 | 0.250 |
| (3,9) | 5 | 0.540 | 0.200 |
| (5,45) | 7 | **0.683** | 0.143 |

At (1,3) it is exact: per-gap capacity is `3//3 = 1`, so with two Distractors
each gap holds exactly one and the real Fact is *always* the middle block.

## How it was found, and what it cost

Not by reading the code. The corrected baseline's grid showed hop_count=1 cells
climbing past the 33.3% ceiling ADR 0017 had claimed was structural -- (1,9)
reached 0.412 and was still rising. That contradiction was the thread. Pulling
it showed the rank distribution at hop_count=1 is
`0.215 / 0.581 / 0.204` across the three same-subject Facts, matching the
`0.2 / 0.6 / 0.2` that the ticket-based Distractor placement predicts, and the
same skew exists at every hop_count.

**ADR 0017 is corrected by this.** Its claim that "no architecture can exceed
33.3%" at hop_count=1 was an overreach: what was demonstrated there is a
*content-only* ceiling -- the answer is not identifiable from subject-matching
plus chain-continuation. It is identifiable from *ordering*, which that
analysis did not consider. The cells are still excluded from the valid surface,
but for a different reason than recorded: not "unanswerable" but "answerable by
a shortcut rather than by traversal".

The cost was one abandoned run. The gaps-only baseline reached 4,745 of 12,000
steps and read **0.5801** on the supposed valid surface -- squarely inside the
0.45-0.69 band the rank heuristic reaches, hence indistinguishable from
rank-following. It is archived as `*_gapsonly_ABANDONED` and its checkpoints
deleted. Had it not been caught, the second experiment would have failed the
same way as the first, with a plausible-looking number.

## The fix

`randomize_fact_order` emits the Chain Facts in a random order rather than in
Chain order. The answer is `chain_entities[-1]` regardless of emission order --
only the Chain's structure defines it, not its layout -- so the task is
unchanged in what it asks, only in what it leaks.

Measured effect on the best fixed-rank guess:

| cell | Chain order | random order | chance |
|---|---:|---:|---:|
| (2,9) | 0.497 | 0.299 | 0.250 |
| (3,9) | 0.554 | 0.225 | 0.200 |
| (5,45) | 0.676 | 0.155 | 0.143 |

End-to-end through the training path at cell (5,21):

| flags | distinct answer offsets | best rank guess |
|---|---:|---:|
| neither (original) | 1 | 0.69 |
| `randomize_gaps` only | 40 | 0.64 |
| **both** | 54 | **0.19** |

The middle row is the whole point of this ADR: the first fix moved the offset
40-fold and barely touched the leak.

## Kept as two flags, not one

`randomize_gaps` and `randomize_fact_order` are separate fields even though
`train_variant.py --fixed` always sets both, so each shortcut can be tested in
isolation. Two regression tests guard the pair, in opposite directions:

- with both flags on, the best fixed-rank guess must be near chance
- **with the flag off, the leak must remain reproducible** (asserted > 0.4)

The second matters as much as the first. A fix whose absence cannot be
demonstrated is not falsifiable, and this generator has now produced four
shortcuts that all looked fine until someone asked what a number could be
produced by.

## Standing lesson, fourth instance

ADR 0006 (silent NaN), ADR 0008 (tied-embedding memorisation), ADR 0012
(appears-once), ADR 0015 (fixed offset), and now this. Every one shares a
shape: a property of *data construction* that bounds or inflates what any model
can score, invisible in the accuracy number, and passing every type check and
unit test.

The specific lesson this instance adds: **fixing one shortcut does not mean the
task now requires the intended capability.** The right check after a fix is not
"is the old shortcut gone" but "what is the best achievable score without the
capability I am trying to measure" -- computed fresh, against the fixed
generator, before spending GPU time on it.
