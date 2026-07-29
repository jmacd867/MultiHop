# hop_count=1 with Distractors is unanswerable by construction; the hop axis cannot be read as difficulty

Surfaced from the corrected baseline's grid at step 4,000, which showed
accuracy **increasing** with hop_count -- the opposite of what a multi-hop task
should do:

```
hop\dist        0        3        9       21       45
      1     0.658    0.502    0.223    0.166    0.168
      5     0.734    0.479    0.484    0.457    0.436
```

That is not the models finding long chains easier. hop_count=1 has a lower
ceiling than every other row, and the ceiling is structural.

## The cause

`generate_example` picks each Distractor's subject with
`decoy_subject_hop = rng.integers(0, config.hop_count)`. At `hop_count=1` that
range is `[0, 1)`, so **every Distractor's subject is `chain_entities[0]` --
which is the Query entity itself.**

The sequence therefore contains three Facts of the form `query -> X`: one real
(`query -> chain_entities[1]`, the answer) and two decoys. And because there is
no second hop, the true target **never reappears as a subject**, which is the
only feature that distinguishes a Chain step from a decoy anywhere else in this
task.

Measured over 400 examples per hop_count at distance=9 on the corrected
generator:

| hop_count | candidates reachable from the Query | survive "is it also a subject?" | ceiling |
|---|---:|---:|---:|
| 1 | 3.00 | 3.00 | **33.3%** |
| 2 | 2.02 | 1.00 | 100% |
| 5 | 1.45 | 1.00 | 100% |

At hop_count >= 2 the answer is uniquely identifiable and the ceiling is 100%.
At hop_count = 1 it is a three-way guess for **any** model, however capable.
This is not a shortcut a model exploits (ADR 0012, ADR 0015) -- it is missing
information. No architecture can exceed 33.3% there.

## Consequences

**The hop=1 row must be excluded from every reading.** Four cells
((1,9), (1,21), (1,45), and (1,3) below) are capped at 33.3% while every other
row is capped at 100%, so including them in a mean or a per-hop marginal
compares quantities with different ceilings.

**(1,3) is separately contaminated.** ADR 0015 records it as the only fully
packed cell in the grid, so `randomize_gaps` has no spare Filler to
redistribute there and the fixed-offset positional shortcut *survives* it.
That is visible in the data: (1,3) reads 0.502 at step 4,000, above the 33.3%
ceiling that binds the rest of its row, because position is still usable.
It is the one cell where both a floor problem and a ceiling problem coincide.

**Degradation with hop_count is not measurable on this grid.** The axis is
confounded by a ceiling that moves in the opposite direction to the difficulty
it is supposed to represent. Any apparent "improvement with depth" is this
artifact. The honest statement is that this experiment cannot answer how
performance varies with chain length; it can only compare architectures at
matched hop_count.

**The valid comparison surface is 16 cells: hop_count >= 2 and distance >= 3.**
distance=0 remains degenerate per ADR 0012 (no Distractors fit, so the
appears-once shortcut scores 1.0), and hop_count=1 is capped as above. On those
16 cells the ceiling is a uniform 100%, the appears-once floor is a uniform
33.3%, and the positional shortcut is dead -- so accuracy above 33.3% there is
genuine evidence of Chain traversal. That surface is what the cross-Variant
comparison is computed on, and headline means over all 25 cells are not to be
quoted.

At step 4,000 the corrected baseline reads **0.4103** on that surface with
13 of 16 cells above the floor -- the first evidence in this project of a model
doing something that requires traversal.

## Why the fix is not applied

Changing `decoy_subject_hop` to exclude the Query entity at hop_count=1 would
fix the ambiguity, but it changes the training distribution, and the corrected
baseline run was already several hours in when this was found. Restarting to
recover four cells that are excluded from the analysis anyway would cost more
than it returns, against a hard deadline on borrowed hardware.

Recorded as a named limitation instead. Any future run of this task should draw
Distractor subjects from `chain_entities[0:hop_count]` **excluding whichever
entity the Query names**, so that a Distractor never produces a second
same-subject Fact -- which would make hop_count=1 answerable and the hop axis
readable.

## Status

Third structural flaw found in this generator, after ADR 0012's appears-once
shortcut and ADR 0015's positional invariant. All three share the same shape:
a property of the *data construction* that determines what any model can score,
independent of architecture, and invisible in an accuracy number read on its
own. The standing lesson holds a fourth time -- check what a number could be
produced by, and check what it could not exceed.
