# The distance axis is confounded with distractor density; d=3 is the hardest column, not the easiest

The full-attention baseline's Degradation Grid is **non-monotonic in
distance**. At step 2500 the per-distance means were:

| distance | 0 | 3 | 9 | 21 | 45 |
|---|---:|---:|---:|---:|---:|
| mean accuracy | 0.825 | **0.696** | 0.983 | 0.988 | 0.938 |

distance=3 is the *worst* column, worse than distance=45 whose sequences are
eight times longer. The `steps_to_criterion` view agrees independently: d=3
cells need 3000-3500 steps where every other distance needs 2500. Two
different views of the same data pointing the same way makes this a property
of the task, not sampling noise.

`(1,3)` is the single worst cell in the whole grid -- the only one still at or
below the ADR 0012 shortcut floor once the rest had passed it.

## Cause: distractor *density*, not distance

The reference distractor count is a fixed absolute 2 at every distance>=3
(ADR 0004), but the space available to put them in grows with distance. Placed
against `max_distractor_capacity`:

| cell | capacity | placed | packed |
|---|---:|---:|---:|
| (1,3) | 2 | 2 | **100%** |
| (2,3) | 3 | 2 | 67% |
| (3,3) | 4 | 2 | 50% |
| (5,3) | 6 | 2 | 33% |
| (1,9) | 6 | 2 | 33% |
| (5,45) | 90 | 2 | **2%** |

At distance=3 the two Distractors consume most or all of the gap space, so
they sit immediately adjacent to the real Chain Facts with little or no Filler
between them. At distance=45 the same two Distractors are diluted across 90
slots and are almost never adjacent to anything that matters. `(1,3)` is the
only **fully packed** cell in the grid, which is exactly why it is the worst.

So the distance axis does not isolate "how far apart are the Chain Facts". It
varies gap length and interference density together, in opposite directions.
Low distance means tightly packed, high-interference sequences; high distance
means sparse, low-interference ones. The two effects work against each other,
and at distance=3 the density effect wins.

## This is ADR 0004's predicted trade-off, now measured

ADR 0004 chose a fixed absolute count over a capacity-scaled one, and recorded
the consequence explicitly: *"the ratio of distractor density to available
capacity shrinks sharply at large distance (2 of 90 possible slots at
hop_count=5, distance=45). The main grid therefore measures degradation under
constant absolute interference, not constant relative interference."*

That reasoning was right, and the decision is not being reversed -- a
capacity-scaled count would have made distractor_count a function of distance,
reintroducing the coupling Filler exists to remove. What is new is the
magnitude: the density effect is large enough to **invert** the expected
ordering of the distance axis, not merely to perturb it. ADR 0004 framed it as
a known limitation of what the grid measures; in practice it dominates the
low-distance end.

## How to read the distance axis

**Do not read the distance column as a pure distance effect.** A Variant doing
worse at distance=3 than at distance=45 is not evidence that it handles short
gaps badly. The honest description is that distance=3 is the
highest-interference condition in the grid and distance=45 the lowest.

**The comparison across Variants is unaffected**, for the same reason as
ADR 0012: all three Variants face identical cells with identical packing, so a
difference between them at any given distance is still an architecture effect.
The confound is in what the *axis* means, not in whether Variants can be
compared along it.

**hop_count remains the cleaner axis.** Capacity grows with hop_count too, so
packing varies along it as well ((1,3) is 100% packed while (5,3) is 33%), but
far less sharply than along distance.

The clean way to separate the two is the separate 1D distractor-count ablation
ADR 0004 already scopes -- swept at a single fixed distance, so density varies
while gap length does not. That remains the right follow-up and is out of
scope here.
