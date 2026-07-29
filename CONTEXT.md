# Multi-Hop Linear-Attention Degradation Experiment

A controlled study of whether linear-attention state compression produces a different (or worse) hop-distance failure pattern than full attention, using a synthetic chained key-value retrieval task.

## Language

**Chain**:
A sequence of N entity-pair Facts forming a path (e.g. A→B, B→C, C→D) that a Query must be resolved by traversing start-to-end. N is the chain's Hop Count.

**Hop Count**:
The number of edges in the Chain being queried, equal to the number of hops required to resolve the Query from start to end. Ranges 1–5. Always equals the full length of the queried chain (no partial-depth queries on longer chains).

**Fact**:
A single (subject, object) pair in the sequence under one implicit link relation. No relation typing — every Fact means the same kind of link.
_Avoid_: Edge, triple (triple implies a relation label, which this task doesn't have).

**Query**:
The single request at the end of a sequence naming the Chain's start entity; the answer is the Chain's terminal entity. Exactly one Query per sequence.

**Distractor**:
A decoy Fact that shares one endpoint with a real Chain Fact but does not continue the queried Chain (e.g. real Chain has B→C, Distractor has B→Z). Distinct from Filler — a Distractor is a well-formed Fact, just not part of the answer path.
_Avoid_: Noise (too vague — see Filler for the non-Fact case), decoy edge (used interchangeably but Distractor is canonical).

**Filler**:
Non-Fact tokens (or noise entity-pairs disjoint from all real/Distractor entities) used to pad gaps to a requested Distance or to reach the requested total sequence length. Never mistaken for a Fact.
_Avoid_: Padding (reserve for tokenizer/framework padding, a different concept), noise.

**Distance**:
The uniform gap (applied identically between every consecutive pair of Chain Facts in one example) separating chained Facts in the sequence. One scalar per example, not a per-hop profile.
Not a clean axis: because the Distractor count is a fixed absolute 2 (ADR 0004) while the space to place them grows with Distance, low-Distance cells are densely packed with Distractors and high-Distance cells are sparse. Distance and interference density therefore vary together in opposite directions, and empirically the density effect wins at the low end — distance=3 is the *hardest* column, not the easiest. See ADR 0013.

**Sequence Length**:
The total token length of one generated example. An independently controllable target; the generator raises an error rather than silently relaxing Hop Count, Distance, or Distractor count when the combination can't fit.

**Variant**:
One of the exactly three architectures under comparison — full attention, pure KDA, and hybrid. All three share one identical depth, width, and head count; a Variant differs from the others *only* in its sequence-mixing mechanism (and in the positional encoding that mechanism dictates). Anything else that differs between Variants is a confound, not a Variant difference.
_Avoid_: Model, architecture (both are broader — a Variant is specifically one arm of this comparison).

**Grid Cell**:
One (Hop Count, Distance) combination — 25 in total, from 5 Hop Counts × 5 Distances. The unit a Variant is scored on.
_Avoid_: Configuration, setting (too vague — those also describe hyperparameters, which a Grid Cell is not).

**Shortcut Floor**:
The accuracy a model reaches in a Grid Cell *without traversing the Chain at
all*. **The floor is 1.0 in every cell** — the task is entirely solvable
without reasoning. Two distinct shortcuts exist, and the stronger one dominates:

- **Positional (ADR 0015, the binding one).** Every gap is exactly Distance
  tokens long whatever is packed into it, and every Fact is exactly 3 tokens,
  so `answer_index == query_position - (Distance + 3)` always. Copying that
  position scores **100% in every cell, at every Hop Count and every Distractor
  count** (measured: 0 violations in 7,500 examples, including at maximum
  Distractor capacity). No eval-time parameter removes it.
- **Appears-exactly-once (ADR 0012).** The Chain's terminal entity is the only
  one never used as a subject. Scores 1/(1 + Distractor count): 100% at
  distance=0, 33.3% at distance≥3.

An earlier version of this entry named the 33.3%/100% appears-once figure as
"the floor every result must be read against". That is **false** and is kept
here as a correction, because it is the sentence a reader would most reasonably
trust: the appears-once figure is a floor, not *the* floor.

Consequence: **no accuracy number from the current generator distinguishes
traversal from shortcut execution.** Which mechanism a trained model actually
uses is a separate empirical question — see `scripts/probe_mechanism.py`.
_Avoid_: Baseline (means the full-attention Variant here), chance (~0.000125
over the vocabulary, which is the wrong reference for this task).

**Degradation Grid**:
The experiment's output: one accuracy figure per Grid Cell for one Variant. Comparing Degradation Grids across Variants is the point of the study. A Degradation Grid measures capability on difficulty levels the Variant was trained on, not extrapolation to unseen ones.
_Avoid_: Results, eval grid (the latter names the sweep that produces the numbers, not the numbers themselves).
