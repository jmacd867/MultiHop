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

**Sequence Length**:
The total token length of one generated example. An independently controllable target; the generator raises an error rather than silently relaxing Hop Count, Distance, or Distractor count when the combination can't fit.

**Variant**:
One of the exactly three architectures under comparison — full attention, pure KDA, and hybrid. All three share one identical depth, width, and head count; a Variant differs from the others *only* in its sequence-mixing mechanism (and in the positional encoding that mechanism dictates). Anything else that differs between Variants is a confound, not a Variant difference.
_Avoid_: Model, architecture (both are broader — a Variant is specifically one arm of this comparison).

**Grid Cell**:
One (Hop Count, Distance) combination — 25 in total, from 5 Hop Counts × 5 Distances. The unit a Variant is scored on.
_Avoid_: Configuration, setting (too vague — those also describe hyperparameters, which a Grid Cell is not).

**Degradation Grid**:
The experiment's output: one accuracy figure per Grid Cell for one Variant. Comparing Degradation Grids across Variants is the point of the study. A Degradation Grid measures capability on difficulty levels the Variant was trained on, not extrapolation to unseen ones.
_Avoid_: Results, eval grid (the latter names the sweep that produces the numbers, not the numbers themselves).
