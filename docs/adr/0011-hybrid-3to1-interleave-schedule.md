# Hybrid layer schedule: strict repeating 3:1 KDA-to-full-attention, no pinned layers

The hybrid variant needs a concrete rule for which of its 12 layers are KDA and
which are full attention. Kimi Linear's design (arXiv:2510.26692) is a 3:1 ratio
of linear-attention layers to full-attention layers, and this project matches
that ratio -- but a ratio alone doesn't determine a schedule, and the schedule
is baked into every trained checkpoint. Getting it wrong is not a
configuration change; it's a re-run of the variant.

**Decision: `KDA, KDA, KDA, FullAttention` repeating from layer 0**, giving
full-attention layers at indices **3, 7, and 11** and KDA everywhere else --
9 KDA layers and 3 full-attention layers. `n_layers=12` divides evenly into
three complete groups of four, so there is no remainder and no special-casing
of the final group. `HybridModel` **enforces** that `n_layers` is a whole
number of groups, because a partial one silently changes the ratio this Variant
is defined by: `n_layers=6` would be 5:1, `n_layers=10` would be 4:1, and
anything below 4 would contain no full-attention layer at all -- a
`PureKDAModel` under the hybrid's name. Each of those builds, trains, and
checkpoints normally, so construction is the only place it can be caught.

## No pinned layers

The rejected alternative was **pinning**: forcing a full-attention layer into a
specific position on principle -- most commonly the first layer (so that token
embeddings are mixed by unrestricted attention before any state compression) or
the last (so the final retrieval before the output head has full access to the
sequence) -- and then distributing the remainder to hit the 3:1 ratio.

Pinning was rejected because it introduces a second, independent architectural
choice into the one variant whose entire purpose is to interpolate between the
other two. A pinned schedule would make the hybrid "3:1 KDA/full-attention
*plus* a hand-placed attention layer," and any difference in its degradation
grid would then be ambiguous between the mixing ratio and the placement
heuristic -- the same class of confound ADR 0001 avoids for positional encoding
and ADR 0009 avoids by excluding ShortConv. A strict repeat is the schedule with
the fewest free parameters: state the ratio and the phase, and the whole stack
follows.

## Consequence: layer 11 is full attention, and that is accepted, not accidental

A strict repeat starting at layer 0 places a full-attention layer immediately
before `final_norm` and the output head. This looks like it could be the result
of pinning-the-last-layer, and a future reader may reasonably wonder which it
was. It is not: it falls out of the phase choice, and the phase was chosen
first.

The consequence is nonetheless favorable and worth stating rather than
discovering. The query token's final act before the output head happens in a
layer with unrestricted access to every prior position, so the hybrid is not
asked to complete its last retrieval step through a compressed state. If the
hybrid's degradation grid later looks better than the 3:1 ratio alone would
suggest -- particularly at high hop counts, where the final retrieval is the
one most likely to fail -- this is the first thing to check, and the honest
comparison would be against the opposite phase
(`FullAttention, KDA, KDA, KDA`, full attention at 0/4/8), not against pure KDA.
That ablation is out of scope here; it is recorded as the specific follow-up
this decision makes possible.

## What this schedule does not change

Depth, width, head count, and `ModelConfig` are shared with the other two
variants unchanged (ADR 0009). The KDA layers are the existing `KDABlock`
and the full-attention layers the existing `TransformerBlock`, both reused
as-is rather than forked. Positional encoding follows ADR 0001's per-variant
rule: because the hybrid contains KDA layers, it is NoPE throughout --
including its three full-attention layers, which do *not* get RoPE back. That
is ADR 0001's explicit intent and the reading it warns against "fixing" into
per-layer-type consistency.

Because both block types share the same input and output shape, a wrong
schedule is silent -- the model builds, trains, and produces plausible losses
with the layers in the wrong order or the wrong ratio. `tests/models/test_hybrid.py`
therefore asserts the concrete layer-type ordering, not just stack shape.
