# Model sizing: 8,000-entity vocab per split, tied embeddings, ~125M params

We need a concrete `vocab_size` for the generator's `GeneratorConfig`, which sets the size of the disjoint train and eval entity pools, plus a decision on whether the model's input embedding and output head share weights. These are really one connected sizing decision — vocab size, tying, and the ~125M target all trade off against each other — so they're recorded together rather than split across separate notes.

## Vocab size: 8,000 per split (16,003 total)

Unlike the RoPE/NoPE choice in ADR 0001, there's no principled derivation for this number — it's a balance point chosen by feel between two competing pressures, so it's worth recording *why* this value rather than another.

**The trade-off:**

- **Too small** and the training set can realistically cover a large fraction of the entity-pair space, letting a model shortcut the task via pair memorization instead of genuine chain traversal — this would confound the hop-distance degradation results we're trying to measure.
- **Too large** and the embedding table (and tied/untied output head) dominates the ~125M param budget shared across all three variants, leaving too little capacity in the actual mechanism under study (full attention vs. KDA vs. hybrid).

**The choice:** 8,000 entities per split, giving `NUM_SPECIAL_TOKENS + 2*vocab_size = 16,003` total vocab.

- 8,000 entities ⇒ ~64M possible (subject, object) pairs — far larger than any training run will cover, so pair memorization isn't a viable shortcut.
- 16,003 rows in the embedding table is a minor fraction of 125M params (≈12.3M at an embed dim of ~768, under 10% of budget).

## Embedding/output-head tying: tied

Tying the input embedding and output projection is the standard choice at this model scale, and there's no domain reason here to expect input/output token semantics to diverge — this is a closed synthetic vocab of entities and special tokens, not natural language with asymmetric input/output statistics. Tying keeps more of the ~125M budget in the transformer blocks themselves (the attention mechanism actually under study across the three variants) rather than paying for two separate 16,003×768 matrices.

## The connected budget

16,003 vocab, tied embeddings, ~768 embed dim, ~125M param target. If the degradation curves look strange later (e.g. suspiciously flat with hop count, or a variant doing implausibly well at high distance), one of the first things to check is whether this sizing is confounding the result — either the vocab still being small enough to leak partial memorization, or the embedding table (tied or not) eating enough of the budget to starve the attention mechanism relative to what the paper's baselines assume.
