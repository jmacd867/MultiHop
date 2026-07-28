# Shared entity pool for binding generalization (reverses ADR 0002's disjoint train/eval pools)

The first full 20,000-step baseline run (`docs/runs/baseline_run_20260727/baseline_run.log`)
completed with normally-decreasing training loss but a flat `0.0000` eval accuracy across
every one of the 25 (hop_count, distance) grid cells, from step 5500 through step 20000.
This is not a training-loop or eval-indexing bug -- eval's indexing matches
`compute_loss`'s exactly, and the failure signature (diagnosed via ad hoc scripts against
the `step_20000` checkpoint) was eval-vocab predictions being **input-invariant**: the
model always predicted from a small fixed set of tokens regardless of what chain was
actually in the input.

## Root cause

ADR 0002 partitioned the entity vocabulary into disjoint, fixed train and eval id ranges
(`entity_pool(split, vocab_size)`). With tied embeddings, output logits are
`hidden_state @ embedding_matrix^T` -- so the eval-range rows of that same matrix are read
at every forward pass, but only the train-range rows ever receive a gradient. After 20,000
steps, the eval-range rows still sit at their random init, and argmax over the full
vocabulary essentially never selects one of them. The model wasn't failing to reason about
multi-hop chains; it had no way to ever be evaluated correctly, because eval was scored
against embedding rows the optimizer was structurally forbidden from touching.

## Precedent: MQAR (Zoology/Based, and the DeltaNet/Gated-DeltaNet/KDA lineage)

This project's whole premise is comparing full attention against the KDA/hybrid lineage
descended from that literature, and Multi-Query Associative Recall is exactly this failure
mode's namesake benchmark. MQAR's design draws keys and values uniformly from **one shared
vocabulary, resampled fresh per example**, specifically because a fixed train/eval
partition lets a model shortcut via a static key->value lookup table instead of doing
genuine in-context binding. Our disjoint pools accidentally recreated the failure MQAR's
design exists to prevent -- and did so more severely, since ours made correct eval
prediction not just discouraged but structurally impossible under tied embeddings.

## The fix

**One shared entity pool**, size 8,000, used by every generated example -- train and eval
alike. `entity_pool` and the `Split` type are deleted; `generate_example`/`generate_batch`
no longer take a `split` argument. `total_vocab_size` changes from
`NUM_SPECIAL_TOKENS + 2 * entity_vocab_size` (two disjoint pools) to
`NUM_SPECIAL_TOKENS + entity_vocab_size` (one pool), so `ModelConfig.vocab_size`'s default
moves from 16,003 to 8,003 and the model's actual param count drops from the previously
"~125M" figure to ~119.41M (measured directly from the instantiated model, not
recomputed by hand). ADR 0002's memorization-infeasibility argument (~64M possible
(subject, object) pairs from an 8,000-entity pool, against a run of a few million examples)
carries over unchanged -- that math was already anchored to one pool's size, not the
combined 16,000, so nothing about pair-memorization risk gets worse by dropping the second
pool. The embedding table shrinking is a bonus in the direction ADR 0002 already wanted
(more of the param budget staying in the transformer blocks under study).

**"Eval" no longer means a reserved id range.** Chain entities in `generate_example` are
already resampled fresh every call (this was never the bug) -- the fix removes the
train/eval id partition, not the per-example resampling, which already existed. Given the
combinatorics above, the odds of an eval example's exact chain having appeared verbatim
during training are already negligible, so held-out-ness now comes from binding
combinatorics rather than a reserved token range. This keeps ADR 0003's "capacity, not
extrapolation" framing intact: we still aren't holding out (hop_count, distance) cells,
just no longer holding out tokens either.

**A fixed, shared eval RNG seed (`EVAL_SEED = 1` in `eval.py`) replaces the reserved pool**
as the mechanism protecting what disjoint pools were actually needed for: not overfitting
protection (the combinatorics already make that a non-issue), but **cross-variant
comparability**. The three architectures (full attention, KDA, hybrid) must be scored on
the identical freshly-generated examples per cell, or comparing their degradation grids
would conflate an architecture effect with independent per-variant sampling noise, on top
of the ~2.2pp per-cell measurement noise the 512-examples-per-cell decision already
accounts for. `evaluate_grid`'s `rng` parameter now defaults to
`np.random.default_rng(EVAL_SEED)` so this is the default behavior, not something every
caller (including the not-yet-written KDA/hybrid eval scripts) has to remember to opt into.

## What's explicitly not changing

Symbolic tokens, hop_count 1-5, the distance set `{0, 3, 9, 21, 45}`, decoy-edge
distractors (ADR 0004), and hard-erroring on infeasible configs are all unchanged -- this
is a fix to the vocab-binding mechanism, not a redesign of the task structure.

## Disposition of the failed run

The 20,000-step checkpoint directory (~29GB) was deleted from the GB10 box -- weights of a
model whose core mechanism was memorization have no future use (no fine-tuning, no
resumption, no attention-pattern comparison against the eventual fixed baseline, since that
will be a genuinely different training run on a genuinely different data distribution). The
run's log and the `timing_run.py` wall-clock measurement were pulled back into
`docs/runs/baseline_run_20260727/` before deletion, since they're small and have real
referential value (this ADR, and the eventual real run's wall-clock budget). The two ad hoc
diagnostic scripts used to root-cause this (`diagnose_eval.py`, `diagnose_eval2.py`) were
not committed -- their core check (counting unique predicted tokens to detect
input-invariance) is what `tests/test_binding_generalization.py` now covers as a standing
regression test, and their one other check (query_position indexing) was already covered by
existing tests in `tests/test_train.py`.

## Status

ADR 0002 is left unedited -- it stands as a record of the disjoint-pool design that was
tried and why it failed, per this project's convention of not rewriting history when a
decision is reversed. This ADR supersedes ADR 0002's vocab-partitioning decision (not its
tying decision, which is unaffected) and is the current source of truth for `vocab_size`
and pool structure.
