"""Hop-count x distance accuracy grid, generic across the three model variants.

Examples are generated fresh from the shared entity pool each time
`evaluate_grid` runs (no fixed eval set, no reserved id range -- see
ADR 0008), matching the on-the-fly training setup in `train.py`. 512
examples per cell (ADR-level decision) keeps per-cell standard error under
~2.2 percentage points at worst case.
"""

from collections.abc import Mapping

import jax.numpy as jnp
import numpy as np

from multihop.data.generator import DISTANCES, MAX_HOP_COUNT, MIN_HOP_COUNT, generate_batch, total_vocab_size

HOP_COUNTS: tuple[int, ...] = tuple(range(MIN_HOP_COUNT, MAX_HOP_COUNT + 1))
EVAL_EXAMPLES_PER_CELL = 512

# Shared across every evaluation of every model variant (baseline, KDA,
# hybrid) -- see ADR 0008. Comparability of the three degradation grids
# requires all three to be scored on the identical freshly-generated
# examples per cell, isolating architecture as the only variable rather
# than adding independent per-variant sampling noise on top of the
# per-cell measurement noise EVAL_EXAMPLES_PER_CELL already accounts for.
EVAL_SEED = 1


def accuracy_at_cell(
    model: object,
    hop_count: int,
    distance: int,
    entity_vocab_size: int,
    n_examples: int,
    rng: np.random.Generator,
) -> float:
    tokens, query_positions, answers = generate_batch(
        rng, hop_count, distance, entity_vocab_size, n_examples
    )

    logits = model(jnp.asarray(tokens))  # type: ignore[operator]
    query_logits = logits[jnp.arange(n_examples), jnp.asarray(query_positions)]
    predicted = jnp.argmax(query_logits, axis=-1)

    return float(jnp.mean(predicted == jnp.asarray(answers)))


def evaluate_grid(
    model: object,
    entity_vocab_size: int,
    rng: np.random.Generator | None = None,
    n_examples_per_cell: int = EVAL_EXAMPLES_PER_CELL,
) -> Mapping[tuple[int, int], float]:
    """Accuracy broken down by (hop_count, distance) at the fixed reference distractor count.

    `rng` defaults to a fresh `np.random.default_rng(EVAL_SEED)` so
    cross-variant comparability (ADR 0008) is the default behavior, not
    something every caller has to remember to opt into. Pass an explicit
    `rng` to override (e.g. test isolation).
    """
    if rng is None:
        rng = np.random.default_rng(EVAL_SEED)
    expected_vocab_size = total_vocab_size(entity_vocab_size)
    model_vocab_size = model.config.vocab_size  # type: ignore[attr-defined]
    if model_vocab_size != expected_vocab_size:
        raise ValueError(
            f"model.config.vocab_size={model_vocab_size} does not match "
            f"total_vocab_size(entity_vocab_size={entity_vocab_size})={expected_vocab_size} "
            "-- the model's embedding table won't cover the token ids the generator produces"
        )

    return {
        (hop_count, distance): accuracy_at_cell(
            model, hop_count, distance, entity_vocab_size, n_examples_per_cell, rng
        )
        for hop_count in HOP_COUNTS
        for distance in DISTANCES
    }
