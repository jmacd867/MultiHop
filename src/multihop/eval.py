"""Hop-count x distance accuracy grid, generic across the three model variants.

Examples are generated fresh from the held-out eval-vocab pool each time
`evaluate_grid` runs (no fixed eval set), matching the on-the-fly training
setup in `train.py`. 512 examples per cell (ADR-level decision) keeps
per-cell standard error under ~2.2 percentage points at worst case.
"""

from collections.abc import Mapping

import jax.numpy as jnp
import numpy as np

from multihop.data.generator import DISTANCES, MAX_HOP_COUNT, MIN_HOP_COUNT, generate_batch, total_vocab_size

HOP_COUNTS: tuple[int, ...] = tuple(range(MIN_HOP_COUNT, MAX_HOP_COUNT + 1))
EVAL_EXAMPLES_PER_CELL = 512


def accuracy_at_cell(
    model: object,
    hop_count: int,
    distance: int,
    entity_vocab_size: int,
    n_examples: int,
    rng: np.random.Generator,
) -> float:
    tokens, query_positions, answers = generate_batch(
        rng, hop_count, distance, entity_vocab_size, n_examples, split="eval"
    )

    logits = model(jnp.asarray(tokens))  # type: ignore[operator]
    query_logits = logits[jnp.arange(n_examples), jnp.asarray(query_positions)]
    predicted = jnp.argmax(query_logits, axis=-1)

    return float(jnp.mean(predicted == jnp.asarray(answers)))


def evaluate_grid(
    model: object,
    entity_vocab_size: int,
    rng: np.random.Generator,
    n_examples_per_cell: int = EVAL_EXAMPLES_PER_CELL,
) -> Mapping[tuple[int, int], float]:
    """Accuracy broken down by (hop_count, distance) at the fixed reference distractor count."""
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
