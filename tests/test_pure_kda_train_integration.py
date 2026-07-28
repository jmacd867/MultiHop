"""Proves the pure-KDA variant plugs into the *existing* training/eval code unmodified
(other than train.py's model type going from concrete-class to Protocol-typed, ADR-adjacent
housekeeping needed for genuine cross-variant reuse) -- no KDA-specific copy of
train_step_accum, compute_loss, or evaluate_grid. See the pure-KDA task's integration step.
"""

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from multihop.data.generator import NUM_SPECIAL_TOKENS, generate_batch
from multihop.eval import evaluate_grid
from multihop.models.config import ModelConfig
from multihop.models.pure_kda import PureKDAModel
from multihop.train import (
    TrainConfig,
    build_optimizer_tx,
    compute_loss,
    train_step_accum,
)

ENTITY_VOCAB_SIZE = 20


def make_model() -> PureKDAModel:
    model_vocab_size = NUM_SPECIAL_TOKENS + ENTITY_VOCAB_SIZE
    config = ModelConfig(
        n_layers=1,
        embed_dim=16,
        n_heads=2,
        ffn_dim=32,
        max_seq_len=287,
        vocab_size=model_vocab_size,
        positional_encoding="none",
    )
    return PureKDAModel(config, rngs=nnx.Rngs(0))


def test_compute_loss_is_finite_for_kda_model() -> None:
    model = make_model()
    rng = np.random.default_rng(0)
    tokens, _query_positions, answers = generate_batch(
        rng, hop_count=3, distance=9, entity_vocab_size=ENTITY_VOCAB_SIZE, batch_size=4
    )
    loss = compute_loss(model, jnp.asarray(tokens), jnp.asarray(answers))
    assert jnp.isfinite(loss)


def test_train_step_accum_produces_finite_loss_and_params_for_kda_model() -> None:
    model = make_model()
    train_config = TrainConfig(batch_size=8, grad_accum_steps=2, total_steps=5, warmup_steps=1)
    optimizer = nnx.Optimizer(model, build_optimizer_tx(train_config), wrt=nnx.Param)

    rng = np.random.default_rng(0)
    micro_batches = [
        (jnp.asarray(tokens), jnp.asarray(answers))
        for tokens, _query_positions, answers in (
            generate_batch(rng, hop_count=5, distance=45, entity_vocab_size=ENTITY_VOCAB_SIZE, batch_size=4)
            for _ in range(2)
        )
    ]

    loss = train_step_accum(model, optimizer, micro_batches)
    assert jnp.isfinite(loss)

    params = jax.tree.leaves(nnx.to_pure_dict(nnx.state(model)))
    assert all(bool(jnp.all(jnp.isfinite(param))) for param in params)


def test_evaluate_grid_runs_end_to_end_for_kda_model() -> None:
    model = make_model()
    grid = evaluate_grid(model, ENTITY_VOCAB_SIZE, n_examples_per_cell=4)
    assert len(grid) == 25
    assert all(0.0 <= accuracy <= 1.0 for accuracy in grid.values())
