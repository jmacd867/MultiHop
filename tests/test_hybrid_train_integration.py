"""Proves the hybrid Variant plugs into the *existing* training/eval/checkpoint code unmodified.

Mirrors tests/test_pure_kda_train_integration.py, with one addition that
matters more for this Variant than for the other two: a **checkpoint
save/load round-trip**.

The hybrid is the only Variant whose `nnx.List` of blocks is heterogeneous --
`KDABlock` and `TransformerBlock` interleaved -- so it is the only one whose
state tree exercises `_flatten_state`/`_unflatten_state`'s str<->int key
normalization (ADR 0007) against two different block structures under one
list. The 20-step GB10 sanity run showed `save_checkpoint` works for the
hybrid (576 finite tensors on disk), but nothing had ever loaded one back.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from multihop.data.generator import NUM_SPECIAL_TOKENS, generate_batch
from multihop.eval import evaluate_grid
from multihop.models.config import ModelConfig
from multihop.models.hybrid import HybridModel
from multihop.train import (
    TrainConfig,
    build_optimizer_tx,
    compute_loss,
    load_checkpoint,
    save_checkpoint,
    train_step_accum,
)

ENTITY_VOCAB_SIZE = 20


def make_model() -> HybridModel:
    model_vocab_size = NUM_SPECIAL_TOKENS + ENTITY_VOCAB_SIZE
    config = ModelConfig(
        # 4 is the smallest legal hybrid depth -- one whole group, so the stack
        # actually contains both block types (ADR 0011).
        n_layers=4,
        embed_dim=16,
        n_heads=2,
        ffn_dim=32,
        max_seq_len=287,
        vocab_size=model_vocab_size,
        positional_encoding="none",
    )
    return HybridModel(config, rngs=nnx.Rngs(0))


def make_micro_batches(rng: np.random.Generator, count: int) -> list[tuple[jnp.ndarray, jnp.ndarray]]:
    return [
        (jnp.asarray(tokens), jnp.asarray(answers))
        for tokens, _query_positions, answers in (
            generate_batch(
                rng, hop_count=5, distance=45, entity_vocab_size=ENTITY_VOCAB_SIZE, batch_size=4
            )
            for _ in range(count)
        )
    ]


def test_compute_loss_is_finite_for_hybrid_model() -> None:
    model = make_model()
    rng = np.random.default_rng(0)
    tokens, _query_positions, answers = generate_batch(
        rng, hop_count=3, distance=9, entity_vocab_size=ENTITY_VOCAB_SIZE, batch_size=4
    )
    loss = compute_loss(model, jnp.asarray(tokens), jnp.asarray(answers))
    assert jnp.isfinite(loss)


def test_train_step_accum_produces_finite_loss_and_params_for_hybrid_model() -> None:
    model = make_model()
    train_config = TrainConfig(batch_size=8, grad_accum_steps=2, total_steps=5, warmup_steps=1)
    optimizer = nnx.Optimizer(model, build_optimizer_tx(train_config), wrt=nnx.Param)

    loss = train_step_accum(model, optimizer, make_micro_batches(np.random.default_rng(0), 2))
    assert jnp.isfinite(loss)

    params = jax.tree.leaves(nnx.to_pure_dict(nnx.state(model)))
    assert all(bool(jnp.all(jnp.isfinite(param))) for param in params)


def test_evaluate_grid_runs_end_to_end_for_hybrid_model() -> None:
    model = make_model()
    grid = evaluate_grid(model, ENTITY_VOCAB_SIZE, n_examples_per_cell=4)
    assert len(grid) == 25
    assert all(0.0 <= accuracy <= 1.0 for accuracy in grid.values())


def test_checkpoint_round_trip_restores_the_heterogeneous_block_stack(tmp_path: Path) -> None:
    """Save then load a *trained* hybrid and get identical predictions back.

    Trained, not freshly-initialized: two identically-seeded fresh models would
    produce identical logits even if load_checkpoint silently restored nothing,
    so the assertion would pass on a broken loader.
    """
    train_config = TrainConfig(batch_size=8, grad_accum_steps=2, total_steps=5, warmup_steps=1)
    model = make_model()
    optimizer = nnx.Optimizer(model, build_optimizer_tx(train_config), wrt=nnx.Param)
    train_step_accum(model, optimizer, make_micro_batches(np.random.default_rng(0), 2))

    tokens, _query_positions, _answers = generate_batch(
        np.random.default_rng(7), hop_count=2, distance=3, entity_vocab_size=ENTITY_VOCAB_SIZE, batch_size=4
    )
    expected_logits = model(jnp.asarray(tokens))

    save_checkpoint(model, optimizer, step=5, path=tmp_path / "step_5")

    restored = make_model()
    restored_optimizer = nnx.Optimizer(restored, build_optimizer_tx(train_config), wrt=nnx.Param)
    step = load_checkpoint(restored, restored_optimizer, tmp_path / "step_5")

    assert step == 5
    np.testing.assert_allclose(
        np.asarray(restored(jnp.asarray(tokens))), np.asarray(expected_logits), atol=1e-6
    )
