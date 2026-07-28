import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from safetensors import safe_open

from multihop.data.generator import (
    DISTANCES,
    NUM_SPECIAL_TOKENS,
    generate_batch,
    max_distractor_capacity,
    reference_distractor_count,
)
from multihop.models.baseline import FullAttentionBaseline
from multihop.models.config import ModelConfig
from multihop.train import (
    TrainConfig,
    build_optimizer_tx,
    compute_loss,
    load_checkpoint,
    load_model_checkpoint,
    sample_cell,
    sample_training_batch,
    sample_training_microbatches,
    save_checkpoint,
    train_step,
    train_step_accum,
)

ENTITY_VOCAB_SIZE = 20


def make_model() -> FullAttentionBaseline:
    model_vocab_size = NUM_SPECIAL_TOKENS + ENTITY_VOCAB_SIZE
    config = ModelConfig(
        # max_seq_len must cover the longest cell these tests exercise:
        # hop_count=5, distance=45 -> 287 tokens (the production default).
        n_layers=1, embed_dim=16, n_heads=2, ffn_dim=32, max_seq_len=287, vocab_size=model_vocab_size
    )
    return FullAttentionBaseline(config, rngs=nnx.Rngs(0))


def test_reference_distractor_count_zero_at_distance_zero() -> None:
    # distance=0 has zero distractor capacity (ADR 0004); the reference
    # count must be forced to 0 there rather than raising.
    assert reference_distractor_count(0) == 0
    assert max_distractor_capacity(hop_count=5, distance=0) == 0


@pytest.mark.parametrize("distance", [3, 9, 21, 45])
def test_reference_distractor_count_fits_capacity_at_every_hop_count(distance: int) -> None:
    count = reference_distractor_count(distance)
    assert count == 2
    for hop_count in range(1, 6):
        assert count <= max_distractor_capacity(hop_count, distance)


def test_cell_sampling_covers_full_grid_and_is_roughly_balanced() -> None:
    # Guards against a sampling bug silently starving a cell, which would be
    # a subtle confound in the degradation grid (easy to miss by eye, hard
    # to distinguish from a genuine architecture effect after the fact).
    rng = np.random.default_rng(0)
    n_draws = 20_000
    cells = [sample_cell(rng) for _ in range(n_draws)]

    hop_counts = [hop_count for hop_count, _distance in cells]
    distances = [distance for _hop_count, distance in cells]

    seen_cells = set(cells)
    assert seen_cells == {
        (hop_count, distance) for hop_count in range(1, 6) for distance in DISTANCES
    }

    # 5 hop_count values, 5 distance values: expected count per bucket is
    # n_draws / 5 = 4000; allow generous slack (well beyond sampling noise
    # at this N) so this only fires on an actual imbalance bug, not variance.
    expected_per_bucket = n_draws / 5
    for hop_count in range(1, 6):
        count = hop_counts.count(hop_count)
        assert abs(count - expected_per_bucket) < expected_per_bucket * 0.15
    for distance in DISTANCES:
        count = distances.count(distance)
        assert abs(count - expected_per_bucket) < expected_per_bucket * 0.15


def test_sample_training_batch_shapes_are_consistent() -> None:
    rng = np.random.default_rng(0)
    tokens, query_positions, answers, hop_count, distance = sample_training_batch(
        rng, batch_size=4, vocab_size=ENTITY_VOCAB_SIZE
    )
    assert distance in DISTANCES
    assert 1 <= hop_count <= 5
    assert tokens.shape[0] == 4
    assert query_positions.shape == (4,)
    assert answers.shape == (4,)
    # token at query_position is the query's start entity (chain_entities[0]);
    # `answers` is the terminal entity the model must predict as the *next*
    # token, so it's metadata alongside `tokens`, not itself in the sequence.
    for i in range(4):
        assert tokens[i, query_positions[i]] != answers[i]


def test_compute_loss_is_finite() -> None:
    model = make_model()
    rng = np.random.default_rng(0)
    tokens, _query_positions, answers, *_ = sample_training_batch(
        rng, batch_size=4, vocab_size=ENTITY_VOCAB_SIZE
    )
    loss = compute_loss(model, jnp.asarray(tokens), jnp.asarray(answers))
    assert jnp.isfinite(loss)


def test_compute_loss_actually_supervises_the_answer_token() -> None:
    # The answer is never a token in `tokens` (it's the next-token target
    # for the position right after query_position, which has no real
    # "next" token in the sequence itself). Guard against the loss quietly
    # excluding that final position: swapping in wrong answers, holding
    # tokens fixed, must change the loss.
    model = make_model()
    rng = np.random.default_rng(0)
    tokens, _query_positions, answers, *_ = sample_training_batch(
        rng, batch_size=8, vocab_size=ENTITY_VOCAB_SIZE
    )
    model_vocab_size = NUM_SPECIAL_TOKENS + ENTITY_VOCAB_SIZE
    wrong_answers = (answers + 1) % model_vocab_size

    loss_correct = compute_loss(model, jnp.asarray(tokens), jnp.asarray(answers))
    loss_wrong = compute_loss(model, jnp.asarray(tokens), jnp.asarray(wrong_answers))

    assert not jnp.allclose(loss_correct, loss_wrong)


def test_answer_loss_weight_does_not_dilute_at_long_sequences() -> None:
    # Sequences in mixed-cell training range from 5 tokens (hop_count=1,
    # distance=0) to 287 (hop_count=5, distance=45). A naive concatenate-
    # then-mean-over-all-positions loss would let the single answer term's
    # effective weight shrink to ~1/seq_len, so the marginal contribution
    # of answer_loss_weight would be ~57x smaller at the long end than the
    # short end. Isolate that contribution directly (loss at weight=1 minus
    # loss at weight=0, i.e. exactly the batch-mean answer_loss) and check
    # it stays the same order of magnitude at both ends.
    model = make_model()
    rng = np.random.default_rng(0)
    short_tokens, _sqp, short_answers = generate_batch(
        rng, hop_count=1, distance=0, entity_vocab_size=ENTITY_VOCAB_SIZE, batch_size=8
    )
    long_tokens, _lqp, long_answers = generate_batch(
        rng, hop_count=5, distance=45, entity_vocab_size=ENTITY_VOCAB_SIZE, batch_size=8
    )
    assert short_tokens.shape[1] == 5
    assert long_tokens.shape[1] == 287

    def answer_term_contribution(tokens: np.ndarray, answers: np.ndarray) -> float:
        loss_with_answer_term = compute_loss(
            model, jnp.asarray(tokens), jnp.asarray(answers), answer_loss_weight=1.0
        )
        loss_without_answer_term = compute_loss(
            model, jnp.asarray(tokens), jnp.asarray(answers), answer_loss_weight=0.0
        )
        return float(loss_with_answer_term - loss_without_answer_term)

    short_contribution = answer_term_contribution(short_tokens, short_answers)
    long_contribution = answer_term_contribution(long_tokens, long_answers)

    # Cross-entropy over a ~43-class vocab for an untrained model is O(1)
    # (~log(vocab_size) nats); both must stay well above a floor that a
    # ~57x dilution would have crushed the long-sequence one below.
    assert short_contribution > 0.5
    assert long_contribution > 0.5


def test_train_step_produces_finite_loss_and_params_at_longest_sequence() -> None:
    # hop_count=5, distance=45 is the longest grid cell (287 tokens). A
    # literal -inf fill value in the causal mask (jnp.where(mask, scores,
    # -inf)) produces NaN gradients during backprop -- a known JAX gotcha
    # where the forward-pass value is correctly masked but the backward
    # pass still evaluates 0 * -inf terms. None of the other tests here
    # take an actual gradient through this specific (longest) sequence, so
    # this is the one that would have caught it.
    model = make_model()
    train_config = TrainConfig(batch_size=4, grad_accum_steps=1, total_steps=5, warmup_steps=1)
    optimizer = nnx.Optimizer(model, build_optimizer_tx(train_config), wrt=nnx.Param)

    rng = np.random.default_rng(0)
    tokens, _query_positions, answers = generate_batch(
        rng, hop_count=5, distance=45, entity_vocab_size=ENTITY_VOCAB_SIZE, batch_size=4
    )

    loss = train_step(model, optimizer, jnp.asarray(tokens), jnp.asarray(answers))
    assert jnp.isfinite(loss)

    params = jax.tree.leaves(nnx.to_pure_dict(nnx.state(model)))
    assert all(bool(jnp.all(jnp.isfinite(param))) for param in params)


def test_checkpoint_round_trip_restores_step_params_and_optimizer_state() -> None:
    model = make_model()
    train_config = TrainConfig(batch_size=4, grad_accum_steps=1, total_steps=5, warmup_steps=1)
    optimizer = nnx.Optimizer(model, build_optimizer_tx(train_config), wrt=nnx.Param)

    rng = np.random.default_rng(0)
    tokens, _query_positions, answers, *_ = sample_training_batch(
        rng, batch_size=4, vocab_size=ENTITY_VOCAB_SIZE
    )
    train_step(model, optimizer, jnp.asarray(tokens), jnp.asarray(answers))

    trained_params = nnx.to_pure_dict(nnx.state(model))
    trained_opt_state = nnx.to_pure_dict(nnx.state(optimizer))

    with tempfile.TemporaryDirectory() as tmp_dir:
        checkpoint_path = Path(tmp_dir) / "step_7"
        save_checkpoint(model, optimizer, step=7, path=checkpoint_path)

        model_file = Path(tmp_dir) / "step_7.model.safetensors"
        optimizer_file = Path(tmp_dir) / "step_7.optimizer.safetensors"
        assert model_file.exists()
        assert optimizer_file.exists()
        for file in (model_file, optimizer_file):
            with safe_open(file, framework="numpy") as f:
                assert f.metadata() == {"step": "7"}

        fresh_model = make_model()
        fresh_optimizer = nnx.Optimizer(fresh_model, build_optimizer_tx(train_config), wrt=nnx.Param)
        restored_step = load_checkpoint(fresh_model, fresh_optimizer, checkpoint_path)

    assert restored_step == 7

    restored_params = nnx.to_pure_dict(nnx.state(fresh_model))
    trained_leaves = jax.tree.leaves(trained_params)
    restored_leaves = jax.tree.leaves(restored_params)
    assert len(trained_leaves) == len(restored_leaves)
    for trained_leaf, restored_leaf in zip(trained_leaves, restored_leaves, strict=True):
        np.testing.assert_allclose(np.asarray(trained_leaf), np.asarray(restored_leaf))

    restored_opt_state = nnx.to_pure_dict(nnx.state(fresh_optimizer))
    trained_opt_leaves = jax.tree.leaves(trained_opt_state)
    restored_opt_leaves = jax.tree.leaves(restored_opt_state)
    assert len(trained_opt_leaves) == len(restored_opt_leaves)
    for trained_leaf, restored_leaf in zip(trained_opt_leaves, restored_opt_leaves, strict=True):
        np.testing.assert_allclose(np.asarray(trained_leaf), np.asarray(restored_leaf))


def test_load_model_checkpoint_restores_weights_without_touching_optimizer_file() -> None:
    model = make_model()
    train_config = TrainConfig(batch_size=4, grad_accum_steps=1, total_steps=5, warmup_steps=1)
    optimizer = nnx.Optimizer(model, build_optimizer_tx(train_config), wrt=nnx.Param)

    rng = np.random.default_rng(0)
    tokens, _query_positions, answers, *_ = sample_training_batch(
        rng, batch_size=4, vocab_size=ENTITY_VOCAB_SIZE
    )
    train_step(model, optimizer, jnp.asarray(tokens), jnp.asarray(answers))
    trained_params = nnx.to_pure_dict(nnx.state(model))

    with tempfile.TemporaryDirectory() as tmp_dir:
        checkpoint_path = Path(tmp_dir) / "step_7"
        save_checkpoint(model, optimizer, step=7, path=checkpoint_path)

        # Prove this is genuinely a model-only load, not just a load_checkpoint
        # that happens to work: delete the optimizer file first.
        optimizer_file = Path(tmp_dir) / "step_7.optimizer.safetensors"
        optimizer_file.unlink()

        fresh_model = make_model()
        restored_step = load_model_checkpoint(fresh_model, checkpoint_path)

    assert restored_step == 7
    restored_params = nnx.to_pure_dict(nnx.state(fresh_model))
    trained_leaves = jax.tree.leaves(trained_params)
    restored_leaves = jax.tree.leaves(restored_params)
    assert len(trained_leaves) == len(restored_leaves)
    for trained_leaf, restored_leaf in zip(trained_leaves, restored_leaves, strict=True):
        np.testing.assert_allclose(np.asarray(trained_leaf), np.asarray(restored_leaf))


def test_train_config_rejects_batch_size_not_divisible_by_grad_accum_steps() -> None:
    with pytest.raises(ValueError, match="divisible"):
        TrainConfig(batch_size=10, grad_accum_steps=3)


def test_train_config_derives_micro_batch_size() -> None:
    config = TrainConfig(batch_size=256, grad_accum_steps=16)
    assert config.micro_batch_size == 16


def test_sample_training_microbatches_are_identically_shaped_and_share_a_cell() -> None:
    rng = np.random.default_rng(0)
    micro_batches, hop_count, distance = sample_training_microbatches(
        rng, micro_batch_size=4, grad_accum_steps=6, vocab_size=ENTITY_VOCAB_SIZE
    )
    assert len(micro_batches) == 6
    assert distance in DISTANCES
    assert 1 <= hop_count <= 5
    expected_shape = micro_batches[0][0].shape
    for tokens, answers in micro_batches:
        assert tokens.shape == expected_shape
        assert tokens.shape[0] == 4
        assert answers.shape == (4,)


def test_train_step_accum_matches_a_true_batch_step_of_the_same_data() -> None:
    # The load-bearing correctness check for gradient accumulation: averaging
    # gradients from grad_accum_steps micro-batches must produce (up to
    # float summation-order noise) the same parameter update as a single
    # true-batch step over the identical examples. generate_batch draws
    # examples sequentially from `rng` (one generate_example call per
    # example), so a fixed-seed rng produces the same example stream whether
    # drawn as 4x4 micro-batches or 1x16 -- this is what makes an exact
    # (not just distributionally similar) comparison possible.
    hop_count, distance = 3, 3

    def make_optimizer(model: FullAttentionBaseline) -> nnx.Optimizer[FullAttentionBaseline]:
        config = TrainConfig(batch_size=16, grad_accum_steps=4, total_steps=5, warmup_steps=1)
        return nnx.Optimizer(model, build_optimizer_tx(config), wrt=nnx.Param)

    accum_model = make_model()
    accum_optimizer = make_optimizer(accum_model)
    rng = np.random.default_rng(0)
    micro_batches = [
        (jnp.asarray(tokens), jnp.asarray(answers))
        for tokens, _query_positions, answers in (
            generate_batch(rng, hop_count, distance, ENTITY_VOCAB_SIZE, batch_size=4)
            for _ in range(4)
        )
    ]
    train_step_accum(accum_model, accum_optimizer, micro_batches)

    true_model = make_model()
    true_optimizer = make_optimizer(true_model)
    true_rng = np.random.default_rng(0)
    true_tokens, _query_positions, true_answers = generate_batch(
        true_rng, hop_count, distance, ENTITY_VOCAB_SIZE, batch_size=16
    )
    train_step(true_model, true_optimizer, jnp.asarray(true_tokens), jnp.asarray(true_answers))

    accum_leaves = jax.tree.leaves(nnx.to_pure_dict(nnx.state(accum_model)))
    true_leaves = jax.tree.leaves(nnx.to_pure_dict(nnx.state(true_model)))
    assert len(accum_leaves) == len(true_leaves)
    for accum_leaf, true_leaf in zip(accum_leaves, true_leaves, strict=True):
        # Tight tolerance deliberately: this is checking a specific
        # correctness property (averaged accumulation == true batch), not
        # "roughly similar training dynamics". A wide tolerance here would
        # mask real bugs (e.g. summing instead of averaging gradients, or an
        # off-by-one dropping a micro-batch) as float noise.
        np.testing.assert_allclose(np.asarray(accum_leaf), np.asarray(true_leaf), rtol=1e-5, atol=1e-6)
