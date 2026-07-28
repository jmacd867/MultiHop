"""Regression test for the train/eval-vocab memorization bug (ADR 0008).

Under the old disjoint-pool design, tied embeddings meant the eval-range
entity rows were never touched by gradient descent, so the trained model
always predicted from a small fixed set of trained tokens regardless of
input -- accuracy silently flatlined at 0.0000 for an entire 20,000-step
run (docs/runs/baseline_run_20260727/baseline_run.log). A model that's
actually resolving chains from context should predict *different* answer
tokens for different freshly-generated chains, not the same one(s) every
time.

This trains a real (if tiny) model for real -- not a constructed-weights
stand-in -- because the bug was an emergent property of the interaction
between the generator, `compute_loss`, tied embeddings, and gradient
descent; a test that only checked variance-detection logic in isolation
would pass even if a future regression reintroduced the same failure
through a different mechanism.

Training is a direct loop over `train_step` at the single easiest grid
cell (hop_count=1, distance=0), not the full `train()` mixed-grid loop --
`train()` samples uniformly across all 25 (hop_count, distance) cells,
which would spend most of a short step budget on much harder cells than
this test needs to make its point quickly and reliably.
"""

import jax.numpy as jnp
import numpy as np
from flax import nnx

from multihop.data.generator import generate_batch, total_vocab_size
from multihop.models.baseline import FullAttentionBaseline
from multihop.models.config import ModelConfig
from multihop.train import TrainConfig, build_optimizer_tx, train_step

ENTITY_VOCAB_SIZE = 200
HOP_COUNT = 1
DISTANCE = 0


def test_predictions_vary_with_context_after_brief_training() -> None:
    model_config = ModelConfig(
        vocab_size=total_vocab_size(ENTITY_VOCAB_SIZE),
        n_layers=2,
        embed_dim=32,
        n_heads=4,
        ffn_dim=64,
        max_seq_len=10,
    )
    model = FullAttentionBaseline(model_config, rngs=nnx.Rngs(0))
    train_config = TrainConfig(
        batch_size=32, grad_accum_steps=1, total_steps=800, warmup_steps=50, peak_lr=1e-3
    )
    optimizer = nnx.Optimizer(model, build_optimizer_tx(train_config), wrt=nnx.Param)

    rng = np.random.default_rng(0)
    for _ in range(train_config.total_steps):
        tokens, _query_positions, answers = generate_batch(
            rng, HOP_COUNT, DISTANCE, ENTITY_VOCAB_SIZE, batch_size=train_config.batch_size
        )
        train_step(model, optimizer, jnp.asarray(tokens), jnp.asarray(answers))

    # A fresh rng stream, disjoint in *draw sequence* from training's, but
    # (per ADR 0008) drawing from the exact same shared entity pool -- this
    # is what "eval" now means. If disjoint pools were accidentally
    # reintroduced, these entities would still sit at random init and
    # predictions would collapse to a small fixed set regardless of input,
    # same as the original bug.
    n_examples = 64
    eval_tokens, eval_query_positions, eval_answers = generate_batch(
        np.random.default_rng(999), HOP_COUNT, DISTANCE, ENTITY_VOCAB_SIZE, batch_size=n_examples
    )
    logits = model(jnp.asarray(eval_tokens))
    query_logits = logits[jnp.arange(n_examples), jnp.asarray(eval_query_positions)]
    predicted = jnp.argmax(query_logits, axis=-1)

    accuracy = float(jnp.mean(predicted == jnp.asarray(eval_answers)))
    assert accuracy > 0.0, "accuracy should move off zero after real training on the shared pool"

    unique_predictions = len(set(np.asarray(predicted).tolist()))
    # The original bug's signature: argmax collapsing to a small fixed set
    # of trained-vocab tokens no matter what the input chain actually was.
    # Real context-dependent prediction should spread across many distinct
    # answer tokens given 64 examples with independently random chains.
    assert unique_predictions > 1, (
        f"predicted only {unique_predictions} distinct token(s) across {n_examples} "
        "independently-sampled examples -- this is the input-invariance signature "
        "of the train/eval-vocab memorization bug (ADR 0008), not genuine "
        "context-dependent chain resolution"
    )
