"""Short sanity run: full 125M-param baseline, on-the-fly mixed-cell training.

Not a real experimental run (see TrainConfig below for the reduced step
budget) -- this exists to confirm the training loop actually decreases loss
and moves the eval grid off its untrained baseline on real hardware (the
DGX Spark / GB10), after the answer-supervision and causal-mask-NaN fixes.
"""

from pathlib import Path

import numpy as np
from flax import nnx

from multihop.data.generator import total_vocab_size
from multihop.eval import evaluate_grid
from multihop.models.baseline import FullAttentionBaseline
from multihop.models.config import ModelConfig
from multihop.train import TrainConfig, build_optimizer_tx, train

ENTITY_VOCAB_SIZE = 8_000  # ADR 0002

if __name__ == "__main__":
    # vocab_size is derived from ENTITY_VOCAB_SIZE rather than left at
    # ModelConfig's default -- train()/evaluate_grid now raise if the two
    # drift apart, so this can't silently fall out of sync again.
    model_config = ModelConfig(vocab_size=total_vocab_size(ENTITY_VOCAB_SIZE))
    model = FullAttentionBaseline(model_config, rngs=nnx.Rngs(0))

    train_config = TrainConfig(
        total_steps=300,
        warmup_steps=30,
        eval_every=100,
        checkpoint_every=150,
        # batch_size=256 (the TrainConfig default) OOMs on GB10 for this
        # 125M model with no gradient checkpointing -- an isolated grad
        # computation failed to allocate even after backing off from 91GiB
        # down through 31GiB+, and the fused train_step (which doesn't
        # crash outright, since XLA can donate/reuse buffers more
        # aggressively when forward+backward+optimizer-update are one
        # traced program) was observed to silently produce NaN params
        # instead, right at that memory ceiling. batch_size=16 is
        # confirmed finite across 10 steps including the two previously
        # broken shapes. The real experimental run will need an actual
        # memory strategy (gradient accumulation to reach an effective
        # batch of 256, activation checkpointing, or mixed precision) --
        # this is a placeholder to unblock the sanity check, not a fix
        # for the real run's batch size. grad_accum_steps=1 keeps this a
        # plain batch_size=16 step (TrainConfig's new default of 16 would
        # otherwise turn this into 16 micro-batches of size 1) -- gradient
        # accumulation itself is exercised by sanity_run_accum.py instead.
        batch_size=16,
        grad_accum_steps=1,
    )
    optimizer = nnx.Optimizer(model, build_optimizer_tx(train_config), wrt=nnx.Param)

    eval_rng = np.random.default_rng(1)

    def eval_fn(m: FullAttentionBaseline) -> dict[tuple[int, int], float]:
        return dict(evaluate_grid(m, ENTITY_VOCAB_SIZE, eval_rng, n_examples_per_cell=32))

    checkpoint_dir = Path(__file__).parent.parent / "checkpoints" / "sanity_run"
    history = train(
        model,
        optimizer,
        train_config,
        ENTITY_VOCAB_SIZE,
        np.random.default_rng(0),
        checkpoint_dir,
        eval_fn=eval_fn,
    )

    print("=== final loss history (last 10) ===")
    for step, loss in history.loss[-10:]:
        print(step, loss)

    print("=== eval grids ===")
    for step, grid in history.eval_grids:
        mean_acc = sum(grid.values()) / len(grid)
        print(f"step {step}: mean_accuracy={mean_acc:.4f}")
        print(grid)
