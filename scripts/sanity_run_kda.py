"""Short sanity run for the pure-KDA variant at TrainConfig's real effective batch_size=256,
via gradient accumulation. Mirrors scripts/sanity_run_accum.py's role for the baseline:
confirms no OOM/NaN on real hardware (the DGX Spark / GB10) at production batch size. Not a
real experimental run -- see the reduced step budget below, and the pure-KDA task's "don't
attempt a real training run at full step count".

Runs at TrainConfig's shared defaults (batch_size=256, grad_accum_steps=16),
identical to the baseline's. Two GB10 memory failures had to be fixed to get
there -- an XLA compile blowup from a Python-unrolled chunk loop (ADR 0009) and
a second one from train_step_accum's Python-unrolled micro-batch loop
(ADR 0010) -- both resolved by scanning rather than unrolling, so no
KDA-specific hyperparameter divergence was needed.
"""

from pathlib import Path

import numpy as np
from flax import nnx

from multihop.data.generator import total_vocab_size
from multihop.eval import EVAL_SEED, evaluate_grid
from multihop.models.config import ModelConfig
from multihop.models.pure_kda import PureKDAModel
from multihop.train import TrainConfig, build_optimizer_tx, train

ENTITY_VOCAB_SIZE = 8_000  # ADR 0008

if __name__ == "__main__":
    model_config = ModelConfig(
        vocab_size=total_vocab_size(ENTITY_VOCAB_SIZE), positional_encoding="none"  # ADR 0001
    )
    model = PureKDAModel(model_config, rngs=nnx.Rngs(0))

    train_config = TrainConfig(
        total_steps=20,
        warmup_steps=2,
        eval_every=10,
        checkpoint_every=20,
        # grad_accum_steps stays at TrainConfig's default 16 -- see ADR 0010 for
        # why raising it was tried, measured worse, and rejected.
    )
    optimizer = nnx.Optimizer(model, build_optimizer_tx(train_config), wrt=nnx.Param)

    eval_rng = np.random.default_rng(EVAL_SEED)

    def eval_fn(m: PureKDAModel) -> dict[tuple[int, int], float]:
        return dict(evaluate_grid(m, ENTITY_VOCAB_SIZE, eval_rng, n_examples_per_cell=32))

    checkpoint_dir = Path(__file__).parent.parent / "checkpoints" / "sanity_run_kda"
    history = train(
        model,
        optimizer,
        train_config,
        ENTITY_VOCAB_SIZE,
        np.random.default_rng(0),
        checkpoint_dir,
        eval_fn=eval_fn,
    )

    print("=== loss history ===")
    for step, loss in history.loss:
        print(step, loss)

    print("=== eval grids ===")
    for step, grid in history.eval_grids:
        mean_acc = sum(grid.values()) / len(grid)
        print(f"step {step}: mean_accuracy={mean_acc:.4f}")
        print(grid)
