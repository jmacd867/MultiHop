"""Short sanity run at TrainConfig's real effective batch_size=256, via gradient accumulation.

Confirms ADR 0006's resolution (grad_accum_steps=16, micro_batch_size=16)
avoids the OOM / silent-NaN failure that batch_size=256 hit without
accumulation, on real hardware (the DGX Spark / GB10). Not a real
experimental run -- see the reduced step budget below.
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
    model_config = ModelConfig(vocab_size=total_vocab_size(ENTITY_VOCAB_SIZE))
    model = FullAttentionBaseline(model_config, rngs=nnx.Rngs(0))

    train_config = TrainConfig(
        total_steps=20,
        warmup_steps=2,
        eval_every=10,
        checkpoint_every=20,
        # TrainConfig's real defaults: batch_size=256, grad_accum_steps=16
        # (micro_batch_size=16, the value ADR 0006 confirmed safe).
    )
    optimizer = nnx.Optimizer(model, build_optimizer_tx(train_config), wrt=nnx.Param)

    eval_rng = np.random.default_rng(1)

    def eval_fn(m: FullAttentionBaseline) -> dict[tuple[int, int], float]:
        return dict(evaluate_grid(m, ENTITY_VOCAB_SIZE, eval_rng, n_examples_per_cell=32))

    checkpoint_dir = Path(__file__).parent.parent / "checkpoints" / "sanity_run_accum"
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
