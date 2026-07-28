"""Real experimental run: full-attention baseline, TrainConfig defaults (20,000 steps,
effective batch_size=256 via grad_accum_steps=16), producing one of the three
degradation grids (full-attention -- the reference/control) for the hop_count x
distance study. KDA and hybrid variants are separate, later runs -- see ADR 0001/0003.

See ADR 0006 for why gradient accumulation is required at this batch size on
GB10, and scripts/timing_run.py / timing_run.log for the empirical wall-clock
budget this was expected to take (~6.3h steady-state + compile overhead for
train_step_accum alone; evaluate_grid's 512-example x 25-cell eval passes,
run every eval_every=500 steps, are additional and not included in that
estimate).
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

    train_config = TrainConfig()  # real defaults: batch_size=256, grad_accum_steps=16, total_steps=20_000
    optimizer = nnx.Optimizer(model, build_optimizer_tx(train_config), wrt=nnx.Param)

    eval_rng = np.random.default_rng(1)

    def eval_fn(m: FullAttentionBaseline) -> dict[tuple[int, int], float]:
        return dict(evaluate_grid(m, ENTITY_VOCAB_SIZE, eval_rng))

    checkpoint_dir = Path(__file__).parent.parent / "checkpoints" / "baseline_run"
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
