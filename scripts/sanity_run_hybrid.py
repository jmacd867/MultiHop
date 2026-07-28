"""Short sanity run for the 3:1 hybrid variant at TrainConfig's real effective batch_size=256.

Mirrors scripts/sanity_run_kda.py exactly, including its step budget and eval
cadence, so the two variants' sanity runs are directly comparable. Confirms no
OOM/NaN on real hardware (the DGX Spark / GB10) at production batch size, and
nothing more -- this establishes nothing about task performance, and the eval
grids it prints are not meaningful signal at 20 optimizer steps.

Runs at TrainConfig's shared defaults (batch_size=256, grad_accum_steps=16),
unchanged. Per ADR 0010, micro_batch_size is not a per-variant knob: diverging
from the baseline's batch/LR conditions would erode exactly the cross-variant
comparability the Degradation Grid comparison depends on. If this run does NOT
fit at 256/16, that is a finding requiring its own ADR, not something to work
around here by editing the numbers -- the hybrid has 9 KDA layers against
pure-KDA's 12, so it should be strictly cheaper to compile than the pure-KDA
run that already fit.
"""

from pathlib import Path

import numpy as np
from flax import nnx

from multihop.data.generator import total_vocab_size
from multihop.eval import EVAL_SEED, evaluate_grid
from multihop.models.config import ModelConfig
from multihop.models.hybrid import HybridModel, is_full_attention_layer
from multihop.train import TrainConfig, build_optimizer_tx, train

ENTITY_VOCAB_SIZE = 8_000  # ADR 0008

if __name__ == "__main__":
    model_config = ModelConfig(
        vocab_size=total_vocab_size(ENTITY_VOCAB_SIZE), positional_encoding="none"  # ADR 0001
    )
    model = HybridModel(model_config, rngs=nnx.Rngs(0))

    full_attention_layers = [
        i for i in range(model_config.n_layers) if is_full_attention_layer(i)
    ]
    print(f"hybrid schedule (ADR 0011): full-attention at {full_attention_layers}, KDA elsewhere")

    train_config = TrainConfig(
        total_steps=20,
        warmup_steps=2,
        eval_every=10,
        checkpoint_every=20,
        # batch_size/grad_accum_steps stay at TrainConfig's defaults -- see ADR 0010.
    )
    optimizer = nnx.Optimizer(model, build_optimizer_tx(train_config), wrt=nnx.Param)

    eval_rng = np.random.default_rng(EVAL_SEED)

    def eval_fn(m: HybridModel) -> dict[tuple[int, int], float]:
        return dict(evaluate_grid(m, ENTITY_VOCAB_SIZE, eval_rng, n_examples_per_cell=32))

    checkpoint_dir = Path(__file__).parent.parent / "checkpoints" / "sanity_run_hybrid"
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
