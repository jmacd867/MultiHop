"""Full-scale single-cell learning probe for the hybrid variant: does accuracy leave zero?

The 20-step sanity run (scripts/sanity_run_hybrid.py) establishes no-OOM/no-NaN
and nothing else. This probe answers the one further question worth spending
GB10 time on before the real runs: at full model scale, does the hybrid
actually learn anything at all?

**Why a single cell rather than the mixed grid.** `train()` samples uniformly
over all 25 Grid Cells, so a few-thousand-step budget would give the easiest
cell only ~1/25 of the steps while paying 25 distinct XLA shape compiles
(~65s each, measured -- docs/runs/baseline_run_20260727/timing_run.csv). Fixing
the cell at hop_count=1, distance=0 makes this one compile and ~5-token
sequences, turning hours into minutes. This is the same single-easiest-cell
design tests/test_binding_generalization.py already uses, scaled up from that
test's tiny 4-layer/32-dim stand-in to the real 131M-param model.

**Why this LR schedule.** `build_optimizer_tx` derives the cosine schedule from
`total_steps`, so the probe gets a complete warmup-peak-decay cycle at the real
`peak_lr` (3e-4, unchanged), with warmup scaled to ~5% of the budget rather
than TrainConfig's default 500 steps -- which would otherwise spend a sixth of
the probe still warming up. A null result then means "the hybrid did not learn",
not "the LR never got anywhere".

**Reading the result -- see ADR 0012 before drawing any conclusion.** The bar is
binary and deliberately low: accuracy > 0, with predictions spread over more
than one distinct token. Passing means only that the composed stack trains --
gradients flow, the shared harness drives it, loss falls.

It does **not** mean the model retrieves anything. ADR 0012 established that
distance=0 is a degenerate cell: the traversal-free "output the entity
appearing exactly once" shortcut scores **100%** there, and at this cell's
five-token sequence (`A B | ? A`) the even cheaper "copy the second token" also
scores 100% with no binding and no attention to the Query. A perfect score here
is what a working optimizer looks like, not a working retrieval mechanism.
Treat this script as a smoke test of the forward/backward path only.

Failing does NOT by itself mean the hybrid is broken -- when this was written no
Variant had cleared the bar at full scale, so there was no reference for how
many steps should suffice. If this probe returns zero, run the identical probe
on the full-attention baseline before concluding anything about the hybrid.
"""

import time

import jax.numpy as jnp
import numpy as np
from flax import nnx

from multihop.data.generator import generate_batch, total_vocab_size
from multihop.models.config import ModelConfig
from multihop.models.hybrid import HybridModel
from multihop.train import TrainConfig, build_optimizer_tx, train_step_accum

ENTITY_VOCAB_SIZE = 8_000  # ADR 0008
HOP_COUNT = 1
DISTANCE = 0
PROBE_STEPS = 3_000
WARMUP_STEPS = 150  # ~5% of PROBE_STEPS
EVAL_EVERY = 250
N_EVAL_EXAMPLES = 512


def evaluate_single_cell(
    model: HybridModel, rng: np.random.Generator
) -> tuple[float, int]:
    """Accuracy and distinct-prediction count at the probe's fixed cell."""
    tokens, query_positions, answers = generate_batch(
        rng, HOP_COUNT, DISTANCE, ENTITY_VOCAB_SIZE, N_EVAL_EXAMPLES
    )
    logits = model(jnp.asarray(tokens))
    query_logits = logits[jnp.arange(N_EVAL_EXAMPLES), jnp.asarray(query_positions)]
    predicted = jnp.argmax(query_logits, axis=-1)
    accuracy = float(jnp.mean(predicted == jnp.asarray(answers)))
    return accuracy, len(set(np.asarray(predicted).tolist()))


if __name__ == "__main__":
    model_config = ModelConfig(
        vocab_size=total_vocab_size(ENTITY_VOCAB_SIZE), positional_encoding="none"  # ADR 0001
    )
    model = HybridModel(model_config, rngs=nnx.Rngs(0))

    train_config = TrainConfig(
        total_steps=PROBE_STEPS,
        warmup_steps=WARMUP_STEPS,
        # batch_size=256 / grad_accum_steps=16 stay at the shared defaults (ADR 0010).
    )
    optimizer = nnx.Optimizer(model, build_optimizer_tx(train_config), wrt=nnx.Param)

    rng = np.random.default_rng(0)
    start = time.perf_counter()

    for step in range(1, PROBE_STEPS + 1):
        # Fixed cell, so every micro-batch across every step is identically
        # shaped -- train_step_accum's @nnx.jit compiles exactly once.
        micro_batches = []
        for _ in range(train_config.grad_accum_steps):
            tokens, _query_positions, answers = generate_batch(
                rng, HOP_COUNT, DISTANCE, ENTITY_VOCAB_SIZE, train_config.micro_batch_size
            )
            micro_batches.append((jnp.asarray(tokens), jnp.asarray(answers)))

        loss = train_step_accum(model, optimizer, micro_batches)

        if step % 50 == 0 or step == 1:
            print(f"step {step}: loss={float(loss):.4f} elapsed={time.perf_counter() - start:.1f}s")

        if step % EVAL_EVERY == 0:
            accuracy, distinct = evaluate_single_cell(model, np.random.default_rng(999))
            print(
                f"step {step}: PROBE accuracy={accuracy:.4f} "
                f"distinct_predictions={distinct}/{N_EVAL_EXAMPLES}"
            )

    accuracy, distinct = evaluate_single_cell(model, np.random.default_rng(999))
    elapsed = time.perf_counter() - start
    print(f"=== probe complete in {elapsed / 60:.1f} min ===")
    print(f"final accuracy at (hop_count={HOP_COUNT}, distance={DISTANCE}): {accuracy:.4f}")
    print(f"distinct predicted tokens: {distinct}/{N_EVAL_EXAMPLES}")
    print(
        "VERDICT: "
        + (
            "PASS -- the composed stack trains (forward/backward path works). "
            "NOT evidence of retrieval or binding: distance=0 is degenerate, "
            "see ADR 0012"
            if accuracy > 0.0 and distinct > 1
            else "NULL -- see this script's docstring; run the same probe on the "
            "baseline before concluding the hybrid is at fault"
        )
    )
