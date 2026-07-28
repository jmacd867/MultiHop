"""Real experimental run: one Variant, 20,000 steps, producing one Degradation Grid.

Usage: `python scripts/train_variant.py {baseline,kda,hybrid}`.

Supersedes `scripts/train_baseline.py`, which hardcoded the full-attention
Variant. One parameterized script rather than three near-duplicates is the
same choice made for `scripts/timing_run.py`, and for the same reason it
matters more here than as a tidiness preference: the three Degradation Grids
are only comparable if the three runs differ in *nothing* but the Variant
under study (CONTEXT.md's definition of Variant: "Anything else that differs
between Variants is a confound, not a Variant difference"). Three separate
scripts make an accidental divergence in step count, seed, eval cadence, or
batch settings invisible; one script makes it impossible.

## Everything below is shared across all three Variants, deliberately

`TrainConfig()` defaults, unmodified: 20,000 steps, effective
`batch_size=256` via `grad_accum_steps=16`, `peak_lr=3e-4`, warmup 500,
`eval_every=500`, `checkpoint_every=1000`.

ADR 0010 is the reason `grad_accum_steps` is 16 here for *every* Variant,
including pure-KDA. Raising it to 32 (micro_batch_size=8) is a documented
FAILURE, not a recommendation: compile-time memory scales with
`grad_accum_steps` while only runtime activation memory scales with
`micro_batch_size`, so raising it consumed ~121GiB of the box's ~124GiB and
had to be killed by hand. The actual fix was making `train_step_accum`'s
micro-batch loop a `lax.scan`, which let all three Variants stay at 256/16.
Do not "fix" this file by lowering micro_batch_size for KDA.

Data ordering is identical across Variants too: `np.random.default_rng(0)`
drives cell sampling and example generation, so all three see the same
sequence of (hop_count, distance) cells. Eval uses `EVAL_SEED` (ADR 0008),
so all three are scored on identical freshly-generated examples per cell.

Positional encoding is the one thing that legitimately varies, per ADR 0001:
the baseline is the only Variant with no KDA layer to carry positional
information through gating, so it alone keeps RoPE.

## What this script adds over its predecessors

**Timestamps.** No prior run script in this project timestamped its steps,
which is why `docs/runs/hybrid_sanity_run_20260728/README.md` had to record
"No comparison against the pure-KDA run is available... both scripts need
timestamps first." Per-step elapsed time is logged here so that comparison
is possible and so a stalled run is distinguishable from a slow one.

**Persisted grids.** Every eval grid is written to
`runs/<variant>_grids.json` as it is produced, not left only in stdout. The
cross-variant comparison reads those files rather than re-parsing logs, and
an interrupted run still leaves every grid it completed.
"""

import json
import sys
import time
from pathlib import Path
from typing import Protocol

import numpy as np
from flax import nnx

from multihop.data.generator import total_vocab_size
from multihop.eval import EVAL_SEED, evaluate_grid
from multihop.models.baseline import FullAttentionBaseline
from multihop.models.config import ModelConfig, PositionalEncoding
from multihop.models.hybrid import HybridModel, is_full_attention_layer
from multihop.models.pure_kda import PureKDAModel
from multihop.train import (
    MultihopModel,
    TrainConfig,
    build_optimizer_tx,
    latest_checkpoint,
    load_checkpoint,
    restore_rngs,
    train,
)

ENTITY_VOCAB_SIZE = 8_000  # ADR 0008


class ModelFactory(Protocol):
    """The constructor signature all three Variant classes share."""

    def __call__(self, config: ModelConfig, *, rngs: nnx.Rngs) -> MultihopModel: ...


VARIANTS: dict[str, tuple[ModelFactory, PositionalEncoding]] = {
    "baseline": (FullAttentionBaseline, "rope"),  # ADR 0001: only Variant keeping RoPE
    "kda": (PureKDAModel, "none"),
    "hybrid": (HybridModel, "none"),
}


def main(variant: str) -> None:
    model_cls, positional_encoding = VARIANTS[variant]
    model_config = ModelConfig(
        vocab_size=total_vocab_size(ENTITY_VOCAB_SIZE),
        positional_encoding=positional_encoding,
    )
    model = model_cls(model_config, rngs=nnx.Rngs(0))

    if variant == "hybrid":
        schedule = [i for i in range(model_config.n_layers) if is_full_attention_layer(i)]
        print(f"hybrid schedule (ADR 0011): full-attention at {schedule}, KDA elsewhere", flush=True)

    train_config = TrainConfig()  # real defaults: 20,000 steps, batch_size=256, grad_accum_steps=16
    print(
        f"variant={variant} positional_encoding={positional_encoding} "
        f"total_steps={train_config.total_steps} batch_size={train_config.batch_size} "
        f"grad_accum_steps={train_config.grad_accum_steps} "
        f"micro_batch_size={train_config.micro_batch_size}",
        flush=True,
    )

    optimizer = nnx.Optimizer(model, build_optimizer_tx(train_config), wrt=nnx.Param)

    repo_root = Path(__file__).parent.parent
    runs_dir = repo_root / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    grids_path = runs_dir / f"{variant}_grids.json"
    grids: list[dict[str, object]] = []

    train_rng = np.random.default_rng(0)
    eval_rng = np.random.default_rng(EVAL_SEED)
    # Both generators are checkpointed and both must be restored together:
    # the training one drives the Grid Cell sequence, and the eval one is
    # threaded across all 40 eval passes, so its position determines which
    # examples the *final* Degradation Grid is scored on. See save_checkpoint.
    rngs = {"train": train_rng, "eval": eval_rng}

    checkpoint_dir = repo_root / "checkpoints" / f"{variant}_run"
    start_step = 0
    existing = latest_checkpoint(checkpoint_dir)
    if existing is not None:
        start_step = load_checkpoint(model, optimizer, existing)
        if not restore_rngs(existing, rngs):
            raise SystemExit(
                f"{existing} predates RNG-state checkpointing, so resuming from it would "
                "silently give this run a different data stream (and a Degradation Grid "
                "scored on different examples) than the other Variants. Delete "
                f"{checkpoint_dir} to start this Variant over."
            )
        # Reload the grids already produced, so an interrupted run's JSON keeps
        # its earlier evals instead of being truncated to the post-resume ones.
        if grids_path.exists():
            previous: list[dict[str, object]] = list(json.loads(grids_path.read_text())["evals"])
            # Drop any grid recorded after the checkpoint being resumed from:
            # those steps are about to be replayed.
            grids = [g for g in previous if int(str(g.get("step", 0))) <= start_step]
        print(f"RESUMING {variant} from {existing.name} ({len(grids)} eval grids kept)", flush=True)

    start = time.perf_counter()
    grids_before_resume = len(grids)

    def eval_fn(m: MultihopModel) -> dict[tuple[int, int], float]:
        grid = dict(evaluate_grid(m, ENTITY_VOCAB_SIZE, eval_rng))
        # Written after every eval rather than once at the end: a 20,000-step
        # run is many hours, and an interrupted one should still leave every
        # grid it managed to produce. JSON keys must be strings, so cells are
        # serialized as "hop_count,distance".
        grids.append(
            {
                "elapsed_s": time.perf_counter() - start,
                "grid": {f"{hop},{dist}": acc for (hop, dist), acc in grid.items()},
            }
        )
        grids_path.write_text(json.dumps({"variant": variant, "evals": grids}, indent=2))
        return grid

    history = train(
        model,
        optimizer,
        train_config,
        ENTITY_VOCAB_SIZE,
        train_rng,
        checkpoint_dir,
        eval_fn=eval_fn,
        start_step=start_step,
        checkpoint_rngs=rngs,
    )

    # `train()` records the step index alongside each eval grid; the eval_fn
    # closure above cannot see it, so attach it here before the final write.
    # Only the grids this invocation produced are zipped -- on a resume,
    # `grids` also holds the earlier run's entries, which already have steps.
    for entry, (step, _grid) in zip(grids[grids_before_resume:], history.eval_grids, strict=True):
        entry["step"] = step
    grids_path.write_text(json.dumps({"variant": variant, "evals": grids}, indent=2))

    total_hours = (time.perf_counter() - start) / 3600
    print(f"\n=== {variant}: complete in {total_hours:.2f}h ===", flush=True)
    print(f"wrote {len(grids)} eval grids to {grids_path}", flush=True)

    print("=== final loss history (last 10) ===")
    for step, loss in history.loss[-10:]:
        print(step, loss)

    print("=== eval grid mean accuracy by step ===")
    for step, grid in history.eval_grids:
        print(f"step {step}: mean_accuracy={sum(grid.values()) / len(grid):.4f}")


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in VARIANTS:
        raise SystemExit(f"usage: {sys.argv[0]} {{{','.join(VARIANTS)}}}")
    main(sys.argv[1])
