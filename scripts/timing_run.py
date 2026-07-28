"""Empirical wall-clock budget for any Variant at the real effective batch_size=256.

Produces the two numbers a 20,000-step run's budget actually needs, per
Variant: steady-state seconds per optimizer step, and seconds per full
`evaluate_grid` pass. Both are measured, not extrapolated, via
`jax.block_until_ready` -- JAX dispatch is async, so a naive
`time.perf_counter()` around the call would time Python-side dispatch
rather than the device computation.

Usage: `python scripts/timing_run.py {baseline,kda,hybrid}`.

## Why a deterministic sweep, not the original random draw

This script first measured the baseline by sampling cells at random for
150 steps (`docs/runs/baseline_run_20260727/timing_run.csv`), then
averaging the steady-state steps. That average is biased: `train()` samples
the 25 Grid Cells *uniformly*, but a 150-step random window does not hit
them uniformly (the baseline window ranged from 1 to 10 hits per cell), and
per-step time varies ~18x across cells (0.20s at seq_len=5 to 3.76s at
seq_len=287). The raw mean of that window (1.0781s, the "5.99h" the script
originally printed) therefore reflects which cells happened to come up, not
the distribution the real run draws from.

The fix is to weight the 25 per-cell means equally rather than weighting by
observed frequency. A deterministic sweep -- every cell hit
`STEPS_PER_CELL` times -- measures exactly that quantity directly, at a
third of the steps, with no cell left at n=1. It is also *retroactively*
comparable: the committed baseline CSV records per-cell times, so the same
uniform-weighted estimate can be recomputed from it (1.0231s -> 5.68h)
without re-running the baseline.

Note that 25 cells share only 17 distinct sequence lengths (e.g. hop_count=2/
distance=45 and hop_count=5/distance=21 are both 287//2+... = 143 tokens), so
a cell whose shape a previous cell already compiled costs no recompile. The
`compiled` column tracks first occurrence of a *shape*, not of a cell, and
compiled steps are excluded from the steady-state statistics and reported
separately as the run's one-time overhead.

## Why the eval grid is timed here at all

No budget in this project has ever included `evaluate_grid`, but the real
runs call it every `eval_every=500` steps -- 40 times over 20,000 steps, at
`EVAL_EXAMPLES_PER_CELL=512` across all 25 cells. That is a real line item,
and for the KDA-containing Variants it is also an unmeasured *memory* risk:
they have only ever been evaluated at 32 examples/cell (the sanity runs),
so a 512-example forward at seq_len=287 through 12 KDA layers is untested.
Surfacing an OOM here costs minutes; surfacing it at step 500 of a 20,000-
step run costs hours.

The grid is timed twice. The first pass includes one-time XLA compilation
of the 25 forward shapes; the second is the steady-state cost the remaining
39 passes actually pay. Eval cells are walked longest-sequence-first so a
memory failure surfaces on the first cell rather than the last, and the
per-cell loop here calls `accuracy_at_cell` directly rather than
`evaluate_grid` so each cell can be timed and flushed individually. That
reordering changes which examples each cell draws from the shared rng, which
is irrelevant here -- this measures wall clock and memory on an untrained
model, and is not a scored evaluation.
"""

import csv
import sys
import time
from pathlib import Path
from typing import Protocol

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from multihop.data.generator import (
    DISTANCES,
    MAX_HOP_COUNT,
    MIN_HOP_COUNT,
    generate_batch,
    required_sequence_length,
    total_vocab_size,
)
from multihop.eval import EVAL_EXAMPLES_PER_CELL, EVAL_SEED, accuracy_at_cell
from multihop.models.baseline import FullAttentionBaseline
from multihop.models.config import ModelConfig, PositionalEncoding
from multihop.models.hybrid import HybridModel
from multihop.models.pure_kda import PureKDAModel
from multihop.train import MultihopModel, TrainConfig, build_optimizer_tx, train_step_accum

ENTITY_VOCAB_SIZE = 8_000  # ADR 0008
STEPS_PER_CELL = 4
REAL_TOTAL_STEPS = 20_000


class ModelFactory(Protocol):
    """The constructor signature all three Variant classes share."""

    def __call__(self, config: ModelConfig, *, rngs: nnx.Rngs) -> MultihopModel: ...


# Per ADR 0001, positional encoding is a per-variant rule, not a per-layer-type
# one: the baseline is the only Variant with no KDA layer to carry positional
# information through gating, so it is the only one that keeps RoPE.
VARIANTS: dict[str, tuple[ModelFactory, PositionalEncoding]] = {
    "baseline": (FullAttentionBaseline, "rope"),
    "kda": (PureKDAModel, "none"),
    "hybrid": (HybridModel, "none"),
}

ALL_CELLS = [
    (hop_count, distance)
    for hop_count in range(MIN_HOP_COUNT, MAX_HOP_COUNT + 1)
    for distance in DISTANCES
]


def system_memory_used_gb() -> float:
    """System-wide memory in use, in GB -- the figure the shared-box ceiling is about.

    Deliberately *not* this process's RSS. GB10's memory is unified, and JAX's
    device arena does not appear in `/proc/self/status`: measured mid-run here,
    VmHWM read 5.4GB while the box was actually 100GB deep, ~91.5GB of which
    `nvidia-smi` attributed to this PID's preallocated arena. Reporting RSS
    would understate the load ~17x and would have made the 118-121GB kill zone
    of ADR 0009 and ADR 0010 look like idle.

    System-wide (MemTotal - MemAvailable) is what those two ADRs were actually
    tracking. On a shared box it also correctly includes anyone else's load,
    which is the right conservatism: the ceiling is the box's, not this run's.
    """
    meminfo: dict[str, int] = {}
    with open("/proc/meminfo") as f:
        for line in f:
            key, _, rest = line.partition(":")
            meminfo[key] = int(rest.split()[0])
    used_kb = meminfo["MemTotal"] - meminfo["MemAvailable"]
    return used_kb / (1024 * 1024)


def build_microbatches_at_cell(
    rng: np.random.Generator, hop_count: int, distance: int, config: TrainConfig
) -> list[tuple[jnp.ndarray, jnp.ndarray]]:
    """`sample_training_microbatches` for an explicitly chosen cell.

    That function picks the cell itself (uniformly), which is what `train()`
    wants and what this deterministic sweep specifically does not. The
    micro-batch construction is otherwise identical, including its guarantee
    that every micro-batch in a step shares one shape -- here that holds by
    construction, since `required_sequence_length` is a function of
    (hop_count, distance) alone.
    """
    micro_batches = []
    for _ in range(config.grad_accum_steps):
        tokens, _query_positions, answers = generate_batch(
            rng, hop_count, distance, ENTITY_VOCAB_SIZE, config.micro_batch_size
        )
        micro_batches.append((jnp.asarray(tokens), jnp.asarray(answers)))
    return micro_batches


def main(variant: str) -> None:
    model_cls, positional_encoding = VARIANTS[variant]
    model_config = ModelConfig(
        vocab_size=total_vocab_size(ENTITY_VOCAB_SIZE),
        positional_encoding=positional_encoding,
    )
    model = model_cls(model_config, rngs=nnx.Rngs(0))

    param_count = sum(int(np.asarray(p).size) for p in jax.tree.leaves(nnx.state(model, nnx.Param)))
    print(f"variant={variant} positional_encoding={positional_encoding} params={param_count:,}", flush=True)

    train_config = TrainConfig()  # real defaults: batch_size=256, grad_accum_steps=16
    optimizer = nnx.Optimizer(model, build_optimizer_tx(train_config), wrt=nnx.Param)

    rng = np.random.default_rng(0)
    seen_shapes: set[int] = set()

    repo_root = Path(__file__).parent.parent
    csv_path = repo_root / f"timing_run_{variant}.csv"
    # Flushed every step: a full sweep is tens of minutes of compiles, and
    # this file is the only way to recover partial results if the process is
    # killed (which is how two earlier GB10 runs ended -- ADR 0009, ADR 0010).
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(
        ["phase", "hop_count", "distance", "seq_len", "compiled", "elapsed_s", "loss", "sys_mem_gb"]
    )
    csv_file.flush()

    per_cell_steady: dict[tuple[int, int], list[float]] = {}
    compile_seconds = 0.0
    num_compiled = 0

    print("\n=== phase 1: train_step_accum sweep ===", flush=True)
    for hop_count, distance in ALL_CELLS:
        seq_len = required_sequence_length(hop_count, distance)
        for _ in range(STEPS_PER_CELL):
            micro_batches = build_microbatches_at_cell(rng, hop_count, distance, train_config)
            compiled = seq_len not in seen_shapes
            seen_shapes.add(seq_len)

            t0 = time.perf_counter()
            loss = train_step_accum(model, optimizer, micro_batches)
            loss.block_until_ready()
            elapsed = time.perf_counter() - t0
            loss_value = float(loss)
            peak_gb = system_memory_used_gb()

            if compiled:
                compile_seconds += elapsed
                num_compiled += 1
            else:
                per_cell_steady.setdefault((hop_count, distance), []).append(elapsed)

            csv_writer.writerow(
                ["train", hop_count, distance, seq_len, compiled, elapsed, loss_value, peak_gb]
            )
            csv_file.flush()
            print(
                f"train hop_count={hop_count} distance={distance} seq_len={seq_len} "
                f"compiled={compiled} elapsed={elapsed:.4f}s loss={loss_value:.4f} "
                f"sys_mem={peak_gb:.1f}GB",
                flush=True,
            )

    print("\n=== phase 2: evaluate_grid, 512 examples/cell, longest sequence first ===", flush=True)
    eval_cells = sorted(ALL_CELLS, key=lambda c: required_sequence_length(*c), reverse=True)
    grid_seconds: list[float] = []
    for pass_index in (1, 2):
        eval_rng = np.random.default_rng(EVAL_SEED)
        total = 0.0
        for hop_count, distance in eval_cells:
            seq_len = required_sequence_length(hop_count, distance)
            t0 = time.perf_counter()
            accuracy = accuracy_at_cell(
                model, hop_count, distance, ENTITY_VOCAB_SIZE, EVAL_EXAMPLES_PER_CELL, eval_rng
            )
            elapsed = time.perf_counter() - t0
            total += elapsed
            peak_gb = system_memory_used_gb()
            csv_writer.writerow(
                [f"eval{pass_index}", hop_count, distance, seq_len, "", elapsed, accuracy, peak_gb]
            )
            csv_file.flush()
            print(
                f"eval{pass_index} hop_count={hop_count} distance={distance} seq_len={seq_len} "
                f"elapsed={elapsed:.4f}s accuracy={accuracy:.4f} sys_mem={peak_gb:.1f}GB",
                flush=True,
            )
        grid_seconds.append(total)
        print(f"eval pass {pass_index} total: {total:.1f}s", flush=True)

    csv_file.close()
    print(f"\nwrote per-step log to {csv_path}", flush=True)

    print("\n=== per-cell steady-state (compiled steps excluded) ===")
    for cell in ALL_CELLS:
        times = per_cell_steady.get(cell, [])
        seq_len = required_sequence_length(*cell)
        if times:
            print(f"{cell}: seq_len={seq_len} mean={np.mean(times):.4f}s n={len(times)}")
        else:
            print(f"{cell}: seq_len={seq_len} NO STEADY-STATE SAMPLES")

    missing = [c for c in ALL_CELLS if not per_cell_steady.get(c)]
    if missing:
        print(f"\nWARNING: {len(missing)} cells have no steady-state sample: {missing}")
        return

    # Equal weight per cell, because train() samples the 25 cells uniformly --
    # see this module's docstring on why a frequency-weighted mean is biased.
    uniform_mean_step = float(np.mean([float(np.mean(per_cell_steady[c])) for c in ALL_CELLS]))
    train_hours = uniform_mean_step * REAL_TOTAL_STEPS / 3600
    num_eval_passes = REAL_TOTAL_STEPS // TrainConfig().eval_every
    first_grid, steady_grid = grid_seconds
    eval_hours = (first_grid + steady_grid * (num_eval_passes - 1)) / 3600
    compile_hours = compile_seconds / 3600

    print(f"\n=== {variant}: {REAL_TOTAL_STEPS}-step budget ===")
    print(f"uniform-cell-weighted per-step: {uniform_mean_step:.4f}s")
    print(f"train steady-state:      {train_hours:6.2f}h")
    print(f"train compile ({num_compiled} shapes): {compile_hours:6.2f}h ({compile_seconds:.0f}s, one-time)")
    print(f"eval ({num_eval_passes} grids):        {eval_hours:6.2f}h "
          f"(first {first_grid:.0f}s incl. compile, steady {steady_grid:.0f}s each)")
    print(f"TOTAL:                   {train_hours + compile_hours + eval_hours:6.2f}h "
          "(excludes checkpoint writes, ~20 x a few seconds)")
    print(f"peak system memory: {system_memory_used_gb():.1f}GB")


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in VARIANTS:
        raise SystemExit(f"usage: {sys.argv[0]} {{{','.join(VARIANTS)}}}")
    main(sys.argv[1])
