"""Empirical wall-clock timing for train_step_accum at the real effective batch_size=256.

Runs a short window of real optimizer steps (TrainConfig's default
batch_size=256 / grad_accum_steps=16) and records, per step: the
(hop_count, distance) cell sampled, whether that step's sequence-length
shape had been seen before in this run (a first occurrence forces an XLA
recompile of train_step_accum, since JIT traces per input shape -- see
sample_training_microbatches' one-cell-per-step design in train.py), and
wall-clock elapsed time (via jax.block_until_ready, since JAX dispatch is
async and a naive time.perf_counter() around the call would only measure
Python-side dispatch, not the actual device computation).

This is diagnostic, not a real training run: it exists to produce a
20,000-step wall-clock budget estimate for the actual experimental run, and
to make the cell-sampling distribution actually exercised during the
timing window inspectable -- with only ~150 steps across 25 grid cells
(~6 hits/cell on average), a skewed draw could bias the average step time
toward whichever cells happened to come up more, so the per-cell hit counts
and per-step (cell, elapsed) log are printed and saved precisely so that can
be checked before trusting the resulting estimate.
"""

import csv
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from flax import nnx

from multihop.data.generator import DISTANCES, MAX_HOP_COUNT, MIN_HOP_COUNT, total_vocab_size
from multihop.models.baseline import FullAttentionBaseline
from multihop.models.config import ModelConfig
from multihop.train import TrainConfig, build_optimizer_tx, sample_training_microbatches, train_step_accum

ENTITY_VOCAB_SIZE = 8_000  # ADR 0002
NUM_TIMING_STEPS = 150
REAL_TOTAL_STEPS = 20_000

if __name__ == "__main__":
    model_config = ModelConfig(vocab_size=total_vocab_size(ENTITY_VOCAB_SIZE))
    model = FullAttentionBaseline(model_config, rngs=nnx.Rngs(0))

    train_config = TrainConfig()  # real defaults: batch_size=256, grad_accum_steps=16
    optimizer = nnx.Optimizer(model, build_optimizer_tx(train_config), wrt=nnx.Param)

    rng = np.random.default_rng(0)
    seen_shapes: set[int] = set()

    steps: list[int] = []
    hop_counts: list[int] = []
    distances: list[int] = []
    seq_lens: list[int] = []
    compiled_flags: list[bool] = []
    elapsed_times: list[float] = []
    losses: list[float] = []

    # Written incrementally (flushed every step), not just at the end -- a
    # 150-step run with ~15-25 one-time ~60-80s compiles can take the better
    # part of an hour, and this file is the only way to recover partial
    # results if the process is interrupted before the final summary prints.
    csv_path = Path(__file__).parent.parent / "timing_run.csv"
    fieldnames = ["step", "hop_count", "distance", "seq_len", "compiled", "elapsed_s", "loss"]
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(fieldnames)
    csv_file.flush()

    for step in range(1, NUM_TIMING_STEPS + 1):
        micro_batches, hop_count, distance = sample_training_microbatches(
            rng, train_config.micro_batch_size, train_config.grad_accum_steps, ENTITY_VOCAB_SIZE
        )
        jnp_micro_batches = [
            (jnp.asarray(tokens), jnp.asarray(answers)) for tokens, answers in micro_batches
        ]
        seq_len = jnp_micro_batches[0][0].shape[1]
        compiled = seq_len not in seen_shapes
        seen_shapes.add(seq_len)

        t0 = time.perf_counter()
        loss = train_step_accum(model, optimizer, jnp_micro_batches)
        loss.block_until_ready()
        elapsed = time.perf_counter() - t0
        loss_value = float(loss)

        steps.append(step)
        hop_counts.append(hop_count)
        distances.append(distance)
        seq_lens.append(seq_len)
        compiled_flags.append(compiled)
        elapsed_times.append(elapsed)
        losses.append(loss_value)
        csv_writer.writerow([step, hop_count, distance, seq_len, compiled, elapsed, loss_value])
        csv_file.flush()
        print(
            f"step {step}: hop_count={hop_count} distance={distance} seq_len={seq_len} "
            f"compiled={compiled} elapsed={elapsed:.4f}s loss={loss:.4f}",
            flush=True,
        )

    csv_file.close()
    print(f"\nwrote per-step log to {csv_path}")

    print("\n=== cell hit counts (hop_count, distance) -> count ===")
    hit_counts: dict[tuple[int, int], int] = {}
    for i in range(len(steps)):
        cell = (hop_counts[i], distances[i])
        hit_counts[cell] = hit_counts.get(cell, 0) + 1
    all_cells = [(h, d) for h in range(MIN_HOP_COUNT, MAX_HOP_COUNT + 1) for d in DISTANCES]
    for cell in all_cells:
        print(f"{cell}: {hit_counts.get(cell, 0)}")
    missing = [cell for cell in all_cells if cell not in hit_counts]
    if missing:
        print(f"\nWARNING: {len(missing)}/{len(all_cells)} cells never sampled in this window: {missing}")

    steady_state = [elapsed_times[i] for i in range(len(steps)) if not compiled_flags[i]]
    num_compiled = sum(1 for flag in compiled_flags if flag)
    print(f"\n=== timing: {num_compiled} compiled steps (excluded), {len(steady_state)} steady-state steps ===")
    if steady_state:
        arr = np.asarray(steady_state, dtype=np.float64)
        print(
            f"steady-state per-step: mean={arr.mean():.4f}s median={np.median(arr):.4f}s "
            f"min={arr.min():.4f}s max={arr.max():.4f}s"
        )
        estimated_total_seconds = float(arr.mean()) * REAL_TOTAL_STEPS
        compile_overhead_seconds = sum(elapsed_times[i] for i in range(len(steps)) if compiled_flags[i])
        print(
            f"\n{REAL_TOTAL_STEPS}-step estimate: {estimated_total_seconds / 3600:.2f}h steady-state "
            f"+ ~{compile_overhead_seconds:.1f}s one-time compile overhead "
            f"({num_compiled} unique shapes seen so far, out of {len(all_cells)} possible)"
        )
    else:
        print("No steady-state steps observed -- every step in this window hit a new shape.")
