"""Attribute each training step's loss to its Grid Cell, and compare Variants per cell.

Usage: `python scripts/loss_by_cell.py [runs_dir]`

## Why this exists

Both Variants saturate the eval grid at 1.0000 accuracy in every cell, so the
Degradation Grid -- the experiment's intended output -- has no signal left to
compare. Training loss does: it is a continuous measure of how well the model
fits the whole sequence, not a thresholded argmax over one position, so it
keeps discriminating long after accuracy stops.

The comparison is **exactly controlled**. `train_variant.py` drives every
Variant from `np.random.default_rng(0)`, and `sample_training_microbatches`
consumes that generator deterministically, so at any given step every Variant
trained on the *same* (hop_count, distance) cell and the *same* generated
examples. A step-matched loss difference is therefore an architecture
difference, with the data held fixed -- no pairing assumptions needed.

The step logs record loss but not which cell produced it. That is recoverable:
the cell sequence is a pure function of the seed, so replaying
`sample_training_microbatches` with a fresh `default_rng(0)` reproduces it
exactly. Replay costs ~2.4 minutes for 20,000 steps on CPU and touches no GPU,
so it can run while training continues.

## Resolution

~800 steps land in each of the 25 cells over a 20,000-step run, against 40 eval
grids total for the accuracy view. That is the whole point: this sees per-cell
structure the eval grid cannot resolve.

## What it does not establish

Loss is not accuracy. A Variant can carry higher loss while still answering
every query correctly -- which is precisely the situation here, and why this
should be reported as "fits the data less tightly", not as "gets more answers
wrong". Both Variants answer every cell correctly.
"""

import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np

from multihop.data.generator import DISTANCES, MAX_HOP_COUNT, MIN_HOP_COUNT
from multihop.train import TrainConfig, sample_training_microbatches

ENTITY_VOCAB_SIZE = 8_000  # ADR 0008
VARIANTS = ("baseline", "kda", "hybrid")
HOP_COUNTS = tuple(range(MIN_HOP_COUNT, MAX_HOP_COUNT + 1))

# Steps before this are the acquisition phase, where Variants differ hugely for
# reasons of learning speed rather than converged fit. Reported separately.
CONVERGED_FROM = 10_000


def replay_cells(total_steps: int) -> dict[int, tuple[int, int]]:
    """Reproduce the (hop_count, distance) cell each step trained on."""
    config = TrainConfig()
    rng = np.random.default_rng(0)
    cells = {}
    for step in range(1, total_steps + 1):
        _micro_batches, hop_count, distance = sample_training_microbatches(
            rng, config.micro_batch_size, config.grad_accum_steps, ENTITY_VOCAB_SIZE
        )
        cells[step] = (hop_count, distance)
    return cells


def parse_losses(path: Path) -> dict[int, float]:
    pattern = re.compile(r"^step (\d+): loss=([0-9.eE+-]+)")
    out: dict[int, float] = {}
    with open(path, errors="ignore") as handle:
        for line in handle:
            match = pattern.match(line)
            if match:
                out[int(match.group(1))] = float(match.group(2))
    return out


def render(grid: dict[tuple[int, int], float], title: str, fmt: str = "{:>10.4f}") -> None:
    print(f"\n{title}")
    header = "hop \\ dist" + "".join(f"{d:>10}" for d in DISTANCES)
    print(header)
    print("-" * len(header))
    for hop in HOP_COUNTS:
        row = ""
        for dist in DISTANCES:
            value = grid.get((hop, dist))
            row += f"{'--':>10}" if value is None else fmt.format(value)
        print(f"{hop:>9} " + row)


def main(runs_dir: Path) -> None:
    losses = {}
    for variant in VARIANTS:
        path = runs_dir / f"{variant}_run.log"
        if not path.exists():
            print(f"NOTE: {path.name} missing -- skipping {variant}")
            continue
        series = parse_losses(path)
        if not series:
            # The orchestrator creates the log before the run starts, so an
            # empty file means "not started yet", not "failed".
            print(f"NOTE: {path.name} has no steps yet -- skipping {variant}")
            continue
        losses[variant] = series
        print(f"{variant}: {len(series)} logged steps")
    if not losses:
        raise SystemExit("no run logs found")

    max_step = max(max(v) for v in losses.values())
    print(f"\nreplaying cell sequence for {max_step} steps (CPU only, no GPU contention)...")
    cells = replay_cells(max_step)

    per_cell: dict[str, dict[tuple[int, int], list[float]]] = {}
    for variant, series in losses.items():
        buckets: dict[tuple[int, int], list[float]] = defaultdict(list)
        for step, loss in series.items():
            if step >= CONVERGED_FROM and step in cells:
                buckets[cells[step]].append(loss)
        per_cell[variant] = buckets

    for variant in VARIANTS:
        if variant not in per_cell:
            continue
        grid = {cell: mean(values) for cell, values in per_cell[variant].items()}
        counts = [len(v) for v in per_cell[variant].values()]
        render(grid, f"=== {variant}: mean training loss per cell (step >= {CONVERGED_FROM}) ===")
        print(f"  samples per cell: min={min(counts)} max={max(counts)}")

    if "baseline" in per_cell:
        for variant in ("kda", "hybrid"):
            if variant not in per_cell:
                continue
            shared = set(per_cell[variant]) & set(per_cell["baseline"])
            delta = {
                cell: mean(per_cell[variant][cell]) - mean(per_cell["baseline"][cell])
                for cell in shared
            }
            render(
                delta,
                f"=== {variant} minus baseline: loss difference per cell "
                f"(positive = {variant} fits worse) ===",
                fmt="{:>+10.4f}",
            )
            print(f"  mean over cells: {mean(delta.values()):+.4f}")
            worst = sorted(delta.items(), key=lambda kv: -kv[1])[:5]
            print("  worst cells for " + variant + ": "
                  + ", ".join(f"{c}:{v:+.3f}" for c, v in worst))
            print("  NOTE: both Variants answer every cell at 1.0000 accuracy. A positive"
                  " value means a looser fit, NOT more wrong answers.")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("docs/runs/experiment_20260728"))
