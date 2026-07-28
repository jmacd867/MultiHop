"""Side-by-side Degradation Grid comparison across the three Variants.

Usage: `python scripts/compare_grids.py [runs_dir]` (default `runs/`).

Reads the `<variant>_grids.json` files written by `scripts/train_variant.py`
and renders, for each Variant, the final 25-cell Degradation Grid, plus the
pairwise differences that the experiment's research question is actually
about: hybrid vs. baseline (does the 3:1 ratio preserve full-attention-level
multi-hop chaining?) and kda vs. baseline (what does removing full attention
entirely cost?).

## Reading these numbers responsibly

Per ADR 0003, every (hop_count, distance) cell here was *trained on*. These
grids measure each architecture's capability ceiling on difficulty levels it
has seen, not extrapolation to unseen ones.

Per `eval.py`, each cell is scored on 512 examples, which puts the per-cell
standard error at up to ~2.2 percentage points. `SIGNIFICANCE_PP` below is
set from that: a cell-level difference smaller than it is not distinguishable
from sampling noise, and this script marks such cells rather than letting a
reader treat a 1pp gap as a finding. The three Variants are scored on
identical examples per cell (a shared `EVAL_SEED`, ADR 0008), which removes
*between-variant* sampling noise but not the absolute per-cell error.
"""

import json
import sys
from pathlib import Path

from multihop.data.generator import (
    DISTANCES,
    MAX_HOP_COUNT,
    MIN_HOP_COUNT,
    reference_distractor_count,
)

VARIANT_ORDER = ("baseline", "kda", "hybrid")
HOP_COUNTS = tuple(range(MIN_HOP_COUNT, MAX_HOP_COUNT + 1))

# Two standard errors at 512 examples/cell is ~4.4pp at p=0.5, but the
# comparison is paired (identical examples per cell across Variants), so the
# relevant threshold is smaller. 2.2pp -- one standard error, the figure
# eval.py's docstring quotes -- is used as the "don't read anything into
# this" floor rather than a formal significance test.
SIGNIFICANCE_PP = 2.2


def shortcut_floor(distance: int) -> float:
    """Accuracy reachable with no Chain traversal at all (ADR 0012).

    The Chain's terminal entity is the only entity never used as a subject, so
    it always appears exactly once. "Output the entity appearing exactly once"
    therefore solves the task without traversal, and its accuracy is 1 over the
    number of once-appearing entities: the answer plus one per Distractor.

    At distance=0 `max_distractor_capacity` is 0, so ADR 0004 forces
    `distractor_count=0` and the shortcut is exact -- those five cells are
    degenerate and cannot discriminate between Variants. Everywhere else the
    reference count is 2, giving 3 candidates and a 33.3% floor, identical at
    every hop_count.
    """
    return 1.0 if distance == 0 else 1.0 / (1 + reference_distractor_count(distance))


def load_final_grid(runs_dir: Path, variant: str) -> tuple[dict[tuple[int, int], float], int]:
    """Final grid for one Variant, plus the step it came from.

    Grids are appended in order, so the last entry is the most recent. Sorting
    by the `step` field would be equivalent *and* would silently pick the
    **first** grid for a file written before steps were labelled as they were
    produced -- exactly the case when reading a run that is still in progress.
    Order is the reliable signal; `step` is only used for the label.
    """
    payload = json.loads((runs_dir / f"{variant}_grids.json").read_text())
    evals = payload["evals"]
    if not evals:
        raise ValueError(f"{variant}: no eval grids recorded")
    final = evals[-1]
    grid = {
        (int(key.split(",")[0]), int(key.split(",")[1])): float(value)
        for key, value in final["grid"].items()
    }
    return grid, int(final.get("step", -1))


def render(grid: dict[tuple[int, int], float], title: str, as_delta: bool = False) -> None:
    print(f"\n{title}")
    header = "hop \\ dist" + "".join(f"{d:>9}" for d in DISTANCES)
    print(header)
    print("-" * len(header))
    for hop in HOP_COUNTS:
        cells = []
        for dist in DISTANCES:
            value = grid.get((hop, dist))
            if value is None:
                cells.append(f"{'--':>9}")
            elif as_delta:
                pp = value * 100
                marker = " " if abs(pp) >= SIGNIFICANCE_PP else "~"
                cells.append(f"{pp:>+8.1f}{marker}")
            else:
                cells.append(f"{value:>9.4f}")
        print(f"{hop:>9} " + "".join(cells))


def render_vs_floor(grid: dict[tuple[int, int], float], title: str) -> None:
    """Each cell as accuracy above its traversal-free shortcut floor (ADR 0012).

    This is the quantity that actually reflects Chain traversal. Raw accuracy
    understates the difficulty of the distance>=3 cells (a third of which is
    free) and wildly overstates the distance=0 column (all of which is free).
    """
    print(f"\n{title}")
    header = "hop \\ dist" + "".join(f"{d:>10}" for d in DISTANCES)
    print(header)
    print("-" * len(header))
    for hop in HOP_COUNTS:
        cells = []
        for dist in DISTANCES:
            value = grid.get((hop, dist))
            floor = shortcut_floor(dist)
            if value is None:
                cells.append(f"{'--':>10}")
            elif dist == 0:
                # Degenerate: the shortcut is exact here, so there is no
                # headroom to report. Show raw accuracy, flagged.
                cells.append(f"{value:>9.3f}*")
            else:
                # Fraction of the available headroom above the floor that this
                # cell captured: 0.0 = shortcut-only, 1.0 = perfect.
                cells.append(f"{(value - floor) / (1.0 - floor):>10.3f}")
        print(f"{hop:>9} " + "".join(cells))
    print("  * distance=0 is degenerate (shortcut floor = 1.000); raw accuracy shown")
    print(f"  elsewhere: fraction of headroom above the {shortcut_floor(3):.3f} floor "
          "(0.0 = shortcut only, 1.0 = perfect)")


def mean_by_axis(grid: dict[tuple[int, int], float]) -> None:
    print("  by hop_count: " + "  ".join(
        f"{hop}:{sum(grid[(hop, d)] for d in DISTANCES) / len(DISTANCES):.3f}" for hop in HOP_COUNTS
    ))
    print("  by distance:  " + "  ".join(
        f"{d}:{sum(grid[(h, d)] for h in HOP_COUNTS) / len(HOP_COUNTS):.3f}" for d in DISTANCES
    ))


def main(runs_dir: Path) -> None:
    grids: dict[str, dict[tuple[int, int], float]] = {}
    for variant in VARIANT_ORDER:
        path = runs_dir / f"{variant}_grids.json"
        if not path.exists():
            print(f"NOTE: {path} missing -- skipping {variant}")
            continue
        grid, step = load_final_grid(runs_dir, variant)
        grids[variant] = grid
        render(grid, f"=== {variant} (step {step}) -- raw accuracy ===")
        print(f"  overall mean: {sum(grid.values()) / len(grid):.4f}")
        mean_by_axis(grid)
        render_vs_floor(grid, f"=== {variant} (step {step}) -- above shortcut floor (ADR 0012) ===")
        scoring = [c for c in grid if c[1] != 0]
        below = [c for c in scoring if grid[c] < shortcut_floor(c[1])]
        print(f"  cells at/below the shortcut floor: {len(below)}/{len(scoring)}"
              + (f" -> {sorted(below)}" if below else ""))

    for variant in ("hybrid", "kda"):
        if variant in grids and "baseline" in grids:
            delta = {cell: grids[variant][cell] - grids["baseline"][cell] for cell in grids["baseline"]}
            render(
                delta,
                f"=== {variant} minus baseline (percentage points; "
                f"'~' = below {SIGNIFICANCE_PP}pp, i.e. within per-cell noise) ===",
                as_delta=True,
            )
            print(f"  mean delta: {sum(delta.values()) / len(delta) * 100:+.2f}pp")

    if "hybrid" in grids and "kda" in grids:
        delta = {cell: grids["hybrid"][cell] - grids["kda"][cell] for cell in grids["kda"]}
        render(
            delta,
            "=== hybrid minus kda (percentage points; what the 3 full-attention layers buy) ===",
            as_delta=True,
        )
        print(f"  mean delta: {sum(delta.values()) / len(delta) * 100:+.2f}pp")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent.parent / "runs")
