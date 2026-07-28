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

from multihop.data.generator import DISTANCES, MAX_HOP_COUNT, MIN_HOP_COUNT

VARIANT_ORDER = ("baseline", "kda", "hybrid")
HOP_COUNTS = tuple(range(MIN_HOP_COUNT, MAX_HOP_COUNT + 1))

# Two standard errors at 512 examples/cell is ~4.4pp at p=0.5, but the
# comparison is paired (identical examples per cell across Variants), so the
# relevant threshold is smaller. 2.2pp -- one standard error, the figure
# eval.py's docstring quotes -- is used as the "don't read anything into
# this" floor rather than a formal significance test.
SIGNIFICANCE_PP = 2.2


def load_final_grid(runs_dir: Path, variant: str) -> tuple[dict[tuple[int, int], float], int]:
    """Final (highest-step) grid for one Variant, plus the step it came from."""
    payload = json.loads((runs_dir / f"{variant}_grids.json").read_text())
    evals = payload["evals"]
    if not evals:
        raise ValueError(f"{variant}: no eval grids recorded")
    final = max(evals, key=lambda e: e.get("step", 0))
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
        render(grid, f"=== {variant} (step {step}) -- accuracy ===")
        print(f"  overall mean: {sum(grid.values()) / len(grid):.4f}")
        mean_by_axis(grid)

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
