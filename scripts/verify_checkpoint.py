"""Check a saved checkpoint's tensors directly for non-finite values.

Usage: `python scripts/verify_checkpoint.py <path-to-checkpoint-dir-or-base>`
e.g. `... checkpoints/kda_run` (checks every step found) or
`... checkpoints/kda_run/step_20000` (checks one step).

This project's standing verification bar is that a run's health is
established by reading the saved tensors, **not** inferred from the loss
looking finite. That distinction is not pedantry here: ADR 0006 documents a
baseline configuration that produced NaN *parameters* after the first step
while raising no error at all, and the two sanity-run READMEs both record
tensor-level finiteness counts (218 tensors for pure-KDA, 576 for hybrid)
precisely because a finite printed loss had already been shown to be
insufficient evidence.

Both files are checked, model and optimizer. ADR 0007 split them so that
eval-time consumers never pay for AdamW's moment arrays, but a corrupted
optimizer moment will silently poison every subsequent step even while the
model file still looks clean, so verification -- unlike eval -- wants both.

Exits non-zero if any tensor contains NaN or Inf, so this can gate a run
being declared successful.
"""

import sys
from pathlib import Path

import numpy as np
from safetensors import safe_open
from safetensors.numpy import load_file


def verify_file(path: Path) -> tuple[int, list[str]]:
    """Return (tensor count, names of tensors containing NaN/Inf) for one safetensors file."""
    tensors = load_file(path)
    bad = []
    for name, array in tensors.items():
        if not np.isfinite(np.asarray(array, dtype=np.float64)).all():
            bad.append(name)
    return len(tensors), bad


def verify_step(base: Path) -> bool:
    """Verify one checkpoint. A model-only checkpoint is valid, not a failure.

    ADR 0007 splits model and optimizer into separate files specifically so
    that eval, ADR 0005 attention capture, and cross-variant weight comparison
    never load optimizer state. An archived or copied checkpoint therefore
    legitimately consists of the model file alone. Reporting that as
    "NON-FINITE TENSORS FOUND" -- as this script originally did -- is both
    false (nothing non-finite was found) and actively misleading, since the
    exit code gates whether a run is treated as healthy.

    Only a missing *model* file, or an actual non-finite tensor, is a failure.
    """
    ok = True
    model_path = base.parent / f"{base.name}.model.safetensors"
    if not model_path.exists():
        print(f"  model: MISSING ({model_path.name}) -- cannot verify")
        return False

    for kind in ("model", "optimizer"):
        path = base.parent / f"{base.name}.{kind}.safetensors"
        if not path.exists():
            print(f"  {kind}: absent (model-only checkpoint -- expected for an archive)")
            continue
        count, bad = verify_file(path)
        with safe_open(path, framework="numpy") as f:
            step = (f.metadata() or {}).get("step", "?")
        if bad:
            ok = False
            print(f"  {kind}: {count} tensors, step={step} -- {len(bad)} NON-FINITE: {bad[:5]}")
        else:
            print(f"  {kind}: {count} tensors, step={step} -- all finite")
    return ok


def main(target: Path) -> int:
    if target.is_dir():
        bases = sorted(
            {p.parent / p.name.split(".")[0] for p in target.glob("step_*.safetensors")},
            key=lambda p: int(p.name.removeprefix("step_")),
        )
        if not bases:
            print(f"no checkpoints found in {target}")
            return 1
    else:
        bases = [target]

    all_ok = True
    for base in bases:
        print(f"{base.name}:")
        all_ok &= verify_step(base)

    # Distinguish the two failure modes rather than labelling both as
    # non-finite: a missing file and a corrupt tensor call for different
    # responses, and mislabelling one as the other is how a healthy run gets
    # discarded (or a corrupt one kept).
    print("\nRESULT: " + ("all tensors finite" if all_ok else "FAILED -- see per-file lines above"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} <checkpoint-dir-or-base-path>")
    raise SystemExit(main(Path(sys.argv[1])))
