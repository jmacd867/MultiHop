# The real experiment: three 20,000-step runs, GB10, 2026-07-28/29

The full-length runs the whole project exists to produce. One Variant at a
time on the shared GB10, orchestrated by `scripts/run_all_variants.sh`, each
gated on its final checkpoint verifying finite before the next begins.

**Read ADR 0012, ADR 0013, and ADR 0014 before drawing any conclusion from
the numbers here.** All three were discovered *during* these runs and each one
changes how a figure in this directory must be read. In particular, accuracy
must not be read against chance.

## Status

| Variant | steps | wall clock | final checkpoint | Degradation Grid |
|---|---|---|---|---|
| baseline (full attention) | 20,000 | 5.72h (20:02Z -> 01:45Z) | verified finite (110 model + 223 optimizer tensors) | **1.0000 in all 25 cells** |
| pure-KDA | in progress (18,819/20,000 at time of writing) | ~11.5h so far | pending | **1.0000 in all 25 cells** since step 16,500 |
| hybrid 3:1 | not started | budget 10.74h | pending | pending |

Param counts, matching the sanity runs exactly: baseline 119,411,712,
pure-KDA 134,877,072, hybrid 131,010,732.

## Headline result

**The Degradation Grid is saturated and does not discriminate.** Baseline and
pure-KDA are identical to +0.00pp in every one of the 25 cells. See ADR 0014
for what is reported instead and why.

The two metrics that do carry signal:

- **Acquisition rate.** KDA needs ~1.66x more steps to criterion (4420 vs
  2660), and slows with distance (+29%) where the baseline is flat.
- **Converged per-cell loss.** KDA matches the baseline almost everywhere
  (mean +0.0085 nats, most cells slightly negative) with one outlier at
  **(1,3), +0.201** -- the fully packed cell of ADR 0013.

Both Variants answer every cell correctly. A loss delta is a looser fit, not
more wrong answers.

## What these runs establish

- Both architectures **completely solve** 1-5 hop chained retrieval at
  distances 0-45 on this task at ~125M parameters, well before 20,000 steps
  (baseline by ~step 4,000).
- KDA pays for that in **learning speed**, not final capability, and its
  penalty scales with **sequence length**, not hop count.
- No NaN or non-finite value in any logged step or any verified checkpoint.

## What they explicitly do NOT establish

- **Nothing about hop-depth degradation.** Both Variants are flat across
  hop_count on every measure. The experiment's motivating question -- does the
  hybrid's 3:1 ratio preserve multi-hop chaining, and where does it break --
  cannot be answered by a task that neither architecture finds hard.
- **Nothing about extrapolation** (ADR 0003): every cell scored here was
  trained on.
- **Nothing about production KDA** (ADR 0009): this is core-mechanism KDA
  without ShortConv, not Kimi Linear as deployed.
- **The five distance=0 cells are degenerate** (ADR 0012) and could not have
  discriminated between Variants under any outcome.

## Operational record

**A network outage, not a failure.** The box became unreachable at
2026-07-28 22:55Z (ARP FAILED at layer 2 while the gateway on the same segment
stayed REACHABLE) and returned at 2026-07-29 13:04Z. It **never rebooted** --
boot time stayed 2026-07-20 11:12 throughout -- and **the chain never stopped**:
the baseline completed and verified at 01:45Z and KDA ran through the entire
outage. The second Spark on the same segment was also unreachable while
powered and warm, which is consistent with an upstream cause rather than a
per-host one. Nothing was lost and nothing was resumed.

**A latent bug that would have silently ended the experiment.** Replacing the
orchestrator mid-run (to add the hybrid memory cap) left it rejected by its
own lock. The orchestrator holds `flock` on fd 200, and bash passes open
descriptors to children, so the KDA trainer had inherited it -- and `flock` is
held while *any* process keeps the descriptor open. The lock therefore
outlived the orchestrator that took it. For a period the chain had **no
supervisor at all**: KDA would have finished and hybrid would simply never
have run, with nothing reporting it. Fixed with `200>&-`.

**Memory.** Pure-KDA's real run sits at 119GB of 121GB, against the 105GB its
100-step timing sweep measured -- ~14GB accumulates over 11 hours that a short
sweep never shows. Because this is a shared box and that is inside the
118-121GB range ADR 0009 and ADR 0010 both had to kill runs in, the hybrid run
caps `XLA_PYTHON_CLIENT_MEM_FRACTION` at 0.65 (escalating to 0.80 on a retry).
That is an allocator setting only; numerics and cross-variant comparability are
untouched.

## Files

- `baseline_grids.json`, `kda_grids.json` -- every eval grid, 40 per completed
  run, with step and elapsed time. `scripts/compare_grids.py` reads these.
- `baseline_run.log`, `kda_run.log` -- per-step loss, per-step seconds,
  cumulative hours. `scripts/loss_by_cell.py` reads these.
- `*_verify.log` -- tensor-level finiteness output.
- `orchestrator.log` -- the chain's own record.

Final model-only checkpoints are archived outside the repo in
`checkpoints_final/` (gitignored, ~500MB each). Per ADR 0007 that is the
artifact eval, attention capture, and weight comparison all need; optimizer
state is not archived.

## The follow-up this run makes necessary

The saturation is fixable without retraining, because eval examples are
generated on the fly: score the existing checkpoints at a higher
`distractor_count`, which attacks saturation and the ADR 0012 shortcut floor
together. It is a stress test outside the training distribution, not an
extension of this grid, and must be reported separately (ADR 0014).
