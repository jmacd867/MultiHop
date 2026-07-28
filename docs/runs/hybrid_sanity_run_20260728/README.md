# Hybrid (3:1) sanity run, GB10, 2026-07-28

First real-hardware run of the 3:1 hybrid Variant (`scripts/sanity_run_hybrid.py`).
20 steps at `TrainConfig`'s shared defaults (effective `batch_size=256`,
`grad_accum_steps=16`), eval at steps 10 and 20. Deliberately identical in
budget and cadence to the pure-KDA sanity run
(`docs/runs/kda_sanity_run_20260728/`) so the two are directly comparable.

## What this run does and does not establish

**Establishes (the sanity-run bar, the same one the other two Variants were held to):**

- **No OOM** across all 20 steps. Peak host memory held flat at ~104GB/121GB,
  of which **93.7GB was JAX's preallocated arena** (confirmed via `nvidia-smi`
  against the run's PID, on GB10's unified memory) -- i.e. a fixed allocation
  made at startup, not growth under load. This is nowhere near the
  118-121GB climb that killed the two earlier KDA attempts (ADR 0009, ADR 0010).
- **No NaN:** all 20 logged losses are finite, and **all 576 tensors** in the
  `step_20` checkpoint (191 model + 385 optimizer) are finite -- checked
  directly against the saved files, not inferred from the loss.
- **Runs at the mandated batch settings unchanged.** ADR 0010 fixes
  `batch_size=256`/`grad_accum_steps=16` across all three Variants, since
  divergent batch/LR conditions would erode the cross-variant comparability the
  Degradation Grid comparison depends on. The hybrid fits there with no
  per-variant tuning, so no ADR-worthy memory finding arose.
- **Param count 131,010,732**, read back from the checkpoint. This exactly
  matches the prediction derived from the other two Variants
  (baseline 119,411,712 + 9 x 1,288,780 per-layer KDA gating delta), confirming
  the depth/width match required by ADR 0009 is intact and that the 3:1 mix is
  structurally what ADR 0011 specifies -- not a number accepted after the fact.
- **The ADR 0011 schedule is what actually ran:** the run logs
  `full-attention at [3, 7, 11], KDA elsewhere` at startup.
- The full shared pipeline runs end to end with **no hybrid-specific copies** --
  `train_step_accum`, `sample_training_microbatches`, `evaluate_grid`, and
  safetensors checkpointing all unmodified.

**Does NOT establish anything about task performance.** The eval grids here
(mean accuracy 0.0000 at both step 10 and step 20) are **not meaningful
signal**, for exactly the reasons the pure-KDA run's README sets out: at the
reduced 32-examples-per-cell used in this script, 25 cells is 800 scored
examples, and chance over the 8,003-token vocabulary is ~0.000125. A model that
has taken 20 optimizer steps scoring 0.0000 is indistinguishable from chance.
In particular this is **not** the input-invariance signature ADR 0008
diagnosed -- that concerned a *20,000*-step run whose training loss had
converged.

Training loss does move in the right direction (first five 17.04, 16.87, 14.61,
14.33, 17.09; last five 10.77, 16.60, 16.78, 11.98, 10.72), but
per-step loss here is dominated by which Grid Cell was sampled, not by learning
progress. The question this run leaves open -- does the hybrid learn at all at
full scale? -- is what `scripts/probe_hybrid_single_cell.py` answers below.

## Compile cost

Wall clock was **~20 minutes** end to end (approximate -- the script does not
timestamp its output), and the log contains **zero** XLA `slow_operation_alarm`
warnings. Compile, not execution, dominates that figure: 20 optimizer steps at
~1s each is negligible against one `train_step_accum` compile per distinct
sequence shape plus the eval grid's 25 forward-only shapes.

**No comparison against the pure-KDA run is available.** Neither script
timestamps its steps, and the pure-KDA log carries no slow-compile warnings
either, so there is no measured basis for claiming the hybrid compiles more
cheaply -- only a structural expectation that it should (9 KDA layers to
pure-KDA's 12, and compile cost here has been dominated by the KDA layer's
`lax.scan` body -- matrix inversion plus WY/UT-transform intermediates,
ADR 0009). If that comparison ever matters, both scripts need timestamps first.

No new failures had to be fixed to get here. Both scan fixes (ADR 0009's chunk
loop, ADR 0010's micro-batch loop) were already in the shared code, and ADR 0010
explicitly anticipated this: *"It also generalizes: the hybrid variant will
contain KDA layers too and would hit the same wall."* It did not hit the wall,
because the wall had already been removed.

## The single-cell probe: the hybrid does learn at full scale

`scripts/probe_hybrid_single_cell.py`, 3,000 steps at hop_count=1, distance=0,
**20.8 minutes** wall clock. This is the run that answers the question the
20-step sanity run leaves open.

| step | accuracy | distinct predictions |
|-----:|---------:|---------------------:|
|  250 |   0.3184 |              420/512 |
|  500 |   0.9980 |              496/512 |
|  750 |   1.0000 |              496/512 |
| 1750 |   0.9980 |              496/512 |
| 3000 |   1.0000 |              496/512 |

(1000-1500 and 2000-2750 were all 1.0000; the single 0.9980 readings are one
example of 512.)

> **Corrected by ADR 0012 (2026-07-28, after this run).** This section
> originally read the 1.0000 as "the first demonstration of real task
> performance at full scale by any Variant", benchmarked against ~0.000125
> chance over the vocabulary. Both are wrong. ADR 0012 established that the
> task has a traversal-free "output the entity appearing exactly once"
> shortcut whose floor at **distance=0 is 100%**, not chance -- so this cell
> is *degenerate* and a perfect score there is the expected result for a model
> doing no retrieval at all. The claims below are the corrected ones; the
> original text is preserved in git history rather than silently rewritten.

**What this establishes.** The composition works end to end at full model
scale: gradients flow through a stack that alternates two different
token-mixing mechanisms, the shared training and eval harness drives it
unmodified, and loss falls steadily to a stable optimum. That was the probe's
actual purpose -- a smoke test of the forward/backward path through a
heterogeneous stack -- and that conclusion stands.

**What it does NOT establish -- including things this README first claimed it
did.** hop_count=1, distance=0 is not merely the easiest cell; per ADR 0012 it
is the *most degenerate* cell in the grid. Its entire sequence is five tokens
(`A B | ? A`, answer `B`), the answer is the only once-appearing entity in
512/512 examples, and an even cheaper strategy also scores 100%: **copy the
second token**, which needs no binding, no retrieval, and no attention to the
Query at all. So this run is evidence of **neither multi-hop reasoning nor
in-context binding nor retrieval of any kind**. Reaching 1.0000 here is what a
working optimizer looks like, not what a working retrieval mechanism looks
like.

**On the `distinct_predictions` column.** It is pinned at 496/512 from step 500
onward because the probe scores a *fixed* seeded batch, so 496 is simply the
number of distinct true answers in that batch -- no information once accuracy
is high. Step 250's 420 distinct at 31.8% does rule out ADR 0008's
input-invariance collapse (predictions genuinely vary with input), but note
that "copy the second token" also produces input-varying predictions, so this
rules out a specific pathology rather than establishing binding.

**Interpreting a hypothetical failure.** Had this probe returned zero, it would
*not* have implicated the hybrid on its own: no Variant had previously cleared
this bar at full scale, so there was no reference for how many steps should
suffice. The probe's docstring records the contingency (run the identical probe
on the baseline first). That contingency was not needed.

## Files

- `sanity_run_hybrid.log` -- full stdout of the 20-step sanity run, including
  both eval grids.
- `probe_hybrid_single_cell.log` -- full stdout of the 3,000-step probe.

## Checkpoint disposition

`checkpoints/sanity_run_hybrid/step_20.{model,optimizer}.safetensors` (~1.5GB
total) was left on the GB10 box, not pulled back. It is a 20-step checkpoint
with no experimental value -- kept only until the real hybrid training run
supersedes it, and safe to delete at any time.
