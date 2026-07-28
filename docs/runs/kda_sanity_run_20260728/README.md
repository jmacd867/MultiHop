# Pure-KDA sanity run, GB10, 2026-07-28

First successful real-hardware run of the pure-KDA variant
(`scripts/sanity_run_kda.py`). 20 steps at `TrainConfig`'s shared defaults
(effective `batch_size=256`, `grad_accum_steps=16`), eval at steps 10 and 20.

## What this run does and does not establish

**Establishes (the sanity-run bar, same one the baseline was held to):**

- No OOM across all 20 steps, including the long-sequence cells that killed
  two earlier attempts. Peak host memory stayed clear of the ceiling.
- No NaN: every one of the 20 logged losses is finite, and all 218 tensors in
  the `step_20` checkpoint are finite (checked directly, not inferred from the
  loss).
- The full pipeline runs end to end against the unmodified shared code --
  `train_step_accum`, `sample_training_microbatches`, `evaluate_grid`, and
  safetensors checkpointing -- with no KDA-specific copies.
- Param count 134,877,072, vs the baseline's ~119.41M. The +15.47M delta is
  fully accounted for by KDA's extra gating projections (alpha_down/up,
  gate_down/up, beta_proj, alpha_log_decay, dt_bias, per-head norm; 1,288,780
  per layer x 12), so the depth/width match with the baseline is intact and
  nothing structural differs unexpectedly.

**Does NOT establish anything about task performance.** The eval grids here
(mean accuracy 0.0013 at step 10, 0.0000 at step 20) are **not meaningful
signal**. At 512-per-cell... actually at the reduced 32-per-cell used in this
script, 25 cells is 800 scored examples total, and chance over the 8,003-token
vocabulary is ~0.000125 -- so 0.0013 is one lucky example and 0.0000 is zero,
both indistinguishable from chance for a model that has taken 20 optimizer
steps. In particular, the 0.0000 at step 20 is **not** the pathological
input-invariance signature ADR 0008 diagnosed; that diagnosis concerned a
*20,000*-step run whose training loss had converged. Drawing any conclusion
about KDA's multi-hop capability from this run would be unfounded.

Training loss does move in the right direction over the run (first five steps
17.06, 16.94, 15.91, 12.29, 17.21; last five 10.86, 17.52, 17.56, 12.18,
10.81), but per-step loss here is dominated by which (hop_count, distance)
cell was sampled, not by learning progress, so even that is weak evidence at
this step count.

## Two GB10 failures fixed to get here

Both were compile-time memory blowups from Python-unrolled loops, both fixed
by scanning instead of unrolling. Neither was a bug in the KDA math.

1. **Chunk loop** (ADR 0009): `kda_chunked_scan` unrolled its per-chunk loop,
   which multiplied against 12 layers and 16 grad-accum steps into an enormous
   jaxpr. Drove host RAM to 118GB/121GB with the GPU idle; killed manually.
   Fixed with `jax.lax.scan` over chunks.
2. **Micro-batch loop** (ADR 0010): `train_step_accum` unrolled its
   grad-accumulation loop (deliberate per ADR 0006, fine for the baseline).
   With KDA's much larger per-layer graph this exhausted the box during
   compilation at ~121GB, again before any step ran. An intermediate attempt
   to fix this by *raising* `grad_accum_steps` to 32 made it worse -- compile
   memory scales with `grad_accum_steps` while runtime activation memory
   scales with `micro_batch_size`. Fixed by scanning the accumulation loop,
   verified not to change baseline numerics via
   `test_train_step_accum_matches_a_true_batch_step_of_the_same_data`.

## Files

- `sanity_run_kda.log` -- full stdout, including the XLA slow-compile warnings.

## Checkpoint disposition

`checkpoints/sanity_run_kda/step_20.{model,optimizer}.safetensors` (~1.6GB
total) was left on the GB10 box, not pulled back. It's a 20-step checkpoint
with no experimental value -- kept only until the real pure-KDA training run
supersedes it, and safe to delete at any time.
