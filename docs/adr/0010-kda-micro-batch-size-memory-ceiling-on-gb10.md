# KDA's compile-time memory forces train_step_accum's micro-batch loop to become a lax.scan

ADR 0006 established `micro_batch_size=16` (`TrainConfig`'s default
`grad_accum_steps=16` at `batch_size=256`) as safe on GB10 for the
full-attention baseline. The pure-KDA sanity run (`scripts/sanity_run_kda.py`,
`TrainConfig` defaults, unmodified) reused that same value — consistent
with the task's "reuse train_step_accum... as-is" — and it OOM'd on GB10 at
step 2, the first step to sample a long sequence.

**What happened:** step 1 (seq_len=143) completed cleanly (loss=17.0652,
matching a standalone smoke test at the same shape). Step 2 sampled a
different, longer-sequence (hop_count, distance) cell and
`jit_train_step_accum` raised a clean `RESOURCE_EXHAUSTED`:  XLA's
rematerialization pass could only shrink the required allocation from
126.12GiB to 102.77GiB, still short of what a single `micro_batch_size=16`
backward pass needed (~100GiB requested from the allocator) against GB10's
~121GiB unified memory (already carrying JAX's own upfront pool
reservation). This is a clean, informative crash — not the baseline's
original silent-NaN failure mode (ADR 0006) — but it's still a real block
on completing a sanity run at the shared `TrainConfig` defaults.

**Why KDA needs more memory per micro-batch than the baseline at the same
`micro_batch_size`:** the chunked-scan recurrence (`kda_chunked_scan` /
`_kda_chunk`, ADR 0009) carries substantially more intermediate state per
layer than plain causal self-attention -- per-chunk matrix inversions
(`jnp.linalg.inv`), the W/U/M auxiliary matrices from the WY/UT-transform,
and the `lax.scan` carry itself, all subject to autodiff through
`train_step_accum`'s backward pass. Plain attention's backward pass only
needs to retain the attention scores/weights per layer; KDA's backward
pass needs to retain (or rematerialize) that whole per-chunk auxiliary
structure, for every one of `ceil(287/32)=9` chunks per layer, across all
12 layers. It's the same qualitative shape of problem ADR 0006 diagnosed
for the baseline (naive backward-pass activation memory exceeding GB10's
budget at effective batch_size=256), but the safe micro-batch size is
smaller because KDA's per-example footprint is larger.

**Empirical check (mirroring ADR 0006's own diagnostic method):** a
standalone single-step backward pass at the worst-case shape
(hop_count=5, distance=45, seq_len=287) was tried directly at decreasing
`micro_batch_size`, independent of `train_step_accum`/grad-accumulation,
to isolate the memory ceiling:

- `micro_batch_size=16` (ADR 0006's baseline-safe value): OOM, matching
  the sanity run's crash.
- `micro_batch_size=8`: succeeded (finite loss).
- `micro_batch_size=4`: succeeded (finite loss).

**First attempted fix (FAILED -- do not repeat):**
`scripts/sanity_run_kda.py` was set to `grad_accum_steps=32` (->
`micro_batch_size=8`), keeping effective `batch_size=256`. The isolated
per-micro-batch measurements above say this *should* fit, and as a
statement about runtime activation memory it's correct. The rerun still
died on GB10 -- this time consuming ~121GiB of the box's ~124GiB and
never reaching step 2, killed manually before the OS OOM killer could
fire on what is a shared machine.

**Why it failed, and the real constraint:** the two memory costs here are
distinct and pull in opposite directions.

- *Runtime activation memory* scales with `micro_batch_size`. Halving it
  16 -> 8 helps, exactly as the isolated tests showed.
- *Compile-time memory* scales with `grad_accum_steps`, because
  `train_step_accum` unrolls its micro-batch loop in plain Python (a
  deliberate choice per ADR 0006, fine for the baseline). Doubling
  `grad_accum_steps` 16 -> 32 doubles the number of full
  forward+backward passes inlined into a single jaxpr.

For the full-attention baseline the second cost is small enough not to
matter. For KDA it dominates: each layer's graph already contains a
`lax.scan` body with a matrix inversion and the WY/UT-transform
intermediates (ADR 0009), so inlining 32 copies of a 12-layer backward
pass is what actually exhausted the box. XLA's own
`slow_operation_alarm` reported ~2m56s to compile a single
`jit_train_step_accum` module, corroborating that compilation -- not
execution -- was the bottleneck.

Raising `grad_accum_steps` to reduce memory is therefore
**counterproductive for this variant**: it trades a cost KDA can afford
for one it can't.

**Options considered.** Reducing `micro_batch_size` without also raising
`grad_accum_steps` would mean abandoning the effective `batch_size=256`
that the baseline's LR schedule and other hyperparameters were tuned for
-- a cross-variant comparability problem, not just a throughput one, so
it was not a decision to make silently inside a sanity-run script.

1. Make `train_step_accum`'s micro-batch loop a `jax.lax.scan` (as
   `kda_chunked_scan` already does for chunks), so compile cost stops
   scaling with `grad_accum_steps`. Principled and addresses the actual
   dominant cost, but edits training code shared with the *already
   validated* baseline, so it needs its own verification that baseline
   numerics are unchanged.
2. Apply rematerialization (`nnx.remat`/`jax.checkpoint`) to the KDA
   layer or block, cutting activation memory so a larger
   `micro_batch_size` (and hence smaller `grad_accum_steps`) fits.
   ADR 0006 explicitly deferred rematerialization for the baseline; this
   would reopen it for KDA only.
3. Accept a smaller effective batch size for KDA and document the
   cross-variant hyperparameter divergence as a named limitation.

**Decision: option 1.** It targets the cost that actually dominates
(compile-time graph size) rather than working around it, and it keeps
every variant on the identical effective `batch_size=256` and
hyperparameters -- preserving the cross-variant comparability that
options 2 and 3 would each erode in different ways (option 2 changes what
KDA recomputes vs. stores relative to the baseline; option 3 changes
KDA's batch/LR conditions outright). It also generalizes: the hybrid
variant will contain KDA layers too and would hit the same wall.

`train_step_accum` now stacks the micro-batches along a leading axis and
`lax.scan`s a gradient-accumulation body over them, splitting the model
into a static graphdef plus params so the body is a pure function of
arrays. This is safe to do generically because
`sample_training_microbatches` already guarantees every micro-batch in a
step is identically shaped (it draws them all from one (hop_count,
distance) cell) -- an invariant that predates this change and was
previously needed to keep `@nnx.jit` from retracing.

**Verification that the baseline is unaffected:**
`tests/test_train.py::test_train_step_accum_matches_a_true_batch_step_of_the_same_data`
is the load-bearing check -- it asserts at tight tolerance (rtol=1e-5,
atol=1e-6) that gradient-accumulated parameters after a step match a
single true-batch step over the identical examples, and it was written
specifically to catch accumulation bugs (summing instead of averaging, a
dropped micro-batch) rather than "roughly similar" dynamics. It passes
unchanged against the scanned implementation, as does the rest of the
suite (64 tests) and mypy --strict.

**Also not yet answered:** whether whatever configuration eventually
works for a 20-step sanity run remains viable at the real training step
count, where per-shape compile cost is amortized differently. Out of
scope for this task (see the KDA implementation task's "don't attempt a
real training run at full step count").