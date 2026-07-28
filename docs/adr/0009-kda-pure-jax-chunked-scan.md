# KDA layer: custom chunked-scan recurrence in pure JAX, no borrowed Triton/CUDA kernels

The pure-KDA model variant needs an implementation of the Kimi Delta
Attention (gated delta-rule) recurrence. The natural default would be to
reach for an existing implementation — `fla` (flash-linear-attention) and
similar libraries ship Triton kernels for DeltaNet/Gated-DeltaNet/KDA that
are faster and already validated against the reference papers than
anything written from scratch here.

That default was rejected for this project. Two options were considered:

1. **Borrow an existing Triton/CUDA kernel** (e.g. `fla`'s chunked
   gated-delta-rule kernel). Fastest path to a correct, fast
   implementation, but Triton kernel support on aarch64 (GB10/DGX Spark is
   an aarch64 box) is a live compatibility risk — Triton's aarch64 +
   CUDA13 support is newer and less battle-tested than its x86_64 path,
   and this project's `pyproject.toml` already carries one aarch64/CUDA13
   forward-compatibility caveat for plain `jax[cuda12]` (see the `cuda`
   extra's comment). Adding a second, independent piece of aarch64-fragile
   compiled-kernel infrastructure stacks that risk rather than isolating
   it, and a kernel failure would be indistinguishable at first from a
   math bug in the variant we're actually trying to evaluate.
2. **Custom chunked-scan implementation in pure JAX**, compiled through
   XLA like every other layer in this codebase (`layers.py`,
   `rope.py`). Slower than a hand-tuned Triton kernel, and the
   implementation burden (and correctness burden — see the numeric
   unit-test requirement below) falls on this project instead of a
   maintained upstream library.

**Decision: option 2.** The whole point of the three-way comparison is to
attribute the hop-distance degradation pattern to genuine architectural
differences (full attention vs. delta-rule linear attention vs. hybrid),
not to differences in numerical implementation, compiled-kernel behavior,
or platform compatibility between variants. A borrowed Triton kernel that
silently underperforms, miscompiles, or behaves differently on aarch64
than on the x86_64 boxes it was primarily validated on would confound
exactly the comparison this experiment exists to make — and would do so
in a way that's hard to distinguish from a genuine finding about KDA's
degradation behavior. Pure JAX, compiled through the same XLA path as the
baseline, keeps every variant on identical platform-compatibility and
numerics footing, which matters more here than raw throughput.

**Known trade-off, accepted:** a chunked-scan implementation in plain JAX
will be materially slower than a fused Triton kernel at the same sequence
lengths, and correctness is now this project's responsibility rather than
an upstream library's. This is judged acceptable because the task's
sequence lengths are small (max_seq_len=287, per `ModelConfig`) and this
phase is explicitly scoped to a short sanity run, not a full training run
at production throughput (see the KDA implementation task's "what not to
do yet"). If KDA throughput becomes a real bottleneck once full training
runs are in scope, revisit — but that is a future decision, not this one.

**Consequence for verification:** because there's no upstream
implementation to defer correctness to, the KDA layer's unit tests must
include numeric correctness against a small hand-computable or
reference-matched case (not shape checks alone) — the same bar RoPE
(`tests/models/test_rope.py`) was already held to in this codebase, now
extended to a recurrence rather than a closed-form encoding.

## Scope: core delta-rule mechanism, not Kimi Linear's full production layer

Kimi Delta Attention as specified in the source paper (arXiv:2510.26692,
"Kimi Linear: An Expressive, Efficient Attention Architecture") is two
separable things: (1) the delta-rule recurrence itself —
`S_t = (I - β_t k_t k_t^T) Diag(α_t) S_{t-1} + β_t k_t v_t^T`,
`o_t = S_t^T q_t` — with its chunkwise-parallel (WY/UT-transform) form,
and (2) a production neural parameterization built around that
recurrence for their 48B-param deployed model: a ShortConv (short
depthwise causal convolution) before the Swish/L2Norm-activated q/k/v
projections, low-rank projections for the per-channel decay gate and a
separate sigmoid output gate, and per-head RMSNorm before the output
projection.

This project implements (1) faithfully and only borrows the parts of (2)
that are load-bearing for the recurrence to function at all: L2Norm on
q/k (the paper ties this to eigenvalue stability of the state transition,
not a peripheral tuning choice), Swish on v, a low-rank projection for
the per-channel decay gate α (needed for α to have a sensible number of
learnable parameters at all — a full-rank `embed_dim -> head_dim`
projection per channel would work but the low-rank form is how the paper
defines α's parameterization, not an add-on), and a sigmoid output gate.
**ShortConv is deliberately excluded.** It's optional machinery on top of
the recurrence, not part of it — the paper itself frames it as one of
several components layered onto KDA for their production model, the same
category as their MoE channel-mixing layers, which this project also
isn't replicating. Including it would give the KDA variant a
structurally different local-context-mixing capability than the
full-attention baseline has (the baseline's `CausalSelfAttention` is
plain linear q/k/v/o projections with no equivalent), reintroducing
exactly the kind of asymmetric-capability confound that matching
depth/width across variants (this ADR, ModelConfig) exists to avoid. Any
accuracy gap later observed between variants must be attributable to the
attention/gating mechanism alone, not to one variant quietly getting an
extra local-mixing primitive the other lacks.

**Named limitation, not an oversight:** this makes the pure-KDA variant
*core-mechanism KDA*, not *production-tuned KDA as Moonshot AI actually
deploys it*. If this variant underperforms meaningfully in the eventual
degradation-grid comparison, that result describes the delta-rule/
channel-wise-gating mechanism in isolation — it should not be read as a
claim about how the real, ShortConv-equipped, MoE-hybridized Kimi Linear
architecture would perform on this task. A future reader comparing this
project's numbers to the paper's benchmarks should keep that scope
boundary in mind, the same way ADR 0003's "capacity, not extrapolation"
framing bounds what the hop-distance grid can and can't be read to show.

## head_dim and chunk size

**head_dim = 64, n_heads = 12** (identical to `ModelConfig`'s existing
values, i.e. `embed_dim // n_heads`), not the paper's fixed
`d_k = d_v = 128`. 128 was tuned for their 48B-param model; reusing the
baseline's existing head geometry keeps depth/width/head-count identical
across every variant so only the sequence-mixing mechanism varies,
consistent with `config.py`'s stated goal of one shared config across all
three variants.

**Chunk size C = 32** rather than the paper's C=64. The chunked
recurrence's math only requires that state carry correctly from one
chunk's end to the next chunk's start, and this project's sequences
(`max_seq_len=287`, down to a handful of tokens at `hop_count=1,
distance=0`) are far shorter than what C=64 was tuned for — C=32 keeps
compute (and chunk count, see the `lax.scan` note below) proportionate to
these shorter sequences.

`kda_chunked_scan` pads the final chunk with neutral entries (alpha=1,
beta=0, so padded steps carry state through unchanged and their discarded
output is well-defined) rather than leaving it genuinely ragged. This
started as a genuinely ragged final chunk with no padding at all (avoiding
"wasted compute on short sequences" was the original reasoning) — see the
`lax.scan` entry below for why that changed. The padding here is cheap
either way (at most `chunk_size - 1` = 31 wasted positions per sequence,
not the paper's C=64 waste this ADR originally argued against), so it
doesn't reopen that trade-off.

## Correction: chunk loop is `jax.lax.scan`, not an unrolled Python loop

The first implementation unrolled the chunk loop in plain Python, matching
this codebase's existing convention of unrolling other static-length
loops at trace time (e.g. `train_step_accum`'s grad-accumulation loop).
That reasoning didn't account for *composition*: `train_step_accum`
already unrolls `grad_accum_steps=16` full forward+backward passes in one
`nnx.jit`, and the model has `n_layers=12` KDA layers, so an additionally
Python-unrolled chunk loop multiplied out to `16 * 12 * num_chunks`
unrolled matrix-inversion blocks (`jnp.linalg.inv` in `_kda_chunk`) inside
a single XLA program. Compiling the first real GB10 sanity run at this
size drove host RAM from idle to 118GB/121GB used with the GPU sitting at
0% utilization for 30+ minutes — i.e. it was still compiling, not
training, and on the verge of triggering the OOM killer on a shared box.
The run was killed before it got there.

**Fix: `kda_chunked_scan` uses `jax.lax.scan` over chunks.** This compiles
the per-chunk body once and loops it at runtime, independent of chunk
count, layer count, or grad-accum steps — eliminating the multiplicative
blowup regardless of how the layer gets composed by code outside this
module's control. `lax.scan` requires uniform per-step shapes, which is
also why the final chunk is now padded (see above) rather than genuinely
ragged: a ragged final chunk can't be a `lax.scan` step.

## Known limitation: GB10 float32 matmul precision widens chunked-vs-sequential drift

`tests/models/test_kda.py`'s chunked-vs-sequential-reference tests
(atol/rtol=1e-4) pass cleanly on CPU but fail on the GB10 at that
tolerance — not from a logic bug (the actual/expected values agree to
~2-6% relative error, not the order-of-magnitude divergence a real bug
produced during development), but from GPU float32 matmul precision:
GB10's Blackwell tensor cores default to reduced-precision (TF32-class)
accumulation for `jnp.einsum`/`jnp.linalg.inv`, and this recurrence chains
many chunk-boundary matmuls, so per-step precision loss compounds along
the sequence. Setting `JAX_DEFAULT_MATMUL_PRECISION=highest` reproduces
the CPU-tight tolerances exactly on GB10, confirming the diagnosis. This
project doesn't set that env var by default (the baseline doesn't either,
and this ADR's mandate is about kernel choice, not numeric precision
policy) — noted here as a known platform trade-off to revisit if a real
KDA training run shows precision-related instability that a short sanity
run wouldn't surface.
