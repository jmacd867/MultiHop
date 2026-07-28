"""Kimi Delta Attention (KDA): gated delta-rule linear attention, chunked-scan in pure JAX.

Implements the exact recurrence and chunkwise-parallel algorithm from
"Kimi Linear: An Expressive, Efficient Attention Architecture"
(arXiv:2510.26692), Eq. 1-9:

    S_t = (I - beta_t k_t k_t^T) Diag(alpha_t) S_{t-1} + beta_t k_t v_t^T
    o_t = S_t^T q_t

with a per-channel (not scalar) decay gate `alpha_t`, which is KDA's
extension over Gated DeltaNet's scalar gate. See ADR 0009 for why this
project implements the recurrence and chunkwise algorithm faithfully but
deliberately excludes the paper's ShortConv (an optional production
add-on, not part of the recurrence), and for the head_dim/chunk_size
choices below.
"""

import jax
import jax.numpy as jnp
from flax import nnx
from jax import Array

from multihop.models.config import ModelConfig
from multihop.models.layers import RMSNorm, SwiGLU

CHUNK_SIZE = 32
_DECAY_FLOOR = 1e-6  # numerical floor for dividing by cumulative decay (paper Sec 3.2's division risk)


def l2norm(x: Array, *, eps: float = 1e-6) -> Array:
    """L2-normalize along the last axis (paper: applied to q/k for state-transition eigenvalue stability)."""
    norm = jnp.sqrt(jnp.sum(jnp.square(x), axis=-1, keepdims=True))
    return x / (norm + eps)


def _kda_chunk(
    state: Array,
    q_c: Array,
    k_c: Array,
    v_c: Array,
    alpha_c: Array,
    beta_c: Array,
) -> tuple[Array, Array]:
    """Process one chunk for every (batch, head) at once.

    Shapes: q_c/k_c/alpha_c (B, H, C, dk), v_c (B, H, C, dv), beta_c (B, H, C),
    state (B, H, dk, dv). Returns (new_state, chunk_output) with chunk_output (B, H, C, dv).

    Directly implements paper Eq. 3-9 (the WY/UT-transform chunkwise form), with
    `state` playing the role of `S^0_[t]` (this chunk's initial state) throughout.
    """
    chunk_len = q_c.shape[2]
    gamma = jnp.cumprod(alpha_c, axis=2)  # Gamma^{1->r} for r=1..C, shape (B,H,C,dk)
    gamma_full = gamma[:, :, -1, :]  # Gamma^{1->C}, the whole-chunk decay, shape (B,H,dk)
    gamma_safe = jnp.clip(gamma, min=_DECAY_FLOOR)

    gamma_k = gamma * k_c  # Gamma^{1->C} \odot K, shape (B,H,C,dk)
    gamma_q = gamma * q_c
    k_over_gamma = k_c / gamma_safe

    # A[r, s] = (gamma_k)_r . (k_over_gamma)_s : (B,H,C,C)
    scores = jnp.einsum("bhrd,bhsd->bhrs", gamma_k, k_over_gamma)
    strict_lower = jnp.tril(jnp.ones((chunk_len, chunk_len), dtype=bool), k=-1)
    scores = jnp.where(strict_lower, scores, 0.0)
    transition = jnp.eye(chunk_len, dtype=scores.dtype) + beta_c[:, :, :, None] * scores

    # M = transition^{-1} @ Diag(beta); Diag(beta) is diagonal, so this is just a
    # per-column scale of transition^{-1} by beta (see kda.py docstring / ADR 0009).
    transition_inv = jnp.linalg.inv(transition)
    m = transition_inv * beta_c[:, :, None, :]

    w = jnp.einsum("bhrs,bhsd->bhrd", m, gamma_k)  # (B,H,C,dk)
    u = jnp.einsum("bhrs,bhsd->bhrd", m, v_c)  # (B,H,C,dv)

    w_state = jnp.einsum("bhrd,bhde->bhre", w, state)  # W @ S^0, (B,H,C,dv)
    pseudo_value = u - w_state  # "U - W S", (B,H,C,dv)

    # The k_i v_i^T contribution added at step i is undecayed at i itself (only steps
    # *after* i decay it), so this uses k_over_gamma (k_i / gamma^i) rather than
    # gamma_k (gamma^i * k_i) -- gamma^i would spuriously decay the i=chunk_len term
    # that must reduce to exactly beta*k*v^T with no decay applied at all.
    new_state = gamma_full[:, :, :, None] * (
        state + jnp.einsum("bhrd,bhre->bhde", k_over_gamma, pseudo_value)
    )

    inter_chunk = jnp.einsum("bhrd,bhde->bhre", gamma_q, state)  # uses S^0 (start-of-chunk state)
    intra_scores = jnp.einsum("bhrd,bhsd->bhrs", gamma_q, k_over_gamma)
    causal = jnp.tril(jnp.ones((chunk_len, chunk_len), dtype=bool))
    intra_scores = jnp.where(causal, intra_scores, 0.0)
    intra_chunk = jnp.einsum("bhrs,bhse->bhre", intra_scores, pseudo_value)

    chunk_output = inter_chunk + intra_chunk
    return new_state, chunk_output


def kda_chunked_scan(
    q: Array,
    k: Array,
    v: Array,
    alpha: Array,
    beta: Array,
    *,
    chunk_size: int = CHUNK_SIZE,
) -> Array:
    """Gated delta-rule attention output via a chunkwise-parallel scan.

    q, k, alpha: (batch, seq_len, n_heads, d_k). v: (batch, seq_len, n_heads, d_v).
    beta: (batch, seq_len, n_heads), in [0, 1]. alpha: (batch, seq_len, n_heads, d_k), in [0, 1].
    Returns o: (batch, seq_len, n_heads, d_v).

    Implemented as a `jax.lax.scan` over chunks (pad `seq_len` up to a multiple of
    `chunk_size` with neutral entries -- alpha=1, beta=0, so padded steps carry state
    through unchanged and contribute nothing -- then slice the padding back off).
    An earlier version unrolled the chunk loop in plain Python (matching this
    codebase's other static-length-loop convention, e.g. train.py's
    train_step_accum). That compiled fine standalone, but combined with
    train_step_accum's own 16-way grad-accumulation unroll and 12 KDA layers, it
    produced an enormous XLA program (num_layers * grad_accum_steps * num_chunks
    unrolled matrix-inversion blocks in one jaxpr) that came close to exhausting
    host RAM compiling on the GB10 -- see ADR 0009. `lax.scan` compiles the
    per-chunk body once and loops it, independent of chunk count.
    """
    batch, seq_len, n_heads, d_k = q.shape
    d_v = v.shape[-1]

    num_chunks = -(-seq_len // chunk_size)  # ceil division
    padded_len = num_chunks * chunk_size
    pad_amount = padded_len - seq_len

    def pad_seq(x: Array, pad_value: float) -> Array:
        if pad_amount == 0:
            return x
        pad_width = [(0, 0)] * x.ndim
        pad_width[1] = (0, pad_amount)
        return jnp.pad(x, pad_width, constant_values=pad_value)

    # alpha=1 (no decay) and beta=0 (no update) on padded steps means they carry
    # state through unchanged and their (discarded) outputs are well-defined.
    q_p, k_p, v_p = pad_seq(q, 0.0), pad_seq(k, 0.0), pad_seq(v, 0.0)
    alpha_p = pad_seq(alpha, 1.0)
    beta_p = pad_seq(beta, 0.0)

    def to_chunks(x: Array) -> Array:
        # (batch, padded_len, n_heads, d) -> (num_chunks, batch, n_heads, chunk_size, d)
        x = x.reshape(batch, num_chunks, chunk_size, n_heads, x.shape[-1])
        return x.transpose(1, 0, 3, 2, 4)

    def to_chunks_beta(x: Array) -> Array:
        # (batch, padded_len, n_heads) -> (num_chunks, batch, n_heads, chunk_size)
        x = x.reshape(batch, num_chunks, chunk_size, n_heads)
        return x.transpose(1, 0, 3, 2)

    q_c, k_c, v_c, alpha_c = to_chunks(q_p), to_chunks(k_p), to_chunks(v_p), to_chunks(alpha_p)
    beta_c = to_chunks_beta(beta_p)

    init_state = jnp.zeros((batch, n_heads, d_k, d_v), dtype=q.dtype)

    def scan_body(
        state: Array, chunk_inputs: tuple[Array, Array, Array, Array, Array]
    ) -> tuple[Array, Array]:
        return _kda_chunk(state, *chunk_inputs)

    _final_state, outputs = jax.lax.scan(scan_body, init_state, (q_c, k_c, v_c, alpha_c, beta_c))
    # outputs: (num_chunks, batch, n_heads, chunk_size, d_v)
    outputs = outputs.transpose(1, 0, 3, 2, 4).reshape(batch, padded_len, n_heads, d_v)
    return outputs[:, :seq_len, :, :]


def kda_sequential_reference(
    q: Array,
    k: Array,
    v: Array,
    alpha: Array,
    beta: Array,
) -> Array:
    """Token-by-token reference implementation of Eq. 1, for testing the chunked scan against.

    Deliberately independent of `kda_chunked_scan`'s chunking/WY-transform machinery --
    a direct transcription of S_t = (I - beta_t k_t k_t^T) Diag(alpha_t) S_{t-1} + beta_t k_t v_t^T,
    o_t = S_t^T q_t, applied one token at a time via jax.lax.scan.
    """
    batch, seq_len, n_heads, d_k = q.shape
    d_v = v.shape[-1]

    q_h = q.transpose(1, 0, 2, 3)
    k_h = k.transpose(1, 0, 2, 3)
    v_h = v.transpose(1, 0, 2, 3)
    alpha_h = alpha.transpose(1, 0, 2, 3)
    beta_h = beta.transpose(1, 0, 2)

    init_state = jnp.zeros((batch, n_heads, d_k, d_v), dtype=q.dtype)

    def step(
        state: Array, inputs: tuple[Array, Array, Array, Array, Array]
    ) -> tuple[Array, Array]:
        q_t, k_t, v_t, alpha_t, beta_t = inputs
        decayed = alpha_t[:, :, :, None] * state  # Diag(alpha_t) S_{t-1}, (B,H,dk,dv)
        k_decayed = jnp.einsum("bhd,bhde->bhe", k_t, decayed)  # k_t^T Diag(alpha_t) S_{t-1}, (B,H,dv)
        correction = jnp.einsum("bhd,bhe->bhde", k_t, k_decayed)  # k_t k_t^T Diag(alpha_t) S_{t-1}
        new_value = jnp.einsum("bhd,bhe->bhde", k_t, v_t)  # k_t v_t^T
        new_state = decayed - beta_t[:, :, None, None] * correction + beta_t[:, :, None, None] * new_value
        o_t = jnp.einsum("bhd,bhde->bhe", q_t, new_state)  # S_t^T q_t
        return new_state, o_t

    _final_state, outputs = jax.lax.scan(step, init_state, (q_h, k_h, v_h, alpha_h, beta_h))
    return outputs.transpose(1, 0, 2, 3)  # (seq, B, H, dv) -> (B, seq, H, dv)


class _PerHeadLinear(nnx.Module):
    """A separate `in_dim -> out_dim` linear map per head, no bias.

    Used for the low-rank "up" projections (decay gate, output gate) in `KDALayer`
    -- the paper specifies these as per-head projections of rank == head_dim, which
    `nnx.Linear` can't express directly since it has no head axis.
    """

    def __init__(self, n_heads: int, in_dim: int, out_dim: int, *, init_std: float, rngs: nnx.Rngs) -> None:
        init = nnx.initializers.normal(init_std)
        self.weight = nnx.Param(init(rngs.params(), (n_heads, in_dim, out_dim)))

    def __call__(self, x: Array) -> Array:
        return jnp.einsum("...hi,hio->...ho", x, self.weight[...])


class KDALayer(nnx.Module):
    """Kimi Delta Attention layer: q/k/v/gate projections around `kda_chunked_scan`.

    See ADR 0009 for what's included vs. excluded relative to the paper's full
    production parameterization (ShortConv excluded; L2Norm/Swish/low-rank gating/
    per-head RMSNorm included as load-bearing for the recurrence itself).
    """

    def __init__(self, config: ModelConfig, *, rngs: nnx.Rngs) -> None:
        self.config = config
        n_heads, head_dim = config.n_heads, config.head_dim
        init = nnx.initializers.normal(config.init_std)
        residual_init = nnx.initializers.normal(config.init_std * config.residual_init_scale)

        self.q_proj = nnx.Linear(config.embed_dim, config.embed_dim, use_bias=False, kernel_init=init, rngs=rngs)
        self.k_proj = nnx.Linear(config.embed_dim, config.embed_dim, use_bias=False, kernel_init=init, rngs=rngs)
        self.v_proj = nnx.Linear(config.embed_dim, config.embed_dim, use_bias=False, kernel_init=init, rngs=rngs)
        self.beta_proj = nnx.Linear(config.embed_dim, n_heads, use_bias=True, kernel_init=init, rngs=rngs)

        self.alpha_down = nnx.Linear(
            config.embed_dim, n_heads * head_dim, use_bias=False, kernel_init=init, rngs=rngs
        )
        self.alpha_up = _PerHeadLinear(n_heads, head_dim, head_dim, init_std=config.init_std, rngs=rngs)
        # Mamba2/GDN-style log-space decay parameterization (GDN paper's g = exp(-exp(A_log) *
        # softplus(a + dt_bias))), made per-channel here rather than per-head scalar -- this is
        # KDA's actual extension over GDN. A_log init follows Mamba2's convention of sampling the
        # per-channel decay *rate* from a moderate range so initial decay isn't degenerate (alpha
        # near 0 or 1 for every channel at init).
        self.alpha_log_decay = nnx.Param(
            jnp.log(nnx.initializers.uniform(scale=1.0)(rngs.params(), (n_heads, head_dim)) * 15.0 + 1.0)
        )
        self.dt_bias = nnx.Param(jnp.zeros((n_heads, head_dim)))

        self.gate_down = nnx.Linear(
            config.embed_dim, n_heads * head_dim, use_bias=False, kernel_init=init, rngs=rngs
        )
        self.gate_up = _PerHeadLinear(n_heads, head_dim, head_dim, init_std=config.init_std, rngs=rngs)

        self.head_norm = RMSNorm(head_dim)
        self.o_proj = nnx.Linear(
            config.embed_dim, config.embed_dim, use_bias=False, kernel_init=residual_init, rngs=rngs
        )

    def __call__(self, x: Array) -> Array:
        batch, seq_len, _ = x.shape
        n_heads, head_dim = self.config.n_heads, self.config.head_dim

        q = l2norm(jax.nn.silu(self.q_proj(x)).reshape(batch, seq_len, n_heads, head_dim))
        k = l2norm(jax.nn.silu(self.k_proj(x)).reshape(batch, seq_len, n_heads, head_dim))
        v = jax.nn.silu(self.v_proj(x)).reshape(batch, seq_len, n_heads, head_dim)
        beta = jax.nn.sigmoid(self.beta_proj(x))

        alpha_pre = self.alpha_up(self.alpha_down(x).reshape(batch, seq_len, n_heads, head_dim))
        decay_rate = jnp.exp(self.alpha_log_decay[...])
        alpha = jnp.exp(-decay_rate * jax.nn.softplus(alpha_pre + self.dt_bias[...]))

        o = kda_chunked_scan(q, k, v, alpha, beta)
        o = self.head_norm(o)

        gate_pre = self.gate_up(self.gate_down(x).reshape(batch, seq_len, n_heads, head_dim))
        o = jax.nn.sigmoid(gate_pre) * o

        return self.o_proj(o.reshape(batch, seq_len, n_heads * head_dim))


class KDABlock(nnx.Module):
    """Transformer block with `KDALayer` token-mixing instead of causal self-attention.

    No RoPE/cos/sin argument (ADR 0001: NoPE on KDA layers -- position is carried
    implicitly through the gating mechanism's decay).
    """

    def __init__(self, config: ModelConfig, *, rngs: nnx.Rngs) -> None:
        self.attn_norm = RMSNorm(config.embed_dim)
        self.attn = KDALayer(config, rngs=rngs)
        self.ffn_norm = RMSNorm(config.embed_dim)
        self.ffn = SwiGLU(config, rngs=rngs)

    def __call__(self, x: Array) -> Array:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x
