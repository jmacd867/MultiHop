"""RMSNorm, SwiGLU FFN, and causal self-attention blocks (Flax NNX).

Norm/activation/bias choices (RMSNorm, SwiGLU, no linear bias) and the
1/sqrt(2*n_layers) residual init scaling follow the precedent set in the
KDA/linear-attention literature this project compares against.
"""

import math

import jax
import jax.numpy as jnp
from flax import nnx
from jax import Array

from multihop.models.config import ModelConfig
from multihop.models.rope import apply_rope


class RMSNorm(nnx.Module):
    def __init__(self, dim: int, *, eps: float = 1e-6) -> None:
        self.eps = eps
        self.weight = nnx.Param(jnp.ones((dim,)))

    def __call__(self, x: Array) -> Array:
        variance = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        normed = x * jax.lax.rsqrt(variance + self.eps)
        return normed * self.weight[...]


class SwiGLU(nnx.Module):
    def __init__(self, config: ModelConfig, *, rngs: nnx.Rngs) -> None:
        init = nnx.initializers.normal(config.init_std)
        residual_init = nnx.initializers.normal(config.init_std * config.residual_init_scale)
        self.gate_proj = nnx.Linear(
            config.embed_dim, config.ffn_dim, use_bias=False, kernel_init=init, rngs=rngs
        )
        self.up_proj = nnx.Linear(
            config.embed_dim, config.ffn_dim, use_bias=False, kernel_init=init, rngs=rngs
        )
        self.down_proj = nnx.Linear(
            config.ffn_dim, config.embed_dim, use_bias=False, kernel_init=residual_init, rngs=rngs
        )

    def __call__(self, x: Array) -> Array:
        return self.down_proj(jax.nn.silu(self.gate_proj(x)) * self.up_proj(x))


class CausalSelfAttention(nnx.Module):
    """Standard multi-head causal self-attention with RoPE.

    When `capture_attention=True`, sows per-head pre-softmax scores and
    post-softmax weights into `nnx.Intermediate` (ADR 0005) at zero cost
    when disabled. Retrieve via `nnx.capture(model, nnx.Intermediate)(...)`.
    """

    def __init__(self, config: ModelConfig, *, rngs: nnx.Rngs) -> None:
        self.config = config
        init = nnx.initializers.normal(config.init_std)
        residual_init = nnx.initializers.normal(config.init_std * config.residual_init_scale)
        self.q_proj = nnx.Linear(
            config.embed_dim, config.embed_dim, use_bias=False, kernel_init=init, rngs=rngs
        )
        self.k_proj = nnx.Linear(
            config.embed_dim, config.embed_dim, use_bias=False, kernel_init=init, rngs=rngs
        )
        self.v_proj = nnx.Linear(
            config.embed_dim, config.embed_dim, use_bias=False, kernel_init=init, rngs=rngs
        )
        self.o_proj = nnx.Linear(
            config.embed_dim, config.embed_dim, use_bias=False, kernel_init=residual_init, rngs=rngs
        )

    def __call__(
        self,
        x: Array,
        cos: Array | None = None,
        sin: Array | None = None,
        *,
        capture_attention: bool = False,
    ) -> Array:
        batch, seq_len, _ = x.shape
        n_heads = self.config.n_heads
        head_dim = self.config.head_dim

        q = self.q_proj(x).reshape(batch, seq_len, n_heads, head_dim)
        k = self.k_proj(x).reshape(batch, seq_len, n_heads, head_dim)
        v = self.v_proj(x).reshape(batch, seq_len, n_heads, head_dim)

        # cos/sin are optional because a NoPE variant has no rotary tables to
        # pass at all (ADR 0001's rule is per-variant: the hybrid's
        # full-attention layers run NoPE, same as its KDA layers). Erroring on
        # the rope-without-tables case keeps that from degrading silently into
        # an unrotated -- i.e. positionless -- attention layer.
        if self.config.positional_encoding == "rope":
            if cos is None or sin is None:
                raise ValueError(
                    "positional_encoding='rope' requires cos and sin, got None -- "
                    "a RoPE layer cannot run without rotary tables"
                )
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)

        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        scores = jnp.einsum("bhqd,bhkd->bhqk", q, k) * (1.0 / math.sqrt(head_dim))
        causal_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))

        # `where=` masks without ever materializing -inf, so the masked branch
        # can't produce the NaN gradients that `jnp.where(mask, scores, -inf)`
        # would (0 * -inf terms in the softmax backward pass). `scores` itself
        # stays the genuine, unmasked dot product at every position -- what
        # ADR 0005's capture below actually wants.
        weights = jax.nn.softmax(scores, axis=-1, where=causal_mask)

        if capture_attention:
            self.sow(nnx.Intermediate, "attn_scores", scores)
            self.sow(nnx.Intermediate, "attn_weights", weights)

        out = jnp.einsum("bhqk,bhkd->bhqd", weights, v)
        out = out.transpose(0, 2, 1, 3).reshape(batch, seq_len, n_heads * head_dim)
        return self.o_proj(out)


class TransformerBlock(nnx.Module):
    def __init__(self, config: ModelConfig, *, rngs: nnx.Rngs) -> None:
        self.attn_norm = RMSNorm(config.embed_dim)
        self.attn = CausalSelfAttention(config, rngs=rngs)
        self.ffn_norm = RMSNorm(config.embed_dim)
        self.ffn = SwiGLU(config, rngs=rngs)

    def __call__(
        self,
        x: Array,
        cos: Array | None = None,
        sin: Array | None = None,
        *,
        capture_attention: bool = False,
    ) -> Array:
        x = x + self.attn(self.attn_norm(x), cos, sin, capture_attention=capture_attention)
        x = x + self.ffn(self.ffn_norm(x))
        return x
