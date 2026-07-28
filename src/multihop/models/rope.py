"""Rotary positional embeddings (RoPE), rotate-half convention.

Used only by the full-attention baseline (ADR 0001): this variant has no
KDA layer to carry positional information through data-dependent gating,
so it needs its own explicit positional encoding.
"""

import jax.numpy as jnp
from jax import Array


def rope_freqs(seq_len: int, head_dim: int, theta: float) -> tuple[Array, Array]:
    """cos/sin tables of shape (seq_len, head_dim // 2)."""
    inv_freq = 1.0 / (theta ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
    positions = jnp.arange(seq_len, dtype=jnp.float32)
    freqs = jnp.einsum("s,d->sd", positions, inv_freq)
    return jnp.cos(freqs), jnp.sin(freqs)


def apply_rope(x: Array, cos: Array, sin: Array) -> Array:
    """Apply RoPE to x of shape (batch, seq, n_heads, head_dim).

    cos/sin have shape (seq, head_dim // 2), broadcast over batch and heads.
    Position 0 has freqs == 0 (cos=1, sin=0), so it's an identity rotation.
    """
    x1, x2 = jnp.split(x, 2, axis=-1)
    cos_b = cos[None, :, None, :]
    sin_b = sin[None, :, None, :]
    rotated_1 = x1 * cos_b - x2 * sin_b
    rotated_2 = x2 * cos_b + x1 * sin_b
    return jnp.concatenate([rotated_1, rotated_2], axis=-1)
