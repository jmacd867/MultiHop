import math

import jax.numpy as jnp
import numpy as np

from multihop.models.rope import apply_rope, rope_freqs


def test_rope_freqs_shape() -> None:
    cos, sin = rope_freqs(seq_len=8, head_dim=4, theta=10_000.0)
    assert cos.shape == (8, 2)
    assert sin.shape == (8, 2)


def test_rope_identity_at_position_zero() -> None:
    # All frequency angles are 0 at position 0, so cos=1, sin=0 everywhere:
    # RoPE must be a no-op at the first sequence position.
    head_dim = 8
    cos, sin = rope_freqs(seq_len=1, head_dim=head_dim, theta=10_000.0)
    x = jnp.arange(head_dim, dtype=jnp.float32).reshape(1, 1, 1, head_dim)
    rotated = apply_rope(x, cos, sin)
    np.testing.assert_allclose(rotated, x, atol=1e-6)


def test_rope_known_angle_at_position_one() -> None:
    # head_dim=2 has a single frequency pair with inv_freq = theta**0 = 1,
    # so at position 1 the rotation angle is exactly 1 radian regardless of
    # theta. Expected output computed independently via math.cos/math.sin,
    # not by re-deriving the implementation's own formula.
    head_dim = 2
    cos, sin = rope_freqs(seq_len=2, head_dim=head_dim, theta=10_000.0)
    x = jnp.array([[[[0.0, 0.0]], [[1.0, 0.0]]]])  # (batch=1, seq=2, heads=1, head_dim=2)

    rotated = apply_rope(x, cos, sin)

    expected_position_one = jnp.array([math.cos(1.0), math.sin(1.0)])
    np.testing.assert_allclose(rotated[0, 1, 0], expected_position_one, atol=1e-6)
