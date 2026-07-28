import jax.numpy as jnp
import numpy as np

from multihop.models.kda import kda_chunked_scan, kda_sequential_reference, l2norm


def _random_inputs(
    rng: np.random.Generator, batch: int, seq_len: int, n_heads: int, d_k: int, d_v: int
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    q = jnp.asarray(rng.normal(size=(batch, seq_len, n_heads, d_k)), dtype=jnp.float32)
    k = jnp.asarray(rng.normal(size=(batch, seq_len, n_heads, d_k)), dtype=jnp.float32)
    v = jnp.asarray(rng.normal(size=(batch, seq_len, n_heads, d_v)), dtype=jnp.float32)
    # alpha in (0, 1): a decay gate at exactly 0 or 1 is an edge case tested separately.
    alpha = jnp.asarray(rng.uniform(0.5, 0.999, size=(batch, seq_len, n_heads, d_k)), dtype=jnp.float32)
    beta = jnp.asarray(rng.uniform(0.1, 0.9, size=(batch, seq_len, n_heads)), dtype=jnp.float32)
    return q, k, v, alpha, beta


def test_shapes() -> None:
    rng = np.random.default_rng(0)
    batch, seq_len, n_heads, d_k, d_v = 2, 17, 3, 8, 5
    q, k, v, alpha, beta = _random_inputs(rng, batch, seq_len, n_heads, d_k, d_v)
    o = kda_chunked_scan(q, k, v, alpha, beta, chunk_size=8)
    assert o.shape == (batch, seq_len, n_heads, d_v)


def test_hand_computed_single_token() -> None:
    # One token, one head: S_0 = 0, so S_1 = beta * k v^T exactly (the decay/correction
    # terms both vanish against a zero initial state), and o_1 = S_1^T q = beta * (k.q) * v.
    q = jnp.array([[[[1.0, 2.0]]]])  # (batch=1, seq=1, heads=1, d_k=2)
    k = jnp.array([[[[0.5, -1.0]]]])
    v = jnp.array([[[[3.0, 4.0, 5.0]]]])  # d_v=3
    alpha = jnp.array([[[[0.7, 0.9]]]])
    beta = jnp.array([[[0.4]]])

    expected = float(beta[0, 0, 0]) * (0.5 * 1.0 + -1.0 * 2.0) * jnp.array([3.0, 4.0, 5.0])

    chunked = kda_chunked_scan(q, k, v, alpha, beta, chunk_size=32)
    sequential = kda_sequential_reference(q, k, v, alpha, beta)

    np.testing.assert_allclose(chunked[0, 0, 0], expected, atol=1e-5)
    np.testing.assert_allclose(sequential[0, 0, 0], expected, atol=1e-5)


def test_chunked_scan_matches_sequential_reference() -> None:
    rng = np.random.default_rng(1)
    batch, seq_len, n_heads, d_k, d_v = 2, 50, 3, 6, 4
    q, k, v, alpha, beta = _random_inputs(rng, batch, seq_len, n_heads, d_k, d_v)

    chunked = kda_chunked_scan(q, k, v, alpha, beta, chunk_size=8)
    sequential = kda_sequential_reference(q, k, v, alpha, beta)

    np.testing.assert_allclose(chunked, sequential, atol=1e-4, rtol=1e-4)


def test_chunked_scan_matches_reference_with_ragged_final_chunk() -> None:
    # seq_len=50 with chunk_size=32 leaves a final chunk of length 18 -- exercises the
    # ragged-final-chunk path (ADR 0009) rather than only exact multiples of chunk_size.
    rng = np.random.default_rng(2)
    batch, seq_len, n_heads, d_k, d_v = 1, 50, 2, 4, 4
    q, k, v, alpha, beta = _random_inputs(rng, batch, seq_len, n_heads, d_k, d_v)

    chunked = kda_chunked_scan(q, k, v, alpha, beta, chunk_size=32)
    sequential = kda_sequential_reference(q, k, v, alpha, beta)

    np.testing.assert_allclose(chunked, sequential, atol=1e-4, rtol=1e-4)


def test_result_independent_of_chunk_size() -> None:
    # The chunked-parallel algorithm is a reorganization of the same recurrence, so
    # the output must not depend on the (arbitrary) chunking granularity.
    rng = np.random.default_rng(3)
    batch, seq_len, n_heads, d_k, d_v = 2, 40, 2, 6, 3
    q, k, v, alpha, beta = _random_inputs(rng, batch, seq_len, n_heads, d_k, d_v)

    out_c8 = kda_chunked_scan(q, k, v, alpha, beta, chunk_size=8)
    out_c16 = kda_chunked_scan(q, k, v, alpha, beta, chunk_size=16)
    out_c1 = kda_chunked_scan(q, k, v, alpha, beta, chunk_size=1)

    np.testing.assert_allclose(out_c8, out_c16, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(out_c8, out_c1, atol=1e-4, rtol=1e-4)


def test_causality_future_tokens_do_not_affect_earlier_outputs() -> None:
    rng = np.random.default_rng(4)
    batch, seq_len, n_heads, d_k, d_v = 1, 20, 2, 4, 4
    q, k, v, alpha, beta = _random_inputs(rng, batch, seq_len, n_heads, d_k, d_v)

    v_perturbed = v.at[:, -1, :, :].set(v[:, -1, :, :] + 100.0)

    out_a = kda_chunked_scan(q, k, v, alpha, beta, chunk_size=8)
    out_b = kda_chunked_scan(q, k, v_perturbed, alpha, beta, chunk_size=8)

    np.testing.assert_allclose(out_a[:, :-1], out_b[:, :-1], atol=1e-5)


def test_l2norm_produces_unit_vectors() -> None:
    rng = np.random.default_rng(5)
    x = jnp.asarray(rng.normal(size=(3, 5, 4, 8)), dtype=jnp.float32)
    normed = l2norm(x)
    norms = jnp.sqrt(jnp.sum(jnp.square(normed), axis=-1))
    np.testing.assert_allclose(norms, jnp.ones_like(norms), atol=1e-4)
