import jax.numpy as jnp
import numpy as np
from flax import nnx

from multihop.models.baseline import FullAttentionBaseline
from multihop.models.config import ModelConfig


def make_model(**overrides: object) -> tuple[FullAttentionBaseline, ModelConfig]:
    defaults: dict[str, object] = dict(
        n_layers=2, embed_dim=32, n_heads=4, ffn_dim=64, max_seq_len=20, vocab_size=50
    )
    defaults.update(overrides)
    config = ModelConfig(**defaults)  # type: ignore[arg-type]
    return FullAttentionBaseline(config, rngs=nnx.Rngs(0)), config


def test_forward_pass_output_shape() -> None:
    model, config = make_model()
    tokens = jnp.zeros((3, 10), dtype=jnp.int32)
    logits = model(tokens)
    assert logits.shape == (3, 10, config.vocab_size)


def test_tied_embeddings_have_no_separate_head() -> None:
    model, _config = make_model(tie_embeddings=True)
    assert not hasattr(model, "head")


def test_untied_embeddings_create_separate_head() -> None:
    model, _config = make_model(tie_embeddings=False)
    assert hasattr(model, "head")


def test_causal_masking_future_tokens_do_not_affect_earlier_logits() -> None:
    model, config = make_model()
    rng = np.random.default_rng(0)
    tokens_a = jnp.asarray(rng.integers(0, config.vocab_size, size=(1, 8)), dtype=jnp.int32)
    last_token = int(tokens_a[0, -1])
    tokens_b = tokens_a.at[0, -1].set((last_token + 1) % config.vocab_size)

    logits_a = model(tokens_a)
    logits_b = model(tokens_b)
    np.testing.assert_allclose(logits_a[:, :-1, :], logits_b[:, :-1, :], atol=1e-5)


def test_capture_attention_produces_per_layer_per_head_pre_and_post_softmax() -> None:
    model, config = make_model()
    batch, seq_len = 2, 6
    tokens = jnp.zeros((batch, seq_len), dtype=jnp.int32)

    _logits, intermediates = nnx.capture(model, nnx.Intermediate)(tokens, capture_attention=True)

    for layer_index in range(config.n_layers):
        layer_attn = intermediates["blocks"][layer_index]["attn"]
        scores = layer_attn["attn_scores"].get_value()[0]  # type: ignore[operator]
        weights = layer_attn["attn_weights"].get_value()[0]  # type: ignore[operator]

        assert scores.shape == (batch, config.n_heads, seq_len, seq_len)
        assert weights.shape == (batch, config.n_heads, seq_len, seq_len)
        # post-softmax weights are a valid probability distribution over keys
        np.testing.assert_allclose(np.asarray(weights).sum(axis=-1), 1.0, atol=1e-5)


def test_capture_attention_off_by_default() -> None:
    model, _config = make_model()
    tokens = jnp.zeros((2, 6), dtype=jnp.int32)
    _logits, intermediates = nnx.capture(model, nnx.Intermediate)(tokens)
    assert len(intermediates) == 0
