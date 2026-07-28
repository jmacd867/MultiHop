import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from multihop.models.config import ModelConfig
from multihop.models.pure_kda import PureKDAModel


def make_model(**overrides: object) -> tuple[PureKDAModel, ModelConfig]:
    defaults: dict[str, object] = dict(
        n_layers=2,
        embed_dim=32,
        n_heads=4,
        ffn_dim=64,
        max_seq_len=20,
        vocab_size=50,
        positional_encoding="none",
    )
    defaults.update(overrides)
    config = ModelConfig(**defaults)  # type: ignore[arg-type]
    return PureKDAModel(config, rngs=nnx.Rngs(0)), config


def test_forward_pass_output_shape() -> None:
    model, config = make_model()
    tokens = jnp.zeros((3, 10), dtype=jnp.int32)
    logits = model(tokens)
    assert logits.shape == (3, 10, config.vocab_size)


def test_rejects_rope_positional_encoding() -> None:
    config = ModelConfig(
        n_layers=1, embed_dim=16, n_heads=2, ffn_dim=32, max_seq_len=8, vocab_size=20,
        positional_encoding="rope",
    )
    with pytest.raises(ValueError, match="positional_encoding"):
        PureKDAModel(config, rngs=nnx.Rngs(0))


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
    np.testing.assert_allclose(logits_a[:, :-1, :], logits_b[:, :-1, :], atol=1e-4)


def test_exceeding_max_seq_len_raises() -> None:
    model, config = make_model()
    tokens = jnp.zeros((1, config.max_seq_len + 1), dtype=jnp.int32)
    with pytest.raises(ValueError, match="max_seq_len"):
        model(tokens)
