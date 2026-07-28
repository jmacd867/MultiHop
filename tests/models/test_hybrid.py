"""Tests for the 3:1 hybrid variant (ADR 0011).

The layer-type-ordering test carries unusual weight here. `KDABlock` and
`TransformerBlock` take the same input shape and return the same output
shape, so a wrong interleave -- wrong ratio, wrong phase, or an off-by-one
in the schedule -- builds cleanly, trains, and produces plausible losses.
Nothing else in the pipeline would catch it, and it would be baked into
every checkpoint of the variant.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from multihop.models.baseline import FullAttentionBaseline
from multihop.models.config import ModelConfig
from multihop.models.hybrid import HybridModel, is_full_attention_layer
from multihop.models.kda import KDABlock
from multihop.models.layers import TransformerBlock
from multihop.models.pure_kda import PureKDAModel


def make_model(**overrides: object) -> tuple[HybridModel, ModelConfig]:
    defaults: dict[str, object] = dict(
        n_layers=8,
        embed_dim=32,
        n_heads=4,
        ffn_dim=64,
        max_seq_len=20,
        vocab_size=50,
        positional_encoding="none",
    )
    defaults.update(overrides)
    config = ModelConfig(**defaults)  # type: ignore[arg-type]
    return HybridModel(config, rngs=nnx.Rngs(0)), config


def param_count(model: nnx.Module) -> int:
    return sum(int(np.asarray(p).size) for p in jax.tree.leaves(nnx.state(model, nnx.Param)))


def test_forward_pass_output_shape() -> None:
    model, config = make_model()
    tokens = jnp.zeros((3, 10), dtype=jnp.int32)
    logits = model(tokens)
    assert logits.shape == (3, 10, config.vocab_size)


def test_layer_types_follow_the_adr_0011_schedule() -> None:
    """KDA,KDA,KDA,FullAttention repeating from layer 0 -- full attention at 3/7/11."""
    model, config = make_model(n_layers=12)
    actual = ["full" if isinstance(b, TransformerBlock) else "kda" for b in model.blocks]
    expected = ["kda", "kda", "kda", "full"] * 3
    assert actual == expected
    assert [i for i, kind in enumerate(actual) if kind == "full"] == [3, 7, 11]
    assert actual.count("kda") == 9
    assert config.n_layers == 12


def test_every_block_is_one_of_the_two_reused_block_types() -> None:
    """No forked or bespoke block type crept in (ADR 0011: reuse both as-is)."""
    model, _config = make_model(n_layers=12)
    assert all(isinstance(b, (KDABlock, TransformerBlock)) for b in model.blocks)


def test_final_layer_is_full_attention() -> None:
    """An accepted consequence of the schedule's phase, asserted so it can't drift silently."""
    model, _config = make_model(n_layers=12)
    assert isinstance(model.blocks[-1], TransformerBlock)


@pytest.mark.parametrize(
    ("layer_index", "expected"),
    [(0, False), (1, False), (2, False), (3, True), (4, False), (7, True), (11, True), (12, False)],
)
def test_is_full_attention_layer(layer_index: int, expected: bool) -> None:
    assert is_full_attention_layer(layer_index) is expected


@pytest.mark.parametrize("n_layers", [1, 2, 3, 5, 6, 10, 13])
def test_rejects_n_layers_that_is_not_a_whole_number_of_groups(n_layers: int) -> None:
    """A partial final group silently changes the ratio that defines this Variant.

    n_layers=6 would be 5:1 and n_layers=10 would be 4:1, while anything below
    4 yields zero full-attention layers -- a PureKDAModel under the hybrid's
    name. Every one of those builds and trains normally, so this has to be
    caught at construction or not at all.
    """
    with pytest.raises(ValueError, match="multiple of 4"):
        make_model(n_layers=n_layers)


@pytest.mark.parametrize("n_layers", [4, 8, 12, 16])
def test_accepts_whole_numbers_of_groups_and_keeps_the_3to1_ratio(n_layers: int) -> None:
    model, _config = make_model(n_layers=n_layers)
    kinds = ["full" if isinstance(b, TransformerBlock) else "kda" for b in model.blocks]
    assert kinds.count("full") == n_layers // 4
    assert kinds.count("kda") == 3 * (n_layers // 4)


def test_rejects_rope_positional_encoding() -> None:
    """ADR 0001 is per-variant: a hybrid's full-attention layers stay NoPE."""
    config = ModelConfig(
        n_layers=4, embed_dim=16, n_heads=2, ffn_dim=32, max_seq_len=8, vocab_size=20,
        positional_encoding="rope",
    )
    with pytest.raises(ValueError, match="positional_encoding"):
        HybridModel(config, rngs=nnx.Rngs(0))


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


def test_capture_attention_sows_only_from_the_full_attention_layers() -> None:
    """ADR 0005 instrumentation reaches the 3 full-attention layers; KDA layers sow nothing."""
    model, config = make_model(n_layers=8)
    tokens = jnp.zeros((2, 10), dtype=jnp.int32)
    _logits, intermediates = nnx.capture(model, nnx.Intermediate)(tokens, capture_attention=True)
    leaves = jax.tree.leaves(nnx.to_pure_dict(intermediates))
    n_full_attention_layers = sum(is_full_attention_layer(i) for i in range(config.n_layers))
    # Two sows per full-attention layer: attn_scores and attn_weights.
    assert len(leaves) == 2 * n_full_attention_layers
    for leaf in leaves:
        assert leaf.shape == (2, config.n_heads, 10, 10)


def test_param_count_is_the_baseline_plus_nine_kda_layer_deltas() -> None:
    """The hybrid's size is *derived* from the shared ModelConfig, not tuned to a target.

    ADR 0009 fixes depth/width/head-count identical across variants, so the
    hybrid's param count is fully determined: it is the baseline's, plus the
    per-layer KDA gating delta for each of the 9 layers that became KDA. A
    mismatch means something structural differs between the variants, which is
    exactly the confound ModelConfig exists to prevent -- not a delta to accept.
    """
    shared: dict[str, object] = dict(
        n_layers=12, embed_dim=32, n_heads=4, ffn_dim=64, max_seq_len=20, vocab_size=50
    )
    baseline = FullAttentionBaseline(
        ModelConfig(**shared, positional_encoding="rope"),  # type: ignore[arg-type]
        rngs=nnx.Rngs(0),
    )
    pure_kda = PureKDAModel(
        ModelConfig(**shared, positional_encoding="none"),  # type: ignore[arg-type]
        rngs=nnx.Rngs(0),
    )
    hybrid = HybridModel(
        ModelConfig(**shared, positional_encoding="none"),  # type: ignore[arg-type]
        rngs=nnx.Rngs(0),
    )

    per_layer_kda_delta = (param_count(pure_kda) - param_count(baseline)) // 12
    n_kda_layers = 12 - sum(is_full_attention_layer(i) for i in range(12))
    assert n_kda_layers == 9
    assert param_count(hybrid) == param_count(baseline) + n_kda_layers * per_layer_kda_delta
