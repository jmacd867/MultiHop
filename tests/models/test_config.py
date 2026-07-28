import pytest

from multihop.models.config import ModelConfig


def test_head_dim_computed_from_embed_dim_and_n_heads() -> None:
    config = ModelConfig(embed_dim=768, n_heads=12)
    assert config.head_dim == 64


def test_residual_init_scale_matches_1_over_sqrt_2n() -> None:
    config = ModelConfig(n_layers=12)
    assert config.residual_init_scale == pytest.approx((2 * 12) ** -0.5)
