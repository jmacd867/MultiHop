import numpy as np
from flax import nnx

from multihop.data.generator import DISTANCES, NUM_SPECIAL_TOKENS
from multihop.eval import HOP_COUNTS, accuracy_at_cell, evaluate_grid
from multihop.models.baseline import FullAttentionBaseline
from multihop.models.config import ModelConfig

ENTITY_VOCAB_SIZE = 20


def make_model() -> FullAttentionBaseline:
    model_vocab_size = NUM_SPECIAL_TOKENS + ENTITY_VOCAB_SIZE
    config = ModelConfig(
        n_layers=1, embed_dim=16, n_heads=2, ffn_dim=32, max_seq_len=300, vocab_size=model_vocab_size
    )
    return FullAttentionBaseline(config, rngs=nnx.Rngs(0))


def test_accuracy_at_cell_is_a_valid_fraction() -> None:
    model = make_model()
    rng = np.random.default_rng(0)
    accuracy = accuracy_at_cell(
        model, hop_count=2, distance=3, entity_vocab_size=ENTITY_VOCAB_SIZE, n_examples=8, rng=rng
    )
    assert 0.0 <= accuracy <= 1.0


def test_evaluate_grid_covers_every_hop_count_distance_cell() -> None:
    model = make_model()
    rng = np.random.default_rng(0)
    grid = evaluate_grid(model, ENTITY_VOCAB_SIZE, rng, n_examples_per_cell=4)

    expected_keys = {(hop_count, distance) for hop_count in HOP_COUNTS for distance in DISTANCES}
    assert set(grid.keys()) == expected_keys
    assert all(0.0 <= accuracy <= 1.0 for accuracy in grid.values())
