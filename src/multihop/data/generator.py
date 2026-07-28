"""Synthetic multi-hop chained key-value retrieval task generator.

See CONTEXT.md at the repo root for the domain vocabulary (Chain, Hop
Count, Fact, Query, Distractor, Filler, Distance, Sequence Length) this
module implements.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

FILLER_TOKEN = 0
SEP_TOKEN = 1
QUERY_TOKEN = 2
NUM_SPECIAL_TOKENS = 3

# The studied hop_count range. Single source of truth for _validate below,
# train.py's cell sampling, and eval.py's degradation grid -- all three must
# stay in lockstep or the eval grid silently stops covering what training
# actually sampled.
MIN_HOP_COUNT = 1
MAX_HOP_COUNT = 5

Split = Literal["train", "eval"]

# The distance grid swept by both training's mixed-cell sampling and eval's
# degradation grid (train.py's sample_cell, eval.py's evaluate_grid).
DISTANCES: tuple[int, ...] = (0, 3, 9, 21, 45)
_REFERENCE_DISTRACTOR_COUNT = 2  # ADR 0004


class InfeasibleConfigError(ValueError):
    """Raised when a GeneratorConfig's parameters can't be realized together."""


@dataclass(frozen=True)
class GeneratorConfig:
    hop_count: int
    distance: int
    distractor_count: int
    sequence_length: int
    vocab_size: int


@dataclass(frozen=True)
class Example:
    tokens: npt.NDArray[np.int32]
    answer: int
    query_position: int
    chain_entities: tuple[int, ...]
    fact_spans: tuple[tuple[int, int], ...]
    distractor_spans: tuple[tuple[int, int], ...]


def required_sequence_length(hop_count: int, distance: int) -> int:
    """Token length implied by hop_count/distance: (hop_count+1) gaps + 3 tokens/fact + 2-token query."""
    return (hop_count + 1) * distance + 3 * hop_count + 2


def max_distractor_capacity(hop_count: int, distance: int) -> int:
    """Total distractor blocks (3 tokens each) that fit across all hop_count+1 gaps."""
    return (hop_count + 1) * (distance // 3)


def entity_pool(split: Split, vocab_size: int) -> tuple[int, int]:
    """Return (start_id, size) of the disjoint entity id range for a split."""
    if split == "train":
        return (NUM_SPECIAL_TOKENS, vocab_size)
    if split == "eval":
        return (NUM_SPECIAL_TOKENS + vocab_size, vocab_size)
    raise ValueError(f"unknown split {split!r}")


def total_vocab_size(entity_vocab_size: int) -> int:
    """Model vocab size implied by an entity_vocab_size: special tokens + disjoint train/eval pools."""
    return NUM_SPECIAL_TOKENS + 2 * entity_vocab_size


def _validate(config: GeneratorConfig) -> int:
    if not (MIN_HOP_COUNT <= config.hop_count <= MAX_HOP_COUNT):
        raise InfeasibleConfigError(
            f"hop_count must be in the studied range {MIN_HOP_COUNT}..{MAX_HOP_COUNT}"
        )
    if config.distance < 0:
        raise InfeasibleConfigError("distance must be >= 0")
    if config.distractor_count < 0:
        raise InfeasibleConfigError("distractor_count must be >= 0")

    required_len = required_sequence_length(config.hop_count, config.distance)
    if config.sequence_length != required_len:
        raise InfeasibleConfigError(
            f"sequence_length={config.sequence_length} does not match the length "
            f"implied by hop_count={config.hop_count}, distance={config.distance} "
            f"(required {required_len})"
        )

    capacity = max_distractor_capacity(config.hop_count, config.distance)
    if config.distractor_count > capacity:
        raise InfeasibleConfigError(
            f"distractor_count={config.distractor_count} exceeds capacity={capacity} "
            f"for hop_count={config.hop_count}, distance={config.distance}"
        )

    needed_pool = config.hop_count + 1 + (1 if config.distractor_count > 0 else 0)
    if config.vocab_size < needed_pool:
        raise InfeasibleConfigError(
            f"vocab_size={config.vocab_size} is too small to hold a {config.hop_count}-hop "
            f"chain plus distractor entities (need >= {needed_pool})"
        )

    return required_len


def generate_example(
    config: GeneratorConfig, split: Split, rng: np.random.Generator
) -> Example:
    _validate(config)

    pool_start, pool_size = entity_pool(split, config.vocab_size)
    chain_entities = tuple(
        int(pool_start + i) for i in rng.choice(pool_size, size=config.hop_count + 1, replace=False)
    )
    chain_set = set(chain_entities)

    distractor_count = config.distractor_count
    if distractor_count > 0:
        chain_array = np.fromiter(chain_set, dtype=np.int64, count=len(chain_set))
        decoy_objects = pool_start + rng.integers(0, pool_size, size=distractor_count)
        collides = np.isin(decoy_objects, chain_array)
        while collides.any():
            decoy_objects[collides] = pool_start + rng.integers(
                0, pool_size, size=int(collides.sum())
            )
            collides = np.isin(decoy_objects, chain_array)
        decoy_subject_hop = rng.integers(0, config.hop_count, size=distractor_count)
    else:
        decoy_objects = np.empty(0, dtype=int)
        decoy_subject_hop = np.empty(0, dtype=int)

    num_gaps = config.hop_count + 1
    per_gap_capacity = config.distance // 3
    tickets = [gap for gap in range(num_gaps) for _ in range(per_gap_capacity)]
    rng.shuffle(tickets)
    chosen_gaps = tickets[:distractor_count]

    gap_to_distractors: dict[int, list[int]] = {gap: [] for gap in range(num_gaps)}
    for distractor_index, gap in enumerate(chosen_gaps):
        gap_to_distractors[gap].append(distractor_index)

    tokens: list[int] = []
    fact_spans: list[tuple[int, int]] = []
    distractor_spans: list[tuple[int, int]] = []

    def emit_gap(gap_index: int) -> None:
        distractor_indices = gap_to_distractors[gap_index]
        filler_count = config.distance - 3 * len(distractor_indices)
        items: list[tuple[str, int]] = [("filler", -1)] * filler_count + [
            ("distractor", d) for d in distractor_indices
        ]
        order = rng.permutation(len(items)) if items else np.empty(0, dtype=int)
        for item_index in order:
            kind, distractor_index = items[int(item_index)]
            if kind == "filler":
                tokens.append(FILLER_TOKEN)
            else:
                subject = chain_entities[int(decoy_subject_hop[distractor_index])]
                obj = int(decoy_objects[distractor_index])
                start = len(tokens)
                tokens.extend([subject, obj, SEP_TOKEN])
                distractor_spans.append((start, start + 3))

    emit_gap(0)
    for hop in range(config.hop_count):
        start = len(tokens)
        tokens.extend([chain_entities[hop], chain_entities[hop + 1], SEP_TOKEN])
        fact_spans.append((start, start + 3))
        emit_gap(hop + 1)

    tokens.append(QUERY_TOKEN)
    tokens.append(chain_entities[0])
    query_position = len(tokens) - 1

    return Example(
        tokens=np.array(tokens, dtype=np.int32),
        answer=chain_entities[-1],
        query_position=query_position,
        chain_entities=chain_entities,
        fact_spans=tuple(fact_spans),
        distractor_spans=tuple(distractor_spans),
    )


def reference_distractor_count(distance: int) -> int:
    """Fixed reference distractor count per ADR 0004: 2 for distance>=3, 0 at distance=0."""
    return 0 if distance == 0 else _REFERENCE_DISTRACTOR_COUNT


def generate_batch(
    rng: np.random.Generator,
    hop_count: int,
    distance: int,
    entity_vocab_size: int,
    batch_size: int,
    split: Split = "train",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a batch for one (hop_count, distance) cell. Shared by training (train.py) and eval (eval.py)."""
    distractor_count = reference_distractor_count(distance)
    sequence_length = required_sequence_length(hop_count, distance)

    gen_config = GeneratorConfig(
        hop_count=hop_count,
        distance=distance,
        distractor_count=distractor_count,
        sequence_length=sequence_length,
        vocab_size=entity_vocab_size,
    )
    examples = [generate_example(gen_config, split=split, rng=rng) for _ in range(batch_size)]
    tokens = np.stack([example.tokens for example in examples])
    query_positions = np.array([example.query_position for example in examples])
    answers = np.array([example.answer for example in examples])
    return tokens, query_positions, answers
