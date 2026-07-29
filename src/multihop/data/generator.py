"""Synthetic multi-hop chained key-value retrieval task generator.

See CONTEXT.md at the repo root for the domain vocabulary (Chain, Hop
Count, Fact, Query, Distractor, Filler, Distance, Sequence Length) this
module implements.
"""

from dataclasses import dataclass

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
    randomize_fact_order: bool = False
    """Destroy the Fact-ordering shortcut (ADR 0018).

    Chain Facts are emitted between gaps while Distractors are emitted inside
    them, so with Facts in Chain order the answer-bearing Fact's *rank* among
    the Fact-shaped blocks stays concentrated even when gap lengths are
    randomised. Measured: "always answer the object of the k-th block" scores
    0.68 at (5,45) against 0.14 chance, and 1.00 at (1,3). Emitting Chain
    Facts in a random order collapses that to 0.155 -- essentially chance.

    `randomize_gaps` alone is therefore NOT sufficient to force traversal, and
    the two flags are meant to be set together. They are separate fields so
    each shortcut can be tested in isolation.
    """

    randomize_gaps: bool = False
    """Destroy the fixed-offset positional shortcut (ADR 0015).

    With the default `False`, every gap is exactly `distance` tokens long, so
    every Fact's index is a deterministic function of (hop_count, distance) and
    `answer_index == query_position - (distance + 3)` always holds. That makes
    the task 100% solvable by copying a fixed position, and both trained
    Variants were measured doing exactly that -- their accuracy under
    fact-order shuffling tracks the positional rule's to three decimals.

    With `True`, the same total gap budget `(hop_count + 1) * distance` is
    randomly partitioned across the gaps instead of split evenly. Total
    sequence length is therefore **unchanged**, which matters for two reasons:
    `required_sequence_length` stays correct, and
    `sample_training_microbatches`' guarantee that every micro-batch in a step
    shares one shape (which `train_step_accum`'s jit depends on) still holds.
    What changes is that neither the start-relative nor the end-relative offset
    to the answer is fixed any more.

    Defaults to False so every existing run, test, and recorded result keeps
    its exact semantics; this is opt-in for the corrected re-runs.
    """


@dataclass(frozen=True)
class Example:
    tokens: npt.NDArray[np.int32]
    answer: int
    query_position: int
    chain_entities: tuple[int, ...]
    fact_spans: tuple[tuple[int, int], ...]
    """Spans of the Chain Facts, **in emission order, not hop order**.

    With `randomize_fact_order` set (ADR 0018) the two differ: `fact_spans[i]`
    is the i-th Fact as laid out in the sequence, which is not hop i. Nothing
    consumes this today, but ADR 0005's attention capture is the obvious future
    consumer and would read per-hop attention from the wrong span. Zip it
    against the emission order, or re-derive the hop from the span's subject.
    """
    distractor_spans: tuple[tuple[int, int], ...]


def required_sequence_length(hop_count: int, distance: int) -> int:
    """Token length implied by hop_count/distance: (hop_count+1) gaps + 3 tokens/fact + 2-token query."""
    return (hop_count + 1) * distance + 3 * hop_count + 2


def max_distractor_capacity(hop_count: int, distance: int) -> int:
    """Total distractor blocks (3 tokens each) that fit across all hop_count+1 gaps."""
    return (hop_count + 1) * (distance // 3)


def total_vocab_size(entity_vocab_size: int) -> int:
    """Model vocab size implied by an entity_vocab_size: special tokens + the one shared entity pool.

    See ADR 0008: entities are drawn from a single shared pool for every
    example (train and freshly-generated eval alike), not disjoint
    train/eval ranges -- so there is exactly one pool to size, not two.
    """
    return NUM_SPECIAL_TOKENS + entity_vocab_size


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


def generate_example(config: GeneratorConfig, rng: np.random.Generator) -> Example:
    """Draw one example's entity bindings fresh from the single shared pool (ADR 0008).

    Every call -- whether this example is used for training or for eval --
    draws from the same entity id range. There is no reserved/held-out id
    range: genuine generalization here comes from the combinatorics of which
    entities get bound to which chain/distractor roles in this specific
    example, not from novel token ids.
    """
    _validate(config)

    pool_start, pool_size = NUM_SPECIAL_TOKENS, config.vocab_size
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

    # Per-gap lengths. Uniform `distance` by default; randomly partitioned when
    # `randomize_gaps` is set, keeping the total identical so sequence length
    # and shape-uniformity are preserved (see GeneratorConfig.randomize_gaps).
    gap_lengths = [config.distance] * num_gaps
    if config.randomize_gaps and config.distance > 0:
        # Each gap must still hold the Distractors assigned to it, so only the
        # Filler above that floor is free to move.
        floors = [3 * len(gap_to_distractors[gap]) for gap in range(num_gaps)]
        spare = num_gaps * config.distance - sum(floors)
        if spare > 0:
            cuts = sorted(int(c) for c in rng.integers(0, spare + 1, size=num_gaps - 1))
        else:
            # A fully packed cell -- (1,3) is the only one at the reference
            # count (ADR 0013) -- has no Filler to redistribute, so it stays
            # uniform and remains positionally solvable. Named in ADR 0015.
            cuts = [0] * (num_gaps - 1)
        previous = 0
        gap_lengths = []
        for cut in [*cuts, spare]:
            gap_lengths.append(cut - previous)
            previous = cut
        gap_lengths = [floors[gap] + gap_lengths[gap] for gap in range(num_gaps)]

    tokens: list[int] = []
    fact_spans: list[tuple[int, int]] = []
    distractor_spans: list[tuple[int, int]] = []

    def emit_gap(gap_index: int) -> None:
        distractor_indices = gap_to_distractors[gap_index]
        filler_count = gap_lengths[gap_index] - 3 * len(distractor_indices)
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

    # Emission order of the Chain Facts. In Chain order by default; shuffled
    # when `randomize_fact_order` is set, so the answer-bearing Fact's rank
    # among the Fact-shaped blocks carries no information (ADR 0018). The
    # answer is `chain_entities[-1]` regardless of the order they are emitted
    # in -- only the Chain's structure defines it, not its layout.
    fact_order = list(range(config.hop_count))
    if config.randomize_fact_order:
        rng.shuffle(fact_order)

    emit_gap(0)
    for gap_index, hop in enumerate(fact_order, start=1):
        start = len(tokens)
        tokens.extend([chain_entities[hop], chain_entities[hop + 1], SEP_TOKEN])
        fact_spans.append((start, start + 3))
        emit_gap(gap_index)

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
    randomize_gaps: bool = False,
    randomize_fact_order: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a batch for one (hop_count, distance) cell. Shared by training (train.py) and eval (eval.py).

    No `split` argument: every example draws from the same shared entity
    pool (ADR 0008). Training and eval differ only in which `rng` the
    caller passes in -- eval.py uses a fixed shared seed across model
    variants for cross-variant comparability, not a separate id range.
    """
    distractor_count = reference_distractor_count(distance)
    sequence_length = required_sequence_length(hop_count, distance)

    gen_config = GeneratorConfig(
        hop_count=hop_count,
        distance=distance,
        distractor_count=distractor_count,
        sequence_length=sequence_length,
        vocab_size=entity_vocab_size,
        randomize_gaps=randomize_gaps,
        randomize_fact_order=randomize_fact_order,
    )
    examples = [generate_example(gen_config, rng=rng) for _ in range(batch_size)]
    tokens = np.stack([example.tokens for example in examples])
    query_positions = np.array([example.query_position for example in examples])
    answers = np.array([example.answer for example in examples])
    return tokens, query_positions, answers
