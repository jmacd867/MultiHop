import numpy as np
import pytest

from multihop.data.generator import (
    FILLER_TOKEN,
    NUM_SPECIAL_TOKENS,
    QUERY_TOKEN,
    SEP_TOKEN,
    Example,
    GeneratorConfig,
    InfeasibleConfigError,
    generate_example,
    max_distractor_capacity,
    reference_distractor_count,
    required_sequence_length,
)


def make_config(**overrides: object) -> GeneratorConfig:
    defaults: dict[str, object] = dict(
        hop_count=3,
        distance=6,
        distractor_count=2,
        sequence_length=required_sequence_length(3, 6),
        vocab_size=50,
    )
    defaults.update(overrides)
    return GeneratorConfig(**defaults)  # type: ignore[arg-type]


def test_required_sequence_length_formula() -> None:
    # (hop_count + 1) gaps of `distance` tokens, 3 tokens per chain fact, 2-token query block
    assert required_sequence_length(hop_count=3, distance=6) == 4 * 6 + 3 * 3 + 2
    assert required_sequence_length(hop_count=1, distance=0) == 2 * 0 + 3 * 1 + 2


def test_max_distractor_capacity_formula() -> None:
    # each gap holds floor(distance / 3) distractor blocks, across (hop_count + 1) gaps
    assert max_distractor_capacity(hop_count=3, distance=6) == 4 * 2
    assert max_distractor_capacity(hop_count=1, distance=0) == 0


def test_generate_example_raises_on_sequence_length_mismatch() -> None:
    config = make_config(sequence_length=required_sequence_length(3, 6) + 1)
    rng = np.random.default_rng(0)
    with pytest.raises(InfeasibleConfigError):
        generate_example(config, rng=rng)


def test_generate_example_raises_on_distractor_capacity_exceeded() -> None:
    distance = 6
    hop_count = 3
    capacity = max_distractor_capacity(hop_count, distance)
    config = make_config(
        hop_count=hop_count,
        distance=distance,
        distractor_count=capacity + 1,
        sequence_length=required_sequence_length(hop_count, distance),
    )
    rng = np.random.default_rng(0)
    with pytest.raises(InfeasibleConfigError):
        generate_example(config, rng=rng)


def test_generate_example_raises_on_distractors_with_no_capacity_gaps() -> None:
    # hop_count=1, distance=0 leaves zero distractor capacity
    config = make_config(
        hop_count=1,
        distance=0,
        distractor_count=1,
        sequence_length=required_sequence_length(1, 0),
    )
    rng = np.random.default_rng(0)
    with pytest.raises(InfeasibleConfigError):
        generate_example(config, rng=rng)


def test_generate_example_raises_on_hop_count_above_studied_range() -> None:
    config = make_config(
        hop_count=6,
        distance=6,
        sequence_length=required_sequence_length(6, 6),
        distractor_count=0,
    )
    rng = np.random.default_rng(0)
    with pytest.raises(InfeasibleConfigError):
        generate_example(config, rng=rng)


def test_generate_example_raises_on_insufficient_vocab_pool() -> None:
    config = make_config(hop_count=3, vocab_size=3)  # needs at least hop_count + 1 entities
    rng = np.random.default_rng(0)
    with pytest.raises(InfeasibleConfigError):
        generate_example(config, rng=rng)


def test_chain_entities_are_distinct_and_correct_count() -> None:
    config = make_config()
    rng = np.random.default_rng(0)
    example = generate_example(config, rng=rng)
    assert len(example.chain_entities) == config.hop_count + 1
    assert len(set(example.chain_entities)) == config.hop_count + 1


def test_fact_spans_encode_correct_chain_transitions() -> None:
    config = make_config()
    rng = np.random.default_rng(0)
    example = generate_example(config, rng=rng)
    assert len(example.fact_spans) == config.hop_count
    for i, (start, end) in enumerate(example.fact_spans):
        assert end - start == 3
        subj, obj, sep = example.tokens[start:end]
        assert subj == example.chain_entities[i]
        assert obj == example.chain_entities[i + 1]
        assert sep == SEP_TOKEN


def test_answer_is_chain_terminal_and_query_position_correct() -> None:
    config = make_config()
    rng = np.random.default_rng(0)
    example = generate_example(config, rng=rng)
    assert example.answer == example.chain_entities[-1]
    assert example.tokens[example.query_position] == example.chain_entities[0]
    assert example.tokens[example.query_position - 1] == QUERY_TOKEN
    assert example.query_position == len(example.tokens) - 1


def test_distractor_spans_share_subject_with_chain_but_do_not_continue_it() -> None:
    config = make_config()
    rng = np.random.default_rng(0)
    example = generate_example(config, rng=rng)
    assert len(example.distractor_spans) == config.distractor_count
    chain_subjects = set(example.chain_entities[:-1])
    chain_pairs = {
        (example.chain_entities[i], example.chain_entities[i + 1])
        for i in range(config.hop_count)
    }
    for start, end in example.distractor_spans:
        assert end - start == 3
        subj, obj, sep = example.tokens[start:end]
        assert sep == SEP_TOKEN
        assert subj in chain_subjects
        assert (subj, obj) not in chain_pairs


def test_spans_do_not_overlap() -> None:
    config = make_config()
    rng = np.random.default_rng(0)
    example = generate_example(config, rng=rng)
    occupied = np.zeros(len(example.tokens), dtype=bool)
    for start, end in [*example.fact_spans, *example.distractor_spans]:
        assert not occupied[start:end].any()
        occupied[start:end] = True


def test_total_sequence_length_matches_config() -> None:
    config = make_config()
    rng = np.random.default_rng(0)
    example = generate_example(config, rng=rng)
    assert len(example.tokens) == config.sequence_length


def test_all_examples_draw_from_the_single_shared_entity_pool() -> None:
    # ADR 0008: there is exactly one entity pool now, not disjoint
    # train/eval ranges -- every drawn entity id must fall inside
    # [NUM_SPECIAL_TOKENS, NUM_SPECIAL_TOKENS + vocab_size), regardless of
    # which call produced it.
    config = make_config(distractor_count=0)
    rng = np.random.default_rng(0)
    examples = [generate_example(config, rng=rng) for _ in range(20)]
    entities = {e for ex in examples for e in ex.chain_entities}
    assert entities.issubset(range(NUM_SPECIAL_TOKENS, NUM_SPECIAL_TOKENS + config.vocab_size))


def test_generation_is_reproducible_given_seeded_rng() -> None:
    config = make_config()
    example_a = generate_example(config, rng=np.random.default_rng(42))
    example_b = generate_example(config, rng=np.random.default_rng(42))
    assert np.array_equal(example_a.tokens, example_b.tokens)
    assert example_a.answer == example_b.answer


def test_distractor_objects_valid_with_minimal_vocab_pool() -> None:
    # vocab_size = hop_count + 2 leaves exactly one non-chain entity, forcing the
    # rejection-sampling loop in generate_example to resample repeatedly.
    hop_count = 3
    distance = 6
    config = make_config(
        hop_count=hop_count,
        distance=distance,
        distractor_count=max_distractor_capacity(hop_count, distance),
        sequence_length=required_sequence_length(hop_count, distance),
        vocab_size=hop_count + 2,
    )
    rng = np.random.default_rng(0)
    example = generate_example(config, rng=rng)
    chain_set = set(example.chain_entities)
    for start, end in example.distractor_spans:
        _subj, obj, _sep = example.tokens[start:end]
        assert obj not in chain_set


def test_no_filler_tokens_leak_into_answer_or_query_block() -> None:
    config = make_config()
    rng = np.random.default_rng(0)
    example = generate_example(config, rng=rng)
    assert example.tokens[example.query_position] != FILLER_TOKEN
    assert example.tokens[example.query_position - 1] == QUERY_TOKEN


def _answer_index(example: Example) -> int:
    """Index of the answer token within the sequence (it is the last Fact's object)."""
    positions = np.where(example.tokens[: example.query_position] == example.answer)[0]
    return int(positions[-1])


@pytest.mark.parametrize("hop_count", [1, 2, 3, 5])
@pytest.mark.parametrize("distance", [3, 9, 45])
def test_default_generation_has_the_fixed_offset_positional_shortcut(
    hop_count: int, distance: int
) -> None:
    """ADR 0015: with uniform gaps the answer is always query_position - (distance + 3).

    Asserted rather than merely documented because it is the property that
    invalidated the first full experiment -- both trained Variants solved the
    task by copying this position instead of traversing the Chain. A future
    change that silently removed it would make old results incomparable, and
    one that silently reintroduced it elsewhere would repeat the failure.
    """
    rng = np.random.default_rng(0)
    config = GeneratorConfig(
        hop_count=hop_count,
        distance=distance,
        distractor_count=reference_distractor_count(distance),
        sequence_length=required_sequence_length(hop_count, distance),
        vocab_size=200,
    )
    for _ in range(50):
        example = generate_example(config, rng)
        assert _answer_index(example) == example.query_position - (distance + 3)


@pytest.mark.parametrize("hop_count", [2, 3, 5])
@pytest.mark.parametrize("distance", [9, 45])
def test_randomize_gaps_destroys_both_positional_offsets_without_changing_length(
    hop_count: int, distance: int
) -> None:
    """The corrected generator must break start- AND end-relative offsets.

    Randomising only the final gap would break the end-relative offset while
    leaving every Fact at a fixed absolute index, so a start-relative rule
    would survive. Both are checked here. Sequence length must be unchanged, or
    `required_sequence_length` becomes wrong and
    `sample_training_microbatches`' one-shape-per-step guarantee (which
    `train_step_accum`'s jit relies on) breaks.
    """
    rng = np.random.default_rng(0)
    config = GeneratorConfig(
        hop_count=hop_count,
        distance=distance,
        distractor_count=reference_distractor_count(distance),
        sequence_length=required_sequence_length(hop_count, distance),
        vocab_size=200,
        randomize_gaps=True,
    )
    expected_length = required_sequence_length(hop_count, distance)
    absolute, relative = set(), set()
    for _ in range(60):
        example = generate_example(config, rng)
        assert len(example.tokens) == expected_length
        index = _answer_index(example)
        absolute.add(index)
        relative.add(example.query_position - index)

    assert len(absolute) > 1, "answer stayed at a fixed absolute index (start-relative rule intact)"
    assert len(relative) > 1, "answer stayed at a fixed offset from the query (end-relative intact)"


def test_randomize_gaps_still_places_every_requested_distractor() -> None:
    """Redistributing Filler must not squeeze out a Distractor."""
    rng = np.random.default_rng(0)
    for hop_count, distance in [(1, 9), (3, 21), (5, 45)]:
        count = reference_distractor_count(distance)
        config = GeneratorConfig(
            hop_count=hop_count,
            distance=distance,
            distractor_count=count,
            sequence_length=required_sequence_length(hop_count, distance),
            vocab_size=200,
            randomize_gaps=True,
        )
        for _ in range(30):
            example = generate_example(config, rng)
            assert len(example.distractor_spans) == count
            assert len(example.fact_spans) == hop_count
