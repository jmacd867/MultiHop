"""Best score reachable on the corrected task WITHOUT traversing the Chain.

Usage: `python scripts/shortcut_ceiling.py [--original]`

## Why this exists

This generator has produced five shortcuts, each invisible in the accuracy
number and each found only after a run had already been spent on it (ADR 0008,
0012, 0015, 0017, 0018). ADR 0018 states the lesson: after fixing a shortcut,
the question is not "is the old shortcut gone" but "what is the best achievable
score without the capability I am trying to measure" -- computed fresh, against
the fixed generator, *before* spending GPU time.

This script is that computation. Every strategy below is a concrete decision
rule that a model could implement using only surface statistics -- occurrence
counts, subject/object roles, block positions -- and none of them follow the
Chain from the Query to its terminal. The maximum over them is the bar any
reported accuracy must clear before it is evidence of retrieval.

Strategies are scored per Grid Cell so the bar can be applied cell-wise, which
matters because the cells differ enormously: distance=0 admits no Distractors
at all, and hop_count=1 admits no chain-continuation signal (ADR 0017).
"""

import sys
from collections import Counter

import numpy as np

from multihop.data.generator import (
    DISTANCES,
    MAX_HOP_COUNT,
    MIN_HOP_COUNT,
    NUM_SPECIAL_TOKENS,
    SEP_TOKEN,
    GeneratorConfig,
    generate_example,
    reference_distractor_count,
    required_sequence_length,
)

ENTITY_VOCAB_SIZE = 8_000
N = 400
HOP_COUNTS = tuple(range(MIN_HOP_COUNT, MAX_HOP_COUNT + 1))


def blocks_of(tokens: list[int], query_position: int) -> list[tuple[int, int, int]]:
    """(index, subject, object) for every Fact-shaped block before the Query."""
    return [
        (i, int(tokens[i]), int(tokens[i + 1]))
        for i in range(query_position - 1)
        if tokens[i + 2] == SEP_TOKEN and tokens[i] >= NUM_SPECIAL_TOKENS
    ]


def strategy_scores(
    hop_count: int, distance: int, rng: np.random.Generator, randomized: bool
) -> dict[str, float]:
    """Accuracy of each traversal-free strategy on one cell."""
    config = GeneratorConfig(
        hop_count=hop_count,
        distance=distance,
        distractor_count=reference_distractor_count(distance),
        sequence_length=required_sequence_length(hop_count, distance),
        vocab_size=ENTITY_VOCAB_SIZE,
        randomize_gaps=randomized,
        randomize_fact_order=randomized,
    )
    hits: dict[str, float] = {}
    rank_hits: Counter[int] = Counter()

    for _ in range(N):
        example = generate_example(config, rng)
        tokens = list(example.tokens)
        answer = example.answer
        query = int(tokens[example.query_position])
        blocks = blocks_of(tokens, example.query_position)
        subjects = {s for _i, s, _o in blocks}
        objects = [o for _i, _s, o in blocks]

        # 1. Appears exactly once (ADR 0012): the Chain terminal is never a
        #    subject, but neither is any Distractor target.
        counts = Counter(t for t in tokens[: example.query_position] if t >= NUM_SPECIAL_TOKENS)
        once = [e for e, c in counts.items() if c == 1 and e != query]
        if once and answer in once:
            hits["appears_once"] = hits.get("appears_once", 0.0) + 1.0 / len(once)

        # 2. Never appears as a subject -- same candidate set by construction,
        #    kept separate because a model could implement either.
        never_subject = [o for o in set(objects) if o not in subjects]
        if never_subject and answer in never_subject:
            hits["never_subject"] = hits.get("never_subject", 0.0) + 1.0 / len(never_subject)

        # 3. Never a subject AND its own subject appears as an object. This
        #    filters Distractors hanging off the Query entity, whose subject
        #    (c_0) is never an object. Uses role statistics only -- no Chain
        #    following -- but is the strongest surface rule available.
        refined = [
            o for _i, s, o in blocks if o not in subjects and s in set(objects)
        ]
        refined = list(set(refined))
        if refined and answer in refined:
            hits["never_subject_linked"] = (
                hits.get("never_subject_linked", 0.0) + 1.0 / len(refined)
            )

        # 4. Last block's object, and 5. first block's object.
        if blocks:
            hits["last_block"] = hits.get("last_block", 0.0) + float(blocks[-1][2] == answer)
            hits["first_block"] = hits.get("first_block", 0.0) + float(blocks[0][2] == answer)
            for rank, (_i, _s, obj) in enumerate(blocks):
                if obj == answer:
                    rank_hits[rank] += 1

    scores = {k: v / N for k, v in hits.items()}
    scores["best_fixed_rank"] = (max(rank_hits.values()) / N) if rank_hits else 0.0
    return scores


def main(randomized: bool) -> None:
    label = "CORRECTED (randomize_gaps + randomize_fact_order)" if randomized else "ORIGINAL"
    print(f"Best traversal-free accuracy per cell -- {label}\n")
    names = [
        "appears_once",
        "never_subject",
        "never_subject_linked",
        "last_block",
        "first_block",
        "best_fixed_rank",
    ]
    header = f"{'cell':<9}" + "".join(f"{n[:13]:>15}" for n in names) + f"{'MAX':>8}"
    print(header)
    print("-" * len(header))

    per_cell_max: dict[tuple[int, int], float] = {}
    for hop in HOP_COUNTS:
        for distance in DISTANCES:
            rng = np.random.default_rng(100 + hop * 10 + distance)
            scores = strategy_scores(hop, distance, rng, randomized)
            best = max(scores.values())
            per_cell_max[(hop, distance)] = best
            row = "".join(f"{scores.get(n, 0.0):>15.3f}" for n in names)
            print(f"({hop},{distance}){'':<4}" + row + f"{best:>8.3f}")

    valid = [c for c in per_cell_max if c[1] != 0 and c[0] != 1]
    print()
    print(f"mean of per-cell maxima, all 25 cells      : "
          f"{sum(per_cell_max.values()) / 25:.4f}")
    print(f"mean of per-cell maxima, VALID surface     : "
          f"{sum(per_cell_max[c] for c in valid) / len(valid):.4f}   "
          f"(hop>=2, distance>=3; {len(valid)} cells)")
    print()
    print("Any reported accuracy at or below its cell's MAX is consistent with a")
    print("traversal-free strategy and is NOT evidence of Chain retrieval.")


if __name__ == "__main__":
    main(randomized="--original" not in sys.argv[1:])
