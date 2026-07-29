"""Falsify what the trained models actually learned: traversal, or a shortcut?

Usage: `python scripts/probe_mechanism.py {baseline,kda,hybrid} [checkpoint_dir]`

## The claim under test

`scripts/verify_positional_invariant.py` establishes (by construction and by
measurement) that this task admits a fixed-offset positional solution:
`emit_gap` emits exactly `distance` tokens whatever it packs into them, and
Facts are exactly 3 tokens, so the answer always sits at
`query_position - (distance + 3)`. Copying that position scores 100% in every
Grid Cell, at every hop_count and every distractor_count, with no traversal.

That a shortcut *exists* does not establish that the models *use* it. Both
things have been true before in this project and the difference mattered: ADR
0008's failure was diagnosed only by testing the mechanism, not by inferring
it. This script tests it.

## Conditions

Each is generated fresh, evaluated on an archived final checkpoint. No
retraining; eval examples are generated on the fly, so this costs one forward
pass per cell.

- **control** -- unmodified generation. Must reproduce ~1.0000, otherwise the
  harness is wrong and nothing below means anything.
- **shuffle** -- the same Facts emitted in random order rather than chain
  order. Gaps, Distractors and the Query are untouched.
- **jitter** -- Facts stay in chain order; only the *final* gap's length is
  randomised, so the answer is no longer a fixed distance from the Query.
  A smaller perturbation than `shuffle`: it breaks the positional rule alone,
  leaving order intact.

## Reading the outcomes (at distance >= 3, where Distractors exist)

| condition | traversal | appears-once (ADR 0012) | positional rule |
|---|---|---|---|
| control | ~1.00 | ~0.33 | ~1.00 |
| shuffle | ~1.00 | ~0.33 | **~1/hop_count** |
| jitter  | ~1.00 | ~0.33 | **~0.00** |

The positional rule's signature under `shuffle` is not zero: the fixed offset
still lands on the answer whenever the answer's Fact happens to be emitted
last, which is 1/hop_count of the time. A *declining* profile across hop_count
is therefore the positive signature of the positional rule, and is more
diagnostic than a flat zero.

distance=0 cells cannot separate traversal from the appears-once shortcut
(no Distractors fit, so both score 1.0) and are reported but not interpreted.

## What a null result would and would not mean

These conditions are out of the training distribution. ADR 0003 forbids
reading OOD performance as a *capacity* measurement, and this is not one --
it is falsification of a mechanism claim already asserted in ADR 0014. Those
carry different evidentiary weight, and conflating them would be the actual
motivated reasoning. But the converse caveat stands and must be reported: a
low score under `shuffle` shows the model did not learn an order-invariant
solution; it does not by itself prove the model cannot traverse.
"""

import sys
from pathlib import Path
from typing import Literal, Protocol

import jax.numpy as jnp
import numpy as np
from flax import nnx

from multihop.data.generator import (
    DISTANCES,
    FILLER_TOKEN,
    MAX_HOP_COUNT,
    MIN_HOP_COUNT,
    NUM_SPECIAL_TOKENS,
    QUERY_TOKEN,
    SEP_TOKEN,
    reference_distractor_count,
    total_vocab_size,
)
from multihop.models.baseline import FullAttentionBaseline
from multihop.models.config import ModelConfig, PositionalEncoding
from multihop.models.hybrid import HybridModel
from multihop.models.pure_kda import PureKDAModel
from multihop.train import MultihopModel, load_model_checkpoint

ENTITY_VOCAB_SIZE = 8_000  # ADR 0008
N_EXAMPLES = 512  # matches EVAL_EXAMPLES_PER_CELL
PROBE_SEED = 7
HOP_COUNTS = tuple(range(MIN_HOP_COUNT, MAX_HOP_COUNT + 1))
Condition = Literal["control", "shuffle", "jitter_end", "jitter_all", "query_ablate"]


class ModelFactory(Protocol):
    def __call__(self, config: ModelConfig, *, rngs: nnx.Rngs) -> MultihopModel: ...


VARIANTS: dict[str, tuple[ModelFactory, PositionalEncoding]] = {
    "baseline": (FullAttentionBaseline, "rope"),  # ADR 0001
    "kda": (PureKDAModel, "none"),
    "hybrid": (HybridModel, "none"),
}


def make_example(
    rng: np.random.Generator, hop_count: int, distance: int, condition: Condition
) -> tuple[list[int], int, int]:
    """One example under a probe condition. Returns (tokens, query_position, answer).

    Mirrors `generate_example`'s layout deliberately rather than importing it,
    because the probe needs to vary emission order and gap length -- neither of
    which the shared generator exposes, and which must not be added to it while
    real runs depend on its exact behaviour.
    """
    distractor_count = reference_distractor_count(distance)
    pool = NUM_SPECIAL_TOKENS
    chain = [int(pool + i) for i in rng.choice(ENTITY_VOCAB_SIZE, size=hop_count + 1, replace=False)]
    chain_set = set(chain)

    decoys: list[int] = []
    while len(decoys) < distractor_count:
        candidate = int(pool + rng.integers(0, ENTITY_VOCAB_SIZE))
        if candidate not in chain_set:
            decoys.append(candidate)
    decoy_subject = [int(rng.integers(0, hop_count)) for _ in range(distractor_count)]

    per_gap_capacity = distance // 3
    num_gaps = hop_count + 1
    tickets = [g for g in range(num_gaps) for _ in range(per_gap_capacity)]
    rng.shuffle(tickets)
    chosen = tickets[:distractor_count]
    gap_to_distractors: dict[int, list[int]] = {g: [] for g in range(num_gaps)}
    for index, gap in enumerate(chosen):
        gap_to_distractors[gap].append(index)

    tokens: list[int] = []

    def emit_gap(gap_index: int, length: int) -> None:
        indices = gap_to_distractors[gap_index]
        filler = max(0, length - 3 * len(indices))
        items: list[tuple[str, int]] = [("filler", -1)] * filler + [("d", i) for i in indices]
        for position in rng.permutation(len(items)) if items else []:
            kind, index = items[int(position)]
            if kind == "filler":
                tokens.append(FILLER_TOKEN)
            else:
                tokens.extend([chain[decoy_subject[index]], decoys[index], SEP_TOKEN])

    # Fact k links chain[k] -> chain[k+1]; chain order is 0..hop_count-1.
    fact_order = list(range(hop_count))
    if condition == "shuffle":
        rng.shuffle(fact_order)

    # Gap lengths.
    #
    # `jitter_end` randomises only the FINAL gap. That breaks the *end*-relative
    # offset (query_position - (distance + 3)) but leaves every Fact's absolute
    # index untouched -- measured: the answer sits at one fixed index (e.g. 142
    # at (3,45)) while only qpos-answer varies. A start-relative rule therefore
    # survives it completely, which is why this condition alone cannot separate
    # "positional" from "traversal".
    #
    # `jitter_all` randomly partitions the same total budget (hop_count+1)*distance
    # across all gaps. Total sequence length is preserved exactly -- so cell
    # structure and the one-shape-per-batch invariant survive -- while both the
    # start-relative and end-relative offsets are destroyed.
    lengths = [distance] * (hop_count + 1)
    if condition == "jitter_end":
        lengths[hop_count] = int(rng.integers(0, 2 * distance + 4))
    elif condition == "jitter_all" and distance > 0:
        total = (hop_count + 1) * distance
        # Every gap must still hold the Distractors assigned to it.
        floors = [3 * len(gap_to_distractors[g]) for g in range(num_gaps)]
        spare = total - sum(floors)
        # spare == 0 happens at fully packed cells -- (1,3) is the only one at
        # the reference count (ADR 0013) -- where Distractors already consume
        # the entire gap budget and no redistribution is possible. That cell is
        # therefore un-jitterable and its `jitter_all` result equals `control`;
        # it must not be read as evidence either way.
        cuts = (
            sorted(int(x) for x in rng.integers(0, spare + 1, size=num_gaps - 1))
            if spare > 0
            else [0] * (num_gaps - 1)
        )
        parts, previous = [], 0
        for cut in [*cuts, spare]:
            parts.append(cut - previous)
            previous = cut
        lengths = [floors[g] + parts[g] for g in range(num_gaps)]

    emit_gap(0, lengths[0])
    for gap_index, hop in enumerate(fact_order, start=1):
        tokens.extend([chain[hop], chain[hop + 1], SEP_TOKEN])
        emit_gap(gap_index, lengths[gap_index])

    tokens.append(QUERY_TOKEN)
    # `query_ablate` asks the Query about an entity that never appears. A model
    # doing content-based retrieval has nothing to retrieve; one executing a
    # positional or appears-once rule is unaffected and still emits the old
    # answer. One token changed, every position preserved.
    if condition == "query_ablate":
        absent = int(pool + rng.integers(0, ENTITY_VOCAB_SIZE))
        while absent in chain_set or absent in decoys:
            absent = int(pool + rng.integers(0, ENTITY_VOCAB_SIZE))
        tokens.append(absent)
    else:
        tokens.append(chain[0])
    return tokens, len(tokens) - 1, chain[-1]


def accuracy(
    model: MultihopModel, rng: np.random.Generator, hop_count: int, distance: int, condition: Condition
) -> tuple[float, float]:
    """Returns (model accuracy, accuracy the fixed-offset positional rule would get)."""
    built = [make_example(rng, hop_count, distance, condition) for _ in range(N_EXAMPLES)]
    width = max(len(t) for t, _q, _a in built)
    # Right-pad past the Query. Both mechanisms under study are causal, so a
    # token after query_position cannot influence the logits read at it.
    tokens = np.full((len(built), width), FILLER_TOKEN, dtype=np.int32)
    for row, (seq, _q, _a) in enumerate(built):
        tokens[row, : len(seq)] = seq
    query_positions = np.array([q for _t, q, _a in built])
    answers = np.array([a for _t, _q, a in built])

    logits = model(jnp.asarray(tokens))
    predicted = np.asarray(jnp.argmax(logits[jnp.arange(len(built)), jnp.asarray(query_positions)], -1))

    offset = distance + 3
    rule = np.array([
        seq[q - offset] if q - offset >= 0 else -1 for seq, q, _a in built
    ])
    return float(np.mean(predicted == answers)), float(np.mean(rule == answers))


def main(variant: str, checkpoint: Path) -> None:
    factory, positional_encoding = VARIANTS[variant]
    config = ModelConfig(
        vocab_size=total_vocab_size(ENTITY_VOCAB_SIZE), positional_encoding=positional_encoding
    )
    model = factory(config, rngs=nnx.Rngs(0))
    step = load_model_checkpoint(model, checkpoint)
    print(f"variant={variant} checkpoint={checkpoint.name} step={step}\n", flush=True)

    conditions: tuple[Condition, ...] = ("control", "shuffle", "jitter_end", "jitter_all", "query_ablate")
    for condition in conditions:
        rng = np.random.default_rng(PROBE_SEED)  # same examples for every variant
        got: dict[tuple[int, int], float] = {}
        rule: dict[tuple[int, int], float] = {}
        for hop in HOP_COUNTS:
            for distance in DISTANCES:
                got[(hop, distance)], rule[(hop, distance)] = accuracy(
                    model, rng, hop, distance, condition
                )

        print(f"=== {condition} ===")
        header = "hop \\ dist" + "".join(f"{d:>16}" for d in DISTANCES)
        print(header)
        print("-" * len(header))
        for hop in HOP_COUNTS:
            print(f"{hop:>9} " + "".join(
                f"{got[(hop, d)]:>9.3f}/{rule[(hop, d)]:<6.3f}" for d in DISTANCES
            ))
        print("  cells show  model_accuracy/positional_rule_accuracy")

        scoring = [c for c in got if c[1] != 0]
        print(f"  model mean over scoring cells (distance>=3): "
              f"{np.mean([got[c] for c in scoring]):.4f}")
        # The positional rule's signature under `shuffle` is a decline across
        # hop_count (~1/hop_count), so the marginal is what distinguishes it
        # from traversal (flat ~1.0) and from appears-once (flat ~0.33).
        print("  model by hop_count (distance>=3): " + "  ".join(
            f"{h}:{np.mean([got[(h, d)] for d in DISTANCES if d != 0]):.3f}" for h in HOP_COUNTS
        ))
        print("  rule  by hop_count (distance>=3): " + "  ".join(
            f"{h}:{np.mean([rule[(h, d)] for d in DISTANCES if d != 0]):.3f}" for h in HOP_COUNTS
        ), flush=True)
        print()


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in VARIANTS:
        raise SystemExit(f"usage: {sys.argv[0]} {{{','.join(VARIANTS)}}} [checkpoint_base]")
    name = sys.argv[1]
    base = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(f"checkpoints/{name}_run/step_20000")
    main(name, base)
