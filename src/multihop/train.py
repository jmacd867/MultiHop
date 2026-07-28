"""Training loop for the full-attention baseline (and, eventually, the KDA/hybrid variants).

Everything here is step-indexed, not epoch-indexed: examples are generated
on-the-fly per step (no fixed training corpus, no epoch concept), so LR
schedule, checkpoint cadence, and eval cadence are all expressed in steps.
See ADR 0003 for why on-the-fly generation is what makes the degradation
grid measure capacity rather than being confounded by example-level
memorization, and ADR 0008 for why entities are drawn from one shared
pool rather than disjoint train/eval ranges.

Each training step samples a single (hop_count, distance) cell for the
whole batch (uniform over hop_count 1-5, and over the ADR-justified
distance set), rather than mixing cells within a batch. This sidesteps
padding/length-bucketing entirely -- every example in a batch already has
the same `required_sequence_length` -- while still mixing across the full
grid over the course of training, which is what ADR 0003 actually requires.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx, traverse_util
from safetensors import safe_open
from safetensors.numpy import load_file, save_file

from multihop.data.generator import (
    DISTANCES,
    MAX_HOP_COUNT,
    MIN_HOP_COUNT,
    generate_batch,
    reference_distractor_count,
    total_vocab_size,
)
from multihop.models.baseline import FullAttentionBaseline

# Weight on the answer-position loss term relative to the mean shifted-position
# loss (see compute_loss). 1.0: the single answer position is deliberately
# weighted as heavily as the average of every other position combined, since
# it's the one prediction the whole experiment measures. Not empirically
# tuned -- a starting point to revisit if training curves look sluggish on
# accuracy specifically despite falling overall loss.
ANSWER_LOSS_WEIGHT = 1.0


@dataclass(frozen=True)
class TrainConfig:
    """`batch_size` is the effective batch size a single optimizer update is over.

    See ADR 0006: at batch_size=256 the naive (no-checkpointing, no mixed
    precision) backward pass OOMs / silently corrupts params to NaN on GB10.
    `grad_accum_steps` splits each effective batch into
    `batch_size // grad_accum_steps`-sized micro-batches, each run through its
    own forward+backward pass, with the resulting gradients averaged before a
    single optimizer update -- reaching the effective batch size the LR
    schedule and other hyperparameters were chosen for without the peak
    memory cost of one batch_size=256 backward pass.
    """

    peak_lr: float = 3e-4
    warmup_steps: int = 500
    total_steps: int = 20_000
    batch_size: int = 256
    grad_accum_steps: int = 16
    grad_clip: float = 1.0
    weight_decay: float = 0.1
    adam_b1: float = 0.9
    adam_b2: float = 0.95
    eval_every: int = 500
    checkpoint_every: int = 1000

    def __post_init__(self) -> None:
        if self.batch_size % self.grad_accum_steps != 0:
            raise ValueError(
                f"batch_size={self.batch_size} must be divisible by "
                f"grad_accum_steps={self.grad_accum_steps}"
            )

    @property
    def micro_batch_size(self) -> int:
        return self.batch_size // self.grad_accum_steps


def build_optimizer_tx(config: TrainConfig) -> optax.GradientTransformation:
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=config.peak_lr,
        warmup_steps=config.warmup_steps,
        decay_steps=config.total_steps,
        end_value=config.peak_lr * 0.1,
    )
    return optax.chain(
        optax.clip_by_global_norm(config.grad_clip),
        optax.adamw(
            learning_rate=schedule,
            b1=config.adam_b1,
            b2=config.adam_b2,
            weight_decay=config.weight_decay,
        ),
    )


def sample_cell(rng: np.random.Generator) -> tuple[int, int]:
    """Sample one (hop_count, distance) grid cell: hop_count uniform 1-5, distance uniform over DISTANCES."""
    hop_count = int(rng.integers(MIN_HOP_COUNT, MAX_HOP_COUNT + 1))
    distance = int(rng.choice(DISTANCES))
    return hop_count, distance


def sample_training_batch(
    rng: np.random.Generator, batch_size: int, vocab_size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Sample one (hop_count, distance) cell and generate a full batch from it."""
    hop_count, distance = sample_cell(rng)
    tokens, query_positions, answers = generate_batch(
        rng, hop_count, distance, vocab_size, batch_size
    )
    return tokens, query_positions, answers, hop_count, distance


def sample_training_microbatches(
    rng: np.random.Generator,
    micro_batch_size: int,
    grad_accum_steps: int,
    vocab_size: int,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], int, int]:
    """Sample one (hop_count, distance) cell, then draw `grad_accum_steps` independent micro-batches from it.

    All micro-batches share `required_sequence_length` by construction (a
    deterministic function of hop_count/distance), so they're always the
    same shape -- necessary for `train_step_accum`'s `@nnx.jit` to avoid
    retracing on every step. Asserted explicitly here so a violation of that
    invariant surfaces at the point of sampling, not as a confusing shape
    error deep inside train_step_accum.
    """
    hop_count, distance = sample_cell(rng)
    micro_batches = []
    expected_shape: tuple[int, ...] | None = None
    for _ in range(grad_accum_steps):
        tokens, _query_positions, answers = generate_batch(
            rng, hop_count, distance, vocab_size, micro_batch_size
        )
        if expected_shape is None:
            expected_shape = tokens.shape
        elif tokens.shape != expected_shape:
            raise AssertionError(
                f"micro-batch shape {tokens.shape} != expected {expected_shape} "
                f"for cell (hop_count={hop_count}, distance={distance}) -- "
                "micro-batches from the same cell must be identically shaped"
            )
        micro_batches.append((tokens, answers))
    return micro_batches, hop_count, distance


def compute_loss(
    model: FullAttentionBaseline,
    tokens: jnp.ndarray,
    answers: jnp.ndarray,
    answer_loss_weight: float = ANSWER_LOSS_WEIGHT,
) -> jnp.ndarray:
    """Full-sequence next-token cross-entropy, over every position including the terminal one.

    `tokens` ends at `query_position` (the query's start entity) with no
    token for the answer that follows it -- the model must predict the
    answer as the *next* token, which never appears in the input sequence.
    The standard tokens[:-1]->tokens[1:] shift therefore has no target for
    the last position at all, silently excluding it from the loss. This
    completes the "loss on every position" decision by using `answers` as
    the genuine next-token target for that final, otherwise-unsupervised
    position -- the one eval.py actually reads to score accuracy.

    The answer term is combined via `mean(shifted_loss) + ANSWER_LOSS_WEIGHT
    * answer_loss`, not concatenated into one flat array before averaging.
    Sequences in this mixed-cell training setup range from 5 to 287 tokens
    (ADR 0003's distance sweep); a flat concatenate-then-mean would let the
    single answer term's weight shrink to ~1/seq_len, diluting it more at
    long distances than short ones -- exactly the confound this experiment
    is trying to measure, reintroduced into the loss itself. Averaging the
    shifted positions on their own first makes that term length-invariant,
    so ANSWER_LOSS_WEIGHT sets a fixed, deliberate ratio between "the rest
    of the sequence" and "the answer" regardless of which cell was sampled.
    """
    logits = model(tokens)
    shifted_logits = logits[:, :-1, :]
    shifted_targets = tokens[:, 1:]
    shifted_loss = optax.softmax_cross_entropy_with_integer_labels(shifted_logits, shifted_targets)
    per_example_shifted_loss = jnp.mean(shifted_loss, axis=1)

    answer_logits = logits[:, -1, :]
    answer_loss = optax.softmax_cross_entropy_with_integer_labels(answer_logits, answers)

    per_example_loss = per_example_shifted_loss + answer_loss_weight * answer_loss
    return jnp.mean(per_example_loss)


@nnx.jit
def train_step(
    model: FullAttentionBaseline,
    optimizer: nnx.Optimizer[FullAttentionBaseline],
    tokens: jnp.ndarray,
    answers: jnp.ndarray,
) -> jnp.ndarray:
    loss, grads = nnx.value_and_grad(compute_loss)(model, tokens, answers)
    optimizer.update(model, grads)
    return jnp.asarray(loss)


@nnx.jit
def train_step_accum(
    model: FullAttentionBaseline,
    optimizer: nnx.Optimizer[FullAttentionBaseline],
    micro_batches: list[tuple[jnp.ndarray, jnp.ndarray]],
) -> jnp.ndarray:
    """Gradient-accumulated step: average gradients over `micro_batches`, then one optimizer update.

    `micro_batches` is a static-length Python list (its length is
    `grad_accum_steps`, fixed by TrainConfig), so this loop unrolls at trace
    time into `len(micro_batches)` forward+backward passes -- see ADR 0006.
    Averaging (not summing) the per-microbatch gradients, each already a
    mean over its own examples via compute_loss, matches the gradient a
    single true batch_size=256 step would produce (all micro-batches are
    equal-sized), so the existing peak_lr/grad_clip/schedule hyperparameters
    -- chosen assuming batch=256 semantics -- don't need re-tuning.
    """
    total_loss = jnp.zeros(())
    accumulated_grads = None
    for tokens, answers in micro_batches:
        loss, grads = nnx.value_and_grad(compute_loss)(model, tokens, answers)
        total_loss += loss
        if accumulated_grads is None:
            accumulated_grads = grads
        else:
            accumulated_grads = jax.tree.map(jnp.add, accumulated_grads, grads)

    grad_accum_steps = len(micro_batches)
    averaged_grads = jax.tree.map(lambda g: g / grad_accum_steps, accumulated_grads)
    optimizer.update(model, averaged_grads)
    return total_loss / grad_accum_steps


def _flatten_state(pure_state: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Flatten an nnx pure-dict pytree into safetensors' flat name->array form.

    optax's tuple-indexed state (e.g. `opt_state.1.0.mu.w`) surfaces as
    integer dict keys in the pure dict, which `traverse_util.flatten_dict`
    can't join into a path string -- stringify them first and restore the
    int keys on the way back in `_unflatten_state`.
    """

    def stringify_keys(node: Any) -> Any:
        if isinstance(node, dict):
            return {str(key): stringify_keys(value) for key, value in node.items()}
        return node

    flat = traverse_util.flatten_dict(stringify_keys(pure_state), sep=".")  # type: ignore[no-untyped-call]
    return {key: np.asarray(value) for key, value in flat.items()}


def _unflatten_state(flat: Mapping[str, np.ndarray]) -> dict[str, Any]:
    def intify_keys(node: Any) -> Any:
        if isinstance(node, dict):
            return {(int(key) if key.isdigit() else key): intify_keys(value) for key, value in node.items()}
        return node

    return dict(intify_keys(traverse_util.unflatten_dict(dict(flat), sep=".")))  # type: ignore[no-untyped-call]


def _checkpoint_paths(path: Path) -> tuple[Path, Path]:
    """Derive the two checkpoint filenames from an extensionless base path (e.g. `step_7`)."""
    return path.parent / f"{path.name}.model.safetensors", path.parent / f"{path.name}.optimizer.safetensors"


def save_checkpoint(model: FullAttentionBaseline, optimizer: nnx.Optimizer[FullAttentionBaseline], step: int, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model_path, optimizer_path = _checkpoint_paths(path)
    metadata = {"step": str(step)}

    model_state = _flatten_state(nnx.to_pure_dict(nnx.state(model)))
    save_file(model_state, model_path, metadata=metadata)

    optimizer_state = _flatten_state(nnx.to_pure_dict(nnx.state(optimizer)))
    save_file(optimizer_state, optimizer_path, metadata=metadata)


def load_checkpoint(model: FullAttentionBaseline, optimizer: nnx.Optimizer[FullAttentionBaseline], path: Path) -> int:
    model_path, optimizer_path = _checkpoint_paths(path)

    model_state = _unflatten_state(load_file(model_path))
    nnx.update(model, model_state)

    optimizer_state = _unflatten_state(load_file(optimizer_path))
    nnx.update(optimizer, optimizer_state)

    with safe_open(optimizer_path, framework="numpy") as f:
        metadata = f.metadata() or {}
    return int(metadata["step"])


def load_model_checkpoint(model: FullAttentionBaseline, path: Path) -> int:
    """Load only the trained weights from a checkpoint, never touching the optimizer file.

    This is the ADR 0007 model-only path: eval, ADR 0005 attention capture,
    and cross-variant weight comparison all want weights without paying for
    AdamW's ~2x-model-size moment arrays, and the step is readable from the
    model file's own metadata (duplicated there for exactly this reason).
    """
    model_path, _optimizer_path = _checkpoint_paths(path)

    model_state = _unflatten_state(load_file(model_path))
    nnx.update(model, model_state)

    with safe_open(model_path, framework="numpy") as f:
        metadata = f.metadata() or {}
    return int(metadata["step"])


EvalFn = Callable[[FullAttentionBaseline], Mapping[tuple[int, int], float]]


@dataclass
class TrainingHistory:
    """Record of what happened during a `train()` run, so loss and eval results are observable."""

    loss: list[tuple[int, float]] = field(default_factory=list)
    eval_grids: list[tuple[int, Mapping[tuple[int, int], float]]] = field(default_factory=list)


def train(
    model: FullAttentionBaseline,
    optimizer: nnx.Optimizer[FullAttentionBaseline],
    train_config: TrainConfig,
    vocab_size: int,
    rng: np.random.Generator,
    checkpoint_dir: Path,
    eval_fn: EvalFn | None = None,
) -> TrainingHistory:
    expected_vocab_size = total_vocab_size(vocab_size)
    if model.config.vocab_size != expected_vocab_size:
        raise ValueError(
            f"model.config.vocab_size={model.config.vocab_size} does not match "
            f"total_vocab_size(entity_vocab_size={vocab_size})={expected_vocab_size} "
            "-- the model's embedding table won't cover the token ids the generator produces"
        )

    history = TrainingHistory()

    for step in range(1, train_config.total_steps + 1):
        micro_batches, _hop_count, _distance = sample_training_microbatches(
            rng,
            train_config.micro_batch_size,
            train_config.grad_accum_steps,
            vocab_size,
        )
        jnp_micro_batches = [
            (jnp.asarray(tokens), jnp.asarray(answers)) for tokens, answers in micro_batches
        ]
        loss = train_step_accum(model, optimizer, jnp_micro_batches)
        loss_value = float(loss)
        history.loss.append((step, loss_value))
        print(f"step {step}: loss={loss_value:.4f}")

        if eval_fn is not None and step % train_config.eval_every == 0:
            grid = eval_fn(model)
            history.eval_grids.append((step, grid))
            mean_accuracy = sum(grid.values()) / len(grid)
            print(f"step {step}: eval mean accuracy={mean_accuracy:.4f}")

        if step % train_config.checkpoint_every == 0:
            save_checkpoint(model, optimizer, step, checkpoint_dir / f"step_{step}")

    return history
