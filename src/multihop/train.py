"""Training loop shared by all three Variants (full attention, pure KDA, hybrid).

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

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, TypeVar

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
from multihop.models.config import ModelConfig


class MultihopModel(Protocol):
    """Structural type every Variant (full attention, pure KDA, hybrid) satisfies.

    Everything in this module is written against this Protocol rather than a
    concrete model class so `train_step_accum`, `compute_loss`, checkpointing, etc.
    are genuinely shared across variants -- not re-implemented per variant, only
    called with a different model instance. See the pure-KDA task's "reuse
    train_step_accum, the shared generator, and the eval harness as-is".
    """

    config: ModelConfig

    def __call__(self, token_ids: jnp.ndarray) -> jnp.ndarray: ...


ModelT = TypeVar("ModelT", bound=MultihopModel)

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
    model: ModelT,
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
    model: ModelT,
    optimizer: nnx.Optimizer[ModelT],
    tokens: jnp.ndarray,
    answers: jnp.ndarray,
) -> jnp.ndarray:
    loss, grads = nnx.value_and_grad(compute_loss)(model, tokens, answers)
    optimizer.update(model, grads)
    return jnp.asarray(loss)


@nnx.jit
def train_step_accum(
    model: ModelT,
    optimizer: nnx.Optimizer[ModelT],
    micro_batches: list[tuple[jnp.ndarray, jnp.ndarray]],
) -> jnp.ndarray:
    """Gradient-accumulated step: average gradients over `micro_batches`, then one optimizer update.

    Averaging (not summing) the per-microbatch gradients, each already a
    mean over its own examples via compute_loss, matches the gradient a
    single true batch_size=256 step would produce (all micro-batches are
    equal-sized), so the existing peak_lr/grad_clip/schedule hyperparameters
    -- chosen assuming batch=256 semantics -- don't need re-tuning.

    The accumulation loop is a `jax.lax.scan` over the stacked micro-batches,
    not the Python-unrolled loop ADR 0006 originally described. Unrolling
    inlined `grad_accum_steps` copies of a full forward+backward pass into one
    jaxpr, which the full-attention baseline could afford but the KDA variant
    could not: its per-layer graph already contains a scanned matrix inversion
    and WY-transform intermediates (ADR 0009), and 16-32 inlined copies of a
    12-layer backward pass exhausted GB10's memory *during compilation*, before
    any step could run. See ADR 0010. Scanning compiles the accumulation body
    once regardless of `grad_accum_steps`, so compile cost no longer scales
    with it.

    `micro_batches` is a static-length Python list of identically-shaped
    (tokens, answers) pairs -- guaranteed by `sample_training_microbatches`,
    which draws every micro-batch from one (hop_count, distance) cell -- so
    stacking them into leading-axis arrays for the scan is always well-defined.
    """
    grad_accum_steps = len(micro_batches)
    stacked_tokens = jnp.stack([tokens for tokens, _answers in micro_batches])
    stacked_answers = jnp.stack([answers for _tokens, answers in micro_batches])

    # lax.scan needs a pure function of arrays, so split the model into a static
    # graphdef plus its differentiable params and re-merge inside the body. The
    # graphdef is a trace-time constant, so this costs nothing at runtime.
    graphdef, params, rest = nnx.split(model, nnx.Param, ...)  # type: ignore[misc]

    def micro_batch_loss(
        params_state: Any, tokens: jnp.ndarray, answers: jnp.ndarray
    ) -> jnp.ndarray:
        merged = nnx.merge(graphdef, params_state, rest)
        return compute_loss(merged, tokens, answers)

    def accumulate(
        carry: tuple[Any, jnp.ndarray], batch: tuple[jnp.ndarray, jnp.ndarray]
    ) -> tuple[tuple[Any, jnp.ndarray], None]:
        accumulated_grads, total_loss = carry
        tokens, answers = batch
        loss, grads = jax.value_and_grad(micro_batch_loss)(params, tokens, answers)
        return (jax.tree.map(jnp.add, accumulated_grads, grads), total_loss + loss), None

    init_grads = jax.tree.map(jnp.zeros_like, params)
    (accumulated_grads, total_loss), _ = jax.lax.scan(
        accumulate, (init_grads, jnp.zeros(())), (stacked_tokens, stacked_answers)
    )

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


def save_checkpoint(
    model: ModelT,
    optimizer: nnx.Optimizer[ModelT],
    step: int,
    path: Path,
    rngs: Mapping[str, np.random.Generator] | None = None,
) -> None:
    """Write the model and optimizer files, plus enough state to resume exactly.

    `rngs` is what makes resumption *exact* rather than merely possible, and
    it is not optional in practice for this experiment. `train()` draws every
    (hop_count, distance) cell and every generated example from one
    `np.random.Generator`, and the eval harness draws its per-cell examples
    from another. Restarting either from its seed after an interruption would
    give the resumed run a different data stream than an uninterrupted one --
    a different sequence of Grid Cells to train on, and (because the eval
    generator is threaded across all 40 eval passes) a different set of
    examples in the *final* Degradation Grid. That last one silently breaks
    the cross-variant comparison this project exists to make: CONTEXT.md
    defines a Variant as differing from the others only in its sequence-mixing
    mechanism, and ADR 0008 makes identical per-cell eval examples the
    mechanism protecting that. A resume without RNG state would produce a
    grid that looks fine and is not comparable.

    Generator state (`bit_generator.state`) is a plain JSON-serializable dict,
    so it rides in the safetensors metadata map ADR 0007 already established
    for `step` -- no sidecar file, and written to both files for the same
    reason `step` is.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    model_path, optimizer_path = _checkpoint_paths(path)
    metadata = {"step": str(step)}
    if rngs is not None:
        metadata["rng_states"] = json.dumps(
            {name: generator.bit_generator.state for name, generator in rngs.items()}
        )

    model_state = _flatten_state(nnx.to_pure_dict(nnx.state(model)))
    save_file(model_state, model_path, metadata=metadata)

    optimizer_state = _flatten_state(nnx.to_pure_dict(nnx.state(optimizer)))
    save_file(optimizer_state, optimizer_path, metadata=metadata)


def load_checkpoint(model: ModelT, optimizer: nnx.Optimizer[ModelT], path: Path) -> int:
    model_path, optimizer_path = _checkpoint_paths(path)

    model_state = _unflatten_state(load_file(model_path))
    nnx.update(model, model_state)

    optimizer_state = _unflatten_state(load_file(optimizer_path))
    nnx.update(optimizer, optimizer_state)

    with safe_open(optimizer_path, framework="numpy") as f:
        metadata = f.metadata() or {}
    return int(metadata["step"])


def restore_rngs(path: Path, rngs: Mapping[str, np.random.Generator]) -> bool:
    """Restore saved generator states in place, returning whether any were found.

    Deliberately mutates the caller's generators rather than returning new
    ones: `train()` and the eval closure both hold references to specific
    Generator objects, so handing back fresh instances would leave the
    originals -- the ones actually being drawn from -- untouched.

    Returns False for a checkpoint written before `save_checkpoint` recorded
    RNG state, so a caller can refuse to resume rather than silently continue
    with a divergent data stream (see `save_checkpoint` on why that matters).
    """
    _model_path, optimizer_path = _checkpoint_paths(path)
    with safe_open(optimizer_path, framework="numpy") as f:
        metadata = f.metadata() or {}
    raw = metadata.get("rng_states")
    if raw is None:
        return False

    saved = json.loads(raw)
    missing = set(rngs) - set(saved)
    if missing:
        raise ValueError(
            f"checkpoint {path.name} has no saved state for rng(s) {sorted(missing)} "
            f"-- resuming would give them a different data stream than an uninterrupted run"
        )
    for name, generator in rngs.items():
        generator.bit_generator.state = saved[name]
    return True


def latest_checkpoint(checkpoint_dir: Path) -> Path | None:
    """Highest-step checkpoint base path in `checkpoint_dir`, or None if there are none.

    Only counts steps whose *both* files exist -- a run killed midway through
    writing a checkpoint pair would otherwise be resumed from a half-written
    step.
    """
    if not checkpoint_dir.is_dir():
        return None
    bases = {p.parent / p.name.split(".")[0] for p in checkpoint_dir.glob("step_*.safetensors")}
    complete = [b for b in bases if all(p.exists() for p in _checkpoint_paths(b))]
    if not complete:
        return None
    return max(complete, key=lambda p: int(p.name.removeprefix("step_")))


def load_model_checkpoint(model: ModelT, path: Path) -> int:
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


EvalFn = Callable[[ModelT], Mapping[tuple[int, int], float]]


@dataclass
class TrainingHistory:
    """Record of what happened during a `train()` run, so loss and eval results are observable."""

    loss: list[tuple[int, float]] = field(default_factory=list)
    eval_grids: list[tuple[int, Mapping[tuple[int, int], float]]] = field(default_factory=list)


def train(
    model: ModelT,
    optimizer: nnx.Optimizer[ModelT],
    train_config: TrainConfig,
    vocab_size: int,
    rng: np.random.Generator,
    checkpoint_dir: Path,
    eval_fn: "EvalFn[ModelT] | None" = None,
    start_step: int = 0,
    checkpoint_rngs: Mapping[str, np.random.Generator] | None = None,
) -> TrainingHistory:
    """Train from `start_step + 1` to `train_config.total_steps`.

    `start_step` and `checkpoint_rngs` exist together and should be used
    together: resuming a run means continuing the *same* data stream, which
    requires the caller to have restored every generator's state (via
    `restore_rngs`) before calling. `checkpoint_rngs` is the set of generators
    written into each checkpoint so that a later resume can do so -- pass the
    same mapping on the original run and on the resumed one. See
    `save_checkpoint` for why an unrestored generator silently breaks
    cross-variant comparability rather than failing loudly.

    Note `start_step` shifts only which steps execute, not the optimizer's LR
    schedule: that is driven by optax's own step count, which rides in the
    restored optimizer state, so a resumed run continues the warmup/cosine
    curve where it left off rather than restarting it.
    """
    expected_vocab_size = total_vocab_size(vocab_size)
    if model.config.vocab_size != expected_vocab_size:
        raise ValueError(
            f"model.config.vocab_size={model.config.vocab_size} does not match "
            f"total_vocab_size(entity_vocab_size={vocab_size})={expected_vocab_size} "
            "-- the model's embedding table won't cover the token ids the generator produces"
        )

    history = TrainingHistory()

    for step in range(start_step + 1, train_config.total_steps + 1):
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
            save_checkpoint(
                model, optimizer, step, checkpoint_dir / f"step_{step}", rngs=checkpoint_rngs
            )

    return history
