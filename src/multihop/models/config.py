"""Shared depth/width config for the full-attention, KDA, and hybrid variants.

See ADR 0008 (docs/adr/0008-shared-entity-pool-for-binding-generalization.md)
for the current vocab size (superseding ADR 0002's disjoint-pool sizing,
which is preserved as a record of the design that was tried and reverted),
and ADR 0001 (docs/adr/0001-nope-full-attention-positional-info-from-gating.md)
for why `positional_encoding` is a per-variant field rather than a shared constant.
"""

from dataclasses import dataclass
from typing import Literal

NormType = Literal["rmsnorm"]
ActivationType = Literal["swiglu"]
PositionalEncoding = Literal["rope", "none"]


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 8_003
    embed_dim: int = 768
    n_layers: int = 12
    n_heads: int = 12
    ffn_dim: int = 3072
    max_seq_len: int = 287
    tie_embeddings: bool = True
    norm: NormType = "rmsnorm"
    activation: ActivationType = "swiglu"
    positional_encoding: PositionalEncoding = "rope"
    rope_theta: float = 10_000.0
    init_std: float = 0.02

    @property
    def head_dim(self) -> int:
        return self.embed_dim // self.n_heads

    @property
    def residual_init_scale(self) -> float:
        return float(1.0 / (2 * self.n_layers) ** 0.5)
