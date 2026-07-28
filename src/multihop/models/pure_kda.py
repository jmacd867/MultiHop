"""Pure-KDA decoder-only variant: every layer is Kimi Delta Attention, NoPE (ADR 0001, ADR 0009)."""

from flax import nnx
from jax import Array

from multihop.models.config import ModelConfig
from multihop.models.kda import KDABlock
from multihop.models.layers import RMSNorm


class PureKDAModel(nnx.Module):
    def __init__(self, config: ModelConfig, *, rngs: nnx.Rngs) -> None:
        if config.positional_encoding != "none":
            raise ValueError(
                f"PureKDAModel requires positional_encoding='none' (ADR 0001), got {config.positional_encoding!r}"
            )
        self.config = config
        embed_init = nnx.initializers.normal(config.init_std)
        self.embed = nnx.Embed(
            config.vocab_size, config.embed_dim, embedding_init=embed_init, rngs=rngs
        )
        self.blocks = nnx.List([KDABlock(config, rngs=rngs) for _ in range(config.n_layers)])
        self.final_norm = RMSNorm(config.embed_dim)
        if not config.tie_embeddings:
            self.head = nnx.Linear(
                config.embed_dim,
                config.vocab_size,
                use_bias=False,
                kernel_init=embed_init,
                rngs=rngs,
            )

    def __call__(self, token_ids: Array) -> Array:
        _batch, seq_len = token_ids.shape
        if seq_len > self.config.max_seq_len:
            raise ValueError(
                f"input sequence length {seq_len} exceeds config.max_seq_len={self.config.max_seq_len}"
            )

        x = self.embed(token_ids)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)

        if self.config.tie_embeddings:
            return self.embed.attend(x)
        return self.head(x)
