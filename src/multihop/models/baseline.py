"""Pure full-attention decoder-only baseline, ~119M params (ADR 0002, ADR 0008)."""

from flax import nnx
from jax import Array

from multihop.models.config import ModelConfig
from multihop.models.layers import RMSNorm, TransformerBlock
from multihop.models.rope import rope_freqs


class FullAttentionBaseline(nnx.Module):
    def __init__(self, config: ModelConfig, *, rngs: nnx.Rngs) -> None:
        self.config = config
        embed_init = nnx.initializers.normal(config.init_std)
        self.embed = nnx.Embed(
            config.vocab_size, config.embed_dim, embedding_init=embed_init, rngs=rngs
        )
        self.blocks = nnx.List(
            [TransformerBlock(config, rngs=rngs) for _ in range(config.n_layers)]
        )
        self.final_norm = RMSNorm(config.embed_dim)
        if not config.tie_embeddings:
            self.head = nnx.Linear(
                config.embed_dim,
                config.vocab_size,
                use_bias=False,
                kernel_init=embed_init,
                rngs=rngs,
            )

    def __call__(self, token_ids: Array, *, capture_attention: bool = False) -> Array:
        _batch, seq_len = token_ids.shape
        if seq_len > self.config.max_seq_len:
            raise ValueError(
                f"input sequence length {seq_len} exceeds config.max_seq_len={self.config.max_seq_len}"
            )
        cos, sin = rope_freqs(seq_len, self.config.head_dim, self.config.rope_theta)

        x = self.embed(token_ids)
        for block in self.blocks:
            x = block(x, cos, sin, capture_attention=capture_attention)
        x = self.final_norm(x)

        if self.config.tie_embeddings:
            return self.embed.attend(x)
        return self.head(x)
