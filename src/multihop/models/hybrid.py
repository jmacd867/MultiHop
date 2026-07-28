"""Hybrid decoder-only variant: 3:1 KDA-to-full-attention interleave, NoPE (ADR 0011, ADR 0001).

Composes the two already-validated block types unchanged -- `KDABlock` from
`kda.py` and `TransformerBlock` from `layers.py` -- rather than reimplementing
either. The only thing this module decides is the order.
"""

from flax import nnx
from jax import Array

from multihop.models.config import ModelConfig
from multihop.models.kda import KDABlock
from multihop.models.layers import RMSNorm, TransformerBlock

# ADR 0011: strict repeating KDA,KDA,KDA,FullAttention from layer 0. Full
# attention therefore lands wherever `layer_index % PERIOD == PERIOD - 1`,
# i.e. 3/7/11 at n_layers=12. Expressed as a rule rather than a literal index
# list so it stays correct for any n_layers that is a whole number of groups --
# which `HybridModel` enforces, because a partial final group would silently
# change the ratio the whole Variant is defined by.
INTERLEAVE_PERIOD = 4


def is_full_attention_layer(layer_index: int) -> bool:
    """True where ADR 0011's schedule puts a full-attention layer."""
    return layer_index % INTERLEAVE_PERIOD == INTERLEAVE_PERIOD - 1


class HybridModel(nnx.Module):
    def __init__(self, config: ModelConfig, *, rngs: nnx.Rngs) -> None:
        if config.positional_encoding != "none":
            raise ValueError(
                "HybridModel requires positional_encoding='none' (ADR 0001: the rule is "
                "per-variant, so a hybrid's full-attention layers are NoPE too), got "
                f"{config.positional_encoding!r}"
            )
        # A partial final group silently changes the ratio that defines this
        # Variant: n_layers=6 gives 5:1, n_layers=10 gives 4:1, and anything
        # below 4 gives *zero* full-attention layers -- a PureKDAModel wearing
        # the hybrid's name. All of those build, train, and checkpoint
        # normally, so nothing downstream would catch it (ADR 0011: both block
        # types share input and output shape, which is why the schedule is
        # asserted rather than assumed).
        if config.n_layers % INTERLEAVE_PERIOD != 0:
            raise ValueError(
                f"HybridModel requires n_layers to be a multiple of {INTERLEAVE_PERIOD} "
                f"(ADR 0011's strict {INTERLEAVE_PERIOD - 1}:1 repeat), got "
                f"n_layers={config.n_layers}, which would leave a partial final group "
                "and silently change the KDA-to-full-attention ratio"
            )
        self.config = config
        embed_init = nnx.initializers.normal(config.init_std)
        self.embed = nnx.Embed(
            config.vocab_size, config.embed_dim, embedding_init=embed_init, rngs=rngs
        )
        self.blocks = nnx.List(
            [
                TransformerBlock(config, rngs=rngs)
                if is_full_attention_layer(i)
                else KDABlock(config, rngs=rngs)
                for i in range(config.n_layers)
            ]
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

        x = self.embed(token_ids)
        for block in self.blocks:
            # No cos/sin: NoPE throughout (ADR 0001), so the full-attention
            # blocks take their optional rotary tables as None.
            if isinstance(block, TransformerBlock):
                x = block(x, capture_attention=capture_attention)
            else:
                x = block(x)
        x = self.final_norm(x)

        if self.config.tie_embeddings:
            return self.embed.attend(x)
        return self.head(x)
