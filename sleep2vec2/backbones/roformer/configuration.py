from __future__ import annotations

"""Configuration for standalone RoFormer encoder."""

from dataclasses import dataclass


@dataclass
class RoFormerConfig:
    """Lightweight configuration for RoFormer encoder-only usage."""

    vocab_size: int = 50000
    embedding_size: int | None = None
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "gelu"
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    max_position_embeddings: int = 1536
    type_vocab_size: int = 2
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    pad_token_id: int = 0
    rotary_value: bool = False
    attention_backend: str = "eager"

    def get(self, key: str, default=None):
        return getattr(self, key, default)

    def __post_init__(self) -> None:
        if self.attention_backend not in ("eager", "sdpa"):
            raise ValueError("attention_backend must be one of eager, sdpa.")
        if self.embedding_size is None:
            self.embedding_size = self.hidden_size
        if self.hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {self.hidden_size}")
        if self.num_attention_heads <= 0:
            raise ValueError(f"num_attention_heads must be positive, got {self.num_attention_heads}")
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                "hidden_size must be divisible by num_attention_heads: "
                f"hidden_size={self.hidden_size}, num_attention_heads={self.num_attention_heads}"
            )
        if self.num_hidden_layers <= 0:
            raise ValueError(f"num_hidden_layers must be positive, got {self.num_hidden_layers}")
        if self.max_position_embeddings <= 0:
            raise ValueError(f"max_position_embeddings must be positive, got {self.max_position_embeddings}")
