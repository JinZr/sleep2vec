import torch

from sleep2vec.downstreams.heads import AttnPooling

from .base import TemporalAggregator


class AttnAggregator(TemporalAggregator):
    name = "attn"

    def __init__(self, hidden_size: int, heads: int = 1, temp: float = 1.0, dropout: float = 0.0):
        super().__init__()
        self.pool = AttnPooling(hidden_size, heads=heads, temp=temp, dropout=dropout)

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        pooled, _ = self.pool(hidden, mask)
        return pooled


__all__ = ["AttnAggregator"]
