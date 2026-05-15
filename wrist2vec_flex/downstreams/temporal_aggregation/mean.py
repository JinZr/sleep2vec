import torch

from .base import TemporalAggregator


class MeanAggregator(TemporalAggregator):
    name = "mean"

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # mask: True for valid positions
        hidden_masked = hidden * mask.unsqueeze(-1)
        denom = mask.sum(dim=1).clamp(min=1).unsqueeze(-1).float()
        return hidden_masked.sum(dim=1) / denom


__all__ = ["MeanAggregator"]
