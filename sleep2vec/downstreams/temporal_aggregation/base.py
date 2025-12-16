import torch
import torch.nn as nn


class TemporalAggregator(nn.Module):
    """Base class for temporal aggregation over token sequences."""

    name: str = "base"

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden: [B, L, D] token embeddings
            mask:   [B, L] boolean, True for valid tokens
        Returns:
            pooled: [B, D] aggregated representation
        """
        raise NotImplementedError


__all__ = ["TemporalAggregator"]
