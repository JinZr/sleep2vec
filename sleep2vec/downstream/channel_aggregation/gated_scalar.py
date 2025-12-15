import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import ChannelAggregator


class GatedScalarChannelAggregator(ChannelAggregator):
    name = "gated_scalar"

    def __init__(self, feature_dim: int, n_mods: int):
        super().__init__(feature_dim, n_mods)
        self.gates = nn.Parameter(torch.zeros(n_mods))

    def forward(self, feats):
        has_L, feats = self._validate_shapes(feats)
        weights = F.softmax(self.gates, dim=0)  # [n_mods]
        stack = torch.stack(feats, dim=0)  # [n_mods, B, L, D]
        fused = (weights[:, None, None, None] * stack).sum(dim=0)
        if not has_L:
            fused = fused.squeeze(1)
        return fused, has_L


__all__ = ["GatedScalarChannelAggregator"]
