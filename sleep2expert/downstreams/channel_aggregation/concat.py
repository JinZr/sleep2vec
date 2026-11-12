import torch
import torch.nn as nn

from .base import ChannelAggregator


class ConcatChannelAggregator(ChannelAggregator):
    name = "concat"

    def __init__(self, feature_dim: int, n_mods: int, *, projection_dim: int | None = None):
        super().__init__(feature_dim, n_mods)
        self.projection = nn.Linear(feature_dim, projection_dim) if projection_dim is not None else None
        self.output_dim = n_mods * (projection_dim if projection_dim is not None else feature_dim)

    def forward(self, feats):
        has_L, feats = self._validate_shapes(feats)
        if self.projection is not None:
            feats = [self.projection(feat) for feat in feats]
        fused = torch.cat(feats, dim=-1)
        if not has_L:
            fused = fused.squeeze(1)
        return fused, has_L


__all__ = ["ConcatChannelAggregator"]
