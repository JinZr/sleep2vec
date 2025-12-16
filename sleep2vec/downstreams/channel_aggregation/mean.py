import torch

from .base import ChannelAggregator


class MeanChannelAggregator(ChannelAggregator):
    name = "mean"

    def forward(self, feats):
        has_L, feats = self._validate_shapes(feats)
        fused = torch.stack(feats, dim=0).mean(dim=0)
        if not has_L:
            fused = fused.squeeze(1)
        return fused, has_L


__all__ = ["MeanChannelAggregator"]
