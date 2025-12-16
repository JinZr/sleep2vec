import torch

from .base import ChannelAggregator


class ConcatChannelAggregator(ChannelAggregator):
    name = "concat"

    def forward(self, feats):
        has_L, feats = self._validate_shapes(feats)
        fused = torch.cat(feats, dim=-1)
        if not has_L:
            fused = fused.squeeze(1)
        return fused, has_L


__all__ = ["ConcatChannelAggregator"]
