import torch

from .base import ChannelAggregator


class MeanChannelAggregator(ChannelAggregator):
    name = "mean"

    def forward(self, feats, *, channel_mask=None):
        has_L, stack, mask = self._prepare_inputs(feats, channel_mask)
        weights = mask.to(dtype=stack.dtype).unsqueeze(-1).unsqueeze(-1)
        fused = (stack * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return self._restore_rank(fused, has_L), has_L


__all__ = ["MeanChannelAggregator"]
