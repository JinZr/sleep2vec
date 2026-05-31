import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import ChannelAggregator


class GatedScalarChannelAggregator(ChannelAggregator):
    name = "gated_scalar"

    def __init__(self, feature_dim: int, n_mods: int):
        super().__init__(feature_dim, n_mods)
        self.gates = nn.Parameter(torch.zeros(n_mods))

    def forward(self, feats, *, channel_mask=None):
        has_L, stack, mask = self._prepare_inputs(feats, channel_mask)
        logits = self.gates.view(1, self.n_mods, 1).expand(stack.size(0), self.n_mods, stack.size(2))
        logits = logits.to(dtype=stack.dtype)
        weights = F.softmax(logits.masked_fill(~mask[:, :, None], torch.finfo(stack.dtype).min), dim=1)
        fused = (stack * weights.unsqueeze(-1)).sum(dim=1)
        return self._restore_rank(fused, has_L), has_L


__all__ = ["GatedScalarChannelAggregator"]
