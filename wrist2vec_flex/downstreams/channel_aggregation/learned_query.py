import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import ChannelAggregator


class LearnedQueryChannelAggregator(ChannelAggregator):
    name = "learned_query"

    def __init__(self, feature_dim: int, n_mods: int, dropout: float = 0.0):
        super().__init__(feature_dim, n_mods)
        self.query = nn.Parameter(torch.empty(feature_dim))
        self.dropout = nn.Dropout(float(dropout))
        nn.init.normal_(self.query, std=feature_dim**-0.5)

    def forward(self, feats, *, channel_mask=None):
        has_L, stack, mask = self._prepare_inputs(feats, channel_mask)
        scores = (stack * self.query.view(1, 1, 1, self.feature_dim)).sum(dim=-1) / math.sqrt(self.feature_dim)
        weights = F.softmax(scores.masked_fill(~mask[:, :, None], torch.finfo(stack.dtype).min), dim=1)
        fused = self.dropout((stack * weights.unsqueeze(-1)).sum(dim=1))
        return self._restore_rank(fused, has_L), has_L


__all__ = ["LearnedQueryChannelAggregator"]
