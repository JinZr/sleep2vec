import typing as t

import torch
import torch.nn as nn

from wrist2vec_flex.downstreams.channel_aggregation import build_channel_aggregator


class FeatureFusion(nn.Module):
    """Shared multi-modal fusion block using pluggable channel aggregators."""

    def __init__(self, feature_dim: int, n_mods: int, agg: str, agg_kwargs: dict | None = None):
        super().__init__()
        if n_mods < 1:
            raise ValueError("n_mods must be >= 1.")
        agg_kwargs = dict(agg_kwargs or {})
        if n_mods == 1 and agg != "concat":
            agg = "concat"
            agg_kwargs = {}
        self.aggregator = build_channel_aggregator(agg, feature_dim=feature_dim, n_mods=n_mods, **agg_kwargs)
        self.output_dim = feature_dim * n_mods if agg == "concat" else feature_dim


__all__ = ["FeatureFusion"]
