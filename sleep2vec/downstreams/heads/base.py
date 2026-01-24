import typing as t

import torch
import torch.nn as nn

from sleep2vec.downstreams.channel_aggregation import build_channel_aggregator


class FeatureFusion(nn.Module):
    """Shared multi-modal fusion block using pluggable channel aggregators."""

    def __init__(self, feature_dim: int, n_mods: int, agg: str):
        super().__init__()
        if n_mods < 1:
            raise ValueError("n_mods must be >= 1.")
        if n_mods == 1 and agg != "concat":
            agg = "concat"  # fall back to concatenation for single modality
        self.aggregator = build_channel_aggregator(agg, feature_dim=feature_dim, n_mods=n_mods)
        self.output_dim = feature_dim * n_mods if agg == "concat" else feature_dim


__all__ = ["FeatureFusion"]
