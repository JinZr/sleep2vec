import typing as t

import torch.nn as nn

from sleep2expert.downstreams.channel_aggregation import build_channel_aggregator


class FeatureFusion(nn.Module):
    """Shared multi-modal fusion block using pluggable channel aggregators."""

    def __init__(self, feature_dim: int, n_mods: int, agg: str, channel_agg_kwargs: dict[str, t.Any] | None = None):
        super().__init__()
        if n_mods < 1:
            raise ValueError("n_mods must be >= 1.")
        if n_mods == 1 and agg != "concat":
            agg = "concat"  # fall back to concatenation for single modality
        self.aggregator = build_channel_aggregator(
            agg, feature_dim=feature_dim, n_mods=n_mods, **(channel_agg_kwargs or {})
        )
        self.output_dim = self.aggregator.output_dim


__all__ = ["FeatureFusion"]
