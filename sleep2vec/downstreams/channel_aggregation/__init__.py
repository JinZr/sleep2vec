from typing import Any

from .base import ChannelAggregator
from .concat import ConcatChannelAggregator
from .gated_scalar import GatedScalarChannelAggregator
from .mean import MeanChannelAggregator

_REGISTRY = {
    MeanChannelAggregator.name: MeanChannelAggregator,
    ConcatChannelAggregator.name: ConcatChannelAggregator,
    GatedScalarChannelAggregator.name: GatedScalarChannelAggregator,
}


def build_channel_aggregator(name: str | None, feature_dim: int, n_mods: int, **kwargs: Any) -> ChannelAggregator:
    resolved = (name or MeanChannelAggregator.name).lower()
    if resolved not in _REGISTRY:
        raise ValueError(f"Unknown channel aggregator '{name}'. Supported: {sorted(_REGISTRY)}")
    cls = _REGISTRY[resolved]
    return cls(feature_dim=feature_dim, n_mods=n_mods, **kwargs)


__all__ = [
    "ChannelAggregator",
    "MeanChannelAggregator",
    "ConcatChannelAggregator",
    "GatedScalarChannelAggregator",
    "build_channel_aggregator",
]
