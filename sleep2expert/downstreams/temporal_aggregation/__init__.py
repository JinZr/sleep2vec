from typing import Any

from .attn import AttnAggregator
from .base import TemporalAggregator
from .mean import MeanAggregator

_REGISTRY = {
    MeanAggregator.name: MeanAggregator,
    AttnAggregator.name: AttnAggregator,
}


def build_temporal_aggregator(name: str | None, hidden_size: int, **kwargs: Any) -> TemporalAggregator:
    resolved = (name or MeanAggregator.name).lower()
    if resolved not in _REGISTRY:
        raise ValueError(f"Unknown temporal aggregator '{name}'. Supported: {sorted(_REGISTRY)}")
    cls = _REGISTRY[resolved]
    if cls is AttnAggregator:
        return cls(hidden_size=hidden_size, **kwargs)
    return cls()


__all__ = ["TemporalAggregator", "MeanAggregator", "AttnAggregator", "build_temporal_aggregator"]
