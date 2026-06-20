from typing import Any

from .attn import AttnAggregator
from .base import TemporalAggregator
from .lstm import LSTMAggregator
from .mean import MeanAggregator

_REGISTRY = {
    MeanAggregator.name: MeanAggregator,
    AttnAggregator.name: AttnAggregator,
    LSTMAggregator.name: LSTMAggregator,
}


def build_temporal_aggregator(name: str | None, hidden_size: int, **kwargs: Any) -> TemporalAggregator:
    resolved = (name or MeanAggregator.name).lower()
    if resolved not in _REGISTRY:
        raise ValueError(f"Unknown temporal aggregator '{name}'. Supported: {sorted(_REGISTRY)}")
    cls = _REGISTRY[resolved]
    if cls in {AttnAggregator, LSTMAggregator}:
        return cls(hidden_size=hidden_size, **kwargs)
    return cls()


__all__ = ["TemporalAggregator", "MeanAggregator", "AttnAggregator", "LSTMAggregator", "build_temporal_aggregator"]
