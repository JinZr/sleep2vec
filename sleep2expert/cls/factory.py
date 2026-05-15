from typing import Any

from .base import ClsEmbedding
from .bert import BertClsEmbedding
from .none import NoClsEmbedding

_REGISTRY = {
    "none": NoClsEmbedding,
    "null": NoClsEmbedding,
    "bert": BertClsEmbedding,
}


def build_cls_embedding(strategy: str | None, hidden_size: int | None, **kwargs: Any) -> ClsEmbedding:
    normalized = (strategy or "none").lower()
    if normalized not in _REGISTRY:
        raise ValueError(f"Unknown CLS strategy '{strategy}'. Supported: null/none, 'bert'.")
    cls = _REGISTRY[normalized]
    if cls is BertClsEmbedding:
        if hidden_size is None:
            raise ValueError("hidden_size is required to build a bert CLS embedding.")
        return cls(hidden_size=hidden_size, **kwargs)
    return cls(**kwargs)


__all__ = ["build_cls_embedding"]
