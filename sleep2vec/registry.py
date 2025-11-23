from __future__ import annotations

import typing as t

from sleep2vec.config import BackboneConfig, ProjectionConfig

BackboneBuilder = t.Callable[[BackboneConfig], t.Any]
TokenizerBuilder = t.Callable[..., t.Any]
ProjectionBuilder = t.Callable[[ProjectionConfig | None], t.Any]

BACKBONE_REGISTRY: t.Dict[str, BackboneBuilder] = {}
TOKENIZER_REGISTRY: t.Dict[str, TokenizerBuilder] = {}
PROJECTION_REGISTRY: t.Dict[str, ProjectionBuilder] = {}


def _register(registry: t.Dict[str, t.Any], name: str, obj: t.Any):
    if name in registry:
        raise ValueError(f"'{name}' is already registered.")
    registry[name] = obj
    return obj


def register_backbone(name: str):
    def decorator(fn: BackboneBuilder):
        return _register(BACKBONE_REGISTRY, name, fn)

    return decorator


def register_tokenizer(name: str):
    def decorator(fn: TokenizerBuilder):
        return _register(TOKENIZER_REGISTRY, name, fn)

    return decorator


def register_projection(name: str):
    def decorator(fn: ProjectionBuilder):
        return _register(PROJECTION_REGISTRY, name, fn)

    return decorator


def get_backbone_builder(name: str) -> BackboneBuilder:
    if name not in BACKBONE_REGISTRY:
        raise KeyError(f"Unknown backbone '{name}'. Available: {sorted(BACKBONE_REGISTRY)}")
    return BACKBONE_REGISTRY[name]


def get_tokenizer_builder(name: str) -> TokenizerBuilder:
    if name not in TOKENIZER_REGISTRY:
        raise KeyError(f"Unknown tokenizer '{name}'. Available: {sorted(TOKENIZER_REGISTRY)}")
    return TOKENIZER_REGISTRY[name]


def get_projection_builder(name: str) -> ProjectionBuilder:
    if name not in PROJECTION_REGISTRY:
        raise KeyError(f"Unknown projection '{name}'. Available: {sorted(PROJECTION_REGISTRY)}")
    return PROJECTION_REGISTRY[name]


def available_backbones() -> t.List[str]:
    return sorted(BACKBONE_REGISTRY)


def available_tokenizers() -> t.List[str]:
    return sorted(TOKENIZER_REGISTRY)


def available_projections() -> t.List[str]:
    return sorted(PROJECTION_REGISTRY)


__all__ = [
    "available_backbones",
    "available_projections",
    "available_tokenizers",
    "get_backbone_builder",
    "get_projection_builder",
    "get_tokenizer_builder",
    "register_backbone",
    "register_projection",
    "register_tokenizer",
]
