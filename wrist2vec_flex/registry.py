from __future__ import annotations

import typing as t

from wrist2vec_flex.config import BackboneConfig, ModelAveragingConfig, ProjectionConfig

BackboneBuilder = t.Callable[[BackboneConfig], t.Any]
TokenizerBuilder = t.Callable[..., t.Any]
ProjectionBuilder = t.Callable[[t.Optional[ProjectionConfig]], t.Any]
ModelAveragerBuilder = t.Callable[[ModelAveragingConfig, t.Any], t.Any]
SourceFusionBuilder = t.Callable[..., t.Any]

BACKBONE_REGISTRY: t.Dict[str, BackboneBuilder] = {}
TOKENIZER_REGISTRY: t.Dict[str, TokenizerBuilder] = {}
PROJECTION_REGISTRY: t.Dict[str, ProjectionBuilder] = {}
MODEL_AVERAGING_REGISTRY: t.Dict[str, ModelAveragerBuilder] = {}
SOURCE_FUSION_REGISTRY: t.Dict[str, SourceFusionBuilder] = {}


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


def register_model_averager(name: str):
    def decorator(fn: ModelAveragerBuilder):
        return _register(MODEL_AVERAGING_REGISTRY, name, fn)

    return decorator


def register_source_fusion(name: str):
    def decorator(fn: SourceFusionBuilder):
        return _register(SOURCE_FUSION_REGISTRY, name, fn)

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


def get_model_averager_builder(name: str) -> ModelAveragerBuilder:
    if name not in MODEL_AVERAGING_REGISTRY:
        raise KeyError(f"Unknown model averaging '{name}'. Available: {sorted(MODEL_AVERAGING_REGISTRY)}")
    return MODEL_AVERAGING_REGISTRY[name]


def get_source_fusion_builder(name: str) -> SourceFusionBuilder:
    if name not in SOURCE_FUSION_REGISTRY:
        raise KeyError(f"Unknown source fusion '{name}'. Available: {sorted(SOURCE_FUSION_REGISTRY)}")
    return SOURCE_FUSION_REGISTRY[name]


def available_backbones() -> t.List[str]:
    return sorted(BACKBONE_REGISTRY)


def available_tokenizers() -> t.List[str]:
    return sorted(TOKENIZER_REGISTRY)


def available_projections() -> t.List[str]:
    return sorted(PROJECTION_REGISTRY)


def available_model_averagers() -> t.List[str]:
    return sorted(MODEL_AVERAGING_REGISTRY)


def available_source_fusions() -> t.List[str]:
    return sorted(SOURCE_FUSION_REGISTRY)


__all__ = [
    "available_backbones",
    "available_projections",
    "available_tokenizers",
    "available_model_averagers",
    "get_backbone_builder",
    "get_model_averager_builder",
    "available_source_fusions",
    "get_projection_builder",
    "get_source_fusion_builder",
    "get_tokenizer_builder",
    "register_backbone",
    "register_model_averager",
    "register_projection",
    "register_source_fusion",
    "register_tokenizer",
]
