from __future__ import annotations

import torch.nn as nn

# Ensure backbone modules are registered.
import sleep2vec2.backbones  # noqa: F401
from sleep2vec2.config import BackboneConfig, ModelConfig, ProjectionConfig, validate_model_config
from sleep2vec2.modules.projection import build_projection_head
from sleep2vec2.modules.tokenizers import build_tokenizer_mapping
from sleep2vec2.registry import get_backbone_builder


def build_encoder(backbone_cfg: BackboneConfig) -> nn.Module:
    """Build the encoder module for the given backbone config."""

    builder = get_backbone_builder(backbone_cfg.name)
    encoder = builder(backbone_cfg)
    if not isinstance(encoder, nn.Module):
        raise TypeError(
            f"Backbone builder '{backbone_cfg.name}' must return nn.Module, got {type(encoder)}"
        )
    return encoder


def build_tokenizers_and_dim(
    model_cfg: ModelConfig, *, device: str = "cuda"
) -> tuple[dict[str, nn.Module], int]:
    feature_dim = validate_model_config(model_cfg)
    tokenizer_mapping = build_tokenizer_mapping(model_cfg.channels, device=device)
    return tokenizer_mapping, feature_dim


def build_projection(projection_cfg: ProjectionConfig, *, in_dim: int) -> nn.Module | None:
    return build_projection_head(projection_cfg, in_dim=in_dim)


__all__ = [
    "build_encoder",
    "build_projection",
    "build_tokenizers_and_dim",
]
