from __future__ import annotations

import torch.nn as nn

from sleep2wave.backbones.encoder_factory import TransformerEncoderFactory
from sleep2wave.config import BackboneConfig, ModelConfig, ProjectionConfig, validate_model_config
from sleep2wave.modules.projection import build_projection_head
from sleep2wave.modules.tokenizers import build_tokenizer_mapping
from sleep2wave.registry import get_backbone_builder


def build_encoder_factory(backbone_cfg: BackboneConfig) -> TransformerEncoderFactory:
    builder = get_backbone_builder(backbone_cfg.name)
    return builder(backbone_cfg)


def build_tokenizers_and_dim(
    model_cfg: ModelConfig, *, device: str = "cuda"
) -> tuple[dict[str, nn.Module], int]:
    feature_dim = validate_model_config(model_cfg)
    tokenizer_mapping = build_tokenizer_mapping(model_cfg.channels, device=device)
    return tokenizer_mapping, feature_dim


def build_projection(
    projection_cfg: ProjectionConfig, *, in_dim: int
) -> nn.Module | None:
    return build_projection_head(projection_cfg, in_dim=in_dim)


__all__ = [
    "build_encoder_factory",
    "build_projection",
    "build_tokenizers_and_dim",
]
