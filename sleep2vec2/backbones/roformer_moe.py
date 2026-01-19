from __future__ import annotations

import typing as t

import torch.nn as nn

from sleep2vec2.config import BackboneConfig
from sleep2vec2.registry import register_backbone

from .roformer import _validate_overrides
from .roformer_encoder import RoFormerEncoderConfig, RoFormerEncoderModel


@register_backbone("roformer_moe")
def build_roformer_moe(cfg: BackboneConfig) -> nn.Module:
    """Build a RoFormer encoder with MoE enabled by default."""

    overrides = _validate_overrides(dict(cfg.config_overrides or {}))
    overrides.setdefault("use_moe", True)

    intermediate_size = cfg.intermediate_size
    if intermediate_size is None:
        intermediate_size = int(cfg.hidden_size * 4)

    params: dict[str, t.Any] = dict(
        embedding_size=cfg.embedding_size,
        hidden_size=cfg.hidden_size,
        num_hidden_layers=cfg.num_hidden_layers,
        num_attention_heads=cfg.num_attention_heads,
        intermediate_size=intermediate_size,
        hidden_act=cfg.hidden_act,
        hidden_dropout_prob=cfg.hidden_dropout_prob,
        attention_probs_dropout_prob=cfg.attention_probs_dropout_prob,
        max_position_embeddings=cfg.max_position_embeddings,
        num_token_types=cfg.num_token_types,
        initializer_range=cfg.initializer_range,
        layer_norm_eps=cfg.layer_norm_eps,
        rotary_value=cfg.rotary_value,
        chunk_size_feed_forward=cfg.chunk_size_feed_forward,
        use_return_dict=cfg.use_return_dict,
    )
    params.update(overrides)

    # RoFormerEncoderModel accepts kwargs validated against RoFormerEncoderConfig.
    allowed = set(RoFormerEncoderConfig.__dataclass_fields__.keys())  # type: ignore[attr-defined]
    unknown = sorted(k for k in params.keys() if k not in allowed)
    if unknown:
        raise ValueError(
            "Unknown backbone.config_overrides keys for RoFormerEncoderModel: "
            f"{unknown}. Allowed keys: {sorted(allowed)}"
        )

    return RoFormerEncoderModel(**params)


__all__ = ["build_roformer_moe"]
