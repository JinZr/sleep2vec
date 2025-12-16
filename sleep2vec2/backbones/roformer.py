from __future__ import annotations

import typing as t

import torch.nn as nn

from sleep2vec2.config import BackboneConfig
from sleep2vec2.registry import register_backbone

from .roformer_encoder import RoFormerEncoderConfig, RoFormerEncoderModel


def _validate_overrides(overrides: dict[str, t.Any]) -> dict[str, t.Any]:
    """Validate and sanitize user-provided ``config_overrides``.

    Historically, the original recipe supported Hugging Face RoFormer configs,
    which commonly included keys like ``vocab_size``. The standalone encoder
    used in sleep2vec2 does not take token IDs and therefore does not use
    vocabulary related options.

    We accept (and ignore) a small set of legacy keys to keep older YAML files
    working, but we fail fast on unknown keys to avoid silently misconfiguring
    the model.
    """

    # Keys that may exist in legacy configs but are irrelevant for embeddings-only encoders.
    legacy_ignored = {"vocab_size", "type_vocab_size"}
    cleaned = {k: v for k, v in overrides.items() if k not in legacy_ignored}

    # RoFormerEncoderModel accepts arbitrary kwargs and forwards them to
    # RoFormerEncoderConfig(...), so validate against the config dataclass fields.
    allowed = set(RoFormerEncoderConfig.__dataclass_fields__.keys())  # type: ignore[attr-defined]
    unknown = sorted(k for k in cleaned.keys() if k not in allowed)
    if unknown:
        raise ValueError(
            "Unknown backbone.config_overrides keys for RoFormerEncoderModel: "
            f"{unknown}. Allowed keys: {sorted(allowed)}"
        )

    return cleaned


@register_backbone("roformer")
def build_roformer(cfg: BackboneConfig) -> nn.Module:
    """Build the standalone RoFormer encoder.

    The returned module follows a HuggingFace-like forward signature:
    ``encoder(inputs_embeds=..., attention_mask=...)`` and returns an object
    with ``last_hidden_state``.
    """

    overrides = _validate_overrides(dict(cfg.config_overrides or {}))

    intermediate_size = cfg.intermediate_size
    if intermediate_size is None:
        intermediate_size = int(cfg.hidden_size * 4)

    # Build param dict first, then apply overrides so YAML can override any field.
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
    return RoFormerEncoderModel(**params)


__all__ = ["build_roformer"]
