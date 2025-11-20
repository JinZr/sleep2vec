from __future__ import annotations

import typing as t
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class ChannelConfig:
    name: str
    input_dim: int
    out_dim: int
    tokenizer: str = "linear"
    tokenizer_kwargs: dict[str, t.Any] = field(default_factory=dict)


@dataclass
class BackboneConfig:
    name: str = "roformer"
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 16
    vocab_size: int = 1
    config_overrides: dict[str, t.Any] = field(default_factory=dict)


@dataclass
class ProjectionConfig:
    name: str = "simclr"
    enabled: bool = True
    hidden_dim: int | None = None
    out_dim: int = 128
    kwargs: dict[str, t.Any] = field(default_factory=dict)


@dataclass
class HeadConfig:
    name: str = "classification"
    agg: str = "gated_scalar"
    hidden_dim: int | None = None
    dropout: float = 0.1
    act: str | None = None
    kwargs: dict[str, t.Any] = field(default_factory=dict)


@dataclass
class ModelConfig:
    channels: t.List[ChannelConfig]
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    projection: ProjectionConfig = field(default_factory=ProjectionConfig)
    head: HeadConfig | None = None


@dataclass
class LossConfig:
    name: str
    temperature: float = 0.2
    params: dict[str, t.Any] = field(default_factory=dict)


@dataclass
class PretrainConfigBundle:
    model: ModelConfig
    loss: LossConfig


@dataclass
class FinetuneConfigBundle:
    model: ModelConfig


def _require_channels(model_block: dict[str, t.Any]) -> t.List[ChannelConfig]:
    channels_raw = model_block.get("channels")
    if not channels_raw:
        raise ValueError("YAML config must supply model.channels list.")
    if not isinstance(channels_raw, list):
        raise ValueError("model.channels must be a list of channel specs.")
    return [ChannelConfig(**item) for item in channels_raw]


def _build_head_config(model_block: dict[str, t.Any]) -> HeadConfig | None:
    head_raw = model_block.get("head")
    if head_raw is None:
        return None
    if not isinstance(head_raw, dict):
        raise ValueError("model.head must be a mapping if provided.")
    return HeadConfig(**head_raw)


def _build_loss(loss_block: dict[str, t.Any]) -> LossConfig:
    if "name" not in loss_block:
        raise ValueError("loss.name is required in YAML config.")
    return LossConfig(**loss_block)


def validate_model_config(model_cfg: ModelConfig) -> int:
    """Checks model config sanity and returns the shared channel feature dim."""
    out_dims = {ch.out_dim for ch in model_cfg.channels}
    if len(out_dims) != 1:
        raise ValueError(
            "All channels must share the same out_dim for now. "
            f"Got: {sorted(out_dims)}"
        )
    return next(iter(out_dims))


def load_pretrain_config(path: str | Path) -> PretrainConfigBundle:
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError("Top-level YAML must be a mapping with model/loss blocks.")

    model_block = data.get("model", {})
    loss_block = data.get("loss", {})

    channels = _require_channels(model_block)
    backbone = BackboneConfig(**(model_block.get("backbone") or {}))
    projection = ProjectionConfig(**(model_block.get("projection") or {}))
    head = _build_head_config(model_block)
    model_cfg = ModelConfig(
        channels=channels,
        backbone=backbone,
        projection=projection,
        head=head,
    )

    loss_cfg = _build_loss(loss_block)
    return PretrainConfigBundle(model=model_cfg, loss=loss_cfg)


def load_finetune_config(path: str | Path) -> FinetuneConfigBundle:
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError("Top-level YAML must be a mapping with a model block.")
    model_block = data.get("model", {})
    channels = _require_channels(model_block)
    backbone = BackboneConfig(**(model_block.get("backbone") or {}))
    projection = ProjectionConfig(**(model_block.get("projection") or {}))
    head = _build_head_config(model_block)
    model_cfg = ModelConfig(
        channels=channels,
        backbone=backbone,
        projection=projection,
        head=head,
    )
    return FinetuneConfigBundle(model=model_cfg)


__all__ = [
    "FinetuneConfigBundle",
    "PretrainConfigBundle",
    "BackboneConfig",
    "ChannelConfig",
    "PretrainConfigBundle",
    "HeadConfig",
    "LossConfig",
    "ModelConfig",
    "ProjectionConfig",
    "load_finetune_config",
    "load_pretrain_config",
    "validate_model_config",
]
