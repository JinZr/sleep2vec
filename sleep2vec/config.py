from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import typing as t

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
class EmaConfig:
    enabled: bool = False
    base_momentum: float = 0.996
    final_momentum: float = 1.0
    use_for_eval: bool = True


@dataclass
class PretrainConfigBundle:
    model: ModelConfig
    loss: LossConfig
    data: "PretrainDataConfig"
    ema: "EmaConfig" = EmaConfig()


@dataclass
class FinetuneConfigBundle:
    model: ModelConfig
    data: "FinetuneDataConfig"
    lora: "LoraConfig"


@dataclass
class FinetuneDataConfig:
    max_tokens: int = 120
    data_channel_names: t.List[str] | None = None
    finetune_data_index: str | None = None
    finetune_preset_path: str | None = None
    train_dataset_names: t.List[str] | None = None
    test_dataset_names: t.List[str] | None = None
    n_few_shot: int = 1280


@dataclass
class LoraConfig:
    freeze_backbone_and_insert_lora: bool = False
    insert_lora: bool = True
    separate_adapters: bool = False


@dataclass
class PretrainDataConfig:
    mask_rate: float = 0.15
    max_tokens: int = 120


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
        raise ValueError("All channels must share the same out_dim for now. " f"Got: {sorted(out_dims)}")
    return next(iter(out_dims))


def load_pretrain_config(path: str | Path) -> PretrainConfigBundle:
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError("Top-level YAML must be a mapping with model/loss blocks.")

    model_block = data.get("model", {})
    loss_block = data.get("loss", {})
    data_block = data.get("data", {})
    ema_block = data.get("ema", {}) or {}

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
    data_cfg = PretrainDataConfig(**data_block)
    ema_cfg = EmaConfig(**ema_block)
    return PretrainConfigBundle(model=model_cfg, loss=loss_cfg, data=data_cfg, ema=ema_cfg)


def load_finetune_config(path: str | Path) -> FinetuneConfigBundle:
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError("Top-level YAML must be a mapping with a model block.")
    model_block = data.get("model", {})
    data_block = data.get("data", {})
    lora_block = data.get("lora", {})
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
    data_cfg = FinetuneDataConfig(**data_block)
    lora_cfg = LoraConfig(**lora_block)
    return FinetuneConfigBundle(model=model_cfg, data=data_cfg, lora=lora_cfg)


__all__ = [
    "FinetuneConfigBundle",
    "PretrainConfigBundle",
    "FinetuneDataConfig",
    "PretrainDataConfig",
    "BackboneConfig",
    "ChannelConfig",
    "HeadConfig",
    "LossConfig",
    "ModelConfig",
    "ProjectionConfig",
    "LoraConfig",
    "EmaConfig",
    "load_finetune_config",
    "load_pretrain_config",
    "validate_model_config",
]
