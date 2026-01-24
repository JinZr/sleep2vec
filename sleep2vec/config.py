from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import typing as t

import yaml


@dataclass
class TokenizerConfig:
    name: str = "linear"
    out_dim: int | None = None
    kwargs: dict[str, t.Any] = field(default_factory=dict)


@dataclass
class ChannelConfig:
    name: str
    input_dim: int
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)


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
class ClsConfig:
    """
    CLS handling strategy
      embedding_type: null / "bert"
      downstream: "cls" / "tokens"
    """

    downstream: str
    embedding_type: str | None = None
    kwargs: dict[str, t.Any] = field(default_factory=dict)


@dataclass
class HeadConfig:
    channel_agg: "ChannelAggConfig"
    temporal_agg: "TemporalAggConfig"
    name: str = "classification"
    hidden_dim: int | None = None
    dropout: float = 0.1
    act: str | None = None
    kwargs: dict[str, t.Any] = field(default_factory=dict)


@dataclass
class TemporalAggConfig:
    name: str = "mean"  # "mean" or "attn"
    kwargs: dict[str, t.Any] = field(default_factory=dict)


@dataclass
class ChannelAggConfig:
    name: str = "gated_scalar"  # "mean" | "concat" | "gated_scalar"
    kwargs: dict[str, t.Any] = field(default_factory=dict)


@dataclass
class ModelConfig:
    channels: t.List[ChannelConfig]
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    projection: ProjectionConfig = field(default_factory=ProjectionConfig)
    cls: ClsConfig | None = None
    head: HeadConfig | None = None


@dataclass
class LossConfig:
    name: str
    temperature: float = 0.2
    params: dict[str, t.Any] = field(default_factory=dict)


@dataclass
class ModelAveragingConfig:
    name: str | None = None
    params: dict[str, t.Any] = field(default_factory=dict)


@dataclass
class PretrainConfigBundle:
    model: ModelConfig
    loss: LossConfig
    data: "PretrainDataConfig"
    averaging: "ModelAveragingConfig | None" = None


@dataclass
class FinetuneConfigBundle:
    model: ModelConfig
    data: "FinetuneDataConfig"
    lora: "LoraConfig"
    averaging: "ModelAveragingConfig | None" = None


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
    channels: list[ChannelConfig] = []
    for item in channels_raw:
        if not isinstance(item, dict):
            raise ValueError("Each channel must be a mapping.")

        item = dict(item)  # shallow copy
        tok_raw = item.pop("tokenizer", None)
        if not isinstance(tok_raw, dict):
            raise ValueError("channel.tokenizer must be a mapping with keys: name, out_dim[, kwargs].")

        tok_cfg = TokenizerConfig(
            name=tok_raw.get("name") or tok_raw.get("type") or "linear",
            out_dim=tok_raw.get("out_dim"),
            kwargs=tok_raw.get("kwargs") or {},
        )

        if tok_cfg.out_dim is None:
            raise ValueError(f"channel '{item.get('name', '?')}' must set tokenizer.out_dim.")

        channels.append(ChannelConfig(tokenizer=tok_cfg, **item))
    return channels


def _build_cls_config(model_block: dict[str, t.Any]) -> ClsConfig | None:
    raw = model_block.get("cls")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("model.cls must be a mapping when provided.")
    downstream = raw.get("downstream")
    if downstream not in {"cls", "tokens"}:
        raise ValueError("model.cls.downstream is required and must be 'cls' or 'tokens'.")
    return ClsConfig(**raw)


def _build_head_config(model_block: dict[str, t.Any], *, required: bool) -> HeadConfig | None:
    head_raw = model_block.get("head")
    if head_raw is None:
        if required:
            raise ValueError("model.head must be a mapping and is required.")
        return None
    if not isinstance(head_raw, dict):
        raise ValueError("model.head must be a mapping when provided.")

    temporal_block = head_raw.pop("temporal_agg", None)
    channel_block = head_raw.pop("channel_agg", None)
    if temporal_block is None:
        raise ValueError("model.head.temporal_agg is required; specify name: mean|attn.")
    if channel_block is None:
        raise ValueError("model.head.channel_agg is required; specify name: mean|concat|gated_scalar.")

    temporal_cfg = (
        TemporalAggConfig(**temporal_block)
        if isinstance(temporal_block, dict)
        else TemporalAggConfig(name=temporal_block)
    )
    channel_cfg = (
        ChannelAggConfig(**channel_block) if isinstance(channel_block, dict) else ChannelAggConfig(name=channel_block)
    )

    return HeadConfig(temporal_agg=temporal_cfg, channel_agg=channel_cfg, **head_raw)


def _build_loss(loss_block: dict[str, t.Any]) -> LossConfig:
    if "name" not in loss_block:
        raise ValueError("loss.name is required in YAML config.")
    return LossConfig(**loss_block)


def validate_model_config(model_cfg: ModelConfig) -> int:
    """Checks model config sanity and returns the shared channel feature dim."""
    out_dims = {ch.tokenizer.out_dim for ch in model_cfg.channels}
    if None in out_dims:
        raise ValueError("All channels must specify tokenizer.out_dim.")
    if len(out_dims) != 1:
        raise ValueError("All channels must share the same out_dim for now. " f"Got: {sorted(out_dims)}")

    if model_cfg.cls:
        if model_cfg.cls.embedding_type not in {None, "bert", "null", "none"}:
            raise ValueError("model.cls.embedding_type must be null/none or 'bert'.")
        if model_cfg.cls.downstream not in {"cls", "tokens"}:
            raise ValueError("model.cls.downstream must be 'cls' or 'tokens'.")
        if model_cfg.cls.downstream == "cls" and model_cfg.cls.embedding_type in {None, "null", "none"}:
            raise ValueError("model.cls.embedding_type must be set when model.cls.downstream is 'cls'.")

    if model_cfg.head is not None:
        if model_cfg.head.temporal_agg.name not in {"mean", "attn"}:
            raise ValueError("model.head.temporal_agg.name must be 'mean' or 'attn'.")
        if model_cfg.head.channel_agg.name not in {"mean", "concat", "gated_scalar"}:
            raise ValueError("model.head.channel_agg.name must be 'mean', 'concat', or 'gated_scalar'.")
    return next(iter(out_dims))


def _build_model_averaging_config(data: dict[str, t.Any]) -> ModelAveragingConfig | None:
    """Parses the model_averaging block; returns None when absent."""
    averaging_block = data.get("model_averaging")
    if averaging_block is None:
        return None
    if not isinstance(averaging_block, dict):
        raise ValueError("model_averaging block must be a mapping when provided.")
    if not averaging_block:
        return None

    params = dict(averaging_block.get("params") or {})
    for key, value in averaging_block.items():
        if key in {"name", "params"}:
            continue
        params.setdefault(key, value)

    name = averaging_block.get("name")
    if name is None:
        raise ValueError("model_averaging.name is required when model_averaging block is provided.")

    return ModelAveragingConfig(name=name, params=params)


def load_pretrain_config(path: str | Path) -> PretrainConfigBundle:
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError("Top-level YAML must be a mapping with model/loss blocks.")

    model_block = data.get("model", {})
    loss_block = data.get("loss", {})
    data_block = data.get("data", {})

    channels = _require_channels(model_block)
    backbone = BackboneConfig(**(model_block.get("backbone") or {}))
    projection = ProjectionConfig(**(model_block.get("projection") or {}))
    cls_cfg = _build_cls_config(model_block)
    head = _build_head_config(model_block, required=False)
    model_cfg = ModelConfig(
        channels=channels,
        backbone=backbone,
        projection=projection,
        cls=cls_cfg,
        head=head,
    )

    loss_cfg = _build_loss(loss_block)
    data_cfg = PretrainDataConfig(**data_block)
    averaging_cfg = _build_model_averaging_config(data)
    return PretrainConfigBundle(model=model_cfg, loss=loss_cfg, data=data_cfg, averaging=averaging_cfg)


def load_finetune_config(path: str | Path) -> FinetuneConfigBundle:
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError("Top-level YAML must be a mapping with a model block.")
    model_block = data.get("model", {})
    data_block = data.get("data", {})
    lora_block = data.get("lora", {})
    averaging_cfg = _build_model_averaging_config(data)
    channels = _require_channels(model_block)
    backbone = BackboneConfig(**(model_block.get("backbone") or {}))
    projection = ProjectionConfig(**(model_block.get("projection") or {}))
    cls_cfg = _build_cls_config(model_block)
    head = _build_head_config(model_block, required=True)
    model_cfg = ModelConfig(
        channels=channels,
        backbone=backbone,
        projection=projection,
        cls=cls_cfg,
        head=head,
    )
    data_cfg = FinetuneDataConfig(**data_block)
    lora_cfg = LoraConfig(**lora_block)
    return FinetuneConfigBundle(model=model_cfg, data=data_cfg, lora=lora_cfg, averaging=averaging_cfg)


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
    "ClsConfig",
    "TemporalAggConfig",
    "ModelAveragingConfig",
    "ProjectionConfig",
    "LoraConfig",
    "load_finetune_config",
    "load_pretrain_config",
    "validate_model_config",
]
