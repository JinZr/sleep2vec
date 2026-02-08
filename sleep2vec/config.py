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
class LayerMixConfig:
    """Learned scalar mix across transformer layers (blocks 1..L)."""

    enabled: bool = False
    shared_across_modalities: bool = False
    layer_indices: t.List[int] | None = None


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
    finetune: "FinetuneConfig"
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
    token_sec: int = 30


@dataclass
class LoraConfig:
    freeze_backbone_and_insert_lora: bool = False
    insert_lora: bool = True
    separate_adapters: bool = False


@dataclass
class TaskConfig:
    type: str
    output_dim: int
    is_seq: bool
    monitor: str
    monitor_mod: str


@dataclass
class FinetuneConfig:
    freeze_tokenizer: bool = True
    lora: LoraConfig = field(default_factory=LoraConfig)
    layer_mix: LayerMixConfig | None = None
    task: TaskConfig | None = None


@dataclass
class PretrainDataConfig:
    mask_rate: float = 0.15
    max_tokens: int = 120
    token_sec: int = 30


_PSG_SAMPLE_RATES: dict[str, int] = {
    "eeg_original": 128,
    "ecg_original": 128,
    "eog_original": 128,
    "emg_original": 128,
    "heartbeat": 4,
    "breath": 4,
    "spo2": 4,
    "resp_original": 4,
    "resp_nasal_original": 4,
}


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

        tok_name = tok_raw.get("name")
        if not tok_name:
            raise ValueError(f"channel '{item.get('name', '?')}' must set tokenizer.name.")

        tok_cfg = TokenizerConfig(
            name=tok_name,
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


def _build_layer_mix_config(raw: t.Any) -> LayerMixConfig | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("layer_mix must be a mapping.")

    layer_indices = raw.get("layer_indices")
    if layer_indices is not None:
        if not isinstance(layer_indices, list) or not layer_indices:
            raise ValueError("layer_mix.layer_indices must be a non-empty list when provided.")
        if not all(isinstance(idx, int) for idx in layer_indices):
            raise ValueError("layer_mix.layer_indices must be a list of integers.")
        if any(idx < 1 for idx in layer_indices):
            raise ValueError("layer_mix.layer_indices values must be >= 1 (transformer blocks are 1-indexed).")
        if len(set(layer_indices)) != len(layer_indices):
            raise ValueError("layer_mix.layer_indices must not contain duplicates.")
        raw = dict(raw)
        raw["layer_indices"] = layer_indices

    return LayerMixConfig(**raw)


def _build_task_config(raw: t.Any) -> TaskConfig | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("finetune.task must be a mapping when provided.")

    required = {"type", "output_dim", "is_seq", "monitor", "monitor_mod"}
    missing = sorted(required - set(raw.keys()))
    if missing:
        raise ValueError(f"finetune.task missing required fields: {missing}")

    task_type = raw.get("type")
    if task_type not in {"classification", "regression"}:
        raise ValueError("finetune.task.type must be 'classification' or 'regression'.")

    output_dim = raw.get("output_dim")
    if not isinstance(output_dim, int) or output_dim < 1:
        raise ValueError("finetune.task.output_dim must be a positive integer.")

    is_seq = raw.get("is_seq")
    if not isinstance(is_seq, bool):
        raise ValueError("finetune.task.is_seq must be a boolean.")

    monitor = raw.get("monitor")
    if not isinstance(monitor, str) or not monitor:
        raise ValueError("finetune.task.monitor must be a non-empty string.")

    monitor_mod = raw.get("monitor_mod")
    if monitor_mod not in {"min", "max"}:
        raise ValueError("finetune.task.monitor_mod must be 'min' or 'max'.")

    extra = sorted(set(raw.keys()) - required)
    if extra:
        raise ValueError(f"finetune.task has unsupported fields: {extra}")

    if task_type == "classification" and output_dim < 2:
        raise ValueError("finetune.task.output_dim must be >= 2 for classification tasks.")
    if task_type == "regression" and output_dim != 1:
        raise ValueError("finetune.task.output_dim must be 1 for regression tasks.")

    return TaskConfig(
        type=task_type,
        output_dim=output_dim,
        is_seq=is_seq,
        monitor=monitor,
        monitor_mod=monitor_mod,
    )


def _validate_layer_mix_config(layer_mix_cfg: LayerMixConfig | None, backbone_cfg: BackboneConfig) -> None:
    if layer_mix_cfg is None or not layer_mix_cfg.layer_indices:
        return
    max_layers = backbone_cfg.num_hidden_layers
    if any(idx > max_layers for idx in layer_mix_cfg.layer_indices):
        raise ValueError(
            f"layer_mix.layer_indices must be <= num_hidden_layers ({max_layers}); "
            f"got {layer_mix_cfg.layer_indices}."
        )


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


def validate_token_sec_config(model_cfg: ModelConfig, token_sec: int, *, context: str) -> None:
    if token_sec <= 0:
        raise ValueError(f"{context}: data.token_sec must be a positive integer. Got: {token_sec}")

    mismatches: list[str] = []
    for ch in model_cfg.channels:
        rate = _PSG_SAMPLE_RATES.get(ch.name)
        if rate is None:
            continue
        expected = rate * token_sec
        if ch.input_dim != expected:
            mismatches.append(
                f"{ch.name}: input_dim={ch.input_dim}, expected={expected} "
                f"(rate={rate}Hz, token_sec={token_sec})"
            )

    if mismatches:
        msg = "\n".join(mismatches)
        raise ValueError(f"{context}: token_sec/input_dim mismatch:\n{msg}")


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

    if "backbone" not in model_block:
        raise ValueError("model.backbone is required in YAML.")
    if "projection" not in model_block:
        raise ValueError("model.projection is required in YAML.")
    if "cls" not in model_block:
        raise ValueError("model.cls is required in YAML.")

    channels = _require_channels(model_block)
    backbone = BackboneConfig(**model_block.get("backbone"))
    projection = ProjectionConfig(**model_block.get("projection"))
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
    validate_token_sec_config(model_cfg, data_cfg.token_sec, context="pretrain")
    averaging_cfg = _build_model_averaging_config(data)
    return PretrainConfigBundle(model=model_cfg, loss=loss_cfg, data=data_cfg, averaging=averaging_cfg)


def load_finetune_config(path: str | Path) -> FinetuneConfigBundle:
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError("Top-level YAML must be a mapping with a model block.")
    model_block = data.get("model", {})
    data_block = data.get("data", {})
    finetune_block = data.get("finetune")
    if finetune_block is None:
        raise ValueError("Finetune YAML must include a top-level 'finetune' block.")
    if not isinstance(finetune_block, dict):
        raise ValueError("finetune block must be a mapping.")
    lora_block = finetune_block.get("lora", {})
    averaging_cfg = _build_model_averaging_config(data)
    if "backbone" not in model_block:
        raise ValueError("model.backbone is required in YAML.")
    if "projection" not in model_block:
        raise ValueError("model.projection is required in YAML.")
    if "cls" not in model_block:
        raise ValueError("model.cls is required in YAML.")

    channels = _require_channels(model_block)
    backbone = BackboneConfig(**model_block.get("backbone"))
    projection = ProjectionConfig(**model_block.get("projection"))
    cls_cfg = _build_cls_config(model_block)
    layer_mix_cfg = _build_layer_mix_config(finetune_block.get("layer_mix"))
    task_cfg = _build_task_config(finetune_block.get("task"))
    head = _build_head_config(model_block, required=True)
    model_cfg = ModelConfig(
        channels=channels,
        backbone=backbone,
        projection=projection,
        cls=cls_cfg,
        head=head,
    )
    _validate_layer_mix_config(layer_mix_cfg, backbone)
    data_cfg = FinetuneDataConfig(**data_block)
    validate_token_sec_config(model_cfg, data_cfg.token_sec, context="finetune")
    lora_cfg = LoraConfig(**lora_block)
    finetune_cfg = FinetuneConfig(
        freeze_tokenizer=finetune_block.get("freeze_tokenizer", True),
        lora=lora_cfg,
        layer_mix=layer_mix_cfg,
        task=task_cfg,
    )
    return FinetuneConfigBundle(model=model_cfg, data=data_cfg, finetune=finetune_cfg, averaging=averaging_cfg)


__all__ = [
    "FinetuneConfigBundle",
    "FinetuneConfig",
    "PretrainConfigBundle",
    "FinetuneDataConfig",
    "PretrainDataConfig",
    "BackboneConfig",
    "ChannelConfig",
    "HeadConfig",
    "LossConfig",
    "ModelConfig",
    "ClsConfig",
    "LayerMixConfig",
    "TemporalAggConfig",
    "ModelAveragingConfig",
    "ProjectionConfig",
    "LoraConfig",
    "TaskConfig",
    "load_finetune_config",
    "load_pretrain_config",
    "validate_model_config",
]
