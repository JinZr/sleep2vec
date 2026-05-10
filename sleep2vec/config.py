from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import typing as t

import yaml

DATA_BACKEND_CHOICES = ("npz", "kaldi")


def _validate_data_backend(backend: str) -> str:
    if backend not in DATA_BACKEND_CHOICES:
        raise ValueError(f"data.backend must be one of {DATA_BACKEND_CHOICES}, got {backend!r}.")
    return backend


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
class EvalVisualizationPlotConfig:
    enabled: bool = False


@dataclass
class ConfusionMatrixVisualizationConfig:
    enabled: bool = False
    show_raw_counts: bool = False


@dataclass
class EvalVisualizationsConfig:
    enabled: bool = False
    stages: t.List[str] = field(default_factory=lambda: ["val", "test"])
    confusion_matrix: ConfusionMatrixVisualizationConfig = field(default_factory=ConfusionMatrixVisualizationConfig)
    roc_curve: EvalVisualizationPlotConfig = field(default_factory=EvalVisualizationPlotConfig)
    regression_scatter: EvalVisualizationPlotConfig = field(default_factory=EvalVisualizationPlotConfig)


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
class AdaptStage1Config:
    train_shared_projection: bool = False


@dataclass
class AdaptLrScalesConfig:
    encoder: float = 0.1
    shared_legacy: float = 0.5
    new_modalities: float = 1.0


@dataclass
class AdaptPairSchedulePoint:
    until: float
    new_pair_ratio: float


def _default_adapt_pair_schedule() -> list["AdaptPairSchedulePoint"]:
    return [
        AdaptPairSchedulePoint(until=0.25, new_pair_ratio=1.0),
        AdaptPairSchedulePoint(until=0.50, new_pair_ratio=0.7),
        AdaptPairSchedulePoint(until=0.75, new_pair_ratio=0.5),
        AdaptPairSchedulePoint(until=1.0, new_pair_ratio=0.0),
    ]


@dataclass
class AdaptStage2Config:
    lr_scales: AdaptLrScalesConfig = field(default_factory=AdaptLrScalesConfig)
    pair_schedule: list[AdaptPairSchedulePoint] = field(default_factory=_default_adapt_pair_schedule)


@dataclass
class AdaptConfig:
    new_channels: list[str]
    stage1: AdaptStage1Config = field(default_factory=AdaptStage1Config)
    stage2: AdaptStage2Config = field(default_factory=AdaptStage2Config)


@dataclass
class PretrainConfigBundle:
    model: ModelConfig
    loss: LossConfig
    data: "PretrainDataConfig"
    averaging: "ModelAveragingConfig | None" = None
    adapt: "AdaptConfig | None" = None


@dataclass
class FinetuneConfigBundle:
    model: ModelConfig
    data: "FinetuneDataConfig"
    finetune: "FinetuneConfig"
    averaging: "ModelAveragingConfig | None" = None


@dataclass
class FinetuneDataConfig:
    backend: str = "npz"
    kaldi_data_root: str | None = None
    kaldi_manifest: str | None = None
    max_tokens: int = 120
    data_channel_names: t.List[str] | None = None
    finetune_data_index: str | None = None
    finetune_preset_path: str | None = None
    train_dataset_names: t.List[str] | None = None
    test_dataset_names: t.List[str] | None = None
    n_few_shot: int = 1280

    def __post_init__(self) -> None:
        self.backend = _validate_data_backend(self.backend)


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
    eval_visualizations: EvalVisualizationsConfig | None = None


@dataclass
class PretrainDataConfig:
    backend: str = "npz"
    kaldi_data_root: str | None = None
    kaldi_manifest: str | None = None
    mask_rate: float = 0.15
    max_tokens: int = 120

    def __post_init__(self) -> None:
        self.backend = _validate_data_backend(self.backend)


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


def _build_model_config(model_block: t.Any, *, require_head: bool) -> ModelConfig:
    if not isinstance(model_block, dict):
        raise ValueError("model block must be a mapping.")
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
    head = _build_head_config(model_block, required=require_head)
    return ModelConfig(
        channels=channels,
        backbone=backbone,
        projection=projection,
        cls=cls_cfg,
        head=head,
    )


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


def _build_eval_visualization_plot_config(raw: t.Any, *, field_name: str) -> EvalVisualizationPlotConfig:
    if raw is None:
        return EvalVisualizationPlotConfig()
    if not isinstance(raw, dict):
        raise ValueError(f"finetune.eval_visualizations.{field_name} must be a mapping when provided.")

    extra = sorted(set(raw.keys()) - {"enabled"})
    if extra:
        raise ValueError(f"finetune.eval_visualizations.{field_name} has unsupported fields: {extra}")

    enabled = raw.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError(f"finetune.eval_visualizations.{field_name}.enabled must be a boolean.")

    return EvalVisualizationPlotConfig(enabled=enabled)


def _build_confusion_matrix_visualization_config(raw: t.Any) -> ConfusionMatrixVisualizationConfig:
    if raw is None:
        return ConfusionMatrixVisualizationConfig()
    if not isinstance(raw, dict):
        raise ValueError("finetune.eval_visualizations.confusion_matrix must be a mapping when provided.")

    extra = sorted(set(raw.keys()) - {"enabled", "show_raw_counts"})
    if extra:
        raise ValueError(f"finetune.eval_visualizations.confusion_matrix has unsupported fields: {extra}")

    enabled = raw.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("finetune.eval_visualizations.confusion_matrix.enabled must be a boolean.")

    show_raw_counts = raw.get("show_raw_counts", False)
    if not isinstance(show_raw_counts, bool):
        raise ValueError("finetune.eval_visualizations.confusion_matrix.show_raw_counts must be a boolean.")

    return ConfusionMatrixVisualizationConfig(enabled=enabled, show_raw_counts=show_raw_counts)


def _build_eval_visualizations_config(raw: t.Any) -> EvalVisualizationsConfig | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("finetune.eval_visualizations must be a mapping when provided.")

    allowed = {"enabled", "stages", "confusion_matrix", "roc_curve", "regression_scatter"}
    extra = sorted(set(raw.keys()) - allowed)
    if extra:
        raise ValueError(f"finetune.eval_visualizations has unsupported fields: {extra}")

    enabled = raw.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("finetune.eval_visualizations.enabled must be a boolean.")

    stages = raw.get("stages", ["val", "test"])
    if not isinstance(stages, list) or not stages:
        raise ValueError("finetune.eval_visualizations.stages must be a non-empty list.")
    if not all(isinstance(stage, str) for stage in stages):
        raise ValueError("finetune.eval_visualizations.stages must be a list of strings.")
    if len(set(stages)) != len(stages):
        raise ValueError("finetune.eval_visualizations.stages must not contain duplicates.")
    invalid_stages = [stage for stage in stages if stage not in {"val", "test"}]
    if invalid_stages:
        raise ValueError(
            "finetune.eval_visualizations.stages only supports 'val' and 'test'. " f"Got: {invalid_stages}"
        )

    return EvalVisualizationsConfig(
        enabled=enabled,
        stages=list(stages),
        confusion_matrix=_build_confusion_matrix_visualization_config(raw.get("confusion_matrix")),
        roc_curve=_build_eval_visualization_plot_config(raw.get("roc_curve"), field_name="roc_curve"),
        regression_scatter=_build_eval_visualization_plot_config(
            raw.get("regression_scatter"),
            field_name="regression_scatter",
        ),
    )


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


def _build_adapt_stage1_config(raw: t.Any) -> AdaptStage1Config:
    if raw is None:
        return AdaptStage1Config()
    if not isinstance(raw, dict):
        raise ValueError("adapt.stage1 must be a mapping when provided.")
    return AdaptStage1Config(**raw)


def _build_adapt_lr_scales(raw: t.Any) -> AdaptLrScalesConfig:
    if raw is None:
        return AdaptLrScalesConfig()
    if not isinstance(raw, dict):
        raise ValueError("adapt.stage2.lr_scales must be a mapping when provided.")
    return AdaptLrScalesConfig(**raw)


def _build_adapt_pair_schedule(raw: t.Any) -> list[AdaptPairSchedulePoint]:
    if raw is None:
        return _default_adapt_pair_schedule()
    if not isinstance(raw, list) or not raw:
        raise ValueError("adapt.stage2.pair_schedule must be a non-empty list when provided.")

    schedule: list[AdaptPairSchedulePoint] = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("Each adapt.stage2.pair_schedule item must be a mapping.")
        if "until" not in item or "new_pair_ratio" not in item:
            raise ValueError("Each adapt.stage2.pair_schedule item must contain 'until' and 'new_pair_ratio'.")
        schedule.append(
            AdaptPairSchedulePoint(
                until=float(item["until"]),
                new_pair_ratio=float(item["new_pair_ratio"]),
            )
        )
    return schedule


def _build_adapt_stage2_config(raw: t.Any) -> AdaptStage2Config:
    if raw is None:
        return AdaptStage2Config()
    if not isinstance(raw, dict):
        raise ValueError("adapt.stage2 must be a mapping when provided.")
    return AdaptStage2Config(
        lr_scales=_build_adapt_lr_scales(raw.get("lr_scales")),
        pair_schedule=_build_adapt_pair_schedule(raw.get("pair_schedule")),
    )


def _build_adapt_config(raw: t.Any) -> AdaptConfig | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("adapt block must be a mapping when provided.")
    new_channels = raw.get("new_channels")
    if not isinstance(new_channels, list) or not new_channels:
        raise ValueError("adapt.new_channels is required and must be a non-empty list.")
    if not all(isinstance(name, str) and name for name in new_channels):
        raise ValueError("adapt.new_channels must contain non-empty strings.")
    if len(set(new_channels)) != len(new_channels):
        raise ValueError("adapt.new_channels must not contain duplicates.")

    return AdaptConfig(
        new_channels=list(new_channels),
        stage1=_build_adapt_stage1_config(raw.get("stage1")),
        stage2=_build_adapt_stage2_config(raw.get("stage2")),
    )


def _validate_adapt_config(adapt_cfg: AdaptConfig | None, model_cfg: ModelConfig) -> None:
    if adapt_cfg is None:
        return

    channel_names = {channel.name for channel in model_cfg.channels}
    missing = [name for name in adapt_cfg.new_channels if name not in channel_names]
    if missing:
        raise ValueError(
            "adapt.new_channels must be present in model.channels. "
            f"Missing: {missing}; available: {sorted(channel_names)}"
        )

    schedule = adapt_cfg.stage2.pair_schedule
    last_until = 0.0
    for point in schedule:
        if not (0.0 < point.until <= 1.0):
            raise ValueError("adapt.stage2.pair_schedule.until values must be in (0, 1].")
        if point.until < last_until:
            raise ValueError("adapt.stage2.pair_schedule.until values must be non-decreasing.")
        if not (0.0 <= point.new_pair_ratio <= 1.0):
            raise ValueError("adapt.stage2.pair_schedule.new_pair_ratio values must be in [0, 1].")
        last_until = point.until

    if schedule and abs(schedule[-1].until - 1.0) > 1e-8:
        raise ValueError("adapt.stage2.pair_schedule must end with until=1.0.")
    for field_name, value in vars(adapt_cfg.stage2.lr_scales).items():
        if value < 0.0:
            raise ValueError(f"adapt.stage2.lr_scales.{field_name} must be >= 0.")


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


def _load_yaml_mapping(path: str | Path, *, error_message: str) -> dict[str, t.Any]:
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError(error_message)
    return data


def load_model_config(path: str | Path, *, require_head: bool = False) -> ModelConfig:
    data = _load_yaml_mapping(path, error_message="Top-level YAML must be a mapping with a model block.")
    return _build_model_config(data.get("model", {}), require_head=require_head)


def load_pretrain_config(path: str | Path) -> PretrainConfigBundle:
    data = _load_yaml_mapping(path, error_message="Top-level YAML must be a mapping with model/loss blocks.")

    model_block = data.get("model", {})
    loss_block = data.get("loss", {})
    data_block = data.get("data", {})
    model_cfg = _build_model_config(model_block, require_head=False)

    loss_cfg = _build_loss(loss_block)
    data_cfg = PretrainDataConfig(**data_block)
    averaging_cfg = _build_model_averaging_config(data)
    adapt_cfg = _build_adapt_config(data.get("adapt"))
    _validate_adapt_config(adapt_cfg, model_cfg)
    return PretrainConfigBundle(model=model_cfg, loss=loss_cfg, data=data_cfg, averaging=averaging_cfg, adapt=adapt_cfg)


def load_finetune_config(path: str | Path) -> FinetuneConfigBundle:
    data = _load_yaml_mapping(path, error_message="Top-level YAML must be a mapping with a model block.")
    model_block = data.get("model", {})
    data_block = data.get("data", {})
    finetune_block = data.get("finetune")
    if finetune_block is None:
        raise ValueError("Finetune YAML must include a top-level 'finetune' block.")
    if not isinstance(finetune_block, dict):
        raise ValueError("finetune block must be a mapping.")
    lora_block = finetune_block.get("lora", {})
    averaging_cfg = _build_model_averaging_config(data)
    model_cfg = _build_model_config(model_block, require_head=True)
    layer_mix_cfg = _build_layer_mix_config(finetune_block.get("layer_mix"))
    eval_visualizations_cfg = _build_eval_visualizations_config(finetune_block.get("eval_visualizations"))
    task_cfg = _build_task_config(finetune_block.get("task"))
    _validate_layer_mix_config(layer_mix_cfg, model_cfg.backbone)
    data_cfg = FinetuneDataConfig(**data_block)
    lora_cfg = LoraConfig(**lora_block)
    finetune_cfg = FinetuneConfig(
        freeze_tokenizer=finetune_block.get("freeze_tokenizer", True),
        lora=lora_cfg,
        layer_mix=layer_mix_cfg,
        task=task_cfg,
        eval_visualizations=eval_visualizations_cfg,
    )
    return FinetuneConfigBundle(model=model_cfg, data=data_cfg, finetune=finetune_cfg, averaging=averaging_cfg)


__all__ = [
    "DATA_BACKEND_CHOICES",
    "FinetuneConfigBundle",
    "FinetuneConfig",
    "PretrainConfigBundle",
    "FinetuneDataConfig",
    "PretrainDataConfig",
    "BackboneConfig",
    "AdaptConfig",
    "AdaptLrScalesConfig",
    "AdaptPairSchedulePoint",
    "AdaptStage1Config",
    "AdaptStage2Config",
    "ChannelConfig",
    "HeadConfig",
    "LossConfig",
    "ModelConfig",
    "ClsConfig",
    "LayerMixConfig",
    "EvalVisualizationPlotConfig",
    "EvalVisualizationsConfig",
    "TemporalAggConfig",
    "ModelAveragingConfig",
    "ProjectionConfig",
    "LoraConfig",
    "TaskConfig",
    "load_finetune_config",
    "load_pretrain_config",
    "validate_model_config",
]
