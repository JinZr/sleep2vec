from __future__ import annotations

from dataclasses import dataclass, field
import math
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
class MoeConfig:
    enabled: bool = False
    layer_indices: list[int] | None = None
    num_experts: int = 16
    top_k: int = 2
    expert_hidden_size: int | None = None
    router_type: str = "learned"
    router_noise: float = 0.0
    router_z_loss_coef: float = 0.0
    router_entropy_coef: float = 0.0
    load_balance_coef: float = 0.0
    modality_balance_coef: float = 0.0
    route_consistency_coef: float = 0.0
    expert_diversity_coef: float = 0.0
    expert_dropout_prob: float = 0.0
    use_modality_group_mask: bool = True
    expert_groups: dict[str, list[int]] = field(default_factory=dict)
    modality_to_groups: dict[str, list[str]] = field(default_factory=dict)
    route_consistency_layers: list[int] | None = None


@dataclass
class BackboneConfig:
    name: str = "roformer"
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 16
    vocab_size: int = 1
    attention_backend: str = "eager"
    config_overrides: dict[str, t.Any] = field(default_factory=dict)
    moe: MoeConfig | None = None


def _validate_moe_int_list(value: t.Any, field_name: str, *, required: bool = False) -> list[int] | None:
    if value is None:
        if required:
            raise ValueError(f"backbone.moe.{field_name} must be a non-empty list when MoE is enabled.")
        return None
    if not isinstance(value, list) or not value:
        raise ValueError(f"backbone.moe.{field_name} must be a non-empty list when provided.")
    if not all(type(idx) is int for idx in value):
        raise ValueError(f"backbone.moe.{field_name} must contain only integers.")
    if len(set(value)) != len(value):
        raise ValueError(f"backbone.moe.{field_name} must not contain duplicates.")
    return value


def _validate_moe_nonnegative_number(value: t.Any, field_name: str) -> None:
    if type(value) not in {int, float}:
        raise ValueError(f"backbone.moe.{field_name} must be a number.")
    if not math.isfinite(float(value)):
        raise ValueError(f"backbone.moe.{field_name} must be finite.")
    if value < 0:
        raise ValueError(f"backbone.moe.{field_name} must be >= 0.")


def _validate_moe_config(
    moe_cfg: MoeConfig,
    backbone_cfg: BackboneConfig,
    channel_names: t.Sequence[str] | None = None,
) -> None:
    if type(moe_cfg.router_type) is not str or moe_cfg.router_type not in {
        "learned",
        "random",
        "hard_modality",
        "hard_group",
    }:
        raise ValueError("backbone.moe.router_type must be one of learned, random, hard_modality, hard_group.")
    if type(moe_cfg.enabled) is not bool:
        raise ValueError("backbone.moe.enabled must be a boolean.")
    if type(moe_cfg.use_modality_group_mask) is not bool:
        raise ValueError("backbone.moe.use_modality_group_mask must be a boolean.")
    if type(moe_cfg.num_experts) is not int:
        raise ValueError("backbone.moe.num_experts must be an integer.")
    if moe_cfg.num_experts <= 0:
        raise ValueError("backbone.moe.num_experts must be > 0.")
    if type(moe_cfg.top_k) is not int:
        raise ValueError("backbone.moe.top_k must be an integer.")
    if moe_cfg.top_k < 1:
        raise ValueError("backbone.moe.top_k must be >= 1.")
    if moe_cfg.top_k > moe_cfg.num_experts:
        raise ValueError("backbone.moe.top_k must be <= backbone.moe.num_experts.")
    if moe_cfg.expert_hidden_size is not None:
        if type(moe_cfg.expert_hidden_size) is not int:
            raise ValueError("backbone.moe.expert_hidden_size must be an integer when provided.")
        if moe_cfg.expert_hidden_size <= 0:
            raise ValueError("backbone.moe.expert_hidden_size must be positive when provided.")
    for field_name in (
        "router_noise",
        "router_z_loss_coef",
        "router_entropy_coef",
        "load_balance_coef",
        "modality_balance_coef",
        "route_consistency_coef",
        "expert_diversity_coef",
    ):
        _validate_moe_nonnegative_number(getattr(moe_cfg, field_name), field_name)
    _validate_moe_nonnegative_number(moe_cfg.expert_dropout_prob, "expert_dropout_prob")
    if moe_cfg.expert_dropout_prob > 1:
        raise ValueError("backbone.moe.expert_dropout_prob must be <= 1.")
    if moe_cfg.enabled and moe_cfg.expert_diversity_coef > 0:
        raise ValueError("backbone.moe.expert_diversity_coef is not supported yet and must be 0.0.")

    layer_indices = _validate_moe_int_list(
        moe_cfg.layer_indices,
        "layer_indices",
        required=moe_cfg.enabled,
    )
    if layer_indices is not None and (min(layer_indices) < 1 or max(layer_indices) > backbone_cfg.num_hidden_layers):
        raise ValueError("backbone.moe.layer_indices values must be within [1, backbone.num_hidden_layers].")

    route_consistency_layers = _validate_moe_int_list(moe_cfg.route_consistency_layers, "route_consistency_layers")
    if moe_cfg.enabled and moe_cfg.route_consistency_coef > 0 and route_consistency_layers is None:
        raise ValueError("backbone.moe.route_consistency_layers is required when route_consistency_coef is positive.")
    if route_consistency_layers is not None:
        if layer_indices is None or not set(route_consistency_layers).issubset(set(layer_indices)):
            raise ValueError("backbone.moe.route_consistency_layers must be a subset of backbone.moe.layer_indices.")

    if not (moe_cfg.enabled and moe_cfg.use_modality_group_mask):
        return

    if not isinstance(moe_cfg.expert_groups, dict):
        raise ValueError("backbone.moe.expert_groups must be a mapping.")
    if not isinstance(moe_cfg.modality_to_groups, dict):
        raise ValueError("backbone.moe.modality_to_groups must be a mapping.")
    if not moe_cfg.expert_groups:
        raise ValueError("backbone.moe.expert_groups is required when use_modality_group_mask is enabled.")
    if not moe_cfg.modality_to_groups:
        raise ValueError("backbone.moe.modality_to_groups is required when use_modality_group_mask is enabled.")
    if channel_names is not None:
        missing_modalities = sorted(set(channel_names) - set(moe_cfg.modality_to_groups))
        if missing_modalities:
            raise ValueError(
                "backbone.moe.modality_to_groups must include every configured channel when "
                f"use_modality_group_mask is enabled; missing: {missing_modalities}."
            )

    for group_name, expert_ids in moe_cfg.expert_groups.items():
        if not isinstance(expert_ids, list) or not expert_ids:
            raise ValueError(f"backbone.moe.expert_groups.{group_name} must be a non-empty list.")
        if not all(type(expert_id) is int for expert_id in expert_ids):
            raise ValueError(f"backbone.moe.expert_groups.{group_name} must contain only integer expert ids.")
        if any(expert_id < 0 or expert_id >= moe_cfg.num_experts for expert_id in expert_ids):
            raise ValueError(
                f"backbone.moe.expert_groups.{group_name} expert ids must be within "
                "[0, backbone.moe.num_experts - 1]."
            )

    valid_groups = set(moe_cfg.expert_groups)
    for modality_name, group_names in moe_cfg.modality_to_groups.items():
        if not isinstance(group_names, list) or not group_names:
            raise ValueError(f"backbone.moe.modality_to_groups.{modality_name} must be a non-empty list.")
        missing_groups = [group_name for group_name in group_names if group_name not in valid_groups]
        if missing_groups:
            raise ValueError(
                f"backbone.moe.modality_to_groups.{modality_name} references unknown groups: {missing_groups}."
            )
        allowed_experts = {expert_id for group_name in group_names for expert_id in moe_cfg.expert_groups[group_name]}
        if len(allowed_experts) < moe_cfg.top_k:
            raise ValueError(f"backbone.moe.modality_to_groups.{modality_name} must expose at least top_k experts.")


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
    name: str = "mean"  # "mean", "attn", or "lstm"
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
    insert_lora: bool = False
    separate_adapters: bool = False
    r: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target_modules: t.List[str] = field(default_factory=lambda: ["query", "key", "value"])
    use_dora: bool = False


@dataclass
class TaskConfig:
    type: str
    output_dim: int
    is_seq: bool
    monitor: str
    monitor_mod: str


@dataclass
class FinetuneLrScalesConfig:
    head: float = 1.0
    backbone: float = 0.1
    experts: float = 0.1
    routers: float = 0.0
    tokenizers: float = 0.0
    projection: float = 0.0
    lora: float = 1.0


@dataclass
class FinetuneMoeRegularizationConfig:
    enabled: bool = False
    collect_train_moe_aux: bool = False
    router_z_loss_coef: float = 0.0
    load_balance_coef: float = 0.0
    modality_balance_coef: float = 0.0
    route_consistency_coef: float = 0.0
    entropy_coef: float = 0.0


@dataclass
class FinetuneMoeTuningConfig:
    mode: str = "conservative_full_router_frozen"
    freeze_router: bool | None = None
    freeze_experts: bool | None = None
    train_moe_layer_indices: list[int] | None = None
    lr_scales: FinetuneLrScalesConfig = field(default_factory=FinetuneLrScalesConfig)
    moe_regularization: FinetuneMoeRegularizationConfig = field(default_factory=FinetuneMoeRegularizationConfig)


@dataclass
class FinetuneLossConfig:
    class_weights: t.List[float] | None = None
    pos_weight: float | t.List[float] | None = None


@dataclass
class FinetuneSamplerConfig:
    weighted_random: bool = False


@dataclass
class SurvivalConfig:
    key_column: str
    disease_columns_index: str
    event_time_index: str
    is_event_index: str
    has_label_index: str


@dataclass
class FinetuneConfig:
    freeze_tokenizer: bool = True
    lora: LoraConfig = field(default_factory=LoraConfig)
    layer_mix: LayerMixConfig | None = None
    loss: FinetuneLossConfig = field(default_factory=FinetuneLossConfig)
    sampler: FinetuneSamplerConfig = field(default_factory=FinetuneSamplerConfig)
    task: TaskConfig | None = None
    survival: SurvivalConfig | None = None
    eval_visualizations: EvalVisualizationsConfig | None = None
    moe_tuning: FinetuneMoeTuningConfig | None = None


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
        raise ValueError("model.head.temporal_agg is required; specify name: mean|attn|lstm.")
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


def _build_backbone_config(raw: t.Any, *, channel_names: t.Sequence[str] | None = None) -> BackboneConfig:
    if not isinstance(raw, dict):
        raise ValueError("model.backbone must be a mapping.")

    config_overrides = raw.get("config_overrides") or {}
    if not isinstance(config_overrides, dict):
        raise ValueError("model.backbone.config_overrides must be a mapping when provided.")
    attention_backend = raw.get("attention_backend", "eager")
    if attention_backend not in ("eager", "sdpa"):
        raise ValueError("model.backbone.attention_backend must be one of eager, sdpa.")
    if "attention_backend" in config_overrides:
        raise ValueError(
            "model.backbone.config_overrides.attention_backend is not supported; "
            "use model.backbone.attention_backend."
        )
    if "moe" in config_overrides:
        raise ValueError("model.backbone.config_overrides.moe is not supported; use model.backbone.moe.")

    raw = dict(raw)
    moe_raw = raw.pop("moe", None)
    moe_cfg = None
    if moe_raw is not None:
        if not isinstance(moe_raw, dict):
            raise ValueError("model.backbone.moe must be a mapping when provided.")
        moe_cfg = MoeConfig(**moe_raw)

    backbone = BackboneConfig(**raw, moe=moe_cfg)
    if moe_cfg is not None:
        if backbone.name != "roformer":
            raise ValueError("model.backbone.moe is only supported for backbone.name='roformer'.")
        _validate_moe_config(moe_cfg, backbone, channel_names=channel_names)
    return backbone


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
    backbone = _build_backbone_config(model_block.get("backbone"), channel_names=[channel.name for channel in channels])
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
    if task_type not in {"classification", "regression", "survival"}:
        raise ValueError("finetune.task.type must be 'classification', 'regression', or 'survival'.")

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
    if task_type == "survival":
        if is_seq:
            raise ValueError("finetune.task.is_seq must be false for survival tasks.")
        if monitor != "val_loss" or monitor_mod != "min":
            raise ValueError("survival tasks must monitor val_loss with monitor_mod min.")

    return TaskConfig(
        type=task_type,
        output_dim=output_dim,
        is_seq=is_seq,
        monitor=monitor,
        monitor_mod=monitor_mod,
    )


def _is_positive_number(value: t.Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and float(value) > 0.0


def _build_positive_float_list(raw: t.Any, *, field_name: str) -> t.List[float] | None:
    if raw is None:
        return None
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{field_name} must be null or a non-empty list of positive numbers.")
    if not all(_is_positive_number(value) for value in raw):
        raise ValueError(f"{field_name} must contain only positive numbers.")
    return [float(value) for value in raw]


def _build_finetune_loss_config(raw: t.Any) -> FinetuneLossConfig:
    if raw is None:
        return FinetuneLossConfig()
    if not isinstance(raw, dict):
        raise ValueError("finetune.loss must be a mapping when provided.")

    allowed = {"class_weights", "pos_weight"}
    extra = sorted(set(raw.keys()) - allowed)
    if extra:
        raise ValueError(f"finetune.loss has unsupported fields: {extra}")

    pos_weight = raw.get("pos_weight")
    if pos_weight is not None and not _is_positive_number(pos_weight):
        pos_weight = _build_positive_float_list(pos_weight, field_name="finetune.loss.pos_weight")
    elif pos_weight is not None:
        pos_weight = float(pos_weight)

    return FinetuneLossConfig(
        class_weights=_build_positive_float_list(raw.get("class_weights"), field_name="finetune.loss.class_weights"),
        pos_weight=pos_weight,
    )


def _build_finetune_sampler_config(raw: t.Any) -> FinetuneSamplerConfig:
    if raw is None:
        return FinetuneSamplerConfig()
    if not isinstance(raw, dict):
        raise ValueError("finetune.sampler must be a mapping when provided.")

    allowed = {"weighted_random"}
    extra = sorted(set(raw.keys()) - allowed)
    if extra:
        raise ValueError(f"finetune.sampler has unsupported fields: {extra}")

    weighted_random = raw.get("weighted_random", False)
    if not isinstance(weighted_random, bool):
        raise ValueError("finetune.sampler.weighted_random must be a boolean.")
    return FinetuneSamplerConfig(weighted_random=weighted_random)


def _build_survival_config(raw: t.Any, task_cfg: TaskConfig | None) -> SurvivalConfig | None:
    if raw is None:
        if task_cfg is not None and task_cfg.type == "survival":
            raise ValueError("finetune.survival is required for survival tasks.")
        return None
    if task_cfg is None or task_cfg.type != "survival":
        raise ValueError("finetune.survival is only supported when finetune.task.type is survival.")
    if not isinstance(raw, dict):
        raise ValueError("finetune.survival must be a mapping when provided.")

    required = {"key_column", "disease_columns_index", "event_time_index", "is_event_index", "has_label_index"}
    missing = sorted(required - set(raw.keys()))
    if missing:
        raise ValueError(f"finetune.survival missing required fields: {missing}")
    extra = sorted(set(raw.keys()) - required)
    if extra:
        raise ValueError(f"finetune.survival has unsupported fields: {extra}")
    for field_name in required:
        value = raw[field_name]
        if not isinstance(value, str) or not value:
            raise ValueError(f"finetune.survival.{field_name} must be a non-empty string.")

    return SurvivalConfig(**raw)


_FINETUNE_MOE_TUNING_MODES = {
    "head_only",
    "conservative_full_router_frozen",
    "conservative_full_router_trainable",
    "top_moe_layer_expert_only",
    "custom",
}


def _reject_extra_fields(raw: dict[str, t.Any], allowed: set[str], field_name: str) -> None:
    extra = sorted(set(raw.keys()) - allowed)
    if extra:
        raise ValueError(f"{field_name} has unsupported fields: {extra}")


def _validate_finetune_moe_bool(value: t.Any, field_name: str) -> None:
    if type(value) is not bool:
        raise ValueError(f"{field_name} must be a boolean.")


def _validate_finetune_moe_nonnegative_number(value: t.Any, field_name: str) -> None:
    if type(value) not in {int, float}:
        raise ValueError(f"{field_name} must be a number.")
    if not math.isfinite(float(value)):
        raise ValueError(f"{field_name} must be finite.")
    if value < 0:
        raise ValueError(f"{field_name} must be >= 0.")


def _validate_finetune_moe_layer_indices(value: t.Any) -> list[int] | None:
    if value is None:
        return None
    field_name = "finetune.moe_tuning.train_moe_layer_indices"
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field_name} must be a non-empty list when provided.")
    if not all(type(idx) is int for idx in value):
        raise ValueError(f"{field_name} must contain only integers.")
    if len(set(value)) != len(value):
        raise ValueError(f"{field_name} must not contain duplicates.")
    return list(value)


def _default_finetune_moe_lr_scales(mode: str) -> dict[str, float]:
    if mode == "head_only":
        return {
            "head": 1.0,
            "backbone": 0.0,
            "experts": 0.0,
            "routers": 0.0,
            "tokenizers": 0.0,
            "projection": 0.0,
            "lora": 1.0,
        }
    if mode == "conservative_full_router_trainable":
        return {
            "head": 1.0,
            "backbone": 0.1,
            "experts": 0.1,
            "routers": 0.01,
            "tokenizers": 0.0,
            "projection": 0.0,
            "lora": 1.0,
        }
    if mode == "top_moe_layer_expert_only":
        return {
            "head": 1.0,
            "backbone": 0.0,
            "experts": 0.1,
            "routers": 0.0,
            "tokenizers": 0.0,
            "projection": 0.0,
            "lora": 1.0,
        }
    return {
        "head": 1.0,
        "backbone": 0.1,
        "experts": 0.1,
        "routers": 0.0,
        "tokenizers": 0.0,
        "projection": 0.0,
        "lora": 1.0,
    }


def _build_finetune_lr_scales_config(raw: t.Any, mode: str) -> FinetuneLrScalesConfig:
    allowed = {"head", "backbone", "experts", "routers", "tokenizers", "projection", "lora"}
    values = _default_finetune_moe_lr_scales(mode)
    if raw is not None:
        if not isinstance(raw, dict):
            raise ValueError("finetune.moe_tuning.lr_scales must be a mapping when provided.")
        _reject_extra_fields(raw, allowed, "finetune.moe_tuning.lr_scales")
        values.update(raw)

    cfg = FinetuneLrScalesConfig(**values)
    for field_name in allowed:
        _validate_finetune_moe_nonnegative_number(
            getattr(cfg, field_name),
            f"finetune.moe_tuning.lr_scales.{field_name}",
        )
    return cfg


def _build_finetune_moe_regularization_config(raw: t.Any) -> FinetuneMoeRegularizationConfig:
    allowed = {
        "enabled",
        "collect_train_moe_aux",
        "router_z_loss_coef",
        "load_balance_coef",
        "modality_balance_coef",
        "route_consistency_coef",
        "entropy_coef",
    }
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("finetune.moe_tuning.moe_regularization must be a mapping when provided.")
    _reject_extra_fields(raw, allowed, "finetune.moe_tuning.moe_regularization")

    cfg = FinetuneMoeRegularizationConfig(**raw)
    _validate_finetune_moe_bool(cfg.enabled, "finetune.moe_tuning.moe_regularization.enabled")
    _validate_finetune_moe_bool(
        cfg.collect_train_moe_aux,
        "finetune.moe_tuning.moe_regularization.collect_train_moe_aux",
    )
    for field_name in allowed - {"enabled", "collect_train_moe_aux"}:
        _validate_finetune_moe_nonnegative_number(
            getattr(cfg, field_name),
            f"finetune.moe_tuning.moe_regularization.{field_name}",
        )

    if cfg.enabled and not cfg.collect_train_moe_aux:
        raise ValueError(
            "finetune.moe_tuning.moe_regularization.collect_train_moe_aux must be true "
            "when downstream MoE regularization is enabled."
        )
    unsupported = {
        "route_consistency_coef": "downstream route consistency is not supported yet",
        "load_balance_coef": "downstream load balancing is not supported yet",
        "modality_balance_coef": "downstream modality balancing is not supported yet",
        "entropy_coef": "downstream entropy regularization is not supported yet",
    }
    for field_name, message in unsupported.items():
        if getattr(cfg, field_name) > 0:
            raise ValueError(message)
    return cfg


def _build_finetune_moe_tuning_config(raw: t.Any) -> FinetuneMoeTuningConfig | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("finetune.moe_tuning must be a mapping when provided.")
    allowed = {
        "mode",
        "freeze_router",
        "freeze_experts",
        "train_moe_layer_indices",
        "lr_scales",
        "moe_regularization",
    }
    _reject_extra_fields(raw, allowed, "finetune.moe_tuning")

    if "mode" not in raw:
        raise ValueError("finetune.moe_tuning.mode is required when finetune.moe_tuning is provided.")
    mode = raw.get("mode")
    if type(mode) is not str or mode not in _FINETUNE_MOE_TUNING_MODES:
        raise ValueError("finetune.moe_tuning.mode must be one of " f"{sorted(_FINETUNE_MOE_TUNING_MODES)}.")

    freeze_router = raw.get("freeze_router")
    freeze_experts = raw.get("freeze_experts")
    if mode == "custom":
        if "freeze_router" not in raw or "freeze_experts" not in raw:
            raise ValueError("finetune.moe_tuning.custom requires explicit freeze_router and freeze_experts.")
        _validate_finetune_moe_bool(freeze_router, "finetune.moe_tuning.freeze_router")
        _validate_finetune_moe_bool(freeze_experts, "finetune.moe_tuning.freeze_experts")
    else:
        if "freeze_router" in raw or "freeze_experts" in raw:
            raise ValueError("finetune.moe_tuning.freeze_router/freeze_experts are only supported in custom mode.")
        freeze_router = mode in {"head_only", "conservative_full_router_frozen", "top_moe_layer_expert_only"}
        freeze_experts = mode == "head_only"

    if mode != "top_moe_layer_expert_only" and "train_moe_layer_indices" in raw:
        raise ValueError(
            "finetune.moe_tuning.train_moe_layer_indices is only supported when " "mode is top_moe_layer_expert_only."
        )

    return FinetuneMoeTuningConfig(
        mode=mode,
        freeze_router=freeze_router,
        freeze_experts=freeze_experts,
        train_moe_layer_indices=_validate_finetune_moe_layer_indices(raw.get("train_moe_layer_indices")),
        lr_scales=_build_finetune_lr_scales_config(raw.get("lr_scales"), mode),
        moe_regularization=_build_finetune_moe_regularization_config(raw.get("moe_regularization")),
    )


def _validate_finetune_moe_tuning_config(cfg: FinetuneMoeTuningConfig | None, model_cfg: ModelConfig) -> None:
    if cfg is None:
        return
    moe_cfg = model_cfg.backbone.moe
    moe_enabled = moe_cfg is not None and moe_cfg.enabled
    if cfg.mode != "head_only" and not moe_enabled:
        raise ValueError("finetune.moe_tuning.mode requires model.backbone.moe.enabled=true unless mode is head_only.")
    if cfg.moe_regularization.enabled and not moe_enabled:
        raise ValueError("finetune.moe_tuning.moe_regularization.enabled requires model.backbone.moe.enabled=true.")

    moe_layers = moe_cfg.layer_indices if moe_cfg is not None and moe_cfg.layer_indices is not None else []
    if cfg.train_moe_layer_indices is not None:
        if not moe_enabled:
            raise ValueError("finetune.moe_tuning.train_moe_layer_indices requires model.backbone.moe.enabled=true.")
        invalid = sorted(set(cfg.train_moe_layer_indices) - set(moe_layers))
        if invalid:
            raise ValueError(
                "finetune.moe_tuning.train_moe_layer_indices must be a subset of "
                f"model.backbone.moe.layer_indices. Invalid: {invalid}."
            )

    if cfg.mode == "top_moe_layer_expert_only" and cfg.train_moe_layer_indices is None:
        cfg.train_moe_layer_indices = [max(moe_layers)]


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
        if model_cfg.head.temporal_agg.name not in {"mean", "attn", "lstm"}:
            raise ValueError("model.head.temporal_agg.name must be 'mean', 'attn', or 'lstm'.")
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
    loss_cfg = _build_finetune_loss_config(finetune_block.get("loss"))
    sampler_cfg = _build_finetune_sampler_config(finetune_block.get("sampler"))
    eval_visualizations_cfg = _build_eval_visualizations_config(finetune_block.get("eval_visualizations"))
    task_cfg = _build_task_config(finetune_block.get("task"))
    survival_cfg = _build_survival_config(finetune_block.get("survival"), task_cfg)
    moe_tuning_cfg = _build_finetune_moe_tuning_config(finetune_block.get("moe_tuning"))
    _validate_layer_mix_config(layer_mix_cfg, model_cfg.backbone)
    _validate_finetune_moe_tuning_config(moe_tuning_cfg, model_cfg)
    data_cfg = FinetuneDataConfig(**data_block)
    lora_cfg = LoraConfig(**lora_block)
    router_targets = [target for target in lora_cfg.target_modules if "router" in target.lower()]
    if router_targets:
        raise ValueError("sleep2expert LoRA does not support router target modules.")
    finetune_cfg = FinetuneConfig(
        freeze_tokenizer=finetune_block.get("freeze_tokenizer", True),
        lora=lora_cfg,
        layer_mix=layer_mix_cfg,
        loss=loss_cfg,
        sampler=sampler_cfg,
        task=task_cfg,
        survival=survival_cfg,
        eval_visualizations=eval_visualizations_cfg,
        moe_tuning=moe_tuning_cfg,
    )
    return FinetuneConfigBundle(model=model_cfg, data=data_cfg, finetune=finetune_cfg, averaging=averaging_cfg)


__all__ = [
    "DATA_BACKEND_CHOICES",
    "FinetuneConfigBundle",
    "FinetuneConfig",
    "FinetuneLossConfig",
    "FinetuneSamplerConfig",
    "FinetuneLrScalesConfig",
    "FinetuneMoeRegularizationConfig",
    "FinetuneMoeTuningConfig",
    "SurvivalConfig",
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
    "MoeConfig",
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
