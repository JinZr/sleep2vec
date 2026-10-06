from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import yaml

from sleep2vec.config import COVARIATE_FIELDS, parse_covariate_fields


@dataclass(frozen=True)
class HeadConfig:
    name: str
    hidden_dim: int
    dropout: float
    act: str
    kwargs: dict[str, int]


@dataclass(frozen=True)
class ModelConfig:
    name: str
    head: HeadConfig


@dataclass(frozen=True)
class DataConfig:
    """The signal model's data inputs, with the same spellings and dataset-name substring semantics."""

    backend: str
    finetune_data_index: str | None
    finetune_preset_path: str | None
    kaldi_data_root: str | None
    kaldi_manifest: str | None
    train_dataset_names: list[str] | None
    test_dataset_names: list[str] | None


@dataclass(frozen=True)
class TaskConfig:
    type: str
    output_dim: int
    is_seq: bool
    monitor: str
    monitor_mod: str


@dataclass(frozen=True)
class SurvivalConfig:
    key_column: str
    disease_columns_index: str
    event_time_index: str
    is_event_index: str
    has_label_index: str
    covariates: list[str]
    covariate_embedding_dim: int
    covariate_normalization: dict[str, dict[str, float]]


@dataclass(frozen=True)
class MultilabelConfig:
    key_column: str
    disease_columns_index: str
    label_index: str
    has_label_index: str
    covariates: list[str]
    covariate_embedding_dim: int
    covariate_normalization: dict[str, dict[str, float]]


@dataclass(frozen=True)
class FinetuneLossConfig:
    pos_weight: float | list[float] | None = None
    eps: float = 1e-9


@dataclass(frozen=True)
class FinetuneConfig:
    task: TaskConfig
    survival: SurvivalConfig | None = None
    multilabel: MultilabelConfig | None = None
    loss: FinetuneLossConfig | None = None


@dataclass(frozen=True)
class BaselineConfig:
    model: ModelConfig
    data: DataConfig
    finetune: FinetuneConfig


def load_config(path: str | Path) -> BaselineConfig:
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError("Sex/age baseline config must contain a YAML mapping.")
    return _build_config(raw)


def load_finetune_config(path: str | Path) -> BaselineConfig:
    return load_config(path)


def load_pretrain_config(path: str | Path):
    raise ValueError("sex_age_baseline does not support pretraining configs.")


def validate_model_config(model_cfg: ModelConfig | BaselineConfig) -> int:
    """Loading already validates the model block; returns the head width for the shared variant contract."""
    model = model_cfg.model if isinstance(model_cfg, BaselineConfig) else model_cfg
    return model.head.hidden_dim


def covariate_task_config(cfg: BaselineConfig) -> SurvivalConfig | MultilabelConfig:
    """The task block that owns the label sidecars, key column and covariates."""
    task_cfg = cfg.finetune.survival if cfg.finetune.task.type == "survival" else cfg.finetune.multilabel
    assert task_cfg is not None
    return task_cfg


def _build_config(raw: dict[str, Any]) -> BaselineConfig:
    extra = sorted(set(raw) - {"model", "data", "finetune"})
    if extra:
        # Artifact switches are CLI flags (e.g. --export-predictions) as for sleep2vec, not YAML blocks.
        raise ValueError(f"config contains unsupported top-level fields: {extra}")
    model = _build_model(_mapping(raw, "model"))
    data = _build_data(_mapping(raw, "data"))
    finetune = _build_finetune(_mapping(raw, "finetune"))
    return BaselineConfig(model=model, data=data, finetune=finetune)


def _build_model(raw: dict[str, Any]) -> ModelConfig:
    extra = sorted(set(raw) - {"name", "head"})
    if extra:
        raise ValueError(
            f"model contains unsupported fields: {extra}; covariates belong in finetune.survival or "
            "finetune.multilabel."
        )
    name = _string(raw, "name")
    if name != "sex_age_mlp":
        raise ValueError("model.name must be 'sex_age_mlp'.")
    return ModelConfig(name=name, head=_build_head(_mapping(raw, "head")))


def _build_head(raw: dict[str, Any]) -> HeadConfig:
    if set(raw) - {"name", "hidden_dim", "dropout", "act", "kwargs"}:
        raise ValueError("model.head contains unsupported fields.")
    name = _string(raw, "name")
    if name != "classification":
        raise ValueError("model.head.name must be classification (raw logits/log-risk).")
    activation = _string(raw, "act")
    if activation not in {"elu", "gelu", "silu"}:
        # The head applies its activation to the zero-initialized covariate embeddings first; relu would block them.
        raise ValueError("model.head.act must be one of elu, gelu, or silu.")
    dropout = _float(raw, "dropout")
    if dropout < 0.0 or dropout >= 1.0:
        raise ValueError("model.head.dropout must be in [0, 1).")
    kwargs = _mapping(raw, "kwargs")
    if set(kwargs) != {"num_layers"} or _positive_int(kwargs, "num_layers") not in {1, 2, 3}:
        raise ValueError("model.head.kwargs must specify num_layers as 1, 2, or 3.")
    return HeadConfig(
        name=name, hidden_dim=_positive_int(raw, "hidden_dim"), dropout=dropout, act=activation, kwargs=dict(kwargs)
    )


def _build_data(raw: dict[str, Any]) -> DataConfig:
    extra = sorted(
        set(raw)
        - {
            "backend",
            "finetune_data_index",
            "finetune_preset_path",
            "kaldi_data_root",
            "kaldi_manifest",
            "train_dataset_names",
            "test_dataset_names",
        }
    )
    if extra:
        raise ValueError(
            f"data contains unsupported fields: {extra}; the baseline reads the signal model's split column and "
            "always collapses rows by the task key column."
        )
    backend = _string(raw, "backend")
    if backend not in {"npz", "kaldi"}:
        raise ValueError("data.backend must be 'npz' or 'kaldi'.")
    # Like the signal configs, the data source may stay null here; the loader requires one after CLI overrides.
    return DataConfig(
        backend=backend,
        finetune_data_index=_optional_string(raw, "finetune_data_index"),
        finetune_preset_path=_optional_string(raw, "finetune_preset_path"),
        kaldi_data_root=_optional_string(raw, "kaldi_data_root"),
        kaldi_manifest=_optional_string(raw, "kaldi_manifest"),
        train_dataset_names=_optional_list_of_strings(raw, "train_dataset_names"),
        test_dataset_names=_optional_list_of_strings(raw, "test_dataset_names"),
    )


def _build_finetune(raw: dict[str, Any]) -> FinetuneConfig:
    extra = sorted(set(raw) - {"task", "survival", "multilabel", "loss"})
    if extra:
        raise ValueError(f"finetune contains unsupported fields: {extra}; training options belong in recipe runtime.")
    task = _build_task(_mapping(raw, "task"))
    if task.type == "survival":
        if "multilabel" in raw:
            raise ValueError("finetune.multilabel is only supported for multilabel_classification tasks.")
        survival = _build_survival(_mapping(raw, "survival"))
        loss = FinetuneLossConfig()
        if "loss" in raw:
            loss_raw = _mapping(raw, "loss")
            if set(loss_raw) - {"eps"}:
                raise ValueError("Survival finetune.loss supports only eps.")
            if "eps" in loss_raw:
                loss = FinetuneLossConfig(eps=_positive_float(loss_raw, "eps"))
        return FinetuneConfig(task=task, survival=survival, loss=loss)
    if task.type == "multilabel_classification":
        if "survival" in raw:
            raise ValueError("finetune.survival is only supported for survival tasks.")
        multilabel = _build_multilabel(_mapping(raw, "multilabel"))
        loss = _build_loss(_mapping(raw, "loss"), task.output_dim) if "loss" in raw else FinetuneLossConfig()
        return FinetuneConfig(task=task, multilabel=multilabel, loss=loss)
    raise ValueError(f"Unsupported sex_age_baseline task type: {task.type}")


def _build_task(raw: dict[str, Any]) -> TaskConfig:
    task_type = _string(raw, "type")
    if task_type not in {"survival", "multilabel_classification"}:
        raise ValueError(f"Unsupported sex_age_baseline task type: {task_type}")
    is_seq = _bool(raw, "is_seq")
    if is_seq:
        raise ValueError("sex_age_baseline only supports non-sequence downstream tasks.")
    monitor_mod = _string(raw, "monitor_mod")
    if monitor_mod not in {"min", "max"}:
        raise ValueError("finetune.task.monitor_mod must be 'min' or 'max'.")
    return TaskConfig(
        type=task_type,
        output_dim=_positive_int(raw, "output_dim"),
        is_seq=is_seq,
        monitor=_string(raw, "monitor"),
        monitor_mod=monitor_mod,
    )


def _build_survival(raw: dict[str, Any]) -> SurvivalConfig:
    sidecars = ("key_column", "disease_columns_index", "event_time_index", "is_event_index")
    fields: dict[str, Any] = {
        **_sidecar_fields(raw, "finetune.survival", sidecars),
        **_covariate_fields(raw, "finetune.survival"),
    }
    return SurvivalConfig(**fields)


def _build_multilabel(raw: dict[str, Any]) -> MultilabelConfig:
    sidecars = ("key_column", "disease_columns_index", "label_index")
    fields: dict[str, Any] = {
        **_sidecar_fields(raw, "finetune.multilabel", sidecars),
        **_covariate_fields(raw, "finetune.multilabel"),
    }
    return MultilabelConfig(**fields)


def _sidecar_fields(raw: dict[str, Any], prefix: str, names: tuple[str, ...]) -> dict[str, str]:
    required = {*names, "has_label_index"}
    extra = sorted(set(raw) - required - COVARIATE_FIELDS)
    if extra:
        raise ValueError(f"{prefix} contains unsupported fields: {extra}")
    return {name: _string(raw, name) for name in sorted(required)}


def _covariate_fields(raw: dict[str, Any], prefix: str) -> dict[str, Any]:
    fields = parse_covariate_fields(raw, prefix)
    if not fields["covariates"]:
        raise ValueError(f"sex_age_baseline requires a non-empty {prefix}.covariates list.")
    return fields


def _build_loss(raw: dict[str, Any], output_dim: int) -> FinetuneLossConfig:
    extra = sorted(set(raw) - {"pos_weight"})
    if extra:
        raise ValueError(f"finetune.loss has unsupported fields: {extra}")
    pos_weight = raw.get("pos_weight")
    if pos_weight is None:
        return FinetuneLossConfig()
    if isinstance(pos_weight, (int, float)) and not isinstance(pos_weight, bool):
        if not math.isfinite(pos_weight) or pos_weight <= 0:
            raise ValueError("finetune.loss.pos_weight must contain only positive numbers.")
        return FinetuneLossConfig(pos_weight=float(pos_weight))
    if isinstance(pos_weight, list):
        if len(pos_weight) != output_dim:
            raise ValueError(
                "finetune.loss.pos_weight length must match finetune.task.output_dim "
                f"({output_dim}); got {len(pos_weight)}."
            )
        if not all(
            isinstance(item, (int, float)) and not isinstance(item, bool) and math.isfinite(item) and item > 0
            for item in pos_weight
        ):
            raise ValueError("finetune.loss.pos_weight must contain only positive numbers.")
        return FinetuneLossConfig(pos_weight=[float(item) for item in pos_weight])
    raise ValueError("finetune.loss.pos_weight must be a positive number or list of positive numbers.")


def _mapping(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping.")
    return value


def _string(raw: dict[str, Any], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string.")
    return value


def _optional_string(raw: dict[str, Any], key: str) -> str | None:
    value = raw.get(key)
    if value in (None, ""):
        return None
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string or null.")
    return value


def _optional_list_of_strings(raw: dict[str, Any], key: str) -> list[str] | None:
    value = raw.get(key)
    if value is None:
        return None
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise ValueError(f"{key} must be a list of non-empty strings or null.")
    return list(value)


def _bool(raw: dict[str, Any], key: str) -> bool:
    value = raw.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be a boolean.")
    return value


def _float(raw: dict[str, Any], key: str) -> float:
    value = raw.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(f"{key} must be a number.")
    return float(value)


def _positive_float(raw: dict[str, Any], key: str) -> float:
    value = _float(raw, key)
    if value <= 0:
        raise ValueError(f"{key} must be positive.")
    return value


def _positive_int(raw: dict[str, Any], key: str) -> int:
    value = raw.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{key} must be a positive integer.")
    return int(value)
