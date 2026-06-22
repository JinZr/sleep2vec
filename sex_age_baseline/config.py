from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class AgeConfig:
    transform: str
    scale: float
    embedding_dim: int


@dataclass(frozen=True)
class SexConfig:
    encoding: str
    embedding_dim: int


@dataclass(frozen=True)
class HeadConfig:
    hidden_dim: int
    dropout: float
    activation: str


@dataclass(frozen=True)
class ModelConfig:
    name: str
    features: list[str]
    age: AgeConfig
    sex: SexConfig
    head: HeadConfig


@dataclass(frozen=True)
class DataConfig:
    backend: str
    finetune_data_index: str | None
    finetune_preset_path: str | None
    kaldi_data_root: str | None
    kaldi_manifest: str | None
    split_column: str
    key_column: str
    deduplicate_by_key: bool


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


@dataclass(frozen=True)
class MultilabelConfig:
    key_column: str
    disease_columns_index: str
    label_index: str
    has_label_index: str


@dataclass(frozen=True)
class FinetuneLossConfig:
    pos_weight: float | list[float] | None = None


@dataclass(frozen=True)
class FinetuneConfig:
    task: TaskConfig
    survival: SurvivalConfig | None = None
    multilabel: MultilabelConfig | None = None
    loss: FinetuneLossConfig | None = None


@dataclass(frozen=True)
class OutputsConfig:
    prediction_csv: bool
    per_disease_metrics_csv: bool


@dataclass(frozen=True)
class BaselineConfig:
    model: ModelConfig
    data: DataConfig
    finetune: FinetuneConfig
    outputs: OutputsConfig


def load_config(path: str | Path, *, validate_sidecars: bool = False) -> BaselineConfig:
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError("Sex/age baseline config must contain a YAML mapping.")
    cfg = _build_config(raw)
    if validate_sidecars:
        validate_sidecar_shapes(cfg)
    return cfg


def load_finetune_config(path: str | Path) -> BaselineConfig:
    return load_config(path)


def load_pretrain_config(path: str | Path):
    raise ValueError("sex_age_baseline does not support pretraining configs.")


def validate_model_config(model_cfg: ModelConfig | BaselineConfig) -> int:
    model = model_cfg.model if isinstance(model_cfg, BaselineConfig) else model_cfg
    return int(model.age.embedding_dim) + int(model.sex.embedding_dim)


def validate_sidecar_shapes(cfg: BaselineConfig) -> None:
    task = cfg.finetune.task
    if task.type == "survival":
        from data.survival import load_survival_label_table

        load_survival_label_table(cfg.finetune.survival, expected_output_dim=task.output_dim)
        return
    if task.type == "multilabel_classification":
        from data.multilabel import load_multilabel_label_table

        load_multilabel_label_table(cfg.finetune.multilabel, expected_output_dim=task.output_dim)


def _build_config(raw: dict[str, Any]) -> BaselineConfig:
    model = _build_model(_mapping(raw, "model"))
    data = _build_data(_mapping(raw, "data"))
    finetune = _build_finetune(_mapping(raw, "finetune"), data)
    outputs = _build_outputs(_mapping(raw, "outputs"))
    return BaselineConfig(model=model, data=data, finetune=finetune, outputs=outputs)


def _build_model(raw: dict[str, Any]) -> ModelConfig:
    name = _string(raw, "name")
    if name != "sex_age_mlp":
        raise ValueError("model.name must be 'sex_age_mlp'.")
    features = _list_of_strings(raw, "features")
    if len(set(features)) != len(features):
        raise ValueError("model.features contains duplicate entries.")
    if features != ["age", "sex"]:
        raise ValueError("sex_age_baseline v1 requires model.features: [age, sex].")
    age = _build_age(_mapping(raw, "age"))
    sex = _build_sex(_mapping(raw, "sex"))
    head = _build_head(_mapping(raw, "head"))
    return ModelConfig(name=name, features=features, age=age, sex=sex, head=head)


def _build_age(raw: dict[str, Any]) -> AgeConfig:
    transform = _string(raw, "transform")
    if transform != "divide":
        raise ValueError("model.age.transform must be 'divide'.")
    scale = _positive_float(raw, "scale")
    return AgeConfig(transform=transform, scale=scale, embedding_dim=_positive_int(raw, "embedding_dim"))


def _build_sex(raw: dict[str, Any]) -> SexConfig:
    encoding = _string(raw, "encoding")
    if encoding != "binary":
        raise ValueError("model.sex.encoding must be 'binary'.")
    return SexConfig(encoding=encoding, embedding_dim=_positive_int(raw, "embedding_dim"))


def _build_head(raw: dict[str, Any]) -> HeadConfig:
    activation = _string(raw, "activation")
    if activation not in {"elu", "gelu", "relu", "silu"}:
        raise ValueError("model.head.activation must be one of elu, gelu, relu, or silu.")
    dropout = _float(raw, "dropout")
    if dropout < 0.0 or dropout >= 1.0:
        raise ValueError("model.head.dropout must be in [0, 1).")
    return HeadConfig(hidden_dim=_positive_int(raw, "hidden_dim"), dropout=dropout, activation=activation)


def _build_data(raw: dict[str, Any]) -> DataConfig:
    backend = _string(raw, "backend")
    if backend not in {"npz", "kaldi"}:
        raise ValueError("data.backend must be 'npz' or 'kaldi'.")
    finetune_data_index = _optional_string(raw, "finetune_data_index")
    finetune_preset_path = _optional_string(raw, "finetune_preset_path")
    kaldi_data_root = _optional_string(raw, "kaldi_data_root")
    kaldi_manifest = _optional_string(raw, "kaldi_manifest")
    if backend == "npz":
        sources = [value for value in (finetune_data_index, finetune_preset_path) if value]
        if len(sources) != 1:
            raise ValueError("data.backend=npz requires exactly one of finetune_data_index or finetune_preset_path.")
        if kaldi_data_root or kaldi_manifest:
            raise ValueError("data.backend=npz must not set kaldi_data_root or kaldi_manifest.")
    if backend == "kaldi":
        if finetune_data_index or finetune_preset_path:
            raise ValueError("data.backend=kaldi must not set finetune_data_index or finetune_preset_path.")
        if not kaldi_data_root or not kaldi_manifest:
            raise ValueError("data.backend=kaldi requires kaldi_data_root and kaldi_manifest.")
    deduplicate_by_key = _bool(raw, "deduplicate_by_key")
    if not deduplicate_by_key:
        raise ValueError("sex_age_baseline v1 requires data.deduplicate_by_key=true.")
    return DataConfig(
        backend=backend,
        finetune_data_index=finetune_data_index,
        finetune_preset_path=finetune_preset_path,
        kaldi_data_root=kaldi_data_root,
        kaldi_manifest=kaldi_manifest,
        split_column=_string(raw, "split_column"),
        key_column=_string(raw, "key_column"),
        deduplicate_by_key=deduplicate_by_key,
    )


def _build_finetune(raw: dict[str, Any], data: DataConfig) -> FinetuneConfig:
    task = _build_task(_mapping(raw, "task"))
    if task.type == "survival":
        if "loss" in raw:
            raise ValueError("finetune.loss is only supported for multilabel_classification tasks.")
        survival = _build_survival(_mapping(raw, "survival"))
        if survival.key_column != data.key_column:
            raise ValueError("finetune.survival.key_column must match data.key_column.")
        return FinetuneConfig(task=task, survival=survival)
    if task.type == "multilabel_classification":
        multilabel = _build_multilabel(_mapping(raw, "multilabel"))
        if multilabel.key_column != data.key_column:
            raise ValueError("finetune.multilabel.key_column must match data.key_column.")
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
    return SurvivalConfig(
        key_column=_string(raw, "key_column"),
        disease_columns_index=_string(raw, "disease_columns_index"),
        event_time_index=_string(raw, "event_time_index"),
        is_event_index=_string(raw, "is_event_index"),
        has_label_index=_string(raw, "has_label_index"),
    )


def _build_multilabel(raw: dict[str, Any]) -> MultilabelConfig:
    return MultilabelConfig(
        key_column=_string(raw, "key_column"),
        disease_columns_index=_string(raw, "disease_columns_index"),
        label_index=_string(raw, "label_index"),
        has_label_index=_string(raw, "has_label_index"),
    )


def _build_loss(raw: dict[str, Any], output_dim: int) -> FinetuneLossConfig:
    extra = sorted(set(raw) - {"pos_weight"})
    if extra:
        raise ValueError(f"finetune.loss has unsupported fields: {extra}")
    pos_weight = raw.get("pos_weight")
    if pos_weight is None:
        return FinetuneLossConfig()
    if isinstance(pos_weight, (int, float)) and not isinstance(pos_weight, bool):
        if pos_weight <= 0:
            raise ValueError("finetune.loss.pos_weight must contain only positive numbers.")
        return FinetuneLossConfig(pos_weight=float(pos_weight))
    if isinstance(pos_weight, list):
        if len(pos_weight) != output_dim:
            raise ValueError(
                "finetune.loss.pos_weight length must match finetune.task.output_dim "
                f"({output_dim}); got {len(pos_weight)}."
            )
        if not all(isinstance(item, (int, float)) and not isinstance(item, bool) and item > 0 for item in pos_weight):
            raise ValueError("finetune.loss.pos_weight must contain only positive numbers.")
        return FinetuneLossConfig(pos_weight=[float(item) for item in pos_weight])
    raise ValueError("finetune.loss.pos_weight must be a positive number or list of positive numbers.")


def _build_outputs(raw: dict[str, Any]) -> OutputsConfig:
    return OutputsConfig(
        prediction_csv=_bool(raw, "prediction_csv"),
        per_disease_metrics_csv=_bool(raw, "per_disease_metrics_csv"),
    )


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


def _list_of_strings(raw: dict[str, Any], key: str) -> list[str]:
    value = raw.get(key)
    if not isinstance(value, list) or not value or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{key} must be a non-empty list of strings.")
    return list(value)


def _bool(raw: dict[str, Any], key: str) -> bool:
    value = raw.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be a boolean.")
    return value


def _float(raw: dict[str, Any], key: str) -> float:
    value = raw.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
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
