from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import logging
from pathlib import Path
import shutil
import typing as t

import yaml

from sleep2vec.config import DATA_BACKEND_CHOICES, FinetuneConfig, ModelConfig, TaskConfig, load_finetune_config

_BUILTIN_TASK_SPECS = {
    "stage3": {
        "type": "classification",
        "output_dim": 3,
        "is_seq": True,
        "monitor": "val_accuracy",
        "monitor_mod": "max",
        "label_source_name": "stage5",
        "stage_names": ["W", "NREM", "REM"],
        "label_merge_map": {0: 0, 1: 1, 2: 1, 3: 1, 4: 2},
    },
    "stage4": {
        "type": "classification",
        "output_dim": 4,
        "is_seq": True,
        "monitor": "val_accuracy",
        "monitor_mod": "max",
        "label_source_name": "stage5",
        "stage_names": ["W", "N1N2", "N3", "REM"],
        "label_merge_map": {0: 0, 1: 1, 2: 1, 3: 2, 4: 3},
    },
    "stage5": {
        "type": "classification",
        "output_dim": 5,
        "is_seq": True,
        "monitor": "val_accuracy",
        "monitor_mod": "max",
        "label_source_name": "stage5",
        "stage_names": ["W", "N1", "N2", "N3", "REM"],
    },
    "ahi": {
        "type": "classification",
        "output_dim": 30,
        "is_seq": True,
        "is_multilabel": True,
        "monitor": "val_ahi_pearson",
        "monitor_mod": "max",
        "label_source_name": "ahi",
        "auxiliary_label_source_names": ["stage5"],
    },
    "sex": {
        "type": "classification",
        "output_dim": 2,
        "is_seq": False,
        "monitor": "val_accuracy",
        "monitor_mod": "max",
        "class_labels": ["female", "male"],
    },
    "age": {
        "type": "regression",
        "output_dim": 1,
        "is_seq": False,
        "monitor": "val_mae",
        "monitor_mod": "min",
    },
}


def is_builtin_seq_task(label_name: str | None) -> bool:
    if label_name is None:
        return False
    spec = _BUILTIN_TASK_SPECS.get(label_name)
    return bool(spec is not None and spec["is_seq"])


def is_builtin_stage_task(label_name: str | None) -> bool:
    if not is_builtin_seq_task(label_name):
        return False
    spec = _BUILTIN_TASK_SPECS.get(label_name)
    return bool(spec is not None and spec.get("label_source_name") == "stage5")


def get_task_label_source_name(label_name: str) -> str:
    spec = _BUILTIN_TASK_SPECS.get(label_name)
    if spec is None:
        return label_name
    return str(spec.get("label_source_name", label_name))


def get_task_stage_names(label_name: str) -> list[str] | None:
    spec = _BUILTIN_TASK_SPECS.get(label_name)
    if spec is None or "stage_names" not in spec:
        return None
    return list(spec["stage_names"])


def get_task_class_labels(label_name: str) -> list[str] | None:
    spec = _BUILTIN_TASK_SPECS.get(label_name)
    if spec is None:
        return None
    if "class_labels" in spec:
        return list(spec["class_labels"])
    if "stage_names" in spec:
        return list(spec["stage_names"])
    return None


def get_task_label_merge_map(label_name: str) -> dict[int, int] | None:
    spec = _BUILTIN_TASK_SPECS.get(label_name)
    if spec is None or "label_merge_map" not in spec:
        return None
    return {int(k): int(v) for k, v in spec["label_merge_map"].items()}


def get_task_is_multilabel(label_name: str) -> bool:
    spec = _BUILTIN_TASK_SPECS.get(label_name)
    return bool(spec is not None and spec.get("is_multilabel", False))


def get_task_auxiliary_label_source_names(label_name: str) -> list[str]:
    spec = _BUILTIN_TASK_SPECS.get(label_name)
    if spec is None or "auxiliary_label_source_names" not in spec:
        return []
    return [str(name) for name in spec["auxiliary_label_source_names"]]


def remap_stage_labels(labels, label_name: str):
    label_merge_map = get_task_label_merge_map(label_name)
    if label_merge_map is None:
        return labels

    remapped = labels.clone()
    for raw_label, merged_label in label_merge_map.items():
        remapped[labels == raw_label] = merged_label
    return remapped


def channel_input_dims_from_model_config(model_cfg: ModelConfig) -> dict[str, int]:
    return {channel.name: int(channel.input_dim) for channel in model_cfg.channels}


def apply_model_config_args(args, model_cfg: ModelConfig, *, set_backbone_arch: bool = False) -> None:
    args.channel_names = [c.name for c in model_cfg.channels]
    args.channel_input_dims = channel_input_dims_from_model_config(model_cfg)
    if set_backbone_arch:
        args.backbone_arch = model_cfg.backbone.name


def _copy_file(src: Path, dest: Path, *, label: str) -> None:
    try:
        shutil.copy2(src, dest)
        logging.info(f"Copied {label} to {dest}")
    except Exception as exc:  # pragma: no cover - best-effort
        logging.warning(f"Failed to copy {label} to {dest}: {exc}")


def _write_cli_args(args: argparse.Namespace, dest: Path) -> None:
    try:
        dump_cli_args_yaml(args, dest)
        logging.info(f"Saved CLI args to {dest}")
    except Exception as exc:  # pragma: no cover - best-effort
        logging.warning(f"Failed to write CLI args YAML to {dest}: {exc}")


def persist_run_config_and_args(
    args: argparse.Namespace,
    exp_dir: Path,
    *,
    phase_name: str | None = None,
    write_root_files: bool = True,
) -> None:
    exp_dir.mkdir(parents=True, exist_ok=True)
    config_src = Path(args.config)

    if write_root_files:
        _copy_file(config_src, exp_dir / "config.yaml", label="config")
        _write_cli_args(args, exp_dir / "cli_args.yaml")

    if phase_name:
        suffix = f".{phase_name}"
        _copy_file(config_src, exp_dir / f"config{suffix}.yaml", label=f"{phase_name} config")
        _write_cli_args(args, exp_dir / f"cli_args{suffix}.yaml")


def _validate_metadata_label_support(args) -> None:
    """Fail fast for unsupported metadata task semantics."""
    if (
        getattr(args, "is_classification", False)
        and int(getattr(args, "output_dim", 0)) > 2
        and not getattr(args, "is_multilabel", False)
        and not is_builtin_seq_task(getattr(args, "label_name", None))
    ):
        raise ValueError(
            "Metadata classification currently supports only binary labels (output_dim=2) "
            "for non-built-in sequence tasks. "
            f"Got --label-name '{args.label_name}' with finetune.task.output_dim={args.output_dim}. "
            "Extend metadata label encoding before using multiclass metadata targets."
        )


def _validate_builtin_task_cfg(label_name: str, task_cfg: TaskConfig, spec: dict[str, t.Any]) -> None:
    if task_cfg.output_dim != spec["output_dim"]:
        raise ValueError(f"finetune.task.output_dim must be {spec['output_dim']} when --label-name is '{label_name}'.")
    if task_cfg.type != spec["type"]:
        raise ValueError(f"finetune.task.type must be '{spec['type']}' when --label-name is '{label_name}'.")
    if task_cfg.is_seq != spec["is_seq"]:
        raise ValueError(f"finetune.task.is_seq must be {spec['is_seq']} when --label-name is '{label_name}'.")
    if label_name == "ahi":
        allowed_ahi_monitors = {
            "val_ahi_pearson": "max",
        }
        expected_monitor_mod = allowed_ahi_monitors.get(task_cfg.monitor)
        if expected_monitor_mod is None:
            raise ValueError(
                "finetune.task.monitor must be one of "
                f"{sorted(allowed_ahi_monitors)} when --label-name is '{label_name}'."
            )
        if task_cfg.monitor_mod != expected_monitor_mod:
            raise ValueError(
                f"finetune.task.monitor_mod must be '{expected_monitor_mod}' when "
                f"finetune.task.monitor is '{task_cfg.monitor}' and --label-name is '{label_name}'."
            )
        return
    if task_cfg.type == "classification":
        allowed_classification_monitors = {
            "val_accuracy": "max",
            "val_f1_macro": "max",
            "val_f1_weighted": "max",
            "val_cohen_kappa": "max",
        }
        if task_cfg.output_dim == 2:
            allowed_classification_monitors["val_roc_auc"] = "max"
        expected_monitor_mod = allowed_classification_monitors.get(task_cfg.monitor)
        if expected_monitor_mod is None:
            raise ValueError(
                "finetune.task.monitor must be one of "
                f"{sorted(allowed_classification_monitors)} when --label-name is '{label_name}'."
            )
        if task_cfg.monitor_mod != expected_monitor_mod:
            raise ValueError(
                f"finetune.task.monitor_mod must be '{expected_monitor_mod}' when "
                f"finetune.task.monitor is '{task_cfg.monitor}' and --label-name is '{label_name}'."
            )
        return
    if task_cfg.monitor != spec["monitor"]:
        raise ValueError(f"finetune.task.monitor must be '{spec['monitor']}' when --label-name is '{label_name}'.")
    if task_cfg.monitor_mod != spec["monitor_mod"]:
        raise ValueError(
            f"finetune.task.monitor_mod must be '{spec['monitor_mod']}' when --label-name is '{label_name}'."
        )


def _apply_builtin_task_attrs(args: argparse.Namespace, label_name: str) -> None:
    args.label_source_name = get_task_label_source_name(label_name)
    args.stage_names = get_task_stage_names(label_name)
    args.class_labels = get_task_class_labels(label_name)
    args.label_merge_map = get_task_label_merge_map(label_name)
    args.is_multilabel = get_task_is_multilabel(label_name)
    args.is_survival = False
    args.auxiliary_label_source_names = get_task_auxiliary_label_source_names(label_name)


def _apply_custom_task_attrs(args: argparse.Namespace) -> None:
    args.label_source_name = args.label_name
    args.stage_names = None
    args.class_labels = None
    args.label_merge_map = None
    args.is_multilabel = False
    args.is_survival = False
    args.auxiliary_label_source_names = []


def apply_task_flags(args, task_cfg: TaskConfig | None = None) -> None:
    """Infer downstream task attributes from label_name or finetune.task."""
    builtin_spec = _BUILTIN_TASK_SPECS.get(args.label_name)
    if task_cfg is not None:
        if builtin_spec is not None:
            _validate_builtin_task_cfg(args.label_name, task_cfg, builtin_spec)
            _apply_builtin_task_attrs(args, args.label_name)
        else:
            _apply_custom_task_attrs(args)
        args.output_dim = task_cfg.output_dim
        args.is_classification = task_cfg.type in {"classification", "multilabel_classification"}
        args.is_survival = task_cfg.type == "survival"
        args.is_multilabel = task_cfg.type == "multilabel_classification" or getattr(args, "is_multilabel", False)
        args.is_seq = task_cfg.is_seq
        args.monitor = task_cfg.monitor
        args.monitor_mod = task_cfg.monitor_mod
        if is_builtin_seq_task(args.label_name) and not args.is_seq:
            raise ValueError(
                "finetune.task.is_seq must be true when --label-name is one of: stage3, stage4, stage5, ahi."
            )
        if args.is_seq and not is_builtin_seq_task(args.label_name):
            raise ValueError(
                "finetune.task.is_seq is only supported for built-in sequence labels (stage3, stage4, stage5, ahi). "
                "Extend the dataloader if you need token-level labels for other targets."
            )
        _validate_metadata_label_support(args)
        return

    if builtin_spec is not None:
        _apply_builtin_task_attrs(args, args.label_name)
        args.output_dim = builtin_spec["output_dim"]
        args.is_classification = builtin_spec["type"] == "classification"
        args.is_survival = False
        args.is_seq = builtin_spec["is_seq"]
        args.monitor = builtin_spec["monitor"]
        args.monitor_mod = builtin_spec["monitor_mod"]
        _validate_metadata_label_support(args)
        return

    _apply_custom_task_attrs(args)
    raise ValueError(
        f"Unknown label_name '{args.label_name}'. "
        "Define finetune.task in the YAML to specify task semantics for custom labels."
    )


def _validate_and_apply_imbalance_config(args: argparse.Namespace, finetune_cfg: FinetuneConfig) -> None:
    loss_cfg = finetune_cfg.loss
    sampler_cfg = finetune_cfg.sampler

    class_weights = loss_cfg.class_weights
    if class_weights is not None:
        if not getattr(args, "is_classification", False) or getattr(args, "is_multilabel", False):
            raise ValueError("finetune.loss.class_weights is only supported for single-label classification tasks.")
        if len(class_weights) != int(args.output_dim):
            raise ValueError(
                "finetune.loss.class_weights length must match finetune.task.output_dim "
                f"({args.output_dim}); got {len(class_weights)}."
            )

    pos_weight = loss_cfg.pos_weight
    if pos_weight is not None:
        if not getattr(args, "is_classification", False) or not getattr(args, "is_multilabel", False):
            raise ValueError("finetune.loss.pos_weight is only supported for multilabel classification tasks.")
        if isinstance(pos_weight, list):
            if len(pos_weight) != int(args.output_dim):
                raise ValueError(
                    "finetune.loss.pos_weight length must match finetune.task.output_dim "
                    f"({args.output_dim}); got {len(pos_weight)}."
                )
        else:
            pos_weight = [float(pos_weight)] * int(args.output_dim)

    if sampler_cfg.weighted_random and (
        not getattr(args, "is_classification", False) or getattr(args, "is_seq", False) or int(args.output_dim) != 2
    ):
        raise ValueError(
            "finetune.sampler.weighted_random is only supported for binary non-sequence classification tasks."
        )

    args.class_weights = class_weights
    args.pos_weight = pos_weight
    args.weighted_random_sampler = bool(sampler_cfg.weighted_random)


def apply_finetune_config(args) -> tuple[t.Any, t.Any]:
    """
    Load finetune YAML and populate argparse Namespace in-place.

    Returns:
        (config_bundle, model_cfg) for convenience in callers.
    """
    config_bundle = load_finetune_config(args.config)
    model_cfg = config_bundle.model
    data_cfg = config_bundle.data
    finetune_cfg = config_bundle.finetune
    lora_cfg = finetune_cfg.lora

    apply_model_config_args(args, model_cfg)
    args.data_channel_names = data_cfg.data_channel_names or args.channel_names
    args.max_tokens = data_cfg.max_tokens
    args.finetune_data_index = Path(data_cfg.finetune_data_index) if data_cfg.finetune_data_index else None
    args.finetune_preset_path = Path(data_cfg.finetune_preset_path) if data_cfg.finetune_preset_path else None
    args.train_dataset_names = data_cfg.train_dataset_names or []
    args.test_dataset_names = data_cfg.test_dataset_names or []
    args.n_few_shot = data_cfg.n_few_shot
    apply_data_backend_args(args, data_cfg, preset_attr="finetune_preset_path")

    args.freeze_backbone_and_insert_lora = lora_cfg.freeze_backbone_and_insert_lora
    args.insert_lora = lora_cfg.insert_lora
    args.separate_adapters = lora_cfg.separate_adapters
    args.lora_r = lora_cfg.r
    args.lora_alpha = lora_cfg.alpha
    args.lora_dropout = lora_cfg.dropout
    args.lora_target_modules = lora_cfg.target_modules
    args.lora_use_dora = lora_cfg.use_dora
    args.freeze_tokenizer = finetune_cfg.freeze_tokenizer
    args.eval_visualizations = finetune_cfg.eval_visualizations
    args.survival = finetune_cfg.survival
    args.multilabel = finetune_cfg.multilabel
    args.head_kwargs = {}

    # Fail fast if requested dataloader channels differ from model/backbone channels.
    if set(args.data_channel_names) != set(args.channel_names):
        raise ValueError(
            "data.data_channel_names in YAML must match model.channels. "
            f"Model channels: {args.channel_names}; data channels: {args.data_channel_names}."
        )

    apply_task_flags(args, config_bundle.finetune.task)
    _validate_and_apply_imbalance_config(args, finetune_cfg)
    return config_bundle, model_cfg


def _optional_path(value: t.Any) -> Path | None:
    if value is None:
        return None
    if isinstance(value, Path):
        return value
    return Path(value)


def apply_data_backend_args(args, data_cfg, *, preset_attr: str | None = None) -> None:
    backend = getattr(args, "data_backend", None) or getattr(data_cfg, "backend", "npz") or "npz"
    if backend not in DATA_BACKEND_CHOICES:
        raise ValueError(f"Unknown data backend: {backend!r}. Expected one of {DATA_BACKEND_CHOICES}.")

    kaldi_data_root = getattr(args, "kaldi_data_root", None)
    if kaldi_data_root is None:
        kaldi_data_root = getattr(data_cfg, "kaldi_data_root", None)
    kaldi_manifest = getattr(args, "kaldi_manifest", None)
    if kaldi_manifest is None:
        kaldi_manifest = getattr(data_cfg, "kaldi_manifest", None)

    args.data_backend = backend
    args.kaldi_data_root = _optional_path(kaldi_data_root)
    args.kaldi_manifest = _optional_path(kaldi_manifest)

    if backend != "kaldi":
        return

    missing = []
    if args.kaldi_data_root is None:
        missing.append("kaldi_data_root")
    if args.kaldi_manifest is None:
        missing.append("kaldi_manifest")
    if missing:
        raise ValueError(
            f"Kaldi backend requires explicit kaldi_data_root and kaldi_manifest; missing {', '.join(missing)}."
        )

    if preset_attr and getattr(args, preset_attr, None):
        raise ValueError("Kaldi backend uses manifest.json; legacy NPZ preset pickles are unsupported.")


def _to_yamlable(obj: t.Any) -> t.Any:
    """Convert argparse / pathlib objects into YAML-safe primitives."""

    if is_dataclass(obj):
        return _to_yamlable(asdict(obj))

    if isinstance(obj, argparse.Namespace):
        obj = vars(obj)

    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, dict):
        converted: dict[t.Any, t.Any] = {}
        for key, value in obj.items():
            yaml_key = _to_yamlable(key)
            if isinstance(yaml_key, list):
                yaml_key = str(yaml_key)
            elif not isinstance(yaml_key, (str, int, float, bool)) and yaml_key is not None:
                yaml_key = str(yaml_key)
            converted[yaml_key] = _to_yamlable(value)
        return converted

    if isinstance(obj, (list, tuple)):
        return [_to_yamlable(v) for v in obj]

    return obj


def dump_cli_args_yaml(args: argparse.Namespace, dest_path: Path) -> Path:
    """Persist CLI/derived args to ``dest_path`` as YAML for experiment tracking."""

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_safe_obj = _to_yamlable(args)
    dest_path.write_text(yaml.safe_dump(yaml_safe_obj, sort_keys=True))
    return dest_path
