from __future__ import annotations

import argparse
from pathlib import Path
import typing as t

import yaml

from sleep2vec.config import TaskConfig, load_finetune_config


def apply_task_flags(args, task_cfg: TaskConfig | None = None) -> None:
    """Infer downstream task attributes from label_name or finetune.task."""
    if task_cfg is not None:
        args.output_dim = task_cfg.output_dim
        args.is_classification = task_cfg.type == "classification"
        args.is_seq = task_cfg.is_seq
        args.monitor = task_cfg.monitor
        args.monitor_mod = task_cfg.monitor_mod
        if args.label_name == "stage5" and not args.is_seq:
            raise ValueError("finetune.task.is_seq must be true when --label-name is 'stage5'.")
        if args.is_seq and args.label_name != "stage5":
            raise ValueError(
                "finetune.task.is_seq is only supported for --label-name stage5. "
                "Extend the dataloader if you need token-level labels for other targets."
            )
        return

    if args.label_name == "stage5":
        args.output_dim = 5
        args.is_classification = True
        args.is_seq = True
        args.monitor = "val_accuracy"
        args.monitor_mod = "max"
        return
    if args.label_name == "sex":
        args.output_dim = 2
        args.is_classification = True
        args.is_seq = False
        args.monitor = "val_accuracy"
        args.monitor_mod = "max"
        return
    if args.label_name == "age":
        args.output_dim = 1
        args.is_classification = False
        args.is_seq = False
        args.monitor = "val_mae"
        args.monitor_mod = "min"
        return

    raise ValueError(
        f"Unknown label_name '{args.label_name}'. "
        "Define finetune.task in the YAML to specify task semantics for custom labels."
    )


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

    args.channel_names = [c.name for c in model_cfg.channels]
    args.data_channel_names = data_cfg.data_channel_names or args.channel_names
    args.max_tokens = data_cfg.max_tokens
    args.finetune_data_index = Path(data_cfg.finetune_data_index) if data_cfg.finetune_data_index else None
    args.finetune_preset_path = Path(data_cfg.finetune_preset_path) if data_cfg.finetune_preset_path else None
    args.train_dataset_names = data_cfg.train_dataset_names or []
    args.test_dataset_names = data_cfg.test_dataset_names or []
    args.n_few_shot = data_cfg.n_few_shot

    args.freeze_backbone_and_insert_lora = lora_cfg.freeze_backbone_and_insert_lora
    args.insert_lora = lora_cfg.insert_lora
    args.separate_adapters = lora_cfg.separate_adapters
    args.freeze_tokenizer = finetune_cfg.freeze_tokenizer
    args.head_kwargs = {}

    # Fail fast if requested dataloader channels differ from model/backbone channels.
    if set(args.data_channel_names) != set(args.channel_names):
        raise ValueError(
            "data.data_channel_names in YAML must match model.channels. "
            f"Model channels: {args.channel_names}; data channels: {args.data_channel_names}."
        )

    apply_task_flags(args, config_bundle.finetune.task)
    return config_bundle, model_cfg


def _to_yamlable(obj: t.Any) -> t.Any:
    """Convert argparse / pathlib objects into YAML-safe primitives."""

    if isinstance(obj, argparse.Namespace):
        obj = vars(obj)

    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, dict):
        return {k: _to_yamlable(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [_to_yamlable(v) for v in obj]

    return obj


def dump_cli_args_yaml(args: argparse.Namespace, dest_path: Path) -> Path:
    """Persist CLI/derived args to ``dest_path`` as YAML for experiment tracking."""

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_safe_obj = _to_yamlable(args)
    dest_path.write_text(yaml.safe_dump(yaml_safe_obj, sort_keys=True))
    return dest_path
