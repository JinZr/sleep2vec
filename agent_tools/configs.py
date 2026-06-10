from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .models import repo_relative, resolve_repo_path

BUILTIN_LABELS = ("stage3", "stage4", "stage5", "ahi", "sex", "age")


def load_yaml(path: str | Path) -> dict[str, Any]:
    resolved = resolve_repo_path(path)
    if resolved is None:
        raise FileNotFoundError("Config path is required.")
    data = yaml.safe_load(resolved.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"YAML must be a mapping: {resolved}")
    return data


def guess_variant(config_path: str | Path) -> str:
    parts = Path(config_path).parts
    if "sleep2expert" in parts:
        return "sleep2expert"
    if "sleep2vec2" in parts:
        return "sleep2vec2"
    return "sleep2vec"


def _channel_summary(item: dict[str, Any]) -> dict[str, Any]:
    tokenizer = item.get("tokenizer") if isinstance(item.get("tokenizer"), dict) else {}
    return {
        "name": item.get("name"),
        "input_dim": item.get("input_dim"),
        "tokenizer": tokenizer.get("name"),
        "out_dim": tokenizer.get("out_dim"),
    }


def config_summary(config_path: str | Path) -> dict[str, Any]:
    resolved = resolve_repo_path(config_path)
    if resolved is None:
        raise FileNotFoundError("Config path is required.")
    data = load_yaml(resolved)
    model = data.get("model") if isinstance(data.get("model"), dict) else {}
    data_block = data.get("data") if isinstance(data.get("data"), dict) else {}
    finetune = data.get("finetune") if isinstance(data.get("finetune"), dict) else {}
    task = finetune.get("task") if isinstance(finetune.get("task"), dict) else {}
    preset_build = data.get("preset_build") if isinstance(data.get("preset_build"), dict) else {}
    head = model.get("head") if isinstance(model.get("head"), dict) else {}
    temporal_agg = head.get("temporal_agg") if isinstance(head.get("temporal_agg"), dict) else {}
    channel_agg = head.get("channel_agg") if isinstance(head.get("channel_agg"), dict) else {}
    layer_mix = finetune.get("layer_mix") if isinstance(finetune.get("layer_mix"), dict) else {}
    lora = finetune.get("lora") if isinstance(finetune.get("lora"), dict) else {}
    averaging = data.get("model_averaging") if isinstance(data.get("model_averaging"), dict) else None
    channels_raw = model.get("channels") if isinstance(model.get("channels"), list) else []
    channels = [_channel_summary(item) for item in channels_raw if isinstance(item, dict)]
    model_channel_names = [item["name"] for item in channels if item.get("name")]
    data_channel_names = data_block.get("data_channel_names") or model_channel_names
    backend = data_block.get("backend") or "npz"
    warnings: list[str] = []
    blocking_issues: list[str] = []

    if data_channel_names and model_channel_names and list(data_channel_names) != model_channel_names:
        blocking_issues.append("data.data_channel_names differs from model.channels.")
    if backend == "kaldi" and not data_block.get("kaldi_manifest"):
        blocking_issues.append("data.backend=kaldi but data.kaldi_manifest is missing.")
    if (
        backend == "npz"
        and finetune
        and not data_block.get("finetune_data_index")
        and not data_block.get("finetune_preset_path")
    ):
        blocking_issues.append("data.backend=npz but both finetune_data_index and finetune_preset_path are missing.")
    if finetune and task == {}:
        warnings.append("finetune.task is missing; custom label semantics may be ambiguous.")
    if model_channel_names == ["ppg"] and finetune and "required_channels" not in preset_build:
        warnings.append("single-channel PPG finetune config has no preset_build.required_channels.")

    summary = {
        "config_path": repo_relative(resolved),
        "variant_guess": guess_variant(resolved),
        "is_finetune": bool(finetune),
        "is_pretrain": not bool(finetune),
        "data_backend": backend,
        "model": {
            "backbone": (model.get("backbone") or {}).get("name") if isinstance(model.get("backbone"), dict) else None,
            "hidden_size": (
                (model.get("backbone") or {}).get("hidden_size") if isinstance(model.get("backbone"), dict) else None
            ),
            "channels": channels,
            "cls": {
                "embedding_type": (
                    (model.get("cls") or {}).get("embedding_type") if isinstance(model.get("cls"), dict) else None
                ),
                "downstream": (
                    (model.get("cls") or {}).get("downstream") if isinstance(model.get("cls"), dict) else None
                ),
            },
            "head": {"name": (model.get("head") or {}).get("name") if isinstance(model.get("head"), dict) else None},
            "head_details": {
                "name": head.get("name"),
                "dropout": head.get("dropout"),
                "hidden_dim": head.get("hidden_dim"),
                "channel_agg": {
                    "name": channel_agg.get("name"),
                    "kwargs": channel_agg.get("kwargs") if isinstance(channel_agg.get("kwargs"), dict) else {},
                },
                "temporal_agg": {
                    "name": temporal_agg.get("name"),
                    "kwargs": temporal_agg.get("kwargs") if isinstance(temporal_agg.get("kwargs"), dict) else {},
                },
            },
            "layer_mix": {
                "enabled": layer_mix.get("enabled"),
                "shared_across_modalities": layer_mix.get("shared_across_modalities"),
                "layer_indices": layer_mix.get("layer_indices"),
            },
            "freeze": {
                "freeze_tokenizer": finetune.get("freeze_tokenizer"),
                "freeze_backbone_and_insert_lora": lora.get("freeze_backbone_and_insert_lora"),
                "insert_lora": lora.get("insert_lora"),
            },
            "model_averaging": {
                "present": averaging is not None,
                "name": averaging.get("name") if averaging else None,
                "enabled": (
                    (averaging.get("params") or {}).get("enabled")
                    if averaging and isinstance(averaging.get("params"), dict)
                    else None
                ),
            },
        },
        "data": {
            "max_tokens": data_block.get("max_tokens"),
            "data_channel_names": list(data_channel_names or []),
            "finetune_data_index": data_block.get("finetune_data_index"),
            "finetune_preset_path": data_block.get("finetune_preset_path"),
            "kaldi_data_root": data_block.get("kaldi_data_root"),
            "kaldi_manifest": data_block.get("kaldi_manifest"),
        },
        "finetune": {
            "task": {
                "type": task.get("type"),
                "output_dim": task.get("output_dim"),
                "is_seq": task.get("is_seq"),
                "monitor": task.get("monitor"),
                "monitor_mod": task.get("monitor_mod"),
            },
            "lora": lora,
            "loss": finetune.get("loss") if isinstance(finetune.get("loss"), dict) else {},
        },
        "preset_build": {
            "required_channels": preset_build.get("required_channels"),
            "min_channels": preset_build.get("min_channels"),
        },
        "plausible_labels": list(BUILTIN_LABELS),
        "warnings": warnings,
        "blocking_issues": blocking_issues,
    }
    return summary
