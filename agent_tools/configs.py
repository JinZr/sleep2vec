from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
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


def _looks_like_sleep2stat_config_data(data: dict[str, Any]) -> bool:
    return {"run", "data", "signals", "analyzers", "reducers", "outputs"}.issubset(set(data))


def _looks_like_placeholder_path(value: str | Path | None) -> bool:
    if value in (None, ""):
        return True
    text = str(value).strip()
    lowered = text.lower()
    return (
        lowered in {"ask_user", "none", "null", "todo", "tbd", "placeholder"}
        or text.startswith("/path/to")
        or text.startswith("<")
        or "ASK_USER" in text
    )


def _survival_summary(
    finetune: dict[str, Any],
    task: dict[str, Any],
    *,
    validate_local_paths: bool = True,
) -> dict[str, Any] | None:
    if task.get("type") != "survival":
        return None

    raw = finetune.get("survival") if isinstance(finetune.get("survival"), dict) else {}
    path_fields = ("disease_columns_index", "event_time_index", "is_event_index", "has_label_index")
    summary: dict[str, Any] = {
        "key_column": raw.get("key_column"),
        "disease_columns_index": raw.get("disease_columns_index"),
        "event_time_index": raw.get("event_time_index"),
        "is_event_index": raw.get("is_event_index"),
        "has_label_index": raw.get("has_label_index"),
        "output_dim": task.get("output_dim"),
        "valid": False,
        "disease_count": None,
        "sidecar_key_count": None,
        "issues": [],
    }

    issues = summary["issues"]
    if not isinstance(finetune.get("survival"), dict):
        issues.append("finetune.survival must be a mapping for survival tasks.")
        return summary
    if not isinstance(raw.get("key_column"), str) or not raw.get("key_column"):
        issues.append("finetune.survival.key_column must be a non-empty string.")
    resolved_paths: dict[str, str] = {}
    for field in path_fields:
        value = raw.get(field)
        if not isinstance(value, str) or _looks_like_placeholder_path(value):
            issues.append(f"finetune.survival.{field} must point to a real file.")
            continue
        if not validate_local_paths:
            continue
        resolved = resolve_repo_path(value)
        if resolved is None or not resolved.exists():
            issues.append(f"finetune.survival.{field} does not exist: {value}")
        else:
            resolved_paths[field] = str(resolved)

    if issues or not validate_local_paths:
        return summary

    try:
        from data.survival import load_survival_label_table

        labels = load_survival_label_table(
            SimpleNamespace(key_column=raw["key_column"], **resolved_paths),
            task.get("output_dim"),
        )
    except Exception as exc:
        issues.append(str(exc))
        return summary

    if labels is not None:
        summary["valid"] = True
        summary["disease_count"] = len(labels.label_names)
        summary["sidecar_key_count"] = len(labels.event_time)
    return summary


def sleep2stat_config_summary(config_path: str | Path) -> dict[str, Any]:
    from sleep2stat.config import SUPPORTED_ANALYZER_TYPES, SUPPORTED_REDUCER_TYPES, load_config

    resolved = resolve_repo_path(config_path)
    if resolved is None:
        raise FileNotFoundError("Config path is required.")
    supported = {
        "supported_analyzer_types": sorted(SUPPORTED_ANALYZER_TYPES),
        "supported_reducer_types": sorted(SUPPORTED_REDUCER_TYPES),
    }
    try:
        cfg = load_config(resolved)
    except Exception as exc:
        return {
            "config_path": repo_relative(resolved),
            "is_sleep2stat": True,
            "data_backend": None,
            "sleep2stat": supported,
            "warnings": [],
            "blocking_issues": [str(exc)],
            "agent_risk_issues": [],
        }

    analyzers = []
    reducers = []
    agent_risk_issues = []
    for item in cfg.analyzers:
        analyzer = {
            "name": item.name,
            "type": item.type,
            "enabled": item.enabled,
            "namespace": item.namespace,
            "label_name": item.label_name,
            "config": str(item.config) if item.config else None,
            "ckpt_path": str(item.ckpt_path) if item.ckpt_path else None,
            "input_channels": list(item.input_channels),
            "stage_source": item.stage_source,
            "event_source": item.event_source,
        }
        analyzers.append(analyzer)
        if item.enabled and item.type == "sleep2vec_downstream":
            if _looks_like_placeholder_path(item.config):
                agent_risk_issues.append(
                    f"Analyzer {item.name} downstream config is missing or placeholder: {item.config}"
                )
            if _looks_like_placeholder_path(item.ckpt_path):
                agent_risk_issues.append(f"Analyzer {item.name} ckpt_path is missing or placeholder: {item.ckpt_path}")
    for item in cfg.reducers:
        reducers.append(
            {
                "name": item.name,
                "type": item.type,
                "enabled": item.enabled,
                "source": item.source,
                "left": item.left,
                "right": item.right,
                "age_prediction": item.age_prediction,
                "sex_prediction": item.sex_prediction,
                "metadata_age_column": item.metadata_age_column,
                "metadata_sex_column": item.metadata_sex_column,
                "options": dict(item.options),
            }
        )

    return {
        "config_path": repo_relative(resolved),
        "is_sleep2stat": True,
        "data_backend": cfg.data.backend,
        "sleep2stat": {
            "run": {
                "name": cfg.run.name,
                "output_dir": str(cfg.run.output_dir),
            },
            "data": {
                "backend": cfg.data.backend,
                "index": str(cfg.data.index) if cfg.data.index else None,
                "kaldi_data_root": str(cfg.data.kaldi_data_root) if cfg.data.kaldi_data_root else None,
                "kaldi_manifest": str(cfg.data.kaldi_manifest) if cfg.data.kaldi_manifest else None,
                "split": list(cfg.data.split),
                "metadata_columns": list(cfg.data.metadata_columns),
                "token_sec": cfg.data.token_sec,
                "max_tokens": cfg.data.max_tokens,
            },
            "analyzers": analyzers,
            "reducers": reducers,
            **supported,
            "outputs": {
                "write_global_tables": cfg.outputs.write_global_tables,
                "write_per_record": cfg.outputs.write_per_record,
                "compression": cfg.outputs.compression,
                "global_tables": dict(cfg.outputs.global_tables),
            },
        },
        "warnings": [],
        "blocking_issues": [],
        "agent_risk_issues": agent_risk_issues,
    }


def config_summary(config_path: str | Path, *, validate_survival_local_paths: bool = True) -> dict[str, Any]:
    resolved = resolve_repo_path(config_path)
    if resolved is None:
        raise FileNotFoundError("Config path is required.")
    data = load_yaml(resolved)
    if _looks_like_sleep2stat_config_data(data):
        return sleep2stat_config_summary(resolved)
    model = data.get("model") if isinstance(data.get("model"), dict) else {}
    data_block = data.get("data") if isinstance(data.get("data"), dict) else {}
    finetune = data.get("finetune") if isinstance(data.get("finetune"), dict) else {}
    task = finetune.get("task") if isinstance(finetune.get("task"), dict) else {}
    survival = _survival_summary(finetune, task, validate_local_paths=validate_survival_local_paths)
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

    finetune_summary = {
        "task": {
            "type": task.get("type"),
            "output_dim": task.get("output_dim"),
            "is_seq": task.get("is_seq"),
            "monitor": task.get("monitor"),
            "monitor_mod": task.get("monitor_mod"),
        },
        "lora": lora,
        "loss": finetune.get("loss") if isinstance(finetune.get("loss"), dict) else {},
    }
    if survival is not None:
        finetune_summary["survival"] = survival

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
        "finetune": finetune_summary,
        "preset_build": {
            "required_channels": preset_build.get("required_channels"),
            "min_channels": preset_build.get("min_channels"),
        },
        "plausible_labels": list(BUILTIN_LABELS),
        "warnings": warnings,
        "blocking_issues": blocking_issues,
    }
    return summary
