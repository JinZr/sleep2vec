from __future__ import annotations

from pathlib import Path
from typing import Any

from ..models import CONFIG_FINETUNE_SECTION, load_yaml, repo_relative, resolve_repo_path
from .sidecar_summaries import multilabel_summary, survival_summary

BUILTIN_LABELS = ("stage3", "stage4", "stage5", "ahi", "sex", "age")


def guess_variant(config_path: str | Path) -> str:
    parts = Path(config_path).parts
    if "sex_age_baseline" in parts:
        return "sex_age_baseline"
    if "sleep2expert" in parts:
        return "sleep2expert"
    if "sleep2vec2" in parts:
        return "sleep2vec2"
    return "sleep2vec"


def _channel_summary(item: dict[str, Any]) -> dict[str, Any]:
    tokenizer_value = item.get("tokenizer")
    tokenizer = tokenizer_value if isinstance(tokenizer_value, dict) else {}
    return {
        "name": item.get("name"),
        "input_dim": item.get("input_dim"),
        "tokenizer": tokenizer.get("name"),
        "out_dim": tokenizer.get("out_dim"),
    }


def finetune_summary_body(
    config_path: str | Path,
    *,
    validate_survival_local_paths: bool = True,
    local_path_base: str | Path | None = None,
    validated_sidecar_keys: dict[str, set[str]] | None = None,
) -> dict[str, Any]:
    resolved = resolve_repo_path(config_path)
    if resolved is None:
        raise FileNotFoundError("Config path is required.")
    data = load_yaml(resolved)
    model_value = data.get("model")
    model = model_value if isinstance(model_value, dict) else {}
    data_block_value = data.get("data")
    data_block = data_block_value if isinstance(data_block_value, dict) else {}
    # A `finetune:` block that is present but empty is still a finetune config, so the checks
    # below ask this rather than `if finetune` -- `{}` is falsy and would skip every one of
    # them while the summary still reports `is_finetune: true`.
    raw_finetune = data.get(CONFIG_FINETUNE_SECTION)
    is_finetune = isinstance(raw_finetune, dict)
    finetune = raw_finetune if isinstance(raw_finetune, dict) else {}
    task_value = finetune.get("task")
    task = task_value if isinstance(task_value, dict) else {}
    survival = survival_summary(
        finetune,
        task,
        validate_local_paths=validate_survival_local_paths,
        local_path_base=local_path_base,
        validated_sidecar_keys=validated_sidecar_keys,
    )
    multilabel = multilabel_summary(
        finetune,
        task,
        validate_local_paths=validate_survival_local_paths,
        local_path_base=local_path_base,
        validated_sidecar_keys=validated_sidecar_keys,
    )
    preset_build_value = data.get("preset_build")
    preset_build = preset_build_value if isinstance(preset_build_value, dict) else {}
    head_value = model.get("head")
    head = head_value if isinstance(head_value, dict) else {}
    raw_head_kwargs_value = head.get("kwargs")
    raw_head_kwargs = raw_head_kwargs_value if isinstance(raw_head_kwargs_value, dict) else {}
    head_kwargs = {
        field: raw_head_kwargs[field] for field in ("attn_dropout", "temporal_dropout") if field in raw_head_kwargs
    }
    temporal_agg_value = head.get("temporal_agg")
    temporal_agg = temporal_agg_value if isinstance(temporal_agg_value, dict) else {}
    channel_agg_value = head.get("channel_agg")
    channel_agg = channel_agg_value if isinstance(channel_agg_value, dict) else {}
    layer_mix_value = finetune.get("layer_mix")
    layer_mix = layer_mix_value if isinstance(layer_mix_value, dict) else {}
    tuning_value = finetune.get("tuning")
    tuning = tuning_value if isinstance(tuning_value, dict) else {}
    backbone_value = model.get("backbone")
    backbone = backbone_value if isinstance(backbone_value, dict) else {}
    averaging_value = data.get("model_averaging")
    averaging = averaging_value if isinstance(averaging_value, dict) else None
    channels_raw_value = model.get("channels")
    channels_raw = channels_raw_value if isinstance(channels_raw_value, list) else []
    channels = [_channel_summary(item) for item in channels_raw if isinstance(item, dict)]
    model_channel_names = [item["name"] for item in channels if item.get("name")]
    data_channel_names = data_block.get("data_channel_names") or model_channel_names
    backend = data_block.get("backend") or "npz"
    warnings: list[str] = []
    blocking_issues: list[str] = []

    if data_channel_names and model_channel_names and list(data_channel_names) != model_channel_names:
        blocking_issues.append("data.data_channel_names differs from model.channels.")
    if backend == "kaldi":
        if not data_block.get("kaldi_data_root"):
            blocking_issues.append("data.backend=kaldi but data.kaldi_data_root is missing.")
        if not data_block.get("kaldi_manifest"):
            blocking_issues.append("data.backend=kaldi but data.kaldi_manifest is missing.")
        if data_block.get("finetune_preset_path"):
            blocking_issues.append("data.backend=kaldi does not support data.finetune_preset_path.")
    if (
        backend == "npz"
        and is_finetune
        and not data_block.get("finetune_data_index")
        and not data_block.get("finetune_preset_path")
    ):
        blocking_issues.append("data.backend=npz but both finetune_data_index and finetune_preset_path are missing.")
    if is_finetune and not isinstance(finetune.get("tuning"), dict):
        # Every variant loader requires this block, and agent_tools cannot call those loaders
        # (enforced forks). Without the check, `plan` emits a command that dies at config load.
        blocking_issues.append("finetune.tuning is missing; the config loader requires it.")
    elif is_finetune and not (type(tuning.get("preset")) is str and tuning["preset"]):
        # A block that names no preset states no policy, so it is the same gap as an absent
        # one -- `finetune.tuning: {}` reaches here as "present". Whether the named preset
        # exists is the loader's question; this only asks that the config named one.
        blocking_issues.append("finetune.tuning.preset is missing; the config loader requires it.")
    if is_finetune and task == {}:
        warnings.append("finetune.task is missing; custom label semantics may be ambiguous.")
    if model_channel_names == ["ppg"] and is_finetune and "required_channels" not in preset_build:
        warnings.append("single-channel PPG finetune config has no preset_build.required_channels.")

    finetune_summary = {
        "task": {
            "type": task.get("type"),
            "output_dim": task.get("output_dim"),
            "is_seq": task.get("is_seq"),
            "monitor": task.get("monitor"),
            "monitor_mod": task.get("monitor_mod"),
        },
        "tuning": tuning,
        "tuning_present": isinstance(finetune.get("tuning"), dict),
        "loss": finetune.get("loss") if isinstance(finetune.get("loss"), dict) else {},
    }
    if survival is not None:
        finetune_summary["survival"] = survival
    if multilabel is not None:
        finetune_summary["multilabel"] = multilabel

    summary: dict[str, Any] = {
        "config_path": repo_relative(resolved),
        "variant_guess": guess_variant(resolved),
        "is_finetune": is_finetune,
        "is_pretrain": not is_finetune,
        "data_backend": backend,
        "model": {
            "backbone": (model.get("backbone") or {}).get("name") if isinstance(model.get("backbone"), dict) else None,
            "hidden_size": (
                (model.get("backbone") or {}).get("hidden_size") if isinstance(model.get("backbone"), dict) else None
            ),
            "backbone_depth": backbone.get("num_hidden_layers"),
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
                "kwargs": head_kwargs,
                "channel_agg": {
                    "name": channel_agg.get("name"),
                    "kwargs": channel_agg.get("kwargs") if isinstance(channel_agg.get("kwargs"), dict) else {},
                },
                "temporal_agg": {
                    "name": temporal_agg.get("name"),
                    "kwargs": temporal_agg.get("kwargs") if isinstance(temporal_agg.get("kwargs"), dict) else {},
                },
            },
            "layer_mix_present": isinstance(finetune.get("layer_mix"), dict),
            "layer_mix": layer_mix,
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
            "train_dataset_names": list(data_block.get("train_dataset_names") or []),
            "test_dataset_names": list(data_block.get("test_dataset_names") or []),
            "kaldi_data_root": data_block.get("kaldi_data_root"),
            "kaldi_manifest": data_block.get("kaldi_manifest"),
        },
        CONFIG_FINETUNE_SECTION: finetune_summary,
        "preset_build": {
            "required_channels": preset_build.get("required_channels"),
            "min_channels": preset_build.get("min_channels"),
        },
        "plausible_labels": list(BUILTIN_LABELS),
        "warnings": warnings,
        "blocking_issues": blocking_issues,
    }
    return summary
