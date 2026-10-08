"""The ``sex_age_baseline`` config summary, claimed by config shape.

Layer 0 domain leaf, reached through ``adapters.config_providers`` rather than
by task name: this variant is recognized from the loaded mapping, so a config
cannot be summarized under the wrong family just by being pointed at the wrong
task.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..models import (
    CONFIG_FINETUNE_SECTION,
    SexAgeConfigSummary,
    SexAgeDataSummary,
    TaskConfigSummary,
    load_yaml,
    repo_relative,
    resolve_repo_path,
)
from .sidecar_summaries import looks_like_placeholder_path, multilabel_summary, survival_summary


def _looks_like_sex_age_baseline_config_data(data: dict[str, Any]) -> bool:
    model = data.get("model")
    if not isinstance(model, dict):
        model = {}
    return model.get("name") == "sex_age_mlp"


def sex_age_baseline_config_summary(
    config_path: str | Path,
    *,
    validate_survival_local_paths: bool = True,
    local_path_base: str | Path | None = None,
    validated_sidecar_keys: dict[str, set[str]] | None = None,
) -> SexAgeConfigSummary:
    resolved = resolve_repo_path(config_path)
    if resolved is None:
        raise FileNotFoundError("Config path is required.")
    data = load_yaml(resolved)
    try:
        from sex_age_baseline.config import covariate_task_config, load_config

        cfg = load_config(resolved)
    except Exception as exc:
        return {
            "config_path": repo_relative(resolved),
            "variant_guess": "sex_age_baseline",
            "is_finetune": True,
            "is_pretrain": False,
            "data_backend": None,
            "model": {"name": "sex_age_mlp"},
            "data": {},
            CONFIG_FINETUNE_SECTION: {},
            "preset_build": {},
            "plausible_labels": [],
            "warnings": [],
            "blocking_issues": [str(exc)],
        }

    raw_finetune = data.get(CONFIG_FINETUNE_SECTION)
    if not isinstance(raw_finetune, dict):
        raw_finetune = {}
    raw_task = raw_finetune.get("task")
    if not isinstance(raw_task, dict):
        raw_task = {}
    survival = survival_summary(
        raw_finetune,
        raw_task,
        validate_local_paths=validate_survival_local_paths,
        local_path_base=local_path_base,
        validated_sidecar_keys=validated_sidecar_keys,
    )
    multilabel = multilabel_summary(
        raw_finetune,
        raw_task,
        validate_local_paths=validate_survival_local_paths,
        local_path_base=local_path_base,
        validated_sidecar_keys=validated_sidecar_keys,
    )
    finetune_data_index = cfg.data.finetune_data_index
    finetune_preset_path = cfg.data.finetune_preset_path
    kaldi_data_root = cfg.data.kaldi_data_root
    kaldi_manifest = cfg.data.kaldi_manifest
    data_summary: SexAgeDataSummary = {
        "backend": cfg.data.backend,
        "finetune_data_index": None if looks_like_placeholder_path(finetune_data_index) else finetune_data_index,
        "finetune_preset_path": None if looks_like_placeholder_path(finetune_preset_path) else finetune_preset_path,
        "kaldi_data_root": None if looks_like_placeholder_path(kaldi_data_root) else kaldi_data_root,
        "kaldi_manifest": None if looks_like_placeholder_path(kaldi_manifest) else kaldi_manifest,
    }
    blocking_issues: list[str] = []
    # The generic finetune summary's data-source rules, which this config family does not reach.
    if cfg.data.backend == "kaldi":
        if not data_summary["kaldi_data_root"]:
            blocking_issues.append("data.backend=kaldi but data.kaldi_data_root is missing.")
        if not data_summary["kaldi_manifest"]:
            blocking_issues.append("data.backend=kaldi but data.kaldi_manifest is missing.")
        if data_summary["finetune_preset_path"]:
            blocking_issues.append("data.backend=kaldi does not support data.finetune_preset_path.")
    elif not data_summary["finetune_data_index"] and not data_summary["finetune_preset_path"]:
        blocking_issues.append("data.backend=npz but both finetune_data_index and finetune_preset_path are missing.")
    if survival is not None and validate_survival_local_paths:
        # Unlike the signal loaders, the baseline reads survival label names in preset mode too: they belong to its
        # checkpoint label contract, so the preset exemption from survival sidecar checks does not cover this file.
        names_path = covariate_task_config(cfg).disease_columns_index
        resolved_names = resolve_repo_path(names_path, relative_to=local_path_base)
        if resolved_names is None or not resolved_names.exists():
            blocking_issues.append(f"finetune.survival.disease_columns_index does not exist: {names_path}")
    raw_loss = raw_finetune.get("loss")
    finetune_summary: TaskConfigSummary = {
        "task": {
            "type": cfg.finetune.task.type,
            "output_dim": cfg.finetune.task.output_dim,
            "is_seq": cfg.finetune.task.is_seq,
            "monitor": cfg.finetune.task.monitor,
            "monitor_mod": cfg.finetune.task.monitor_mod,
        },
        "loss": raw_loss if isinstance(raw_loss, dict) else {},
    }
    if survival is not None:
        finetune_summary["survival"] = survival
    if multilabel is not None:
        finetune_summary["multilabel"] = multilabel
    return {
        "config_path": repo_relative(resolved),
        "variant_guess": "sex_age_baseline",
        "is_finetune": True,
        "is_pretrain": False,
        "data_backend": cfg.data.backend,
        "model": {
            "name": cfg.model.name,
            "head_details": {
                "name": cfg.model.head.name,
                "hidden_dim": cfg.model.head.hidden_dim,
                "dropout": cfg.model.head.dropout,
                "act": cfg.model.head.act,
                "kwargs": dict(cfg.model.head.kwargs),
            },
        },
        "data": data_summary,
        CONFIG_FINETUNE_SECTION: finetune_summary,
        "preset_build": {},
        "plausible_labels": [],
        "warnings": [],
        "blocking_issues": blocking_issues,
    }
