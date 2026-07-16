from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, NamedTuple

from ..models import CONFIG_FINETUNE_SECTION, repo_relative, resolve_repo_path
from ..domain.sidecar_summaries import looks_like_placeholder_path, multilabel_summary, survival_summary


class ConfigSummaryProvider(NamedTuple):
    #: Variant name that forces this provider even when the config shape does
    #: not match (None disables forcing).
    force_variant: str | None
    #: Config-shape probe on the raw loaded mapping.
    matches: Callable[[dict[str, Any]], bool]
    #: Produce the structured summary for a resolved config path.
    summarize: Callable[..., dict[str, Any]]


def _looks_like_sex_age_baseline_config_data(data: dict[str, Any]) -> bool:
    model = data.get("model") if isinstance(data.get("model"), dict) else {}
    return model.get("name") == "sex_age_mlp"


def sex_age_baseline_config_summary(
    config_path: str | Path,
    *,
    validate_survival_local_paths: bool = True,
) -> dict[str, Any]:
    from ..models import load_yaml

    resolved = resolve_repo_path(config_path)
    if resolved is None:
        raise FileNotFoundError("Config path is required.")
    data = load_yaml(resolved)
    try:
        from sex_age_baseline.config import load_config

        cfg = load_config(resolved)
    except Exception as exc:
        return {
            "config_path": repo_relative(resolved),
            "variant_guess": "sex_age_baseline",
            "is_finetune": True,
            "is_pretrain": False,
            "data_backend": None,
            "model": {"name": "sex_age_mlp", "features": []},
            "data": {},
            CONFIG_FINETUNE_SECTION: {},
            "preset_build": {},
            "plausible_labels": [],
            "warnings": [],
            "blocking_issues": [str(exc)],
        }

    raw_finetune = data.get(CONFIG_FINETUNE_SECTION) if isinstance(data.get(CONFIG_FINETUNE_SECTION), dict) else {}
    raw_task = raw_finetune.get("task") if isinstance(raw_finetune.get("task"), dict) else {}
    survival = survival_summary(
        raw_finetune,
        raw_task,
        validate_local_paths=validate_survival_local_paths,
    )
    multilabel = multilabel_summary(
        raw_finetune,
        raw_task,
        validate_local_paths=validate_survival_local_paths,
    )
    finetune_data_index = cfg.data.finetune_data_index
    finetune_preset_path = cfg.data.finetune_preset_path
    kaldi_data_root = cfg.data.kaldi_data_root
    kaldi_manifest = cfg.data.kaldi_manifest
    finetune_summary = {
        "task": {
            "type": cfg.finetune.task.type,
            "output_dim": cfg.finetune.task.output_dim,
            "is_seq": cfg.finetune.task.is_seq,
            "monitor": cfg.finetune.task.monitor,
            "monitor_mod": cfg.finetune.task.monitor_mod,
        },
        "loss": raw_finetune.get("loss") if isinstance(raw_finetune.get("loss"), dict) else {},
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
            "features": list(cfg.model.features),
            "head_details": {
                "hidden_dim": cfg.model.head.hidden_dim,
                "dropout": cfg.model.head.dropout,
                "activation": cfg.model.head.activation,
            },
        },
        "data": {
            "backend": cfg.data.backend,
            "finetune_data_index": None if looks_like_placeholder_path(finetune_data_index) else finetune_data_index,
            "finetune_preset_path": (
                None if looks_like_placeholder_path(finetune_preset_path) else finetune_preset_path
            ),
            "kaldi_data_root": None if looks_like_placeholder_path(kaldi_data_root) else kaldi_data_root,
            "kaldi_manifest": None if looks_like_placeholder_path(kaldi_manifest) else kaldi_manifest,
            "split_column": cfg.data.split_column,
            "key_column": cfg.data.key_column,
            "deduplicate_by_key": cfg.data.deduplicate_by_key,
        },
        CONFIG_FINETUNE_SECTION: finetune_summary,
        "preset_build": {},
        "plausible_labels": [],
        "warnings": [],
        "blocking_issues": [],
    }


CONFIG_SUMMARY_PROVIDERS: tuple[ConfigSummaryProvider, ...] = (
    ConfigSummaryProvider(
        force_variant="sex_age_baseline",
        matches=_looks_like_sex_age_baseline_config_data,
        summarize=sex_age_baseline_config_summary,
    ),
)
