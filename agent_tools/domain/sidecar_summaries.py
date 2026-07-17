from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from ..models import resolve_repo_path


def looks_like_placeholder_path(value: str | Path | None) -> bool:
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


def survival_summary(
    finetune: dict[str, Any],
    task: dict[str, Any],
    *,
    validate_local_paths: bool = True,
) -> dict[str, Any] | None:
    if task.get("type") != "survival":
        return None

    raw = finetune.get("survival") if isinstance(finetune.get("survival"), dict) else {}
    covariates = raw.get("covariates", [])
    if isinstance(covariates, list):
        covariates = list(covariates)
    path_fields = ("disease_columns_index", "event_time_index", "is_event_index", "has_label_index")
    summary: dict[str, Any] = {
        "key_column": raw.get("key_column"),
        "disease_columns_index": raw.get("disease_columns_index"),
        "event_time_index": raw.get("event_time_index"),
        "is_event_index": raw.get("is_event_index"),
        "has_label_index": raw.get("has_label_index"),
        "covariates": covariates,
        "covariate_embedding_dim": raw.get("covariate_embedding_dim", 16),
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
        if not isinstance(value, str) or looks_like_placeholder_path(value):
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


def multilabel_summary(
    finetune: dict[str, Any],
    task: dict[str, Any],
    *,
    validate_local_paths: bool = True,
) -> dict[str, Any] | None:
    if task.get("type") != "multilabel_classification":
        return None

    raw = finetune.get("multilabel") if isinstance(finetune.get("multilabel"), dict) else {}
    path_fields = ("disease_columns_index", "label_index", "has_label_index")
    summary: dict[str, Any] = {
        "key_column": raw.get("key_column"),
        "disease_columns_index": raw.get("disease_columns_index"),
        "label_index": raw.get("label_index"),
        "has_label_index": raw.get("has_label_index"),
        "output_dim": task.get("output_dim"),
        "valid": False,
        "disease_count": None,
        "sidecar_key_count": None,
        "issues": [],
    }

    issues = summary["issues"]
    if not isinstance(finetune.get("multilabel"), dict):
        issues.append("finetune.multilabel must be a mapping for multilabel tasks.")
        return summary
    if not isinstance(raw.get("key_column"), str) or not raw.get("key_column"):
        issues.append("finetune.multilabel.key_column must be a non-empty string.")
    resolved_paths: dict[str, str] = {}
    for field in path_fields:
        value = raw.get(field)
        if not isinstance(value, str) or looks_like_placeholder_path(value):
            issues.append(f"finetune.multilabel.{field} must point to a real file.")
            continue
        if not validate_local_paths:
            continue
        resolved = resolve_repo_path(value)
        if resolved is None or not resolved.exists():
            issues.append(f"finetune.multilabel.{field} does not exist: {value}")
        else:
            resolved_paths[field] = str(resolved)

    if issues or not validate_local_paths:
        return summary

    try:
        from data.multilabel import load_multilabel_label_table

        labels = load_multilabel_label_table(
            SimpleNamespace(key_column=raw["key_column"], **resolved_paths),
            task.get("output_dim"),
        )
    except Exception as exc:
        issues.append(str(exc))
        return summary

    if labels is not None:
        summary["valid"] = True
        summary["disease_count"] = len(labels.label_names)
        summary["sidecar_key_count"] = len(labels.disease_label)
    return summary
