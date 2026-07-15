from __future__ import annotations

from pathlib import Path
import subprocess  # noqa: F401 -- tests patch decision_paths.subprocess.run (stdlib global)
from typing import Any

from . import transport
from .decision_models import DecisionIssue, DecisionStatus
from .models import REPO_ROOT

_EXECUTION_FIELDS = {"host", "path_context", "path_validation", "target"}


def execution_contract_issues(recipe: dict, *, source_layer: str) -> list[DecisionIssue]:
    if "execution" not in recipe:
        return []
    execution = recipe["execution"]
    if not isinstance(execution, dict):
        return [_execution_contract_issue("execution", "execution must be a mapping.", execution, source_layer)]
    issues = []
    for field in sorted(set(execution) - _EXECUTION_FIELDS):
        issues.append(
            _execution_contract_issue(
                f"execution.{field}",
                f"Unknown execution field for this task: {field}.",
                execution[field],
                source_layer,
            )
        )
    return issues


def _execution_contract_issue(field: str, message: str, value: Any, source_layer: str) -> DecisionIssue:
    return DecisionIssue(
        DecisionStatus.FAIL,
        field,
        message,
        None,
        {"value": value, "source_layer": source_layer, "preflight_before_workspace": True},
    )


def _config_data(config_summary: dict | None) -> dict[str, Any]:
    data = config_summary.get("data") if isinstance(config_summary, dict) else {}
    return data if isinstance(data, dict) else {}


def _config_finetune(config_summary: dict | None) -> dict[str, Any]:
    finetune = config_summary.get("finetune") if isinstance(config_summary, dict) else {}
    return finetune if isinstance(finetune, dict) else {}


def _effective_preset_path(task: str, recipe: dict, config_summary: dict | None) -> tuple[str, Any]:
    inputs = recipe.get("inputs") if isinstance(recipe.get("inputs"), dict) else {}
    if task in {"infer", "evaluate"}:
        value = inputs.get("inference_preset_path")
        if value not in (None, "", "ASK_USER"):
            return "inference_preset_path", value
    value = _config_data(config_summary).get("finetune_preset_path")
    if task in {"finetune", "hparam_tune", "infer", "evaluate"} and value not in (None, "", "ASK_USER"):
        return "finetune_preset_path", value
    return "", None


def survival_sidecar_issue(
    task: str,
    recipe: dict,
    config_summary: dict | None,
) -> DecisionIssue | None:
    if not _requires_survival_sidecars(task, recipe, config_summary):
        return None
    survival = _config_finetune(config_summary).get("survival")
    if not isinstance(survival, dict) or not survival.get("issues"):
        return None
    return DecisionIssue(
        DecisionStatus.NEEDS_USER_INPUT,
        "survival_sidecars",
        "Survival sidecar files are missing or inconsistent.",
        (
            "Please provide valid disease_columns_index, event_time_index, is_event_index, and "
            "has_label_index files, and keep output_dim equal to the disease column count."
        ),
        {"survival": survival},
    )


def multilabel_sidecar_issue(
    task: str,
    recipe: dict,
    config_summary: dict | None,
) -> DecisionIssue | None:
    if not _requires_multilabel_sidecars(task, recipe, config_summary):
        return None
    multilabel = _config_finetune(config_summary).get("multilabel")
    if not isinstance(multilabel, dict) or not multilabel.get("issues"):
        return None
    return DecisionIssue(
        DecisionStatus.NEEDS_USER_INPUT,
        "multilabel_sidecars",
        "Multilabel sidecar files are missing or inconsistent.",
        (
            "Please provide valid disease_columns_index, label_index, and has_label_index files, "
            "and keep output_dim equal to the disease column count."
        ),
        {"multilabel": multilabel},
    )


def _requires_survival_sidecars(task: str, recipe: dict, config_summary: dict | None) -> bool:
    task_cfg = _config_finetune(config_summary).get("task")
    if not isinstance(task_cfg, dict) or task_cfg.get("type") != "survival":
        return False
    if task == "preset_prepare":
        return True
    if config_summary and config_summary.get("variant_guess") == "sex_age_baseline":
        return task in {"finetune", "hparam_tune", "infer", "evaluate"}
    if task in {"finetune", "hparam_tune", "infer", "evaluate"}:
        _field, preset_path = _effective_preset_path(task, recipe, config_summary)
        return preset_path in (None, "")
    return False


def _requires_multilabel_sidecars(task: str, recipe: dict, config_summary: dict | None) -> bool:
    task_cfg = _config_finetune(config_summary).get("task")
    if not isinstance(task_cfg, dict) or task_cfg.get("type") != "multilabel_classification":
        return False
    return task in {"finetune", "hparam_tune", "infer", "evaluate"}


def _append_remote_survival_sidecar_issues(
    issues: list[DecisionIssue],
    task: str,
    recipe: dict,
    config_summary: dict | None,
) -> None:
    if not _requires_survival_sidecars(task, recipe, config_summary):
        return
    survival = _config_finetune(config_summary).get("survival")
    if not isinstance(survival, dict):
        return
    for data_field in ("disease_columns_index", "event_time_index", "is_event_index", "has_label_index"):
        value = survival.get(data_field)
        if not value:
            continue
        context = path_context(recipe, value)
        validation = path_validation(recipe, context)
        if context == "remote" and validation in {"ssh", "remote"}:
            issue = validate_input_path(recipe, f"finetune.survival.{data_field}", value, configured=True)
            if issue is not None:
                issues.append(issue)


def _append_remote_multilabel_sidecar_issues(
    issues: list[DecisionIssue],
    task: str,
    recipe: dict,
    config_summary: dict | None,
) -> None:
    if not _requires_multilabel_sidecars(task, recipe, config_summary):
        return
    multilabel = _config_finetune(config_summary).get("multilabel")
    if not isinstance(multilabel, dict):
        return
    for data_field in ("disease_columns_index", "label_index", "has_label_index"):
        value = multilabel.get(data_field)
        if not value:
            continue
        context = path_context(recipe, value)
        validation = path_validation(recipe, context)
        if context == "remote" and validation in {"ssh", "remote"}:
            issue = validate_input_path(recipe, f"finetune.multilabel.{data_field}", value, configured=True)
            if issue is not None:
                issues.append(issue)


def path_issues(
    task: str,
    recipe: dict,
    config_summary: dict | None,
) -> list[DecisionIssue]:
    issues: list[DecisionIssue] = []
    inputs = recipe.get("inputs") if isinstance(recipe.get("inputs"), dict) else {}
    required_paths: list[tuple[str, Any]] = []
    if inputs.get("config"):
        required_paths.append(("config", inputs.get("config")))
    if task == "preset_prepare":
        for path in inputs.get("index") or []:
            required_paths.append(("index", path))
    if task in {"infer", "evaluate"}:
        ckpt_path = inputs.get("ckpt_path")
        if ckpt_path not in (None, "", "ASK_USER"):
            required_paths.append(("ckpt_path", ckpt_path))
    if task == "finetune":
        for input_field in ("pretrained_backbone_path", "ckpt_path"):
            if recipe.get("variant") == "sex_age_baseline" and input_field == "pretrained_backbone_path":
                continue
            value = inputs.get(input_field)
            if value not in (None, "", "ASK_USER"):
                required_paths.append((input_field, value))

    for path_field, raw_path in required_paths:
        issue = validate_input_path(recipe, path_field, raw_path, configured=False)
        if issue is not None:
            issues.append(issue)

    if task in {"finetune", "infer", "evaluate"} and config_summary and config_summary.get("data_backend") == "npz":
        data = _config_data(config_summary)
        preset_field, preset_path = _effective_preset_path(task, recipe, config_summary)
        if preset_path not in (None, ""):
            issue = validate_input_path(
                recipe, preset_field, preset_path, configured=preset_field == "finetune_preset_path"
            )
            if issue is not None:
                issues.append(issue)
        else:
            value = data.get("finetune_data_index")
            if value:
                issue = validate_input_path(recipe, "finetune_data_index", value, configured=True)
                if issue is not None:
                    issues.append(issue)
    if (
        task in {"finetune", "infer", "evaluate"}
        and config_summary
        and config_summary.get("variant_guess") == "sex_age_baseline"
        and config_summary.get("data_backend") == "kaldi"
    ):
        data = _config_data(config_summary)
        for data_field in ("kaldi_data_root", "kaldi_manifest"):
            value = data.get(data_field)
            if value:
                issue = validate_input_path(recipe, data_field, value, configured=True)
                if issue is not None:
                    issues.append(issue)
    _append_remote_survival_sidecar_issues(issues, task, recipe, config_summary)
    _append_remote_multilabel_sidecar_issues(issues, task, recipe, config_summary)
    if task == "sleep2stat" and config_summary and config_summary.get("is_sleep2stat"):
        sleep2stat = config_summary.get("sleep2stat") or {}
        data = sleep2stat.get("data") or {}
        for data_field in ("index", "kaldi_data_root", "kaldi_manifest"):
            value = data.get(data_field)
            if value:
                issue = validate_input_path(recipe, f"sleep2stat.data.{data_field}", value, configured=True)
                if issue is not None:
                    issues.append(issue)
        for analyzer in sleep2stat.get("analyzers", []):
            if analyzer.get("enabled") is False:
                continue
            for analyzer_field in ("config", "ckpt_path"):
                value = analyzer.get(analyzer_field)
                if not value or _looks_like_placeholder_path(value):
                    continue
                issue = validate_input_path(
                    recipe,
                    f"sleep2stat.analyzer.{analyzer.get('name')}.{analyzer_field}",
                    value,
                    configured=True,
                )
                if issue is not None:
                    issues.append(issue)
    return issues


def validate_input_path(recipe: dict, field: str, raw_path: Any, *, configured: bool) -> DecisionIssue | None:
    context = path_context(recipe, raw_path)
    validation = path_validation(recipe, context)
    if context not in {"local", "remote"}:
        return DecisionIssue(
            DecisionStatus.FAIL,
            "execution.path_context",
            "execution.path_context must be local or remote.",
            None,
            {"path_context": context},
        )
    if validation not in {"local", "remote", "defer", "ssh"}:
        return DecisionIssue(
            DecisionStatus.FAIL,
            "execution.path_validation",
            "execution.path_validation must be local, remote, defer, or ssh.",
            None,
            {"path_validation": validation},
        )
    if validation == "remote":
        validation = "ssh"
    if context == "remote" and validation == "defer":
        return DecisionIssue(
            DecisionStatus.WARN,
            field,
            f"{_path_label(configured)} path validation deferred for remote path: {raw_path}",
            None,
            {"path": str(raw_path), "path_context": "remote", "path_validation": "defer"},
        )
    if context == "remote" and validation == "ssh":
        host = _execution(recipe).get("host")
        if not host:
            return DecisionIssue(
                DecisionStatus.FAIL,
                "execution.host",
                "execution.host is required for remote path validation.",
                None,
                {"path": str(raw_path)},
            )
        result = transport.run_ssh(str(host), f"test -e {_sh(raw_path)}", text=True, timeout=None)
        if result.returncode != 0:
            return DecisionIssue(
                DecisionStatus.FAIL,
                field,
                f"{_path_label(configured)} remote path does not exist: {raw_path}",
                None,
                {"path": str(raw_path), "host": str(host), "stderr": result.stderr.strip()},
            )
        return None

    path = Path(str(raw_path)).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    if path.exists():
        return None
    return DecisionIssue(
        DecisionStatus.FAIL,
        field,
        f"{_path_label(configured)} path does not exist: {raw_path}",
        None,
        {"path": str(raw_path), "path_context": "local", "path_validation": validation},
    )


def sleep2stat_existing_run_dir_issue(recipe: dict, raw_path: Any) -> DecisionIssue | None:
    context = path_context(recipe, raw_path)
    validation = path_validation(recipe, context)
    if context != "local" or validation in {"remote", "ssh"}:
        return None
    path = Path(str(raw_path)).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.exists() or (path.is_dir() and not any(path.iterdir())):
        return None
    return DecisionIssue(
        DecisionStatus.NEEDS_USER_INPUT,
        "sleep2stat.run.output_dir",
        "sleep2stat run.output_dir already exists and is not empty; sleep2stat run directories are single-use.",
        "Use a fresh run.output_dir or manually clear the existing directory before generating commands.",
        {"path": str(raw_path), "resolved_path": str(path)},
    )


def _path_label(configured: bool) -> str:
    return "Configured input" if configured else "Required input"


def path_context(recipe: dict, raw_path: Any) -> str:
    execution = _execution(recipe)
    explicit = execution.get("path_context")
    if explicit:
        return str(explicit)
    if execution.get("target") == "ssh" and Path(str(raw_path)).expanduser().is_absolute():
        return "remote"
    return "local"


def path_validation(recipe: dict, context: str) -> str:
    explicit = _execution(recipe).get("path_validation")
    if explicit:
        return str(explicit)
    return "defer" if context == "remote" else "local"


def _execution(recipe: dict) -> dict[str, Any]:
    return recipe.get("execution") if isinstance(recipe.get("execution"), dict) else {}


_sh = transport.sh


def _looks_like_placeholder_path(value: Any) -> bool:
    text = str(value).strip()
    lowered = text.lower()
    return (
        lowered in {"", "ask_user", "none", "null", "todo", "tbd", "placeholder"}
        or text.startswith("/path/to")
        or text.startswith("<")
        or "ASK_USER" in text
    )
