from __future__ import annotations

from pathlib import Path
import re
import subprocess  # noqa: F401 -- tests patch decision_paths.subprocess.run (stdlib global)
from typing import Any

from . import transport
from .decision_models import DecisionIssue, DecisionStatus
from .models import CONFIG_FINETUNE_SECTION, REPO_ROOT

_EXECUTION_FIELDS = {"host", "path_context", "path_validation", "target", "workdir"}
_RUNTIME_IDENTITY_FIELDS = {"python", "runtime_commit"}
_RUNTIME_IDENTITY_REQUIRED_FIELDS = {*_RUNTIME_IDENTITY_FIELDS, "workdir"}


def execution_contract_issues(
    recipe: dict, *, source_layer: str, supports_runtime_identity: bool
) -> list[DecisionIssue]:
    if "execution" not in recipe:
        return []
    execution = recipe["execution"]
    if not isinstance(execution, dict):
        return [_execution_contract_issue("execution", "execution must be a mapping.", execution, source_layer)]
    allowed_fields = _EXECUTION_FIELDS | (_RUNTIME_IDENTITY_FIELDS if supports_runtime_identity else set())
    issues = []
    for field in sorted(set(execution) - allowed_fields):
        issues.append(
            _execution_contract_issue(
                f"execution.{field}",
                f"Unknown execution field for this task: {field}.",
                execution[field],
                source_layer,
            )
        )
    identity_fields = set(execution) & _RUNTIME_IDENTITY_FIELDS
    workdir = execution.get("workdir")
    if "workdir" in execution and (
        not isinstance(workdir, str) or not workdir.strip() or workdir == "ASK_USER" or not Path(workdir).is_absolute()
    ):
        issues.append(
            _execution_contract_issue(
                "execution.workdir",
                "execution.workdir must be an explicit absolute path.",
                workdir,
                source_layer,
            )
        )
    if not supports_runtime_identity or not identity_fields:
        return issues
    for field in sorted(_RUNTIME_IDENTITY_REQUIRED_FIELDS - set(execution)):
        issues.append(
            _execution_contract_issue(
                f"execution.{field}",
                "execution.python, execution.runtime_commit, and execution.workdir must be provided together.",
                None,
                source_layer,
            )
        )
    target = execution.get("target")
    if target not in (None, "", "local"):
        issues.append(
            _execution_contract_issue(
                "execution.target",
                "Explicit runtime identity supports only local execution.",
                target,
                source_layer,
            )
        )
    python_command = execution.get("python")
    if "python" in execution and (
        not isinstance(python_command, str)
        or not python_command.strip()
        or python_command == "ASK_USER"
        or python_command.startswith("~")
        or re.search(r"\s", python_command) is not None
    ):
        issues.append(
            _execution_contract_issue(
                "execution.python",
                "execution.python must be a single executable name or path without whitespace, arguments, "
                "or ~ shorthand.",
                python_command,
                source_layer,
            )
        )
    runtime_commit = execution.get("runtime_commit")
    if "runtime_commit" in execution and (
        not isinstance(runtime_commit, str) or re.fullmatch(r"[0-9a-f]{40}", runtime_commit) is None
    ):
        issues.append(
            _execution_contract_issue(
                "execution.runtime_commit",
                "execution.runtime_commit must be a lowercase 40-character Git commit SHA.",
                runtime_commit,
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
    finetune = config_summary.get(CONFIG_FINETUNE_SECTION) if isinstance(config_summary, dict) else {}
    return finetune if isinstance(finetune, dict) else {}


def _effective_preset_path(
    task: str,
    recipe: dict,
    config_summary: dict | None,
    recipe_field: str | None = None,
    *,
    uses_finetune_config: bool = False,
) -> tuple[str, Any]:
    inputs = recipe.get("inputs") if isinstance(recipe.get("inputs"), dict) else {}
    if recipe_field is not None:
        value = inputs.get(recipe_field)
        if value not in (None, "", "ASK_USER"):
            return recipe_field, value
    value = _config_data(config_summary).get("finetune_preset_path")
    if uses_finetune_config and value not in (None, "", "ASK_USER"):
        return "finetune_preset_path", value
    return "", None


def survival_sidecar_issue(
    task: str,
    recipe: dict,
    config_summary: dict | None,
    *,
    required: bool | None = None,
    preset_path_recipe_field: str | None = None,
    uses_finetune_config: bool = False,
) -> DecisionIssue | None:
    if not _requires_survival_sidecars(
        task, recipe, config_summary, required, preset_path_recipe_field, uses_finetune_config
    ):
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
    *,
    required: bool | None = None,
    preset_path_recipe_field: str | None = None,
    uses_finetune_config: bool = False,
) -> DecisionIssue | None:
    if not _requires_multilabel_sidecars(
        task, recipe, config_summary, required, preset_path_recipe_field, uses_finetune_config
    ):
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


def _requires_survival_sidecars(
    task: str,
    recipe: dict,
    config_summary: dict | None,
    required: bool | None = None,
    preset_path_recipe_field: str | None = None,
    uses_finetune_config: bool = False,
) -> bool:
    task_cfg = _config_finetune(config_summary).get("task")
    if not isinstance(task_cfg, dict) or task_cfg.get("type") != "survival":
        return False
    if required is not None:
        return required
    if config_summary and config_summary.get("authoritative_variant") == "sex_age_baseline":
        return uses_finetune_config
    if uses_finetune_config:
        _field, preset_path = _effective_preset_path(
            task, recipe, config_summary, preset_path_recipe_field, uses_finetune_config=uses_finetune_config
        )
        return preset_path in (None, "")
    return False


def _requires_multilabel_sidecars(
    task: str,
    recipe: dict,
    config_summary: dict | None,
    required: bool | None = None,
    preset_path_recipe_field: str | None = None,
    uses_finetune_config: bool = False,
) -> bool:
    task_cfg = _config_finetune(config_summary).get("task")
    if not isinstance(task_cfg, dict) or task_cfg.get("type") != "multilabel_classification":
        return False
    if required is not None:
        return required
    return uses_finetune_config


def _append_remote_survival_sidecar_issues(
    issues: list[DecisionIssue],
    task: str,
    recipe: dict,
    config_summary: dict | None,
    required: bool | None = None,
    preset_path_recipe_field: str | None = None,
    uses_finetune_config: bool = False,
) -> None:
    if not _requires_survival_sidecars(
        task, recipe, config_summary, required, preset_path_recipe_field, uses_finetune_config
    ):
        return
    survival = _config_finetune(config_summary).get("survival")
    if not isinstance(survival, dict):
        return
    for data_field in ("disease_columns_index", "event_time_index", "is_event_index", "has_label_index"):
        value = survival.get(data_field)
        if not value:
            continue
        context = path_context(recipe, value, relative_to_workdir=True)
        validation = path_validation(recipe, context)
        if str(value).startswith("~") or (context == "remote" and validation in {"ssh", "remote"}):
            issue = validate_input_path(
                recipe,
                f"finetune.survival.{data_field}",
                value,
                configured=True,
                require_file=True,
            )
            if issue is not None:
                issues.append(issue)


def _append_remote_multilabel_sidecar_issues(
    issues: list[DecisionIssue],
    task: str,
    recipe: dict,
    config_summary: dict | None,
    required: bool | None = None,
    preset_path_recipe_field: str | None = None,
    uses_finetune_config: bool = False,
) -> None:
    if not _requires_multilabel_sidecars(
        task, recipe, config_summary, required, preset_path_recipe_field, uses_finetune_config
    ):
        return
    multilabel = _config_finetune(config_summary).get("multilabel")
    if not isinstance(multilabel, dict):
        return
    for data_field in ("disease_columns_index", "label_index", "has_label_index"):
        value = multilabel.get(data_field)
        if not value:
            continue
        context = path_context(recipe, value, relative_to_workdir=True)
        validation = path_validation(recipe, context)
        if str(value).startswith("~") or (context == "remote" and validation in {"ssh", "remote"}):
            issue = validate_input_path(
                recipe,
                f"finetune.multilabel.{data_field}",
                value,
                configured=True,
                require_file=True,
            )
            if issue is not None:
                issues.append(issue)


def path_issues(
    task: str,
    recipe: dict,
    config_summary: dict | None,
    *,
    required_input_paths: list[tuple[str, Any]] | None = None,
    requires_survival_sidecars: bool | None = None,
    requires_multilabel_sidecars: bool | None = None,
    preset_path_recipe_field: str | None = None,
    validates_dataset_paths: bool = False,
    uses_finetune_config: bool = False,
) -> list[DecisionIssue]:
    issues: list[DecisionIssue] = []
    inputs = recipe.get("inputs") if isinstance(recipe.get("inputs"), dict) else {}
    required_paths: list[tuple[str, Any, bool]] = []
    if inputs.get("config"):
        required_paths.append(("config", inputs.get("config"), False))
    required_paths.extend((field, path, True) for field, path in required_input_paths or [])

    for path_field, raw_path, relative_to_workdir in required_paths:
        issue = validate_input_path(
            recipe,
            path_field,
            raw_path,
            configured=False,
            relative_to_workdir=relative_to_workdir,
            require_file=True,
        )
        if issue is not None:
            issues.append(issue)

    if validates_dataset_paths and config_summary and config_summary.get("data_backend") == "npz":
        data = _config_data(config_summary)
        preset_field, preset_path = _effective_preset_path(
            task, recipe, config_summary, preset_path_recipe_field, uses_finetune_config=uses_finetune_config
        )
        if preset_path not in (None, ""):
            issue = validate_input_path(
                recipe,
                preset_field,
                preset_path,
                configured=preset_field == "finetune_preset_path",
                require_file=True,
            )
            if issue is not None:
                issues.append(issue)
        else:
            value = data.get("finetune_data_index")
            if value:
                issue = validate_input_path(recipe, "finetune_data_index", value, configured=True, require_file=True)
                if issue is not None:
                    issues.append(issue)
    if validates_dataset_paths and config_summary and config_summary.get("data_backend") == "kaldi":
        data = _config_data(config_summary)
        preset_field, preset_path = _effective_preset_path(
            task, recipe, config_summary, preset_path_recipe_field, uses_finetune_config=uses_finetune_config
        )
        if preset_path not in (None, ""):
            issues.append(
                DecisionIssue(
                    DecisionStatus.FAIL,
                    preset_field,
                    "Kaldi backend does not support an NPZ inference or finetune preset.",
                    None,
                    {"data_backend": "kaldi", "preset_path": str(preset_path)},
                )
            )
        for data_field in ("kaldi_data_root", "kaldi_manifest"):
            value = data.get(data_field)
            if value:
                issue = validate_input_path(
                    recipe,
                    data_field,
                    value,
                    configured=True,
                    require_directory=data_field == "kaldi_data_root",
                    require_file=data_field == "kaldi_manifest",
                )
                if issue is not None:
                    issues.append(issue)
    _append_remote_survival_sidecar_issues(
        issues,
        task,
        recipe,
        config_summary,
        requires_survival_sidecars,
        preset_path_recipe_field,
        uses_finetune_config,
    )
    _append_remote_multilabel_sidecar_issues(
        issues,
        task,
        recipe,
        config_summary,
        requires_multilabel_sidecars,
        preset_path_recipe_field,
        uses_finetune_config,
    )
    return issues


def validate_input_path(
    recipe: dict,
    field: str,
    raw_path: Any,
    *,
    configured: bool,
    relative_to_workdir: bool = True,
    require_directory: bool = False,
    require_file: bool = False,
) -> DecisionIssue | None:
    if relative_to_workdir and str(raw_path).startswith("~"):
        return DecisionIssue(
            DecisionStatus.FAIL,
            field,
            "Runtime input paths must not use ~ home-directory shorthand; use an absolute or workdir-relative path.",
            None,
            {"path": str(raw_path), "preflight_before_workspace": True},
        )
    execution = _execution(recipe)
    context = path_context(recipe, raw_path, relative_to_workdir=relative_to_workdir)
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
        host = execution.get("host")
        if not host:
            return DecisionIssue(
                DecisionStatus.FAIL,
                "execution.host",
                "execution.host is required for remote path validation.",
                None,
                {"path": str(raw_path)},
            )
        validation_path = raw_path
        if relative_to_workdir and not Path(str(raw_path)).is_absolute():
            workdir = execution.get("workdir") or REPO_ROOT
            if Path(str(workdir)).is_absolute():
                validation_path = Path(str(workdir)) / str(raw_path)
        test_flag = "-d" if require_directory else "-f" if require_file else "-e"
        result = transport.run_ssh(str(host), f"test {test_flag} {_sh(validation_path)}", text=True, timeout=None)
        if result.returncode != 0:
            expected = "directory" if require_directory else "file" if require_file else "path"
            return DecisionIssue(
                DecisionStatus.FAIL,
                field,
                f"{_path_label(configured)} remote {expected} does not exist: {raw_path}",
                None,
                {"path": str(raw_path), "host": str(host), "stderr": result.stderr.strip()},
            )
        return None

    path = Path(str(raw_path))
    if not relative_to_workdir:
        path = path.expanduser()
    if not path.is_absolute():
        base = REPO_ROOT
        workdir = execution.get("workdir")
        if relative_to_workdir and workdir not in (None, "") and Path(str(workdir)).is_absolute():
            base = Path(str(workdir))
        path = base / path
    path_exists = path.is_dir() if require_directory else path.is_file() if require_file else path.exists()
    if path_exists:
        return None
    expected = "directory" if require_directory else "file" if require_file else "path"
    return DecisionIssue(
        DecisionStatus.FAIL,
        field,
        f"{_path_label(configured)} {expected} does not exist: {raw_path}",
        None,
        {"path": str(raw_path), "path_context": "local", "path_validation": validation},
    )


def inference_checkpoint_averaging_issue(recipe: dict, ckpt_path: Any) -> DecisionIssue | None:
    inputs = recipe.get("inputs") if isinstance(recipe.get("inputs"), dict) else {}
    runtime = recipe.get("runtime") if isinstance(recipe.get("runtime"), dict) else {}
    avg_ckpts_value = runtime.get("avg_ckpts", 1)
    avg_ckpts = avg_ckpts_value if type(avg_ckpts_value) is int and avg_ckpts_value > 0 else 1
    if recipe.get("variant") == "sex_age_baseline" and avg_ckpts != 1:
        return DecisionIssue(
            DecisionStatus.FAIL,
            "runtime.avg_ckpts",
            "sex_age_baseline inference does not support checkpoint averaging.",
            None,
            {"avg_ckpts": runtime.get("avg_ckpts")},
        )
    if avg_ckpts <= 1:
        return None
    if inputs.get("label_name") == "ahi":
        return DecisionIssue(
            DecisionStatus.FAIL,
            "runtime.avg_ckpts",
            "AHI inference does not support checkpoint averaging.",
            None,
            {"avg_ckpts": runtime.get("avg_ckpts")},
        )
    avg_ckpt_dir = runtime.get("avg_ckpt_dir")
    if ckpt_path in ("best", "last") and avg_ckpt_dir in (None, "", "ASK_USER"):
        return DecisionIssue(
            DecisionStatus.FAIL,
            "runtime.avg_ckpt_dir",
            "Checkpoint averaging with ckpt_path=best/last requires avg_ckpt_dir.",
            None,
            {"ckpt_path": ckpt_path, "avg_ckpts": avg_ckpts},
        )
    if avg_ckpt_dir not in (None, "", "ASK_USER"):
        return validate_input_path(
            recipe,
            "avg_ckpt_dir",
            avg_ckpt_dir,
            configured=False,
            require_directory=True,
        )
    return None


def sex_age_pretrained_backbone_issue(recipe: dict) -> DecisionIssue | None:
    if recipe.get("variant") != "sex_age_baseline":
        return None
    inputs = recipe.get("inputs") if isinstance(recipe.get("inputs"), dict) else {}
    value = inputs.get("pretrained_backbone_path")
    if value in (None, "", "ASK_USER"):
        return None
    return DecisionIssue(
        DecisionStatus.FAIL,
        "pretrained_backbone_path",
        "sex_age_baseline does not support pretrained_backbone_path.",
        None,
        {"variant": "sex_age_baseline", "pretrained_backbone_path": value},
    )


def _path_label(configured: bool) -> str:
    return "Configured input" if configured else "Required input"


def path_context(recipe: dict, raw_path: Any, *, relative_to_workdir: bool = False) -> str:
    execution = _execution(recipe)
    explicit = execution.get("path_context")
    if explicit:
        return str(explicit)
    path = Path(str(raw_path))
    if not relative_to_workdir:
        path = path.expanduser()
    if execution.get("target") == "ssh" and (relative_to_workdir or path.is_absolute()):
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
