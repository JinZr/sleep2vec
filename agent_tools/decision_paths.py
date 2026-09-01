from __future__ import annotations

from pathlib import Path
import re
import subprocess  # noqa: F401 -- tests patch decision_paths.subprocess.run (stdlib global)
from typing import Any

from . import gpu_rules, slurm, transport
from .decision_models import DecisionIssue, DecisionStatus
from .models import CONFIG_FINETUNE_SECTION, REPO_ROOT, is_full_git_object_id

_EXECUTION_FIELDS = {"host", "path_context", "path_validation", "target", "workdir"}
_RUNTIME_IDENTITY_FIELDS = {"python", "runtime_commit"}
_RUNTIME_IDENTITY_REQUIRED_FIELDS = {*_RUNTIME_IDENTITY_FIELDS, "workdir"}


def execution_contract_issues(
    recipe: dict,
    *,
    source_layer: str,
    supports_runtime_identity: bool,
    supports_slurm: bool = False,
    supports_direct: bool = False,
) -> list[DecisionIssue]:
    if "execution" not in recipe:
        return []
    execution = recipe["execution"]
    if not isinstance(execution, dict):
        return [_execution_contract_issue("execution", "execution must be a mapping.", execution, source_layer)]
    allowed_fields = _EXECUTION_FIELDS | (_RUNTIME_IDENTITY_FIELDS if supports_runtime_identity else set())
    scheduler = execution.get("scheduler")
    is_slurm = supports_slurm and isinstance(scheduler, dict) and scheduler.get("type") == "slurm"
    is_direct = supports_direct and isinstance(scheduler, dict) and scheduler.get("type") == "direct"
    if is_slurm:
        allowed_fields |= {"scheduler", "gpus_per_run", "env"}
    if is_direct:
        allowed_fields.add("scheduler")
    issues = []
    if is_direct:
        for field in sorted(set(scheduler) - {"type"}):
            issues.append(
                _execution_contract_issue(
                    f"execution.scheduler.{field}",
                    f"Unknown execution.scheduler field for direct execution: {field}.",
                    scheduler[field],
                    source_layer,
                )
            )
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
    if is_slurm:
        for field in sorted(_RUNTIME_IDENTITY_REQUIRED_FIELDS):
            if execution.get(field) in (None, "", "ASK_USER"):
                issues.append(
                    _execution_contract_issue(
                        f"execution.{field}",
                        f"Slurm execution requires explicit execution.{field}.",
                        execution.get(field),
                        source_layer,
                    )
                )
        runtime = recipe.get("runtime") if isinstance(recipe.get("runtime"), dict) else {}
        issues.extend(
            issue
            for issue in managed_runtime_resource_issues(
                execution, runtime, scheduler=scheduler, is_slurm=True, variant=str(recipe.get("variant"))
            )
            if issue.status == DecisionStatus.FAIL
        )
        # Unknown scheduler fields must fail at the recipe boundary, not first during compilation.
        for field in sorted(set(scheduler) - slurm.RESOURCE_FIELDS):
            issues.append(
                _execution_contract_issue(
                    f"execution.scheduler.{field}",
                    f"Unknown execution.scheduler field: {field}.",
                    scheduler[field],
                    source_layer,
                )
            )
        gpus = execution.get("gpus_per_run", 1)
        if type(gpus) is int and gpus > 0 and "devices" in runtime and runtime["devices"] != list(range(gpus)):
            issues.append(
                _execution_contract_issue(
                    "runtime.devices",
                    "Slurm runtime.devices must match logical devices derived from execution.gpus_per_run.",
                    runtime["devices"],
                    source_layer,
                )
            )
        for field, allowed in (("accelerator", {"gpu", "auto"}), ("device", {"cuda", "cuda:0"})):
            if field in runtime and runtime[field] not in allowed:
                issues.append(
                    _execution_contract_issue(
                        f"runtime.{field}",
                        f"Slurm runtime.{field} must use GPU execution.",
                        runtime[field],
                        source_layer,
                    )
                )
        issues.extend(managed_runtime_env_issues(execution, is_slurm=True))
    if not supports_runtime_identity or (not identity_fields and not is_slurm):
        return issues
    missing_fields = _RUNTIME_IDENTITY_REQUIRED_FIELDS - set(execution) if not is_slurm else set()
    for field in sorted(missing_fields):
        issues.append(
            _execution_contract_issue(
                f"execution.{field}",
                "execution.python, execution.runtime_commit, and execution.workdir must be provided together.",
                None,
                source_layer,
            )
        )
    target = execution.get("target")
    if is_slurm:
        if target not in (None, "local", "ssh"):
            issues.append(
                _execution_contract_issue(
                    "execution.target", "execution.target must be local or ssh.", target, source_layer
                )
            )
        if target == "ssh" and not execution.get("host"):
            issues.append(
                _execution_contract_issue(
                    "execution.host",
                    "execution.host is required when execution.target=ssh.",
                    execution.get("host"),
                    source_layer,
                )
            )
    elif target not in (None, "", "local"):
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
        not isinstance(runtime_commit, str) or not is_full_git_object_id(runtime_commit.lower())
    ):
        issues.append(
            _execution_contract_issue(
                "execution.runtime_commit",
                "execution.runtime_commit must be a full 40-character Git commit ID.",
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


def managed_runtime_resource_issues(
    execution: dict[str, Any],
    runtime: dict[str, Any],
    *,
    scheduler: Any,
    is_slurm: bool,
    variant: str,
) -> list[DecisionIssue]:
    issues: list[DecisionIssue] = []
    max_concurrent = None
    if "max_concurrent" in execution and not is_slurm:
        raw_max_concurrent = execution["max_concurrent"]
        if type(raw_max_concurrent) is not int or raw_max_concurrent <= 0:
            issues.append(
                DecisionIssue(
                    DecisionStatus.FAIL,
                    "execution.max_concurrent",
                    "execution.max_concurrent must be a positive integer.",
                    None,
                    {"max_concurrent": execution.get("max_concurrent")},
                )
            )
        else:
            max_concurrent = raw_max_concurrent
    if not is_slurm and "gpu_pool" in execution and not isinstance(execution["gpu_pool"], list):
        issues.append(
            DecisionIssue(
                DecisionStatus.FAIL,
                "execution.gpu_pool",
                "execution.gpu_pool must be a list.",
                None,
                {"gpu_pool": execution.get("gpu_pool")},
            )
        )
    gpus_per_run = None
    if "gpus_per_run" in execution:
        raw_gpus_per_run = execution["gpus_per_run"]
        if type(raw_gpus_per_run) is not int or raw_gpus_per_run <= 0:
            issues.append(
                DecisionIssue(
                    DecisionStatus.FAIL,
                    "execution.gpus_per_run",
                    "execution.gpus_per_run must be a positive integer.",
                    None,
                    {"gpus_per_run": execution.get("gpus_per_run")},
                )
            )
        else:
            gpus_per_run = raw_gpus_per_run
    if is_slurm and variant == "sex_age_baseline" and gpus_per_run is not None and gpus_per_run > 1:
        issues.append(
            DecisionIssue(
                DecisionStatus.FAIL,
                "execution.gpus_per_run",
                "sex_age_baseline does not support multi-GPU Slurm execution.",
                None,
                {"gpus_per_run": gpus_per_run, "variant": variant, "preflight_before_workspace": True},
            )
        )
    if is_slurm and isinstance(scheduler, dict) and not (set(scheduler) - slurm.RESOURCE_FIELDS):
        try:
            slurm.normalize_resources(scheduler, gpus_per_run if gpus_per_run is not None else 1)
        except ValueError as exc:
            issues.append(
                DecisionIssue(
                    DecisionStatus.FAIL,
                    "execution.scheduler",
                    str(exc),
                    None,
                    {"scheduler": scheduler, "preflight_before_workspace": True},
                )
            )
        else:
            priority_message = (
                "Slurm priority is cluster-managed and cannot be guaranteed by this tool. Keep nice=0, submit "
                "frozen jobs promptly, request the shortest credible walltime and only necessary CPU, memory, and "
                "GPU resources, and avoid nodelist unless it is required."
            )
            if scheduler.get("nice", 0) > 0:
                priority_message += f" The configured nice={scheduler['nice']} voluntarily lowers priority."
            if scheduler.get("nodelist"):
                priority_message += " The configured nodelist narrows eligible nodes and may delay backfill."
            issues.append(
                DecisionIssue(
                    DecisionStatus.WARN,
                    "execution.scheduler.priority",
                    priority_message,
                    None,
                    {
                        "nice": scheduler.get("nice", 0),
                        "nodelist": scheduler.get("nodelist", ""),
                    },
                )
            )
    # An invalid (non-list) gpu_pool already failed above; drop it so the shared rules
    # fall back to runtime.devices, matching the previous inline behaviour. An invalid
    # gpus_per_run skips the pool rules entirely (type failure already reported).
    if not is_slurm and (gpus_per_run is not None or "gpus_per_run" not in execution):
        invalid_gpu_pool = "gpu_pool" in execution and not isinstance(execution["gpu_pool"], list)
        gpu_execution = (
            {key: value for key, value in execution.items() if key != "gpu_pool"} if invalid_gpu_pool else execution
        )
        _groups, gpu_issues = gpu_rules.gpu_group_plan(gpu_execution, runtime, max_concurrent=max_concurrent)
        issues.extend(
            DecisionIssue(
                DecisionStatus.WARN if issue.warning else DecisionStatus.FAIL,
                issue.field,
                issue.message,
                None,
                issue.evidence,
            )
            for issue in gpu_issues
        )
    return issues


def managed_runtime_env_issues(
    execution: dict[str, Any],
    *,
    is_slurm: bool,
) -> list[DecisionIssue]:
    issues: list[DecisionIssue] = []
    if "env" in execution and not isinstance(execution["env"], dict):
        issues.append(
            DecisionIssue(
                DecisionStatus.FAIL,
                "execution.env",
                "execution.env must be a mapping.",
                None,
                {"env": execution.get("env")},
            )
        )
    if isinstance(execution.get("env"), dict):
        for env_name, value in execution["env"].items():
            if not isinstance(env_name, str) or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", env_name) is None:
                issues.append(
                    DecisionIssue(
                        DecisionStatus.FAIL,
                        f"execution.env.{env_name}",
                        "execution.env keys must be POSIX environment variable names.",
                        None,
                        {"name": env_name},
                    )
                )
            if not isinstance(value, (str, int, float, bool)):
                issues.append(
                    DecisionIssue(
                        DecisionStatus.FAIL,
                        f"execution.env.{env_name}",
                        "execution.env values must be scalar strings, numbers, or booleans.",
                        None,
                        {"value": value},
                    )
                )
            if (
                isinstance(env_name, str)
                and is_slurm
                and (
                    env_name.startswith("SLURM_")
                    or env_name == "CUDA_VISIBLE_DEVICES"
                    or env_name in {"RANK", "LOCAL_RANK", "WORLD_SIZE"}
                    or env_name.startswith("MASTER_")
                )
            ):
                issues.append(
                    DecisionIssue(
                        DecisionStatus.FAIL,
                        f"execution.env.{env_name}",
                        f"{env_name} is owned by Slurm and cannot be overridden in execution.env.",
                        None,
                        {env_name: value},
                    )
                )
        for env_name, field in {
            "PYTHONPATH": "execution.workdir",
            "WANDB_PROJECT": "execution.wandb_project",
            "WANDB_GROUP": "execution.wandb_group",
            "WANDB_RUN_GROUP": "execution.wandb_group",
            "WANDB_MODE": "runtime.wandb_mode",
        }.items():
            if env_name in execution["env"]:
                issues.append(
                    DecisionIssue(
                        DecisionStatus.FAIL,
                        f"execution.env.{env_name}",
                        f"{env_name} is not supported in execution.env; use {field}.",
                        None,
                        {env_name: execution["env"][env_name]},
                    )
                )
    return issues


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
    if inputs.get("config") not in (None, "", "ASK_USER"):
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
        command = f"test {test_flag} {_sh(validation_path)} && printf 'path-present\\n'"
        result = transport.run_ssh(str(host), command, text=True, timeout=None)
        # Some SSH endpoints hide the child exit status; an empty result cannot prove existence.
        if result.returncode != 0 or result.stdout != "path-present\n":
            expected = "directory" if require_directory else "file" if require_file else "path"
            return DecisionIssue(
                DecisionStatus.FAIL,
                field,
                f"{_path_label(configured)} remote {expected} could not be verified: {raw_path}",
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
