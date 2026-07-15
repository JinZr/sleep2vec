from __future__ import annotations

from pathlib import Path
import re
from typing import Any

from . import gpu_rules
from .decision_models import DecisionIssue, DecisionStatus, ResolvedDecision, needs_issue, question_for
from .decision_paths import multilabel_sidecar_issue
from .models import REPO_ROOT

_HPARAM_EXECUTION_FIELDS = {
    "target",
    "host",
    "workdir",
    "path_context",
    "path_validation",
    "max_concurrent",
    "gpu_pool",
    "gpus_per_run",
    "env",
    "conda_env",
    "python",
    "runtime_commit",
    "wandb_project",
    "wandb_group",
}
_HPARAM_EVALUATION_FIELDS = {
    "selection_metric",
    "selection_mode",
    "selection_split",
    "external_test_locked",
    "test_after_fit",
    "final_eval_split",
    "final_test_unlocked",
    "require_manual_unlock_for_final_test",
}
_HPARAM_ADAPTIVE_FIELDS = {
    "enabled",
    "max_rounds",
    "max_runs_total",
    "objective_metric",
    "objective_mode",
    "poll_seconds",
    "replacement",
    "round_size",
    "suggest",
    "test_feedback_for_selection",
}
_HPARAM_ADAPTIVE_REPLACEMENT_FIELDS = {
    "allow_running_stop",
    "enabled",
    "grace_epochs",
    "grace_minutes",
    "kill_margin",
}
_HPARAM_ADAPTIVE_SUGGEST_FIELDS = {"strategy"}
_HPARAM_INPUT_FIELDS = {
    "ckpt_path",
    "config",
    "data_backend",
    "final_eval_config_path",
    "inference_preset_path",
    "label_name",
    "override_dataset_names",
    "pretrained_backbone_path",
}
_HPARAM_SEARCH_FIELDS = {"max_runs", "max_trials", "method", "parameters"}


def hparam_recipe_contract_issues(recipe: dict, *, source_layer: str) -> list[DecisionIssue]:
    issues: list[DecisionIssue] = []
    for section, allowed_fields in {
        "inputs": _HPARAM_INPUT_FIELDS,
        "search": _HPARAM_SEARCH_FIELDS,
        "evaluation_policy": _HPARAM_EVALUATION_FIELDS,
        "execution": _HPARAM_EXECUTION_FIELDS | {"gpus_per_trial", "log_dir", "pid_dir"},
    }.items():
        if section not in recipe:
            continue
        value = recipe[section]
        if not isinstance(value, dict):
            issues.append(_contract_issue(section, f"{section} must be a mapping.", value, source_layer))
            continue
        for field in sorted(set(value) - allowed_fields):
            issues.append(
                _contract_issue(
                    f"{section}.{field}",
                    f"Unknown hparam {section} field: {field}.",
                    value[field],
                    source_layer,
                )
            )

    if "adaptive" not in recipe:
        return issues
    adaptive = recipe["adaptive"]
    if not isinstance(adaptive, dict):
        issues.append(_contract_issue("adaptive", "adaptive must be a mapping.", adaptive, source_layer))
        return issues
    for field in sorted(set(adaptive) - _HPARAM_ADAPTIVE_FIELDS - {"max_trials_total"}):
        issues.append(
            _contract_issue(
                f"adaptive.{field}",
                f"Unknown adaptive field: {field}.",
                adaptive[field],
                source_layer,
            )
        )
    for section, allowed_fields in {
        "replacement": _HPARAM_ADAPTIVE_REPLACEMENT_FIELDS,
        "suggest": _HPARAM_ADAPTIVE_SUGGEST_FIELDS,
    }.items():
        if section not in adaptive:
            continue
        value = adaptive[section]
        if not isinstance(value, dict):
            issues.append(
                _contract_issue(f"adaptive.{section}", f"adaptive.{section} must be a mapping.", value, source_layer)
            )
            continue
        for field in sorted(set(value) - allowed_fields):
            issues.append(
                _contract_issue(
                    f"adaptive.{section}.{field}",
                    f"Unknown adaptive {section} field: {field}.",
                    value[field],
                    source_layer,
                )
            )
    return issues


def _contract_issue(field: str, message: str, value: Any, source_layer: str) -> DecisionIssue:
    return DecisionIssue(
        DecisionStatus.FAIL,
        field,
        message,
        None,
        {"value": value, "source_layer": source_layer, "preflight_before_workspace": True},
    )


def hparam_tune_issues(
    recipe: dict,
    config_summary: dict | None,
    decisions: dict[str, ResolvedDecision],
    high_impact: dict[str, dict[str, Any]],
) -> list[DecisionIssue]:
    issues = hparam_recipe_contract_issues(recipe, source_layer="effective")
    evaluation = recipe.get("evaluation_policy") if isinstance(recipe.get("evaluation_policy"), dict) else {}
    search = recipe.get("search") if isinstance(recipe.get("search"), dict) else {}
    execution = recipe.get("execution") if isinstance(recipe.get("execution"), dict) else {}
    runtime = recipe.get("runtime") if isinstance(recipe.get("runtime"), dict) else {}
    adaptive = recipe.get("adaptive") if isinstance(recipe.get("adaptive"), dict) else {}

    local_recipe = recipe.get("_local_recipe") if isinstance(recipe.get("_local_recipe"), dict) else recipe
    local_evaluation = (
        local_recipe.get("evaluation_policy") if isinstance(local_recipe.get("evaluation_policy"), dict) else {}
    )
    local_decisions = local_recipe.get("decisions") if isinstance(local_recipe.get("decisions"), dict) else {}
    if config_summary:
        for issue in config_summary.get("blocking_issues", []):
            issues.append(
                DecisionIssue(
                    DecisionStatus.NEEDS_USER_INPUT,
                    "config",
                    issue,
                    "Which corrected config, preset path, or index path should this task use?",
                    {"config_path": config_summary.get("config_path")},
                )
            )
    if config_summary is None:
        inputs = recipe.get("inputs") if isinstance(recipe.get("inputs"), dict) else {}
        config = inputs.get("config")
        if config:
            issues.append(
                DecisionIssue(
                    DecisionStatus.FAIL,
                    "config",
                    (
                        "Hparam plan generation needs local config YAML content; remote path validation may be "
                        "deferred, but YAML overrides cannot be generated from an unreadable config."
                    ),
                    None,
                    {"config": config},
                )
            )
    multilabel_issue = multilabel_sidecar_issue("hparam_tune", recipe, config_summary)
    if multilabel_issue is not None:
        issues.append(multilabel_issue)
    local_field_map = {
        "selection_metric": ("evaluation_policy.selection_metric", "selection_metric"),
        "selection_mode": ("evaluation_policy.selection_mode", "selection_mode"),
        "selection_split": ("evaluation_policy.selection_split", "train_val_test_policy"),
        "external_test_locked": ("evaluation_policy.external_test_locked", "external_test_locked"),
        "test_after_fit": ("evaluation_policy.test_after_fit", "test_after_fit"),
        "final_eval_split": ("evaluation_policy.final_eval_split", "final_eval_split"),
        "final_test_unlocked": ("evaluation_policy.final_test_unlocked", "final_eval_unlock"),
        "require_manual_unlock_for_final_test": (
            "evaluation_policy.require_manual_unlock_for_final_test",
            "final_eval_unlock",
        ),
    }
    for eval_field, (path, decision_field) in local_field_map.items():
        if eval_field not in local_evaluation and not _has_explicit_user_or_local_decision(
            decisions, local_decisions, decision_field
        ):
            issues.append(
                DecisionIssue(
                    DecisionStatus.NEEDS_USER_INPUT,
                    decision_field,
                    f"{path} must be explicit in the hparam recipe or user-decision file.",
                    question_for(high_impact, decision_field)
                    or f"What should {path} be for this hyper-parameter tuning task?",
                    {"local_recipe": "missing"},
                )
            )
    if not recipe.get("base_recipe"):
        issues.append(needs_issue("base_recipe", "base_recipe is required for hyper-parameter tuning.", high_impact))
    if "max_trials" in search:
        issues.append(
            DecisionIssue(
                DecisionStatus.FAIL,
                "search.max_trials",
                "search.max_trials is no longer supported; use search.max_runs.",
                None,
                {"max_trials": search.get("max_trials")},
            )
        )
    if not search.get("method"):
        issues.append(needs_issue("search_method", "search.method is required.", high_impact))
    elif search.get("method") != "grid":
        issues.append(
            DecisionIssue(
                DecisionStatus.FAIL,
                "search_method",
                "Only search.method=grid is supported.",
                None,
                {"method": search.get("method")},
            )
        )
    if not search.get("parameters"):
        issues.append(needs_issue("hparam_search_space", "search.parameters is required.", high_impact))
    else:
        issues.extend(_hparam_search_parameter_issues(search.get("parameters")))
    issues.extend(_hparam_execution_issues(execution, runtime))
    issues.extend(_hparam_adaptive_issues(adaptive))
    max_runs = search.get("max_runs")
    if max_runs in (None, ""):
        issues.append(needs_issue("hparam_budget", "search.max_runs is required.", high_impact))
    else:
        try:
            if int(max_runs) <= 0:
                raise ValueError
        except (TypeError, ValueError):
            issues.append(
                DecisionIssue(
                    DecisionStatus.FAIL,
                    "hparam_budget",
                    "search.max_runs must be a positive integer.",
                    None,
                    {"max_runs": max_runs},
                )
            )
    if evaluation.get("selection_split") == "test":
        issues.append(
            DecisionIssue(
                DecisionStatus.NEEDS_USER_INPUT,
                "selection_split",
                "selection_split=test is not allowed for model selection.",
                "Which validation split should be used for model selection?",
                {"evaluation_policy": evaluation},
            )
        )
    user_external_lock = decisions.get("external_test_locked")
    has_external_lock = (
        "external_test_locked" in local_evaluation
        or "external_test_locked" in local_decisions
        or (user_external_lock is not None and user_external_lock.source == "explicit_user")
    )
    if not has_external_lock:
        issues.append(needs_issue("external_test_locked", "external_test_locked must be explicit.", high_impact))
    test_after_fit = evaluation.get("test_after_fit")
    final_test_unlocked = evaluation.get("final_test_unlocked")
    external_test_locked = evaluation.get("external_test_locked")
    if test_after_fit is True and not (external_test_locked is False and final_test_unlocked is True):
        issues.append(
            DecisionIssue(
                DecisionStatus.NEEDS_USER_INPUT,
                "test_after_fit",
                "Run commands would evaluate test data without an explicit test unlock.",
                "Should test_after_fit be false, or should external_test_locked=false and final_test_unlocked=true?",
                {"evaluation_policy": evaluation},
            )
        )
    if evaluation.get("final_eval_split") == "test" and "require_manual_unlock_for_final_test" not in evaluation:
        issues.append(
            needs_issue("final_eval_unlock", "Final test evaluation requires manual unlock policy.", high_impact)
        )
    return issues


def _hparam_execution_issues(execution: dict[str, Any], runtime: dict[str, Any]) -> list[DecisionIssue]:
    issues: list[DecisionIssue] = []
    if "gpus_per_trial" in execution:
        issues.append(
            DecisionIssue(
                DecisionStatus.FAIL,
                "execution.gpus_per_trial",
                "execution.gpus_per_trial is no longer supported; use execution.gpus_per_run.",
                None,
                {"gpus_per_trial": execution.get("gpus_per_trial")},
            )
        )
    target = execution.get("target", "local")
    for field in ("log_dir", "pid_dir"):
        if field in execution:
            issues.append(
                DecisionIssue(
                    DecisionStatus.FAIL,
                    f"execution.{field}",
                    f"execution.{field} is not supported; logs and PIDs are stored in each managed run directory.",
                    None,
                    {field: execution.get(field)},
                )
            )
    if target not in {"local", "ssh"}:
        issues.append(
            DecisionIssue(
                DecisionStatus.FAIL,
                "execution.target",
                "execution.target must be local or ssh.",
                None,
                {"target": target},
            )
        )
    if target == "ssh" and not execution.get("host"):
        issues.append(
            DecisionIssue(
                DecisionStatus.FAIL,
                "execution.host",
                "execution.host is required when execution.target=ssh.",
                None,
                {},
            )
        )
    workdir = execution.get("workdir")
    if workdir not in (None, "") and not Path(str(workdir)).is_absolute():
        issues.append(
            DecisionIssue(
                DecisionStatus.FAIL,
                "execution.workdir",
                "execution.workdir must be an absolute path when set.",
                None,
                {"workdir": workdir},
            )
        )
    python = execution.get("python")
    if python not in (None, "ASK_USER") and (not isinstance(python, str) or not python.strip()):
        issues.append(
            DecisionIssue(
                DecisionStatus.FAIL,
                "execution.python",
                "execution.python must be a non-empty command or path when set.",
                None,
                {"python": python},
            )
        )
    runtime_commit = execution.get("runtime_commit")
    if runtime_commit not in (None, "ASK_USER") and (
        not isinstance(runtime_commit, str) or re.fullmatch(r"[0-9a-fA-F]{40}|[0-9a-fA-F]{64}", runtime_commit) is None
    ):
        issues.append(
            DecisionIssue(
                DecisionStatus.FAIL,
                "execution.runtime_commit",
                "execution.runtime_commit must be a full Git commit hash when set.",
                None,
                {"runtime_commit": runtime_commit},
            )
        )
    manager_runtime = (
        target == "local" and workdir in (None, "", str(REPO_ROOT)) and execution.get("conda_env") in (None, "")
    )
    if not manager_runtime:
        for field, question in (
            ("python", "What Python command or absolute path should the target runtime use?"),
            ("runtime_commit", "What full Git commit hash should the target runtime use?"),
        ):
            if field not in execution or execution.get(field) in (None, "ASK_USER"):
                issues.append(
                    DecisionIssue(
                        DecisionStatus.NEEDS_USER_INPUT,
                        f"execution.{field}",
                        f"execution.{field} must be explicit when the target runtime is not local REPO_ROOT.",
                        question,
                        {"target": target, "workdir": workdir, "conda_env": execution.get("conda_env")},
                    )
                )
    if execution.get("path_context") not in (None, "local", "remote"):
        issues.append(
            DecisionIssue(
                DecisionStatus.FAIL,
                "execution.path_context",
                "execution.path_context must be local or remote.",
                None,
                {"path_context": execution.get("path_context")},
            )
        )
    if execution.get("path_validation") not in (None, "local", "remote", "defer", "ssh"):
        issues.append(
            DecisionIssue(
                DecisionStatus.FAIL,
                "execution.path_validation",
                "execution.path_validation must be local, remote, defer, or ssh.",
                None,
                {"path_validation": execution.get("path_validation")},
            )
        )
    max_concurrent = None
    if "max_concurrent" in execution:
        try:
            max_concurrent = int(execution["max_concurrent"])
            if max_concurrent <= 0:
                raise ValueError
        except (TypeError, ValueError):
            issues.append(
                DecisionIssue(
                    DecisionStatus.FAIL,
                    "execution.max_concurrent",
                    "execution.max_concurrent must be a positive integer.",
                    None,
                    {"max_concurrent": execution.get("max_concurrent")},
                )
            )
    if "gpu_pool" in execution and not isinstance(execution["gpu_pool"], list):
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
        try:
            raw_gpus_per_run = execution["gpus_per_run"]
            gpus_per_run = int(raw_gpus_per_run)
            if (
                isinstance(raw_gpus_per_run, bool)
                or gpus_per_run <= 0
                or isinstance(raw_gpus_per_run, float)
                and not raw_gpus_per_run.is_integer()
            ):
                raise ValueError
        except (TypeError, ValueError):
            gpus_per_run = None
            issues.append(
                DecisionIssue(
                    DecisionStatus.FAIL,
                    "execution.gpus_per_run",
                    "execution.gpus_per_run must be a positive integer.",
                    None,
                    {"gpus_per_run": execution.get("gpus_per_run")},
                )
            )
    # An invalid (non-list) gpu_pool already failed above; drop it so the shared rules
    # fall back to runtime.devices, matching the previous inline behaviour. An invalid
    # gpus_per_run skips the pool rules entirely (type failure already reported).
    if gpus_per_run is not None or "gpus_per_run" not in execution:
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


def _hparam_adaptive_issues(adaptive: dict[str, Any]) -> list[DecisionIssue]:
    issues: list[DecisionIssue] = []
    if not adaptive:
        return issues
    if "max_trials_total" in adaptive:
        issues.append(
            DecisionIssue(
                DecisionStatus.FAIL,
                "adaptive.max_trials_total",
                "adaptive.max_trials_total is no longer supported; use adaptive.max_runs_total.",
                None,
                {"max_trials_total": adaptive.get("max_trials_total")},
            )
        )
    replacement = adaptive.get("replacement") if isinstance(adaptive.get("replacement"), dict) else {}
    if adaptive.get("enabled") is not True:
        return issues
    objective = str(adaptive.get("objective_metric") or "test_auroc")
    if (objective.startswith("test_") or objective.startswith("external_")) and adaptive.get(
        "test_feedback_for_selection"
    ) is not True:
        issues.append(
            DecisionIssue(
                DecisionStatus.FAIL,
                "adaptive.test_feedback_for_selection",
                (
                    "adaptive.test_feedback_for_selection=true is required when adaptive objective "
                    "uses test/external metrics."
                ),
                None,
                {"objective_metric": objective},
            )
        )
    if adaptive.get("objective_mode", "max") not in {"max", "min"}:
        issues.append(
            DecisionIssue(
                DecisionStatus.FAIL,
                "adaptive.objective_mode",
                "adaptive.objective_mode must be max or min.",
                None,
                {"objective_mode": adaptive.get("objective_mode")},
            )
        )
    for adaptive_field in ("max_rounds", "max_runs_total", "round_size", "poll_seconds"):
        if adaptive_field not in adaptive:
            continue
        try:
            if int(adaptive[adaptive_field]) <= 0:
                raise ValueError
        except (TypeError, ValueError):
            issues.append(
                DecisionIssue(
                    DecisionStatus.FAIL,
                    f"adaptive.{adaptive_field}",
                    f"adaptive.{adaptive_field} must be a positive integer.",
                    None,
                    {adaptive_field: adaptive.get(adaptive_field)},
                )
            )
    if replacement and replacement.get("kill_margin") is not None:
        try:
            if float(replacement["kill_margin"]) < 0:
                raise ValueError
        except (TypeError, ValueError):
            issues.append(
                DecisionIssue(
                    DecisionStatus.FAIL,
                    "adaptive.replacement.kill_margin",
                    "adaptive.replacement.kill_margin must be a non-negative number.",
                    None,
                    {"kill_margin": replacement.get("kill_margin")},
                )
            )
    return issues


def _has_explicit_user_or_local_decision(
    decisions: dict[str, ResolvedDecision],
    local_decisions: dict[str, Any],
    field: str,
) -> bool:
    decision = decisions.get(field)
    return field in local_decisions or decision is not None and decision.source == "explicit_user"


def _hparam_search_parameter_issues(parameters: Any) -> list[DecisionIssue]:
    issues: list[DecisionIssue] = []
    if not isinstance(parameters, dict):
        return [
            DecisionIssue(
                DecisionStatus.FAIL,
                "hparam_search_space",
                "search.parameters must be a mapping.",
                None,
                {"parameters": parameters},
            )
        ]
    allowed_runtime = {
        "lr",
        "weight_decay",
        "batch_size",
        "epochs",
        "num_workers",
        "precision",
        "gradient_clip_val",
        "accumulate_grad_batches",
        "warmup_steps",
        "patience",
        "check_val_every_n_epoch",
        "ckpt_every_n_epochs",
    }
    for key, values in parameters.items():
        if not isinstance(values, list) or not values:
            issues.append(
                DecisionIssue(
                    DecisionStatus.FAIL,
                    "hparam_search_space",
                    "Each search parameter must have a non-empty list of values.",
                    None,
                    {"parameter": key, "value": values},
                )
            )
        if isinstance(key, str) and key.startswith("runtime."):
            runtime_name = key.split(".", 1)[1]
            if runtime_name not in allowed_runtime:
                issues.append(
                    DecisionIssue(
                        DecisionStatus.FAIL,
                        "hparam_search_space",
                        "Unsupported runtime search parameter.",
                        None,
                        {"parameter": key, "allowed_runtime": sorted(allowed_runtime)},
                    )
                )
        elif isinstance(key, str) and key.startswith("yaml:/"):
            continue
        else:
            issues.append(
                DecisionIssue(
                    DecisionStatus.FAIL,
                    "hparam_search_space",
                    "Search parameters must use runtime.<name> or yaml:/json/pointer/path keys.",
                    None,
                    {"parameter": key},
                )
            )
    return issues
