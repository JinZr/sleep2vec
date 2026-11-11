"""The hyper-parameter recipe and search-space decision contract.

Mixed bridge: the generic hparam gates plus the multilabel coupling inherited
from ``decision_paths``. Owns what a well-formed search declaration is --
axes, budget, and objective -- before any plan is materialized. Materialization
itself belongs to ``plan_hparam``.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

from .decision_models import DecisionIssue, DecisionStatus, ResolvedDecision, needs_issue, question_for
from .decision_paths import (
    finetune_init_checkpoint_issue,
    managed_runtime_env_issues,
    managed_runtime_resource_issues,
    multilabel_sidecar_issue,
    sex_age_pretrained_backbone_issue,
)
from .models import REPO_ROOT, ConfigSummaryInput, is_full_git_object_id
from .plan_rendering import FINETUNE_SCHEDULER_FIELDS

_HPARAM_EXECUTION_FIELDS = {
    "workdir",
    "max_concurrent",
    "gpu_pool",
    "gpus_per_run",
    "env",
    "conda_env",
    "python",
    "runtime_commit",
    "scheduler",
    "wandb_project",
    "wandb_group",
}
_HPARAM_SCHEDULER_FIELDS = {
    "type",
    "partition",
    "cpus_per_task",
    "memory",
    "walltime",
    "nice",
    "nodelist",
    "direct_controller",
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
_HPARAM_INPUT_FIELDS = {
    "init_checkpoint_path",
    "ckpt_path",
    "config",
    "data_backend",
    "final_eval_config_path",
    "inference_preset_path",
    "label_name",
    "override_dataset_names",
    "pretrained_backbone_path",
}
_HPARAM_SEARCH_FIELDS = {"configurations", "max_runs", "max_trials", "method", "parameters"}


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
    execution_value = recipe.get("execution")
    execution = execution_value if isinstance(execution_value, dict) else {}
    if "scheduler" in execution:
        scheduler = execution["scheduler"]
        if not isinstance(scheduler, dict):
            issues.append(
                _contract_issue(
                    "execution.scheduler",
                    "execution.scheduler must be a mapping.",
                    scheduler,
                    source_layer,
                )
            )
        else:
            for field in sorted(set(scheduler) - _HPARAM_SCHEDULER_FIELDS):
                issues.append(
                    _contract_issue(
                        f"execution.scheduler.{field}",
                        f"Unknown hparam execution.scheduler field: {field}.",
                        scheduler[field],
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


def _hparam_config_issues(
    recipe: dict,
    config_summary: ConfigSummaryInput | None,
    decisions: dict[str, ResolvedDecision],
    high_impact: dict[str, dict[str, Any]],
) -> list[DecisionIssue]:
    issues = []
    local_recipe_value = recipe.get("_local_recipe")
    local_recipe = local_recipe_value if isinstance(local_recipe_value, dict) else recipe
    local_evaluation_value = local_recipe.get("evaluation_policy")
    local_evaluation = local_evaluation_value if isinstance(local_evaluation_value, dict) else {}
    local_decisions_value = local_recipe.get("decisions")
    local_decisions = local_decisions_value if isinstance(local_decisions_value, dict) else {}
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
        inputs_value = recipe.get("inputs")
        inputs = inputs_value if isinstance(inputs_value, dict) else {}
        config = inputs.get("config")
        if config:
            issues.append(
                DecisionIssue(
                    DecisionStatus.FAIL,
                    "config",
                    (
                        "Hparam plan generation needs local config YAML content; YAML overrides cannot be "
                        "generated from an unreadable config."
                    ),
                    None,
                    {"config": config},
                )
            )
    multilabel_issue = multilabel_sidecar_issue("hparam_tune", recipe, config_summary, uses_finetune_config=True)
    if multilabel_issue is not None:
        issues.append(multilabel_issue)
    pretrained_issue = sex_age_pretrained_backbone_issue(recipe)
    if pretrained_issue is not None:
        issues.append(pretrained_issue)
    init_issue = finetune_init_checkpoint_issue(recipe)
    if init_issue is not None:
        issues.append(init_issue)
    local_field_map = {
        "selection_metric": ("evaluation_policy.selection_metric", "selection_metric"),
        "selection_mode": ("evaluation_policy.selection_mode", "selection_mode"),
        "selection_split": ("evaluation_policy.selection_split", "train_val_test_policy"),
        "external_test_locked": ("evaluation_policy.external_test_locked", "external_test_locked"),
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
    return issues


def hparam_search_issues(recipe: dict, *, high_impact: dict[str, dict[str, Any]]) -> list[DecisionIssue]:
    search_value = recipe.get("search")
    search = search_value if isinstance(search_value, dict) else {}
    issues = []
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
    elif search.get("method") not in (None, "grid"):
        issues.append(
            DecisionIssue(
                DecisionStatus.FAIL,
                "search_method",
                "Only search.method=grid is supported.",
                None,
                {"method": search.get("method")},
            )
        )
    configurations = search.get("configurations")
    if "configurations" in search and "parameters" in search:
        issues.append(
            DecisionIssue(
                DecisionStatus.FAIL,
                "hparam_search_space",
                "search.parameters and search.configurations are mutually exclusive.",
                None,
                {"parameters": search.get("parameters"), "configurations": configurations},
            )
        )
    elif "configurations" in search:
        issues.extend(_hparam_search_configurations_issues(configurations))
    elif not search.get("parameters"):
        issues.append(needs_issue("hparam_search_space", "search.parameters is required.", high_impact))
    elif "parameters" in search:
        issues.extend(_hparam_search_parameter_issues(search.get("parameters")))
    return issues


def _hparam_search_budget_issues(
    search: dict[str, Any],
    *,
    high_impact: dict[str, dict[str, Any]],
) -> list[DecisionIssue]:
    issues = []
    max_runs = search.get("max_runs")
    if max_runs in (None, ""):
        issues.append(needs_issue("hparam_budget", "search.max_runs is required.", high_impact))
    elif max_runs not in (None, "") and (type(max_runs) is not int or max_runs <= 0):
        issues.append(
            DecisionIssue(
                DecisionStatus.FAIL,
                "hparam_budget",
                "search.max_runs must be a positive integer.",
                None,
                {"max_runs": max_runs},
            )
        )
    return issues


def _hparam_evaluation_issues(
    recipe: dict,
    decisions: dict[str, ResolvedDecision],
    high_impact: dict[str, dict[str, Any]],
) -> list[DecisionIssue]:
    issues = []
    evaluation_value = recipe.get("evaluation_policy")
    evaluation = evaluation_value if isinstance(evaluation_value, dict) else {}
    local_recipe_value = recipe.get("_local_recipe")
    local_recipe = local_recipe_value if isinstance(local_recipe_value, dict) else recipe
    local_evaluation_value = local_recipe.get("evaluation_policy")
    local_evaluation = local_evaluation_value if isinstance(local_evaluation_value, dict) else {}
    local_decisions_value = local_recipe.get("decisions")
    local_decisions = local_decisions_value if isinstance(local_decisions_value, dict) else {}
    user_external_lock = decisions.get("external_test_locked")
    has_external_lock = (
        "external_test_locked" in local_evaluation
        or "external_test_locked" in local_decisions
        or (user_external_lock is not None and user_external_lock.source == "explicit_user")
    )
    if not has_external_lock:
        issues.append(needs_issue("external_test_locked", "external_test_locked must be explicit.", high_impact))
    test_after_fit = decisions["test_after_fit"].value
    external_test_locked = evaluation.get("external_test_locked")
    if type(test_after_fit) is not bool:
        issues.append(
            DecisionIssue(
                DecisionStatus.NEEDS_USER_INPUT,
                "test_after_fit",
                "test_after_fit must be true or false when provided.",
                "Should test evaluation run after fit for this task?",
                {"value": test_after_fit, "evaluation_policy": evaluation},
            )
        )
    elif test_after_fit and external_test_locked is True:
        issues.append(
            DecisionIssue(
                DecisionStatus.NEEDS_USER_INPUT,
                "test_after_fit",
                "test_after_fit=true would evaluate test while external_test_locked=true.",
                "Should test_after_fit be false, or should external_test_locked=false?",
                {"evaluation_policy": evaluation},
            )
        )
    if evaluation.get("final_eval_split") == "test" and "require_manual_unlock_for_final_test" not in evaluation:
        issues.append(
            needs_issue("final_eval_unlock", "Final test evaluation requires manual unlock policy.", high_impact)
        )
    return issues


def hparam_tune_issues(
    recipe: dict,
    config_summary: ConfigSummaryInput | None,
    decisions: dict[str, ResolvedDecision],
    high_impact: dict[str, dict[str, Any]],
) -> list[DecisionIssue]:
    search_value = recipe.get("search")
    search = search_value if isinstance(search_value, dict) else {}
    execution_value = recipe.get("execution")
    execution = execution_value if isinstance(execution_value, dict) else {}
    runtime_value = recipe.get("runtime")
    runtime = runtime_value if isinstance(runtime_value, dict) else {}
    local_recipe_value = recipe.get("_local_recipe")
    local_recipe = local_recipe_value if isinstance(local_recipe_value, dict) else recipe
    local_runtime_value = local_recipe.get("runtime")
    local_runtime = local_runtime_value if isinstance(local_runtime_value, dict) else {}

    issues = hparam_recipe_contract_issues(recipe, source_layer="effective")
    issues.extend(_hparam_config_issues(recipe, config_summary, decisions, high_impact))
    issues.extend(hparam_search_issues(recipe, high_impact=high_impact))
    issues.extend(
        _hparam_execution_issues(
            execution,
            runtime,
            local_runtime=local_runtime,
            variant=str(recipe.get("variant") or ""),
        )
    )
    issues.extend(_hparam_search_budget_issues(search, high_impact=high_impact))
    issues.extend(_hparam_evaluation_issues(recipe, decisions, high_impact))
    return issues


def _hparam_execution_issues(
    execution: dict[str, Any],
    runtime: dict[str, Any],
    *,
    local_runtime: dict[str, Any] | None = None,
    variant: str = "",
) -> list[DecisionIssue]:
    scheduler = execution.get("scheduler") if "scheduler" in execution else {"type": "direct"}
    scheduler_type = scheduler.get("type") if isinstance(scheduler, dict) else None
    is_slurm = scheduler_type == "slurm"
    issues = _hparam_scheduler_issues(
        execution,
        scheduler=scheduler,
        is_slurm=is_slurm,
        local_runtime=local_runtime,
    )
    issues.extend(_hparam_runtime_identity_issues(execution))
    issues.extend(
        managed_runtime_resource_issues(
            execution,
            runtime,
            scheduler=scheduler,
            is_slurm=is_slurm,
            variant=variant,
        )
    )
    issues.extend(managed_runtime_env_issues(execution, is_slurm=is_slurm))
    return issues


def _hparam_scheduler_issues(
    execution: dict[str, Any],
    *,
    scheduler: Any,
    is_slurm: bool,
    local_runtime: dict[str, Any] | None,
) -> list[DecisionIssue]:
    issues: list[DecisionIssue] = []
    scheduler_type = scheduler.get("type") if isinstance(scheduler, dict) else None
    if scheduler_type not in {"direct", "slurm"}:
        issues.append(
            DecisionIssue(
                DecisionStatus.FAIL,
                "execution.scheduler.type",
                "execution.scheduler.type must be direct or slurm.",
                None,
                {"type": scheduler_type, "preflight_before_workspace": True},
            )
        )
    if scheduler_type == "direct" and set(scheduler) - {"type"}:
        issues.append(
            DecisionIssue(
                DecisionStatus.FAIL,
                "execution.scheduler",
                "Direct execution.scheduler accepts only type: direct.",
                None,
                {"scheduler": scheduler, "preflight_before_workspace": True},
            )
        )
    if is_slurm:
        for field in ("gpu_pool", "max_concurrent"):
            if field in execution:
                issues.append(
                    DecisionIssue(
                        DecisionStatus.FAIL,
                        f"execution.{field}",
                        f"execution.{field} is not used with execution.scheduler.type=slurm.",
                        None,
                        {field: execution[field], "preflight_before_workspace": True},
                    )
                )
        if execution.get("conda_env") not in (None, ""):
            issues.append(
                DecisionIssue(
                    DecisionStatus.FAIL,
                    "execution.conda_env",
                    "Slurm execution requires an explicit execution.python path instead of execution.conda_env.",
                    None,
                    {"conda_env": execution.get("conda_env"), "preflight_before_workspace": True},
                )
            )
        if local_runtime is not None and "devices" in local_runtime:
            issues.append(
                DecisionIssue(
                    DecisionStatus.FAIL,
                    "runtime.devices",
                    "Slurm hparam recipes derive logical runtime.devices from execution.gpus_per_run.",
                    None,
                    {"devices": local_runtime.get("devices"), "preflight_before_workspace": True},
                )
            )
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
    return issues


def _hparam_runtime_identity_issues(execution: dict[str, Any]) -> list[DecisionIssue]:
    issues: list[DecisionIssue] = []
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
    if python not in (None, "ASK_USER") and (
        not isinstance(python, str)
        or not python.strip()
        or python.startswith("~")
        or re.search(r"\s", python) is not None
    ):
        issues.append(
            DecisionIssue(
                DecisionStatus.FAIL,
                "execution.python",
                "execution.python must be a single executable name or path without whitespace, arguments, "
                "or ~ shorthand.",
                None,
                {"python": python},
            )
        )
    runtime_commit = execution.get("runtime_commit")
    if runtime_commit not in (None, "ASK_USER") and (
        not isinstance(runtime_commit, str) or not is_full_git_object_id(runtime_commit.lower())
    ):
        issues.append(
            DecisionIssue(
                DecisionStatus.FAIL,
                "execution.runtime_commit",
                "execution.runtime_commit must be a full 40-character Git commit ID when set.",
                None,
                {"runtime_commit": runtime_commit},
            )
        )
    manager_runtime = workdir in (None, "", str(REPO_ROOT)) and execution.get("conda_env") in (None, "")
    if not manager_runtime:
        for field, question in (
            ("python", "What Python executable name or absolute path should the target runtime use?"),
            ("runtime_commit", "What full Git commit hash should be recorded as the target runtime baseline?"),
        ):
            if field not in execution or execution.get(field) in (None, "ASK_USER"):
                issues.append(
                    DecisionIssue(
                        DecisionStatus.NEEDS_USER_INPUT,
                        f"execution.{field}",
                        f"execution.{field} must be explicit when the target runtime is not local REPO_ROOT.",
                        question,
                        {"workdir": workdir, "conda_env": execution.get("conda_env")},
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
        key_issue = _search_key_issue(key)
        if key_issue is not None:
            issues.append(key_issue)
    return issues


_ALLOWED_RUNTIME_SEARCH_FIELDS = frozenset(
    {
        *FINETUNE_SCHEDULER_FIELDS,
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
)


def _search_key_issue(key: Any) -> DecisionIssue | None:
    if isinstance(key, str) and key.startswith("runtime."):
        runtime_name = key.split(".", 1)[1]
        if runtime_name not in _ALLOWED_RUNTIME_SEARCH_FIELDS:
            return DecisionIssue(
                DecisionStatus.FAIL,
                "hparam_search_space",
                "Unsupported runtime search parameter.",
                None,
                {"parameter": key, "allowed_runtime": sorted(_ALLOWED_RUNTIME_SEARCH_FIELDS)},
            )
        return None
    if isinstance(key, str) and key.startswith("yaml:/"):
        return None
    return DecisionIssue(
        DecisionStatus.FAIL,
        "hparam_search_space",
        "Search parameters must use runtime.<name> or yaml:/json/pointer/path keys.",
        None,
        {"parameter": key},
    )


def _hparam_search_configurations_issues(configurations: Any) -> list[DecisionIssue]:
    if not isinstance(configurations, list) or not configurations:
        return [
            DecisionIssue(
                DecisionStatus.FAIL,
                "hparam_search_space",
                "search.configurations must be a non-empty list of configuration points.",
                None,
                {"configurations": configurations},
            )
        ]
    issues: list[DecisionIssue] = []
    for index, point in enumerate(configurations):
        if not isinstance(point, dict) or not point:
            issues.append(
                DecisionIssue(
                    DecisionStatus.FAIL,
                    "hparam_search_space",
                    "Each search configuration must be a non-empty mapping of search keys to values.",
                    None,
                    {"configuration_index": index, "value": point},
                )
            )
            continue
        for key in point:
            key_issue = _search_key_issue(key)
            if key_issue is not None:
                issues.append(key_issue)
    return issues
