from __future__ import annotations

import math
from pathlib import Path
import re
from typing import Any

from .adaptive_proposals import validate_parameter_envelopes
from .decision_models import DecisionIssue, DecisionStatus, ResolvedDecision, needs_issue, question_for
from .decision_paths import managed_runtime_env_issues, managed_runtime_resource_issues, multilabel_sidecar_issue
from .models import REPO_ROOT, is_full_git_object_id

DEFAULT_ADAPTIVE_SUGGEST_STRATEGY = "agent_proposal"

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
_HPARAM_ADAPTIVE_SUGGEST_FIELDS = {"bounds", "strategy"}
_HPARAM_AGENT_PROPOSAL_REQUIRED_FIELDS = (
    "objective_metric",
    "objective_mode",
    "round_size",
    "max_rounds",
    "max_runs_total",
)
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
_HPARAM_SEARCH_FIELDS = {"configurations", "max_runs", "max_trials", "method", "parameters", "profile"}


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
    execution = recipe.get("execution") if isinstance(recipe.get("execution"), dict) else {}
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
    suggest = adaptive.get("suggest") if isinstance(adaptive.get("suggest"), dict) else {}
    strategy = suggest.get("strategy", DEFAULT_ADAPTIVE_SUGGEST_STRATEGY)
    if not isinstance(strategy, str) or strategy not in {"best_neighborhood", "agent_proposal"}:
        issues.append(
            _contract_issue(
                "adaptive.suggest.strategy",
                "adaptive.suggest.strategy must be best_neighborhood or agent_proposal.",
                strategy,
                source_layer,
            )
        )
    if "bounds" in suggest and strategy != "agent_proposal":
        issues.append(
            _contract_issue(
                "adaptive.suggest.bounds",
                "adaptive.suggest.bounds is supported only with strategy=agent_proposal.",
                suggest["bounds"],
                source_layer,
            )
        )
    # A disabled adaptive block never starts the suggestion protocol.
    if strategy == "agent_proposal" and adaptive.get("enabled") is True:
        for field in _HPARAM_AGENT_PROPOSAL_REQUIRED_FIELDS:
            value = adaptive.get(field)
            if field == "objective_metric" and field in adaptive and value is not None and not isinstance(value, str):
                issues.append(
                    _contract_issue(
                        "adaptive.objective_metric",
                        (
                            "adaptive.objective_metric must be a non-blank string when "
                            "adaptive.suggest.strategy=agent_proposal."
                        ),
                        value,
                        source_layer,
                    )
                )
                continue
            if field in adaptive and value not in (None, "") and (field != "objective_metric" or value.strip()):
                continue
            field_path = f"adaptive.{field}"
            issues.append(
                DecisionIssue(
                    DecisionStatus.NEEDS_USER_INPUT,
                    field_path,
                    f"{field_path} must be explicit when adaptive.suggest.strategy=agent_proposal.",
                    f"What should {field_path} be for this agent-proposal workflow?",
                    {
                        "value": value,
                        "source_layer": source_layer,
                        "preflight_before_workspace": True,
                    },
                )
            )
        replacement = adaptive.get("replacement")
        if replacement is not None and not (
            isinstance(replacement, dict) and set(replacement) == {"enabled"} and replacement.get("enabled") is False
        ):
            issues.append(
                _contract_issue(
                    "adaptive.replacement",
                    "strategy=agent_proposal requires adaptive.replacement to be omitted or exactly {enabled: false}.",
                    replacement,
                    source_layer,
                )
            )
        parameters = recipe.get("search", {}).get("parameters") if isinstance(recipe.get("search"), dict) else None
        if isinstance(parameters, dict) and parameters:
            try:
                validate_parameter_envelopes(parameters, suggest.get("bounds"))
            except ValueError as exc:
                issues.append(
                    _contract_issue(
                        "adaptive.suggest.bounds",
                        str(exc),
                        suggest.get("bounds"),
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
    config_summary: dict | None,
    decisions: dict[str, ResolvedDecision],
    high_impact: dict[str, dict[str, Any]],
) -> list[DecisionIssue]:
    issues = []
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
    multilabel_issue = multilabel_sidecar_issue("hparam_tune", recipe, config_summary, uses_finetune_config=True)
    if multilabel_issue is not None:
        issues.append(multilabel_issue)
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


def hparam_search_issues(
    search: dict[str, Any],
    *,
    profile_mode: bool,
    high_impact: dict[str, dict[str, Any]],
) -> list[DecisionIssue]:
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
    if not search.get("method") and not profile_mode:
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
    elif not search.get("parameters") and not profile_mode:
        issues.append(needs_issue("hparam_search_space", "search.parameters is required.", high_impact))
    elif "parameters" in search:
        issues.extend(_hparam_search_parameter_issues(search.get("parameters")))
    return issues


def _hparam_search_budget_issues(
    search: dict[str, Any],
    *,
    profile_mode: bool,
    high_impact: dict[str, dict[str, Any]],
) -> list[DecisionIssue]:
    issues = []
    max_runs = search.get("max_runs")
    if max_runs in (None, "") and not profile_mode:
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
    evaluation = recipe.get("evaluation_policy") if isinstance(recipe.get("evaluation_policy"), dict) else {}
    search = recipe.get("search") if isinstance(recipe.get("search"), dict) else {}
    runtime = recipe.get("runtime") if isinstance(recipe.get("runtime"), dict) else {}
    adaptive = recipe.get("adaptive") if isinstance(recipe.get("adaptive"), dict) else {}
    local_recipe = recipe.get("_local_recipe") if isinstance(recipe.get("_local_recipe"), dict) else recipe
    local_evaluation = (
        local_recipe.get("evaluation_policy") if isinstance(local_recipe.get("evaluation_policy"), dict) else {}
    )
    local_decisions = local_recipe.get("decisions") if isinstance(local_recipe.get("decisions"), dict) else {}
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
    selection_split = evaluation.get("selection_split")
    selection_metric = evaluation.get("selection_metric")
    max_runs = search.get("max_runs")
    configurations = search.get("configurations")
    if (
        selection_split == "test"
        and selection_metric not in (None, "", "ASK_USER")
        and (not isinstance(selection_metric, str) or not selection_metric.startswith("test_"))
    ):
        issues.append(
            DecisionIssue(
                DecisionStatus.FAIL,
                "selection_metric",
                "selection_split=test requires a test_* selection_metric.",
                None,
                {"evaluation_policy": evaluation},
            )
        )
    if selection_split == "test":
        parameters = search.get("parameters")
        search_space_is_plannable = (
            search.get("method") == "grid"
            and type(max_runs) is int
            and max_runs > 0
            and not ("configurations" in search and "parameters" in search)
            and (
                isinstance(configurations, list)
                and bool(configurations)
                and all(isinstance(point, dict) and point for point in configurations)
                or isinstance(parameters, dict)
                and bool(parameters)
                and all(isinstance(values, list) and values for values in parameters.values())
            )
        )
        if search_space_is_plannable:
            from .plan_hparam import hparam_combos

            planned_combos = hparam_combos(recipe)
            checkpoint_intervals = [
                combo.get("runtime.ckpt_every_n_epochs", runtime.get("ckpt_every_n_epochs", 1))
                for combo in planned_combos
            ]
            epoch_counts = [combo.get("runtime.epochs", runtime.get("epochs", 30)) for combo in planned_combos]
            validation_intervals = [
                combo.get("runtime.check_val_every_n_epoch", runtime.get("check_val_every_n_epoch", 1))
                for combo in planned_combos
            ]
        else:
            checkpoint_intervals = []
            epoch_counts = []
            validation_intervals = []
        # Early stopping can occur before a wider interval fires, so test-selected tuning must save every epoch.
        if any(type(interval) is not int or interval != 1 for interval in checkpoint_intervals):
            issues.append(
                DecisionIssue(
                    DecisionStatus.FAIL,
                    "runtime.ckpt_every_n_epochs",
                    "selection_split=test requires effective runtime.ckpt_every_n_epochs=1 for every trial.",
                    None,
                    {"effective_values": checkpoint_intervals},
                )
            )
        if any(type(epochs) is not int or epochs <= 0 for epochs in epoch_counts):
            issues.append(
                DecisionIssue(
                    DecisionStatus.FAIL,
                    "runtime.epochs",
                    (
                        "selection_split=test requires effective runtime.epochs to be a positive integer for "
                        "every trial so each trial has an epoch checkpoint opportunity."
                    ),
                    None,
                    {"effective_values": epoch_counts},
                )
            )
        if decisions["label_name"].value in {"ahi", "arousal"} and any(
            type(interval) is not int or interval != 1 for interval in validation_intervals
        ):
            issues.append(
                DecisionIssue(
                    DecisionStatus.FAIL,
                    "runtime.check_val_every_n_epoch",
                    (
                        "selection_split=test requires effective runtime.check_val_every_n_epoch=1 for ahi/arousal "
                        "so every tested checkpoint contains validation-fitted thresholds."
                    ),
                    None,
                    {"effective_values": validation_intervals},
                )
            )
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
    if selection_split == "test" and test_after_fit is False:
        issues.append(
            DecisionIssue(
                DecisionStatus.NEEDS_USER_INPUT,
                "test_after_fit",
                (
                    "selection_split=test requires test_after_fit=true so every trial produces the frozen "
                    "selection metric."
                ),
                "Should test_after_fit be true for test-selected hyper-parameter tuning?",
                {"evaluation_policy": evaluation},
            )
        )
    objective_metric = str(adaptive.get("objective_metric") or "test_auroc")
    if adaptive.get("enabled") is True and objective_metric.startswith("test_") and test_after_fit is False:
        issues.append(
            DecisionIssue(
                DecisionStatus.NEEDS_USER_INPUT,
                "test_after_fit",
                "A test-metric adaptive objective requires test_after_fit=true so trials produce objective evidence.",
                "Should test_after_fit be true for this adaptive test objective?",
                {"objective_metric": objective_metric, "evaluation_policy": evaluation},
            )
        )
    if evaluation.get("final_eval_split") == "test" and "require_manual_unlock_for_final_test" not in evaluation:
        issues.append(
            needs_issue("final_eval_unlock", "Final test evaluation requires manual unlock policy.", high_impact)
        )
    return issues


def hparam_tune_issues(
    recipe: dict,
    config_summary: dict | None,
    decisions: dict[str, ResolvedDecision],
    high_impact: dict[str, dict[str, Any]],
) -> list[DecisionIssue]:
    search = recipe.get("search") if isinstance(recipe.get("search"), dict) else {}
    profile_mode = "profile" in search
    execution = recipe.get("execution") if isinstance(recipe.get("execution"), dict) else {}
    runtime = recipe.get("runtime") if isinstance(recipe.get("runtime"), dict) else {}
    adaptive = recipe.get("adaptive") if isinstance(recipe.get("adaptive"), dict) else {}
    local_recipe = recipe.get("_local_recipe") if isinstance(recipe.get("_local_recipe"), dict) else recipe
    local_runtime = local_recipe.get("runtime") if isinstance(local_recipe.get("runtime"), dict) else {}

    issues = hparam_recipe_contract_issues(recipe, source_layer="effective")
    issues.extend(_hparam_config_issues(recipe, config_summary, decisions, high_impact))
    issues.extend(hparam_search_issues(search, profile_mode=profile_mode, high_impact=high_impact))
    issues.extend(
        _hparam_execution_issues(
            execution,
            runtime,
            local_runtime=local_runtime,
            variant=str(recipe.get("variant") or ""),
        )
    )
    if not (profile_mode and adaptive.get("enabled") is True):
        issues.extend(_hparam_adaptive_issues(adaptive))
    issues.extend(_hparam_search_budget_issues(search, profile_mode=profile_mode, high_impact=high_impact))
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
                "execution.runtime_commit must be a full 40- or 64-character Git object ID when set.",
                None,
                {"runtime_commit": runtime_commit},
            )
        )
    manager_runtime = (
        target == "local" and workdir in (None, "", str(REPO_ROOT)) and execution.get("conda_env") in (None, "")
    )
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
    for field in ("enabled", "test_feedback_for_selection"):
        if field in adaptive and type(adaptive[field]) is not bool:
            issues.append(
                DecisionIssue(
                    DecisionStatus.FAIL,
                    f"adaptive.{field}",
                    f"adaptive.{field} must be a boolean.",
                    None,
                    {field: adaptive.get(field)},
                )
            )
    replacement = adaptive.get("replacement") if isinstance(adaptive.get("replacement"), dict) else {}
    for field in ("enabled", "allow_running_stop"):
        if field in replacement and type(replacement[field]) is not bool:
            issues.append(
                DecisionIssue(
                    DecisionStatus.FAIL,
                    f"adaptive.replacement.{field}",
                    f"adaptive.replacement.{field} must be a boolean.",
                    None,
                    {field: replacement.get(field)},
                )
            )
    for field in ("grace_epochs", "grace_minutes", "kill_margin"):
        value = replacement.get(field)
        if field in replacement and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or value < 0
        ):
            issues.append(
                DecisionIssue(
                    DecisionStatus.FAIL,
                    f"adaptive.replacement.{field}",
                    f"adaptive.replacement.{field} must be a finite non-negative number.",
                    None,
                    {field: value},
                )
            )
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
    objective_mode = adaptive.get("objective_mode", "max")
    if not isinstance(objective_mode, str) or objective_mode not in {"max", "min"}:
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
        if type(adaptive[adaptive_field]) is not int or adaptive[adaptive_field] <= 0:
            issues.append(
                DecisionIssue(
                    DecisionStatus.FAIL,
                    f"adaptive.{adaptive_field}",
                    f"adaptive.{adaptive_field} must be a positive integer.",
                    None,
                    {adaptive_field: adaptive.get(adaptive_field)},
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
