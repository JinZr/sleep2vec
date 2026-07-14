from __future__ import annotations

from pathlib import Path
from typing import Any

from .decision_models import DecisionIssue, DecisionStatus, ResolvedDecision, needs_issue, question_for
from .decision_paths import multilabel_sidecar_issue

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


def hparam_tune_issues(
    recipe: dict,
    config_summary: dict | None,
    decisions: dict[str, ResolvedDecision],
    high_impact: dict[str, dict[str, Any]],
) -> list[DecisionIssue]:
    issues: list[DecisionIssue] = []
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
    for field in sorted(set(evaluation) - _HPARAM_EVALUATION_FIELDS):
        issues.append(
            DecisionIssue(
                DecisionStatus.FAIL,
                f"evaluation_policy.{field}",
                f"Unknown hparam evaluation_policy field: {field}.",
                None,
                {field: evaluation[field], "preflight_before_workspace": True},
            )
        )
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
    finetune_task = config_summary.get("finetune", {}).get("task", {}) if config_summary else {}
    if evaluation.get("selection_metric") and finetune_task.get("monitor"):
        if evaluation["selection_metric"] != finetune_task["monitor"]:
            issues.append(
                DecisionIssue(
                    DecisionStatus.NEEDS_USER_INPUT,
                    "selection_metric",
                    "Hparam selection_metric differs from config finetune.task.monitor.",
                    question_for(high_impact, "selection_metric"),
                    {"recipe": evaluation["selection_metric"], "config": finetune_task["monitor"]},
                )
            )
    if evaluation.get("selection_mode") and finetune_task.get("monitor_mod"):
        if evaluation["selection_mode"] != finetune_task["monitor_mod"]:
            issues.append(
                DecisionIssue(
                    DecisionStatus.NEEDS_USER_INPUT,
                    "selection_mode",
                    "Hparam selection_mode differs from config finetune.task.monitor_mod.",
                    question_for(high_impact, "selection_mode"),
                    {"recipe": evaluation["selection_mode"], "config": finetune_task["monitor_mod"]},
                )
            )
    return issues


def _hparam_execution_issues(execution: dict[str, Any], runtime: dict[str, Any]) -> list[DecisionIssue]:
    issues: list[DecisionIssue] = []
    legacy_fields = {"gpus_per_trial", "log_dir", "pid_dir"}
    for field in sorted(set(execution) - _HPARAM_EXECUTION_FIELDS - legacy_fields):
        issues.append(
            DecisionIssue(
                DecisionStatus.FAIL,
                f"execution.{field}",
                f"Unknown hparam execution field: {field}.",
                None,
                {field: execution[field], "preflight_before_workspace": True},
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
    devices_value = runtime.get("devices")
    if devices_value in (None, "", "ASK_USER"):
        devices = []
    elif isinstance(devices_value, (list, tuple)):
        devices = list(devices_value)
    else:
        devices = [devices_value]
    gpu_pool = execution.get("gpu_pool") if isinstance(execution.get("gpu_pool"), list) else None
    pool = list(gpu_pool) if gpu_pool else devices
    per_run = gpus_per_run if gpus_per_run is not None else len(devices) or 1
    if gpus_per_run is not None and not pool:
        issues.append(
            DecisionIssue(
                DecisionStatus.FAIL,
                "execution.gpus_per_run",
                "execution.gpus_per_run requires a non-empty execution.gpu_pool or runtime.devices.",
                None,
                {
                    "gpus_per_run": gpus_per_run,
                    "preflight_before_workspace": True,
                },
            )
        )
    elif pool and (gpus_per_run is not None or "gpus_per_run" not in execution):
        pool_field = "execution.gpu_pool" if gpu_pool else "runtime.devices"
        if len({str(item) for item in pool}) != len(pool):
            issues.append(
                DecisionIssue(
                    DecisionStatus.FAIL,
                    pool_field,
                    f"{pool_field} must not contain duplicate GPU identifiers.",
                    None,
                    {"gpu_pool": pool},
                )
            )
        elif per_run > len(pool):
            issues.append(
                DecisionIssue(
                    DecisionStatus.FAIL,
                    "execution.gpus_per_run",
                    "execution.gpus_per_run cannot exceed the effective GPU pool size.",
                    None,
                    {"gpus_per_run": per_run, "gpu_pool": pool},
                )
            )
        elif len(pool) % per_run != 0:
            issues.append(
                DecisionIssue(
                    DecisionStatus.FAIL,
                    "execution.gpus_per_run",
                    "The effective GPU pool must divide evenly into disjoint per-run GPU groups.",
                    None,
                    {"gpus_per_run": per_run, "gpu_pool": pool},
                )
            )
        elif max_concurrent is not None and max_concurrent > len(pool) // per_run:
            group_count = len(pool) // per_run
            issues.append(
                DecisionIssue(
                    DecisionStatus.WARN,
                    "execution.max_concurrent",
                    (
                        f"execution.max_concurrent={max_concurrent} exceeds the {group_count} available GPU "
                        "group(s); GPU oversubscription is explicitly enabled."
                    ),
                    None,
                    {"max_concurrent": max_concurrent, "gpu_group_count": group_count},
                )
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
        for env_name, field in {
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
    replacement = adaptive.get("replacement") if isinstance(adaptive.get("replacement"), dict) else {}
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
