from __future__ import annotations

from typing import Any

from . import decision_hparam as hparam_rules, decision_paths as paths, decision_rules as rules
from .decision_models import (
    DecisionIssue,
    DecisionReport,
    DecisionStatus,
    ResolvedDecision,
    merge_status,
    needs_issue,
    question_for,
)
from .experiment_workspace import experiment_metadata_issues
from .models import SUPPORTED_VARIANTS, task_requires_variant

__all__ = [
    "DecisionIssue",
    "DecisionReport",
    "DecisionStatus",
    "ResolvedDecision",
    "evaluate_consultation_gates",
    "merge_status",
]

_EXPLICIT_HIGH_IMPACT_SOURCES = {"explicit_user", "explicit_cli", "explicit_recipe", "explicit_config"}
_RUNTIME_FIELDS = {
    "accelerator",
    "accumulate_grad_batches",
    "avg_ckpt_dir",
    "avg_ckpts",
    "batch_size",
    "check_val_every_n_epoch",
    "ckpt_every_n_epochs",
    "data_backend",
    "device",
    "devices",
    "dry_run",
    "epochs",
    "gradient_clip_val",
    "limit_records",
    "lr",
    "num_workers",
    "patience",
    "plot_adjust_covariates",
    "plot_cohort_after_run",
    "plot_group_column",
    "plot_stage_source",
    "precision",
    "seed",
    "summarize_after_run",
    "wandb_mode",
    "warmup_steps",
    "weight_decay",
}


def evaluate_consultation_gates(
    task: str | None,
    recipe: dict | None,
    config_summary: dict | None,
    cli_args: dict | None,
    policy: dict,
    *,
    require_experiment: bool = True,
) -> DecisionReport:
    recipe = recipe or {}
    cli_args = cli_args or {}
    user_decisions = cli_args.get("user_decisions") or {}
    issues: list[DecisionIssue] = []
    decisions: dict[str, ResolvedDecision] = {}
    high_impact = _high_impact_by_id(policy)
    supported_tasks = _supported_tasks(high_impact)
    task_decision = _resolve_decision(
        "task",
        recipe,
        config_summary,
        cli_args,
        user_decisions,
        task_override=task,
    )
    task_value = task_decision.value
    decisions["task"] = task_decision
    if task_value in (None, ""):
        issues.append(needs_issue("task", "Task is missing.", high_impact))
        return DecisionReport(status=merge_status(issues), issues=issues, decisions=decisions)
    if task_value not in supported_tasks:
        issues.append(
            DecisionIssue(
                DecisionStatus.FAIL,
                "task",
                f"Unsupported task: {task_value}",
                question_for(high_impact, "task"),
                {"supported_tasks": sorted(supported_tasks)},
            )
        )
        return DecisionReport(status=merge_status(issues), issues=issues, decisions=decisions)
    variant = recipe.get("variant")
    if task_requires_variant(str(task_value)):
        if variant not in SUPPORTED_VARIANTS:
            issues.append(
                DecisionIssue(
                    DecisionStatus.NEEDS_USER_INPUT,
                    "variant",
                    "Recipe variant is missing or unsupported.",
                    "Which variant should this task use: sleep2vec, sleep2vec2, sleep2expert, or sex_age_baseline?",
                    {"variant": variant, "allowed_values": list(SUPPORTED_VARIANTS)},
                )
            )
    elif variant not in (None, ""):
        issues.append(
            DecisionIssue(
                DecisionStatus.FAIL,
                "variant",
                "task=sleep2stat must omit variant or set it to null; sleep2stat is not a model variant.",
                None,
                {"variant": variant},
            )
        )
    if task_value == "preset_prepare" and variant == "sex_age_baseline":
        issues.append(
            DecisionIssue(
                DecisionStatus.FAIL,
                "variant",
                "sex_age_baseline does not support preset_prepare.",
                None,
                {"variant": variant, "task": task_value},
            )
        )

    runtime = recipe.get("runtime") if isinstance(recipe.get("runtime"), dict) else {}
    for field in sorted(set(runtime) - _RUNTIME_FIELDS):
        issues.append(
            DecisionIssue(
                DecisionStatus.FAIL,
                f"runtime.{field}",
                f"Unknown runtime field: {field}.",
                None,
                {field: runtime[field], "preflight_before_workspace": True},
            )
        )

    if require_experiment:
        metadata_recipe = recipe
        if task_value == "hparam_tune" and isinstance(recipe.get("_local_recipe"), dict):
            metadata_recipe = recipe["_local_recipe"]
        for issue in experiment_metadata_issues(metadata_recipe):
            issues.append(
                DecisionIssue(
                    DecisionStatus(issue["status"]),
                    issue["field"],
                    issue["message"],
                    issue.get("question"),
                    issue.get("evidence", {}),
                )
            )

    for decision_field, rule in high_impact.items():
        if task_value not in rule.get("required_for_tasks", []):
            continue
        if decision_field == "task":
            continue
        decision = _resolve_decision(decision_field, recipe, config_summary, cli_args, user_decisions)
        decisions[decision_field] = decision
        if decision.value == "ASK_USER":
            issues.append(
                DecisionIssue(
                    DecisionStatus.NEEDS_USER_INPUT,
                    decision_field,
                    f"{decision_field} is marked ASK_USER.",
                    decision.evidence.get("question") or rule.get("question"),
                    decision.evidence,
                )
            )
            continue
        if decision.source not in _EXPLICIT_HIGH_IMPACT_SOURCES:
            issues.append(
                DecisionIssue(
                    DecisionStatus.NEEDS_USER_INPUT,
                    decision_field,
                    f"{decision_field} is not explicitly resolved.",
                    rule.get("question"),
                    decision.evidence,
                )
            )
            continue
        allowed_values = rule.get("allowed_values")
        if allowed_values and decision.value not in allowed_values:
            issues.append(
                DecisionIssue(
                    DecisionStatus.FAIL,
                    decision_field,
                    f"{decision_field} must be one of {allowed_values}.",
                    rule.get("question"),
                    {"value": decision.value},
                )
            )

    for optional_field in (
        "test_after_fit",
        "ckpt_path",
        "config",
        "final_eval_config_path",
        "eval_split",
        "external_test_locked",
    ):
        decisions[optional_field] = _resolve_decision(optional_field, recipe, config_summary, cli_args, user_decisions)

    if str(task_value) == "hparam_tune":
        issues.extend(_base_finetune_issues(recipe, config_summary, cli_args, policy))
    issues.extend(_task_specific_issues(str(task_value), recipe, config_summary, decisions, high_impact))
    issues.extend(paths.path_issues(str(task_value), recipe, config_summary))
    if _output_paths_missing(recipe):
        issues.append(
            DecisionIssue(
                DecisionStatus.WARN,
                "output_dir",
                "Output paths are absent and agent defaults may be used under artifacts/.",
                None,
                {},
            )
        )
    return DecisionReport(status=merge_status(issues), issues=issues, decisions=decisions)


def _high_impact_by_id(policy: dict) -> dict[str, dict[str, Any]]:
    return {
        str(item["id"]): item
        for item in policy.get("high_impact_fields", [])
        if isinstance(item, dict) and "id" in item
    }


def _supported_tasks(high_impact: dict[str, dict[str, Any]]) -> set[str]:
    tasks: set[str] = set()
    for rule in high_impact.values():
        tasks.update(rule.get("required_for_tasks", []))
    return tasks


def _resolve_decision(
    field: str,
    recipe: dict,
    config_summary: dict | None,
    cli_args: dict,
    user_decisions: dict,
    *,
    task_override: str | None = None,
) -> ResolvedDecision:
    if field in user_decisions:
        return _decision_from_mapping(field, user_decisions[field], "explicit_user")
    if task_override not in (None, "") and field == "task":
        return ResolvedDecision(field, task_override, "explicit_cli", "high", {"task": task_override})
    if field in cli_args and cli_args[field] not in (None, ""):
        return ResolvedDecision(field, cli_args[field], "explicit_cli", "high", {"cli": cli_args[field]})

    recipe_decisions = recipe.get("decisions") if isinstance(recipe.get("decisions"), dict) else {}
    if field in recipe_decisions:
        return _decision_from_mapping(field, recipe_decisions[field], "explicit_recipe")

    recipe_value = _recipe_field_value(field, recipe)
    if recipe_value is not _MISSING:
        return ResolvedDecision(field, recipe_value, "explicit_recipe", "high", {"recipe": recipe_value})

    config_value = _config_field_value(field, config_summary)
    if config_value is not _MISSING:
        source = "explicit_config"
        if field in {"selection_metric", "selection_mode"}:
            source = "explicit_config"
        return ResolvedDecision(field, config_value, source, "medium", {"config": config_value})

    return ResolvedDecision(
        field,
        None,
        "missing",
        "none",
        {"cli": "missing", "recipe": "missing", "config": "missing"},
    )


def _decision_from_mapping(field: str, raw: Any, fallback_source: str) -> ResolvedDecision:
    if isinstance(raw, dict):
        value = raw.get("value")
        source = raw.get("source") or fallback_source
        evidence = {key: value for key, value in raw.items() if key != "value"}
        return ResolvedDecision(field, value, source, "high", evidence)
    return ResolvedDecision(field, raw, fallback_source, "high", {"value": raw})


class _Missing:
    pass


_MISSING = _Missing()


def _recipe_field_value(field: str, recipe: dict) -> Any:
    inputs = recipe.get("inputs") if isinstance(recipe.get("inputs"), dict) else {}
    runtime = recipe.get("runtime") if isinstance(recipe.get("runtime"), dict) else {}
    evaluation = recipe.get("evaluation_policy") if isinstance(recipe.get("evaluation_policy"), dict) else {}
    artifacts = recipe.get("artifacts") if isinstance(recipe.get("artifacts"), dict) else {}
    preset = recipe.get("preset") if isinstance(recipe.get("preset"), dict) else {}
    search = recipe.get("search") if isinstance(recipe.get("search"), dict) else {}

    pretrained_value = inputs.get("pretrained_backbone_path", _MISSING)
    if pretrained_value is None:
        pretrained_value = _MISSING
    mapping = {
        "task": recipe.get("task", _MISSING),
        "label_name": inputs.get("label_name", _MISSING),
        "data_backend": inputs.get("data_backend", runtime.get("data_backend", _MISSING)),
        "train_val_test_policy": evaluation.get("selection_split", _MISSING),
        "external_test_locked": evaluation.get("external_test_locked", _MISSING),
        "selection_metric": evaluation.get("selection_metric", _MISSING),
        "selection_mode": evaluation.get("selection_mode", _MISSING),
        "pretrained_backbone_path": pretrained_value,
        "config": inputs.get("config", _MISSING),
        "ckpt_path": inputs.get("ckpt_path", _MISSING),
        "eval_split": inputs.get("eval_split", _MISSING),
        "final_eval_config_path": inputs.get("final_eval_config_path", _MISSING),
        "preset_regeneration": preset.get("regenerate", _MISSING),
        "overwrite_policy": artifacts.get("overwrite", preset.get("overwrite", _MISSING)),
        "required_channels": preset.get("required_channels", preset.get("channels", _MISSING)),
        "min_channels": preset.get("min_channels", _MISSING),
        "hparam_search_space": search.get("parameters", _MISSING),
        "hparam_budget": search.get("max_runs", _MISSING),
        "final_eval_unlock": evaluation.get("final_test_unlocked", _MISSING),
        "test_after_fit": evaluation.get("test_after_fit", _MISSING),
    }
    return mapping.get(field, _MISSING)


def _config_field_value(field: str, config_summary: dict | None) -> Any:
    if not config_summary:
        return _MISSING
    data = config_summary.get("data", {})
    finetune_task = config_summary.get("finetune", {}).get("task", {})
    preset_build = config_summary.get("preset_build", {})
    mapping = {
        "data_backend": config_summary.get("data_backend"),
        "selection_metric": finetune_task.get("monitor"),
        "selection_mode": finetune_task.get("monitor_mod"),
        "required_channels": preset_build.get("required_channels"),
        "min_channels": preset_build.get("min_channels"),
    }
    value = mapping.get(field, _MISSING)
    if (
        field == "data_backend"
        and value is None
        and (data.get("finetune_preset_path") or data.get("finetune_data_index"))
    ):
        return "npz"
    if value in (None, ""):
        return _MISSING
    return value


def _task_specific_issues(
    task: str,
    recipe: dict,
    config_summary: dict | None,
    decisions: dict[str, ResolvedDecision],
    high_impact: dict[str, dict[str, Any]],
) -> list[DecisionIssue]:
    if task == "sleep2stat":
        return rules.sleep2stat_issues(recipe, config_summary, high_impact)
    if task == "preset_prepare":
        return rules.preset_prepare_issues(recipe, config_summary, high_impact)
    if task == "finetune":
        return rules.finetune_task_issues(recipe, config_summary, decisions, high_impact)
    if task == "hparam_tune":
        return hparam_rules.hparam_tune_issues(recipe, config_summary, decisions, high_impact)
    if task in {"infer", "evaluate"}:
        return rules.infer_evaluate_issues(recipe, config_summary, decisions, high_impact)
    return []


def _base_finetune_issues(
    recipe: dict,
    config_summary: dict | None,
    cli_args: dict,
    policy: dict,
) -> list[DecisionIssue]:
    base_recipe = recipe.get("_base_recipe") if isinstance(recipe.get("_base_recipe"), dict) else None
    if not base_recipe:
        return []
    local_recipe = recipe.get("_local_recipe") if isinstance(recipe.get("_local_recipe"), dict) else recipe
    base_gate = {key: value for key, value in recipe.items() if not key.startswith("_")}
    base_gate["task"] = "finetune"

    local_decisions = local_recipe.get("decisions") if isinstance(local_recipe.get("decisions"), dict) else {}
    base_decisions = dict(base_recipe.get("decisions") or {})
    for decision_field, value in local_decisions.items():
        if decision_field != "task" and decision_field not in base_decisions:
            base_decisions[decision_field] = value
    base_gate["decisions"] = base_decisions

    base_cli_args = dict(cli_args)
    if isinstance(base_cli_args.get("user_decisions"), dict):
        base_cli_args["user_decisions"] = {
            field: value for field, value in base_cli_args["user_decisions"].items() if field != "task"
        }
    report = evaluate_consultation_gates(
        "finetune",
        base_gate,
        config_summary,
        base_cli_args,
        policy,
        require_experiment=False,
    )
    return [
        DecisionIssue(
            issue.status,
            f"base_finetune.{issue.field}",
            f"Base finetune readiness issue: {issue.message}",
            issue.question,
            issue.evidence,
        )
        for issue in report.blocking_issues()
    ]


def _output_paths_missing(recipe: dict) -> bool:
    artifacts = recipe.get("artifacts") if isinstance(recipe.get("artifacts"), dict) else {}
    return not bool(artifacts)
