from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import subprocess
from typing import Any

from .models import REPO_ROOT, SUPPORTED_VARIANTS, task_requires_variant

_EXPLICIT_HIGH_IMPACT_SOURCES = {"explicit_user", "explicit_cli", "explicit_recipe", "explicit_config"}


class DecisionStatus(str, Enum):
    PASS = "PASS"
    WARN = "WARN"
    NEEDS_USER_INPUT = "NEEDS_USER_INPUT"
    FAIL = "FAIL"


@dataclass
class DecisionIssue:
    status: DecisionStatus
    field: str
    message: str
    question: str | None = None
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass
class DecisionReport:
    status: DecisionStatus
    issues: list[DecisionIssue] = field(default_factory=list)
    decisions: dict[str, "ResolvedDecision"] = field(default_factory=dict)

    @property
    def exit_code(self) -> int:
        if any(issue.status == DecisionStatus.FAIL for issue in self.issues):
            return 1
        if any(issue.status == DecisionStatus.NEEDS_USER_INPUT for issue in self.issues):
            return 2
        return 0

    def blocking_issues(self) -> list[DecisionIssue]:
        return [
            issue for issue in self.issues if issue.status in {DecisionStatus.NEEDS_USER_INPUT, DecisionStatus.FAIL}
        ]


@dataclass
class ResolvedDecision:
    field: str
    value: Any
    source: str
    confidence: str
    evidence: dict[str, Any] = field(default_factory=dict)


def merge_status(issues: list[DecisionIssue]) -> DecisionStatus:
    if any(i.status == DecisionStatus.FAIL for i in issues):
        return DecisionStatus.FAIL
    if any(i.status == DecisionStatus.NEEDS_USER_INPUT for i in issues):
        return DecisionStatus.NEEDS_USER_INPUT
    if any(i.status == DecisionStatus.WARN for i in issues):
        return DecisionStatus.WARN
    return DecisionStatus.PASS


def evaluate_consultation_gates(
    task: str | None,
    recipe: dict | None,
    config_summary: dict | None,
    cli_args: dict | None,
    policy: dict,
    approved_defaults: dict,
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
        approved_defaults,
        task_override=task,
    )
    task_value = task_decision.value
    decisions["task"] = task_decision
    if task_value in (None, ""):
        issues.append(_needs("task", "Task is missing.", high_impact))
        return DecisionReport(status=merge_status(issues), issues=issues, decisions=decisions)
    if task_value not in supported_tasks:
        issues.append(
            DecisionIssue(
                DecisionStatus.FAIL,
                "task",
                f"Unsupported task: {task_value}",
                _question(high_impact, "task"),
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
                    "Which variant should this task use: sleep2vec, sleep2vec2, or sleep2expert?",
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

    for decision_field, rule in high_impact.items():
        if task_value not in rule.get("required_for_tasks", []):
            continue
        if decision_field == "task":
            continue
        decision = _resolve_decision(
            decision_field, recipe, config_summary, cli_args, user_decisions, approved_defaults
        )
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
        decisions[optional_field] = _resolve_decision(
            optional_field, recipe, config_summary, cli_args, user_decisions, approved_defaults
        )

    if str(task_value) == "hparam_tune":
        issues.extend(_base_finetune_issues(recipe, config_summary, cli_args, policy, approved_defaults))
    issues.extend(_task_specific_issues(str(task_value), recipe, config_summary, decisions, high_impact))
    issues.extend(_path_issues(str(task_value), recipe, config_summary, decisions))
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


def _question(high_impact: dict[str, dict[str, Any]], field: str) -> str | None:
    return high_impact.get(field, {}).get("question")


def _needs(
    field: str,
    message: str,
    high_impact: dict[str, dict[str, Any]],
    evidence: dict | None = None,
) -> DecisionIssue:
    return DecisionIssue(
        DecisionStatus.NEEDS_USER_INPUT,
        field,
        message,
        _question(high_impact, field),
        evidence or {},
    )


def _resolve_decision(
    field: str,
    recipe: dict,
    config_summary: dict | None,
    cli_args: dict,
    user_decisions: dict,
    approved_defaults: dict,
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

    defaults = approved_defaults.get("approved_defaults", {}) if isinstance(approved_defaults, dict) else {}
    if field in defaults:
        value = defaults[field].get("value") if isinstance(defaults[field], dict) else defaults[field]
        return ResolvedDecision(field, value, "approved_default", "medium", {"approved_default": defaults[field]})

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
        "hparam_budget": search.get("max_trials", _MISSING),
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
    issues: list[DecisionIssue] = []
    evaluation = recipe.get("evaluation_policy") if isinstance(recipe.get("evaluation_policy"), dict) else {}
    inputs = recipe.get("inputs") if isinstance(recipe.get("inputs"), dict) else {}
    preset = recipe.get("preset") if isinstance(recipe.get("preset"), dict) else {}
    search = recipe.get("search") if isinstance(recipe.get("search"), dict) else {}
    execution = recipe.get("execution") if isinstance(recipe.get("execution"), dict) else {}
    adaptive = recipe.get("adaptive") if isinstance(recipe.get("adaptive"), dict) else {}

    if task == "sleep2stat":
        if not inputs.get("config"):
            issues.append(_needs("config", "sleep2stat requires inputs.config.", high_impact))
            return issues
        if not config_summary or not config_summary.get("is_sleep2stat"):
            issues.append(
                DecisionIssue(
                    DecisionStatus.FAIL,
                    "config",
                    "task=sleep2stat requires a sleep2stat config.",
                    None,
                    {"config_summary": config_summary},
                )
            )
            return issues
        for message in config_summary.get("blocking_issues", []):
            issues.append(
                DecisionIssue(
                    DecisionStatus.NEEDS_USER_INPUT,
                    "sleep2stat_config",
                    message,
                    "Please fix the sleep2stat config before the agent generates commands.",
                    {"config_path": config_summary.get("config_path")},
                )
            )
        sleep2stat = config_summary.get("sleep2stat") or {}
        cfg_run = sleep2stat.get("run") or {}
        cfg_data = sleep2stat.get("data") or {}
        recipe_run_dir = (recipe.get("artifacts") if isinstance(recipe.get("artifacts"), dict) else {}).get("run_dir")
        config_run_dir = cfg_run.get("output_dir")
        if recipe_run_dir and config_run_dir and str(recipe_run_dir) != str(config_run_dir):
            issues.append(
                DecisionIssue(
                    DecisionStatus.NEEDS_USER_INPUT,
                    "artifacts.run_dir",
                    (
                        "Recipe artifacts.run_dir differs from sleep2stat config run.output_dir. "
                        "The sleep2stat CLI uses config run.output_dir, so commands would target the wrong directory."
                    ),
                    (
                        "Should artifacts.run_dir be changed to match config run.output_dir, or should the "
                        "sleep2stat config run.output_dir be changed?"
                    ),
                    {"recipe": recipe_run_dir, "config": config_run_dir},
                )
            )
        overwrite_decision = decisions.get("overwrite_policy")
        if cfg_run.get("overwrite") is True and overwrite_decision is not None and overwrite_decision.value is False:
            issues.append(
                DecisionIssue(
                    DecisionStatus.NEEDS_USER_INPUT,
                    "overwrite_policy",
                    "sleep2stat config run.overwrite=true conflicts with overwrite_policy=false.",
                    "Should config run.overwrite be false, or should overwrite_policy be changed to true?",
                    {"config_run_overwrite": cfg_run.get("overwrite"), "overwrite_policy": overwrite_decision.value},
                )
            )
        effective_split = _as_list(inputs.get("split") or cfg_data.get("split"))
        if not effective_split:
            issues.append(
                DecisionIssue(
                    DecisionStatus.NEEDS_USER_INPUT,
                    "sleep2stat_split_policy",
                    "sleep2stat split is not explicit in recipe or config.",
                    "Which split(s) should sleep2stat process?",
                    {"recipe_split": inputs.get("split"), "config_split": cfg_data.get("split")},
                )
            )
        external_test_locked = evaluation.get("external_test_locked")
        external_test_decision = decisions.get("external_test_locked")
        if external_test_decision is not None and external_test_decision.value not in (None, ""):
            external_test_locked = external_test_decision.value
        if "test" in {str(value) for value in effective_split} and external_test_locked is not True:
            issues.append(
                DecisionIssue(
                    DecisionStatus.NEEDS_USER_INPUT,
                    "external_test_locked",
                    "sleep2stat is configured for test split, but external_test_locked is not explicitly true.",
                    "Is this test split external/locked, and should outputs be descriptive-only?",
                    {"effective_split": effective_split, "external_test_locked": external_test_locked},
                )
            )
        for message in config_summary.get("agent_risk_issues", []):
            issues.append(
                DecisionIssue(
                    DecisionStatus.NEEDS_USER_INPUT,
                    "sleep2stat_config",
                    message,
                    "Please provide a concrete path, adjust path context, or disable the analyzer.",
                    {"config_path": config_summary.get("config_path")},
                )
            )

    if task == "preset_prepare":
        for input_field, value in {
            "index": inputs.get("index"),
            "dataset_name": inputs.get("dataset_name"),
            "split": preset.get("split"),
            "n_tokens": preset.get("n_tokens"),
            "allow_missing_channels": preset.get("allow_missing_channels"),
        }.items():
            if value in (None, "", []):
                issues.append(
                    _needs(
                        input_field,
                        f"{input_field} is required for preset preparation.",
                        high_impact,
                        {"recipe": value},
                    )
                )
        if preset.get("allow_missing_channels") is True and preset.get("min_channels") is None:
            issues.append(
                _needs("min_channels", "min_channels is required when missing channels are allowed.", high_impact)
            )

    if task == "finetune":
        if not inputs.get("config"):
            issues.append(_needs("config", "Config path is required for finetune.", high_impact))
        if "test_after_fit" not in evaluation and not _has_explicit_decision(decisions, "test_after_fit"):
            issues.append(
                DecisionIssue(
                    DecisionStatus.NEEDS_USER_INPUT,
                    "test_after_fit",
                    "test_after_fit policy is required for finetune command generation.",
                    "Should test evaluation run after fit for this task?",
                    {"evaluation_policy": evaluation},
                )
            )
        if "external_test_locked" not in evaluation and not _has_explicit_decision(decisions, "external_test_locked"):
            issues.append(
                _needs("external_test_locked", "external_test_locked must be explicit for finetune.", high_impact)
            )
        data = config_summary.get("data", {}) if config_summary else {}
        if config_summary and config_summary.get("data_backend") == "npz":
            if not data.get("finetune_data_index") and not data.get("finetune_preset_path"):
                issues.append(
                    DecisionIssue(
                        DecisionStatus.NEEDS_USER_INPUT,
                        "data_input",
                        "NPZ finetune requires finetune_preset_path or finetune_data_index.",
                        "Which preset or index should this run use?",
                        {"config": data},
                    )
                )
        if evaluation.get("external_test_locked") is True and evaluation.get("test_after_fit") is True:
            issues.append(
                DecisionIssue(
                    DecisionStatus.NEEDS_USER_INPUT,
                    "test_after_fit",
                    "test_after_fit=true would evaluate test while external_test_locked=true.",
                    "Should test evaluation be disabled during model selection?",
                    {"evaluation_policy": evaluation},
                )
            )

    if task == "hparam_tune":
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
            base_recipe = recipe.get("_base_recipe") if isinstance(recipe.get("_base_recipe"), dict) else {}
            base_inputs = base_recipe.get("inputs") if isinstance(base_recipe.get("inputs"), dict) else {}
            recipe_inputs = recipe.get("inputs") if isinstance(recipe.get("inputs"), dict) else {}
            base_config = base_inputs.get("config") or recipe_inputs.get("config")
            if base_config:
                issues.append(
                    DecisionIssue(
                        DecisionStatus.FAIL,
                        "config",
                        (
                            "Hparam plan generation needs local config YAML content; remote path validation may be "
                            "deferred, but YAML overrides cannot be generated from an unreadable config."
                        ),
                        None,
                        {"config": base_config},
                    )
                )
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
                        _question(high_impact, decision_field)
                        or f"What should {path} be for this hyper-parameter tuning task?",
                        {"local_recipe": "missing"},
                    )
                )
        if not recipe.get("base_recipe"):
            issues.append(_needs("base_recipe", "base_recipe is required for hyper-parameter tuning.", high_impact))
        if not search.get("method"):
            issues.append(_needs("search_method", "search.method is required.", high_impact))
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
            issues.append(_needs("hparam_search_space", "search.parameters is required.", high_impact))
        else:
            issues.extend(_hparam_search_parameter_issues(search.get("parameters")))
        issues.extend(_hparam_execution_issues(execution))
        issues.extend(_hparam_adaptive_issues(adaptive))
        max_trials = search.get("max_trials")
        if max_trials in (None, ""):
            issues.append(_needs("hparam_budget", "search.max_trials is required.", high_impact))
        else:
            try:
                if int(max_trials) <= 0:
                    raise ValueError
            except (TypeError, ValueError):
                issues.append(
                    DecisionIssue(
                        DecisionStatus.FAIL,
                        "hparam_budget",
                        "search.max_trials must be a positive integer.",
                        None,
                        {"max_trials": max_trials},
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
            issues.append(_needs("external_test_locked", "external_test_locked must be explicit.", high_impact))
        test_after_fit_decision = decisions.get("test_after_fit")
        test_after_fit = evaluation.get(
            "test_after_fit",
            test_after_fit_decision.value if test_after_fit_decision else None,
        )
        if test_after_fit is True:
            issues.append(
                DecisionIssue(
                    DecisionStatus.NEEDS_USER_INPUT,
                    "test_after_fit",
                    "Trial commands would evaluate test data.",
                    "Should test_after_fit be false during hyper-parameter tuning?",
                    {"evaluation_policy": evaluation},
                )
            )
        if evaluation.get("final_eval_split") == "test" and "require_manual_unlock_for_final_test" not in evaluation:
            issues.append(
                _needs("final_eval_unlock", "Final test evaluation requires manual unlock policy.", high_impact)
            )
        finetune_task = config_summary.get("finetune", {}).get("task", {}) if config_summary else {}
        if local_evaluation.get("selection_metric") and finetune_task.get("monitor"):
            if local_evaluation["selection_metric"] != finetune_task["monitor"]:
                issues.append(
                    DecisionIssue(
                        DecisionStatus.NEEDS_USER_INPUT,
                        "selection_metric",
                        "Hparam selection_metric differs from config finetune.task.monitor.",
                        _question(high_impact, "selection_metric"),
                        {"recipe": local_evaluation["selection_metric"], "config": finetune_task["monitor"]},
                    )
                )
        if local_evaluation.get("selection_mode") and finetune_task.get("monitor_mod"):
            if local_evaluation["selection_mode"] != finetune_task["monitor_mod"]:
                issues.append(
                    DecisionIssue(
                        DecisionStatus.NEEDS_USER_INPUT,
                        "selection_mode",
                        "Hparam selection_mode differs from config finetune.task.monitor_mod.",
                        _question(high_impact, "selection_mode"),
                        {"recipe": local_evaluation["selection_mode"], "config": finetune_task["monitor_mod"]},
                    )
                )

    if task in {"infer", "evaluate"}:
        if inputs.get("eval_split") == "test":
            final_eval_unlock = decisions.get("final_eval_unlock")
            unlocked = evaluation.get("final_test_unlocked") is True or (
                final_eval_unlock is not None and final_eval_unlock.value is True
            )
            if not unlocked:
                issues.append(
                    _needs("final_eval_unlock", "Test evaluation requires explicit final unlock.", high_impact)
                )
        if not inputs.get("eval_split"):
            issues.append(
                DecisionIssue(
                    DecisionStatus.NEEDS_USER_INPUT,
                    "eval_split",
                    "eval_split is required for inference/evaluation.",
                    "Which split should be evaluated?",
                    {"inputs": inputs},
                )
            )
    return issues


def _hparam_execution_issues(execution: dict[str, Any]) -> list[DecisionIssue]:
    issues: list[DecisionIssue] = []
    if not execution:
        return issues
    target = execution.get("target", "local")
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
    if "max_concurrent" in execution:
        try:
            if int(execution["max_concurrent"]) <= 0:
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
    return issues


def _hparam_adaptive_issues(adaptive: dict[str, Any]) -> list[DecisionIssue]:
    issues: list[DecisionIssue] = []
    if not adaptive:
        return issues
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
    for adaptive_field in ("max_rounds", "max_trials_total", "round_size", "poll_seconds"):
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


def _base_finetune_issues(
    recipe: dict,
    config_summary: dict | None,
    cli_args: dict,
    policy: dict,
    approved_defaults: dict,
) -> list[DecisionIssue]:
    base_recipe = recipe.get("_base_recipe") if isinstance(recipe.get("_base_recipe"), dict) else None
    if not base_recipe:
        return []
    local_recipe = recipe.get("_local_recipe") if isinstance(recipe.get("_local_recipe"), dict) else recipe
    base_gate = dict(base_recipe)
    base_gate["task"] = "finetune"
    if recipe.get("variant") and not base_gate.get("variant"):
        base_gate["variant"] = recipe.get("variant")

    local_evaluation = (
        local_recipe.get("evaluation_policy") if isinstance(local_recipe.get("evaluation_policy"), dict) else {}
    )
    base_evaluation = dict(base_gate.get("evaluation_policy") or {})
    for evaluation_field in (
        "selection_metric",
        "selection_mode",
        "selection_split",
        "external_test_locked",
        "test_after_fit",
        "final_eval_split",
        "require_manual_unlock_for_final_test",
    ):
        if evaluation_field not in base_evaluation and evaluation_field in local_evaluation:
            base_evaluation[evaluation_field] = local_evaluation[evaluation_field]
    base_gate["evaluation_policy"] = base_evaluation

    local_decisions = local_recipe.get("decisions") if isinstance(local_recipe.get("decisions"), dict) else {}
    base_decisions = dict(base_gate.get("decisions") or {})
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
        "finetune", base_gate, config_summary, base_cli_args, policy, approved_defaults
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


def _has_explicit_decision(decisions: dict[str, ResolvedDecision], field: str) -> bool:
    decision = decisions.get(field)
    return decision is not None and decision.source in {
        "explicit_user",
        "explicit_cli",
        "explicit_recipe",
        "explicit_config",
    }


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


def _path_issues(
    task: str,
    recipe: dict,
    config_summary: dict | None,
    decisions: dict[str, ResolvedDecision],
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
        ckpt_decision = decisions.get("ckpt_path")
        ckpt_path = (
            ckpt_decision.value
            if ckpt_decision is not None and ckpt_decision.value not in (None, "")
            else inputs.get("ckpt_path")
        )
        if ckpt_path not in (None, "", "ASK_USER"):
            required_paths.append(("ckpt_path", ckpt_path))

    for path_field, raw_path in required_paths:
        issue = _validate_input_path(recipe, path_field, raw_path, configured=False)
        if issue is not None:
            issues.append(issue)

    if task in {"finetune", "hparam_tune"} and config_summary and config_summary.get("data_backend") == "npz":
        data = config_summary.get("data", {})
        for data_field in ("finetune_preset_path", "finetune_data_index"):
            value = data.get(data_field)
            if value:
                issue = _validate_input_path(recipe, data_field, value, configured=True)
                if issue is not None:
                    issues.append(issue)
    if task == "sleep2stat" and config_summary and config_summary.get("is_sleep2stat"):
        sleep2stat = config_summary.get("sleep2stat") or {}
        data = sleep2stat.get("data") or {}
        for data_field in ("index", "kaldi_data_root", "kaldi_manifest"):
            value = data.get(data_field)
            if value:
                check_value = value
                if data_field == "kaldi_manifest":
                    root = data.get("kaldi_data_root")
                    manifest_path = Path(str(value)).expanduser()
                    if root and not manifest_path.is_absolute():
                        check_value = Path(str(root)).expanduser() / manifest_path
                issue = _validate_input_path(recipe, f"sleep2stat.data.{data_field}", check_value, configured=True)
                if issue is not None:
                    issues.append(issue)
        for analyzer in sleep2stat.get("analyzers", []):
            if analyzer.get("enabled") is False:
                continue
            for analyzer_field in ("config", "ckpt_path"):
                value = analyzer.get(analyzer_field)
                if not value or _looks_like_placeholder_path(value):
                    continue
                issue = _validate_input_path(
                    recipe,
                    f"sleep2stat.analyzer.{analyzer.get('name')}.{analyzer_field}",
                    value,
                    configured=True,
                )
                if issue is not None:
                    issues.append(issue)
    return issues


def _validate_input_path(recipe: dict, field: str, raw_path: Any, *, configured: bool) -> DecisionIssue | None:
    context = _path_context(recipe, raw_path)
    validation = _path_validation(recipe, context)
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
        result = subprocess.run(["ssh", str(host), f"test -e {_sh(raw_path)}"], text=True, capture_output=True)
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


def _path_label(configured: bool) -> str:
    return "Configured input" if configured else "Required input"


def _path_context(recipe: dict, raw_path: Any) -> str:
    execution = _execution(recipe)
    explicit = execution.get("path_context")
    if explicit:
        return str(explicit)
    if execution.get("target") == "ssh" and Path(str(raw_path)).expanduser().is_absolute():
        return "remote"
    return "local"


def _path_validation(recipe: dict, context: str) -> str:
    explicit = _execution(recipe).get("path_validation")
    if explicit:
        return str(explicit)
    return "defer" if context == "remote" else "local"


def _execution(recipe: dict) -> dict[str, Any]:
    return recipe.get("execution") if isinstance(recipe.get("execution"), dict) else {}


def _sh(value: Any) -> str:
    import shlex

    return shlex.quote(str(value))


def _as_list(value: Any) -> list[Any]:
    if value in (None, "", []):
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _looks_like_placeholder_path(value: Any) -> bool:
    text = str(value).strip()
    lowered = text.lower()
    return (
        lowered in {"", "ask_user", "none", "null", "todo", "tbd", "placeholder"}
        or text.startswith("/path/to")
        or text.startswith("<")
        or "ASK_USER" in text
    )


def _output_paths_missing(recipe: dict) -> bool:
    artifacts = recipe.get("artifacts") if isinstance(recipe.get("artifacts"), dict) else {}
    return not bool(artifacts)
