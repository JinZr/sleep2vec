from __future__ import annotations

from typing import Any

from . import decision_paths as paths, plan_rendering as rendering, schema_map
from .adapters import SUPPORTED_TASKS, all_adapters, get_adapter
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
_DECISION_ENTRY_FIELDS = {"meaning", "question", "rationale", "source", "value"}


def consultation_contract_issues(
    task: str | None,
    recipe: dict,
    policy: dict,
    *,
    source_layer: str,
) -> list[DecisionIssue]:
    issues: list[DecisionIssue] = []
    runtime_value = recipe.get("runtime")
    if "runtime" in recipe:
        if not isinstance(runtime_value, dict):
            issues.append(_contract_issue("runtime", "runtime must be a mapping.", runtime_value, source_layer))
        else:
            allowed_runtime = _runtime_fields_for_task(task, recipe.get("variant"))
            for field in sorted(set(runtime_value) - allowed_runtime):
                issues.append(
                    _contract_issue(
                        f"runtime.{field}",
                        f"Unknown runtime field for task={task}: {field}.",
                        runtime_value[field],
                        source_layer,
                    )
                )
            if "avg_ckpts" in runtime_value and (
                type(runtime_value["avg_ckpts"]) is not int or runtime_value["avg_ckpts"] < 1
            ):
                issues.append(
                    _contract_issue(
                        "runtime.avg_ckpts",
                        "runtime.avg_ckpts must be a positive integer.",
                        runtime_value["avg_ckpts"],
                        source_layer,
                    )
                )

    decisions_value = recipe.get("decisions")
    if "decisions" not in recipe:
        return issues
    if not isinstance(decisions_value, dict):
        issues.append(_contract_issue("decisions", "decisions must be a mapping.", decisions_value, source_layer))
        return issues
    allowed_decisions = _decision_fields_for_task(task, policy)
    for field, value in decisions_value.items():
        if field not in allowed_decisions:
            issues.append(
                _contract_issue(
                    f"decisions.{field}",
                    f"Decision field is not supported for task={task}: {field}.",
                    value,
                    source_layer,
                )
            )
            continue
        if not isinstance(value, dict):
            continue
        for entry_field in sorted(set(value) - _DECISION_ENTRY_FIELDS):
            issues.append(
                _contract_issue(
                    f"decisions.{field}.{entry_field}",
                    f"Unknown decision entry field: {entry_field}.",
                    value[entry_field],
                    source_layer,
                )
            )
    return issues


def _runtime_fields_for_task(task: str | None, variant: Any) -> frozenset[str]:
    adapter = get_adapter(task)
    if adapter is not None:
        return adapter.runtime_fields(variant)
    if task is not None:
        return frozenset()
    union = rendering.FINETUNE_RUNTIME_FIELDS | rendering.INFER_RUNTIME_FIELDS
    for registered in all_adapters():
        union = union | registered.runtime_fields(variant)
    return union


def _decision_fields_for_task(task: str | None, policy: dict) -> set[str]:
    task_scope = {task}
    scope_adapter = get_adapter(task)
    if scope_adapter is not None and scope_adapter.base_task is not None:
        task_scope.add(scope_adapter.base_task)
    allowed = {
        str(item["id"])
        for item in policy.get("high_impact_fields", [])
        if isinstance(item, dict)
        and "id" in item
        and (task is None or task_scope.intersection(item.get("required_for_tasks", [])))
    }
    if task is None:
        for adapter in all_adapters():
            allowed |= adapter.extra_decision_fields
    elif scope_adapter is not None:
        allowed |= scope_adapter.extra_decision_fields
    return allowed


def _contract_issue(field: str, message: str, value: Any, source_layer: str) -> DecisionIssue:
    return DecisionIssue(
        DecisionStatus.FAIL,
        field,
        message,
        None,
        {"value": value, "source_layer": source_layer, "preflight_before_workspace": True},
    )


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
    policy_tasks = _policy_tasks(high_impact)
    unknown_policy_tasks = sorted(policy_tasks - SUPPORTED_TASKS)
    missing_policy_tasks = sorted(SUPPORTED_TASKS - policy_tasks)
    if unknown_policy_tasks or missing_policy_tasks:
        issues.append(
            DecisionIssue(
                DecisionStatus.FAIL,
                "consultation_policy",
                "Consultation policy task coverage differs from the registered adapters.",
                None,
                {"unknown_tasks": unknown_policy_tasks, "missing_tasks": missing_policy_tasks},
            )
        )
        return DecisionReport(status=merge_status(issues), issues=issues, decisions=decisions)
    supported_tasks = SUPPORTED_TASKS
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
    task_adapter = get_adapter(str(task_value))
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
                f"task={task_value} must omit variant or set it to null; {task_value} is not a model variant.",
                None,
                {"variant": variant},
            )
        )
    if task_adapter is not None and isinstance(variant, str) and variant in task_adapter.unsupported_variants:
        issues.append(
            DecisionIssue(
                DecisionStatus.FAIL,
                "variant",
                f"{variant} does not support {task_value}.",
                None,
                {"variant": variant, "task": task_value},
            )
        )

    issues.extend(consultation_contract_issues(str(task_value), recipe, policy, source_layer="effective"))

    if require_experiment:
        metadata_recipe = recipe
        if (
            task_adapter is not None
            and task_adapter.base_task is not None
            and isinstance(recipe.get("_local_recipe"), dict)
        ):
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

    external_test_locked = decisions["external_test_locked"]
    unresolved_recipe_value = (
        external_test_locked.source != "missing"
        and external_test_locked.value in (None, "")
        and "recipe" in external_test_locked.evidence
    )
    if external_test_locked.value == "ASK_USER" or unresolved_recipe_value:
        if not any(issue.field == "external_test_locked" for issue in issues):
            issues.append(
                needs_issue(
                    "external_test_locked",
                    "external_test_locked must be explicitly true or false.",
                    high_impact,
                    {
                        "value": external_test_locked.value,
                        "source": external_test_locked.source,
                        "preflight_before_workspace": True,
                    },
                )
            )
    elif external_test_locked.value not in (None, "") and type(external_test_locked.value) is not bool:
        issues.append(
            _contract_issue(
                "external_test_locked",
                "external_test_locked must be a YAML boolean.",
                external_test_locked.value,
                external_test_locked.source,
            )
        )

    if task_adapter is not None and task_adapter.base_task is not None:
        issues.extend(_base_task_issues(task_adapter.base_task, recipe, config_summary, cli_args, policy))
    issues.extend(_task_specific_issues(str(task_value), recipe, config_summary, decisions, high_impact))
    issues.extend(
        paths.path_issues(
            str(task_value),
            recipe,
            config_summary,
            required_input_paths=task_adapter.required_input_paths(recipe) if task_adapter else None,
            requires_survival_sidecars=task_adapter.requires_survival_sidecars if task_adapter else None,
            requires_multilabel_sidecars=task_adapter.requires_multilabel_sidecars if task_adapter else None,
            preset_path_recipe_field=task_adapter.preset_path_recipe_field if task_adapter else None,
            validates_dataset_paths=task_adapter.validates_dataset_paths if task_adapter else False,
            uses_finetune_config=task_adapter.uses_finetune_config if task_adapter else None,
        )
    )
    if task_adapter is not None:
        issues.extend(task_adapter.configured_input_issues(recipe, config_summary))
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


def _policy_tasks(high_impact: dict[str, dict[str, Any]]) -> set[str]:
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
    spec = schema_map.BASE_RECIPE_FIELDS.get(field)
    if spec is None:
        return _MISSING
    if field == "task":
        return recipe.get("task", _MISSING)
    value: Any = _MISSING
    for section, key in spec.read_path:
        owner = recipe.get(section) if isinstance(recipe.get(section), dict) else {}
        if key in owner:  # present-but-falsy values (e.g. overwrite=False) count as a hit
            value = owner[key]
            break
    if spec.none_is_missing and value is None:
        return _MISSING
    return value


def _config_field_value(field: str, config_summary: dict | None) -> Any:
    if not config_summary:
        return _MISSING
    spec = schema_map.CONFIG_FIELDS.get(field)
    if spec is None:
        return _MISSING
    node: Any = config_summary
    for part in spec.summary_path[:-1]:
        node = node.get(part, {})
    value = node.get(spec.summary_path[-1])
    if (
        field == "data_backend"
        and value is None
        and (
            config_summary.get("data", {}).get("finetune_preset_path")
            or config_summary.get("data", {}).get("finetune_data_index")
        )
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
    adapter = get_adapter(task)
    if adapter is not None:
        return adapter.task_issues(recipe, config_summary, decisions, high_impact)
    return []


def _base_task_issues(
    base_task: str,
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
    base_gate["task"] = base_task
    if isinstance(base_gate.get("runtime"), dict):
        base_runtime_fields = _runtime_fields_for_task(base_task, recipe.get("variant"))
        base_gate["runtime"] = {
            field: value for field, value in base_gate["runtime"].items() if field in base_runtime_fields
        }

    local_decisions = local_recipe.get("decisions") if isinstance(local_recipe.get("decisions"), dict) else {}
    base_decisions = dict(base_recipe.get("decisions") or {})
    base_decision_fields = _decision_fields_for_task(base_task, policy)
    for decision_field, value in local_decisions.items():
        if decision_field in base_decision_fields:
            base_decisions[decision_field] = value
    base_gate["decisions"] = base_decisions

    base_cli_args = dict(cli_args)
    if isinstance(base_cli_args.get("user_decisions"), dict):
        base_cli_args["user_decisions"] = {
            field: value for field, value in base_cli_args["user_decisions"].items() if field in base_decision_fields
        }
    report = evaluate_consultation_gates(
        base_task,
        base_gate,
        config_summary,
        base_cli_args,
        policy,
        require_experiment=False,
    )
    return [
        DecisionIssue(
            issue.status,
            f"base_{base_task}.{issue.field}",
            f"Base {base_task} readiness issue: {issue.message}",
            issue.question,
            issue.evidence,
        )
        for issue in report.blocking_issues()
    ]


def _output_paths_missing(recipe: dict) -> bool:
    artifacts = recipe.get("artifacts") if isinstance(recipe.get("artifacts"), dict) else {}
    return not bool(artifacts)
