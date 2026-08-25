from __future__ import annotations

from typing import Any

from . import decision_paths as paths, plan_rendering as rendering
from .adapters import SUPPORTED_TASKS, all_adapters, get_adapter
from .decision_models import DecisionIssue, DecisionStatus
from .models import SUPPORTED_VARIANTS

_COMMON_RECIPE_FIELDS = frozenset({"decisions", "experiment", "input_snapshots", "name", "step", "task", "variant"})


def recipe_structure_issues(task: Any, recipe: dict[str, Any], *, source_layer: str) -> list[DecisionIssue]:
    if not isinstance(task, str) or task not in SUPPORTED_TASKS:
        return [_contract_issue("task", f"Unsupported task: {task}", task, source_layer)]
    adapter = get_adapter(task)
    assert adapter is not None
    issues = variant_structure_issues(task, recipe.get("variant"), source_layer=source_layer)
    allowed_top_level = _COMMON_RECIPE_FIELDS | adapter.recipe_extra_fields
    issues.extend(
        _contract_issue(
            str(field),
            f"Unknown recipe field for task={task}: {field}.",
            recipe[field],
            source_layer,
        )
        for field in sorted(set(recipe) - allowed_top_level)
        if not str(field).startswith("_")
    )

    if "decisions" in recipe and not isinstance(recipe["decisions"], dict):
        issues.append(_contract_issue("decisions", "decisions must be a mapping.", recipe["decisions"], source_layer))
    issues.extend(runtime_structure_issues(task, recipe, source_layer=source_layer))

    adapter_contract = adapter.section_contract_issues(recipe, source_layer=source_layer)
    if adapter_contract is not None:
        issues.extend(adapter_contract)
    else:
        issues.extend(task_recipe_contract_issues(task, recipe, source_layer=source_layer))
        issues.extend(
            paths.execution_contract_issues(
                recipe,
                source_layer=source_layer,
                supports_runtime_identity=adapter.supports_runtime_identity,
            )
        )
    issues.extend(_artifact_contract_issues(task, recipe, source_layer=source_layer))
    return issues


def variant_structure_issues(task: str, variant: Any, *, source_layer: str) -> list[DecisionIssue]:
    adapter = get_adapter(task)
    if adapter is None:
        return []
    if adapter.requires_variant and variant not in SUPPORTED_VARIANTS:
        return [
            DecisionIssue(
                DecisionStatus.NEEDS_USER_INPUT,
                "variant",
                "Recipe variant is missing or unsupported.",
                "Which variant should this task use: sleep2vec, sleep2vec2, sleep2expert, or sex_age_baseline?",
                {
                    "variant": variant,
                    "allowed_values": list(SUPPORTED_VARIANTS),
                    "source_layer": source_layer,
                    "preflight_before_workspace": True,
                },
            )
        ]
    if not adapter.requires_variant and variant not in (None, ""):
        return [
            _contract_issue(
                "variant",
                f"task={task} must omit variant or set it to null; {task} is not a model variant.",
                variant,
                source_layer,
            )
        ]
    if isinstance(variant, str) and variant in adapter.unsupported_variants:
        return [
            _contract_issue(
                "variant",
                f"{variant} does not support {task}.",
                variant,
                source_layer,
            )
        ]
    return []


def runtime_structure_issues(task: str | None, recipe: dict, *, source_layer: str) -> list[DecisionIssue]:
    if "runtime" not in recipe:
        return []
    runtime = recipe["runtime"]
    if not isinstance(runtime, dict):
        return [_contract_issue("runtime", "runtime must be a mapping.", runtime, source_layer)]
    adapter = get_adapter(task)
    if adapter is not None:
        allowed_fields = adapter.runtime_fields(recipe.get("variant"))
    elif task is None:
        allowed_fields = rendering.FINETUNE_RUNTIME_FIELDS | rendering.INFER_RUNTIME_FIELDS
        for registered in all_adapters():
            allowed_fields = allowed_fields | registered.runtime_fields(recipe.get("variant"))
    else:
        allowed_fields = frozenset()
    issues = [
        _contract_issue(
            f"runtime.{field}",
            f"Unknown runtime field for task={task}: {field}.",
            runtime[field],
            source_layer,
        )
        for field in sorted(set(runtime) - allowed_fields)
    ]
    if "avg_ckpts" in runtime and (type(runtime["avg_ckpts"]) is not int or runtime["avg_ckpts"] < 1):
        issues.append(
            _contract_issue(
                "runtime.avg_ckpts",
                "runtime.avg_ckpts must be a positive integer.",
                runtime["avg_ckpts"],
                source_layer,
            )
        )
    return issues


def task_recipe_contract_issues(task: str, recipe: dict, *, source_layer: str) -> list[DecisionIssue]:
    issues: list[DecisionIssue] = []
    adapter = get_adapter(task)
    if adapter is None:
        return issues
    for section, allowed_fields in adapter.contract_sections.items():
        if section not in recipe or allowed_fields is None:
            continue
        value = recipe[section]
        if not isinstance(value, dict):
            issues.append(_contract_issue(section, f"{section} must be a mapping.", value, source_layer))
            continue
        for field in sorted(set(value) - allowed_fields):
            issues.append(
                _contract_issue(
                    f"{section}.{field}",
                    f"Unknown {section} field for task={task}: {field}.",
                    value[field],
                    source_layer,
                )
            )
    return issues


def _artifact_contract_issues(task: str, recipe: dict, *, source_layer: str) -> list[DecisionIssue]:
    if "artifacts" not in recipe:
        return []
    artifacts = recipe["artifacts"]
    if not isinstance(artifacts, dict):
        return [_contract_issue("artifacts", "artifacts must be a mapping.", artifacts, source_layer)]
    adapter = get_adapter(task)
    allowed_fields = adapter.artifact_fields if adapter is not None else frozenset()
    return [
        _contract_issue(
            f"artifacts.{field}",
            f"Unknown artifacts field for task={task}: {field}.",
            artifacts[field],
            source_layer,
        )
        for field in sorted(set(artifacts) - allowed_fields)
    ]


def _contract_issue(field: str, message: str, value: Any, source_layer: str) -> DecisionIssue:
    return DecisionIssue(
        DecisionStatus.FAIL,
        field,
        message,
        None,
        {"value": value, "source_layer": source_layer, "preflight_before_workspace": True},
    )
