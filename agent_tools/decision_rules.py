from __future__ import annotations

from typing import Any

from .adapters import get_adapter
from .decision_models import DecisionIssue, DecisionStatus


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


def _contract_issue(field: str, message: str, value: Any, source_layer: str) -> DecisionIssue:
    return DecisionIssue(
        DecisionStatus.FAIL,
        field,
        message,
        None,
        {"value": value, "source_layer": source_layer, "preflight_before_workspace": True},
    )
