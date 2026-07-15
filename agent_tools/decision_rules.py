from __future__ import annotations

from typing import Any

from .adapters import get_adapter
from .decision_models import DecisionIssue, DecisionStatus

_INPUT_FIELDS: dict[str, set[str]] = {}
_EVALUATION_FIELDS: dict[str, set[str]] = {}


def task_recipe_contract_issues(task: str, recipe: dict, *, source_layer: str) -> list[DecisionIssue]:
    issues: list[DecisionIssue] = []
    adapter = get_adapter(task)
    if adapter is not None:
        sections = {
            section: adapter.contract_sections.get(section) for section in ("inputs", "evaluation_policy", "preset")
        }
    else:
        sections = {
            "inputs": _INPUT_FIELDS.get(task),
            "evaluation_policy": _EVALUATION_FIELDS.get(task),
            "preset": None,
        }
    for section, allowed_fields in sections.items():
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
