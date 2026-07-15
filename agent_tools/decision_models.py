from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


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
    evidence: dict[str, Any] = field(default_factory=dict)


def merge_status(issues: list[DecisionIssue]) -> DecisionStatus:
    if any(issue.status == DecisionStatus.FAIL for issue in issues):
        return DecisionStatus.FAIL
    if any(issue.status == DecisionStatus.NEEDS_USER_INPUT for issue in issues):
        return DecisionStatus.NEEDS_USER_INPUT
    if any(issue.status == DecisionStatus.WARN for issue in issues):
        return DecisionStatus.WARN
    return DecisionStatus.PASS


def question_for(high_impact: dict[str, dict[str, Any]], field: str) -> str | None:
    return high_impact.get(field, {}).get("question")


def needs_issue(
    field: str,
    message: str,
    high_impact: dict[str, dict[str, Any]],
    evidence: dict | None = None,
) -> DecisionIssue:
    return DecisionIssue(
        DecisionStatus.NEEDS_USER_INPUT,
        field,
        message,
        question_for(high_impact, field),
        evidence or {},
    )


def contract_issue(field: str, message: str, value: Any, source_layer: str) -> DecisionIssue:
    return DecisionIssue(
        DecisionStatus.FAIL,
        field,
        message,
        None,
        {"value": value, "source_layer": source_layer, "preflight_before_workspace": True},
    )
