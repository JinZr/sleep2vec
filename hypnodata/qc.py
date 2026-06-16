from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QCIssue:
    record_id: str
    scope: str
    canonical_channel: str
    code: str
    severity: str
    message: str


def issue_row(issue: QCIssue) -> dict[str, str]:
    return {
        "record_id": issue.record_id,
        "scope": issue.scope,
        "canonical_channel": issue.canonical_channel,
        "code": issue.code,
        "severity": issue.severity,
        "message": issue.message,
    }
