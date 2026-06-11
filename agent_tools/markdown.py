from __future__ import annotations

from .decisions import DecisionReport, DecisionStatus


def questions_payload(report: DecisionReport) -> list[dict]:
    return [
        {
            "field": issue.field,
            "status": issue.status.value,
            "message": issue.message,
            "question": issue.question,
            "evidence": issue.evidence,
        }
        for issue in report.blocking_issues()
    ]


def questions_markdown(report: DecisionReport) -> str:
    questions = questions_payload(report)
    if not questions:
        return "# Questions\n\nNo user input is required.\n"
    lines = ["# Questions", "", "The agent must ask the user these questions before continuing.", ""]
    for idx, item in enumerate(questions, start=1):
        lines.extend(
            [
                f"{idx}. {item['field']}",
                f"   {item.get('question') or item.get('message')}",
                "   Evidence:",
            ]
        )
        evidence = item.get("evidence") or {}
        if evidence:
            for key, value in evidence.items():
                lines.append(f"   - {key}: {value}")
        else:
            lines.append("   - none")
        lines.append("")
    return "\n".join(lines)


def report_text(report: DecisionReport) -> str:
    lines = [f"Status: {report.status.value}", ""]
    blocking = report.blocking_issues()
    if blocking:
        lines.append(
            f"The agent cannot safely continue because {len(blocking)} high-impact decision(s) are unclear or invalid."
        )
        lines.append("")
        lines.append("Questions for user:")
        lines.append("")
        for idx, issue in enumerate(blocking, start=1):
            lines.append(f"{idx}. {issue.field}")
            lines.append(f"   {issue.message}")
            if issue.question and issue.question != issue.message:
                lines.append(f"   Question: {issue.question}")
            if issue.evidence:
                lines.append("   Evidence:")
                for key, value in issue.evidence.items():
                    lines.append(f"   - {key}: {value}")
            lines.append("")
    else:
        for issue in report.issues:
            if issue.status == DecisionStatus.WARN:
                lines.append(f"WARN {issue.field}: {issue.message}")
    return "\n".join(lines).rstrip() + "\n"
