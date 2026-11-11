"""Plan-time index summary checks and the plan/blocked-plan Markdown renderings.

Layer 2 kernel, mixed bridge -- it imports ``domain.index_csv`` directly. The
index summary runs for every task with one implementation, so it stays a direct
import rather than a per-task adapter hook; adapters only vary its inputs
(``index_summary_inputs_override``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypedDict

from .adapters.registry import all_adapters
from .configs import runtime_path_base
from .decision_models import DecisionIssue, DecisionReport, DecisionStatus
from .domain.index_csv import IndexSummary, index_summary
from .models import CONFIG_FINETUNE_SECTION, ConfigSummaryInput, coerce_list


class SummaryFailure(TypedDict):
    blocking_issues: list[str]


def context_index_summary(
    recipe: dict,
    cfg: ConfigSummaryInput | None,
    *,
    validated_sidecar_keys: dict[str, set[str]] | None = None,
) -> IndexSummary | SummaryFailure | None:
    paths, config, split_values = _index_summary_inputs(recipe, cfg)
    finetune: Any = (cfg or {}).get(CONFIG_FINETUNE_SECTION) or {}
    task_type = (finetune.get("task") or {}).get("type")
    label_sidecars_valid = False
    if task_type == "survival":
        label_sidecars_valid = (finetune.get("survival") or {}).get("valid") is True
    elif task_type == "multilabel_classification":
        label_sidecars_valid = (finetune.get("multilabel") or {}).get("valid") is True
    if not paths:
        return None
    validated_summary = None
    if (
        config
        and validated_sidecar_keys is not None
        and cfg is not None
        and not cfg.get("blocking_issues")
        and isinstance(cfg.get("_source_config_bytes"), bytes)
        and (task_type not in {"survival", "multilabel_classification"} or label_sidecars_valid)
    ):
        # Reuse only a completed validation view; failed or deferred checks keep their original path.
        validated_summary = (cfg, validated_sidecar_keys)
    try:
        # Index and sidecar checks must stay bound to the config snapshot accepted by plan preflight.
        return index_summary(
            paths,
            config=config,
            config_bytes=(cfg or {}).get("_source_config_bytes"),
            local_path_base=runtime_path_base(recipe),
            split_values=split_values,
            validated_summary=validated_summary,
        )
    except Exception as exc:
        return {"blocking_issues": [f"Failed to summarize index: {exc}"]}


def _index_summary_inputs(recipe: dict, cfg: ConfigSummaryInput | None) -> tuple[list[Any], Any, list[Any]]:
    inputs = recipe["inputs"] if isinstance(recipe.get("inputs"), dict) else {}
    config = inputs.get("config")
    for adapter in all_adapters():
        override = adapter.index_summary_inputs_override(recipe, cfg)
        if override is not None:
            return override

    paths = coerce_list(inputs.get("index"))
    if not paths and cfg:
        data: Any = cfg.get("data") or {}
        paths = coerce_list(data.get("finetune_data_index"))
    return paths, config, []


def index_summary_issues(
    recipe: dict,
    cfg: ConfigSummaryInput | None,
    *,
    index_payload: IndexSummary | SummaryFailure | None = None,
    validated_sidecar_keys: dict[str, set[str]] | None = None,
) -> list[DecisionIssue]:
    if index_payload is None:
        index_payload = context_index_summary(recipe, cfg, validated_sidecar_keys=validated_sidecar_keys)
    blocking = (index_payload.get("blocking_issues") or []) if index_payload is not None else []
    return [
        DecisionIssue(
            DecisionStatus.FAIL,
            "data_input",
            issue,
            None,
            {"index_summary": index_payload},
        )
        for issue in blocking
    ]


def blocked_plan_markdown(
    report: DecisionReport,
    allow_unresolved: bool,
    *,
    user_decisions_path: Path | None = None,
) -> str:
    lines = ["# Agent Plan Blocked", "", f"Status: {report.status.value}", ""]
    if allow_unresolved:
        lines.append("A draft plan may be written, but executable commands are not generated.")
        lines.append("")
    lines.append("## Questions")
    for issue in report.blocking_issues():
        lines.append(f"- {issue.field}: {issue.question or issue.message}")
    if user_decisions_path is not None:
        lines.extend(
            [
                "",
                "## Next action",
                "",
                f"Fill `{user_decisions_path}` and pass it with `--user-decisions`.",
                "Use a fresh `--output-dir` for the retry.",
            ]
        )
    return "\n".join(lines) + "\n"


def plan_markdown(report: DecisionReport, commands: list[str]) -> str:
    lines = ["# Agent Plan", "", f"Status: {report.status.value}", "", "## Commands", ""]
    lines.extend(f"```bash\n{command}\n```" for command in commands)
    return "\n\n".join(lines) + "\n"
