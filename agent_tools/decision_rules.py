from __future__ import annotations

from typing import Any

from .adapters import get_adapter
from .decision_models import DecisionIssue, DecisionStatus, ResolvedDecision, needs_issue
from .decision_paths import multilabel_sidecar_issue, sex_age_pretrained_backbone_issue, survival_sidecar_issue

_INPUT_FIELDS = {
    "finetune": {"ckpt_path", "config", "data_backend", "label_name", "pretrained_backbone_path"},
}
_EVALUATION_FIELDS = {
    "finetune": {"external_test_locked", "selection_metric", "selection_mode", "selection_split", "test_after_fit"},
}


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


def finetune_task_issues(
    recipe: dict,
    config_summary: dict | None,
    decisions: dict[str, ResolvedDecision],
    high_impact: dict[str, dict[str, Any]],
) -> list[DecisionIssue]:
    issues: list[DecisionIssue] = []
    evaluation = recipe.get("evaluation_policy") if isinstance(recipe.get("evaluation_policy"), dict) else {}
    inputs = recipe.get("inputs") if isinstance(recipe.get("inputs"), dict) else {}

    if not inputs.get("config"):
        issues.append(needs_issue("config", "Config path is required for finetune.", high_impact))
    if config_summary:
        for issue in config_summary.get("blocking_issues", []):
            issues.append(
                DecisionIssue(
                    DecisionStatus.NEEDS_USER_INPUT,
                    "config",
                    issue,
                    "Please fix the config before the agent generates commands.",
                    {"config_path": config_summary.get("config_path")},
                )
            )
    test_after_fit_decision = decisions.get("test_after_fit")
    test_after_fit = (
        test_after_fit_decision.value if test_after_fit_decision is not None else evaluation.get("test_after_fit")
    )
    if type(test_after_fit) is not bool:
        issues.append(
            DecisionIssue(
                DecisionStatus.NEEDS_USER_INPUT,
                "test_after_fit",
                "test_after_fit must be explicitly true or false for finetune command generation.",
                "Should test evaluation run after fit for this task?",
                {"value": test_after_fit, "evaluation_policy": evaluation},
            )
        )
    if "external_test_locked" not in evaluation:
        issues.append(
            needs_issue("external_test_locked", "external_test_locked must be explicit for finetune.", high_impact)
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
    if (
        config_summary
        and config_summary.get("variant_guess") == "sex_age_baseline"
        and config_summary.get("data_backend") == "kaldi"
    ):
        if not data.get("kaldi_data_root") or not data.get("kaldi_manifest"):
            issues.append(
                DecisionIssue(
                    DecisionStatus.NEEDS_USER_INPUT,
                    "data_input",
                    "Kaldi-backed sex_age_baseline finetune requires kaldi_data_root and kaldi_manifest.",
                    "Which Kaldi data root and manifest should this sex/age baseline use?",
                    {"config": data},
                )
            )
    pretrained_issue = sex_age_pretrained_backbone_issue("finetune", recipe)
    if pretrained_issue is not None:
        issues.append(pretrained_issue)
    survival_issue = survival_sidecar_issue("finetune", recipe, config_summary)
    if survival_issue is not None:
        issues.append(survival_issue)
    multilabel_issue = multilabel_sidecar_issue("finetune", recipe, config_summary)
    if multilabel_issue is not None:
        issues.append(multilabel_issue)
    external_test_locked = evaluation.get("external_test_locked")
    if external_test_locked is True and test_after_fit is True:
        issues.append(
            DecisionIssue(
                DecisionStatus.NEEDS_USER_INPUT,
                "test_after_fit",
                "test_after_fit=true would evaluate test while external_test_locked=true.",
                "Should test evaluation be disabled during model selection?",
                {"evaluation_policy": evaluation, "external_test_locked": external_test_locked},
            )
        )
    return issues
