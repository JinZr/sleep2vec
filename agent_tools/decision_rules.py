from __future__ import annotations

from typing import Any

from . import plan_rendering as rendering
from .adapters import get_adapter
from .decision_models import DecisionIssue, DecisionStatus, ResolvedDecision, needs_issue
from .decision_paths import multilabel_sidecar_issue, sleep2stat_existing_run_dir_issue, survival_sidecar_issue

_INPUT_FIELDS = {
    "preset_prepare": {"config", "dataset_name", "index"},
    "finetune": {"ckpt_path", "config", "data_backend", "label_name", "pretrained_backbone_path"},
    "infer": {
        "ckpt_path",
        "config",
        "data_backend",
        "eval_split",
        "inference_preset_path",
        "label_name",
        "override_dataset_names",
        "pretrained_backbone_path",
    },
    "evaluate": {
        "ckpt_path",
        "config",
        "data_backend",
        "eval_split",
        "inference_preset_path",
        "label_name",
        "override_dataset_names",
        "pretrained_backbone_path",
    },
}
_EVALUATION_FIELDS = {
    "finetune": {"external_test_locked", "selection_metric", "selection_mode", "selection_split", "test_after_fit"},
    "infer": {"external_test_locked", "final_test_unlocked"},
    "evaluate": {"external_test_locked", "final_test_unlocked"},
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
            "preset": rendering.PRESET_FIELDS if task == "preset_prepare" else None,
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


def sleep2stat_issues(
    recipe: dict,
    config_summary: dict | None,
    high_impact: dict[str, dict[str, Any]],
) -> list[DecisionIssue]:
    issues: list[DecisionIssue] = []
    evaluation = recipe.get("evaluation_policy") if isinstance(recipe.get("evaluation_policy"), dict) else {}
    inputs = recipe.get("inputs") if isinstance(recipe.get("inputs"), dict) else {}

    if not inputs.get("config"):
        issues.append(needs_issue("config", "sleep2stat requires inputs.config.", high_impact))
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
    if config_run_dir:
        existing_run_dir_issue = sleep2stat_existing_run_dir_issue(recipe, config_run_dir)
        if existing_run_dir_issue is not None:
            issues.append(existing_run_dir_issue)
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
    return issues


def preset_prepare_issues(
    recipe: dict, config_summary: dict | None, high_impact: dict[str, dict[str, Any]]
) -> list[DecisionIssue]:
    issues: list[DecisionIssue] = []
    inputs = recipe.get("inputs") if isinstance(recipe.get("inputs"), dict) else {}
    preset = recipe.get("preset") if isinstance(recipe.get("preset"), dict) else {}

    for input_field, value in {
        "index": inputs.get("index"),
        "dataset_name": inputs.get("dataset_name"),
        "split": preset.get("split"),
        "n_tokens": preset.get("n_tokens"),
        "allow_missing_channels": preset.get("allow_missing_channels"),
    }.items():
        if value in (None, "", []):
            issues.append(
                needs_issue(
                    input_field,
                    f"{input_field} is required for preset preparation.",
                    high_impact,
                    {"recipe": value},
                )
            )
    if preset.get("allow_missing_channels") is True and preset.get("min_channels") is None:
        issues.append(
            needs_issue("min_channels", "min_channels is required when missing channels are allowed.", high_impact)
        )
    if recipe.get("variant") in {"sleep2vec2", "sleep2expert"}:
        if preset.get("manifest_output") not in (None, ""):
            issues.append(
                DecisionIssue(
                    DecisionStatus.FAIL,
                    "manifest_output",
                    f"{recipe['variant']} preset preparation does not support manifest_output.",
                    None,
                    {"variant": recipe["variant"]},
                )
            )
        if "write_sidecar_manifest" in preset:
            issues.append(
                DecisionIssue(
                    DecisionStatus.FAIL,
                    "write_sidecar_manifest",
                    f"{recipe['variant']} preset preparation does not support write_sidecar_manifest.",
                    None,
                    {"variant": recipe["variant"]},
                )
            )
    survival_issue = survival_sidecar_issue("preset_prepare", recipe, config_summary)
    if survival_issue is not None:
        issues.append(survival_issue)
    return issues


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
    pretrained_issue = _sex_age_pretrained_backbone_issue("finetune", recipe)
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


def infer_evaluate_issues(
    recipe: dict,
    config_summary: dict | None,
    decisions: dict[str, ResolvedDecision],
    high_impact: dict[str, dict[str, Any]],
) -> list[DecisionIssue]:
    issues: list[DecisionIssue] = []
    evaluation = recipe.get("evaluation_policy") if isinstance(recipe.get("evaluation_policy"), dict) else {}
    inputs = recipe.get("inputs") if isinstance(recipe.get("inputs"), dict) else {}

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
    if inputs.get("eval_split") == "test":
        if evaluation.get("final_test_unlocked") is not True:
            issues.append(
                needs_issue("final_eval_unlock", "Test evaluation requires explicit final unlock.", high_impact)
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
    pretrained_issue = _sex_age_pretrained_backbone_issue(str(recipe.get("task")), recipe)
    if pretrained_issue is not None:
        issues.append(pretrained_issue)
    override_issue = _sex_age_override_dataset_names_issue(str(recipe.get("task")), recipe)
    if override_issue is not None:
        issues.append(override_issue)
    survival_issue = survival_sidecar_issue(str(recipe.get("task")), recipe, config_summary)
    if survival_issue is not None:
        issues.append(survival_issue)
    multilabel_issue = multilabel_sidecar_issue(str(recipe.get("task")), recipe, config_summary)
    if multilabel_issue is not None:
        issues.append(multilabel_issue)
    return issues


def _sex_age_pretrained_backbone_issue(
    task: str,
    recipe: dict,
) -> DecisionIssue | None:
    if recipe.get("variant") != "sex_age_baseline" or task not in {"finetune", "infer", "evaluate"}:
        return None
    inputs = recipe.get("inputs") if isinstance(recipe.get("inputs"), dict) else {}
    value = inputs.get("pretrained_backbone_path")
    if value in (None, "", "ASK_USER"):
        return None
    return DecisionIssue(
        DecisionStatus.FAIL,
        "pretrained_backbone_path",
        "sex_age_baseline does not support pretrained_backbone_path.",
        None,
        {"variant": "sex_age_baseline", "pretrained_backbone_path": value},
    )


def _sex_age_override_dataset_names_issue(task: str, recipe: dict) -> DecisionIssue | None:
    if recipe.get("variant") != "sex_age_baseline" or task not in {"infer", "evaluate"}:
        return None
    inputs = recipe.get("inputs") if isinstance(recipe.get("inputs"), dict) else {}
    value = inputs.get("override_dataset_names")
    if value in (None, "", "ASK_USER"):
        return None
    return DecisionIssue(
        DecisionStatus.FAIL,
        "override_dataset_names",
        "sex_age_baseline does not support override_dataset_names.",
        None,
        {"variant": "sex_age_baseline", "override_dataset_names": value},
    )


def _as_list(value: Any) -> list[Any]:
    if value in (None, "", []):
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]
