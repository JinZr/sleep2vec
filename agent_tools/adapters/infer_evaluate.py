from __future__ import annotations

from typing import Any

from ..decision_models import DecisionIssue, DecisionStatus, ResolvedDecision, needs_issue
from ..decision_paths import (
    inference_checkpoint_averaging_issue,
    multilabel_sidecar_issue,
    sex_age_pretrained_backbone_issue,
    survival_sidecar_issue,
)
from ..models import coerce_list
from ..plan_rendering import (
    INFER_RUNTIME_FIELDS,
    infer_input_cli_args,
    infer_runtime_cli_args,
    render_command,
    variant_module,
)
from .base import TaskAdapter

_INFER_EVALUATE_TASKS = frozenset({"infer", "evaluate"})
# Byte-compat guard for sex_age_pretrained_backbone_issue: the pre-adapter
# kernel gated this helper on the recipe's own task string being one of the
# model tasks (a finetune recipe dispatched as infer still produced the
# issue; a task-less recipe did not).
_SEX_AGE_PRETRAINED_GUARD_TASKS = frozenset({"finetune", "infer", "evaluate"})
# Byte-compat guard for the sidecar helpers' finetune-config membership, keyed
# on the recipe's own task string like the pre-adapter kernel sets were.
_FINETUNE_CONFIG_GUARD_TASKS = frozenset({"finetune", "hparam_tune", "infer", "evaluate"})

_INPUT_FIELDS = frozenset(
    {
        "ckpt_path",
        "config",
        "data_backend",
        "eval_split",
        "inference_preset_path",
        "label_name",
        "override_dataset_names",
        "pretrained_backbone_path",
    }
)
_EVALUATION_FIELDS = frozenset({"external_test_locked", "final_test_unlocked"})


def _inputs(recipe: dict[str, Any]) -> dict[str, Any]:
    return recipe.get("inputs") if isinstance(recipe.get("inputs"), dict) else {}


def _recipe_preset_field(recipe: dict[str, Any]) -> str | None:
    # Byte-compat with the pre-adapter kernel: these helpers were keyed on the
    # recipe's own task string, not the dispatch task.
    return "inference_preset_path" if str(recipe.get("task")) in _INFER_EVALUATE_TASKS else None


def sex_age_override_dataset_names_issue(task: str, recipe: dict) -> DecisionIssue | None:
    if recipe.get("variant") != "sex_age_baseline" or task not in _INFER_EVALUATE_TASKS:
        return None
    inputs = _inputs(recipe)
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


class InferEvaluateAdapter(TaskAdapter):
    recipe_extra_fields = frozenset({"artifacts", "evaluation_policy", "execution", "inputs", "runtime"})
    artifact_fields = frozenset({"overwrite"})
    contract_sections = {"inputs": _INPUT_FIELDS, "evaluation_policy": _EVALUATION_FIELDS}
    preset_path_recipe_field = "inference_preset_path"
    validates_dataset_paths = True
    uses_finetune_config = True
    supports_runtime_identity = True

    def __init__(self, task: str, extra_decision_fields: frozenset[str]) -> None:
        self.task = task
        self.extra_decision_fields = extra_decision_fields

    def runtime_fields(self, variant: Any) -> frozenset[str]:
        return INFER_RUNTIME_FIELDS

    def required_input_paths(self, recipe: dict[str, Any]) -> list[tuple[str, Any]]:
        inputs = _inputs(recipe)
        runtime = recipe.get("runtime") if isinstance(recipe.get("runtime"), dict) else {}
        avg_ckpts = runtime.get("avg_ckpts", 1)
        averages_checkpoints = type(avg_ckpts) is int and avg_ckpts > 1
        required: list[tuple[str, Any]] = []
        for input_field in ("ckpt_path", "pretrained_backbone_path"):
            if recipe.get("variant") == "sex_age_baseline" and input_field == "pretrained_backbone_path":
                continue
            value = inputs.get(input_field)
            if input_field == "ckpt_path" and averages_checkpoints and value in ("best", "last"):
                continue
            if value not in (None, "", "ASK_USER"):
                required.append((input_field, value))
        return required

    def task_issues(
        self,
        recipe: dict[str, Any],
        config_summary: dict[str, Any] | None,
        decisions: dict[str, ResolvedDecision],
        high_impact: dict[str, dict[str, Any]],
    ) -> list[DecisionIssue]:
        issues: list[DecisionIssue] = []
        evaluation = recipe.get("evaluation_policy") if isinstance(recipe.get("evaluation_policy"), dict) else {}
        inputs = _inputs(recipe)

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
            # Only the structural config-family marker is a routing gate; variant_guess can be path-derived.
            config_variant = config_summary.get("authoritative_variant")
            recipe_variant = recipe.get("variant")
            # An unresolved variant belongs to the consultation gate; only an explicit conflict is invalid.
            if (
                config_variant is not None
                and recipe_variant not in (None, "", "ASK_USER")
                and recipe_variant != config_variant
            ):
                issues.append(
                    DecisionIssue(
                        DecisionStatus.FAIL,
                        "variant",
                        f"Config family requires variant={config_variant}.",
                        None,
                        {"config_variant": config_variant, "recipe_variant": recipe_variant},
                    )
                )
        if inputs.get("eval_split") == "test":
            if "external_test_locked" not in evaluation or evaluation["external_test_locked"] is True:
                issues.append(
                    DecisionIssue(
                        DecisionStatus.NEEDS_USER_INPUT,
                        "external_test_locked",
                        "Test evaluation requires external_test_locked=false.",
                        "Should the external test set be unlocked for this inference/evaluation run?",
                        {
                            "evaluation_policy": evaluation,
                            "external_test_locked": evaluation.get("external_test_locked"),
                        },
                    )
                )
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
        averaging_issue = inference_checkpoint_averaging_issue(recipe, inputs.get("ckpt_path"))
        if averaging_issue is not None:
            issues.append(averaging_issue)
        if str(recipe.get("task")) in _SEX_AGE_PRETRAINED_GUARD_TASKS:
            pretrained_issue = sex_age_pretrained_backbone_issue(recipe)
            if pretrained_issue is not None:
                issues.append(pretrained_issue)
        override_issue = sex_age_override_dataset_names_issue(str(recipe.get("task")), recipe)
        if override_issue is not None:
            issues.append(override_issue)
        survival_issue = survival_sidecar_issue(
            str(recipe.get("task")),
            recipe,
            config_summary,
            preset_path_recipe_field=_recipe_preset_field(recipe),
            uses_finetune_config=str(recipe.get("task")) in _FINETUNE_CONFIG_GUARD_TASKS,
        )
        if survival_issue is not None:
            issues.append(survival_issue)
        multilabel_issue = multilabel_sidecar_issue(
            str(recipe.get("task")),
            recipe,
            config_summary,
            preset_path_recipe_field=_recipe_preset_field(recipe),
            uses_finetune_config=str(recipe.get("task")) in _FINETUNE_CONFIG_GUARD_TASKS,
        )
        if multilabel_issue is not None:
            issues.append(multilabel_issue)
        return issues

    def commands(self, recipe: dict[str, Any], config_summary: dict[str, Any] | None) -> list[str]:
        inputs = _inputs(recipe)
        runtime = recipe.get("runtime") if isinstance(recipe.get("runtime"), dict) else {}
        execution = recipe.get("execution") if isinstance(recipe.get("execution"), dict) else {}
        return [
            render_command(
                [
                    execution.get("python") or "python",
                    "-m",
                    variant_module(recipe, "infer"),
                    "--config",
                    inputs.get("config"),
                    "--ckpt-path",
                    inputs.get("ckpt_path"),
                    "--label-name",
                    inputs.get("label_name"),
                    "--eval-split",
                    inputs.get("eval_split"),
                    *infer_runtime_cli_args(runtime),
                    *infer_input_cli_args(inputs, variant=str(recipe.get("variant"))),
                ]
            )
        ]

    def index_summary_inputs_override(
        self, recipe: dict[str, Any], config_summary: dict[str, Any] | None
    ) -> tuple[list[Any], Any, list[Any]] | None:
        if recipe.get("task") != self.task:
            return None
        inputs = _inputs(recipe)
        split_values = coerce_list(inputs.get("eval_split"))
        if self._effective_preset_path(recipe, config_summary) not in (None, ""):
            return [], inputs.get("config"), split_values
        data = (config_summary or {}).get("data") or {}
        return coerce_list(data.get("finetune_data_index")), inputs.get("config"), split_values

    @staticmethod
    def _effective_preset_path(recipe: dict[str, Any], cfg: dict[str, Any] | None) -> Any:
        value = _inputs(recipe).get("inference_preset_path")
        if value not in (None, "", "ASK_USER"):
            return value
        if cfg:
            value = (cfg.get("data") or {}).get("finetune_preset_path")
            if value not in (None, "", "ASK_USER"):
                return value
        return None


INFER_ADAPTER = InferEvaluateAdapter(
    "infer", frozenset({"config", "external_test_locked", "final_eval_unlock", "pretrained_backbone_path"})
)
EVALUATE_ADAPTER = InferEvaluateAdapter(
    "evaluate", frozenset({"config", "external_test_locked", "pretrained_backbone_path"})
)
