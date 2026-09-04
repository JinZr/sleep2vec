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
from .base import TaskAdapter, config_summary_issues, recipe_inputs

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


def _recipe_preset_field(recipe: dict[str, Any]) -> str | None:
    # Byte-compat with the pre-adapter kernel: these helpers were keyed on the
    # recipe's own task string, not the dispatch task.
    return "inference_preset_path" if str(recipe.get("task")) in _INFER_EVALUATE_TASKS else None


def sex_age_override_dataset_names_issue(task: str, recipe: dict) -> DecisionIssue | None:
    if recipe.get("variant") != "sex_age_baseline" or task not in _INFER_EVALUATE_TASKS:
        return None
    inputs = recipe_inputs(recipe)
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
    slurm_launch_subcommand = "infer-launch"

    def __init__(self, task: str, extra_decision_fields: frozenset[str]) -> None:
        self.task = task
        self.extra_decision_fields = extra_decision_fields

    def runtime_fields(self, variant: Any) -> frozenset[str]:
        return INFER_RUNTIME_FIELDS

    def bind_effective_recipe(
        self,
        recipe: dict[str, Any],
        config_summary: dict[str, Any] | None,
        *,
        source_recipe: dict[str, Any] | None = None,
    ) -> list[DecisionIssue]:
        execution = recipe.get("execution") or {}
        if isinstance(execution.get("scheduler"), dict) and execution["scheduler"].get("type") == "slurm":
            gpus = execution.get("gpus_per_run", 1)
            if type(gpus) is int and gpus > 0:
                recipe.setdefault("runtime", {}).setdefault("devices", list(range(gpus)))
        return []

    def frozen_command_prefix(self, recipe: dict[str, Any]) -> tuple[str, ...]:
        execution = recipe.get("execution") if isinstance(recipe.get("execution"), dict) else {}
        return (str(execution.get("python") or "python"), "-m", variant_module(recipe, "infer"))

    def required_input_paths(self, recipe: dict[str, Any]) -> list[tuple[str, Any]]:
        inputs = recipe_inputs(recipe)
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
        inputs = recipe_inputs(recipe)

        issues.extend(config_summary_issues(recipe, config_summary))
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
        inputs = recipe_inputs(recipe)
        runtime = recipe.get("runtime") if isinstance(recipe.get("runtime"), dict) else {}
        return [
            render_command(
                [
                    *self.frozen_command_prefix(recipe),
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

    def index_summary_split_values(self, recipe: dict[str, Any]) -> list[Any]:
        # infer/evaluate name their split in the recipe rather than loading the
        # finetune split set.
        return coerce_list(recipe_inputs(recipe).get("eval_split"))


INFER_ADAPTER = InferEvaluateAdapter(
    "infer", frozenset({"config", "external_test_locked", "final_eval_unlock", "pretrained_backbone_path"})
)
EVALUATE_ADAPTER = InferEvaluateAdapter(
    "evaluate", frozenset({"config", "external_test_locked", "pretrained_backbone_path"})
)
