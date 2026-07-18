from __future__ import annotations

from pathlib import Path
from typing import Any

from ..decision_models import DecisionIssue, DecisionStatus, ResolvedDecision, needs_issue
from ..decision_paths import multilabel_sidecar_issue, sex_age_pretrained_backbone_issue, survival_sidecar_issue
from ..models import REPO_ROOT, coerce_list, recipe_name
from ..plan_rendering import (
    FINETUNE_RUNTIME_FIELDS,
    finetune_input_cli_args,
    finetune_loaded_split_values,
    render_command,
    runtime_cli_args,
    variant_module,
)
from .base import TaskAdapter


def _inputs(recipe: dict[str, Any]) -> dict[str, Any]:
    return recipe.get("inputs") if isinstance(recipe.get("inputs"), dict) else {}


class FinetuneAdapter(TaskAdapter):
    task = "finetune"

    recipe_extra_fields = frozenset({"artifacts", "evaluation_policy", "execution", "inputs", "runtime"})
    artifact_fields = frozenset({"overwrite", "results_csv_path", "version_name"})
    contract_sections = {
        "inputs": frozenset({"ckpt_path", "config", "data_backend", "label_name", "pretrained_backbone_path"}),
        "evaluation_policy": frozenset(
            {"external_test_locked", "selection_metric", "selection_mode", "selection_split", "test_after_fit"}
        ),
    }
    extra_decision_fields = frozenset({"ckpt_path", "config", "external_test_locked", "test_after_fit"})
    validates_dataset_paths = True
    uses_finetune_config = True
    enforces_required_channels = True

    def runtime_fields(self, variant: Any) -> frozenset[str]:
        fields = FINETUNE_RUNTIME_FIELDS
        if variant == "sex_age_baseline":
            fields = fields - {"wandb_mode"}
        return fields

    def required_input_paths(self, recipe: dict[str, Any]) -> list[tuple[str, Any]]:
        inputs = _inputs(recipe)
        required: list[tuple[str, Any]] = []
        for input_field in ("pretrained_backbone_path", "ckpt_path"):
            if recipe.get("variant") == "sex_age_baseline" and input_field == "pretrained_backbone_path":
                continue
            value = inputs.get(input_field)
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
            # Only the structural config-family marker is a routing gate; variant_guess can be path-derived.
            config_variant = config_summary.get("authoritative_variant")
            if config_variant is not None and recipe.get("variant") != config_variant:
                issues.append(
                    DecisionIssue(
                        DecisionStatus.FAIL,
                        "variant",
                        f"Config family requires variant={config_variant}.",
                        None,
                        {"config_variant": config_variant, "recipe_variant": recipe.get("variant")},
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
        pretrained_issue = sex_age_pretrained_backbone_issue(recipe)
        if pretrained_issue is not None:
            issues.append(pretrained_issue)
        # self.task, not the recipe's own task string: the pre-adapter kernel
        # hard-coded "finetune" for these helpers.
        survival_issue = survival_sidecar_issue(self.task, recipe, config_summary, uses_finetune_config=True)
        if survival_issue is not None:
            issues.append(survival_issue)
        multilabel_issue = multilabel_sidecar_issue(self.task, recipe, config_summary, uses_finetune_config=True)
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

    def commands(self, recipe: dict[str, Any], config_summary: dict[str, Any] | None) -> list[str]:
        inputs = _inputs(recipe)
        runtime = recipe.get("runtime") if isinstance(recipe.get("runtime"), dict) else {}
        artifacts = recipe.get("artifacts") if isinstance(recipe.get("artifacts"), dict) else {}
        evaluation = recipe.get("evaluation_policy") if isinstance(recipe.get("evaluation_policy"), dict) else {}
        test_after_fit = evaluation.get("test_after_fit")
        pieces = [
            "python",
            "-m",
            variant_module(recipe, "finetune"),
            "--config",
            inputs.get("config"),
            "--label-name",
            inputs.get("label_name"),
            "--version-name",
            artifacts.get("version_name", recipe_name(recipe)),
            "--results-csv-path",
            artifacts.get("results_csv_path", "results/agent_results.csv"),
            *runtime_cli_args(runtime, variant=str(recipe.get("variant"))),
            *finetune_input_cli_args(
                inputs,
                variant=str(recipe.get("variant")),
            ),
        ]
        if test_after_fit is False or evaluation.get("external_test_locked") is True:
            pieces.append("--no-test-after-fit")
        return [render_command(pieces)]

    def managed_runtime_dir(self, recipe: dict[str, Any], version: str) -> Path | None:
        return REPO_ROOT / "log-finetune" / version

    def index_summary_inputs_override(
        self, recipe: dict[str, Any], config_summary: dict[str, Any] | None
    ) -> tuple[list[Any], Any, list[Any]] | None:
        if recipe.get("task") != self.task:
            return None
        inputs = _inputs(recipe)
        split_values = finetune_loaded_split_values(recipe)
        if self._effective_preset_path(config_summary) not in (None, ""):
            return [], inputs.get("config"), split_values
        data = (config_summary or {}).get("data") or {}
        return coerce_list(data.get("finetune_data_index")), inputs.get("config"), split_values

    @staticmethod
    def _effective_preset_path(cfg: dict[str, Any] | None) -> Any:
        # finetune has no recipe-level preset field (inference_preset_path is
        # infer/evaluate-only); only the config's finetune_preset_path applies.
        if cfg:
            value = (cfg.get("data") or {}).get("finetune_preset_path")
            if value not in (None, "", "ASK_USER"):
                return value
        return None


FINETUNE_ADAPTER = FinetuneAdapter()
