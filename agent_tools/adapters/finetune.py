from __future__ import annotations

from pathlib import Path
from typing import Any

from ..decision_models import DecisionIssue, DecisionReport, DecisionStatus, ResolvedDecision, merge_status, needs_issue
from ..decision_paths import multilabel_sidecar_issue, sex_age_pretrained_backbone_issue, survival_sidecar_issue
from ..models import REPO_ROOT, recipe_name
from ..plan_rendering import (
    FINETUNE_RUNTIME_FIELDS,
    finetune_input_cli_args,
    render_command,
    runtime_cli_args,
    variant_module,
)
from .base import TaskAdapter, config_summary_issues, recipe_inputs


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

    def frozen_command_prefix(self, recipe: dict[str, Any]) -> tuple[str, ...]:
        return ("python", "-m", variant_module(recipe, "finetune"))

    def required_input_paths(self, recipe: dict[str, Any]) -> list[tuple[str, Any]]:
        inputs = recipe_inputs(recipe)
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
        evaluation = recipe.get("evaluation_policy")
        if not isinstance(evaluation, dict):
            evaluation = {}

        issues.extend(config_summary_issues(recipe, config_summary))
        test_after_fit = decisions["test_after_fit"].value
        if type(test_after_fit) is not bool:
            issues.append(
                DecisionIssue(
                    DecisionStatus.NEEDS_USER_INPUT,
                    "test_after_fit",
                    "test_after_fit must be true or false when provided.",
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
            and config_summary.get("authoritative_variant") == "sex_age_baseline"
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

    def preflight_issues(
        self,
        recipe: dict[str, Any],
        config_summary: dict[str, Any] | None,
        *,
        unlock_final_test: bool,
        output_dir: Path | None = None,
    ) -> list[DecisionIssue]:
        evaluation = recipe.get("evaluation_policy")
        if not isinstance(evaluation, dict):
            evaluation = {}
        if evaluation.get("selection_split") != "test":
            return []
        return [
            DecisionIssue(
                DecisionStatus.FAIL,
                "evaluation_policy.selection_split",
                "Direct finetune cannot select checkpoints on test. Use task=hparam_tune with one configuration "
                "and max_runs: 1 to test and rank every saved epoch checkpoint.",
                None,
                {"selection_split": "test", "preflight_before_workspace": True},
            )
        ]

    def prepare_doctor_report(self, recipe: dict[str, Any], report: DecisionReport) -> DecisionReport:
        if report.exit_code != 0:
            return report
        issues = [*report.issues, *self.preflight_issues(recipe, None, unlock_final_test=False)]
        return DecisionReport(status=merge_status(issues), issues=issues, decisions=report.decisions)

    def commands(self, recipe: dict[str, Any], config_summary: dict[str, Any] | None) -> list[str]:
        inputs = recipe_inputs(recipe)
        runtime = recipe.get("runtime")
        if not isinstance(runtime, dict):
            runtime = {}
        artifacts = recipe.get("artifacts")
        if not isinstance(artifacts, dict):
            artifacts = {}
        evaluation = recipe.get("evaluation_policy")
        if not isinstance(evaluation, dict):
            evaluation = {}
        test_after_fit = evaluation["test_after_fit"]
        pieces = [
            *self.frozen_command_prefix(recipe),
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
        if test_after_fit:
            pieces.append("--test-after-fit")
        else:
            pieces.append("--no-test-after-fit")
        return [render_command(pieces)]

    def managed_runtime_dir(self, recipe: dict[str, Any], version: str) -> Path | None:
        execution = recipe.get("execution")
        if not isinstance(execution, dict):
            execution = {}
        return Path(str(execution.get("workdir") or REPO_ROOT)) / "log-finetune" / version


FINETUNE_ADAPTER = FinetuneAdapter()
