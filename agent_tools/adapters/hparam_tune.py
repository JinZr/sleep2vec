"""The ``hparam_tune`` task adapter.

The single composite adapter: its ``base_task`` is ``finetune``, so recipes
carrying ``_base_recipe`` / ``_local_recipe`` layers close under its contract.

Binds the hparam decision contract for hyper-parameter plans; ``plan_hparam``
materializes and compiles their plan contract.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .. import plan_hparam
from ..decision_hparam import hparam_recipe_contract_issues, hparam_search_issues, hparam_tune_issues
from ..decision_models import DecisionIssue, DecisionStatus, ResolvedDecision
from ..models import ConfigSummaryInput, JsonValue
from ..plan_rendering import FINETUNE_RUNTIME_FIELDS, INFER_RUNTIME_FIELDS, variant_module
from .base import TaskAdapter
from .finetune import FINETUNE_ADAPTER


class HparamTuneAdapter(TaskAdapter):
    task = "hparam_tune"

    recipe_extra_fields = frozenset(
        {"artifacts", "base_recipe", "evaluation_policy", "execution", "inputs", "runtime", "search"}
    )
    artifact_fields = frozenset({"results_csv_path"})
    extra_decision_fields = frozenset(
        {
            "ckpt_path",
            "init_checkpoint_path",
            "config",
            "data_backend",
            "final_eval_config_path",
            "pretrained_backbone_path",
            "required_channels",
            "test_after_fit",
        }
    )
    base_task = "finetune"
    uses_finetune_config = True
    enforces_required_channels = True
    materializes_plan = True
    supports_doctor_runtime_diagnostics = True
    decision_recipe_targets = {
        "hparam_search_space": ("search", "parameters"),
        "hparam_budget": ("search", "max_runs"),
    }

    def runtime_fields(self, variant: Any) -> frozenset[str]:
        return FINETUNE_RUNTIME_FIELDS | INFER_RUNTIME_FIELDS

    def required_input_paths(self, recipe: dict[str, JsonValue]) -> list[tuple[str, JsonValue]]:
        return [item for item in FINETUNE_ADAPTER.required_input_paths(recipe) if item[0] == "init_checkpoint_path"]

    def frozen_input_paths(self, recipe: dict[str, JsonValue]) -> list[tuple[str, Path]]:
        return FINETUNE_ADAPTER.frozen_input_paths(recipe)

    def frozen_command_prefix(self, recipe: dict[str, Any]) -> tuple[str, ...]:
        execution = recipe.get("execution")
        if not isinstance(execution, dict):
            execution = {}
        return (str(execution.get("python") or "python"), "-m", variant_module(recipe, "finetune"))

    def section_contract_issues(self, recipe: dict[str, Any], *, source_layer: str) -> list[DecisionIssue] | None:
        return hparam_recipe_contract_issues(recipe, source_layer=source_layer)

    def recipe_input_issues(self, recipe: dict[str, JsonValue]) -> list[DecisionIssue]:
        return [issue for issue in hparam_search_issues(recipe, high_impact={}) if issue.status == DecisionStatus.FAIL]

    def task_issues(
        self,
        recipe: dict[str, JsonValue],
        config_summary: ConfigSummaryInput | None,
        decisions: dict[str, ResolvedDecision],
        high_impact: dict[str, dict[str, Any]],
    ) -> list[DecisionIssue]:
        return hparam_tune_issues(recipe, config_summary, decisions, high_impact)

    def config_override_issues(
        self, recipe: dict[str, Any], config_summary: ConfigSummaryInput | None
    ) -> list[DecisionIssue] | None:
        # Override checks must consume the same snapshot that build_plan will freeze, not reopen a mutable path.
        config_bytes = (config_summary or {}).get("_source_config_bytes")
        if not isinstance(config_bytes, bytes):
            return [
                DecisionIssue(
                    DecisionStatus.FAIL,
                    "config",
                    "Hparam YAML override validation requires bound source config bytes.",
                    None,
                    {"preflight_before_workspace": True},
                )
            ]
        return plan_hparam.hparam_yaml_override_issues(recipe, config_bytes=config_bytes)

    def preflight_issues(
        self,
        recipe: dict[str, Any],
        config_summary: ConfigSummaryInput | None,
        *,
        unlock_final_test: bool,
        output_dir: Path | None = None,
    ) -> list[DecisionIssue]:
        return plan_hparam.final_test_checkpoint_issues(
            recipe,
            config_summary,
            unlock_final_test=unlock_final_test,
        )

    def registration_rows(self, plan: dict[str, Any]) -> list[dict[str, Any]]:
        return plan_hparam.hparam_manifest_rows(plan)


HPARAM_TUNE_ADAPTER = HparamTuneAdapter()
