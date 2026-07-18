from __future__ import annotations

from pathlib import Path
from typing import Any

from ..decision_hparam import hparam_recipe_contract_issues, hparam_tune_issues
from ..decision_models import DecisionIssue, DecisionReport, DecisionStatus, ResolvedDecision
from ..models import coerce_list
from ..plan_rendering import FINETUNE_RUNTIME_FIELDS, INFER_RUNTIME_FIELDS, finetune_loaded_split_values
from .base import TaskAdapter


class HparamTuneAdapter(TaskAdapter):
    task = "hparam_tune"

    recipe_extra_fields = frozenset(
        {"adaptive", "artifacts", "base_recipe", "evaluation_policy", "execution", "inputs", "runtime", "search"}
    )
    artifact_fields = frozenset({"overwrite", "results_csv_path"})
    extra_decision_fields = frozenset(
        {
            "ckpt_path",
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
    decision_recipe_targets = {
        "hparam_search_space": ("search", "parameters"),
        "hparam_budget": ("search", "max_runs"),
    }

    def runtime_fields(self, variant: Any) -> frozenset[str]:
        return FINETUNE_RUNTIME_FIELDS | INFER_RUNTIME_FIELDS

    def section_contract_issues(self, recipe: dict[str, Any], *, source_layer: str) -> list[DecisionIssue] | None:
        return hparam_recipe_contract_issues(recipe, source_layer=source_layer)

    def task_issues(
        self,
        recipe: dict[str, Any],
        config_summary: dict[str, Any] | None,
        decisions: dict[str, ResolvedDecision],
        high_impact: dict[str, dict[str, Any]],
    ) -> list[DecisionIssue]:
        return hparam_tune_issues(recipe, config_summary, decisions, high_impact)

    def config_override_issues(
        self, recipe: dict[str, Any], config_summary: dict[str, Any] | None
    ) -> list[DecisionIssue] | None:
        from .. import plan_hparam

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
        self, recipe: dict[str, Any], config_summary: dict[str, Any] | None, *, unlock_final_test: bool
    ) -> list[DecisionIssue]:
        from .. import plan_hparam

        return plan_hparam.final_test_checkpoint_issues(recipe, unlock_final_test=unlock_final_test)

    def write_plan(
        self,
        recipe: dict[str, Any],
        out: Path,
        *,
        unlock_final_test: bool,
        source_config_bytes: bytes,
        source_config_sha256: str,
    ) -> None:
        from .. import plan_hparam

        plan_hparam.write_hparam_plan(
            recipe,
            out,
            unlock_final_test=unlock_final_test,
            source_config_bytes=source_config_bytes,
            source_config_sha256=source_config_sha256,
        )

    def planned_plan_paths(
        self,
        recipe: dict[str, Any],
        out: Path,
        report: DecisionReport,
        *,
        allow_unresolved: bool,
        unlock_final_test: bool,
    ) -> list[Path] | None:
        from .. import plan_hparam
        from ..experiment_workspace import next_run_index, run_identity

        if report.exit_code != 0:
            paths = [out / "questions.json", out / "questions.md", out / "plan.blocked.md"]
            evaluation = recipe.get("evaluation_policy") or {}
            if plan_hparam.final_test_unlocked(evaluation, unlock_final_test):
                paths.append(out / "final_external_test.sh")
            if allow_unresolved and report.exit_code == 2:
                paths.append(out / "plan.draft.json")
            return paths
        paths = [
            out / "plan.json",
            out / "plan.md",
            out / "run_all.sh",
            out / "validation.sh",
            out / "recipe.resolved.yaml",
            out / "config.source.yaml",
        ]
        offset = next_run_index(recipe)
        for idx, combo in enumerate(plan_hparam.hparam_combos(recipe)):
            identity = run_identity(recipe, offset + idx, combo)
            run_dir = out / "runs" / f"{identity['run_id']}--{identity['run_name']}"
            paths.extend(
                [run_dir / "launch.sh", run_dir / "config.yaml", run_dir / "run.json", run_dir / "artifacts.json"]
            )
        paths.append(out / "final_external_test.sh")
        return paths

    def index_summary_inputs_override(
        self, recipe: dict[str, Any], config_summary: dict[str, Any] | None
    ) -> tuple[list[Any], Any, list[Any]] | None:
        if recipe.get("task") != self.task:
            return None
        inputs = recipe.get("inputs") if isinstance(recipe.get("inputs"), dict) else {}
        split_values = finetune_loaded_split_values(recipe, test_split_opt_in=True)
        if self._effective_preset_path(config_summary) not in (None, ""):
            return [], inputs.get("config"), split_values
        data = (config_summary or {}).get("data") or {}
        return coerce_list(data.get("finetune_data_index")), inputs.get("config"), split_values

    @staticmethod
    def _effective_preset_path(cfg: dict[str, Any] | None) -> Any:
        if cfg:
            value = (cfg.get("data") or {}).get("finetune_preset_path")
            if value not in (None, "", "ASK_USER"):
                return value
        return None


HPARAM_TUNE_ADAPTER = HparamTuneAdapter()
