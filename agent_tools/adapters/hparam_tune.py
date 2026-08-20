from __future__ import annotations

from pathlib import Path
import subprocess
from typing import Any

from .. import slurm
from ..decision_hparam import hparam_recipe_contract_issues, hparam_tune_issues
from ..decision_models import DecisionIssue, DecisionReport, DecisionStatus, ResolvedDecision, merge_status
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

        return plan_hparam.final_test_checkpoint_issues(
            recipe,
            config_summary,
            unlock_final_test=unlock_final_test,
        )

    def prepare_doctor_report(self, recipe: dict[str, Any], report: DecisionReport) -> DecisionReport:
        execution = recipe.get("execution") if isinstance(recipe.get("execution"), dict) else {}
        scheduler = execution.get("scheduler") if isinstance(execution.get("scheduler"), dict) else {}
        if scheduler.get("type") != "slurm" or report.blocking_issues():
            return report
        try:
            capabilities = slurm.cluster_scheduling_capabilities(execution, str(scheduler["partition"]))
        except (slurm.SlurmCommandError, subprocess.TimeoutExpired, ValueError) as exc:
            issue = DecisionIssue(
                DecisionStatus.WARN,
                "execution.scheduler.capabilities",
                f"Read-only Slurm capability inspection was unavailable: {exc}",
                None,
                {"error": str(exc)},
            )
            issues = [*report.issues, issue]
            return DecisionReport(status=merge_status(issues), issues=issues, decisions=report.decisions)

        priority_type = capabilities["priority_type"] or "unknown priority policy"
        scheduler_type = capabilities["scheduler_type"] or "unknown scheduler"
        accounting_type = capabilities["accounting_storage_type"] or "unknown accounting storage"
        message = (
            f"Read-only Slurm inspection reports {priority_type}, {scheduler_type}, {accounting_type}, and "
            f"{capabilities['reservation_count']} visible reservation(s). "
        )
        if priority_type == "priority/basic":
            message += (
                "Submit time and fitting credible short resource requests into backfill are the practical user-side "
                "levers; no setting can guarantee first priority."
            )
        elif priority_type == "priority/multifactor":
            message += (
                "Priority may also depend on cluster-controlled age, fair-share, QOS, and association policy; no "
                "setting can guarantee first priority."
            )
        else:
            message += "Cluster policy determines ordering; no user-side setting can guarantee first priority."
        issue = DecisionIssue(
            DecisionStatus.WARN,
            "execution.scheduler.capabilities",
            message,
            None,
            capabilities,
        )
        issues = [*report.issues, issue]
        return DecisionReport(status=merge_status(issues), issues=issues, decisions=report.decisions)

    def write_plan(
        self,
        recipe: dict[str, Any],
        out: Path,
        *,
        write_out: Path | None = None,
        unlock_final_test: bool,
        source_config_bytes: bytes,
        source_config_sha256: str,
    ) -> None:
        from .. import plan_hparam

        plan_hparam.write_hparam_plan(
            recipe,
            out,
            write_out=write_out,
            unlock_final_test=unlock_final_test,
            source_config_bytes=source_config_bytes,
            source_config_sha256=source_config_sha256,
        )

    def commit_plan(self, out: Path) -> None:
        from .. import plan_hparam

        plan_hparam.commit_hparam_plan(out)

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
                paths.extend(
                    [
                        out / "final_external_test.sh",
                        out / plan_hparam.FROZEN_FINAL_EVAL_CONFIG_NAME,
                    ]
                )
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
            out / plan_hparam.FROZEN_FINAL_EVAL_CONFIG_NAME,
        ]
        offset = next_run_index(recipe)
        scheduler = (recipe.get("execution") or {}).get("scheduler") or {}
        for idx, combo in enumerate(plan_hparam.hparam_combos(recipe)):
            identity = run_identity(recipe, offset + idx, combo)
            run_dir = out / "runs" / f"{identity['run_id']}--{identity['run_name']}"
            paths.extend(
                [run_dir / "launch.sh", run_dir / "config.yaml", run_dir / "run.json", run_dir / "artifacts.json"]
            )
            if scheduler.get("type") == "slurm":
                paths.extend(
                    [
                        run_dir / "job.sbatch",
                        run_dir / "slurm_terminal.json",
                        run_dir / "allocation_identity.json",
                        run_dir / "slurm.log",
                    ]
                )
        paths.append(out / "final_external_test.sh")
        return paths

    def index_summary_inputs_override(
        self, recipe: dict[str, Any], config_summary: dict[str, Any] | None
    ) -> tuple[list[Any], Any, list[Any]] | None:
        if recipe.get("task") != self.task:
            return None
        inputs = recipe.get("inputs") if isinstance(recipe.get("inputs"), dict) else {}
        split_values = finetune_loaded_split_values(recipe, load_test=False)
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
