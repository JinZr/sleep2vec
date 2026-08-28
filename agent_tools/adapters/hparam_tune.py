from __future__ import annotations

from pathlib import Path
import subprocess
from typing import Any

from .. import plan_contract, slurm
from ..decision_hparam import hparam_recipe_contract_issues, hparam_tune_issues
from ..decision_models import DecisionIssue, DecisionReport, DecisionStatus, ResolvedDecision, merge_status
from ..models import coerce_list
from ..plan_rendering import FINETUNE_RUNTIME_FIELDS, INFER_RUNTIME_FIELDS, finetune_loaded_split_values, variant_module
from .base import PlanRegistrationPreflightError, TaskAdapter


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

    def frozen_command_prefix(self, recipe: dict[str, Any]) -> tuple[str, ...]:
        execution = recipe.get("execution") if isinstance(recipe.get("execution"), dict) else {}
        return (str(execution.get("python") or "python"), "-m", variant_module(recipe, "finetune"))

    def section_contract_issues(self, recipe: dict[str, Any], *, source_layer: str) -> list[DecisionIssue] | None:
        return hparam_recipe_contract_issues(recipe, source_layer=source_layer)

    def bind_effective_recipe(
        self,
        recipe: dict[str, Any],
        config_summary: dict[str, Any] | None,
        *,
        source_recipe: dict[str, Any] | None = None,
    ) -> list[DecisionIssue]:
        source = source_recipe or recipe
        if isinstance(source.get("_local_recipe"), dict):
            source = source["_local_recipe"]
        authored_search = source.get("search") if isinstance(source.get("search"), dict) else {}
        if "profile" not in authored_search:
            return []
        from ..domain.finetune_hparam_profile import compile_finetune_balanced_profile

        compiled, issues = compile_finetune_balanced_profile(recipe, config_summary)
        if compiled is not None:
            recipe["search"] = compiled
        return issues

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
        self,
        recipe: dict[str, Any],
        config_summary: dict[str, Any] | None,
        *,
        unlock_final_test: bool,
        output_dir: Path | None = None,
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
        else:
            priority_type = capabilities["priority_type"] or "unknown priority policy"
            scheduler_type = capabilities["scheduler_type"] or "unknown scheduler"
            accounting_type = capabilities["accounting_storage_type"] or "unknown accounting storage"
            message = (
                f"Read-only Slurm inspection reports {priority_type}, {scheduler_type}, {accounting_type}, and "
                f"{capabilities['reservation_count']} visible reservation(s). "
            )
            if priority_type == "priority/basic":
                message += (
                    "Submit time and fitting credible short resource requests into backfill are the practical "
                    "user-side levers; no setting can guarantee first priority."
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
        run_index_offset: int | None = None,
        unlock_final_test: bool,
        source_config_bytes: bytes,
        source_config_sha256: str,
    ) -> None:
        from .. import plan_hparam
        from ..domain.finetune_hparam_profile import finetune_balanced_profile_audit

        search = recipe.get("search") if isinstance(recipe.get("search"), dict) else {}
        profile_audit = (
            finetune_balanced_profile_audit(search) if search.get("profile") == "finetune_balanced" else None
        )

        plan_hparam.write_hparam_plan(
            recipe,
            out,
            write_out=write_out,
            unlock_final_test=unlock_final_test,
            source_config_bytes=source_config_bytes,
            source_config_sha256=source_config_sha256,
            profile_audit=profile_audit,
            run_index_offset=run_index_offset,
        )

    def commit_plan(self, out: Path, *, preflight_validated: bool = False) -> None:
        from .. import plan_hparam

        try:
            plan_hparam.commit_hparam_plan(out, preflight_validated=preflight_validated)
        except plan_hparam.HparamRegistrationPreflightError as exc:
            raise PlanRegistrationPreflightError(str(exc)) from exc

    def registration_rows(self, plan: dict[str, Any]) -> list[dict[str, Any]]:
        from .. import plan_hparam

        return plan_hparam.hparam_manifest_rows(plan)

    def precommit_plan(self, out: Path, *, write_out: Path) -> str:
        from .. import plan_hparam

        try:
            return plan_hparam.preflight_hparam_plan(write_out, semantic_out=out)
        except OSError as exc:
            raise RuntimeError(f"Target execution preflight failed: {exc}") from exc

    def compile_plan_contract(
        self,
        recipe: dict[str, Any],
        out: Path,
        *,
        run_index_offset: int,
        config_bytes: bytes,
    ) -> dict[str, Any]:
        from .. import plan_contract, plan_hparam

        contracts = plan_hparam.compile_hparam_run_contracts(
            recipe,
            out,
            run_index_offset,
            source_config_bytes=config_bytes or None,
        )
        final_command = plan_hparam.compile_hparam_final_command(recipe, out)
        final_config_required = final_command is not None and plan_hparam.has_explicit_final_eval_config(recipe)
        final_snapshot = None
        if final_config_required:
            final_snapshot = plan_contract.frozen_input_snapshot(recipe, "inputs.final_eval_config_path")
            if final_snapshot["path"] != str((recipe.get("inputs") or {}).get("final_eval_config_path") or ""):
                raise ValueError("Frozen final evaluation config differs from its recipe path.")
        return {
            "runs": [contract["row"] for contract in contracts],
            "run_files": contracts,
            "launch_script_text": plan_hparam.compile_hparam_run_all_script(recipe, out),
            "final_command": final_command,
            "final_script_text": (
                plan_hparam.render_hparam_final_script(recipe, final_command) if final_command is not None else None
            ),
            "final_eval_config_required": final_config_required,
            "final_eval_config_sha256": final_snapshot["sha256"] if final_snapshot is not None else None,
        }

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
        from ..experiment_workspace import next_run_index

        if report.exit_code != 0:
            paths = plan_contract.blocked_plan_control_paths(out)
            evaluation = recipe.get("evaluation_policy") or {}
            if plan_hparam.final_test_unlocked(evaluation, unlock_final_test):
                paths.extend(
                    [
                        out / "final_external_test.sh",
                        out / plan_hparam.FROZEN_FINAL_EVAL_CONFIG_NAME,
                    ]
                )
            return paths
        paths = [
            out / "plan.json",
            out / "plan.md",
            out / "run_all.sh",
            out / "validation.sh",
            out / "execution_snapshot.json",
            out / "recipe.resolved.yaml",
            out / "config.source.yaml",
            out / plan_hparam.FROZEN_FINAL_EVAL_CONFIG_NAME,
        ]
        scheduler = (recipe.get("execution") or {}).get("scheduler") or {}
        for layout in plan_hparam.hparam_run_layouts(recipe, out, next_run_index(recipe)):
            run_dir = layout["run_dir"]
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
        split_values = finetune_loaded_split_values(recipe)
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
