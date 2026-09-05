from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from .. import plan_contract, slurm
from ..decision_hparam import hparam_recipe_contract_issues, hparam_search_issues, hparam_tune_issues
from ..decision_models import DecisionIssue, DecisionReport, DecisionStatus, ResolvedDecision, merge_status
from ..plan_rendering import FINETUNE_RUNTIME_FIELDS, INFER_RUNTIME_FIELDS, variant_module
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
    supports_doctor_runtime_diagnostics = True
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

    def recipe_input_issues(self, recipe: dict[str, Any]) -> list[DecisionIssue]:
        search = recipe.get("search") if isinstance(recipe.get("search"), dict) else {}
        return [
            issue
            for issue in hparam_search_issues(search, profile_mode="profile" in search, high_impact={})
            if issue.status == DecisionStatus.FAIL
        ]

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
        effective_search = recipe.get("search") if isinstance(recipe.get("search"), dict) else {}
        if "profile" not in authored_search or "profile" not in effective_search:
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
        from .. import plan_hparam

        execution = recipe.get("execution") if isinstance(recipe.get("execution"), dict) else {}
        scheduler = execution.get("scheduler") if isinstance(execution.get("scheduler"), dict) else {}
        if scheduler.get("type") != "slurm" or report.blocking_issues():
            return report
        try:
            capabilities = slurm.cluster_scheduling_capabilities(execution, str(scheduler["partition"]))
        except (slurm.SlurmCommandError, subprocess.TimeoutExpired, ValueError) as exc:
            capability_issue = DecisionIssue(
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
            capability_issue = DecisionIssue(
                DecisionStatus.WARN,
                "execution.scheduler.capabilities",
                message,
                None,
                capabilities,
            )
        planned_runs = len(plan_hparam.hparam_combos(recipe))
        resources = slurm.normalize_resources(scheduler, execution.get("gpus_per_run", 1))
        try:
            capacity = slurm.fixed_node_resource_capacity(execution, resources, planned_runs)
        except (slurm.SlurmCommandError, subprocess.TimeoutExpired, ValueError) as exc:
            capacity = {"status": "unknown", "reason": str(exc), "planned_runs": planned_runs}
            capacity_issue = DecisionIssue(
                DecisionStatus.WARN,
                "execution.scheduler.capacity",
                f"Slurm fixed-node resource capacity unknown: read-only inspection was unavailable: {exc}",
                None,
                capacity,
            )
        else:
            if capacity["status"] == "unknown":
                capacity_issue = DecisionIssue(
                    DecisionStatus.WARN,
                    "execution.scheduler.capacity",
                    f"Slurm fixed-node resource capacity unknown: {capacity['reason']}.",
                    None,
                    capacity,
                )
            else:
                per_run = capacity["per_run"]
                limits = capacity["limits"]
                limiting = ", ".join(capacity["limiting_resources"])
                resource_word = "resource" if len(capacity["limiting_resources"]) == 1 else "resources"
                details = (
                    f"Per run: {per_run['gpus']} GPUs, {per_run['cpus']} CPUs, {per_run['memory']}. "
                    "Empty-node theoretical limits if co-resident: "
                    f"GPU={limits['gpu']}, CPU={limits['cpu']}, memory={limits['memory']}, "
                    f"overall={capacity['overall_empty_node_limit']}; limiting {resource_word}: {limiting}."
                )
                if capacity["overall_empty_node_limit"] == 0:
                    capacity_issue = DecisionIssue(
                        DecisionStatus.FAIL,
                        "execution.scheduler.capacity",
                        f"Fixed node {capacity['node']}: one job cannot fit. {details}",
                        None,
                        capacity,
                    )
                elif planned_runs > capacity["overall_empty_node_limit"]:
                    capacity_issue = DecisionIssue(
                        DecisionStatus.WARN,
                        "execution.scheduler.capacity",
                        (
                            f"Fixed node {capacity['node']} has an empty-node theoretical limit of "
                            f"{capacity['overall_empty_node_limit']} run(s); {planned_runs} planned run(s), "
                            f"if co-resident, require at least {capacity['minimum_waves']} waves. {details}"
                        ),
                        None,
                        capacity,
                    )
                else:
                    capacity_issue = DecisionIssue(
                        DecisionStatus.PASS,
                        "execution.scheduler.capacity",
                        (
                            f"All {planned_runs} planned run(s) fit if co-resident on otherwise empty fixed node "
                            f"{capacity['node']}. {details}"
                        ),
                        None,
                        capacity,
                    )
        issues = [*report.issues, capability_issue, capacity_issue]
        return DecisionReport(status=merge_status(issues), issues=issues, decisions=report.decisions)

    def doctor_runtime_card(self, recipe: dict[str, Any]) -> str | None:
        from .. import managed_scheduler

        execution = recipe.get("execution") if isinstance(recipe.get("execution"), dict) else {}
        python = str(execution.get("python") or sys.executable)
        program = (
            "import importlib.metadata, json, socket, sys; "
            "pl_version = next((dist.version for dist in importlib.metadata.distributions() "
            "if str(dist.metadata.get('Name', '')).lower() == 'pytorch-lightning'), 'unavailable'); "
            "print(json.dumps({'host': socket.gethostname(), 'python': sys.executable, "
            "'python_version': sys.version.split()[0], 'pytorch_lightning_version': pl_version}, sort_keys=True))"
        )
        try:
            result = managed_scheduler.run_execution_command(execution, [python, "-c", program])
        except subprocess.TimeoutExpired:
            return "Doctor runtime unavailable: diagnostic probe timed out"
        except (OSError, ValueError):
            return "Doctor runtime unavailable: diagnostic probe could not start"
        if result.returncode != 0:
            return f"Doctor runtime unavailable: diagnostic probe exited with code {result.returncode}"
        try:
            payload = json.loads(result.stdout)
        except (json.JSONDecodeError, TypeError):
            return "Doctor runtime unavailable: malformed diagnostic output"
        fields = ("host", "python", "python_version", "pytorch_lightning_version")
        if not isinstance(payload, dict) or any(
            not isinstance(payload.get(field), str) or not payload[field] for field in fields
        ):
            return "Doctor runtime unavailable: incomplete diagnostic output"
        target = str(execution.get("target") or "local")
        if execution.get("host"):
            target += f":{execution['host']}"
        return (
            f"Doctor runtime: transport={target}, host={payload['host']}, python={payload['python']}, "
            f"python_version={payload['python_version']}, "
            f"pytorch-lightning={payload['pytorch_lightning_version']}"
        )

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


HPARAM_TUNE_ADAPTER = HparamTuneAdapter()
