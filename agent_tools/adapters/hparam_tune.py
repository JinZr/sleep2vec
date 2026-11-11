"""The ``hparam_tune`` task adapter.

Layer 1, domain, and the single composite adapter: its ``base_task`` is
``finetune``, so recipes carrying ``_base_recipe`` / ``_local_recipe`` layers
close under its contract.

Binds the hparam decision contract and compiles the plan contract for
hyper-parameter plans.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from .. import execution_snapshot, experiment_workspace, plan_contract, plan_hparam
from ..decision_hparam import hparam_recipe_contract_issues, hparam_search_issues, hparam_tune_issues
from ..decision_models import DecisionIssue, DecisionReport, DecisionStatus, ResolvedDecision
from ..models import BoundFinalEvalConfigSnapshot, ConfigSummaryInput, JsonValue
from ..plan_rendering import FINETUNE_RUNTIME_FIELDS, INFER_RUNTIME_FIELDS, variant_module
from .base import TaskAdapter
from .finetune import FINETUNE_ADAPTER


class HparamTuneAdapter(TaskAdapter):
    task = "hparam_tune"

    recipe_extra_fields = frozenset(
        {"artifacts", "base_recipe", "evaluation_policy", "execution", "inputs", "runtime", "search"}
    )
    artifact_fields = frozenset({"overwrite", "results_csv_path"})
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

    def doctor_runtime_card(self, recipe: dict[str, Any]) -> str | None:
        execution = recipe.get("execution")
        if not isinstance(execution, dict):
            execution = {}
        python = str(execution.get("python") or sys.executable)
        program = (
            "import importlib.metadata, json, socket, sys; "
            "pl_version = next((dist.version for dist in importlib.metadata.distributions() "
            "if str(dist.metadata.get('Name', '')).lower() == 'pytorch-lightning'), 'unavailable'); "
            "print(json.dumps({'host': socket.gethostname(), 'python': sys.executable, "
            "'python_version': sys.version.split()[0], 'pytorch_lightning_version': pl_version}, sort_keys=True))"
        )
        try:
            result = execution_snapshot.run_execution_command(execution, [python, "-c", program])
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
        return (
            f"Doctor runtime: host={payload['host']}, python={payload['python']}, "
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
        final_eval_config: BoundFinalEvalConfigSnapshot | None,
    ) -> None:
        plan_hparam.write_hparam_plan(
            recipe,
            out,
            write_out=write_out,
            unlock_final_test=unlock_final_test,
            source_config_bytes=source_config_bytes,
            source_config_sha256=source_config_sha256,
            final_eval_config=final_eval_config,
            run_index_offset=run_index_offset,
        )

    def registration_rows(self, plan: dict[str, Any]) -> list[dict[str, Any]]:
        return plan_hparam.hparam_manifest_rows(plan)

    def compile_plan_contract(
        self,
        recipe: dict[str, Any],
        out: Path,
        *,
        run_index_offset: int,
        config_bytes: bytes,
    ) -> plan_contract.HparamCompiledPlanContract:
        return plan_hparam.compile_hparam_plan_contract(
            recipe, out, run_index_offset=run_index_offset, config_bytes=config_bytes
        )

    def planned_plan_paths(
        self,
        recipe: dict[str, JsonValue],
        out: Path,
        report: DecisionReport,
        *,
        allow_unresolved: bool,
        unlock_final_test: bool,
    ) -> list[Path] | None:
        if report.exit_code != 0:
            paths = plan_contract.blocked_plan_control_paths(out)
            evaluation_value = recipe.get("evaluation_policy")
            evaluation = evaluation_value if isinstance(evaluation_value, dict) else {}
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
        execution_value = recipe.get("execution")
        execution = execution_value if isinstance(execution_value, dict) else {}
        scheduler_value = execution.get("scheduler")
        scheduler = scheduler_value if isinstance(scheduler_value, dict) else {}
        for layout in plan_hparam.hparam_run_layouts(recipe, out, experiment_workspace.next_run_index(recipe)):
            run_dir = layout["run_dir"]
            paths.extend(
                [run_dir / "launch.sh", run_dir / "config.yaml", run_dir / "run.json", run_dir / "artifacts.json"]
            )
            if scheduler.get("type") == "slurm":
                paths.extend([run_dir / "job.sbatch", run_dir / "slurm.log"])
        paths.append(out / "final_external_test.sh")
        return paths


HPARAM_TUNE_ADAPTER = HparamTuneAdapter()
