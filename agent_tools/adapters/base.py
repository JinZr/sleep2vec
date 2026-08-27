"""Structured per-task boundary for the agent_tools kernel.

Layering contract (import directions are one-way):

- Layer 0 (leaf modules adapters MAY import): models, decision_models,
  transport, plan_rendering, decision_paths, gpu_rules, decision_hparam,
  plan_hparam, experiment_workspace, manifests, repo, slurm.
- Layer 1 (this package): adapters/base.py, adapters/<task>.py,
  adapters/registry.py.
- Layer 2 (kernel orchestration, imports the registry): configs,
  decision_rules, decisions, plan_context, plans.

Adapters must never import layer-2 modules. decision_paths is layer 0 and
must never import the registry -- task-specific dispatch that used to live
there is hoisted into decisions.py instead.

For the full module ownership map (kernel vs domain vs mixed bridges), the CLI
command triage, and the tolerated reverse edges, see ../ARCHITECTURE.md and the
machine-readable partition in ../layering.py.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

from ..decision_models import DecisionIssue, DecisionReport, ResolvedDecision


class PlanRegistrationPreflightError(ValueError):
    pass


class TaskAdapter:
    """One agent task's structured boundary.

    Declarative members are class attributes; behavioural hooks are methods
    with safe defaults ([] / None / False means "fall back to the kernel's
    generic path"). Kernel dispatch points resolve the adapter through
    adapters.registry and never hard-code task names.
    """

    #: Registry key; must equal the recipe's ``task`` value.
    task: str
    #: False requires membership in models.VARIANTLESS_TASKS (asserted at
    #: registry import time).
    requires_variant: bool = True

    #: Top-level recipe fields allowed beyond the kernel's common set.
    recipe_extra_fields: frozenset[str] = frozenset()
    #: Allowed ``artifacts.*`` fields.
    artifact_fields: frozenset[str] = frozenset()
    #: Section name -> allowed fields. A missing section key means the kernel
    #: does not validate that section for this task.
    contract_sections: Mapping[str, frozenset[str]] = {}
    #: Decision fields allowed beyond the consultation policy's
    #: required_for_tasks entries.
    extra_decision_fields: frozenset[str] = frozenset()
    #: Decision field -> (recipe section, field) materialization target.
    #: Only declared fields are (re)targeted; kernel defaults (e.g.
    #: overwrite_policy -> (artifacts, overwrite)) apply otherwise.
    decision_recipe_targets: Mapping[str, tuple[str, str]] = {}
    #: Variants rejected with FAIL "{variant} does not support {task}.".
    unsupported_variants: frozenset[str] = frozenset()
    #: True/False forces the survival-sidecar requirement for this task;
    #: None keeps the kernel's own inference (decision_paths).
    requires_survival_sidecars: bool | None = None
    #: True/False forces the multilabel-sidecar requirement for this task;
    #: None keeps the kernel's own inference (decision_paths).
    requires_multilabel_sidecars: bool | None = None
    #: Recipe inputs field holding this task's preset path override
    #: (e.g. inference_preset_path); None means the task has no
    #: recipe-level preset override and the kernel's config fallback applies.
    preset_path_recipe_field: str | None = None
    #: True enables path_issues' dataset-source existence checks (npz
    #: effective preset/index; sex_age kaldi data root/manifest).
    validates_dataset_paths: bool = False
    #: Composite task's base-layer task name; non-None means recipes may
    #: carry two layers (_base_recipe/_local_recipe), the base layer closes
    #: under this task's contract and the kernel runs a recursive base gate.
    base_task: str | None = None
    #: Task consumes datasets through the finetune-family config: the
    #: finetune_preset_path fallback, survival/multilabel sidecar inference,
    #: and the explicit config-decision check apply.
    uses_finetune_config: bool = False
    #: required_channels decision vs config preset_build consistency check.
    enforces_required_channels: bool = False
    #: Task writes its own complete plan bundle (multi-run) via write_plan;
    #: the kernel skips the generic single-run materialization and the flat
    #: command-emptiness preflight check.
    materializes_plan: bool = False
    #: Task accepts a frozen Python/workdir/commit execution identity.
    supports_runtime_identity: bool = False
    #: Task accepts either a pretrain or finetune model config.
    accepts_pretrain_config: bool = False
    #: Run preflight_issues while consultation choices remain unresolved.
    preflight_on_unresolved: bool = False

    def section_contract_issues(self, recipe: dict[str, Any], *, source_layer: str) -> list[DecisionIssue] | None:
        """Full replacement for the kernel's per-section recipe contract walk
        (task_recipe_contract_issues + execution_contract_issues); None means
        use the generic path."""
        return None

    def config_override_issues(
        self, recipe: dict[str, Any], config_summary: dict[str, Any] | None
    ) -> list[DecisionIssue] | None:
        """None: the kernel runs its generic flat config-contract block.
        Non-None: the kernel skips that block and appends these issues at the
        original override position (after index issues)."""
        return None

    def bind_effective_recipe(
        self,
        recipe: dict[str, Any],
        config_summary: dict[str, Any] | None,
        *,
        source_recipe: dict[str, Any] | None = None,
    ) -> list[DecisionIssue]:
        """Bind config-owned fields into the in-memory effective recipe.

        Domain adapters may defer a domain-leaf compiler import inside this
        hook; generic adapters must keep it domain-free.
        """
        return []

    def preflight_issues(
        self,
        recipe: dict[str, Any],
        config_summary: dict[str, Any] | None,
        *,
        unlock_final_test: bool,
        output_dir: Path | None = None,
    ) -> list[DecisionIssue]:
        """Extra issues evaluated only during preflight_plan (not doctor)."""
        return []

    def prepare_doctor_report(self, recipe: dict[str, Any], report: DecisionReport) -> DecisionReport:
        """Add task-specific read-only doctor findings."""
        return report

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
        """Materialize the full plan bundle; ``out`` is the final semantic
        root and ``write_out`` may be a physical staging root."""
        raise NotImplementedError

    def commit_plan(self, out: Path, *, preflight_validated: bool = False) -> None:
        """Register a fully materialized plan; called only when
        materializes_plan is True."""
        raise NotImplementedError

    def precommit_plan(self, out: Path, *, write_out: Path) -> str | None:
        """Validate a materialized plan before publication or registration."""
        return None

    def planned_plan_paths(
        self,
        recipe: dict[str, Any],
        out: Path,
        report: DecisionReport,
        *,
        allow_unresolved: bool,
        unlock_final_test: bool,
    ) -> list[Path] | None:
        """Full replacement for the kernel's expected-output path list
        (both the blocked and success branches); None means use the generic
        single-run path list."""
        return None

    def managed_runtime_dir(self, recipe: dict[str, Any], version: str) -> Path | None:
        """Externally-managed runtime directory for a planned managed run;
        None means the kernel records empty runtime/checkpoint dirs."""
        return None

    def required_input_paths(self, recipe: dict[str, Any]) -> list[tuple[str, Any]]:
        """Task-specific required input paths, validated by
        decision_paths.path_issues; passed through decisions.py because
        decision_paths cannot import the registry."""
        return []

    def frozen_input_paths(self, recipe: dict[str, Any]) -> list[tuple[str, Path]]:
        """Local external inputs whose content identity is frozen in the plan."""
        return []

    def runtime_fields(self, variant: Any) -> frozenset[str]:
        """Allowed ``runtime.*`` fields; variant-sensitive for some tasks."""
        return frozenset()

    def frozen_command_prefix(self, recipe: dict[str, Any]) -> tuple[str, ...]:
        """Task-owned prefix required for every command in a frozen plan."""
        raise NotImplementedError

    def matches_config_data(self, data: dict[str, Any]) -> bool:
        """Whether a loaded config mapping belongs to this task's domain."""
        return False

    def config_summary(self, config_path: str | Path) -> dict[str, Any]:
        """Structured summary of a domain config. Domain-leaf imports used by
        adapter hooks must stay inside the method body (deferred)."""
        raise NotImplementedError

    def task_issues(
        self,
        recipe: dict[str, Any],
        config_summary: dict[str, Any] | None,
        decisions: dict[str, ResolvedDecision],
        high_impact: dict[str, dict[str, Any]],
    ) -> list[DecisionIssue]:
        """Task-specific consultation issues (adapters may ignore arguments
        they do not need; the signature is uniform across tasks)."""
        return []

    def configured_input_issues(
        self, recipe: dict[str, Any], config_summary: dict[str, Any] | None
    ) -> list[DecisionIssue]:
        """Existence checks for task-specific configured input paths."""
        return []

    def commands(self, recipe: dict[str, Any], config_summary: dict[str, Any] | None) -> list[str]:
        """Runnable commands for this task; [] means the recipe cannot be
        rendered (the kernel reports it as unsupported)."""
        return []

    def frozen_commands(self, recipe: dict[str, Any], config_bytes: bytes) -> list[str]:
        """Rebuild commands from a frozen plan-owned config snapshot."""
        return self.commands(recipe, None)

    def compile_plan_contract(
        self,
        recipe: dict[str, Any],
        out: Path,
        *,
        run_index_offset: int,
        config_bytes: bytes,
    ) -> dict[str, Any]:
        from .. import plan_contract

        frozen_inputs = plan_contract.frozen_input_snapshots(recipe)
        source_config = plan_contract.resolve_frozen_repo_path(recipe, (recipe.get("inputs") or {}).get("config"))
        if source_config is None:
            raise ValueError("Frozen generic input snapshots differ from required recipe inputs.")
        expected_input_paths = [("inputs.config", str(source_config))]
        expected_input_paths.extend((field, str(path)) for field, path in self.frozen_input_paths(recipe))
        expected_input_paths.sort()
        frozen_input_paths = [(snapshot["field"], snapshot["path"]) for snapshot in frozen_inputs]
        if frozen_input_paths != expected_input_paths:
            raise ValueError("Frozen generic input snapshots differ from required recipe inputs.")
        config_snapshot = next(snapshot for snapshot in frozen_inputs if snapshot["field"] == "inputs.config")
        config_sha256 = hashlib.sha256(config_bytes).hexdigest()
        if config_sha256 != config_snapshot["sha256"]:
            raise ValueError("Frozen generic config differs from its recipe digest.")
        run = plan_contract.generic_run_contract(recipe, out, run_index_offset, self)
        input_snapshots = [snapshot for snapshot in frozen_inputs if snapshot["field"] != "inputs.config"]
        run["config_sha256"] = config_sha256
        if input_snapshots:
            run["input_snapshots"] = input_snapshots
        commands = plan_contract.generic_commands(recipe, run, self, config_bytes)
        return {
            "runs": [run],
            "commands": commands,
            "script_text": plan_contract.generic_script_text(
                recipe,
                run,
                self,
                commands,
                input_snapshots,
            ),
        }

    def validation_commands(self, recipe: dict[str, Any]) -> list[str] | None:
        """Full replacement for the kernel's generic validation command list;
        None means use the generic path."""
        return None

    def expected_artifacts(self, recipe: dict[str, Any], config_summary: dict[str, Any] | None) -> list[dict[str, str]]:
        """Expected output artifacts for context/plan documents."""
        return []

    def index_summary_inputs_override(
        self, recipe: dict[str, Any], config_summary: dict[str, Any] | None
    ) -> tuple[list[Any], Any, list[Any]] | None:
        """(index_paths, config, split_values) when this adapter claims the
        recipe/config combination, else None. Claiming is by config shape,
        not task name, so config-probing adapters must be registered before
        task-keyed ones (registration order is the probing order)."""
        return None
