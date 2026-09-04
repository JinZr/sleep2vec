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

from ..decision_models import DecisionIssue, DecisionReport, DecisionStatus, ResolvedDecision
from ..models import coerce_list
from ..plan_rendering import finetune_loaded_split_values


class PlanRegistrationPreflightError(ValueError):
    pass


def recipe_inputs(recipe: dict[str, Any]) -> dict[str, Any]:
    """The recipe's ``inputs`` mapping, or an empty one when absent or malformed."""
    inputs = recipe.get("inputs")
    return inputs if isinstance(inputs, dict) else {}


def config_summary_issues(recipe: dict[str, Any], config_summary: dict[str, Any] | None) -> list[DecisionIssue]:
    """Issues a task raises from its loaded config summary: the config's own
    blocking issues, plus an explicit recipe/config variant conflict."""
    if not config_summary:
        return []
    issues = [
        DecisionIssue(
            DecisionStatus.NEEDS_USER_INPUT,
            "config",
            issue,
            "Please fix the config before the agent generates commands.",
            {"config_path": config_summary.get("config_path")},
        )
        for issue in config_summary.get("blocking_issues", [])
    ]
    # Only the structural config-family marker is a routing gate; variant_guess can be path-derived.
    config_variant = config_summary.get("authoritative_variant")
    recipe_variant = recipe.get("variant")
    # An unresolved variant belongs to the consultation gate; only an explicit conflict is invalid.
    if config_variant is not None and recipe_variant not in (None, "", "ASK_USER") and recipe_variant != config_variant:
        issues.append(
            DecisionIssue(
                DecisionStatus.FAIL,
                "variant",
                f"Config family requires variant={config_variant}.",
                None,
                {"config_variant": config_variant, "recipe_variant": recipe_variant},
            )
        )
    return issues


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
    #: Public manager entrypoint for a generic task's single Slurm run.
    slurm_launch_subcommand: str | None = None
    #: Public manager entrypoint for a generic task's local detached run.
    direct_launch_subcommand: str | None = None
    #: Task accepts either a pretrain or finetune model config.
    accepts_pretrain_config: bool = False
    #: Run preflight_issues while consultation choices remain unresolved.
    preflight_on_unresolved: bool = False
    #: Task owns a read-only target runtime diagnostic for doctor.
    supports_doctor_runtime_diagnostics: bool = False

    def section_contract_issues(self, recipe: dict[str, Any], *, source_layer: str) -> list[DecisionIssue] | None:
        """Full replacement for the kernel's per-section recipe contract walk
        (task_recipe_contract_issues + execution_contract_issues); None means
        use the generic path."""
        return None

    def recipe_input_issues(self, recipe: dict[str, Any]) -> list[DecisionIssue]:
        """Known hard failures after decision materialization, without config or I/O."""
        return []

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

    def doctor_runtime_card(self, recipe: dict[str, Any]) -> str | None:
        """Return a read-only target runtime diagnostic for doctor."""
        return None

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

    def registration_rows(self, plan: dict[str, Any]) -> list[dict[str, Any]]:
        """Project frozen plan runs into their canonical registration rows."""
        return [
            {
                "parameter_summary": "single resolved recipe",
                **{key: value for key, value in run.items() if key != "command"},
            }
            for run in plan["runs"]
        ]

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
        from .. import plan_contract, plan_rendering, slurm

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
        contract = {
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
        if run.get("scheduler_type") == "slurm":
            execution = recipe["execution"]
            resources = slurm.normalize_resources(execution["scheduler"], execution.get("gpus_per_run", 1))
            run.update(command=commands[0], script_sha256=hashlib.sha256(contract["script_text"].encode()).hexdigest())
            token = slurm.submit_token(run, resources, execution["runtime_commit"])
            scheduler_text = slurm.render_batch_script(
                run=run,
                execution=execution,
                resources=resources,
                token=token,
                result_path=run["scheduler_result_path"],
                allocation_identity_path=run["allocation_identity_path"],
                execution_snapshot_path=out / "execution_snapshot.json",
                log_path=run["log_path"],
                module=self.frozen_command_prefix(recipe)[2],
            )
            run.update(
                scheduler_submit_token=token,
                scheduler_script_sha256=hashlib.sha256(scheduler_text.encode()).hexdigest(),
            )
            contract["scheduler_script_text"] = scheduler_text
        launch_subcommand = None
        if run.get("scheduler_type") == "slurm":
            launch_subcommand = self.slurm_launch_subcommand
        elif run.get("scheduler_type") == "direct":
            launch_subcommand = self.direct_launch_subcommand
        if launch_subcommand:
            manager_command = plan_rendering.render_command(
                [
                    plan_contract.frozen_plan_context(recipe)["python"],
                    "-m",
                    "agent_tools",
                    launch_subcommand,
                    "--plan-dir",
                    out,
                ]
            )
            contract["launch_script_text"] = (
                "\n".join(
                    plan_rendering.script_lines(
                        [manager_command + ' "$@"'], run_cwd=plan_contract.frozen_plan_context(recipe)["repo_root"]
                    )
                )
                + "\n"
            )
        return contract

    def validation_commands(self, recipe: dict[str, Any]) -> list[str] | None:
        """Full replacement for the kernel's generic validation command list;
        None means use the generic path."""
        return None

    def expected_artifacts(self, recipe: dict[str, Any], config_summary: dict[str, Any] | None) -> list[dict[str, str]]:
        """Expected output artifacts for context/plan documents."""
        return []

    def effective_preset_path(self, recipe: dict[str, Any], config_summary: dict[str, Any] | None) -> Any:
        """The preset this task actually loads: the recipe's declared
        ``preset_path_recipe_field`` override when concrete, else the config's
        ``finetune_preset_path``. None when neither is."""
        if self.preset_path_recipe_field is not None:
            value = recipe_inputs(recipe).get(self.preset_path_recipe_field)
            if value not in (None, "", "ASK_USER"):
                return value
        if not self.uses_finetune_config:
            return None
        value = ((config_summary or {}).get("data") or {}).get("finetune_preset_path")
        return value if value not in (None, "", "ASK_USER") else None

    def index_summary_split_values(self, recipe: dict[str, Any]) -> list[Any]:
        """Splits the index summary reports for this task. The default is the
        finetune family's loaded splits; tasks that name their evaluation split
        in the recipe override it."""
        return finetune_loaded_split_values(recipe)

    def index_summary_inputs_override(
        self, recipe: dict[str, Any], config_summary: dict[str, Any] | None
    ) -> tuple[list[Any], Any, list[Any]] | None:
        """(index_paths, config, split_values) when this adapter claims the
        recipe/config combination, else None. Claiming is by config shape,
        not task name, so config-probing adapters must be registered before
        task-keyed ones (registration order is the probing order).

        The default covers the finetune-family config shape shared by finetune,
        hparam_tune, infer, and evaluate: an effective preset carries its own
        records, so no index path is reported; otherwise the config's
        ``finetune_data_index`` applies. Tasks that do not read that config
        claim nothing.
        """
        if not self.uses_finetune_config or recipe.get("task") != self.task:
            return None
        inputs = recipe_inputs(recipe)
        split_values = self.index_summary_split_values(recipe)
        if self.effective_preset_path(recipe, config_summary) is not None:
            return [], inputs.get("config"), split_values
        data = (config_summary or {}).get("data") or {}
        return coerce_list(data.get("finetune_data_index")), inputs.get("config"), split_values
