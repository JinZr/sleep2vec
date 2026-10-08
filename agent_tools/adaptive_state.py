"""Durable state of an adaptive hparam workflow, and its validation and recovery.

Owns the ``adaptive/`` layout under a workflow root: round directories, the
frozen ``workflow.json`` payload, the append-only ``run_registry.tsv``, the
workflow's experiment events, and reconciliation of a launch that was
interrupted before its round was committed, plus the recipe's adaptive settings,
suggest strategy and workflow objective accessors and the round-terminal check
over that state. ``adaptive_hparam`` drives the
digest, proposal, registration and launch steps on top of this state.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import csv
import io
import json
import os
from pathlib import Path
from typing import Any, Literal, TypedDict, TypeVar

from . import (
    adaptive_proposals,
    experiment_io as exp_io,
    experiment_workspace,
    hparam_runtime,
    managed_scheduler,
    plan_hparam,
    run_artifacts as artifacts,
    run_evidence as evidence,
)
from .decision_hparam import DEFAULT_ADAPTIVE_SUGGEST_STRATEGY
from .experiment_workspace import (
    TERMINAL_STATUSES,
    AdaptiveEventPayload,
    AdaptiveInitEvent,
    PlanCreatedEvent,
    append_event as _write_experiment_event,
    event_matches,
    experiment_root,
    managed_run_key,
    merge_run_manifest,
    plan_registration_rows_state,
    read_experiment_events,
    read_run_manifest,
    scheduler_direct_controller,
    scheduler_type,
    validate_frozen_run_update,
    validate_managed_run_rows,
    validated_run_key,
)
from .manifests import read_json, read_rows, utc_now, validate_managed_header
from .models import REPO_ROOT, JsonValue

EXECUTION_IDENTITY_FIELDS = ("python", "runtime_commit")


FROZEN_EXECUTION_IDENTITY_FIELDS = ("python",)


EXECUTION_ROUTE_FIELDS = (
    "target",
    "host",
    "workdir",
    "conda_env",
    "scheduler.type",
    "scheduler.direct_controller",
    "scheduler.partition",
    "scheduler.nodelist",
)


_SLURM_ACCEPTED_STATUSES = {"queued", "running", "stopping", "completed", "finished", "failed", "stopped"}


def _is_accepted_start(row: Mapping[str, JsonValue]) -> bool:
    if scheduler_type(row) == "direct":
        return row.get("status") in {"launched", "running"}
    job_id = str(row.get("scheduler_job_id") or "")
    return row.get("status") in _SLURM_ACCEPTED_STATUSES and job_id.isdigit() and int(job_id) > 0


def accepted_start_keys(rows: Sequence[Mapping[str, JsonValue]]) -> set[tuple[str, str]]:
    return {validated_run_key(row) for row in rows if _is_accepted_start(row)}


class InitialAdaptiveWorkflow(TypedDict):
    recipe_path: str
    execution_identity: dict[str, Any]
    root: str
    external_optimized: Literal[True]
    objective_metric: str
    objective_mode: str


_WorkflowPayload = TypeVar("_WorkflowPayload", bound=InitialAdaptiveWorkflow | dict[str, Any])


def append_event(root: Path, event_type: str, payload: dict[str, Any]) -> None:
    workflow_path = root / "adaptive" / "workflow.json"
    target = root
    if workflow_path.exists():
        target = workflow_workspace(root)
    _write_experiment_event(target, event_type, payload)


def workflow_workspace(root: Path) -> Path:
    initial_plan = hparam_runtime.read_hparam_plan_under_run_lock(round_path(root, 0))
    recipe_value = initial_plan.get("recipe")
    recipe = recipe_value if isinstance(recipe_value, dict) else {}
    workspace = experiment_root(recipe)
    if workspace is None:
        raise ValueError("Adaptive workflow is not bound to an experiment workspace.")
    return workspace


def read_workflow(root: Path) -> dict[str, Any]:
    path = root / "adaptive" / "workflow.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing adaptive workflow: {path}")
    workflow = read_json(path)
    return validate_workflow_payload(root, workflow)


def adaptive_settings(recipe: dict[str, Any]) -> dict[str, Any]:
    adaptive = recipe.get("adaptive")
    return adaptive if isinstance(adaptive, dict) else {}


def workflow_objective(root: Path, recipe: dict[str, Any]) -> adaptive_proposals.ProposalObjective:
    workflow = read_workflow(root) if (root / "adaptive" / "workflow.json").exists() else {}
    adaptive = adaptive_settings(recipe)
    return {
        "metric": str(workflow.get("objective_metric") or adaptive.get("objective_metric") or "test_auroc"),
        "mode": str(workflow.get("objective_mode") or adaptive.get("objective_mode") or "max"),
    }


def suggest_strategy(recipe: dict[str, Any]) -> str:
    suggest = adaptive_settings(recipe).get("suggest")
    if not isinstance(suggest, dict):
        return DEFAULT_ADAPTIVE_SUGGEST_STRATEGY
    return str(suggest.get("strategy", DEFAULT_ADAPTIVE_SUGGEST_STRATEGY))


def round_is_terminal(
    round_dir: Path, workspace: Path, *, read_run_manifest: Callable[[Path], list[dict[str, str]]]
) -> bool:
    # Launches and run scripts replace run_manifest.tsv under the run lock; no adaptive caller holds it here.
    with experiment_workspace.managed_run_lock(workspace):
        plan = artifacts.read_hparam_plan(round_dir)
        canonical_by_key = {managed_run_key(row): row for row in read_run_manifest(workspace)}
    run_keys = [managed_run_key(run) for run in plan.get("runs", [])]
    return bool(run_keys) and all(canonical_by_key.get(key, {}).get("status") in TERMINAL_STATUSES for key in run_keys)


def validate_workflow_payload(
    root: Path,
    workflow: _WorkflowPayload,
    *,
    require_adaptive_commit: bool = True,
    registry_rows: Sequence[Mapping[str, str]] | None = None,
) -> _WorkflowPayload:
    path = root / "adaptive" / "workflow.json"
    if not isinstance(workflow, dict):
        raise ValueError(f"Adaptive workflow must contain a mapping: {path}")
    if str(workflow.get("root") or "") != str(root):
        raise ValueError(f"Adaptive workflow root differs from the requested workspace: {root}")
    recipe_path = Path(str(workflow.get("recipe_path") or ""))
    if not recipe_path.is_absolute():
        raise ValueError(f"Adaptive workflow recipe_path must be absolute: {path}")
    execution_identity = workflow.get("execution_identity")
    if (
        not isinstance(execution_identity, dict)
        or set(execution_identity) != set(EXECUTION_IDENTITY_FIELDS)
        or any(execution_identity.get(field) in (None, "") for field in EXECUTION_IDENTITY_FIELDS)
    ):
        raise ValueError(f"Adaptive workflow lacks frozen execution identity: {path}")
    legacy_registry = root / "adaptive" / "trial_registry.tsv"
    if legacy_registry.exists():
        raise ValueError(f"Legacy adaptive registry is read-only and cannot be managed: {legacy_registry}")
    registry_path = root / "adaptive" / "run_registry.tsv"
    if registry_rows is None:
        if not registry_path.exists():
            raise FileNotFoundError(f"Missing adaptive run registry: {registry_path}")
        registry_rows = read_rows(registry_path, require_managed_identity=True)
    validate_managed_run_rows(registry_rows, source=str(registry_path), cardinality="one_per_run")
    round_index = latest_round_index(root) if require_adaptive_commit else 0
    round_dir = round_path(root, round_index)
    frozen_plan = artifacts.read_hparam_plan(round_dir, require_workspace_state=False, require_adaptive_commit=False)
    workspace = experiment_root(frozen_plan["recipe"])
    assert workspace is not None  # read_hparam_plan rejects a plan without experiment.root.
    # Launches and run scripts replace run_manifest.tsv under the run lock; no adaptive caller holds it here.
    with experiment_workspace.managed_run_lock(workspace):
        plan = artifacts.read_hparam_plan(round_dir, require_adaptive_commit=require_adaptive_commit)
        recipe_value = plan.get("recipe")
        recipe = recipe_value if isinstance(recipe_value, dict) else {}
        plan_execution_value = recipe.get("execution")
        plan_execution = plan_execution_value if isinstance(plan_execution_value, dict) else {}
        if any(plan_execution.get(field) != execution_identity[field] for field in FROZEN_EXECUTION_IDENTITY_FIELDS):
            raise ValueError(f"Adaptive workflow execution identity differs from the current round plan: {round_dir}")
        initial_plan = plan if round_index == 0 else artifacts.read_hparam_plan(round_path(root, 0))
        initial_recipe_value = initial_plan.get("recipe")
        initial_recipe = initial_recipe_value if isinstance(initial_recipe_value, dict) else {}
        initial_execution_value = initial_recipe.get("execution")
        initial_execution = initial_execution_value if isinstance(initial_execution_value, dict) else {}
        if any(initial_execution.get(field) != execution_identity[field] for field in EXECUTION_IDENTITY_FIELDS):
            raise ValueError(f"Adaptive workflow baseline execution identity differs from round 000: {path}")
        frozen_route = execution_route(initial_execution)
        current_route = execution_route(plan_execution)
        changed_route = [field for field in EXECUTION_ROUTE_FIELDS if current_route[field] != frozen_route[field]]
        if changed_route:
            raise ValueError(f"Adaptive workflow execution route differs from round 000: {', '.join(changed_route)}")
        canonical_by_key = {managed_run_key(row): row for row in read_run_manifest(workspace)}
    for registered in registry_rows:
        canonical = canonical_by_key.get(managed_run_key(registered))
        if canonical is None:
            raise ValueError(
                f"Adaptive registry row is outside the canonical manifest: "
                f"{registered.get('step_id', '')} / {registered.get('run_id', '')}"
            )
        validate_frozen_run_update(canonical, registered)
    validate_round_registry(root, round_index, plan, registry_rows)
    return workflow


def validate_round_registry(
    root: Path,
    round_index: int,
    plan: Mapping[str, Any],
    registry_rows: Sequence[Mapping[str, str]],
) -> None:
    round_dir = round_path(root, round_index)
    registry_by_key = {validated_run_key(row): row for row in registry_rows}
    plan_runs = plan.get("runs", [])
    plan_keys = {validated_run_key(run) for run in plan_runs}
    for run in plan_runs:
        key = validated_run_key(run)
        registered = registry_by_key.get(key)
        if registered is None:
            raise ValueError(f"Adaptive registry is missing the current plan run: {key[0]} / {key[1]}")
        if str(registered.get("round") or "") != str(round_index) or str(registered.get("round_dir") or "") != str(
            round_dir
        ):
            raise ValueError(f"Adaptive registry round binding differs for run: {key[0]} / {key[1]}")
        validate_frozen_run_update(run, registered)
    registered_keys = {
        validated_run_key(row) for row in registry_rows if str(row.get("round") or "") == str(round_index)
    }
    if registered_keys != plan_keys:
        raise ValueError(f"Adaptive registry has runs outside the current plan: {round_dir}")


def _parse_registry(text: str | None, path: Path) -> list[dict[str, str]]:
    if text is None:
        raise ValueError(f"Managed table is not valid UTF-8: {path}")
    try:
        reader = csv.DictReader(io.StringIO(text), delimiter="\t", strict=True)
        fieldnames = reader.fieldnames
        if not fieldnames:
            raise ValueError(f"Managed table has no header: {path}")
        if len(fieldnames) != len(set(fieldnames)):
            raise ValueError(f"Managed table has duplicate header fields: {path}")
        validate_managed_header(fieldnames, path)
        rows = list(reader)
    except csv.Error as exc:
        raise ValueError(f"Managed table is malformed: {path}") from exc
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f"Managed table has a non-rectangular row: {path}")
    validate_managed_run_rows(rows, source=str(path), cardinality="one_per_run")
    return rows


def _registry_text(rows: Sequence[Mapping[str, JsonValue]]) -> str:
    fieldnames = sorted({key for row in rows for key in row})
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, delimiter="\t")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def validate_public_initial_workflow(root: Path, expected: InitialAdaptiveWorkflow, expected_readme: str) -> None:
    workflow_path = root / "adaptive" / "workflow.json"
    registry_path = root / "adaptive" / "run_registry.tsv"
    readme_path = root / "adaptive" / "README.md"
    snapshots = exp_io.read_managed_files_at(root, [workflow_path, registry_path, readme_path])
    try:
        actual = json.loads(snapshots[str(workflow_path)]["text"])
    except json.JSONDecodeError as exc:
        raise ValueError(f"Adaptive workflow is malformed: {workflow_path}") from exc
    if actual != expected:
        raise ValueError(f"Existing adaptive workflow differs from requested initialization: {workflow_path}")
    validate_initial_support_snapshots(root, actual, expected_readme, snapshots)


def validate_initial_support_snapshots(
    root: Path,
    workflow: InitialAdaptiveWorkflow | dict[str, Any],
    expected_readme: str,
    snapshots: dict[str, exp_io.ManagedFileSnapshot],
) -> None:
    registry_path = root / "adaptive" / "run_registry.tsv"
    readme_path = root / "adaptive" / "README.md"
    if snapshots[str(readme_path)]["text"] != expected_readme:
        raise ValueError(f"Existing adaptive README differs from requested initialization: {readme_path}")
    registry_rows = _parse_registry(snapshots[str(registry_path)]["text"], registry_path)
    validate_workflow_payload(
        root,
        workflow,
        require_adaptive_commit=False,
        registry_rows=registry_rows,
    )


def reconcile_event(
    workspace: Path,
    event_type: str,
    payload: dict[str, Any] | AdaptiveEventPayload,
    *,
    identity_field: str,
) -> None:
    if validate_event_history(workspace, event_type, payload, identity_field=identity_field):
        return
    _write_experiment_event(workspace, event_type, payload)
    if not validate_event_history(workspace, event_type, payload, identity_field=identity_field):
        raise ValueError(f"Experiment event was not committed exactly once: {event_type}")


def validate_event_history(
    workspace: Path,
    event_type: str,
    payload: dict[str, Any] | AdaptiveEventPayload,
    *,
    identity_field: str,
) -> bool:
    related = [
        event
        for event in read_experiment_events(workspace)
        if _is_related_event(event, event_type, payload, identity_field)
    ]
    exact = [event for event in related if event_matches(event, event_type, payload)]
    if len(related) != len(exact) or len(exact) > 1:
        raise ValueError(f"Experiment event history conflicts: {event_type}")
    return bool(exact)


def agent_proposal_accepted_event(events: Sequence[Mapping[str, JsonValue]], round_index: int) -> dict[str, Any]:
    related = [
        event
        for event in events
        if event.get("event_type") == "agent_proposal_accepted"
        and type(event.get("round")) is int
        and event["round"] == round_index
    ]
    fields = ("round", "request_id", "proposal_path", "proposal_sha256", "suggestion", "suggestion_sha256")
    if len(related) != 1 or any(field not in related[0] for field in fields):
        raise ValueError(f"Committed agent proposal round {round_index:03d} lacks one exact acceptance event.")
    return {field: related[0][field] for field in fields}


def _round_event_index(
    events: Sequence[Mapping[str, JsonValue]],
    event_type: str,
    payload: Mapping[str, Any],
) -> int | None:
    related = [
        (index, event)
        for index, event in enumerate(events)
        if event.get("event_type") == event_type
        and type(event.get("round")) is int
        and event["round"] == payload["round"]
    ]
    exact = [(index, event) for index, event in related if event_matches(event, event_type, payload)]
    if len(related) != len(exact) or len(exact) > 1:
        raise ValueError(f"Experiment event history conflicts: {event_type}")
    return exact[0][0] if exact else None


def validate_agent_proposal_execute_events(
    events: Sequence[Mapping[str, JsonValue]],
    accepted_event: Mapping[str, Any],
    round_dir: Path,
) -> None:
    accepted_index = _round_event_index(events, "agent_proposal_accepted", accepted_event)
    if accepted_index is None:
        raise ValueError("Committed agent proposal lacks its acceptance event.")
    launch_index = _round_event_index(
        events,
        "launch_round",
        {"round": accepted_event["round"], "round_dir": str(round_dir)},
    )
    if launch_index is None:
        raise ValueError("Committed agent proposal lacks its launch event.")
    completion_index = _round_event_index(events, "agent_proposal_execute_completed", accepted_event)
    if completion_index is None:
        raise ValueError("Committed agent proposal lacks its successful completion event.")
    if not accepted_index < launch_index < completion_index:
        raise ValueError("Agent proposal execute event order conflicts.")


def _is_related_event(
    event: Mapping[str, JsonValue],
    event_type: str,
    payload: Mapping[str, Any],
    identity_field: str,
) -> bool:
    return event.get("event_type") == event_type and event.get(identity_field) == payload[identity_field]


def validate_initial_event_order(
    workspace: Path,
    plan_event: PlanCreatedEvent,
    ready_event: AdaptiveInitEvent,
    *,
    allow_ready_event: bool,
) -> None:
    events = read_experiment_events(workspace)
    plan_positions = [
        index for index, event in enumerate(events) if _is_related_event(event, "plan_created", plan_event, "plan_dir")
    ]
    ready_positions = [
        index
        for index, event in enumerate(events)
        if _is_related_event(event, "adaptive_init", ready_event, "round_dir")
    ]
    if ready_positions and not plan_positions:
        raise ValueError("Adaptive initialization event exists without its plan-created event.")
    if ready_positions and not allow_ready_event:
        raise ValueError("Adaptive readiness event exists without its workflow marker.")
    if plan_positions and ready_positions and max(plan_positions) >= min(ready_positions):
        raise ValueError("Adaptive initialization events are out of order.")


def plan_created_payload(round_dir: Path, plan: Mapping[str, Any]) -> PlanCreatedEvent:
    recipe_value = plan.get("recipe")
    recipe = recipe_value if isinstance(recipe_value, dict) else {}
    return {
        "step_id": (recipe.get("step") or {}).get("id"),
        "plan_dir": str(round_dir),
        "run_count": len(plan.get("runs", [])),
    }


def reconcile_plan_event(workspace: Path, round_dir: Path, plan: Mapping[str, Any]) -> None:
    reconcile_event(
        workspace,
        "plan_created",
        plan_created_payload(round_dir, plan),
        identity_field="plan_dir",
    )


def ensure_initial_registry(root: Path, round_dir: Path, plan: Mapping[str, Any]) -> None:
    registry_path = root / "adaptive" / "run_registry.tsv"
    registered_at = utc_now()
    rows = [
        {
            "round": 0,
            "experiment_id": run.get("experiment_id"),
            "step_id": run.get("step_id"),
            "run_id": run.get("run_id"),
            "run_name": run.get("run_name"),
            "version": run.get("version"),
            "config": run.get("config"),
            "script": run.get("script"),
            "round_dir": str(round_dir),
            "registered_at": registered_at,
        }
        for run in plan.get("runs", [])
    ]
    validate_managed_run_rows(rows, source=str(registry_path), cardinality="one_per_run")
    exp_io.validate_managed_output_paths(root, [registry_path])
    existing = []
    existing_sha256 = None
    existing_is_invalid = False
    if os.path.lexists(registry_path):
        snapshot = exp_io.read_managed_files_at(root, [registry_path], allow_invalid_utf8=True)[str(registry_path)]
        exp_io.validate_managed_output_paths(root, [registry_path])
        existing_sha256 = snapshot["sha256"]
        try:
            existing = _parse_registry(snapshot["text"], registry_path)
        except ValueError:
            existing_is_invalid = True
    stable_fields = tuple(field for field in rows[0] if field != "registered_at")
    expected_fields = set(rows[0])
    if existing_sha256 is not None and not existing_is_invalid:
        expected_stable = [
            {field: "" if row[field] is None else str(row[field]) for field in stable_fields} for row in rows
        ]
        if len(existing) == len(rows) and all(set(row) == expected_fields for row in existing):
            existing_stable = [{field: row[field] for field in stable_fields} for row in existing]
            if existing_stable == expected_stable:
                return
        raise ValueError(f"Existing adaptive initial registry differs from the frozen round: {registry_path}")
    if not exp_io.conditional_atomic_replace_text_at(
        registry_path,
        _registry_text(rows),
        existing_sha256,
        managed_root=root,
    ):
        raise RuntimeError(f"Adaptive initial registry changed during recovery: {registry_path}")
    exp_io.validate_managed_output_paths(root, [registry_path])


def execution_route(execution: Mapping[str, JsonValue]) -> dict[str, str]:
    scheduler_value = execution.get("scheduler")
    scheduler = scheduler_value if isinstance(scheduler_value, dict) else {}
    scheduler_type = str(scheduler.get("type") or "direct")
    slurm_scheduler = scheduler if scheduler_type == "slurm" else {}
    return {
        "target": str(execution.get("target", "local") or "local"),
        "host": str(execution.get("host") or ""),
        "workdir": str(execution.get("workdir") or REPO_ROOT),
        "conda_env": str(execution.get("conda_env") or ""),
        "scheduler.type": scheduler_type,
        "scheduler.direct_controller": str(slurm_scheduler.get("direct_controller") is True).lower(),
        "scheduler.partition": str(slurm_scheduler.get("partition") or ""),
        "scheduler.nodelist": str(slurm_scheduler.get("nodelist") or ""),
    }


def append_registry_rows(root: Path, round_index: int, round_dir: Path) -> None:
    path = root / "adaptive" / "run_registry.tsv"
    plan = hparam_runtime.read_hparam_plan_under_run_lock(round_dir)
    snapshot = exp_io.read_managed_files_at(root, [path])[str(path)]
    rows = _parse_registry(snapshot["text"], path)
    registered_at = utc_now()
    expected = [
        {
            "round": round_index,
            "experiment_id": run.get("experiment_id"),
            "step_id": run.get("step_id"),
            "run_id": run.get("run_id"),
            "run_name": run.get("run_name"),
            "version": run.get("version"),
            "config": run.get("config"),
            "script": run.get("script"),
            "round_dir": str(round_dir),
            "registered_at": registered_at,
        }
        for run in plan.get("runs", [])
    ]
    validate_managed_run_rows(expected, source=str(path), cardinality="one_per_run")
    stable_fields = tuple(field for field in expected[0] if field != "registered_at")
    existing_round = [row for row in rows if str(row.get("round")) == str(round_index)]
    expected_stable = [
        {field: "" if row[field] is None else str(row[field]) for field in stable_fields} for row in expected
    ]
    existing_stable = [{field: row.get(field, "") for field in stable_fields} for row in existing_round]
    if existing_round:
        if (
            len(existing_round) == len(expected)
            and all(set(row) == set(expected[0]) for row in existing_round)
            and existing_stable == expected_stable
        ):
            return
        raise ValueError(f"Existing adaptive registry round differs from the frozen plan: {round_dir}")
    expected_keys = {managed_run_key(row) for row in expected}
    if any(managed_run_key(row) in expected_keys for row in rows):
        raise ValueError(f"Adaptive registry run identity is already bound to another round: {round_dir}")
    replacement = _registry_text([*rows, *expected])
    if not exp_io.conditional_atomic_replace_text_at(
        path,
        replacement,
        snapshot["sha256"],
        managed_root=root,
    ):
        raise RuntimeError(f"Adaptive registry changed during round registration: {path}")
    committed = exp_io.read_managed_files_at(root, [path])[str(path)]
    if committed["text"] != replacement:
        raise RuntimeError(f"Adaptive registry changed after round registration: {path}")


def reconcile_interrupted_launch(
    workspace: Path, plan_dir: Path, plan_keys: set[tuple[str, str]]
) -> tuple[list[dict[str, str]], set[tuple[str, str]], set[tuple[str, str]]]:
    # Slurm observation may reach a remote scheduler and merge_run_manifest takes the run lock, so only the reads
    # hold it; the merge re-applies lifecycle rules to the rows it reads under the lock.
    with experiment_workspace.managed_run_lock(workspace):
        artifacts.read_hparam_plan(plan_dir)
        canonical_rows = read_run_manifest(workspace)
    updates: list[Mapping[str, JsonValue]] = []
    unresolved = set()
    reconciled = set()
    for row in canonical_rows:
        key = validated_run_key(row)
        if key not in plan_keys:
            continue
        if scheduler_type(row) == "slurm":
            if _is_accepted_start(row):
                reconciled.add(key)
                continue
            if row.get("status") not in {"submitting", "unknown_scheduler"}:
                continue
            execution: dict[str, JsonValue] = {"target": row["target"]}
            if row["target"] == "ssh":
                execution["host"] = row["host"]
            if scheduler_direct_controller(row):
                execution["scheduler"] = {"direct_controller": True}
            observed = managed_scheduler.observe_slurm_run(plan_dir, execution, row)
            if _is_accepted_start(observed):
                updates.append(observed)
                reconciled.add(key)
            else:
                unresolved.add(key)
            continue
        if row.get("status") not in {"planned", "pending"}:
            continue
        if row.get("target") in (None, ""):
            continue
        try:
            process_identity = evidence.read_process_identity(row.get("pid_path"), row)
        except RuntimeError:
            unresolved.add(key)
            continue
        if process_identity is not None:
            reconciled.add(key)
            update: dict[str, JsonValue] = {"step_id": row["step_id"], "run_id": row["run_id"], "status": "launched"}
            # The launch may have outlived its first commit; recover the complete immutable identity.
            update.update(**process_identity)
            update["launched_at"] = row.get("launched_at") or utc_now()
            updates.append(update)
    if updates:
        canonical_rows = merge_run_manifest(workspace, updates)
    return canonical_rows, unresolved, reconciled


def finish_interrupted_launch(
    round_dir: Path, workspace: Path, started_keys: set[tuple[str, str]]
) -> list[dict[str, str]]:
    hparam_runtime.reconcile_hparam_launch_artifacts(round_dir, started_keys)
    with experiment_workspace.managed_run_lock(workspace):
        return read_run_manifest(workspace)


def uncommitted_launch_attempts(
    root: Path,
    workspace: Path,
) -> tuple[list[tuple[int, str]], list[tuple[int, dict[str, str]]]]:
    committed_rounds = committed_round_indexes(root)
    registry = read_rows(root / "adaptive" / "run_registry.tsv", require_managed_identity=True)
    # Process-identity reads and plan_registration_rows_state below must not hold the run lock.
    with experiment_workspace.managed_run_lock(workspace):
        canonical_by_key = {validated_run_key(row): row for row in read_run_manifest(workspace)}
    registered_by_round: dict[int, set[tuple[str, str]]] = {}
    for registered in registry:
        round_index = int(registered["round"])
        if round_index not in committed_rounds:
            registered_by_round.setdefault(round_index, set()).add(validated_run_key(registered))
    round_dirs = {
        int(path.name.removeprefix("round_")): path
        for path in (root / "adaptive" / "rounds").glob("round_*")
        if path.name.removeprefix("round_").isdigit() and int(path.name.removeprefix("round_")) not in committed_rounds
    }
    for round_index in registered_by_round:
        round_dirs.setdefault(round_index, round_path(root, round_index))
    unresolved = []
    abandoned = []
    for round_index, round_dir in sorted(round_dirs.items()):
        plan_path = round_dir / "plan.json"
        if not plan_path.exists() and round_index not in registered_by_round:
            continue
        registered_keys = set(registered_by_round.get(round_index, set()))
        plan = (
            hparam_runtime.read_hparam_plan_under_run_lock(round_dir)
            if registered_keys
            else artifacts.read_hparam_plan(
                round_dir,
                require_workspace_state=False,
                require_adaptive_commit=False,
            )
        )
        plan_keys = {validated_run_key(run) for run in plan.get("runs", [])}
        if registered_keys and registered_keys != plan_keys:
            raise ValueError(f"Adaptive registry differs from the frozen round: {round_dir}")
        if not registered_keys:
            row_state = plan_registration_rows_state(
                workspace,
                plan_hparam.hparam_manifest_rows(plan),
                source="Canonical adaptive round",
            )
            if row_state == "missing":
                continue
        run_keys = registered_keys or plan_keys
        for run_key in sorted(run_keys):
            row = canonical_by_key.get(run_key)
            if row is None:
                raise ValueError(f"Uncommitted adaptive run is missing from the canonical manifest: {run_key}")
            status = str(row.get("status") or "")
            if not registered_keys:
                if status not in {"planned", "pending"}:
                    unresolved.append((round_index, str(row["run_id"])))
                continue
            if scheduler_type(row) == "slurm":
                if status in {"submitting", "unknown_scheduler"} or _is_accepted_start(row):
                    unresolved.append((round_index, str(row["run_id"])))
                elif status in {"planned", "pending"}:
                    abandoned.append((round_index, row))
                continue
            pid: int | str | None = row.get("pid")
            if row.get("target") not in (None, ""):
                try:
                    process_identity = evidence.read_process_identity(row.get("pid_path"), row)
                    pid = process_identity["pid"] if process_identity is not None else pid
                except RuntimeError:
                    unresolved.append((round_index, str(row["run_id"])))
                    continue
            if pid not in (None, "") or status not in {"planned", "pending", "launch_failed", "superseded"}:
                unresolved.append((round_index, str(row["run_id"])))
                continue
            if status in {"planned", "pending"}:
                abandoned.append((round_index, row))
    return unresolved, abandoned


def reject_unresolved_launch_attempts(root: Path, workspace: Path) -> None:
    unresolved, abandoned = uncommitted_launch_attempts(root, workspace)
    if unresolved:
        detail = ", ".join(f"round {round_index:03d} {run_id}" for round_index, run_id in unresolved)
        raise RuntimeError(
            f"Uncommitted adaptive launch evidence remains for {detail}; resolve the canonical launch state before "
            "creating another round."
        )
    if not abandoned:
        return
    committed = merge_run_manifest(
        workspace,
        [
            {"step_id": row["step_id"], "run_id": row["run_id"], "status": "superseded"}
            for _round_index, row in abandoned
        ],
    )
    committed_by_key = {managed_run_key(row): row for row in committed}
    unchanged = [
        (round_index, row)
        for round_index, row in abandoned
        if committed_by_key[managed_run_key(row)].get("status") != "superseded"
    ]
    if unchanged:
        detail = ", ".join(f"round {round_index:03d} {row['run_id']}" for round_index, row in unchanged)
        raise RuntimeError(f"Uncommitted adaptive launch state changed before supersede: {detail}")
    for round_index, row in abandoned:
        append_event(
            root,
            "supersede_pending_run",
            {
                "round_dir": str(round_path(root, round_index)),
                "run_id": row["run_id"],
                "status": row["status"],
            },
        )


def committed_round_indexes(root: Path) -> set[int]:
    committed = {0}
    registry_path = root / "adaptive" / "run_registry.tsv"
    if not registry_path.exists():
        return committed
    registry = read_rows(registry_path, require_managed_identity=True)
    registered_rounds = {int(row["round"]) for row in registry}
    initial_plan = hparam_runtime.read_hparam_plan_under_run_lock(round_path(root, 0))
    recipe_value = initial_plan.get("recipe")
    recipe = recipe_value if isinstance(recipe_value, dict) else {}
    workspace = experiment_root(recipe)
    if workspace is None:
        return committed
    # The launch_round event's round is consumed numerically below.
    events: list[dict[str, Any]] = read_experiment_events(workspace)
    for event in events:
        if event.get("event_type") != "launch_round":
            continue
        round_index = int(event["round"])
        if round_index in registered_rounds and event.get("round_dir") == str(round_path(root, round_index)):
            committed.add(round_index)
    return committed


def latest_round_index(root: Path) -> int:
    return max(committed_round_indexes(root))


def round_path(root: Path, index: int) -> Path:
    return root / "adaptive" / "rounds" / f"round_{index:03d}"


def next_round_index(root: Path) -> int:
    registry_path = root / "adaptive" / "run_registry.tsv"
    registry = read_rows(registry_path, require_managed_identity=True) if registry_path.exists() else []
    registered = [int(row["round"]) for row in registry]
    return max([0, *registered]) + 1
