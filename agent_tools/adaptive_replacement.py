"""Launch an adaptive round as a replacement for the current one.

Domain-free kernel step used by ``adaptive_hparam``. Launches the next round, finds running
current-round runs that fail or trail the incumbent past their grace period, stops them
to free capacity for further next-round starts, supersedes pending current runs, and
commits the round once a replacement start is confirmed. A launch interrupted part way
is reconciled before any failure is raised, so the round is either committed with its
confirmed starts or left uncommitted with prospective run states preserved.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import (
    adaptive_evidence,
    adaptive_state,
    experiment_io as exp_io,
    experiment_workspace,
    run_artifacts as artifacts,
    run_evidence as evidence,
)
from .experiment_workspace import (
    experiment_root,
    managed_run_key,
    merge_run_manifest,
    read_run_manifest,
    scheduler_type,
    validated_run_key,
)
from .hparam_runtime import launch_hparam_runs, monitor_hparam_runs, read_hparam_plan_under_run_lock, stop_hparam_run
from .manifests import read_rows, write_rows


def _ids_or_none(keys: list[tuple[str, str]]) -> str:
    return ", ".join(key[1] for key in keys) or "none"


def _committed_phrase(round_committed: bool) -> str:
    return "is already committed" if round_committed else "was not committed"


def _preserved_tail(
    stopped_run_keys: list[tuple[str, str]],
    superseded_current_keys: list[tuple[str, str]],
    next_dir: Path,
) -> str:
    stopped_ids = _ids_or_none(stopped_run_keys)
    superseded_ids = _ids_or_none(superseded_current_keys)
    return (
        f"Confirmed stopped current runs: {stopped_ids}. "
        f"Superseded current pending runs: {superseded_ids}. "
        f"Prospective run states were preserved at {next_dir}."
    )


def _not_superseded_launch_error(
    kind: str,
    next_round: int,
    *,
    attempt_run_id: str | None = None,
    unresolved_ids: str = "",
) -> RuntimeError:
    if attempt_run_id is None:
        lead = f"Adaptive replacement launch failed and round {next_round:03d}"
    else:
        lead = (
            f"Adaptive replacement launch failed after the stop attempt for {attempt_run_id}, and round "
            f"{next_round:03d}"
        )
    if kind == "commit":
        body = f"{lead} launch evidence could not be committed."
        scope = "canonical launch state"
    elif kind == "unresolved":
        body = f"{lead} has unresolved launch evidence for {unresolved_ids}."
        scope = "launch state"
    else:  # kind == "mirrors"
        body = f"{lead} launch mirrors or events could not be reconciled."
        scope = "canonical launch state"
    return RuntimeError(f"{body} Prospective runs were not superseded; resolve the {scope} before retry.")


@dataclass
class _ReplacementState:
    next_round: int
    next_dir: Path
    next_plan_keys: set[tuple[str, str]]
    started_keys: set[tuple[str, str]]
    launch_failed_keys: set[tuple[str, str]]
    round_committed: bool = False
    retirement_credit: int = 0
    superseded_current_keys: list[tuple[str, str]] = dataclass_field(default_factory=list)
    stopped_run_keys: list[tuple[str, str]] = dataclass_field(default_factory=list)


def _commit_round(root: Path, round_dir: Path, state: _ReplacementState) -> None:
    """Commit after a replacement start is confirmed, with launch_round written last."""
    state.superseded_current_keys = _supersede_pending_runs(root, round_dir)
    adaptive_state.append_event(root, "launch_round", {"round": state.next_round, "round_dir": str(state.next_dir)})
    state.round_committed = True


def _launch_with_recovery(
    root: Path,
    workspace: Path,
    state: _ReplacementState,
    round_dir: Path,
    *,
    attempt_run_id: str | None,
    before_launch: set[tuple[str, str]],
) -> None:
    """One launch_hparam_runs attempt plus the interrupted-launch recovery block:
    reconcile -> reject unresolved -> finish mirrors -> commit confirmed starts ->
    raise the family A/B failure. Recovery steps run in that order (invariant)."""
    try:
        launch_hparam_runs(state.next_dir, dry_run=False)
    except Exception as exc:
        try:
            canonical_rows, unresolved_launches, reconciled_starts = adaptive_state.reconcile_interrupted_launch(
                workspace, state.next_dir, state.next_plan_keys
            )
        except Exception as reconcile_exc:
            raise _not_superseded_launch_error(
                "commit", state.next_round, attempt_run_id=attempt_run_id
            ) from reconcile_exc
        if unresolved_launches:
            unresolved_ids = ", ".join(sorted(key[1] for key in unresolved_launches))
            raise _not_superseded_launch_error(
                "unresolved", state.next_round, attempt_run_id=attempt_run_id, unresolved_ids=unresolved_ids
            ) from exc
        next_round_rows = [row for row in canonical_rows if managed_run_key(row) in state.next_plan_keys]
        refreshed_started_keys = adaptive_state.accepted_start_keys(next_round_rows)
        confirmed_starts = (refreshed_started_keys - before_launch) | reconciled_starts
        if confirmed_starts:
            try:
                adaptive_state.finish_interrupted_launch(state.next_dir, workspace, confirmed_starts)
            except Exception as reconcile_exc:
                raise _not_superseded_launch_error(
                    "mirrors", state.next_round, attempt_run_id=attempt_run_id
                ) from reconcile_exc
        if confirmed_starts and not state.round_committed:
            _commit_round(root, round_dir, state)
        if attempt_run_id is None:
            lead = (
                f"Adaptive replacement launch failed; round {state.next_round:03d} "
                f"{_committed_phrase(state.round_committed)}."
            )
        else:
            lead = (
                f"Adaptive replacement launch failed after the stop attempt for {attempt_run_id}; "
                f"round {state.next_round:03d} "
                f"{_committed_phrase(state.round_committed)}."
            )
        raise RuntimeError(
            f"{lead} " + _preserved_tail(state.stopped_run_keys, state.superseded_current_keys, state.next_dir)
        ) from exc


def _launch_initial_replacement(
    root: Path, workspace: Path, state: _ReplacementState, round_dir: Path
) -> list[dict[str, Any]]:
    before_launch = state.started_keys
    _launch_with_recovery(root, workspace, state, round_dir, attempt_run_id=None, before_launch=before_launch)
    # Launches, stops and run scripts replace run_manifest.tsv under the run lock. Every launch, stop, monitor and
    # recovery call in this module takes that lock itself, so only the reads between them hold it.
    with experiment_workspace.managed_run_lock(workspace):
        canonical_rows = read_run_manifest(workspace)
    next_round_rows = [row for row in canonical_rows if validated_run_key(row) in state.next_plan_keys]
    refreshed_started_keys = adaptive_state.accepted_start_keys(next_round_rows)
    newly_launch_failed = {
        validated_run_key(row) for row in next_round_rows if row.get("status") == "launch_failed"
    } - state.launch_failed_keys
    state.retirement_credit = len(refreshed_started_keys - before_launch)
    state.started_keys = refreshed_started_keys
    if state.retirement_credit:
        _commit_round(root, round_dir, state)
    if newly_launch_failed:
        failed_ids = ", ".join(sorted(key[1] for key in newly_launch_failed))
        raise RuntimeError(
            f"Adaptive replacement launch failed for {failed_ids}; round {state.next_round:03d} "
            f"{_committed_phrase(state.round_committed)}. "
            + _preserved_tail([], state.superseded_current_keys, state.next_dir)
        )
    return next_round_rows


def _drain_bad_runs(
    root: Path,
    workspace: Path,
    state: _ReplacementState,
    round_dir: Path,
    recipe: dict[str, Any],
    ordered_bad_run_keys: list[tuple[str, str]],
    next_round_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    bad_index = 0
    while bad_index < len(ordered_bad_run_keys):
        with experiment_workspace.managed_run_lock(workspace):
            canonical_rows = read_run_manifest(workspace)
        next_round_rows = [row for row in canonical_rows if validated_run_key(row) in state.next_plan_keys]
        pending = any(row.get("status") in {"planned", "pending"} for row in next_round_rows)
        if state.retirement_credit <= 0 and not pending:
            break
        run_key = ordered_bad_run_keys[bad_index]
        before_stop = dict({validated_run_key(row): row for row in canonical_rows}[run_key])
        if scheduler_type(before_stop) == "slurm" and state.retirement_credit <= 0:
            break
        bad_index += 1
        try:
            stopped = _stop_bad_running_runs(root, round_dir, recipe, run_keys={run_key})
        except Exception as exc:
            with experiment_workspace.managed_run_lock(workspace):
                canonical_by_key = {validated_run_key(row): row for row in read_run_manifest(workspace)}
            if canonical_by_key[run_key].get("status") == "stopped" and run_key not in state.stopped_run_keys:
                state.stopped_run_keys.append(run_key)
            raise RuntimeError(
                f"Adaptive replacement failed while stopping {run_key[1]}; round {state.next_round:03d} "
                f"{_committed_phrase(state.round_committed)}. "
                + _preserved_tail(state.stopped_run_keys, state.superseded_current_keys, state.next_dir)
            ) from exc
        if not stopped:
            with experiment_workspace.managed_run_lock(workspace):
                canonical_by_key = {validated_run_key(row): row for row in read_run_manifest(workspace)}
            after_stop = canonical_by_key[run_key]
            if (
                before_stop.get("stop_requested_at") in (None, "")
                and after_stop.get("status") == "stopping"
                and after_stop.get("stop_requested_at") not in (None, "")
                and after_stop.get("stop_reason") == "adaptive replacement"
            ):
                state.retirement_credit -= 1
                if state.retirement_credit <= 0:
                    break
                continue
            break
        state.stopped_run_keys.extend(stopped)
        state.retirement_credit -= len(stopped)
        with experiment_workspace.managed_run_lock(workspace):
            canonical_rows = read_run_manifest(workspace)
        next_round_rows = [row for row in canonical_rows if validated_run_key(row) in state.next_plan_keys]
        if not any(row.get("status") in {"planned", "pending"} for row in next_round_rows):
            continue
        before_launch = state.started_keys
        before_launch_failed = {
            validated_run_key(row) for row in next_round_rows if row.get("status") == "launch_failed"
        }
        _launch_with_recovery(root, workspace, state, round_dir, attempt_run_id=run_key[1], before_launch=before_launch)
        with experiment_workspace.managed_run_lock(workspace):
            canonical_rows = read_run_manifest(workspace)
        next_round_rows = [row for row in canonical_rows if validated_run_key(row) in state.next_plan_keys]
        state.started_keys = adaptive_state.accepted_start_keys(next_round_rows)
        newly_launch_failed = {
            validated_run_key(row) for row in next_round_rows if row.get("status") == "launch_failed"
        } - before_launch_failed
        newly_started = state.started_keys - before_launch
        if newly_started and not state.round_committed:
            _commit_round(root, round_dir, state)
        if newly_launch_failed:
            failed_ids = ", ".join(sorted(key[1] for key in newly_launch_failed))
            raise RuntimeError(
                f"Adaptive replacement launch failed for {failed_ids} after the stop attempt for {run_key[1]}; "
                f"round {state.next_round:03d} {_committed_phrase(state.round_committed)}. "
                + _preserved_tail(state.stopped_run_keys, state.superseded_current_keys, state.next_dir)
            )
        if not newly_started:
            statuses = ", ".join(sorted({str(row.get("status") or "") for row in next_round_rows})) or "none"
            raise RuntimeError(
                f"Round {state.next_round:03d} started no additional runs after stopping {run_key[1]} "
                f"(statuses: {statuses}); the round {_committed_phrase(state.round_committed)}. "
                + _preserved_tail(state.stopped_run_keys, state.superseded_current_keys, state.next_dir)
            )
        state.retirement_credit += len(newly_started)
    return next_round_rows


def launch_replacement_round(
    root: Path,
    workspace: Path,
    round_dir: Path,
    recipe: dict[str, Any],
    next_round: int,
    next_dir: Path,
) -> None:
    execution_value = recipe.get("execution")
    execution = execution_value if isinstance(execution_value, dict) else {}
    scheduler_value = execution.get("scheduler")
    scheduler = scheduler_value if isinstance(scheduler_value, dict) else {}
    if scheduler.get("type") == "slurm":
        monitor_hparam_runs(round_dir)
    # Both rounds are frozen into this workspace; monitoring and _bad_running_run_keys take its run lock themselves.
    with experiment_workspace.managed_run_lock(workspace):
        current_plan = artifacts.read_hparam_plan(round_dir)
    bad_run_keys = _bad_running_run_keys(root, round_dir, recipe)
    ordered_bad_run_keys = [
        validated_run_key(run) for run in current_plan["runs"] if validated_run_key(run) in bad_run_keys
    ]
    with experiment_workspace.managed_run_lock(workspace):
        next_plan_keys = {validated_run_key(run) for run in artifacts.read_hparam_plan(next_dir)["runs"]}
        canonical_rows = read_run_manifest(workspace)
    state = _ReplacementState(
        next_round=next_round,
        next_dir=next_dir,
        next_plan_keys=next_plan_keys,
        started_keys=adaptive_state.accepted_start_keys(
            [row for row in canonical_rows if validated_run_key(row) in next_plan_keys]
        ),
        launch_failed_keys={
            validated_run_key(row)
            for row in canonical_rows
            if validated_run_key(row) in next_plan_keys and row.get("status") == "launch_failed"
        },
    )
    next_round_rows = _launch_initial_replacement(root, workspace, state, round_dir)
    next_round_rows = _drain_bad_runs(root, workspace, state, round_dir, recipe, ordered_bad_run_keys, next_round_rows)

    if not state.round_committed:
        statuses = ", ".join(sorted({str(row.get("status") or "") for row in next_round_rows})) or "none"
        raise RuntimeError(
            f"Round {next_round:03d} started no runs (statuses: {statuses}); the round was not committed and "
            f"current runs were not retired. Prospective run states were preserved at {next_dir}."
        )


def _supersede_pending_runs(root: Path, round_dir: Path) -> list[tuple[str, str]]:
    plan = read_hparam_plan_under_run_lock(round_dir)
    recipe_value = plan.get("recipe")
    recipe = recipe_value if isinstance(recipe_value, dict) else {}
    workspace = experiment_root(recipe)
    if workspace is None:
        raise ValueError("Hparam plan is not bound to an experiment workspace.")
    launch_path = round_dir / "launch_manifest.tsv"
    targets = [
        workspace / "run_manifest.tsv",
        workspace / "run_matrix.csv",
        workspace / "reports" / "run_matrix.md",
        workspace / "events.jsonl",
        round_dir / "run_status.tsv",
    ]
    if launch_path.exists():
        targets.append(launch_path)
    exp_io.validate_managed_output_paths(workspace, targets)
    # merge_run_manifest takes the run lock and re-applies lifecycle rules, so only this read holds it.
    with experiment_workspace.managed_run_lock(workspace):
        canonical_rows = read_run_manifest(workspace)
    canonical_by_key = {validated_run_key(row): row for row in canonical_rows}
    transitions = []
    for run in plan["runs"]:
        row = canonical_by_key[validated_run_key(run)]
        if row.get("status") in {"planned", "pending"}:
            transitions.append(row)
    if transitions:
        committed = merge_run_manifest(
            workspace,
            [{"step_id": row["step_id"], "run_id": row["run_id"], "status": "superseded"} for row in transitions],
        )
    else:
        committed = canonical_rows
    committed_by_key = {validated_run_key(row): row for row in committed}
    round_rows = [committed_by_key[validated_run_key(run)] for run in plan["runs"]]
    write_rows(round_dir / "run_status.tsv", round_rows)
    if launch_path.exists():
        write_rows(launch_path, round_rows)
    for row in transitions:
        if committed_by_key[validated_run_key(row)].get("status") != "superseded":
            continue
        adaptive_state.append_event(
            root,
            "supersede_pending_run",
            {"round_dir": str(round_dir), "run_id": row["run_id"], "status": row["status"]},
        )
    return [
        validated_run_key(row)
        for row in transitions
        if committed_by_key[validated_run_key(row)]["status"] == "superseded"
    ]


def _bad_running_run_keys(root: Path, round_dir: Path, recipe: dict[str, Any]) -> set[tuple[str, str]]:
    adaptive = adaptive_state.adaptive_settings(recipe)
    replacement_value = adaptive.get("replacement")
    replacement = replacement_value if isinstance(replacement_value, dict) else {}
    if replacement.get("enabled", True) is not True or replacement.get("allow_running_stop", False) is not True:
        return set()
    objective = adaptive_state.workflow_objective(root, recipe)
    incumbent = _latest_incumbent_score(root)
    margin = float(replacement.get("kill_margin") or 0.0)
    plan = read_hparam_plan_under_run_lock(round_dir)
    plan_recipe_value = plan.get("recipe")
    plan_recipe = plan_recipe_value if isinstance(plan_recipe_value, dict) else {}
    evaluation_value = plan_recipe.get("evaluation_policy")
    evaluation = evaluation_value if isinstance(evaluation_value, dict) else {}
    selection_split = str(evaluation.get("selection_split") or "")
    workspace = experiment_root(plan_recipe)
    if workspace is None:
        raise ValueError("Hparam plan is not bound to an experiment workspace.")
    plan_keys = {validated_run_key(run) for run in plan["runs"]}
    bad_keys = set()
    # Runtime evidence may be read over SSH, so only the manifest read holds the run lock.
    with experiment_workspace.managed_run_lock(workspace):
        canonical_rows = read_run_manifest(workspace)
    for row in canonical_rows:
        key = validated_run_key(row)
        if key not in plan_keys:
            continue
        if row.get("status") != "running":
            continue
        if scheduler_type(row) == "slurm" and row.get("scheduler_exit_code") not in (None, ""):
            continue
        should_stop = evidence.log_has_failure(row.get("log_path"), row)
        uses_checkpoint_test_objective = selection_split == "test" and objective["metric"].startswith("test_")
        data: dict[str, Any] = {}
        if not should_stop and not uses_checkpoint_test_objective:
            observed_artifacts = evidence.runtime_artifacts(row)
            if observed_artifacts is not None:
                _manifest_path, data, _checkpoint_names = observed_artifacts
        if uses_checkpoint_test_objective:
            # Checkpoint test objectives become selection evidence only after canonical successful completion.
            score = None
        else:
            score = adaptive_evidence.manifest_metrics(data).get(
                objective["metric"], artifacts.metric_value(data, objective["metric"])
            )
        if (
            not should_stop
            and incumbent is not None
            and score is not None
            and score != ""
            and _grace_satisfied(row, data, replacement)
        ):
            try:
                value = float(score)
                should_stop = value < incumbent - margin if objective["mode"] == "max" else value > incumbent + margin
            except (TypeError, ValueError):
                should_stop = False
        if should_stop:
            bad_keys.add(key)
    return bad_keys


def _stop_bad_running_runs(
    root: Path,
    round_dir: Path,
    recipe: dict[str, Any],
    *,
    run_keys: set[tuple[str, str]] | None = None,
) -> list[tuple[str, str]]:
    keys = _bad_running_run_keys(root, round_dir, recipe) if run_keys is None else run_keys
    if not keys:
        return []
    plan = read_hparam_plan_under_run_lock(round_dir)
    plan_recipe_value = plan.get("recipe")
    plan_recipe = plan_recipe_value if isinstance(plan_recipe_value, dict) else {}
    workspace = experiment_root(plan_recipe)
    if workspace is None:
        raise ValueError("Hparam plan is not bound to an experiment workspace.")
    # stop_hparam_run takes the run lock itself, so only the reads around it hold it.
    with experiment_workspace.managed_run_lock(workspace):
        canonical_by_key = {managed_run_key(row): row for row in read_run_manifest(workspace)}
    stopped = []
    for run in plan["runs"]:
        key = managed_run_key(run)
        row = canonical_by_key[key]
        if key not in keys or row.get("status") != "running":
            continue
        stop_hparam_run(round_dir, str(row["run_id"]), reason="adaptive replacement")
        with experiment_workspace.managed_run_lock(workspace):
            canonical_by_key = {managed_run_key(item): item for item in read_run_manifest(workspace)}
        if canonical_by_key[key].get("status") == "stopped":
            adaptive_state.append_event(
                root, "stop_bad_running_run", {"round_dir": str(round_dir), "run_id": row["run_id"]}
            )
            stopped.append(key)
    return stopped


def _grace_satisfied(row: dict[str, Any], manifest: dict[str, Any], replacement: dict[str, Any]) -> bool:
    grace_epochs = replacement.get("grace_epochs")
    if grace_epochs is not None:
        try:
            if float(manifest.get("epoch", "")) < float(grace_epochs):
                return False
        except (TypeError, ValueError):
            return False
    grace_minutes = replacement.get("grace_minutes")
    if grace_minutes is not None:
        started_at = (
            row.get("scheduler_started_at", "") if scheduler_type(row) == "slurm" else row.get("launched_at", "")
        )
        minutes = _minutes_since(started_at)
        if minutes is None or minutes < float(grace_minutes):
            return False
    return True


def _minutes_since(timestamp: str) -> float | None:
    try:
        start = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None
    return (datetime.now(timezone.utc) - start).total_seconds() / 60


def _latest_incumbent_score(root: Path) -> float | None:
    rows = read_rows(root / "adaptive" / "incumbents.tsv")
    if not rows:
        return None
    try:
        return float(rows[-1]["objective_score"])
    except (KeyError, TypeError, ValueError):
        return None
