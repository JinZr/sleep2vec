"""Backend selection and reusable managed launch and observation primitives.

Owns the reusable direct GPU-capacity and process lifecycle and the Slurm
submit/adopt pass, state table and stop. Each launch checks its target against
the frozen execution snapshot owned by ``execution_snapshot``, and writes that
snapshot when the plan has none yet.

``_launch_managed_runs`` selects the direct or Slurm backend for one frozen
``LaunchOptions``; ``SchedulerHooks`` inside it supplies persistence and
execution callbacks.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any, Literal, TypedDict

import yaml

# Aliased because build_launch_command's public keyword and a launch-path local are named execution_snapshot.
from . import (
    execution_snapshot as exec_snapshot,
    experiment_io as exp_io,
    experiment_workspace,
    gpu_rules,
    python_programs,
    run_evidence as evidence,
    slurm,
    transport,
)
from .experiment_workspace import (
    EXECUTION_IDENTITY_FIELDS,
    LAUNCHABLE_STATUSES,
    PROCESS_IDENTITY_FIELDS,
    SCHEDULER_PLAN_IDENTITY_FIELDS,
    TERMINAL_STATUSES,
    RunKey,
    append_event,
    merge_run_manifest,
    merge_run_row,
    read_run_manifest,
    scheduler_type,
    validate_frozen_run_update,
    validated_run_key,
)
from .manifests import utc_now
from .models import REPO_ROOT, YAML_SAFE_LOADER, JsonValue, is_full_git_object_id

ACTIVE_STATUSES = frozenset({"queued", "launched", "running", "stopping", "lost", "missing_pid"})
LAUNCH_TIMEOUT_SECONDS = 60
QUEUE_RUNNER_LOCK_NAME = "queue_runner.lock"
QUEUE_RUNNER_LOG_NAME = "queue_runner.log"
SLURM_EXIT_CODE_NAME = "exit_code"
SLURM_RUNTIME_COMMIT_NAME = "runtime_commit"


class MissingPidCapacityError(RuntimeError):
    def __init__(self, step_id: str, run_id: str):
        self.step_id = step_id
        self.run_id = run_id
        super().__init__(f"Managed launch capacity is blocked because {step_id} / {run_id} has status missing_pid.")


__all__ = [
    "ACTIVE_STATUSES",
    "CapacityState",
    "LAUNCHABLE_STATUSES",
    "LAUNCH_TIMEOUT_SECONDS",
    "LaunchResult",
    "MissingPidCapacityError",
    "QUEUE_RUNNER_LOCK_NAME",
    "QUEUE_RUNNER_LOG_NAME",
    "QueueRunnerLiveness",
    "SchedulerHooks",
    "build_launch_command",
    "capacity_state",
    "gpu_groups",
    "launch_managed_runs",
    "observe_run",
    "observe_runs",
    "observe_slurm_run",
    "queue_runner_liveness",
    "queue_runner_lock",
    "script_commits_terminal_status",
    "shares_capacity",
    "slurm_job_snapshot",
    "slurm_state_needed",
    "start_process",
    "stop_slurm_run_locked",
]


class QueueRunnerLiveness(TypedDict):
    plan_dir: str
    active: bool
    pid: str | None
    log_path: str


@contextmanager
def queue_runner_lock(plan_dir: Path, *, record_pid: bool) -> Iterator[None]:
    """Hold a direct plan's exclusive queue-runner lock, failing at once if another runner holds it.

    The lock is separate from the workspace run lock: take it before, never while holding,
    ``managed_run_lock``. With record_pid the holder's PID replaces the lock file's text, for
    liveness display only; the flock, not the PID, decides liveness. Closing the descriptor
    (or the holder's exit) releases the lock."""
    path = plan_dir / QUEUE_RUNNER_LOCK_NAME
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o644)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                f"A queue runner is already active for {plan_dir} (pid {_queue_runner_pid(descriptor) or 'unknown'}); "
                f"see its log {plan_dir / QUEUE_RUNNER_LOG_NAME}."
            ) from exc
        if record_pid:
            os.ftruncate(descriptor, 0)
            os.pwrite(descriptor, f"{os.getpid()}\n".encode(), 0)
        yield
    finally:
        os.close(descriptor)


def queue_runner_liveness(plan_dir: Path) -> QueueRunnerLiveness:
    """Probe a plan's queue runner with a non-blocking shared lock; never creates the lock file or starts a runner.

    A held lock means a runner is alive; a missing lock file means none has run."""
    liveness: QueueRunnerLiveness = {
        "plan_dir": str(plan_dir),
        "active": False,
        "pid": None,
        "log_path": str(plan_dir / QUEUE_RUNNER_LOG_NAME),
    }
    try:
        descriptor = os.open(plan_dir / QUEUE_RUNNER_LOCK_NAME, os.O_RDONLY | os.O_NOFOLLOW)
    except FileNotFoundError:
        return liveness
    try:
        fcntl.flock(descriptor, fcntl.LOCK_SH | fcntl.LOCK_NB)
    except BlockingIOError:
        liveness["active"] = True
        liveness["pid"] = _queue_runner_pid(descriptor)
    finally:
        os.close(descriptor)
    return liveness


def _queue_runner_pid(descriptor: int) -> str | None:
    return os.pread(descriptor, 32, 0).decode("ascii", errors="replace").strip() or None


@dataclass(frozen=True)
class StatusChanges:
    rows_by_key: dict[RunKey, Mapping[str, JsonValue]]
    changes: dict[RunKey, tuple[JsonValue, JsonValue]]


@dataclass
class CapacityState:
    gpu_groups: list[list[Any]]
    group_loads: list[int]
    assigned_group_by_key: dict[RunKey, int]
    slots: int
    allow_gpu_oversubscription: bool
    external_missing_pid: list[RunKey]

    def next_allocation(
        self,
        candidates: Sequence[tuple[int, dict[str, JsonValue]]],
    ) -> tuple[int, dict[str, JsonValue], int | None] | None:
        eligible: list[tuple[int, int, dict[str, JsonValue], int | None]] = []
        for index, row in candidates:
            frozen_group_index = self.assigned_group_by_key.get(validated_run_key(row))
            if frozen_group_index is not None:
                group_indexes: Iterable[int | None] = [frozen_group_index]
            elif self.gpu_groups:
                group_indexes = range(len(self.gpu_groups))
            else:
                group_indexes = [None]
            for group_index in group_indexes:
                load = self.group_loads[group_index] if group_index is not None else 0
                if group_index is not None and not self.allow_gpu_oversubscription and load >= 1:
                    continue
                eligible.append((load, index, row, group_index))
        if not eligible:
            return None
        _load, index, row, group_index = min(
            eligible,
            key=lambda item: (item[0], item[1], item[3] if item[3] is not None else -1),
        )
        return index, row, group_index

    def record_started(self, group_index: int | None) -> None:
        if group_index is not None:
            self.group_loads[group_index] += 1
        self.slots -= 1


@dataclass(frozen=True)
class LaunchResult:
    committed_rows: list[dict[str, str]]
    launch_rows: Sequence[Mapping[str, JsonValue]]
    started_keys: frozenset[RunKey]
    status_changes: dict[RunKey, tuple[JsonValue, JsonValue]]
    external_status_changes: dict[RunKey, tuple[JsonValue, JsonValue]]


class _LaunchIdentityExtras(TypedDict, total=False):
    pid: Literal[""]
    process_group_id: Literal[""]
    process_start_token: Literal[""]
    planned_runtime_commit: str


class _LaunchIdentityFields(_LaunchIdentityExtras):
    target: str
    workdir: Any
    gpus: str
    pid_path: str
    command: str


class DirectLaunchIdentity(_LaunchIdentityFields):
    log_path: str


class SlurmLaunchIdentity(_LaunchIdentityFields):
    log_path: JsonValue


class LaunchVerificationOptions(TypedDict, total=False):
    checkpoint_path: Path | None
    checkpoint_sha256: str | None
    planned_command: str
    run_id: str


class FrozenLaunchArtifact(TypedDict):
    path: str
    sha256: str


@dataclass(frozen=True)
class SchedulerHooks:
    merge_manifest: Callable[..., list[dict[str, str]]] = merge_run_manifest
    append_event: Callable[..., None] = append_event
    validate_run_update: Callable[..., None] = validate_frozen_run_update
    validated_snapshot: Callable[..., exec_snapshot.ExecutionSnapshotResult] | None = None
    build_command: Callable[..., str] | None = None
    start_process: Callable[..., tuple[str, str]] | None = None


@dataclass(frozen=True)
class LaunchOptions:
    """Per-call launch settings shared by the direct and Slurm backends."""

    dry_run: bool
    fail_on_missing_pid_blocker: bool
    hooks: SchedulerHooks


def gpu_groups(execution: Mapping[str, JsonValue], runtime: Mapping[str, Any]) -> list[list[Any]]:
    groups, issues = gpu_rules.gpu_group_plan(execution, runtime)
    errors = [issue for issue in issues if not issue.warning]
    if errors:
        raise ValueError(errors[0].message)
    return groups


def script_commits_terminal_status(row: Mapping[str, JsonValue], *, default: bool = False) -> bool:
    owner = row.get("terminal_status_owner")
    if owner in (None, ""):
        return default
    if owner == "script":
        return True
    if owner == "monitor":
        return False
    raise ValueError("terminal_status_owner must be 'script' or 'monitor'.")


def observe_run(
    run_dir: str | Path,
    row: Mapping[str, JsonValue],
    previous: Mapping[str, JsonValue] | None = None,
    *,
    health: bool = False,
) -> dict[str, JsonValue]:
    """Return a direct-run observation merged with previous (or row) evidence.

    Passes the row's observation fields to status_row and derives terminal-status
    ownership from prior metadata, which defaults to monitor-owned.
    health enables additional probes. Reads local evidence but does not
    commit manifests, launch or stop runs; the caller persists the returned row.
    This entrypoint does not dispatch Slurm rows to observe_slurm_run. Unhandled
    observer errors propagate.
    """
    prior = previous or row
    observation = {field: row[field] for field in evidence.RUN_EVIDENCE_FIELDS if field in row}
    observation.update(
        {
            "step_id": row.get("step_id", ""),
            "run_id": row.get("run_id", ""),
            "status": row.get("status", ""),
        }
    )
    return evidence.status_row(
        Path(run_dir),
        observation,
        prior,
        script_commits_terminal_status=script_commits_terminal_status(prior),
        health=health,
    )


def observe_runs(
    run_dir: str | Path,
    rows_by_key: Mapping[RunKey, Mapping[str, JsonValue]],
    keys: Iterable[RunKey],
    *,
    dry_run: bool,
    health: bool = False,
) -> StatusChanges:
    refreshed: dict[RunKey, Mapping[str, JsonValue]] = {}
    changes: dict[RunKey, tuple[JsonValue, JsonValue]] = {}
    for key in keys:
        previous = rows_by_key.get(key)
        if previous is None:
            raise ValueError(f"Canonical run is missing: {key[0]} / {key[1]}")
        if dry_run or previous.get("target") in (None, ""):
            refreshed[key] = previous
            continue
        observed = observe_run(run_dir, previous, previous, health=health)
        refreshed[key] = observed
        if observed.get("status") != previous.get("status"):
            changes[key] = (previous.get("status"), observed.get("status"))
    return StatusChanges(refreshed, changes)


def capacity_state(
    execution: Mapping[str, JsonValue],
    runtime: Mapping[str, Any],
    expected_rows: Mapping[RunKey, Mapping[str, JsonValue]],
    workspace_rows: Mapping[RunKey, Mapping[str, JsonValue]],
    *,
    expected_keys: set[RunKey],
) -> CapacityState:
    groups = gpu_groups(execution, runtime)
    raw_max_concurrent = execution.get("max_concurrent", max(len(groups), 1))
    if type(raw_max_concurrent) is not int or raw_max_concurrent <= 0:
        raise ValueError("execution.max_concurrent must be a positive integer.")
    max_concurrent = raw_max_concurrent
    allow_gpu_oversubscription = bool(groups) and max_concurrent > len(groups)
    group_values = [{str(item) for item in group} for group in groups]
    current_gpu_pool = set().union(*group_values) if group_values else set()
    other_active_gpu_sets: list[set[str]] = []
    unknown_other_active = 0
    external_missing_pid: list[RunKey] = []
    for key, row in workspace_rows.items():
        if (
            not groups
            or key in expected_keys
            or row.get("status") not in ACTIVE_STATUSES
            or scheduler_type(row) == "slurm"
        ):
            continue
        if not row.get("target"):
            unknown_other_active += 1
            continue
        assigned = {part.strip() for part in str(row.get("gpus") or "").split(",") if part.strip()}
        if assigned and not assigned.intersection(current_gpu_pool):
            continue
        if row.get("status") == "missing_pid":
            external_missing_pid.append(key)
        if not assigned:
            unknown_other_active += 1
            continue
        other_active_gpu_sets.append(assigned)

    active = (
        sum(row.get("status") in ACTIVE_STATUSES for row in expected_rows.values())
        + len(other_active_gpu_sets)
        + unknown_other_active
    )
    group_by_value = {",".join(str(item) for item in group): index for index, group in enumerate(groups)}
    group_loads = [unknown_other_active] * len(groups)
    for assigned in other_active_gpu_sets:
        for group_index, group in enumerate(group_values):
            if assigned.intersection(group):
                group_loads[group_index] += 1
    assigned_group_by_key: dict[RunKey, int] = {}
    for key, previous in expected_rows.items():
        assigned_gpus = ",".join(part.strip() for part in str(previous.get("gpus") or "").split(",") if part.strip())
        if not assigned_gpus:
            if previous.get("status") in ACTIVE_STATUSES:
                for group_index in range(len(groups)):
                    group_loads[group_index] += 1
            continue
        assigned_group_index = group_by_value.get(assigned_gpus)
        if assigned_group_index is None:
            raise ValueError(f"Frozen GPUs are not one configured GPU group for {key[0]} / {key[1]}: {assigned_gpus}")
        assigned_group_by_key[key] = assigned_group_index
        if previous.get("status") in ACTIVE_STATUSES:
            group_loads[assigned_group_index] += 1
    return CapacityState(
        gpu_groups=groups,
        group_loads=group_loads,
        assigned_group_by_key=assigned_group_by_key,
        slots=max(max_concurrent - active, 0),
        allow_gpu_oversubscription=allow_gpu_oversubscription,
        external_missing_pid=external_missing_pid,
    )


def shares_capacity(
    execution: Mapping[str, JsonValue],
    groups: list[list[Any]],
    row: Mapping[str, JsonValue],
) -> bool:
    if scheduler_type(row) == "slurm":
        return False
    assigned = {part.strip() for part in str(row.get("gpus") or "").split(",") if part.strip()}
    current_gpu_pool = {str(item) for group in groups for item in group}
    return not assigned or bool(assigned.intersection(current_gpu_pool))


def launch_managed_runs(
    workspace: str | Path,
    owner_dir: str | Path,
    runs: Sequence[Mapping[str, JsonValue]],
    execution: Mapping[str, JsonValue],
    runtime: Mapping[str, Any],
    *,
    dry_run: bool = True,
    fail_on_missing_pid_blocker: bool = False,
    hooks: SchedulerHooks | None = None,
    lock_held: bool = False,
) -> LaunchResult:
    root = Path(workspace)
    managed_dir = Path(owner_dir)
    options = LaunchOptions(
        dry_run=dry_run,
        fail_on_missing_pid_blocker=fail_on_missing_pid_blocker,
        hooks=hooks or SchedulerHooks(),
    )
    if lock_held:
        return _launch_managed_runs(root, managed_dir, runs, execution, runtime, options)
    with experiment_workspace.managed_run_lock(root):
        return _launch_managed_runs(root, managed_dir, runs, execution, runtime, options)


def _launch_managed_runs(
    workspace: Path,
    owner_dir: Path,
    runs: Sequence[Mapping[str, JsonValue]],
    execution: Mapping[str, JsonValue],
    runtime: Mapping[str, Any],
    options: LaunchOptions,
) -> LaunchResult:
    backend = _managed_scheduler_type(execution, runs)
    if backend == "slurm":
        return _launch_slurm_runs(workspace, owner_dir, runs, execution, options)
    planned_by_key = {validated_run_key(run): run for run in runs}
    snapshot_path, expected_keys, workspace_by_key = _managed_launch_preflight(workspace, owner_dir, runs)
    observed = observe_runs(
        owner_dir,
        workspace_by_key,
        expected_keys,
        dry_run=options.dry_run,
    )
    refreshed = observed.rows_by_key
    groups = gpu_groups(execution, runtime)
    workspace_by_key, external_status_changes = _refresh_external_capacity_runs(
        workspace=workspace,
        owner_dir=owner_dir,
        workspace_by_key=workspace_by_key,
        expected_keys=expected_keys,
        groups=groups,
        execution=execution,
        dry_run=options.dry_run,
        hooks=options.hooks,
    )

    capacity = capacity_state(
        execution,
        runtime,
        refreshed,
        workspace_by_key,
        expected_keys=expected_keys,
    )
    missing_pid_blocker = None
    if not options.dry_run and options.fail_on_missing_pid_blocker:
        current_missing_pid = [key for key, row in refreshed.items() if row.get("status") == "missing_pid"]
        capacity_needed = any(row.get("status") in ACTIVE_STATUSES | LAUNCHABLE_STATUSES for row in refreshed.values())
        external_missing_pid = capacity.external_missing_pid if capacity_needed else []
        blockers = sorted(set(current_missing_pid) | set(external_missing_pid))
        if blockers:
            missing_pid_blocker = MissingPidCapacityError(*blockers[0])

    rows, launch_identity_by_key = _prepare_direct_launch_rows(runs, execution, refreshed, hooks=options.hooks)

    launchable = [(index, row) for index, row in enumerate(rows) if row["status"] in LAUNCHABLE_STATUSES]
    output_path_fields: tuple[Literal["log_path", "pid_path"], ...] = ("log_path", "pid_path")
    run_output_paths = [
        Path(str(launch_identity_by_key[validated_run_key(row)][field])) for row in rows for field in output_path_fields
    ]
    exp_io.validate_managed_output_paths(workspace, run_output_paths)

    execution_snapshot = None
    if not options.dry_run and missing_pid_blocker is None:
        has_launch_candidate = False
        if capacity.slots > 0:
            has_launch_candidate = capacity.next_allocation(launchable) is not None
        if has_launch_candidate:
            runtime_roots = [
                Path(str(row[field]))
                for _index, row in launchable
                for field in ("runtime_dir", "checkpoint_dir")
                if row.get(field) not in (None, "")
            ]
            runtime_root = Path(str(execution.get("workdir") or REPO_ROOT))
            exp_io.validate_managed_output_paths(runtime_root, runtime_roots)
            validated_snapshot = options.hooks.validated_snapshot or exec_snapshot.validated_execution_snapshot
            snapshot_result = validated_snapshot(
                owner_dir,
                execution,
                runs,
                workspace_by_key,
            )
            execution_snapshot = snapshot_result[0]
            if snapshot_result[1]:
                exec_snapshot.write_execution_snapshot_file(snapshot_path, snapshot_result[0])
    for row in rows:
        Path(str(row["run_dir"])).mkdir(parents=True, exist_ok=True)

    build_command = options.hooks.build_command or build_launch_command
    start = options.hooks.start_process or start_process
    started_keys = _start_direct_launch_rows(
        workspace=workspace,
        rows=rows,
        launchable=launchable,
        capacity=capacity,
        execution=execution,
        launch_identity_by_key=launch_identity_by_key,
        workspace_by_key=workspace_by_key,
        planned_by_key=planned_by_key,
        execution_snapshot=execution_snapshot,
        missing_pid_blocker=missing_pid_blocker,
        dry_run=options.dry_run,
        build_command=build_command,
        start=start,
        hooks=options.hooks,
    )

    committed_rows, launch_rows = _commit_direct_launch_rows(
        workspace, runs, rows, workspace_by_key, dry_run=options.dry_run, hooks=options.hooks
    )
    result = LaunchResult(
        committed_rows=committed_rows,
        launch_rows=launch_rows,
        started_keys=frozenset(started_keys),
        status_changes=observed.changes,
        external_status_changes=external_status_changes,
    )
    for committed_row in committed_rows:
        key = validated_run_key(committed_row)
        if key in observed.changes:
            before, after = observed.changes[key]
            options.hooks.append_event(
                workspace,
                "run_status_changed",
                {"step_id": key[0], "run_id": key[1], "from": before, "to": after},
            )
        if key in started_keys:
            options.hooks.append_event(
                workspace,
                "run_launched",
                {"step_id": key[0], "run_id": key[1], "gpus": committed_row.get("gpus", "")},
            )
    if missing_pid_blocker is not None:
        raise missing_pid_blocker
    return result


def _prepare_direct_launch_rows(
    runs: Sequence[Mapping[str, JsonValue]],
    execution: Mapping[str, JsonValue],
    refreshed: Mapping[RunKey, Mapping[str, JsonValue]],
    *,
    hooks: SchedulerHooks,
) -> tuple[list[dict[str, JsonValue]], dict[RunKey, DirectLaunchIdentity]]:
    launch_identity_by_key: dict[RunKey, DirectLaunchIdentity] = {}
    rows: list[dict[str, JsonValue]] = []
    for run in runs:
        key = validated_run_key(run)
        previous = refreshed[key]
        script = Path(str(run["script"]))
        semantic_run_dir = Path(str(run.get("run_dir") or script.parent))
        launch_identity: DirectLaunchIdentity = {
            "target": "local",
            "workdir": execution.get("workdir") or str(REPO_ROOT),
            "gpus": "",
            "log_path": str(semantic_run_dir / "stdout.log"),
            "pid_path": str(semantic_run_dir / "pid"),
            "command": "",
        }
        for field in PROCESS_IDENTITY_FIELDS:
            launch_identity[field] = ""
        launch_identity_by_key[key] = launch_identity
        execution_identity = (
            {field: previous.get(field, "") for field in launch_identity_by_key[key]}
            if previous.get("target") not in (None, "")
            else {field: "" for field in launch_identity_by_key[key]}
        )
        row: dict[str, JsonValue] = {
            **previous,
            **execution_identity,
            "status": previous.get("status") or "planned",
            "launched_at": previous.get("launched_at", ""),
        }
        rows.append(row)
        planned_semantics = {
            field: run[field]
            for field in (
                "experiment_id",
                "step_id",
                "run_id",
                "run_name",
                "parameter_summary",
                "version",
                "config",
                "config_sha256",
                "script",
                "script_sha256",
                "run_dir",
                "artifacts",
                "runtime_dir",
                "checkpoint_dir",
                "terminal_status_owner",
            )
            if field in run
        }
        hooks.validate_run_update(previous, planned_semantics, allow_execution_identity_fill=True)
        hooks.validate_run_update(previous, row, allow_execution_identity_fill=True)
    return rows, launch_identity_by_key


def _commit_direct_launch_rows(
    workspace: Path,
    runs: Sequence[Mapping[str, JsonValue]],
    rows: list[dict[str, JsonValue]],
    workspace_by_key: Mapping[RunKey, Mapping[str, JsonValue]],
    *,
    dry_run: bool,
    hooks: SchedulerHooks,
) -> tuple[list[dict[str, str]], Sequence[Mapping[str, JsonValue]]]:
    commit_rows = []
    for row in rows:
        commit_row = dict(row)
        if dry_run and workspace_by_key[validated_run_key(row)].get("target") in (None, ""):
            commit_row.update({field: "" for field in EXECUTION_IDENTITY_FIELDS})
        commit_rows.append(commit_row)
    committed = hooks.merge_manifest(workspace, commit_rows, lock_held=True)
    committed_by_key = {validated_run_key(row): row for row in committed}
    committed_rows = [committed_by_key[validated_run_key(run)] for run in runs]
    launch_rows: Sequence[Mapping[str, JsonValue]]
    if dry_run:
        preview_by_key = {validated_run_key(row): row for row in rows}
        preview_rows: list[Mapping[str, JsonValue]] = []
        for committed_row in committed_rows:
            preview = preview_by_key[validated_run_key(committed_row)]
            if committed_row.get("target") in (None, ""):
                preview_rows.append(
                    {
                        **committed_row,
                        **{field: preview.get(field, "") for field in EXECUTION_IDENTITY_FIELDS},
                    }
                )
            else:
                preview_rows.append(committed_row)
        launch_rows = preview_rows
    else:
        launch_rows = committed_rows
    return committed_rows, launch_rows


def _refresh_external_capacity_runs(
    *,
    workspace: Path,
    owner_dir: Path,
    workspace_by_key: dict[RunKey, dict[str, str]],
    expected_keys: set[RunKey],
    groups: list[list[Any]],
    execution: Mapping[str, JsonValue],
    dry_run: bool,
    hooks: SchedulerHooks,
) -> tuple[dict[RunKey, dict[str, str]], dict[RunKey, tuple[JsonValue, JsonValue]]]:
    external_keys = []
    if groups:
        external_keys = [
            key
            for key, row in workspace_by_key.items()
            if key not in expected_keys
            and row.get("status") in ACTIVE_STATUSES
            and shares_capacity(execution, groups, row)
            and all(
                row.get(field) not in (None, "")
                for field in ("target", "workdir", "pid_path", "log_path", "command", "script")
            )
        ]
    external_observed = observe_runs(owner_dir, workspace_by_key, external_keys, dry_run=dry_run)
    external_status_changes = external_observed.changes
    if external_status_changes:
        committed = hooks.merge_manifest(
            workspace,
            [external_observed.rows_by_key[key] for key in external_status_changes],
            lock_held=True,
        )
        workspace_by_key = {validated_run_key(row): row for row in committed}
        for key, (before, after) in external_status_changes.items():
            hooks.append_event(
                workspace,
                "run_status_changed",
                {"step_id": key[0], "run_id": key[1], "from": before, "to": after},
            )

    return workspace_by_key, external_status_changes


def _start_direct_launch_rows(
    *,
    workspace: Path,
    rows: list[dict[str, JsonValue]],
    launchable: list[tuple[int, dict[str, JsonValue]]],
    capacity: CapacityState,
    execution: Mapping[str, JsonValue],
    launch_identity_by_key: dict[RunKey, DirectLaunchIdentity],
    workspace_by_key: Mapping[RunKey, Mapping[str, JsonValue]],
    planned_by_key: Mapping[RunKey, Mapping[str, JsonValue]],
    execution_snapshot: exec_snapshot.ExecutionSnapshot | None,
    missing_pid_blocker: MissingPidCapacityError | None,
    dry_run: bool,
    build_command: Callable[..., str],
    start: Callable[..., tuple[str, str]],
    hooks: SchedulerHooks,
) -> set[RunKey]:
    started_keys: set[RunKey] = set()
    if dry_run:
        preview_loads = list(capacity.group_loads)
        for row in rows:
            if row["status"] not in LAUNCHABLE_STATUSES or row.get("target") not in (None, ""):
                continue
            group_index = (
                min(range(len(capacity.gpu_groups)), key=lambda index: (preview_loads[index], index))
                if capacity.gpu_groups
                else None
            )
            gpus = list(capacity.gpu_groups[group_index]) if group_index is not None else []
            identity = launch_identity_by_key[validated_run_key(row)].copy()
            identity["gpus"] = ",".join(str(item) for item in gpus)
            identity["command"] = build_command(
                execution,
                Path(str(row["script"])),
                identity["log_path"],
                identity["pid_path"],
                gpus,
            )
            row.update(**identity)
            hooks.validate_run_update(
                workspace_by_key[validated_run_key(row)],
                row,
                allow_execution_identity_fill=True,
            )
            if group_index is not None:
                preview_loads[group_index] += 1
    else:
        while missing_pid_blocker is None and launchable and capacity.slots > 0:
            allocation = capacity.next_allocation(launchable)
            if allocation is None:
                break
            index, row, group_index = allocation
            launchable = [
                (candidate_index, candidate) for candidate_index, candidate in launchable if candidate_index != index
            ]
            if row.get("target") in (None, ""):
                gpus = list(capacity.gpu_groups[group_index]) if group_index is not None else []
                identity = launch_identity_by_key[validated_run_key(row)].copy()
                identity["gpus"] = ",".join(str(item) for item in gpus)
                planned = planned_by_key[validated_run_key(row)]
                checkpoint_path = planned.get("checkpoint")
                checkpoint_sha256 = planned.get("checkpoint_sha256")
                launch_kwargs: LaunchVerificationOptions = {}
                # Keep these verification arguments absent when the frozen plan does not bind them.
                if checkpoint_path not in (None, "") or checkpoint_sha256 not in (None, ""):
                    launch_kwargs["checkpoint_path"] = (
                        Path(str(checkpoint_path)) if checkpoint_path not in (None, "") else None
                    )
                    launch_kwargs["checkpoint_sha256"] = (
                        str(checkpoint_sha256) if checkpoint_sha256 not in (None, "") else None
                    )
                if execution_snapshot is not None:
                    launch_kwargs["planned_command"] = str(planned["command"])
                    launch_kwargs["run_id"] = str(planned["run_id"])
                identity["command"] = build_command(
                    execution,
                    Path(str(row["script"])),
                    identity["log_path"],
                    identity["pid_path"],
                    gpus,
                    execution_snapshot=execution_snapshot,
                    config_path=Path(str(row["config"])),
                    script_sha256=str(row["script_sha256"]),
                    config_sha256=str(row["config_sha256"]),
                    **launch_kwargs,
                )
                if execution.get("runtime_commit") not in (None, ""):
                    identity["planned_runtime_commit"] = str(execution["runtime_commit"])
                row.update(**identity)
                hooks.validate_run_update(
                    workspace_by_key[validated_run_key(row)],
                    row,
                    allow_execution_identity_fill=True,
                )
            key = validated_run_key(row)
            committed = hooks.merge_manifest(workspace, [row], lock_held=True)
            committed_by_key = {validated_run_key(item): item for item in committed}
            row.clear()
            row.update(committed_by_key[key])
            if row["status"] not in LAUNCHABLE_STATUSES:
                continue
            row["status"], reason = start(execution, row["command"])
            row["launched_at"] = utc_now() if row["status"] == "launched" else ""
            if reason:
                row["scheduler_reason"] = reason
            if row["status"] == "launched":
                try:
                    process_identity = evidence.read_process_identity(row["pid_path"], row)
                except RuntimeError:
                    process_identity = None
                if process_identity is not None:
                    row.update(**process_identity)
            committed = hooks.merge_manifest(workspace, [row], lock_held=True)
            committed_by_key = {validated_run_key(item): item for item in committed}
            row.clear()
            row.update(committed_by_key[key])
            if row["status"] == "launched":
                started_keys.add(key)
                capacity.record_started(group_index)
        for _index, row in launchable:
            if row["status"] == "planned":
                row["status"] = "pending"

    return started_keys


def _managed_scheduler_type(execution: Mapping[str, JsonValue], runs: Sequence[Mapping[str, JsonValue]]) -> str:
    scheduler = execution.get("scheduler") or {}
    if not isinstance(scheduler, dict):
        raise ValueError("execution.scheduler must be a mapping.")
    configured = str(scheduler.get("type") or "direct")
    if configured not in {"direct", "slurm"}:
        raise ValueError("execution.scheduler.type must be direct or slurm.")
    planned = {scheduler_type(run) for run in runs}
    if len(planned) != 1 or planned != {configured}:
        raise ValueError("Frozen run scheduler identity differs from execution.scheduler.type.")
    return configured


def _managed_launch_preflight(
    workspace: Path,
    owner_dir: Path,
    runs: Sequence[Mapping[str, JsonValue]],
) -> tuple[Path, set[RunKey], dict[RunKey, dict[str, str]]]:
    snapshot_path = owner_dir / exec_snapshot.EXECUTION_SNAPSHOT_NAME
    exp_io.validate_managed_output_paths(workspace, [snapshot_path])
    experiment_manifest = yaml.load((workspace / "experiment.yaml").read_text(), Loader=YAML_SAFE_LOADER) or {}
    experiment = experiment_manifest.get("experiment") if isinstance(experiment_manifest, dict) else None
    if isinstance(experiment, dict) and experiment.get("status") == "completed":
        raise ValueError(f"Experiment is completed and cannot launch runs: {workspace}")
    expected_keys = {validated_run_key(run) for run in runs}
    workspace_by_key = {validated_run_key(row): row for row in read_run_manifest(workspace)}
    return snapshot_path, expected_keys, workspace_by_key


def _preflight_slurm_launch(
    workspace: Path,
    owner_dir: Path,
    runs: Sequence[Mapping[str, JsonValue]],
    *,
    hooks: SchedulerHooks,
) -> tuple[Path, dict[RunKey, dict[str, str]]]:
    snapshot_path, expected_keys, workspace_by_key = _managed_launch_preflight(workspace, owner_dir, runs)
    missing = expected_keys - set(workspace_by_key)
    if missing:
        step_id, run_id = sorted(missing)[0]
        raise ValueError(f"Canonical run is missing: {step_id} / {run_id}")
    planned_fields = {
        "experiment_id",
        "step_id",
        "run_id",
        "run_name",
        "parameter_summary",
        "version",
        "config",
        "config_sha256",
        "script",
        "script_sha256",
        "run_dir",
        "artifacts",
        "runtime_dir",
        "checkpoint_dir",
        "terminal_status_owner",
        "log_path",
        *SCHEDULER_PLAN_IDENTITY_FIELDS,
    }
    for run in runs:
        previous = workspace_by_key[validated_run_key(run)]
        hooks.validate_run_update(
            previous,
            {field: run[field] for field in planned_fields if field in run},
            allow_execution_identity_fill=True,
        )
    return snapshot_path, workspace_by_key


def _launch_slurm_runs(
    workspace: Path,
    owner_dir: Path,
    runs: Sequence[Mapping[str, JsonValue]],
    execution: Mapping[str, JsonValue],
    options: LaunchOptions,
) -> LaunchResult:
    """Make one submit pass over the plan's launchable Slurm runs; the caller holds managed_run_lock.

    One ``squeue --me`` snapshot decides each run: adopt the one job carrying its name,
    record ``lost`` when its log or exit code exists without a job, otherwise submit it
    and commit the job id at once. Already submitted runs are left to the monitor."""
    snapshot_path, workspace_by_key = _preflight_slurm_launch(workspace, owner_dir, runs, hooks=options.hooks)
    launchable = [run for run in runs if workspace_by_key[validated_run_key(run)].get("status") in LAUNCHABLE_STATUSES]
    status_changes: dict[RunKey, tuple[JsonValue, JsonValue]] = {}
    started_keys: set[RunKey] = set()
    pending_error: Exception | None = None
    if not options.dry_run and launchable:
        _verify_slurm_submission_inputs(
            owner_dir, runs, launchable, execution, snapshot_path, workspace_by_key, options
        )
        # Launch evidence precedes squeue and sbatch, so a later failure is never recorded as definitely unsubmitted.
        committed = options.hooks.merge_manifest(
            workspace,
            [
                {**workspace_by_key[validated_run_key(run)], **_slurm_execution_identity(execution, run)}
                for run in launchable
            ],
            lock_held=True,
        )
        workspace_by_key = {validated_run_key(row): row for row in committed}
        jobs_by_name: dict[str, list[slurm.JobObservation]] = {}
        for job in slurm.user_jobs(timeout=LAUNCH_TIMEOUT_SECONDS):
            jobs_by_name.setdefault(job.name, []).append(job)
        for run in launchable:
            named_jobs = jobs_by_name.get(slurm.job_name(run), [])
            if len(named_jobs) > 1:
                job_ids = ", ".join(job.job_id for job in named_jobs)
                raise RuntimeError(
                    f"Several Slurm jobs are named {slurm.job_name(run)} ({job_ids}); resolve them before launching."
                )
        for run in launchable:
            key = validated_run_key(run)
            previous = workspace_by_key[key]
            try:
                submitted = _submit_slurm_run(owner_dir, run, previous, jobs_by_name.get(slurm.job_name(run), []))
            except Exception as exc:
                # A timed-out or unreadable sbatch may have queued the job; the next pass adopts it by name.
                pending_error = exc
                break
            committed = options.hooks.merge_manifest(workspace, [submitted], lock_held=True)
            workspace_by_key = {validated_run_key(row): row for row in committed}
            row = workspace_by_key[key]
            if row.get("status") != previous.get("status"):
                status_changes[key] = (previous.get("status"), row.get("status"))
            if row.get("scheduler_job_id") not in (None, ""):
                started_keys.add(key)

    committed_rows = [workspace_by_key[validated_run_key(run)] for run in runs]
    launch_rows: Sequence[Mapping[str, JsonValue]] = committed_rows
    if options.dry_run:
        launch_rows = [
            dict(row, **_slurm_execution_identity(execution, run)) if row.get("target") in (None, "") else row
            for run, row in zip(runs, committed_rows)
        ]
    result = LaunchResult(
        committed_rows=committed_rows,
        launch_rows=launch_rows,
        started_keys=frozenset(started_keys),
        status_changes=status_changes,
        external_status_changes={},
    )
    for key, (before, after) in status_changes.items():
        options.hooks.append_event(
            workspace,
            "run_status_changed",
            {"step_id": key[0], "run_id": key[1], "from": before, "to": after},
        )
    for key in sorted(started_keys):
        options.hooks.append_event(
            workspace,
            "run_launched",
            {"step_id": key[0], "run_id": key[1], "scheduler_job_id": workspace_by_key[key]["scheduler_job_id"]},
        )
    if pending_error is not None:
        raise pending_error
    return result


def _verify_slurm_submission_inputs(
    owner_dir: Path,
    runs: Sequence[Mapping[str, JsonValue]],
    launchable: Sequence[Mapping[str, JsonValue]],
    execution: Mapping[str, JsonValue],
    snapshot_path: Path,
    workspace_by_key: Mapping[RunKey, Mapping[str, JsonValue]],
    options: LaunchOptions,
) -> None:
    runtime_roots = [
        Path(str(run[field]))
        for run in launchable
        for field in ("runtime_dir", "checkpoint_dir")
        if run.get(field) not in (None, "")
    ]
    exp_io.validate_managed_output_paths(Path(str(execution.get("workdir") or REPO_ROOT)), runtime_roots)
    frozen_paths = [
        path
        for run in launchable
        for path in (
            Path(str(run["scheduler_script"])),
            Path(str(run["log_path"])),
            Path(str(run["run_dir"])) / SLURM_EXIT_CODE_NAME,
            Path(str(run["run_dir"])) / SLURM_RUNTIME_COMMIT_NAME,
        )
    ]
    exp_io.validate_managed_output_paths(owner_dir, frozen_paths)
    for run in launchable:
        script_text = exp_io.read_text_at(str(run["scheduler_script"]))
        if hashlib.sha256(script_text.encode()).hexdigest() != run["scheduler_script_sha256"]:
            raise ValueError(f"Frozen Slurm script changed before submission: {run['scheduler_script']}")
    validated_snapshot = options.hooks.validated_snapshot or exec_snapshot.validated_execution_snapshot
    snapshot_result = validated_snapshot(owner_dir, execution, runs, workspace_by_key)
    if snapshot_result[1]:
        exec_snapshot.write_execution_snapshot_file(snapshot_path, snapshot_result[0])


def _submit_slurm_run(
    owner_dir: Path,
    run: Mapping[str, JsonValue],
    row: Mapping[str, JsonValue],
    named_jobs: Sequence[slurm.JobObservation],
) -> dict[str, JsonValue]:
    if named_jobs:
        (job,) = named_jobs
        adopted: dict[str, JsonValue] = {
            **row,
            "scheduler_job_id": job.job_id,
            "launched_at": row.get("launched_at") or utc_now(),
        }
        adopted.update(_slurm_state(owner_dir, adopted, job))
        return adopted
    run_dir = Path(str(run["run_dir"]))
    if any(os.path.lexists(path) for path in (str(run["log_path"]), run_dir / SLURM_EXIT_CODE_NAME)):
        # The job finished and aged out of squeue; without accounting its id cannot be recovered.
        return {**row, "status": "lost", "scheduler_reason": "job ran earlier; id unknown"}
    try:
        job_id = slurm.submit(str(run["scheduler_script"]), timeout=LAUNCH_TIMEOUT_SECONDS)
    except slurm.SlurmCommandError as exc:
        return {**row, "status": "launch_failed", "scheduler_reason": str(exc)}
    return {
        **row,
        "status": "queued",
        "scheduler_job_id": job_id,
        "scheduler_raw_state": "",
        "scheduler_reason": "",
        "launched_at": utc_now(),
    }


def _slurm_execution_identity(execution: Mapping[str, JsonValue], run: Mapping[str, JsonValue]) -> SlurmLaunchIdentity:
    identity: SlurmLaunchIdentity = {
        "target": "local",
        "workdir": execution.get("workdir") or str(REPO_ROOT),
        "gpus": "",
        "pid_path": "",
        "log_path": run["log_path"],
        "command": slurm.submission_command(str(run["scheduler_script"])),
    }
    for field in PROCESS_IDENTITY_FIELDS:
        identity[field] = ""
    if execution.get("runtime_commit") not in (None, ""):
        identity["planned_runtime_commit"] = str(execution["runtime_commit"])
    return identity


def stop_slurm_run_locked(
    workspace: Path,
    workspace_rows: list[dict[str, str]],
    key: RunKey,
    *,
    reason: str,
    hooks: SchedulerHooks,
    now: Callable[[], str],
) -> tuple[list[dict[str, str]], bool]:
    """Stop one Slurm run; return the committed rows and whether it was stopped without scancel.

    A run with a job id records its stopping intent, then scancel; the monitor later commits
    stopped. A launchable run is looked up by name and cancelled when found, otherwise stopped
    in metadata. A lost run is stopped in metadata without scancel and is never resubmitted."""
    # The caller holds managed_run_lock and supplies its freshly read, validated canonical rows.
    previous = next(row for row in workspace_rows if validated_run_key(row) == key)
    run_id = key[1]
    if previous.get("status") in TERMINAL_STATUSES:
        raise ValueError(f"Run is already terminal and cannot be stopped: {run_id} ({previous['status']})")
    pending_stop = previous.get("stop_requested_at") not in (None, "")
    if pending_stop and previous.get("stop_reason") != reason:
        raise ValueError(f"Run already has a pending stop request with a different reason: {run_id}")
    job_id = str(previous.get("scheduler_job_id") or "")
    if previous.get("status") in LAUNCHABLE_STATUSES:
        name = slurm.job_name(previous)
        matches = [job for job in slurm.user_jobs() if job.name == name]
        if len(matches) > 1:
            raise ValueError(f"Several Slurm jobs are named {name}; cancel the right one by hand: {run_id}")
        job_id = matches[0].job_id if matches else ""
    if previous.get("status") == "lost" or not job_id:
        final = merge_run_row(
            previous,
            {"step_id": key[0], "run_id": run_id, "status": "stopped", "stopped_at": now(), "stop_reason": reason},
        )
        return hooks.merge_manifest(workspace, [final], lock_held=True), True
    if pending_stop:
        committed = workspace_rows
    else:
        final = merge_run_row(
            previous,
            {
                "step_id": key[0],
                "run_id": run_id,
                "scheduler_job_id": job_id,
                "status": "stopping",
                "stop_requested_at": now(),
                "stop_reason": reason,
            },
        )
        committed = hooks.merge_manifest(workspace, [final], lock_held=True)
        hooks.append_event(
            workspace,
            "run_stop_requested",
            {"step_id": key[0], "run_id": run_id, "reason": reason},
        )
    slurm.cancel(job_id)
    return committed, False


def slurm_state_needed(row: Mapping[str, JsonValue]) -> bool:
    """Whether the Slurm state table applies: a non-terminal Slurm run bound to a job id."""
    return (
        scheduler_type(row) == "slurm"
        and row.get("scheduler_job_id") not in (None, "")
        and row.get("status") not in TERMINAL_STATUSES
    )


def slurm_job_snapshot(rows: Iterable[Mapping[str, JsonValue]]) -> dict[str, slurm.JobObservation]:
    """Return one ``squeue --me`` snapshot keyed by job id; no query when no row needs one."""
    if not any(slurm_state_needed(row) for row in rows):
        return {}
    return {job.job_id: job for job in slurm.user_jobs()}


def observe_slurm_run(
    owner_dir: str | Path,
    row: Mapping[str, JsonValue],
    jobs: Mapping[str, slurm.JobObservation] | None = None,
    *,
    health: bool = False,
) -> dict[str, JsonValue]:
    """Observe one Slurm run and return evidence for the caller to commit.

    For a non-terminal run with a job id, applies the Slurm state table to
    run_dir/exit_code, the jobs snapshot (one ``squeue --me`` query when jobs is None)
    and the recorded stop request, and copies run_dir/runtime_commit. Other rows only
    refresh artifacts. Never submits, cancels or persists."""
    observation: dict[str, JsonValue] = dict(row)
    if slurm_state_needed(row):
        snapshot = jobs if jobs is not None else slurm_job_snapshot([row])
        observation.update(_slurm_state(Path(owner_dir), row, snapshot.get(str(row["scheduler_job_id"]))))
    return _slurm_artifact_observation(observation, health=health)


def _slurm_state(
    owner_dir: Path,
    row: Mapping[str, JsonValue],
    job: slurm.JobObservation | None,
) -> dict[str, JsonValue]:
    run_dir = Path(str(row["run_dir"]))
    exit_path = run_dir / SLURM_EXIT_CODE_NAME
    commit_path = run_dir / SLURM_RUNTIME_COMMIT_NAME
    texts = exp_io.read_managed_output_texts_at(owner_dir, [exit_path, commit_path])
    update: dict[str, JsonValue] = {}
    runtime_commit = (texts[str(commit_path)] or "").strip()
    if runtime_commit:
        if not is_full_git_object_id(runtime_commit):
            raise ValueError(f"Slurm runtime_commit must be a full 40-character Git commit ID: {commit_path}")
        update["runtime_commit"] = runtime_commit
    if job is not None:
        update.update(
            scheduler_raw_state=job.state,
            scheduler_reason=job.reason,
            scheduler_node=job.node_list or row.get("scheduler_node", ""),
        )
    stop_requested = row.get("stop_requested_at") not in (None, "")
    exit_text = texts[str(exit_path)]
    if exit_text is not None:
        exit_code = exit_text.strip()
        if not (exit_code.isascii() and exit_code.isdecimal() and int(exit_code) <= 255):
            raise ValueError(f"Slurm exit_code must be an integer from 0 to 255: {exit_path}")
        update["scheduler_exit_code"] = int(exit_code)
        update["status"] = "completed" if int(exit_code) == 0 else "stopped" if stop_requested else "failed"
    elif job is not None and job.state in slurm.RUNNING_STATES:
        update["status"] = "stopping" if stop_requested else "running"
    elif job is not None and job.state in slurm.QUEUED_STATES:
        update["status"] = "stopping" if stop_requested else "queued"
    elif stop_requested:
        update["status"] = "stopped"
    else:
        # Not terminal: a later exit_code upgrades it; otherwise only a reasoned stop resolves it.
        update["status"] = "lost"
        update["scheduler_reason"] = (
            f"Slurm reports {job.state} but the job wrote no exit_code."
            if job is not None
            else "Slurm no longer reports the job and it wrote no exit_code."
        )
    if update["status"] == "stopped":
        update["stopped_at"] = row.get("stopped_at") or utc_now()
    return update


def _slurm_artifact_observation(row: dict[str, JsonValue], *, health: bool = False) -> dict[str, JsonValue]:
    run_manifest, _manifest, checkpoints = evidence.runtime_artifacts(row)
    row.update({"run_manifest": run_manifest, "checkpoints": ";".join(checkpoints)})
    row["log_tail"] = evidence.log_tail(row.get("log_path"))
    if health:
        status = str(row.get("status") or "")
        log_age = evidence.log_age_seconds(row.get("log_path"))
        row["health_status"] = {"queued": "scheduler_queued", "running": "scheduler_running"}.get(status, status)
        row["log_age_seconds"] = "" if log_age is None else log_age
    row["monitored_at"] = utc_now()
    return row


def build_launch_command(
    execution: Mapping[str, Any],
    script: Path,
    log_path: str | Path,
    pid_path: str | Path,
    gpus: list[Any],
    *,
    execution_snapshot: exec_snapshot.ExecutionSnapshot | None = None,
    config_path: Path | None = None,
    script_sha256: str | None = None,
    config_sha256: str | None = None,
    checkpoint_path: Path | None = None,
    checkpoint_sha256: str | None = None,
    planned_command: str | None = None,
    run_id: str = "",
) -> str:
    workdir = str(execution.get("workdir") or REPO_ROOT)
    env = dict(execution.get("env") or {})
    if gpus:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(item) for item in gpus)
    run = [
        str(execution.get("python") or sys.executable),
        "-c",
        python_programs.source("managed_scheduler.process_launch"),
        str(script),
        str(log_path),
        str(pid_path),
        workdir,
        str(execution.get("runtime_commit") or ""),
    ]
    verification_commands = []
    if execution_snapshot is not None or any(
        value is not None for value in (config_path, script_sha256, config_sha256, checkpoint_path, checkpoint_sha256)
    ):
        if config_path is None or not script_sha256 or not config_sha256:
            raise ValueError("Verified launch requires frozen script and config hashes.")
        artifacts: list[FrozenLaunchArtifact] = [
            {"path": str(script), "sha256": script_sha256},
            {"path": str(config_path), "sha256": config_sha256},
        ]
        if (checkpoint_path is None) != (checkpoint_sha256 is None) or checkpoint_sha256 == "":
            raise ValueError("Verified launch requires both frozen checkpoint path and hash.")
        if checkpoint_path is not None and checkpoint_sha256 is not None:
            artifacts.append({"path": str(checkpoint_path), "sha256": checkpoint_sha256})
        if execution_snapshot is not None:
            if planned_command is None:
                raise ValueError("Verified module launch requires its frozen command.")
            tokens = shlex.split(planned_command)
            try:
                module_index = tokens.index("-m") + 1
            except (ValueError, IndexError) as exc:
                raise ValueError("Verified module launch command has no Python module.") from exc
            if module_index != 2 or tokens[module_index] != execution_snapshot["module"]:
                raise ValueError("Verified module launch command differs from its execution snapshot.")
            planned_argv: list[exec_snapshot.PlannedArgv] = [
                {"run_id": run_id or script.stem, "args": tokens[module_index + 1 :]}
            ]
            verification_commands.extend(
                [
                    (
                        execution["python"],
                        "-c",
                        python_programs.source("managed_scheduler.runtime_identity"),
                        execution_snapshot["module"],
                        json.dumps(execution_snapshot, sort_keys=True),
                        json.dumps(artifacts, sort_keys=True),
                        str(execution_snapshot.get("expected_runtime_commit") or execution["runtime_commit"]),
                    ),
                    (
                        execution["python"],
                        "-c",
                        python_programs.source("managed_scheduler.cli_preflight"),
                        execution_snapshot["module"],
                        json.dumps(planned_argv, sort_keys=True),
                        execution_snapshot["module_origin"],
                    ),
                ]
            )
        else:
            # Script-based plans verify the lifecycle runtime without claiming the workload's CLI contract.
            verification_commands.append(
                (
                    run[0],
                    "-c",
                    python_programs.source("managed_scheduler.runtime_identity"),
                    "agent_tools.experiment_workspace",
                    "{}",
                    json.dumps(artifacts, sort_keys=True),
                    str(execution["runtime_commit"]),
                )
            )
    run.append(json.dumps(verification_commands))
    if execution.get("conda_env"):
        wrapper = ["conda", "run", "--no-capture-output", "-n", str(execution["conda_env"])]
        run = [*wrapper, *run]
    run_command = " ".join(_sh(part) for part in run)
    if env:
        env_prefix = " ".join(f"{key}={_sh(value)}" for key, value in sorted(env.items()))
        run_command = f"env {env_prefix} {run_command}"
    return f"cd {_sh(workdir)} && {run_command}"


def start_process(execution: Mapping[str, JsonValue], command: str) -> tuple[str, str]:
    """Run the launch wrapper and return ``(status, scheduler_reason)``.

    A non-zero wrapper exit is a definite ``launch_failed``: before spawn no process exists, and after spawn the
    wrapper stopped the unrecorded group. The reason is a bounded tail of the wrapper's stderr."""
    result = subprocess.run(["bash", "-lc", command], text=True, capture_output=True)
    if result.returncode == 0:
        return "launched", ""
    # One line keeps the reason readable in the TSV manifest and the monitor/status tables.
    tail = " / ".join(line.strip() for line in result.stderr.splitlines() if line.strip())[-1000:]
    return "launch_failed", f"launch exit {result.returncode}" + (f": {tail}" if tail else "")


_sh = transport.sh
