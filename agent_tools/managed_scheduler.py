from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
from typing import Any

import yaml

from . import experiment_io as exp_io, gpu_rules, python_programs, run_evidence as evidence, slurm, transport
from .experiment_workspace import (
    EXECUTION_IDENTITY_FIELDS,
    LAUNCHABLE_STATUSES,
    PROCESS_IDENTITY_FIELDS,
    SCHEDULER_PLAN_IDENTITY_FIELDS,
    SUBMISSION_CLUSTER_MISMATCH,
    TERMINAL_STATUSES,
    append_event,
    file_sha256,
    has_managed_launch_evidence,
    managed_run_key,
    merge_run_manifest,
    merge_run_row,
    read_run_manifest,
    scheduler_direct_controller,
    scheduler_type,
    validate_frozen_run_update,
    write_status_report,
)
from .manifests import read_json, utc_now
from .models import REPO_ROOT, is_full_git_object_id

RunKey = tuple[str, str]
ACTIVE_STATUSES = frozenset(
    {"submitting", "queued", "launched", "running", "stopping", "unknown_remote", "unknown_scheduler", "missing_pid"}
)
LAUNCH_TIMEOUT_SECONDS = 60
EXECUTION_SNAPSHOT_NAME = "execution_snapshot.json"
DIRECT_LAUNCH_CAPABILITIES = ("commit_run_start", "runtime_lock")
SLURM_LAUNCH_CAPABILITIES = (*DIRECT_LAUNCH_CAPABILITIES, "slurm_runtime_lock_fd")


class MissingPidCapacityError(RuntimeError):
    def __init__(self, step_id: str, run_id: str):
        self.step_id = step_id
        self.run_id = run_id
        super().__init__(f"Managed launch capacity is blocked because {step_id} / {run_id} has status missing_pid.")


__all__ = [
    "ACTIVE_STATUSES",
    "CapacityState",
    "EXECUTION_SNAPSHOT_NAME",
    "LAUNCHABLE_STATUSES",
    "LAUNCH_TIMEOUT_SECONDS",
    "LaunchResult",
    "MissingPidCapacityError",
    "SchedulerHooks",
    "SlurmMonitorContext",
    "build_launch_command",
    "capacity_state",
    "gpu_groups",
    "inspect_execution_target",
    "launch_managed_runs",
    "managed_run_lock",
    "observe_run",
    "observe_runs",
    "observe_slurm_run",
    "run_execution_command",
    "script_commits_terminal_status",
    "shares_capacity",
    "start_process",
    "stop_slurm_run_locked",
    "validated_execution_snapshot",
]


@dataclass(frozen=True)
class StatusChanges:
    rows_by_key: dict[RunKey, dict[str, Any]]
    changes: dict[RunKey, tuple[Any, Any]]


@dataclass
class CapacityState:
    gpu_groups: list[list[Any]]
    group_loads: list[int]
    assigned_group_by_key: dict[RunKey, int]
    slots: int
    allow_gpu_oversubscription: bool
    external_missing_pid: list[RunKey]

    def preview_group(self) -> int | None:
        if not self.gpu_groups:
            return None
        return min(range(len(self.gpu_groups)), key=lambda index: (self.group_loads[index], index))

    def next_allocation(
        self,
        candidates: Sequence[tuple[int, dict[str, Any]]],
    ) -> tuple[int, dict[str, Any], int | None] | None:
        eligible: list[tuple[int, int, dict[str, Any], int | None]] = []
        for index, row in candidates:
            frozen_group_index = self.assigned_group_by_key.get(managed_run_key(row))
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
    committed_rows: list[dict[str, Any]]
    launch_rows: list[dict[str, Any]]
    started_keys: frozenset[RunKey]
    status_changes: dict[RunKey, tuple[Any, Any]]
    external_status_changes: dict[RunKey, tuple[Any, Any]]


@dataclass(frozen=True)
class SchedulerHooks:
    merge_manifest: Callable[..., list[dict[str, Any]]] = merge_run_manifest
    append_event: Callable[..., Path] = append_event
    write_status_report: Callable[..., Path] = write_status_report
    validate_run_update: Callable[..., None] = validate_frozen_run_update
    validated_snapshot: Callable[..., tuple[dict[str, Any] | None, bool]] | None = None
    build_command: Callable[..., str] | None = None
    start_process: Callable[..., str] | None = None


@contextmanager
def managed_run_lock(workspace: str | Path):
    root = Path(workspace)
    lock_path = root / "run_manifest.tsv.lock"
    exp_io.validate_managed_output_paths(root, [lock_path])
    with exp_io.blocking_file_lock(lock_path):
        yield


def gpu_groups(execution: dict[str, Any], runtime: dict[str, Any]) -> list[list[Any]]:
    groups, issues = gpu_rules.gpu_group_plan(execution, runtime)
    errors = [issue for issue in issues if not issue.warning]
    if errors:
        raise ValueError(errors[0].message)
    return groups


def script_commits_terminal_status(row: dict[str, Any], *, default: bool = False) -> bool:
    owner = row.get("terminal_status_owner")
    if owner in (None, ""):
        return default
    if owner == "script":
        return True
    if owner in {"monitor", "scheduler_sidecar"}:
        return False
    raise ValueError("terminal_status_owner must be 'script', 'monitor', or 'scheduler_sidecar'.")


def observe_run(
    run_dir: str | Path,
    row: dict[str, Any],
    previous: dict[str, Any] | None = None,
    *,
    health: bool = False,
    default_script_commits_terminal_status: bool = False,
) -> dict[str, Any]:
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
        script_commits_terminal_status=script_commits_terminal_status(
            prior,
            default=default_script_commits_terminal_status,
        ),
        health=health,
    )


def observe_runs(
    run_dir: str | Path,
    rows_by_key: dict[RunKey, dict[str, Any]],
    keys: Iterable[RunKey],
    *,
    dry_run: bool,
    health: bool = False,
    default_script_commits_terminal_status: bool = False,
) -> StatusChanges:
    refreshed: dict[RunKey, dict[str, Any]] = {}
    changes: dict[RunKey, tuple[Any, Any]] = {}
    for key in keys:
        previous = rows_by_key.get(key)
        if previous is None:
            raise ValueError(f"Canonical run is missing: {key[0]} / {key[1]}")
        if dry_run or previous.get("target") in (None, ""):
            refreshed[key] = previous
            continue
        observed = observe_run(
            run_dir,
            previous,
            previous,
            health=health,
            default_script_commits_terminal_status=default_script_commits_terminal_status,
        )
        refreshed[key] = observed
        if observed.get("status") != previous.get("status"):
            changes[key] = (previous.get("status"), observed.get("status"))
    return StatusChanges(refreshed, changes)


def capacity_state(
    execution: dict[str, Any],
    runtime: dict[str, Any],
    expected_rows: dict[RunKey, dict[str, Any]],
    workspace_rows: dict[RunKey, dict[str, Any]],
    *,
    expected_keys: set[RunKey],
) -> CapacityState:
    groups = gpu_groups(execution, runtime)
    raw_max_concurrent = execution.get("max_concurrent", max(len(groups), 1))
    if type(raw_max_concurrent) is not int or raw_max_concurrent <= 0:
        raise ValueError("execution.max_concurrent must be a positive integer.")
    max_concurrent = raw_max_concurrent
    allow_gpu_oversubscription = bool(groups) and max_concurrent > len(groups)
    target = str(execution.get("target", "local") or "local")
    current_host = str(execution.get("host") or "") if target == "ssh" else ""
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
        row_target = str(row.get("target") or "")
        if not row_target:
            unknown_other_active += 1
            continue
        if row_target != target:
            continue
        if target == "ssh":
            row_host = str(row.get("host") or "")
            if not row_host:
                unknown_other_active += 1
                continue
            if row_host != current_host:
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
        assigned = ",".join(part.strip() for part in str(previous.get("gpus") or "").split(",") if part.strip())
        if not assigned:
            if previous.get("status") in ACTIVE_STATUSES:
                for group_index in range(len(groups)):
                    group_loads[group_index] += 1
            continue
        group_index = group_by_value.get(assigned)
        if group_index is None:
            raise ValueError(f"Frozen GPUs are not one configured GPU group for {key[0]} / {key[1]}: {assigned}")
        assigned_group_by_key[key] = group_index
        if previous.get("status") in ACTIVE_STATUSES:
            group_loads[group_index] += 1
    return CapacityState(
        gpu_groups=groups,
        group_loads=group_loads,
        assigned_group_by_key=assigned_group_by_key,
        slots=max(max_concurrent - active, 0),
        allow_gpu_oversubscription=allow_gpu_oversubscription,
        external_missing_pid=external_missing_pid,
    )


def shares_capacity(
    execution: dict[str, Any],
    groups: list[list[Any]],
    row: dict[str, Any],
) -> bool:
    if scheduler_type(row) == "slurm":
        return False
    row_target = str(row.get("target") or "")
    target = str(execution.get("target", "local") or "local")
    if row_target and row_target != target:
        return False
    if target == "ssh" and row_target:
        row_host = str(row.get("host") or "")
        current_host = str(execution.get("host") or "")
        if row_host and row_host != current_host:
            return False
    assigned = {part.strip() for part in str(row.get("gpus") or "").split(",") if part.strip()}
    current_gpu_pool = {str(item) for group in groups for item in group}
    return not assigned or bool(assigned.intersection(current_gpu_pool))


def launch_managed_runs(
    workspace: str | Path,
    owner_dir: str | Path,
    runs: list[dict[str, Any]],
    execution: dict[str, Any],
    runtime: dict[str, Any],
    *,
    dry_run: bool = True,
    fail_on_missing_pid_blocker: bool = False,
    default_script_commits_terminal_status: bool = False,
    runtime_output_fields: tuple[str, ...] = ("runtime_dir", "checkpoint_dir"),
    runtime_output_root: str | Path | None = None,
    projection_writer: Callable[[LaunchResult], None] | None = None,
    hooks: SchedulerHooks | None = None,
    lock_held: bool = False,
) -> LaunchResult:
    root = Path(workspace)
    managed_dir = Path(owner_dir)
    if lock_held:
        return _launch_managed_runs(
            root,
            managed_dir,
            runs,
            execution,
            runtime,
            dry_run=dry_run,
            fail_on_missing_pid_blocker=fail_on_missing_pid_blocker,
            default_script_commits_terminal_status=default_script_commits_terminal_status,
            runtime_output_fields=runtime_output_fields,
            runtime_output_root=runtime_output_root,
            projection_writer=projection_writer,
            hooks=hooks or SchedulerHooks(),
        )
    with managed_run_lock(root):
        return _launch_managed_runs(
            root,
            managed_dir,
            runs,
            execution,
            runtime,
            dry_run=dry_run,
            fail_on_missing_pid_blocker=fail_on_missing_pid_blocker,
            default_script_commits_terminal_status=default_script_commits_terminal_status,
            runtime_output_fields=runtime_output_fields,
            runtime_output_root=runtime_output_root,
            projection_writer=projection_writer,
            hooks=hooks or SchedulerHooks(),
        )


def _launch_managed_runs(
    workspace: Path,
    owner_dir: Path,
    runs: list[dict[str, Any]],
    execution: dict[str, Any],
    runtime: dict[str, Any],
    *,
    dry_run: bool,
    fail_on_missing_pid_blocker: bool,
    default_script_commits_terminal_status: bool,
    runtime_output_fields: tuple[str, ...],
    runtime_output_root: str | Path | None,
    projection_writer: Callable[[LaunchResult], None] | None,
    hooks: SchedulerHooks,
) -> LaunchResult:
    backend = _managed_scheduler_type(execution, runs)
    if backend == "slurm":
        return _launch_slurm_runs(
            workspace,
            owner_dir,
            runs,
            execution,
            dry_run=dry_run,
            runtime_output_fields=runtime_output_fields,
            runtime_output_root=runtime_output_root,
            projection_writer=projection_writer,
            hooks=hooks,
        )
    planned_by_key = {managed_run_key(run): run for run in runs}
    snapshot_path, expected_keys, workspace_by_key = _managed_launch_preflight(workspace, owner_dir, runs)
    if (
        not dry_run
        and not snapshot_path.exists()
        and any(
            workspace_by_key[managed_run_key(run)].get("target") not in (None, "")
            or workspace_by_key[managed_run_key(run)].get("status") not in LAUNCHABLE_STATUSES
            for run in runs
        )
    ):
        validated_snapshot = hooks.validated_snapshot or validated_execution_snapshot
        validated_snapshot(owner_dir, execution, runs, workspace_by_key)

    observed = observe_runs(
        owner_dir,
        workspace_by_key,
        expected_keys,
        dry_run=dry_run,
        default_script_commits_terminal_status=default_script_commits_terminal_status,
    )
    refreshed = observed.rows_by_key
    external_status_changes: dict[RunKey, tuple[Any, Any]] = {}
    groups = gpu_groups(execution, runtime)
    if groups:
        for key, row in list(workspace_by_key.items()):
            if (
                key in expected_keys
                or row.get("status") not in ACTIVE_STATUSES
                or not shares_capacity(execution, groups, row)
            ):
                continue
            observable = all(
                row.get(field) not in (None, "")
                for field in ("target", "workdir", "pid_path", "log_path", "command", "script")
            ) and (row.get("target") != "ssh" or row.get("host") not in (None, ""))
            if not dry_run and observable:
                observed_row = observe_run(owner_dir, row, row)
                if observed_row.get("status") != row.get("status"):
                    external_status_changes[key] = (row.get("status"), observed_row.get("status"))
                    workspace_by_key[key] = observed_row
    if external_status_changes:
        committed = hooks.merge_manifest(
            workspace,
            [workspace_by_key[key] for key in external_status_changes],
            lock_held=True,
        )
        workspace_by_key = {managed_run_key(row): row for row in committed}
        for key, (before, after) in external_status_changes.items():
            hooks.append_event(
                workspace,
                "run_status_changed",
                {"step_id": key[0], "run_id": key[1], "from": before, "to": after},
            )
        hooks.write_status_report(workspace)

    capacity = capacity_state(
        execution,
        runtime,
        refreshed,
        workspace_by_key,
        expected_keys=expected_keys,
    )
    missing_pid_blocker = None
    if not dry_run and fail_on_missing_pid_blocker:
        current_missing_pid = [key for key, row in refreshed.items() if row.get("status") == "missing_pid"]
        capacity_needed = any(row.get("status") in ACTIVE_STATUSES | LAUNCHABLE_STATUSES for row in refreshed.values())
        external_missing_pid = capacity.external_missing_pid if capacity_needed else []
        blockers = sorted(set(current_missing_pid) | set(external_missing_pid))
        if blockers:
            missing_pid_blocker = MissingPidCapacityError(*blockers[0])

    target = str(execution.get("target", "local") or "local")
    launch_identity_by_key: dict[RunKey, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for run in runs:
        key = managed_run_key(run)
        previous = refreshed[key]
        script = Path(str(run["script"]))
        semantic_run_dir = Path(str(run.get("run_dir") or script.parent))
        launch_identity_by_key[key] = {
            "target": target,
            "host": execution.get("host", ""),
            "workdir": execution.get("workdir") or str(REPO_ROOT),
            "gpus": "",
            "log_path": str(semantic_run_dir / "stdout.log"),
            "pid_path": str(semantic_run_dir / "pid"),
            "command": "",
            **{field: "" for field in PROCESS_IDENTITY_FIELDS},
        }
        execution_identity = (
            {field: previous.get(field, "") for field in launch_identity_by_key[key]}
            if previous.get("target") not in (None, "")
            else {field: "" for field in launch_identity_by_key[key]}
        )
        row = {
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
                "pipeline_id",
                "job_id",
                "attempt",
                "result_root",
                "terminal_status_owner",
            )
            if field in run
        }
        hooks.validate_run_update(previous, planned_semantics, allow_execution_identity_fill=True)
        hooks.validate_run_update(previous, row, allow_execution_identity_fill=True)

    run_output_paths = [
        Path(str(launch_identity_by_key[managed_run_key(row)][field]))
        for row in rows
        for field in ("log_path", "pid_path")
    ]
    if target == "ssh":
        if not dry_run:
            exp_io.validate_managed_output_paths(workspace, run_output_paths, remote=str(execution["host"]))
    else:
        exp_io.validate_managed_output_paths(workspace, run_output_paths)

    execution_snapshot = None
    if not dry_run and missing_pid_blocker is None:
        launchable = [row for row in rows if row["status"] in LAUNCHABLE_STATUSES]
        has_launch_candidate = False
        if capacity.slots > 0:
            has_launch_candidate = capacity.next_allocation(list(enumerate(launchable))) is not None
        if has_launch_candidate:
            runtime_roots = [
                Path(str(row[field]))
                for row in launchable
                for field in runtime_output_fields
                if row.get(field) not in (None, "")
            ]
            runtime_root = (
                Path(runtime_output_root)
                if runtime_output_root is not None
                else Path(str(execution.get("workdir") or REPO_ROOT))
            )
            remote_host = str(execution["host"]) if target == "ssh" else None
            exp_io.validate_managed_output_paths(runtime_root, runtime_roots, remote=remote_host)
            validated_snapshot = hooks.validated_snapshot or validated_execution_snapshot
            execution_snapshot, write_execution_snapshot = validated_snapshot(
                owner_dir,
                execution,
                runs,
                workspace_by_key,
            )
            if write_execution_snapshot:
                write_execution_snapshot_file(snapshot_path, execution_snapshot)
    if target != "ssh":
        for row in rows:
            Path(str(row["run_dir"])).mkdir(parents=True, exist_ok=True)

    build_command = hooks.build_command or build_launch_command
    start = hooks.start_process or start_process
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
            identity = dict(launch_identity_by_key[managed_run_key(row)])
            identity["gpus"] = ",".join(str(item) for item in gpus)
            identity["command"] = build_command(
                execution,
                Path(str(row["script"])),
                identity["log_path"],
                identity["pid_path"],
                gpus,
            )
            row.update(identity)
            hooks.validate_run_update(
                workspace_by_key[managed_run_key(row)],
                row,
                allow_execution_identity_fill=True,
            )
            if group_index is not None:
                preview_loads[group_index] += 1
    else:
        launchable = [(index, row) for index, row in enumerate(rows) if row["status"] in LAUNCHABLE_STATUSES]
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
                identity = dict(launch_identity_by_key[managed_run_key(row)])
                identity["gpus"] = ",".join(str(item) for item in gpus)
                planned = planned_by_key[managed_run_key(row)]
                checkpoint_path = planned.get("checkpoint")
                checkpoint_sha256 = planned.get("checkpoint_sha256")
                checkpoint_args = (
                    {
                        "checkpoint_path": Path(str(checkpoint_path)) if checkpoint_path not in (None, "") else None,
                        "checkpoint_sha256": (str(checkpoint_sha256) if checkpoint_sha256 not in (None, "") else None),
                    }
                    if checkpoint_path not in (None, "") or checkpoint_sha256 not in (None, "")
                    else {}
                )
                runtime_verification_args = (
                    {
                        "planned_command": str(planned["command"]),
                        "run_id": str(planned["run_id"]),
                    }
                    if execution_snapshot is not None
                    else {}
                )
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
                    **runtime_verification_args,
                    **checkpoint_args,
                )
                if execution.get("runtime_commit") not in (None, ""):
                    identity["planned_runtime_commit"] = str(execution["runtime_commit"])
                row.update(identity)
                hooks.validate_run_update(
                    workspace_by_key[managed_run_key(row)],
                    row,
                    allow_execution_identity_fill=True,
                )
            key = managed_run_key(row)
            committed = hooks.merge_manifest(workspace, [row], lock_held=True)
            committed_by_key = {managed_run_key(item): item for item in committed}
            row.clear()
            row.update(committed_by_key[key])
            if row["status"] not in LAUNCHABLE_STATUSES:
                continue
            row["status"] = start(execution, row["command"])
            row["launched_at"] = utc_now() if row["status"] == "launched" else ""
            if row["status"] == "launched":
                try:
                    process_identity = evidence.read_process_identity(row["pid_path"], row)
                except RuntimeError:
                    process_identity = None
                if process_identity is not None:
                    row.update(process_identity)
            committed = hooks.merge_manifest(workspace, [row], lock_held=True)
            committed_by_key = {managed_run_key(item): item for item in committed}
            row.clear()
            row.update(committed_by_key[key])
            if row["status"] == "launched":
                started_keys.add(key)
                capacity.record_started(group_index)
        for _index, row in launchable:
            if row["status"] == "planned":
                row["status"] = "pending"

    commit_rows = []
    for row in rows:
        committed_row = dict(row)
        if dry_run and workspace_by_key[managed_run_key(row)].get("target") in (None, ""):
            committed_row.update({field: "" for field in EXECUTION_IDENTITY_FIELDS})
        commit_rows.append(committed_row)
    committed = hooks.merge_manifest(workspace, commit_rows, lock_held=True)
    committed_by_key = {managed_run_key(row): row for row in committed}
    committed_rows = [committed_by_key[managed_run_key(run)] for run in runs]
    if dry_run:
        preview_by_key = {managed_run_key(row): row for row in rows}
        launch_rows = []
        for committed_row in committed_rows:
            preview = preview_by_key[managed_run_key(committed_row)]
            if committed_row.get("target") in (None, ""):
                launch_rows.append(
                    {
                        **committed_row,
                        **{field: preview.get(field, "") for field in EXECUTION_IDENTITY_FIELDS},
                    }
                )
            else:
                launch_rows.append(committed_row)
    else:
        launch_rows = committed_rows
    result = LaunchResult(
        committed_rows=committed_rows,
        launch_rows=launch_rows,
        started_keys=frozenset(started_keys),
        status_changes=observed.changes,
        external_status_changes=external_status_changes,
    )
    if projection_writer is not None:
        projection_writer(result)
    for row in committed_rows:
        key = managed_run_key(row)
        if key in observed.changes:
            before, after = observed.changes[key]
            hooks.append_event(
                workspace,
                "run_status_changed",
                {"step_id": key[0], "run_id": key[1], "from": before, "to": after},
            )
        if key in started_keys:
            hooks.append_event(
                workspace,
                "run_launched",
                {"step_id": key[0], "run_id": key[1], "gpus": row.get("gpus", "")},
            )
    hooks.write_status_report(workspace)
    if missing_pid_blocker is not None:
        raise missing_pid_blocker
    return result


def _managed_scheduler_type(execution: dict[str, Any], runs: list[dict[str, Any]]) -> str:
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
    runs: list[dict[str, Any]],
) -> tuple[Path, set[RunKey], dict[RunKey, dict[str, Any]]]:
    snapshot_path = owner_dir / EXECUTION_SNAPSHOT_NAME
    exp_io.validate_managed_output_paths(
        workspace,
        [
            workspace / "run_manifest.tsv",
            workspace / "run_matrix.csv",
            workspace / "reports" / "run_matrix.md",
            workspace / "events.jsonl",
            workspace / "reports" / "status.md",
            snapshot_path,
        ],
    )
    experiment_manifest = yaml.safe_load((workspace / "experiment.yaml").read_text()) or {}
    experiment = experiment_manifest.get("experiment") if isinstance(experiment_manifest, dict) else None
    if isinstance(experiment, dict) and experiment.get("status") == "completed":
        raise ValueError(f"Experiment is completed and cannot launch runs: {workspace}")
    expected_keys = {managed_run_key(run) for run in runs}
    workspace_by_key = {managed_run_key(row): row for row in read_run_manifest(workspace)}
    return snapshot_path, expected_keys, workspace_by_key


def _launch_slurm_runs(
    workspace: Path,
    owner_dir: Path,
    runs: list[dict[str, Any]],
    execution: dict[str, Any],
    *,
    dry_run: bool,
    runtime_output_fields: tuple[str, ...],
    runtime_output_root: str | Path | None,
    projection_writer: Callable[[LaunchResult], None] | None,
    hooks: SchedulerHooks,
) -> LaunchResult:
    snapshot_path, expected_keys, workspace_by_key = _managed_launch_preflight(workspace, owner_dir, runs)
    missing = expected_keys - set(workspace_by_key)
    if missing:
        step_id, run_id = sorted(missing)[0]
        raise ValueError(f"Canonical run is missing: {step_id} / {run_id}")
    if not dry_run:
        for key in expected_keys:
            if workspace_by_key[key].get("scheduler_raw_state") == SUBMISSION_CLUSTER_MISMATCH:
                raise ValueError(f"Slurm plan is blocked by {SUBMISSION_CLUSTER_MISMATCH}: {key[0]} / {key[1]}")
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
        previous = workspace_by_key[managed_run_key(run)]
        hooks.validate_run_update(
            previous,
            {field: run[field] for field in planned_fields if field in run},
            allow_execution_identity_fill=True,
        )

    status_changes: dict[RunKey, tuple[Any, Any]] = {}
    if not dry_run:
        observed_rows = []
        previous_statuses = {}
        for run in runs:
            key = managed_run_key(run)
            previous = workspace_by_key[key]
            previous_statuses[key] = previous.get("status")
            observed = (
                observe_slurm_run(owner_dir, execution, previous)
                if previous.get("target") not in (None, "")
                else previous
            )
            observed_rows.append(observed)
        committed = hooks.merge_manifest(workspace, observed_rows, lock_held=True)
        workspace_by_key = {managed_run_key(row): row for row in committed}
        status_changes = {
            key: (before, workspace_by_key[key].get("status"))
            for key, before in previous_statuses.items()
            if workspace_by_key[key].get("status") != before
        }

    preview_rows = []
    for run in runs:
        key = managed_run_key(run)
        previous = workspace_by_key[key]
        identity = _slurm_execution_identity(execution, run)
        if dry_run and previous.get("target") in (None, ""):
            preview_rows.append({**previous, **identity})
        else:
            preview_rows.append(previous)

    launchable = [run for run in runs if workspace_by_key[managed_run_key(run)].get("status") in LAUNCHABLE_STATUSES]
    execution_snapshot_sha256 = ""
    if launchable and not dry_run:
        remote = str(execution["host"]) if execution.get("target", "local") == "ssh" else None
        runtime_roots = [
            Path(str(run[field]))
            for run in launchable
            for field in runtime_output_fields
            if run.get(field) not in (None, "")
        ]
        runtime_root = (
            Path(runtime_output_root)
            if runtime_output_root is not None
            else Path(str(execution.get("workdir") or REPO_ROOT))
        )
        exp_io.validate_managed_output_paths(runtime_root, runtime_roots, remote=remote)
        frozen_paths = [
            Path(str(run[field]))
            for run in launchable
            for field in ("scheduler_script", "scheduler_result_path", "allocation_identity_path", "log_path")
        ]
        exp_io.validate_managed_output_paths(owner_dir, frozen_paths, remote=remote)
        for run in launchable:
            script_text = exp_io.read_text_at(run["scheduler_script"], remote=remote)
            if hashlib.sha256(script_text.encode()).hexdigest() != run["scheduler_script_sha256"]:
                raise ValueError(f"Frozen Slurm script changed before submission: {run['scheduler_script']}")
        validated_snapshot = hooks.validated_snapshot or validated_execution_snapshot
        execution_snapshot, write_execution_snapshot = validated_snapshot(
            owner_dir,
            execution,
            runs,
            workspace_by_key,
        )
        if write_execution_snapshot:
            write_execution_snapshot_file(snapshot_path, execution_snapshot)
        execution_snapshot_sha256 = file_sha256(snapshot_path)
        snapshot_rows = [
            {
                **workspace_by_key[managed_run_key(run)],
                "execution_snapshot_sha256": execution_snapshot_sha256,
            }
            for run in runs
        ]
        committed = hooks.merge_manifest(workspace, snapshot_rows, lock_held=True)
        workspace_by_key = {managed_run_key(row): row for row in committed}

    started_keys: set[RunKey] = set()
    uncertain_error: RuntimeError | None = None
    if not dry_run:
        for run in launchable:
            key = managed_run_key(run)
            previous = workspace_by_key[key]
            cluster = slurm.controller_cluster(execution, timeout=LAUNCH_TIMEOUT_SECONDS)
            submitting = {
                **previous,
                **_slurm_execution_identity(execution, run, execution_snapshot_sha256),
                "status": "submitting",
                "scheduler_cluster": cluster,
                "scheduler_observed_at": utc_now(),
            }
            committed = hooks.merge_manifest(workspace, [submitting], lock_held=True)
            workspace_by_key = {managed_run_key(row): row for row in committed}
            submitting = workspace_by_key[key]
            if submitting.get("status") != "submitting":
                continue
            try:
                identity = slurm.submit(
                    execution,
                    str(run["scheduler_script"]),
                    str(run["scheduler_submit_token"]),
                    execution_snapshot_sha256=execution_snapshot_sha256,
                    timeout=LAUNCH_TIMEOUT_SECONDS,
                )
                submitted = _submitted_slurm_row(submitting, identity, raw_state="SUBMITTED")
                if identity.cluster and identity.cluster != submitting["scheduler_cluster"]:
                    reason = (
                        f"{SUBMISSION_CLUSTER_MISMATCH}: submitted job {identity.job_id} returned cluster "
                        f"{identity.cluster!r}, differing from frozen controller {submitting['scheduler_cluster']!r}."
                    )
                    submitted.update(
                        status="unknown_scheduler",
                        scheduler_raw_state=SUBMISSION_CLUSTER_MISMATCH,
                        scheduler_reason=reason,
                    )
                    uncertain_error = RuntimeError(reason)
            except slurm.SlurmCommandError as exc:
                if exc.returncode != 255:
                    failed = {
                        **submitting,
                        "status": "launch_failed",
                        "scheduler_reason": str(exc),
                        "scheduler_observed_at": utc_now(),
                    }
                    committed = hooks.merge_manifest(workspace, [failed], lock_held=True)
                    workspace_by_key = {managed_run_key(row): row for row in committed}
                    continue
                submitted, uncertain_error = _reconcile_slurm_submission(owner_dir, execution, submitting, exc)
            except (subprocess.TimeoutExpired, ValueError) as exc:
                submitted, uncertain_error = _reconcile_slurm_submission(owner_dir, execution, submitting, exc)
            committed = hooks.merge_manifest(workspace, [submitted], lock_held=True)
            workspace_by_key = {managed_run_key(row): row for row in committed}
            if workspace_by_key[key].get("scheduler_job_id") not in (None, ""):
                started_keys.add(key)
            if uncertain_error is not None:
                break

    committed_rows = [workspace_by_key[managed_run_key(run)] for run in runs]
    launch_rows = preview_rows if dry_run else committed_rows
    result = LaunchResult(
        committed_rows=committed_rows,
        launch_rows=launch_rows,
        started_keys=frozenset(started_keys),
        status_changes=status_changes,
        external_status_changes={},
    )
    if projection_writer is not None:
        projection_writer(result)
    for key, (before, after) in status_changes.items():
        hooks.append_event(
            workspace,
            "run_status_changed",
            {"step_id": key[0], "run_id": key[1], "from": before, "to": after},
        )
    for key in started_keys:
        row = workspace_by_key[key]
        hooks.append_event(
            workspace,
            "run_launched",
            {
                "step_id": key[0],
                "run_id": key[1],
                "scheduler_job_id": row["scheduler_job_id"],
            },
        )
    hooks.write_status_report(workspace)
    if uncertain_error is not None:
        raise uncertain_error
    return result


def _slurm_execution_identity(
    execution: dict[str, Any], run: dict[str, Any], execution_snapshot_sha256: str | None = None
) -> dict[str, Any]:
    target = str(execution.get("target", "local") or "local")
    inner = slurm.submission_command(
        str(run["scheduler_script"]),
        str(run["scheduler_submit_token"]),
        execution_snapshot_sha256,
    )
    command = f"ssh {transport.sh(execution['host'])} {transport.sh(inner)}" if target == "ssh" else inner
    identity = {
        "target": target,
        "host": execution.get("host", ""),
        "workdir": execution.get("workdir") or str(REPO_ROOT),
        "gpus": "",
        "pid_path": "",
        "log_path": run["log_path"],
        "command": command,
        **{field: "" for field in PROCESS_IDENTITY_FIELDS},
    }
    if execution.get("runtime_commit") not in (None, ""):
        identity["planned_runtime_commit"] = str(execution["runtime_commit"])
    if execution_snapshot_sha256:
        identity["execution_snapshot_sha256"] = execution_snapshot_sha256
    return identity


def _submitted_slurm_row(
    row: dict[str, Any],
    identity: slurm.JobIdentity,
    *,
    raw_state: str,
    status: str = "queued",
) -> dict[str, Any]:
    return {
        **row,
        "status": status,
        "scheduler_job_id": identity.job_id,
        "scheduler_cluster": row.get("scheduler_cluster") or identity.cluster,
        "scheduler_raw_state": raw_state,
        "scheduler_reason": "",
        "scheduler_observed_at": utc_now(),
        "launched_at": row.get("launched_at") or utc_now(),
    }


def _reconcile_slurm_submission(
    owner_dir: Path,
    execution: dict[str, Any],
    row: dict[str, Any],
    cause: BaseException,
) -> tuple[dict[str, Any], RuntimeError | None]:
    terminal = _read_slurm_json(owner_dir, execution, row["scheduler_result_path"])
    if terminal:
        observed = observe_slurm_run(owner_dir, execution, row)
        if observed.get("scheduler_job_id"):
            return observed, None
        detail = f"{cause}; {observed['scheduler_reason']}"
        observed["scheduler_reason"] = detail
        return observed, RuntimeError(f"Slurm submission outcome is uncertain: {detail}")
    try:
        matches = slurm.active_jobs(
            execution, submit_token=row["scheduler_submit_token"], cluster=row.get("scheduler_cluster") or None
        )
    except Exception as reconcile_error:
        detail = f"{cause}; reconciliation failed: {reconcile_error}"
        unresolved = {**row, "scheduler_reason": detail, "scheduler_observed_at": utc_now()}
        return unresolved, RuntimeError(f"Slurm submission outcome is uncertain: {detail}")
    if len(matches) == 1:
        observed = matches[0]
        category = slurm.state_category(observed.state)
        status = category if category in {"queued", "running"} else "unknown_scheduler"
        reason = observed.reason
        if slurm.normalize_state(observed.state) == "REVOKED":
            reason = "Slurm reports REVOKED federation sibling state; sibling-cluster rebinding is unsupported."
            if observed.reason:
                reason = f"{reason} Scheduler reason: {observed.reason}"
        return (
            {
                **_submitted_slurm_row(
                    row,
                    slurm.JobIdentity(observed.job_id, str(row.get("scheduler_cluster") or "")),
                    raw_state=observed.state,
                    status=status,
                ),
                "scheduler_reason": reason,
                "scheduler_node": observed.node_list,
            },
            None,
        )
    detail = f"{cause}; reconciliation found {len(matches)} jobs for token {row['scheduler_submit_token']}"
    unresolved = {**row, "scheduler_reason": detail, "scheduler_observed_at": utc_now()}
    return unresolved, RuntimeError(f"Slurm submission outcome is uncertain: {detail}")


def stop_slurm_run_locked(
    workspace: Path,
    workspace_rows: list[dict[str, Any]],
    key: RunKey,
    *,
    reason: str,
    hooks: SchedulerHooks,
    now: Callable[[], str],
) -> tuple[list[dict[str, Any]], bool]:
    # The caller holds managed_run_lock and supplies its freshly read, validated canonical rows.
    previous = next(row for row in workspace_rows if managed_run_key(row) == key)
    run_id = key[1]
    if previous.get("scheduler_raw_state") == SUBMISSION_CLUSTER_MISMATCH:
        raise ValueError(f"Slurm stop is blocked by {SUBMISSION_CLUSTER_MISMATCH}: {run_id}")
    if previous.get("status") in TERMINAL_STATUSES:
        raise ValueError(f"Run is already terminal and cannot be stopped: {run_id} ({previous['status']})")
    metadata_stop = previous.get("status") in LAUNCHABLE_STATUSES and not has_managed_launch_evidence(previous)
    if metadata_stop:
        final = merge_run_row(
            previous,
            {
                "step_id": key[0],
                "run_id": key[1],
                "status": "stopped",
                "stopped_at": now(),
                "stop_reason": reason,
            },
        )
        return hooks.merge_manifest(workspace, [final], lock_held=True), True

    pending_stop = previous.get("stop_requested_at") not in (None, "")
    if pending_stop and previous.get("stop_reason") != reason:
        raise ValueError(f"Run already has a pending stop request with a different reason: {run_id}")
    target = previous.get("target")
    host = previous.get("host")
    if target not in {"local", "ssh"}:
        raise ValueError(f"Canonical run target must be local or ssh for run_id: {run_id}")
    if target == "ssh" and (not isinstance(host, str) or not host.strip()):
        raise ValueError(f"Canonical SSH run requires a non-empty host for run_id: {run_id}")
    execution = {"target": target, "host": host}
    if scheduler_direct_controller(previous):
        execution["scheduler"] = {"direct_controller": True}
    job_id = str(previous.get("scheduler_job_id") or "")
    cluster = str(previous.get("scheduler_cluster") or "")
    if pending_stop and not job_id:
        raise ValueError(f"Pending Slurm stop request is missing scheduler job identity: {run_id}")
    if not job_id:
        matches = slurm.active_jobs(
            execution,
            submit_token=str(previous["scheduler_submit_token"]),
            cluster=cluster or None,
        )
        if len(matches) != 1:
            raise ValueError(f"Cannot resolve one Slurm job for run_id {run_id}; found {len(matches)} matching jobs.")
        job_id = matches[0].job_id
    if pending_stop:
        committed = workspace_rows
    else:
        final = merge_run_row(
            previous,
            {
                "step_id": key[0],
                "run_id": key[1],
                "scheduler_job_id": job_id,
                "scheduler_cluster": cluster,
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
    slurm.cancel(execution, job_id, cluster=cluster or None)
    return committed, False


class SlurmMonitorContext:
    def __init__(self, rows: Iterable[dict[str, Any]], *, owner_dir: str | Path, remote: str | None = None):
        self.owner_dir = Path(owner_dir)
        groups: dict[tuple[str, str, str, bool], set[str]] = {}
        file_groups: dict[tuple[str, str], list[tuple[str, str]]] = {}
        for row in rows:
            if remote and not row.get("host"):
                continue
            route = self._route(row)
            if route is not None:
                groups.setdefault(route, set()).add(str(row["scheduler_job_id"]))
            file_route = self._file_route(row)
            paths = (str(row.get("scheduler_result_path") or ""), str(row.get("allocation_identity_path") or ""))
            if file_route is not None and all(paths):
                file_groups.setdefault(file_route, []).append(paths)
        self.groups = {route: tuple(sorted(job_ids)) for route, job_ids in groups.items() if len(job_ids) > 1}
        self.snapshots: dict[tuple[str, str, str, bool], dict[str, slurm.JobObservation] | None] = {}
        self.file_groups = {route: paths for route, paths in file_groups.items() if len(paths) > 1}
        self.file_snapshots: dict[tuple[str, str, str], dict[str, str | None] | None] = {}

    @staticmethod
    def _file_route(row: dict[str, Any]) -> tuple[str, str] | None:
        target = row.get("target")
        host = str(row.get("host") or "").strip() if target == "ssh" else ""
        if (
            row.get("scheduler_type") != "slurm"
            or row.get("scheduler_raw_state") == SUBMISSION_CLUSTER_MISMATCH
            or target not in {"local", "ssh"}
            or (target == "ssh" and not host)
            or not row.get("scheduler_submit_token")
        ):
            return None
        return target, host

    def sidecar(self, owner_dir: Path, execution: dict[str, Any], row: dict[str, Any], field: str) -> dict[str, Any]:
        path = str(row[field])
        route = self._file_route(row)
        group = self.file_groups.get(route, ())
        row_paths = (str(row.get("scheduler_result_path") or ""), str(row.get("allocation_identity_path") or ""))
        target = execution.get("target", "local")
        host = str(execution.get("host") or "").strip() if target == "ssh" else ""
        if owner_dir != self.owner_dir or route != (target, host) or row_paths not in group:
            return _read_slurm_json(owner_dir, execution, path)
        key = (*route, field)
        if key not in self.file_snapshots:
            if field == "scheduler_result_path":
                paths = [terminal for terminal, _allocation in group]
            else:
                terminals = self.file_snapshots.get((*route, "scheduler_result_path"))
                if terminals is None:
                    return _read_slurm_json(owner_dir, execution, path)
                paths = []
                for terminal, allocation in group:
                    try:
                        payload = _parse_slurm_json(terminals[terminal], terminal)
                    except ValueError:
                        # A future row's bad terminal must neither fail this row nor prefetch its allocation.
                        continue
                    if not payload:
                        paths.append(allocation)
            try:
                self.file_snapshots[key] = exp_io.read_managed_output_texts_at(
                    self.owner_dir, list(dict.fromkeys(paths)), remote=host if target == "ssh" else None
                )
            except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired):
                # Discard the whole batch; exact reads retain per-run errors and the phase is not retried.
                self.file_snapshots[key] = None
        snapshot = self.file_snapshots[key]
        if snapshot is None:
            return _read_slurm_json(owner_dir, execution, path)
        # Missing text is a snapshot only for this round; the next context rechecks newly published sidecars.
        return _parse_slurm_json(snapshot[path], path)

    @staticmethod
    def _route(row: dict[str, Any]) -> tuple[str, str, str, bool] | None:
        target = row.get("target")
        host = str(row.get("host") or "").strip() if target == "ssh" else ""
        topology = row.get("scheduler_direct_controller")
        if (
            row.get("scheduler_type") != "slurm"
            or row.get("scheduler_raw_state") == SUBMISSION_CLUSTER_MISMATCH
            or target not in {"local", "ssh"}
            or (target == "ssh" and not host)
            or topology not in {"true", "false"}
            or not all(row.get(field) for field in ("scheduler_job_id", "scheduler_cluster", "scheduler_submit_token"))
        ):
            return None
        return target, host, str(row["scheduler_cluster"]), topology == "true"

    def active_job(self, execution: dict[str, Any], row: dict[str, Any]) -> slurm.JobObservation | None:
        route = self._route(row)
        job_ids = self.groups.get(route, ())
        job_id = str(row.get("scheduler_job_id") or "")
        if job_id not in job_ids:
            return None
        if route not in self.snapshots:
            try:
                matches = slurm.active_jobs(execution, job_id=job_ids, cluster=route[2])
                snapshot = {item.job_id: item for item in matches}
                if len(snapshot) != len(matches):
                    raise ValueError("A Slurm monitor batch returned duplicate job ids.")
            except (slurm.SlurmCommandError, subprocess.TimeoutExpired, RuntimeError, ValueError):
                # A failed batch is not negative scheduler evidence; exact queries retain their original errors.
                self.snapshots[route] = None
            else:
                self.snapshots[route] = snapshot
        snapshot = self.snapshots[route]
        return snapshot.get(job_id) if snapshot is not None else None


def observe_slurm_run(
    owner_dir: str | Path,
    execution: dict[str, Any],
    row: dict[str, Any],
    *,
    health: bool = False,
    monitor_context: SlurmMonitorContext | None = None,
) -> dict[str, Any]:
    if row.get("scheduler_raw_state") == SUBMISSION_CLUSTER_MISMATCH:
        return _slurm_artifact_observation({**row, "scheduler_observed_at": utc_now()}, health=health)
    owner = Path(owner_dir)
    token = str(row["scheduler_submit_token"])
    canonical_job_id = str(row.get("scheduler_job_id") or "")
    job_id = canonical_job_id
    canonical_cluster = str(row.get("scheduler_cluster") or "")
    cluster = canonical_cluster
    execution_target = str(execution.get("target", "local") or "local")
    execution_host = str(execution.get("host") or "").strip()
    execution_scheduler = execution.get("scheduler") if isinstance(execution.get("scheduler"), dict) else {}
    canonical_direct_controller = str(row.get("scheduler_direct_controller") or "") == "true"
    routing_identity_matches = (
        execution_target == row.get("target")
        and (execution_target != "ssh" or execution_host == str(row.get("host") or "").strip())
        and (execution_scheduler.get("direct_controller") is True) == canonical_direct_controller
    )
    stop_requested = row.get("stop_requested_at") not in (None, "")
    terminal = (
        monitor_context.sidecar(owner, execution, row, "scheduler_result_path")
        if monitor_context is not None
        else _read_slurm_json(owner, execution, row["scheduler_result_path"])
    )
    observation: dict[str, Any] = {**row, "scheduler_observed_at": utc_now()}
    terminal_exit_code: int | None = None
    terminal_identity: slurm.JobIdentity | None = None
    if terminal:
        identity = slurm.sidecar_identity(terminal, token, expected_job_id=job_id or None)
        if cluster and identity.cluster and identity.cluster != cluster:
            raise ValueError("Slurm terminal sidecar cluster differs from the canonical run.")
        terminal_identity = identity
        job_id = identity.job_id
        terminal_exit_code = slurm.terminal_exit_code(terminal)
        observation.update(
            {
                "scheduler_node": terminal.get("node", ""),
                "scheduler_exit_code": terminal_exit_code,
                "scheduler_started_at": terminal.get("started_at", ""),
            }
        )
        runtime_commit = _slurm_sidecar_runtime_commit(terminal)
        if runtime_commit:
            observation["runtime_commit"] = runtime_commit
    else:
        allocation = (
            monitor_context.sidecar(owner, execution, row, "allocation_identity_path")
            if monitor_context is not None
            else _read_slurm_json(owner, execution, row["allocation_identity_path"])
        )
        if allocation:
            allocation_identity = slurm.sidecar_identity(allocation, token, expected_job_id=job_id or None)
            if cluster and allocation_identity.cluster and allocation_identity.cluster != cluster:
                raise ValueError("Slurm allocation sidecar cluster differs from the canonical run.")
            if not job_id:
                job_id = allocation_identity.job_id
            observation["scheduler_node"] = allocation.get("node", "")
            observation["scheduler_started_at"] = allocation.get("started_at", "")
            runtime_commit = _slurm_sidecar_runtime_commit(allocation)
            if runtime_commit:
                observation["runtime_commit"] = runtime_commit
    health_error = ""
    try:
        if not job_id:
            matches = slurm.active_jobs(execution, submit_token=token, cluster=cluster or None)
            if len(matches) > 1:
                raise ValueError(f"Multiple Slurm jobs match frozen submit token {token}.")
            if not matches:
                observation.update({"status": "submitting", "scheduler_reason": "submission_unresolved"})
                return _slurm_artifact_observation(observation, health=health)
            active = matches[0]
            job_id = active.job_id
        else:
            active = None
            if monitor_context is not None and monitor_context.owner_dir == owner and routing_identity_matches:
                active = monitor_context.active_job(execution, row)
            if active is None:
                matches = slurm.active_jobs(execution, job_id=job_id, cluster=cluster or None)
                active = matches[0] if matches else None
        controller_observation = None
        if active is None:
            controller_observation = slurm.show_job(execution, job_id, cluster=cluster or None)
            active = controller_observation
        from_accounting = False
        accounting_error = ""
        accounting_disabled = False
        if active is None:
            try:
                active = slurm.accounting_job(
                    execution,
                    job_id,
                    submit_token=token,
                    cluster=cluster or None,
                )
            except slurm.SlurmCommandError as exc:
                accounting_error = str(exc)
                accounting_output = f"{exc.stdout}\n{exc.stderr}".lower()
                accounting_disabled = not (execution_target == "ssh" and exc.returncode == 255) and (
                    "slurm accounting storage is disabled" in accounting_output
                )
            except (subprocess.TimeoutExpired, RuntimeError) as exc:
                accounting_error = str(exc)
            from_accounting = active is not None
        if active is None:
            fallback_identity_is_frozen = (
                bool(canonical_job_id)
                and bool(canonical_cluster)
                and bool(token)
                and str(row.get("scheduler_direct_controller") or "") in {"false", "true"}
                and (row.get("target") == "local" or (row.get("target") == "ssh" and row.get("host") not in (None, "")))
                and routing_identity_matches
                and terminal_identity is not None
                and terminal_identity.cluster == canonical_cluster
            )
            if terminal and accounting_disabled and fallback_identity_is_frozen:
                observation.update(
                    {
                        "status": "completed" if terminal_exit_code == 0 else "failed",
                        "scheduler_raw_state": "MISSING",
                        "scheduler_reason": (
                            "Slurm controller no longer retains the bound job and accounting storage is disabled; "
                            "terminal status recovered from the authenticated terminal sidecar."
                        ),
                    }
                )
                return _slurm_artifact_observation(observation, health=health)
            if terminal:
                reason = "Slurm job disappeared before terminal scheduler state was observed."
            else:
                reason = "Slurm job disappeared without a terminal sidecar."
            if terminal_exit_code not in (None, 0):
                reason = f"{reason} Authenticated terminal sidecar reports non-zero exit code {terminal_exit_code}."
            if accounting_error:
                reason = f"{reason} {accounting_error}"
            observation.update(
                {
                    "status": "unknown_scheduler" if canonical_job_id else "submitting",
                    "scheduler_raw_state": "MISSING",
                    "scheduler_reason": reason,
                }
            )
            return _slurm_artifact_observation(
                observation,
                health=health,
                health_error=accounting_error,
            )
        # Reuse successful details, but keep the fresh controller retry after accounting.
        if health and controller_observation is None:
            try:
                detailed = slurm.show_job(execution, job_id, cluster=cluster or None)
                if detailed is None:
                    health_error = "Slurm job details are unavailable."
                else:
                    active = detailed
                    from_accounting = False
            except (slurm.SlurmCommandError, subprocess.TimeoutExpired, RuntimeError, ValueError) as exc:
                health_error = str(exc)
        if not from_accounting and active.comment != token:
            raise ValueError("Observed Slurm job comment differs from the frozen submit token.")
        # Sidecars supply lookup candidates, never first-bind identity without scheduler evidence on the frozen route.
        if not canonical_job_id:
            if not routing_identity_matches:
                raise ValueError("Slurm query route differs from the canonical run.")
            observation["scheduler_job_id"] = job_id
            observation["launched_at"] = row.get("launched_at") or utc_now()
        category = slurm.state_category(active.state)
        reason = active.reason
        if slurm.normalize_state(active.state) == "REVOKED":
            reason = "Slurm reports REVOKED federation sibling state; sibling-cluster rebinding is unsupported."
            if active.reason:
                reason = f"{reason} Scheduler reason: {active.reason}"
        if terminal_exit_code is not None:
            if category == "cancelled" and stop_requested:
                status = "stopped"
            elif category == "completed":
                status = "completed" if terminal_exit_code == 0 else "failed"
            elif category in {"failed", "cancelled"}:
                status = "failed"
            elif category in {"queued", "running"}:
                status = "stopping" if stop_requested else category
            else:
                status = "unknown_scheduler"
                reason = reason or "Slurm terminal sidecar is present but scheduler state is not recognized."
        else:
            if category == "cancelled" and stop_requested:
                status = "stopped"
            else:
                status = (
                    "stopping"
                    if stop_requested and category in {"queued", "running"}
                    else category if category in {"queued", "running"} else "unknown_scheduler"
                )
            if category in {"completed", "failed", "cancelled"} and status != "stopped":
                reason = reason or "Terminal scheduler state is missing the matching terminal sidecar."
        observation.update(
            {
                "scheduler_raw_state": active.state,
                "scheduler_reason": reason,
                "scheduler_node": active.node_list or observation.get("scheduler_node", ""),
                "status": status,
                **{f"scheduler_{field}": value for field, value in active.details.items()},
            }
        )
        if status == "stopped":
            observation["stopped_at"] = row.get("stopped_at") or observation["scheduler_observed_at"]
    except (slurm.SlurmCommandError, subprocess.TimeoutExpired, RuntimeError) as exc:
        observation.update(
            {
                "status": "unknown_scheduler" if canonical_job_id else "submitting",
                "scheduler_reason": str(exc),
            }
        )
        health_error = str(exc)
    return _slurm_artifact_observation(observation, health=health, health_error=health_error)


def _slurm_sidecar_runtime_commit(payload: dict[str, Any]) -> str:
    snapshot = payload.get("execution_snapshot")
    value = snapshot.get("runtime_commit") if isinstance(snapshot, dict) else payload.get("runtime_commit")
    if value in (None, ""):
        return ""
    if not is_full_git_object_id(value):
        raise ValueError("Slurm sidecar runtime_commit must be a full lowercase 40- or 64-character Git object ID.")
    return value


def _read_slurm_json(owner_dir: Path, execution: dict[str, Any], path: str | Path) -> dict[str, Any]:
    remote = str(execution["host"]) if execution.get("target", "local") == "ssh" else None
    text = exp_io.read_managed_output_texts_at(owner_dir, [path], remote=remote)[str(path)]
    return _parse_slurm_json(text, path)


def _parse_slurm_json(text: str | None, path: str | Path) -> dict[str, Any]:
    if not text:
        return {}
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Slurm sidecar is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Slurm sidecar must be a mapping: {path}")
    return payload


def _slurm_artifact_observation(row: dict[str, Any], *, health: bool = False, health_error: str = "") -> dict[str, Any]:
    observed_artifacts = evidence.runtime_artifacts(row)
    if observed_artifacts is not None:
        run_manifest, _manifest, checkpoints = observed_artifacts
        row.update({"run_manifest": run_manifest, "checkpoints": ";".join(checkpoints)})
    if health:
        row["log_tail"], log_age = evidence.log_tail_and_age(row.get("log_path"), row)
    else:
        row["log_tail"] = evidence.log_tail(row.get("log_path"), row)
    if health:
        status = str(row.get("status") or "")
        if health_error or status in {"submitting", "unknown_scheduler"}:
            health_status = "health_unknown"
        elif status == "queued":
            health_status = "scheduler_queued"
        elif status == "running":
            health_status = "scheduler_running"
        else:
            health_status = status
        queue_age = _timestamp_age_seconds(row.get("launched_at")) if status == "queued" else None
        allocation_age = _timestamp_age_seconds(row.get("scheduler_started_at"))
        row.update(
            {
                "health_status": health_status,
                "scheduler_health_error": health_error,
                "scheduler_queue_age_seconds": "" if queue_age is None else queue_age,
                "scheduler_allocation_age_seconds": "" if allocation_age is None else allocation_age,
                "log_age_seconds": "" if log_age is None else log_age,
            }
        )
    row["monitored_at"] = utc_now()
    return row


def _timestamp_age_seconds(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return max(int((datetime.now(timezone.utc) - parsed.astimezone(timezone.utc)).total_seconds()), 0)


def validated_execution_snapshot(
    owner_dir: str | Path,
    execution: dict[str, Any],
    runs: list[dict[str, Any]],
    workspace_by_key: dict[RunKey, dict[str, Any]],
    *,
    inspector: Callable[[dict[str, Any], list[dict[str, Any]]], dict[str, Any]] | None = None,
    plan_label: str = "managed",
) -> tuple[dict[str, Any], bool]:
    root = Path(owner_dir)
    snapshot_path = root / EXECUTION_SNAPSHOT_NAME
    inspect = inspector or inspect_execution_target
    if snapshot_path.exists():
        frozen = read_json(snapshot_path)
        if not isinstance(frozen, dict):
            raise ValueError(f"Execution snapshot must be a mapping: {snapshot_path}")
        actual = inspect(execution, runs)
        # A rolling checkout may advance after registration; its live commit is recorded per run at launch.
        rolling_evidence_fields = {"runtime_commit", "supported_options", "cli_options_sha256"}
        changed = sorted(
            key
            for key in set(frozen) | set(actual)
            if key not in rolling_evidence_fields and frozen.get(key) != actual.get(key)
        )
        if changed:
            raise ValueError(f"Frozen execution snapshot changed: {', '.join(changed)}")
        return actual, False
    for run in runs:
        row = workspace_by_key[managed_run_key(run)]
        if row.get("target") not in (None, "") or row.get("status") not in LAUNCHABLE_STATUSES:
            raise ValueError(
                f"Cannot establish an execution snapshot after a {plan_label} run has started; create a new plan."
            )
    return inspect(execution, runs), True


def write_execution_snapshot_file(path: str | Path, snapshot: dict[str, Any]) -> None:
    snapshot_path = Path(path)
    payload = (json.dumps(snapshot, indent=2, sort_keys=True) + "\n").encode()
    descriptor, temporary = tempfile.mkstemp(prefix=f".{snapshot_path.name}.", dir=snapshot_path.parent)
    try:
        with os.fdopen(descriptor, "wb") as file_obj:
            file_obj.write(payload)
            file_obj.flush()
            os.fsync(file_obj.fileno())
        os.replace(temporary, snapshot_path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def inspect_execution_target(
    execution: dict[str, Any],
    runs: list[dict[str, Any]],
    *,
    command_runner: Callable[[dict[str, Any], list[str]], subprocess.CompletedProcess] | None = None,
    plan_label: str = "managed",
) -> dict[str, Any]:
    modules: set[str] = set()
    python_commands: set[str] = set()
    planned_argv: list[dict[str, Any]] = []
    required_options: set[str] = set()
    for run in runs:
        command = str(run.get("command") or "")
        if command not in Path(str(run["script"])).read_text().splitlines():
            raise ValueError(f"Frozen {plan_label} command differs from its launch script: {run['run_id']}")
        tokens = shlex.split(command)
        try:
            module_flag_index = tokens.index("-m")
            module_index = module_flag_index + 1
            modules.add(tokens[module_index])
        except (IndexError, ValueError) as exc:
            raise ValueError(f"Frozen {plan_label} command has no Python module: {run['run_id']}") from exc
        if module_flag_index != 1:
            raise ValueError(f"Frozen {plan_label} command has an unsupported Python invocation: {run['run_id']}")
        python_commands.add(tokens[0])
        planned_argv.append({"run_id": str(run["run_id"]), "args": tokens[module_index + 1 :]})
        required_options.update(token for token in tokens[module_index + 1 :] if token.startswith("--"))
    if len(modules) != 1:
        raise ValueError(f"A {plan_label} plan must use exactly one target runtime module.")
    if len(python_commands) != 1:
        raise ValueError(f"A {plan_label} plan must use exactly one target Python executable.")
    module = next(iter(modules))
    python_command = next(iter(python_commands))
    backend = _managed_scheduler_type(execution, runs)
    expected_python = execution.get("python")
    planned_commit = execution.get("runtime_commit")
    if expected_python in (None, "") or planned_commit in (None, ""):
        raise ValueError(
            f"Frozen {plan_label} plan lacks execution.python or execution.runtime_commit; create a new plan."
        )
    if python_command != str(expected_python):
        raise ValueError(f"Frozen {plan_label} commands differ from execution.python.")
    run_command = command_runner or run_execution_command
    identity_result = run_command(
        execution,
        [
            python_command,
            "-c",
            python_programs.source("managed_scheduler.runtime_identity"),
            module,
            "{}",
            "[]",
            json.dumps(SLURM_LAUNCH_CAPABILITIES if backend == "slurm" else DIRECT_LAUNCH_CAPABILITIES),
        ],
    )
    if identity_result.returncode != 0:
        detail = (
            identity_result.stderr.strip()
            or identity_result.stdout.strip()
            or f"exit code {identity_result.returncode}"
        )
        raise RuntimeError(f"Target execution identity preflight failed: {detail}")
    try:
        identity = json.loads(identity_result.stdout)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError("Target execution identity preflight returned malformed JSON.") from exc
    identity_fields = (
        "python",
        "python_version",
        "runtime_commit",
        "runtime_repo_root",
        "runtime_hostname",
        "module",
        "module_origin",
    )
    if not isinstance(identity, dict) or any(identity.get(field) in (None, "") for field in identity_fields):
        raise ValueError("Target execution identity preflight returned incomplete evidence.")
    if not is_full_git_object_id(identity["runtime_commit"]):
        raise ValueError("Target execution identity preflight returned an invalid runtime commit.")
    parse_result = run_command(
        execution,
        [
            python_command,
            "-c",
            python_programs.source("managed_scheduler.cli_preflight"),
            module,
            json.dumps(planned_argv),
            identity["module_origin"],
        ],
    )
    if parse_result.returncode != 0:
        detail = parse_result.stderr.strip() or parse_result.stdout.strip() or f"exit code {parse_result.returncode}"
        raise ValueError(f"Target runtime rejected frozen arguments: {detail}")
    marker = "AGENT_CLI_PREFLIGHT="
    evidence_lines = [line.removeprefix(marker) for line in parse_result.stdout.splitlines() if line.startswith(marker)]
    if len(evidence_lines) != 1:
        raise ValueError("Target runtime CLI preflight returned malformed evidence.")
    try:
        cli_evidence = json.loads(evidence_lines[0])
    except json.JSONDecodeError as exc:
        raise ValueError("Target runtime CLI preflight returned malformed evidence.") from exc
    supported_options = set(cli_evidence.get("supported_options") or []) if isinstance(cli_evidence, dict) else set()
    missing_options = sorted(required_options - supported_options)
    if missing_options:
        raise ValueError(f"Target runtime CLI {module} does not accept planned options: {', '.join(missing_options)}")
    cli_options_sha256 = cli_evidence.get("cli_options_sha256") if isinstance(cli_evidence, dict) else None
    if not isinstance(cli_options_sha256, str) or not cli_options_sha256:
        raise ValueError("Target runtime CLI preflight returned malformed evidence.")
    execution_env = execution.get("env") if isinstance(execution.get("env"), dict) else {}
    return {
        "target": str(execution.get("target", "local") or "local"),
        "host": str(execution.get("host") or ""),
        "workdir": str(execution.get("workdir") or REPO_ROOT),
        "conda_env": str(execution.get("conda_env") or ""),
        "python_command": python_command,
        "expected_runtime_commit": str(planned_commit),
        "execution_env_sha256": hashlib.sha256(
            json.dumps(execution_env, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        **identity,
        "module": module,
        "required_options": sorted(required_options),
        "supported_options": sorted(supported_options),
        "cli_options_sha256": cli_options_sha256,
        "validated_argv_sha256": hashlib.sha256(
            json.dumps(planned_argv, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }


def run_execution_command(execution: dict[str, Any], command: list[str]) -> subprocess.CompletedProcess:
    workdir = str(execution.get("workdir") or REPO_ROOT)
    inner = f"export PYTHONPATH={_sh(workdir)} && " + " ".join(_sh(part) for part in command)
    run = ["bash", "-c", inner]
    if execution.get("conda_env"):
        run = ["conda", "run", "--no-capture-output", "-n", str(execution["conda_env"]), *run]
    run_command = " ".join(_sh(part) for part in run)
    env = dict(execution.get("env") or {})
    if env:
        env_prefix = " ".join(f"{key}={_sh(value)}" for key, value in sorted(env.items()))
        run_command = f"env {env_prefix} {run_command}"
    run_command = f"cd {_sh(workdir)} && {run_command}"
    host = str(execution["host"]) if execution.get("target", "local") == "ssh" else None
    return transport.run_shell(host, run_command, timeout=LAUNCH_TIMEOUT_SECONDS)


def build_launch_command(
    execution: dict[str, Any],
    script: Path,
    log_path: str | Path,
    pid_path: str | Path,
    gpus: list[Any],
    *,
    execution_snapshot: dict[str, Any] | None = None,
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
        artifacts = [
            {"path": str(script), "sha256": script_sha256},
            {"path": str(config_path), "sha256": config_sha256},
        ]
        if (checkpoint_path is None) != (checkpoint_sha256 is None) or checkpoint_sha256 == "":
            raise ValueError("Verified launch requires both frozen checkpoint path and hash.")
        if checkpoint_path is not None:
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
            planned_argv = [{"run_id": run_id or script.stem, "args": tokens[module_index + 1 :]}]
            verification_commands.extend(
                [
                    (
                        execution["python"],
                        "-c",
                        python_programs.source("managed_scheduler.runtime_identity"),
                        execution_snapshot["module"],
                        json.dumps(execution_snapshot, sort_keys=True),
                        json.dumps(artifacts, sort_keys=True),
                        json.dumps(DIRECT_LAUNCH_CAPABILITIES),
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
            # Script-based plans retain artifact checks without claiming a module execution snapshot.
            verification_commands.append(
                (
                    run[0],
                    "-c",
                    python_programs.source("plan_rendering.verify_input_snapshots"),
                    json.dumps(artifacts, sort_keys=True),
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
    if execution.get("target", "local") == "ssh":
        mkdir = f"mkdir -p {_sh(_parent_path(log_path))} {_sh(_parent_path(pid_path))}"
        inner = f"{mkdir} && cd {_sh(workdir)} && {run_command}"
        return f"ssh {_sh(execution['host'])} {_sh(inner)}"
    return f"cd {_sh(workdir)} && {run_command}"


def start_process(execution: dict[str, Any], command: str) -> str:
    try:
        result = subprocess.run(
            ["bash", "-lc", command],
            text=True,
            capture_output=True,
            timeout=LAUNCH_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        # A detached child may already exist when the transport times out; monitoring must reconcile it.
        return "launched"
    if execution.get("target", "local") == "ssh" and result.returncode == 255:
        # SSH may disconnect after starting the detached child; monitoring must reconcile its identity.
        return "launched"
    return "launched" if result.returncode == 0 else "launch_failed"


def _parent_path(path: str | Path) -> str:
    text = str(path)
    parent = text.rsplit("/", 1)[0] if "/" in text else "."
    return parent or "/"


_sh = transport.sh
