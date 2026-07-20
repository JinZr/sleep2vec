from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
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
import tempfile
from typing import Any

import yaml

from . import experiment_io as exp_io, gpu_rules, run_evidence as evidence, transport
from .experiment_workspace import (
    EXECUTION_IDENTITY_FIELDS,
    PROCESS_IDENTITY_FIELDS,
    append_event,
    managed_run_key,
    merge_run_manifest,
    read_run_manifest,
    validate_frozen_run_update,
    write_status_report,
)
from .manifests import read_json, utc_now
from .models import REPO_ROOT

RunKey = tuple[str, str]
ACTIVE_STATUSES = frozenset({"launched", "running", "unknown_remote", "missing_pid"})
LAUNCHABLE_STATUSES = frozenset({"planned", "pending"})
LAUNCH_TIMEOUT_SECONDS = 60
EXECUTION_SNAPSHOT_NAME = "execution_snapshot.json"


class MissingPidCapacityError(RuntimeError):
    def __init__(self, step_id: str, run_id: str):
        self.step_id = step_id
        self.run_id = run_id
        super().__init__(f"Managed launch capacity is blocked because {step_id} / {run_id} has status missing_pid.")


__all__ = [
    "ACTIVE_STATUSES",
    "CapacityState",
    "EXECUTION_SNAPSHOT_NAME",
    "LAUNCH_TIMEOUT_SECONDS",
    "LaunchResult",
    "MissingPidCapacityError",
    "SchedulerHooks",
    "build_launch_command",
    "capacity_state",
    "gpu_groups",
    "inspect_execution_target",
    "launch_managed_runs",
    "managed_run_lock",
    "observe_run",
    "observe_runs",
    "run_execution_command",
    "script_commits_terminal_status",
    "shares_capacity",
    "start_process",
    "validated_execution_snapshot",
]

_RUNTIME_IDENTITY_SCRIPT = """
import hashlib
import importlib.util
import json
from pathlib import Path
import socket
import subprocess
import sys

module = sys.argv[1]
expected = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
artifacts = json.loads(sys.argv[3]) if len(sys.argv) > 3 else []
commit = subprocess.run(["git", "rev-parse", "HEAD"], text=True, capture_output=True)
repo_root = subprocess.run(["git", "rev-parse", "--show-toplevel"], text=True, capture_output=True)
dirty = subprocess.run(
    ["git", "status", "--porcelain", "--untracked-files=no"],
    text=True,
    capture_output=True,
)
untracked_code = subprocess.run(
    ["git", "ls-files", "--others", "--exclude-standard", "--", "*.py", "*.pyi", "*.so"],
    text=True,
    capture_output=True,
)
ignored_code = subprocess.run(
    ["git", "ls-files", "--others", "--ignored", "--exclude-standard", "--", "*.py", "*.pyi", "*.so"],
    text=True,
    capture_output=True,
)
if (
    commit.returncode != 0
    or repo_root.returncode != 0
    or dirty.returncode != 0
    or untracked_code.returncode != 0
    or ignored_code.returncode != 0
):
    print(
        commit.stderr or repo_root.stderr or dirty.stderr or untracked_code.stderr or ignored_code.stderr,
        file=sys.stderr,
    )
    raise SystemExit(2)
if dirty.stdout.strip():
    print("Target runtime has tracked worktree changes; launch requires a clean commit.", file=sys.stderr)
    raise SystemExit(2)

def importable_code(output):
    paths = []
    for raw in output.splitlines():
        path = Path(raw)
        module_name = path.name.split(".", 1)[0]
        if module_name.isidentifier() and all(part.isidentifier() for part in path.parts[:-1]):
            paths.append(raw)
    return paths

if importable_code(untracked_code.stdout) or importable_code(ignored_code.stdout):
    print(
        "Target runtime has untracked or ignored Python code; launch requires commit-defined import roots.",
        file=sys.stderr,
    )
    raise SystemExit(2)
runtime_repo_root = Path(repo_root.stdout.strip()).resolve()
spec = importlib.util.find_spec(module)
origin = Path(spec.origin).resolve() if spec is not None and spec.origin else None
if origin is None:
    print(f"Target runtime module has no file origin: {module}", file=sys.stderr)
    raise SystemExit(2)
try:
    origin.relative_to(runtime_repo_root)
except ValueError:
    print(f"Target runtime module is outside the verified repository: {origin}", file=sys.stderr)
    raise SystemExit(2)
payload = {
    "python": sys.executable,
    "python_version": sys.version.split()[0],
    "runtime_commit": commit.stdout.strip(),
    "runtime_repo_root": str(runtime_repo_root),
    "runtime_hostname": socket.gethostname(),
    "module": module,
    "module_origin": str(origin),
}
identity_fields = (
    "python",
    "python_version",
    "runtime_commit",
    "runtime_repo_root",
    "runtime_hostname",
    "module",
    "module_origin",
)
changed = [field for field in identity_fields if expected and payload[field] != expected.get(field)]
if changed:
    print("Target runtime identity changed before process start: " + ", ".join(changed), file=sys.stderr)
    raise SystemExit(2)
for artifact in artifacts:
    path = Path(artifact["path"])
    if path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1:
        print(f"Frozen run artifact is not an independent file: {path}", file=sys.stderr)
        raise SystemExit(2)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != artifact["sha256"]:
        print(f"Frozen run artifact changed before process start: {path}", file=sys.stderr)
        raise SystemExit(2)
print(json.dumps(payload, sort_keys=True))
""".strip()

_PROCESS_LAUNCH_SCRIPT = "\n\n".join(
    [
        """
import json
import os
import signal
import subprocess
import sys
""".strip(),
        evidence._PROCESS_START_TOKEN_CODE,
        """

script, log_path, pid_path, workdir = sys.argv[1:]

with open(log_path, "ab", buffering=0) as log_file:
    process = subprocess.Popen(
        ["bash", script],
        cwd=workdir,
        env=os.environ.copy(),
        stdin=subprocess.DEVNULL,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        identity = {
            "pid": process.pid,
            "process_group_id": os.getpgid(process.pid),
            "process_start_token": start_token(process.pid),
        }
        if identity["process_start_token"] is None:
            raise RuntimeError(f"Cannot read process start time for PID {process.pid}")
        descriptor = os.open(pid_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as file_obj:
            json.dump(identity, file_obj, sort_keys=True)
            file_obj.write("\\n")
            file_obj.flush()
            os.fsync(file_obj.fileno())
    except BaseException:
        # An unrecorded process group cannot be managed safely, so stop it before surfacing launch failure.
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        raise
""".strip(),
    ]
)


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
    with lock_path.open("a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


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
    if owner == "monitor":
        return False
    raise ValueError("terminal_status_owner must be 'script' or 'monitor'.")


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
    max_concurrent = int(execution["max_concurrent"]) if "max_concurrent" in execution else max(len(groups), 1)
    if max_concurrent <= 0:
        raise ValueError("execution.max_concurrent must be a positive integer.")
    allow_gpu_oversubscription = bool(groups) and max_concurrent > len(groups)
    target = str(execution.get("target", "local") or "local")
    current_host = str(execution.get("host") or "") if target == "ssh" else ""
    group_values = [{str(item) for item in group} for group in groups]
    current_gpu_pool = set().union(*group_values) if group_values else set()
    other_active_gpu_sets: list[set[str]] = []
    unknown_other_active = 0
    external_missing_pid: list[RunKey] = []
    for key, row in workspace_rows.items():
        if not groups or key in expected_keys or row.get("status") not in ACTIVE_STATUSES:
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
    planned_by_key = {managed_run_key(run): run for run in runs}
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
                    **checkpoint_args,
                )
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
        if frozen != actual:
            changed = sorted(key for key in set(frozen) | set(actual) if frozen.get(key) != actual.get(key))
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
    expected_python = execution.get("python")
    expected_commit = execution.get("runtime_commit")
    if expected_python in (None, "") or expected_commit in (None, ""):
        raise ValueError(
            f"Frozen {plan_label} plan lacks execution.python or execution.runtime_commit; create a new plan."
        )
    if python_command != str(expected_python):
        raise ValueError(f"Frozen {plan_label} commands differ from execution.python.")
    run_command = command_runner or run_execution_command
    identity_result = run_command(execution, [python_command, "-c", _RUNTIME_IDENTITY_SCRIPT, module])
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
    if identity["runtime_commit"] != str(expected_commit):
        raise ValueError(
            "Target runtime commit differs from the frozen plan: "
            f"expected {expected_commit}, observed {identity['runtime_commit']}."
        )

    parser_script = """
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import runpy
import sys

module = sys.argv[1]
planned_argv = json.loads(sys.argv[2])
expected_origin = sys.argv[3]
spec = importlib.util.find_spec(module)
origin = str(Path(spec.origin).resolve()) if spec is not None and spec.origin else ""
if origin != expected_origin:
    print("Target runtime module origin changed before argparse validation.", file=sys.stderr)
    raise SystemExit(2)
original_parse_args = argparse.ArgumentParser.parse_args

class ArgumentsValidated(Exception):
    pass

def validate(self, args=None, namespace=None):
    for planned in planned_argv:
        try:
            original_parse_args(self, planned["args"], namespace)
        except SystemExit:
            print("Frozen argv rejected for " + planned["run_id"], file=sys.stderr)
            raise
    supported_options = sorted({option for action in self._actions for option in action.option_strings})
    normalized = json.dumps(supported_options, separators=(",", ":"))
    evidence = {
        "supported_options": supported_options,
        "cli_options_sha256": hashlib.sha256(normalized.encode()).hexdigest(),
    }
    print("AGENT_CLI_PREFLIGHT=" + json.dumps(evidence, sort_keys=True))
    raise ArgumentsValidated

argparse.ArgumentParser.parse_args = validate
sys.argv = [module, *planned_argv[0]["args"]]
try:
    runpy.run_module(module, run_name="__main__")
except ArgumentsValidated:
    raise SystemExit(0)
print("Target runtime did not validate arguments through argparse.", file=sys.stderr)
raise SystemExit(2)
""".strip()
    parse_result = run_command(
        execution,
        [python_command, "-c", parser_script, module, json.dumps(planned_argv), identity["module_origin"]],
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
        "expected_runtime_commit": str(expected_commit),
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
) -> str:
    workdir = str(execution.get("workdir") or REPO_ROOT)
    env = dict(execution.get("env") or {})
    if gpus:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(item) for item in gpus)
    run = [
        str(execution.get("python") or sys.executable),
        "-c",
        _PROCESS_LAUNCH_SCRIPT,
        str(script),
        str(log_path),
        str(pid_path),
        workdir,
    ]
    verification = None
    if execution_snapshot is not None:
        if config_path is None or not script_sha256 or not config_sha256:
            raise ValueError("Verified launch requires frozen script and config hashes.")
        artifacts = [
            {"path": str(script), "sha256": script_sha256},
            {"path": str(config_path), "sha256": config_sha256},
        ]
        if (checkpoint_path is None) != (checkpoint_sha256 is None):
            raise ValueError("Verified launch requires both frozen checkpoint path and hash.")
        if checkpoint_path is not None:
            artifacts.append({"path": str(checkpoint_path), "sha256": checkpoint_sha256})
        verification_inner = (
            "export PYTHONPATH="
            + _sh(workdir)
            + " && "
            + " ".join(
                _sh(part)
                for part in (
                    execution["python"],
                    "-c",
                    _RUNTIME_IDENTITY_SCRIPT,
                    execution_snapshot["module"],
                    json.dumps(execution_snapshot, sort_keys=True),
                    json.dumps(artifacts, sort_keys=True),
                )
            )
        )
        verification = ["bash", "-c", verification_inner]
    if execution.get("conda_env"):
        wrapper = ["conda", "run", "--no-capture-output", "-n", str(execution["conda_env"])]
        run = [*wrapper, *run]
        if verification is not None:
            verification = [*wrapper, *verification]
    run_command = " ".join(_sh(part) for part in run)
    verification_command = " ".join(_sh(part) for part in verification) if verification is not None else ""
    if env:
        env_prefix = " ".join(f"{key}={_sh(value)}" for key, value in sorted(env.items()))
        run_command = f"env {env_prefix} {run_command}"
        if verification_command:
            verification_command = f"env {env_prefix} {verification_command}"
    guard = f"{verification_command} >/dev/null && " if verification_command else ""
    if execution.get("target", "local") == "ssh":
        mkdir = f"mkdir -p {_sh(_parent_path(log_path))} {_sh(_parent_path(pid_path))}"
        inner = f"{mkdir} && cd {_sh(workdir)} && {guard}{run_command}"
        return f"ssh {_sh(execution['host'])} {_sh(inner)}"
    return f"cd {_sh(workdir)} && {guard}{run_command}"


def start_process(execution: dict[str, Any], command: str) -> str:
    del execution
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
    return "launched" if result.returncode == 0 else "launch_failed"


def _parent_path(path: str | Path) -> str:
    text = str(path)
    parent = text.rsplit("/", 1)[0] if "/" in text else "."
    return parent or "/"


_sh = transport.sh
