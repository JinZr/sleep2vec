from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import time
from typing import Any

import yaml

from . import experiment_io as exp_io, gpu_rules, run_artifacts as artifacts, run_evidence as evidence, transport
from .experiment_workspace import (
    EXECUTION_IDENTITY_FIELDS,
    PROCESS_IDENTITY_FIELDS,
    TERMINAL_STATUSES,
    append_event,
    experiment_root,
    managed_run_key,
    merge_run_manifest,
    merge_run_row,
    read_run_manifest,
    validate_frozen_run_update,
    write_status_report,
)
from .manifests import read_json, utc_now, write_rows
from .models import REPO_ROOT

LAUNCH_TIMEOUT_SECONDS = 60
EXECUTION_SNAPSHOT_NAME = "execution_snapshot.json"
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
    if path.is_symlink() or not path.is_file():
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


def launch_hparam_runs(
    plan_dir: str | Path,
    *,
    dry_run: bool = True,
    fail_on_missing_pid_blocker: bool = False,
) -> Path:
    run_dir = Path(plan_dir).expanduser()
    if not run_dir.is_absolute():
        run_dir = run_dir.resolve()
    plan = artifacts.read_hparam_plan(run_dir)
    recipe = plan.get("recipe") if isinstance(plan.get("recipe"), dict) else {}
    workspace = experiment_root(recipe)
    if workspace is None:
        raise ValueError("Hparam plan is not bound to an experiment workspace.")
    lock_path = workspace / "run_manifest.tsv.lock"
    exp_io.validate_managed_output_paths(workspace, [lock_path])
    with lock_path.open("a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            return _launch_hparam_runs(
                run_dir,
                dry_run=dry_run,
                manifest_lock_held=True,
                fail_on_missing_pid_blocker=fail_on_missing_pid_blocker,
            )
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def run_hparam_queue(
    plan_dir: str | Path,
    *,
    dry_run: bool = True,
    poll_seconds: float = 60,
) -> Path:
    if not math.isfinite(poll_seconds) or poll_seconds <= 0:
        raise ValueError("poll_seconds must be positive.")
    run_dir = Path(plan_dir).expanduser()
    if not run_dir.is_absolute():
        run_dir = run_dir.resolve()
    if dry_run:
        return launch_hparam_runs(run_dir, dry_run=True)

    plan = artifacts.read_hparam_plan(run_dir)
    recipe = plan.get("recipe") if isinstance(plan.get("recipe"), dict) else {}
    workspace = experiment_root(recipe)
    if workspace is None:
        raise ValueError("Hparam plan is not bound to an experiment workspace.")
    expected_keys = {managed_run_key(run) for run in plan["runs"]}
    status_path = run_dir / "run_status.tsv"
    exp_io.validate_managed_output_paths(workspace, [status_path])
    while True:
        rows_by_key = {managed_run_key(row): row for row in read_run_manifest(workspace)}
        if all(rows_by_key[key].get("status") in TERMINAL_STATUSES for key in expected_keys):
            write_rows(status_path, [rows_by_key[managed_run_key(run)] for run in plan["runs"]])
            return status_path
        missing_pid = sorted(key for key in expected_keys if rows_by_key[key].get("status") == "missing_pid")
        if missing_pid:
            step_id, run_id = missing_pid[0]
            raise RuntimeError(f"Hparam queue cannot advance because {step_id} / {run_id} has status missing_pid.")

        monitor_hparam_runs(run_dir)
        rows_by_key = {managed_run_key(row): row for row in read_run_manifest(workspace)}
        if all(rows_by_key[key].get("status") in TERMINAL_STATUSES for key in expected_keys):
            return status_path
        missing_pid = sorted(key for key in expected_keys if rows_by_key[key].get("status") == "missing_pid")
        if missing_pid:
            step_id, run_id = missing_pid[0]
            raise RuntimeError(f"Hparam queue cannot advance because {step_id} / {run_id} has status missing_pid.")

        launch_hparam_runs(run_dir, dry_run=False, fail_on_missing_pid_blocker=True)
        rows_by_key = {managed_run_key(row): row for row in read_run_manifest(workspace)}
        if all(rows_by_key[key].get("status") in TERMINAL_STATUSES for key in expected_keys):
            return status_path
        missing_pid = sorted(key for key in expected_keys if rows_by_key[key].get("status") == "missing_pid")
        if missing_pid:
            step_id, run_id = missing_pid[0]
            raise RuntimeError(f"Hparam queue cannot advance because {step_id} / {run_id} has status missing_pid.")
        time.sleep(poll_seconds)


def reconcile_hparam_launch_artifacts(plan_dir: str | Path, started_keys: set[tuple[str, str]]) -> list[dict[str, Any]]:
    run_dir = Path(plan_dir).expanduser()
    if not run_dir.is_absolute():
        run_dir = run_dir.resolve()
    plan = artifacts.read_hparam_plan(run_dir)
    recipe = plan.get("recipe") if isinstance(plan.get("recipe"), dict) else {}
    workspace = experiment_root(recipe)
    if workspace is None:
        raise ValueError("Hparam plan is not bound to an experiment workspace.")
    exp_io.validate_managed_output_paths(
        workspace,
        [
            workspace / "events.jsonl",
            workspace / "reports" / "status.md",
            run_dir / "launch_manifest.tsv",
            run_dir / "run_status.tsv",
            run_dir / EXECUTION_SNAPSHOT_NAME,
        ],
    )
    canonical_by_key = {managed_run_key(row): row for row in read_run_manifest(workspace)}
    expected_keys = {managed_run_key(run) for run in plan["runs"]}
    if not started_keys.issubset(expected_keys):
        raise ValueError("Interrupted launch evidence is outside the current hparam plan.")
    rows = [canonical_by_key[managed_run_key(run)] for run in plan["runs"]]
    write_rows(run_dir / "launch_manifest.tsv", rows)
    write_rows(run_dir / "run_status.tsv", rows)
    events_path = workspace / "events.jsonl"

    def launched_event_keys() -> set[tuple[str, str]]:
        if not events_path.exists():
            return set()
        keys = set()
        for line in events_path.read_text().splitlines():
            event = json.loads(line)
            if event.get("event_type") == "run_launched":
                keys.add((str(event.get("step_id") or ""), str(event.get("run_id") or "")))
        return keys

    launched_events = launched_event_keys()
    for key in sorted(started_keys - launched_events):
        try:
            append_event(
                workspace,
                "run_launched",
                {"step_id": key[0], "run_id": key[1], "gpus": canonical_by_key[key].get("gpus", "")},
            )
        except Exception:
            if key not in launched_event_keys():
                raise
    write_status_report(workspace)
    return rows


def _launch_hparam_runs(
    plan_dir: str | Path,
    *,
    dry_run: bool = True,
    manifest_lock_held: bool,
    fail_on_missing_pid_blocker: bool,
) -> Path:
    run_dir = Path(plan_dir).expanduser()
    if not run_dir.is_absolute():
        run_dir = run_dir.resolve()
    plan = artifacts.read_hparam_plan(run_dir)
    recipe = plan.get("recipe") if isinstance(plan.get("recipe"), dict) else {}
    workspace = experiment_root(recipe)
    if workspace is None:
        raise ValueError("Hparam plan is not bound to an experiment workspace.")
    exp_io.validate_managed_output_paths(
        workspace,
        [
            workspace / "run_manifest.tsv",
            workspace / "run_matrix.csv",
            workspace / "reports" / "run_matrix.md",
            workspace / "events.jsonl",
            workspace / "reports" / "status.md",
            run_dir / "launch_manifest.tsv",
            run_dir / "run_status.tsv",
            run_dir / EXECUTION_SNAPSHOT_NAME,
        ],
    )
    experiment_manifest = yaml.safe_load((workspace / "experiment.yaml").read_text()) or {}
    experiment = experiment_manifest.get("experiment") if isinstance(experiment_manifest, dict) else None
    if isinstance(experiment, dict) and experiment.get("status") == "completed":
        raise ValueError(f"Experiment is completed and cannot launch runs: {workspace}")
    execution = recipe.get("execution") if isinstance(recipe.get("execution"), dict) else {}
    runs = plan["runs"]
    target = str(execution.get("target", "local") or "local")
    gpu_groups = _gpu_groups(recipe)
    max_concurrent = int(execution["max_concurrent"]) if "max_concurrent" in execution else max(len(gpu_groups), 1)
    if max_concurrent <= 0:
        raise ValueError("execution.max_concurrent must be a positive integer.")
    allow_gpu_oversubscription = bool(gpu_groups) and max_concurrent > len(gpu_groups)
    expected_keys = {managed_run_key(run) for run in runs}
    manifest = run_dir / "launch_manifest.tsv"
    status_path = run_dir / "run_status.tsv"
    workspace_by_key = {managed_run_key(row): row for row in read_run_manifest(workspace)}
    snapshot_path = run_dir / EXECUTION_SNAPSHOT_NAME
    if (
        not dry_run
        and not snapshot_path.exists()
        and any(
            workspace_by_key[managed_run_key(run)].get("target") not in (None, "")
            or workspace_by_key[managed_run_key(run)].get("status") not in {"planned", "pending"}
            for run in runs
        )
    ):
        # Fail closed before probing when the durable execution marker is missing.
        _validated_execution_snapshot(run_dir, execution, runs, workspace_by_key)
    refreshed = {}
    observed_status_changes = {}
    for key in expected_keys:
        previous = workspace_by_key.get(key)
        if previous is None:
            raise ValueError(f"Canonical run is missing for the current hparam plan: {key[0]} / {key[1]}")
        # Dry-run only renders launch metadata; it must never probe PID, logs, or SSH.
        if dry_run or previous.get("target") in (None, ""):
            refreshed[key] = previous
            continue
        observation = {field: previous[field] for field in evidence.RUN_EVIDENCE_FIELDS if field in previous}
        observation.update({"step_id": key[0], "run_id": key[1], "status": previous.get("status", "")})
        refreshed[key] = evidence.status_row(
            run_dir,
            observation,
            previous,
            script_commits_terminal_status=False,
            health=False,
        )
        if refreshed[key].get("status") != previous.get("status"):
            observed_status_changes[key] = (previous.get("status"), refreshed[key].get("status"))
    active_statuses = {"launched", "running", "unknown_remote", "missing_pid"}
    current_host = str(execution.get("host") or "") if target == "ssh" else ""
    gpu_group_values = [{str(item) for item in group} for group in gpu_groups]
    current_gpu_pool = set().union(*gpu_group_values) if gpu_group_values else set()
    other_active_gpu_sets = []
    unknown_other_active = 0
    external_status_changes = {}
    external_missing_pid = []
    for key, row in list(workspace_by_key.items()):
        if not gpu_groups or key in expected_keys or row.get("status") not in active_statuses:
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
        observable = all(
            row.get(field) not in (None, "")
            for field in ("target", "workdir", "pid_path", "log_path", "command", "script")
        )
        if not dry_run and observable:
            observation = {field: row[field] for field in evidence.RUN_EVIDENCE_FIELDS if field in row}
            observation.update({"step_id": key[0], "run_id": key[1], "status": row.get("status", "")})
            observed = evidence.status_row(
                run_dir,
                observation,
                row,
                script_commits_terminal_status=False,
                health=False,
            )
            if observed.get("status") != row.get("status"):
                external_status_changes[key] = (row.get("status"), observed.get("status"))
                workspace_by_key[key] = observed
                row = observed
            if row.get("status") not in active_statuses:
                continue
        if row.get("status") == "missing_pid":
            external_missing_pid.append(key)
        if not assigned:
            unknown_other_active += 1
            continue
        other_active_gpu_sets.append(assigned)
    if external_status_changes:
        committed = merge_run_manifest(
            workspace,
            [workspace_by_key[key] for key in external_status_changes],
            lock_held=manifest_lock_held,
        )
        workspace_by_key = {managed_run_key(row): row for row in committed}
        for key, (before, after) in external_status_changes.items():
            append_event(
                workspace,
                "run_status_changed",
                {"step_id": key[0], "run_id": key[1], "from": before, "to": after},
            )
        write_status_report(workspace)
    if not dry_run and fail_on_missing_pid_blocker and external_missing_pid:
        step_id, run_id = sorted(external_missing_pid)[0]
        raise RuntimeError(f"Hparam launch capacity is blocked because {step_id} / {run_id} has status missing_pid.")
    active = (
        sum(row.get("status") in active_statuses for row in refreshed.values())
        + len(other_active_gpu_sets)
        + unknown_other_active
    )
    slots = max(max_concurrent - active, 0)
    gpu_group_by_value = {",".join(str(item) for item in group): index for index, group in enumerate(gpu_groups)}
    active_gpu_loads = [unknown_other_active] * len(gpu_groups)
    for assigned in other_active_gpu_sets:
        for group_index, group in enumerate(gpu_group_values):
            if assigned.intersection(group):
                active_gpu_loads[group_index] += 1
    assigned_group_by_key = {}
    for key, previous in refreshed.items():
        assigned = ",".join(part.strip() for part in str(previous.get("gpus") or "").split(",") if part.strip())
        if not assigned:
            if previous.get("status") in active_statuses:
                for group_index in range(len(gpu_groups)):
                    active_gpu_loads[group_index] += 1
            continue
        group_index = gpu_group_by_value.get(assigned)
        if group_index is None:
            raise ValueError(f"Frozen GPUs are not one configured GPU group for {key[0]} / {key[1]}: {assigned}")
        assigned_group_by_key[key] = group_index
        if previous.get("status") in active_statuses:
            active_gpu_loads[group_index] += 1
    rows = []
    launch_identity_by_key = {}
    for run in runs:
        key = managed_run_key(run)
        run_id = str(run["run_id"])
        script = Path(str(run["script"]))
        previous = refreshed.get(key, {})
        semantic_run_dir = Path(str(run.get("run_dir") or script.parent))
        log_path = semantic_run_dir / "stdout.log"
        pid_path = semantic_run_dir / "pid"
        launch_identity_by_key[key] = {
            "target": target,
            "host": execution.get("host", ""),
            "workdir": execution.get("workdir") or str(REPO_ROOT),
            "gpus": "",
            "log_path": str(log_path),
            "pid_path": str(pid_path),
            "command": "",
            **{field: "" for field in PROCESS_IDENTITY_FIELDS},
        }
        execution_identity = (
            {field: previous.get(field, "") for field in launch_identity_by_key[key]}
            if previous.get("target") not in (None, "")
            else {field: "" for field in launch_identity_by_key[key]}
        )
        status = previous.get("status") or "planned"
        launched_at = previous.get("launched_at", "")
        row = {
            "experiment_id": run["experiment_id"],
            "step_id": run["step_id"],
            "run_id": run_id,
            "run_name": run["run_name"],
            "parameter_summary": run.get("parameter_summary", ""),
            "version": run["version"],
            "config": run.get("config"),
            "config_sha256": run.get("config_sha256"),
            "script": str(script),
            "script_sha256": run.get("script_sha256"),
            "run_dir": str(semantic_run_dir),
            "runtime_dir": run["runtime_dir"],
            "checkpoint_dir": run["checkpoint_dir"],
            **execution_identity,
            "status": status,
            "launched_at": launched_at,
        }
        for field in ("stopped_at", "stop_reason"):
            if previous.get(field):
                row[field] = previous[field]
        rows.append(row)
    for row in rows:
        validate_frozen_run_update(
            workspace_by_key[managed_run_key(row)],
            row,
            allow_execution_identity_fill=True,
        )
    run_output_paths = [
        Path(str(launch_identity_by_key[managed_run_key(row)][field]))
        for row in rows
        for field in ("log_path", "pid_path")
    ]
    if target == "ssh":
        if not dry_run:
            exp_io.validate_managed_output_paths(
                workspace,
                run_output_paths,
                remote=str(execution["host"]),
            )
    else:
        exp_io.validate_managed_output_paths(workspace, run_output_paths)
    execution_snapshot = None
    if not dry_run:
        launchable = [row for row in rows if row["status"] in {"planned", "pending"}]
        has_launch_candidate = False
        if slots > 0:
            for row in launchable:
                frozen_group_index = assigned_group_by_key.get(managed_run_key(row))
                if frozen_group_index is not None:
                    group_indexes = [frozen_group_index]
                elif gpu_groups:
                    group_indexes = list(range(len(gpu_groups)))
                else:
                    group_indexes = [None]
                if any(
                    group_index is None or allow_gpu_oversubscription or active_gpu_loads[group_index] < 1
                    for group_index in group_indexes
                ):
                    has_launch_candidate = True
                    break
        if has_launch_candidate:
            runtime_roots = [Path(str(row[field])) for row in launchable for field in ("runtime_dir", "checkpoint_dir")]
            runtime_root = Path(str(execution.get("workdir") or REPO_ROOT))
            remote_host = str(execution["host"]) if target == "ssh" else None
            # Trainer artifact directories are single-use; aliases or prior contents must fail before any start.
            exp_io.validate_managed_output_paths(runtime_root, runtime_roots, remote=remote_host)
            execution_snapshot, write_execution_snapshot = _validated_execution_snapshot(
                run_dir,
                execution,
                runs,
                workspace_by_key,
            )
            if write_execution_snapshot:
                snapshot_path = run_dir / EXECUTION_SNAPSHOT_NAME
                payload = (json.dumps(execution_snapshot, indent=2, sort_keys=True) + "\n").encode()
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
    if target != "ssh":
        for row in rows:
            Path(str(row["run_dir"])).mkdir(parents=True, exist_ok=True)
    started_keys = set()
    if dry_run:
        preview_gpu_loads = list(active_gpu_loads)
        for row in rows:
            if row["status"] not in {"planned", "pending"} or row.get("target") not in (None, ""):
                continue
            group_index = (
                min(range(len(gpu_groups)), key=lambda index: (preview_gpu_loads[index], index)) if gpu_groups else None
            )
            gpus = list(gpu_groups[group_index]) if group_index is not None else []
            identity = dict(launch_identity_by_key[managed_run_key(row)])
            identity["gpus"] = ",".join(str(item) for item in gpus)
            identity["command"] = _launch_command(
                execution,
                Path(str(row["script"])),
                identity["log_path"],
                identity["pid_path"],
                gpus,
            )
            row.update(identity)
            validate_frozen_run_update(
                workspace_by_key[managed_run_key(row)],
                row,
                allow_execution_identity_fill=True,
            )
            if group_index is not None:
                preview_gpu_loads[group_index] += 1
    else:
        launchable = [(index, row) for index, row in enumerate(rows) if row["status"] in {"planned", "pending"}]
        while launchable and slots > 0:
            eligible = []
            for index, row in launchable:
                frozen_group_index = assigned_group_by_key.get(managed_run_key(row))
                if frozen_group_index is not None:
                    group_indexes = [frozen_group_index]
                elif gpu_groups:
                    group_indexes = list(range(len(gpu_groups)))
                else:
                    group_indexes = [None]
                for group_index in group_indexes:
                    load = active_gpu_loads[group_index] if group_index is not None else 0
                    if group_index is not None and not allow_gpu_oversubscription and load >= 1:
                        continue
                    eligible.append((load, index, row, group_index))
            if not eligible:
                break
            _load, index, row, group_index = min(
                eligible,
                key=lambda item: (item[0], item[1], item[3] if item[3] is not None else -1),
            )
            launchable = [
                (candidate_index, candidate) for candidate_index, candidate in launchable if candidate_index != index
            ]
            if row.get("target") in (None, ""):
                gpus = list(gpu_groups[group_index]) if group_index is not None else []
                identity = dict(launch_identity_by_key[managed_run_key(row)])
                identity["gpus"] = ",".join(str(item) for item in gpus)
                identity["command"] = _launch_command(
                    execution,
                    Path(str(row["script"])),
                    identity["log_path"],
                    identity["pid_path"],
                    gpus,
                    execution_snapshot=execution_snapshot,
                    config_path=Path(str(row["config"])),
                    script_sha256=str(row["script_sha256"]),
                    config_sha256=str(row["config_sha256"]),
                )
                row.update(identity)
                validate_frozen_run_update(
                    workspace_by_key[managed_run_key(row)],
                    row,
                    allow_execution_identity_fill=True,
                )
            key = managed_run_key(row)
            committed = merge_run_manifest(workspace, [row], lock_held=manifest_lock_held)
            committed_by_key = {managed_run_key(item): item for item in committed}
            row.clear()
            row.update(committed_by_key[key])
            if row["status"] not in {"planned", "pending"}:
                continue
            row["status"] = _start_process(execution, row["command"])
            row["launched_at"] = utc_now() if row["status"] == "launched" else ""
            if row["status"] == "launched":
                try:
                    process_identity = evidence.read_process_identity(row["pid_path"], row)
                except RuntimeError:
                    process_identity = None
                if process_identity is not None:
                    row.update(process_identity)
            committed = merge_run_manifest(workspace, [row], lock_held=manifest_lock_held)
            committed_by_key = {managed_run_key(item): item for item in committed}
            row.clear()
            row.update(committed_by_key[key])
            if row["status"] == "launched":
                started_keys.add(managed_run_key(row))
                if group_index is not None:
                    active_gpu_loads[group_index] += 1
                slots -= 1
        for _index, row in launchable:
            if row["status"] == "planned":
                row["status"] = "pending"
    commit_rows = []
    for row in rows:
        committed_row = dict(row)
        if dry_run and workspace_by_key[managed_run_key(row)].get("target") in (None, ""):
            committed_row.update({field: "" for field in EXECUTION_IDENTITY_FIELDS})
        commit_rows.append(committed_row)
    committed = merge_run_manifest(workspace, commit_rows, lock_held=manifest_lock_held)
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
    write_rows(manifest, launch_rows)
    write_rows(status_path, committed_rows)
    for row in committed_rows:
        key = managed_run_key(row)
        if key in observed_status_changes:
            before, after = observed_status_changes[key]
            append_event(
                workspace,
                "run_status_changed",
                {
                    "step_id": row["step_id"],
                    "run_id": row["run_id"],
                    "from": before,
                    "to": after,
                },
            )
        if key in started_keys:
            append_event(
                workspace,
                "run_launched",
                {"step_id": (recipe.get("step") or {}).get("id"), "run_id": row["run_id"], "gpus": row["gpus"]},
            )
    write_status_report(workspace)
    return manifest


def _validated_execution_snapshot(
    run_dir: Path,
    execution: dict[str, Any],
    runs: list[dict[str, Any]],
    workspace_by_key: dict[tuple[str, str], dict[str, Any]],
) -> tuple[dict[str, Any], bool]:
    snapshot_path = run_dir / EXECUTION_SNAPSHOT_NAME
    if snapshot_path.exists():
        frozen = read_json(snapshot_path)
        if not isinstance(frozen, dict):
            raise ValueError(f"Execution snapshot must be a mapping: {snapshot_path}")
        actual = _inspect_execution_target(execution, runs)
        if frozen != actual:
            changed = sorted(key for key in set(frozen) | set(actual) if frozen.get(key) != actual.get(key))
            raise ValueError(f"Frozen execution snapshot changed: {', '.join(changed)}")
        return actual, False

    for run in runs:
        row = workspace_by_key[managed_run_key(run)]
        if row.get("target") not in (None, "") or row.get("status") not in {"planned", "pending"}:
            raise ValueError(
                "Cannot establish an execution snapshot after a hparam run has started; create a new plan."
            )
    actual = _inspect_execution_target(execution, runs)
    return actual, True


def _inspect_execution_target(execution: dict[str, Any], runs: list[dict[str, Any]]) -> dict[str, Any]:
    modules = set()
    python_commands = set()
    planned_argv = []
    required_options = set()
    for run in runs:
        command = str(run.get("command") or "")
        if command not in Path(str(run["script"])).read_text().splitlines():
            raise ValueError(f"Frozen hparam command differs from its launch script: {run['run_id']}")
        tokens = shlex.split(command)
        try:
            module_flag_index = tokens.index("-m")
            module_index = module_flag_index + 1
            modules.add(tokens[module_index])
        except (IndexError, ValueError) as exc:
            raise ValueError(f"Frozen hparam command has no Python module: {run['run_id']}") from exc
        if module_flag_index != 1:
            raise ValueError(f"Frozen hparam command has an unsupported Python invocation: {run['run_id']}")
        python_commands.add(tokens[0])
        planned_argv.append({"run_id": str(run["run_id"]), "args": tokens[module_index + 1 :]})
        required_options.update(token for token in tokens[module_index + 1 :] if token.startswith("--"))
    if len(modules) != 1:
        raise ValueError("A hparam plan must use exactly one target runtime module.")
    if len(python_commands) != 1:
        raise ValueError("A hparam plan must use exactly one target Python command.")
    module = next(iter(modules))
    python_command = next(iter(python_commands))
    expected_python = execution.get("python")
    expected_commit = execution.get("runtime_commit")
    if expected_python in (None, "") or expected_commit in (None, ""):
        raise ValueError("Frozen hparam plan lacks execution.python or execution.runtime_commit; create a new plan.")
    if python_command != str(expected_python):
        raise ValueError("Frozen hparam commands differ from execution.python.")

    identity_result = _run_execution_command(execution, [python_command, "-c", _RUNTIME_IDENTITY_SCRIPT, module])
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
    if not isinstance(identity, dict) or any(
        identity.get(field) in (None, "")
        for field in (
            "python",
            "python_version",
            "runtime_commit",
            "runtime_repo_root",
            "runtime_hostname",
            "module",
            "module_origin",
        )
    ):
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
    parse_result = _run_execution_command(
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


def _run_execution_command(execution: dict[str, Any], command: list[str]) -> subprocess.CompletedProcess:
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


def monitor_hparam_runs(run_dir: str | Path, *, once: bool = True, health: bool = False) -> Path:
    root = Path(run_dir)
    plan = artifacts.read_hparam_plan(root)
    recipe = plan.get("recipe") if isinstance(plan.get("recipe"), dict) else {}
    expected_keys = {managed_run_key(run) for run in plan["runs"]}
    status_path = root / "run_status.tsv"
    workspace = experiment_root(recipe)
    if workspace is None:
        raise ValueError("Hparam plan is not bound to an experiment workspace.")
    exp_io.validate_managed_output_paths(
        workspace,
        [
            workspace / "run_manifest.tsv",
            workspace / "run_matrix.csv",
            workspace / "reports" / "run_matrix.md",
            workspace / "events.jsonl",
            workspace / "reports" / "status.md",
            root / "launch_manifest.tsv",
            root / "run_status.tsv",
        ],
    )
    workspace_rows = read_run_manifest(workspace)
    workspace_by_key = {managed_run_key(row): row for row in workspace_rows}
    missing = expected_keys - set(workspace_by_key)
    if missing:
        step_id, run_id = sorted(missing)[0]
        raise ValueError(f"Canonical run is missing for the current hparam plan: {step_id} / {run_id}")
    previous_rows = {key: workspace_by_key[key] for key in expected_keys}
    rows = []
    for run in plan["runs"]:
        key = managed_run_key(run)
        prior = previous_rows[key]
        if prior.get("target") in (None, ""):
            rows.append(prior)
            continue
        observation = {field: prior[field] for field in evidence.RUN_EVIDENCE_FIELDS if field in prior}
        observation.update({"step_id": key[0], "run_id": key[1], "status": prior.get("status", "")})
        rows.append(
            evidence.status_row(
                root,
                observation,
                prior,
                script_commits_terminal_status=False,
                health=health,
            )
        )
    out = status_path
    committed = merge_run_manifest(workspace, rows)
    committed_by_key = {managed_run_key(row): row for row in committed}
    rows = [committed_by_key[managed_run_key(run)] for run in plan["runs"]]
    write_rows(out, rows)
    for row in rows:
        before = previous_rows[managed_run_key(row)].get("status")
        after = row.get("status")
        if before and after and before != after:
            append_event(
                workspace,
                "run_status_changed",
                {
                    "step_id": row["step_id"],
                    "run_id": row["run_id"],
                    "from": before,
                    "to": after,
                },
            )
    write_status_report(workspace)
    if not once:
        print(f"wrote {out}")
    return out


def stop_hparam_run(run_dir: str | Path, run_id: str, *, reason: str) -> Path:
    if not reason.strip():
        raise ValueError("Stopping a run requires a non-empty reason.")
    root = Path(run_dir)
    plan = artifacts.read_hparam_plan(root)
    recipe = plan.get("recipe") if isinstance(plan.get("recipe"), dict) else {}
    expected_keys = {managed_run_key(run) for run in plan["runs"]}
    manifest_path = root / "launch_manifest.tsv"
    status_path = root / "run_status.tsv"
    workspace = experiment_root(recipe)
    if workspace is None:
        raise ValueError("Hparam plan is not bound to an experiment workspace.")
    exp_io.validate_managed_output_paths(
        workspace,
        [
            workspace / "run_manifest.tsv",
            workspace / "run_matrix.csv",
            workspace / "reports" / "run_matrix.md",
            workspace / "events.jsonl",
            workspace / "reports" / "status.md",
            root / "launch_manifest.tsv",
            root / "run_status.tsv",
        ],
    )
    workspace_rows = read_run_manifest(workspace)
    workspace_by_key = {managed_run_key(item): item for item in workspace_rows}
    missing = expected_keys - set(workspace_by_key)
    if missing:
        step_id, missing_run_id = sorted(missing)[0]
        raise ValueError(f"Canonical run is missing for the current hparam plan: {step_id} / {missing_run_id}")
    matched = [run for run in plan["runs"] if run.get("run_id") == run_id]
    if not matched:
        raise ValueError(f"Unknown run_id: {run_id}")
    if len(matched) > 1:
        raise ValueError(f"Ambiguous run_id in hparam plan: {run_id}")
    key = managed_run_key(matched[0])
    previous = workspace_by_key[key]
    missing_execution_identity = {
        field for field in EXECUTION_IDENTITY_FIELDS - PROCESS_IDENTITY_FIELDS if field not in previous
    }
    if previous.get("target") in (None, ""):
        missing_execution_identity.add("target")
    if missing_execution_identity:
        raise ValueError(
            f"Canonical run is missing execution identity for {run_id}: {', '.join(sorted(missing_execution_identity))}"
        )
    if previous.get("status") in TERMINAL_STATUSES:
        raise ValueError(f"Run is already terminal and cannot be stopped: {run_id} ({previous['status']})")
    target = previous.get("target")
    if target not in {"local", "ssh"}:
        raise ValueError(f"Canonical run target must be local or ssh for run_id: {run_id}")
    host = previous.get("host")
    if target == "ssh" and (not isinstance(host, str) or not host.strip()):
        raise ValueError(f"Canonical SSH run requires a non-empty host for run_id: {run_id}")
    populated_process_fields = {field for field in PROCESS_IDENTITY_FIELDS if previous.get(field) not in (None, "")}
    if populated_process_fields and populated_process_fields != PROCESS_IDENTITY_FIELDS:
        missing = ", ".join(sorted(PROCESS_IDENTITY_FIELDS - populated_process_fields))
        raise ValueError(f"Canonical run has partial process identity for {run_id}; missing: {missing}")
    remote_host = str(host) if target == "ssh" else None
    exp_io.validate_managed_output_paths(
        workspace,
        [previous["pid_path"]],
        remote=remote_host,
    )
    if populated_process_fields:
        process_identity = evidence.read_process_identity(previous.get("pid_path"), previous)
    else:
        process_identity = evidence.read_process_identity(
            previous.get("pid_path"),
            previous,
            expected_script=previous.get("script"),
        )
    if process_identity is None:
        raise ValueError(f"No recorded PID for run_id: {run_id}")
    for field in PROCESS_IDENTITY_FIELDS:
        frozen_value = previous.get(field)
        if frozen_value not in (None, "") and str(frozen_value) != str(process_identity[field]):
            raise RuntimeError(f"Recorded process identity differs from canonical {field} for run_id: {run_id}")
    evidence.stop_process_group(previous, process_identity)
    stopped_at = utc_now()
    final = merge_run_row(
        previous,
        {
            "step_id": key[0],
            "run_id": key[1],
            **process_identity,
            "status": "stopped",
            "stopped_at": stopped_at,
            "stop_reason": reason,
        },
    )
    committed = merge_run_manifest(workspace, [final])
    committed_by_key = {managed_run_key(item): item for item in committed}
    final_status_rows = [committed_by_key[managed_run_key(run)] for run in plan["runs"]]
    write_rows(status_path, final_status_rows)
    write_rows(manifest_path, final_status_rows)
    append_event(
        workspace,
        "run_stopped",
        {"step_id": key[0], "run_id": run_id, "reason": reason},
    )
    write_status_report(workspace)
    return status_path


def _gpu_groups(recipe: dict[str, Any]) -> list[list[Any]]:
    execution = recipe.get("execution") if isinstance(recipe.get("execution"), dict) else {}
    runtime = recipe.get("runtime") if isinstance(recipe.get("runtime"), dict) else {}
    groups, issues = gpu_rules.gpu_group_plan(execution, runtime)
    errors = [issue for issue in issues if not issue.warning]
    if errors:
        raise ValueError(errors[0].message)
    return groups


def _launch_command(
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
        wrapper = [
            "conda",
            "run",
            "--no-capture-output",
            "-n",
            str(execution["conda_env"]),
        ]
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
    inner = f"cd {_sh(workdir)} && {guard}{run_command}"
    return inner


def _parent_path(path: str | Path) -> str:
    text = str(path)
    parent = text.rsplit("/", 1)[0] if "/" in text else "."
    return parent or "/"


def _start_process(execution: dict[str, Any], command: str) -> str:
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


_sh = transport.sh
