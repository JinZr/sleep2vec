from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
from typing import Any

import yaml

from . import experiment_io as exp_io, run_artifacts as artifacts, run_evidence as evidence
from .experiment_workspace import (
    EXECUTION_IDENTITY_FIELDS,
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
from .manifests import utc_now, write_rows
from .models import REPO_ROOT

LAUNCH_TIMEOUT_SECONDS = 60


def launch_hparam_runs(
    plan_dir: str | Path,
    *,
    dry_run: bool = True,
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
            )
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


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
    refreshed = {}
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
        refreshed[key] = evidence.status_row(run_dir, observation, previous, health=False)
    active_statuses = {"launched", "running", "unknown_remote", "missing_pid"}
    current_host = str(execution.get("host") or "") if target == "ssh" else ""
    gpu_group_values = [{str(item) for item in group} for group in gpu_groups]
    current_gpu_pool = set().union(*gpu_group_values) if gpu_group_values else set()
    other_active_gpu_sets = []
    unknown_other_active = 0
    for key, row in workspace_by_key.items():
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
        if not assigned:
            unknown_other_active += 1
            continue
        if not assigned.intersection(current_gpu_pool):
            continue
        other_active_gpu_sets.append(assigned)
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
    if not dry_run:
        launchable = [row for row in rows if row["status"] in {"planned", "pending"}]
        runtime_roots = [Path(str(row[field])) for row in launchable for field in ("runtime_dir", "checkpoint_dir")]
        runtime_root = Path(str(execution.get("workdir") or REPO_ROOT))
        remote_host = str(execution["host"]) if target == "ssh" else None
        # Trainer artifact directories are single-use; aliases or prior contents must fail before any start.
        exp_io.validate_managed_output_paths(runtime_root, runtime_roots, remote=remote_host)
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
        if managed_run_key(row) in started_keys:
            append_event(
                workspace,
                "run_launched",
                {"step_id": (recipe.get("step") or {}).get("id"), "run_id": row["run_id"], "gpus": row["gpus"]},
            )
    write_status_report(workspace)
    return manifest


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
        rows.append(evidence.status_row(root, observation, prior, health=health))
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
    missing_execution_identity = {field for field in EXECUTION_IDENTITY_FIELDS if field not in previous}
    if previous.get("target") in (None, ""):
        missing_execution_identity.add("target")
    if missing_execution_identity:
        raise ValueError(
            f"Canonical run is missing execution identity for {run_id}: {', '.join(sorted(missing_execution_identity))}"
        )
    if previous.get("status") in TERMINAL_STATUSES:
        raise ValueError(f"Run is already terminal and cannot be stopped: {run_id} ({previous['status']})")
    pid = evidence.read_pid(previous.get("pid_path"), previous)
    if pid is None:
        raise ValueError(f"No recorded PID for run_id: {run_id}")
    if previous.get("target") == "ssh":
        result = subprocess.run(
            ["ssh", previous["host"], f"kill -TERM {pid}"],
            check=False,
            timeout=evidence.SSH_TIMEOUT_SECONDS,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to stop remote run {run_id} on {previous['host']}.")
    else:
        os.kill(pid, signal.SIGTERM)
    stopped_at = utc_now()
    final = merge_run_row(
        previous,
        {
            "step_id": key[0],
            "run_id": key[1],
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
    devices = _as_list(runtime.get("devices"))
    pool = _as_list(execution.get("gpu_pool")) or devices
    if not pool:
        if "gpus_per_run" in execution:
            raise ValueError("execution.gpus_per_run requires a non-empty execution.gpu_pool or runtime.devices.")
        return []
    if len({str(item) for item in pool}) != len(pool):
        raise ValueError("The effective GPU pool must not contain duplicate GPU identifiers.")
    per_run = int(execution["gpus_per_run"]) if "gpus_per_run" in execution else len(devices) or 1
    if per_run <= 0:
        raise ValueError("execution.gpus_per_run must be a positive integer.")
    if per_run > len(pool):
        raise ValueError("execution.gpus_per_run cannot exceed the effective GPU pool size.")
    if len(pool) % per_run != 0:
        raise ValueError("The effective GPU pool must divide evenly into disjoint per-run GPU groups.")
    return [pool[index : index + per_run] for index in range(0, len(pool), per_run)]


def _as_list(value: Any) -> list[Any]:
    if value in (None, "", "ASK_USER"):
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _launch_command(
    execution: dict[str, Any],
    script: Path,
    log_path: str | Path,
    pid_path: str | Path,
    gpus: list[Any],
) -> str:
    env = dict(execution.get("env") or {})
    if gpus:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(item) for item in gpus)
    run = ["bash", str(script)]
    if execution.get("conda_env"):
        run = [
            "conda",
            "run",
            "--no-capture-output",
            "-n",
            str(execution["conda_env"]),
            *run,
        ]
    run_command = " ".join(_sh(part) for part in run)
    if env:
        env_prefix = " ".join(f"{key}={_sh(value)}" for key, value in sorted(env.items()))
        run_command = f"env {env_prefix} {run_command}"
    workdir = execution.get("workdir") or str(REPO_ROOT)
    if execution.get("target", "local") == "ssh":
        mkdir = f"mkdir -p {_sh(_parent_path(log_path))} {_sh(_parent_path(pid_path))}"
        inner = (
            f"{mkdir} && cd {_sh(workdir)} && "
            f"(nohup {run_command} > {_sh(log_path)} 2>&1 & echo $! > {_sh(pid_path)})"
        )
        return f"ssh {_sh(execution['host'])} {_sh(inner)}"
    inner = (
        f"cd {_sh(workdir)} && (nohup {run_command} > {_sh(log_path)} 2>&1 & echo $! > {_sh(pid_path)})"  # noqa: E501
    )
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
        return "launch_failed"
    return "launched" if result.returncode == 0 else "launch_failed"


def _sh(value: Any) -> str:
    return shlex.quote(str(value))
