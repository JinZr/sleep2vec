from __future__ import annotations

import fcntl
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


def launch_hparam_runs(plan_dir: str | Path, *, dry_run: bool = True) -> Path:
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
            return _launch_hparam_runs(run_dir, dry_run=dry_run, manifest_lock_held=True)
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _launch_hparam_runs(plan_dir: str | Path, *, dry_run: bool = True, manifest_lock_held: bool) -> Path:
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
    active = sum(row.get("status") in active_statuses for row in refreshed.values())
    slots = max(max_concurrent - active, 0)
    gpu_group_by_value = {",".join(str(item) for item in group): index for index, group in enumerate(gpu_groups)}
    active_gpu_loads = [0] * len(gpu_groups)
    assigned_gpu_loads = [0] * len(gpu_groups)
    assigned_group_by_key = {}
    for key, previous in refreshed.items():
        assigned = ",".join(part.strip() for part in str(previous.get("gpus") or "").split(",") if part.strip())
        if not assigned:
            continue
        group_index = gpu_group_by_value.get(assigned)
        if group_index is None:
            raise ValueError(f"Frozen GPUs are not one configured GPU group for {key[0]} / {key[1]}: {assigned}")
        assigned_group_by_key[key] = group_index
        if previous.get("status") in active_statuses:
            active_gpu_loads[group_index] += 1
            assigned_gpu_loads[group_index] += 1
    rows = []
    for run in runs:
        key = managed_run_key(run)
        run_id = str(run["run_id"])
        script = Path(str(run["script"]))
        previous = refreshed.get(key, {})
        group_index = assigned_group_by_key.get(key)
        if group_index is None and gpu_groups:
            group_index = min(range(len(gpu_groups)), key=lambda index: (assigned_gpu_loads[index], index))
            assigned_gpu_loads[group_index] += 1
            gpus = list(gpu_groups[group_index])
            assigned_group_by_key[key] = group_index
            if previous.get("status") in active_statuses:
                active_gpu_loads[group_index] += 1
        else:
            gpus = list(gpu_groups[group_index]) if group_index is not None else []
        semantic_run_dir = Path(str(run.get("run_dir") or script.parent))
        log_path = semantic_run_dir / "stdout.log"
        pid_path = semantic_run_dir / "pid"
        command = _launch_command(execution, script, log_path, pid_path, gpus)
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
            "target": execution.get("target", "local"),
            "host": execution.get("host", ""),
            "workdir": execution.get("workdir") or str(REPO_ROOT),
            "gpus": ",".join(str(item) for item in gpus),
            "log_path": str(log_path),
            "pid_path": str(pid_path),
            "command": command,
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
    run_output_paths = [Path(str(row[field])) for row in rows for field in ("log_path", "pid_path")]
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
    if not dry_run:
        launchable = [(index, row) for index, row in enumerate(rows) if row["status"] in {"planned", "pending"}]
        while launchable and slots > 0:
            eligible = []
            for index, row in launchable:
                group_index = assigned_group_by_key.get(managed_run_key(row))
                if group_index is not None:
                    if not allow_gpu_oversubscription and active_gpu_loads[group_index] >= 1:
                        continue
                    load = active_gpu_loads[group_index]
                else:
                    load = 0
                eligible.append((load, index, row, group_index))
            if not eligible:
                break
            _load, index, row, group_index = min(eligible, key=lambda item: (item[0], item[1]))
            launchable = [
                (candidate_index, candidate) for candidate_index, candidate in launchable if candidate_index != index
            ]
            row["status"] = _start_process(execution, row["command"])
            row["launched_at"] = utc_now() if row["status"] == "launched" else ""
            if row["status"] == "launched":
                started_keys.add(managed_run_key(row))
                if group_index is not None:
                    active_gpu_loads[group_index] += 1
                slots -= 1
        for _index, row in launchable:
            if row["status"] == "planned":
                row["status"] = "pending"
    committed = merge_run_manifest(workspace, rows, lock_held=manifest_lock_held)
    committed_by_key = {managed_run_key(row): row for row in committed}
    rows = [committed_by_key[managed_run_key(run)] for run in runs]
    write_rows(manifest, rows)
    write_rows(status_path, rows)
    for row in rows:
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
