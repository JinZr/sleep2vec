from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
import time
from typing import Any

from . import (
    experiment_io as exp_io,
    managed_scheduler as scheduler,
    run_artifacts as artifacts,
    run_evidence as evidence,
)
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
from .manifests import utc_now, write_rows

LAUNCH_TIMEOUT_SECONDS = scheduler.LAUNCH_TIMEOUT_SECONDS
EXECUTION_SNAPSHOT_NAME = scheduler.EXECUTION_SNAPSHOT_NAME


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
    with scheduler.managed_run_lock(workspace):
        return _launch_hparam_runs(
            run_dir,
            dry_run=dry_run,
            manifest_lock_held=True,
            fail_on_missing_pid_blocker=fail_on_missing_pid_blocker,
        )


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
    manifest = run_dir / "launch_manifest.tsv"
    status_path = run_dir / "run_status.tsv"
    exp_io.validate_managed_output_paths(
        workspace,
        [
            workspace / "run_manifest.tsv",
            workspace / "run_matrix.csv",
            workspace / "reports" / "run_matrix.md",
            workspace / "events.jsonl",
            workspace / "reports" / "status.md",
            manifest,
            status_path,
            run_dir / EXECUTION_SNAPSHOT_NAME,
        ],
    )
    execution = recipe.get("execution") if isinstance(recipe.get("execution"), dict) else {}
    runtime = recipe.get("runtime") if isinstance(recipe.get("runtime"), dict) else {}

    def write_projections(result: scheduler.LaunchResult) -> None:
        write_rows(manifest, result.launch_rows)
        write_rows(status_path, result.committed_rows)

    hooks = scheduler.SchedulerHooks(
        merge_manifest=merge_run_manifest,
        append_event=append_event,
        write_status_report=write_status_report,
        validate_run_update=validate_frozen_run_update,
        validated_snapshot=_validated_execution_snapshot,
        build_command=_launch_command,
        start_process=_start_process,
    )
    scheduler.launch_managed_runs(
        workspace,
        run_dir,
        plan["runs"],
        execution,
        runtime,
        dry_run=dry_run,
        fail_on_missing_pid_blocker=fail_on_missing_pid_blocker,
        default_script_commits_terminal_status=False,
        projection_writer=write_projections,
        hooks=hooks,
        lock_held=manifest_lock_held,
    )
    return manifest


def _validated_execution_snapshot(
    run_dir: Path,
    execution: dict[str, Any],
    runs: list[dict[str, Any]],
    workspace_by_key: dict[tuple[str, str], dict[str, Any]],
) -> tuple[dict[str, Any], bool]:
    return scheduler.validated_execution_snapshot(
        run_dir,
        execution,
        runs,
        workspace_by_key,
        inspector=_inspect_execution_target,
        plan_label="hparam",
    )


def _inspect_execution_target(execution: dict[str, Any], runs: list[dict[str, Any]]) -> dict[str, Any]:
    return scheduler.inspect_execution_target(
        execution,
        runs,
        command_runner=_run_execution_command,
        plan_label="hparam",
    )


def _run_execution_command(execution: dict[str, Any], command: list[str]) -> subprocess.CompletedProcess:
    return scheduler.run_execution_command(execution, command)


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
        rows.append(
            scheduler.observe_run(
                root,
                prior,
                prior,
                health=health,
                default_script_commits_terminal_status=False,
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
    with scheduler.managed_run_lock(workspace):
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
        committed = merge_run_manifest(workspace, [final], lock_held=True)
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
    return scheduler.gpu_groups(execution, runtime)


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
    return scheduler.build_launch_command(
        execution,
        script,
        log_path,
        pid_path,
        gpus,
        execution_snapshot=execution_snapshot,
        config_path=config_path,
        script_sha256=script_sha256,
        config_sha256=config_sha256,
    )


def _parent_path(path: str | Path) -> str:
    return scheduler._parent_path(path)


def _start_process(execution: dict[str, Any], command: str) -> str:
    return scheduler.start_process(execution, command)
