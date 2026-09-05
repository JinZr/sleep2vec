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
    plan_hparam,
    run_artifacts as artifacts,
    run_evidence as evidence,
)
from .experiment_workspace import (
    EXECUTION_IDENTITY_FIELDS,
    LAUNCHABLE_STATUSES,
    PROCESS_IDENTITY_FIELDS,
    TERMINAL_STATUSES,
    append_event,
    experiment_root,
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
from .manifests import utc_now, write_rows

LAUNCH_TIMEOUT_SECONDS = scheduler.LAUNCH_TIMEOUT_SECONDS
EXECUTION_SNAPSHOT_NAME = scheduler.EXECUTION_SNAPSHOT_NAME


def launch_hparam_runs(
    plan_dir: str | Path,
    *,
    dry_run: bool = True,
    fail_on_missing_pid_blocker: bool = False,
) -> Path:
    """Attempt one scheduling pass for a registered hparam plan and return launch_manifest.tsv.

    Validates frozen inputs and serializes the pass with the workspace run lock.
    dry_run=True previews scheduling without starting processes or submitting
    jobs, but still writes launch/status projections and may merge manifest
    rows; it is not a read-only operation. Execution can observe local/remote
    runs, launch eligible work within capacity and record lifecycle evidence.

    fail_on_missing_pid_blocker makes an unresolved process-identity blocker an
    execution error for queue callers. Handled observation or launch failures can
    be recorded as statuses such as unknown_scheduler or launch_failed and return
    normally. Inspect canonical run status: the returned path does not imply a
    successful launch. Unhandled exceptions propagate; already recorded launches
    are not rolled back on later failure."""
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
    """Advance a registered hparam queue and return its status-table path.

    With dry_run=True, performs one launch preview and returns launch_manifest.tsv,
    including that preview's projection writes. With dry_run=False, repeatedly
    monitors and launches eligible runs, writing canonical state and projections,
    until every run in this plan is terminal; returns run_status.tsv then.
    Terminal includes failed/stopped runs and does not imply selection succeeded.

    poll_seconds must be finite and positive. Missing process identity and
    unresolved scheduler/launch outcomes can raise instead of advancing the
    queue; other validation and runtime errors propagate. This function launches
    work only in execution mode and does not select or finalize the experiment."""
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
        unresolved_scheduler = sorted(
            key
            for key in expected_keys
            if scheduler_type(rows_by_key[key]) == "slurm"
            and rows_by_key[key].get("status") in {"submitting", "unknown_scheduler"}
        )
        if unresolved_scheduler:
            step_id, run_id = unresolved_scheduler[0]
            raise RuntimeError(
                f"Hparam queue cannot advance because {step_id} / {run_id} has unresolved Slurm state "
                f"{rows_by_key[unresolved_scheduler[0]].get('status')}."
            )
        missing_pid = sorted(key for key in expected_keys if rows_by_key[key].get("status") == "missing_pid")
        if missing_pid:
            step_id, run_id = missing_pid[0]
            raise RuntimeError(f"Hparam queue cannot advance because {step_id} / {run_id} has status missing_pid.")
        unbound_unknown_remote = sorted(
            key
            for key in expected_keys
            if rows_by_key[key].get("status") == "unknown_remote"
            and any(rows_by_key[key].get(field) in (None, "") for field in PROCESS_IDENTITY_FIELDS)
        )
        if unbound_unknown_remote:
            step_id, run_id = unbound_unknown_remote[0]
            raise RuntimeError(
                f"Hparam queue cannot advance because {step_id} / {run_id} has status "
                "unknown_remote without complete process identity; launch outcome is uncertain."
            )

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
        event = {"step_id": key[0], "run_id": key[1], "gpus": canonical_by_key[key].get("gpus", "")}
        if scheduler_type(canonical_by_key[key]) == "slurm":
            event["scheduler_job_id"] = canonical_by_key[key]["scheduler_job_id"]
        try:
            append_event(
                workspace,
                "run_launched",
                event,
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
    canonical_by_key = {managed_run_key(row): row for row in read_run_manifest(workspace)}
    launchable_runs = [
        run
        for run in plan["runs"]
        if canonical_by_key[managed_run_key(run)].get("status") in scheduler.LAUNCHABLE_STATUSES
    ]
    if launchable_runs:
        # Observation and capacity can only narrow this set; dry-run must validate the same prospective candidates.
        plan_hparam.validate_hparam_run_configs(
            recipe,
            [(run, Path(str(run["config"])).read_bytes()) for run in launchable_runs],
        )
        if not dry_run:
            plan_hparam.validate_hparam_output_paths(run_dir, plan, runs=launchable_runs)

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


def monitor_hparam_runs(
    run_dir: str | Path,
    *,
    once: bool = True,
    health: bool = False,
    poll_seconds: float = 60,
) -> Path:
    """Observe a registered hparam plan, commit status updates and return run_status.tsv.

    Reads local/remote execution evidence and merges observations into the
    canonical run manifest, refreshing projections/reports and recording status
    transitions. Never starts pending runs. health requests additional health
    evidence from the observers.

    The Python default once=True performs one round; once=False polls until all
    plan runs are terminal, including failures/stops. poll_seconds must be finite
    and positive even for one round. Missing canonical runs, invalid frozen
    inputs and unhandled observation/I/O errors raise; earlier writes can remain."""
    if not math.isfinite(poll_seconds) or poll_seconds <= 0:
        raise ValueError("poll_seconds must be positive.")
    root = Path(run_dir)
    plan = artifacts.read_hparam_plan(root)
    recipe = plan.get("recipe") if isinstance(plan.get("recipe"), dict) else {}
    expected_keys = {managed_run_key(run) for run in plan["runs"]}
    status_path = root / "run_status.tsv"
    workspace = experiment_root(recipe)
    if workspace is None:
        raise ValueError("Hparam plan is not bound to an experiment workspace.")
    managed_output_paths = [
        workspace / "run_manifest.tsv",
        workspace / "run_matrix.csv",
        workspace / "reports" / "run_matrix.md",
        workspace / "events.jsonl",
        workspace / "reports" / "status.md",
        root / "launch_manifest.tsv",
        root / "run_status.tsv",
    ]
    exp_io.validate_managed_output_paths(workspace, managed_output_paths)
    while True:
        workspace_rows = read_run_manifest(workspace)
        workspace_by_key = {managed_run_key(row): row for row in workspace_rows}
        missing = expected_keys - set(workspace_by_key)
        if missing:
            step_id, run_id = sorted(missing)[0]
            raise ValueError(f"Canonical run is missing for the current hparam plan: {step_id} / {run_id}")
        previous_rows = {key: workspace_by_key[key] for key in expected_keys}
        # Anchor CLI-relative owners without resolving aliases or removing raw '..' components.
        monitor_context = scheduler.SlurmMonitorContext(previous_rows.values(), owner_dir=root.absolute())
        rows = []
        for run in plan["runs"]:
            key = managed_run_key(run)
            prior = previous_rows[key]
            if prior.get("target") in (None, ""):
                rows.append(prior)
                continue
            if scheduler_type(prior) == "slurm":
                execution = {"target": prior["target"]}
                if prior["target"] == "ssh":
                    execution["host"] = prior["host"]
                if scheduler_direct_controller(prior):
                    execution["scheduler"] = {"direct_controller": True}
                rows.append(
                    scheduler.observe_slurm_run(
                        monitor_context.owner_dir, execution, prior, health=health, monitor_context=monitor_context
                    )
                )
            else:
                rows.append(
                    scheduler.observe_run(
                        root,
                        prior,
                        prior,
                        health=health,
                        default_script_commits_terminal_status=False,
                    )
                )
        exp_io.validate_managed_output_paths(workspace, managed_output_paths)
        committed = merge_run_manifest(workspace, rows)
        committed_by_key = {managed_run_key(row): row for row in committed}
        rows = [committed_by_key[managed_run_key(run)] for run in plan["runs"]]
        write_rows(status_path, rows)
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
                # A stale pre-cancel observation may merge to stopped without confirming a scheduler stop.
                if (
                    after == "stopped"
                    and scheduler_type(row) == "slurm"
                    and previous_rows[managed_run_key(row)].get("stop_requested_at") not in (None, "")
                ):
                    append_event(
                        workspace,
                        "run_stopped",
                        {
                            "step_id": row["step_id"],
                            "run_id": row["run_id"],
                            "reason": row.get("stop_reason", ""),
                        },
                    )
        write_status_report(workspace)
        if once or all(row.get("status") in TERMINAL_STATUSES for row in rows):
            return status_path
        print(f"wrote {status_path}")
        time.sleep(poll_seconds)


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
        backend = scheduler_type(previous)
        if backend == "slurm":
            committed, metadata_stop = scheduler.stop_slurm_run_locked(
                workspace,
                workspace_rows,
                key,
                reason=reason,
                hooks=scheduler.SchedulerHooks(
                    merge_manifest=merge_run_manifest,
                    append_event=append_event,
                ),
                now=utc_now,
            )
        else:
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
                        "stopped_at": utc_now(),
                        "stop_reason": reason,
                    },
                )
                committed = merge_run_manifest(workspace, [final], lock_held=True)
            else:
                missing_execution_identity = {
                    field for field in EXECUTION_IDENTITY_FIELDS - PROCESS_IDENTITY_FIELDS if field not in previous
                }
                if previous.get("target") in (None, ""):
                    missing_execution_identity.add("target")
                if missing_execution_identity:
                    raise ValueError(
                        f"Canonical run is missing execution identity for {run_id}: "
                        f"{', '.join(sorted(missing_execution_identity))}"
                    )

                target = previous.get("target")
                host = previous.get("host")
                if target not in {"local", "ssh"}:
                    raise ValueError(f"Canonical run target must be local or ssh for run_id: {run_id}")
                if target == "ssh" and (not isinstance(host, str) or not host.strip()):
                    raise ValueError(f"Canonical SSH run requires a non-empty host for run_id: {run_id}")
                populated_process_fields = {
                    field for field in PROCESS_IDENTITY_FIELDS if previous.get(field) not in (None, "")
                }
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
                        raise RuntimeError(
                            f"Recorded process identity differs from canonical {field} for run_id: {run_id}"
                        )
                evidence.stop_process_group(previous, process_identity)
                final = merge_run_row(
                    previous,
                    {
                        "step_id": key[0],
                        "run_id": key[1],
                        **process_identity,
                        "status": "stopped",
                        "stopped_at": utc_now(),
                        "stop_reason": reason,
                    },
                )
                committed = merge_run_manifest(workspace, [final], lock_held=True)
        committed_by_key = {managed_run_key(item): item for item in committed}
        final_status_rows = [committed_by_key[managed_run_key(run)] for run in plan["runs"]]
        write_rows(status_path, final_status_rows)
        write_rows(manifest_path, final_status_rows)
        if backend != "slurm" or metadata_stop:
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
    planned_command: str | None = None,
    run_id: str = "",
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
        planned_command=planned_command,
        run_id=run_id,
    )


def _parent_path(path: str | Path) -> str:
    return scheduler._parent_path(path)


def _start_process(execution: dict[str, Any], command: str) -> str:
    return scheduler.start_process(execution, command)
