"""Launch, queue, and stop for a registered hyper-parameter plan's runs.

Dispatches through ``managed_scheduler`` for capacity and process lifecycle. A
direct plan's queue runs as one detached runner per plan, guarded by the plan's
``queue_runner.lock``; a Slurm plan's queue is one submit pass, since Slurm
itself queues the jobs.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import dataclasses
import functools
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import cast

from . import (
    execution_snapshot,
    experiment_io as exp_io,
    experiment_workspace,
    managed_scheduler as scheduler,
    plan_contract,
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
    file_sha256,
    has_managed_launch_evidence,
    merge_run_manifest,
    merge_run_row,
    read_run_manifest,
    scheduler_type,
    validate_frozen_run_update,
    validated_run_key,
)
from .manifests import read_json, utc_now
from .models import JsonValue

LAUNCH_TIMEOUT_SECONDS = scheduler.LAUNCH_TIMEOUT_SECONDS


@dataclasses.dataclass(frozen=True)
class QueueRunnerStart:
    pid: int
    log_path: Path


def read_hparam_plan_under_run_lock(run_dir: Path) -> plan_contract.HparamPlan:
    """Return the registered hparam plan, reading workspace state under the workspace run lock.

    plan.json first only locates the workspace. Launches, stops and run scripts
    replace run_manifest.tsv under the run lock, so the registered read holds it.
    The caller must not already hold that lock: it is not reentrant.
    """
    with experiment_workspace.managed_run_lock(artifacts.hparam_plan_workspace(run_dir)):
        return artifacts.read_hparam_plan(run_dir)


def launch_hparam_runs(
    plan_dir: str | Path,
    *,
    dry_run: bool = True,
    fail_on_missing_pid_blocker: bool = False,
    hooks: scheduler.SchedulerHooks | None = None,
) -> scheduler.LaunchResult:
    """Attempt one scheduling pass for a registered hparam plan and return its LaunchResult.

    Reads the registered plan and serializes the pass with the workspace run lock;
    frozen configs, scripts and inputs are verified by the launch command itself.
    dry_run=True previews scheduling without starting processes or submitting
    jobs: its launch_rows carry each untargeted launchable run's prospective GPU
    group and command, which are never committed. It still re-commits the plan's
    canonical rows and creates run directories, so it is not a read-only
    operation. Execution can observe local
    runs, launch eligible work within capacity and record lifecycle evidence;
    for a Slurm plan it is one submit pass that submits or adopts every
    launchable run.

    fail_on_missing_pid_blocker makes an unresolved process-identity blocker an
    execution error for queue callers. Handled observation or launch failures can
    be recorded as statuses such as lost or launch_failed and return
    normally. Inspect canonical run status: a returned result does not imply a
    successful launch. Unhandled exceptions propagate; already recorded launches
    are not rolled back on later failure.

    hooks lets Python callers such as tests replace the process effects. Only
    three fields are used: ``validated_snapshot`` replaces the frozen execution
    snapshot (read from execution_snapshot.json for a direct plan, re-probed at
    Slurm submission), ``build_command`` the launch-command builder and
    ``start_process`` the process start; a ``None`` field uses the canonical
    owner, looked up when the pass starts. Manifest merges, event appends
    and frozen-run validation are not replaceable."""
    run_dir = Path(plan_dir).expanduser()
    if not run_dir.is_absolute():
        run_dir = run_dir.resolve()
    # plan.json only locates the workspace to lock; a concurrent lock holder may replace run_manifest.tsv
    # and events.jsonl, so _launch_hparam_runs reads the registered plan under the lock.
    with experiment_workspace.managed_run_lock(artifacts.hparam_plan_workspace(run_dir)):
        return _launch_hparam_runs(
            run_dir,
            dry_run=dry_run,
            fail_on_missing_pid_blocker=fail_on_missing_pid_blocker,
            hooks=hooks,
        )


def start_hparam_queue_runner(plan_dir: str | Path, *, poll_seconds: float = 60) -> QueueRunnerStart | None:
    """Start one detached queue runner for a direct hparam plan and return its PID and log path.

    Validates the plan under the workspace run lock, then fails without blocking if
    another runner holds the plan's queue_runner.lock. The runner is
    ``hparam-run-queue --execute --foreground`` in a new session, with stdin from
    /dev/null and stdout/stderr appended to queue_runner.log, so it outlives the
    caller's terminal or SSH session; the runner takes the lock itself. Returns
    None without starting anything for a Slurm plan, whose execution is one
    submit pass in the caller's process."""
    if not math.isfinite(poll_seconds) or poll_seconds <= 0:
        raise ValueError("poll_seconds must be positive.")
    run_dir = Path(plan_dir).expanduser()
    if not run_dir.is_absolute():
        run_dir = run_dir.resolve()
    plan = read_hparam_plan_under_run_lock(run_dir)
    if not _is_direct_plan(plan):
        return None
    workspace = experiment_root(plan["recipe"])
    assert workspace is not None  # read_hparam_plan rejects a plan without experiment.root.
    log_path = run_dir / scheduler.QUEUE_RUNNER_LOG_NAME
    exp_io.validate_managed_output_paths(workspace, [run_dir / scheduler.QUEUE_RUNNER_LOCK_NAME, log_path])
    with scheduler.queue_runner_lock(run_dir, record_pid=False):
        pass
    argv = [
        sys.executable,
        "-m",
        "agent_tools",
        "hparam-run-queue",
        "--plan-dir",
        str(run_dir),
        "--execute",
        "--foreground",
        "--poll-seconds",
        str(poll_seconds),
    ]
    log_descriptor = os.open(log_path, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW, 0o644)
    try:
        os.write(log_descriptor, f"{utc_now()} starting queue runner for {run_dir}\n".encode())
        process = subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=log_descriptor,
            stderr=log_descriptor,
            start_new_session=True,
        )
    finally:
        os.close(log_descriptor)
    return QueueRunnerStart(pid=process.pid, log_path=log_path)


def run_hparam_queue(
    plan_dir: str | Path,
    *,
    dry_run: bool = True,
    poll_seconds: float = 60,
) -> Sequence[Mapping[str, JsonValue]]:
    """Advance a registered hparam queue and return the plan's run rows.

    With dry_run=True, performs one launch preview and returns its launch_rows:
    the canonical rows with each untargeted launchable run's prospective GPU
    group and command. With dry_run=False, a direct plan repeatedly runs
    launch passes in the calling process, each of which observes the plan's runs
    and launches eligible ones, until every run in this plan is terminal.
    Terminal includes failed/stopped runs and does not imply selection succeeded.
    A direct plan's loop holds the plan's queue_runner.lock (recording this PID)
    and fails at once if another runner holds it. A Slurm plan makes one submit
    pass, which submits or adopts every launchable run, and returns without
    polling; Slurm queues the jobs. The returned rows are read from
    run_manifest.tsv in plan order once the pass or loop ends.

    poll_seconds must be finite and positive. Missing process identity can raise
    instead of advancing the queue; other validation and runtime errors propagate.
    This function launches work only in execution mode and does not select or
    finalize the experiment."""
    if not math.isfinite(poll_seconds) or poll_seconds <= 0:
        raise ValueError("poll_seconds must be positive.")
    run_dir = Path(plan_dir).expanduser()
    if not run_dir.is_absolute():
        run_dir = run_dir.resolve()
    if dry_run:
        return launch_hparam_runs(run_dir, dry_run=True).launch_rows
    # Launched run scripts and concurrent monitors replace run_manifest.tsv under the run lock, so every
    # workspace-state read below holds it. Each launch pass rereads only the registered plan, and the
    # scheduler checks every run it starts against its canonical row.
    plan = read_hparam_plan_under_run_lock(run_dir)
    workspace = experiment_root(plan["recipe"])
    assert workspace is not None  # The registered read rejects a plan without experiment.root.
    if not _is_direct_plan(plan):
        launch_hparam_runs(run_dir, dry_run=False)
        return _canonical_plan_rows(workspace, plan)
    # Taken outside the run lock and held for the whole loop, which takes the run lock repeatedly.
    exp_io.validate_managed_output_paths(workspace, [run_dir / scheduler.QUEUE_RUNNER_LOCK_NAME])
    with scheduler.queue_runner_lock(run_dir, record_pid=True):
        return _drain_hparam_queue(run_dir, workspace, plan, poll_seconds)


def _is_direct_plan(plan: plan_contract.HparamPlan) -> bool:
    return {scheduler_type(run) for run in plan["runs"]} == {"direct"}


def _canonical_plan_rows(workspace: Path, plan: plan_contract.HparamPlan) -> list[dict[str, str]]:
    with experiment_workspace.managed_run_lock(workspace):
        rows_by_key = {validated_run_key(row): row for row in read_run_manifest(workspace)}
    return [rows_by_key[validated_run_key(run)] for run in plan["runs"]]


def _queue_is_terminal(rows: Sequence[Mapping[str, str]]) -> bool:
    """Return whether every plan run is terminal; a missing_pid run blocks the queue instead of waiting."""
    if all(row.get("status") in TERMINAL_STATUSES for row in rows):
        return True
    missing_pid = sorted(validated_run_key(row) for row in rows if row.get("status") == "missing_pid")
    if missing_pid:
        step_id, run_id = missing_pid[0]
        raise RuntimeError(f"Hparam queue cannot advance because {step_id} / {run_id} has status missing_pid.")
    return False


def _drain_hparam_queue(
    run_dir: Path,
    workspace: Path,
    plan: plan_contract.HparamPlan,
    poll_seconds: float,
) -> list[dict[str, str]]:
    # Each launch pass observes the plan's runs under the run lock before it fills capacity.
    while True:
        rows = _canonical_plan_rows(workspace, plan)
        if _queue_is_terminal(rows):
            return rows
        launch_hparam_runs(run_dir, dry_run=False, fail_on_missing_pid_blocker=True)
        rows = _canonical_plan_rows(workspace, plan)
        if _queue_is_terminal(rows):
            return rows
        time.sleep(poll_seconds)


def _launch_hparam_runs(
    run_dir: Path,
    *,
    dry_run: bool,
    fail_on_missing_pid_blocker: bool,
    hooks: scheduler.SchedulerHooks | None,
) -> scheduler.LaunchResult:
    plan = artifacts.read_hparam_plan(run_dir)
    recipe = plan["recipe"]
    workspace = experiment_root(recipe)
    assert workspace is not None  # The registered read rejects a plan without experiment.root.
    execution_value = recipe.get("execution")
    execution = execution_value if isinstance(execution_value, dict) else {}
    runtime_value = recipe.get("runtime")
    runtime = runtime_value if isinstance(runtime_value, dict) else {}
    # Every launch check compares the live target against this file, so it must be the one registered with the plan.
    snapshot_path = run_dir / execution_snapshot.EXECUTION_SNAPSHOT_NAME
    if file_sha256(snapshot_path) != plan["execution_snapshot"]["sha256"]:
        raise ValueError(f"Hparam execution snapshot changed after registration: {snapshot_path}")
    validated_snapshot: Callable[..., execution_snapshot.ExecutionSnapshotResult]
    if _is_direct_plan(plan):

        def validated_snapshot(*_scheduler_state: object) -> execution_snapshot.ExecutionSnapshotResult:
            # The launch command's runtime_identity and cli_preflight programs check this frozen snapshot against
            # the live target, and the frozen config and launch.sh hashes, right before process start; a mismatch is
            # a definite launch_failed. launch.sh then checks its frozen input snapshots, such as an init checkpoint.
            return cast(execution_snapshot.ExecutionSnapshot, read_json(snapshot_path)), False

    else:
        validated_snapshot = functools.partial(
            execution_snapshot.validated_execution_snapshot,
            inspector=_inspect_execution_target,
            plan_label="hparam",
        )

    # Built per pass so a patched public owner takes effect; hooks replace only the process effects.
    launch_hooks = scheduler.SchedulerHooks(
        merge_manifest=merge_run_manifest,
        append_event=append_event,
        validate_run_update=validate_frozen_run_update,
        validated_snapshot=validated_snapshot,
        build_command=scheduler.build_launch_command,
        start_process=scheduler.start_process,
    )
    if hooks is not None:
        launch_hooks = dataclasses.replace(
            launch_hooks,
            validated_snapshot=hooks.validated_snapshot or launch_hooks.validated_snapshot,
            build_command=hooks.build_command or launch_hooks.build_command,
            start_process=hooks.start_process or launch_hooks.start_process,
        )
    return scheduler.launch_managed_runs(
        workspace,
        run_dir,
        plan["runs"],
        execution,
        runtime,
        dry_run=dry_run,
        fail_on_missing_pid_blocker=fail_on_missing_pid_blocker,
        hooks=launch_hooks,
        lock_held=True,
    )


def _inspect_execution_target(
    execution: Mapping[str, JsonValue], runs: Sequence[Mapping[str, JsonValue]]
) -> execution_snapshot.ExecutionSnapshot:
    return execution_snapshot.inspect_execution_target(execution, runs, plan_label="hparam")


def stop_hparam_run(run_dir: str | Path, run_id: str, *, reason: str) -> Path:
    if not reason.strip():
        raise ValueError("Stopping a run requires a non-empty reason.")
    root = Path(run_dir)
    # plan.json only locates the workspace to lock; a concurrent lock holder may replace run_manifest.tsv
    # and events.jsonl, so the registered plan is read under the lock.
    workspace = artifacts.hparam_plan_workspace(root)
    with experiment_workspace.managed_run_lock(workspace):
        plan = artifacts.read_hparam_plan(root)
        expected_keys = {validated_run_key(run) for run in plan["runs"]}
        workspace_rows = read_run_manifest(workspace)
        workspace_by_key = {validated_run_key(item): item for item in workspace_rows}
        missing = expected_keys - set(workspace_by_key)
        if missing:
            step_id, missing_run_id = sorted(missing)[0]
            raise ValueError(f"Canonical run is missing for the current hparam plan: {step_id} / {missing_run_id}")
        matched = [run for run in plan["runs"] if run.get("run_id") == run_id]
        if not matched:
            raise ValueError(f"Unknown run_id: {run_id}")
        if len(matched) > 1:
            raise ValueError(f"Ambiguous run_id in hparam plan: {run_id}")
        key = validated_run_key(matched[0])
        previous = workspace_by_key[key]
        backend = scheduler_type(previous)
        if backend == "slurm":
            _committed, metadata_stop = scheduler.stop_slurm_run_locked(
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
                merge_run_manifest(workspace, [final], lock_held=True)
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

                populated_process_fields = {
                    field for field in PROCESS_IDENTITY_FIELDS if previous.get(field) not in (None, "")
                }
                if populated_process_fields and populated_process_fields != PROCESS_IDENTITY_FIELDS:
                    missing_process_fields = ", ".join(sorted(PROCESS_IDENTITY_FIELDS - populated_process_fields))
                    raise ValueError(
                        f"Canonical run has partial process identity for {run_id}; missing: {missing_process_fields}"
                    )
                exp_io.validate_managed_output_paths(workspace, [previous["pid_path"]])
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
                merge_run_manifest(workspace, [final], lock_held=True)
        if backend != "slurm" or metadata_stop:
            append_event(
                workspace,
                "run_stopped",
                {"step_id": key[0], "run_id": run_id, "reason": reason},
            )
    return workspace / "run_manifest.tsv"
