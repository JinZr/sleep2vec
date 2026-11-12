"""Facade for the experiment lifecycle commands the CLI drives.

Domain-free facade: init, note, finalize, monitor, status, and the preset/infer
launch and stop paths.

A facade, not a second implementation: lifecycle rules live in
``experiment_workspace`` and evidence in ``experiment_tracking``. What stays
here is the sequencing a command performs across those owners -- most visibly
in ``finalize_experiment``.
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import nullcontext
from datetime import datetime, timezone
import functools
import hashlib
from pathlib import Path
from typing import Any, Literal, TypedDict

import yaml

from . import (
    execution_snapshot,
    experiment_io as exp_io,
    experiment_tracking as tracking,
    experiment_workspace,
    managed_scheduler,
    run_artifacts as artifacts,
    run_evidence as evidence,
)
from .experiment_workspace import (
    EXECUTION_IDENTITY_FIELDS,
    FROZEN_RUN_FIELDS,
    MONITOR_OBSERVATION_FIELDS,
    PROCESS_IDENTITY_FIELDS,
    SHA256_RE,
    TERMINAL_STATUSES,
    append_event,
    append_research_log,
    canonical_local_experiment_root,
    experiment_metadata_issues,
    experiment_readme_text,
    experiment_root,
    has_managed_launch_evidence,
    managed_run_key,
    managed_run_parameters,
    merge_run_manifest,
    read_managed_yaml_mapping,
    read_registered_steps,
    read_run_manifest,
    scheduler_type,
    stopped_runs_without_reason,
    validate_existing_experiment_manifest,
    validate_frozen_run_update,
    validated_run_key,
    write_initial_experiment_manifest,
)
from .manifests import read_json, utc_now


class ExperimentNoteResult(TypedDict):
    path: str
    entry_id: str
    appended: bool


class ExperimentCompletionFields(TypedDict):
    status: Literal["completed"]
    completed_at: str
    final_report: str
    final_report_sha256: str


def _preset_direct_workspace(plan_dir: Path) -> Path:
    # The locator is untrusted and only names the workspace to lock; run_manifest.tsv may be replaced by a concurrent
    # lock holder, so callers read the plan and workspace state through _read_preset_direct_plan under the run lock.
    initial = read_json(plan_dir / "plan.json")
    recipe = initial.get("recipe")
    if not isinstance(recipe, dict) or recipe.get("task") != "preset_prepare":
        raise ValueError("preset-launch and run-stop require a preset preparation plan.")
    issues = experiment_metadata_issues(recipe)
    if issues:
        raise ValueError("Invalid preset workspace binding: " + "; ".join(issue["message"] for issue in issues))
    workspace = experiment_root(recipe)
    assert workspace is not None
    return workspace


def _read_target_plan(plan_dir: Path, workspace: Path) -> tuple[artifacts.RegisteredPlanSummary, list[dict[str, str]]]:
    """Read only the registered plan a launch or stop names, under the caller's run lock."""
    experiment, rows = _managed_workspace(workspace, lock_held=True)
    for manifest in read_registered_steps(workspace, experiment_id=str(experiment["id"])):
        if str(plan_dir) in manifest["plans"]:
            plan = artifacts.read_registered_plan(
                plan_dir, workspace_experiment=experiment, step_manifest=manifest, workspace_rows=rows
            )
            return plan, rows
    raise ValueError(f"Plan is not registered in its experiment workspace: {plan_dir}")


def _read_preset_direct_plan(
    plan_dir: Path, workspace: Path
) -> tuple[artifacts.RegisteredPlanSummary, list[dict[str, Any]]]:
    # All execution fields below come from the registered frozen plan, not the untrusted locator.
    plan, rows = _read_target_plan(plan_dir, workspace)
    execution = plan["recipe"].get("execution") or {}
    if (
        plan["task"] != "preset_prepare"
        or len(plan["runs"]) != 1
        or execution.get("scheduler") != {"type": "direct"}
        or any(not execution.get(field) for field in ("python", "runtime_commit", "workdir"))
        or plan["runs"][0].get("terminal_status_owner") != "script"
    ):
        raise ValueError("preset-launch and run-stop require one new local managed direct preset run.")
    return plan, rows


def _record_definitely_unlaunched_failure(workspace: Path, key: tuple[str, str], reason: str) -> None:
    previous = {managed_run_key(row): row for row in read_run_manifest(workspace)}[key]
    # A lost receipt or post-submit failure must never become a definitely-unsubmitted failure.
    if previous["status"] in managed_scheduler.LAUNCHABLE_STATUSES and not has_managed_launch_evidence(previous):
        merge_run_manifest(
            workspace,
            [
                {
                    "step_id": key[0],
                    "run_id": key[1],
                    "status": "launch_failed",
                    "scheduler_reason": reason,
                }
            ],
            lock_held=True,
        )
        append_event(
            workspace,
            "run_status_changed",
            {"step_id": key[0], "run_id": key[1], "from": previous["status"], "to": "launch_failed"},
        )


def launch_preset_run(plan_dir: str | Path, *, dry_run: bool = True) -> managed_scheduler.LaunchResult:
    owner_dir = Path(plan_dir).absolute()
    workspace = _preset_direct_workspace(owner_dir)
    with experiment_workspace.managed_run_lock(workspace):
        plan, rows = _read_preset_direct_plan(owner_dir, workspace)
        run = plan["runs"][0]
        key = validated_run_key(run)
        previous = {managed_run_key(row): row for row in rows}[key]
        execution = plan["recipe"]["execution"]

        def preflight(*_scheduler_state: object) -> execution_snapshot.ExecutionSnapshotResult:
            # The scheduler passes the manifest state read under this same lock, which is previous for this run.
            # Script-based preset runs freeze no module execution snapshot. The launch command itself runs the
            # runtime_identity check (planned commit, clean runtime, frozen artifacts) right before process start,
            # and its failure is recorded as launch_failed.
            if previous["status"] not in managed_scheduler.LAUNCHABLE_STATUSES:
                return None, False
            # A preset spawns only after its launched claim persists, so identity the scheduler bound to a still
            # launchable row before that claim proves no process; any other launch evidence is refused.
            pre_claim_identity = dict.fromkeys(EXECUTION_IDENTITY_FIELDS - PROCESS_IDENTITY_FIELDS, "")
            if has_managed_launch_evidence({**previous, **pre_claim_identity}):
                raise ValueError("Preset run already has launch evidence; refusing another launch attempt.")
            if (Path(run["run_dir"]) / "pid").exists():
                raise ValueError("Preset PID receipt already exists; refusing another launch attempt.")
            return None, False

        def claim_and_start(_frozen_execution: object, command: str) -> tuple[str, str]:
            # Claim before fork: an interrupted manager must not turn an uncertain launch into a retry.
            merge_run_manifest(
                workspace,
                [{"step_id": key[0], "run_id": key[1], "status": "launched", "launched_at": utc_now()}],
                lock_held=True,
            )
            return managed_scheduler.start_process(execution, command)

        try:
            return managed_scheduler.launch_managed_runs(
                workspace,
                owner_dir,
                plan["runs"],
                execution,
                plan["recipe"].get("runtime", {}),
                dry_run=dry_run,
                hooks=managed_scheduler.SchedulerHooks(
                    validated_snapshot=preflight,
                    # Dry run previews the same artifact-verifying command that execute records and starts.
                    build_command=functools.partial(
                        managed_scheduler.build_launch_command,
                        config_path=Path(run["config"]),
                        script_sha256=run["script_sha256"],
                        config_sha256=run["config_sha256"],
                    ),
                    start_process=claim_and_start,
                ),
                lock_held=True,
            )
        except Exception as exc:
            # An unbound PID receipt may belong to an interrupted live launch, so it never becomes launch_failed.
            if not dry_run and not (Path(run["run_dir"]) / "pid").exists():
                _record_definitely_unlaunched_failure(workspace, key, f"Pre-launch guard failed: {exc}")
            raise


def stop_preset_run(plan_dir: str | Path, run_id: str, *, reason: str) -> Path:
    if not reason.strip():
        raise ValueError("Stopping a run requires a non-empty reason.")
    owner_dir = Path(plan_dir).absolute()
    workspace = _preset_direct_workspace(owner_dir)
    with experiment_workspace.managed_run_lock(workspace):
        plan, rows = _read_preset_direct_plan(owner_dir, workspace)
        run = plan["runs"][0]
        key = validated_run_key(run)
        if key[1] != run_id:
            raise ValueError(f"Unknown run_id: {run_id}")
        previous = {managed_run_key(row): row for row in rows}[key]
        if previous["status"] in TERMINAL_STATUSES:
            raise ValueError(f"Run is already terminal and cannot be stopped: {key[1]} ({previous['status']})")
        if previous["status"] in managed_scheduler.LAUNCHABLE_STATUSES and not has_managed_launch_evidence(previous):
            merge_run_manifest(
                workspace,
                [
                    {
                        "step_id": key[0],
                        "run_id": key[1],
                        "status": "stopped",
                        "stopped_at": utc_now(),
                        "stop_reason": reason,
                    }
                ],
                lock_held=True,
            )
            append_event(workspace, "run_stopped", {"step_id": key[0], "run_id": key[1], "reason": reason})
            return workspace / "run_manifest.tsv"
        if previous.get("pid_path") != str(Path(run["run_dir"]) / "pid"):
            raise ValueError("Preset run lacks its frozen local launch identity.")
        populated = {field for field in PROCESS_IDENTITY_FIELDS if previous.get(field) not in (None, "")}
        if populated and populated != PROCESS_IDENTITY_FIELDS:
            raise ValueError("Preset run has partial canonical process identity.")
        exp_io.validate_managed_output_paths(workspace, [Path(previous["pid_path"])])
        identity = evidence.read_process_identity(
            previous["pid_path"], previous, expected_script=run["script"] if not populated else None
        )
        if identity is None:
            raise ValueError("Preset run has no recorded process identity; refusing to signal.")
        if evidence.process_identity_running(previous, identity) is None:
            raise RuntimeError("Cannot verify preset process group before stop.")
        stopping = {
            **previous,
            **identity,
            "status": "stopping",
            "stop_requested_at": previous.get("stop_requested_at") or utc_now(),
            "stop_reason": previous.get("stop_reason") or reason,
        }
        merge_run_manifest(workspace, [stopping], lock_held=True)
    # The worker's EXIT trap takes this same manifest lock; never hold it while waiting for exit.
    running = evidence.process_identity_running(stopping, identity)
    if running is None:
        raise RuntimeError("Cannot verify preset process group after stop intent.")
    if running:
        try:
            evidence.stop_process_group(stopping, identity)
        except (RuntimeError, ProcessLookupError) as exc:
            # A short task can exit between the authenticated probe and signal. Uncertainty or reuse is not exit.
            if (
                isinstance(exc, evidence.ProcessIdentityError)
                or evidence.process_identity_running(stopping, identity) is not False
            ):
                raise
    with experiment_workspace.managed_run_lock(workspace):
        _experiment, rows = _managed_workspace(workspace, lock_held=True)
        latest = {managed_run_key(row): row for row in rows}[key]
        validate_frozen_run_update(latest, stopping)
        merge_run_manifest(
            workspace,
            [{**latest, "status": "stopped", "stopped_at": utc_now()}],
            lock_held=True,
        )
        append_event(workspace, "run_stopped", {"step_id": key[0], "run_id": key[1], "reason": stopping["stop_reason"]})
    return workspace / "run_manifest.tsv"


def _infer_slurm_workspace(plan_dir: Path) -> Path:
    # The initial document only locates the workspace to lock; run_manifest.tsv may be replaced by a concurrent lock
    # holder, so callers read the plan and workspace state through _read_infer_slurm_plan under the run lock.
    initial = read_json(plan_dir / "plan.json")
    recipe = initial.get("recipe")
    if not isinstance(recipe, dict) or recipe.get("task") != "infer":
        raise ValueError("infer-launch and run-stop require an ordinary infer plan.")
    issues = experiment_metadata_issues(recipe)
    if issues:
        raise ValueError("Invalid inference workspace binding: " + "; ".join(issue["message"] for issue in issues))
    workspace = experiment_root(recipe)
    assert workspace is not None
    return workspace


def _read_infer_slurm_plan(
    plan_dir: Path, workspace: Path
) -> tuple[artifacts.RegisteredPlanSummary, list[dict[str, str]]]:
    # Execution uses the registered plan read under the run lock, not the locating document.
    plan, rows = _read_target_plan(plan_dir, workspace)
    if plan["task"] != "infer" or len(plan["runs"]) != 1 or scheduler_type(plan["runs"][0]) != "slurm":
        raise ValueError("infer-launch and run-stop require one ordinary managed Slurm inference run.")
    return plan, rows


def launch_infer_run(plan_dir: str | Path, *, dry_run: bool = True) -> managed_scheduler.LaunchResult:
    """Make one Slurm submit pass for a registered inference plan's run under the workspace run lock.

    A pre-submission gate failure records launch_failed; once the run carries launch
    evidence, a failed squeue or sbatch raises and the next pass adopts the job by name."""
    owner_dir = Path(plan_dir).absolute()
    workspace = _infer_slurm_workspace(owner_dir)
    with experiment_workspace.managed_run_lock(workspace):
        plan, _rows = _read_infer_slurm_plan(owner_dir, workspace)
        recipe = plan["recipe"]
        try:
            return managed_scheduler.launch_managed_runs(
                workspace,
                owner_dir,
                plan["runs"],
                recipe["execution"],
                recipe.get("runtime", {}),
                dry_run=dry_run,
                lock_held=True,
            )
        except Exception as exc:
            if not dry_run:
                key = validated_run_key(plan["runs"][0])
                _record_definitely_unlaunched_failure(workspace, key, f"Pre-submission guard failed: {exc}")
            raise


def stop_infer_run(plan_dir: str | Path, run_id: str, *, reason: str) -> Path:
    if not reason.strip():
        raise ValueError("Stopping a run requires a non-empty reason.")
    owner_dir = Path(plan_dir).absolute()
    workspace = _infer_slurm_workspace(owner_dir)
    with experiment_workspace.managed_run_lock(workspace):
        plan, rows = _read_infer_slurm_plan(owner_dir, workspace)
        key = validated_run_key(plan["runs"][0])
        if key[1] != run_id:
            raise ValueError(f"Unknown run_id: {run_id}")
        _committed, metadata_stop = managed_scheduler.stop_slurm_run_locked(
            workspace,
            rows,
            key,
            reason=reason,
            hooks=managed_scheduler.SchedulerHooks(merge_manifest=merge_run_manifest, append_event=append_event),
            now=utc_now,
        )
        if metadata_stop:
            append_event(workspace, "run_stopped", {"step_id": key[0], "run_id": key[1], "reason": reason})
    return workspace / "run_manifest.tsv"


def init_experiment(run_dir: str | Path, spec_path: str | Path) -> Path:
    root = _target_root(run_dir)
    raw = read_managed_yaml_mapping(Path(spec_path).read_text(), source=f"Experiment spec {spec_path}")
    experiment = raw.get("experiment") if isinstance(raw, dict) and isinstance(raw.get("experiment"), dict) else raw
    if not isinstance(experiment, dict):
        raise ValueError("Experiment spec must be a YAML mapping.")
    experiment = dict(experiment)
    experiment["root"] = str(root)
    issues = experiment_metadata_issues(
        {
            "experiment": experiment,
            "step": {"id": "init", "phase": "prepare", "purpose": "initialize experiment"},
        }
    )
    if issues:
        raise ValueError("; ".join(issue["message"] for issue in issues))
    existing_text = exp_io.read_text_at(root / "experiment.yaml")
    if root.exists() and any(root.iterdir()) and not existing_text:
        raise ValueError(f"Experiment root is non-empty: {root}")
    manifest = root / "experiment_manifest.tsv"
    rows = []
    if existing_text:
        validate_existing_experiment_manifest(existing_text, experiment, root)
        _managed_rows(root)
        rows = exp_io.read_rows_at(manifest, strict=True)
    exp_io.mkdir_experiment_dirs(root)
    if not existing_text:
        write_initial_experiment_manifest(root, experiment)
        append_event(root, "experiment_initialized", {"experiment_id": experiment["id"]})
    exp_io.write_text_at(root / "README.md", experiment_readme_text(experiment))
    if rows:
        row = rows[0]
        row.update(
            {
                "experiment_id": experiment["id"],
                "experiment_root": str(root),
                "title": experiment["title"],
                "objective": experiment["objective"],
                "updated_at": utc_now(),
            }
        )
    else:
        row = {
            "experiment_id": experiment["id"],
            "experiment_root": str(root),
            "title": experiment["title"],
            "objective": experiment["objective"],
            "task": "",
            "selection_metric": "",
            "selection_mode": "",
            "wandb_entity": "",
            "wandb_project": "",
            "wandb_group": "",
            "created_at": utc_now(),
            "updated_at": utc_now(),
        }
    exp_io.write_rows_at(manifest, [row])
    return manifest


def append_experiment_note(
    run_dir: str | Path,
    entry_path: str | Path,
) -> ExperimentNoteResult:
    root = _target_root(run_dir)
    experiment, rows = _managed_workspace(root, allow_completed=True)
    entry = read_managed_yaml_mapping(
        Path(entry_path).read_text(),
        source=f"Research log entry {entry_path}",
    )
    path, entry_id, appended = append_research_log(
        root,
        entry,
        experiment_id=str(experiment["id"]),
        managed_rows=rows,
    )
    return {"path": str(path), "entry_id": entry_id, "appended": appended}


def _validate_hparam_checkpoints(
    rows: list[dict[str, str]],
    selected_steps: Sequence[tracking.HparamSelectionReportStep],
) -> None:
    checkpoint_rows_by_path: dict[tuple[str, str, str], dict[str, Any]] = {}
    for step in selected_steps:
        for row in step["ranked"]:
            checkpoint_rows_by_path.setdefault((*validated_run_key(row), str(row["checkpoint_path"])), row)
    checkpoint_rows = list(checkpoint_rows_by_path.values())
    tracking.validate_checkpoint_evidence_rows(rows, checkpoint_rows)
    for row in checkpoint_rows:
        if evidence.checkpoint_file_sha256(row["checkpoint_path"]) != row["checkpoint_sha256"]:
            raise ValueError(f"Frozen checkpoint SHA-256 differs: {row['checkpoint_path']}")


def _final_report_text(
    report_path: str | Path,
    hparam: tracking.HparamSelectionLifecycle,
    selection_report: tracking.HparamSelectionReportSnapshot | None,
) -> str:
    """The final report text, enforcing which report an hparam experiment must finalize from."""
    report_path_is_selection = str(Path(report_path)) == hparam["report_path"]
    if hparam["selected_steps"] and hparam["automatic_report_final"]:
        if not report_path_is_selection:
            raise ValueError(f"Successful hparam experiments must finalize from {hparam['report_path']}")
        # finalize_experiment requires the selection snapshot whenever a step is selected.
        assert selection_report is not None
        return str(selection_report["text"])
    report_text = exp_io.read_text_at(report_path)
    report_is_selection = report_path_is_selection or bool(
        selection_report is not None and hashlib.sha256(report_text.encode()).hexdigest() == selection_report["sha256"]
    )
    if hparam["selected_steps"] and report_is_selection:
        raise ValueError("The hparam selection report cannot replace the required combined experiment report.")
    if hparam["hparam_steps"] and not hparam["selected_steps"] and report_is_selection:
        raise ValueError("The hparam selection report cannot replace the required hparam failure report.")
    return report_text


def finalize_experiment(run_dir: str | Path, report_path: str | Path) -> Path:
    """Publish a final report and commit the managed experiment as completed.

    One check under the workspace run lock: every run is terminal, every
    stopped run has a stop_reason, every canonical step is materialized, every
    successful hparam step is selected with a selection report that matches its
    canonical evidence, selected checkpoints still hash to their frozen SHA-256,
    and the final report is non-empty. Failed runs are allowed when the required
    failure or combined report is supplied; completion does not mean every run
    succeeded.

    Still under the lock, writes reports/final.md, appends a preparation event,
    then commits experiment.yaml with the final report path and SHA-256, plus the
    selection report SHA-256 when selection exists. Both writes are atomic
    replacements and experiment.yaml is the terminal commit, so a failure between
    them leaves a published report without completed status. Returns the
    canonical final report path. Invalid prerequisites raise ValueError; I/O
    errors propagate. Does not launch runs or refresh their observed statuses."""
    root = _target_root(run_dir)
    target = root / "reports" / "final.md"
    with experiment_workspace.managed_run_lock(root):
        experiment, rows = _managed_workspace(root, lock_held=True)
        if not rows:
            raise ValueError("Experiment has no managed runs to finalize.")
        unresolved = [row["run_id"] for row in rows if row.get("status") not in TERMINAL_STATUSES]
        if unresolved:
            raise ValueError(f"Experiment still has unresolved runs: {unresolved}")
        missing_stop_reasons = stopped_runs_without_reason(rows)
        if missing_stop_reasons:
            run_ids = [f"{row['step_id']} / {row['run_id']}" for row in missing_stop_reasons]
            raise ValueError(f"Stopped runs are missing required stop_reason: {run_ids}")
        registered_steps = _registered_plan_steps(root, experiment, rows, require_registered_rows=False)
        selection_report = (
            _hparam_selection_report(root)
            if any(plan.get("task") == "hparam_tune" for step in registered_steps for plan in step["plans"])
            else None
        )
        snapshot = tracking.experiment_status_snapshot(
            experiment, registered_steps, rows, root=root, hparam_selection_report=selection_report
        )
        if any(blocker["code"] == "unmaterialized_step" for blocker in snapshot["blockers"]):
            raise ValueError("Experiment cannot be finalized with incomplete canonical steps: unmaterialized_step")
        hparam = tracking.hparam_selection_lifecycle(registered_steps, rows, root=root, report=selection_report)
        if hparam["pending_steps"]:
            raise ValueError("Successful hparam runs must be selected before experiment finalization.")
        selected_report = selection_report if hparam["selected_steps"] else None
        if hparam["selected_steps"] and (selected_report is None or not hparam["report_valid"]):
            raise ValueError("The hparam selection report is missing or differs from canonical selection evidence.")
        report_text = _final_report_text(report_path, hparam, selection_report)
        if not report_text.strip():
            raise ValueError("Final report is missing or empty.")
        if hparam["selected_steps"]:
            _validate_hparam_checkpoints(rows, hparam["selected_steps"])

        exp_io.atomic_replace(target, report_text)
        append_event(root, "experiment_finalization_prepared", {"report": str(target)})
        completion: ExperimentCompletionFields = {
            "status": "completed",
            "completed_at": utc_now(),
            "final_report": str(target),
            "final_report_sha256": hashlib.sha256(report_text.encode()).hexdigest(),
        }
        experiment.update(completion)
        if selected_report is not None:
            experiment["selection_report_sha256"] = selected_report["sha256"]
        # The experiment manifest is the terminal commit, so publish it only after the report is durable.
        exp_io.atomic_replace(root / "experiment.yaml", yaml.safe_dump({"experiment": experiment}, sort_keys=False))
    return target


def monitor_experiment(run_dir: str | Path) -> tracking.ExperimentMonitorResult:
    """Observe managed runs once and commit their current status without launching work.

    Requires an active, valid managed workspace. Reads local process,
    artifact and Slurm evidence (one ``squeue --me`` query when a Slurm run
    needs a state), merges observations into run_manifest.tsv, then writes
    reports/monitor.md. Appends a run_status_changed event for each status this
    poll changes, plus run_stopped when it confirms a requested Slurm stop.
    Per-poll observation fields (MONITOR_OBSERVATION_FIELDS, such as log_tail
    and log_age_seconds) appear only in that report and the returned runs,
    never in run_manifest.tsv. Also probes the queue runner of
    every registered plan that has a queue_runner.lock (only a direct hparam
    queue creates one) without starting any runner. Returns run_dir, the
    committed runs with this poll's observation fields, that runner liveness and
    the report path; these are observations rather than an experiment_status
    lifecycle decision.

    Invalid ownership/state and unhandled
    observation or write errors propagate; a report-write failure can occur
    after the canonical manifest has already been updated."""
    root = _target_root(run_dir)
    run_rows = _managed_rows(root)
    report_path = root / "reports" / "monitor.md"
    # Observe the owner-validated input snapshot; the commit still rereads canonical rows under the run lock.
    slurm_jobs = managed_scheduler.slurm_job_snapshot(run_rows)
    observations = [tracking.monitor_run_row(root, row, run_rows, slurm_jobs=slurm_jobs) for row in run_rows]
    observed_by_key = {validated_run_key(row): row for row in observations}
    committed_rows = merge_run_manifest(root, observations)
    _record_status_transitions(root, run_rows, committed_rows)
    # run_manifest.tsv drops per-poll observation fields, so the report and returned runs carry this poll's values.
    committed = [
        {
            **row,
            **{
                field: str(value)
                for field, value in observed_by_key.get(validated_run_key(row), {}).items()
                if field in MONITOR_OBSERVATION_FIELDS
            },
        }
        for row in committed_rows
    ]
    # Registered plans own canonical rows, and _managed_rows checked every row's experiment_id.
    step_manifests = read_registered_steps(root, experiment_id=str(run_rows[0]["experiment_id"])) if run_rows else []
    queue_runners = [
        managed_scheduler.queue_runner_liveness(Path(plan_path))
        for manifest in step_manifests
        for plan_path in manifest["plans"]
        if (Path(plan_path) / managed_scheduler.QUEUE_RUNNER_LOCK_NAME).exists()
    ]
    report = tracking.monitor_report(committed, queue_runners)
    exp_io.write_text_at(report_path, report)
    return {"run_dir": str(root), "runs": committed, "queue_runners": queue_runners, "report": str(report_path)}


def _record_status_transitions(
    root: Path, previous_rows: Sequence[dict[str, str]], committed_rows: Sequence[dict[str, str]]
) -> None:
    previous_by_key = {validated_run_key(row): row for row in previous_rows}
    for row in committed_rows:
        previous = previous_by_key.get(validated_run_key(row), {})
        before, after = previous.get("status"), row.get("status")
        if not before or not after or before == after:
            continue
        run = {"step_id": row["step_id"], "run_id": row["run_id"]}
        append_event(root, "run_status_changed", {**run, "from": before, "to": after})
        # A stale pre-cancel observation may merge to stopped without confirming a scheduler stop.
        if after == "stopped" and scheduler_type(row) == "slurm" and previous.get("stop_requested_at"):
            append_event(root, "run_stopped", {**run, "reason": row.get("stop_reason", "")})


def experiment_status(run_dir: str | Path) -> tracking.ExperimentStatusSnapshot:
    """Return a read-only lifecycle snapshot from the managed workspace's recorded state.

    Validates registered plan bindings and available selection/report/checkpoint
    evidence, including completed workspaces. Does not poll
    jobs or refresh canonical run statuses: live_observation is False. The one
    live read is a shared-lock probe of each hparam plan with launchable runs:
    an active queue runner adds a queue_runner_active blocker, and that plan's
    runs no longer make the state ready_to_launch.

    The mapping contains experiment identity, lifecycle_source, summary (with the
    lifecycle state), steps, runs and blockers; it names no commands to run.
    Lifecycle blockers appear in that snapshot; invalid workspace/plan contracts and
    unhandled read errors raise. No reports or manifests are written."""
    root = _target_root(run_dir)
    experiment, rows = _managed_workspace(
        root,
        allow_completed=True,
        validate_experiment_index=False,
    )
    registered_steps = _registered_plan_steps(root, experiment, rows, require_registered_rows=True)
    selection_report = (
        _hparam_selection_report(root)
        if any(plan.get("task") == "hparam_tune" for step in registered_steps for plan in step["plans"])
        else None
    )

    return tracking.experiment_status_snapshot(
        experiment,
        registered_steps,
        rows,
        root=root,
        hparam_selection_report=selection_report,
    )


def _registered_plan_steps(
    root: Path,
    experiment: dict[str, Any],
    rows: list[dict[str, str]],
    *,
    require_registered_rows: bool,
) -> list[artifacts.RegisteredPlanStep]:
    step_manifests = read_registered_steps(root, experiment_id=str(experiment["id"]))
    legacy_run_identity_fields = {"experiment_id", "step_id", "run_id", "run_name", "version"}
    managed_plan_fields = (FROZEN_RUN_FIELDS - legacy_run_identity_fields) | tracking.HPARAM_SELECTION_METADATA_FIELDS
    has_managed_plan_rows = any(
        managed_run_parameters(row) or any(row.get(field) not in (None, "") for field in managed_plan_fields)
        for row in rows
    )
    if not step_manifests and not require_registered_rows and not has_managed_plan_rows:
        return []
    if (
        not require_registered_rows
        and not has_managed_plan_rows
        and not any(manifest["plans"] for manifest in step_manifests)
    ):
        registered_step_ids = {str(manifest["step"]["id"]) for manifest in step_manifests}
        if not ({str(row["step_id"]) for row in rows} & registered_step_ids):
            return []
    step_ids = {str(manifest["step"]["id"]) for manifest in step_manifests}
    orphaned_steps = sorted({str(row["step_id"]) for row in rows} - step_ids)
    if orphaned_steps:
        raise ValueError(f"run_manifest.tsv references unregistered steps: {', '.join(orphaned_steps)}")

    plan_owners: dict[str, str] = {}
    for manifest in step_manifests:
        step_id = str(manifest["step"]["id"])
        for plan_path in manifest["plans"]:
            owner = plan_owners.setdefault(str(plan_path), step_id)
            if owner != step_id:
                raise ValueError(f"Registered plan belongs to more than one managed step: {plan_path}")

    registered_steps: list[artifacts.RegisteredPlanStep] = []
    for manifest in step_manifests:
        step_id = str(manifest["step"]["id"])
        step_rows = [row for row in rows if str(row["step_id"]) == step_id]
        plans: list[artifacts.RegisteredPlanSummary] = []
        plan_keys: list[tuple[str, str]] = []
        for plan_path in manifest["plans"]:
            if artifacts.is_registered_blocked_plan(plan_path, workspace=root):
                continue
            plan = artifacts.read_registered_plan(
                plan_path, workspace_experiment=experiment, step_manifest=manifest, workspace_rows=rows
            )
            plans.append(plan)
            plan_keys.extend(plan["run_keys"])
        if len(plan_keys) != len(set(plan_keys)):
            raise ValueError(f"Managed step registers duplicate run keys across plans: {step_id}")
        canonical_keys = {managed_run_key(row) for row in step_rows}
        if set(plan_keys) != canonical_keys:
            raise ValueError(f"Managed step plans differ from canonical run keys: {step_id}")
        registered_steps.append({"manifest": manifest, "plans": plans})

    return registered_steps


def _hparam_selection_report(root: Path) -> tracking.HparamSelectionReportSnapshot | None:
    path = root / "reports" / "hparam_selection.md"
    ranking_path = root / "reports" / "ranking.csv"
    if not exp_io.path_exists_at(path):
        return None
    read_paths: list[str | Path] = [path]
    if exp_io.path_exists_at(ranking_path):
        read_paths.append(ranking_path)
    files = exp_io.read_managed_files_at(root, read_paths)
    ranking = files.get(str(ranking_path))
    return {
        "path": str(path),
        **files[str(path)],
        "ranking_path": str(ranking_path),
        "ranking_text": ranking["text"] if ranking is not None else None,
        "ranking_sha256": ranking["sha256"] if ranking is not None else None,
    }


def _target_root(run_dir: str | Path) -> Path:
    return canonical_local_experiment_root(Path(run_dir), Path.cwd())


def _managed_rows(root: Path) -> list[dict[str, str]]:
    _experiment, rows = _managed_workspace(root)
    return rows


def _normalized_completed_metadata(
    raw_experiment: Any, *, root: Path, allow_completed: bool
) -> tuple[Any, list[str | Path]]:
    validated_experiment = raw_experiment
    completed_bindings: list[str | Path] = []
    if (
        allow_completed
        and isinstance(raw_experiment, dict)
        and ("status" in raw_experiment or "completed_at" in raw_experiment)
    ):
        terminal_fields = set(raw_experiment) - {"id", "title", "objective", "root", "baseline"}
        allowed_terminal_fields = (
            {"status", "completed_at"},
            {"status", "completed_at", "final_report", "final_report_sha256"},
            {
                "status",
                "completed_at",
                "final_report",
                "final_report_sha256",
                "selection_report_sha256",
            },
        )
        if terminal_fields not in allowed_terminal_fields:
            raise ValueError("Completed experiment metadata has incomplete or unexpected terminal fields.")
        completed_at = raw_experiment.get("completed_at")
        if raw_experiment.get("status") != "completed" or not isinstance(completed_at, str):
            raise ValueError("Completed experiment metadata is invalid.")
        try:
            completed_time = datetime.fromisoformat(completed_at.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("Completed experiment completed_at must be an ISO timestamp.") from exc
        if completed_time.tzinfo is None or completed_time.utcoffset() != timezone.utc.utcoffset(completed_time):
            raise ValueError("Completed experiment completed_at must be in UTC.")
        if "final_report" in raw_experiment:
            final_report = root / "reports" / "final.md"
            if raw_experiment["final_report"] != str(final_report) or not SHA256_RE.fullmatch(
                str(raw_experiment["final_report_sha256"])
            ):
                raise ValueError("Completed experiment final report binding is invalid.")
            completed_bindings.append(final_report)
            selection_sha256 = raw_experiment.get("selection_report_sha256")
            if selection_sha256 is not None:
                if not SHA256_RE.fullmatch(str(selection_sha256)):
                    raise ValueError("Completed experiment selection report binding is invalid.")
                completed_bindings.append(root / "reports" / "hparam_selection.md")
        validated_experiment = {field: value for field, value in raw_experiment.items() if field not in terminal_fields}
    return validated_experiment, completed_bindings


def _managed_workspace(
    root: Path,
    *,
    allow_completed: bool = False,
    validate_experiment_index: bool = True,
    lock_held: bool = False,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    manifest_path = root / "experiment.yaml"
    if not exp_io.path_exists_at(manifest_path):
        raise ValueError("experiment.yaml is missing. Initialize the experiment first.")
    files = exp_io.read_managed_files_at(root, [manifest_path])
    experiment_text = files[str(manifest_path)]["text"]
    manifest = read_managed_yaml_mapping(experiment_text, source=f"Managed experiment manifest {manifest_path}")
    if set(manifest) != {"experiment"}:
        raise ValueError("experiment.yaml must contain only the experiment owner mapping.")
    raw_experiment = manifest.get("experiment")
    validated_experiment, completed_bindings = _normalized_completed_metadata(
        raw_experiment, root=root, allow_completed=allow_completed
    )
    issues = experiment_metadata_issues(
        {
            "experiment": validated_experiment,
            "step": {"id": "preflight", "phase": "prepare", "purpose": "validate experiment workspace"},
        }
    )
    if issues:
        raise ValueError("; ".join(issue["message"] for issue in issues))
    experiment: dict[str, Any] = manifest["experiment"]
    if str(experiment["root"]) != str(root):
        raise ValueError(f"experiment.root differs from the target workspace: {root}")

    if completed_bindings:
        bound_files = exp_io.read_managed_files_at(root, completed_bindings)
        if bound_files[str(completed_bindings[0])]["sha256"] != experiment["final_report_sha256"]:
            raise ValueError("Completed experiment final report differs from its terminal binding.")
        if len(completed_bindings) == 2 and (
            bound_files[str(completed_bindings[1])]["sha256"] != experiment["selection_report_sha256"]
        ):
            raise ValueError("Completed experiment selection report differs from its terminal binding.")

    legacy_path = root / "trial_status.tsv"
    if exp_io.path_exists_at(legacy_path):
        raise ValueError(f"Historical experiment artifacts are read-only: {legacy_path}")

    experiment_manifest = root / "experiment_manifest.tsv"
    if validate_experiment_index and exp_io.path_exists_at(experiment_manifest):
        manifest_rows = exp_io.read_rows_at(experiment_manifest, strict=True)
        if len(manifest_rows) != 1:
            raise ValueError("experiment_manifest.tsv must contain exactly one row.")
        manifest_row = manifest_rows[0]
        if manifest_row.get("experiment_id") != experiment["id"]:
            raise ValueError("experiment_manifest.tsv belongs to a different experiment.")
        if manifest_row.get("experiment_root") != str(root):
            raise ValueError("experiment_manifest.tsv root differs from the target workspace.")

    # lock_held means the caller already owns the run lock.
    with nullcontext() if lock_held else experiment_workspace.managed_run_lock(root):
        rows = read_run_manifest(root)
    for row in rows:
        if row["experiment_id"] != experiment["id"]:
            raise ValueError("run_manifest.tsv contains a run owned by a different experiment.")
    return experiment, rows
