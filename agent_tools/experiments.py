from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from pathlib import Path
from typing import Any

import yaml

from . import (
    experiment_io as exp_io,
    experiment_tracking as tracking,
    managed_scheduler,
    python_programs,
    run_artifacts as artifacts,
    run_evidence as evidence,
)
from .experiment_workspace import (
    FROZEN_RUN_FIELDS,
    PROCESS_IDENTITY_FIELDS,
    RESEARCH_LOG_NAME,
    SHA256_RE,
    TERMINAL_STATUSES,
    append_event,
    append_research_log,
    canonical_local_experiment_root,
    commit_step_manifest,
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
    validate_managed_run_rows,
    verify_run_snapshot,
    write_initial_experiment_manifest,
    write_status_report,
)
from .manifests import read_json, utc_now


def _read_preset_direct_plan(plan_dir: Path) -> tuple[Path, artifacts.RegisteredPlanSummary, list[dict[str, Any]]]:
    # The locator is untrusted; all execution fields below come from the registered frozen plan.
    initial = read_json(plan_dir / "plan.json")
    recipe = initial.get("recipe") if isinstance(initial, dict) else None
    if not isinstance(recipe, dict) or recipe.get("task") != "preset_prepare":
        raise ValueError("preset-launch and preset-stop require a preset preparation plan.")
    issues = experiment_metadata_issues(recipe)
    if issues:
        raise ValueError("Invalid preset workspace binding: " + "; ".join(issue["message"] for issue in issues))
    workspace = experiment_root(recipe)
    assert workspace is not None
    experiment, rows = _managed_workspace(workspace, remote=None)
    registered_steps = _registered_plan_steps(workspace, experiment, rows, remote=None, require_registered_rows=True)
    for registered in registered_steps:
        for plan in registered["plans"]:
            if plan["path"] != str(plan_dir):
                continue
            execution = plan["recipe"].get("execution") or {}
            if (
                registered["manifest"]["plan_controller"] != "ordinary"
                or plan["task"] != "preset_prepare"
                or len(plan["runs"]) != 1
                or execution.get("scheduler") != {"type": "direct"}
                or (execution.get("target") or "local") != "local"
                or any(not execution.get(field) for field in ("python", "runtime_commit", "workdir"))
                or plan["runs"][0].get("terminal_status_owner") != "script"
            ):
                raise ValueError("preset-launch and preset-stop require one new local managed direct preset run.")
            return workspace, plan, rows
    raise ValueError(f"Preset plan is not registered in its experiment workspace: {plan_dir}")


def launch_preset_run(plan_dir: str | Path, *, dry_run: bool = True) -> managed_scheduler.LaunchResult:
    owner_dir = Path(plan_dir).absolute()
    workspace, _plan, _rows = _read_preset_direct_plan(owner_dir)
    with managed_scheduler.managed_run_lock(workspace):
        locked_workspace, plan, rows = _read_preset_direct_plan(owner_dir)
        if locked_workspace != workspace:
            raise ValueError("Preset plan workspace changed before launch.")
        run = plan["runs"][0]
        key = (str(run["step_id"]), str(run["run_id"]))
        previous = {managed_run_key(row): row for row in rows}[key]
        if previous["status"] not in managed_scheduler.LAUNCHABLE_STATUSES:
            return managed_scheduler.LaunchResult(rows, [previous], frozenset(), {}, {})
        if has_managed_launch_evidence(previous):
            raise ValueError("Preset run already has launch evidence; refusing another launch attempt.")
        execution = plan["recipe"]["execution"]
        run_dir = Path(run["run_dir"])
        identity = {
            "target": "local",
            "host": "",
            "workdir": execution["workdir"],
            "gpus": "",
            "pid_path": str(run_dir / "pid"),
            "log_path": str(run_dir / "stdout.log"),
        }
        identity["command"] = managed_scheduler.build_launch_command(
            execution,
            Path(run["script"]),
            identity["log_path"],
            identity["pid_path"],
            [],
            config_path=Path(run["config"]),
            script_sha256=run["script_sha256"],
            config_sha256=run["config_sha256"],
        )
        exp_io.validate_managed_output_paths(
            workspace,
            [
                workspace / "run_manifest.tsv",
                workspace / "run_matrix.csv",
                workspace / "reports" / "run_matrix.md",
                workspace / "events.jsonl",
                workspace / "reports" / "status.md",
                Path(identity["pid_path"]),
                Path(identity["log_path"]),
            ],
        )
        verify_run_snapshot(run)
        preview = {**previous, **identity}
        validate_frozen_run_update(previous, preview, allow_execution_identity_fill=True)
        if dry_run:
            return managed_scheduler.LaunchResult(rows, [preview], frozenset(), {}, {})
        if Path(identity["pid_path"]).exists():
            raise ValueError("Preset PID receipt already exists; refusing another launch attempt.")
        probe = managed_scheduler.run_execution_command(
            execution,
            [
                execution["python"],
                "-c",
                python_programs.source("managed_scheduler.runtime_identity"),
                "agent_tools.experiment_workspace",
                "{}",
                "[]",
                str(execution["runtime_commit"]),
            ],
        )
        if probe.returncode != 0:
            detail = probe.stderr.strip() or probe.stdout.strip() or f"exit code {probe.returncode}"
            raise RuntimeError(f"Preset runtime preflight failed: {detail}")
        verify_run_snapshot(run)
        # Claim before fork: an interrupted manager must not turn an uncertain launch into a retry.
        attempted = {
            **preview,
            "planned_runtime_commit": str(execution["runtime_commit"]),
            "status": "launched",
            "launched_at": utc_now(),
        }
        merge_run_manifest(workspace, [attempted], lock_held=True)
        status = managed_scheduler.start_process(
            execution,
            identity["command"],
            retry_pre_spawn_failure=False,
        )
        attempted["status"] = status
        process_identity = evidence.read_process_identity(identity["pid_path"], attempted)
        if process_identity is not None:
            attempted.update(process_identity)
        committed = merge_run_manifest(workspace, [attempted], lock_held=True)
        final = {managed_run_key(row): row for row in committed}[key]
        append_event(
            workspace,
            "run_status_changed",
            {"step_id": key[0], "run_id": key[1], "from": previous["status"], "to": final["status"]},
        )
        write_status_report(workspace)
        return managed_scheduler.LaunchResult(
            committed,
            [final],
            frozenset({key}) if status == "launched" else frozenset(),
            {key: (previous["status"], final["status"])},
            {},
        )


def stop_preset_run(plan_dir: str | Path, *, reason: str) -> Path:
    if not reason.strip():
        raise ValueError("Stopping a run requires a non-empty reason.")
    owner_dir = Path(plan_dir).absolute()
    workspace, _plan, _rows = _read_preset_direct_plan(owner_dir)
    with managed_scheduler.managed_run_lock(workspace):
        locked_workspace, plan, rows = _read_preset_direct_plan(owner_dir)
        if locked_workspace != workspace:
            raise ValueError("Preset plan workspace changed before stop.")
        exp_io.validate_managed_output_paths(
            workspace,
            [
                workspace / "run_manifest.tsv",
                workspace / "run_matrix.csv",
                workspace / "reports" / "run_matrix.md",
                workspace / "events.jsonl",
                workspace / "reports" / "status.md",
            ],
        )
        run = plan["runs"][0]
        key = (str(run["step_id"]), str(run["run_id"]))
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
            write_status_report(workspace)
            return workspace / "run_manifest.tsv"
        if previous.get("target") != "local" or previous.get("pid_path") != str(Path(run["run_dir"]) / "pid"):
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
    with managed_scheduler.managed_run_lock(workspace):
        locked_workspace, _plan, rows = _read_preset_direct_plan(owner_dir)
        if locked_workspace != workspace:
            raise ValueError("Preset plan workspace changed before stop completion.")
        latest = {managed_run_key(row): row for row in rows}[key]
        validate_frozen_run_update(latest, stopping)
        merge_run_manifest(
            workspace,
            [{**latest, "status": "stopped", "stopped_at": utc_now()}],
            lock_held=True,
        )
        append_event(workspace, "run_stopped", {"step_id": key[0], "run_id": key[1], "reason": stopping["stop_reason"]})
        write_status_report(workspace)
    return workspace / "run_manifest.tsv"


def _read_infer_slurm_plan(plan_dir: Path) -> tuple[Path, artifacts.RegisteredPlanSummary, list[dict[str, Any]]]:
    # The initial document only locates the workspace; execution uses the strict registered-plan reader below.
    initial = read_json(plan_dir / "plan.json")
    recipe = initial.get("recipe") if isinstance(initial, dict) else None
    if not isinstance(recipe, dict) or recipe.get("task") not in {"infer", "evaluate"}:
        raise ValueError("infer-launch and infer-stop require an ordinary infer/evaluate plan.")
    issues = experiment_metadata_issues(recipe)
    if issues:
        raise ValueError("Invalid inference workspace binding: " + "; ".join(issue["message"] for issue in issues))
    workspace = experiment_root(recipe)
    assert workspace is not None
    experiment, rows = _managed_workspace(workspace, remote=None)
    registered_steps = _registered_plan_steps(workspace, experiment, rows, remote=None, require_registered_rows=True)
    for registered in registered_steps:
        for plan in registered["plans"]:
            if plan["path"] != str(plan_dir):
                continue
            if (
                registered["manifest"]["plan_controller"] != "ordinary"
                or plan["task"] not in {"infer", "evaluate"}
                or len(plan["runs"]) != 1
                or scheduler_type(plan["runs"][0]) != "slurm"
            ):
                raise ValueError("infer-launch and infer-stop require one ordinary managed Slurm inference run.")
            return workspace, plan, rows
    raise ValueError(f"Inference plan is not registered in its experiment workspace: {plan_dir}")


def launch_infer_run(plan_dir: str | Path, *, dry_run: bool = True) -> managed_scheduler.LaunchResult:
    owner_dir = Path(plan_dir).absolute()
    workspace, _plan, _rows = _read_infer_slurm_plan(owner_dir)
    with managed_scheduler.managed_run_lock(workspace):
        locked_workspace, plan, _rows = _read_infer_slurm_plan(owner_dir)
        if locked_workspace != workspace:
            raise ValueError("Inference plan workspace changed before launch.")
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
                key = (str(plan["runs"][0]["step_id"]), str(plan["runs"][0]["run_id"]))
                previous = {managed_run_key(row): row for row in read_run_manifest(workspace)}[key]
                # A lost receipt or post-submit failure must never become a definitely-unsubmitted failure.
                if previous["status"] in managed_scheduler.LAUNCHABLE_STATUSES and not has_managed_launch_evidence(
                    previous
                ):
                    merge_run_manifest(
                        workspace,
                        [
                            {
                                "step_id": key[0],
                                "run_id": key[1],
                                "status": "launch_failed",
                                "scheduler_reason": f"Pre-submission guard failed: {exc}",
                                "scheduler_observed_at": utc_now(),
                            }
                        ],
                        lock_held=True,
                    )
                    append_event(
                        workspace,
                        "run_status_changed",
                        {"step_id": key[0], "run_id": key[1], "from": previous["status"], "to": "launch_failed"},
                    )
                    write_status_report(workspace)
            raise


def stop_infer_run(plan_dir: str | Path, *, reason: str) -> Path:
    if not reason.strip():
        raise ValueError("Stopping a run requires a non-empty reason.")
    owner_dir = Path(plan_dir).absolute()
    workspace, _plan, _rows = _read_infer_slurm_plan(owner_dir)
    with managed_scheduler.managed_run_lock(workspace):
        locked_workspace, plan, rows = _read_infer_slurm_plan(owner_dir)
        if locked_workspace != workspace:
            raise ValueError("Inference plan workspace changed before stop.")
        exp_io.validate_managed_output_paths(
            workspace,
            [
                workspace / "run_manifest.tsv",
                workspace / "run_matrix.csv",
                workspace / "reports" / "run_matrix.md",
                workspace / "events.jsonl",
                workspace / "reports" / "status.md",
            ],
        )
        key = (str(plan["runs"][0]["step_id"]), str(plan["runs"][0]["run_id"]))
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
        write_status_report(workspace)
    return workspace / "run_manifest.tsv"


def run_experiment_pipeline(
    run_dir: str | Path,
    spec_path: str | Path,
    *,
    unlock_final_test: bool = False,
    execute: bool = False,
    resume: bool = False,
    poll_seconds: float = 60,
) -> dict[str, Any]:
    from .experiment_pipeline import run_experiment_pipeline as run_pipeline

    return run_pipeline(
        run_dir,
        spec_path,
        unlock_final_test=unlock_final_test,
        execute=execute,
        resume=resume,
        poll_seconds=poll_seconds,
        finalize_callback=lambda root, report: finalize_experiment(root, report),
    )


def init_experiment(run_dir: str | Path, spec_path: str | Path, *, remote: str | None = None) -> Path:
    root = _target_root(run_dir, remote)
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
    existing_text = exp_io.read_text_at(root / "experiment.yaml", remote=remote)
    if not remote and root.exists() and any(root.iterdir()) and not existing_text:
        raise ValueError(f"Experiment root is non-empty: {root}")
    if remote and not existing_text and exp_io.remote_dir_nonempty(root, remote):
        raise ValueError(f"Experiment root is non-empty: {root}")
    manifest = root / "experiment_manifest.tsv"
    rows = []
    if existing_text:
        validate_existing_experiment_manifest(existing_text, experiment, root)
        _managed_rows(root, remote=remote)
        rows = exp_io.read_rows_at(manifest, remote=remote, strict=True)
    exp_io.validate_managed_output_paths(
        root,
        [
            root / "experiment.yaml",
            root / "run_manifest.tsv",
            root / RESEARCH_LOG_NAME,
            root / "events.jsonl",
            root / "README.md",
            manifest,
        ],
        remote=remote,
    )
    exp_io.mkdir_experiment_dirs(root, remote=remote)
    if not existing_text:
        write_initial_experiment_manifest(root, experiment, remote=remote)
        exp_io.append_event_at(root, "experiment_initialized", {"experiment_id": experiment["id"]}, remote=remote)
    exp_io.write_text_at(root / "README.md", experiment_readme_text(experiment), remote=remote)
    if rows:
        row = rows[0]
        row.update(
            {
                "experiment_id": experiment["id"],
                "experiment_root": str(root),
                "title": experiment["title"],
                "objective": experiment["objective"],
                "remote_host": remote or row.get("remote_host", ""),
                "updated_at": utc_now(),
            }
        )
    else:
        row = {
            "experiment_id": experiment["id"],
            "experiment_root": str(root),
            "title": experiment["title"],
            "objective": experiment["objective"],
            "remote_host": remote or "",
            "task": "",
            "selection_metric": "",
            "selection_mode": "",
            "wandb_entity": "",
            "wandb_project": "",
            "wandb_group": "",
            "created_at": utc_now(),
            "updated_at": utc_now(),
        }
    exp_io.write_rows_at(manifest, [row], remote=remote)
    return manifest


def append_experiment_note(
    run_dir: str | Path,
    entry_path: str | Path,
    *,
    remote: str | None = None,
) -> dict[str, Any]:
    root = _target_root(run_dir, remote)
    experiment, rows = _managed_workspace(root, remote=remote, allow_completed=True)
    entry = read_managed_yaml_mapping(
        Path(entry_path).read_text(),
        source=f"Research log entry {entry_path}",
    )
    path, entry_id, appended = append_research_log(
        root,
        entry,
        experiment_id=str(experiment["id"]),
        managed_rows=rows,
        remote=remote,
    )
    return {"path": str(path), "entry_id": entry_id, "appended": appended}


def register_experiment_step(run_dir: str | Path, spec_path: str | Path, *, remote: str | None = None) -> Path:
    root = _target_root(run_dir, remote)
    _managed_rows(root, remote=remote)
    experiment_text = exp_io.read_text_at(root / "experiment.yaml", remote=remote)
    experiment_manifest = (
        read_managed_yaml_mapping(experiment_text, source=f"Managed experiment manifest {root / 'experiment.yaml'}")
        if experiment_text
        else {}
    )
    experiment = experiment_manifest.get("experiment") if isinstance(experiment_manifest, dict) else None
    if not isinstance(experiment, dict):
        raise ValueError("experiment.yaml is missing. Initialize the experiment first.")
    raw = read_managed_yaml_mapping(Path(spec_path).read_text(), source=f"Step spec {spec_path}")
    step = raw.get("step") if isinstance(raw, dict) and isinstance(raw.get("step"), dict) else raw
    if not isinstance(step, dict):
        raise ValueError("Step spec must be a YAML mapping.")
    for field in ("id", "phase", "purpose", "inputs", "outputs"):
        if step.get(field) in (None, "", "ASK_USER"):
            raise ValueError(f"step.{field} is required.")
    issues = experiment_metadata_issues({"experiment": experiment, "step": step}, allow_step_io=True)
    if issues:
        raise ValueError("; ".join(issue["message"] for issue in issues))
    path = root / "steps" / str(step["id"]) / "step.yaml"
    exp_io.validate_managed_output_paths(root, [path, root / "events.jsonl"], remote=remote)
    _merged, created = commit_step_manifest(
        root,
        {
            "step": step,
            "experiment_id": experiment["id"],
            "plan_controller": "unassigned",
            "recipe_path": "",
            "plans": [],
        },
        remote=remote,
    )
    if created:
        exp_io.append_event_at(root, "step_registered", {"step_id": step["id"], "phase": step["phase"]}, remote=remote)
    return path


def _validate_hparam_checkpoints(
    rows: list[dict[str, Any]],
    selected_steps: list[dict[str, Any]],
    *,
    remote: str | None,
) -> None:
    canonical_by_key = {managed_run_key(row): row for row in rows}
    checkpoint_rows_by_path: dict[tuple[str, str, str], dict[str, Any]] = {}
    for step in selected_steps:
        for row in [*step["ranked"], *step.get("checkpoint_audit_rows", [])]:
            checkpoint_rows_by_path.setdefault(
                (str(row["step_id"]), str(row["run_id"]), str(row["checkpoint_path"])), row
            )
    checkpoint_rows = list(checkpoint_rows_by_path.values())
    tracking.validate_checkpoint_evidence_rows(rows, checkpoint_rows, remote=remote)
    for row in checkpoint_rows:
        owner = canonical_by_key[managed_run_key(row)]
        evidence_host = tracking.checkpoint_evidence_host(owner, remote)
        evidence_row = owner if evidence_host is None else {**owner, "target": "ssh", "host": evidence_host}
        if evidence.checkpoint_file_sha256(evidence_row, row["checkpoint_path"]) != row["checkpoint_sha256"]:
            raise ValueError(f"Frozen checkpoint SHA-256 differs: {row['checkpoint_path']}")


def _validate_hparam_selection_files_unchanged(
    root: Path,
    selection_report: tracking.HparamSelectionReportSnapshot,
    checkpoint_audits: dict[str, exp_io.ManagedFileSnapshot | None],
    *,
    remote: str | None,
) -> None:
    expected = {
        str(selection_report["path"]): selection_report["sha256"],
        str(selection_report["ranking_path"]): selection_report["ranking_sha256"],
        **{path: audit["sha256"] for path, audit in checkpoint_audits.items() if isinstance(audit, dict)},
    }
    current = exp_io.read_managed_files_at(root, [Path(path) for path in expected], remote=remote)
    for path, sha256 in expected.items():
        if current[path]["sha256"] != sha256:
            if path == str(selection_report["path"]):
                raise ValueError("The hparam selection report changed during finalization.")
            if path == str(selection_report["ranking_path"]):
                raise ValueError("The hparam ranking changed during finalization.")
            raise ValueError(f"The frozen checkpoint test ranking changed during finalization: {path}")


def _finalizable_rows(root: Path, *, remote: str | None) -> list[dict[str, str]]:
    """The managed rows, refusing an experiment that is not ready to finalize.
    Read twice -- once before the run-manifest snapshot and once after -- so a
    run that changes during finalization cannot slip past the check."""
    rows = _managed_rows(root, remote=remote)
    if not rows:
        raise ValueError("Experiment has no managed runs to finalize.")
    unresolved = [row["run_id"] for row in rows if row.get("status") not in TERMINAL_STATUSES]
    if unresolved:
        raise ValueError(f"Experiment still has unresolved runs: {unresolved}")
    missing_stop_reasons = stopped_runs_without_reason(rows)
    if missing_stop_reasons:
        run_ids = [f"{row['step_id']} / {row['run_id']}" for row in missing_stop_reasons]
        raise ValueError(f"Stopped runs are missing required stop_reason: {run_ids}")
    return rows


def finalize_experiment(run_dir: str | Path, report_path: str | Path, *, remote: str | None = None) -> Path:
    """Publish a final report and commit the managed experiment as completed.

    Requires nonempty terminal run state, reasons for stopped runs, materialized
    canonical steps and the applicable hparam selection/report/checkpoint
    evidence. Failed runs are allowed when the required failure or combined
    report is supplied; completion does not mean every run succeeded.

    Reads report_path locally or on remote (where it must be absolute), writes
    reports/final.md and a preparation event, then commits experiment.yaml with
    report hash bindings. Returns the canonical final report path. The manifest
    is the terminal commit: a later failure can leave a published report without
    completed status. Invalid prerequisites and changed hparam evidence raise
    ValueError; run-manifest changes detected during finalization raise RuntimeError,
    and other I/O errors propagate. Does not launch runs or refresh their observed
    statuses."""
    if remote and not Path(report_path).is_absolute():
        raise ValueError("Remote final report path must be absolute.")
    root = _target_root(run_dir, remote)
    run_manifest_path = root / "run_manifest.tsv"
    rows = _finalizable_rows(root, remote=remote)
    run_manifest_snapshot = exp_io.read_managed_files_at(root, [run_manifest_path], remote=remote)[
        str(run_manifest_path)
    ]
    rows = _finalizable_rows(root, remote=remote)
    if (
        exp_io.read_managed_files_at(root, [run_manifest_path], remote=remote)[str(run_manifest_path)]["sha256"]
        != run_manifest_snapshot["sha256"]
    ):
        raise RuntimeError("Run manifest changed during finalization.")
    manifest_text = exp_io.read_text_at(root / "experiment.yaml", remote=remote)
    manifest = read_managed_yaml_mapping(
        manifest_text, source=f"Managed experiment manifest {root / 'experiment.yaml'}"
    )
    if not isinstance(manifest.get("experiment"), dict):
        raise ValueError("experiment.yaml is missing.")
    registered_steps = _registered_plan_steps(
        root,
        manifest["experiment"],
        rows,
        remote=remote,
        require_registered_rows=False,
    )
    selection_report = (
        _hparam_selection_report(root, remote=remote)
        if any(plan.get("task") == "hparam_tune" for step in registered_steps for plan in step["plans"])
        else None
    )
    checkpoint_audits = _hparam_checkpoint_audits(root, registered_steps, remote=remote)
    snapshot = tracking.experiment_status_snapshot(
        manifest["experiment"],
        registered_steps,
        rows,
        root=root,
        remote=remote,
        hparam_selection_report=selection_report,
        hparam_checkpoint_audits=checkpoint_audits,
    )
    blocking_codes = sorted(
        {blocker["code"] for blocker in snapshot["blockers"] if blocker["code"] == "unmaterialized_step"}
    )
    if blocking_codes:
        raise ValueError("Experiment cannot be finalized with incomplete canonical steps: " + ", ".join(blocking_codes))
    hparam = tracking.hparam_selection_lifecycle(
        registered_steps,
        rows,
        root=root,
        report=selection_report,
        checkpoint_audits=checkpoint_audits,
    )
    if hparam["pending_steps"]:
        raise ValueError("Successful hparam runs must be selected before experiment finalization.")
    selected_report = selection_report if hparam["selected_steps"] else None
    if hparam["selected_steps"] and (selected_report is None or not hparam["report_valid"]):
        raise ValueError("The hparam selection report is missing or differs from canonical selection evidence.")
    if selected_report is not None:
        _validate_hparam_selection_files_unchanged(root, selected_report, checkpoint_audits, remote=remote)
    report_path_is_selection = str(Path(report_path)) == hparam["report_path"]
    if selected_report is not None:
        if hparam["automatic_report_final"]:
            if not report_path_is_selection:
                raise ValueError(f"Successful hparam experiments must finalize from {hparam['report_path']}")
            report_text = str(selected_report["text"])
        elif report_path_is_selection:
            raise ValueError("The hparam selection report cannot replace the required combined experiment report.")
    elif hparam["hparam_steps"] and report_path_is_selection:
        raise ValueError("The hparam selection report cannot replace the required hparam failure report.")
    if not (hparam["selected_steps"] and hparam["automatic_report_final"]):
        report_text = exp_io.read_text_at(report_path, remote=remote)
        report_content_is_selection = bool(
            selection_report is not None
            and hashlib.sha256(report_text.encode()).hexdigest() == selection_report["sha256"]
        )
        if hparam["selected_steps"] and report_content_is_selection:
            raise ValueError("The hparam selection report cannot replace the required combined experiment report.")
        if not hparam["selected_steps"] and hparam["hparam_steps"] and report_content_is_selection:
            raise ValueError("The hparam selection report cannot replace the required hparam failure report.")
    if not report_text.strip():
        raise ValueError("Final report is missing or empty.")
    if hparam["selected_steps"]:
        _validate_hparam_checkpoints(rows, hparam["selected_steps"], remote=remote)
    target = root / "reports" / "final.md"
    exp_io.validate_managed_output_paths(
        root,
        [
            target,
            root / "experiment.yaml",
            root / "events.jsonl",
            run_manifest_path.with_name(run_manifest_path.name + ".lock"),
        ],
        remote=remote,
    )
    target_exists = exp_io.path_exists_at(target, remote=remote)
    target_text = exp_io.read_text_at(target, remote=remote) if target_exists else ""
    target_sha256 = hashlib.sha256(target_text.encode()).hexdigest() if target_exists else None
    manifest_sha256 = hashlib.sha256(manifest_text.encode()).hexdigest()
    report_sha256 = hashlib.sha256(report_text.encode()).hexdigest()
    if not exp_io.conditional_atomic_replace_text_at(
        target,
        report_text,
        target_sha256,
        managed_root=root,
        remote=remote,
        dependency_path=run_manifest_path,
        expected_dependency_sha256=run_manifest_snapshot["sha256"],
        guard_path=root / "experiment.yaml",
        expected_guard_sha256=manifest_sha256,
    ):
        raise RuntimeError(f"Final report changed during publication: {target}")
    manifest["experiment"].update(
        {
            "status": "completed",
            "completed_at": utc_now(),
            "final_report": str(target),
            "final_report_sha256": report_sha256,
        }
    )
    if selected_report is not None:
        manifest["experiment"]["selection_report_sha256"] = selected_report["sha256"]
    exp_io.append_event_at(root, "experiment_finalization_prepared", {"report": str(target)}, remote=remote)
    if selected_report is not None:
        _validate_hparam_selection_files_unchanged(root, selected_report, checkpoint_audits, remote=remote)
    if hparam["selected_steps"]:
        _validate_hparam_checkpoints(rows, hparam["selected_steps"], remote=remote)
    # The experiment manifest is the terminal commit, so publish it only after the report is durable.
    if not exp_io.conditional_atomic_replace_text_at(
        root / "experiment.yaml",
        yaml.safe_dump(manifest, sort_keys=False),
        manifest_sha256,
        managed_root=root,
        remote=remote,
        dependency_path=run_manifest_path,
        expected_dependency_sha256=run_manifest_snapshot["sha256"],
        guard_path=target,
        expected_guard_sha256=report_sha256,
    ):
        raise RuntimeError("Experiment or run manifest changed during finalization.")
    return target


def sync_wandb_runs(
    run_dir: str | Path,
    *,
    entity: str,
    project: str,
    group: str | None = None,
    remote: str | None = None,
) -> Path:
    root = _target_root(run_dir, remote)
    managed_rows = _managed_rows(root, remote=remote)
    existing_metrics = exp_io.read_rows_at(root / "metrics_manifest.tsv", remote=remote, require_managed_identity=True)
    _validate_evidence_rows(managed_rows, existing_metrics, "metrics_manifest.tsv")
    blocked = root / "reports" / "wandb_blocked.md"
    exp_io.validate_managed_output_paths(
        root,
        [
            blocked,
            root / "wandb" / "summaries.jsonl",
            root / "wandb" / "runs.tsv",
            root / "metrics_manifest.tsv",
            root / "experiment_manifest.tsv",
            root / "reports" / "wandb.md",
            root / "run_manifest.tsv",
            root / "run_matrix.csv",
            root / "reports" / "run_matrix.md",
            root / "events.jsonl",
        ],
        remote=remote,
    )
    try:
        runs = tracking.wandb_runs(entity, project, group)
    except Exception as exc:
        exp_io.write_text_at(blocked, f"# W&B Sync Blocked\n\n{type(exc).__name__}: {exc}\n", remote=remote)
        raise RuntimeError(f"W&B sync blocked; wrote {blocked}") from exc

    payloads = [tracking.wandb_run_payload(run, entity=entity, project=project) for run in runs]
    run_rows = [payload["run_row"] for payload in payloads]
    metric_rows = [row for payload in payloads for row in payload["metric_rows"]]
    summary_lines = [payload["summary_line"] for payload in payloads]
    observations = tracking.wandb_run_observations(managed_rows, run_rows)
    managed_metrics = tracking.managed_metric_rows(managed_rows, metric_rows)
    merged_metrics = tracking.merge_rows(existing_metrics, managed_metrics)
    exp_io.validate_managed_output_paths(
        root,
        [root / "wandb" / "history" / payload["history_filename"] for payload in payloads],
        remote=remote,
    )

    exp_io.mkdir_experiment_dirs(root, remote=remote)
    for payload in payloads:
        tracking.write_history_csv(
            root / "wandb" / "history" / payload["history_filename"],
            payload["history_rows"],
            remote=remote,
        )

    exp_io.write_text_at(
        root / "wandb" / "summaries.jsonl",
        "\n".join(summary_lines) + ("\n" if summary_lines else ""),
        remote=remote,
    )
    exp_io.write_rows_at(root / "wandb" / "runs.tsv", run_rows, remote=remote)
    metrics_path = root / "metrics_manifest.tsv"
    if merged_metrics:
        exp_io.write_rows_at(metrics_path, merged_metrics, remote=remote)
    else:
        exp_io.write_text_at(metrics_path, "step_id\trun_id\n", remote=remote)
    tracking.update_experiment_wandb(root, entity=entity, project=project, group=group or "", remote=remote)
    tracking.write_wandb_report(root, run_rows, remote=remote)
    merge_run_manifest(root, observations, remote=remote)
    return root / "wandb" / "runs.tsv"


def index_checkpoints(run_dir: str | Path, *, remote: str | None = None) -> Path:
    root = _target_root(run_dir, remote)
    managed_rows = _managed_rows(root, remote=remote)
    metrics_path = root / "metrics_manifest.tsv"
    checkpoint_path = root / "checkpoint_manifest.tsv"
    # Workspace location proves evidence ownership only when the table path is not an alias.
    exp_io.validate_managed_output_paths(root, [metrics_path, checkpoint_path], remote=remote)
    metrics = exp_io.read_rows_at(metrics_path, remote=remote, require_managed_identity=True)
    _validate_evidence_rows(managed_rows, metrics, "metrics_manifest.tsv")
    rows = tracking.checkpoint_rows(root, remote=remote)
    for row in rows:
        row.update(tracking.best_metric_for_checkpoint(row, metrics))
    validate_managed_run_rows(rows, source="checkpoint_manifest.tsv", cardinality="many_per_run")
    if rows:
        exp_io.write_rows_at(checkpoint_path, rows, remote=remote)
    else:
        exp_io.write_text_at(checkpoint_path, "step_id\trun_id\n", remote=remote)
    return root / "checkpoint_manifest.tsv"


def monitor_experiment(run_dir: str | Path, *, remote: str | None = None) -> dict[str, Any]:
    """Observe managed runs once and commit their current status without launching work.

    Requires an active, valid managed workspace. Reads local/remote process,
    scheduler and artifact evidence, merges observations into run_manifest.tsv
    and its projections, then writes reports/monitor.md. Returns run_dir, the
    committed runs and the report path; these are committed observations rather
    than an experiment_status lifecycle decision.

    remote selects the workspace host. Invalid ownership/state and unhandled
    observation or write errors propagate; a report-write failure can occur
    after the canonical manifest has already been updated."""
    root = _target_root(run_dir, remote)
    previous_rows = _managed_rows(root, remote=remote)
    report_path = root / "reports" / "monitor.md"
    exp_io.validate_managed_output_paths(
        root,
        [
            report_path,
            root / "run_manifest.tsv",
            root / "run_matrix.csv",
            root / "reports" / "run_matrix.md",
            root / "events.jsonl",
        ],
        remote=remote,
    )
    # Observe the owner-validated input snapshot; the commit still reads fresh under its lock/CAS.
    run_rows = previous_rows
    monitor_context = managed_scheduler.SlurmMonitorContext(run_rows, owner_dir=root, remote=remote)
    observations = [
        tracking.monitor_run_row(root, row, previous_rows, remote=remote, monitor_context=monitor_context)
        for row in run_rows
    ]
    committed = merge_run_manifest(root, observations, remote=remote)
    report = tracking.monitor_report(committed)
    exp_io.write_text_at(report_path, report, remote=remote)
    return {"run_dir": str(root), "runs": committed, "report": str(report_path)}


def experiment_status(run_dir: str | Path, *, remote: str | None = None) -> tracking.ExperimentStatusSnapshot:
    """Return a read-only lifecycle snapshot from the managed workspace's recorded state.

    Validates registered plan bindings and available selection/report/checkpoint
    evidence locally or on remote, including completed workspaces. Does not poll
    jobs or refresh canonical run statuses: live_observation is False.

    The mapping contains experiment identity, lifecycle_source, summary, steps,
    runs, blockers and advisory next actions, which do not authorize execution.
    Lifecycle blockers appear in that snapshot; invalid workspace/plan contracts and
    unhandled read errors raise. No reports or manifests are written."""
    root = _target_root(run_dir, remote)
    experiment, rows = _managed_workspace(
        root,
        remote=remote,
        allow_completed=True,
        validate_experiment_index=False,
    )
    registered_steps = _registered_plan_steps(root, experiment, rows, remote=remote, require_registered_rows=True)
    selection_report = (
        _hparam_selection_report(root, remote=remote)
        if any(plan.get("task") == "hparam_tune" for step in registered_steps for plan in step["plans"])
        else None
    )
    checkpoint_audits = _hparam_checkpoint_audits(root, registered_steps, remote=remote)

    return tracking.experiment_status_snapshot(
        experiment,
        registered_steps,
        rows,
        root=root,
        remote=remote,
        hparam_selection_report=selection_report,
        hparam_checkpoint_audits=checkpoint_audits,
    )


def _registered_plan_steps(
    root: Path,
    experiment: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    remote: str | None,
    require_registered_rows: bool,
) -> list[artifacts.RegisteredPlanStep]:
    step_manifests = read_registered_steps(root, experiment_id=str(experiment["id"]), remote=remote)
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
        run_index_offset = 0
        for plan_path in manifest["plans"]:
            if artifacts.is_registered_blocked_plan(plan_path, workspace=root, remote=remote):
                continue
            plan = artifacts.read_registered_plan(
                plan_path,
                workspace=root,
                workspace_experiment=experiment,
                step_manifest=manifest,
                workspace_rows=rows,
                expected_recipe_path=(
                    manifest["recipe_path"] if manifest["plan_controller"] == "ordinary" and not plans else None
                ),
                remote=remote,
                run_index_offset=run_index_offset,
            )
            plans.append(plan)
            plan_keys.extend(plan["run_keys"])
            run_index_offset += len(plan["run_keys"])
        if len(plan_keys) != len(set(plan_keys)):
            raise ValueError(f"Managed step registers duplicate run keys across plans: {step_id}")
        canonical_keys = {managed_run_key(row) for row in step_rows}
        if set(plan_keys) != canonical_keys:
            raise ValueError(f"Managed step plans differ from canonical run keys: {step_id}")
        registered_steps.append({"manifest": manifest, "plans": plans})

    return registered_steps


def _hparam_selection_report(root: Path, *, remote: str | None) -> tracking.HparamSelectionReportSnapshot | None:
    path = root / "reports" / "hparam_selection.md"
    ranking_path = root / "reports" / "ranking.csv"
    exp_io.validate_managed_output_paths(root, [path, ranking_path], remote=remote)
    if not exp_io.path_exists_at(path, remote=remote):
        return None
    read_paths: list[str | Path] = [path]
    if exp_io.path_exists_at(ranking_path, remote=remote):
        read_paths.append(ranking_path)
    files = exp_io.read_managed_files_at(root, read_paths, remote=remote)
    ranking = files.get(str(ranking_path))
    return {
        "path": str(path),
        **files[str(path)],
        "ranking_path": str(ranking_path),
        "ranking_text": ranking["text"] if ranking is not None else None,
        "ranking_sha256": ranking["sha256"] if ranking is not None else None,
    }


def _hparam_checkpoint_audits(
    root: Path,
    registered_steps: list[artifacts.RegisteredPlanStep],
    *,
    remote: str | None,
) -> dict[str, exp_io.ManagedFileSnapshot | None]:
    checkpoint_audit_paths = sorted(
        {
            Path(plan["path"]) / "checkpoint_test_ranking.csv"
            for registered in registered_steps
            for plan in registered["plans"]
            if plan["task"] == "hparam_tune" and plan["selection"] is not None and plan["selection"]["split"] == "test"
        }
    )
    if not checkpoint_audit_paths:
        return {}
    exp_io.validate_managed_output_paths(root, checkpoint_audit_paths, remote=remote)
    existing: list[str | Path] = [path for path in checkpoint_audit_paths if exp_io.path_exists_at(path, remote=remote)]
    files = exp_io.read_managed_files_at(root, existing, remote=remote)
    return {str(path): files.get(str(path)) for path in checkpoint_audit_paths}


def rank_experiment_candidates(run_dir: str | Path, *, metric: str, mode: str, remote: str | None = None) -> Path:
    root = _target_root(run_dir, remote)
    managed_rows = _managed_rows(root, remote=remote)
    out = root / "reports" / "experiment_ranking.csv"
    exp_io.validate_managed_output_paths(
        root,
        [
            out,
            root / "reports" / "experiment_ranking.md",
            root / "metrics_manifest.tsv",
            root / "checkpoint_manifest.tsv",
        ],
        remote=remote,
    )
    metric_rows = exp_io.read_rows_at(root / "metrics_manifest.tsv", remote=remote, require_managed_identity=True)
    checkpoint_rows = exp_io.read_rows_at(
        root / "checkpoint_manifest.tsv", remote=remote, require_managed_identity=True
    )
    _validate_evidence_rows(managed_rows, metric_rows, "metrics_manifest.tsv")
    _validate_evidence_rows(
        managed_rows,
        checkpoint_rows,
        "checkpoint_manifest.tsv",
        checkpoint_evidence=True,
        remote=remote,
    )
    run_rows = tracking.experiment_run_rows(root, remote=remote)
    rows = tracking.candidate_rows(run_rows, metric_rows, metric)
    ranked = tracking.rank_candidates(rows, checkpoint_rows, mode=mode)
    validate_managed_run_rows(ranked, source="experiment_ranking.csv", cardinality="one_per_run")
    if ranked:
        exp_io.write_rows_at(out, ranked, remote=remote)
    else:
        exp_io.write_text_at(out, "step_id,run_id\n", remote=remote)
    tracking.write_rank_report(root, metric, mode, ranked, remote=remote)
    return out


def _target_root(run_dir: str | Path, remote: str | None) -> Path:
    root = Path(run_dir)
    if remote:
        if not root.is_absolute():
            raise ValueError("Remote experiment root must be an absolute path.")
        return root
    return canonical_local_experiment_root(root, Path.cwd())


def _managed_rows(root: Path, *, remote: str | None) -> list[dict[str, str]]:
    _experiment, rows = _managed_workspace(root, remote=remote)
    return rows


def _managed_workspace(  # noqa: C901
    root: Path,
    *,
    remote: str | None,
    allow_completed: bool = False,
    validate_experiment_index: bool = True,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    manifest_path = root / "experiment.yaml"
    if not exp_io.path_exists_at(manifest_path, remote=remote):
        raise ValueError("experiment.yaml is missing. Initialize the experiment first.")
    files = exp_io.read_managed_files_at(root, [manifest_path], remote=remote)
    experiment_text = files[str(manifest_path)]["text"]
    manifest = read_managed_yaml_mapping(experiment_text, source=f"Managed experiment manifest {manifest_path}")
    if set(manifest) != {"experiment"}:
        raise ValueError("experiment.yaml must contain only the experiment owner mapping.")
    raw_experiment = manifest.get("experiment")
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
        bound_files = exp_io.read_managed_files_at(root, completed_bindings, remote=remote)
        if bound_files[str(completed_bindings[0])]["sha256"] != experiment["final_report_sha256"]:
            raise ValueError("Completed experiment final report differs from its terminal binding.")
        if len(completed_bindings) == 2 and (
            bound_files[str(completed_bindings[1])]["sha256"] != experiment["selection_report_sha256"]
        ):
            raise ValueError("Completed experiment selection report differs from its terminal binding.")

    for legacy_path in (root / "trial_status.tsv", root / "adaptive" / "trial_registry.tsv"):
        if exp_io.path_exists_at(legacy_path, remote=remote):
            raise ValueError(f"Historical experiment artifacts are read-only: {legacy_path}")

    experiment_manifest = root / "experiment_manifest.tsv"
    if validate_experiment_index and exp_io.path_exists_at(experiment_manifest, remote=remote):
        manifest_rows = exp_io.read_rows_at(experiment_manifest, remote=remote, strict=True)
        if len(manifest_rows) != 1:
            raise ValueError("experiment_manifest.tsv must contain exactly one row.")
        manifest_row = manifest_rows[0]
        if manifest_row.get("experiment_id") != experiment["id"]:
            raise ValueError("experiment_manifest.tsv belongs to a different experiment.")
        if manifest_row.get("experiment_root") != str(root):
            raise ValueError("experiment_manifest.tsv root differs from the target workspace.")

    rows = read_run_manifest(root, remote=remote)
    for row in rows:
        if row["experiment_id"] != experiment["id"]:
            raise ValueError("run_manifest.tsv contains a run owned by a different experiment.")
    return experiment, rows


def _validate_evidence_rows(
    managed_rows: list[dict[str, Any]],
    evidence_rows: list[dict[str, Any]],
    source: str,
    *,
    checkpoint_evidence: bool = False,
    remote: str | None = None,
) -> None:
    validate_managed_run_rows(evidence_rows, source=source, cardinality="many_per_run")
    managed_by_key = {managed_run_key(row): row for row in managed_rows}
    for row in evidence_rows:
        managed = managed_by_key.get(managed_run_key(row))
        if managed is None:
            raise ValueError(f"{source} contains a run outside the canonical manifest.")
        validate_frozen_run_update(managed, row)
    if checkpoint_evidence:
        tracking.validate_checkpoint_evidence_rows(managed_rows, evidence_rows, remote=remote)
