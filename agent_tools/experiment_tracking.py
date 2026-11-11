"""Projections, status advice, and hparam ranking validation over recorded evidence.

Layer 0 leaf. Consumes the run manifest plus ``experiment_sources`` checkpoint
evidence checks and produces the experiment status snapshot, monitor results,
and the hparam-selection lifecycle view.

Recipe fields are consistency guards here; ``run_manifest.tsv`` remains the
lifecycle owner, so nothing in this module advances a run's status.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import csv
import io
import json
from pathlib import Path
import re
import shlex
from typing import Any, TypedDict

from . import experiment_sources, managed_scheduler, run_artifacts as artifacts, run_evidence as evidence
from .experiment_workspace import (
    SUCCESS_STATUSES,
    TERMINAL_STATUSES,
    managed_run_key,
    managed_run_parameters,
    resolve_run_row,
    scheduler_direct_controller,
    scheduler_type,
    stopped_runs_without_reason,
    validate_scheduler_run_identity,
    validated_run_key,
)
from .manifests import utc_now
from .models import JsonValue


class HparamSelectionReportSnapshot(TypedDict):
    path: str
    text: str
    sha256: str
    ranking_path: str
    ranking_text: str | None
    ranking_sha256: str | None


class _HparamSelectionStepCore(TypedDict):
    step_id: str
    selection: artifacts.RegisteredPlanSelection
    rows: list[dict[str, str]]


class HparamSelectionReportStep(_HparamSelectionStepCore, total=False):
    ranked: list[dict[str, Any]]
    legacy_selection: bool


class HparamSelectionStep(HparamSelectionReportStep):
    plan_path: str
    plans: list[artifacts.RegisteredPlanSummary]


class HparamSelectionLifecycle(TypedDict):
    hparam_steps: list[HparamSelectionStep]
    pending_steps: list[HparamSelectionStep]
    selected_steps: list[HparamSelectionStep]
    expected_report: str | None
    report_path: str
    report_valid: bool
    automatic_report_final: bool


class _ExperimentStatusActionCore(TypedDict):
    id: str
    step_id: str | None
    control_host: str | None
    reason: str
    advisory: bool
    mutates: bool
    requires_authorization: bool
    argv: list[str]


class ExperimentStatusAction(_ExperimentStatusActionCore, total=False):
    required_inputs: list[str]


class ExperimentStatusDecision(TypedDict):
    manual_choice_required: bool
    recommended_next: ExperimentStatusAction | None
    other_legal_actions: list[ExperimentStatusAction]
    blocked_actions: list[str]


class ExperimentStatusBlocker(TypedDict):
    code: str
    step_id: str | None
    run_ids: list[str]
    message: str
    blocked_actions: list[str]


class ExperimentStatusSummary(TypedDict):
    state: str
    run_count: int
    status_counts: dict[str, int]


class ExperimentStatusMetadata(TypedDict):
    id: str
    title: str
    root: str
    status: str
    remote: str | None


class ExperimentStatusStep(TypedDict):
    id: str
    phase: str
    purpose: str
    plan_controller: str
    plans: list[str]
    status_counts: dict[str, int]


class ExperimentStatusExecution(TypedDict):
    target: str | None
    host: str | None


class ExperimentStatusScheduler(TypedDict):
    type: str | None
    job_id: str | None
    cluster: str | None
    node: str | None
    raw_state: str | None
    reason: str | None
    observed_at: str | None


class ExperimentStatusProcess(TypedDict):
    pid: str | None
    process_group_id: str | None
    identity_error: str | None
    monitored_at: str | None


class ExperimentStatusEvidence(TypedDict):
    health_status: str | None
    gpu_summary: str | None
    log_age_seconds: str | None
    checkpoint_dir: str | None
    checkpoint_count: str | None
    run_manifest: str | None


class ExperimentStatusRun(TypedDict):
    step_id: str
    run_id: str
    run_name: str
    parameter_summary: str
    status: str
    execution: ExperimentStatusExecution
    scheduler: ExperimentStatusScheduler
    process: ExperimentStatusProcess
    evidence: ExperimentStatusEvidence
    blockers: list[str]


class ExperimentMonitorResult(TypedDict):
    run_dir: str
    runs: list[dict[str, str]]
    report: str


class ExperimentStatusSnapshot(TypedDict):
    experiment: ExperimentStatusMetadata
    lifecycle_source: str
    live_observation: bool
    summary: ExperimentStatusSummary
    steps: list[ExperimentStatusStep]
    runs: list[ExperimentStatusRun]
    blockers: list[ExperimentStatusBlocker]
    decision: ExperimentStatusDecision


checkpoint_evidence_host = experiment_sources.checkpoint_evidence_host
validate_checkpoint_evidence_rows = experiment_sources.validate_checkpoint_evidence_rows

HPARAM_SELECTION_METADATA_FIELDS = {
    "selection_task",
    "selection_mode",
    "selection_split",
    "selection_report",
    "selection_report_sha256",
}
HPARAM_SELECTION_RESULT_FIELDS = {
    "score",
    "rank",
    "checkpoint_path",
    "checkpoint_sha256",
    "epoch",
    "source",
}
_HPARAM_RANKING_BASE_FIELDS = (
    "run_name",
    "parameter_summary",
    "version",
    "config",
    "metric",
    "score",
    "rank",
    "checkpoint_path",
    "checkpoint_sha256",
)
_HPARAM_RANKING_OPTIONAL_FIELDS = ("epoch", "run_manifest", "source", "status")


def monitor_run_row(
    root: Path,
    row: Mapping[str, str],
    previous_rows: Sequence[Mapping[str, str]],
    *,
    remote: str | None = None,
    monitor_context: managed_scheduler.SlurmMonitorContext | None = None,
) -> dict[str, JsonValue]:
    previous = resolve_run_row(previous_rows, row) or {}
    observation_row = dict(row)
    transport_override = False
    if remote and not observation_row.get("host"):
        transport_override = True
        observation_row["target"] = "ssh"
        observation_row["host"] = remote
    scheduler_row = previous if previous.get("scheduler_type") not in (None, "") else row
    if scheduler_type(scheduler_row) == "slurm":
        has_execution_identity = any(source.get("target") not in (None, "") for source in (row, previous))
        if has_execution_identity:
            execution: dict[str, JsonValue] = {"target": observation_row.get("target") or "local"}
            if execution["target"] == "ssh":
                execution["host"] = observation_row.get("host")
            if scheduler_direct_controller(scheduler_row):
                execution["scheduler"] = {"direct_controller": True}
            status = managed_scheduler.observe_slurm_run(
                root,
                execution,
                observation_row,
                health=True,
                monitor_context=None if transport_override else monitor_context,
            )
        else:
            status = {**observation_row}
    else:
        has_managed_script = any(source.get("script") not in (None, "") for source in (row, previous))
        has_process_identity = any(source.get("pid_path") not in (None, "") for source in (row, previous))
        has_execution_evidence = any(
            source.get(field) not in (None, "") for source in (row, previous) for field in ("pid_path", "state")
        )
        # Hparam scripts expose process identity for monitor-owned exit inference; lifecycle scripts self-commit.
        legacy_script_owner = has_managed_script and not has_process_identity
        owner_row = previous if previous.get("terminal_status_owner") not in (None, "") else row
        script_commits_terminal_status = managed_scheduler.script_commits_terminal_status(
            owner_row,
            default=legacy_script_owner,
        )
        if (
            (previous.get("status") or row.get("status")) == "running"
            and has_managed_script
            and not has_execution_evidence
        ):
            status = {
                "step_id": row["step_id"],
                "run_id": row["run_id"],
                "status": "running",
                "health_status": "running",
                "monitored_at": utc_now(),
            }
        else:
            status = evidence.status_row(
                root,
                observation_row,
                previous,
                script_commits_terminal_status=script_commits_terminal_status,
                health=True,
            )
    if status.get("status") == "finished":
        status["status"] = "completed"
    if status.get("health_status") == "finished":
        status["health_status"] = "completed"
    observation_fields = evidence.RUN_STATUS_FIELDS | {
        "scheduler_job_id",
        "scheduler_cluster",
        "scheduler_raw_state",
        "scheduler_reason",
        "scheduler_node",
        "scheduler_exit_code",
        "scheduler_observed_at",
        "scheduler_started_at",
        "scheduler_priority",
        "scheduler_nice",
        "scheduler_partition",
        "scheduler_account",
        "scheduler_qos",
        "scheduler_reservation",
        "scheduler_submit_time",
        "scheduler_eligible_time",
        "scheduler_start_time",
        "scheduler_time_limit",
        "scheduler_requested_nodes",
        "scheduler_features",
        "scheduler_requested_tres",
        "scheduler_tres_per_node",
        "scheduler_health_error",
        "scheduler_queue_age_seconds",
        "scheduler_allocation_age_seconds",
    }
    observation = {
        "step_id": row["step_id"],
        "run_id": row["run_id"],
        **{field: status[field] for field in observation_fields if field in status},
    }
    if transport_override:
        observation.pop("target", None)
        observation.pop("host", None)
    return observation


def monitor_report(rows: Sequence[Mapping[str, Any]]) -> str:
    lines = ["# Experiment Monitor", ""]
    if not rows:
        return "# Experiment Monitor\n\nNo runs found.\n"
    lines.append("| run | setting | status | scheduler | health | gpu | log age | checkpoints |")
    lines.append("|---|---|---|---|---|---|---:|---:|")
    for row in rows:
        scheduler = " ".join(
            f"{label}={row[field]}"
            for label, field in (
                ("job", "scheduler_job_id"),
                ("state", "scheduler_raw_state"),
                ("reason", "scheduler_reason"),
                ("node", "scheduler_node"),
                ("priority", "scheduler_priority"),
            )
            if row.get(field) not in (None, "")
        )
        lines.append(
            "| {run} | {setting} | {status} | {scheduler} | {health} | {gpu} | {log_age} | {ckpts} |".format(
                run=f"{row.get('step_id', '')} / {row.get('run_id', '')} — {row.get('run_name', '')}",
                setting=str(row.get("parameter_summary", "")).replace("|", "/"),
                status=row.get("status", ""),
                scheduler=scheduler.replace("|", "/"),
                health=row.get("health_status", ""),
                gpu=str(row.get("gpu_summary", "")).replace("|", "/"),
                log_age=row.get("log_age_seconds", ""),
                ckpts=row.get("checkpoint_count", ""),
            )
        )
    lines.append("")
    return "\n".join(lines)


def _experiment_lifecycle_decision(
    *,
    completed: bool,
    rows: Sequence[Mapping[str, str]],
    sorted_rows: Sequence[Mapping[str, str]],
    plan_blockers: list[ExperimentStatusBlocker],
    missing_stop_reason_rows: Sequence[Mapping[str, str]],
    hparam: HparamSelectionLifecycle,
    candidates: list[ExperimentStatusAction],
    blockers: list[ExperimentStatusBlocker],
    root: Path,
    remote: str | None,
) -> tuple[str, ExperimentStatusDecision]:
    decision: ExperimentStatusDecision = {
        "manual_choice_required": False,
        "recommended_next": None,
        "other_legal_actions": [],
        "blocked_actions": [],
    }
    if completed:
        state = "completed"
    elif not rows:
        state = "empty"
        blockers.append(
            _status_blocker(
                "no_managed_runs",
                "The experiment has no canonical managed runs. No plan or launch command can be inferred.",
            )
        )
        decision["manual_choice_required"] = True
    else:
        uncertain = [
            row
            for row in sorted_rows
            if row["status"] in {"unknown_scheduler", "unknown_remote", "missing_pid", "submitting"}
        ]
        active = [row for row in sorted_rows if row["status"] in managed_scheduler.ACTIVE_STATUSES]
        if uncertain:
            state = "blocked"
            groups = sorted({(str(row["status"]), str(row["step_id"])) for row in uncertain})
            for status, step_id in groups:
                matching = [row for row in uncertain if str(row["status"]) == status and str(row["step_id"]) == step_id]
                blockers.append(
                    _status_blocker(
                        status,
                        _status_reason(matching),
                        rows=matching,
                        blocked_actions=["finalize", "launch", "resubmit"],
                    )
                )
            uncertain_keys = {managed_run_key(row) for row in uncertain}
            remaining_active = [row for row in active if managed_run_key(row) not in uncertain_keys]
            for step_id in sorted({str(row["step_id"]) for row in remaining_active}):
                blockers.append(
                    _status_blocker(
                        "active_runs",
                        "Canonical runs are still active. Refresh recorded evidence before another lifecycle decision.",
                        rows=[row for row in remaining_active if str(row["step_id"]) == step_id],
                        blocked_actions=["finalize", "launch", "resubmit"],
                    )
                )
            decision["recommended_next"] = _monitor_action(root, remote)
        elif active:
            state = "in_progress"
            for step_id in sorted({str(row["step_id"]) for row in active}):
                blockers.append(
                    _status_blocker(
                        "active_runs",
                        "Canonical runs are still active. Refresh recorded evidence before another lifecycle decision.",
                        rows=[row for row in active if str(row["step_id"]) == step_id],
                        blocked_actions=["finalize", "launch", "resubmit"],
                    )
                )
            decision["recommended_next"] = _monitor_action(root, remote)
        elif all(row["status"] in TERMINAL_STATUSES for row in sorted_rows):
            if plan_blockers or missing_stop_reason_rows:
                state = "blocked"
                decision["manual_choice_required"] = True
            elif hparam["pending_steps"]:
                state = "ready_to_select"
                select_actions = [
                    _hparam_select_action(step["step_id"], step["plan_path"], remote)
                    for step in hparam["pending_steps"]
                ]
                for step in hparam["pending_steps"]:
                    blockers.append(
                        _status_blocker(
                            "hparam_selection_required",
                            "Successful terminal hparam runs must be ranked before finalization.",
                            step_id=step["step_id"],
                            blocked_actions=["finalize"],
                        )
                    )
                if len(select_actions) == 1:
                    decision["recommended_next"] = select_actions[0]
                else:
                    decision["manual_choice_required"] = True
                    decision["other_legal_actions"] = select_actions
            elif hparam["selected_steps"] and not hparam["report_valid"]:
                state = "ready_to_report"
                blockers.append(
                    _status_blocker(
                        "hparam_selection_report_required",
                        "The deterministic hparam selection report is missing or differs from "
                        "canonical selection evidence.",
                        blocked_actions=["finalize"],
                    )
                )
                first = hparam["selected_steps"][0]
                decision["recommended_next"] = _hparam_select_action(first["step_id"], first["plan_path"], remote)
            elif hparam["selected_steps"] and hparam["automatic_report_final"]:
                state = "ready_to_finalize"
                decision["recommended_next"] = _finalize_action(root, hparam["report_path"], remote)
            elif hparam["selected_steps"]:
                state = "ready_to_report"
                blockers.append(
                    _status_blocker(
                        "combined_report_required",
                        "The hparam selection report does not cover every ordinary experiment step, so "
                        "finalization requires a non-empty combined report.",
                        blocked_actions=["finalize"],
                    )
                )
                finalize = _finalize_action(root, "{report_path}", remote)
                finalize["required_inputs"] = ["report_path"]
                decision["manual_choice_required"] = True
                decision["other_legal_actions"] = [finalize]
            elif hparam["hparam_steps"]:
                state = "ready_to_report"
                blockers.append(
                    _status_blocker(
                        "failure_report_required",
                        "No hparam run completed successfully; finalization requires a non-empty failure report.",
                        blocked_actions=["finalize"],
                    )
                )
                finalize = _finalize_action(root, "{report_path}", remote)
                finalize["required_inputs"] = ["report_path"]
                decision["manual_choice_required"] = True
                decision["other_legal_actions"] = [finalize]
            else:
                state = "ready_to_finalize"
                finalize = _finalize_action(root, "{report_path}", remote)
                finalize["required_inputs"] = ["report_path"]
                # The report is experiment-wide human input, not evidence attached to any managed run.
                blockers.append(
                    _status_blocker(
                        "final_report_required",
                        "Finalization requires a user-selected non-empty report path.",
                    )
                )
                decision["manual_choice_required"] = True
                decision["other_legal_actions"] = [finalize]
        else:
            state = "ready_to_launch" if candidates else "blocked"
            if len(candidates) == 1:
                decision["recommended_next"] = candidates[0]
            elif len(candidates) > 1:
                decision["manual_choice_required"] = True
                decision["other_legal_actions"] = candidates
            else:
                decision["manual_choice_required"] = True
    return state, decision


def experiment_status_snapshot(
    experiment: dict[str, Any],
    registered_steps: list[artifacts.RegisteredPlanStep],
    rows: list[dict[str, str]],
    *,
    root: Path,
    remote: str | None = None,
    hparam_selection_report: HparamSelectionReportSnapshot | None = None,
) -> ExperimentStatusSnapshot:
    allowed_statuses = TERMINAL_STATUSES | managed_scheduler.ACTIVE_STATUSES | managed_scheduler.LAUNCHABLE_STATUSES
    for row in rows:
        validate_scheduler_run_identity(row)
        status = row.get("status")
        if status not in allowed_statuses:
            raise ValueError(
                f"run_manifest.tsv contains an unsupported status: {row.get('step_id')} / "
                f"{row.get('run_id')}: {status}"
            )
    sorted_rows = sorted(rows, key=lambda row: (str(row["step_id"]), str(row["run_id"])))
    plan_blockers, candidates = _plan_advice(registered_steps, sorted_rows, remote=remote)
    hparam = hparam_selection_lifecycle(
        registered_steps,
        sorted_rows,
        root=root,
        report=hparam_selection_report,
    )
    missing_stop_reason_rows = stopped_runs_without_reason(sorted_rows)
    completed = experiment.get("status") == "completed"
    if completed:
        if not rows or any(row["status"] not in TERMINAL_STATUSES for row in rows):
            raise ValueError("Completed experiment metadata conflicts with canonical run lifecycle state.")
        if missing_stop_reason_rows:
            raise ValueError("Completed experiment metadata conflicts with stopped runs missing stop_reason.")
        if plan_blockers:
            raise ValueError("Completed experiment metadata cannot be verified for unmaterialized registered steps.")
        has_terminal_report_binding = experiment.get("final_report_sha256") not in (None, "")
        # A modern terminal binding makes disappearing selection evidence corruption, not legacy state.
        incomplete_steps = (
            hparam["pending_steps"]
            if has_terminal_report_binding
            else [step for step in hparam["pending_steps"] if not step.get("legacy_selection")]
        )
        if incomplete_steps or (hparam["selected_steps"] and not hparam["report_valid"]):
            raise ValueError("Completed experiment metadata conflicts with incomplete hparam selection evidence.")
        if hparam["selected_steps"] and not has_terminal_report_binding:
            raise ValueError("Completed hparam experiment metadata is missing terminal report bindings.")
        if has_terminal_report_binding:
            terminal_selection_sha256 = experiment.get("selection_report_sha256")
            current_selection_sha256 = (
                hparam_selection_report.get("sha256") if hparam_selection_report is not None else None
            )
            if hparam["selected_steps"] and terminal_selection_sha256 != current_selection_sha256:
                raise ValueError("Completed experiment metadata conflicts with its hparam selection report binding.")
            if not hparam["selected_steps"] and terminal_selection_sha256 not in (None, ""):
                raise ValueError("Completed experiment metadata has an unexpected hparam selection report binding.")
    row_payloads = [_status_run_payload(row) for row in sorted_rows]
    step_payloads: list[ExperimentStatusStep] = []
    for registered in sorted(registered_steps, key=lambda item: str(item["manifest"]["step"]["id"])):
        manifest = registered["manifest"]
        step_id = str(manifest["step"]["id"])
        step_rows = [row for row in sorted_rows if str(row["step_id"]) == step_id]
        step_payloads.append(
            {
                "id": step_id,
                "phase": str(manifest["step"]["phase"]),
                "purpose": str(manifest["step"]["purpose"]),
                "plan_controller": str(manifest["plan_controller"]),
                "plans": sorted(plan["path"] for plan in registered["plans"]),
                "status_counts": _status_counts(step_rows),
            }
        )

    blockers = list(plan_blockers)
    for step_id in sorted({str(row["step_id"]) for row in missing_stop_reason_rows}):
        blockers.append(
            _status_blocker(
                "missing_stop_reason",
                "Stopped canonical runs require a non-empty recorded stop_reason before finalization.",
                rows=[row for row in missing_stop_reason_rows if str(row["step_id"]) == step_id],
                blocked_actions=["finalize"],
            )
        )
    state, decision = _experiment_lifecycle_decision(
        completed=completed,
        rows=rows,
        sorted_rows=sorted_rows,
        plan_blockers=plan_blockers,
        missing_stop_reason_rows=missing_stop_reason_rows,
        hparam=hparam,
        candidates=candidates,
        blockers=blockers,
        root=root,
        remote=remote,
    )

    blocker_codes_by_key: dict[tuple[str, str], list[str]] = {
        (str(row["step_id"]), str(row["run_id"])): [] for row in sorted_rows
    }
    for blocker in blockers:
        for key in blocker_codes_by_key:
            if blocker["step_id"] == key[0] and key[1] in blocker["run_ids"]:
                blocker_codes_by_key[key].append(blocker["code"])
    for payload in row_payloads:
        payload["blockers"] = sorted(blocker_codes_by_key[(payload["step_id"], payload["run_id"])])
    decision["blocked_actions"] = sorted(
        {
            *decision["blocked_actions"],
            *(action for blocker in blockers for action in blocker["blocked_actions"]),
        }
    )

    return {
        "experiment": {
            "id": str(experiment["id"]),
            "title": str(experiment["title"]),
            "root": str(root),
            "status": "completed" if completed else "active",
            "remote": remote,
        },
        "lifecycle_source": str(root / "run_manifest.tsv"),
        "live_observation": False,
        "summary": {
            "state": state,
            "run_count": len(sorted_rows),
            "status_counts": _status_counts(sorted_rows),
        },
        "steps": step_payloads,
        "runs": row_payloads,
        "blockers": sorted(
            blockers,
            key=lambda blocker: (blocker["code"], blocker["step_id"] or "", blocker["run_ids"]),
        ),
        "decision": decision,
    }


def hparam_selection_lifecycle(
    registered_steps: list[artifacts.RegisteredPlanStep],
    rows: list[dict[str, str]],
    *,
    root: Path,
    report: HparamSelectionReportSnapshot | None = None,
) -> HparamSelectionLifecycle:
    hparam_run_keys = {
        tuple(key)
        for registered in registered_steps
        for plan in registered["plans"]
        if plan.get("task") == "hparam_tune"
        for key in plan["run_keys"]
    }
    # Controller-owned hparam plans may carry selection metadata; non-hparam runs never may.
    misowned_selection = sorted(
        validated_run_key(row)
        for row in rows
        if managed_run_key(row) not in hparam_run_keys
        and any(row.get(field) not in (None, "") for field in HPARAM_SELECTION_METADATA_FIELDS)
    )
    if misowned_selection:
        rendered = ", ".join(f"{step_id} / {run_id}" for step_id, run_id in misowned_selection)
        raise ValueError(f"Canonical hparam selection metadata is not owned by a registered hparam plan: {rendered}")
    rows_by_step: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        rows_by_step.setdefault(str(row["step_id"]), []).append(row)
    hparam_steps: list[HparamSelectionStep] = []
    pending_steps: list[HparamSelectionStep] = []
    selected_steps: list[HparamSelectionStep] = []
    for registered in sorted(registered_steps, key=lambda item: str(item["manifest"]["step"]["id"])):
        manifest = registered["manifest"]
        if manifest["plan_controller"] != "ordinary":
            continue
        plans = [plan for plan in registered["plans"] if plan.get("task") == "hparam_tune"]
        if not plans:
            continue
        step_id = str(manifest["step"]["id"])
        selections = {tuple(sorted((plan.get("selection") or {}).items())) for plan in plans}
        if len(selections) != 1:
            raise ValueError(f"Registered hparam plans disagree on selection policy: {step_id}")
        selection = dict(next(iter(selections)))
        if not selection.get("metric") or selection.get("mode") not in {"min", "max"} or not selection.get("split"):
            raise ValueError(f"Registered hparam plan has an incomplete selection policy: {step_id}")
        plan_keys = {tuple(key) for plan in plans for key in plan["run_keys"]}
        step: HparamSelectionStep = {
            "step_id": step_id,
            "plan_path": min(str(plan["path"]) for plan in plans),
            "plans": plans,
            "selection": {
                "metric": str(selection["metric"]),
                "mode": str(selection["mode"]),
                "split": str(selection["split"]),
            },
            "rows": sorted(
                [row for row in rows_by_step.get(step_id, []) if managed_run_key(row) in plan_keys],
                key=lambda row: str(row["run_id"]),
            ),
        }
        hparam_steps.append(step)
        if not any(row["status"] in SUCCESS_STATUSES for row in step["rows"]):
            stale_fields = (
                "selection_task",
                "metric",
                "selection_mode",
                "selection_split",
                "score",
                "rank",
                "checkpoint_path",
                "checkpoint_sha256",
                "selection_report",
                "selection_report_sha256",
            )
            if any(row.get(field) not in (None, "") for row in step["rows"] for field in stale_fields):
                raise ValueError(f"Canonical hparam selection evidence is stale for all-failed step {step_id}")
            continue
        has_selection_metadata = any(
            row.get(field) not in (None, "") for row in step["rows"] for field in HPARAM_SELECTION_METADATA_FIELDS
        )
        if not has_selection_metadata:
            step["legacy_selection"] = True
            pending_steps.append(step)
            continue
        if not all(
            row.get("selection_task") == "hparam_tune"
            and row.get("selection_mode") not in (None, "")
            and row.get("selection_split") not in (None, "")
            for row in step["rows"]
        ):
            raise ValueError(f"Canonical hparam selection metadata is only partially materialized for step {step_id}")
        ranked = validated_hparam_ranking(step)
        if ranked is None:
            pending_steps.append(step)
        else:
            step["ranked"] = ranked
            selected_steps.append(step)

    expected_report = hparam_selection_report_text(selected_steps, root=root) if selected_steps else None
    report_path = str(root / "reports" / "hparam_selection.md")
    report_valid = bool(
        expected_report is not None
        and report is not None
        and report.get("path") == report_path
        and report.get("text") == expected_report
        and _hparam_ranking_matches(selected_steps, report.get("ranking_text"))
        and all(
            str(row.get("selection_report") or "") == report_path
            and str(row.get("selection_report_sha256") or "") == str(report.get("sha256") or "")
            for step in selected_steps
            for row in step["rows"]
        )
    )
    automatic_report_final = (
        bool(hparam_steps)
        and len(selected_steps) == len(hparam_steps)
        and all(
            registered["manifest"]["plan_controller"] == "ordinary"
            and bool(registered["plans"])
            and all(plan.get("task") == "hparam_tune" for plan in registered["plans"])
            for registered in registered_steps
        )
    )
    return {
        "hparam_steps": hparam_steps,
        "pending_steps": pending_steps,
        "selected_steps": selected_steps,
        "expected_report": expected_report,
        "report_path": report_path,
        "report_valid": report_valid,
        "automatic_report_final": automatic_report_final,
    }


def _hparam_ranking_matches(selected_steps: Sequence[HparamSelectionReportStep], ranking_text: str | None) -> bool:
    ranked_rows = [row for step in selected_steps for row in step["ranked"]]
    expected_rows = hparam_ranking_projection(ranked_rows)
    if not expected_rows:
        return False
    expected_fields = sorted(expected_rows[0])
    if not isinstance(ranking_text, str):
        return False
    try:
        reader = csv.DictReader(io.StringIO(ranking_text), strict=True)
        fieldnames = reader.fieldnames or []
        if fieldnames != expected_fields:
            return False
        ranking_rows = list(reader)
    except csv.Error:
        return False
    if any(None in row or any(value is None for value in row.values()) for row in ranking_rows):
        return False
    expected = [tuple(str(row.get(field) or "") for field in expected_fields) for row in expected_rows]
    observed = [tuple(row[field] for field in expected_fields) for row in ranking_rows]
    return observed == expected


def hparam_ranking_projection(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = sorted(
        rows,
        key=lambda row: (str(row["step_id"]), int(row["rank"]), str(row["run_id"])),
    )
    parameter_fields = sorted({field for row in rows for field in managed_run_parameters(row)})
    fields = [
        "step_id",
        "run_id",
        *_HPARAM_RANKING_BASE_FIELDS,
        *parameter_fields,
        *(field for field in _HPARAM_RANKING_OPTIONAL_FIELDS if any(row.get(field) not in (None, "") for row in rows)),
    ]
    return [{field: row.get(field, "") for field in fields} for row in rows]


def hparam_selection_report_text(selected_steps: Sequence[HparamSelectionReportStep], *, root: Path) -> str:
    lines = [
        "# Hyper-parameter Selection",
        "",
        "Best observed candidates within the frozen search domain, metric, split, and budget.",
        "This report does not claim a global optimum or authorize external-test access.",
        "",
    ]
    ranking_path = root / "reports" / "ranking.csv"
    for step in sorted(selected_steps, key=lambda item: item["step_id"]):
        ranked = step["ranked"]
        winner = ranked[0]
        overrides = json.dumps(
            managed_run_parameters(winner),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        selection = step["selection"]
        lines.extend(
            [
                f"## Step `{step['step_id']}`",
                "",
                f"- Selection metric: `{selection['metric']}`",
                f"- Selection mode: `{selection['mode']}`",
                f"- Selection split: `{selection['split']}`",
                f"- Evaluated candidates: `{len(ranked)}/{len(step['rows'])}`",
                f"- Winner run: `{winner['run_id']}`",
                f"- Winner checkpoint: `{winner['checkpoint_path']}`",
            ]
        )
        if winner.get("checkpoint_sha256") not in (None, ""):
            lines.append(f"- Winner checkpoint SHA-256: `{winner['checkpoint_sha256']}`")
        lines.extend(
            [
                f"- Winner score: `{winner['score']}`",
                f"- Parameter summary: `{winner.get('parameter_summary', '')}`",
                f"- Search overrides: `{overrides}`",
                f"- Frozen config: `{winner['config']}`",
                f"- Frozen config SHA-256: `{winner['config_sha256']}`",
                f"- Frozen script: `{winner['script']}`",
                f"- Frozen script SHA-256: `{winner['script_sha256']}`",
                f"- Ranking: `{ranking_path}`",
                "",
            ]
        )
    return "\n".join(lines)


def validated_hparam_ranking(step: HparamSelectionReportStep) -> list[dict[str, Any]] | None:
    rows = step["rows"]
    selection = step["selection"]
    evidence_fields = ("metric", "selection_mode", "selection_split", "rank", "score", "checkpoint_path")
    if not any(any(row.get(field) not in (None, "") for field in evidence_fields) for row in rows):
        return None
    ranked: list[dict[str, Any]] = []
    for row in rows:
        for field, expected in (
            ("metric", selection["metric"]),
            ("selection_mode", selection["mode"]),
            ("selection_split", selection["split"]),
        ):
            if str(row.get(field) or "") != expected:
                raise ValueError(f"Canonical hparam selection {field} differs for {step['step_id']} / {row['run_id']}")
        rank = row.get("rank")
        if rank is None or rank == "":
            if any(row.get(field) not in (None, "") for field in HPARAM_SELECTION_RESULT_FIELDS):
                raise ValueError(f"Canonical hparam selection is incomplete for {step['step_id']} / {row['run_id']}")
            continue
        try:
            rank_number = int(rank)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Canonical hparam rank is invalid for {step['step_id']} / {row['run_id']}") from exc
        score = artifacts.float_or_none(row.get("score"))
        if (
            rank_number < 1
            or score is None
            or str(row.get("run_manifest") or "") != str(Path(str(row["runtime_dir"])) / "run_manifest.json")
            or row.get("source") not in (None, "")
            or row.get("checkpoint_path") in (None, "")
            or re.fullmatch(r"[0-9a-f]{64}", str(row.get("checkpoint_sha256") or "")) is None
            or row.get("config") in (None, "")
            or re.fullmatch(r"[0-9a-f]{64}", str(row.get("config_sha256") or "")) is None
            or row.get("script") in (None, "")
            or re.fullmatch(r"[0-9a-f]{64}", str(row.get("script_sha256") or "")) is None
            or row["status"] not in SUCCESS_STATUSES
        ):
            raise ValueError(f"Canonical hparam selection evidence is invalid for {step['step_id']} / {row['run_id']}")
        ranked.append({**row, "rank": rank_number})
    ranked.sort(key=lambda row: row["rank"])
    if [row["rank"] for row in ranked] != list(range(1, len(ranked) + 1)):
        raise ValueError(f"Canonical hparam ranks are incomplete or duplicated for step {step['step_id']}")
    scores = [float(row["score"]) for row in ranked]
    reverse = selection["mode"] == "max"
    if any((left < right if reverse else left > right) for left, right in zip(scores, scores[1:])):
        raise ValueError(f"Canonical hparam ranks disagree with selection mode for step {step['step_id']}")
    return ranked or None


def _hparam_select_action(step_id: str, plan_path: str, remote: str | None) -> ExperimentStatusAction:
    return _status_action(
        "hparam-select",
        "Rank terminal successful candidates and regenerate the deterministic selection report.",
        ["python", "-m", "agent_tools", "hparam-select", "--run-dir", plan_path],
        step_id=step_id,
        control_host=remote,
    )


def _finalize_action(root: Path, report_path: str, remote: str | None) -> ExperimentStatusAction:
    return _status_action(
        "experiment-finalize",
        "Publish the verified non-empty experiment report.",
        [
            "python",
            "-m",
            "agent_tools",
            "experiment-finalize",
            "--run-dir",
            str(root),
            "--report",
            report_path,
            *(["--remote", remote] if remote else []),
        ],
    )


def format_experiment_status(snapshot: ExperimentStatusSnapshot) -> str:
    experiment = snapshot["experiment"]
    lines = [
        f"# Experiment Status: {experiment['title']}",
        "",
        f"- Experiment: `{experiment['id']}`",
        f"- Root: `{experiment['root']}`",
        f"- State: `{snapshot['summary']['state']}`",
        f"- Lifecycle source: `{snapshot['lifecycle_source']}`",
        "- Evidence mode: recorded evidence, not live",
        "",
    ]
    if snapshot["steps"]:
        lines.extend(
            [
                "| Step | Phase | Plan controller | Status counts |",
                "|---|---|---|---|",
            ]
        )
        for step in snapshot["steps"]:
            counts = ", ".join(f"{status}={count}" for status, count in sorted(step["status_counts"].items()))
            lines.append(
                "| {step} | {phase} | {controller} | {counts} |".format(
                    step=_table_text(step["id"]),
                    phase=_table_text(step["phase"]),
                    controller=_table_text(step["plan_controller"]),
                    counts=_table_text(counts),
                )
            )
        lines.append("")
    if snapshot["runs"]:
        lines.extend(
            [
                "| Run | Canonical | Execution transport | Scheduler | Process | Checkpoints | Runtime manifest | "
                "Blocker |",
                "|---|---|---|---|---|---:|---|---|",
            ]
        )
        for run in snapshot["runs"]:
            execution = run["execution"]
            execution_text = execution["target"] or ""
            if execution["host"]:
                execution_text += f":{execution['host']}"
            scheduler = run["scheduler"]
            scheduler_text = " ".join(
                f"{name}={value}"
                for name, value in (
                    ("job", scheduler["job_id"]),
                    ("node", scheduler["node"]),
                    ("state", scheduler["raw_state"]),
                    ("reason", scheduler["reason"]),
                    ("observed", scheduler["observed_at"]),
                )
                if value is not None
            )
            process = run["process"]
            process_text = " ".join(
                f"{name}={value}"
                for name, value in (
                    ("pid", process["pid"]),
                    ("pgid", process["process_group_id"]),
                    ("error", process["identity_error"]),
                    ("observed", process["monitored_at"]),
                )
                if value is not None
            )
            lines.append(
                "| {run_id} | {status} | {execution} | {scheduler} | {process} | {checkpoints} | {test} | "
                "{blocker} |".format(
                    run_id=_table_text(f"{run['step_id']} / {run['run_id']} — {run['run_name']}"),
                    status=_table_text(run["status"]),
                    execution=_table_text(execution_text),
                    scheduler=_table_text(scheduler_text),
                    process=_table_text(process_text),
                    checkpoints=_table_text(run["evidence"]["checkpoint_count"]),
                    test=_table_text(run["evidence"]["run_manifest"]),
                    blocker=_table_text(", ".join(run["blockers"])),
                )
            )
        lines.append("")
    else:
        lines.extend(["No canonical managed runs.", ""])

    if snapshot["blockers"]:
        lines.extend(["## Blockers", ""])
        for blocker in snapshot["blockers"]:
            scope = []
            if blocker["step_id"] is not None:
                scope.append(f"step={blocker['step_id']}")
            if blocker["run_ids"]:
                scope.append(f"runs={', '.join(blocker['run_ids'])}")
            scope_text = f" [{'; '.join(scope)}]" if scope else ""
            lines.append(f"- `{blocker['code']}`{scope_text}: {blocker['message']}")
        lines.append("")

    lines.extend(["## Next legal action", ""])
    decision = snapshot["decision"]
    actions = []
    if decision["recommended_next"] is not None:
        actions.append(decision["recommended_next"])
    actions.extend(decision["other_legal_actions"])
    if decision["manual_choice_required"]:
        lines.append("Manual choice required.")
    if actions:
        for action in actions:
            control_host = action.get("control_host")
            host_text = f" (control host: `{control_host}`)" if control_host else ""
            lines.append(f"- `{action['id']}`{host_text}: `{shlex.join(action['argv'])}` — {action['reason']}")
            if action.get("required_inputs"):
                lines.append(f"  Required inputs: {', '.join(action['required_inputs'])}")
    else:
        lines.append("No advisory command is available from the current canonical state.")
    if decision["blocked_actions"]:
        lines.append("Blocked actions: " + ", ".join(decision["blocked_actions"]))
    lines.extend(["", "Advisory only; this output does not authorize execution.", ""])
    return "\n".join(lines)


def _plan_advice(
    registered_steps: list[artifacts.RegisteredPlanStep],
    rows: Sequence[Mapping[str, str]],
    *,
    remote: str | None = None,
) -> tuple[list[ExperimentStatusBlocker], list[ExperimentStatusAction]]:
    rows_by_key = {managed_run_key(row): row for row in rows}
    blockers = []
    candidates = []
    for registered in sorted(registered_steps, key=lambda item: str(item["manifest"]["step"]["id"])):
        manifest = registered["manifest"]
        step_id = str(manifest["step"]["id"])
        if not registered["plans"]:
            blockers.append(
                _status_blocker(
                    "unmaterialized_step",
                    "The registered step has no materialized plan or canonical runs; experiment-status cannot "
                    "prove controller completion.",
                    step_id=step_id,
                    blocked_actions=["finalize"],
                )
            )
            continue
        for plan in sorted(registered["plans"], key=lambda item: str(item["path"])):
            plan_rows = [rows_by_key[key] for key in plan["run_keys"]]
            if not any(row["status"] in managed_scheduler.LAUNCHABLE_STATUSES for row in plan_rows):
                continue
            if plan["task"] == "hparam_tune":
                argv = [
                    "python",
                    "-m",
                    "agent_tools",
                    "hparam-run-queue",
                    "--plan-dir",
                    plan["path"],
                    "--execute",
                ]
                action_id = "hparam-run-queue"
                control_host = remote
            elif plan["task"] == "infer" and all(scheduler_type(row) == "slurm" for row in plan_rows):
                argv = ["python", "-m", "agent_tools", "infer-launch", "--plan-dir", plan["path"], "--execute"]
                action_id = "infer-launch"
                control_host = remote
            elif plan["task"] == "preset_prepare" and (plan["recipe"].get("execution") or {}).get("scheduler") == {
                "type": "direct"
            }:
                argv = ["python", "-m", "agent_tools", "preset-launch", "--plan-dir", plan["path"], "--execute"]
                action_id = "preset-launch"
                control_host = remote
            else:
                argv = ["bash", plan["launch_script"]]
                action_id = "run-plan"
                control_host = remote
            candidates.append(
                _status_action(
                    action_id,
                    "Launch only after explicit user authorization and the command's own preflight succeeds.",
                    argv,
                    step_id=step_id,
                    control_host=control_host,
                )
            )
    candidates.sort(key=lambda action: (action["step_id"] or "", action["id"], action["argv"]))
    return blockers, candidates


def _status_run_payload(row: Mapping[str, str]) -> ExperimentStatusRun:
    return {
        "step_id": str(row["step_id"]),
        "run_id": str(row["run_id"]),
        "run_name": str(row.get("run_name") or ""),
        "parameter_summary": str(row.get("parameter_summary") or ""),
        "status": str(row["status"]),
        "execution": {
            "target": _optional_text(row.get("target")),
            "host": _optional_text(row.get("host")),
        },
        "scheduler": {
            "type": _optional_text(row.get("scheduler_type")),
            "job_id": _optional_text(row.get("scheduler_job_id")),
            "cluster": _optional_text(row.get("scheduler_cluster")),
            "node": _optional_text(row.get("scheduler_node")),
            "raw_state": _optional_text(row.get("scheduler_raw_state")),
            "reason": _optional_text(row.get("scheduler_reason")),
            "observed_at": _optional_text(row.get("scheduler_observed_at")),
        },
        "process": {
            "pid": _optional_text(row.get("pid")),
            "process_group_id": _optional_text(row.get("process_group_id")),
            "identity_error": _optional_text(row.get("process_identity_error")),
            "monitored_at": _optional_text(row.get("monitored_at")),
        },
        "evidence": {
            "health_status": _optional_text(row.get("health_status")),
            "gpu_summary": _optional_text(row.get("gpu_summary")),
            "log_age_seconds": _optional_text(row.get("log_age_seconds")),
            "checkpoint_dir": _optional_text(row.get("checkpoint_dir")),
            "checkpoint_count": _optional_text(row.get("checkpoint_count")),
            "run_manifest": _optional_text(row.get("run_manifest")),
        },
        "blockers": [],
    }


def _status_counts(rows: Sequence[Mapping[str, str]]) -> dict[str, int]:
    return dict(sorted(Counter(str(row["status"]) for row in rows).items()))


def _status_reason(rows: Sequence[Mapping[str, str]]) -> str:
    reasons = sorted({str(row["scheduler_reason"]) for row in rows if row.get("scheduler_reason") not in (None, "")})
    return "; ".join(reasons) if reasons else "Canonical execution identity is uncertain and must be refreshed."


def _status_blocker(
    code: str,
    message: str,
    *,
    step_id: str | None = None,
    rows: Sequence[Mapping[str, str]] | None = None,
    blocked_actions: list[str] | None = None,
) -> ExperimentStatusBlocker:
    rows = rows or []
    step_ids = sorted({str(row["step_id"]) for row in rows})
    return {
        "code": code,
        "step_id": step_id or (step_ids[0] if len(step_ids) == 1 else None),
        "run_ids": sorted(str(row["run_id"]) for row in rows),
        "message": message,
        "blocked_actions": sorted(blocked_actions or []),
    }


def _monitor_action(root: Path, remote: str | None) -> ExperimentStatusAction:
    argv = ["python", "-m", "agent_tools", "experiment-monitor", "--run-dir", str(root)]
    if remote:
        argv.extend(["--remote", remote])
    return _status_action(
        "experiment-monitor",
        "Refresh canonical lifecycle evidence.",
        argv,
    )


def _status_action(
    action_id: str,
    reason: str,
    argv: list[str],
    *,
    step_id: str | None = None,
    control_host: str | None = None,
) -> ExperimentStatusAction:
    return {
        "id": action_id,
        "step_id": step_id,
        "control_host": control_host,
        "reason": reason,
        "advisory": True,
        "mutates": True,
        "requires_authorization": True,
        "argv": argv,
    }


def _optional_text(value: Any) -> str | None:
    return None if value in (None, "") else str(value)


def _table_text(value: Any) -> str:
    return "" if value is None else str(value).replace("|", "/")
