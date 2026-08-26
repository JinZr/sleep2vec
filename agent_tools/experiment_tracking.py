from __future__ import annotations

from collections import Counter
import csv
import io
import json
import math
from pathlib import Path
import re
import shlex
import stat
import subprocess
from typing import Any

from . import (
    experiment_io as exp_io,
    managed_scheduler,
    run_artifacts as artifacts,
    run_evidence as evidence,
    transport,
)
from .experiment_workspace import (
    SUCCESS_STATUSES,
    TERMINAL_STATUSES,
    managed_run_key,
    managed_run_parameters,
    merge_run_row,
    read_managed_yaml_mapping,
    read_run_manifest,
    resolve_external_run_row,
    resolve_run_row,
    scheduler_direct_controller,
    scheduler_type,
    stopped_runs_without_reason,
    validate_checkpoint_ownership,
    validate_frozen_run_update,
    validate_managed_run_rows,
    validate_scheduler_run_identity,
)
from .manifests import read_json, utc_now
from .models import json_ready

WANDB_RUN_FIELDS = {
    "status",
    "state",
    "wandb_run_id",
    "wandb_url",
    "wandb_entity",
    "wandb_project",
    "wandb_group",
    "created_at",
    "updated_at",
}


def wandb_runs(entity: str, project: str, group: str | None) -> list[Any]:
    import wandb

    api = wandb.Api()
    filters = {"group": group} if group else None
    return list(api.runs(f"{entity}/{project}", filters=filters))


def wandb_run_payload(run: Any, *, entity: str, project: str) -> dict[str, Any]:
    wandb_run_id = str(getattr(run, "id", ""))
    version = str(getattr(run, "name", "") or wandb_run_id)
    run_group = str(getattr(run, "group", "") or "")
    summary = _safe_dict(getattr(run, "summary", {}))
    config = _safe_dict(getattr(run, "config", {}))
    url = str(getattr(run, "url", "") or "")
    state = str(getattr(run, "state", "") or "")
    status = {
        "finished": "completed",
        "failed": "failed",
        "crashed": "failed",
        "killed": "stopped",
        "running": "running",
    }.get(state)
    row = {
        "version": version,
        "state": state,
        "wandb_run_id": wandb_run_id,
        "wandb_url": url,
        "wandb_entity": entity,
        "wandb_project": project,
        "wandb_group": run_group,
        "created_at": str(getattr(run, "created_at", "") or ""),
        "updated_at": str(getattr(run, "updated_at", "") or ""),
    }
    for field in ("experiment_id", "step_id", "run_id"):
        if config.get(field) not in (None, ""):
            row[field] = str(config[field])
    if status:
        row["status"] = status
    metric_rows = []
    for metric, value in summary.items():
        if _is_scalar_number(value):
            metric_rows.append(
                {
                    **{field: row[field] for field in ("experiment_id", "step_id", "run_id") if field in row},
                    "version": version,
                    "epoch": _summary_epoch(summary),
                    "split": _metric_split(metric),
                    "metric": metric,
                    "value": value,
                    "source": "wandb_summary",
                    "metric_scope": _metric_scope(metric),
                    "wandb_run_id": wandb_run_id,
                    "updated_at": utc_now(),
                }
            )
    history_rows = _history_rows_for_run(run)
    metric_rows.extend(_history_metric_rows(wandb_run_id, version, row, history_rows))
    return {
        "run_row": row,
        "metric_rows": metric_rows,
        "summary_line": json.dumps(json_ready({"run": row, "summary": summary}), sort_keys=True),
        "history_rows": history_rows,
        "history_filename": f"{_safe_filename(wandb_run_id or version)}.csv",
    }


def update_experiment_wandb(root: Path, *, entity: str, project: str, group: str, remote: str | None = None) -> None:
    path = root / "experiment_manifest.tsv"
    rows = exp_io.read_rows_at(path, remote=remote)
    if not rows:
        experiment_path = root / "experiment.yaml"
        manifest = read_managed_yaml_mapping(
            exp_io.read_text_at(experiment_path, remote=remote),
            source=f"Managed experiment manifest {experiment_path}",
        )
        experiment = manifest.get("experiment") if isinstance(manifest, dict) else {}
        rows = [
            {
                "experiment_id": experiment["id"],
                "experiment_root": str(root),
                "remote_host": "",
                "task": "",
                "selection_metric": "",
                "selection_mode": "",
                "created_at": utc_now(),
            }
        ]
    rows[0].update(
        {
            "wandb_entity": entity,
            "wandb_project": project,
            "wandb_group": group,
            "updated_at": utc_now(),
        }
    )
    exp_io.write_rows_at(path, rows, remote=remote)


def wandb_run_observations(run_rows: list[dict[str, Any]], wandb_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    validate_managed_run_rows(run_rows, source="run_manifest.tsv", cardinality="one_per_run")
    observations: dict[tuple[str, str], dict[str, Any]] = {}
    wandb_run_ids: dict[tuple[str, str], str] = {}
    for row in wandb_rows:
        if "trial_id" in row:
            raise ValueError("Historical trial_id W&B evidence is unsupported.")
        existing = resolve_external_run_row(run_rows, row)
        if existing is None:
            continue
        key = managed_run_key(existing)
        incoming_wandb_run_id = str(row.get("wandb_run_id") or "")
        known_wandb_run_id = wandb_run_ids.get(key) or str(existing.get("wandb_run_id") or "")
        if incoming_wandb_run_id and known_wandb_run_id and incoming_wandb_run_id != known_wandb_run_id:
            raise ValueError(
                f"Ambiguous W&B runs for managed run {key[0]} / {key[1]}: "
                f"{known_wandb_run_id} and {incoming_wandb_run_id}"
            )
        if incoming_wandb_run_id:
            wandb_run_ids[key] = incoming_wandb_run_id
        fields = WANDB_RUN_FIELDS
        if scheduler_type(existing) == "slurm" or existing.get("terminal_status_owner") == "scheduler_sidecar":
            fields = fields - {"status"}
        update = {
            "step_id": key[0],
            "run_id": key[1],
            **{field: row[field] for field in fields if field in row},
        }
        observations[key] = merge_run_row(observations.get(key, {}), update)
    rows = list(observations.values())
    validate_managed_run_rows(rows, source="W&B run observations", cardinality="one_per_run")
    return rows


def experiment_run_rows(root: Path, *, remote: str | None = None) -> list[dict[str, Any]]:
    return read_run_manifest(root, remote=remote)


def managed_metric_rows(run_rows: list[dict[str, Any]], metric_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    validate_managed_run_rows(run_rows, source="run_manifest.tsv", cardinality="one_per_run")
    rows = []
    wandb_run_ids: dict[tuple[str, str], str] = {}
    for metric_row in metric_rows:
        if "trial_id" in metric_row:
            raise ValueError("Historical trial_id metric evidence is unsupported.")
        run_row = resolve_external_run_row(run_rows, metric_row)
        if run_row is None:
            continue
        key = managed_run_key(run_row)
        incoming_wandb_run_id = str(metric_row.get("wandb_run_id") or "")
        known_wandb_run_id = wandb_run_ids.get(key) or str(run_row.get("wandb_run_id") or "")
        if incoming_wandb_run_id and known_wandb_run_id and incoming_wandb_run_id != known_wandb_run_id:
            raise ValueError(
                f"Ambiguous W&B runs for managed run {key[0]} / {key[1]}: "
                f"{known_wandb_run_id} and {incoming_wandb_run_id}"
            )
        if incoming_wandb_run_id:
            wandb_run_ids[key] = incoming_wandb_run_id
        row = dict(metric_row)
        row.update(
            {
                field: run_row[field]
                for field in ("experiment_id", "step_id", "run_id", "run_name", "parameter_summary", "version")
                if run_row.get(field) not in (None, "")
            }
        )
        rows.append(row)
    validate_managed_run_rows(rows, source="managed metrics", cardinality="many_per_run")
    return rows


def checkpoint_rows(root: Path, *, remote: str | None = None) -> list[dict[str, Any]]:
    previous_rows = exp_io.read_rows_at(root / "checkpoint_manifest.tsv", remote=remote, require_managed_identity=True)
    validate_managed_run_rows(previous_rows, source="checkpoint_manifest.tsv", cardinality="many_per_run")
    runs = read_run_manifest(root, remote=remote)
    eligible_runs = []
    for run in runs:
        if "runtime_dir" not in run or "checkpoint_dir" not in run:
            raise ValueError(f"Managed run is missing frozen artifact paths: {run['step_id']} / {run['run_id']}")
        if bool(run["runtime_dir"]) != bool(run["checkpoint_dir"]):
            raise ValueError(f"Managed run has partial frozen artifact paths: {run['step_id']} / {run['run_id']}")
        if run["runtime_dir"]:
            eligible_runs.append(run)
    eligible_keys = {managed_run_key(run) for run in eligible_runs}
    runs_by_key = {managed_run_key(run): run for run in runs}
    for row in previous_rows:
        run = runs_by_key.get(managed_run_key(row))
        if run is None:
            raise ValueError(
                f"Checkpoint manifest row does not belong to an eligible managed run: "
                f"{row['step_id']} / {row['run_id']}"
            )
        if managed_run_key(run) not in eligible_keys:
            raise ValueError(
                f"Checkpoint manifest row does not belong to an eligible managed run: "
                f"{row['step_id']} / {row['run_id']}"
            )
        evidence_host = checkpoint_evidence_host(run, remote)
        validate_frozen_run_update(run, row, require_checkpoint_ownership=evidence_host is None)
        if evidence_host:
            validate_checkpoint_ownership(run, row)
    previous_checkpoint_keys = {
        managed_run_key(row) for row in previous_rows if row.get("checkpoint_path") not in (None, "")
    }
    for run in eligible_runs:
        if managed_run_key(run) not in previous_checkpoint_keys:
            continue
        evidence_host = checkpoint_evidence_host(run, remote)
        if evidence_host:
            try:
                runtime_exists = exp_io.path_exists_at(run["runtime_dir"], remote=evidence_host)
                checkpoint_exists = exp_io.path_exists_at(run["checkpoint_dir"], remote=evidence_host)
            except RuntimeError as exc:
                raise RuntimeError(f"SSH checkpoint scan failed on {evidence_host}: {exc}") from exc
            if not runtime_exists or not checkpoint_exists:
                raise RuntimeError(
                    f"SSH checkpoint scan found a missing frozen artifact directory with existing inventory on "
                    f"{evidence_host}: {run['step_id']} / {run['run_id']}"
                )
        else:
            for field in ("runtime_dir", "checkpoint_dir"):
                try:
                    info = Path(str(run[field])).lstat()
                except FileNotFoundError as exc:
                    raise ValueError(
                        f"Managed {field} is missing for a run with existing checkpoint inventory: "
                        f"{run['step_id']} / {run['run_id']} / {run[field]}"
                    ) from exc
                if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
                    raise ValueError(
                        f"Managed {field} is not a directory: " f"{run['step_id']} / {run['run_id']} / {run[field]}"
                    )
    validate_checkpoint_evidence_rows(runs, previous_rows, remote=remote)
    if not eligible_runs:
        return []
    rows = []
    for run in eligible_runs:
        evidence_host = checkpoint_evidence_host(run, remote)
        if evidence_host:
            rows.extend(_remote_checkpoint_rows([run], evidence_host))
        else:
            rows.extend(_local_checkpoint_rows([run]))
    validate_managed_run_rows(rows, source="checkpoint scan", cardinality="many_per_run")
    return rows


def best_metric_for_checkpoint(row: dict[str, Any], metrics: list[dict[str, str]]) -> dict[str, Any]:
    epoch = artifacts.epoch_number(row.get("epoch"))
    key = managed_run_key(row)
    same_run = [item for item in metrics if managed_run_key(item) == key]
    for item in same_run:
        validate_frozen_run_update(row, item)
    matches = [
        item
        for item in same_run
        if artifacts.epoch_number(item.get("epoch")) == epoch and item.get("metric_scope") == "validation"
    ]
    if not matches:
        return {"metric": "", "value": ""}
    chosen = matches[0]
    return {"metric": chosen.get("metric", ""), "value": chosen.get("value", "")}


def monitor_run_row(
    root: Path,
    row: dict[str, Any],
    previous_rows: list[dict[str, str]],
    *,
    remote: str | None = None,
) -> dict[str, Any]:
    previous = resolve_run_row(previous_rows, row) or {}
    observation_row = dict(row)
    transport_override = bool(remote and not observation_row.get("host"))
    if transport_override:
        observation_row["target"] = "ssh"
        observation_row["host"] = remote
    scheduler_row = previous if previous.get("scheduler_type") not in (None, "") else row
    if scheduler_type(scheduler_row) == "slurm":
        has_execution_identity = any(source.get("target") not in (None, "") for source in (row, previous))
        if has_execution_identity:
            execution = {"target": observation_row.get("target") or "local"}
            if execution["target"] == "ssh":
                execution["host"] = observation_row.get("host")
            if scheduler_direct_controller(scheduler_row):
                execution["scheduler"] = {"direct_controller": True}
            status = managed_scheduler.observe_slurm_run(root, execution, observation_row, health=True)
        else:
            status = observation_row
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


def candidate_rows(
    run_rows: list[dict[str, Any]], metric_rows: list[dict[str, str]], metric: str
) -> list[dict[str, Any]]:
    validate_managed_run_rows(run_rows, source="run_manifest.tsv", cardinality="one_per_run")
    validate_managed_run_rows(metric_rows, source="metrics_manifest.tsv", cardinality="many_per_run")
    runs_by_key = {managed_run_key(run): run for run in run_rows}
    owned_metrics = []
    for metric_row in metric_rows:
        run_row = runs_by_key.get(managed_run_key(metric_row))
        if run_row is None:
            raise ValueError(
                f"Metric row is not managed by run_manifest.tsv: "
                f"{metric_row.get('step_id', '')} / {metric_row.get('run_id', '')}"
            )
        validate_frozen_run_update(run_row, metric_row)
        owned_metrics.append((metric_row, run_row))
    rows = []
    for metric_row, run_row in owned_metrics:
        if metric_row.get("metric") != metric:
            continue
        score = artifacts.float_or_none(metric_row.get("value"))
        if score is None:
            continue
        rows.append(
            {
                "experiment_id": run_row.get("experiment_id", ""),
                "step_id": run_row["step_id"],
                "run_id": run_row["run_id"],
                "run_name": run_row.get("run_name", ""),
                "parameter_summary": run_row.get("parameter_summary", ""),
                "version": run_row.get("version", ""),
                "epoch": metric_row.get("epoch", ""),
                "metric": metric,
                "score": score,
                "metric_scope": metric_row.get("metric_scope") or _metric_scope(metric),
                "source": metric_row.get("source", ""),
                "wandb_run_id": metric_row.get("wandb_run_id", ""),
            }
        )
    validate_managed_run_rows(rows, source="candidate metrics", cardinality="many_per_run")
    return rows


def rank_candidates(
    rows: list[dict[str, Any]], checkpoints: list[dict[str, str]], *, mode: str
) -> list[dict[str, Any]]:
    validate_managed_run_rows(rows, source="candidate metrics", cardinality="many_per_run")
    validate_managed_run_rows(checkpoints, source="checkpoint_manifest.tsv", cardinality="many_per_run")
    for checkpoint in checkpoints:
        run = resolve_run_row(rows, checkpoint)
        if run is not None:
            validate_frozen_run_update(run, checkpoint)
    reverse = mode == "max"
    ranked = _best_rows(rows, mode=mode)
    for row in ranked:
        row["checkpoint_path"] = _checkpoint_for_metric_row(row, checkpoints)
    ranked = artifacts.assign_ranks(ranked, key="score", reverse=reverse)
    validate_managed_run_rows(ranked, source="experiment ranking", cardinality="one_per_run")
    return ranked


def write_history_csv(path: Path, rows: list[dict[str, Any]], *, remote: str | None = None) -> None:
    if not rows:
        exp_io.write_rows_at(path, [], remote=remote)
        return
    fieldnames = sorted({key for row in rows for key in row})
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    exp_io.write_text_at(path, buffer.getvalue(), remote=remote)


def merge_rows(existing: list[dict[str, str]], new_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    validate_managed_run_rows(existing, source="metrics_manifest.tsv", cardinality="many_per_run")
    validate_managed_run_rows(new_rows, source="incoming metrics", cardinality="many_per_run")
    order = []
    by_key = {}
    for row in [*existing, *new_rows]:
        key = tuple(
            str(row.get(field, ""))
            for field in ("step_id", "run_id", "version", "epoch", "metric", "source", "wandb_run_id")
        )
        if key not in by_key:
            order.append(key)
        by_key[key] = row
    return [by_key[key] for key in order]


def monitor_report(rows: list[dict[str, Any]]) -> str:
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


def experiment_status_snapshot(
    experiment: dict[str, Any],
    registered_steps: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    *,
    root: Path,
    remote: str | None = None,
    hparam_selection_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
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
            raise ValueError(
                "Completed experiment metadata cannot be verified for adaptive or pipeline plans, or for "
                "unmaterialized registered steps."
            )
        new_pending_steps = [step for step in hparam["pending_steps"] if not step.get("legacy_selection")]
        if new_pending_steps or (hparam["selected_steps"] and not hparam["report_valid"]):
            raise ValueError("Completed experiment metadata conflicts with incomplete hparam selection evidence.")
    row_payloads = [_status_run_payload(row) for row in sorted_rows]
    step_payloads = []
    for registered in sorted(registered_steps, key=lambda item: str(item["manifest"]["step"]["id"])):
        manifest = registered["manifest"]
        step_id = str(manifest["step"]["id"])
        step_rows = [row for row in sorted_rows if str(row["step_id"]) == step_id]
        step_payloads.append(
            {
                "id": step_id,
                "phase": str(manifest["step"]["phase"]),
                "purpose": str(manifest["step"]["purpose"]),
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
    decision = {
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
                        blocked_actions=["adaptive_advance", "finalize", "launch", "resubmit"],
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
                        blocked_actions=["adaptive_advance", "finalize", "launch", "resubmit"],
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
                        blocked_actions=["adaptive_advance", "finalize", "launch", "resubmit"],
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

    blocker_codes_by_key = {(str(row["step_id"]), str(row["run_id"])): [] for row in sorted_rows}
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
    registered_steps: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    *,
    root: Path,
    report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rows_by_step: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        rows_by_step.setdefault(str(row["step_id"]), []).append(row)
    hparam_steps = []
    pending_steps = []
    selected_steps = []
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
        step = {
            "step_id": step_id,
            "plan_path": min(str(plan["path"]) for plan in plans),
            "selection": selection,
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
            row.get(field) not in (None, "")
            for row in step["rows"]
            for field in (
                "selection_task",
                "selection_mode",
                "selection_split",
                "selection_report",
                "selection_report_sha256",
            )
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
        ranked = _validated_hparam_ranking(step)
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


def _hparam_ranking_matches(selected_steps: list[dict[str, Any]], ranking_text: Any) -> bool:
    if not isinstance(ranking_text, str):
        return False
    try:
        reader = csv.DictReader(io.StringIO(ranking_text), strict=True)
        fieldnames = reader.fieldnames or []
        required = {"step_id", "run_id", "metric", "score", "rank", "checkpoint_path", "checkpoint_sha256"}
        if len(fieldnames) != len(set(fieldnames)) or not required.issubset(fieldnames):
            return False
        ranking_rows = list(reader)
    except csv.Error:
        return False
    if any(None in row or any(value is None for value in row.values()) for row in ranking_rows):
        return False

    fields = ("metric", "score", "rank", "checkpoint_path", "checkpoint_sha256")
    expected = {
        managed_run_key(row): tuple(str(row.get(field) or "") for field in fields)
        for step in selected_steps
        for row in step["ranked"]
    }
    observed = {managed_run_key(row): tuple(str(row.get(field) or "") for field in fields) for row in ranking_rows}
    return len(observed) == len(ranking_rows) and None not in observed and observed == expected


def hparam_selection_report_text(selected_steps: list[dict[str, Any]], *, root: Path) -> str:
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


def _validated_hparam_ranking(step: dict[str, Any]) -> list[dict[str, Any]] | None:
    rows = step["rows"]
    selection = step["selection"]
    evidence_fields = ("metric", "selection_mode", "selection_split", "rank", "score", "checkpoint_path")
    if not any(any(row.get(field) not in (None, "") for field in evidence_fields) for row in rows):
        return None
    ranked = []
    for row in rows:
        for field, expected in (
            ("metric", selection["metric"]),
            ("selection_mode", selection["mode"]),
            ("selection_split", selection["split"]),
        ):
            if str(row.get(field) or "") != expected:
                raise ValueError(f"Canonical hparam selection {field} differs for {step['step_id']} / {row['run_id']}")
        rank = row.get("rank")
        if rank in (None, ""):
            if row.get("score") not in (None, "") or row.get("checkpoint_path") not in (None, ""):
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
    scores = [artifacts.float_or_none(row["score"]) for row in ranked]
    reverse = selection["mode"] == "max"
    if any((left < right if reverse else left > right) for left, right in zip(scores, scores[1:])):
        raise ValueError(f"Canonical hparam ranks disagree with selection mode for step {step['step_id']}")
    return ranked or None


def _hparam_select_action(step_id: str, plan_path: str, remote: str | None) -> dict[str, Any]:
    return _status_action(
        "hparam-select",
        "Rank terminal successful candidates and regenerate the deterministic selection report.",
        ["python", "-m", "agent_tools", "hparam-select", "--run-dir", plan_path],
        step_id=step_id,
        execution_host=remote,
    )


def _finalize_action(root: Path, report_path: str, remote: str | None) -> dict[str, Any]:
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


def format_experiment_status(snapshot: dict[str, Any]) -> str:
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
    if snapshot["runs"]:
        lines.extend(
            [
                "| Run | Canonical | Scheduler | Process | Checkpoints | Runtime manifest | Blocker |",
                "|---|---|---|---|---:|---|---|",
            ]
        )
        for run in snapshot["runs"]:
            scheduler = run["scheduler"]
            scheduler_text = " ".join(
                f"{name}={value}"
                for name, value in (
                    ("job", scheduler["job_id"]),
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
                "| {run_id} | {status} | {scheduler} | {process} | {checkpoints} | {test} | {blocker} |".format(
                    run_id=_table_text(f"{run['step_id']} / {run['run_id']} — {run['run_name']}"),
                    status=_table_text(run["status"]),
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
            execution_host = action.get("execution_host")
            host_text = f" (execution host: `{execution_host}`)" if execution_host else ""
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
    registered_steps: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    *,
    remote: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows_by_key = {managed_run_key(row): row for row in rows}
    blockers = []
    candidates = []
    for registered in sorted(registered_steps, key=lambda item: str(item["manifest"]["step"]["id"])):
        manifest = registered["manifest"]
        step_id = str(manifest["step"]["id"])
        plan_controller = manifest["plan_controller"]
        if plan_controller in {"adaptive", "pipeline"}:
            code = f"{plan_controller}_phase_deferred"
            message = (
                f"Experiment-status cannot verify {plan_controller} controller completion or interpret its eligibility."
            )
            blocked_actions = [f"{plan_controller}_advance", "finalize"]
            if registered["plans"]:
                for plan in sorted(registered["plans"], key=lambda item: str(item["path"])):
                    blockers.append(
                        _status_blocker(
                            code,
                            message,
                            rows=[rows_by_key[tuple(key)] for key in plan["run_keys"]],
                            blocked_actions=blocked_actions,
                        )
                    )
            else:
                blockers.append(
                    _status_blocker(
                        code,
                        message,
                        step_id=step_id,
                        blocked_actions=blocked_actions,
                    )
                )
            continue
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
            plan_rows = [rows_by_key[tuple(key)] for key in plan["run_keys"]]
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
                execution_host = remote
            else:
                argv = ["bash", plan["launch_script"]]
                action_id = "run-plan"
                hosts = sorted({str(row["host"]) for row in plan_rows if row.get("host") not in (None, "")})
                execution_host = remote or (hosts[0] if len(hosts) == 1 else None)
            candidates.append(
                _status_action(
                    action_id,
                    "Launch only after explicit user authorization and the command's own preflight succeeds.",
                    argv,
                    step_id=step_id,
                    execution_host=execution_host,
                )
            )
    candidates.sort(key=lambda action: (action["step_id"] or "", action["id"], action["argv"]))
    return blockers, candidates


def _status_run_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "step_id": str(row["step_id"]),
        "run_id": str(row["run_id"]),
        "run_name": str(row.get("run_name") or ""),
        "parameter_summary": str(row.get("parameter_summary") or ""),
        "status": str(row["status"]),
        "scheduler": {
            "type": _optional_text(row.get("scheduler_type")),
            "job_id": _optional_text(row.get("scheduler_job_id")),
            "cluster": _optional_text(row.get("scheduler_cluster")),
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


def _status_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(str(row["status"]) for row in rows).items()))


def _status_reason(rows: list[dict[str, Any]]) -> str:
    reasons = sorted({str(row["scheduler_reason"]) for row in rows if row.get("scheduler_reason") not in (None, "")})
    return "; ".join(reasons) if reasons else "Canonical execution identity is uncertain and must be refreshed."


def _status_blocker(
    code: str,
    message: str,
    *,
    step_id: str | None = None,
    rows: list[dict[str, Any]] | None = None,
    blocked_actions: list[str] | None = None,
) -> dict[str, Any]:
    rows = rows or []
    step_ids = sorted({str(row["step_id"]) for row in rows})
    return {
        "code": code,
        "step_id": step_id or (step_ids[0] if len(step_ids) == 1 else None),
        "run_ids": sorted(str(row["run_id"]) for row in rows),
        "message": message,
        "blocked_actions": sorted(blocked_actions or []),
    }


def _monitor_action(root: Path, remote: str | None) -> dict[str, Any]:
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
    execution_host: str | None = None,
) -> dict[str, Any]:
    return {
        "id": action_id,
        "step_id": step_id,
        "execution_host": execution_host,
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


def write_wandb_report(root: Path, rows: list[dict[str, Any]], *, remote: str | None = None) -> None:
    lines = ["# W&B Sync", "", f"Synced runs: {len(rows)}", ""]
    for row in rows[:20]:
        lines.append(f"- `{row.get('version')}`: {row.get('state', '')} {row.get('wandb_url', '')}")
    exp_io.write_text_at(root / "reports" / "wandb.md", "\n".join(lines) + "\n", remote=remote)


def write_rank_report(
    root: Path, metric: str, mode: str, rows: list[dict[str, Any]], *, remote: str | None = None
) -> None:
    lines = ["# Candidate Ranking", "", f"Metric: `{metric}` ({mode})", ""]
    if rows:
        lines.append("| rank | run | setting | score | epoch | scope | checkpoint |")
        lines.append("|---:|---|---|---:|---:|---|---|")
        for row in rows:
            run_label = f"{row.get('step_id', '')} / {row.get('run_id')} — {row.get('run_name', '')}".strip(" /—")
            lines.append(
                f"| {row.get('rank')} | {run_label} | "
                f"{str(row.get('parameter_summary', '')).replace('|', '/')} | {row.get('score')} | "
                f"{row.get('epoch', '')} | {row.get('metric_scope', '')} | `{row.get('checkpoint_path', '')}` |"
            )
    else:
        lines.append("No metric rows matched.")
    exp_io.write_text_at(root / "reports" / "experiment_ranking.md", "\n".join(lines) + "\n", remote=remote)


def _local_checkpoint_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for run in runs:
        runtime_dir = Path(str(run["runtime_dir"]))
        checkpoint_dir = Path(str(run["checkpoint_dir"]))
        try:
            runtime_info = runtime_dir.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(runtime_info.st_mode) or not stat.S_ISDIR(runtime_info.st_mode):
            raise ValueError(
                f"Managed runtime_dir is not a directory: " f"{run['step_id']} / {run['run_id']} / {runtime_dir}"
            )
        try:
            checkpoint_info = checkpoint_dir.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(checkpoint_info.st_mode) or not stat.S_ISDIR(checkpoint_info.st_mode):
            raise ValueError(
                f"Managed checkpoint_dir is not a directory: " f"{run['step_id']} / {run['run_id']} / {checkpoint_dir}"
            )
        manifest_path = artifacts.find_run_manifest(run)
        manifest = read_json(manifest_path) if manifest_path else {}
        best_path = artifacts.fixed_checkpoint_path(manifest, checkpoint_dir)
        has_explicit_epoch = manifest.get("epoch") not in (None, "")
        paths = sorted(checkpoint_dir.glob("*.ckpt"))
        validate_checkpoint_evidence_rows(
            [run],
            [{**run, "checkpoint_path": str(path)} for path in paths],
        )
        for path in paths:
            rows.append(
                {
                    **{
                        field: run.get(field, "")
                        for field in ("experiment_id", "step_id", "run_id", "run_name", "version")
                    },
                    "checkpoint_path": str(path),
                    "epoch": _checkpoint_epoch(path.name),
                    "global_step": _checkpoint_step(path.name),
                    "mtime": str(int(path.stat().st_mtime)),
                    "metric": "",
                    "value": "",
                    "is_best_by_val": str(
                        str(path) == best_path or (not has_explicit_epoch and path.name.startswith("best-"))
                    ).lower(),
                    "is_last": str(path.name == "last.ckpt").lower(),
                }
            )
    return rows


def _remote_checkpoint_rows(runs: list[dict[str, Any]], remote: str | None) -> list[dict[str, Any]]:
    if not remote or not runs:
        return []
    available_runs = []
    for run in runs:
        try:
            if not exp_io.path_exists_at(run["runtime_dir"], remote=remote):
                continue
            if not exp_io.path_exists_at(run["checkpoint_dir"], remote=remote):
                continue
        except RuntimeError as exc:
            raise RuntimeError(f"SSH checkpoint scan failed on {remote}: {exc}") from exc
        available_runs.append(run)
    if not available_runs:
        return []
    runtime_roots = " ".join(
        transport.sh(run["runtime_dir"]) for run in available_runs if run.get("runtime_dir") not in (None, "")
    )
    roots = " ".join(transport.sh(run["checkpoint_dir"]) for run in available_runs)
    command = (
        (
            f"for runtime_root in {runtime_roots}; do "
            'if [ -L "$runtime_root" ] || [ ! -d "$runtime_root" ]; then '
            "printf 'Managed runtime_dir is missing or is not a directory: %s\\n' \"$runtime_root\" >&2; exit 1; "
            "fi; done; "
        )
        if runtime_roots
        else ""
    ) + (
        f"for root in {roots}; do "
        'if [ -L "$root" ] || [ ! -d "$root" ]; then '
        "printf 'Managed checkpoint_dir is missing or is not a directory: %s\\n' \"$root\" >&2; exit 1; "
        "fi; "
        "find \"$root\" -mindepth 1 -maxdepth 1 -name '*.ckpt' -printf '%p\t%T@\n' || exit $?; "
        "done"
    )
    try:
        result = transport.run_ssh(remote, command, text=True)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"SSH checkpoint scan timed out on {remote}") from exc
    if result.returncode != 0:
        detail = result.stderr.strip() or f"exit code {result.returncode}"
        raise RuntimeError(f"SSH checkpoint scan failed on {remote}: {detail}")
    runs_by_checkpoint_dir = {str(run["checkpoint_dir"]): run for run in available_runs}
    rows = {}
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        if "\t" not in line:
            raise RuntimeError(f"SSH checkpoint scan returned malformed output on {remote}: {line}")
        path_text, mtime = line.split("\t", 1)
        if not path_text or not mtime:
            raise RuntimeError(f"SSH checkpoint scan returned malformed output on {remote}: {line}")
        try:
            parsed_mtime = float(mtime)
        except ValueError as exc:
            raise RuntimeError(f"SSH checkpoint scan returned malformed output on {remote}: {line}") from exc
        if not math.isfinite(parsed_mtime):
            raise RuntimeError(f"SSH checkpoint scan returned malformed output on {remote}: {line}")
        name = path_text.rsplit("/", 1)[-1]
        run = runs_by_checkpoint_dir.get(path_text.rsplit("/", 1)[0])
        if run is None:
            raise RuntimeError(f"SSH checkpoint scan returned an undeclared checkpoint path on {remote}: {path_text}")
        rows[path_text] = {
            **{field: run.get(field, "") for field in ("experiment_id", "step_id", "run_id", "run_name", "version")},
            "checkpoint_path": path_text,
            "epoch": _checkpoint_epoch(name),
            "global_step": _checkpoint_step(name),
            "mtime": mtime,
            "metric": "",
            "value": "",
            "is_best_by_val": str(name.startswith("best-")).lower(),
            "is_last": str(name == "last.ckpt").lower(),
        }
    checkpoint_rows = list(rows.values())
    validate_checkpoint_evidence_rows(available_runs, checkpoint_rows, remote=remote)
    for run in available_runs:
        manifest_path = str(run["runtime_dir"]).rstrip("/") + "/run_manifest.json"
        try:
            exp_io.validate_managed_output_paths(run["runtime_dir"], [manifest_path], remote=remote)
            manifest_text = exp_io.read_text_at(manifest_path, remote=remote)
            manifest_exists = bool(manifest_text) or exp_io.path_exists_at(manifest_path, remote=remote)
        except (RuntimeError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError(f"SSH checkpoint scan failed to read {manifest_path} on {remote}") from exc
        if manifest_exists:
            if not manifest_text:
                raise RuntimeError(f"SSH checkpoint scan found a corrupt run manifest on {remote}: {manifest_path}")
            try:
                manifest = json.loads(manifest_text)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"SSH checkpoint scan found a corrupt run manifest on {remote}: {manifest_path}"
                ) from exc
            if not isinstance(manifest, dict):
                raise RuntimeError(f"SSH checkpoint scan found a corrupt run manifest on {remote}: {manifest_path}")
        else:
            manifest = {}
        same_run = [row for row in checkpoint_rows if managed_run_key(row) == managed_run_key(run)]
        best_path = artifacts.fixed_checkpoint_path_from_names(
            manifest,
            run["checkpoint_dir"],
            [Path(str(row["checkpoint_path"])).name for row in same_run],
        )
        has_explicit_epoch = manifest.get("epoch") not in (None, "")
        for row in same_run:
            name = Path(str(row["checkpoint_path"])).name
            row["is_best_by_val"] = str(
                row["checkpoint_path"] == best_path or (not has_explicit_epoch and name.startswith("best-"))
            ).lower()
    return checkpoint_rows


def validate_checkpoint_evidence_rows(
    runs: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    *,
    remote: str | None = None,
) -> None:
    runs_by_key = {managed_run_key(run): run for run in runs}
    grouped_paths: dict[tuple[str | None, str], list[Path]] = {}
    for row in rows:
        checkpoint_path = row.get("checkpoint_path")
        if checkpoint_path in (None, ""):
            continue
        run = runs_by_key.get(managed_run_key(row))
        if run is None:
            raise ValueError(
                f"Checkpoint evidence is outside the canonical manifest: "
                f"{row.get('step_id', '')} / {row.get('run_id', '')}"
            )
        validate_frozen_run_update(run, row)
        validate_checkpoint_ownership(run, row)
        evidence_host = checkpoint_evidence_host(run, remote)
        checkpoint_dir = str(run["checkpoint_dir"])
        grouped_paths.setdefault((evidence_host, checkpoint_dir), []).append(Path(str(checkpoint_path)))
    for (evidence_host, checkpoint_dir), paths in grouped_paths.items():
        exp_io.validate_managed_output_paths(checkpoint_dir, paths, remote=evidence_host)
        for path in paths:
            if not exp_io.path_exists_at(path, remote=evidence_host):
                raise ValueError(f"Checkpoint evidence is missing: {path}")


def checkpoint_evidence_host(run: dict[str, Any], remote: str | None) -> str | None:
    if run.get("target") != "ssh":
        return remote
    host = str(run.get("host") or "")
    if not host:
        raise ValueError(f"Managed SSH run is missing its host: {run.get('step_id', '')} / {run.get('run_id', '')}")
    return host


def _checkpoint_for_metric_row(row: dict[str, Any], checkpoints: list[dict[str, str]]) -> str:
    raw_epoch = row.get("epoch")
    key = managed_run_key(row)
    same_run = [item for item in checkpoints if managed_run_key(item) == key]
    if raw_epoch not in (None, ""):
        try:
            numeric_epoch = float(str(raw_epoch))
        except ValueError:
            return ""
        if not math.isfinite(numeric_epoch) or not numeric_epoch.is_integer():
            return ""
        epoch = int(numeric_epoch)
        for item in same_run:
            if artifacts.epoch_number(item.get("epoch")) == epoch:
                return item.get("checkpoint_path", "")
        return ""
    best = [item for item in same_run if item.get("is_best_by_val") == "true"]
    if best:
        return best[0].get("checkpoint_path", "")
    last = [item for item in same_run if item.get("is_last") == "true"]
    return last[0].get("checkpoint_path", "") if last else ""


def _best_rows(rows: list[dict[str, Any]], *, mode: str) -> list[dict[str, Any]]:
    reverse = mode == "max"
    best: dict[tuple[str, ...], dict[str, Any]] = {}
    for row in rows:
        key = managed_run_key(row)
        if key is None:
            continue
        if key not in best:
            best[key] = row
            continue
        current = artifacts.sortable_score(row.get("score"), reverse)
        previous = artifacts.sortable_score(best[key].get("score"), reverse)
        if (reverse and current > previous) or (not reverse and current < previous):
            best[key] = row
    return list(best.values())


def _history_rows_for_run(run: Any) -> list[dict[str, Any]]:
    try:
        history = run.history(samples=100000, pandas=True)
    except TypeError:
        history = run.history()
    except Exception:
        history = None
    if hasattr(history, "to_dict"):
        return [dict(row) for row in history.to_dict(orient="records")]
    if history:
        return [dict(row) for row in history]
    try:
        return [dict(row) for row in run.scan_history()]
    except Exception:
        return []


def _history_metric_rows(
    wandb_run_id: str,
    version: str,
    run_row: dict[str, Any],
    history: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for record in history:
        epoch = _record_epoch(record)
        for metric, value in record.items():
            if metric.startswith("_") or not _is_scalar_number(value):
                continue
            rows.append(
                {
                    **{field: run_row[field] for field in ("experiment_id", "step_id", "run_id") if field in run_row},
                    "version": version,
                    "epoch": "" if epoch is None else epoch,
                    "split": _metric_split(metric),
                    "metric": metric,
                    "value": value,
                    "source": "wandb_history",
                    "metric_scope": _metric_scope(metric),
                    "wandb_run_id": wandb_run_id,
                    "updated_at": utc_now(),
                }
            )
    return rows


def _safe_dict(value: Any) -> dict[str, Any]:
    try:
        return dict(value)
    except Exception:
        return {}


def _is_scalar_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    try:
        score = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(score)


def _summary_epoch(summary: dict[str, Any]) -> str:
    for key in ("epoch", "trainer/epoch", "current_epoch"):
        value = artifacts.float_or_none(summary.get(key))
        if value is not None:
            return str(int(value))
    return ""


def _record_epoch(record: dict[str, Any]) -> str | None:
    for key in ("epoch", "trainer/epoch", "current_epoch"):
        value = artifacts.float_or_none(record.get(key))
        if value is not None:
            return str(int(value))
    return None


def _metric_split(metric: str) -> str:
    lowered = metric.lower()
    if lowered.startswith("train") or "/train" in lowered:
        return "train"
    if lowered.startswith("val") or "/val" in lowered or "validation" in lowered:
        return "val"
    if lowered.startswith("test") or "/test" in lowered:
        return "test"
    if lowered.startswith("external") or "/external" in lowered:
        return "external"
    return ""


def _metric_scope(metric: str) -> str:
    split = _metric_split(metric)
    if split == "val":
        return "validation"
    if split in {"test", "external"}:
        return "test_or_external"
    if split == "train":
        return "train"
    return "unknown"


def _checkpoint_epoch(name: str) -> str:
    clean = name.removeprefix("best-")
    if clean == "last.ckpt":
        return ""
    return artifacts.epoch_from_checkpoint_name(clean)


def _checkpoint_step(name: str) -> str:
    match = re.search(r"step=(\d+)", name)
    return match.group(1) if match else ""


def _safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_") or "run"
