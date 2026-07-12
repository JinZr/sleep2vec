from __future__ import annotations

import csv
import io
import json
import math
from pathlib import Path
import re
import shlex
import subprocess
from typing import Any

from . import experiment_io as exp_io, run_artifacts as artifacts, run_evidence as evidence
from .experiment_workspace import (
    managed_run_key,
    merge_run_row,
    read_managed_yaml_mapping,
    read_run_manifest,
    resolve_external_run_row,
    resolve_run_row,
    validate_frozen_run_update,
    validate_managed_run_rows,
)
from .manifests import read_json, utc_now
from .models import json_ready

AUXILIARY_RUN_FIELDS = {
    "target",
    "host",
    "workdir",
    "gpus",
    "pid_path",
    "pid",
    "log_path",
    "log_tail",
    "log_age_seconds",
    "command",
    "launched_at",
    "monitored_at",
    "stopped_at",
    "stop_reason",
    "run_manifest",
    "checkpoints",
    "checkpoint_count",
    "health_status",
    "gpu_summary",
    "io_read_bytes",
    "io_write_bytes",
    "io_read_delta_bytes",
    "io_write_delta_bytes",
    "progress_dir",
    "progress_status",
    "progress_processed",
    "progress_total",
    "progress_updated_at",
    "progress_age_seconds",
}
RUN_STATUS_FIELDS = AUXILIARY_RUN_FIELDS | {"status"}
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
    for row in wandb_rows:
        if "trial_id" in row:
            raise ValueError("Historical trial_id W&B evidence is unsupported.")
        existing = resolve_external_run_row(run_rows, row)
        if existing is None:
            continue
        key = managed_run_key(existing)
        update = {
            "step_id": key[0],
            "run_id": key[1],
            **{field: row[field] for field in WANDB_RUN_FIELDS if field in row},
        }
        observations[key] = merge_run_row(observations.get(key, {}), update)
    rows = list(observations.values())
    validate_managed_run_rows(rows, source="W&B run observations", cardinality="one_per_run")
    return rows


def experiment_run_rows(root: Path, *, remote: str | None = None) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, ...], dict[str, Any]] = {}
    canonical_rows = read_run_manifest(root, remote=remote)
    for row in canonical_rows:
        key = managed_run_key(row)
        by_key[key] = merge_run_row(by_key.get(key, {}), row)

    merged_rows = list(by_key.values())
    for path in (root / "launch_manifest.tsv", root / "run_status.tsv"):
        auxiliary_rows = exp_io.read_rows_at(path, remote=remote, require_managed_identity=True)
        validate_managed_run_rows(auxiliary_rows, source=path.name, cardinality="one_per_run")
        allowed_fields = RUN_STATUS_FIELDS if path.name == "run_status.tsv" else AUXILIARY_RUN_FIELDS
        for row in auxiliary_rows:
            key = managed_run_key(row)
            existing = next((candidate for candidate in merged_rows if managed_run_key(candidate) == key), None)
            if existing is None:
                raise ValueError(
                    f"{path.name} row is not managed by run_manifest.tsv: "
                    f"{row.get('step_id', '')} / {row.get('run_id', '')}"
                )
            validate_frozen_run_update(existing, row)
            update = {field: row[field] for field in allowed_fields if field in row}
            merged = merge_run_row(existing, update)
            index = next(index for index, candidate in enumerate(merged_rows) if candidate is existing)
            merged_rows[index] = merged

    if remote:
        for row in merged_rows:
            if not row.get("host"):
                row["target"] = "ssh"
                row["host"] = remote
    return merged_rows


def managed_metric_rows(run_rows: list[dict[str, Any]], metric_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    validate_managed_run_rows(run_rows, source="run_manifest.tsv", cardinality="one_per_run")
    rows = []
    for metric_row in metric_rows:
        if "trial_id" in metric_row:
            raise ValueError("Historical trial_id metric evidence is unsupported.")
        run_row = resolve_external_run_row(run_rows, metric_row)
        if run_row is None:
            continue
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
        validate_frozen_run_update(run, row, require_checkpoint_ownership=True)
    if not eligible_runs:
        return []
    rows = _remote_checkpoint_rows(eligible_runs, remote) if remote else _local_checkpoint_rows(eligible_runs)
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
    if remote and not row.get("host"):
        row["target"] = "ssh"
        row["host"] = remote
    status = evidence.status_row(root, row, previous, health=True)
    if status.get("status") == "finished":
        status["status"] = "completed"
    if status.get("health_status") == "finished":
        status["health_status"] = "completed"
    return {
        "step_id": row["step_id"],
        "run_id": row["run_id"],
        **{field: status[field] for field in RUN_STATUS_FIELDS if field in status},
    }


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
    ranked = sorted(ranked, key=lambda row: artifacts.sortable_score(row.get("score"), reverse), reverse=reverse)
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
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
    seen = set()
    merged = []
    for row in [*existing, *new_rows]:
        key = tuple(
            str(row.get(field, ""))
            for field in ("step_id", "run_id", "version", "epoch", "metric", "source", "wandb_run_id")
        )
        if key in seen:
            continue
        seen.add(key)
        merged.append(row)
    return merged


def monitor_report(rows: list[dict[str, Any]]) -> str:
    lines = ["# Experiment Monitor", ""]
    if not rows:
        return "# Experiment Monitor\n\nNo runs found.\n"
    lines.append("| run | setting | status | health | gpu | log age | checkpoints |")
    lines.append("|---|---|---|---|---|---:|---:|")
    for row in rows:
        lines.append(
            "| {run} | {setting} | {status} | {health} | {gpu} | {log_age} | {ckpts} |".format(
                run=f"{row.get('step_id', '')} / {row.get('run_id', '')} — {row.get('run_name', '')}",
                setting=str(row.get("parameter_summary", "")).replace("|", "/"),
                status=row.get("status", ""),
                health=row.get("health_status", ""),
                gpu=str(row.get("gpu_summary", "")).replace("|", "/"),
                log_age=row.get("log_age_seconds", ""),
                ckpts=row.get("checkpoint_count", ""),
            )
        )
    lines.append("")
    return "\n".join(lines)


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
        checkpoint_dir = Path(str(run["checkpoint_dir"]))
        manifest_path = Path(str(run["runtime_dir"])) / "run_manifest.json"
        manifest = read_json(manifest_path) if manifest_path.exists() else {}
        best_path = artifacts.fixed_checkpoint_path(manifest, checkpoint_dir)
        for path in sorted(checkpoint_dir.glob("*.ckpt")):
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
                    "is_best_by_val": str(str(path) == best_path or path.name.startswith("best-")).lower(),
                    "is_last": str(path.name == "last.ckpt").lower(),
                }
            )
    return rows


def _remote_checkpoint_rows(runs: list[dict[str, Any]], remote: str | None) -> list[dict[str, Any]]:
    if not remote or not runs:
        return []
    roots = " ".join(shlex.quote(str(run["checkpoint_dir"])) for run in runs)
    command = f"find {roots} -maxdepth 1 -type f -name '*.ckpt' -printf '%p\t%T@\n' 2>/dev/null"
    try:
        result = subprocess.run(
            ["ssh", remote, command],
            text=True,
            capture_output=True,
            timeout=exp_io.SSH_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"SSH checkpoint scan timed out on {remote}") from exc
    if result.returncode != 0:
        detail = result.stderr.strip() or f"exit code {result.returncode}"
        raise RuntimeError(f"SSH checkpoint scan failed on {remote}: {detail}")
    runs_by_checkpoint_dir = {str(run["checkpoint_dir"]): run for run in runs}
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
    return list(rows.values())


def _checkpoint_for_metric_row(row: dict[str, Any], checkpoints: list[dict[str, str]]) -> str:
    epoch = artifacts.epoch_number(row.get("epoch"))
    key = managed_run_key(row)
    same_run = [item for item in checkpoints if managed_run_key(item) == key]
    for item in same_run:
        if artifacts.epoch_number(item.get("epoch")) == epoch:
            return item.get("checkpoint_path", "")
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
