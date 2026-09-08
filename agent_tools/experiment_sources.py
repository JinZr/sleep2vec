"""Acquisition of W&B run history and remote checkpoint evidence.

Layer 0 leaf. Owns experiment-wide W&B history acquisition and remote or local
checkpoint evidence collection; per-run checkpoint inventory and epoch
resolution live in ``run_artifacts``. ``experiment_tracking`` consumes what it
returns. Acquisition and interpretation stay split so status, ranking, and
projection logic remain testable without a network or a cluster.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
import stat
import subprocess
from typing import Any, Literal, TypedDict

from . import experiment_io as exp_io, run_artifacts as artifacts, transport
from .experiment_workspace import managed_run_key, validate_checkpoint_ownership, validate_frozen_run_update
from .manifests import read_json, utc_now
from .models import json_ready


class _WandbIdentity(TypedDict, total=False):
    experiment_id: str
    step_id: str
    run_id: str


class _WandbRunCore(_WandbIdentity):
    version: str
    state: str
    wandb_run_id: str
    wandb_url: str
    wandb_entity: str
    wandb_project: str
    wandb_group: str
    created_at: str
    updated_at: str


class WandbRunObservation(_WandbRunCore, total=False):
    status: str


class WandbMetricObservation(_WandbIdentity):
    version: str
    epoch: str
    split: str
    metric: str
    value: Any
    source: str
    metric_scope: str
    wandb_run_id: str
    updated_at: str


class WandbRunPayload(TypedDict):
    run_row: WandbRunObservation
    metric_rows: list[WandbMetricObservation]
    summary_line: str
    history_rows: list[dict[str, Any]]
    history_filename: str


class WandbTrainingHistory(TypedDict):
    source_path: str
    source_sha256: str
    wandb_run_id: str
    observations: list[dict[str, Any]]
    learning_rate_ranges: dict[str, dict[str, float]]


class CheckpointObservation(TypedDict):
    experiment_id: str
    step_id: str
    run_id: str
    run_name: str
    version: str
    checkpoint_path: str
    epoch: str
    global_step: str
    mtime: str
    metric: str
    value: str
    is_best_by_val: str
    is_last: str


def wandb_runs(entity: str, project: str, group: str | None) -> list[Any]:
    import wandb

    api = wandb.Api()
    filters = {"group": group} if group else None
    return list(api.runs(f"{entity}/{project}", filters=filters))


def read_wandb_training_history(
    workspace: Path, run: Mapping[str, Any], *, monitor: str, objective: str
) -> WandbTrainingHistory | None:
    """Read an already-synced history bound by the canonical run's W&B id; never sync or infer completion."""
    run_id = str(run.get("wandb_run_id") or "")
    if not run_id:
        return None
    path = workspace / "wandb" / "history" / f"{_safe_filename(run_id)}.csv"
    if not path.exists():
        return None
    content = path.read_bytes()
    records = csv.DictReader(io.StringIO(content.decode("utf-8")), strict=True)
    metrics = {"train_loss_epoch", "val_loss"}
    metrics.update(name for name in (monitor, objective) if name.startswith("val_"))
    observations, learning_rates = _training_history_observations(records, metrics)
    return {
        "source_path": str(path),
        "source_sha256": hashlib.sha256(content).hexdigest(),
        "wandb_run_id": run_id,
        "observations": observations,
        "learning_rate_ranges": learning_rates,
    }


def _training_history_observations(
    records: Iterable[dict[str, Any]], metrics: set[str]
) -> tuple[list[dict[str, Any]], dict[str, dict[str, float]]]:
    observations = []
    learning_rates: dict[str, dict[str, float]] = {}
    for record in records:
        values = {name: float(record[name]) for name in sorted(metrics) if _is_scalar_number(record.get(name))}
        if values:
            observation: dict[str, Any] = {"metrics": values}
            for key in ("epoch", "trainer/epoch", "current_epoch", "_step", "trainer/global_step"):
                if _is_scalar_number(record.get(key)):
                    observation[key] = float(record[key])
            observations.append(observation)
        for name, value in record.items():
            if isinstance(name, str) and name.startswith("lr-") and _is_scalar_number(value):
                score = float(value)
                limits = learning_rates.setdefault(name, {"min": score, "max": score})
                limits["min"] = min(limits["min"], score)
                limits["max"] = max(limits["max"], score)
    return observations, learning_rates


def wandb_run_payload(run: Any, *, entity: str, project: str) -> WandbRunPayload:
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
    row: WandbRunObservation = {
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
    field: Literal["experiment_id", "step_id", "run_id"]
    for field in ("experiment_id", "step_id", "run_id"):
        if config.get(field) not in (None, ""):
            row[field] = str(config[field])
    if status:
        row["status"] = status
    metric_rows: list[WandbMetricObservation] = []
    for metric, value in summary.items():
        if _is_scalar_number(value):
            identity: _WandbIdentity = {}
            for field in ("experiment_id", "step_id", "run_id"):
                if field in row:
                    identity[field] = row[field]
            metric_rows.append(
                {
                    **identity,
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


def _local_checkpoint_rows(runs: list[dict[str, str]]) -> list[CheckpointObservation]:
    rows: list[CheckpointObservation] = []
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
                    "experiment_id": run.get("experiment_id", ""),
                    "step_id": run.get("step_id", ""),
                    "run_id": run.get("run_id", ""),
                    "run_name": run.get("run_name", ""),
                    "version": run.get("version", ""),
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


def _parse_remote_checkpoint_rows(
    stdout: str, available_runs: list[dict[str, str]], *, remote: str
) -> list[CheckpointObservation]:
    runs_by_checkpoint_dir = {str(run["checkpoint_dir"]): run for run in available_runs}
    rows: dict[str, CheckpointObservation] = {}
    for line in stdout.splitlines():
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
        owner_run = runs_by_checkpoint_dir.get(path_text.rsplit("/", 1)[0])
        if owner_run is None:
            raise RuntimeError(f"SSH checkpoint scan returned an undeclared checkpoint path on {remote}: {path_text}")
        rows[path_text] = {
            "experiment_id": owner_run.get("experiment_id", ""),
            "step_id": owner_run.get("step_id", ""),
            "run_id": owner_run.get("run_id", ""),
            "run_name": owner_run.get("run_name", ""),
            "version": owner_run.get("version", ""),
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


def _remote_checkpoint_rows(runs: list[dict[str, str]], remote: str | None) -> list[CheckpointObservation]:
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
    checkpoint_rows = _parse_remote_checkpoint_rows(result.stdout, available_runs, remote=remote)
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
    rows: Sequence[Mapping[str, Any]],
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
    run_row: WandbRunObservation,
    history: list[dict[str, Any]],
) -> list[WandbMetricObservation]:
    rows: list[WandbMetricObservation] = []
    for record in history:
        epoch = _record_epoch(record)
        for metric, value in record.items():
            if metric.startswith("_") or not _is_scalar_number(value):
                continue
            identity: _WandbIdentity = {}
            field: Literal["experiment_id", "step_id", "run_id"]
            for field in ("experiment_id", "step_id", "run_id"):
                if field in run_row:
                    identity[field] = run_row[field]
            rows.append(
                {
                    **identity,
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
