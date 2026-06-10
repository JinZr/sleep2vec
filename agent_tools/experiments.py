from __future__ import annotations

import csv
import io
import json
import math
from pathlib import Path
import re
import subprocess
from typing import Any

from .hparam import (
    SSH_TIMEOUT_SECONDS,
    _epoch_from_checkpoint_name,
    _epoch_number,
    _fixed_checkpoint_path,
    _float_or_none,
    _now,
    _read_json,
    _read_rows,
    _sh,
    _sortable_score,
    _status_row,
    _write_rows,
)
from .manifests import write_text
from .models import json_ready


def init_experiment(run_dir: str | Path, name: str, *, remote: str | None = None) -> Path:
    root = Path(run_dir)
    _mkdir_experiment_dirs(root, remote=remote)
    manifest = root / "experiment_manifest.tsv"
    rows = _read_rows_at(manifest, remote=remote)
    if rows:
        row = rows[0]
        row.update(
            {
                "experiment_id": name,
                "experiment_root": str(root),
                "remote_host": remote or row.get("remote_host", ""),
                "updated_at": _now(),
            }
        )
    else:
        row = {
            "experiment_id": name,
            "experiment_root": str(root),
            "remote_host": remote or "",
            "task": "",
            "selection_metric": "",
            "selection_mode": "",
            "wandb_entity": "",
            "wandb_project": "",
            "wandb_group": "",
            "created_at": _now(),
            "updated_at": _now(),
        }
    _write_rows_at(manifest, [row], remote=remote)
    return manifest


def sync_wandb_runs(
    run_dir: str | Path,
    *,
    entity: str,
    project: str,
    group: str | None = None,
    remote: str | None = None,
) -> Path:
    root = Path(run_dir)
    _mkdir_experiment_dirs(root, remote=remote)
    try:
        import wandb

        api = wandb.Api()
        filters = {"group": group} if group else None
        runs = list(api.runs(f"{entity}/{project}", filters=filters))
    except Exception as exc:
        blocked = root / "reports" / "wandb_blocked.md"
        _write_text_at(blocked, f"# W&B Sync Blocked\n\n{type(exc).__name__}: {exc}\n", remote=remote)
        raise RuntimeError(f"W&B sync blocked; wrote {blocked}") from exc

    run_rows = []
    metric_rows = []
    summary_path = root / "wandb" / "summaries.jsonl"
    summary_lines = []
    for run in runs:
        run_id = str(getattr(run, "id", ""))
        run_name = str(getattr(run, "name", "") or run_id)
        run_group = str(getattr(run, "group", "") or "")
        summary = _safe_dict(getattr(run, "summary", {}))
        config = _safe_dict(getattr(run, "config", {}))
        url = str(getattr(run, "url", "") or "")
        state = str(getattr(run, "state", "") or "")
        row = {
            "trial_id": str(config.get("trial_id") or run_name),
            "version": run_name,
            "state": state,
            "wandb_run_id": run_id,
            "wandb_url": url,
            "wandb_entity": entity,
            "wandb_project": project,
            "wandb_group": run_group,
            "created_at": str(getattr(run, "created_at", "") or ""),
            "updated_at": str(getattr(run, "updated_at", "") or ""),
        }
        run_rows.append(row)
        summary_lines.append(json.dumps(json_ready({"run": row, "summary": summary}), sort_keys=True))
        for metric, value in summary.items():
            if _is_scalar_number(value):
                metric_rows.append(
                    {
                        "trial_id": row["trial_id"],
                        "version": run_name,
                        "epoch": _summary_epoch(summary),
                        "split": _metric_split(metric),
                        "metric": metric,
                        "value": value,
                        "source": "wandb_summary",
                        "metric_scope": _metric_scope(metric),
                        "wandb_run_id": run_id,
                        "updated_at": _now(),
                    }
                )
        history_rows = _history_rows_for_run(run)
        _write_history_csv(
            root / "wandb" / "history" / f"{_safe_filename(run_id or run_name)}.csv",
            history_rows,
            remote=remote,
        )
        metric_rows.extend(_history_metric_rows(run_id, run_name, row["trial_id"], history_rows))

    _write_text_at(summary_path, "\n".join(summary_lines) + ("\n" if summary_lines else ""), remote=remote)
    _write_rows_at(root / "wandb" / "runs.tsv", run_rows, remote=remote)
    _write_rows_at(
        root / "metrics_manifest.tsv",
        _merge_rows(_read_rows_at(root / "metrics_manifest.tsv", remote=remote), metric_rows),
        remote=remote,
    )
    _write_rows_at(root / "run_manifest.tsv", _merge_run_rows(root, run_rows, remote=remote), remote=remote)
    _update_experiment_wandb(root, entity=entity, project=project, group=group or "", remote=remote)
    _write_wandb_report(root, run_rows, remote=remote)
    return root / "wandb" / "runs.tsv"


def index_checkpoints(run_dir: str | Path, *, remote: str | None = None) -> Path:
    root = Path(run_dir)
    rows = _remote_checkpoint_rows(root, remote) if remote else _local_checkpoint_rows(root)
    metrics = _read_rows_at(root / "metrics_manifest.tsv", remote=remote)
    for row in rows:
        metric = _best_metric_for_checkpoint(row, metrics)
        row.update(metric)
    _write_rows_at(root / "checkpoint_manifest.tsv", rows, remote=remote)
    return root / "checkpoint_manifest.tsv"


def monitor_experiment(run_dir: str | Path, *, remote: str | None = None) -> dict[str, Any]:
    root = Path(run_dir)
    run_rows = _experiment_run_rows(root, remote=remote)
    previous = {
        row.get("trial_id") or row.get("version"): row
        for row in _read_rows_at(root / "run_manifest.tsv", remote=remote)
    }
    monitored = []
    for row in run_rows:
        key = row.get("trial_id") or row.get("version")
        if remote and not row.get("host"):
            row["target"] = "ssh"
            row["host"] = remote
        status = _status_row(root, row, previous.get(key, {}), health=True)
        if status.get("status") == "finished":
            status["status"] = "completed"
        if status.get("health_status") == "finished":
            status["health_status"] = "completed"
        monitored.append(status)
    _write_rows_at(root / "run_manifest.tsv", monitored, remote=remote)
    report = _monitor_report(monitored)
    _write_text_at(root / "reports" / "monitor.md", report, remote=remote)
    return {"run_dir": str(root), "runs": monitored, "report": str(root / "reports" / "monitor.md")}


def rank_experiment_candidates(run_dir: str | Path, *, metric: str, mode: str, remote: str | None = None) -> Path:
    root = Path(run_dir)
    reverse = mode == "max"
    rows = []
    for metric_row in _read_rows_at(root / "metrics_manifest.tsv", remote=remote):
        if metric_row.get("metric") != metric:
            continue
        score = _float_or_none(metric_row.get("value"))
        if score is None:
            continue
        rows.append(
            {
                "trial_id": metric_row.get("trial_id"),
                "version": metric_row.get("version"),
                "epoch": metric_row.get("epoch", ""),
                "metric": metric,
                "score": score,
                "metric_scope": metric_row.get("metric_scope") or _metric_scope(metric),
                "source": metric_row.get("source", ""),
                "wandb_run_id": metric_row.get("wandb_run_id", ""),
            }
        )
    ranked = _best_rows(rows, mode=mode)
    checkpoints = _read_rows_at(root / "checkpoint_manifest.tsv", remote=remote)
    for row in ranked:
        row["checkpoint_path"] = _checkpoint_for_metric_row(row, checkpoints)
    ranked = sorted(ranked, key=lambda row: _sortable_score(row.get("score"), reverse), reverse=reverse)
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
    out = root / "candidate_ranking.tsv"
    _write_rows_at(out, ranked, remote=remote)
    _write_rank_report(root, metric, mode, ranked, remote=remote)
    return out


def _update_experiment_wandb(root: Path, *, entity: str, project: str, group: str, remote: str | None = None) -> None:
    path = root / "experiment_manifest.tsv"
    rows = _read_rows_at(path, remote=remote)
    if not rows:
        rows = [
            {
                "experiment_id": root.name,
                "experiment_root": str(root),
                "remote_host": "",
                "task": "",
                "selection_metric": "",
                "selection_mode": "",
                "created_at": _now(),
            }
        ]
    rows[0].update(
        {
            "wandb_entity": entity,
            "wandb_project": project,
            "wandb_group": group,
            "updated_at": _now(),
        }
    )
    _write_rows_at(path, rows, remote=remote)


def _merge_run_rows(root: Path, wandb_rows: list[dict[str, Any]], *, remote: str | None = None) -> list[dict[str, Any]]:
    existing = _experiment_run_rows(root, remote=remote)
    by_key = {row.get("version") or row.get("trial_id"): dict(row) for row in existing}
    for row in wandb_rows:
        key = row.get("version") or row.get("trial_id")
        merged = by_key.get(key, {})
        merged.update(row)
        by_key[key] = merged
    return list(by_key.values())


def _experiment_run_rows(root: Path, *, remote: str | None = None) -> list[dict[str, Any]]:
    by_key: dict[str, dict[str, Any]] = {}
    for path in (root / "run_manifest.tsv", root / "launch_manifest.tsv", root / "trial_status.tsv"):
        for row in _read_rows_at(path, remote=remote):
            key = row.get("trial_id") or row.get("version")
            if not key:
                continue
            merged = by_key.get(key, {})
            merged.update(row)
            if remote and not merged.get("host"):
                merged["target"] = "ssh"
                merged["host"] = remote
            by_key[key] = merged
    return list(by_key.values())


def _mkdir_experiment_dirs(root: Path, *, remote: str | None = None) -> None:
    dirs = [root / "reports", root / "wandb" / "history"]
    if remote:
        command = "mkdir -p " + " ".join(_sh(path) for path in dirs)
        subprocess.run(
            ["ssh", remote, command],
            check=True,
            text=True,
            capture_output=True,
            timeout=SSH_TIMEOUT_SECONDS,
        )
        return
    for path in dirs:
        path.mkdir(parents=True, exist_ok=True)


def _read_rows_at(path: str | Path, *, remote: str | None = None) -> list[dict[str, str]]:
    if not remote:
        return _read_rows(path)
    result = subprocess.run(
        ["ssh", remote, f"cat {_sh(path)}"],
        text=True,
        capture_output=True,
        timeout=SSH_TIMEOUT_SECONDS,
    )
    if result.returncode != 0:
        return []
    delimiter = "\t" if Path(str(path)).suffix == ".tsv" else ","
    return list(csv.DictReader(io.StringIO(result.stdout), delimiter=delimiter))


def _write_rows_at(path: str | Path, rows: list[dict[str, Any]], *, remote: str | None = None) -> None:
    if not remote:
        _write_rows(path, rows)
        return
    target = Path(str(path))
    fieldnames = sorted({key for row in rows for key in row}) if rows else ["trial_id"]
    delimiter = "\t" if target.suffix == ".tsv" else ","
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, delimiter=delimiter)
    writer.writeheader()
    writer.writerows(rows)
    _write_text_at(path, buffer.getvalue(), remote=remote)


def _write_text_at(path: str | Path, text: str, *, remote: str | None = None) -> None:
    if not remote:
        write_text(path, text)
        return
    target = Path(str(path))
    command = f"mkdir -p {_sh(target.parent)} && cat > {_sh(target)}"
    subprocess.run(
        ["ssh", remote, command],
        input=text,
        text=True,
        capture_output=True,
        check=True,
        timeout=SSH_TIMEOUT_SECONDS,
    )


def _local_checkpoint_rows(root: Path) -> list[dict[str, Any]]:
    rows = []
    best_paths = _best_checkpoint_paths(root)
    for path in sorted(root.glob("**/checkpoints/*.ckpt")):
        version = path.parent.parent.name
        rows.append(
            {
                "trial_id": version,
                "version": version,
                "checkpoint_path": str(path),
                "epoch": _checkpoint_epoch(path.name),
                "global_step": _checkpoint_step(path.name),
                "mtime": str(int(path.stat().st_mtime)),
                "metric": "",
                "value": "",
                "is_best_by_val": str(path in best_paths or path.name.startswith("best-")).lower(),
                "is_last": str(path.name == "last.ckpt").lower(),
            }
        )
    return rows


def _remote_checkpoint_rows(root: Path, remote: str | None) -> list[dict[str, Any]]:
    if not remote:
        return []
    command = f"find {_sh(root)} -path '*/checkpoints/*.ckpt' -printf '%p\\t%T@\\n' 2>/dev/null"
    try:
        result = subprocess.run(
            ["ssh", remote, command],
            text=True,
            capture_output=True,
            timeout=SSH_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return []
    if result.returncode != 0:
        return []
    rows = []
    for line in result.stdout.splitlines():
        if "\t" not in line:
            continue
        path_text, mtime = line.split("\t", 1)
        name = path_text.rsplit("/", 1)[-1]
        version = path_text.rsplit("/checkpoints/", 1)[0].rsplit("/", 1)[-1]
        rows.append(
            {
                "trial_id": version,
                "version": version,
                "checkpoint_path": path_text,
                "epoch": _checkpoint_epoch(name),
                "global_step": _checkpoint_step(name),
                "mtime": mtime,
                "metric": "",
                "value": "",
                "is_best_by_val": str(name.startswith("best-")).lower(),
                "is_last": str(name == "last.ckpt").lower(),
            }
        )
    return rows


def _best_checkpoint_paths(root: Path) -> set[Path]:
    best = set()
    for manifest_path in root.glob("**/run_manifest.json"):
        manifest = _read_json(manifest_path)
        fixed = _fixed_checkpoint_path(manifest, manifest_path)
        if fixed:
            best.add(Path(fixed))
    return best


def _best_metric_for_checkpoint(row: dict[str, Any], metrics: list[dict[str, str]]) -> dict[str, Any]:
    epoch = _epoch_number(row.get("epoch"))
    version = row.get("version")
    matches = [
        item
        for item in metrics
        if item.get("version") == version
        and _epoch_number(item.get("epoch")) == epoch
        and item.get("metric_scope") == "validation"
    ]
    if not matches:
        return {"metric": "", "value": ""}
    chosen = matches[0]
    return {"metric": chosen.get("metric", ""), "value": chosen.get("value", "")}


def _checkpoint_for_metric_row(row: dict[str, Any], checkpoints: list[dict[str, str]]) -> str:
    epoch = _epoch_number(row.get("epoch"))
    version = row.get("version")
    same_version = [item for item in checkpoints if item.get("version") == version]
    for item in same_version:
        if _epoch_number(item.get("epoch")) == epoch:
            return item.get("checkpoint_path", "")
    best = [item for item in same_version if item.get("is_best_by_val") == "true"]
    if best:
        return best[0].get("checkpoint_path", "")
    last = [item for item in same_version if item.get("is_last") == "true"]
    return last[0].get("checkpoint_path", "") if last else ""


def _best_rows(rows: list[dict[str, Any]], *, mode: str) -> list[dict[str, Any]]:
    reverse = mode == "max"
    best: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = row.get("trial_id") or row.get("version")
        if key not in best:
            best[key] = row
            continue
        current = _sortable_score(row.get("score"), reverse)
        previous = _sortable_score(best[key].get("score"), reverse)
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
    run_id: str,
    version: str,
    trial_id: str,
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
                    "trial_id": trial_id,
                    "version": version,
                    "epoch": "" if epoch is None else epoch,
                    "split": _metric_split(metric),
                    "metric": metric,
                    "value": value,
                    "source": "wandb_history",
                    "metric_scope": _metric_scope(metric),
                    "wandb_run_id": run_id,
                    "updated_at": _now(),
                }
            )
    return rows


def _write_history_csv(path: Path, rows: list[dict[str, Any]], *, remote: str | None = None) -> None:
    if not rows:
        _write_rows_at(path, [], remote=remote)
        return
    fieldnames = sorted({key for row in rows for key in row})
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    _write_text_at(path, buffer.getvalue(), remote=remote)


def _merge_rows(existing: list[dict[str, str]], new_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    merged = []
    for row in [*existing, *new_rows]:
        key = tuple(
            str(row.get(field, "")) for field in ("trial_id", "version", "epoch", "metric", "source", "wandb_run_id")
        )
        if key in seen:
            continue
        seen.add(key)
        merged.append(row)
    return merged


def _monitor_report(rows: list[dict[str, Any]]) -> str:
    lines = ["# Experiment Monitor", ""]
    if not rows:
        return "# Experiment Monitor\n\nNo runs found.\n"
    lines.append("| trial | version | status | health | gpu | log age | checkpoints |")
    lines.append("|---|---|---|---|---|---:|---:|")
    for row in rows:
        lines.append(
            "| {trial} | {version} | {status} | {health} | {gpu} | {log_age} | {ckpts} |".format(
                trial=row.get("trial_id", ""),
                version=row.get("version", ""),
                status=row.get("status", ""),
                health=row.get("health_status", ""),
                gpu=str(row.get("gpu_summary", "")).replace("|", "/"),
                log_age=row.get("log_age_seconds", ""),
                ckpts=row.get("checkpoint_count", ""),
            )
        )
    lines.append("")
    return "\n".join(lines)


def _write_wandb_report(root: Path, rows: list[dict[str, Any]], *, remote: str | None = None) -> None:
    lines = ["# W&B Sync", "", f"Synced runs: {len(rows)}", ""]
    for row in rows[:20]:
        lines.append(f"- `{row.get('version')}`: {row.get('state', '')} {row.get('wandb_url', '')}")
    _write_text_at(root / "reports" / "wandb_rank.md", "\n".join(lines) + "\n", remote=remote)


def _write_rank_report(
    root: Path, metric: str, mode: str, rows: list[dict[str, Any]], *, remote: str | None = None
) -> None:
    lines = ["# Candidate Ranking", "", f"Metric: `{metric}` ({mode})", ""]
    if rows:
        lines.append("| rank | version | score | epoch | scope | checkpoint |")
        lines.append("|---:|---|---:|---:|---|---|")
        for row in rows:
            lines.append(
                f"| {row.get('rank')} | `{row.get('version')}` | {row.get('score')} | "
                f"{row.get('epoch', '')} | {row.get('metric_scope', '')} | `{row.get('checkpoint_path', '')}` |"
            )
    else:
        lines.append("No metric rows matched.")
    _write_text_at(root / "reports" / "wandb_rank.md", "\n".join(lines) + "\n", remote=remote)


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
        value = _float_or_none(summary.get(key))
        if value is not None:
            return str(int(value))
    return ""


def _record_epoch(record: dict[str, Any]) -> str | None:
    for key in ("epoch", "trainer/epoch", "current_epoch"):
        value = _float_or_none(record.get(key))
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
    return _epoch_from_checkpoint_name(clean)


def _checkpoint_step(name: str) -> str:
    match = re.search(r"step=(\d+)", name)
    return match.group(1) if match else ""


def _safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_") or "run"
