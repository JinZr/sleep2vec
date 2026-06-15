from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from sleep2stat.io.writers import COMPLETION_MARKER, _utc_now, _write_json


def resume_status(run_dir: Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    if not run_dir.exists():
        return {"run_dir": str(run_dir), "status": "missing", "message": "run directory not found"}
    record_ids = _read_record_ids(run_dir / "record_manifest.csv")
    success_ids = _read_success_ids(run_dir)
    failures = _read_failures(run_dir / "status" / "failures.csv")
    has_global_failure = _has_global_failure(failures)
    failed_ids = {
        str(row.get("record_id"))
        for row in failures
        if row.get("record_id") not in (None, "", "__all__") and str(row.get("record_id")) not in success_ids
    }
    if has_global_failure:
        # "__all__" means setup failed before per-record work; unresolved records failed at run level.
        failed_ids.update(record_ids - success_ids)
    pending_ids = sorted(record_ids - success_ids - failed_ids)
    progress = _read_json(run_dir / "status" / "progress.json")
    pid_info = _read_json(run_dir / "status" / "pid.json")
    pid = _coerce_int(pid_info.get("pid")) if isinstance(pid_info, dict) else None
    pid_state = _pid_state(pid)
    status = _classify_status(
        total_records=len(record_ids),
        success_records=len(success_ids),
        failed_records=len(failed_ids),
        failure_rows=len(failures),
        pending_records=len(pending_ids),
        progress_status=progress.get("status") if isinstance(progress, dict) else None,
        pid_state=pid_state,
    )
    return {
        "run_dir": str(run_dir),
        "status": status,
        "total_records": len(record_ids),
        "success_records": len(success_ids),
        "failed_records": len(failed_ids),
        "failure_rows": len(failures),
        "global_failures": sum(1 for row in failures if row.get("record_id") == "__all__"),
        "pending_records": len(pending_ids),
        "pending_record_ids": pending_ids[:20],
        "progress_status": progress.get("status") if isinstance(progress, dict) else None,
        "pid": pid,
        "pid_state": pid_state,
    }


def scan_resume_status(run_root: Path, pattern: str) -> dict[str, Any]:
    run_root = Path(run_root)
    runs = [resume_status(path) for path in sorted(run_root.glob(pattern)) if path.is_dir()]
    return {
        "run_root": str(run_root),
        "glob": pattern,
        "runs": runs,
        "dead_runs": [run["run_dir"] for run in runs if run.get("status") == "stale_running"],
        "incomplete_runs": [run["run_dir"] for run in runs if run.get("pending_records", 0) > 0],
    }


def repair_run_status(run_dir: Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    status = resume_status(run_dir)
    if status.get("status") == "missing":
        return status
    repaired_status = _repair_status(status)
    progress_path = run_dir / "status" / "progress.json"
    progress = _read_json(progress_path)
    payload = progress if isinstance(progress, dict) else {}
    payload.update(
        {
            "status": repaired_status,
            "total_records": int(status.get("total_records", 0)),
            "completed_records": int(status.get("success_records", 0)),
            "failed_records": int(status.get("failed_records", 0)),
            "pending_records": int(status.get("pending_records", 0)),
            "updated_at_utc": _utc_now(),
        }
    )
    _write_json(payload, progress_path)
    manifest_path = run_dir / "run_manifest.json"
    manifest = _read_json(manifest_path)
    if isinstance(manifest, dict):
        manifest["status"] = repaired_status
        manifest["updated_at_utc"] = _utc_now()
        _write_json(manifest, manifest_path)
    repaired = resume_status(run_dir)
    repaired["repair_status"] = repaired_status
    return repaired


def format_resume_status(data: dict[str, Any]) -> str:
    if "runs" in data:
        lines = [f"run_root: {data['run_root']}", f"glob: {data['glob']}"]
        for run in data["runs"]:
            lines.append(_format_one(run))
        return "\n".join(lines) + "\n"
    return _format_one(data) + "\n"


def _format_one(data: dict[str, Any]) -> str:
    return (
        f"{data.get('run_dir')}: {data.get('status')} "
        f"success={data.get('success_records', 0)} "
        f"failed={data.get('failed_records', 0)} "
        f"pending={data.get('pending_records', 0)} "
        f"pid={data.get('pid') or 'unknown'} "
        f"pid_state={data.get('pid_state') or 'unknown'}"
    )


def _classify_status(
    *,
    total_records: int,
    success_records: int,
    failed_records: int,
    failure_rows: int,
    pending_records: int,
    progress_status: str | None,
    pid_state: str,
) -> str:
    if total_records > 0 and pending_records == 0:
        return "completed_with_failures" if failed_records or failure_rows else "completed"
    if progress_status == "running":
        if pid_state == "dead":
            return "stale_running"
        if pid_state == "alive":
            return "running"
        return "running_liveness_unknown"
    if progress_status == "interrupted":
        return "interrupted"
    if pending_records > 0:
        return "incomplete"
    return progress_status or "unknown"


def _repair_status(status: dict[str, Any]) -> str:
    if status.get("total_records", 0) > 0 and status.get("pending_records", 0) == 0:
        return "completed_with_failures" if status.get("failure_rows", 0) else "completed"
    if status.get("pid_state") == "dead":
        return "interrupted"
    return str(status.get("status") or "unknown")


def _read_record_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    try:
        frame = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return set()
    if "record_id" not in frame.columns:
        return set()
    return set(frame["record_id"].astype(str))


def _read_success_ids(run_dir: Path) -> set[str]:
    per_record = run_dir / "per_record"
    if not per_record.exists():
        return set()
    return {marker.parent.name for marker in per_record.glob(f"*/{COMPLETION_MARKER}")}


def _read_failures(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        frame = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return []
    return frame.fillna("").to_dict("records")


def _has_global_failure(failures: list[dict[str, Any]]) -> bool:
    return any(str(row.get("record_id")) == "__all__" for row in failures)


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _coerce_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _pid_state(pid: int | None) -> str:
    if pid is None:
        return "unknown"
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return "dead"
    except PermissionError:
        return "alive"
    except OSError:
        return "dead"
    return "alive"
