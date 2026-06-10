from __future__ import annotations

import json
from pathlib import Path
import subprocess
import time
from typing import Any

from .models import json_ready

PROGRESS_RELATIVE_PATH = Path("status") / "progress.json"
EVENTS_RELATIVE_PATH = Path("status") / "events.jsonl"


def progress_path(run_dir: str | Path) -> Path:
    return Path(run_dir).expanduser() / PROGRESS_RELATIVE_PATH


def write_progress(
    run_dir: str | Path,
    *,
    status: str,
    task: str,
    processed: int = 0,
    total: int | None = None,
    success: int = 0,
    failed: int = 0,
    start_time: float | None = None,
    current_item: str | None = None,
    message: str | None = None,
) -> Path:
    now = time.time()
    rate_per_min = None
    eta_minutes = None
    if start_time is not None and processed > 0:
        elapsed = max(now - start_time, 1e-9)
        rate_per_min = processed / elapsed * 60
        if total is not None and rate_per_min > 0:
            eta_minutes = max(total - processed, 0) / rate_per_min
    payload = {
        "status": status,
        "task": task,
        "processed": int(processed),
        "total": None if total is None else int(total),
        "success": int(success),
        "failed": int(failed),
        "rate_per_min": rate_per_min,
        "eta_minutes": eta_minutes,
        "current_item": current_item,
        "updated_at": _now(),
        "message": message,
    }
    path = progress_path(run_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n")
    tmp.replace(path)
    return path


def append_event(run_dir: str | Path, event: str, payload: dict[str, Any] | None = None) -> Path:
    path = Path(run_dir).expanduser() / EVENTS_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"event": event, "updated_at": _now(), **(payload or {})}
    with path.open("a") as file_obj:
        file_obj.write(json.dumps(json_ready(row), sort_keys=True) + "\n")
    return path


def read_progress(run_dir: str | Path, *, remote: str | None = None) -> dict[str, Any]:
    path = progress_path(run_dir)
    if remote:
        result = subprocess.run(
            ["ssh", remote, f"cat {_sh(path)}"],
            text=True,
            capture_output=True,
        )
        if result.returncode != 0:
            return {
                "status": "missing",
                "task": None,
                "path": str(path),
                "remote": remote,
                "message": result.stderr.strip() or "progress file not found",
            }
        raw = result.stdout
    else:
        if not path.exists():
            return {"status": "missing", "task": None, "path": str(path), "message": "progress file not found"}
        raw = path.read_text()
    data = json.loads(raw)
    if isinstance(data, dict):
        data.setdefault("path", str(path))
        if remote:
            data.setdefault("remote", remote)
        return data
    raise ValueError(f"Progress file must contain a JSON object: {path}")


def format_progress(data: dict[str, Any]) -> str:
    status = data.get("status") or "unknown"
    task = data.get("task") or "unknown"
    processed = data.get("processed")
    total = data.get("total")
    failed = data.get("failed")
    rate = data.get("rate_per_min")
    eta = data.get("eta_minutes")
    updated = data.get("updated_at")
    parts = [f"{task} {status}"]
    if processed is not None and total is not None:
        parts.append(f"{processed} / {total} done")
    if failed not in (None, ""):
        parts.append(f"{failed} failed")
    if rate:
        parts.append(f"{float(rate):.1f}/min")
    if eta is not None:
        parts.append(f"ETA {float(eta):.1f} min")
    if updated:
        parts.append(f"last update: {updated}")
    message = data.get("message")
    if message:
        parts.append(str(message))
    return "\n".join(parts) + "\n"


def _sh(value: Any) -> str:
    import shlex

    return shlex.quote(str(value))


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
