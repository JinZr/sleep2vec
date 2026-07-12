from __future__ import annotations

import calendar
import os
from pathlib import Path
import shlex
import subprocess
import time
from typing import Any

from . import run_artifacts as artifacts
from .experiment_io import REMOTE_MISSING_RETURN_CODE
from .experiment_workspace import TERMINAL_STATUSES, merge_run_row
from .manifests import utc_now
from .progress import read_progress

SSH_TIMEOUT_SECONDS = 10


def status_row(
    run_dir: Path,
    row: dict[str, Any],
    previous: dict[str, Any] | None = None,
    *,
    health: bool = False,
) -> dict[str, Any]:
    previous = previous or {}
    try:
        pid = read_pid(row.get("pid_path"), row)
    except RuntimeError as exc:
        observed_status = previous.get("status") or row.get("status") or "missing_pid"
        if not is_remote_row(row) and observed_status in {"planned", "pending"} and isinstance(exc.__cause__, OSError):
            raise
        pid = to_int(previous.get("pid") or row.get("pid"))
        running_state = None
        if is_remote_row(row):
            observed_status = "unknown_remote"
        elif observed_status in {"planned", "pending"}:
            observed_status = "missing_pid"
    else:
        running_state = process_running(row, pid) if pid is not None else False
        observed_status = row.get("status") or "unknown"
    running = bool(running_state)
    if observed_status in TERMINAL_STATUSES:
        pass
    elif pid is None and row.get("state") == "running":
        observed_status = "running"
    elif (
        pid is None
        and is_remote_row(row)
        and observed_status
        in {
            "launched",
            "running",
            "unknown_remote",
            "missing_pid",
        }
    ):
        observed_status = "unknown_remote"
    elif pid is None and observed_status == "launched":
        observed_status = "missing_pid"
    elif running_state is None:
        if is_remote_row(row):
            observed_status = "unknown_remote"
    elif running:
        observed_status = "running"
    elif observed_status in {"launched", "running"}:
        observed_status = "failed" if log_has_failure(row.get("log_path"), row) else "finished"
    manifest = artifacts.find_run_manifest(row)
    checkpoints = artifacts.checkpoint_names(row)
    observation = {
        **row,
        "status": observed_status,
        "pid": pid or "",
        "log_tail": log_tail(row.get("log_path"), row),
        "run_manifest": str(manifest or ""),
        "checkpoints": ";".join(checkpoints),
        "monitored_at": utc_now(),
    }
    output = merge_run_row(previous, observation)
    if health:
        output.update(health_fields(run_dir, row, previous, pid, running_state, output["status"], checkpoints))
    return output


def read_pid(path: Any, row: dict[str, Any] | None = None) -> int | None:
    if not path:
        return None
    if is_remote_row(row):
        script = f"""
import os
import sys

path = sys.argv[1]
try:
    os.lstat(path)
except FileNotFoundError:
    raise SystemExit({REMOTE_MISSING_RETURN_CODE})
except OSError as exc:
    print(exc, file=sys.stderr)
    raise SystemExit(1)

try:
    with open(path, encoding="utf-8") as file_obj:
        sys.stdout.write(file_obj.read())
except (OSError, UnicodeError) as exc:
    print(exc, file=sys.stderr)
    raise SystemExit(1)
"""
        result = run_row_command(
            row or {},
            f"python3 -c {shlex.quote(script)} {shlex.quote(str(path))}",
        )
        if result.returncode == REMOTE_MISSING_RETURN_CODE:
            return None
        if result.returncode != 0:
            detail = result.stderr.strip() or f"exit code {result.returncode}"
            raise RuntimeError(f"SSH PID read failed for {path} on {row['host']}: {detail}")
        text = result.stdout.strip()
    else:
        pid_path = Path(str(path))
        try:
            if not pid_path.exists() and not pid_path.is_symlink():
                return None
            text = pid_path.read_text().strip()
        except (OSError, UnicodeError) as exc:
            raise RuntimeError(f"PID file read failed: {path}") from exc
    try:
        pid = int(text)
    except ValueError as exc:
        raise RuntimeError(f"PID file is empty or invalid: {path}") from exc
    if pid <= 0:
        raise RuntimeError(f"PID file is empty or invalid: {path}")
    return pid


def process_running(row: dict[str, Any], pid: int | None) -> bool | None:
    if pid is None:
        return False
    if row.get("target") == "ssh" and row.get("host"):
        result = run_row_command(row, f"ps -p {pid} -o pid=")
        if result.returncode == 0:
            return str(pid) in result.stdout
        if result.returncode == 1:
            return False
        return None
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def log_has_failure(path: Any, row: dict[str, Any] | None = None) -> bool:
    if not path:
        return False
    if is_remote_row(row):
        result = run_row_command(row or {}, f"tail -n 100 {shlex.quote(str(path))}")
        if result.returncode != 0:
            return False
        tail = result.stdout
    else:
        log_path = Path(str(path))
        if not log_path.exists():
            return False
        tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-100:])
    return any(
        marker in tail
        for marker in [
            "Traceback",
            "RuntimeError",
            "CUDA out of memory",
            "Error executing job",
        ]
    )


def log_tail(path: Any, row: dict[str, Any] | None = None, lines: int = 8) -> str:
    if not path:
        return ""
    if is_remote_row(row):
        result = run_row_command(row or {}, f"tail -n {int(lines)} {shlex.quote(str(path))}")
        return result.stdout.strip() if result.returncode == 0 else ""
    log_path = Path(str(path))
    if not log_path.exists():
        return ""
    return "\n".join(log_path.read_text(errors="replace").splitlines()[-lines:])


def health_fields(
    run_dir: Path,
    row: dict[str, Any],
    previous: dict[str, Any],
    pid: int | None,
    running_state: bool | None,
    status: str,
    checkpoints: list[str],
) -> dict[str, Any]:
    progress = read_run_progress(run_dir, row)
    io_counts = proc_io(row, pid)
    read_bytes = io_counts.get("read_bytes")
    write_bytes = io_counts.get("write_bytes")
    read_delta = delta(read_bytes, previous.get("io_read_bytes"))
    write_delta = delta(write_bytes, previous.get("io_write_bytes"))
    log_age = log_age_seconds(row.get("log_path"), row)
    gpu = gpu_summary(row, pid)
    checkpoint_count = len(checkpoints)
    health_status = classify_health(
        status=status,
        running_state=running_state,
        gpu_summary=gpu,
        io_read_delta=read_delta,
        io_write_delta=write_delta,
        progress=progress,
        progress_is_fresh=progress_is_fresh(progress, previous),
        log_age_seconds=log_age,
        checkpoint_count=checkpoint_count,
        previous_checkpoint_count=to_int(previous.get("checkpoint_count")),
    )
    return {
        "health_status": health_status,
        "gpu_summary": gpu,
        "io_read_bytes": "" if read_bytes is None else read_bytes,
        "io_write_bytes": "" if write_bytes is None else write_bytes,
        "io_read_delta_bytes": "" if read_delta is None else read_delta,
        "io_write_delta_bytes": "" if write_delta is None else write_delta,
        "progress_status": progress.get("status", ""),
        "progress_processed": progress.get("processed", ""),
        "progress_total": progress.get("total", ""),
        "progress_updated_at": progress.get("updated_at", ""),
        "progress_age_seconds": progress_age_seconds(progress),
        "log_age_seconds": "" if log_age is None else log_age,
        "checkpoint_count": checkpoint_count,
    }


def read_run_progress(run_dir: Path, row: dict[str, Any]) -> dict[str, Any]:
    progress_dir = row.get("progress_dir") or row.get("workdir") or run_dir
    try:
        return read_progress(progress_dir, remote=row.get("host") if is_remote_row(row) else None)
    except Exception as exc:
        return {"status": "unknown", "message": str(exc)}


def proc_io(row: dict[str, Any], pid: int | None) -> dict[str, int]:
    if pid is None:
        return {}
    result = run_row_command(row, f"cat /proc/{int(pid)}/io")
    if result.returncode != 0:
        return {}
    counts: dict[str, int] = {}
    for line in result.stdout.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        try:
            counts[key.strip()] = int(value.strip())
        except ValueError:
            pass
    return counts


def gpu_summary(row: dict[str, Any], pid: int | None) -> str:
    if pid is None:
        return ""
    apps = run_row_command(
        row,
        "nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader,nounits",
    )
    if apps.returncode != 0:
        return ""
    matched = []
    for line in apps.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if parts and parts[0] == str(pid):
            matched.append(line.strip())
    if not matched:
        return ""
    gpu_state = run_row_command(
        row,
        "nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits",
    )
    summary = "; ".join(matched)
    if gpu_state.returncode == 0 and gpu_state.stdout.strip():
        summary = f"{summary} | gpu={gpu_state.stdout.strip().replace(chr(10), '; ')}"
    return summary


def log_age_seconds(path: Any, row: dict[str, Any]) -> int | None:
    if not path:
        return None
    if is_remote_row(row):
        quoted = shlex.quote(str(path))
        result = run_row_command(
            row,
            f"now=$(date +%s); m=$(stat -c %Y {quoted} 2>/dev/null) || exit 1; echo $((now-m))",
        )
        if result.returncode != 0:
            return None
        return to_int(result.stdout.strip())
    log_path = Path(str(path))
    if not log_path.exists():
        return None
    return int(time.time() - log_path.stat().st_mtime)


def classify_health(
    *,
    status: str,
    running_state: bool | None,
    gpu_summary: str,
    io_read_delta: int | None,
    io_write_delta: int | None,
    progress: dict[str, Any],
    progress_is_fresh: bool,
    log_age_seconds: int | None,
    checkpoint_count: int,
    previous_checkpoint_count: int | None,
) -> str:
    if status == "unknown_remote" or running_state is None:
        return "unknown_remote"
    if status == "failed":
        return "failed"
    if status == "finished":
        return "finished"
    if not running_state:
        return status
    if gpu_summary:
        return "compute_active"
    if (io_read_delta or 0) > 0 or (io_write_delta or 0) > 0:
        return "data_loading"
    if progress.get("status") == "running" and progress_is_fresh:
        return "healthy_running"
    if log_age_seconds is not None and log_age_seconds < 300:
        return "healthy_running"
    if previous_checkpoint_count is not None and checkpoint_count > previous_checkpoint_count:
        return "healthy_running"
    return "possibly_stalled"


def delta(current: int | None, previous: Any) -> int | None:
    old = to_int(previous)
    if current is None or old is None:
        return None
    return max(int(current) - old, 0)


def to_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def progress_is_fresh(progress: dict[str, Any], previous: dict[str, Any]) -> bool:
    if progress.get("status") != "running":
        return False
    processed = to_int(progress.get("processed"))
    previous_processed = to_int(previous.get("progress_processed"))
    if processed is not None and previous_processed is not None and processed > previous_processed:
        return True
    age = progress_age_seconds(progress)
    return age is not None and age < 300


def progress_age_seconds(progress: dict[str, Any]) -> int | None:
    updated = progress.get("updated_at")
    if not updated:
        return None
    try:
        parsed = time.strptime(str(updated), "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        return None
    return max(int(time.time() - calendar.timegm(parsed)), 0)


def run_row_command(row: dict[str, Any], command: str) -> subprocess.CompletedProcess:
    try:
        if is_remote_row(row):
            return subprocess.run(
                ["ssh", str(row["host"]), command],
                text=True,
                capture_output=True,
                timeout=SSH_TIMEOUT_SECONDS,
            )
        return subprocess.run(
            ["bash", "-lc", command],
            text=True,
            capture_output=True,
            timeout=SSH_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        args = exc.cmd if isinstance(exc.cmd, list) else [str(exc.cmd)]
        return subprocess.CompletedProcess(args, 124, "", f"timed out after {SSH_TIMEOUT_SECONDS}s")


def is_remote_row(row: dict[str, Any] | None) -> bool:
    return bool(row and row.get("target") == "ssh" and row.get("host"))
