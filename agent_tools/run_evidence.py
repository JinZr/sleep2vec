from __future__ import annotations

import base64
import calendar
import codecs
import json
import locale
import os
from pathlib import Path
import signal
import stat
import subprocess
import time
from typing import Any, BinaryIO, TypedDict, TypeGuard, cast

from . import run_artifacts as artifacts, transport
from .experiment_io import REMOTE_MISSING_RETURN_CODE
from .experiment_workspace import (
    MONITOR_EXIT_CODE_PREFIX,
    PROCESS_IDENTITY_FIELDS,
    TERMINAL_STATUSES,
    file_sha256,
    merge_run_row,
)
from .manifests import read_json, utc_now
from .models import is_full_git_object_id
from .progress import read_progress
from .transport import SSH_TIMEOUT_SECONDS

RUN_EVIDENCE_FIELDS = {
    "target",
    "host",
    "workdir",
    "gpus",
    "pid_path",
    "pid",
    "process_group_id",
    "process_start_token",
    "runtime_commit",
    "process_identity_error",
    "log_path",
    "log_tail",
    "log_age_seconds",
    "command",
    "launched_at",
    "monitored_at",
    "stop_requested_at",
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
RUN_STATUS_FIELDS = RUN_EVIDENCE_FIELDS | {"status"}


class _RequiredProcessIdentity(TypedDict):
    pid: int
    process_group_id: int
    process_start_token: str


class ProcessIdentity(_RequiredProcessIdentity, total=False):
    runtime_commit: str


class ProcessIdentityError(RuntimeError):
    pass


def status_row(  # noqa: C901
    run_dir: Path,
    row: dict[str, Any],
    previous: dict[str, Any] | None = None,
    *,
    script_commits_terminal_status: bool,
    health: bool = False,
) -> dict[str, Any]:
    previous = previous or {}
    process_identity = None
    committed_process_identity = None
    dead_unbound_process_identity = False
    process_identity_error = None
    managed_process = any(source.get("script") not in (None, "") for source in (row, previous))
    managed_process_identity = managed_process and any(
        source.get("pid_path") not in (None, "") for source in (row, previous)
    )
    try:
        if managed_process:
            canonical_process_identity = {
                field: previous[field] if field in previous else row.get(field) for field in PROCESS_IDENTITY_FIELDS
            }
            populated_process_fields = {
                field for field, value in canonical_process_identity.items() if value not in (None, "")
            }
            if populated_process_fields and populated_process_fields != PROCESS_IDENTITY_FIELDS:
                missing = ", ".join(sorted(PROCESS_IDENTITY_FIELDS - populated_process_fields))
                raise ProcessIdentityError(f"Canonical run has partial process identity; missing: {missing}")
            managed_process_identity = managed_process_identity or populated_process_fields == PROCESS_IDENTITY_FIELDS
            process_identity = read_process_identity(row.get("pid_path"), row)
            pid = process_identity["pid"] if process_identity is not None else None
            running_state = process_identity_running(row, process_identity) if process_identity is not None else False
            if process_identity is not None:
                if populated_process_fields == PROCESS_IDENTITY_FIELDS:
                    committed_process_identity = process_identity
                elif running_state is True:
                    _require_process_script(
                        process_identity["pid"],
                        previous.get("script") or row.get("script"),
                        row,
                    )
                    committed_process_identity = process_identity
                else:
                    # A dead leader cannot bind previously unfrozen PID evidence to the launch script.
                    dead_unbound_process_identity = running_state is False
                    running_state = None
        else:
            pid = read_pid(row.get("pid_path"), row)
            running_state = process_running(row, pid) if pid is not None else False
    except ProcessIdentityError as exc:
        # Corrupt, incomplete, or reused identity is confirmed unsafe evidence, not a transient probe failure.
        observed_status = previous.get("status") or row.get("status") or "missing_pid"
        pid = to_int(previous.get("pid") or row.get("pid"))
        running_state = None
        process_identity_error = str(exc)
        if observed_status not in TERMINAL_STATUSES:
            observed_status = "missing_pid"
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
    elif (
        pid is None
        and running_state is False
        and managed_process_identity
        and observed_status in {"launched", "running"}
    ):
        observed_status = "missing_pid"
    elif pid is None and observed_status == "launched":
        observed_status = "missing_pid"
    elif running_state is None:
        if is_remote_row(row):
            observed_status = "unknown_remote"
        elif dead_unbound_process_identity:
            observed_status = "missing_pid"
    elif running:
        observed_status = "running"
    elif observed_status in {"launched", "running", "unknown_remote"}:
        if script_commits_terminal_status:
            # Lifecycle-enabled scripts commit their own terminal status; disappearance without it is failure.
            observed_status = "failed"
        else:
            owner = previous.get("terminal_status_owner") or row.get("terminal_status_owner")
            log_failed = log_has_failure(
                row.get("log_path"),
                row,
                require_exit_code=owner == "monitor",
            )
            if log_failed is None and is_remote_row(row):
                observed_status = "unknown_remote"
            else:
                observed_status = "failed" if log_failed else "finished"
    if script_commits_terminal_status and previous.get("status") == "stopping" and previous.get("stop_requested_at"):
        # The stop manager owns final stopped evidence; a live probe cannot release its recorded intent.
        observed_status = "stopping"
    # Remote artifacts must be observed on the execution host; transport uncertainty preserves prior evidence.
    manifest = str(previous.get("run_manifest") or row.get("run_manifest") or "")
    checkpoints = [name for name in str(previous.get("checkpoints") or row.get("checkpoints") or "").split(";") if name]
    artifact_row = {
        **row,
        **{field: previous[field] for field in ("runtime_dir", "checkpoint_dir") if field in previous},
    }
    observed_artifacts = runtime_artifacts(artifact_row)
    health_checkpoints = None
    if observed_artifacts is not None:
        manifest, _manifest_data, checkpoints = observed_artifacts
        health_checkpoints = checkpoints
    observation = {
        **row,
        **(committed_process_identity or {}),
        "status": observed_status,
        "pid": (committed_process_identity["pid"] if committed_process_identity else None)
        or previous.get("pid")
        or row.get("pid")
        or "",
        "log_tail": log_tail(row.get("log_path"), row),
        "run_manifest": str(manifest or ""),
        "checkpoints": ";".join(checkpoints),
        "monitored_at": utc_now(),
    }
    if process_identity_error:
        observation["process_identity_error"] = process_identity_error
    output = merge_run_row(previous, observation)
    if health:
        output.update(health_fields(run_dir, row, previous, pid, running_state, output["status"], health_checkpoints))
    return output


def runtime_artifacts(row: dict[str, Any]) -> tuple[str, dict[str, Any], list[str]] | None:
    if is_remote_row(row):
        result = run_row_command(
            row,
            transport.remote_python_program_command(
                "run_evidence.runtime_artifacts",
                str(row.get("runtime_dir") or ""),
                str(row.get("checkpoint_dir") or ""),
            ),
        )
        if result.returncode == 0:
            try:
                artifact_payload = json.loads(result.stdout)
            except (TypeError, json.JSONDecodeError):
                raise RuntimeError(f"SSH runtime artifact observation returned malformed output on {row['host']}.")
            if not isinstance(artifact_payload, dict) or not isinstance(artifact_payload.get("checkpoints"), list):
                raise RuntimeError(f"SSH runtime artifact observation returned malformed output on {row['host']}.")
            manifest = artifact_payload.get("manifest", {})
            if not isinstance(manifest, dict):
                raise RuntimeError(f"SSH runtime artifact observation returned malformed output on {row['host']}.")
            return (
                str(artifact_payload.get("run_manifest") or ""),
                manifest,
                [str(name) for name in artifact_payload["checkpoints"]],
            )
        if result.returncode in {124, 255}:
            return None
        detail = result.stderr.strip() or f"exit code {result.returncode}"
        raise RuntimeError(f"SSH runtime artifact observation failed on {row['host']}: {detail}")
    manifest_path = artifacts.find_run_manifest(row)
    manifest = read_json(manifest_path) if manifest_path else {}
    return str(manifest_path or ""), manifest, artifacts.checkpoint_names(row)


def checkpoint_file_sha256(row: dict[str, Any], checkpoint_path: str | Path) -> str:
    # Checkpoint bytes belong to the frozen execution target; a manager-local namesake is not SSH evidence.
    if row.get("target") == "ssh" and not row.get("host"):
        raise ValueError("Managed SSH checkpoint evidence requires a host.")
    if is_remote_row(row):
        result = run_row_command(
            row,
            transport.remote_python_program_command("run_evidence.checkpoint_file_sha256", str(checkpoint_path)),
        )
        if result.returncode != 0:
            detail = result.stderr.strip() or f"exit code {result.returncode}"
            raise RuntimeError(f"SSH checkpoint SHA-256 failed on {row['host']}: {detail}")
        digest = result.stdout.strip()
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise RuntimeError(f"SSH checkpoint SHA-256 returned malformed output on {row['host']}.")
        return digest
    path = Path(checkpoint_path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Checkpoint is not a physical regular file: {path}")
    return file_sha256(path)


def read_pid(
    path: Any,
    row: dict[str, Any] | None = None,
    *,
    expected_script: str | Path | None = None,
) -> int | None:
    text = _read_pid_text(path, row)
    if text is None:
        return None
    if text.startswith("{"):
        identity = _parse_process_identity(text, path)
        if expected_script is not None:
            state = process_identity_running(row or {}, identity)
            if state is not True:
                raise RuntimeError(f"Cannot verify PID {identity['pid']} process identity.")
            _require_process_script(identity["pid"], expected_script, row)
        return identity["pid"]
    try:
        pid = int(text)
    except ValueError as exc:
        raise RuntimeError(f"PID file is empty or invalid: {path}") from exc
    if pid <= 0:
        raise RuntimeError(f"PID file is empty or invalid: {path}")
    if expected_script is not None:
        raise ProcessIdentityError(f"PID file lacks process group identity: {path}")
    return pid


def read_process_identity(
    path: Any,
    row: dict[str, Any] | None = None,
    *,
    expected_script: str | Path | None = None,
) -> ProcessIdentity | None:
    text = _read_pid_text(path, row)
    if text is None:
        return None
    if not text.startswith("{"):
        try:
            legacy_pid = int(text)
        except ValueError as exc:
            raise ProcessIdentityError(f"PID file is empty or invalid: {path}") from exc
        if legacy_pid <= 0:
            raise ProcessIdentityError(f"PID file is empty or invalid: {path}")
        raise ProcessIdentityError(f"PID file lacks process group identity: {path}")
    identity = _parse_process_identity(text, path)
    if expected_script is not None:
        _require_process_script(identity["pid"], expected_script, row)
    return identity


def _require_process_script(pid: int, expected_script: str | Path | None, row: dict[str, Any] | None) -> None:
    script_path = Path(str(expected_script or ""))
    if not script_path.is_absolute():
        raise RuntimeError(f"Frozen run script is not absolute: {expected_script}")
    result = run_row_command(row or {}, f"ps -ww -p {pid} -o args=")
    if result.returncode != 0 or not result.stdout.strip():
        location = f" on {row['host']}" if is_remote_row(row) else ""
        detail = result.stderr.strip() or f"exit code {result.returncode}"
        raise RuntimeError(f"Cannot verify PID {pid} process identity{location}: {detail}")
    process_args = result.stdout.rstrip("\r\n")
    expected_suffix = f"bash {script_path}"
    prefix = process_args[: -len(expected_suffix)] if process_args.endswith(expected_suffix) else ""
    allowed_prefix = not prefix or (prefix.endswith("/") and not any(character.isspace() for character in prefix))
    if not process_args.endswith(expected_suffix) or not allowed_prefix:
        raise ProcessIdentityError(f"PID {pid} process identity does not match frozen script: {script_path}")


def _read_pid_text(path: Any, row: dict[str, Any] | None) -> str | None:
    if not path:
        return None
    if is_remote_row(row):
        result = run_row_command(
            row or {},
            transport.remote_python_program_command("run_evidence.read_pid_text", str(path)),
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
            info = os.lstat(pid_path)
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise RuntimeError(f"PID file read failed: {path}") from exc
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise RuntimeError(f"PID file is not an independent regular file: {path}")
        try:
            if not pid_path.exists():
                return None
            text = pid_path.read_text().strip()
        except (OSError, UnicodeError) as exc:
            raise RuntimeError(f"PID file read failed: {path}") from exc
    if not text:
        raise RuntimeError(f"PID file is empty or invalid: {path}")
    return text


def _parse_process_identity(text: str, path: Any) -> ProcessIdentity:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ProcessIdentityError(f"PID file is empty or invalid: {path}") from exc
    allowed_fields = PROCESS_IDENTITY_FIELDS | {"runtime_commit"}
    if not isinstance(payload, dict) or not PROCESS_IDENTITY_FIELDS <= set(payload) or set(payload) - allowed_fields:
        raise ProcessIdentityError(f"PID file has incomplete process group identity: {path}")
    pid = payload.get("pid")
    pgid = payload.get("process_group_id")
    token = payload.get("process_start_token")
    if (
        type(pid) is not int
        or type(pgid) is not int
        or pid <= 0
        or pgid != pid
        or not isinstance(token, str)
        or not token
    ):
        raise ProcessIdentityError(f"PID file has invalid process group identity: {path}")
    identity: ProcessIdentity = {"pid": pid, "process_group_id": pgid, "process_start_token": token}
    if "runtime_commit" in payload:
        runtime_commit = payload["runtime_commit"]
        if not is_full_git_object_id(runtime_commit):
            raise ProcessIdentityError(f"PID file has invalid runtime commit: {path}")
        identity["runtime_commit"] = runtime_commit
    return identity


def process_identity_running(row: dict[str, Any], identity: ProcessIdentity) -> bool | None:
    _require_matching_process_identity(row, identity)
    result = run_row_command(
        row,
        transport.remote_python_program_command(
            "run_evidence.process_probe",
            identity["pid"],
            identity["process_group_id"],
        ),
    )
    if result.returncode != 0:
        return None
    try:
        payload = json.loads(result.stdout)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ProcessIdentityError(
            f"Process identity probe returned malformed output for PID {identity['pid']}."
        ) from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("group_running"), bool):
        raise ProcessIdentityError(f"Process identity probe returned malformed output for PID {identity['pid']}.")
    leader = payload.get("leader")
    if leader is not None:
        if not isinstance(leader, dict) or any(
            str(leader.get(field)) != str(identity[field]) for field in PROCESS_IDENTITY_FIELDS
        ):
            raise ProcessIdentityError(f"PID {identity['pid']} was reused by a different process.")
    return payload["group_running"]


def _require_matching_process_identity(row: dict[str, Any], identity: ProcessIdentity) -> None:
    for field in PROCESS_IDENTITY_FIELDS:
        expected = row.get(field)
        if expected not in (None, "") and str(expected) != str(identity[field]):
            raise ProcessIdentityError(f"PID file differs from canonical {field}: {identity['pid']}")


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


def stop_process_group(row: dict[str, Any], identity: ProcessIdentity, *, timeout: float = 5.0) -> None:
    _require_matching_process_identity(row, identity)
    pid = identity["pid"]
    pgid = identity["process_group_id"]
    if is_remote_row(row):
        # Verification and signal share one remote process so PID reuse cannot race a second SSH call.
        result = run_row_command(
            row,
            transport.remote_python_program_command(
                "run_evidence.process_stop",
                pid,
                pgid,
                identity["process_start_token"],
                timeout,
            ),
        )
        if result.returncode != 0:
            detail = result.stderr.strip() or f"exit code {result.returncode}"
            raise RuntimeError(f"Failed to stop remote process group {pgid} on {row['host']}: {detail}")
        return
    if pgid != pid or pgid == os.getpgrp():
        raise RuntimeError(f"Refusing to signal unsafe process group: {pgid}")
    running = process_identity_running(row, identity)
    if running is not True:
        raise RuntimeError(f"Cannot verify managed process group before stop: {pgid}")
    os.killpg(pgid, signal.SIGTERM)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        running = process_identity_running(row, identity)
        if running is None:
            raise RuntimeError(f"Cannot verify managed process group after stop: {pgid}")
        if not running:
            return
        time.sleep(0.05)
    running = process_identity_running(row, identity)
    if running is None:
        raise RuntimeError(f"Cannot verify managed process group after stop: {pgid}")
    if not running:
        return
    raise RuntimeError(f"Managed process group did not stop after SIGTERM: {pgid}")


def log_has_failure(
    path: Any,
    row: dict[str, Any] | None = None,
    *,
    require_exit_code: bool = False,
) -> bool | None:
    if not path:
        return require_exit_code
    if is_remote_row(row):
        result = run_row_command(
            row or {},
            transport.remote_python_program_command("run_evidence.log_tail", str(path)),
        )
        if result.returncode == REMOTE_MISSING_RETURN_CODE:
            tail = ""
        elif result.returncode != 0:
            return None
        else:
            tail = result.stdout
    else:
        log_path = Path(str(path))
        if not log_path.exists():
            tail = ""
        else:
            tail = _local_log_tail(log_path, 100)
    lines = tail.splitlines()
    final_line = lines[-1] if lines else ""
    # srun --label prefixes the rank-zero terminal marker; other ranks cannot own it.
    if str((row or {}).get("scheduler_type") or "") == "slurm" and final_line.startswith("0:"):
        final_line = final_line[2:].lstrip()
    if final_line.startswith(MONITOR_EXIT_CODE_PREFIX):
        raw_exit_code = final_line.removeprefix(MONITOR_EXIT_CODE_PREFIX)
        valid_exit_code = raw_exit_code and raw_exit_code.isascii() and raw_exit_code.isdecimal()
        if valid_exit_code:
            exit_code = int(raw_exit_code)
            if exit_code <= 255:
                return exit_code != 0
        if require_exit_code:
            return True
    elif require_exit_code:
        return True
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
        result = run_row_command(row or {}, f"tail -n {int(lines)} {transport.sh(path)}")
        return result.stdout.strip() if result.returncode == 0 else ""
    log_path = Path(str(path))
    if not log_path.exists():
        return ""
    return _local_log_tail(log_path, lines)


def _local_log_tail(path: Path, lines: int) -> str:
    with path.open(errors="replace") as source:
        info = os.fstat(source.fileno())
        if (
            lines <= 0
            or codecs.lookup(source.encoding).name != "utf-8"
            or not source.seekable()
            or not stat.S_ISREG(info.st_mode)
            or not info.st_size
        ):
            return "\n".join(source.read().splitlines()[-lines:])
        # Path.open uses a buffered binary reader underneath this text stream.
        buffer = cast(BinaryIO, source.buffer)
        end = buffer.seek(0, os.SEEK_END)
        window = 65536
        while True:
            start = max(0, end - window)
            buffer.seek(start)
            parts = buffer.read(end - start).decode("utf-8", errors="replace").splitlines()
            # A nonzero window may start inside a UTF-8 character or line separator; discard that first line.
            if start == 0 or len(parts) > lines:
                return "\n".join(parts[-lines:])
            window *= 2


def log_tail_and_age(path: Any, row: dict[str, Any], lines: int = 8) -> tuple[str, int | None]:
    if not path or not is_remote_row(row):
        return log_tail(path, row, lines), log_age_seconds(path, row)
    result = run_row_command(
        row,
        transport.remote_python_program_command("run_evidence.log_tail_and_age", str(path), int(lines)),
    )
    try:
        if result.returncode != 0:
            raise ValueError("Remote log probe failed.")
        payload = json.loads(result.stdout)
        if (
            not isinstance(payload, list)
            or len(payload) != 2
            or any(
                not isinstance(part, list)
                or len(part) != 3
                or type(part[0]) is not int
                or not all(isinstance(value, str) for value in part[1:])
                for part in payload
            )
        ):
            raise ValueError("Remote log probe returned an incomplete response.")
        decoded = [(part[0], *(base64.b64decode(value, validate=True) for value in part[1:])) for part in payload]
    except (TypeError, ValueError):
        # A broken paired response is not missing evidence; retain the two exact probes for this observation.
        return log_tail(path, row, lines), log_age_seconds(path, row)
    outputs = []
    for returncode, stdout, stderr in decoded:
        encoding = locale.getpreferredencoding(False)
        text = stdout.decode(encoding).replace("\r\n", "\n").replace("\r", "\n")
        stderr.decode(encoding)  # Preserve the original text subprocess's strict decoding, including stderr.
        outputs.append(text.strip() if returncode == 0 else "")
    return outputs[0], to_int(outputs[1])


def health_fields(
    run_dir: Path,
    row: dict[str, Any],
    previous: dict[str, Any],
    pid: int | None,
    running_state: bool | None,
    status: str,
    checkpoints: list[str] | None,
) -> dict[str, Any]:
    progress = read_run_progress(run_dir, row)
    io_counts = proc_io(row, pid)
    read_bytes = io_counts.get("read_bytes")
    write_bytes = io_counts.get("write_bytes")
    read_delta = delta(read_bytes, previous.get("io_read_bytes"))
    write_delta = delta(write_bytes, previous.get("io_write_bytes"))
    log_age = log_age_seconds(row.get("log_path"), row)
    gpu = gpu_summary(row, pid)
    checkpoint_count = len(checkpoints) if checkpoints is not None else None
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
        "gpu_summary": "" if gpu is None else gpu,
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
        "checkpoint_count": "" if checkpoint_count is None else checkpoint_count,
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


def gpu_summary(row: dict[str, Any], pid: int | None) -> str | None:
    if pid is None:
        return ""
    gpu_probe_required = bool(str(row.get("gpus") or "").strip())
    apps = run_row_command(
        row,
        "nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader,nounits",
    )
    if apps.returncode != 0:
        return None if gpu_probe_required else ""
    app_rows = []
    for line in apps.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        app_pid = to_int(parts[0]) if parts else None
        if app_pid is not None:
            app_rows.append((app_pid, line.strip()))
    if not app_rows:
        return ""
    managed_pids = {pid}
    process_group_id = to_int(row.get("process_group_id"))
    if process_group_id is not None:
        processes = run_row_command(row, "ps -eo pid=,pgid=")
        if processes.returncode == 0:
            for line in processes.stdout.splitlines():
                parts = line.split()
                if len(parts) != 2:
                    continue
                process_pid = to_int(parts[0])
                process_pgid = to_int(parts[1])
                if process_pid is not None and process_pgid == process_group_id:
                    managed_pids.add(process_pid)
        elif not any(app_pid == pid for app_pid, _line in app_rows):
            return None if gpu_probe_required else ""
    matched = [line for app_pid, line in app_rows if app_pid in managed_pids]
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
        quoted = transport.sh(path)
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
    gpu_summary: str | None,
    io_read_delta: int | None,
    io_write_delta: int | None,
    progress: dict[str, Any],
    progress_is_fresh: bool,
    log_age_seconds: int | None,
    checkpoint_count: int | None,
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
    if (
        checkpoint_count is not None
        and previous_checkpoint_count is not None
        and checkpoint_count > previous_checkpoint_count
    ):
        return "healthy_running"
    if gpu_summary is None or checkpoint_count is None or previous_checkpoint_count is None:
        return "health_unknown"
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
    host = str(row["host"]) if is_remote_row(row) else None
    return transport.run_shell(host, command, timeout=SSH_TIMEOUT_SECONDS, swallow_timeout=True)


def is_remote_row(row: dict[str, Any] | None) -> TypeGuard[dict[str, Any]]:
    return bool(row and row.get("target") == "ssh" and row.get("host"))
