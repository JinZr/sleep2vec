from __future__ import annotations

import calendar
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import time
from typing import Any

from . import run_artifacts as artifacts, transport
from .experiment_io import REMOTE_MISSING_RETURN_CODE
from .experiment_workspace import PROCESS_IDENTITY_FIELDS, TERMINAL_STATUSES, merge_run_row
from .manifests import read_json, utc_now
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
    "process_identity_error",
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
RUN_STATUS_FIELDS = RUN_EVIDENCE_FIELDS | {"status"}


class ProcessIdentityError(RuntimeError):
    pass


_PROCESS_START_TOKEN_CODE = """
def start_token(pid):
    try:
        stat_text = open(f"/proc/{pid}/stat", encoding="utf-8").read()
    except OSError:
        if sys.platform == "darwin":
            # macOS has no /proc; libproc exposes the kernel start timestamp without a fragile ps parse.
            import ctypes

            class ProcBsdInfo(ctypes.Structure):
                _fields_ = [
                    ("flags", ctypes.c_uint32), ("status", ctypes.c_uint32),
                    ("xstatus", ctypes.c_uint32), ("pid", ctypes.c_uint32),
                    ("ppid", ctypes.c_uint32), ("uid", ctypes.c_uint32),
                    ("gid", ctypes.c_uint32), ("ruid", ctypes.c_uint32),
                    ("rgid", ctypes.c_uint32), ("svuid", ctypes.c_uint32),
                    ("svgid", ctypes.c_uint32), ("rfu", ctypes.c_uint32),
                    ("comm", ctypes.c_char * 16), ("name", ctypes.c_char * 32),
                    ("nfiles", ctypes.c_uint32), ("pgid", ctypes.c_uint32),
                    ("pjobc", ctypes.c_uint32), ("e_tdev", ctypes.c_uint32),
                    ("e_tpgid", ctypes.c_uint32), ("nice", ctypes.c_int32),
                    ("start_sec", ctypes.c_uint64), ("start_usec", ctypes.c_uint64),
                ]

            info = ProcBsdInfo()
            libproc = ctypes.CDLL("/usr/lib/libproc.dylib")
            size = libproc.proc_pidinfo(pid, 3, 0, ctypes.byref(info), ctypes.sizeof(info))
            if size == ctypes.sizeof(info) and info.pid == pid:
                return f"darwin:{info.start_sec}:{info.start_usec}"
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "lstart="],
            text=True,
            capture_output=True,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return None
        return "ps:" + result.stdout.strip()
    return "proc:" + stat_text.rsplit(")", 1)[1].split()[19]
""".strip()


_PROCESS_GROUP_RUNNING_CODE = """
def process_group_running(pgid, proc_root="/proc"):
    leader_uncertain = False
    try:
        leader_stat = open(os.path.join(proc_root, str(pgid), "stat"), encoding="utf-8").read()
    except FileNotFoundError:
        leader_stat = None
    except OSError:
        leader_stat = None
        leader_uncertain = True
    if leader_stat:
        fields = leader_stat.rsplit(")", 1)[-1].split()
        try:
            leader_pgid = int(fields[2])
        except (IndexError, ValueError):
            leader_uncertain = True
        else:
            if leader_pgid == pgid and fields[0] not in {"Z", "X"}:
                return True

    try:
        entries = os.listdir(proc_root)
    except FileNotFoundError:
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True
    except OSError:
        return None

    uncertain = leader_uncertain
    for entry in entries:
        if not entry.isdigit() or entry == str(pgid):
            continue
        try:
            stat_text = open(os.path.join(proc_root, entry, "stat"), encoding="utf-8").read()
        except FileNotFoundError:
            continue
        except OSError:
            uncertain = True
            continue
        fields = stat_text.rsplit(")", 1)[-1].split()
        if len(fields) < 3:
            uncertain = True
            continue
        try:
            process_pgid = int(fields[2])
        except ValueError:
            uncertain = True
            continue
        if process_pgid == pgid and fields[0] not in {"Z", "X"}:
            return True
    return None if uncertain else False
""".strip()


def status_row(
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
            observed_status = "missing_pid" if observed_status in {"planned", "pending"} else "failed"
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
    elif pid is None and running_state is False and managed_process and observed_status in {"launched", "running"}:
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
            log_failed = log_has_failure(row.get("log_path"), row)
            if log_failed is None and is_remote_row(row):
                observed_status = "unknown_remote"
            else:
                observed_status = "failed" if log_failed else "finished"
    # Remote artifacts must be observed on the execution host; transport uncertainty preserves prior evidence.
    manifest = str(previous.get("run_manifest") or row.get("run_manifest") or "")
    checkpoints = [name for name in str(previous.get("checkpoints") or row.get("checkpoints") or "").split(";") if name]
    observed_artifacts = runtime_artifacts(row)
    if observed_artifacts is not None:
        manifest, _manifest_data, checkpoints = observed_artifacts
    observation = {
        **row,
        **(committed_process_identity or {}),
        "status": observed_status,
        "pid": (committed_process_identity or {}).get("pid") or previous.get("pid") or row.get("pid") or "",
        "log_tail": log_tail(row.get("log_path"), row),
        "run_manifest": str(manifest or ""),
        "checkpoints": ";".join(checkpoints),
        "monitored_at": utc_now(),
    }
    if process_identity_error:
        observation["process_identity_error"] = process_identity_error
    output = merge_run_row(previous, observation)
    if health:
        output.update(health_fields(run_dir, row, previous, pid, running_state, output["status"], checkpoints))
    return output


def runtime_artifacts(row: dict[str, Any]) -> tuple[str, dict[str, Any], list[str]] | None:
    if is_remote_row(row):
        script = """
import json
import os
import stat
import sys

runtime_dir = sys.argv[1]
checkpoint_dir = sys.argv[2]
payload = {"run_manifest": "", "manifest": {}, "checkpoints": []}

if runtime_dir:
    try:
        runtime_info = os.lstat(runtime_dir)
    except FileNotFoundError:
        runtime_info = None
    except OSError as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(1)
    if runtime_info is not None and (stat.S_ISLNK(runtime_info.st_mode) or not stat.S_ISDIR(runtime_info.st_mode)):
        print(f"Remote runtime path is not a directory: {runtime_dir}", file=sys.stderr)
        raise SystemExit(1)
    manifest = os.path.join(runtime_dir, "run_manifest.json")
    try:
        manifest_info = os.lstat(manifest)
    except FileNotFoundError:
        pass
    except OSError as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(1)
    else:
        if (
            stat.S_ISLNK(manifest_info.st_mode)
            or not stat.S_ISREG(manifest_info.st_mode)
            or manifest_info.st_nlink != 1
        ):
            print(f"Remote run manifest is not an independent regular file: {manifest}", file=sys.stderr)
            raise SystemExit(1)
        try:
            with open(manifest, encoding="utf-8") as file_obj:
                manifest_payload = json.load(file_obj)
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            print(exc, file=sys.stderr)
            raise SystemExit(1)
        if not isinstance(manifest_payload, dict):
            print(f"Remote run manifest is corrupt: {manifest}", file=sys.stderr)
            raise SystemExit(1)
        payload["run_manifest"] = manifest
        payload["manifest"] = manifest_payload

if checkpoint_dir:
    try:
        checkpoint_info = os.lstat(checkpoint_dir)
    except FileNotFoundError:
        pass
    except OSError as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(1)
    else:
        if stat.S_ISLNK(checkpoint_info.st_mode) or not stat.S_ISDIR(checkpoint_info.st_mode):
            print(f"Remote checkpoint path is not a directory: {checkpoint_dir}", file=sys.stderr)
            raise SystemExit(1)
        try:
            payload["checkpoints"] = sorted(
                name
                for name in os.listdir(checkpoint_dir)
                if name.endswith(".ckpt")
                and stat.S_ISREG(os.lstat(os.path.join(checkpoint_dir, name)).st_mode)
            )
        except OSError as exc:
            print(exc, file=sys.stderr)
            raise SystemExit(1)

sys.stdout.write(json.dumps(payload))
"""
        result = run_row_command(
            row,
            transport.remote_python_command(
                script,
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
) -> dict[str, Any] | None:
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
        script = f"""
import os
import stat
import sys

path = sys.argv[1]
try:
    info = os.lstat(path)
except FileNotFoundError:
    raise SystemExit({REMOTE_MISSING_RETURN_CODE})
except OSError as exc:
    print(exc, file=sys.stderr)
    raise SystemExit(1)

if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
    print(f"PID file is not an independent regular file: {{path}}", file=sys.stderr)
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
            transport.remote_python_command(script, str(path)),
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


def _parse_process_identity(text: str, path: Any) -> dict[str, Any]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ProcessIdentityError(f"PID file is empty or invalid: {path}") from exc
    if not isinstance(payload, dict) or set(payload) != PROCESS_IDENTITY_FIELDS:
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
    return {"pid": pid, "process_group_id": pgid, "process_start_token": token}


_PROCESS_PROBE_SCRIPT = "\n\n".join(
    [
        """
import json
import os
import subprocess
import sys
""".strip(),
        _PROCESS_START_TOKEN_CODE,
        _PROCESS_GROUP_RUNNING_CODE,
        """

pid = int(sys.argv[1])
pgid = int(sys.argv[2])

try:
    leader_pgid = os.getpgid(pid)
except ProcessLookupError:
    leader = None
else:
    token = start_token(pid)
    if token is None:
        print(f"Cannot read process start time for PID {pid}", file=sys.stderr)
        raise SystemExit(1)
    leader = {"pid": pid, "process_group_id": leader_pgid, "process_start_token": token}

group_running = process_group_running(pgid)
if group_running is None:
    print(f"Cannot inspect process group {pgid}", file=sys.stderr)
    raise SystemExit(1)

print(json.dumps({"leader": leader, "group_running": group_running}, sort_keys=True))
""".strip(),
    ]
)


def process_identity_running(row: dict[str, Any], identity: dict[str, Any]) -> bool | None:
    _require_matching_process_identity(row, identity)
    result = run_row_command(
        row,
        transport.remote_python_command(
            _PROCESS_PROBE_SCRIPT,
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


def _require_matching_process_identity(row: dict[str, Any], identity: dict[str, Any]) -> None:
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


_PROCESS_STOP_SCRIPT = "\n\n".join(
    [
        """
import os
import signal
import subprocess
import sys
import time
""".strip(),
        _PROCESS_START_TOKEN_CODE,
        _PROCESS_GROUP_RUNNING_CODE,
        """

pid = int(sys.argv[1])
pgid = int(sys.argv[2])
expected_token = sys.argv[3]
timeout = float(sys.argv[4])

if pgid != pid or pgid == os.getpgrp():
    print("Refusing to signal an unsafe process group", file=sys.stderr)
    raise SystemExit(45)
try:
    leader_pgid = os.getpgid(pid)
except ProcessLookupError:
    leader_pgid = None
if leader_pgid is not None:
    if leader_pgid != pgid or start_token(pid) != expected_token:
        print("Process identity changed before stop", file=sys.stderr)
        raise SystemExit(45)

group_running = process_group_running(pgid)
if group_running is None:
    print("Cannot inspect managed process group", file=sys.stderr)
    raise SystemExit(1)
if not group_running:
    print("Managed process group is no longer running", file=sys.stderr)
    raise SystemExit(44)

os.killpg(pgid, signal.SIGTERM)
deadline = time.monotonic() + timeout
while time.monotonic() < deadline:
    group_running = process_group_running(pgid)
    if group_running is None:
        print("Cannot inspect managed process group", file=sys.stderr)
        raise SystemExit(1)
    if not group_running:
        break
    time.sleep(0.05)
group_running = process_group_running(pgid)
if group_running is None:
    print("Cannot inspect managed process group", file=sys.stderr)
    raise SystemExit(1)
if group_running:
    print("Managed process group did not stop after SIGTERM", file=sys.stderr)
    raise SystemExit(46)
""".strip(),
    ]
)


def stop_process_group(row: dict[str, Any], identity: dict[str, Any], *, timeout: float = 5.0) -> None:
    _require_matching_process_identity(row, identity)
    pid = identity["pid"]
    pgid = identity["process_group_id"]
    if is_remote_row(row):
        # Verification and signal share one remote process so PID reuse cannot race a second SSH call.
        result = run_row_command(
            row,
            transport.remote_python_command(
                _PROCESS_STOP_SCRIPT,
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


def log_has_failure(path: Any, row: dict[str, Any] | None = None) -> bool | None:
    if not path:
        return False
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
    with open(path, encoding="utf-8", errors="replace") as file_obj:
        lines = file_obj.readlines()
except OSError as exc:
    print(exc, file=sys.stderr)
    raise SystemExit(1)

sys.stdout.write("".join(lines[-100:]))
"""
        result = run_row_command(
            row or {},
            transport.remote_python_command(script, str(path)),
        )
        if result.returncode == REMOTE_MISSING_RETURN_CODE:
            return False
        if result.returncode != 0:
            return None
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
        result = run_row_command(row or {}, f"tail -n {int(lines)} {transport.sh(path)}")
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
    host = str(row["host"]) if is_remote_row(row) else None
    return transport.run_shell(host, command, timeout=SSH_TIMEOUT_SECONDS, swallow_timeout=True)


def is_remote_row(row: dict[str, Any] | None) -> bool:
    return bool(row and row.get("target") == "ssh" and row.get("host"))
