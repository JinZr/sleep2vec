from __future__ import annotations

from dataclasses import dataclass
import re
import shlex
import subprocess
from typing import Any

from . import transport


@dataclass(frozen=True)
class JobIdentity:
    job_id: str
    cluster: str = ""


@dataclass(frozen=True)
class JobObservation:
    job_id: str
    state: str
    reason: str = ""
    node_list: str = ""
    comment: str = ""
    exit_code: str = ""


class SlurmCommandError(RuntimeError):
    def __init__(self, action: str, result: subprocess.CompletedProcess):
        self.returncode = result.returncode
        self.stdout = str(result.stdout or "")
        self.stderr = str(result.stderr or "")
        detail = self.stderr.strip() or self.stdout.strip() or f"exit code {self.returncode}"
        super().__init__(f"Slurm {action} failed: {detail}")


def run_command(
    execution: dict[str, Any],
    argv: list[str],
    *,
    timeout: float = transport.SSH_TIMEOUT_SECONDS,
) -> subprocess.CompletedProcess:
    target = str(execution.get("target", "local") or "local")
    if target not in {"local", "ssh"}:
        raise ValueError("execution.target must be local or ssh.")
    host = None
    if target == "ssh":
        host = str(execution.get("host") or "").strip()
        if not host:
            raise ValueError("execution.host is required for an SSH Slurm submission host.")
    command = " ".join(transport.sh(part) for part in argv)
    return transport.run_shell(host, command, timeout=timeout)


def parse_sbatch_output(stdout: str) -> JobIdentity:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise ValueError("sbatch --parsable must return exactly one non-empty line.")
    match = re.fullmatch(r"([1-9][0-9]*)(?:;([A-Za-z0-9_.-]+))?", lines[0])
    if match is None:
        raise ValueError(f"Invalid sbatch --parsable output: {lines[0]!r}")
    return JobIdentity(job_id=match.group(1), cluster=match.group(2) or "")


def submit(
    execution: dict[str, Any],
    script: str,
    submit_token: str,
    *,
    timeout: float = transport.SSH_TIMEOUT_SECONDS,
) -> JobIdentity:
    result = run_command(
        execution,
        ["sbatch", "--parsable", f"--comment={submit_token}", script],
        timeout=timeout,
    )
    if result.returncode != 0:
        raise SlurmCommandError("submission", result)
    return parse_sbatch_output(str(result.stdout or ""))


def active_jobs(
    execution: dict[str, Any],
    *,
    job_id: str | None = None,
    submit_token: str | None = None,
    timeout: float = transport.SSH_TIMEOUT_SECONDS,
) -> list[JobObservation]:
    argv = ["squeue", "--noheader", "--format=%i|%T|%R|%N|%k"]
    if job_id is not None:
        argv.extend(["--jobs", _job_id(job_id)])
    result = run_command(execution, argv, timeout=timeout)
    if result.returncode != 0:
        raise SlurmCommandError("active-job query", result)
    observations = [_parse_squeue_line(line) for line in str(result.stdout or "").splitlines() if line.strip()]
    if submit_token is not None:
        observations = [item for item in observations if item.comment == submit_token]
    if job_id is not None:
        observations = [item for item in observations if item.job_id == str(job_id)]
    return observations


def show_job(
    execution: dict[str, Any],
    job_id: str,
    *,
    timeout: float = transport.SSH_TIMEOUT_SECONDS,
) -> JobObservation | None:
    job_id = _job_id(job_id)
    result = run_command(execution, ["scontrol", "show", "job", "--oneliner", job_id], timeout=timeout)
    if result.returncode != 0:
        detail = f"{result.stdout or ''}\n{result.stderr or ''}".lower()
        if "invalid job id" in detail:
            return None
        raise SlurmCommandError("job query", result)
    fields = _parse_scontrol_fields(str(result.stdout or ""))
    observed_id = fields.get("JobId", "")
    if observed_id != job_id:
        raise ValueError(f"scontrol returned job {observed_id!r} while querying {job_id!r}.")
    return JobObservation(
        job_id=job_id,
        state=normalize_state(fields.get("JobState", "")),
        reason=_null_to_empty(fields.get("Reason", "")),
        node_list=_null_to_empty(fields.get("NodeList", "")),
        comment=_null_to_empty(fields.get("Comment", "")),
        exit_code=_null_to_empty(fields.get("ExitCode", "")),
    )


def cancel(
    execution: dict[str, Any],
    job_id: str,
    *,
    timeout: float = transport.SSH_TIMEOUT_SECONDS,
) -> None:
    result = run_command(execution, ["scancel", _job_id(job_id)], timeout=timeout)
    if result.returncode != 0:
        raise SlurmCommandError("cancellation", result)


def normalize_state(value: str) -> str:
    state = str(value or "").strip().upper()
    while state.endswith("+"):
        state = state[:-1]
    return state


def state_category(value: str) -> str:
    state = normalize_state(value)
    if state in {"PENDING", "CONFIGURING"}:
        return "queued"
    if state in {"RUNNING", "COMPLETING"}:
        return "running"
    if state == "COMPLETED":
        return "completed"
    if state == "CANCELLED":
        return "cancelled"
    if state in {
        "FAILED",
        "TIMEOUT",
        "OUT_OF_MEMORY",
        "NODE_FAIL",
        "BOOT_FAIL",
        "DEADLINE",
        "PREEMPTED",
    }:
        return "failed"
    return "unknown"


def parse_exit_code(value: str) -> tuple[int, int]:
    match = re.fullmatch(r"([0-9]+):([0-9]+)", str(value or "").strip())
    if match is None:
        raise ValueError(f"Invalid Slurm exit code: {value!r}")
    return int(match.group(1)), int(match.group(2))


def _parse_squeue_line(line: str) -> JobObservation:
    parts = line.strip().split("|", 4)
    if len(parts) != 5:
        raise ValueError(f"Invalid squeue output row: {line!r}")
    return JobObservation(
        job_id=_job_id(parts[0]),
        state=normalize_state(parts[1]),
        reason=_null_to_empty(parts[2]),
        node_list=_null_to_empty(parts[3]),
        comment=_null_to_empty(parts[4]),
    )


def _parse_scontrol_fields(output: str) -> dict[str, str]:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if len(lines) != 1:
        raise ValueError("scontrol show job --oneliner must return exactly one row.")
    fields: dict[str, str] = {}
    for token in shlex.split(lines[0]):
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        fields[key] = value
    return fields


def _job_id(value: str) -> str:
    text = str(value or "").strip()
    if re.fullmatch(r"[1-9][0-9]*", text) is None:
        raise ValueError(f"Slurm job id must be a positive integer: {value!r}")
    return text


def _null_to_empty(value: str) -> str:
    return "" if value in {"", "(null)", "None"} else value
