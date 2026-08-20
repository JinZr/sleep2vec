from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import signal
import socket
import subprocess
import tempfile
import traceback
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


RESOURCE_FIELDS = {"type", "partition", "cpus_per_task", "memory", "walltime", "nice", "nodelist"}


def normalize_resources(scheduler: dict[str, Any], gpus_per_run: Any) -> dict[str, Any]:
    unknown = sorted(set(scheduler) - RESOURCE_FIELDS)
    if unknown:
        raise ValueError(f"Unknown execution.scheduler field: {', '.join(unknown)}")
    if scheduler.get("type") != "slurm":
        raise ValueError("execution.scheduler.type must be slurm.")
    required = [field for field in ("partition", "cpus_per_task", "memory", "walltime") if not scheduler.get(field)]
    if required:
        raise ValueError(f"Slurm scheduler resources are missing: {', '.join(required)}")
    partition = str(scheduler["partition"])
    if re.fullmatch(r"[A-Za-z0-9_.-]+", partition) is None:
        raise ValueError("execution.scheduler.partition must be a Slurm partition name.")
    cpus_per_task = _positive_int(scheduler["cpus_per_task"], "execution.scheduler.cpus_per_task")
    gpus = _positive_int(gpus_per_run, "execution.gpus_per_run")
    memory = str(scheduler["memory"])
    if re.fullmatch(r"[1-9][0-9]*[KMGTP]", memory) is None:
        raise ValueError("execution.scheduler.memory must use a positive Slurm size such as 64G.")
    walltime = str(scheduler["walltime"])
    if re.fullmatch(r"(?:[0-9]+-)?[0-9]{1,3}:[0-5][0-9]:[0-5][0-9]", walltime) is None:
        raise ValueError("execution.scheduler.walltime must use HH:MM:SS or D-HH:MM:SS.")
    nice = scheduler.get("nice", 0)
    if type(nice) is not int or not 0 <= nice <= 10000:
        raise ValueError("execution.scheduler.nice must be an integer from 0 to 10000.")
    nodelist = str(scheduler.get("nodelist") or "")
    if nodelist and re.fullmatch(r"[A-Za-z0-9_.\-,\[\]]+", nodelist) is None:
        raise ValueError("execution.scheduler.nodelist must be a Slurm node-list expression.")
    return {
        "partition": partition,
        "cpus_per_task": cpus_per_task,
        "memory": memory,
        "walltime": walltime,
        "nice": nice,
        "nodelist": nodelist,
        "gpus_per_run": gpus,
    }


def submit_token(run: dict[str, Any], resources: dict[str, Any], runtime_commit: str) -> str:
    payload = {
        "experiment_id": run["experiment_id"],
        "step_id": run["step_id"],
        "run_id": run["run_id"],
        "run_dir": run["run_dir"],
        "config_sha256": run["config_sha256"],
        "script_sha256": run["script_sha256"],
        "resources": resources,
        "runtime_commit": runtime_commit,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return f"agent-tools-{digest[:32]}"


def render_batch_script(
    *,
    run: dict[str, Any],
    execution: dict[str, Any],
    resources: dict[str, Any],
    token: str,
    result_path: str | Path,
    allocation_identity_path: str | Path,
    log_path: str | Path,
    module: str,
) -> str:
    workdir = str(execution["workdir"])
    job_name = _job_name(str(run["experiment_id"]), str(run["run_id"]))
    directives = [
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --partition={resources['partition']}",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        f"#SBATCH --cpus-per-task={resources['cpus_per_task']}",
        f"#SBATCH --mem={resources['memory']}",
        f"#SBATCH --time={resources['walltime']}",
        f"#SBATCH --nice={resources['nice']}",
        f"#SBATCH --gres=gpu:{resources['gpus_per_run']}",
        "#SBATCH --no-requeue",
        f"#SBATCH --comment={token}",
        f"#SBATCH --output={log_path}",
        f"#SBATCH --error={log_path}",
    ]
    if resources.get("nodelist"):
        directives.append(f"#SBATCH --nodelist={resources['nodelist']}")
    env_lines = [f"export {key}={transport.sh(value)}" for key, value in sorted((execution.get("env") or {}).items())]
    worker = [
        execution["python"],
        "-m",
        "agent_tools.slurm",
        "run-frozen-job",
        "--run-id",
        run["run_id"],
        "--command",
        run["command"],
        "--script",
        run["script"],
        "--script-sha256",
        run["script_sha256"],
        "--config",
        run["config"],
        "--config-sha256",
        run["config_sha256"],
        "--result-path",
        result_path,
        "--allocation-identity-path",
        allocation_identity_path,
        "--log-path",
        log_path,
        "--submit-token",
        token,
        "--workdir",
        workdir,
        "--python",
        execution["python"],
        "--runtime-commit",
        execution["runtime_commit"],
        "--module",
        module,
    ]
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            *directives,
            "",
            "set -euo pipefail",
            f"cd {transport.sh(workdir)}",
            f"export PYTHONPATH={transport.sh(workdir)}${{PYTHONPATH:+:$PYTHONPATH}}",
            *env_lines,
            "",
            f"exec {' '.join(transport.sh(part) for part in worker)}",
            "",
        ]
    )


def run_frozen_job(
    *,
    run_id: str,
    command: str,
    script: str,
    script_sha256: str,
    config: str,
    config_sha256: str,
    result_path: str,
    allocation_identity_path: str,
    log_path: str,
    submit_token: str,
    workdir: str,
    python: str,
    runtime_commit: str,
    module: str,
) -> int:
    started_at = _utc_now()
    job_id = _job_id(os.environ.get("SLURM_JOB_ID", ""))
    cluster = str(os.environ.get("SLURM_CLUSTER_NAME") or "")
    node = socket.gethostname()
    child: subprocess.Popen | None = None
    received_signal = 0

    def forward_signal(signum, _frame):
        nonlocal received_signal
        received_signal = signum
        if child is not None and child.poll() is None:
            child.send_signal(signum)

    old_handlers = {signum: signal.signal(signum, forward_signal) for signum in (signal.SIGTERM, signal.SIGINT)}
    exit_code = 2
    log_target = Path(log_path)
    log_target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with log_target.open("a") as log:
            try:
                from .managed_scheduler import inspect_execution_target

                for path, expected in ((script, script_sha256), (config, config_sha256)):
                    artifact = Path(path)
                    if artifact.is_symlink() or not artifact.is_file() or artifact.stat().st_nlink != 1:
                        raise ValueError(f"Frozen run artifact is not an independent file: {artifact}")
                    if hashlib.sha256(artifact.read_bytes()).hexdigest() != expected:
                        raise ValueError(f"Frozen run artifact changed before allocation start: {artifact}")
                snapshot = inspect_execution_target(
                    {
                        "target": "local",
                        "workdir": workdir,
                        "python": python,
                        "runtime_commit": runtime_commit,
                    },
                    [{"run_id": run_id, "command": command, "script": script}],
                    plan_label="Slurm",
                )
                if snapshot["module"] != module:
                    raise ValueError("Frozen Slurm module differs from the verified runtime module.")
                _atomic_create_json(
                    allocation_identity_path,
                    {
                        "schema_version": 1,
                        "scheduler_job_id": job_id,
                        "scheduler_cluster": cluster,
                        "scheduler_submit_token": submit_token,
                        "node": node,
                        "started_at": started_at,
                        "execution_snapshot": snapshot,
                    },
                )
                child = subprocess.Popen(
                    [script],
                    cwd=workdir,
                    env=os.environ.copy(),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=False,
                )
                exit_code = child.wait()
                if exit_code < 0:
                    exit_code = 128 + abs(exit_code)
                if received_signal and exit_code == 0:
                    exit_code = 128 + received_signal
            except BaseException:
                traceback.print_exc(file=log)
                if received_signal:
                    exit_code = 128 + received_signal
    finally:
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)
    _atomic_create_json(
        result_path,
        {
            "schema_version": 1,
            "scheduler_job_id": job_id,
            "scheduler_cluster": cluster,
            "scheduler_submit_token": submit_token,
            "node": node,
            "started_at": started_at,
            "ended_at": _utc_now(),
            "exit_code": exit_code,
        },
    )
    return exit_code


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


def submission_command(script: str, submit_token: str) -> str:
    return " ".join(transport.sh(part) for part in ["sbatch", "--parsable", f"--comment={submit_token}", script])


def sidecar_identity(
    payload: dict[str, Any],
    submit_token: str,
    *,
    expected_job_id: str | None = None,
) -> JobIdentity:
    if payload.get("schema_version") != 1:
        raise ValueError("Slurm sidecar has an unsupported schema version.")
    if payload.get("scheduler_submit_token") != submit_token:
        raise ValueError("Slurm sidecar submit token differs from the frozen run.")
    job_id = _job_id(str(payload.get("scheduler_job_id") or ""))
    if expected_job_id is not None and job_id != _job_id(expected_job_id):
        raise ValueError("Slurm sidecar job id differs from the canonical run.")
    cluster = str(payload.get("scheduler_cluster") or "")
    if cluster and re.fullmatch(r"[A-Za-z0-9_.-]+", cluster) is None:
        raise ValueError("Slurm sidecar cluster name is invalid.")
    return JobIdentity(job_id, cluster)


def terminal_exit_code(payload: dict[str, Any]) -> int:
    exit_code = payload.get("exit_code")
    if type(exit_code) is not int or exit_code < 0:
        raise ValueError("Slurm terminal sidecar exit_code must be a non-negative integer.")
    return exit_code


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


def _positive_int(value: Any, field: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{field} must be a positive integer.")
    return value


def _job_name(experiment_id: str, run_id: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "-", f"{experiment_id}-{run_id}").strip("-.")
    return (value or "agent-tools-run")[:128]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _atomic_create_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Slurm result artifact already exists: {target}")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(descriptor, "w") as file_obj:
            json.dump(payload, file_obj, indent=2, sort_keys=True)
            file_obj.write("\n")
            file_obj.flush()
            os.fsync(file_obj.fileno())
        os.replace(temporary, target)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command_name", required=True)
    worker = subparsers.add_parser("run-frozen-job")
    for name in (
        "run_id",
        "command",
        "script",
        "script_sha256",
        "config",
        "config_sha256",
        "result_path",
        "allocation_identity_path",
        "log_path",
        "submit_token",
        "workdir",
        "python",
        "runtime_commit",
        "module",
    ):
        worker.add_argument(f"--{name.replace('_', '-')}", required=True)
    args = parser.parse_args(argv)
    payload = vars(args)
    payload.pop("command_name")
    return run_frozen_job(**payload)


if __name__ == "__main__":
    raise SystemExit(_main())
