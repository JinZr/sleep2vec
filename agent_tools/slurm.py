from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
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
from typing import Any, Literal, TypedDict

from . import manifests, python_programs, transport
from .runtime_lock import runtime_lock


class SlurmResources(TypedDict):
    partition: str
    cpus_per_task: int
    memory: str
    walltime: str
    nice: int
    nodelist: str
    direct_controller: bool
    gpus_per_run: int


class SlurmSchedulingCapabilities(TypedDict):
    slurm_version: str
    priority_type: str
    scheduler_type: str
    accounting_storage_type: str
    preempt_type: str
    multifactor_priority: bool
    backfill_enabled: bool
    accounting_enabled: bool
    preemption_enabled: bool
    partition: str
    partition_state: str
    partition_max_time: str
    reservation_count: int


class PerRunResources(TypedDict):
    gpus: int
    cpus: int
    memory: str
    memory_kib: int


class NodeCapacity(TypedDict):
    gpus: int
    cpus: int
    memory_mib: int
    memory_kib: int


class KnownNodeResourceCapacity(TypedDict):
    status: Literal["known"]
    node: str
    planned_runs: int
    per_run: PerRunResources
    node_capacity: NodeCapacity
    limits: dict[str, int]
    overall_empty_node_limit: int
    limiting_resources: list[str]
    minimum_waves: int | None


class UnknownNodeResourceCapacity(TypedDict):
    status: Literal["unknown"]
    reason: str
    planned_runs: int


NodeResourceCapacity = KnownNodeResourceCapacity | UnknownNodeResourceCapacity


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
    details: dict[str, str] = field(default_factory=dict)


class SlurmCommandError(RuntimeError):
    def __init__(self, action: str, result: subprocess.CompletedProcess):
        self.returncode = result.returncode
        self.stdout = str(result.stdout or "")
        self.stderr = str(result.stderr or "")
        detail = self.stderr.strip() or self.stdout.strip() or f"exit code {self.returncode}"
        super().__init__(f"Slurm {action} failed: {detail}")


RESOURCE_FIELDS = {
    "type",
    "partition",
    "cpus_per_task",
    "memory",
    "walltime",
    "nice",
    "nodelist",
    "direct_controller",
}
_DISTRIBUTED_ENV_FIELDS = {"RANK", "LOCAL_RANK", "WORLD_SIZE"}


def normalize_resources(scheduler: dict[str, Any], gpus_per_run: Any) -> SlurmResources:
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
    direct_controller = scheduler.get("direct_controller", False)
    if type(direct_controller) is not bool:
        raise ValueError("execution.scheduler.direct_controller must be a boolean.")
    return {
        "partition": partition,
        "cpus_per_task": cpus_per_task,
        "memory": memory,
        "walltime": walltime,
        "nice": nice,
        "nodelist": nodelist,
        "direct_controller": direct_controller,
        "gpus_per_run": gpus,
    }


def fixed_node_resource_capacity(
    execution: dict[str, Any],
    resources: SlurmResources,
    planned_runs: int,
    *,
    timeout: float = transport.SSH_TIMEOUT_SECONDS,
) -> NodeResourceCapacity:
    node = str(resources.get("nodelist") or "")
    if re.fullmatch(r"[A-Za-z0-9_.-]+", node) is None:
        return {
            "status": "unknown",
            "reason": "fixed-node capacity requires one literal execution.scheduler.nodelist",
            "planned_runs": planned_runs,
        }

    result = run_command(execution, ["scontrol", "show", "node", node, "-o"], timeout=timeout)
    if result.returncode != 0:
        raise SlurmCommandError("fixed-node resource capacity query", result)
    lines = [line.strip() for line in str(result.stdout or "").splitlines() if line.strip()]
    if len(lines) != 1:
        raise ValueError("scontrol show node must return exactly one row for a literal nodelist.")
    fields = _parse_scontrol_oneline(lines[0])
    if fields.get("NodeName") != node:
        raise ValueError(f"scontrol returned node {fields.get('NodeName')!r} while querying {node!r}.")
    try:
        node_cpus = int(fields["CPUTot"])
        node_memory_mib = int(fields["RealMemory"])
    except (KeyError, ValueError) as exc:
        raise ValueError("scontrol node output lacks valid CPUTot or RealMemory capacity.") from exc
    if node_cpus <= 0 or node_memory_mib <= 0:
        raise ValueError("scontrol node output lacks positive CPUTot or RealMemory capacity.")

    gres = fields.get("Gres")
    if not gres:
        raise ValueError("scontrol node output lacks configured GPU capacity in Gres.")
    node_gpus = 0
    for token in gres.split(","):
        configured = token.strip().split("(", 1)[0]
        if not configured.startswith("gpu:"):
            continue
        match = re.fullmatch(r"gpu:(?:[^:,()]+:)?([1-9][0-9]*)", configured)
        if match is None:
            raise ValueError(f"scontrol node output has malformed GPU capacity in Gres: {token.strip()!r}.")
        node_gpus += int(match.group(1))
    if not node_gpus:
        raise ValueError("scontrol node output lacks configured GPU capacity in Gres.")

    memory = str(resources["memory"])
    memory_match = re.fullmatch(r"([1-9][0-9]*)([KMGTP])", memory)
    if memory_match is None:
        raise ValueError("Frozen Slurm memory does not use a supported size unit.")
    memory_kib = (
        int(memory_match.group(1))
        * {
            "K": 1,
            "M": 1024,
            "G": 1024**2,
            "T": 1024**3,
            "P": 1024**4,
        }[memory_match.group(2)]
    )
    per_run_gpus = int(resources["gpus_per_run"])
    per_run_cpus = per_run_gpus * int(resources["cpus_per_task"])
    node_memory_kib = node_memory_mib * 1024
    limits = {
        "cpu": node_cpus // per_run_cpus,
        "memory": node_memory_kib // memory_kib,
        "gpu": node_gpus // per_run_gpus,
    }
    overall_limit = min(limits.values())
    return {
        "status": "known",
        "node": node,
        "planned_runs": planned_runs,
        "per_run": {
            "gpus": per_run_gpus,
            "cpus": per_run_cpus,
            "memory": memory,
            "memory_kib": memory_kib,
        },
        "node_capacity": {
            "gpus": node_gpus,
            "cpus": node_cpus,
            "memory_mib": node_memory_mib,
            "memory_kib": node_memory_kib,
        },
        "limits": limits,
        "overall_empty_node_limit": overall_limit,
        "limiting_resources": [name for name, limit in limits.items() if limit == overall_limit],
        "minimum_waves": (planned_runs + overall_limit - 1) // overall_limit if overall_limit else None,
    }


def submit_token(run: dict[str, Any], resources: SlurmResources, runtime_commit: str) -> str:
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
    resources: SlurmResources,
    token: str,
    result_path: str | Path,
    allocation_identity_path: str | Path,
    execution_snapshot_path: str | Path,
    log_path: str | Path,
    module: str,
) -> str:
    workdir = str(execution["workdir"])
    job_name = _job_name(str(run["experiment_id"]), str(run["run_id"]))
    directive_log_path = transport.sh(str(log_path).replace("%", "%%"))
    directives = [
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --partition={resources['partition']}",
        "#SBATCH --nodes=1",
        f"#SBATCH --ntasks={resources['gpus_per_run']}",
        f"#SBATCH --ntasks-per-node={resources['gpus_per_run']}",
        f"#SBATCH --cpus-per-task={resources['cpus_per_task']}",
        f"#SBATCH --mem={resources['memory']}",
        f"#SBATCH --time={resources['walltime']}",
        f"#SBATCH --nice={resources['nice']}",
        f"#SBATCH --gres=gpu:{resources['gpus_per_run']}",
        "#SBATCH --no-requeue",
        f"#SBATCH --comment={token}",
        f"#SBATCH --output={directive_log_path}",
        f"#SBATCH --error={directive_log_path}",
    ]
    if resources.get("nodelist"):
        directives.append(f"#SBATCH --nodelist={resources['nodelist']}")
    env_lines = [f"export {key}={transport.sh(value)}" for key, value in sorted((execution.get("env") or {}).items())]
    worker = [
        execution["python"],
        "-c",
        python_programs.source("slurm.worker_bootstrap"),
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
        "--execution-snapshot-path",
        execution_snapshot_path,
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
        "--gpus-per-run",
        resources["gpus_per_run"],
    ]
    worker_command = " ".join(transport.sh(part) for part in worker)
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            *directives,
            "",
            "set -euo pipefail",
            f"cd {transport.sh(workdir)}",
            f"export PYTHONPATH={transport.sh(workdir)}",
            *env_lines,
            "",
            f'exec {worker_command} --execution-snapshot-sha256 "${{1:-}}"',
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
    execution_snapshot_path: str,
    execution_snapshot_sha256: str,
    log_path: str,
    submit_token: str,
    workdir: str,
    python: str,
    runtime_commit: str,
    module: str,
    gpus_per_run: int,
) -> int:
    started_at = _utc_now()
    job_id = _job_id(os.environ.get("SLURM_JOB_ID", ""))
    expected_tasks = gpus_per_run
    cluster = str(os.environ.get("SLURM_CLUSTER_NAME") or "")
    node = socket.gethostname()
    child: subprocess.Popen | None = None
    received_signal = 0
    observed_runtime_commit = ""

    def write_terminal(exit_code: int) -> None:
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
                "runtime_commit": observed_runtime_commit,
            },
        )

    def verify_frozen_artifacts(when: str) -> None:
        for path, expected in ((script, script_sha256), (config, config_sha256)):
            artifact = Path(path)
            if artifact.is_symlink() or not artifact.is_file() or artifact.stat().st_nlink != 1:
                raise ValueError(f"Frozen run artifact is not an independent file: {artifact}")
            if hashlib.sha256(artifact.read_bytes()).hexdigest() != expected:
                raise ValueError(f"Frozen run artifact changed {when}: {artifact}")

    def forward_signal(signum, _frame):
        nonlocal received_signal
        received_signal = signum
        if child is not None and child.poll() is None:
            child.send_signal(signum)

    forwarded_signals = (signal.SIGTERM, signal.SIGINT)
    old_handlers = {signum: signal.signal(signum, forward_signal) for signum in forwarded_signals}
    exit_code = 2
    log_target = Path(log_path)
    log_target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with log_target.open("a") as log:
            try:
                from .managed_scheduler import inspect_execution_target

                expected_tasks = _positive_int(expected_tasks, "gpus_per_run")
                if os.environ.get("SLURM_NTASKS") != str(expected_tasks):
                    raise ValueError(
                        f"Slurm allocation task count must equal gpus_per_run={expected_tasks}; "
                        f"observed SLURM_NTASKS={os.environ.get('SLURM_NTASKS')!r}."
                    )

                verify_frozen_artifacts("before allocation start")
                with runtime_lock(workdir):
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
                    observed_runtime_commit = str(snapshot["runtime_commit"])
                    if snapshot["module"] != module:
                        raise ValueError("Frozen Slurm module differs from the verified runtime module.")
                    expected_snapshot_sha256 = _sha256(execution_snapshot_sha256)
                    snapshot_artifact = Path(execution_snapshot_path)
                    if (
                        snapshot_artifact.is_symlink()
                        or not snapshot_artifact.is_file()
                        or snapshot_artifact.stat().st_nlink != 1
                    ):
                        raise ValueError(f"Frozen execution snapshot is not an independent file: {snapshot_artifact}")
                    snapshot_bytes = snapshot_artifact.read_bytes()
                    if hashlib.sha256(snapshot_bytes).hexdigest() != expected_snapshot_sha256:
                        raise ValueError("Frozen Slurm execution snapshot changed before allocation start.")
                    frozen_snapshot = json.loads(snapshot_bytes)
                    identity_fields = ("python", "python_version")
                    changed = [
                        field
                        for field in identity_fields
                        if not isinstance(frozen_snapshot, dict) or snapshot.get(field) != frozen_snapshot.get(field)
                    ]
                    if changed:
                        raise ValueError("Frozen Slurm execution identity changed in allocation: " + ", ".join(changed))
                    if received_signal:
                        exit_code = 128 + received_signal
                    else:
                        verify_frozen_artifacts("before process start")
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
                        child_env = os.environ.copy()
                        for env_name in tuple(child_env):
                            if env_name in _DISTRIBUTED_ENV_FIELDS or env_name.startswith("MASTER_"):
                                child_env.pop(env_name)
                        child = subprocess.Popen(
                            [
                                "srun",
                                "--nodes=1",
                                f"--ntasks={expected_tasks}",
                                f"--ntasks-per-node={expected_tasks}",
                                "--kill-on-bad-exit=1",
                                "--quit-on-interrupt",
                                "--label",
                                script,
                            ],
                            cwd=workdir,
                            env=child_env,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            start_new_session=False,
                        )
                        if received_signal and child.poll() is None:
                            child.send_signal(received_signal)
                if child is not None:
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
    write_terminal(exit_code)
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
    return transport.run_shell(host, _shell_command(argv), timeout=timeout)


def controller_cluster(
    execution: dict[str, Any],
    *,
    timeout: float = transport.SSH_TIMEOUT_SECONDS,
) -> str:
    result = run_command(execution, ["scontrol", "show", "config"], timeout=timeout)
    if result.returncode != 0:
        raise SlurmCommandError("controller cluster query", result)
    output = str(result.stdout or "")
    cluster_lines = [line for line in output.splitlines() if line.partition("=")[0].strip() == "ClusterName"]
    cluster = _parse_scontrol_config(output).get("ClusterName", "")
    if len(cluster_lines) != 1 or not cluster:
        raise ValueError("scontrol show config must return exactly one non-empty ClusterName.")
    return _cluster_name(cluster)


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
    execution_snapshot_sha256: str,
    timeout: float = transport.SSH_TIMEOUT_SECONDS,
) -> JobIdentity:
    result = run_command(
        execution,
        _submission_argv(script, submit_token, execution_snapshot_sha256),
        timeout=timeout,
    )
    if result.returncode != 0:
        raise SlurmCommandError("submission", result)
    return parse_sbatch_output(str(result.stdout or ""))


def active_jobs(
    execution: dict[str, Any],
    *,
    job_id: str | Sequence[str] | None = None,
    submit_token: str | None = None,
    cluster: str | None = None,
    timeout: float = transport.SSH_TIMEOUT_SECONDS,
) -> list[JobObservation]:
    argv = ["squeue", "--noheader", "--format=%i|%T|%R|%N|%k"]
    cluster_name = _follow_up_cluster_name(execution, cluster)
    if cluster_name:
        argv.append(f"--clusters={cluster_name}")
    batch = job_id is not None and not isinstance(job_id, str)
    if job_id is not None:
        requested_ids = (job_id,) if isinstance(job_id, str) else tuple(job_id)
        if not requested_ids:
            raise ValueError("An active-job query requires at least one Slurm job id.")
        argv.extend(["--jobs", ",".join(_job_id(value) for value in requested_ids)])
    result = run_command(execution, argv, timeout=timeout)
    if result.returncode != 0:
        detail = f"{result.stdout or ''}\n{result.stderr or ''}".lower()
        if job_id is not None and not batch and re.search(r"\b(?:invalid|unknown) job[ _-]?id\b", detail):
            return []
        raise SlurmCommandError("active-job query", result)
    observations = [_parse_squeue_line(line) for line in str(result.stdout or "").splitlines() if line.strip()]
    if submit_token is not None:
        observations = [item for item in observations if item.comment == submit_token]
    if job_id is not None:
        observations = [item for item in observations if item.job_id in requested_ids]
    return observations


def show_job(
    execution: dict[str, Any],
    job_id: str,
    *,
    cluster: str | None = None,
    timeout: float = transport.SSH_TIMEOUT_SECONDS,
) -> JobObservation | None:
    job_id = _job_id(job_id)
    argv = ["scontrol"]
    cluster_name = _follow_up_cluster_name(execution, cluster)
    if cluster_name:
        argv.append(f"--clusters={cluster_name}")
    argv.extend(["show", "job", "--oneliner", job_id])
    result = run_command(execution, argv, timeout=timeout)
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
        details={
            name: _null_to_empty(fields.get(source, ""))
            for name, source in {
                "priority": "Priority",
                "nice": "Nice",
                "partition": "Partition",
                "account": "Account",
                "qos": "QOS",
                "reservation": "Reservation",
                "submit_time": "SubmitTime",
                "eligible_time": "EligibleTime",
                "start_time": "StartTime",
                "time_limit": "TimeLimit",
                "requested_nodes": "ReqNodeList",
                "features": "Features",
                "requested_tres": "ReqTRES",
                "tres_per_node": "TresPerNode",
            }.items()
        },
    )


def accounting_job(
    execution: dict[str, Any],
    job_id: str,
    *,
    submit_token: str,
    cluster: str | None = None,
    timeout: float = transport.SSH_TIMEOUT_SECONDS,
) -> JobObservation | None:
    job_id = _job_id(job_id)
    argv = ["sacct", "--duplicates", "--noheader", "--parsable2", "--allocations"]
    cluster_name = _follow_up_cluster_name(execution, cluster)
    if cluster_name:
        argv.append(f"--clusters={cluster_name}")
    argv.extend(["--jobs", job_id, "--format=JobIDRaw,State%64,ExitCode,NodeList,Comment%64"])
    result = run_command(execution, argv, timeout=timeout)
    if result.returncode != 0:
        raise SlurmCommandError("accounting query", result)
    lines = [line for line in str(result.stdout or "").splitlines() if line.strip()]
    observations = [_parse_sacct_line(line) for line in lines if line.split("|", 1)[0].strip() == job_id]
    if not observations:
        return None
    matches = [item for item in observations if item.comment == submit_token]
    if len(matches) != 1:
        raise ValueError(f"sacct did not return exactly one authenticated allocation row for job {job_id}.")
    return matches[0]


def cluster_scheduling_capabilities(
    execution: dict[str, Any],
    partition: str,
    *,
    timeout: float = transport.SSH_TIMEOUT_SECONDS,
) -> SlurmSchedulingCapabilities:
    outputs: dict[str, str] = {}
    for action, argv in (
        ("version query", ["scontrol", "--version"]),
        ("configuration query", ["scontrol", "show", "config"]),
        ("partition query", ["scontrol", "show", "partition", "-o"]),
        ("reservation query", ["scontrol", "show", "reservation", "-o"]),
    ):
        result = run_command(execution, argv, timeout=timeout)
        if result.returncode != 0:
            raise SlurmCommandError(action, result)
        outputs[action] = str(result.stdout or "")
    return parse_cluster_scheduling_capabilities(
        version_output=outputs["version query"],
        config_output=outputs["configuration query"],
        partition_output=outputs["partition query"],
        reservation_output=outputs["reservation query"],
        partition=partition,
    )


def parse_cluster_scheduling_capabilities(
    *,
    version_output: str,
    config_output: str,
    partition_output: str,
    reservation_output: str,
    partition: str,
) -> SlurmSchedulingCapabilities:
    config = _parse_scontrol_config(config_output)
    partitions = [_parse_scontrol_oneline(line) for line in partition_output.splitlines() if line.strip()]
    selected_partition = next((row for row in partitions if row.get("PartitionName") == partition), {})
    reservation_count = sum(
        "ReservationName" in _parse_scontrol_oneline(line) for line in reservation_output.splitlines()
    )
    priority_type = config.get("PriorityType", "")
    scheduler_type = config.get("SchedulerType", "")
    accounting_storage_type = config.get("AccountingStorageType", "")
    preempt_type = config.get("PreemptType", "")
    return {
        "slurm_version": version_output.strip(),
        "priority_type": priority_type,
        "scheduler_type": scheduler_type,
        "accounting_storage_type": accounting_storage_type,
        "preempt_type": preempt_type,
        "multifactor_priority": priority_type == "priority/multifactor",
        "backfill_enabled": scheduler_type == "sched/backfill",
        "accounting_enabled": accounting_storage_type not in {"", "accounting_storage/none"},
        "preemption_enabled": preempt_type not in {"", "preempt/none"},
        "partition": partition,
        "partition_state": _null_to_empty(selected_partition.get("State", "")),
        "partition_max_time": _null_to_empty(selected_partition.get("MaxTime", "")),
        "reservation_count": reservation_count,
    }


def cancel(
    execution: dict[str, Any],
    job_id: str,
    *,
    cluster: str | None = None,
    timeout: float = transport.SSH_TIMEOUT_SECONDS,
) -> None:
    argv = ["scancel"]
    cluster_name = _follow_up_cluster_name(execution, cluster)
    if cluster_name:
        argv.append(f"--clusters={cluster_name}")
    argv.append(_job_id(job_id))
    result = run_command(execution, argv, timeout=timeout)
    if result.returncode != 0:
        raise SlurmCommandError("cancellation", result)


def _follow_up_cluster_name(execution: dict[str, Any], cluster: str | None) -> str:
    cluster_name = _cluster_name(cluster)
    scheduler = execution["scheduler"] if isinstance(execution.get("scheduler"), dict) else {}
    return "" if scheduler.get("direct_controller") is True else cluster_name


def normalize_state(value: str) -> str:
    state = str(value or "").strip().upper()
    while state.endswith("+"):
        state = state[:-1]
    return state


def state_category(value: str) -> str:
    state = normalize_state(value)
    if state in {
        "PENDING",
        "CONFIGURING",
        "EXPEDITING",
        "POWER_UP_NODE",
        "REQUEUED",
        "REQUEUE_FED",
        "REQUEUE_HOLD",
        "RESV_DEL_HOLD",
        "SPECIAL_EXIT",
    }:
        return "queued"
    if state in {
        "RUNNING",
        "SUSPENDED",
        "COMPLETING",
        "RESIZING",
        "SIGNALING",
        "STAGE_OUT",
        "STOPPED",
        "UPDATE_DB",
    }:
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
        "LAUNCH_FAILED",
        "RECONFIG_FAIL",
    }:
        return "failed"
    return "unknown"


def parse_exit_code(value: str) -> tuple[int, int]:
    match = re.fullmatch(r"([0-9]+):([0-9]+)", str(value or "").strip())
    if match is None:
        raise ValueError(f"Invalid Slurm exit code: {value!r}")
    return int(match.group(1)), int(match.group(2))


def submission_command(script: str, submit_token: str, execution_snapshot_sha256: str | None = None) -> str:
    return _shell_command(_submission_argv(script, submit_token, execution_snapshot_sha256))


def _shell_command(argv: list[str]) -> str:
    return " ".join(transport.sh(part) for part in ["env", "-u", "SLURM_CLUSTERS", *argv])


def _submission_argv(script: str, submit_token: str, execution_snapshot_sha256: str | None = None) -> list[str]:
    argv = ["sbatch", "--parsable", f"--comment={submit_token}", script]
    if execution_snapshot_sha256:
        argv.append(_sha256(execution_snapshot_sha256))
    return [
        "bash",
        "-c",
        'for name in "${!SBATCH_@}"; do unset "$name"; done; exec "$@"',
        "agent-tools-sbatch",
        *argv,
    ]


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
        comment=_null_to_empty(parts[4].strip()),
    )


def _parse_sacct_line(line: str) -> JobObservation:
    parts = line.strip().split("|", 4)
    if len(parts) != 5:
        raise ValueError(f"Invalid sacct output row: {line!r}")
    return JobObservation(
        job_id=_job_id(parts[0]),
        state=normalize_state(parts[1].split(maxsplit=1)[0]),
        node_list=_null_to_empty(parts[3]),
        comment=_null_to_empty(parts[4].strip()),
        exit_code=_null_to_empty(parts[2]),
    )


def _parse_scontrol_fields(output: str) -> dict[str, str]:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if len(lines) != 1:
        raise ValueError("scontrol show job --oneliner must return exactly one row.")
    return _parse_scontrol_oneline(lines[0])


def _parse_scontrol_oneline(line: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for token in shlex.split(line):
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        # Slurm appends unquoted Reason/Comment text, whose later key-like words must not replace real fields.
        fields.setdefault(key, value)
    return fields


def _parse_scontrol_config(output: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in output.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        fields[key.strip()] = value.strip()
    return fields


def _job_id(value: str) -> str:
    text = str(value or "").strip()
    if re.fullmatch(r"[1-9][0-9]*", text) is None:
        raise ValueError(f"Slurm job id must be a positive integer: {value!r}")
    return text


def _sha256(value: str) -> str:
    text = str(value or "")
    if re.fullmatch(r"[0-9a-f]{64}", text) is None:
        raise ValueError("SHA-256 must be 64 lowercase hexadecimal characters.")
    return text


def _cluster_name(value: str | None) -> str:
    text = str(value or "")
    if text and re.fullmatch(r"[A-Za-z0-9_.-]+", text) is None:
        raise ValueError(f"Slurm cluster name is invalid: {value!r}")
    return text


def _null_to_empty(value: str) -> str:
    return "" if value in {"", "(null)", "None", "N/A", "Unknown"} else value


def _positive_int(value: Any, field: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{field} must be a positive integer.")
    return value


def _job_name(experiment_id: str, run_id: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "-", f"{experiment_id}-{run_id}").strip("-.")
    return (value or "agent-tools-run")[:128]


def _utc_now() -> str:
    return manifests.utc_now()


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
        "execution_snapshot_path",
        "execution_snapshot_sha256",
        "log_path",
        "submit_token",
        "workdir",
        "python",
        "runtime_commit",
        "module",
    ):
        worker.add_argument(f"--{name.replace('_', '-')}", required=True)
    worker.add_argument("--gpus-per-run", required=True, type=int)
    args = parser.parse_args(argv)
    payload = vars(args)
    payload.pop("command_name")
    return run_frozen_job(**payload)


if __name__ == "__main__":
    raise SystemExit(_main())
