"""Slurm resource and batch-script contracts and the sbatch / squeue / scancel transport.

Owns the frozen ``job.sbatch`` shape, resource normalization, submission, the
per-user ``squeue`` snapshot parser, state categories and cancellation. Slurm
owns queueing and job state; the batch script records its own exit code and
runtime commit in the run directory.

Every probe requests a machine-readable format (``--parsable``, an explicit
``--format``), so a change to the cluster's human-facing output cannot silently
alter an observed job state.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import re
import subprocess
from typing import Any, TypedDict

from . import transport


class SlurmResources(TypedDict):
    partition: str
    cpus_per_task: int
    memory: str
    walltime: str
    nice: int
    nodelist: str
    gpus_per_run: int


@dataclass(frozen=True)
class JobObservation:
    job_id: str
    name: str
    state: str
    reason: str = ""
    node_list: str = ""


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
}
QUEUED_STATES = frozenset(
    {
        "PENDING",
        "EXPEDITING",
        "POWER_UP_NODE",
        "REQUEUED",
        "REQUEUE_FED",
        "REQUEUE_HOLD",
        "RESV_DEL_HOLD",
        "SPECIAL_EXIT",
    }
)
RUNNING_STATES = frozenset(
    {
        "RUNNING",
        "CONFIGURING",
        "COMPLETING",
        "SUSPENDED",
        "RESIZING",
        "SIGNALING",
        "STAGE_OUT",
        "STOPPED",
        "UPDATE_DB",
    }
)
_SQUEUE_FORMAT = "%i|%j|%T|%r|%N"


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
    return {
        "partition": partition,
        "cpus_per_task": cpus_per_task,
        "memory": memory,
        "walltime": walltime,
        "nice": nice,
        "nodelist": nodelist,
        "gpus_per_run": gpus,
    }


def job_name(run: Mapping[str, Any]) -> str:
    """The Slurm job name that identifies one managed run in ``squeue --me``."""
    return f"{run['experiment_id']}/{run['step_id']}/{run['run_id']}"


def render_batch_script(
    *,
    run: Mapping[str, Any],
    execution: Mapping[str, Any],
    resources: SlurmResources,
) -> str:
    """Render the frozen job.sbatch for one run.

    The script records ``git rev-parse HEAD`` of the workdir into run_dir/runtime_commit
    without blocking on drift, checks the frozen config and launch script digests, runs
    launch.sh once per GPU task through srun, and atomically writes its exit code to
    run_dir/exit_code, including on SIGTERM/SIGINT. It embeds the digests of the other
    frozen files, never its own."""
    workdir = transport.sh(str(execution["workdir"]))
    run_dir = str(run["run_dir"])
    tasks = resources["gpus_per_run"]
    directives = [
        f"#SBATCH --job-name={job_name(run)}",
        f"#SBATCH --partition={resources['partition']}",
        "#SBATCH --nodes=1",
        f"#SBATCH --ntasks={tasks}",
        f"#SBATCH --ntasks-per-node={tasks}",
        f"#SBATCH --gres=gpu:{tasks}",
        f"#SBATCH --cpus-per-task={resources['cpus_per_task']}",
        f"#SBATCH --mem={resources['memory']}",
        f"#SBATCH --time={resources['walltime']}",
        f"#SBATCH --nice={resources['nice']}",
    ]
    if resources["nodelist"]:
        directives.append(f"#SBATCH --nodelist={resources['nodelist']}")
    directives.extend(
        [
            "#SBATCH --no-requeue",
            f"#SBATCH --output={transport.sh(str(run['log_path']).replace('%', '%%'))}",
        ]
    )
    env_lines = [f"export {key}={transport.sh(value)}" for key, value in sorted((execution.get("env") or {}).items())]
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            *directives,
            "set -uo pipefail",
            f"run_dir={transport.sh(run_dir)}",
            "finish() { trap '' TERM INT; printf '%s\\n' \"$1\" > \"$run_dir/exit_code.tmp\" "
            '&& mv -f "$run_dir/exit_code.tmp" "$run_dir/exit_code"; exit "$1"; }',
            # Explicit traps: bash runs an EXIT trap with $?=0 after SIGTERM, which would record success.
            "trap 'finish 143' TERM",
            "trap 'finish 130' INT",
            'echo "agent_tools job=$SLURM_JOB_ID node=$(hostname) start=$(date -u +%FT%TZ)"',
            f"cd {workdir} || finish 3",
            f"export PYTHONPATH={workdir}",
            *env_lines,
            "unset RANK LOCAL_RANK WORLD_SIZE MASTER_ADDR MASTER_PORT",
            # Record the checkout that runs; the monitor reports drift from the planned commit.
            'git rev-parse HEAD > "$run_dir/runtime_commit" || finish 3',
            "sha256sum --check --strict <<'EOF' || finish 3",
            f"{run['config_sha256']}  {run['config']}",
            f"{run['script_sha256']}  {run['script']}",
            "EOF",
            f"srun --nodes=1 --ntasks={tasks} --ntasks-per-node={tasks} --kill-on-bad-exit=1 --label "
            f"bash {transport.sh(str(run['script']))}",
            "finish $?",
            "",
        ]
    )


def run_command(
    argv: list[str],
    *,
    timeout: float = transport.COMMAND_TIMEOUT_SECONDS,
) -> subprocess.CompletedProcess:
    # Slurm commands run on the host where agent_tools runs.
    return transport.run_shell(_shell_command(argv), timeout=timeout)


def parse_sbatch_output(stdout: str) -> str:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise ValueError("sbatch --parsable must return exactly one non-empty line.")
    match = re.fullmatch(r"([1-9][0-9]*)(?:;[A-Za-z0-9_.-]+)?", lines[0])
    if match is None:
        raise ValueError(f"Invalid sbatch --parsable output: {lines[0]!r}")
    return match.group(1)


def submit(script: str, *, timeout: float = transport.COMMAND_TIMEOUT_SECONDS) -> str:
    """Submit one frozen job.sbatch with SBATCH_* stripped and return its job id."""
    result = run_command(_submission_argv(script), timeout=timeout)
    if result.returncode != 0:
        raise SlurmCommandError("submission", result)
    return parse_sbatch_output(str(result.stdout or ""))


def user_jobs(*, timeout: float = transport.COMMAND_TIMEOUT_SECONDS) -> list[JobObservation]:
    """Return every job ``squeue --me --states=all`` reports, including recently finished ones.

    One query; a failed query raises rather than reading as an empty queue."""
    result = run_command(
        ["squeue", "--me", "--states=all", "--noheader", f"--format={_SQUEUE_FORMAT}"], timeout=timeout
    )
    if result.returncode != 0:
        raise SlurmCommandError("queue query", result)
    return [parse_squeue_line(line) for line in str(result.stdout or "").splitlines() if line.strip()]


def parse_squeue_line(line: str) -> JobObservation:
    # Only names can contain the delimiter; job id, state, reason and node list cannot.
    head, separator, tail = line.strip().partition("|")
    fields = tail.rsplit("|", 3)
    if not separator or len(fields) != 4:
        raise ValueError(f"Invalid squeue output row: {line!r}")
    name, state, reason, node_list = fields
    return JobObservation(
        job_id=head.strip(),
        name=name,
        state=normalize_state(state),
        reason=_null_to_empty(reason.strip()),
        node_list=_null_to_empty(node_list.strip()),
    )


def cancel(job_id: str, *, timeout: float = transport.COMMAND_TIMEOUT_SECONDS) -> None:
    result = run_command(["scancel", _job_id(job_id)], timeout=timeout)
    if result.returncode != 0:
        raise SlurmCommandError("cancellation", result)


def normalize_state(value: str) -> str:
    state = str(value or "").strip().upper()
    while state.endswith("+"):
        state = state[:-1]
    return state


def submission_command(script: str) -> str:
    return _shell_command(_submission_argv(script))


def _shell_command(argv: list[str]) -> str:
    return " ".join(transport.sh(part) for part in ["env", "-u", "SLURM_CLUSTERS", *argv])


def _submission_argv(script: str) -> list[str]:
    return [
        "bash",
        "-c",
        'for name in "${!SBATCH_@}"; do unset "$name"; done; exec "$@"',
        "agent-tools-sbatch",
        "sbatch",
        "--parsable",
        script,
    ]


def _job_id(value: str) -> str:
    text = str(value or "").strip()
    if re.fullmatch(r"[1-9][0-9]*", text) is None:
        raise ValueError(f"Slurm job id must be a positive integer: {value!r}")
    return text


def _null_to_empty(value: str) -> str:
    return "" if value in {"", "(null)", "None", "N/A", "Unknown"} else value


def _positive_int(value: Any, field: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{field} must be a positive integer.")
    return value
