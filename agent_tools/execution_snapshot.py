"""Frozen execution snapshot of the target runtime a managed plan launches into.

Layer 0 leaf. The execution snapshot (``execution_snapshot.json``) is the
evidence a managed plan freezes about the runtime its runs execute in: the
execution target, the interpreter and its version, the planned and observed
checkout commits, the one runtime module and its origin, and the CLI options the
planned arguments need and that module accepts. This module owns that shape, the
target probe that produces it, the check that a frozen snapshot still matches
the live target, and the snapshot's atomic write. The embedded
``managed_scheduler.runtime_identity`` and ``managed_scheduler.cli_preflight``
programs it runs are shared with the scheduler's launch-time verification.

Imports only ``models``, ``experiment_workspace``, ``manifests``,
``python_programs`` and ``transport``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
from typing import Any, Literal, TypedDict

from . import python_programs, transport
from .experiment_workspace import LAUNCHABLE_STATUSES, RunKey, validated_run_key
from .manifests import read_json
from .models import REPO_ROOT, JsonValue, is_full_git_object_id

# Complete repository scans can be slow on shared experiment filesystems.
_EXECUTION_PREFLIGHT_TIMEOUT_SECONDS = 300
EXECUTION_SNAPSHOT_NAME = "execution_snapshot.json"


class PlannedArgv(TypedDict):
    run_id: str
    args: list[str]


class ExecutionSnapshot(TypedDict, total=False):
    # External identity fields are passed through, including overrides of execution metadata.
    target: JsonValue
    host: JsonValue
    workdir: JsonValue
    conda_env: JsonValue
    python_command: JsonValue
    expected_runtime_commit: JsonValue
    execution_env_sha256: JsonValue
    python: JsonValue
    python_version: JsonValue
    runtime_repo_root: JsonValue
    runtime_hostname: JsonValue
    module_origin: JsonValue
    runtime_commit: str
    module: str
    required_options: list[str]
    supported_options: list[JsonValue]
    cli_options_sha256: str
    validated_argv_sha256: str


ExecutionSnapshotResult = tuple[ExecutionSnapshot, bool] | tuple[None, Literal[False]]


def validated_execution_snapshot(
    owner_dir: str | Path,
    execution: Mapping[str, JsonValue],
    runs: Sequence[Mapping[str, JsonValue]],
    workspace_by_key: Mapping[RunKey, Mapping[str, JsonValue]],
    *,
    inspector: Callable[[Mapping[str, JsonValue], Sequence[Mapping[str, JsonValue]]], ExecutionSnapshot] | None = None,
    plan_label: str = "managed",
) -> tuple[ExecutionSnapshot, bool]:
    root = Path(owner_dir)
    snapshot_path = root / EXECUTION_SNAPSHOT_NAME
    inspect = inspector or inspect_execution_target
    if snapshot_path.exists():
        frozen = read_json(snapshot_path)
        actual = inspect(execution, runs)
        # A rolling checkout may advance after registration; its live commit is recorded per run at launch.
        rolling_evidence_fields = {"runtime_commit", "supported_options", "cli_options_sha256"}
        changed = sorted(
            key
            for key in set(frozen) | set(actual)
            if key not in rolling_evidence_fields and frozen.get(key) != actual.get(key)
        )
        if changed:
            raise ValueError(f"Frozen execution snapshot changed: {', '.join(changed)}")
        return actual, False
    for run in runs:
        row = workspace_by_key[validated_run_key(run)]
        if row.get("target") not in (None, "") or row.get("status") not in LAUNCHABLE_STATUSES:
            raise ValueError(
                f"Cannot establish an execution snapshot after a {plan_label} run has started; create a new plan."
            )
    return inspect(execution, runs), True


def write_execution_snapshot_file(path: str | Path, snapshot: ExecutionSnapshot) -> None:
    snapshot_path = Path(path)
    payload = (json.dumps(snapshot, indent=2, sort_keys=True) + "\n").encode()
    descriptor, temporary = tempfile.mkstemp(prefix=f".{snapshot_path.name}.", dir=snapshot_path.parent)
    try:
        with os.fdopen(descriptor, "wb") as file_obj:
            file_obj.write(payload)
            file_obj.flush()
            os.fsync(file_obj.fileno())
        os.replace(temporary, snapshot_path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def inspect_execution_target(
    execution: Mapping[str, JsonValue],
    runs: Sequence[Mapping[str, JsonValue]],
    *,
    command_runner: Callable[[Mapping[str, JsonValue], list[str]], subprocess.CompletedProcess] | None = None,
    plan_label: str = "managed",
) -> ExecutionSnapshot:
    modules: set[str] = set()
    python_commands: set[str] = set()
    planned_argv: list[PlannedArgv] = []
    required_options: set[str] = set()
    for run in runs:
        command = str(run.get("command") or "")
        if command not in Path(str(run["script"])).read_text().splitlines():
            raise ValueError(f"Frozen {plan_label} command differs from its launch script: {run['run_id']}")
        tokens = shlex.split(command)
        try:
            module_flag_index = tokens.index("-m")
            module_index = module_flag_index + 1
            modules.add(tokens[module_index])
        except (IndexError, ValueError) as exc:
            raise ValueError(f"Frozen {plan_label} command has no Python module: {run['run_id']}") from exc
        if module_flag_index != 1:
            raise ValueError(f"Frozen {plan_label} command has an unsupported Python invocation: {run['run_id']}")
        python_commands.add(tokens[0])
        planned_argv.append({"run_id": str(run["run_id"]), "args": tokens[module_index + 1 :]})
        required_options.update(token for token in tokens[module_index + 1 :] if token.startswith("--"))
    if len(modules) != 1:
        raise ValueError(f"A {plan_label} plan must use exactly one target runtime module.")
    if len(python_commands) != 1:
        raise ValueError(f"A {plan_label} plan must use exactly one target Python executable.")
    module = next(iter(modules))
    python_command = next(iter(python_commands))
    expected_python = execution.get("python")
    planned_commit = execution.get("runtime_commit")
    if expected_python in (None, "") or planned_commit in (None, ""):
        raise ValueError(
            f"Frozen {plan_label} plan lacks execution.python or execution.runtime_commit; create a new plan."
        )
    if python_command != str(expected_python):
        raise ValueError(f"Frozen {plan_label} commands differ from execution.python.")
    run_command = command_runner or run_execution_command
    identity_result = run_command(
        execution,
        [
            python_command,
            "-c",
            python_programs.source("managed_scheduler.runtime_identity"),
            module,
            "{}",
            "[]",
            str(planned_commit),
        ],
    )
    if identity_result.returncode != 0:
        detail = (
            identity_result.stderr.strip()
            or identity_result.stdout.strip()
            or f"exit code {identity_result.returncode}"
        )
        raise RuntimeError(f"Target execution identity preflight failed: {detail}")
    try:
        identity = json.loads(identity_result.stdout)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError("Target execution identity preflight returned malformed JSON.") from exc
    identity_fields = (
        "python",
        "python_version",
        "runtime_commit",
        "runtime_repo_root",
        "runtime_hostname",
        "module",
        "module_origin",
    )
    if not isinstance(identity, dict) or any(identity.get(field) in (None, "") for field in identity_fields):
        raise ValueError("Target execution identity preflight returned incomplete evidence.")
    if not is_full_git_object_id(identity["runtime_commit"]):
        raise ValueError("Target execution identity preflight returned an invalid runtime commit.")
    parse_result = run_command(
        execution,
        [
            python_command,
            "-c",
            python_programs.source("managed_scheduler.cli_preflight"),
            module,
            json.dumps(planned_argv),
            identity["module_origin"],
        ],
    )
    if parse_result.returncode != 0:
        detail = parse_result.stderr.strip() or parse_result.stdout.strip() or f"exit code {parse_result.returncode}"
        raise ValueError(f"Target runtime rejected frozen arguments: {detail}")
    marker = "AGENT_CLI_PREFLIGHT="
    evidence_lines = [line.removeprefix(marker) for line in parse_result.stdout.splitlines() if line.startswith(marker)]
    if len(evidence_lines) != 1:
        raise ValueError("Target runtime CLI preflight returned malformed evidence.")
    try:
        cli_evidence = json.loads(evidence_lines[0])
    except json.JSONDecodeError as exc:
        raise ValueError("Target runtime CLI preflight returned malformed evidence.") from exc
    supported_options = set(cli_evidence.get("supported_options") or []) if isinstance(cli_evidence, dict) else set()
    missing_options = sorted(required_options - supported_options)
    if missing_options:
        raise ValueError(f"Target runtime CLI {module} does not accept planned options: {', '.join(missing_options)}")
    cli_options_sha256 = cli_evidence.get("cli_options_sha256") if isinstance(cli_evidence, dict) else None
    if not isinstance(cli_options_sha256, str) or not cli_options_sha256:
        raise ValueError("Target runtime CLI preflight returned malformed evidence.")
    execution_env = execution.get("env") if isinstance(execution.get("env"), dict) else {}
    return {
        "target": str(execution.get("target", "local") or "local"),
        "host": str(execution.get("host") or ""),
        "workdir": str(execution.get("workdir") or REPO_ROOT),
        "conda_env": str(execution.get("conda_env") or ""),
        "python_command": python_command,
        "expected_runtime_commit": str(planned_commit),
        "execution_env_sha256": hashlib.sha256(
            json.dumps(execution_env, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        **identity,
        "module": module,
        "required_options": sorted(required_options),
        "supported_options": sorted(supported_options),
        "cli_options_sha256": cli_options_sha256,
        "validated_argv_sha256": hashlib.sha256(
            json.dumps(planned_argv, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }


def run_execution_command(execution: Mapping[str, Any], command: list[str]) -> subprocess.CompletedProcess:
    workdir = str(execution.get("workdir") or REPO_ROOT)
    inner = f"export PYTHONPATH={transport.sh(workdir)} && " + " ".join(transport.sh(part) for part in command)
    run = ["bash", "-c", inner]
    if execution.get("conda_env"):
        run = ["conda", "run", "--no-capture-output", "-n", str(execution["conda_env"]), *run]
    run_command = " ".join(transport.sh(part) for part in run)
    env = dict(execution.get("env") or {})
    if env:
        env_prefix = " ".join(f"{key}={transport.sh(value)}" for key, value in sorted(env.items()))
        run_command = f"env {env_prefix} {run_command}"
    run_command = f"cd {transport.sh(workdir)} && {run_command}"
    host = str(execution["host"]) if execution.get("target", "local") == "ssh" else None
    return transport.run_shell(host, run_command, timeout=_EXECUTION_PREFLIGHT_TIMEOUT_SECONDS)
