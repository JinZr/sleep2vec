from __future__ import annotations

from pathlib import Path
import shlex
import subprocess
from typing import Any

SSH_TIMEOUT_SECONDS = 10
REMOTE_MISSING_RETURN_CODE = 44
REMOTE_CONFLICT_RETURN_CODE = 45


def sh(value: Any) -> str:
    return shlex.quote(str(value))


def ssh_argv(host: str, command: str) -> list[str]:
    return ["ssh", str(host), command]


def run_ssh(
    host: str,
    command: str,
    *,
    input: str | bytes | None = None,
    text: bool | None = None,
    check: bool | None = None,
    capture_output: bool = True,
    timeout: float | None = SSH_TIMEOUT_SECONDS,
) -> subprocess.CompletedProcess:
    """Remote execution primitive. kwargs are forwarded conditionally so that
    subprocess.run receives exactly the keys each call site passed before the
    consolidation (None means "do not pass the key"). TimeoutExpired is never
    swallowed here."""
    kwargs: dict[str, Any] = {}
    if capture_output:
        kwargs["capture_output"] = True
    if input is not None:
        kwargs["input"] = input
    if text is not None:
        kwargs["text"] = text
    if check is not None:
        kwargs["check"] = check
    if timeout is not None:
        kwargs["timeout"] = timeout
    return subprocess.run(ssh_argv(host, command), **kwargs)


def run_shell(
    host: str | None,
    command: str,
    *,
    timeout: float = SSH_TIMEOUT_SECONDS,
    swallow_timeout: bool = False,
) -> subprocess.CompletedProcess:
    """Dispatching primitive: host -> ssh, None -> local bash -lc.
    swallow_timeout=True reproduces the evidence-probe semantics: a timeout is
    reported as returncode 124 instead of raising."""
    argv = ssh_argv(host, command) if host else ["bash", "-lc", command]
    try:
        return subprocess.run(argv, text=True, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        if not swallow_timeout:
            raise
        return subprocess.CompletedProcess(argv, 124, "", f"timed out after {timeout}s")


def remote_python_command(script: str, *args: Any) -> str:
    return " ".join([f"python3 -c {sh(script)}", *(sh(arg) for arg in args)])


def remote_write_command(path: Any) -> str:
    target = Path(str(path))
    return f"mkdir -p {sh(target.parent)} && cat > {sh(target)}"


def remote_append_command(path: Any) -> str:
    target = Path(str(path))
    return f"mkdir -p {sh(target.parent)} && cat >> {sh(target)}"
