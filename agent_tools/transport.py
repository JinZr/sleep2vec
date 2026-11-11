"""Shared local shell command construction and execution.

Provides shell quoting and the bounded local ``bash -lc`` runner and its
default timeout.
"""

from __future__ import annotations

import shlex
import subprocess
from typing import Any

COMMAND_TIMEOUT_SECONDS = 10


def sh(value: Any) -> str:
    return shlex.quote(str(value))


def run_shell(
    command: str,
    *,
    timeout: float | None = COMMAND_TIMEOUT_SECONDS,
    swallow_timeout: bool = False,
) -> subprocess.CompletedProcess:
    """Run one command through local ``bash -lc``.
    swallow_timeout=True reproduces the evidence-probe semantics: a timeout is
    reported as returncode 124 instead of raising."""
    argv = ["bash", "-lc", command]
    try:
        return subprocess.run(argv, text=True, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        if not swallow_timeout:
            raise
        return subprocess.CompletedProcess(argv, 124, "", f"timed out after {timeout}s")
