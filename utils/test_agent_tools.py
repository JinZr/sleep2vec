"""Run agent-tools tests with the same CPU and random-order policy locally and in CI."""

import os
from pathlib import Path
import shlex
import subprocess
import sys

if __name__ == "__main__":
    environment = os.environ.copy()
    environment.setdefault("DS_ACCELERATOR", "cpu")
    arguments = [*shlex.split(environment.get("PYTEST_ADDOPTS", "")), *sys.argv[1:]]
    randomly_disabled = "-pno:randomly" in arguments or any(
        option == "-p" and value == "no:randomly" for option, value in zip(arguments, arguments[1:])
    )
    random_options = [] if randomly_disabled else ["--randomly-dont-reset-seed"]
    sys.exit(
        subprocess.call(
            [
                sys.executable,
                "-m",
                "pytest",
                "-o",
                "testpaths=tests/agent_tools",
                *random_options,
                *sys.argv[1:],
            ],
            cwd=Path(__file__).resolve().parents[1],
            env=environment,
        )
    )
