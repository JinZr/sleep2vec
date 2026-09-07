"""Run agent-tools tests with the same CPU and random-order policy locally and in CI."""

import os
from pathlib import Path
import subprocess
import sys

if __name__ == "__main__":
    environment = os.environ.copy()
    environment.setdefault("DS_ACCELERATOR", "cpu")
    sys.exit(
        subprocess.call(
            [
                sys.executable,
                "-m",
                "pytest",
                "-o",
                "testpaths=tests/agent_tools",
                "--randomly-dont-reset-seed",
                *sys.argv[1:],
            ],
            cwd=Path(__file__).resolve().parents[1],
            env=environment,
        )
    )
