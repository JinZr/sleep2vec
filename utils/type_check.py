#!/usr/bin/env python3
"""Type-check agent_tools once, without global or per-module error suppression.

Scope and settings live in pyproject.toml. Both utils/style_check.sh and CI use
this entrypoint; neither requires Git history or a base revision.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

if sys.version_info >= (3, 11):
    import tomllib
else:
    # mypy supplies tomli on Python versions without stdlib tomllib.
    import tomli as tomllib

PYPROJECT = Path("pyproject.toml")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.parse_args(argv)

    mypy = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["tool"]["mypy"]
    if mypy.get("ignore_errors") or any(override.get("ignore_errors") for override in mypy.get("overrides", [])):
        print("ignore_errors is not allowed in the mypy configuration; fix type errors instead of suppressing them.")
        return 1

    print("== mypy ==", flush=True)
    return subprocess.run([sys.executable, "-m", "mypy", "--config-file", str(PYPROJECT)]).returncode


if __name__ == "__main__":
    raise SystemExit(main())
