#!/usr/bin/env bash
set -euo pipefail

# Run repository-wide formatting and linting, then type-check agent_tools.
# Mirrors the style_check workflow so a clean local run means a clean CI run.
# Tools are expected to be available in the current Python environment.
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

python -m isort .
python -m black .
python -m flake8 .
# agent_tools carries a complexity ceiling the rest of the repo does not: the
# model/training code is legitimately long-bodied, while agent_tools is control
# flow whose branch count is the thing worth capping. Scoped through its own
# checker rather than a repo-wide .flake8 setting plus per-directory C901
# ignores, which would leave the ignored directories unchecked forever. The
# checker also holds the suppression ledger honest and reaches the embedded
# python -c programs, which flake8's directory walk cannot see.
python utils/complexity_check.py
python utils/type_check.py
