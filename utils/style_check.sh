#!/usr/bin/env bash
set -euo pipefail

# Run repository-wide formatting and linting.
# Tools are expected to be available in the current Python environment.
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

python -m isort .
python -m black .
python -m flake8 .
