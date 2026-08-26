from __future__ import annotations

from pathlib import Path
import sys

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[2])]

from agent_tool_test_helpers import run_execution_preflight_fixture

from agent_tools import experiment_io, managed_scheduler
from agent_tools.cli import main

managed_scheduler.run_execution_command = run_execution_preflight_fixture
_validate_managed_output_paths = experiment_io.validate_managed_output_paths


def validate_managed_output_paths(root, paths, *, remote=None):
    if remote is None:
        return _validate_managed_output_paths(root, paths)


experiment_io.validate_managed_output_paths = validate_managed_output_paths
raise SystemExit(main(sys.argv[1:]))
