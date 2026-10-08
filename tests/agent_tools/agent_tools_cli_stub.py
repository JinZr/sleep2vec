from __future__ import annotations

from pathlib import Path
import sys

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[2])]

from agent_tool_test_helpers import run_execution_preflight_fixture, validate_local_managed_output_paths

from agent_tools import execution_snapshot, experiment_io
from agent_tools.cli import main

execution_snapshot.run_execution_command = run_execution_preflight_fixture
experiment_io.validate_managed_output_paths = validate_local_managed_output_paths
raise SystemExit(main(sys.argv[1:]))
