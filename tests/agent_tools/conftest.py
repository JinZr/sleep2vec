from agent_tool_test_helpers import run_execution_preflight_fixture
import pytest

from agent_tools import execution_snapshot


@pytest.fixture
def stub_execution_target(monkeypatch, request):
    """Answer execution-target probes in process; test_execution_probe_* tests keep the real probe.

    Modules opt in with ``pytestmark = pytest.mark.usefixtures("stub_execution_target")``."""
    if not request.node.name.startswith("test_execution_probe_"):
        monkeypatch.setattr(execution_snapshot, "run_execution_command", run_execution_preflight_fixture)
