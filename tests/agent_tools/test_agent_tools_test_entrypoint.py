import os
from pathlib import Path
import subprocess
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = REPO_ROOT / "utils" / "test_agent_tools.py"


@pytest.mark.parametrize("accelerator", [None, "cpu"], ids=["default-cpu", "explicit-cpu"])
def test_runner_child_defaults_and_argument_passthrough(tmp_path, accelerator):
    probe = tmp_path / "test_probe.py"
    probe.write_text(
        "import os\n"
        "from pathlib import Path\n\n"
        "def test_selected(pytestconfig):\n"
        "    assert os.environ['DS_ACCELERATOR'] == 'cpu'\n"
        f"    assert Path.cwd() == Path({str(REPO_ROOT)!r})\n"
        "    assert pytestconfig.getini('testpaths') == ['tests/agent_tools']\n"
        "    assert pytestconfig.option.randomly_reset_seed is False\n"
        "    assert pytestconfig.option.randomly_reorganize is True\n"
        "    assert pytestconfig.option.randomly_seed == 1729\n\n"
        "def test_excluded():\n"
        "    assert False, '-k was not forwarded'\n",
        encoding="utf-8",
    )
    env = os.environ.copy()
    env.pop("PYTEST_ADDOPTS", None)
    env.pop("DS_ACCELERATOR", None)
    if accelerator is not None:
        env["DS_ACCELERATOR"] = accelerator
    result = subprocess.run(
        [sys.executable, str(RUNNER), str(probe), "-q", "-k", "selected", "--randomly-seed=1729"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "1 passed, 1 deselected" in result.stdout


def test_runner_propagates_pytest_failure_exit_code(tmp_path):
    probe = tmp_path / "test_failure.py"
    probe.write_text("def test_failure():\n    assert False, 'intentional probe failure'\n", encoding="utf-8")
    env = os.environ.copy()
    env.pop("PYTEST_ADDOPTS", None)
    result = subprocess.run(
        [sys.executable, str(RUNNER), str(probe), "-q", "--randomly-seed=1729"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == pytest.ExitCode.TESTS_FAILED, result.stdout + result.stderr
    assert "intentional probe failure" in result.stdout
    assert "1 failed" in result.stdout


def test_runner_seed_controls_collection_order_without_changing_membership(tmp_path):
    probe = tmp_path / "test_order.py"
    names = {f"test_item_{index:02d}" for index in range(24)}
    probe.write_text("\n".join(f"def {name}():\n    pass\n" for name in sorted(names)), encoding="utf-8")
    env = os.environ.copy()
    env.pop("PYTEST_ADDOPTS", None)
    orders = []
    for seed in (1729, 1729, 2718):
        result = subprocess.run(
            [sys.executable, str(RUNNER), str(probe), "--collect-only", "-q", f"--randomly-seed={seed}"],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        order = [line.rsplit("::", 1)[1] for line in result.stdout.splitlines() if "test_order.py::" in line]
        assert len(order) == len(names)
        assert set(order) == names
        orders.append(order)
    assert orders[0] == orders[1]
    assert orders[0] != orders[2]
