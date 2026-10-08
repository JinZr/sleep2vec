import os
from pathlib import Path
import subprocess
import sys
import zlib

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


def test_runner_shards_partition_collected_tests():
    # pytest applies tests/conftest.py only to paths below tests/, so the probe is an existing module there.
    module = REPO_ROOT / "tests" / "agent_tools" / "test_agent_layering.py"
    command = [sys.executable, str(RUNNER), str(module), "--collect-only", "-q"]
    env = os.environ.copy()
    env.pop("PYTEST_ADDOPTS", None)
    outputs = []
    # Each shard runs in its own process under a different shuffled collection order.
    for options in (
        ["-p", "no:randomly"],
        ["--randomly-seed=1729", "--shard-count", "2", "--shard-index", "0"],
        ["--randomly-seed=2718", "--shard-count", "2", "--shard-index", "1"],
    ):
        result = subprocess.run([*command, *options], env=env, capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, result.stdout + result.stderr
        outputs.append(result.stdout)
    full, shard_0, shard_1 = [
        {line for line in output.splitlines() if "test_agent_layering.py::" in line} for output in outputs
    ]
    assert shard_0 and shard_1
    assert shard_0.isdisjoint(shard_1)
    assert shard_0 | shard_1 == full
    # Recomputing the documented crc32 rule here pins that membership depends only on the node id.
    assert shard_0 == {node for node in full if zlib.crc32(node.encode()) % 2 == 0}
    assert f"{len(shard_0)}/{len(full)} tests collected ({len(shard_1)} deselected)" in outputs[1]

    result = subprocess.run(
        [*command, "--shard-count", "2", "--shard-index", "2"], env=env, capture_output=True, text=True, timeout=120
    )
    assert result.returncode == pytest.ExitCode.USAGE_ERROR, result.stdout + result.stderr
    assert "--shard-count 2 --shard-index 2" in result.stderr


@pytest.mark.parametrize("source", ["cli", "environment"])
@pytest.mark.parametrize("options", [["-p", "no:randomly"], ["-pno:randomly"]], ids=["separate", "compact"])
def test_runner_honors_fixed_order_plugin_disable(tmp_path, source, options):
    probe = tmp_path / "test_fixed_order.py"
    probe.write_text(
        "def test_z_first(pytestconfig):\n"
        "    assert not pytestconfig.pluginmanager.hasplugin('randomly')\n\n"
        "def test_a_second():\n"
        "    pass\n",
        encoding="utf-8",
    )
    env = os.environ.copy()
    env.pop("PYTEST_ADDOPTS", None)
    arguments = []
    if source == "environment":
        env["PYTEST_ADDOPTS"] = " ".join(options)
    else:
        arguments = options
    result = subprocess.run(
        [sys.executable, str(RUNNER), str(probe), "-v", *arguments],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "2 passed" in result.stdout
    assert result.stdout.index("test_z_first PASSED") < result.stdout.index("test_a_second PASSED")
