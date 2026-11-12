from __future__ import annotations

import json
import os
from pathlib import Path

from agent_tool_test_helpers import FakeLauncher, read_table, run_cli, write_local_hparam_recipe
import pytest
import yaml

from agent_tools import hparam_runtime
from agent_tools.experiment_workspace import MONITOR_EXIT_CODE_PREFIX

pytestmark = pytest.mark.usefixtures("stub_execution_target")


def test_hparam_launch_rejects_plan_without_workspace_binding_before_start(tmp_path: Path):
    recipe = write_local_hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    assert run_cli("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    plan = json.loads((plan_dir / "plan.json").read_text())
    plan["recipe"].pop("experiment")
    (plan_dir / "plan.json").write_text(json.dumps(plan))
    launcher = FakeLauncher()

    with pytest.raises(ValueError, match="workspace binding"):
        hparam_runtime.launch_hparam_runs(plan_dir, dry_run=False, hooks=launcher.hooks())

    launcher.assert_not_started()


def test_hparam_plan_canonicalizes_relative_workspace_root_consistently(tmp_path: Path):
    recipe = write_local_hparam_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    payload["experiment"]["root"] = os.path.relpath(tmp_path, Path.cwd())
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
    plan_dir = tmp_path / "plan"

    plan_result = run_cli("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir))
    launch_result = run_cli("hparam-run-queue", "--plan-dir", str(plan_dir))

    assert plan_result.returncode == 0, plan_result.stderr
    assert launch_result.returncode == 0, launch_result.stderr
    plan = json.loads((plan_dir / "plan.json").read_text())
    manifest = yaml.safe_load((tmp_path / "experiment.yaml").read_text())
    assert plan["recipe"]["experiment"]["root"] == str(tmp_path)
    assert manifest["experiment"]["root"] == str(tmp_path)


def test_hparam_plan_records_monitor_owned_exit_status_contract(tmp_path: Path):
    recipe = write_local_hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"

    result = run_cli("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir))

    assert result.returncode == 0, result.stderr
    run = json.loads((plan_dir / "plan.json").read_text())["runs"][0]
    canonical = read_table(tmp_path / "run_manifest.tsv")[0]
    script = Path(run["script"]).read_text()
    assert run["terminal_status_owner"] == "monitor"
    assert run["scheduler_type"] == "direct"
    assert json.loads((plan_dir / "plan.json").read_text())["recipe"]["execution"]["scheduler"] == {"type": "direct"}
    assert canonical["terminal_status_owner"] == "monitor"
    assert canonical["scheduler_type"] == "direct"
    assert "trap _agent_tools_record_exit EXIT" in script
    assert MONITOR_EXIT_CODE_PREFIX in script


def test_hparam_plan_rejects_duplicate_gpu_assignments_within_a_run(tmp_path: Path):
    recipe = write_local_hparam_recipe(
        tmp_path,
        execution={"workdir": str(tmp_path), "gpu_pool": [0, 0], "gpus_per_run": 2},
    )

    result = run_cli("plan", "--recipe", str(recipe), "--output-dir", str(tmp_path / "plan"))

    assert result.returncode == 1
    assert "must not contain duplicate GPU identifiers" in result.stdout


@pytest.mark.parametrize("env_name", ["PYTHONPATH", "WANDB_PROJECT", "WANDB_GROUP", "WANDB_RUN_GROUP", "WANDB_MODE"])
def test_hparam_plan_rejects_environment_semantic_aliases(tmp_path: Path, env_name: str):
    recipe = write_local_hparam_recipe(
        tmp_path,
        execution={"workdir": str(tmp_path), "env": {env_name: "unit"}},
    )

    result = run_cli("plan", "--recipe", str(recipe), "--output-dir", str(tmp_path / "plan"))

    assert result.returncode == 1
    assert f"execution.env.{env_name}" in result.stdout


@pytest.mark.parametrize("field", ["target", "host", "path_context", "path_validation"])
def test_hparam_doctor_rejects_removed_ssh_execution_fields(tmp_path: Path, field: str):
    value = {"target": "ssh", "host": "runtime-host", "path_context": "remote", "path_validation": "ssh"}[field]
    recipe = write_local_hparam_recipe(tmp_path, execution={"workdir": str(tmp_path), field: value})

    result = run_cli("doctor", "--recipe", str(recipe), "--output-dir", str(tmp_path / "doctor"))

    assert result.returncode == 1
    assert f"execution.{field}" in result.stdout


def test_hparam_doctor_rejects_deprecated_log_and_pid_dirs(tmp_path: Path):
    recipe = write_local_hparam_recipe(tmp_path, execution={"log_dir": "logs", "pid_dir": "pids"})

    result = run_cli("doctor", "--recipe", str(recipe), "--output-dir", str(tmp_path / "doctor"))

    assert result.returncode == 1
    assert "execution.log_dir" in result.stdout
    assert "execution.pid_dir" in result.stdout
