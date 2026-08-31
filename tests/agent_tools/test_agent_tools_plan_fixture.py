from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

from agent_tool_test_helpers import prepare_hparam_plan_fixture, run_execution_preflight_fixture
import pytest
from test_agent_tools_hparam_selection import _hparam_recipe
import yaml

from agent_tools import cli, experiment_io, managed_scheduler
from agent_tools.experiment_workspace import read_run_manifest


def test_plan_fixture_restores_state_after_rejection_then_prepares_a_fresh_plan(tmp_path: Path, monkeypatch):
    rejected_recipe = _hparam_recipe(tmp_path / "rejected")
    payload = yaml.safe_load(rejected_recipe.read_text())
    payload.pop("step")
    rejected_recipe.write_text(yaml.safe_dump(payload))
    valid_recipe = _hparam_recipe(tmp_path / "valid")
    execution_command = managed_scheduler.run_execution_command
    output_validator = experiment_io.validate_managed_output_paths
    process_state = (Path.cwd(), dict(os.environ), list(sys.argv), list(sys.path), sys.stdout, sys.stderr)
    monkeypatch.setattr(subprocess, "Popen", lambda *_args, **_kwargs: pytest.fail("Fixture must not launch a child"))

    with pytest.raises(AssertionError, match="NEEDS_USER_INPUT"):
        prepare_hparam_plan_fixture(rejected_recipe, rejected_recipe.parent / "plan")

    assert managed_scheduler.run_execution_command is execution_command
    assert experiment_io.validate_managed_output_paths is output_validator
    assert (Path.cwd(), dict(os.environ), list(sys.argv), list(sys.path), sys.stdout, sys.stderr) == process_state
    assert read_run_manifest(rejected_recipe.parent) == []

    plan_dir = valid_recipe.parent / "plan"
    assert prepare_hparam_plan_fixture(valid_recipe, plan_dir) is None

    assert managed_scheduler.run_execution_command is execution_command
    assert experiment_io.validate_managed_output_paths is output_validator
    assert (Path.cwd(), dict(os.environ), list(sys.argv), list(sys.path), sys.stdout, sys.stderr) == process_state
    plan = json.loads((plan_dir / "plan.json").read_text())
    assert len(plan["runs"]) == 1
    assert read_run_manifest(valid_recipe.parent)[0]["status"] == "planned"


def test_plan_fixture_restores_exact_stubs_and_streams_after_an_exception(tmp_path: Path, monkeypatch):
    execution_command = managed_scheduler.run_execution_command
    validator_calls = []

    def output_validator(root, paths):
        validator_calls.append((root, paths))

    monkeypatch.setattr(experiment_io, "validate_managed_output_paths", output_validator)
    streams = (sys.stdout, sys.stderr)
    recipe, plan_dir = tmp_path / "recipe.yaml", tmp_path / "plan"

    def failing_main(argv):
        assert argv == ["plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)]
        assert managed_scheduler.run_execution_command is run_execution_preflight_fixture
        experiment_io.validate_managed_output_paths(tmp_path, [plan_dir])
        experiment_io.validate_managed_output_paths(tmp_path, [plan_dir], remote="unused-test-host")
        print("fixture stdout")
        print("fixture stderr", file=sys.stderr)
        raise ValueError("fixture failure")

    monkeypatch.setattr(cli, "main", failing_main)

    with pytest.raises(ValueError, match="fixture failure"):
        prepare_hparam_plan_fixture(recipe, plan_dir)

    assert validator_calls == [(tmp_path, [plan_dir])]
    assert managed_scheduler.run_execution_command is execution_command
    assert experiment_io.validate_managed_output_paths is output_validator
    assert (sys.stdout, sys.stderr) == streams
