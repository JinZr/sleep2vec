from __future__ import annotations

import json
import os
from pathlib import Path

from agent_tool_test_helpers import workspace_files, write_finetune_recipe, write_public_hparam_recipe, write_yaml
import pytest
import yaml

from agent_tools import experiments, plan_contract, plan_hparam, plans
from agent_tools.models import REPO_ROOT

pytestmark = pytest.mark.usefixtures("stub_execution_target")


def test_experiment_status_accepts_public_generic_plan_without_runtime_directories(tmp_path):
    root = tmp_path / "experiment"
    payload = yaml.safe_load((REPO_ROOT / "recipes/examples/tiny_fixture_sleep2stat.yaml").read_text())
    recipe = write_yaml(root / "sleep2stat.yaml", payload)
    plan_dir = root / "plans" / "sleep2stat"

    assert plans.build_plan(recipe_path=recipe, output_dir=plan_dir).exit_code == 0
    planned = json.loads((plan_dir / "plan.json").read_text())["runs"][0]
    assert planned["runtime_dir"] == planned["checkpoint_dir"] == ""

    snapshot = experiments.experiment_status(root)

    assert snapshot["summary"]["state"] == "ready_to_launch"
    assert snapshot["blockers"] == []


def test_experiment_status_accepts_public_finetune_plan(tmp_path):
    root = tmp_path / "experiment"
    recipe = write_finetune_recipe(root)
    plan_dir = root / "plans" / "train"

    assert plans.build_plan(recipe_path=recipe, output_dir=plan_dir).exit_code == 0

    snapshot = experiments.experiment_status(root)

    assert snapshot["summary"]["state"] == "ready_to_launch"
    assert snapshot["blockers"] == []


def test_experiment_status_recompiles_relative_generic_plan_without_controller_host_defaults(tmp_path, monkeypatch):
    root = tmp_path / "experiment"
    recipe = write_finetune_recipe(root)
    recipe_payload = yaml.safe_load(recipe.read_text())
    recipe_payload["inputs"]["config"] = os.path.relpath(root / "config.yaml", REPO_ROOT)
    recipe.write_text(yaml.safe_dump(recipe_payload, sort_keys=False))
    plan_dir = root / "plans" / "train"
    assert plans.build_plan(recipe_path=recipe, output_dir=plan_dir).exit_code == 0
    before = workspace_files(root)

    monkeypatch.setattr(plan_contract, "REPO_ROOT", Path("/controller/repo"))
    monkeypatch.setattr(plan_hparam, "REPO_ROOT", Path("/controller/repo"))
    monkeypatch.setattr(plan_contract.sys, "executable", "/controller/bin/python")

    snapshot = experiments.experiment_status(root)

    assert snapshot["summary"]["state"] == "ready_to_launch"
    assert workspace_files(root) == before


def test_experiment_status_recompiles_relative_hparam_plan_without_controller_host_defaults(tmp_path, monkeypatch):
    root = tmp_path / "experiment"
    recipe = write_public_hparam_recipe(root, {"runtime.lr": [1e-6]})
    recipe_payload = yaml.safe_load(recipe.read_text())
    base_recipe = Path(recipe_payload["base_recipe"])
    base_payload = yaml.safe_load(base_recipe.read_text())
    base_payload["inputs"]["config"] = os.path.relpath(root / "config.yaml", REPO_ROOT)
    base_recipe.write_text(yaml.safe_dump(base_payload, sort_keys=False))
    plan_dir = root / "plans" / "tune"
    assert plans.build_plan(recipe_path=recipe, output_dir=plan_dir).exit_code == 0
    before = workspace_files(root)

    monkeypatch.setattr(plan_contract, "REPO_ROOT", Path("/controller/repo"))
    monkeypatch.setattr(plan_hparam, "REPO_ROOT", Path("/controller/repo"))
    monkeypatch.setattr(plan_contract.sys, "executable", "/controller/bin/python")

    snapshot = experiments.experiment_status(root)

    assert snapshot["summary"]["state"] == "ready_to_launch"
    assert workspace_files(root) == before
