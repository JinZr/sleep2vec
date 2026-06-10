from __future__ import annotations

from pathlib import Path

import yaml

from agent_tools.plans import evaluate_recipe
from agent_tools.recipes import load_recipe_with_base


def test_load_recipe_with_base_preserves_base_inputs():
    recipe = load_recipe_with_base("recipes/templates/hparam_tune_ppg_ahi.yaml")

    assert recipe["task"] == "hparam_tune"
    assert recipe["inputs"]["label_name"] == "ahi"
    assert recipe["_base_recipe"]["task"] == "finetune"


def test_load_recipe_with_base_resolves_relative_to_recipe_file(tmp_path: Path):
    base = tmp_path / "base.yaml"
    base.write_text(yaml.safe_dump({"task": "finetune", "inputs": {"label_name": "ahi"}}))
    nested = tmp_path / "nested"
    nested.mkdir()
    recipe_path = nested / "tune.yaml"
    recipe_path.write_text(yaml.safe_dump({"task": "hparam_tune", "base_recipe": "../base.yaml"}))

    recipe = load_recipe_with_base(recipe_path)

    assert recipe["_base_recipe"]["task"] == "finetune"
    assert recipe["inputs"]["label_name"] == "ahi"


def test_tiny_fixture_examples_pass_consultation_gates():
    for path in [
        "recipes/examples/tiny_fixture_finetune.yaml",
        "recipes/examples/tiny_fixture_hparam.yaml",
    ]:
        _recipe, _cfg, report = evaluate_recipe(path)
        assert report.exit_code == 0, path
