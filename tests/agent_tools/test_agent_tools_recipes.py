from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from agent_tools.plans import evaluate_recipe
from agent_tools.recipes import load_recipe_with_base


def _recipe_case(path: str, exit_code: int, status: str, fields: set[str] | None = None):
    return pytest.param(path, exit_code, status, fields or set(), id=path)


RECIPE_CASES = [
    _recipe_case("recipes/examples/tiny_fixture_finetune.yaml", 0, "PASS"),
    _recipe_case("recipes/examples/tiny_fixture_hparam.yaml", 0, "PASS"),
    _recipe_case("recipes/examples/tiny_fixture_sleep2stat.yaml", 0, "PASS"),
    _recipe_case("recipes/templates/finetune_ppg_ahi_val_only.yaml", 2, "NEEDS_USER_INPUT", {"data_input"}),
    _recipe_case(
        "recipes/templates/finetune_ppg_cox_val_only.yaml",
        2,
        "NEEDS_USER_INPUT",
        {"data_input", "survival_sidecars"},
    ),
    _recipe_case(
        "recipes/templates/finetune_sex_age_cox_val_only.yaml",
        2,
        "NEEDS_USER_INPUT",
        {"data_input", "survival_sidecars"},
    ),
    _recipe_case(
        "recipes/templates/finetune_sex_age_multilabel_val_only.yaml",
        2,
        "NEEDS_USER_INPUT",
        {"data_input", "multilabel_sidecars"},
    ),
    _recipe_case(
        "recipes/templates/finetune_sleep2expert_ahi_val_only.yaml",
        2,
        "NEEDS_USER_INPUT",
        {"data_input"},
    ),
    _recipe_case(
        "recipes/templates/finetune_sleep2expert_cox_val_only.yaml",
        2,
        "NEEDS_USER_INPUT",
        {"data_input", "survival_sidecars"},
    ),
    _recipe_case(
        "recipes/templates/finetune_sleep2vec2_ppg_ahi_val_only.yaml",
        2,
        "NEEDS_USER_INPUT",
        {"data_input"},
    ),
    _recipe_case(
        "recipes/templates/finetune_sleep2vec2_ppg_cox_val_only.yaml",
        2,
        "NEEDS_USER_INPUT",
        {"data_input", "survival_sidecars"},
    ),
    _recipe_case(
        "recipes/templates/hparam_tune_ppg_ahi.yaml",
        2,
        "NEEDS_USER_INPUT",
        {"base_finetune.data_input", "config"},
    ),
    _recipe_case(
        "recipes/templates/hparam_tune_sleep2expert_ahi.yaml",
        2,
        "NEEDS_USER_INPUT",
        {"base_finetune.data_input", "config"},
    ),
    _recipe_case(
        "recipes/templates/hparam_tune_sleep2vec2_ppg_ahi.yaml",
        2,
        "NEEDS_USER_INPUT",
        {"base_finetune.data_input", "config"},
    ),
    _recipe_case(
        "recipes/templates/infer_ppg_ahi_external_test.yaml",
        2,
        "NEEDS_USER_INPUT",
        {"ckpt_path", "final_eval_unlock", "overwrite_policy"},
    ),
    _recipe_case(
        "recipes/templates/infer_sleep2expert_ahi_external_test.yaml",
        2,
        "NEEDS_USER_INPUT",
        {"ckpt_path", "final_eval_unlock"},
    ),
    _recipe_case(
        "recipes/templates/infer_sleep2vec2_ppg_ahi_external_test.yaml",
        2,
        "NEEDS_USER_INPUT",
        {"ckpt_path", "final_eval_unlock"},
    ),
    _recipe_case("recipes/templates/preset_ppg_ahi.yaml", 1, "FAIL", {"index"}),
    _recipe_case(
        "recipes/templates/sleep2stat_model_only.yaml",
        1,
        "FAIL",
        {"sleep2stat.data.index", "sleep2stat_config"},
    ),
    _recipe_case(
        "recipes/templates/sleep2stat_psg_yasa_microstructure.yaml",
        1,
        "FAIL",
        {"sleep2stat.data.index"},
    ),
    _recipe_case(
        "recipes/templates/sleep2stat_spo2_respiratory.yaml",
        1,
        "FAIL",
        {"sleep2stat.data.index"},
    ),
]


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


def test_recipe_cases_cover_checked_in_examples_and_templates():
    expected = [
        str(path)
        for root in (Path("recipes/examples"), Path("recipes/templates"))
        for path in sorted(root.glob("*.yaml"))
    ]

    assert [case.values[0] for case in RECIPE_CASES] == expected


@pytest.mark.parametrize(("path", "expected_exit_code", "expected_status", "expected_issue_fields"), RECIPE_CASES)
def test_checked_in_recipe(path: str, expected_exit_code: int, expected_status: str, expected_issue_fields: set[str]):
    recipe = load_recipe_with_base(path)
    _recipe, cfg, report = evaluate_recipe(path)

    assert report.exit_code == expected_exit_code
    assert report.status.value == expected_status
    issue_fields = {issue.field for issue in report.blocking_issues()}
    assert expected_issue_fields <= issue_fields
    if recipe.get("base_recipe"):
        assert "_base_recipe" in recipe
    if recipe.get("task") == "sleep2stat":
        assert cfg is not None
        assert cfg.get("is_sleep2stat") is True
