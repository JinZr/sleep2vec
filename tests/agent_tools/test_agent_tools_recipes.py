from __future__ import annotations

import datetime
from pathlib import Path
import re

import pytest
import yaml

from agent_tools import configs, execution_snapshot, plan_context
from agent_tools.decision_models import DecisionStatus
from agent_tools.experiment_workspace import experiment_metadata_issues
from agent_tools.models import resolve_repo_path, validate_json_value
from agent_tools.plans import evaluate_recipe
from agent_tools.recipes import load_recipe_with_base, load_yaml_file


def _recipe_case(path: str, exit_code: int, status: str, fields: set[str] | None = None):
    return pytest.param(path, exit_code, status, fields or set(), id=path)


RECIPE_CASES = [
    _recipe_case("recipes/examples/tiny_fixture_finetune.yaml", 0, "PASS"),
    _recipe_case("recipes/examples/tiny_fixture_hparam.yaml", 0, "PASS"),
    _recipe_case("recipes/examples/tiny_fixture_sleep2stat.yaml", 0, "PASS"),
    _recipe_case("recipes/templates/finetune_ppg_ahi.yaml", 2, "NEEDS_USER_INPUT", {"data_input"}),
    _recipe_case(
        "recipes/templates/finetune_ppg_cox.yaml",
        2,
        "NEEDS_USER_INPUT",
        {"data_input", "survival_sidecars"},
    ),
    _recipe_case(
        "recipes/templates/finetune_sex_age_cox.yaml",
        2,
        "NEEDS_USER_INPUT",
        {"data_input", "survival_sidecars"},
    ),
    _recipe_case(
        "recipes/templates/finetune_sex_age_multilabel.yaml",
        2,
        "NEEDS_USER_INPUT",
        {"data_input", "multilabel_sidecars"},
    ),
    _recipe_case(
        "recipes/templates/finetune_sleep2expert_ahi.yaml",
        2,
        "NEEDS_USER_INPUT",
        {"data_input"},
    ),
    _recipe_case(
        "recipes/templates/finetune_sleep2expert_cox.yaml",
        2,
        "NEEDS_USER_INPUT",
        {"data_input", "survival_sidecars"},
    ),
    _recipe_case(
        "recipes/templates/finetune_sleep2vec2_ppg_ahi.yaml",
        2,
        "NEEDS_USER_INPUT",
        {"data_input"},
    ),
    _recipe_case(
        "recipes/templates/finetune_sleep2vec2_ppg_cox.yaml",
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
        {"ckpt_path", "final_eval_unlock"},
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


def test_load_yaml_file_rejects_duplicate_keys(tmp_path: Path):
    path = tmp_path / "duplicate.yaml"
    path.write_text("task: finetune\ntask: infer\n")

    with pytest.raises(ValueError, match="duplicate key: task"):
        load_yaml_file(path)


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        ("{}\n", "non-empty mapping"),
        ("node: &node\n  child: *node\n", "recursive YAML alias"),
    ],
)
def test_load_yaml_file_rejects_empty_mapping_and_recursive_alias(tmp_path: Path, contents: str, message: str):
    path = tmp_path / "invalid.yaml"
    path.write_text(contents)

    with pytest.raises(ValueError, match=message):
        load_yaml_file(path)


def test_validate_json_value_reports_a_non_string_key_at_its_location():
    with pytest.raises(ValueError, match=r"^Recipe\.search\.configurations\[0\] contains a non-string object key\.$"):
        validate_json_value({"search": {"configurations": [{1: 0.1}]}}, "Recipe")


@pytest.mark.parametrize(("value", "type_name"), [(datetime.date(2026, 10, 8), "date"), ((0.1, 0.2), "tuple")])
def test_validate_json_value_rejects_values_outside_the_json_data_model(value, type_name: str):
    with pytest.raises(
        ValueError, match=rf"^Recipe\.experiment\.title contains a non-JSON value of type {type_name}\.$"
    ):
        validate_json_value({"experiment": {"title": value}}, "Recipe")


def test_validate_json_value_allows_non_finite_floats_only_when_finite_is_false():
    document = {"runtime": {"gradient_clip_val": float("nan")}}

    validate_json_value(document, "Recipe", finite=False)
    with pytest.raises(ValueError, match=r"^Recipe\.runtime\.gradient_clip_val contains a non-finite number\.$"):
        validate_json_value(document, "Recipe")


def test_evaluate_recipe_rejects_a_non_string_key_at_the_freeze_check(tmp_path: Path):
    payload = load_yaml_file("recipes/examples/tiny_fixture_hparam.yaml")
    payload["base_recipe"] = str(Path("recipes/examples/tiny_fixture_finetune.yaml").resolve())
    payload["search"]["parameters"] = {1: [0.1]}
    path = tmp_path / "hparam.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False))

    message = (
        f"Cannot freeze recipe {path} as JSON: Recipe.search.parameters contains a non-string object key. "
        "Quote YAML dates/timestamps if a string was intended."
    )
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        evaluate_recipe(path)


def test_evaluate_recipe_rejects_an_unquoted_yaml_date_at_the_freeze_check(tmp_path: Path):
    payload = load_yaml_file("recipes/examples/tiny_fixture_finetune.yaml")
    payload["experiment"]["title"] = datetime.date(2026, 10, 8)
    path = tmp_path / "finetune.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    assert "  title: 2026-10-08\n" in path.read_text()

    message = (
        f"Cannot freeze recipe {path} as JSON: Recipe.experiment.title contains a non-JSON value of type date. "
        "Quote YAML dates/timestamps if a string was intended."
    )
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        evaluate_recipe(path)


def test_recipe_rejects_test_selection_split(tmp_path: Path):
    payload = load_yaml_file("recipes/examples/tiny_fixture_hparam.yaml")
    payload["base_recipe"] = str(Path("recipes/examples/tiny_fixture_finetune.yaml").resolve())
    payload["evaluation_policy"].update({"selection_metric": "test_ahi_pearson", "selection_split": "test"})
    path = tmp_path / "hparam.yaml"
    path.write_text(yaml.safe_dump(payload))

    _recipe, _cfg, report = evaluate_recipe(path)

    assert report.exit_code == 1
    issue = next(issue for issue in report.blocking_issues() if issue.field == "evaluation_policy.selection_split")
    assert "never a selection split" in issue.message
    assert issue.evidence["preflight_before_workspace"] is True


def test_recipe_rejects_search_profile(tmp_path: Path):
    payload = load_yaml_file("recipes/examples/tiny_fixture_hparam.yaml")
    payload["base_recipe"] = str(Path("recipes/examples/tiny_fixture_finetune.yaml").resolve())
    payload["search"] = {"method": "grid", "max_runs": 1, "profile": "balanced"}
    path = tmp_path / "hparam.yaml"
    path.write_text(yaml.safe_dump(payload))

    _recipe, _cfg, report = evaluate_recipe(path)

    assert report.exit_code == 1
    assert "search.profile" in {issue.field for issue in report.blocking_issues()}


def test_recipe_accepts_explicit_search_configurations(tmp_path: Path):
    payload = load_yaml_file("recipes/examples/tiny_fixture_hparam.yaml")
    payload["base_recipe"] = str(Path("recipes/examples/tiny_fixture_finetune.yaml").resolve())
    payload["search"] = {
        "method": "grid",
        "max_runs": 2,
        "configurations": [
            {"runtime.lr": 1.0e-6, "runtime.weight_decay": 1.0e-5},
            {"runtime.lr": 2.0e-6, "runtime.weight_decay": 1.0e-6},
        ],
    }
    path = tmp_path / "hparam.yaml"
    path.write_text(yaml.safe_dump(payload))

    _recipe, _cfg, report = evaluate_recipe(path)

    assert "hparam_search_space" not in {issue.field for issue in report.blocking_issues()}


def test_recipe_rejects_parameters_and_configurations_together(tmp_path: Path):
    payload = load_yaml_file("recipes/examples/tiny_fixture_hparam.yaml")
    payload["base_recipe"] = str(Path("recipes/examples/tiny_fixture_finetune.yaml").resolve())
    payload["search"]["configurations"] = [{"runtime.lr": 1.0e-6}]
    path = tmp_path / "hparam.yaml"
    path.write_text(yaml.safe_dump(payload))

    _recipe, _cfg, report = evaluate_recipe(path)

    assert report.exit_code == 1
    messages = [issue.message for issue in report.blocking_issues() if issue.field == "hparam_search_space"]
    assert any("mutually exclusive" in message for message in messages)


def test_recipe_rejects_empty_parameters_alongside_configurations(tmp_path: Path):
    # Mutual exclusion is a presence check: an empty parameters mapping must
    # not slip past the two-shapes contract just because it is falsy.
    payload = load_yaml_file("recipes/examples/tiny_fixture_hparam.yaml")
    payload["base_recipe"] = str(Path("recipes/examples/tiny_fixture_finetune.yaml").resolve())
    payload["search"]["parameters"] = {}
    payload["search"]["configurations"] = [{"runtime.lr": 1.0e-6}]
    path = tmp_path / "hparam.yaml"
    path.write_text(yaml.safe_dump(payload))

    _recipe, _cfg, report = evaluate_recipe(path)

    assert report.exit_code == 1
    messages = [issue.message for issue in report.blocking_issues() if issue.field == "hparam_search_space"]
    assert any("mutually exclusive" in message for message in messages)


@pytest.mark.parametrize(
    ("configurations", "expected_message_part"),
    [
        ([], "non-empty list"),
        ("not-a-list", "non-empty list"),
        ([[1, 2]], "non-empty mapping"),
        ([{}], "non-empty mapping"),
        ([{"lr": 1.0e-6}], "runtime.<name> or yaml:/"),
        ([{"runtime.not_allowed": 1}], "Unsupported runtime search parameter"),
    ],
)
def test_recipe_rejects_invalid_search_configurations(tmp_path: Path, configurations, expected_message_part: str):
    payload = load_yaml_file("recipes/examples/tiny_fixture_hparam.yaml")
    payload["base_recipe"] = str(Path("recipes/examples/tiny_fixture_finetune.yaml").resolve())
    payload["search"] = {"method": "grid", "max_runs": 1, "configurations": configurations}
    path = tmp_path / "hparam.yaml"
    path.write_text(yaml.safe_dump(payload))

    _recipe, _cfg, report = evaluate_recipe(path)

    assert report.exit_code == 1
    messages = [issue.message for issue in report.blocking_issues() if issue.field == "hparam_search_space"]
    assert any(expected_message_part in message for message in messages)


def test_recipe_rejects_removed_adaptive_block(tmp_path: Path):
    payload = load_yaml_file("recipes/examples/tiny_fixture_hparam.yaml")
    payload["base_recipe"] = str(Path("recipes/examples/tiny_fixture_finetune.yaml").resolve())
    payload["adaptive"] = {"enabled": True, "suggest": {"strategy": "agent_proposal"}, "round_size": 2, "max_rounds": 6}
    path = tmp_path / "hparam.yaml"
    path.write_text(yaml.safe_dump(payload))

    _recipe, _cfg, report = evaluate_recipe(path)

    assert report.exit_code == 1
    issue = next(issue for issue in report.blocking_issues() if issue.field == "adaptive")
    assert issue.message == "Unknown recipe field for task=hparam_tune: adaptive."


@pytest.mark.parametrize("value", [True, 1.5, "2"])
def test_hparam_budget_requires_an_exact_positive_integer(tmp_path: Path, value):
    payload = load_yaml_file("recipes/examples/tiny_fixture_hparam.yaml")
    payload["base_recipe"] = str(Path("recipes/examples/tiny_fixture_finetune.yaml").resolve())
    payload["search"]["max_runs"] = value
    path = tmp_path / "hparam.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False))

    _recipe, _cfg, report = evaluate_recipe(path)

    assert "hparam_budget" in {issue.field for issue in report.blocking_issues()}


def test_recipe_cases_cover_checked_in_examples_and_templates():
    expected = [
        str(path)
        for root in (Path("recipes/examples"), Path("recipes/templates"))
        for path in sorted(root.glob("*.yaml"))
    ]

    assert [case.values[0] for case in RECIPE_CASES] == expected


@pytest.mark.parametrize(("path", "expected_exit_code", "expected_status", "expected_issue_fields"), RECIPE_CASES)
def test_checked_in_recipe(
    path: str,
    expected_exit_code: int,
    expected_status: str,
    expected_issue_fields: set[str],
    tmp_path: Path,
    monkeypatch,
):
    recipe = load_recipe_with_base(path)
    if Path(path).parent == Path("recipes/templates"):
        expected_questions = experiment_metadata_issues(recipe.get("_local_recipe", recipe))
        config_path = resolve_repo_path((recipe.get("inputs") or {}).get("config"))
        original_open = Path.open

        def guarded_open(file_path, mode="r", *args, **kwargs):
            if file_path == config_path and "r" in mode:
                pytest.fail("Unresolved template ownership reached a config read")
            return original_open(file_path, mode, *args, **kwargs)

        with monkeypatch.context() as guarded:
            guarded.setattr(Path, "open", guarded_open)
            for module, name in (
                (configs, "load_config_summary_for_recipe"),
                (plan_context, "context_index_summary"),
            ):
                guarded.setattr(
                    module,
                    name,
                    lambda *_args, **_kwargs: pytest.fail("Unresolved template ownership reached data validation"),
                )
            guarded.setattr(
                execution_snapshot,
                "run_execution_command",
                lambda *_args, **_kwargs: pytest.fail("Unresolved template ownership reached runtime inspection"),
            )
            _recipe, cfg, report = evaluate_recipe(path)

        assert cfg is None
        assert report.exit_code == 2
        assert report.status == DecisionStatus.NEEDS_USER_INPUT
        assert [
            (issue.status.value, issue.field, issue.message, issue.question) for issue in report.blocking_issues()
        ] == [(issue["status"], issue["field"], issue["message"], issue["question"]) for issue in expected_questions]
        assert all(issue.evidence["preflight_before_workspace"] for issue in report.blocking_issues())

        # Resolve only ownership in a test copy so downstream template validation remains covered.
        payload = load_yaml_file(path)
        payload["experiment"] = {
            "id": "template-validation",
            "title": "Template validation",
            "objective": "Exercise the template after resolving workspace ownership.",
            "root": str(tmp_path),
            "baseline": {"type": "none", "rationale": "unit fixture"},
        }
        payload["step"] = {**payload["step"], "id": "validate-template", "purpose": "Validate the authored template."}
        if payload.get("base_recipe"):
            payload["base_recipe"] = str(resolve_repo_path(recipe["_base_recipe"]["_recipe_path"]))
        path = tmp_path / Path(path).name
        path.write_text(yaml.safe_dump(payload))

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
