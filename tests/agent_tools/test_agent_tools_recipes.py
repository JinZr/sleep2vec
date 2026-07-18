from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from agent_tools.decision_hparam import hparam_recipe_contract_issues
from agent_tools.decision_models import DecisionStatus
from agent_tools.plans import evaluate_recipe
from agent_tools.recipes import load_recipe_with_base, load_yaml_file


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


def test_recipe_rejects_unknown_runtime_field(tmp_path: Path):
    payload = load_yaml_file("recipes/examples/tiny_fixture_finetune.yaml")
    payload["runtime"]["lrr"] = payload["runtime"].pop("lr")
    path = tmp_path / "finetune.yaml"
    path.write_text(yaml.safe_dump(payload))

    _recipe, _cfg, report = evaluate_recipe(path)

    assert report.exit_code == 1
    assert "runtime.lrr" in {issue.field for issue in report.blocking_issues()}


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


@pytest.mark.parametrize(
    ("adaptive", "field"),
    [
        ({"enabled": False, "poll_second": 1}, "adaptive.poll_second"),
        (
            {"enabled": False, "replacement": {"grace_epoch": 1}},
            "adaptive.replacement.grace_epoch",
        ),
        (
            {"enabled": False, "suggest": {"stratgey": "best_neighborhood"}},
            "adaptive.suggest.stratgey",
        ),
    ],
)
def test_recipe_rejects_unknown_adaptive_fields(tmp_path: Path, adaptive: dict, field: str):
    payload = load_yaml_file("recipes/examples/tiny_fixture_hparam.yaml")
    payload["base_recipe"] = str(Path("recipes/examples/tiny_fixture_finetune.yaml").resolve())
    payload["adaptive"] = adaptive
    path = tmp_path / "hparam.yaml"
    path.write_text(yaml.safe_dump(payload))

    _recipe, _cfg, report = evaluate_recipe(path)

    assert report.exit_code == 1
    assert field in {issue.field for issue in report.blocking_issues()}


def _adaptive_contract_fields(*, parameters: dict, adaptive: dict) -> set[str]:
    issues = hparam_recipe_contract_issues(
        {"search": {"parameters": parameters}, "adaptive": adaptive},
        source_layer="effective",
    )
    return {issue.field for issue in issues}


def _adaptive_semantic_issue_fields(tmp_path: Path, adaptive: dict, *, max_runs=1) -> set[str]:
    payload = load_yaml_file("recipes/examples/tiny_fixture_hparam.yaml")
    payload["base_recipe"] = str(Path("recipes/examples/tiny_fixture_finetune.yaml").resolve())
    payload["search"]["max_runs"] = max_runs
    payload["adaptive"] = adaptive
    path = tmp_path / "hparam.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    _recipe, _cfg, report = evaluate_recipe(path)
    return {issue.field for issue in report.blocking_issues()}


_AGENT_PROPOSAL_REQUIRED_VALUES = {
    "objective_metric": "val_ahi_pearson",
    "objective_mode": "max",
    "round_size": 1,
    "max_rounds": 2,
    "max_runs_total": 4,
}


def _valid_adaptive_values() -> dict:
    return {
        "enabled": True,
        "objective_metric": "val_ahi_pearson",
        "objective_mode": "max",
        "max_rounds": 2,
        "max_runs_total": 4,
        "round_size": 1,
        "poll_seconds": 1,
        "replacement": {
            "enabled": True,
            "allow_running_stop": True,
            "grace_epochs": 1,
            "grace_minutes": 1,
            "kill_margin": 0.05,
        },
        "suggest": {"strategy": "best_neighborhood"},
    }


@pytest.mark.parametrize("value", [True, 1.5, "2"])
def test_hparam_budget_requires_an_exact_positive_integer(tmp_path: Path, value):
    fields = _adaptive_semantic_issue_fields(tmp_path, _valid_adaptive_values(), max_runs=value)

    assert "hparam_budget" in fields


@pytest.mark.parametrize("field", ["max_rounds", "max_runs_total", "round_size", "poll_seconds"])
@pytest.mark.parametrize("value", [True, 1.5, "2"])
def test_adaptive_integer_fields_require_exact_positive_integers(tmp_path: Path, field: str, value):
    adaptive = _valid_adaptive_values()
    adaptive[field] = value

    fields = _adaptive_semantic_issue_fields(tmp_path, adaptive)

    assert f"adaptive.{field}" in fields


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("enabled", "false"),
        ("test_feedback_for_selection", "false"),
    ],
)
def test_adaptive_flags_require_booleans(tmp_path: Path, field: str, value):
    adaptive = _valid_adaptive_values()
    adaptive[field] = value

    fields = _adaptive_semantic_issue_fields(tmp_path, adaptive)

    assert f"adaptive.{field}" in fields


@pytest.mark.parametrize("field", ["enabled", "allow_running_stop"])
def test_adaptive_replacement_flags_require_booleans(tmp_path: Path, field: str):
    adaptive = _valid_adaptive_values()
    adaptive["replacement"][field] = "false"

    fields = _adaptive_semantic_issue_fields(tmp_path, adaptive)

    assert f"adaptive.replacement.{field}" in fields


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("grace_epochs", -1),
        ("grace_minutes", "invalid"),
        ("kill_margin", float("nan")),
    ],
)
def test_adaptive_replacement_thresholds_must_be_finite_and_non_negative(tmp_path: Path, field: str, value):
    adaptive = _valid_adaptive_values()
    adaptive["replacement"][field] = value

    fields = _adaptive_semantic_issue_fields(tmp_path, adaptive)

    assert f"adaptive.replacement.{field}" in fields


@pytest.mark.parametrize("objective_mode", [["max"], {"value": "max"}])
def test_malformed_adaptive_objective_mode_returns_a_structured_issue(tmp_path: Path, objective_mode):
    adaptive = _valid_adaptive_values()
    adaptive["objective_mode"] = objective_mode

    fields = _adaptive_semantic_issue_fields(tmp_path, adaptive)

    assert "adaptive.objective_mode" in fields


def test_explicit_best_neighborhood_contract_keeps_existing_parameter_value_semantics():
    fields = _adaptive_contract_fields(
        parameters={"runtime.lr": [[1, 2], [3, 4]]},
        adaptive={"enabled": True, "suggest": {"strategy": "best_neighborhood"}},
    )

    assert not fields


def test_omitted_strategy_defaults_to_agent_proposal():
    fields = _adaptive_contract_fields(
        parameters={"runtime.lr": [1e-6, 2e-6]},
        adaptive={
            "enabled": True,
            **_AGENT_PROPOSAL_REQUIRED_VALUES,
            "suggest": {"bounds": {"runtime.lr": [5e-7, 1e-5]}},
        },
    )

    assert not fields


@pytest.mark.parametrize("adaptive", [{}, {"enabled": False}])
def test_inactive_adaptive_block_does_not_start_default_agent_proposal_contract(adaptive: dict):
    issues = hparam_recipe_contract_issues({"adaptive": adaptive}, source_layer="effective")

    assert not issues


def test_agent_proposal_contract_accepts_expanded_numeric_bounds_and_categories():
    fields = _adaptive_contract_fields(
        parameters={
            "runtime.lr": [1e-6, 2e-6],
            "runtime.batch_size": [8, 16],
            "yaml:/model/head/name": ["linear", "mlp"],
            "yaml:/model/use_bias": [True, False],
        },
        adaptive={
            "enabled": True,
            **_AGENT_PROPOSAL_REQUIRED_VALUES,
            "replacement": {"enabled": False},
            "suggest": {
                "strategy": "agent_proposal",
                "bounds": {"runtime.lr": [5e-7, 1e-5], "runtime.batch_size": [4, 64]},
            },
        },
    )

    assert not fields


def test_agent_proposal_contract_accepts_omitted_replacement_and_default_numeric_envelope():
    fields = _adaptive_contract_fields(
        parameters={"runtime.lr": [1e-6, 2e-6]},
        adaptive={
            "enabled": True,
            **_AGENT_PROPOSAL_REQUIRED_VALUES,
            "suggest": {"strategy": "agent_proposal"},
        },
    )

    assert not fields


@pytest.mark.parametrize("field", tuple(_AGENT_PROPOSAL_REQUIRED_VALUES))
@pytest.mark.parametrize("unresolved", ["missing", "null", "empty"])
def test_agent_proposal_requires_explicit_control_fields(field: str, unresolved: str):
    adaptive = {
        "enabled": True,
        **_AGENT_PROPOSAL_REQUIRED_VALUES,
        "suggest": {"strategy": "agent_proposal"},
    }
    if unresolved == "missing":
        adaptive.pop(field)
    else:
        adaptive[field] = None if unresolved == "null" else ""

    issues = hparam_recipe_contract_issues(
        {"search": {"parameters": {"runtime.lr": [1e-6, 2e-6]}}, "adaptive": adaptive},
        source_layer="effective",
    )
    issue = next(issue for issue in issues if issue.field == f"adaptive.{field}")

    assert issue.status == DecisionStatus.NEEDS_USER_INPUT
    assert issue.question
    assert issue.evidence["preflight_before_workspace"] is True


def test_agent_proposal_treats_blank_objective_metric_as_unresolved():
    adaptive = {
        "enabled": True,
        **_AGENT_PROPOSAL_REQUIRED_VALUES,
        "objective_metric": "   ",
        "suggest": {"strategy": "agent_proposal"},
    }

    issues = hparam_recipe_contract_issues(
        {"search": {"parameters": {"runtime.lr": [1e-6, 2e-6]}}, "adaptive": adaptive},
        source_layer="effective",
    )
    issue = next(issue for issue in issues if issue.field == "adaptive.objective_metric")

    assert issue.status == DecisionStatus.NEEDS_USER_INPUT
    assert issue.question
    assert issue.evidence["preflight_before_workspace"] is True


@pytest.mark.parametrize(
    "objective_metric",
    [
        pytest.param(0, id="zero"),
        pytest.param(False, id="false"),
        pytest.param([], id="list"),
        pytest.param({}, id="mapping"),
        pytest.param(1, id="integer"),
    ],
)
def test_agent_proposal_rejects_non_string_objective_metric(objective_metric):
    adaptive = {
        "enabled": True,
        **_AGENT_PROPOSAL_REQUIRED_VALUES,
        "objective_metric": objective_metric,
        "suggest": {"strategy": "agent_proposal"},
    }

    issues = hparam_recipe_contract_issues(
        {"search": {"parameters": {"runtime.lr": [1e-6, 2e-6]}}, "adaptive": adaptive},
        source_layer="effective",
    )
    issue = next(issue for issue in issues if issue.field == "adaptive.objective_metric")

    assert issue.status == DecisionStatus.FAIL
    assert issue.question is None
    assert issue.evidence["preflight_before_workspace"] is True


@pytest.mark.parametrize(
    ("adaptive", "field"),
    [
        ({"suggest": {"strategy": "unknown"}}, "adaptive.suggest.strategy"),
        ({"suggest": {"strategy": []}}, "adaptive.suggest.strategy"),
        (
            {"suggest": {"strategy": "best_neighborhood", "bounds": {"runtime.lr": [1e-7, 1e-5]}}},
            "adaptive.suggest.bounds",
        ),
    ],
)
def test_adaptive_suggest_strategy_and_bounds_are_closed(adaptive: dict, field: str):
    fields = _adaptive_contract_fields(parameters={"runtime.lr": [1e-6]}, adaptive=adaptive)

    assert field in fields


@pytest.mark.parametrize("replacement", [{}, {"enabled": True}, {"enabled": False, "grace_epochs": 1}, {"enabled": 0}])
def test_default_agent_proposal_requires_replacement_to_be_omitted_or_strictly_disabled(replacement: dict):
    fields = _adaptive_contract_fields(
        parameters={"runtime.lr": [1e-6]},
        adaptive={
            "enabled": True,
            **_AGENT_PROPOSAL_REQUIRED_VALUES,
            "replacement": replacement,
        },
    )

    assert "adaptive.replacement" in fields


@pytest.mark.parametrize(
    ("parameters", "bounds"),
    [
        ({"runtime.lr": [1e-6]}, {"runtime.weight_decay": [0.0, 1.0]}),
        ({"runtime.lr": [1e-6]}, {"runtime.lr": [1e-7]}),
        ({"runtime.lr": [1e-6]}, {"runtime.lr": [2e-6, 1e-6]}),
        ({"runtime.lr": [1e-6]}, {"runtime.lr": [1e-7, float("inf")]}),
        ({"runtime.lr": [1e-6]}, {"runtime.lr": [False, 1e-5]}),
        ({"runtime.batch_size": [8, 16]}, {"runtime.batch_size": [4.0, 64]}),
        ({"yaml:/model/head/name": ["linear", "mlp"]}, {"yaml:/model/head/name": [0, 1]}),
        ({"runtime.lr": [1e-6, "auto"]}, {}),
        ({"runtime.lr": [[1, 2], [3, 4]]}, {}),
    ],
)
def test_agent_proposal_rejects_invalid_parameter_envelopes(parameters: dict, bounds: dict):
    fields = _adaptive_contract_fields(
        parameters=parameters,
        adaptive={
            "enabled": True,
            **_AGENT_PROPOSAL_REQUIRED_VALUES,
            "suggest": {"strategy": "agent_proposal", "bounds": bounds},
        },
    )

    assert "adaptive.suggest.bounds" in fields


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
