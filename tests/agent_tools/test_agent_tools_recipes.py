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


_AGENT_PROPOSAL_REQUIRED_VALUES = {
    "objective_metric": "val_ahi_pearson",
    "objective_mode": "max",
    "round_size": 1,
    "max_rounds": 2,
    "max_runs_total": 4,
}


@pytest.mark.parametrize("suggest", [{}, {"strategy": "best_neighborhood"}])
def test_best_neighborhood_contract_keeps_existing_parameter_value_semantics(suggest: dict):
    fields = _adaptive_contract_fields(
        parameters={"runtime.lr": [[1, 2], [3, 4]]},
        adaptive={"enabled": True, "suggest": suggest},
    )

    assert not fields


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
        (
            {"suggest": {"bounds": {"runtime.lr": [1e-7, 1e-5]}}},
            "adaptive.suggest.bounds",
        ),
    ],
)
def test_adaptive_suggest_strategy_and_bounds_are_closed(adaptive: dict, field: str):
    fields = _adaptive_contract_fields(parameters={"runtime.lr": [1e-6]}, adaptive=adaptive)

    assert field in fields


@pytest.mark.parametrize("replacement", [{}, {"enabled": True}, {"enabled": False, "grace_epochs": 1}, {"enabled": 0}])
def test_agent_proposal_requires_replacement_to_be_omitted_or_strictly_disabled(replacement: dict):
    fields = _adaptive_contract_fields(
        parameters={"runtime.lr": [1e-6]},
        adaptive={
            "enabled": True,
            **_AGENT_PROPOSAL_REQUIRED_VALUES,
            "replacement": replacement,
            "suggest": {"strategy": "agent_proposal"},
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
