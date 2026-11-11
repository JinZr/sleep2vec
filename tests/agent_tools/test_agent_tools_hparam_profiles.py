from __future__ import annotations

import copy
import json
from pathlib import Path

from agent_tool_test_helpers import hparam_search_defaults, run_execution_preflight_fixture
import pytest
import yaml

from agent_tools import execution_snapshot, plans
from agent_tools.decision_hparam import hparam_search_issues
from agent_tools.models import REPO_ROOT
from agent_tools.plan_hparam import apply_search_overrides, validate_finetune_config_bytes


@pytest.fixture(autouse=True)
def _stub_execution_target(monkeypatch):
    monkeypatch.setattr(execution_snapshot, "run_execution_command", run_execution_preflight_fixture)


def _search_recipe(tmp_path: Path) -> tuple[Path, Path]:
    inputs_dir = tmp_path / "inputs"
    inputs_dir.mkdir()
    index = inputs_dir / "index.csv"
    index.write_text("path,split,duration,ppg_mask,ah_event_mask,stage_mask\nx.npz,train,60,1,1,1\n")
    config_payload = yaml.safe_load((REPO_ROOT / "configs/ppg_ahi_finetune_large.yaml").read_text())
    config_payload["data"]["finetune_data_index"] = str(index)
    config = inputs_dir / "config.yaml"
    config.write_text(yaml.safe_dump(config_payload, sort_keys=False))

    workspace = tmp_path / "workspace"
    experiment = {
        "id": "explicit-search",
        "title": "Explicit search",
        "objective": "Exercise an explicit bounded search.",
        "root": str(workspace),
        "baseline": {"type": "none", "rationale": "unit fixture"},
    }
    base = yaml.safe_load((REPO_ROOT / "recipes/templates/finetune_ppg_ahi.yaml").read_text())
    base["experiment"] = experiment
    base["inputs"]["config"] = str(config)
    base_path = inputs_dir / "base.yaml"
    base_path.write_text(yaml.safe_dump(base, sort_keys=False))

    recipe = yaml.safe_load((REPO_ROOT / "recipes/templates/hparam_tune_ppg_ahi.yaml").read_text())
    recipe["search"] = {"method": "grid", "max_runs": 1, "parameters": {"runtime.lr": [1.0e-6]}}
    recipe["experiment"] = experiment
    recipe["step"] = {"id": "auto-tune", "phase": "train", "purpose": "Select a bounded candidate."}
    recipe["base_recipe"] = str(base_path)
    recipe["evaluation_policy"].update(
        {
            "selection_metric": "val_ahi_pearson",
            "selection_split": "val",
            "external_test_locked": True,
            "test_after_fit": False,
            "final_test_unlocked": False,
        }
    )
    recipe["decisions"].update(
        {
            "external_test_locked": {"value": True, "source": "explicit_recipe"},
            "train_val_test_policy": {"value": "val", "source": "explicit_recipe"},
            "test_after_fit": {"value": False, "source": "explicit_recipe"},
        }
    )
    recipe_path = inputs_dir / "search.yaml"
    recipe_path.write_text(yaml.safe_dump(recipe, sort_keys=False))
    return recipe_path, workspace


def test_explicit_user_search_replaces_the_authored_search_space(tmp_path: Path):
    authored_search = {"method": "grid", "max_runs": 3, "configurations": [{"runtime.lr": 1.0e-6}]}
    recipe_path, workspace = _search_recipe(tmp_path)
    recipe = yaml.safe_load(recipe_path.read_text())
    recipe["search"] = authored_search
    recipe_path.write_text(yaml.safe_dump(recipe, sort_keys=False))
    recipe_bytes = recipe_path.read_bytes()

    control, _summary, control_report = plans.evaluate_recipe(recipe_path)

    assert control_report.exit_code == 0
    assert control["search"] == authored_search
    assert recipe_path.read_bytes() == recipe_bytes

    decisions_path = tmp_path / "decisions.yaml"
    decisions_path.write_text(
        yaml.safe_dump(
            {
                "decisions": {
                    "hparam_search_space": {
                        "value": {"runtime.lr": [2.0e-6]},
                        "source": "explicit_user",
                    },
                }
            },
            sort_keys=False,
        )
    )
    decision_bytes = decisions_path.read_bytes()
    expected_search = {
        "method": "grid",
        "max_runs": authored_search["max_runs"],
        "parameters": {"runtime.lr": [2.0e-6]},
    }

    effective, _summary, doctor = plans.evaluate_recipe(recipe_path, decisions_path)

    assert doctor.exit_code == 0
    assert effective["search"] == expected_search

    plan_dir = workspace / "plans" / "user-search"
    report = plans.build_plan(recipe_path=recipe_path, output_dir=plan_dir, user_decisions_path=decisions_path)

    assert report.exit_code == 0
    plan = json.loads((plan_dir / "plan.json").read_text())
    resolved = yaml.safe_load((plan_dir / "recipe.resolved.yaml").read_text())
    assert plan["recipe"]["search"] == expected_search
    assert resolved["search"] == expected_search
    assert len(plan["runs"]) == 1
    assert plan["runs"][0]["runtime.lr"] == 2.0e-6
    assert recipe_path.read_bytes() == recipe_bytes
    assert decisions_path.read_bytes() == decision_bytes


def test_unresolved_user_search_preserves_the_authored_search_space(tmp_path: Path):
    authored_search = {"method": "grid", "max_runs": 3, "configurations": [{"runtime.lr": 1.0e-6}]}
    recipe_path, workspace = _search_recipe(tmp_path)
    recipe = yaml.safe_load(recipe_path.read_text())
    recipe["search"] = authored_search
    recipe_path.write_text(yaml.safe_dump(recipe, sort_keys=False))
    recipe_bytes = recipe_path.read_bytes()
    control, _summary, control_report = plans.evaluate_recipe(recipe_path)
    decisions_path = tmp_path / "decisions.yaml"
    decisions_path.write_text(
        yaml.safe_dump(
            {"decisions": {"hparam_search_space": {"value": "ASK_USER", "source": "explicit_user"}}},
            sort_keys=False,
        )
    )
    decision_bytes = decisions_path.read_bytes()

    effective, _summary, doctor = plans.evaluate_recipe(recipe_path, decisions_path)

    assert control_report.exit_code == 0
    assert doctor.exit_code == 2
    assert effective["search"] == control["search"]

    plan_dir = workspace / "plans" / "unresolved-search"
    report = plans.build_plan(recipe_path=recipe_path, output_dir=plan_dir, user_decisions_path=decisions_path)

    assert report.exit_code == 2
    assert not (plan_dir / "plan.json").exists()
    assert not (plan_dir / "runs").exists()
    assert not (plan_dir / "run_all.sh").exists()
    assert recipe_path.read_bytes() == recipe_bytes
    assert decisions_path.read_bytes() == decision_bytes


def test_tiny_fixture_keeps_legacy_explicit_search():
    recipe, _summary, report = plans.evaluate_recipe(REPO_ROOT / "recipes/examples/tiny_fixture_hparam.yaml")

    assert report.exit_code == 0
    assert "profile" not in recipe["search"]
    assert recipe["search"]["parameters"] == {"runtime.lr": [1.0e-6]}


@pytest.mark.parametrize(
    "template_name",
    ["hparam_tune_ppg_ahi.yaml", "hparam_tune_sleep2vec2_ppg_ahi.yaml", "hparam_tune_sleep2expert_ahi.yaml"],
)
def test_checked_in_templates_use_explicit_configurations(template_name):
    template_dir = REPO_ROOT / "recipes/templates"
    template = yaml.safe_load((template_dir / template_name).read_text())
    search = template["search"]
    points = search["configurations"]

    assert "profile" not in search
    assert "parameters" not in search
    assert search["method"] == "grid"
    assert search["max_runs"] == len(points) <= hparam_search_defaults()["max_runs_total"]
    assert not hparam_search_issues(template, high_impact={})
    base = yaml.safe_load((template_dir / template["base_recipe"]).read_text())
    source = REPO_ROOT / base["inputs"]["config"]
    payload = yaml.safe_load(source.read_text())
    for point in points:
        candidate = copy.deepcopy(payload)
        apply_search_overrides(candidate, point)
        validate_finetune_config_bytes({"variant": template["variant"]}, yaml.safe_dump(candidate).encode())

    if template["variant"] in {"sleep2vec", "sleep2vec2"}:
        assert template["evaluation_policy"] == {
            "selection_metric": "val_ahi_pearson",
            "selection_mode": "max",
            "selection_split": "val",
            "final_eval_split": "test",
            "external_test_locked": True,
            "test_after_fit": False,
            "require_manual_unlock_for_final_test": True,
            "final_test_unlocked": False,
        }
        assert template["decisions"]["external_test_locked"]["value"] is True
        assert template["decisions"]["test_after_fit"]["value"] is False
        assert template["decisions"]["train_val_test_policy"]["value"] == "val"
    else:
        assert template["evaluation_policy"]["selection_split"] == "val"
        assert template["evaluation_policy"]["test_after_fit"] is True
        assert template["decisions"]["test_after_fit"]["value"] is True


def test_task_recipe_schema_hparam_skeleton_keeps_test_locked():
    text = (REPO_ROOT / "recipes/schemas/task_recipe.schema.md").read_text()
    block = text.split("```yaml\n", 1)[1].split("\n```", 1)[0]
    skeleton = yaml.safe_load(block)

    assert "profile" not in skeleton["search"]
    assert "adaptive" not in skeleton
    assert skeleton["search"]["max_runs"] == len(skeleton["search"]["configurations"])
    assert not hparam_search_issues(skeleton, high_impact={})
    policy = skeleton["evaluation_policy"]
    assert policy["selection_metric"] == "val_ahi_pearson"
    assert policy["selection_split"] == "val"
    assert policy["external_test_locked"] is True
    assert policy["test_after_fit"] is False
    assert policy["final_test_unlocked"] is False


def test_generated_config_validation_failure_precedes_workspace_mutation(tmp_path: Path, monkeypatch):
    recipe_path, workspace = _search_recipe(tmp_path)
    before = {
        str(path.relative_to(tmp_path)): path.read_bytes() for path in sorted(tmp_path.rglob("*")) if path.is_file()
    }

    def reject_generated_config(_recipe, _config_bytes):
        raise ValueError("generated config is invalid")

    monkeypatch.setattr("agent_tools.plan_hparam.validate_finetune_config_bytes", reject_generated_config)

    report = plans.build_plan(recipe_path=recipe_path, output_dir=workspace / "plans" / "auto")

    after = {
        str(path.relative_to(tmp_path)): path.read_bytes() for path in sorted(tmp_path.rglob("*")) if path.is_file()
    }
    lock_path = workspace.parent / f".{workspace.name}.plan-registration.lock"
    assert report.exit_code == 1
    assert lock_path.is_file() and not lock_path.is_symlink()
    assert after.pop(str(lock_path.relative_to(tmp_path))) == b""
    assert before == after
    assert not workspace.exists()
    issue = next(issue for issue in report.issues if issue.field == "hparam_search_space")
    assert issue.evidence["preflight_before_workspace"] is True
