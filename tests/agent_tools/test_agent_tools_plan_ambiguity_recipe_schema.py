from __future__ import annotations

import json
from pathlib import Path
from shlex import quote as shlex_quote

from agent_tool_test_helpers import write_finetune_recipe, write_yaml
import pytest
from test_agent_plan_blocks_on_ambiguity import _first_run, _hparam_recipe, _run
from test_agent_plan_blocks_on_ambiguity import _stub_execution_target  # noqa: F401
import yaml

from agent_tools import plan_context, plans
from agent_tools.run_artifacts import read_hparam_plan


def test_hparam_recipe_cannot_inherit_experiment_and_step_from_base(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    payload.pop("experiment")
    payload.pop("step")
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
    before = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}

    doctor = _run("doctor", "--recipe", str(recipe))
    plan = _run("plan", "--recipe", str(recipe), "--output-dir", str(tmp_path / "plan"))

    assert doctor.returncode == 2
    assert plan.returncode == 2
    assert "experiment" in doctor.stdout
    assert "experiment" in plan.stdout
    assert {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before


def test_effective_user_config_fails_before_workspace_mutation(tmp_path: Path):
    for case, config_text in (("missing", None), ("invalid", "model: not-a-mapping\n")):
        root = tmp_path / case
        recipe = write_finetune_recipe(root)
        selected_config = root / "selected.yaml"
        if config_text is not None:
            selected_config.write_text(config_text)
        decisions = root / "decisions.yaml"
        decisions.write_text(
            yaml.safe_dump({"decisions": {"config": {"value": str(selected_config), "source": "explicit_user"}}})
        )
        before = {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}

        result = _run(
            "plan",
            "--recipe",
            str(recipe),
            "--user-decisions",
            str(decisions),
            "--output-dir",
            str(root / "plan"),
        )

        assert result.returncode == 1
        assert "config" in result.stdout.lower()
        assert {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()} == before


def test_unresolved_effective_user_config_fails_before_workspace_mutation(tmp_path: Path):
    for case, value in (("null", None), ("empty", ""), ("ask", "ASK_USER")):
        root = tmp_path / case
        recipe = write_finetune_recipe(root)
        decisions = root / "decisions.yaml"
        decisions.write_text(yaml.safe_dump({"decisions": {"config": {"value": value, "source": "explicit_user"}}}))
        before = {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}

        result = _run(
            "plan",
            "--recipe",
            str(recipe),
            "--user-decisions",
            str(decisions),
            "--output-dir",
            str(root / "plan"),
        )

        assert result.returncode == 2
        assert "config" in result.stdout.lower()
        assert {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()} == before


def test_unresolved_hparam_user_config_fails_before_workspace_mutation(tmp_path: Path):
    for case, value in (("null", None), ("empty", ""), ("ask", "ASK_USER")):
        root = tmp_path / case
        recipe = _hparam_recipe(root)
        decisions = root / "decisions.yaml"
        decisions.write_text(yaml.safe_dump({"decisions": {"config": {"value": value, "source": "explicit_user"}}}))
        before = {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}

        result = _run(
            "plan",
            "--recipe",
            str(recipe),
            "--user-decisions",
            str(decisions),
            "--output-dir",
            str(root / "plan"),
        )

        assert result.returncode == 2
        assert "config" in result.stdout.lower()
        assert {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()} == before


def test_resolved_hparam_user_config_owns_consultation_and_snapshot(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path)
    recipe_payload = yaml.safe_load(recipe.read_text())
    base_recipe = yaml.safe_load(Path(recipe_payload["base_recipe"]).read_text())
    base_config = yaml.safe_load(Path(base_recipe["inputs"]["config"]).read_text())
    selected_config = tmp_path / "selected.yaml"
    base_config["data"]["max_tokens"] = 5
    selected_config.write_text(yaml.safe_dump(base_config, sort_keys=False))
    decisions = tmp_path / "decisions.yaml"
    decisions.write_text(
        yaml.safe_dump({"decisions": {"config": {"value": str(selected_config), "source": "explicit_user"}}})
    )

    result = _run(
        "plan",
        "--recipe",
        str(recipe),
        "--user-decisions",
        str(decisions),
        "--output-dir",
        str(tmp_path / "plan"),
    )

    assert result.returncode == 0, result.stderr or result.stdout
    run = _first_run(tmp_path / "plan")
    assert yaml.safe_load(Path(run["config"]).read_text())["data"]["max_tokens"] == 5
    plan = json.loads((tmp_path / "plan" / "plan.json").read_text())
    assert plan["recipe"]["inputs"]["config"] == str(selected_config)
    assert plan["recipe"]["_base_recipe"]["inputs"]["config"] != str(selected_config)


def test_hparam_user_decisions_freeze_one_effective_recipe(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    payload["evaluation_policy"]["selection_metric"] = "val_wrong"
    payload["decisions"]["selection_metric"] = {"value": "val_wrong", "source": "explicit_recipe"}
    write_yaml(recipe, payload)
    decisions = write_yaml(
        tmp_path / "decisions.yaml",
        {
            "decisions": {
                "selection_metric": {"value": "val_ahi_pearson", "source": "explicit_user"},
                "selection_mode": {"value": "max", "source": "explicit_user"},
                "train_val_test_policy": {"value": "val", "source": "explicit_user"},
                "hparam_search_space": {
                    "value": {"runtime.lr": [2e-6]},
                    "source": "explicit_user",
                },
                "hparam_budget": {"value": 1, "source": "explicit_user"},
            }
        },
    )
    plan_dir = tmp_path / "plan"

    result = _run(
        "plan",
        "--recipe",
        str(recipe),
        "--user-decisions",
        str(decisions),
        "--output-dir",
        str(plan_dir),
    )

    assert result.returncode == 0, result.stderr or result.stdout
    plan = json.loads((plan_dir / "plan.json").read_text())
    effective = plan["recipe"]
    resolved = yaml.safe_load((plan_dir / "recipe.resolved.yaml").read_text())
    assert effective["evaluation_policy"]["selection_metric"] == "val_ahi_pearson"
    assert effective["evaluation_policy"]["selection_split"] == "val"
    assert effective["_local_recipe"]["evaluation_policy"]["selection_metric"] == "val_wrong"
    assert effective["decisions"]["selection_metric"]["value"] == "val_ahi_pearson"
    assert effective["_local_recipe"]["decisions"]["selection_metric"]["value"] == "val_wrong"
    assert effective["search"] == {"method": "grid", "max_runs": 1, "parameters": {"runtime.lr": [2e-6]}}
    assert effective["_local_recipe"]["search"] != effective["search"]
    assert "--lr 2e-06" in plan["runs"][0]["command"]
    assert resolved == {key: value for key, value in effective.items() if key != "_recipe_path"}
    reloaded = read_hparam_plan(plan_dir)["recipe"]
    assert reloaded["evaluation_policy"]["selection_metric"] == "val_ahi_pearson"


def test_hparam_user_selection_metric_rechecks_config_monitor(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path)
    decisions = write_yaml(
        tmp_path / "decisions.yaml",
        {
            "decisions": {
                "selection_metric": {"value": "val_other", "source": "explicit_user"},
            }
        },
    )
    plan_dir = tmp_path / "plan"

    result = _run(
        "plan",
        "--recipe",
        str(recipe),
        "--user-decisions",
        str(decisions),
        "--output-dir",
        str(plan_dir),
    )

    assert result.returncode == 1
    assert "selection_metric decision differs" in result.stdout
    assert not (plan_dir / "runs").exists()


def test_resolved_hparam_user_config_rechecks_base_consultation(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path)
    recipe_payload = yaml.safe_load(recipe.read_text())
    base_recipe = yaml.safe_load(Path(recipe_payload["base_recipe"]).read_text())
    base_config = yaml.safe_load(Path(base_recipe["inputs"]["config"]).read_text())
    selected_config = tmp_path / "selected-mismatch.yaml"
    base_config["finetune"]["task"]["monitor"] = "val_other"
    selected_config.write_text(yaml.safe_dump(base_config, sort_keys=False))
    decisions = tmp_path / "decisions.yaml"
    decisions.write_text(
        yaml.safe_dump({"decisions": {"config": {"value": str(selected_config), "source": "explicit_user"}}})
    )

    result = _run(
        "plan",
        "--recipe",
        str(recipe),
        "--user-decisions",
        str(decisions),
        "--output-dir",
        str(tmp_path / "plan"),
    )

    assert result.returncode == 1
    assert "selection_metric decision differs" in result.stdout
    assert not (tmp_path / "plan").exists()
    assert not (tmp_path / "plan" / "runs").exists()


def test_missing_or_unsupported_task_without_workspace_returns_report(tmp_path: Path):
    for name, task, expected_returncode in (("missing", None, 2), ("unsupported", "unknown", 1)):
        root = tmp_path / name
        root.mkdir()
        payload = {"name": name, "variant": "sleep2vec", "inputs": {}}
        if task is not None:
            payload["task"] = task
        recipe = root / "recipe.yaml"
        recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
        before = {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}

        result = _run("plan", "--recipe", str(recipe), "--output-dir", str(root / "plan"))

        assert result.returncode == expected_returncode
        assert "task" in result.stdout.lower()
        assert not result.stderr
        assert {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()} == before


@pytest.mark.parametrize("value", [None, "", "ASK_USER", "false", "true", 0, 1, [], {}])
def test_plan_blocks_non_boolean_test_after_fit_decision(tmp_path: Path, value: object):
    recipe = write_finetune_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    payload["evaluation_policy"].pop("test_after_fit")
    write_yaml(recipe, payload)
    decisions = write_yaml(
        tmp_path / "decisions.yaml",
        {"decisions": {"test_after_fit": {"value": value, "source": "explicit_user"}}},
    )
    output_dir = tmp_path / "plan"

    result = _run("plan", "--recipe", str(recipe), "--user-decisions", str(decisions), "--output-dir", str(output_dir))

    assert result.returncode == 2
    assert "test_after_fit" in result.stdout
    assert "Traceback" not in result.stderr
    assert not (output_dir / "run.sh").exists()


@pytest.mark.parametrize("value", [None, "", "ASK_USER", "true"])
def test_unresolved_recipe_test_after_fit_does_not_load_test_index(tmp_path: Path, monkeypatch, value: object):
    recipe = write_finetune_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    payload["evaluation_policy"].pop("test_after_fit")
    payload["decisions"]["test_after_fit"] = {"value": value, "source": "explicit_recipe"}
    write_yaml(recipe, payload)
    observed_split_values = []
    real_index_summary = plan_context.index_summary

    def record_split_values(*args, **kwargs):
        observed_split_values.append(kwargs.get("split_values"))
        return real_index_summary(*args, **kwargs)

    monkeypatch.setattr(plan_context, "index_summary", record_split_values)

    report = plans.build_plan(recipe_path=recipe, output_dir=tmp_path / "plan")

    assert report.exit_code == 2
    assert any(issue.field == "test_after_fit" for issue in report.issues)
    assert observed_split_values
    assert all("test" not in splits for splits in observed_split_values if splits is not None)
    assert not (tmp_path / "plan" / "run.sh").exists()


@pytest.mark.parametrize("value", ["true", "false", 0, 1, [], {}])
def test_plan_rejects_non_boolean_external_test_lock(tmp_path: Path, value: object):
    source = tmp_path / "source"
    recipe = write_finetune_recipe(source)
    payload = yaml.safe_load(recipe.read_text())
    workspace = tmp_path / "workspace"
    payload["experiment"]["root"] = str(workspace)
    payload["evaluation_policy"]["external_test_locked"] = value
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
    output_dir = workspace / "plans" / "invalid"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 1
    assert "external_test_locked must be a YAML boolean" in result.stdout
    assert not workspace.exists()


@pytest.mark.parametrize("value", [None, "", "ASK_USER"])
def test_plan_blocks_unresolved_external_test_lock(tmp_path: Path, value: object):
    source = tmp_path / "source"
    recipe = write_finetune_recipe(source)
    payload = yaml.safe_load(recipe.read_text())
    workspace = tmp_path / "workspace"
    payload["experiment"]["root"] = str(workspace)
    payload["evaluation_policy"]["external_test_locked"] = value
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
    output_dir = workspace / "plans" / "unresolved"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 2
    assert "external_test_locked must be explicitly true or false" in result.stdout
    assert not workspace.exists()


def test_plan_rejects_non_boolean_external_test_lock_user_decision(tmp_path: Path):
    recipe = write_finetune_recipe(tmp_path)
    decisions = write_yaml(
        tmp_path / "decisions.yaml",
        {"decisions": {"external_test_locked": {"value": 1, "source": "explicit_user"}}},
    )
    output_dir = tmp_path / "plan"

    result = _run("plan", "--recipe", str(recipe), "--user-decisions", str(decisions), "--output-dir", str(output_dir))

    assert result.returncode == 1
    assert "external_test_locked must be a YAML boolean" in result.stdout
    assert not output_dir.exists()


@pytest.mark.parametrize("decision_owner", ["recipe", "user"])
@pytest.mark.parametrize("value", ["false", 0])
def test_plan_rejects_non_boolean_external_test_lock_with_missing_declared_source(
    tmp_path: Path, decision_owner: str, value: object
):
    source = tmp_path / "source"
    recipe = write_finetune_recipe(source)
    payload = yaml.safe_load(recipe.read_text())
    workspace = tmp_path / "workspace"
    payload["experiment"]["root"] = str(workspace)
    decisions = None
    if decision_owner == "recipe":
        payload["decisions"]["external_test_locked"] = {"value": value, "source": "missing"}
    else:
        decisions = write_yaml(
            tmp_path / "decisions.yaml",
            {"decisions": {"external_test_locked": {"value": value, "source": "missing"}}},
        )
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
    output_dir = workspace / "plans" / "invalid"
    args = ["plan", "--recipe", str(recipe), "--output-dir", str(output_dir)]
    if decisions is not None:
        args.extend(["--user-decisions", str(decisions)])

    result = _run(*args)

    assert result.returncode == 1
    assert "external_test_locked must be a YAML boolean" in result.stdout
    assert not workspace.exists()


@pytest.mark.parametrize("decision_owner", ["recipe", "user"])
@pytest.mark.parametrize("value", [None, "", "ASK_USER"])
def test_plan_reports_unresolved_external_test_lock_with_missing_declared_source_once(
    tmp_path: Path, decision_owner: str, value: object
):
    recipe = write_finetune_recipe(tmp_path)
    decisions = None
    if decision_owner == "recipe":
        payload = yaml.safe_load(recipe.read_text())
        payload["decisions"]["external_test_locked"] = {"value": value, "source": "missing"}
        recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
    else:
        decisions = write_yaml(
            tmp_path / "decisions.yaml",
            {"decisions": {"external_test_locked": {"value": value, "source": "missing"}}},
        )

    _, _, report = plans.evaluate_recipe(recipe, decisions)

    matching = [issue for issue in report.blocking_issues() if issue.field == "external_test_locked"]
    assert report.exit_code == 2
    assert len(matching) == 1


def test_plan_keeps_missing_external_test_lock_task_owned(tmp_path: Path):
    recipe = write_finetune_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    payload["evaluation_policy"].pop("external_test_locked")
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))

    _, _, report = plans.evaluate_recipe(recipe)

    matching = [issue for issue in report.blocking_issues() if issue.field == "external_test_locked"]
    assert report.exit_code == 2
    assert [issue.message for issue in matching] == ["external_test_locked must be explicit for finetune."]


@pytest.mark.parametrize("value", [None, "", "ASK_USER"])
def test_plan_reports_unresolved_external_test_lock_user_decision_once(tmp_path: Path, value: object):
    recipe = write_finetune_recipe(tmp_path)
    decisions = write_yaml(
        tmp_path / "decisions.yaml",
        {"decisions": {"external_test_locked": {"value": value, "source": "explicit_user"}}},
    )

    _, _, report = plans.evaluate_recipe(recipe, decisions)

    matching = [issue for issue in report.blocking_issues() if issue.field == "external_test_locked"]
    assert report.exit_code == 2
    assert len(matching) == 1


@pytest.mark.parametrize(
    ("field", "value", "config_field"),
    [
        ("data_backend", "kaldi", "data.backend"),
        ("selection_metric", "val_loss", "finetune.task.monitor"),
        ("selection_mode", "min", "finetune.task.monitor_mod"),
    ],
)
def test_plan_fails_when_decision_differs_from_runtime_config(
    tmp_path: Path,
    field: str,
    value: str,
    config_field: str,
):
    recipe = write_finetune_recipe(tmp_path)
    config = Path(yaml.safe_load(recipe.read_text())["inputs"]["config"])
    config_before = config.read_bytes()
    decisions = write_yaml(
        tmp_path / "decisions.yaml",
        {"decisions": {field: {"value": value, "source": "explicit_user"}}},
    )
    output_dir = tmp_path / "plan"

    result = _run("plan", "--recipe", str(recipe), "--user-decisions", str(decisions), "--output-dir", str(output_dir))

    assert result.returncode == 1
    assert f"{field} decision differs from config {config_field}" in result.stdout
    assert config.read_bytes() == config_before
    assert not (output_dir / "run.sh").exists()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("data_backend", "npz"),
        ("selection_metric", "val_ahi_pearson"),
        ("selection_mode", "max"),
    ],
)
def test_plan_allows_decision_matching_runtime_config(tmp_path: Path, field: str, value: str):
    recipe = write_finetune_recipe(tmp_path)
    config = Path(yaml.safe_load(recipe.read_text())["inputs"]["config"])
    config_before = config.read_bytes()
    decisions = write_yaml(
        tmp_path / "decisions.yaml",
        {"decisions": {field: {"value": value, "source": "explicit_user"}}},
    )
    output_dir = tmp_path / "plan"

    result = _run("plan", "--recipe", str(recipe), "--user-decisions", str(decisions), "--output-dir", str(output_dir))

    assert result.returncode == 0, result.stderr or result.stdout
    assert config.read_bytes() == config_before
    assert (output_dir / "run.sh").exists()


@pytest.mark.parametrize("value", [None, ""])
def test_plan_blocks_unresolved_recipe_label_before_workspace_mutation(tmp_path: Path, value: str | None):
    recipe = write_finetune_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    payload["inputs"]["label_name"] = "raw-label"
    payload["decisions"]["label_name"] = {"value": value, "source": "explicit_recipe"}
    write_yaml(recipe, payload)
    output_dir = tmp_path / "plan"
    before = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 2
    assert "label_name decision is unresolved" in result.stdout
    assert not output_dir.exists()
    assert {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before


def test_plan_materializes_recipe_decisions_before_rendering(tmp_path: Path):
    recipe = write_finetune_recipe(tmp_path)
    raw_pretrained = tmp_path / "raw-pretrained.ckpt"
    selected_pretrained = tmp_path / "selected-pretrained.ckpt"
    raw_resume = tmp_path / "raw-resume.ckpt"
    selected_resume = tmp_path / "selected-resume.ckpt"
    for path in (raw_pretrained, selected_pretrained, raw_resume, selected_resume):
        path.write_text("checkpoint")
    payload = yaml.safe_load(recipe.read_text())
    payload["inputs"].update(
        {
            "label_name": "wrong-label",
            "pretrained_backbone_path": str(raw_pretrained),
            "ckpt_path": str(raw_resume),
        }
    )
    payload["evaluation_policy"]["test_after_fit"] = True
    payload["decisions"].update(
        {
            "label_name": {"value": "ahi", "source": "explicit_recipe"},
            "pretrained_backbone_path": {
                "value": str(selected_pretrained),
                "source": "explicit_recipe",
            },
            "ckpt_path": {"value": str(selected_resume), "source": "explicit_recipe"},
            "test_after_fit": {"value": False, "source": "explicit_recipe"},
        }
    )
    write_yaml(recipe, payload)
    output_dir = tmp_path / "plan"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 0, result.stderr or result.stdout
    script = (output_dir / "run.sh").read_text()
    assert "--label-name ahi" in script
    assert f"--pretrained-backbone-path {shlex_quote(str(selected_pretrained))}" in script
    assert f"--ckpt-path {shlex_quote(str(selected_resume))}" in script
    assert "--no-test-after-fit" in script
    assert "wrong-label" not in script
    assert str(raw_pretrained) not in script
    assert str(raw_resume) not in script
    effective = json.loads((output_dir / "plan.json").read_text())["recipe"]
    assert effective["inputs"]["label_name"] == "ahi"
    assert effective["inputs"]["pretrained_backbone_path"] == str(selected_pretrained)
    assert effective["inputs"]["ckpt_path"] == str(selected_resume)
    assert effective["evaluation_policy"]["test_after_fit"] is False


def test_hparam_rejects_bare_search_parameter(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path, parameters={"lr": [1e-6]})
    output_dir = tmp_path / "plan"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 1
    assert not (output_dir / "run_all.sh").exists()


def test_hparam_rejects_non_positive_max_runs(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path, max_runs=0)
    output_dir = tmp_path / "plan"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 1
    assert "hparam_budget" in (output_dir / "questions.md").read_text()
    assert not (output_dir / "run_all.sh").exists()


def test_hparam_rejects_removed_max_trials_field(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    payload["search"]["max_trials"] = payload["search"].pop("max_runs")
    recipe.write_text(yaml.safe_dump(payload))

    result = _run("doctor", "--recipe", str(recipe))

    assert result.returncode == 1
    assert "search.max_trials is no longer supported" in result.stdout


def test_hparam_rejects_removed_adaptive_field_when_adaptive_is_disabled(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    payload["adaptive"] = {"enabled": False, "max_trials_total": 4}
    recipe.write_text(yaml.safe_dump(payload))

    result = _run("doctor", "--recipe", str(recipe))

    assert result.returncode == 1
    assert "adaptive.max_trials_total is no longer supported" in result.stdout


@pytest.mark.parametrize(
    ("section", "field"),
    [("execution", "max_concurent"), ("evaluation_policy", "selection_metic")],
)
def test_hparam_rejects_unknown_execution_and_evaluation_fields(
    tmp_path: Path,
    section: str,
    field: str,
):
    recipe = _hparam_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    payload.setdefault(section, {})[field] = 1
    recipe.write_text(yaml.safe_dump(payload))
    output_dir = tmp_path / "plan"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 1
    assert f"{section}.{field}" in result.stdout
    assert not output_dir.exists()


def test_hparam_yaml_parameter_updates_run_config(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path, parameters={"yaml:/finetune/task/output_dim": [31]})
    output_dir = tmp_path / "plan"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 0
    run_config = yaml.safe_load(Path(_first_run(output_dir)["config"]).read_text())
    assert run_config["finetune"]["task"]["output_dim"] == 31


@pytest.mark.parametrize(
    ("parameter", "section", "field", "value", "config_path"),
    [
        ("yaml:/data/backend", "inputs", "data_backend", "kaldi", ("data", "backend")),
        (
            "yaml:/finetune/task/monitor",
            "evaluation_policy",
            "selection_metric",
            "val_override",
            ("finetune", "task", "monitor"),
        ),
        (
            "yaml:/finetune/task/monitor_mod",
            "evaluation_policy",
            "selection_mode",
            "min",
            ("finetune", "task", "monitor_mod"),
        ),
    ],
)
def test_hparam_yaml_override_can_match_explicit_decision_when_base_differs(
    tmp_path: Path,
    parameter: str,
    section: str,
    field: str,
    value: str,
    config_path: tuple[str, ...],
):
    recipe = _hparam_recipe(tmp_path, parameters={parameter: [value]})
    payload = yaml.safe_load(recipe.read_text())
    payload[section][field] = value
    write_yaml(recipe, payload)
    output_dir = tmp_path / "plan"

    doctor = _run("doctor", "--recipe", str(recipe))
    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert doctor.returncode == 0, doctor.stderr or doctor.stdout
    assert result.returncode == 0, result.stderr or result.stdout
    run_config = yaml.safe_load(Path(_first_run(output_dir)["config"]).read_text())
    configured = run_config
    for key in config_path:
        configured = configured[key]
    assert configured == value


def test_hparam_yaml_backend_override_rejects_any_combo_conflicting_with_explicit_decision(tmp_path: Path):
    recipe = _hparam_recipe(
        tmp_path,
        parameters={"yaml:/data/backend": ["kaldi", "npz"]},
        max_runs=2,
    )
    payload = yaml.safe_load(recipe.read_text())
    payload["inputs"]["data_backend"] = "kaldi"
    write_yaml(recipe, payload)
    output_dir = tmp_path / "plan"

    doctor = _run("doctor", "--recipe", str(recipe))
    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert doctor.returncode == 1
    assert "data_backend decision differs from config data.backend after hparam YAML overrides" in doctor.stdout
    assert result.returncode == 1
    assert "data_backend decision differs from config data.backend after hparam YAML overrides" in result.stdout
    assert not output_dir.exists()


def test_hparam_backend_decision_must_match_base_without_yaml_override(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    payload["inputs"]["data_backend"] = "kaldi"
    write_yaml(recipe, payload)
    output_dir = tmp_path / "plan"

    doctor = _run("doctor", "--recipe", str(recipe))
    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert doctor.returncode == 1
    assert "data_backend decision differs from config data.backend after hparam YAML overrides" in doctor.stdout
    assert result.returncode == 1
    assert "data_backend decision differs from config data.backend after hparam YAML overrides" in result.stdout
    assert not output_dir.exists()


@pytest.mark.parametrize(
    ("parameter", "value", "field"),
    [
        ("yaml:/data/backend", "kaldi", "data_backend"),
        ("yaml:/finetune/task/monitor", "val_loss", "selection_metric"),
        ("yaml:/finetune/task/monitor_mod", "min", "selection_mode"),
    ],
)
def test_hparam_yaml_parameter_cannot_conflict_with_decision_contract(
    tmp_path: Path,
    parameter: str,
    value: str,
    field: str,
):
    recipe = _hparam_recipe(tmp_path, parameters={parameter: [value]})
    output_dir = tmp_path / "plan"
    manifest = tmp_path / "run_manifest.tsv"
    manifest_before = manifest.read_bytes() if manifest.exists() else None

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 1
    assert f"{field} decision differs from config" in result.stdout
    assert not output_dir.exists()
    if manifest_before is None:
        assert not manifest.exists()
    else:
        assert manifest.read_bytes() == manifest_before
    assert not (tmp_path / "events.jsonl").exists()
    assert not (tmp_path / "steps" / "unit-hparam-tune" / "step.yaml").exists()


@pytest.mark.parametrize(
    ("section", "key", "recipe_value", "parameter", "value", "expected_message"),
    [
        (
            "inputs",
            "data_backend",
            None,
            "yaml:/data/backend",
            "kaldi",
            "data_backend decision differs from config",
        ),
        (
            "evaluation_policy",
            "selection_metric",
            None,
            "runtime.lr",
            1e-6,
            "selection_metric must be a non-empty value",
        ),
        (
            "evaluation_policy",
            "selection_metric",
            "",
            "runtime.lr",
            1e-6,
            "selection_metric must be a non-empty value",
        ),
    ],
)
def test_hparam_yaml_parameter_handles_empty_recipe_semantics(
    tmp_path: Path,
    section: str,
    key: str,
    recipe_value: object,
    parameter: str,
    value: object,
    expected_message: str,
):
    recipe = _hparam_recipe(tmp_path, parameters={parameter: [value]})
    payload = yaml.safe_load(recipe.read_text())
    payload[section][key] = recipe_value
    write_yaml(recipe, payload)
    output_dir = tmp_path / "plan"
    manifest = tmp_path / "run_manifest.tsv"
    manifest_before = manifest.read_bytes() if manifest.exists() else None

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 1
    assert expected_message in result.stdout
    assert not output_dir.exists()
    if manifest_before is None:
        assert not manifest.exists()
    else:
        assert manifest.read_bytes() == manifest_before
    assert not (tmp_path / "events.jsonl").exists()
    assert not (tmp_path / "steps" / "unit-hparam-tune" / "step.yaml").exists()


def test_hparam_yaml_parameter_rejects_ask_user_backend_after_null_input(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path, parameters={"yaml:/data/backend": ["kaldi"]})
    payload = yaml.safe_load(recipe.read_text())
    payload["inputs"]["data_backend"] = None
    payload.setdefault("runtime", {})["data_backend"] = "ASK_USER"
    write_yaml(recipe, payload)
    output_dir = tmp_path / "plan"
    manifest = tmp_path / "run_manifest.tsv"
    manifest_before = manifest.read_bytes() if manifest.exists() else None

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 1
    assert "runtime.data_backend" in result.stdout
    assert not output_dir.exists()
    if manifest_before is None:
        assert not manifest.exists()
    else:
        assert manifest.read_bytes() == manifest_before
    assert not (tmp_path / "events.jsonl").exists()
    assert not (tmp_path / "steps" / "unit-hparam-tune" / "step.yaml").exists()


def test_hparam_yaml_parameter_rejects_negative_list_index(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path, parameters={"yaml:/model/channels/-1/input_dim": [9]})
    output_dir = tmp_path / "plan"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 1
    assert not (output_dir / "run_all.sh").exists()
