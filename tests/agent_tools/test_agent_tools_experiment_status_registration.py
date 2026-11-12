from __future__ import annotations

import json
from pathlib import Path

from agent_tool_test_helpers import (
    add_plan,
    init_status_workspace,
    read_manifest_rows,
    sha256_of,
    workspace_files,
    write_finetune_recipe,
    write_public_hparam_recipe,
)
import pytest
import yaml

from agent_tools import cli, experiment_io, experiment_workspace, experiments, plan_hparam, plans, run_artifacts

pytestmark = pytest.mark.usefixtures("stub_execution_target")


@pytest.mark.parametrize("allow_unresolved", [False, True])
def test_experiment_status_skips_registered_blocked_plan_after_successful_retry(tmp_path, allow_unresolved):
    root = tmp_path / "experiment"
    recipe = write_finetune_recipe(root, include_label=False)
    blocked_dir = root / "plans" / "blocked"

    blocked = plans.build_plan(
        recipe_path=recipe,
        output_dir=blocked_dir,
        allow_unresolved=allow_unresolved,
    )

    assert blocked.exit_code == 2
    blocked_path = blocked_dir / "plan.blocked.md"
    assert blocked_path.exists()
    assert (blocked_dir / "decisions.yaml").exists()
    assert not (blocked_dir / "plan.json").exists()
    assert not (blocked_dir / "recipe.resolved.yaml").exists()
    assert (blocked_dir / "plan.draft.json").exists() is allow_unresolved
    decisions_path = blocked_dir / "decisions.yaml"
    decisions = yaml.safe_load(decisions_path.read_text())
    decisions["decisions"]["label_name"]["value"] = "ahi"
    decisions_path.write_text(yaml.safe_dump(decisions, sort_keys=False))
    retry_dir = root / "plans" / "retry"
    retry = plans.build_plan(
        recipe_path=recipe,
        output_dir=retry_dir,
        user_decisions_path=decisions_path,
    )
    assert retry.exit_code == 0
    before = workspace_files(root)

    snapshot = experiments.experiment_status(root)

    assert snapshot["summary"]["state"] == "ready_to_launch"
    assert snapshot["steps"][0]["plans"] == [str(retry_dir)]
    assert workspace_files(root) == before


def test_experiment_status_skips_registered_blocked_plan_after_successful_plan(tmp_path):
    root = tmp_path / "experiment"
    recipe = write_finetune_recipe(root)
    successful_dir = root / "plans" / "successful"
    assert plans.build_plan(recipe_path=recipe, output_dir=successful_dir).exit_code == 0

    payload = yaml.safe_load(recipe.read_text())
    del payload["inputs"]["label_name"]
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
    blocked_dir = root / "plans" / "blocked"
    assert plans.build_plan(recipe_path=recipe, output_dir=blocked_dir).exit_code == 2
    before = workspace_files(root)

    snapshot = experiments.experiment_status(root)

    assert snapshot["summary"]["state"] == "ready_to_launch"
    assert snapshot["steps"][0]["plans"] == [str(successful_dir)]
    assert workspace_files(root) == before


def test_experiment_status_allows_later_ordinary_plan_from_new_recipe(tmp_path):
    root = tmp_path / "experiment"
    first_recipe = write_finetune_recipe(root)
    second_recipe = root / "second-recipe.yaml"
    second_payload = yaml.safe_load(first_recipe.read_text())
    second_payload["artifacts"]["version_name"] = "unit-second"
    second_recipe.write_text(yaml.safe_dump(second_payload, sort_keys=False))
    first_plan = root / "plans" / "first"
    second_plan = root / "plans" / "second"

    assert plans.build_plan(recipe_path=first_recipe, output_dir=first_plan).exit_code == 0
    assert plans.build_plan(recipe_path=second_recipe, output_dir=second_plan).exit_code == 0

    snapshot = experiments.experiment_status(root)

    assert snapshot["summary"]["state"] == "ready_to_launch"
    assert snapshot["steps"][0]["plans"] == [str(first_plan), str(second_plan)]
    assert snapshot["steps"][0]["status_counts"] == {"planned": 2}


def test_experiment_status_rejects_blocked_artifacts_beside_pass_plan(tmp_path):
    root = tmp_path / "experiment"
    recipe = write_finetune_recipe(root)
    plan_dir = root / "plans" / "train"
    assert plans.build_plan(recipe_path=recipe, output_dir=plan_dir).exit_code == 0
    (plan_dir / "plan.blocked.md").write_text("blocked\n")
    before = workspace_files(root)

    with pytest.raises(ValueError, match="both PASS and blocked"):
        experiments.experiment_status(root)
    assert workspace_files(root) == before


def test_experiment_status_rejects_user_decisions_beside_pass_plan(tmp_path):
    root = tmp_path / "experiment"
    recipe = write_finetune_recipe(root)
    plan_dir = root / "plans" / "train"
    assert plans.build_plan(recipe_path=recipe, output_dir=plan_dir).exit_code == 0
    (plan_dir / "decisions.yaml").write_text("decisions: {}\n")

    with pytest.raises(ValueError, match="both PASS and blocked"):
        experiments.experiment_status(root)


def test_experiment_status_accepts_historical_blocked_plan_without_user_decisions(tmp_path):
    root = tmp_path / "experiment"
    recipe = write_finetune_recipe(root, include_label=False)
    blocked_dir = root / "plans" / "blocked"
    assert plans.build_plan(recipe_path=recipe, output_dir=blocked_dir).exit_code == 2
    (blocked_dir / "decisions.yaml").unlink()

    snapshot = experiments.experiment_status(root)

    assert snapshot["summary"]["state"] == "empty"


@pytest.mark.parametrize("mutation", ["missing_questions_json", "launch_script"])
def test_experiment_status_rejects_partial_registered_blocked_plan(tmp_path, mutation):
    root = tmp_path / "experiment"
    recipe = write_finetune_recipe(root, include_label=False)
    blocked_dir = root / "plans" / "blocked"
    assert plans.build_plan(recipe_path=recipe, output_dir=blocked_dir).exit_code == 2

    if mutation == "missing_questions_json":
        (blocked_dir / "questions.json").unlink()
        error = "Managed file is missing"
    else:
        (blocked_dir / "run.sh").write_text("partial artifact\n")
        error = "directory entries differ"
    before = workspace_files(root)

    with pytest.raises(ValueError, match=error):
        experiments.experiment_status(root)
    assert workspace_files(root) == before


def test_experiment_status_rejects_step_metadata_drift_from_registered_plan(tmp_path):
    root = tmp_path / "experiment"
    recipe = write_finetune_recipe(root)
    recipe_payload = yaml.safe_load(recipe.read_text())
    plan_dir = root / "plans" / "train"
    assert plans.build_plan(recipe_path=recipe, output_dir=plan_dir).exit_code == 0
    before = workspace_files(root)

    snapshot = experiments.experiment_status(root)

    assert snapshot["summary"]["state"] == "ready_to_launch"
    assert workspace_files(root) == before
    step_path = root / "steps" / recipe_payload["step"]["id"] / "step.yaml"
    step_manifest = yaml.safe_load(step_path.read_text())
    step_manifest["step"]["purpose"] = "Different purpose."
    step_path.write_text(yaml.safe_dump(step_manifest, sort_keys=False))
    with pytest.raises(ValueError, match="step metadata differs"):
        experiments.experiment_status(root)


def test_experiment_status_rejects_unknown_registered_step_metadata(tmp_path):
    root = tmp_path / "experiment"
    init_status_workspace(root)
    add_plan(root, step_id="train")
    step_path = root / "steps" / "train" / "step.yaml"
    step_manifest = yaml.safe_load(step_path.read_text())
    step_manifest["step"]["purpsoe"] = "typo"
    step_path.write_text(yaml.safe_dump(step_manifest, sort_keys=False))

    with pytest.raises(ValueError, match="Unknown step field: purpsoe"):
        experiments.experiment_status(root)


@pytest.mark.parametrize("parameters", [{"runtime.lr": [1e-6]}, {"yaml:/data/finetune_preset_path": [None]}])
def test_experiment_status_accepts_public_layered_hparam_plan(tmp_path, parameters):
    root = tmp_path / "experiment"
    recipe = write_public_hparam_recipe(root, parameters)
    plan_dir = root / "plans" / "tune"

    assert plans.build_plan(recipe_path=recipe, output_dir=plan_dir).exit_code == 0
    plan = json.loads((plan_dir / "plan.json").read_text())
    resolved = yaml.safe_load((plan_dir / "recipe.resolved.yaml").read_text())
    assert "_base_recipe" in plan["recipe"] and "_local_recipe" in resolved
    parameter = next(iter(parameters))
    if parameters[parameter] == [None]:
        assert plan["runs"][0][parameter] is None
        assert read_manifest_rows(root)[0][parameter] == ""

    snapshot = experiments.experiment_status(root)

    assert snapshot["summary"]["state"] == "ready_to_launch"


@pytest.mark.parametrize("reverse_keys", [False, True])
def test_hparam_configurations_preserve_frozen_identity_across_json_roundtrip(tmp_path, reverse_keys):
    root = tmp_path / "experiment"
    configurations = [
        {"runtime.lr": 2e-6, "runtime.weight_decay": 0.1},
        {"runtime.lr": 1e-6, "runtime.weight_decay": 0.01},
    ]
    if reverse_keys:
        configurations = [dict(reversed(list(point.items()))) for point in configurations]
    recipe = write_public_hparam_recipe(root, {})
    payload = yaml.safe_load(recipe.read_text())
    payload["search"] = {"method": "grid", "max_runs": 2, "configurations": configurations}
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
    plan_dir = root / "plans" / "tune"

    assert plans.build_plan(recipe_path=recipe, output_dir=plan_dir).exit_code == 0
    plan = json.loads((plan_dir / "plan.json").read_text())
    runs = plan["runs"]
    assert [run["run_id"] for run in runs] == ["run-000", "run-001"]
    assert [run["run_name"] for run in runs] == ["lr-2e-6__weight-decay-0.1", "lr-1e-6__weight-decay-0.01"]
    assert [run["parameter_summary"] for run in runs] == [
        "runtime.lr=2e-06; runtime.weight_decay=0.1",
        "runtime.lr=1e-06; runtime.weight_decay=0.01",
    ]
    assert [{key: run[key] for key in point} for run, point in zip(runs, configurations)] == configurations
    canonical = read_manifest_rows(root)
    identity_fields = (
        *("experiment_id", "step_id", "run_id", "run_name", "parameter_summary", "version"),
        *("config", "config_sha256", "script", "script_sha256", "run_dir", "artifacts", "runtime_dir"),
        "checkpoint_dir",
        *sorted(experiment_workspace.SCHEDULER_PLAN_IDENTITY_FIELDS),
    )
    for run, row in zip(runs, canonical):
        for field in identity_fields:
            assert str(run.get(field) or "") == str(row.get(field) or "")
        assert run["config_sha256"] == sha256_of(Path(run["config"]))
        assert run["script_sha256"] == sha256_of(Path(run["script"]))
        assert run["command"] in Path(run["script"]).read_text()
    before = workspace_files(root)

    assert run_artifacts.read_hparam_plan(plan_dir)["runs"] == runs
    assert experiments.experiment_status(root)["summary"]["state"] == "ready_to_launch"
    assert workspace_files(root) == before


@pytest.mark.parametrize("max_runs", [3, 4])
def test_hparam_grid_key_order_does_not_change_run_indices_or_budget_subset(tmp_path, max_runs):
    expected = [
        {"runtime.lr": 2e-6, "runtime.weight_decay": 0.1},
        {"runtime.lr": 2e-6, "runtime.weight_decay": 0.01},
        {"runtime.lr": 1e-6, "runtime.weight_decay": 0.1},
        {"runtime.lr": 1e-6, "runtime.weight_decay": 0.01},
    ][:max_runs]
    parameters = {"runtime.weight_decay": [0.1, 0.01], "runtime.lr": [2e-6, 1e-6]}
    for index, mapping in enumerate((parameters, dict(reversed(list(parameters.items()))))):
        root = tmp_path / f"experiment-{index}"
        recipe = write_public_hparam_recipe(root, mapping, max_runs=max_runs)
        payload = yaml.safe_load(recipe.read_text())
        payload["search"]["parameters"] = mapping
        recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
        assert plan_hparam.hparam_combos(payload) == expected
        assert plan_hparam.hparam_combos(json.loads(json.dumps(payload, sort_keys=True))) == expected
        plan_dir = root / "plans" / "tune"

        assert plans.build_plan(recipe_path=recipe, output_dir=plan_dir).exit_code == 0
        runs = json.loads((plan_dir / "plan.json").read_text())["runs"]
        assert [run["run_id"] for run in runs] == [f"run-{run_index:03d}" for run_index in range(max_runs)]
        assert [{key: run[key] for key in parameters} for run in runs] == expected
        before = workspace_files(root)

        assert run_artifacts.read_hparam_plan(plan_dir)["runs"] == runs
        assert experiments.experiment_status(root)["summary"]["state"] == "ready_to_launch"
        assert workspace_files(root) == before


def test_experiment_status_requires_hparam_resolved_recipe_digest(tmp_path, capsys):
    root = tmp_path / "experiment"
    init_status_workspace(root)
    plan_dir, _canonical = add_plan(root, step_id="tune", task="hparam_tune")
    plan = json.loads((plan_dir / "plan.json").read_text())
    del plan["resolved_recipe_sha256"]
    (plan_dir / "plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="hparam recipe SHA-256"):
        experiments.experiment_status(root)
    assert cli.main(["experiment-status", "--run-dir", str(root), "--json"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "hparam-run-queue" not in captured.err


def test_experiment_status_allows_unrelated_blank_parameter_columns(tmp_path):
    root = tmp_path / "experiment"
    parameter = "yaml:/data/finetune_preset_path"
    recipe = write_public_hparam_recipe(root, {parameter: [None]})
    plan_dir = root / "plans" / "tune"
    assert plans.build_plan(recipe_path=recipe, output_dir=plan_dir).exit_code == 0
    add_plan(root, step_id="analyze", task="sleep2stat")

    generic_row = next(row for row in read_manifest_rows(root) if row["step_id"] == "analyze")
    assert generic_row[parameter] == ""

    snapshot = experiments.experiment_status(root)

    assert snapshot["summary"]["state"] == "ready_to_launch"
    assert snapshot["blockers"] == []


@pytest.mark.parametrize("drift", ["resolved_recipe", "plan_escape", "canonical_run"])
def test_experiment_status_rejects_registered_plan_binding_drift(tmp_path, drift):
    root = tmp_path / "experiment"
    init_status_workspace(root)
    plan_dir, _canonical = add_plan(root, step_id="train")

    if drift == "resolved_recipe":
        (plan_dir / "recipe.resolved.yaml").write_text("task: changed\n")
    elif drift == "plan_escape":
        step_path = root / "steps" / "train" / "step.yaml"
        step_manifest = yaml.safe_load(step_path.read_text())
        step_manifest["plans"] = [str(tmp_path / "outside")]
        step_path.write_text(yaml.safe_dump(step_manifest, sort_keys=False))
    else:
        (root / "run_manifest.tsv").write_text("step_id\trun_id\n")

    with pytest.raises(ValueError):
        experiments.experiment_status(root)


def test_experiment_status_rejects_plan_escape_before_external_probe(tmp_path, monkeypatch):
    root = tmp_path / "experiment"
    init_status_workspace(root)
    add_plan(root, step_id="train")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "plan.blocked.md").write_text("blocked\n")
    step_path = root / "steps" / "train" / "step.yaml"
    step_manifest = yaml.safe_load(step_path.read_text())
    step_manifest["plans"] = [str(outside)]
    step_path.write_text(yaml.safe_dump(step_manifest, sort_keys=False))
    path_exists_at = experiment_io.path_exists_at

    def reject_external_probe(path):
        try:
            Path(path).relative_to(root)
        except ValueError as exc:
            raise AssertionError(f"status probed outside the canonical workspace: {path}") from exc
        return path_exists_at(path)

    monkeypatch.setattr(experiment_io, "path_exists_at", reject_external_probe)

    with pytest.raises(ValueError, match="outside its managed workspace"):
        experiments.experiment_status(root)


@pytest.mark.parametrize("field", ["title", "objective", "baseline"])
def test_experiment_status_rejects_coherent_registered_plan_experiment_drift(tmp_path, field):
    root = tmp_path / "experiment"
    init_status_workspace(root)
    plan_dir, _canonical = add_plan(root, step_id="train")

    plan_path = plan_dir / "plan.json"
    plan = json.loads(plan_path.read_text())
    plan["recipe"]["experiment"][field] = f"foreign {field}"
    resolved_path = plan_dir / "recipe.resolved.yaml"
    resolved = yaml.safe_load(resolved_path.read_text())
    resolved["experiment"][field] = f"foreign {field}"
    resolved_path.write_text(yaml.safe_dump(resolved, sort_keys=False))
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="experiment metadata differs from the managed workspace"):
        experiments.experiment_status(root)


def test_experiment_status_rejects_plan_registered_by_multiple_steps(tmp_path):
    root = tmp_path / "experiment"
    init_status_workspace(root)
    plan_dir, _canonical = add_plan(root, step_id="train")
    second_step = root / "steps" / "evaluate"
    second_step.mkdir(parents=True)
    (second_step / "step.yaml").write_text(
        yaml.safe_dump(
            {
                "step": {"id": "evaluate", "phase": "evaluate", "purpose": "Evaluate."},
                "experiment_id": "status-unit",
                "recipe_path": "",
                "plans": [str(plan_dir)],
            },
            sort_keys=False,
        )
    )

    with pytest.raises(ValueError, match="more than one managed step"):
        experiments.experiment_status(root)
