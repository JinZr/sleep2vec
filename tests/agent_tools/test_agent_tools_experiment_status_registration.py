from __future__ import annotations

import json
from pathlib import Path

from agent_tool_test_helpers import write_finetune_recipe, write_yaml
import pytest
from test_agent_tools_experiment_status import (
    _add_plan,
    _init_workspace,
    _read_manifest_rows,
    _sha256,
    _workspace_files,
    _write_public_hparam_recipe,
)
from test_agent_tools_experiment_status import _stub_execution_target  # noqa: F401
import yaml

from agent_tools import cli, experiment_io, experiments, plans
from agent_tools.manifests import write_rows


@pytest.mark.parametrize(
    "context",
    [
        None,
        {"home": "/creator/home", "python": "/creator/python", "repo_root": "/creator/repo", "extra": "x"},
        {"home": "/creator/home", "python": "python", "repo_root": "/creator/repo"},
        {"home": "/creator/home", "python": "/creator/python", "repo_root": "relative/repo"},
    ],
)
def test_experiment_status_rejects_invalid_frozen_plan_context(tmp_path, context):
    root = tmp_path / "experiment"
    _init_workspace(root)
    plan_dir, _canonical = _add_plan(root, step_id="train")
    plan_path = plan_dir / "plan.json"
    plan = json.loads(plan_path.read_text())
    resolved_path = plan_dir / "recipe.resolved.yaml"
    resolved = yaml.safe_load(resolved_path.read_text())
    if context is None:
        plan["recipe"].pop("_plan_context")
        resolved.pop("_plan_context")
    else:
        plan["recipe"]["_plan_context"] = context
        resolved["_plan_context"] = context
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    resolved_path.write_text(yaml.safe_dump(resolved, sort_keys=False))

    with pytest.raises(ValueError, match="exact absolute _plan_context"):
        experiments.experiment_status(root)


def test_experiment_status_rejects_tampered_hparam_run_all_script(tmp_path):
    root = tmp_path / "experiment"
    _init_workspace(root)
    plan_dir, _canonical = _add_plan(root, step_id="tune", task="hparam_tune")
    (plan_dir / "run_all.sh").write_text("#!/usr/bin/env bash\nexit 0\n")

    with pytest.raises(ValueError, match="launch script differs from its frozen recipe"):
        experiments.experiment_status(root)


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
    assert not (blocked_dir / "plan.json").exists()
    assert not (blocked_dir / "recipe.resolved.yaml").exists()
    assert (blocked_dir / "plan.draft.json").exists() is allow_unresolved
    decisions_path = write_yaml(
        root / "decisions.yaml",
        {"decisions": {"label_name": {"value": "ahi", "source": "explicit_user"}}},
    )
    retry_dir = root / "plans" / "retry"
    retry = plans.build_plan(
        recipe_path=recipe,
        output_dir=retry_dir,
        user_decisions_path=decisions_path,
    )
    assert retry.exit_code == 0
    before = _workspace_files(root)

    snapshot = experiments.experiment_status(root)

    assert snapshot["summary"]["state"] == "ready_to_launch"
    assert snapshot["steps"][0]["plans"] == [str(retry_dir)]
    assert snapshot["decision"]["recommended_next"]["argv"] == ["bash", str(retry_dir / "run.sh")]
    assert _workspace_files(root) == before


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
    before = _workspace_files(root)

    snapshot = experiments.experiment_status(root)

    assert snapshot["summary"]["state"] == "ready_to_launch"
    assert snapshot["steps"][0]["plans"] == [str(successful_dir)]
    assert snapshot["decision"]["recommended_next"]["argv"] == ["bash", str(successful_dir / "run.sh")]
    assert _workspace_files(root) == before


def test_experiment_status_rejects_recipe_path_drift_on_ordinary_retry(tmp_path):
    root = tmp_path / "experiment"
    recipe = write_finetune_recipe(root, include_label=False)
    blocked_dir = root / "plans" / "blocked"
    assert plans.build_plan(recipe_path=recipe, output_dir=blocked_dir).exit_code == 2

    decisions_path = write_yaml(
        root / "decisions.yaml",
        {"decisions": {"label_name": {"value": "ahi", "source": "explicit_user"}}},
    )
    retry_dir = root / "plans" / "retry"
    assert plans.build_plan(recipe_path=recipe, output_dir=retry_dir, user_decisions_path=decisions_path).exit_code == 0
    plan_path = retry_dir / "plan.json"
    plan = json.loads(plan_path.read_text())
    plan["recipe"]["_recipe_path"] = str(root / "foreign-recipe.yaml")
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="recipe path differs from its managed step"):
        experiments.experiment_status(root)


@pytest.mark.parametrize("missing_from", ["step", "plan"])
def test_experiment_status_rejects_missing_registered_recipe_provenance(tmp_path, missing_from):
    root = tmp_path / "experiment"
    _init_workspace(root)
    plan_dir, _canonical = _add_plan(root, step_id="train")
    if missing_from == "step":
        step_path = root / "steps" / "train" / "step.yaml"
        step = yaml.safe_load(step_path.read_text())
        step["recipe_path"] = ""
        step_path.write_text(yaml.safe_dump(step, sort_keys=False))
        error = "recipe path differs from its managed step"
    else:
        plan_path = plan_dir / "plan.json"
        plan = json.loads(plan_path.read_text())
        del plan["recipe"]["_recipe_path"]
        plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
        error = "recipe path must be absolute"
    before = _workspace_files(root)

    with pytest.raises(ValueError, match=error):
        experiments.experiment_status(root)
    assert _workspace_files(root) == before


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
    assert snapshot["decision"]["manual_choice_required"] is True
    assert len(snapshot["decision"]["other_legal_actions"]) == 2


@pytest.mark.parametrize(("adaptive", "pipeline"), [(True, False), (False, True)])
def test_experiment_status_allows_controller_owned_recipe_path(tmp_path, adaptive, pipeline):
    root = tmp_path / "experiment"
    _init_workspace(root)
    plan_dir, _canonical = _add_plan(
        root,
        step_id="train",
        task="hparam_tune" if adaptive else "finetune",
        adaptive=adaptive,
        pipeline=pipeline,
    )
    plan_path = plan_dir / "plan.json"
    plan = json.loads(plan_path.read_text())
    plan["recipe"]["_recipe_path"] = str(root / "controller-recipe.yaml")
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")

    snapshot = experiments.experiment_status(root)

    assert snapshot["summary"]["state"] == "blocked"


def test_experiment_status_rejects_blocked_artifacts_beside_pass_plan(tmp_path):
    root = tmp_path / "experiment"
    recipe = write_finetune_recipe(root)
    plan_dir = root / "plans" / "train"
    assert plans.build_plan(recipe_path=recipe, output_dir=plan_dir).exit_code == 0
    (plan_dir / "plan.blocked.md").write_text("blocked\n")
    before = _workspace_files(root)

    with pytest.raises(ValueError, match="both PASS and blocked"):
        experiments.experiment_status(root)
    assert _workspace_files(root) == before


@pytest.mark.parametrize(
    "mutation",
    ["missing_questions_json", "missing_questions_md", "launch_script", "config", "runs"],
)
def test_experiment_status_rejects_partial_registered_blocked_plan(tmp_path, mutation):
    root = tmp_path / "experiment"
    recipe = write_finetune_recipe(root, include_label=False)
    blocked_dir = root / "plans" / "blocked"
    assert plans.build_plan(recipe_path=recipe, output_dir=blocked_dir).exit_code == 2

    if mutation.startswith("missing_questions"):
        suffix = ".json" if mutation.endswith("json") else ".md"
        (blocked_dir / f"questions{suffix}").unlink()
        error = "Managed file is missing"
    elif mutation == "runs":
        (blocked_dir / "runs").mkdir()
        error = "directory entries differ"
    else:
        name = "run.sh" if mutation == "launch_script" else "config.yaml"
        (blocked_dir / name).write_text("partial artifact\n")
        error = "directory entries differ"
    before = _workspace_files(root)

    with pytest.raises(ValueError, match=error):
        experiments.experiment_status(root)
    assert _workspace_files(root) == before


@pytest.mark.parametrize("alias_kind", ["symlink", "hardlink"])
def test_experiment_status_rejects_aliased_registered_blocked_plan(tmp_path, alias_kind):
    root = tmp_path / "experiment"
    recipe = write_finetune_recipe(root, include_label=False)
    blocked_dir = root / "plans" / "blocked"
    assert plans.build_plan(recipe_path=recipe, output_dir=blocked_dir).exit_code == 2
    blocked_path = blocked_dir / "plan.blocked.md"
    contents = blocked_path.read_bytes()
    blocked_path.unlink()
    outside = root / "outside-blocked.md"
    outside.write_bytes(contents)
    if alias_kind == "symlink":
        blocked_path.symlink_to(outside)
    else:
        blocked_path.hardlink_to(outside)

    with pytest.raises(ValueError, match="missing or aliased"):
        experiments.experiment_status(root)


def test_experiment_status_preserves_registered_step_io_metadata(tmp_path):
    root = tmp_path / "experiment"
    recipe = write_finetune_recipe(root)
    recipe_payload = yaml.safe_load(recipe.read_text())
    step_spec = write_yaml(
        root / "step-spec.yaml",
        {
            **recipe_payload["step"],
            "inputs": ["config.yaml"],
            "outputs": ["reports/ranking.csv"],
        },
    )
    experiments.register_experiment_step(root, step_spec)
    plan_dir = root / "plans" / "train"
    assert plans.build_plan(recipe_path=recipe, output_dir=plan_dir).exit_code == 0
    before = _workspace_files(root)

    snapshot = experiments.experiment_status(root)

    assert snapshot["summary"]["state"] == "ready_to_launch"
    assert _workspace_files(root) == before
    step_path = root / "steps" / recipe_payload["step"]["id"] / "step.yaml"
    step_manifest = yaml.safe_load(step_path.read_text())
    assert step_manifest["step"]["inputs"] == ["config.yaml"]
    assert step_manifest["step"]["outputs"] == ["reports/ranking.csv"]

    step_manifest["step"]["purpose"] = "Different purpose."
    step_path.write_text(yaml.safe_dump(step_manifest, sort_keys=False))
    with pytest.raises(ValueError, match="step metadata differs"):
        experiments.experiment_status(root)


def test_experiment_status_rejects_unknown_registered_step_metadata(tmp_path):
    root = tmp_path / "experiment"
    _init_workspace(root)
    _add_plan(root, step_id="train")
    step_path = root / "steps" / "train" / "step.yaml"
    step_manifest = yaml.safe_load(step_path.read_text())
    step_manifest["step"]["purpsoe"] = "typo"
    step_path.write_text(yaml.safe_dump(step_manifest, sort_keys=False))

    with pytest.raises(ValueError, match="Unknown step field: purpsoe"):
        experiments.experiment_status(root)


def test_experiment_status_rejects_supported_task_relabel_with_foreign_command(tmp_path):
    root = tmp_path / "experiment"
    _init_workspace(root)
    plan_dir, _canonical = _add_plan(root, step_id="train")

    plan = json.loads((plan_dir / "plan.json").read_text())
    plan["recipe"]["task"] = "infer"
    (plan_dir / "plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    resolved = yaml.safe_load((plan_dir / "recipe.resolved.yaml").read_text())
    resolved["task"] = "infer"
    (plan_dir / "recipe.resolved.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False))

    with pytest.raises(ValueError, match="Invalid registered plan recipe|frozen recipe"):
        experiments.experiment_status(root)


@pytest.mark.parametrize(
    ("task", "variant", "field"),
    [
        ("preset_prepare", "sex_age_baseline", "variant"),
        ("sleep2stat", "sleep2vec", "variant"),
    ],
)
def test_experiment_status_rejects_invalid_frozen_recipe_structure(tmp_path, capsys, task, variant, field):
    root = tmp_path / "experiment"
    _init_workspace(root)
    plan_dir, _canonical = _add_plan(root, step_id="step")
    plan = json.loads((plan_dir / "plan.json").read_text())
    plan["recipe"].update({"task": task, "variant": variant})
    (plan_dir / "plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    resolved = yaml.safe_load((plan_dir / "recipe.resolved.yaml").read_text())
    resolved.update({"task": task, "variant": variant})
    (plan_dir / "recipe.resolved.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False))

    assert cli.main(["experiment-status", "--run-dir", str(root), "--json"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert field in captured.err
    assert "Traceback" not in captured.err


def test_experiment_status_rejects_unknown_frozen_internal_recipe_field(tmp_path):
    root = tmp_path / "experiment"
    _init_workspace(root)
    plan_dir, _canonical = _add_plan(root, step_id="train")
    plan = json.loads((plan_dir / "plan.json").read_text())
    plan["recipe"]["_private"] = True
    (plan_dir / "plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    resolved = yaml.safe_load((plan_dir / "recipe.resolved.yaml").read_text())
    resolved["_private"] = True
    (plan_dir / "recipe.resolved.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False))

    with pytest.raises(ValueError, match="unsupported internal fields"):
        experiments.experiment_status(root)


@pytest.mark.parametrize(
    ("layer", "field", "value"),
    [("_base_recipe", "adaptive", {}), ("_local_recipe", "unknown", True)],
)
def test_experiment_status_rejects_layered_hparam_structure_drift(tmp_path, layer, field, value):
    root = tmp_path / "experiment"
    _init_workspace(root)
    plan_dir, _canonical = _add_plan(root, step_id="tune", task="hparam_tune")
    plan = json.loads((plan_dir / "plan.json").read_text())
    plan["recipe"][layer][field] = value
    resolved = yaml.safe_load((plan_dir / "recipe.resolved.yaml").read_text())
    resolved[layer][field] = value
    resolved_path = plan_dir / "recipe.resolved.yaml"
    resolved_path.write_text(yaml.safe_dump(resolved, sort_keys=False))
    plan["resolved_recipe_sha256"] = _sha256(resolved_path)
    (plan_dir / "plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="Invalid registered plan recipe"):
        experiments.experiment_status(root)


@pytest.mark.parametrize(
    ("layer", "binding"),
    [("_base_recipe", "experiment"), ("_local_recipe", "step")],
)
def test_experiment_status_rejects_layered_hparam_binding_drift(tmp_path, layer, binding):
    root = tmp_path / "experiment"
    _init_workspace(root)
    plan_dir, _canonical = _add_plan(root, step_id="tune", task="hparam_tune")
    plan_path = plan_dir / "plan.json"
    plan = json.loads(plan_path.read_text())
    plan["recipe"][layer][binding] = "invalid"
    resolved_path = plan_dir / "recipe.resolved.yaml"
    resolved = yaml.safe_load(resolved_path.read_text())
    resolved[layer][binding] = "invalid"
    resolved_path.write_text(yaml.safe_dump(resolved, sort_keys=False))
    plan["resolved_recipe_sha256"] = _sha256(resolved_path)
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    before = _workspace_files(root)

    with pytest.raises(ValueError, match=f"Invalid registered .* recipe binding: {binding} must be a mapping"):
        experiments.experiment_status(root)
    assert _workspace_files(root) == before


@pytest.mark.parametrize("section", [None, "execution"])
def test_experiment_status_rejects_layered_hparam_effective_structure_drift(tmp_path, section):
    root = tmp_path / "experiment"
    _init_workspace(root)
    plan_dir, _canonical = _add_plan(root, step_id="tune", task="hparam_tune")
    plan = json.loads((plan_dir / "plan.json").read_text())
    resolved = yaml.safe_load((plan_dir / "recipe.resolved.yaml").read_text())
    if section is None:
        plan["recipe"]["unknown"] = True
        resolved["unknown"] = True
    else:
        plan["recipe"].setdefault(section, {})["unknown"] = True
        resolved.setdefault(section, {})["unknown"] = True
    resolved_path = plan_dir / "recipe.resolved.yaml"
    resolved_path.write_text(yaml.safe_dump(resolved, sort_keys=False))
    plan["resolved_recipe_sha256"] = _sha256(resolved_path)
    (plan_dir / "plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="Invalid registered plan recipe"):
        experiments.experiment_status(root)


def test_experiment_status_rejects_finetune_plan_without_runtime_directories(tmp_path):
    root = tmp_path / "experiment"
    _init_workspace(root)
    plan_dir, canonical = _add_plan(root, step_id="train", task="finetune")
    plan = json.loads((plan_dir / "plan.json").read_text())
    for row in (plan["runs"][0], canonical):
        row["runtime_dir"] = ""
        row["checkpoint_dir"] = ""
    (plan_dir / "plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    write_rows(root / "run_manifest.tsv", [canonical])

    with pytest.raises(ValueError, match="runtime_dir, checkpoint_dir"):
        experiments.experiment_status(root)


def test_experiment_status_rejects_unsupported_registered_plan_task(tmp_path, capsys):
    root = tmp_path / "experiment"
    _init_workspace(root)
    plan_dir, canonical = _add_plan(root, step_id="train", task="finetune")
    plan = json.loads((plan_dir / "plan.json").read_text())
    plan["recipe"]["task"] = "unsupported_task"
    for row in (plan["runs"][0], canonical):
        row["runtime_dir"] = ""
        row["checkpoint_dir"] = ""
    (plan_dir / "plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    resolved = yaml.safe_load((plan_dir / "recipe.resolved.yaml").read_text())
    resolved["task"] = "unsupported_task"
    (plan_dir / "recipe.resolved.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False))
    write_rows(root / "run_manifest.tsv", [canonical])

    with pytest.raises(ValueError, match="(?i)unsupported.*task"):
        experiments.experiment_status(root)
    assert cli.main(["experiment-status", "--run-dir", str(root), "--json"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "run-plan" not in captured.err


@pytest.mark.parametrize("parameters", [{"runtime.lr": [1e-6]}, {"yaml:/data/finetune_preset_path": [None]}])
def test_experiment_status_accepts_public_layered_hparam_plan(tmp_path, parameters):
    root = tmp_path / "experiment"
    recipe = _write_public_hparam_recipe(root, parameters)
    plan_dir = root / "plans" / "tune"

    assert plans.build_plan(recipe_path=recipe, output_dir=plan_dir).exit_code == 0
    plan = json.loads((plan_dir / "plan.json").read_text())
    resolved = yaml.safe_load((plan_dir / "recipe.resolved.yaml").read_text())
    assert "_base_recipe" in plan["recipe"] and "_local_recipe" in resolved
    parameter = next(iter(parameters))
    if parameters[parameter] == [None]:
        assert plan["runs"][0][parameter] is None
        assert _read_manifest_rows(root)[0][parameter] == ""

    snapshot = experiments.experiment_status(root)

    assert snapshot["summary"]["state"] == "ready_to_launch"


def test_experiment_status_requires_hparam_resolved_recipe_digest(tmp_path, capsys):
    root = tmp_path / "experiment"
    _init_workspace(root)
    plan_dir, _canonical = _add_plan(root, step_id="tune", task="hparam_tune")
    plan = json.loads((plan_dir / "plan.json").read_text())
    del plan["resolved_recipe_sha256"]
    (plan_dir / "plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="hparam recipe SHA-256"):
        experiments.experiment_status(root)
    assert cli.main(["experiment-status", "--run-dir", str(root), "--json"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "hparam-run-queue" not in captured.err


@pytest.mark.parametrize("search_kind", ["grid", "configurations"])
def test_experiment_status_rejects_hparam_parameter_drift_shared_by_plan_and_canonical_rows(tmp_path, search_kind):
    root = tmp_path / "experiment"
    recipe = _write_public_hparam_recipe(root, {"runtime.lr": [1e-6, 2e-6]})
    if search_kind == "configurations":
        payload = yaml.safe_load(recipe.read_text())
        payload["search"] = {
            "method": "grid",
            "max_runs": 2,
            "configurations": [{"runtime.lr": 1e-6}, {"runtime.lr": 2e-6}],
        }
        recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
    else:
        payload = yaml.safe_load(recipe.read_text())
        payload["search"]["max_runs"] = 2
        recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
    plan_dir = root / "plans" / "tune"
    assert plans.build_plan(recipe_path=recipe, output_dir=plan_dir).exit_code == 0

    plan_path = plan_dir / "plan.json"
    plan = json.loads(plan_path.read_text())
    plan["runs"][0]["runtime.lr"] = 9e-6
    plan["runs"][0]["parameter_summary"] = "runtime.lr=9e-06"
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    rows = _read_manifest_rows(root)
    rows[0]["runtime.lr"] = "9e-06"
    rows[0]["parameter_summary"] = "runtime.lr=9e-06"
    write_rows(root / "run_manifest.tsv", rows)
    before = _workspace_files(root)

    with pytest.raises(ValueError, match="parameter|canonical"):
        experiments.experiment_status(root)
    assert _workspace_files(root) == before


def test_experiment_status_rejects_missing_declared_blank_hparam_key(tmp_path):
    root = tmp_path / "experiment"
    parameter = "yaml:/data/finetune_preset_path"
    recipe = _write_public_hparam_recipe(root, {parameter: [None]})
    plan_dir = root / "plans" / "tune"
    assert plans.build_plan(recipe_path=recipe, output_dir=plan_dir).exit_code == 0
    plan = json.loads((plan_dir / "plan.json").read_text())
    del plan["runs"][0][parameter]
    (plan_dir / "plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="Workspace run parameters differ from plan"):
        experiments.experiment_status(root)


def test_experiment_status_allows_unrelated_blank_parameter_columns(tmp_path):
    root = tmp_path / "experiment"
    parameter = "yaml:/data/finetune_preset_path"
    recipe = _write_public_hparam_recipe(root, {parameter: [None]})
    plan_dir = root / "plans" / "tune"
    assert plans.build_plan(recipe_path=recipe, output_dir=plan_dir).exit_code == 0
    _add_plan(root, step_id="analyze", task="sleep2stat")

    generic_row = next(row for row in _read_manifest_rows(root) if row["step_id"] == "analyze")
    assert generic_row[parameter] == ""

    snapshot = experiments.experiment_status(root)

    assert snapshot["summary"]["state"] == "ready_to_launch"
    assert snapshot["decision"]["manual_choice_required"] is True


@pytest.mark.parametrize(
    ("drift", "error"),
    [
        ("missing_member", "missing final_eval_config"),
        ("missing_bundle", "missing final_eval_config"),
        ("missing_sha256", "final_eval_config must define"),
        ("file_drift", "frozen file SHA-256 changed"),
        ("coherent_file_drift", "frozen recipe digest"),
        ("extra_script_command", "final external-test script differs"),
    ],
)
def test_experiment_status_requires_final_eval_config_integrity(tmp_path, drift, error):
    root = tmp_path / "experiment"
    recipe = _write_public_hparam_recipe(root, {"yaml:/finetune/task/output_dim": [31]})
    payload = yaml.safe_load(recipe.read_text())
    base_recipe = yaml.safe_load(Path(payload["base_recipe"]).read_text())
    final_config_path = root / "selected-final-config.yaml"
    final_config = yaml.safe_load(Path(base_recipe["inputs"]["config"]).read_text())
    final_config["model"]["head"].update({"channel_agg": {"name": "mean"}, "temporal_agg": {"name": "mean"}})
    final_config_path.write_text(yaml.safe_dump(final_config, sort_keys=False))
    checkpoint = root / "selected.ckpt"
    checkpoint.write_text("checkpoint\n")
    payload["inputs"].update({"ckpt_path": str(checkpoint), "final_eval_config_path": str(final_config_path)})
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
    plan_dir = root / "plans" / "tune"
    assert plans.build_plan(recipe_path=recipe, output_dir=plan_dir, unlock_final_test=True).exit_code == 0
    assert experiments.experiment_status(root)["summary"]["state"] == "ready_to_launch"

    plan_path = plan_dir / "plan.json"
    plan = json.loads(plan_path.read_text())
    frozen_config = Path(plan["final_eval_config"]["path"])
    if drift in {"missing_member", "missing_bundle"}:
        del plan["final_eval_config"]
        if drift == "missing_bundle":
            frozen_config.unlink()
            (plan_dir / "final_external_test.sh").unlink()
        plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    elif drift == "missing_sha256":
        del plan["final_eval_config"]["sha256"]
        plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    elif drift == "file_drift":
        frozen_config.write_text("model: changed\n")
    elif drift == "coherent_file_drift":
        frozen_config.write_text("model: changed\n")
        plan["final_eval_config"]["sha256"] = _sha256(frozen_config)
        plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    else:
        final_script = plan_dir / "final_external_test.sh"
        final_script.write_text(final_script.read_text() + "echo injected\n")

    before = _workspace_files(root)

    with pytest.raises(ValueError, match=error):
        experiments.experiment_status(root)
    assert _workspace_files(root) == before


def test_experiment_status_rejects_coherent_registered_command_drift(tmp_path):
    root = tmp_path / "experiment"
    recipe = _write_public_hparam_recipe(root, {"runtime.lr": [1e-6]})
    plan_dir = root / "plans" / "tune"
    assert plans.build_plan(recipe_path=recipe, output_dir=plan_dir).exit_code == 0

    plan_path = plan_dir / "plan.json"
    plan = json.loads(plan_path.read_text())
    run = plan["runs"][0]
    command = run["command"]
    tokens = command.split()
    lr_index = tokens.index("--lr") + 1
    changed_command = command.replace(f"--lr {tokens[lr_index]}", "--lr 9e-06", 1)
    assert changed_command != command
    script_path = Path(run["script"])
    script_path.write_text(script_path.read_text().replace(command, changed_command, 1))
    run["command"] = changed_command
    run["script_sha256"] = _sha256(script_path)
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    rows = _read_manifest_rows(root)
    rows[0]["script_sha256"] = run["script_sha256"]
    write_rows(root / "run_manifest.tsv", rows)
    before = _workspace_files(root)

    with pytest.raises(ValueError, match="command"):
        experiments.experiment_status(root)
    assert _workspace_files(root) == before


def test_experiment_status_rejects_coherent_generic_command_drift(tmp_path):
    root = tmp_path / "experiment"
    recipe = write_finetune_recipe(root)
    plan_dir = root / "plans" / "train"
    assert plans.build_plan(recipe_path=recipe, output_dir=plan_dir).exit_code == 0

    plan_path = plan_dir / "plan.json"
    plan = json.loads(plan_path.read_text())
    old_command = plan["commands"][0]
    tokens = old_command.split()
    label_index = tokens.index("--label-name") + 1
    changed_command = old_command.replace(f"--label-name {tokens[label_index]}", "--label-name changed", 1)
    assert changed_command != old_command
    run = plan["runs"][0]
    run["command"] = changed_command
    plan["commands"] = [changed_command]
    for script_path in (plan_dir / "run.sh", Path(run["script"])):
        script_path.write_text(script_path.read_text().replace(old_command, changed_command, 1))
    run["script_sha256"] = _sha256(Path(run["script"]))
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    rows = _read_manifest_rows(root)
    rows[0]["script_sha256"] = run["script_sha256"]
    write_rows(root / "run_manifest.tsv", rows)
    before = _workspace_files(root)

    with pytest.raises(ValueError, match="commands differ from its frozen recipe"):
        experiments.experiment_status(root)
    assert _workspace_files(root) == before


@pytest.mark.parametrize("task", ["finetune", "hparam_tune"])
def test_experiment_status_rejects_extra_frozen_script_command(tmp_path, task):
    root = tmp_path / "experiment"
    if task == "hparam_tune":
        recipe = _write_public_hparam_recipe(root, {"runtime.lr": [1e-6]})
        plan_dir = root / "plans" / "tune"
    else:
        recipe = write_finetune_recipe(root)
        plan_dir = root / "plans" / "train"
    assert plans.build_plan(recipe_path=recipe, output_dir=plan_dir).exit_code == 0

    plan_path = plan_dir / "plan.json"
    plan = json.loads(plan_path.read_text())
    run = plan["runs"][0]
    script_path = Path(run["script"])
    script_path.write_text(script_path.read_text() + "echo injected\n")
    if task == "finetune":
        (plan_dir / "run.sh").write_text(script_path.read_text())
    run["script_sha256"] = _sha256(script_path)
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    rows = _read_manifest_rows(root)
    rows[0]["script_sha256"] = run["script_sha256"]
    write_rows(root / "run_manifest.tsv", rows)

    with pytest.raises(
        ValueError,
        match="script differs from its frozen recipe|run.sh differs from its frozen recipe|script_sha256",
    ):
        experiments.experiment_status(root)


def test_experiment_status_rejects_missing_hparam_source_config(tmp_path):
    root = tmp_path / "experiment"
    recipe = _write_public_hparam_recipe(root, {"runtime.lr": [1e-6]})
    plan_dir = root / "plans" / "tune"
    assert plans.build_plan(recipe_path=recipe, output_dir=plan_dir).exit_code == 0
    (plan_dir / "config.source.yaml").unlink()
    before = _workspace_files(root)

    with pytest.raises(ValueError, match="config.source.yaml"):
        experiments.experiment_status(root)
    assert _workspace_files(root) == before


def test_experiment_status_rejects_coherent_hparam_config_drift(tmp_path):
    root = tmp_path / "experiment"
    recipe = _write_public_hparam_recipe(root, {"yaml:/finetune/task/output_dim": [31]})
    plan_dir = root / "plans" / "tune"
    assert plans.build_plan(recipe_path=recipe, output_dir=plan_dir).exit_code == 0
    plan_path = plan_dir / "plan.json"
    plan = json.loads(plan_path.read_text())
    run = plan["runs"][0]
    config_path = Path(run["config"])
    config = yaml.safe_load(config_path.read_text())
    config["finetune"]["task"]["output_dim"] = 32
    config_path.write_text(yaml.safe_dump(config))
    run["config_sha256"] = _sha256(config_path)
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    rows = _read_manifest_rows(root)
    rows[0]["config_sha256"] = run["config_sha256"]
    write_rows(root / "run_manifest.tsv", rows)

    with pytest.raises(
        ValueError, match="config differs from its frozen recipe|canonical expected runs field config_sha256"
    ):
        experiments.experiment_status(root)


def test_experiment_status_rejects_coherent_generic_config_drift(tmp_path):
    root = tmp_path / "experiment"
    recipe = write_finetune_recipe(root)
    plan_dir = root / "plans" / "train"
    assert plans.build_plan(recipe_path=recipe, output_dir=plan_dir).exit_code == 0
    plan_path = plan_dir / "plan.json"
    plan = json.loads(plan_path.read_text())
    run = plan["runs"][0]
    config_path = Path(run["config"])
    config_path.write_text(config_path.read_text() + "\n")
    run["config_sha256"] = _sha256(config_path)
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    rows = _read_manifest_rows(root)
    rows[0]["config_sha256"] = run["config_sha256"]
    write_rows(root / "run_manifest.tsv", rows)

    with pytest.raises(ValueError, match="Frozen generic config differs from its recipe digest"):
        experiments.experiment_status(root)


def test_experiment_status_rejects_coherent_slurm_script_drift(tmp_path):
    root = tmp_path / "experiment"
    recipe = _write_public_hparam_recipe(root, {"runtime.lr": [1e-6]})
    payload = yaml.safe_load(recipe.read_text())
    payload.setdefault("execution", {}).update(
        {
            "gpus_per_run": 1,
            "scheduler": {
                "type": "slurm",
                "partition": "gpu",
                "cpus_per_task": 8,
                "memory": "64G",
                "walltime": "01:00:00",
                "nice": 0,
            },
        }
    )
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
    plan_dir = root / "plans" / "tune"
    assert plans.build_plan(recipe_path=recipe, output_dir=plan_dir).exit_code == 0
    plan_path = plan_dir / "plan.json"
    plan = json.loads(plan_path.read_text())
    run = plan["runs"][0]
    script_path = Path(run["scheduler_script"])
    script_path.write_text(script_path.read_text() + "echo injected\n")
    run["scheduler_script_sha256"] = _sha256(script_path)
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    rows = _read_manifest_rows(root)
    rows[0]["scheduler_script_sha256"] = run["scheduler_script_sha256"]
    write_rows(root / "run_manifest.tsv", rows)

    with pytest.raises(ValueError, match="Slurm script differs from its frozen recipe|scheduler_script_sha256"):
        experiments.experiment_status(root)


def test_experiment_status_rejects_hparam_run_omission_shared_by_plan_and_canonical_rows(tmp_path):
    root = tmp_path / "experiment"
    recipe = _write_public_hparam_recipe(root, {"runtime.lr": [1e-6, 2e-6]})
    payload = yaml.safe_load(recipe.read_text())
    payload["search"]["max_runs"] = 2
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
    plan_dir = root / "plans" / "tune"
    assert plans.build_plan(recipe_path=recipe, output_dir=plan_dir).exit_code == 0

    plan_path = plan_dir / "plan.json"
    plan = json.loads(plan_path.read_text())
    plan["runs"] = plan["runs"][:1]
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    write_rows(root / "run_manifest.tsv", _read_manifest_rows(root)[:1])
    before = _workspace_files(root)

    with pytest.raises(ValueError, match="canonical expected runs"):
        experiments.experiment_status(root)
    assert _workspace_files(root) == before


@pytest.mark.parametrize(
    "drift",
    ["partial_runtime", "hparam_parameter_summary", "input_snapshots", "command"],
)
def test_experiment_status_rejects_incomplete_registered_plan_identity(tmp_path, drift):
    root = tmp_path / "experiment"
    _init_workspace(root)
    if drift in {"hparam_parameter_summary", "command"}:
        task = "hparam_tune"
    elif drift == "partial_runtime":
        task = "finetune"
    else:
        task = "sleep2stat"
    plan_dir, canonical = _add_plan(root, step_id="train", task=task)
    plan = json.loads((plan_dir / "plan.json").read_text())
    if drift == "partial_runtime":
        plan["runs"][0]["runtime_dir"] = ""
        canonical["runtime_dir"] = ""
    elif drift == "hparam_parameter_summary":
        del plan["runs"][0]["parameter_summary"]
    elif drift == "command":
        del plan["runs"][0]["command"]
    else:
        canonical["input_snapshots"] = [{"field": "inputs.config", "path": canonical["config"], "sha256": "0" * 64}]
    (plan_dir / "plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    write_rows(root / "run_manifest.tsv", [canonical])

    with pytest.raises(ValueError):
        experiments.experiment_status(root)


@pytest.mark.parametrize(
    "drift",
    ["config", "run_script", "resolved_recipe", "plan_alias", "plan_escape", "canonical_run"],
)
def test_experiment_status_rejects_registered_plan_drift(tmp_path, drift):
    root = tmp_path / "experiment"
    _init_workspace(root)
    plan_dir, canonical = _add_plan(root, step_id="train")

    if drift == "config":
        Path(canonical["config"]).write_text("model: changed\n")
    elif drift == "run_script":
        (plan_dir / "run.sh").write_text("#!/bin/sh\nexit 1\n")
    elif drift == "resolved_recipe":
        (plan_dir / "recipe.resolved.yaml").write_text("task: changed\n")
    elif drift == "plan_alias":
        plan_path = plan_dir / "plan.json"
        target = plan_dir / "plan.real.json"
        plan_path.rename(target)
        plan_path.symlink_to(target.name)
    elif drift == "plan_escape":
        step_path = root / "steps" / "train" / "step.yaml"
        step_manifest = yaml.safe_load(step_path.read_text())
        step_manifest["plans"] = [str(tmp_path / "outside")]
        step_path.write_text(yaml.safe_dump(step_manifest, sort_keys=False))
    else:
        (root / "run_manifest.tsv").write_text("step_id\trun_id\n")

    with pytest.raises(ValueError):
        experiments.experiment_status(root)


def test_experiment_status_rejects_duplicate_plan_json_keys(tmp_path, capsys):
    root = tmp_path / "experiment"
    _init_workspace(root)
    plan_dir, _canonical = _add_plan(root, step_id="train")
    plan_path = plan_dir / "plan.json"
    source = plan_path.read_text()
    plan_path.write_text('{\n  "status": "FAIL",\n' + source.lstrip()[1:])
    before = _workspace_files(root)

    assert cli.main(["experiment-status", "--run-dir", str(root), "--json"]) == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "duplicate JSON key: status" in captured.err
    assert "Traceback" not in captured.err
    assert _workspace_files(root) == before


def test_experiment_status_rejects_plan_escape_before_external_probe(tmp_path, monkeypatch):
    root = tmp_path / "experiment"
    _init_workspace(root)
    _add_plan(root, step_id="train")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "plan.blocked.md").write_text("blocked\n")
    step_path = root / "steps" / "train" / "step.yaml"
    step_manifest = yaml.safe_load(step_path.read_text())
    step_manifest["plans"] = [str(outside)]
    step_path.write_text(yaml.safe_dump(step_manifest, sort_keys=False))
    path_exists_at = experiment_io.path_exists_at

    def reject_external_probe(path, *, remote=None):
        try:
            Path(path).relative_to(root)
        except ValueError as exc:
            raise AssertionError(f"status probed outside the canonical workspace: {path}") from exc
        return path_exists_at(path, remote=remote)

    monkeypatch.setattr(experiment_io, "path_exists_at", reject_external_probe)

    with pytest.raises(ValueError, match="outside its managed workspace"):
        experiments.experiment_status(root)


def test_experiment_status_rejects_aliased_plan_directory_before_file_probe(tmp_path, monkeypatch):
    root = tmp_path / "experiment"
    _init_workspace(root)
    _add_plan(root, step_id="train")
    outside = tmp_path / "outside"
    outside.mkdir()
    alias = root / "plans" / "alias"
    alias.symlink_to(outside, target_is_directory=True)
    step_path = root / "steps" / "train" / "step.yaml"
    step_manifest = yaml.safe_load(step_path.read_text())
    step_manifest["plans"] = [str(alias)]
    step_path.write_text(yaml.safe_dump(step_manifest, sort_keys=False))
    path_exists_at = experiment_io.path_exists_at

    def reject_aliased_file_probe(path, *, remote=None):
        if Path(path).is_relative_to(alias):
            raise AssertionError(f"status probed through an aliased plan directory: {path}")
        return path_exists_at(path, remote=remote)

    monkeypatch.setattr(experiment_io, "path_exists_at", reject_aliased_file_probe)

    with pytest.raises(ValueError, match="missing or aliased"):
        experiments.experiment_status(root)


@pytest.mark.parametrize("field", ["title", "objective", "baseline"])
def test_experiment_status_rejects_coherent_registered_plan_experiment_drift(tmp_path, field):
    root = tmp_path / "experiment"
    _init_workspace(root)
    plan_dir, _canonical = _add_plan(root, step_id="train")

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


def test_experiment_status_rejects_registered_plan_recipe_path_drift(tmp_path):
    root = tmp_path / "experiment"
    _init_workspace(root)
    plan_dir, _canonical = _add_plan(root, step_id="train")

    step_path = root / "steps" / "train" / "step.yaml"
    step_manifest = yaml.safe_load(step_path.read_text())
    step_manifest["recipe_path"] = str(root / "registered-recipe.yaml")
    step_path.write_text(yaml.safe_dump(step_manifest, sort_keys=False))
    plan_path = plan_dir / "plan.json"
    plan = json.loads(plan_path.read_text())
    plan["recipe"]["_recipe_path"] = str(root / "foreign-recipe.yaml")
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="recipe path differs from its managed step"):
        experiments.experiment_status(root)


def test_experiment_status_rejects_plan_registered_by_multiple_steps(tmp_path):
    root = tmp_path / "experiment"
    _init_workspace(root)
    plan_dir, _canonical = _add_plan(root, step_id="train")
    second_step = root / "steps" / "evaluate"
    second_step.mkdir(parents=True)
    (second_step / "step.yaml").write_text(
        yaml.safe_dump(
            {
                "step": {"id": "evaluate", "phase": "evaluate", "purpose": "Evaluate."},
                "experiment_id": "status-unit",
                "plan_controller": "ordinary",
                "recipe_path": "",
                "plans": [str(plan_dir)],
            },
            sort_keys=False,
        )
    )

    with pytest.raises(ValueError, match="more than one managed step"):
        experiments.experiment_status(root)


def test_experiment_status_routes_all_registered_reads_to_remote(monkeypatch):
    root = Path("/remote/experiment")
    experiment = {
        "id": "status-unit",
        "title": "Status unit experiment",
        "objective": "Remote status.",
        "root": str(root),
        "baseline": "unit baseline",
    }
    step = {"id": "train", "phase": "train", "purpose": "Train."}
    manifest = {
        "step": step,
        "experiment_id": "status-unit",
        "plan_controller": "ordinary",
        "recipe_path": "",
        "plans": [str(root / "plans" / "train")],
    }
    row = {
        "experiment_id": "status-unit",
        "step_id": "train",
        "run_id": "run-000",
        "run_name": "default",
        "status": "unknown_remote",
    }
    calls = []

    def managed_workspace(candidate, *, remote, allow_completed, validate_experiment_index):
        calls.append(("workspace", candidate, remote, allow_completed, validate_experiment_index))
        return experiment, [row]

    def registered_steps(candidate, *, experiment_id, remote):
        calls.append(("steps", candidate, experiment_id, remote))
        return [manifest]

    def registered_blocked_plan(candidate, *, workspace, remote):
        calls.append(("blocked", candidate, workspace, remote))
        return False

    def registered_plan(
        candidate,
        *,
        workspace,
        workspace_experiment,
        step_manifest,
        workspace_rows,
        expected_recipe_path,
        remote,
        run_index_offset,
    ):
        calls.append(
            (
                "plan",
                candidate,
                workspace,
                workspace_experiment,
                step_manifest,
                workspace_rows,
                expected_recipe_path,
                remote,
                run_index_offset,
            )
        )
        candidate_path = Path(candidate)
        return {
            "path": str(candidate),
            "task": "finetune",
            "run_keys": [("train", "run-000")],
            "launch_script": str(candidate_path / "run.sh"),
        }

    monkeypatch.setattr(experiments, "_managed_workspace", managed_workspace)
    monkeypatch.setattr(experiments, "read_registered_steps", registered_steps)
    monkeypatch.setattr(experiments.artifacts, "is_registered_blocked_plan", registered_blocked_plan)
    monkeypatch.setattr(experiments.artifacts, "read_registered_plan", registered_plan)

    def unexpected_write(*_args, **_kwargs):
        raise AssertionError("remote experiment-status attempted a write")

    for name in (
        "write_text_at",
        "write_rows_at",
        "conditional_atomic_replace_text_at",
        "append_event_at",
        "mkdir_experiment_dirs",
    ):
        monkeypatch.setattr(experiment_io, name, unexpected_write)
    monkeypatch.setattr(experiments, "merge_run_manifest", unexpected_write)

    snapshot = experiments.experiment_status(root, remote="baichuan3")

    assert calls[0] == ("workspace", root, "baichuan3", True, False)
    assert calls[1] == ("steps", root, "status-unit", "baichuan3")
    assert calls[2] == ("blocked", str(root / "plans" / "train"), root, "baichuan3")
    assert calls[3][3] == experiment
    assert calls[3][-3:] == ("", "baichuan3", 0)
    assert snapshot["experiment"]["remote"] == "baichuan3"
    action = snapshot["decision"]["recommended_next"]
    assert action["execution_host"] is None
    assert action["argv"][-2:] == ["--remote", "baichuan3"]
