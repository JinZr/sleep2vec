from __future__ import annotations

import os
from pathlib import Path
import sys

from agent_tool_test_helpers import run_cli, tree_files, write_finetune_recipe, write_variant_hparam_recipe, write_yaml
import pytest
import yaml

from agent_tools import experiments, plans, run_artifacts
from agent_tools.experiment_workspace import ensure_experiment_workspace, read_run_manifest

pytestmark = pytest.mark.usefixtures("stub_execution_target")


def test_plan_does_not_create_run_all_when_consultation_required(tmp_path: Path):
    recipe = write_finetune_recipe(tmp_path, include_label=False)
    output_dir = tmp_path / "plan"

    result = run_cli("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 2
    assert "Questions for user" in result.stdout
    assert "label_name" in result.stdout
    assert str(output_dir / "decisions.yaml") in result.stdout
    assert (output_dir / "plan.blocked.md").exists()
    assert (output_dir / "decisions.yaml").exists()
    assert "fresh `--output-dir`" in (output_dir / "plan.blocked.md").read_text()
    assert not (output_dir / "run_all.sh").exists()


def test_plan_with_unresolved_experiment_metadata_does_not_write_output(tmp_path: Path):
    recipe = write_finetune_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    payload["step"]["purpose"] = "ASK_USER"
    recipe.write_text(yaml.safe_dump(payload))
    output_dir = tmp_path / "unresolved-plan"

    result = run_cli("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 2
    assert not output_dir.exists()


@pytest.mark.parametrize("section", ["experiment", "step"])
def test_plan_rejects_non_string_workspace_ids_before_creating_workspace(tmp_path: Path, section: str):
    source = tmp_path / "source"
    recipe = write_finetune_recipe(source)
    payload = yaml.safe_load(recipe.read_text())
    workspace = tmp_path / "workspace"
    payload["experiment"]["root"] = str(workspace)
    payload[section]["id"] = 123
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))

    result = run_cli("plan", "--recipe", str(recipe), "--output-dir", str(workspace / "plans" / "first"))

    assert result.returncode == 1
    assert f"{section}.id must be a string" in result.stdout
    assert not workspace.exists()


def test_blocked_plan_initializes_workspace_and_retry_uses_new_plan_dir(tmp_path: Path):
    source = tmp_path / "source"
    recipe = write_finetune_recipe(source, include_label=False)
    workspace = tmp_path / "workspace"
    payload = yaml.safe_load(recipe.read_text())
    payload["experiment"]["root"] = str(workspace)
    recipe.write_text(yaml.safe_dump(payload))
    blocked_dir = workspace / "plans" / "blocked"

    blocked = run_cli("plan", "--recipe", str(recipe), "--output-dir", str(blocked_dir))

    assert blocked.returncode == 2
    assert (workspace / "experiment.yaml").exists()
    assert (blocked_dir / "plan.blocked.md").exists()
    decisions = blocked_dir / "decisions.yaml"
    decision_payload = yaml.safe_load(decisions.read_text())
    decision_payload["decisions"]["label_name"]["value"] = "ahi"
    decisions.write_text(yaml.safe_dump(decision_payload, sort_keys=False))
    retry_dir = workspace / "plans" / "retry"

    retry = run_cli(
        "plan",
        "--recipe",
        str(recipe),
        "--user-decisions",
        str(decisions),
        "--output-dir",
        str(retry_dir),
    )

    assert retry.returncode == 0, retry.stderr
    assert (retry_dir / "run.sh").exists()


@pytest.mark.parametrize("task", ["finetune", "hparam_tune"])
def test_blocked_plan_directory_may_equal_fresh_experiment_root(tmp_path: Path, task: str):
    source = tmp_path / "source"
    workspace = tmp_path / "workspace"
    if task == "hparam_tune":
        recipe = write_variant_hparam_recipe(source)
        payload = yaml.safe_load(recipe.read_text())
        payload["decisions"]["label_name"]["value"] = "ASK_USER"
    else:
        recipe = write_finetune_recipe(source, include_label=False)
        payload = yaml.safe_load(recipe.read_text())
    payload["experiment"]["root"] = str(workspace)
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))

    result = run_cli("plan", "--recipe", str(recipe), "--output-dir", str(workspace))

    assert result.returncode == 2, (result.stdout, result.stderr)
    assert (workspace / "plan.blocked.md").is_file()
    assert (workspace / "decisions.yaml").is_file()
    assert run_artifacts.is_registered_blocked_plan(workspace, workspace=workspace)
    step = yaml.safe_load((workspace / "steps" / payload["step"]["id"] / "step.yaml").read_text())
    assert step["plans"] == [str(workspace)]
    assert read_run_manifest(workspace) == []
    assert experiments.experiment_status(workspace)["summary"]["state"] == "empty"


@pytest.mark.parametrize(
    ("residue_name", "is_directory"),
    [
        ("run.sh", False),
        ("runs", True),
        (".workspace.interrupted.staging", True),
        (".workspace.interrupted.backup", True),
    ],
    ids=["pass-file", "pass-directory", "staging", "backup"],
)
def test_root_blocked_plan_rejects_interrupted_pass_residue(
    tmp_path: Path,
    residue_name: str,
    is_directory: bool,
):
    source = tmp_path / "source"
    recipe = write_finetune_recipe(source, include_label=False)
    payload = yaml.safe_load(recipe.read_text())
    workspace = tmp_path / "workspace"
    payload["experiment"]["root"] = str(workspace)
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
    ensure_experiment_workspace(payload, workspace, register_step=False)
    residue = workspace / residue_name
    if is_directory:
        residue.mkdir()
    else:
        residue.write_text("interrupted publication\n")

    result = run_cli("plan", "--recipe", str(recipe), "--output-dir", str(workspace))

    assert result.returncode == 1
    assert not (workspace / "plan.blocked.md").exists()
    assert not (workspace / "decisions.yaml").exists()
    assert not (workspace / "steps" / payload["step"]["id"] / "step.yaml").exists()
    assert residue.exists()


def test_registered_root_blocked_plan_reader_rejects_pass_residue(tmp_path: Path):
    source = tmp_path / "source"
    recipe = write_finetune_recipe(source, include_label=False)
    payload = yaml.safe_load(recipe.read_text())
    workspace = tmp_path / "workspace"
    payload["experiment"]["root"] = str(workspace)
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
    assert run_cli("plan", "--recipe", str(recipe), "--output-dir", str(workspace)).returncode == 2
    (workspace / "run.sh").write_text("interrupted publication\n")

    with pytest.raises(ValueError, match="contains PASS planning artifacts"):
        run_artifacts.is_registered_blocked_plan(workspace, workspace=workspace)
    with pytest.raises(ValueError, match="contains PASS planning artifacts"):
        experiments.experiment_status(workspace)


def test_generic_blocked_plan_retry_rejects_same_output_dir(tmp_path: Path):
    source = tmp_path / "source"
    recipe = write_finetune_recipe(source, include_label=False)
    workspace = tmp_path / "workspace"
    payload = yaml.safe_load(recipe.read_text())
    payload["experiment"]["root"] = str(workspace)
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
    blocked_dir = workspace / "plans" / "blocked"

    blocked = run_cli("plan", "--recipe", str(recipe), "--output-dir", str(blocked_dir))

    assert blocked.returncode == 2
    decisions = blocked_dir / "decisions.yaml"
    decision_payload = yaml.safe_load(decisions.read_text())
    decision_payload["decisions"]["label_name"]["value"] = "ahi"
    decisions.write_text(yaml.safe_dump(decision_payload, sort_keys=False))
    blocked_files = {path.name: path.read_bytes() for path in blocked_dir.iterdir() if path.is_file()}
    manifest = workspace / "run_manifest.tsv"
    manifest_bytes = manifest.read_bytes()

    retry = run_cli(
        "plan",
        "--recipe",
        str(recipe),
        "--user-decisions",
        str(decisions),
        "--output-dir",
        str(blocked_dir),
    )

    assert retry.returncode == 1
    assert "fresh --output-dir" in retry.stdout
    assert {path.name: path.read_bytes() for path in blocked_dir.iterdir() if path.is_file()} == blocked_files
    assert manifest.read_bytes() == manifest_bytes
    assert not (blocked_dir / "plan.json").exists()
    assert not (blocked_dir / "runs").exists()


def test_generic_blocked_plan_rejects_foreign_output_entry(tmp_path: Path):
    source = tmp_path / "source"
    recipe = write_finetune_recipe(source, include_label=False)
    workspace = tmp_path / "workspace"
    payload = yaml.safe_load(recipe.read_text())
    payload["experiment"]["root"] = str(workspace)
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
    plan_dir = workspace / "plans" / "blocked"
    ensure_experiment_workspace(payload, plan_dir, register_step=False)
    plan_dir.mkdir(parents=True)
    competitor = plan_dir / "run.sh"
    competitor.write_text("user competitor\n")
    before = tree_files(workspace)

    result = run_cli("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir))

    assert result.returncode == 1
    assert "unexpected entries" in result.stdout
    assert tree_files(workspace) == before
    assert read_run_manifest(workspace) == []


@pytest.mark.parametrize("competitor_kind", ["file", "symlink"])
def test_generic_pass_rolls_back_blocked_artifact_created_after_final_guard(
    tmp_path: Path,
    monkeypatch,
    competitor_kind: str,
):
    source = tmp_path / "source"
    recipe = write_finetune_recipe(source)
    workspace = tmp_path / "workspace"
    payload = yaml.safe_load(recipe.read_text())
    payload["experiment"]["root"] = str(workspace)
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
    plan_dir = workspace / "plans" / "current"
    ensure_experiment_workspace(payload, plan_dir, register_step=False)
    plan_dir.mkdir(parents=True)
    competitor = plan_dir / "decisions.yaml"
    guard_pass = plans._guard_pass_plan_publication
    guard_calls = 0

    def inject_after_final_guard(*args, **kwargs):
        nonlocal guard_calls
        guarded = guard_pass(*args, **kwargs)
        guard_calls += 1
        if guard_calls == 1:
            if competitor_kind == "symlink":
                target = tmp_path / "user-decisions.yaml"
                target.write_text("user competitor\n")
                competitor.symlink_to(target)
            else:
                competitor.write_text("user competitor\n")
        return guarded

    monkeypatch.setattr(plans, "_guard_pass_plan_publication", inject_after_final_guard)

    with pytest.raises(ValueError, match="planning artifacts|Managed output paths"):
        plans.build_plan(recipe_path=recipe, output_dir=plan_dir)

    assert competitor.read_text() == "user competitor\n"
    assert competitor.is_symlink() is (competitor_kind == "symlink")
    assert not (plan_dir / "plan.json").exists()
    assert read_run_manifest(workspace) == []


def test_hparam_blocked_plan_writes_user_decision_template(tmp_path: Path):
    recipe = write_variant_hparam_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    payload["decisions"]["label_name"]["value"] = "ASK_USER"
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
    output_dir = tmp_path / "hparam-blocked"

    result = run_cli("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 2
    assert yaml.safe_load((output_dir / "decisions.yaml").read_text())["decisions"]["label_name"] == {
        "value": "ASK_USER",
        "source": "explicit_user",
        "question": (
            "Which label should this task use, for example ahi, age, sex, stage5, or a custom metadata label?"
        ),
    }


def test_blocked_plan_rejects_user_decisions_created_during_publication(tmp_path: Path, monkeypatch):
    source = tmp_path / "source"
    recipe = write_finetune_recipe(source, include_label=False)
    workspace = tmp_path / "workspace"
    payload = yaml.safe_load(recipe.read_text())
    payload["experiment"]["root"] = str(workspace)
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
    plan_dir = workspace / "plans" / "blocked"
    template = plan_dir / "decisions.yaml"
    real_link = os.link

    def competing_link(source_path, destination, *args, **kwargs):
        if Path(destination) == template:
            template.write_text("user competitor\n")
        return real_link(source_path, destination, *args, **kwargs)

    monkeypatch.setattr(os, "link", competing_link)

    with pytest.raises(ValueError, match="appeared during blocked plan publication"):
        plans.build_plan(recipe_path=recipe, output_dir=plan_dir)

    assert template.read_text() == "user competitor\n"
    assert not (plan_dir / "plan.blocked.md").exists()
    assert not list(plan_dir.glob(".decisions.yaml.*.tmp"))
    assert not (workspace / "steps" / payload["step"]["id"] / "step.yaml").exists()
    assert experiments.experiment_status(workspace)["steps"] == []


def test_blocked_plan_registers_step_after_complete_bundle(tmp_path: Path, monkeypatch):
    source = tmp_path / "source"
    recipe = write_finetune_recipe(source, include_label=False)
    workspace = tmp_path / "workspace"
    payload = yaml.safe_load(recipe.read_text())
    payload["experiment"]["root"] = str(workspace)
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
    plan_dir = workspace / "plans" / "blocked"
    step_manifest = workspace / "steps" / payload["step"]["id"] / "step.yaml"
    real_ensure = plans.ensure_experiment_workspace
    registration_modes = []

    def inspect_registration(*args, **kwargs):
        register_step = kwargs.get("register_step", True)
        registration_modes.append(register_step)
        if register_step:
            assert not step_manifest.exists()
            assert run_artifacts.is_registered_blocked_plan(plan_dir, workspace=workspace)
        return real_ensure(*args, **kwargs)

    monkeypatch.setattr(plans, "ensure_experiment_workspace", inspect_registration)

    report = plans.build_plan(recipe_path=recipe, output_dir=plan_dir)

    assert report.exit_code == 2
    assert registration_modes == [False, True]
    assert step_manifest.is_file()


def test_hparam_blocked_plan_retry_rejects_same_output_dir(tmp_path: Path):
    recipe = write_variant_hparam_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    payload["decisions"]["label_name"]["value"] = "ASK_USER"
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
    blocked_dir = tmp_path / "hparam-blocked"

    blocked = run_cli("plan", "--recipe", str(recipe), "--output-dir", str(blocked_dir))

    assert blocked.returncode == 2
    decisions = blocked_dir / "decisions.yaml"
    decision_payload = yaml.safe_load(decisions.read_text())
    assert decision_payload["decisions"]["label_name"]["value"] == "ASK_USER"
    blocked_files = {path.name: path.read_bytes() for path in blocked_dir.iterdir() if path.is_file()}
    manifest = tmp_path / "run_manifest.tsv"
    manifest_bytes = manifest.read_bytes()

    retry = run_cli(
        "plan",
        "--recipe",
        str(recipe),
        "--user-decisions",
        str(decisions),
        "--output-dir",
        str(blocked_dir),
    )

    assert retry.returncode == 1
    assert "fresh --output-dir" in retry.stdout
    assert {path.name: path.read_bytes() for path in blocked_dir.iterdir() if path.is_file()} == blocked_files
    assert manifest.read_bytes() == manifest_bytes
    assert not (blocked_dir / "plan.json").exists()
    assert not (blocked_dir / "runs").exists()


def test_plan_blocks_user_decision_test_after_fit_when_finetune_lock_stays_resolved(tmp_path: Path):
    recipe = write_finetune_recipe(tmp_path)
    decisions = write_yaml(
        tmp_path / "decisions.yaml",
        {"decisions": {"test_after_fit": {"value": True, "source": "explicit_user"}}},
    )
    output_dir = tmp_path / "plan"

    result = run_cli(
        "plan", "--recipe", str(recipe), "--user-decisions", str(decisions), "--output-dir", str(output_dir)
    )

    assert result.returncode == 2
    assert "test_after_fit=true would evaluate test" in result.stdout
    assert not (output_dir / "run.sh").exists()


def test_importing_decisions_does_not_import_torch_or_lightning(monkeypatch):
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.delitem(sys.modules, "pytorch_lightning", raising=False)

    import agent_tools.decisions  # noqa: F401

    assert "torch" not in sys.modules
    assert "pytorch_lightning" not in sys.modules
