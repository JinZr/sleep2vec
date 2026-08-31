from __future__ import annotations

import json
from pathlib import Path
from shlex import quote as shlex_quote
import sys

from agent_tool_test_helpers import write_finetune_recipe, write_yaml
import pytest
from test_agent_plan_blocks_on_ambiguity import _first_run, _hparam_recipe, _run, _write_preset_recipe
from test_agent_plan_blocks_on_ambiguity import _stub_execution_target  # noqa: F401
import yaml

from agent_tools import configs, plan_context, plans
from agent_tools.models import REPO_ROOT


def test_unlock_final_test_required_for_final_external_script(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path)
    blocked_output = tmp_path / "blocked"
    unlocked_output = tmp_path / "unlocked"

    blocked = _run("plan", "--recipe", str(recipe), "--output-dir", str(blocked_output))
    unlocked = _run("plan", "--recipe", str(recipe), "--output-dir", str(unlocked_output), "--unlock-final-test")

    assert blocked.returncode == 0
    assert not (blocked_output / "final_external_test.sh").exists()
    assert unlocked.returncode == 2
    assert not (unlocked_output / "final_external_test.sh").exists()


@pytest.mark.parametrize("variant", ["sleep2vec", "sleep2vec2", "sleep2expert"])
def test_unlock_final_test_with_explicit_ckpt_generates_final_script(tmp_path: Path, variant: str):
    ckpt = tmp_path / "best model.ckpt"
    ckpt.write_text("checkpoint")
    recipe = _hparam_recipe(tmp_path, variant=variant, ckpt_path=ckpt)
    output_dir = tmp_path / "unlocked"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir), "--unlock-final-test")

    assert result.returncode == 0
    script = (output_dir / "final_external_test.sh").read_text()
    assert f"python -m {variant}.infer" in script
    assert shlex_quote(str(ckpt)) in script
    assert "This script evaluates the configured final test split." in script
    assert "Run commands do not evaluate the external test split." not in script
    trial_script = Path(_first_run(output_dir)["script"]).read_text()
    assert f"python -m {variant}.finetune" in trial_script
    assert "--no-test-after-fit" in trial_script
    assert "--test-after-fit" not in trial_script


def test_unlock_final_test_uses_hparam_user_decision_checkpoint(tmp_path: Path):
    ckpt = tmp_path / "selected model.ckpt"
    ckpt.write_text("checkpoint")
    recipe = _hparam_recipe(tmp_path)
    decisions = write_yaml(
        tmp_path / "decisions.yaml",
        {"decisions": {"ckpt_path": {"value": str(ckpt), "source": "explicit_user"}}},
    )
    output_dir = tmp_path / "unlocked"

    result = _run(
        "plan",
        "--recipe",
        str(recipe),
        "--user-decisions",
        str(decisions),
        "--output-dir",
        str(output_dir),
        "--unlock-final-test",
    )

    assert result.returncode == 0, result.stderr or result.stdout
    assert shlex_quote(str(ckpt)) in (output_dir / "final_external_test.sh").read_text()
    plan = json.loads((output_dir / "plan.json").read_text())
    assert plan["recipe"]["inputs"]["ckpt_path"] == str(ckpt)
    launch_script = Path(plan["runs"][0]["script"]).read_text()
    assert "--ckpt-path" not in launch_script
    assert str(ckpt) not in launch_script


def test_plan_uses_user_decision_label_name_in_command(tmp_path: Path):
    recipe = write_finetune_recipe(tmp_path, include_label=False)
    decisions = write_yaml(
        tmp_path / "decisions.yaml",
        {"decisions": {"label_name": {"value": "ahi", "source": "explicit_user"}}},
    )
    output_dir = tmp_path / "plan"

    result = _run("plan", "--recipe", str(recipe), "--user-decisions", str(decisions), "--output-dir", str(output_dir))

    assert result.returncode == 0
    script = (output_dir / "run.sh").read_text()
    assert "--label-name ahi" in script
    assert "--label-name --version-name" not in script


def test_plan_uses_user_decision_test_after_fit_in_command(tmp_path: Path):
    recipe = write_finetune_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    payload["evaluation_policy"].pop("test_after_fit")
    write_yaml(recipe, payload)
    source_bytes = recipe.read_bytes()
    decisions = write_yaml(
        tmp_path / "decisions.yaml",
        {"decisions": {"test_after_fit": {"value": False, "source": "explicit_user"}}},
    )
    output_dir = tmp_path / "plan"

    result = _run("plan", "--recipe", str(recipe), "--user-decisions", str(decisions), "--output-dir", str(output_dir))

    assert result.returncode == 0
    assert "--no-test-after-fit" in (output_dir / "run.sh").read_text()
    assert recipe.read_bytes() == source_bytes
    effective = json.loads((output_dir / "plan.json").read_text())["recipe"]
    resolved = yaml.safe_load((output_dir / "recipe.resolved.yaml").read_text())
    assert effective["evaluation_policy"]["test_after_fit"] is False
    assert effective["decisions"]["test_after_fit"]["source"] == "explicit_user"
    assert resolved["evaluation_policy"]["test_after_fit"] is False
    assert resolved["decisions"]["test_after_fit"]["source"] == "explicit_user"


@pytest.mark.parametrize("variant", ["sleep2vec", "sleep2vec2", "sleep2expert"])
def test_finetune_plan_materializes_test_after_fit_policy_default(tmp_path: Path, variant: str):
    recipe = write_finetune_recipe(tmp_path, variant=variant)
    payload = yaml.safe_load(recipe.read_text())
    payload["evaluation_policy"].pop("test_after_fit")
    payload["evaluation_policy"]["external_test_locked"] = False
    write_yaml(recipe, payload)
    source_bytes = recipe.read_bytes()
    output_dir = tmp_path / "plan"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 0, result.stderr or result.stdout
    script = (output_dir / "run.sh").read_text()
    assert f"python -m {variant}.finetune" in script
    assert "--test-after-fit" in script
    assert "--no-test-after-fit" not in script
    assert recipe.read_bytes() == source_bytes
    source = yaml.safe_load(recipe.read_text())
    assert "test_after_fit" not in source["evaluation_policy"]
    assert "test_after_fit" not in source["decisions"]
    effective = json.loads((output_dir / "plan.json").read_text())["recipe"]
    resolved = yaml.safe_load((output_dir / "recipe.resolved.yaml").read_text())
    for frozen in (effective, resolved):
        assert frozen["evaluation_policy"]["test_after_fit"] is True
        assert frozen["decisions"]["test_after_fit"]["value"] is True
        assert frozen["decisions"]["test_after_fit"]["source"] == "policy_default"


@pytest.mark.parametrize("test_after_fit", [False, True])
@pytest.mark.parametrize("variant", ["sleep2vec", "sleep2vec2", "sleep2expert"])
def test_finetune_doctor_and_plan_reject_test_selection_before_workspace_mutation(
    tmp_path: Path,
    variant: str,
    test_after_fit: bool,
):
    source_dir = tmp_path / "source"
    recipe = write_finetune_recipe(source_dir, variant=variant)
    workspace = tmp_path / "workspace"
    payload = yaml.safe_load(recipe.read_text())
    payload["experiment"]["root"] = str(workspace)
    payload["evaluation_policy"].update(
        {
            "selection_split": "test",
            "external_test_locked": False,
            "test_after_fit": test_after_fit,
        }
    )
    payload["decisions"].update(
        {
            "train_val_test_policy": {"value": "test", "source": "explicit_user"},
            "external_test_locked": {"value": False, "source": "explicit_user"},
            "test_after_fit": {"value": test_after_fit, "source": "explicit_user"},
        }
    )
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
    source_recipe_bytes = recipe.read_bytes()
    config = Path(payload["inputs"]["config"])
    source_config_bytes = config.read_bytes()

    doctor = _run("doctor", "--recipe", str(recipe))
    planned = _run(
        "plan",
        "--recipe",
        str(recipe),
        "--output-dir",
        str(workspace / "plans" / "direct-finetune"),
    )

    for result in (doctor, planned):
        assert result.returncode == 1
        assert "Direct finetune cannot select checkpoints on test" in result.stdout
        assert "task=hparam_tune" in result.stdout
        assert "max_runs: 1" in result.stdout
    assert not workspace.exists()
    assert recipe.read_bytes() == source_recipe_bytes
    assert config.read_bytes() == source_config_bytes


def test_finetune_lock_does_not_silently_disable_default_test_after_fit(tmp_path: Path):
    recipe = write_finetune_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    payload["evaluation_policy"].pop("test_after_fit")
    write_yaml(recipe, payload)
    output_dir = tmp_path / "plan"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 2
    assert "test_after_fit=true would evaluate test while external_test_locked=true" in result.stdout
    assert not (output_dir / "run.sh").exists()


@pytest.mark.parametrize("task", ["infer", "evaluate"])
@pytest.mark.parametrize("external_test_locked", [False, True, None], ids=["unlocked", "locked", "omitted"])
def test_direct_test_evaluation_requires_external_test_unlock(
    tmp_path: Path, task: str, external_test_locked: bool | None
):
    ckpt = tmp_path / "model.ckpt"
    ckpt.write_text("checkpoint")
    config = yaml.safe_load(write_finetune_recipe(tmp_path).read_text())["inputs"]["config"]
    payload = {
        "name": f"unit_{task}",
        "task": task,
        "variant": "sleep2vec",
        "inputs": {
            "config": config,
            "label_name": "ahi",
            "ckpt_path": str(ckpt),
            "eval_split": "test",
        },
        "evaluation_policy": {"final_test_unlocked": True},
        "artifacts": {"overwrite": False},
        "decisions": {
            "task": {"value": task, "source": "explicit_recipe"},
            "label_name": {"value": "ahi", "source": "explicit_recipe"},
            "ckpt_path": {"value": str(ckpt), "source": "explicit_recipe"},
            "final_eval_unlock": {"value": True, "source": "explicit_recipe"},
            "overwrite_policy": {"value": False, "source": "explicit_recipe"},
        },
    }
    if external_test_locked is not None:
        payload["evaluation_policy"]["external_test_locked"] = external_test_locked
        payload["decisions"]["external_test_locked"] = {
            "value": external_test_locked,
            "source": "explicit_recipe",
        }
    recipe = write_yaml(tmp_path / f"{task}.yaml", payload)
    output_dir = tmp_path / "plan"

    _, _, report = plans.evaluate_recipe(recipe)
    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    matching = [issue.message for issue in report.blocking_issues() if issue.field == "external_test_locked"]
    if external_test_locked is False:
        assert report.exit_code == 0
        assert result.returncode == 0
        assert "--eval-split test" in (output_dir / "run.sh").read_text()
    else:
        assert report.exit_code == 2
        assert matching == ["Test evaluation requires external_test_locked=false."]
        assert result.returncode == 2
        assert "Test evaluation requires external_test_locked=false." in result.stdout
        assert not (output_dir / "run.sh").exists()


def test_variant_controls_generated_finetune_module(tmp_path: Path):
    recipe = write_finetune_recipe(tmp_path, variant="sleep2vec2")
    output_dir = tmp_path / "plan"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 0
    script = (output_dir / "run.sh").read_text()
    assert "python -m sleep2vec2.finetune" in script
    assert "--no-test-after-fit" in script


@pytest.mark.parametrize("misleading_dir", ["sleep2vec2", "sex_age_baseline"])
def test_path_based_variant_guess_does_not_override_recipe_routing(tmp_path: Path, misleading_dir: str):
    work_dir = tmp_path / misleading_dir
    recipe = write_finetune_recipe(work_dir, variant="sleep2vec")
    output_dir = work_dir / "plan"
    config = Path(yaml.safe_load(recipe.read_text())["inputs"]["config"])

    summary = configs.config_summary(config)

    report = plans.build_plan(recipe_path=recipe, output_dir=output_dir)

    assert summary["variant_guess"] == misleading_dir
    assert summary.get("authoritative_variant") is None
    assert report.exit_code == 0
    assert "python -m sleep2vec.finetune" in (output_dir / "run.sh").read_text()


@pytest.mark.parametrize("variant", ["sleep2vec", "sleep2vec2", "sleep2expert"])
def test_model_variant_controls_generated_hparam_module(tmp_path: Path, variant: str):
    recipe = _hparam_recipe(tmp_path, variant=variant)
    output_dir = tmp_path / "plan"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 0
    script = Path(_first_run(output_dir)["script"]).read_text()
    assert f"cd {shlex_quote(str(REPO_ROOT))}" in script
    assert f"export PYTHONPATH={shlex_quote(str(REPO_ROOT))}" in script
    assert f"python -m {variant}.finetune" in script
    assert "--no-test-after-fit" in script
    assert "--test-after-fit" not in script
    assert "--test-all-checkpoints-after-fit" not in script
    assert f"--results-csv-path {shlex_quote(str(tmp_path / 'results.csv'))}" in script


@pytest.mark.parametrize("test_after_fit", [False, True])
def test_hparam_plan_preflight_uses_resolved_test_after_fit(tmp_path: Path, monkeypatch, test_after_fit: bool):
    recipe = _hparam_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    payload["evaluation_policy"]["test_after_fit"] = test_after_fit
    payload["decisions"]["test_after_fit"] = {"value": test_after_fit, "source": "explicit_recipe"}
    if test_after_fit:
        payload["evaluation_policy"]["external_test_locked"] = False
        payload["decisions"]["external_test_locked"] = {"value": False, "source": "explicit_recipe"}
    write_yaml(recipe, payload)
    observed_split_values = []
    real_index_summary = plan_context.index_summary

    def record_split_values(*args, **kwargs):
        observed_split_values.append(kwargs.get("split_values"))
        return real_index_summary(*args, **kwargs)

    monkeypatch.setattr(plan_context, "index_summary", record_split_values)

    report = plans.build_plan(recipe_path=recipe, output_dir=tmp_path / "plan")

    assert report.exit_code == 0
    assert observed_split_values
    assert all(
        ("test" in split_values) is test_after_fit for split_values in observed_split_values if split_values is not None
    )


@pytest.mark.parametrize("variant", ["sleep2vec", "sleep2vec2", "sleep2expert"])
def test_hparam_plan_materializes_test_after_fit_policy_default_for_test_selection(tmp_path: Path, variant: str):
    recipe = _hparam_recipe(tmp_path, variant=variant)
    payload = yaml.safe_load(recipe.read_text())
    payload["evaluation_policy"].update(
        {
            "selection_metric": "test_ahi_pearson",
            "selection_split": "test",
            "external_test_locked": False,
        }
    )
    payload["evaluation_policy"].pop("test_after_fit")
    payload["decisions"]["external_test_locked"] = {"value": False, "source": "explicit_recipe"}
    payload["decisions"]["train_val_test_policy"] = {"value": "test", "source": "explicit_recipe"}
    write_yaml(recipe, payload)
    output_dir = tmp_path / "plan"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 0, result.stderr or result.stdout
    plan = json.loads((output_dir / "plan.json").read_text())
    resolved = yaml.safe_load((output_dir / "recipe.resolved.yaml").read_text())
    for frozen in (plan["recipe"], resolved):
        assert frozen["evaluation_policy"]["test_after_fit"] is True
        assert frozen["decisions"]["test_after_fit"] == {
            "value": True,
            "source": "policy_default",
            "meaning": "Run the configured test split after each trial unless the recipe or user explicitly opts out.",
        }
    script = Path(plan["runs"][0]["script"]).read_text()
    assert f"{shlex_quote(sys.executable)} -m {variant}.finetune" in script
    assert "--test-after-fit" in script
    assert "--no-test-after-fit" not in script
    assert "--test-all-checkpoints-after-fit" in script
    assert "Candidate selection uses the frozen test split metric." in script
    assert not (output_dir / "final_external_test.sh").exists()


def test_hparam_scalar_user_test_after_fit_false_overrides_policy_default(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    payload["evaluation_policy"].pop("test_after_fit")
    write_yaml(recipe, payload)
    decisions = write_yaml(tmp_path / "decisions.yaml", {"decisions": {"test_after_fit": False}})
    output_dir = tmp_path / "plan"

    result = _run(
        "plan",
        "--recipe",
        str(recipe),
        "--user-decisions",
        str(decisions),
        "--output-dir",
        str(output_dir),
    )

    assert result.returncode == 0, result.stderr or result.stdout
    plan = json.loads((output_dir / "plan.json").read_text())
    assert plan["recipe"]["evaluation_policy"]["test_after_fit"] is False
    assert "--no-test-after-fit" in plan["runs"][0]["command"]
    assert "--test-after-fit" not in plan["runs"][0]["command"]
    assert "--test-all-checkpoints-after-fit" not in plan["runs"][0]["command"]


def test_hparam_plan_allows_test_metric_selection_distinct_from_validation_monitor(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    payload["evaluation_policy"].update(
        {
            "selection_metric": "test_ahi_pearson",
            "selection_split": "test",
            "external_test_locked": False,
            "test_after_fit": True,
            "final_test_unlocked": True,
            "require_manual_unlock_for_final_test": False,
        }
    )
    payload["decisions"].update(
        {
            "external_test_locked": {"value": False, "source": "explicit_user"},
            "test_after_fit": {"value": True, "source": "explicit_user"},
            "train_val_test_policy": {"value": "test", "source": "explicit_user"},
            "final_eval_unlock": {"value": True, "source": "explicit_user"},
        }
    )
    write_yaml(recipe, payload)
    output_dir = tmp_path / "plan"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 0, result.stderr or result.stdout
    plan = json.loads((output_dir / "plan.json").read_text())
    assert plan["recipe"]["evaluation_policy"]["selection_metric"] == "test_ahi_pearson"
    assert plan["recipe"]["evaluation_policy"]["selection_split"] == "test"
    assert "--test-after-fit" in plan["runs"][0]["command"]
    assert "--no-test-after-fit" not in plan["runs"][0]["command"]
    assert "--test-all-checkpoints-after-fit" in plan["runs"][0]["command"]


def test_hparam_plan_guards_stale_final_script_when_unlocked_without_ckpt(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path, variant="sleep2vec2")
    payload = yaml.safe_load(recipe.read_text())
    payload["evaluation_policy"].update(
        {
            "external_test_locked": False,
            "test_after_fit": False,
            "final_test_unlocked": True,
            "require_manual_unlock_for_final_test": False,
        }
    )
    payload["decisions"].update(
        {
            "external_test_locked": {"value": False, "source": "explicit_user"},
            "test_after_fit": {"value": False, "source": "explicit_user"},
            "final_eval_unlock": {"value": True, "source": "explicit_user"},
        }
    )
    write_yaml(recipe, payload)
    output_dir = tmp_path / "plan"
    output_dir.mkdir()
    stale_final_script = output_dir / "final_external_test.sh"
    stale_final_script.write_text("# stale final test script\n")
    stale_final_config = output_dir / "config.final_eval.yaml"
    stale_final_config.write_text("stale: true\n")

    result = _run(
        "plan",
        "--recipe",
        str(recipe),
        "--output-dir",
        str(output_dir),
        "--unlock-final-test",
    )

    assert result.returncode == 1
    assert "Output artifacts already exist" in result.stdout
    assert str(stale_final_script) in result.stdout
    assert str(stale_final_config) in result.stdout
    assert not (output_dir / "plan.md").exists()


def test_hparam_plan_removes_stale_final_script_when_overwrite_allowed_without_ckpt(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path, variant="sleep2vec2")
    payload = yaml.safe_load(recipe.read_text())
    payload["evaluation_policy"].update(
        {
            "external_test_locked": False,
            "test_after_fit": False,
            "final_test_unlocked": True,
            "require_manual_unlock_for_final_test": False,
        }
    )
    payload["decisions"].update(
        {
            "external_test_locked": {"value": False, "source": "explicit_user"},
            "test_after_fit": {"value": False, "source": "explicit_user"},
            "final_eval_unlock": {"value": True, "source": "explicit_user"},
            "overwrite_policy": {"value": True, "source": "explicit_user"},
        }
    )
    payload["artifacts"] = {**payload.get("artifacts", {}), "overwrite": True}
    write_yaml(recipe, payload)
    output_dir = tmp_path / "plan"
    output_dir.mkdir()
    stale_final_script = output_dir / "final_external_test.sh"
    stale_final_script.write_text("# stale final test script\n")
    stale_final_config = output_dir / "config.final_eval.yaml"
    stale_final_config.write_text("stale: true\n")
    unrelated = output_dir / "unrelated.txt"
    unrelated.write_text("preserve me\n")

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 0
    assert not stale_final_script.exists()
    assert not stale_final_config.exists()
    assert unrelated.read_text() == "preserve me\n"
    trial_script = Path(_first_run(output_dir)["script"]).read_text()
    assert "--no-test-after-fit" in trial_script
    assert "--test-after-fit" not in trial_script
    plan = (output_dir / "plan.md").read_text()
    assert "explicit checkpoint path is required" in plan
    assert "Final external-test script generated" not in plan


def test_hparam_plan_blocks_user_test_after_fit_when_lock_stays_resolved(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path, variant="sleep2vec2")
    decisions = write_yaml(
        tmp_path / "decisions.yaml",
        {"decisions": {"test_after_fit": {"value": True, "source": "explicit_user"}}},
    )
    output_dir = tmp_path / "plan"

    result = _run(
        "plan",
        "--recipe",
        str(recipe),
        "--user-decisions",
        str(decisions),
        "--output-dir",
        str(output_dir),
    )

    assert result.returncode == 2
    assert "test_after_fit" in (output_dir / "questions.md").read_text()
    assert not (output_dir / "runs").exists()


def test_pretrain_and_adapt_are_not_runnable_recipe_tasks(tmp_path: Path):
    pretrained = tmp_path / "pretrained.ckpt"
    pretrained.write_text("checkpoint")
    recipes = []
    for task in ("pretrain", "adapt"):
        recipe = {
            "name": f"unit_{task}",
            "task": task,
            "variant": "sleep2vec",
            "inputs": {},
            "artifacts": {"output_dir": str(tmp_path / task)},
            "decisions": {"task": {"value": task, "source": "explicit_recipe"}},
        }
        if task == "adapt":
            recipe["inputs"]["pretrained_backbone_path"] = str(pretrained)
            recipe["decisions"]["pretrained_backbone_path"] = {
                "value": str(pretrained),
                "source": "explicit_recipe",
            }
        recipes.append((task, write_yaml(tmp_path / f"{task}.yaml", recipe)))

    for task, recipe in recipes:
        output_dir = tmp_path / f"{task}_plan"
        result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

        assert result.returncode == 1
        assert f"Unsupported task: {task}" in result.stdout
        assert not output_dir.exists()
        assert not (output_dir / "run.sh").exists()


def test_preset_plan_includes_explicit_preset_args(tmp_path: Path):
    base = write_finetune_recipe(tmp_path)
    config = yaml.safe_load(base.read_text())["inputs"]["config"]
    config_payload = yaml.safe_load(Path(config).read_text())
    config_payload.pop("preset_build")
    write_yaml(Path(config), config_payload)
    index = tmp_path / "preset_index.csv"
    index.write_text("path,split,duration,ppg_mask,ah_event_mask\nx.npz,train,60,1,1\n")
    output_template = tmp_path / "{dataset}_{split}_{tokens}.pkl"
    manifest_output = tmp_path / "manifest.json"
    recipe = _write_preset_recipe(
        tmp_path,
        config=config,
        index=index,
        preset={
            "n_tokens": 128,
            "stride_tokens": 64,
            "split": ["train"],
            "channels": ["ppg", "ahi"],
            "meta_data_names": ["age"],
            "include_no_metadata": True,
            "allow_missing_channels": True,
            "min_channels": 2,
            "output_template": str(output_template),
            "overwrite": True,
            "batch_size": 4,
            "shuffle": False,
            "mask_rate": 0.1,
            "dry_run": True,
            "manifest_output": str(manifest_output),
            "write_sidecar_manifest": False,
        },
    )
    output_dir = tmp_path / "plan"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 0
    wrapper = (output_dir / "run.sh").read_text()
    assert f'preset-launch --plan-dir {shlex_quote(str(output_dir))} "$@"' in wrapper
    assert "--execute" not in wrapper
    script = Path(_first_run(output_dir)["script"]).read_text()
    assert "--stride-tokens 64" in script
    assert "--channels ppg ahi" in script
    assert "--meta-data-names age" in script
    assert "--include-no-metadata" in script
    assert "--allow-missing-channels" in script
    assert "--min-channels 2" in script
    assert f"--output-template {shlex_quote(str(output_template))}" in script
    assert "--overwrite" in script
    assert "--batch-size 4" in script
    assert "--no-shuffle" in script
    assert "--mask-rate 0.1" in script
    assert "--dry-run" in script
    assert f"--manifest-output {shlex_quote(str(manifest_output))}" in script
    assert "--no-write-sidecar-manifest" in script


def test_preset_plan_materializes_rendered_recipe_decisions(tmp_path: Path):
    base = write_finetune_recipe(tmp_path)
    config = yaml.safe_load(base.read_text())["inputs"]["config"]
    config_payload = yaml.safe_load(Path(config).read_text())
    config_payload.pop("preset_build")
    write_yaml(Path(config), config_payload)
    index = tmp_path / "preset_index.csv"
    index.write_text("path,split,duration,ppg_mask,ah_event_mask,stage_mask\nx.npz,train,60,1,1,1\n")
    recipe = _write_preset_recipe(
        tmp_path,
        config=config,
        index=index,
        preset={
            "n_tokens": 128,
            "split": ["train"],
            "channels": ["stage5"],
            "allow_missing_channels": True,
            "min_channels": 1,
            "overwrite": False,
        },
    )
    payload = yaml.safe_load(recipe.read_text())
    payload["decisions"].update(
        {
            "required_channels": {"value": ["ppg", "ahi"], "source": "explicit_recipe"},
            "min_channels": {"value": 2, "source": "explicit_recipe"},
            "overwrite_policy": {"value": True, "source": "explicit_recipe"},
        }
    )
    write_yaml(recipe, payload)
    output_dir = tmp_path / "preset-plan"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 0, result.stderr or result.stdout
    wrapper = (output_dir / "run.sh").read_text()
    assert f'preset-launch --plan-dir {shlex_quote(str(output_dir))} "$@"' in wrapper
    assert "--execute" not in wrapper
    script = Path(_first_run(output_dir)["script"]).read_text()
    assert "--channels ppg ahi" in script
    assert "--channels stage5" not in script
    assert "--min-channels 2" in script
    assert "--overwrite" in script
    resolved = yaml.safe_load((output_dir / "recipe.resolved.yaml").read_text())
    assert "regenerate" not in resolved["preset"]
    assert resolved["decisions"]["preset_regeneration"]["value"] is True


@pytest.mark.parametrize(
    ("variant", "expected_script"),
    [
        ("sleep2vec", "preprocess/save_dataset_presets.py"),
        ("sleep2vec2", "sleep2vec2/preprocess/save_dataset_presets.py"),
        ("sleep2expert", "sleep2expert/preprocess/save_dataset_presets.py"),
    ],
)
def test_preset_plan_routes_to_variant_local_script(tmp_path: Path, variant: str, expected_script: str):
    base = write_finetune_recipe(tmp_path, variant=variant)
    config = yaml.safe_load(base.read_text())["inputs"]["config"]
    config_bytes = Path(config).read_bytes()
    index = tmp_path / "preset_index.csv"
    index.write_text("path,split,duration,ppg_mask\nx.npz,train,60,1\n")
    recipe = _write_preset_recipe(
        tmp_path,
        config=config,
        index=index,
        variant=variant,
        preset={"n_tokens": 128, "split": ["train"], "allow_missing_channels": True},
    )
    output_dir = tmp_path / "plan"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 0, result.stderr or result.stdout
    wrapper = (output_dir / "run.sh").read_text()
    assert f'preset-launch --plan-dir {shlex_quote(str(output_dir))} "$@"' in wrapper
    assert "--execute" not in wrapper
    script = Path(_first_run(output_dir)["script"]).read_text()
    assert expected_script in script
    assert "--channels" not in script
    assert "--min-channels" not in script
    resolved = yaml.safe_load((output_dir / "recipe.resolved.yaml").read_text())
    assert "channels" not in resolved["preset"]
    assert "min_channels" not in resolved["preset"]
    assert resolved["decisions"]["required_channels"] == {
        "value": ["ppg", "ahi", "stage5"],
        "source": "explicit_config",
    }
    assert resolved["decisions"]["min_channels"] == {"value": 3, "source": "explicit_config"}
    assert Path(config).read_bytes() == config_bytes
    assert Path(_first_run(output_dir)["config"]).read_bytes() == config_bytes


@pytest.mark.parametrize(
    ("field", "preset_field"),
    [("required_channels", "channels"), ("min_channels", "min_channels")],
)
@pytest.mark.parametrize("ask_user_source", ["preset", "decision"])
def test_preset_plan_accepts_config_owned_user_decision_for_ask_user(
    tmp_path: Path,
    field: str,
    preset_field: str,
    ask_user_source: str,
):
    source_dir = tmp_path / "source"
    base = write_finetune_recipe(source_dir)
    config = yaml.safe_load(base.read_text())["inputs"]["config"]
    config_payload = yaml.safe_load(Path(config).read_text())
    config_value = config_payload["preset_build"][field]
    index = source_dir / "preset_index.csv"
    index.write_text("path,split,duration,ppg_mask\nx.npz,train,60,1\n")
    recipe = _write_preset_recipe(source_dir, config=config, index=index)
    payload = yaml.safe_load(recipe.read_text())
    if ask_user_source == "preset":
        payload["preset"][preset_field] = "ASK_USER"
    else:
        payload["decisions"][field] = {"value": "ASK_USER", "source": "explicit_recipe"}
    write_yaml(recipe, payload)
    user_decisions = write_yaml(
        source_dir / "user_decisions.yaml",
        {"decisions": {field: {"value": config_value, "source": "explicit_user"}}},
    )
    output_dir = source_dir / "plan"

    result = _run(
        "plan",
        "--recipe",
        str(recipe),
        "--user-decisions",
        str(user_decisions),
        "--output-dir",
        str(output_dir),
    )

    assert result.returncode == 0, result.stderr or result.stdout
    script = (output_dir / "run.sh").read_text()
    assert "--channels" not in script
    assert "--min-channels" not in script
    resolved = yaml.safe_load((output_dir / "recipe.resolved.yaml").read_text())
    assert preset_field not in resolved["preset"]
    assert resolved["decisions"][field] == {"value": config_value, "source": "explicit_user"}


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("required_channels", ["ahi", "ppg", "stage5"]),
        ("min_channels", 1),
    ],
)
def test_preset_plan_rejects_config_owned_channel_conflict_before_workspace(
    tmp_path: Path,
    field: str,
    value: list[str] | int,
):
    source_dir = tmp_path / "source"
    base = write_finetune_recipe(source_dir)
    config = yaml.safe_load(base.read_text())["inputs"]["config"]
    index = source_dir / "preset_index.csv"
    index.write_text("path,split,duration,ppg_mask\nx.npz,train,60,1\n")
    recipe = _write_preset_recipe(source_dir, config=config, index=index)
    payload = yaml.safe_load(recipe.read_text())
    payload["decisions"][field] = {"value": value, "source": "explicit_recipe"}
    write_yaml(recipe, payload)
    output_dir = tmp_path / "plan"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 1
    assert f"{field} differs from config preset_build.{field}" in result.stdout
    assert not output_dir.exists()


@pytest.mark.parametrize(
    ("field", "preset_field", "value"),
    [
        ("required_channels", "channels", ["ppg"]),
        ("min_channels", "min_channels", 1),
    ],
)
def test_preset_plan_rejects_authored_recipe_conflict_when_decision_matches_config(
    tmp_path: Path,
    field: str,
    preset_field: str,
    value: list[str] | int,
):
    source_dir = tmp_path / "source"
    base = write_finetune_recipe(source_dir)
    config = yaml.safe_load(base.read_text())["inputs"]["config"]
    index = source_dir / "preset_index.csv"
    index.write_text("path,split,duration,ppg_mask\nx.npz,train,60,1\n")
    recipe = _write_preset_recipe(source_dir, config=config, index=index)
    payload = yaml.safe_load(recipe.read_text())
    payload["preset"][preset_field] = value
    write_yaml(recipe, payload)
    output_dir = tmp_path / "plan"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 1
    assert f"{field} differs from config preset_build.{field}" in result.stdout
    assert not output_dir.exists()


@pytest.mark.parametrize(
    ("field", "value"),
    [("manifest_output", "manifest.json"), ("write_sidecar_manifest", False)],
)
@pytest.mark.parametrize("variant", ["sleep2vec2", "sleep2expert"])
def test_variant_preset_rejects_root_only_manifest_flags_before_writing(
    tmp_path: Path,
    variant: str,
    field: str,
    value: str | bool,
):
    base = write_finetune_recipe(tmp_path, variant=variant)
    config = yaml.safe_load(base.read_text())["inputs"]["config"]
    index = tmp_path / "preset_index.csv"
    index.write_text("path,split,duration,ppg_mask\nx.npz,train,60,1\n")
    recipe = _write_preset_recipe(
        tmp_path,
        config=config,
        index=index,
        variant=variant,
        preset={"n_tokens": 128, "split": ["train"], "allow_missing_channels": False, field: value},
    )
    output_dir = tmp_path / "plan"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 1
    assert f"does not support {field}" in result.stdout
    assert not (output_dir / "run.sh").exists()
    assert not (output_dir / "plan.json").exists()


def test_hparam_runtime_parameter_reaches_run_script(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path, parameters={"runtime.lr": [2e-6]})
    output_dir = tmp_path / "plan"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 0
    assert "--lr 2e-06" in Path(_first_run(output_dir)["script"]).read_text()


def test_hparam_runtime_training_knobs_reach_run_script(tmp_path: Path):
    recipe = _hparam_recipe(
        tmp_path,
        parameters={
            "runtime.gradient_clip_val": [0.5],
            "runtime.accumulate_grad_batches": [2],
            "runtime.warmup_steps": [500],
            "runtime.patience": [4],
            "runtime.check_val_every_n_epoch": [2],
            "runtime.ckpt_every_n_epochs": [3],
        },
    )
    output_dir = tmp_path / "plan"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 0
    script = Path(_first_run(output_dir)["script"]).read_text()
    assert "--gradient-clip-val 0.5" in script
    assert "--accumulate-grad-batches 2" in script
    assert "--warmup-steps 500" in script
    assert "--patience 4" in script
    assert "--check-val-every-n-epoch 2" in script
    assert "--ckpt-every-n-epochs 3" in script


def test_hparam_run_includes_base_input_and_runtime_args(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    base_recipe = Path(payload["base_recipe"])
    base_payload = yaml.safe_load(base_recipe.read_text())
    pretrained = tmp_path / "base pretrained.ckpt"
    pretrained.write_text("checkpoint")
    base_payload["inputs"]["pretrained_backbone_path"] = str(pretrained)
    base_payload["runtime"].update({"warmup_steps": 7, "gradient_clip_val": 0.75, "accumulate_grad_batches": 3})
    base_payload["decisions"]["pretrained_backbone_path"] = {
        "value": str(pretrained),
        "source": "explicit_recipe",
    }
    write_yaml(base_recipe, base_payload)
    output_dir = tmp_path / "plan"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 0
    script = Path(_first_run(output_dir)["script"]).read_text()
    assert f"--pretrained-backbone-path {shlex_quote(str(pretrained))}" in script
    assert "--warmup-steps 7" in script
    assert "--gradient-clip-val 0.75" in script
    assert "--accumulate-grad-batches 3" in script


def test_hparam_blocks_when_base_finetune_pretrained_decision_is_missing(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    base_recipe = Path(payload["base_recipe"])
    base_payload = yaml.safe_load(base_recipe.read_text())
    base_payload["decisions"].pop("pretrained_backbone_path")
    write_yaml(base_recipe, base_payload)
    output_dir = tmp_path / "plan"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 2
    assert "base_finetune.pretrained_backbone_path" in result.stdout
    assert not (output_dir / "run_all.sh").exists()


def test_hparam_local_ask_user_overrides_base_test_after_fit_decision(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    base_recipe = Path(payload["base_recipe"])
    base_payload = yaml.safe_load(base_recipe.read_text())
    base_payload["evaluation_policy"].update({"external_test_locked": False, "test_after_fit": True})
    base_payload["decisions"].update(
        {
            "external_test_locked": {"value": False, "source": "explicit_recipe"},
            "test_after_fit": {"value": True, "source": "explicit_recipe"},
        }
    )
    write_yaml(base_recipe, base_payload)
    payload["evaluation_policy"].pop("test_after_fit")
    payload["evaluation_policy"].update(
        {
            "external_test_locked": False,
            "final_test_unlocked": True,
            "require_manual_unlock_for_final_test": False,
        }
    )
    payload["decisions"].update(
        {
            "external_test_locked": {"value": False, "source": "explicit_recipe"},
            "final_eval_unlock": {"value": True, "source": "explicit_recipe"},
            "test_after_fit": {"value": "ASK_USER", "source": "unresolved"},
        }
    )
    write_yaml(recipe, payload)
    output_dir = tmp_path / "plan"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 2
    assert "base_finetune.test_after_fit" in result.stdout
    assert not (output_dir / "plan.json").exists()
    assert not (output_dir / "runs").exists()
