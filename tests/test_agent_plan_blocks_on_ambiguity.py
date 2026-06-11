from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from agent_tool_test_helpers import write_finetune_recipe, write_yaml
import yaml

from agent_tools.models import REPO_ROOT


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "agent_tools", *args], text=True, capture_output=True)


def _hparam_recipe(
    tmp_path: Path,
    *,
    variant: str = "sleep2vec",
    parameters: dict | None = None,
    max_trials: int | str = 1,
    ckpt_path: Path | None = None,
    final_config_path: Path | None = None,
    selection_metric: str = "val_ahi_pearson",
    selection_mode: str = "max",
) -> Path:
    base = write_finetune_recipe(tmp_path, variant=variant)
    inputs = {"ckpt_path": str(ckpt_path)} if ckpt_path else {}
    if final_config_path is not None:
        inputs["final_eval_config_path"] = str(final_config_path)
    return write_yaml(
        tmp_path / f"tune_{variant}.yaml",
        {
            "schema_version": 1,
            "name": f"unit_tune_{variant}",
            "task": "hparam_tune",
            "variant": variant,
            "base_recipe": str(base),
            "inputs": inputs,
            "search": {"method": "grid", "max_trials": max_trials, "parameters": parameters or {"runtime.lr": [1e-6]}},
            "evaluation_policy": {
                "selection_metric": selection_metric,
                "selection_mode": selection_mode,
                "selection_split": "val",
                "final_eval_split": "test",
                "external_test_locked": True,
                "test_after_fit": False,
                "final_test_unlocked": False,
                "require_manual_unlock_for_final_test": True,
            },
            "decisions": {
                "task": {"value": "hparam_tune", "source": "explicit_recipe"},
                "label_name": {"value": "ahi", "source": "explicit_recipe"},
                "external_test_locked": {"value": True, "source": "explicit_recipe"},
                "train_val_test_policy": {"value": "select on val", "source": "explicit_recipe"},
                "overwrite_policy": {"value": False, "source": "explicit_recipe"},
                "final_eval_unlock": {"value": False, "source": "explicit_recipe"},
            },
        },
    )


def test_plan_does_not_create_run_all_when_consultation_required(tmp_path: Path):
    recipe = write_finetune_recipe(tmp_path, include_label=False)
    output_dir = tmp_path / "plan"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 2
    assert "Questions for user" in result.stdout
    assert "label_name" in result.stdout
    assert (output_dir / "plan.blocked.md").exists()
    assert not (output_dir / "run_all.sh").exists()


def test_context_writes_questions_and_blocked_script(tmp_path: Path):
    config = yaml.safe_load(write_finetune_recipe(tmp_path).read_text())["inputs"]["config"]
    output_dir = tmp_path / "context"

    result = _run("context", "--task", "finetune", "--config", config, "--output-dir", str(output_dir))

    assert result.returncode == 2
    assert (output_dir / "questions.md").exists()
    assert (output_dir / "commands.blocked.sh").exists()
    assert not (output_dir / "commands.sh").exists()
    context = json.loads((output_dir / "context.json").read_text())
    assert context["skill"]["name"] == "finetuning"
    assert "runtime-orchestrator" in context["owners"]
    assert context["relevant_docs"]
    assert context["index_summary"]["rows"] == 1
    assert context["preset_summary"] is None
    assert context["expected_artifacts"]


def test_context_writes_questions_for_mixed_fail_and_user_input(tmp_path: Path):
    recipe = write_finetune_recipe(tmp_path, include_label=False)
    config = Path(yaml.safe_load(recipe.read_text())["inputs"]["config"])
    payload = yaml.safe_load(config.read_text())
    payload["data"]["finetune_data_index"] = None
    payload["data"]["finetune_preset_path"] = str(tmp_path / "missing preset.pkl")
    write_yaml(config, payload)
    output_dir = tmp_path / "context"

    result = _run(
        "context",
        "--task",
        "finetune",
        "--variant",
        "sleep2vec",
        "--config",
        str(config),
        "--output-dir",
        str(output_dir),
    )

    assert result.returncode == 1
    context = json.loads((output_dir / "context.json").read_text())
    assert context["consultation_required"] is True
    assert (output_dir / "questions.md").exists()
    assert (output_dir / "commands.blocked.sh").exists()
    assert not (output_dir / "commands.sh").exists()


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


def test_unlock_final_test_with_explicit_ckpt_generates_final_script(tmp_path: Path):
    ckpt = tmp_path / "best model.ckpt"
    ckpt.write_text("checkpoint")
    recipe = _hparam_recipe(tmp_path, ckpt_path=ckpt)
    output_dir = tmp_path / "unlocked"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir), "--unlock-final-test")

    assert result.returncode == 0
    script = (output_dir / "final_external_test.sh").read_text()
    assert "python -m sleep2vec.infer" in script
    assert shlex_quote(str(ckpt)) in script


def test_plan_uses_user_decision_label_name_in_command(tmp_path: Path):
    recipe = write_finetune_recipe(tmp_path, include_label=False)
    decisions = write_yaml(
        tmp_path / "decisions.yaml",
        {"schema_version": 1, "decisions": {"label_name": {"value": "ahi", "source": "explicit_user"}}},
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
    decisions = write_yaml(
        tmp_path / "decisions.yaml",
        {"schema_version": 1, "decisions": {"test_after_fit": {"value": False, "source": "explicit_user"}}},
    )
    output_dir = tmp_path / "plan"

    result = _run("plan", "--recipe", str(recipe), "--user-decisions", str(decisions), "--output-dir", str(output_dir))

    assert result.returncode == 0
    assert "--no-test-after-fit" in (output_dir / "run.sh").read_text()


def test_plan_normalizes_scalar_runtime_devices(tmp_path: Path):
    for value, expected in [(0, "--devices 0"), ("10", "--devices 10")]:
        recipe = write_finetune_recipe(tmp_path / str(value))
        payload = yaml.safe_load(recipe.read_text())
        payload["runtime"]["devices"] = value
        write_yaml(recipe, payload)
        output_dir = tmp_path / f"plan_{value}"

        result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

        assert result.returncode == 0, result.stderr
        script = (output_dir / "run.sh").read_text()
        assert expected in script
        assert "--devices 1 0" not in script


def test_finetune_plan_honors_runtime_wandb_mode_env(tmp_path: Path):
    recipe = write_finetune_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    payload["runtime"]["wandb_mode"] = "offline"
    write_yaml(recipe, payload)
    output_dir = tmp_path / "plan"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 0
    assert "WANDB_MODE=offline python -m sleep2vec.finetune" in (output_dir / "run.sh").read_text()


def test_hparam_trial_script_honors_base_runtime_wandb_mode_env(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    base_recipe = Path(payload["base_recipe"])
    base_payload = yaml.safe_load(base_recipe.read_text())
    base_payload["runtime"]["wandb_mode"] = "offline"
    write_yaml(base_recipe, base_payload)
    output_dir = tmp_path / "plan"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 0
    assert "WANDB_MODE=offline python -m sleep2vec.finetune" in (output_dir / "trial_000.sh").read_text()


def test_infer_eval_split_ask_user_blocks_command_generation(tmp_path: Path):
    ckpt = tmp_path / "model.ckpt"
    ckpt.write_text("checkpoint")
    config = yaml.safe_load(write_finetune_recipe(tmp_path).read_text())["inputs"]["config"]
    recipe = write_yaml(
        tmp_path / "infer.yaml",
        {
            "schema_version": 1,
            "name": "unit_infer",
            "task": "infer",
            "variant": "sleep2vec",
            "inputs": {
                "config": config,
                "label_name": "ahi",
                "ckpt_path": str(ckpt),
                "eval_split": "ASK_USER",
            },
            "evaluation_policy": {"external_test_locked": True, "final_test_unlocked": False},
            "decisions": {
                "task": {"value": "infer", "source": "explicit_recipe"},
                "label_name": {"value": "ahi", "source": "explicit_recipe"},
                "ckpt_path": {"value": str(ckpt), "source": "explicit_recipe"},
                "overwrite_policy": {"value": False, "source": "explicit_recipe"},
            },
        },
    )
    output_dir = tmp_path / "plan"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 2
    assert "eval_split" in result.stdout
    assert not (output_dir / "run.sh").exists()


def test_infer_invalid_eval_split_blocks_command_generation(tmp_path: Path):
    ckpt = tmp_path / "model.ckpt"
    ckpt.write_text("checkpoint")
    config = yaml.safe_load(write_finetune_recipe(tmp_path).read_text())["inputs"]["config"]
    recipe = write_yaml(
        tmp_path / "infer.yaml",
        {
            "schema_version": 1,
            "name": "unit_infer",
            "task": "infer",
            "variant": "sleep2vec",
            "inputs": {
                "config": config,
                "label_name": "ahi",
                "ckpt_path": str(ckpt),
                "eval_split": "validation",
            },
            "evaluation_policy": {"external_test_locked": False, "final_test_unlocked": False},
            "decisions": {
                "task": {"value": "infer", "source": "explicit_recipe"},
                "label_name": {"value": "ahi", "source": "explicit_recipe"},
                "ckpt_path": {"value": str(ckpt), "source": "explicit_recipe"},
                "overwrite_policy": {"value": False, "source": "explicit_recipe"},
            },
        },
    )
    output_dir = tmp_path / "plan"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 1
    assert "eval_split must be one of" in result.stdout
    assert not (output_dir / "run.sh").exists()


def test_unlock_final_test_with_yaml_search_requires_explicit_final_config(tmp_path: Path):
    ckpt = tmp_path / "best.ckpt"
    ckpt.write_text("checkpoint")
    recipe = _hparam_recipe(tmp_path, parameters={"yaml:/finetune/task/output_dim": [31]}, ckpt_path=ckpt)
    output_dir = tmp_path / "unlocked"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir), "--unlock-final-test")

    assert result.returncode == 2
    assert not (output_dir / "final_external_test.sh").exists()
    assert "final_eval_config_path" in (output_dir / "questions.md").read_text()


def test_unlock_final_test_with_yaml_search_uses_explicit_final_config(tmp_path: Path):
    ckpt = tmp_path / "best.ckpt"
    ckpt.write_text("checkpoint")
    selected_config = tmp_path / "selected_trial.yaml"
    recipe = _hparam_recipe(
        tmp_path,
        parameters={"yaml:/finetune/task/output_dim": [31]},
        ckpt_path=ckpt,
        final_config_path=selected_config,
    )
    selected_config.write_text((tmp_path / "config.yaml").read_text())
    output_dir = tmp_path / "unlocked"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir), "--unlock-final-test")

    assert result.returncode == 0
    script = (output_dir / "final_external_test.sh").read_text()
    assert shlex_quote(str(selected_config)) in script
    assert "trial_000.yaml --ckpt-path" not in script


def test_infer_user_decision_ckpt_path_must_exist(tmp_path: Path):
    config = yaml.safe_load(write_finetune_recipe(tmp_path).read_text())["inputs"]["config"]
    recipe = write_yaml(
        tmp_path / "infer.yaml",
        {
            "schema_version": 1,
            "name": "unit_infer",
            "task": "infer",
            "variant": "sleep2vec",
            "inputs": {
                "config": config,
                "label_name": "ahi",
                "ckpt_path": "ASK_USER",
                "eval_split": "validation",
            },
            "evaluation_policy": {"final_test_unlocked": False},
            "artifacts": {"overwrite": True},
            "decisions": {
                "task": {"value": "infer", "source": "explicit_recipe"},
                "label_name": {"value": "ahi", "source": "explicit_recipe"},
                "overwrite_policy": {"value": True, "source": "explicit_recipe"},
            },
        },
    )
    decisions = write_yaml(
        tmp_path / "decisions.yaml",
        {
            "schema_version": 1,
            "decisions": {"ckpt_path": {"value": str(tmp_path / "missing.ckpt"), "source": "explicit_user"}},
        },
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

    assert result.returncode == 1
    assert "ckpt_path" in (output_dir / "questions.md").read_text()
    assert not (output_dir / "run.sh").exists()


def test_variant_controls_generated_finetune_module(tmp_path: Path):
    recipe = write_finetune_recipe(tmp_path, variant="sleep2vec2")
    output_dir = tmp_path / "plan"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 0
    script = (output_dir / "run.sh").read_text()
    assert "python -m sleep2vec2.finetune" in script
    assert "--no-test-after-fit" in script


def test_sleep2expert_variant_controls_generated_hparam_module(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path, variant="sleep2expert")
    output_dir = tmp_path / "plan"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 0
    script = (output_dir / "trial_000.sh").read_text()
    assert f"cd {shlex_quote(str(REPO_ROOT))}" in script
    assert f"export PYTHONPATH={shlex_quote(str(REPO_ROOT))}" in script
    assert "python -m sleep2expert.finetune" in script
    assert "--no-test-after-fit" in script
    assert f"--results-csv-path {shlex_quote(str(tmp_path / 'results.csv'))}" in script


def test_pretrain_and_adapt_tasks_fail_instead_of_generating_empty_scripts(tmp_path: Path):
    pretrained = tmp_path / "pretrained.ckpt"
    pretrained.write_text("checkpoint")
    recipes = []
    for task in ("pretrain", "adapt"):
        recipe = {
            "schema_version": 1,
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
        assert "No command renderer is implemented" in result.stdout
        assert (output_dir / "plan.blocked.md").exists()
        assert not (output_dir / "run.sh").exists()


def test_context_unsupported_task_writes_blocked_script_instead_of_empty_commands(tmp_path: Path):
    output_dir = tmp_path / "context"

    result = _run("context", "--task", "pretrain", "--variant", "sleep2vec", "--output-dir", str(output_dir))

    assert result.returncode == 1
    assert (output_dir / "commands.blocked.sh").exists()
    assert not (output_dir / "commands.sh").exists()


def test_missing_variant_blocks_command_generation(tmp_path: Path):
    recipe = write_finetune_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    payload.pop("variant")
    write_yaml(recipe, payload)
    output_dir = tmp_path / "plan"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 2
    assert not (output_dir / "run.sh").exists()


def test_plan_refuses_existing_artifact_when_overwrite_false(tmp_path: Path):
    recipe = write_finetune_recipe(tmp_path)
    output_dir = tmp_path / "plan"
    output_dir.mkdir()
    run_script = output_dir / "run.sh"
    run_script.write_text("keep me")

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 1
    assert run_script.read_text() == "keep me"


def test_plan_refuses_existing_blocked_artifact_when_overwrite_missing(tmp_path: Path):
    recipe = write_finetune_recipe(tmp_path, include_label=False)
    payload = yaml.safe_load(recipe.read_text())
    payload["decisions"].pop("overwrite_policy")
    write_yaml(recipe, payload)
    output_dir = tmp_path / "plan"
    output_dir.mkdir()
    blocked_plan = output_dir / "plan.blocked.md"
    blocked_plan.write_text("keep me")

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 2
    assert blocked_plan.read_text() == "keep me"


def test_generated_commands_quote_paths_with_spaces(tmp_path: Path):
    root = tmp_path / "path with space"
    root.mkdir()
    recipe = write_finetune_recipe(root)
    output_dir = root / "plan"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 0
    script = (output_dir / "run.sh").read_text()
    assert shlex_quote(str(root / "config.yaml")) in script


def test_preset_plan_includes_explicit_preset_args(tmp_path: Path):
    base = write_finetune_recipe(tmp_path)
    config = yaml.safe_load(base.read_text())["inputs"]["config"]
    index = tmp_path / "preset_index.csv"
    index.write_text("path,split,duration,ppg_mask,ah_event_mask\nx.npz,train,60,1,1\n")
    output_template = tmp_path / "{dataset}_{split}_{tokens}.pkl"
    manifest_output = tmp_path / "manifest.json"
    recipe = write_yaml(
        tmp_path / "preset.yaml",
        {
            "schema_version": 1,
            "name": "unit_preset",
            "task": "preset_prepare",
            "variant": "sleep2vec",
            "inputs": {"config": config, "index": [str(index)], "dataset_name": "unit"},
            "preset": {
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
            "decisions": {
                "task": {"value": "preset_prepare", "source": "explicit_recipe"},
                "preset_regeneration": {"value": True, "source": "explicit_recipe"},
                "overwrite_policy": {"value": True, "source": "explicit_recipe"},
                "required_channels": {"value": ["ppg", "ahi"], "source": "explicit_recipe"},
                "min_channels": {"value": 2, "source": "explicit_recipe"},
            },
        },
    )
    output_dir = tmp_path / "plan"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 0
    script = (output_dir / "run.sh").read_text()
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


def test_finetune_plan_includes_explicit_input_and_runtime_args(tmp_path: Path):
    recipe = write_finetune_recipe(tmp_path)
    pretrained = tmp_path / "pretrained model.ckpt"
    resume = tmp_path / "resume checkpoint.ckpt"
    pretrained.write_text("checkpoint")
    resume.write_text("checkpoint")
    payload = yaml.safe_load(recipe.read_text())
    payload["inputs"]["pretrained_backbone_path"] = str(pretrained)
    payload["inputs"]["ckpt_path"] = str(resume)
    payload["runtime"].update(
        {
            "device": "cuda",
            "warmup_steps": 11,
            "gradient_clip_val": 0.5,
            "accumulate_grad_batches": 2,
            "patience": 4,
            "check_val_every_n_epoch": 2,
            "ckpt_every_n_epochs": 3,
        }
    )
    payload["decisions"]["pretrained_backbone_path"] = {
        "value": str(pretrained),
        "source": "explicit_recipe",
    }
    write_yaml(recipe, payload)
    output_dir = tmp_path / "plan"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 0
    script = (output_dir / "run.sh").read_text()
    assert f"--pretrained-backbone-path {shlex_quote(str(pretrained))}" in script
    assert f"--ckpt-path {shlex_quote(str(resume))}" in script
    assert "--device cuda" in script
    assert "--warmup-steps 11" in script
    assert "--gradient-clip-val 0.5" in script
    assert "--accumulate-grad-batches 2" in script
    assert "--patience 4" in script
    assert "--check-val-every-n-epoch 2" in script
    assert "--ckpt-every-n-epochs 3" in script


def test_hparam_rejects_bare_search_parameter(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path, parameters={"lr": [1e-6]})
    output_dir = tmp_path / "plan"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 1
    assert not (output_dir / "run_all.sh").exists()


def test_hparam_rejects_non_positive_max_trials(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path, max_trials=0)
    output_dir = tmp_path / "plan"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 1
    assert "hparam_budget" in (output_dir / "questions.md").read_text()
    assert not (output_dir / "run_all.sh").exists()


def test_hparam_runtime_parameter_reaches_trial_script(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path, parameters={"runtime.lr": [2e-6]})
    output_dir = tmp_path / "plan"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 0
    assert "--lr 2e-06" in (output_dir / "trial_000.sh").read_text()


def test_hparam_runtime_training_knobs_reach_trial_script(tmp_path: Path):
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
    script = (output_dir / "trial_000.sh").read_text()
    assert "--gradient-clip-val 0.5" in script
    assert "--accumulate-grad-batches 2" in script
    assert "--warmup-steps 500" in script
    assert "--patience 4" in script
    assert "--check-val-every-n-epoch 2" in script
    assert "--ckpt-every-n-epochs 3" in script


def test_hparam_trial_includes_base_input_and_runtime_args(tmp_path: Path):
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
    script = (output_dir / "trial_000.sh").read_text()
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


def test_hparam_run_all_uses_output_dir_for_trial_scripts(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path)
    output_dir = tmp_path / "plan"
    marker = tmp_path / "ran.txt"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))
    assert result.returncode == 0
    (output_dir / "trial_000.sh").write_text(
        f"#!/usr/bin/env bash\nset -euo pipefail\nprintf ok > {shlex_quote(str(marker))}\n"
    )

    run_all = subprocess.run(["bash", str(output_dir / "run_all.sh")], cwd=tmp_path, text=True, capture_output=True)

    assert run_all.returncode == 0, run_all.stderr
    assert marker.read_text() == "ok"


def test_hparam_yaml_parameter_updates_trial_config(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path, parameters={"yaml:/finetune/task/output_dim": [31]})
    output_dir = tmp_path / "plan"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 0
    trial_config = yaml.safe_load((output_dir / "configs" / "trial_000.yaml").read_text())
    assert trial_config["finetune"]["task"]["output_dim"] == 31


def test_hparam_yaml_parameter_rejects_negative_list_index(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path, parameters={"yaml:/model/channels/-1/input_dim": [9]})
    output_dir = tmp_path / "plan"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 1
    assert not (output_dir / "run_all.sh").exists()


def test_importing_decisions_does_not_import_torch_or_lightning(monkeypatch):
    sys.modules.pop("torch", None)
    sys.modules.pop("pytorch_lightning", None)

    import agent_tools.decisions  # noqa: F401

    assert "torch" not in sys.modules
    assert "pytorch_lightning" not in sys.modules


def shlex_quote(value: str) -> str:
    import shlex

    return shlex.quote(value)
