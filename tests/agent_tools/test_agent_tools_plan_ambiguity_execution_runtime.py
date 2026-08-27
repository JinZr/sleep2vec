from __future__ import annotations

import csv
import json
from pathlib import Path
from shlex import quote as shlex_quote
import subprocess
import sys

from agent_tool_test_helpers import write_finetune_recipe, write_yaml
import pytest
import yaml

from agent_tools import experiments, plan_contract, plan_hparam, plans
from agent_tools.experiment_workspace import file_sha256, merge_run_manifest, read_run_manifest
from agent_tools.models import REPO_ROOT
from agent_tools.plans import collect_runs

from test_agent_plan_blocks_on_ambiguity import (
    _RUNTIME_COMMIT,
    _bound_config_summary,
    _first_run,
    _hparam_recipe,
    _run,
    _stub_execution_target,
)


def test_plan_normalizes_scalar_runtime_devices(tmp_path: Path):
    for value, expected in [(0, "--devices 0"), ("10", "--devices 10")]:
        recipe = write_finetune_recipe(tmp_path / str(value))
        payload = yaml.safe_load(recipe.read_text())
        payload["runtime"]["devices"] = value
        write_yaml(recipe, payload)
        output_dir = recipe.parent / "plan"

        result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

        assert result.returncode == 0, result.stderr
        script = (output_dir / "run.sh").read_text()
        assert expected in script
        assert "--devices 1 0" not in script


def test_finetune_plan_honors_runtime_wandb_mode_cli(tmp_path: Path):
    recipe = write_finetune_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    payload["runtime"]["wandb_mode"] = "offline"
    write_yaml(recipe, payload)
    output_dir = tmp_path / "plan"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 0
    script = (output_dir / "run.sh").read_text()
    assert "--wandb-mode offline" in script
    assert "WANDB_MODE=" not in script


def test_hparam_plan_expands_explicit_configurations_to_exact_runs(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    payload["search"] = {
        "method": "grid",
        "max_runs": 2,
        "configurations": [
            {"runtime.lr": 1.0e-6, "runtime.weight_decay": 1.0e-5},
            {"runtime.lr": 2.0e-6, "runtime.weight_decay": 1.0e-6},
        ],
    }
    write_yaml(recipe, payload)
    output_dir = tmp_path / "plan"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 0, result.stderr or result.stdout
    runs = json.loads((output_dir / "plan.json").read_text())["runs"]
    assert len(runs) == 2  # two points, not the 2x2 cartesian product
    observed = [(run["runtime.lr"], run["runtime.weight_decay"]) for run in runs]
    assert observed == [(1.0e-6, 1.0e-5), (2.0e-6, 1.0e-6)]
    for run, expected_lr in zip(runs, (1.0e-6, 2.0e-6)):
        script = Path(run["script"]).read_text()
        assert f"--lr {expected_lr}" in script


def test_hparam_run_script_honors_base_runtime_wandb_mode_cli(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    base_recipe = Path(payload["base_recipe"])
    base_payload = yaml.safe_load(base_recipe.read_text())
    base_payload["runtime"]["wandb_mode"] = "offline"
    write_yaml(base_recipe, base_payload)
    output_dir = tmp_path / "plan"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 0
    script = Path(_first_run(output_dir)["script"]).read_text()
    assert "--wandb-mode offline" in script
    assert "WANDB_MODE=" not in script


def test_hparam_plan_and_launch_use_merged_effective_recipe(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    payload["inputs"]["label_name"] = "effective-label"
    payload["runtime"] = {"devices": [4], "batch_size": 7}
    payload["artifacts"] = {"results_csv_path": "effective/results.csv"}
    payload["decisions"]["label_name"]["value"] = "effective-label"
    write_yaml(recipe, payload)
    output_dir = tmp_path / "plan"

    planned = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))
    launched = _run("hparam-launch", "--plan-dir", str(output_dir))

    assert planned.returncode == 0, planned.stderr or planned.stdout
    assert launched.returncode == 0, launched.stderr or launched.stdout
    plan = json.loads((output_dir / "plan.json").read_text())
    command = plan["runs"][0]["command"]
    assert "--label-name effective-label" in command
    assert "--batch-size 7" in command
    assert f"--results-csv-path {output_dir / 'effective/results.csv'}" in command
    assert plan["recipe"]["runtime"]["devices"] == [4]
    assert plan["recipe"]["_base_recipe"]["runtime"]["devices"] != [4]
    with (output_dir / "launch_manifest.tsv").open(newline="") as file_obj:
        launch_rows = list(csv.DictReader(file_obj, delimiter="\t"))
    assert launch_rows[0]["gpus"] == "4"


def test_context_rejects_symlink_output_root_before_writing(tmp_path: Path):
    target = tmp_path / "context-target"
    target.mkdir()
    output_dir = tmp_path / "context-alias"
    output_dir.symlink_to(target, target_is_directory=True)

    report = plans.build_context(task="pretrain", variant="sleep2vec", config=None, output_dir=output_dir)

    assert report.exit_code == 1
    assert any(issue.field == "output_artifacts" for issue in report.blocking_issues())
    assert list(target.iterdir()) == []


def test_plan_rejects_symlink_output_root_before_writing(tmp_path: Path):
    recipe = write_finetune_recipe(tmp_path)
    target = tmp_path / "plan-target"
    target.mkdir()
    output_dir = tmp_path / "plan-alias"
    output_dir.symlink_to(target, target_is_directory=True)

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 1
    assert "must not be a symlink" in result.stderr
    assert list(target.iterdir()) == []


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


def test_plan_overwrite_rejects_output_alias_to_canonical_without_writing(tmp_path: Path):
    recipe = write_finetune_recipe(tmp_path / "source")
    workspace = tmp_path / "workspace"
    payload = yaml.safe_load(recipe.read_text())
    payload["experiment"]["root"] = str(workspace)
    payload["decisions"]["overwrite_policy"]["value"] = True
    payload["artifacts"]["overwrite"] = True
    recipe.write_text(yaml.safe_dump(payload))
    initial = _run("plan", "--recipe", str(recipe), "--output-dir", str(workspace / "plan-1"))
    assert initial.returncode == 0, initial.stderr or initial.stdout
    output_dir = workspace / "plan-2"
    output_dir.mkdir()
    canonical_before = (workspace / "run_manifest.tsv").read_bytes()
    step_manifest = workspace / "steps" / payload["step"]["id"] / "step.yaml"
    step_before = step_manifest.read_bytes()
    (output_dir / "plan.json").hardlink_to(workspace / "run_manifest.tsv")

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 1
    assert (workspace / "run_manifest.tsv").read_bytes() == canonical_before
    assert step_manifest.read_bytes() == step_before
    assert not (output_dir / "runs").exists()


def test_plan_allows_existing_workspace_matrix_and_events_for_new_plan(tmp_path: Path):
    recipe = write_finetune_recipe(tmp_path / "source")
    workspace = tmp_path / "workspace"
    payload = yaml.safe_load(recipe.read_text())
    payload["experiment"]["root"] = str(workspace)
    recipe.write_text(yaml.safe_dump(payload))

    first = _run("plan", "--recipe", str(recipe), "--output-dir", str(workspace / "plan-1"))
    second = _run("plan", "--recipe", str(recipe), "--output-dir", str(workspace / "plan-2"))

    assert first.returncode == 0, first.stderr or first.stdout
    assert second.returncode == 0, second.stderr or second.stdout
    assert (workspace / "plan-2" / "plan.json").exists()


def test_plan_refuses_existing_blocked_artifact_when_overwrite_missing(tmp_path: Path):
    recipe = write_finetune_recipe(tmp_path, include_label=False)
    payload = yaml.safe_load(recipe.read_text())
    payload["decisions"].pop("overwrite_policy")
    payload["artifacts"].pop("overwrite")
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
    frozen_config = output_dir / "runs" / "run-000--unit" / "config.yaml"
    assert shlex_quote(str(frozen_config)) in script
    assert shlex_quote(str(root / "config.yaml")) not in script


def test_hparam_materialization_failure_does_not_register_plan(tmp_path: Path, monkeypatch):
    recipe = _hparam_recipe(tmp_path)
    effective_recipe, _, _ = plans.evaluate_recipe(recipe)
    workspace = Path(effective_recipe["experiment"]["root"])
    step_id = effective_recipe["step"]["id"]
    output_dir = workspace / "failed-plan"
    original_write_text = plan_hparam.write_text

    def fail_plan_markdown(path, text, *, executable=False):
        if Path(path).name == "plan.md":
            raise OSError("injected plan materialization failure")
        return original_write_text(path, text, executable=executable)

    monkeypatch.setattr(plan_hparam, "write_text", fail_plan_markdown)

    with pytest.raises(OSError, match="injected plan materialization failure"):
        plans.build_plan(recipe_path=recipe, output_dir=output_dir)

    assert not (workspace / "steps" / step_id / "step.yaml").exists()
    assert read_run_manifest(workspace) == []
    assert not output_dir.exists()

    monkeypatch.setattr(plan_hparam, "write_text", original_write_text)
    successful_dir = workspace / "successful-plan"
    report = plans.build_plan(recipe_path=recipe, output_dir=successful_dir)

    assert report.exit_code == 0
    step_manifest = yaml.safe_load((workspace / "steps" / step_id / "step.yaml").read_text())
    assert step_manifest["plans"] == [str(successful_dir.resolve())]


def test_single_run_materialization_failure_does_not_register_plan(tmp_path: Path, monkeypatch):
    recipe = write_finetune_recipe(tmp_path / "source")
    payload = yaml.safe_load(recipe.read_text())
    workspace = Path(payload["experiment"]["root"])
    output_dir = workspace / "failed-plan"
    original_write_json = plans.write_json

    def fail_plan_manifest(path, value):
        if Path(path).name == "plan.json":
            raise OSError("injected plan materialization failure")
        return original_write_json(path, value)

    monkeypatch.setattr(plans, "write_json", fail_plan_manifest)

    with pytest.raises(OSError, match="injected plan materialization failure"):
        plans.build_plan(recipe_path=recipe, output_dir=output_dir)

    step_path = workspace / "steps" / payload["step"]["id"] / "step.yaml"
    assert not step_path.exists()
    assert read_run_manifest(workspace) == []
    assert output_dir.exists()
    assert not (output_dir / "plan.json").exists()

    monkeypatch.setattr(plans, "write_json", original_write_json)
    successful_dir = workspace / "successful-plan"
    report = plans.build_plan(recipe_path=recipe, output_dir=successful_dir)

    assert report.exit_code == 0
    assert yaml.safe_load(step_path.read_text())["plans"] == [str(successful_dir.resolve())]


def test_hparam_outer_workspace_owns_base_finetune_metadata(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    base_recipe = Path(payload["base_recipe"])
    base_payload = yaml.safe_load(base_recipe.read_text())
    for field in ("id", "title", "objective", "root", "baseline"):
        base_payload["experiment"][field] = "ASK_USER"
    for field in ("id", "phase", "purpose"):
        base_payload["step"][field] = "ASK_USER"
    base_recipe.write_text(yaml.safe_dump(base_payload))
    output_dir = tmp_path / "plan"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 0, result.stdout
    assert "base_finetune.experiment" not in result.stdout


def test_single_finetune_freezes_runtime_and_checkpoint_paths(tmp_path: Path):
    recipe = write_finetune_recipe(tmp_path)
    output_dir = tmp_path / "plan"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 0, result.stderr
    run = _first_run(output_dir)
    expected_runtime = REPO_ROOT / "log-finetune" / run["version"]
    assert run["runtime_dir"] == str(expected_runtime)
    assert run["checkpoint_dir"] == str(expected_runtime / "checkpoints")
    assert json.loads(Path(run["artifacts"]).read_text())["runtime_dir"] == str(expected_runtime)
    script = Path(run["script"])
    script_text = script.read_text()
    assert f"cd {shlex_quote(str(REPO_ROOT))}" in script_text
    assert f"export PYTHONPATH={shlex_quote(str(REPO_ROOT))}${{PYTHONPATH:+:$PYTHONPATH}}" in script_text
    frozen_hash = file_sha256(script)
    assert run["script_sha256"] == frozen_hash
    assert json.loads((Path(run["run_dir"]) / "run.json").read_text())["script_sha256"] == frozen_hash
    with (tmp_path / "run_manifest.tsv").open(newline="") as file_obj:
        manifest = next(csv.DictReader(file_obj, delimiter="\t"))
    assert manifest["script_sha256"] == frozen_hash


def test_single_finetune_uses_explicit_execution_workdir(tmp_path: Path):
    recipe = write_finetune_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    run_cwd = tmp_path / "runtime cwd"
    payload["execution"] = {"workdir": str(run_cwd)}
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
    output_dir = tmp_path / "plan"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 0, result.stderr
    run = _first_run(output_dir)
    expected_runtime = run_cwd / "log-finetune" / run["version"]
    expected_checkpoint = expected_runtime / "checkpoints"
    assert run["runtime_dir"] == str(expected_runtime)
    assert run["checkpoint_dir"] == str(expected_checkpoint)
    artifacts = json.loads(Path(run["artifacts"]).read_text())
    assert artifacts["runtime_dir"] == str(expected_runtime)
    assert artifacts["checkpoint_dir"] == str(expected_checkpoint)
    with (tmp_path / "run_manifest.tsv").open(newline="") as file_obj:
        manifest = next(csv.DictReader(file_obj, delimiter="\t"))
    assert manifest["runtime_dir"] == str(expected_runtime)
    assert manifest["checkpoint_dir"] == str(expected_checkpoint)
    script = Path(run["script"]).read_text()
    assert f"cd {shlex_quote(str(run_cwd))}" in script
    assert f"export PYTHONPATH={shlex_quote(str(run_cwd))}${{PYTHONPATH:+:$PYTHONPATH}}" in script


def test_hparam_workdir_is_verbatim_run_cwd_for_all_generated_scripts(tmp_path: Path):
    checkpoint = tmp_path / "selected.ckpt"
    checkpoint.write_text("checkpoint")
    recipe = _hparam_recipe(tmp_path, ckpt_path=checkpoint)
    payload = yaml.safe_load(recipe.read_text())
    run_cwd = tmp_path / "runtime cwd"
    payload["execution"] = {
        "workdir": str(run_cwd),
        "python": sys.executable,
        "runtime_commit": _RUNTIME_COMMIT,
    }
    recipe.write_text(yaml.safe_dump(payload))
    output_dir = tmp_path / "plan"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir), "--unlock-final-test")

    assert result.returncode == 0, result.stderr
    run = _first_run(output_dir)
    expected_runtime = run_cwd / "log-finetune" / run["version"]
    assert run["runtime_dir"] == str(expected_runtime)
    expected_cwd = f"cd {shlex_quote(str(run_cwd))}"
    expected_pythonpath = f"export PYTHONPATH={shlex_quote(str(run_cwd))}"
    for script in (Path(run["script"]), output_dir / "final_external_test.sh"):
        text = script.read_text()
        assert expected_cwd in text
        assert expected_pythonpath in text
    run_all_text = (output_dir / "run_all.sh").read_text()
    assert f"cd {shlex_quote(str(REPO_ROOT))}" in run_all_text
    assert f"export PYTHONPATH={shlex_quote(str(REPO_ROOT))}" in run_all_text
    assert expected_cwd not in run_all_text
    assert "${PYTHONPATH:+:$PYTHONPATH}" not in run_all_text


def test_hparam_plan_freezes_explicit_target_python_and_commit(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    payload.setdefault("execution", {}).update({"python": "/runtime/bin/python3", "runtime_commit": "A" * 40})
    recipe.write_text(yaml.safe_dump(payload))
    output_dir = tmp_path / "plan"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 0, result.stderr
    plan = json.loads((output_dir / "plan.json").read_text())
    assert plan["recipe"]["execution"]["python"] == "/runtime/bin/python3"
    assert plan["recipe"]["execution"]["runtime_commit"] == "a" * 40
    assert plan["runs"][0]["command"].startswith("/runtime/bin/python3 -m ")
    resolved = yaml.safe_load((output_dir / "recipe.resolved.yaml").read_text())
    assert resolved["execution"]["runtime_commit"] == "a" * 40


def test_hparam_plan_rejects_compound_target_python_before_plan_write(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    payload["execution"] = {
        "workdir": "/separate/runtime",
        "python": "conda run -n exp python",
        "runtime_commit": "a" * 40,
    }
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
    output_dir = tmp_path / "plan"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 1
    assert "execution.python must be a single executable" in result.stdout
    assert not (output_dir / "plan.json").exists()
    assert not (output_dir / "recipe.resolved.yaml").exists()
    assert not (output_dir / "run_all.sh").exists()


@pytest.mark.parametrize(
    "execution",
    [
        {"workdir": "/separate/runtime"},
        {"target": "ssh", "host": "runtime-host", "workdir": "/remote/runtime"},
        {"conda_env": "runtime"},
    ],
)
def test_hparam_plan_requires_explicit_identity_for_non_manager_runtime_before_plan_write(
    tmp_path: Path, execution: dict
):
    recipe = _hparam_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    payload["execution"] = execution
    recipe.write_text(yaml.safe_dump(payload))
    output_dir = tmp_path / "plan"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 2
    assert "execution.python" in result.stdout
    assert "execution.runtime_commit" in result.stdout
    assert not (output_dir / "plan.json").exists()
    assert not (output_dir / "recipe.resolved.yaml").exists()
    assert not (output_dir / "run_all.sh").exists()


@pytest.mark.parametrize(
    ("env", "message"),
    [
        ({"BAD-NAME": "value"}, "POSIX environment variable names"),
        ({"NESTED": ["value"]}, "scalar strings, numbers, or booleans"),
    ],
)
def test_hparam_plan_rejects_unsafe_execution_environment_before_plan_write(tmp_path: Path, env: dict, message: str):
    recipe = _hparam_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    payload["execution"] = {
        "workdir": "/separate/runtime",
        "python": "/runtime/bin/python",
        "runtime_commit": "a" * 40,
        "env": env,
    }
    recipe.write_text(yaml.safe_dump(payload))
    output_dir = tmp_path / "plan"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 1
    assert message in result.stdout
    assert not (output_dir / "plan.json").exists()
    assert not (output_dir / "recipe.resolved.yaml").exists()
    assert not (output_dir / "run_all.sh").exists()


def test_hparam_relative_workdir_fails_before_workspace_creation(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    payload["execution"] = {"workdir": "relative/runtime"}
    recipe.write_text(yaml.safe_dump(payload))
    output_dir = tmp_path / "plan"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 1
    assert "execution.workdir must be an absolute path" in result.stdout
    assert not output_dir.exists()


def test_remote_deferred_config_must_be_locally_freezable_before_single_run_workspace_creation(tmp_path: Path):
    recipe = write_finetune_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    payload["inputs"].update({"config": "/remote/config.yaml", "data_backend": "npz"})
    payload["execution"] = {
        "target": "ssh",
        "host": "unit-host",
        "path_context": "remote",
        "path_validation": "defer",
    }
    payload["decisions"]["required_channels"] = {
        "value": ["ppg", "ahi", "stage5"],
        "source": "explicit_recipe",
    }
    recipe.write_text(yaml.safe_dump(payload))
    output_dir = tmp_path / "plan"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 1
    assert "Config cannot be frozen from a local file" in result.stdout
    assert not output_dir.exists()
    assert not (tmp_path / "steps" / "unit-finetune").exists()
    assert not (tmp_path / "events.jsonl").exists()


def test_remote_deferred_config_must_be_locally_freezable_before_hparam_workspace_creation(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    base_recipe = Path(payload["base_recipe"])
    base_payload = yaml.safe_load(base_recipe.read_text())
    base_payload["inputs"].update({"config": "/remote/config.yaml", "data_backend": "npz"})
    base_payload["decisions"]["required_channels"] = {
        "value": ["ppg", "ahi", "stage5"],
        "source": "explicit_recipe",
    }
    base_recipe.write_text(yaml.safe_dump(base_payload))
    payload["execution"] = {
        "target": "ssh",
        "host": "unit-host",
        "path_context": "remote",
        "path_validation": "defer",
    }
    recipe.write_text(yaml.safe_dump(payload))
    output_dir = tmp_path / "plan"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))

    assert result.returncode == 1
    assert "Config cannot be frozen from a local file" in result.stdout
    assert not output_dir.exists()
    assert not (tmp_path / "steps" / "unit-hparam-tune").exists()
    assert not (tmp_path / "events.jsonl").exists()


def test_collect_runs_only_reads_managed_manifest_paths(tmp_path: Path):
    runtime_dir = tmp_path / "runtime" / "run-000"
    runtime_dir.mkdir(parents=True)
    (runtime_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "version": "runtime-version",
                "config_path": "/runtime/config.yaml",
                "status": "finished",
                "command": "runtime command",
                "epoch": 4,
                "metrics": {"score": 0.8},
            }
        )
    )
    (tmp_path / "run_manifest.tsv").write_text(
        "experiment_id\tstep_id\trun_id\tversion\tconfig\tcommand\truntime_dir\tstatus\n"
        f"unit\ttrain\trun-000\tmanaged-version\t/managed/config.yaml\tmanaged command\t{runtime_dir}\tfailed\n"
    )
    historical = tmp_path / "historical"
    historical.mkdir()
    (historical / "run_manifest.json").write_text(json.dumps({"version": "historical-version"}))
    (historical / "trial_status.tsv").write_text("trial_id\tstatus\ntrial_000\tfinished\n")
    output = tmp_path / "collected.csv"

    collect_runs(tmp_path, "score", output)

    with output.open(newline="") as file_obj:
        rows = list(csv.DictReader(file_obj))
    assert len(rows) == 1
    assert rows[0]["run_id"] == "run-000"
    assert rows[0]["version"] == "managed-version"
    assert rows[0]["config"] == "/managed/config.yaml"
    assert rows[0]["command"] == "managed command"
    assert rows[0]["status"] == "failed"
    assert rows[0]["epoch"] == "4"
    assert rows[0]["score"] == "0.8"
    assert "runtime-version" not in output.read_text()
    assert "historical-version" not in output.read_text()


def test_collect_runs_rejects_invalid_runtime_manifest_without_overwriting_output(tmp_path: Path):
    runtime_dir = tmp_path / "runtime" / "run-000"
    runtime_dir.mkdir(parents=True)
    manifest = runtime_dir / "run_manifest.json"
    manifest.write_text("{")
    (tmp_path / "run_manifest.tsv").write_text(
        "experiment_id\tstep_id\trun_id\truntime_dir\tstatus\n" f"unit\ttrain\trun-000\t{runtime_dir}\tfailed\n"
    )
    output = tmp_path / "collected.csv"
    output.write_bytes(b"existing output\n")

    with pytest.raises(ValueError, match="run manifest"):
        collect_runs(tmp_path, "score", output)

    assert output.read_bytes() == b"existing output\n"


def test_collect_runs_rejects_missing_canonical_manifest_without_overwriting_output(tmp_path: Path):
    output = tmp_path / "collected.csv"
    output.write_bytes(b"existing output\n")

    try:
        collect_runs(tmp_path, None, output)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("collect_runs must require run_manifest.tsv")

    assert output.read_bytes() == b"existing output\n"


def test_collect_runs_rejects_canonical_manifest_output_alias_without_writing(tmp_path: Path):
    manifest = tmp_path / "run_manifest.tsv"
    original = b"step_id\trun_id\n"
    manifest.write_bytes(original)

    with pytest.raises(ValueError, match="cannot overwrite canonical run_manifest.tsv"):
        collect_runs(tmp_path, None, manifest)

    assert manifest.read_bytes() == original


def test_collect_runs_rejects_unsafe_output_topology_without_writing(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    manifest = workspace / "run_manifest.tsv"
    original_manifest = b"step_id\trun_id\n"
    manifest.write_bytes(original_manifest)
    sentinel = tmp_path / "sentinel.csv"
    sentinel.write_bytes(b"keep me\n")
    output = workspace / "collected.csv"
    output.hardlink_to(sentinel)

    with pytest.raises(ValueError, match="Managed output paths"):
        collect_runs(workspace, None, output)

    assert manifest.read_bytes() == original_manifest
    assert sentinel.read_bytes() == b"keep me\n"


def test_collect_runs_allows_header_only_canonical_manifest(tmp_path: Path):
    (tmp_path / "run_manifest.tsv").write_text("step_id\trun_id\n")
    output = tmp_path / "collected.csv"

    collect_runs(tmp_path, None, output)

    assert output.read_text() == "version\n"


def test_hparam_run_all_rejects_tampered_leaf_before_execution(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path)
    output_dir = tmp_path / "plan"
    marker = tmp_path / "ran.txt"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(output_dir))
    assert result.returncode == 0
    run_all_text = (output_dir / "run_all.sh").read_text()
    assert (
        f"{shlex_quote(sys.executable)} -m agent_tools hparam-run-queue "
        f"--plan-dir {shlex_quote(str(output_dir.resolve()))} --execute" in run_all_text
    )
    Path(_first_run(output_dir)["script"]).write_text(
        f"#!/usr/bin/env bash\nset -euo pipefail\nprintf ok > {shlex_quote(str(marker))}\n"
    )

    run_all = subprocess.run(["bash", str(output_dir / "run_all.sh")], cwd=tmp_path, text=True, capture_output=True)

    assert run_all.returncode != 0
    assert "snapshot hash changed" in run_all.stderr
    assert not marker.exists()


@pytest.mark.parametrize("task", ["finetune", "infer", "evaluate", "preset_prepare", "sleep2stat"])
@pytest.mark.parametrize("cwd_kind", ["plan", "outside"])
def test_non_hparam_run_script_commits_lifecycle_from_any_cwd(
    tmp_path: Path,
    monkeypatch,
    task: str,
    cwd_kind: str,
):
    source = tmp_path / "source"
    recipe_path = write_finetune_recipe(source)
    recipe = yaml.safe_load(recipe_path.read_text())
    workspace = tmp_path / "workspace"
    recipe["task"] = task
    recipe["name"] = f"unit_{task}"
    recipe["experiment"]["root"] = str(workspace)
    recipe["step"] = {
        "id": f"unit-{task.replace('_', '-')}",
        "phase": "train" if task == "finetune" else "analyze",
        "purpose": "Exercise managed non-hparam lifecycle.",
    }
    recipe["decisions"]["task"] = {"value": task, "source": "explicit_recipe"}
    if task in {"infer", "evaluate"}:
        recipe["evaluation_policy"] = {"external_test_locked": True}
        recipe["artifacts"] = {"overwrite": False}
    elif task == "preset_prepare":
        recipe.pop("artifacts")
        recipe.pop("evaluation_policy")
        recipe.pop("runtime")
        recipe["inputs"] = {"config": recipe["inputs"]["config"]}
    elif task == "sleep2stat":
        recipe.pop("variant")
        recipe.pop("runtime")
        recipe["inputs"] = {"config": recipe["inputs"]["config"]}
        recipe["evaluation_policy"] = {"external_test_locked": True}
        recipe["artifacts"] = {"overwrite": False}
    recipe["_recipe_path"] = str(recipe_path.resolve())
    plan_contract.bind_plan_context(recipe)
    report = plans.DecisionReport(status=plans.DecisionStatus.PASS, issues=[], decisions={})
    monkeypatch.setattr(plans, "preflight_plan", lambda **_kwargs: (recipe, _bound_config_summary(recipe), report))
    marker = tmp_path / "runtime.txt"
    runtime_code = (
        "import sys; from pathlib import Path; "
        "from agent_tools.experiment_workspace import read_run_manifest; "
        "rows = read_run_manifest(sys.argv[1]); "
        "Path(sys.argv[2]).write_text(rows[0]['status'] + '\\n' + str(Path.cwd()))"
    )
    command = " ".join(shlex_quote(str(value)) for value in (sys.executable, "-c", runtime_code, workspace, marker))
    monkeypatch.setattr(plans, "_commands_for_recipe", lambda *_args, **_kwargs: [command])
    monkeypatch.setattr(plans.get_adapter(recipe["task"]), "frozen_commands", lambda *_args, **_kwargs: [command])
    plan_dir = workspace / "plan"

    assert plans.build_plan(recipe_path=recipe_path, output_dir=plan_dir).exit_code == 0
    outside = tmp_path / "outside"
    outside.mkdir()
    cwd = plan_dir if cwd_kind == "plan" else outside
    result = subprocess.run(["bash", str(plan_dir / "run.sh")], cwd=cwd, text=True, capture_output=True)

    assert result.returncode == 0, result.stderr
    assert marker.read_text().splitlines() == ["running", str(REPO_ROOT)]
    assert read_run_manifest(workspace)[0]["status"] == "completed"
    script = (plan_dir / "run.sh").read_text()
    assert f"cd {shlex_quote(str(REPO_ROOT))}" in script
    assert f"export PYTHONPATH={shlex_quote(str(REPO_ROOT))}${{PYTHONPATH:+:$PYTHONPATH}}" in script
    assert f"  {shlex_quote(sys.executable)} -c " in script
    final_report = tmp_path / "final.md"
    final_report.write_text("# Final\n\nManaged run completed.\n")
    assert experiments.finalize_experiment(workspace, final_report) == workspace / "reports" / "final.md"


def test_infer_plan_uses_frozen_runtime_python_for_workload_and_lifecycle(tmp_path: Path, monkeypatch):
    source = tmp_path / "source"
    recipe_path = write_finetune_recipe(source)
    recipe = yaml.safe_load(recipe_path.read_text())
    workspace = tmp_path / "workspace"
    runtime_python = "/runtime/bin/python"
    recipe.update({"task": "infer", "name": "unit_infer_runtime_identity"})
    recipe["experiment"]["root"] = str(workspace)
    recipe["step"] = {"id": "unit-infer", "phase": "evaluate", "purpose": "Exercise runtime identity."}
    recipe["decisions"]["task"] = {"value": "infer", "source": "explicit_recipe"}
    recipe["execution"] = {
        "target": "local",
        "workdir": str(REPO_ROOT),
        "python": runtime_python,
        "runtime_commit": _RUNTIME_COMMIT,
    }
    plan_contract.bind_plan_context(recipe)
    report = plans.DecisionReport(status=plans.DecisionStatus.PASS, issues=[], decisions={})
    monkeypatch.setattr(plans, "preflight_plan", lambda **_kwargs: (recipe, _bound_config_summary(recipe), report))
    command = f"{runtime_python} -m sleep2vec.infer --unit-runtime-identity"
    monkeypatch.setattr(plans, "_commands_for_recipe", lambda *_args, **_kwargs: [command])
    monkeypatch.setattr(plans.get_adapter(recipe["task"]), "frozen_commands", lambda *_args, **_kwargs: [command])
    plan_dir = workspace / "plan"

    assert plans.build_plan(recipe_path=recipe_path, output_dir=plan_dir).exit_code == 0

    lines = (plan_dir / "run.sh").read_text().splitlines()
    helper_index = lines.index("_agent_commit_status() {")
    running_index = lines.index("_agent_commit_status running")
    workload_index = lines.index(command)
    assert lines[helper_index + 1].startswith(f"  {runtime_python} -c ")
    assert any(line.startswith(f"{runtime_python} -c ") and _RUNTIME_COMMIT in line for line in lines)
    assert helper_index < running_index < workload_index
    plan = json.loads((plan_dir / "plan.json").read_text())
    assert plan["recipe"]["execution"] == recipe["execution"]
    assert yaml.safe_load((plan_dir / "recipe.resolved.yaml").read_text())["execution"] == recipe["execution"]


def test_infer_runtime_commit_mismatch_fails_before_running_or_payload(tmp_path: Path, monkeypatch):
    source = tmp_path / "source"
    recipe_path = write_finetune_recipe(source)
    recipe = yaml.safe_load(recipe_path.read_text())
    workspace = tmp_path / "workspace"
    marker = tmp_path / "payload-ran.txt"
    recipe.update({"task": "infer", "name": "unit_infer_runtime_commit_guard"})
    recipe["experiment"]["root"] = str(workspace)
    recipe["step"] = {"id": "unit-infer", "phase": "evaluate", "purpose": "Exercise commit guard."}
    recipe["decisions"]["task"] = {"value": "infer", "source": "explicit_recipe"}
    recipe["execution"] = {
        "target": "local",
        "workdir": str(REPO_ROOT),
        "python": sys.executable,
        "runtime_commit": "0" * 40,
    }
    plan_contract.bind_plan_context(recipe)
    report = plans.DecisionReport(status=plans.DecisionStatus.PASS, issues=[], decisions={})
    monkeypatch.setattr(plans, "preflight_plan", lambda **_kwargs: (recipe, _bound_config_summary(recipe), report))
    payload_code = "from pathlib import Path; Path(__import__('sys').argv[1]).write_text('ran')"
    command = " ".join(shlex_quote(str(value)) for value in (sys.executable, "-c", payload_code, marker))
    monkeypatch.setattr(plans, "_commands_for_recipe", lambda *_args, **_kwargs: [command])
    monkeypatch.setattr(plans.get_adapter(recipe["task"]), "frozen_commands", lambda *_args, **_kwargs: [command])
    plan_dir = workspace / "plan"

    assert plans.build_plan(recipe_path=recipe_path, output_dir=plan_dir).exit_code == 0
    result = subprocess.run(["bash", str(plan_dir / "run.sh")], text=True, capture_output=True)

    assert result.returncode != 0
    assert "Target runtime commit differs from the frozen plan" in result.stderr
    assert not marker.exists()
    assert read_run_manifest(workspace)[0]["status"] == "planned"


def test_non_hparam_run_script_records_failure_and_preserves_runtime_exit_code(tmp_path: Path, monkeypatch):
    source = tmp_path / "source"
    recipe_path = write_finetune_recipe(source)
    recipe = yaml.safe_load(recipe_path.read_text())
    workspace = tmp_path / "workspace"
    recipe["experiment"]["root"] = str(workspace)
    plan_contract.bind_plan_context(recipe)
    report = plans.DecisionReport(status=plans.DecisionStatus.PASS, issues=[], decisions={})
    monkeypatch.setattr(plans, "preflight_plan", lambda **_kwargs: (recipe, _bound_config_summary(recipe), report))
    command = " ".join(shlex_quote(str(value)) for value in (sys.executable, "-c", "import sys; sys.exit(7)"))
    monkeypatch.setattr(plans, "_commands_for_recipe", lambda *_args, **_kwargs: [command])
    monkeypatch.setattr(plans.get_adapter(recipe["task"]), "frozen_commands", lambda *_args, **_kwargs: [command])
    plan_dir = workspace / "plan"
    plans.build_plan(recipe_path=recipe_path, output_dir=plan_dir)

    result = subprocess.run(["bash", str(plan_dir / "run.sh")], cwd=plan_dir, text=True, capture_output=True)

    assert result.returncode == 7
    assert read_run_manifest(workspace)[0]["status"] == "failed"


def test_non_hparam_run_script_propagates_terminal_commit_failure(tmp_path: Path, monkeypatch):
    source = tmp_path / "source"
    recipe_path = write_finetune_recipe(source)
    recipe = yaml.safe_load(recipe_path.read_text())
    workspace = tmp_path / "workspace"
    recipe["experiment"]["root"] = str(workspace)
    plan_contract.bind_plan_context(recipe)
    report = plans.DecisionReport(status=plans.DecisionStatus.PASS, issues=[], decisions={})
    monkeypatch.setattr(plans, "preflight_plan", lambda **_kwargs: (recipe, _bound_config_summary(recipe), report))
    runtime_code = "import sys; from pathlib import Path; (Path(sys.argv[1]) / 'run_manifest.tsv').unlink()"
    command = " ".join(shlex_quote(str(value)) for value in (sys.executable, "-c", runtime_code, workspace))
    monkeypatch.setattr(plans, "_commands_for_recipe", lambda *_args, **_kwargs: [command])
    monkeypatch.setattr(plans.get_adapter(recipe["task"]), "frozen_commands", lambda *_args, **_kwargs: [command])
    plan_dir = workspace / "plan"
    plans.build_plan(recipe_path=recipe_path, output_dir=plan_dir)

    result = subprocess.run(["bash", str(plan_dir / "run.sh")], cwd=plan_dir, text=True, capture_output=True)

    assert result.returncode != 0
    assert "run_manifest.tsv" in result.stderr


def test_non_hparam_run_script_refuses_to_execute_terminal_run(tmp_path: Path, monkeypatch):
    source = tmp_path / "source"
    recipe_path = write_finetune_recipe(source)
    recipe = yaml.safe_load(recipe_path.read_text())
    workspace = tmp_path / "workspace"
    recipe["experiment"]["root"] = str(workspace)
    plan_contract.bind_plan_context(recipe)
    report = plans.DecisionReport(status=plans.DecisionStatus.PASS, issues=[], decisions={})
    monkeypatch.setattr(plans, "preflight_plan", lambda **_kwargs: (recipe, _bound_config_summary(recipe), report))
    marker = tmp_path / "runtime.txt"
    command = " ".join(
        shlex_quote(str(value))
        for value in (sys.executable, "-c", "import sys; from pathlib import Path; Path(sys.argv[1]).touch()", marker)
    )
    monkeypatch.setattr(plans, "_commands_for_recipe", lambda *_args, **_kwargs: [command])
    monkeypatch.setattr(plans.get_adapter(recipe["task"]), "frozen_commands", lambda *_args, **_kwargs: [command])
    plan_dir = workspace / "plan"
    plans.build_plan(recipe_path=recipe_path, output_dir=plan_dir)
    run = _first_run(plan_dir)
    merge_run_manifest(
        workspace,
        [{"step_id": run["step_id"], "run_id": run["run_id"], "status": "completed"}],
    )

    result = subprocess.run(["bash", str(plan_dir / "run.sh")], cwd=plan_dir, text=True, capture_output=True)

    assert result.returncode != 0
    assert not marker.exists()
    assert read_run_manifest(workspace)[0]["status"] == "completed"
