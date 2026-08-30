from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

from agent_tool_test_helpers import config_payload, run_execution_preflight_fixture, write_yaml
import pytest
import yaml

from agent_tools import (
    experiment_io,
    experiment_workspace,
    managed_scheduler,
    plan_rendering,
    plans,
    python_programs,
    run_artifacts,
    slurm,
)
from agent_tools.adapters import get_adapter
from agent_tools.models import REPO_ROOT


@pytest.fixture(autouse=True)
def _no_external_scheduler(tmp_path: Path, monkeypatch):
    commands = {"ssh", "sbatch", "squeue", "sacct", "scontrol", "scancel", "srun"}
    guarded_bin = tmp_path / "guarded-bin"
    guarded_bin.mkdir()
    attempted = tmp_path / "unexpected-scheduler-command"
    for name in commands:
        executable = guarded_bin / name
        executable.write_text(f"#!/bin/sh\nprintf '%s\\n' \"$0\" >> {shlex.quote(str(attempted))}\nexit 97\n")
        executable.chmod(0o755)
    monkeypatch.setenv("PATH", str(guarded_bin) + os.pathsep + os.environ.get("PATH", ""))
    original_popen = subprocess.Popen

    def guarded_popen(args, *positional, **kwargs):
        executable = kwargs.get("executable")
        if executable is None and isinstance(args, (list, tuple)) and args:
            executable = args[0]
        if executable is not None and Path(str(executable)).name in commands:
            resolved = shutil.which(str(executable), path=(kwargs.get("env") or os.environ).get("PATH"))
            assert resolved == str(guarded_bin / Path(str(executable)).name), "Genuine SSH/Slurm command is forbidden"
        return original_popen(args, *positional, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", guarded_popen)
    yield
    assert not attempted.exists(), "Planning must not invoke SSH or Slurm, including from a child shell"


@pytest.fixture
def _runtime_probe(monkeypatch):
    monkeypatch.setattr(managed_scheduler, "run_execution_command", run_execution_preflight_fixture)


@pytest.fixture(scope="module")
def _runtime_commit():
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True, text=True, capture_output=True
    ).stdout.strip()


def _infer_slurm_recipe(tmp_path: Path, *, variant: str, task: str, runtime_commit: str) -> Path:
    workspace = tmp_path / "workspace"
    inputs_dir = workspace / "inputs"
    inputs_dir.mkdir(parents=True)
    workdir = tmp_path / "runtime"
    workdir.mkdir()
    if variant == "sex_age_baseline":
        from tests.agent_tools.test_agent_sex_age_baseline import _write_survival_config

        config = _write_survival_config(inputs_dir)
        label = "incident_cox"
    else:
        index = inputs_dir / "index.csv"
        index.write_text("path,split,duration,ppg_mask,ah_event_mask,stage_mask\nx.npz,val,60,1,1,1\n")
        config = write_yaml(inputs_dir / "config.yaml", config_payload(index))
        label = "ahi"
    checkpoint = inputs_dir / "model.ckpt"
    checkpoint.write_text("unit checkpoint input, never loaded by a model\n")
    return write_yaml(
        workspace / "recipe.yaml",
        {
            "name": "unit_managed_infer",
            "task": task,
            "variant": variant,
            "step": {"id": "unit-infer", "phase": "evaluate", "purpose": "Exercise ordinary managed inference."},
            "inputs": {
                "config": str(config),
                "ckpt_path": str(checkpoint),
                "label_name": label,
                "eval_split": "val",
            },
            "runtime": {"batch_size": 2, "num_workers": 0},
            "execution": {
                "target": "local",
                "workdir": str(workdir),
                "python": sys.executable,
                "runtime_commit": runtime_commit,
                "gpus_per_run": 1,
                "scheduler": {
                    "type": "slurm",
                    "partition": "unit-gpu",
                    "cpus_per_task": 1,
                    "memory": "1G",
                    "walltime": "00:01:00",
                },
            },
            "artifacts": {"overwrite": False},
            "evaluation_policy": {"external_test_locked": True, "final_test_unlocked": False},
            "decisions": {
                "task": {"value": task, "source": "explicit_recipe"},
                "label_name": {"value": label, "source": "explicit_recipe"},
                "ckpt_path": {"value": str(checkpoint), "source": "explicit_recipe"},
                "eval_split": {"value": "val", "source": "explicit_recipe"},
                "final_eval_unlock": {"value": False, "source": "explicit_recipe"},
                "overwrite_policy": {"value": False, "source": "explicit_recipe"},
            },
        },
    )


def _read_registered_infer_plan(plan_dir: Path):
    workspace = plan_dir.parent
    rows = experiment_workspace.read_run_manifest(workspace)
    (row,) = rows
    experiment_path = workspace / "experiment.yaml"
    experiment = experiment_workspace.read_managed_yaml_mapping(experiment_path.read_text(), source=experiment_path)[
        "experiment"
    ]
    step = experiment_workspace.read_step_manifest(workspace, row["step_id"])
    assert step is not None
    return run_artifacts.read_registered_plan(
        plan_dir,
        workspace=workspace,
        workspace_experiment=experiment,
        step_manifest=step,
        workspace_rows=rows,
        expected_recipe_path=str(workspace / "recipe.yaml"),
    )


@pytest.mark.parametrize("variant", ["sleep2vec", "sleep2vec2", "sleep2expert", "sex_age_baseline"])
@pytest.mark.parametrize("task", ["infer", "evaluate"])
def test_infer_slurm_round_trip_preserves_unsubmitted_identity(
    tmp_path: Path, variant: str, task: str, _runtime_commit, _runtime_probe
):
    recipe_path = _infer_slurm_recipe(tmp_path, variant=variant, task=task, runtime_commit=_runtime_commit)
    workspace = recipe_path.parent
    plan_dir = workspace / "plan"

    result = plans.build_plan(recipe_path=recipe_path, output_dir=plan_dir)

    assert result.exit_code == 0, [(issue.field, issue.message) for issue in result.blocking_issues()]
    plan = json.loads((plan_dir / "plan.json").read_text())
    (run,) = plan["runs"]
    (row,) = experiment_workspace.read_run_manifest(workspace)
    assert row["status"] == "planned"
    assert row["run_id"] == "run-000"
    assert row["scheduler_type"] == "slurm"
    assert row["terminal_status_owner"] == "scheduler_sidecar"
    for field in ("target", "host", "workdir", "command", "scheduler_job_id", "scheduler_cluster"):
        assert row.get(field, "") == ""
    assert not experiment_workspace.has_managed_launch_evidence(row)
    for field in (
        "config",
        "config_sha256",
        "script",
        "script_sha256",
        "scheduler_script",
        "scheduler_script_sha256",
        "scheduler_submit_token",
        "scheduler_result_path",
        "allocation_identity_path",
        "log_path",
    ):
        assert row[field] == run[field]
        assert row[field]

    validated = _read_registered_infer_plan(plan_dir)

    assert validated["recipe"] == plan["recipe"]
    assert validated["runs"] == plan["runs"]
    frozen_recipe = validated["recipe"]
    assert frozen_recipe["runtime"]["devices"] == [0]
    adapter = get_adapter(task)
    assert adapter is not None
    contract = adapter.compile_plan_contract(
        frozen_recipe, plan_dir, run_index_offset=0, config_bytes=Path(run["config"]).read_bytes()
    )
    assert contract == adapter.compile_plan_contract(
        frozen_recipe, plan_dir, run_index_offset=0, config_bytes=Path(run["config"]).read_bytes()
    )
    manager_text = (plan_dir / "run.sh").read_text()
    worker_text = Path(run["script"]).read_text()
    scheduler_text = Path(run["scheduler_script"]).read_text()
    assert manager_text == contract["launch_script_text"]
    assert worker_text == contract["script_text"]
    assert scheduler_text == contract["scheduler_script_text"]
    assert manager_text != worker_text
    assert "infer-launch" in manager_text
    assert "_agent_commit_status" not in worker_text
    assert "SLURM_JOB_ID" in worker_text
    assert "run-frozen-job" in scheduler_text

    (command,) = plan["commands"]
    assert command == run["command"]
    assert command == json.loads((Path(run["run_dir"]) / "run.json").read_text())["command"]
    assert worker_text.splitlines().count(command) == 1
    assert shlex.split(command)[:3] == [sys.executable, "-m", f"{variant}.infer"]
    for path_field, digest_field in (
        ("config", "config_sha256"),
        ("script", "script_sha256"),
        ("scheduler_script", "scheduler_script_sha256"),
    ):
        assert hashlib.sha256(Path(run[path_field]).read_bytes()).hexdigest() == run[digest_field]
    execution = frozen_recipe["execution"]
    resources = slurm.normalize_resources(execution["scheduler"], execution["gpus_per_run"])
    assert run["scheduler_submit_token"] == slurm.submit_token(run, resources, execution["runtime_commit"])


def test_infer_slurm_derives_multiple_logical_devices(tmp_path: Path, _runtime_commit, _runtime_probe):
    recipe_path = _infer_slurm_recipe(tmp_path, variant="sleep2vec", task="infer", runtime_commit=_runtime_commit)
    recipe = yaml.safe_load(recipe_path.read_text())
    recipe["execution"]["gpus_per_run"] = 2
    recipe_path.write_text(yaml.safe_dump(recipe))
    plan_dir = recipe_path.parent / "plan"

    assert plans.build_plan(recipe_path=recipe_path, output_dir=plan_dir).exit_code == 0

    validated = _read_registered_infer_plan(plan_dir)
    assert validated["recipe"]["runtime"]["devices"] == [0, 1]
    (run,) = validated["runs"]
    scheduler_text = Path(run["scheduler_script"]).read_text()
    assert "#SBATCH --nodes=1\n" in scheduler_text
    assert "#SBATCH --ntasks=2\n" in scheduler_text
    assert "#SBATCH --gres=gpu:2\n" in scheduler_text


@pytest.mark.parametrize(
    ("variant", "section", "field", "value", "expected_field"),
    [
        ("sex_age_baseline", "execution", "gpus_per_run", 2, "execution.gpus_per_run"),
        ("sleep2vec", "runtime", "devices", [2], "runtime.devices"),
        ("sleep2vec", "runtime", "accelerator", "cpu", "runtime.accelerator"),
        ("sleep2vec", "execution", "workdir", None, "execution.workdir"),
        ("sleep2vec", "execution", "python", None, "execution.python"),
        ("sleep2vec", "execution", "runtime_commit", None, "execution.runtime_commit"),
    ],
)
def test_infer_slurm_rejects_invalid_execution_contract(
    tmp_path: Path, variant, section, field, value, expected_field, _runtime_commit, _runtime_probe
):
    recipe_path = _infer_slurm_recipe(tmp_path, variant=variant, task="infer", runtime_commit=_runtime_commit)
    recipe = yaml.safe_load(recipe_path.read_text())
    if value is None:
        recipe[section].pop(field)
    else:
        recipe[section][field] = value
    recipe_path.write_text(yaml.safe_dump(recipe))
    plan_dir = recipe_path.parent / "plan"

    result = plans.build_plan(recipe_path=recipe_path, output_dir=plan_dir)

    assert result.exit_code == 1
    assert expected_field in {issue.field for issue in result.issues}
    assert not (plan_dir / "plan.json").exists()
    assert experiment_workspace.read_run_manifest(recipe_path.parent) == []


@pytest.mark.parametrize("artifact", ["manager", "worker", "scheduler"])
def test_infer_slurm_reader_rejects_changed_script(tmp_path: Path, artifact: str, _runtime_commit, _runtime_probe):
    recipe_path = _infer_slurm_recipe(tmp_path, variant="sleep2vec", task="infer", runtime_commit=_runtime_commit)
    plan_dir = recipe_path.parent / "plan"
    assert plans.build_plan(recipe_path=recipe_path, output_dir=plan_dir).exit_code == 0
    (run,) = _read_registered_infer_plan(plan_dir)["runs"]
    path = {
        "manager": plan_dir / "run.sh",
        "worker": Path(run["script"]),
        "scheduler": Path(run["scheduler_script"]),
    }[artifact]
    path.write_text(path.read_text() + "# unregistered edit\n")
    manifest_path = recipe_path.parent / "run_manifest.tsv"
    before = manifest_path.read_bytes()

    with pytest.raises(ValueError, match="differs|changed"):
        _read_registered_infer_plan(plan_dir)

    assert manifest_path.read_bytes() == before


def test_infer_slurm_reader_recompiles_coherently_edited_scheduler(tmp_path: Path, _runtime_commit, _runtime_probe):
    recipe_path = _infer_slurm_recipe(tmp_path, variant="sleep2vec", task="evaluate", runtime_commit=_runtime_commit)
    plan_dir = recipe_path.parent / "plan"
    assert plans.build_plan(recipe_path=recipe_path, output_dir=plan_dir).exit_code == 0
    plan_path = plan_dir / "plan.json"
    plan = json.loads(plan_path.read_text())
    (run,) = plan["runs"]
    scheduler_path = Path(run["scheduler_script"])
    scheduler_path.write_text(scheduler_path.read_text() + "# coherent but unauthorized edit\n")
    edited_digest = hashlib.sha256(scheduler_path.read_bytes()).hexdigest()
    run["scheduler_script_sha256"] = edited_digest
    plan_path.write_text(json.dumps(plan))
    run_path = Path(run["run_dir"]) / "run.json"
    run_payload = json.loads(run_path.read_text())
    run_payload["scheduler_script_sha256"] = edited_digest
    run_path.write_text(json.dumps(run_payload))
    rows = experiment_workspace.read_run_manifest(recipe_path.parent)
    rows[0]["scheduler_script_sha256"] = edited_digest
    experiment_io.write_rows_at(recipe_path.parent / "run_manifest.tsv", rows)

    with pytest.raises(ValueError, match="scheduler_script_sha256|Slurm script"):
        _read_registered_infer_plan(plan_dir)


@pytest.mark.parametrize(
    "identity_case",
    ["matching", "missing-allocation", "missing-job", "wrong-job", "wrong-token", "wrong-cluster", "wrong-run"],
)
def test_infer_slurm_worker_authenticates_allocation_before_workload(
    tmp_path: Path, identity_case: str, _runtime_commit, _runtime_probe
):
    recipe_path = _infer_slurm_recipe(tmp_path, variant="sleep2vec", task="infer", runtime_commit=_runtime_commit)
    plan_dir = recipe_path.parent / "plan"
    assert plans.build_plan(recipe_path=recipe_path, output_dir=plan_dir).exit_code == 0
    validated = _read_registered_infer_plan(plan_dir)
    (run,) = validated["runs"]
    worker_text = Path(run["script"]).read_text()
    words = shlex.split(worker_text, comments=True)
    guard_index = words.index(python_programs.source("plan_rendering.slurm_allocation_guard"))
    guard_argv = words[guard_index - 2 : guard_index + 5]
    assert guard_argv[:2] == [sys.executable, "-c"]
    assert worker_text.index(plan_rendering.render_command(guard_argv)) < worker_text.index(run["command"])

    allocation = {
        "schema_version": 1,
        "scheduler_job_id": "3880",
        "scheduler_cluster": "unit-cluster",
        "scheduler_submit_token": run["scheduler_submit_token"],
    }
    env = {
        **os.environ,
        "SLURM_JOB_ID": "3880",
        "SLURM_CLUSTER_NAME": "unit-cluster",
        "PYTHONPATH": str(REPO_ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""),
    }
    if identity_case == "missing-job":
        env.pop("SLURM_JOB_ID")
    elif identity_case == "wrong-job":
        allocation["scheduler_job_id"] = "3881"
    elif identity_case == "wrong-token":
        allocation["scheduler_submit_token"] = "another-frozen-run"
    elif identity_case == "wrong-cluster":
        allocation["scheduler_cluster"] = "another-cluster"
    elif identity_case == "wrong-run":
        run_path = Path(run["run_dir"]) / "run.json"
        run_payload = json.loads(run_path.read_text())
        run_payload["run_id"] = "run-001"
        run_path.write_text(json.dumps(run_payload))
    if identity_case != "missing-allocation":
        Path(run["allocation_identity_path"]).write_text(json.dumps(allocation))
    manifest_path = recipe_path.parent / "run_manifest.tsv"
    before = manifest_path.read_bytes()

    # Execute the exact generated guard only; even its success case must not start the model or srun.
    result = subprocess.run(
        guard_argv,
        cwd=validated["recipe"]["execution"]["workdir"],
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
    )

    if identity_case == "matching":
        assert result.returncode == 0, result.stderr
    else:
        assert result.returncode != 0
    assert manifest_path.read_bytes() == before
