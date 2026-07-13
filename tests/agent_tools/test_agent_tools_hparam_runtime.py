from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from agent_tool_test_helpers import write_finetune_recipe, write_yaml
import pytest
import yaml

from agent_tools import hparam_runtime, manifests, run_evidence
from agent_tools.experiment_workspace import file_sha256, merge_run_manifest, merge_run_row
from agent_tools.hparam_runtime import monitor_hparam_runs


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "agent_tools", *args], text=True, capture_output=True)


def _hparam_recipe(tmp_path: Path, *, execution: dict | None = None) -> Path:
    base = write_finetune_recipe(tmp_path)
    return write_yaml(
        tmp_path / "tune.yaml",
        {
            "name": "unit_hparam",
            "task": "hparam_tune",
            "variant": "sleep2vec",
            "base_recipe": str(base),
            "search": {
                "method": "grid",
                "max_runs": 1,
                "parameters": {"runtime.lr": [1e-6]},
            },
            "execution": execution if execution is not None else {"workdir": str(tmp_path)},
            "evaluation_policy": {
                "selection_metric": "val_ahi_pearson",
                "selection_mode": "max",
                "selection_split": "val",
                "external_test_locked": True,
                "test_after_fit": False,
                "final_eval_split": "test",
                "final_test_unlocked": False,
                "require_manual_unlock_for_final_test": True,
            },
            "decisions": {
                "task": {"value": "hparam_tune", "source": "explicit_recipe"},
                "label_name": {"value": "ahi", "source": "explicit_recipe"},
                "external_test_locked": {"value": True, "source": "explicit_recipe"},
                "train_val_test_policy": {
                    "value": "select on val",
                    "source": "explicit_recipe",
                },
                "overwrite_policy": {"value": False, "source": "explicit_recipe"},
                "final_eval_unlock": {"value": False, "source": "explicit_recipe"},
            },
        },
    )


def _read_table(path: Path) -> list[dict[str, str]]:
    delimiter = "\t" if path.suffix == ".tsv" else ","
    with path.open(newline="") as file_obj:
        return list(csv.DictReader(file_obj, delimiter=delimiter))


def _write_runtime_rows(root: Path, specs: list[dict]) -> list[dict]:
    experiment = {
        "id": "unit-experiment",
        "title": "Unit experiment",
        "objective": "Exercise hparam runtime state transitions.",
        "root": str(root),
        "baseline": {"type": "none", "rationale": "Unit fixture."},
    }
    step = {"id": "train-model", "phase": "train", "purpose": "Exercise managed runs."}
    (root / "experiment.yaml").write_text(yaml.safe_dump({"experiment": experiment}, sort_keys=False))
    step_dir = root / "steps" / step["id"]
    step_dir.mkdir(parents=True, exist_ok=True)
    (step_dir / "step.yaml").write_text(
        yaml.safe_dump(
            {
                "step": step,
                "experiment_id": experiment["id"],
                "recipe_path": "",
                "plans": [str(root.resolve())],
            },
            sort_keys=False,
        )
    )
    runs = []
    rows = []
    for index, spec in enumerate(specs):
        run_id = str(spec["run_id"])
        managed_dir = root / "runs" / run_id
        managed_dir.mkdir(parents=True, exist_ok=True)
        config = managed_dir / "config.yaml"
        script = managed_dir / "launch.sh"
        artifacts_path = managed_dir / "artifacts.json"
        config.write_text("model: unit\n")
        script.write_text("#!/usr/bin/env bash\ntrue\n")
        artifacts_path.write_text("{}\n")
        version = str(spec.get("version") or f"version-{index}")
        runtime_dir = root / "log-finetune" / version
        run = {
            "experiment_id": "unit-experiment",
            "step_id": "train-model",
            "run_id": run_id,
            "run_name": run_id,
            "version": version,
            "run_dir": str(managed_dir),
            "runtime_dir": str(runtime_dir),
            "checkpoint_dir": str(runtime_dir / "checkpoints"),
            "config": str(config),
            "config_sha256": file_sha256(config),
            "script": str(script),
            "script_sha256": file_sha256(script),
            "artifacts": str(artifacts_path),
        }
        runs.append(run)
        row = {
            **run,
            "target": "local",
            "host": "",
            "workdir": str(root),
            "pid_path": str(managed_dir / "pid"),
            "log_path": str(managed_dir / "stdout.log"),
            "command": f"run {run_id}",
            "status": "planned",
            "launched_at": "",
            **spec,
        }
        rows.append(row)
    (root / "plan.json").write_text(
        json.dumps(
            {
                "runs": runs,
                "recipe": {
                    "experiment": experiment,
                    "step": step,
                    "execution": {"workdir": str(root)},
                },
            }
        )
    )
    manifests.write_rows(
        root / "run_manifest.tsv",
        [{**run, "status": row["status"]} for run, row in zip(runs, rows)],
    )
    manifests.write_rows(root / "launch_manifest.tsv", rows)
    manifests.write_rows(root / "run_status.tsv", rows)
    return rows


def test_hparam_launch_rejects_plan_without_workspace_binding_before_start(tmp_path: Path, monkeypatch):
    recipe = _hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    plan = json.loads((plan_dir / "plan.json").read_text())
    plan["recipe"].pop("experiment")
    (plan_dir / "plan.json").write_text(json.dumps(plan))
    started = []
    monkeypatch.setattr(
        hparam_runtime, "_start_process", lambda _execution, command: started.append(command) or "launched"
    )

    with pytest.raises(ValueError, match="workspace binding"):
        hparam_runtime.launch_hparam_runs(plan_dir, dry_run=False)

    assert started == []
    assert not (plan_dir / "launch_manifest.tsv").exists()


def test_hparam_plan_canonicalizes_relative_workspace_root_consistently(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    payload["experiment"]["root"] = os.path.relpath(tmp_path, Path.cwd())
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
    plan_dir = tmp_path / "plan"

    plan_result = _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir))
    launch_result = _run("hparam-launch", "--plan-dir", str(plan_dir))

    assert plan_result.returncode == 0, plan_result.stderr
    assert launch_result.returncode == 0, launch_result.stderr
    plan = json.loads((plan_dir / "plan.json").read_text())
    manifest = yaml.safe_load((tmp_path / "experiment.yaml").read_text())
    assert plan["recipe"]["experiment"]["root"] == str(tmp_path)
    assert manifest["experiment"]["root"] == str(tmp_path)


def test_registered_step_remains_canonical_through_plan_and_dry_run_launch(tmp_path: Path):
    source = tmp_path / "source"
    recipe = _hparam_recipe(source)
    payload = yaml.safe_load(recipe.read_text())
    workspace = tmp_path / "workspace"
    payload["experiment"]["root"] = str(workspace)
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
    experiment_spec = tmp_path / "experiment.yaml"
    experiment_spec.write_text(yaml.safe_dump(payload["experiment"], sort_keys=False))
    step_spec = tmp_path / "step.yaml"
    step_spec.write_text(
        yaml.safe_dump(
            {
                **payload["step"],
                "inputs": ["reports/ranking.csv"],
                "outputs": ["reports/final.md"],
            },
            sort_keys=False,
        )
    )
    plan_dir = workspace / "plans" / "hparam"

    initialized = _run("experiment-init", "--run-dir", str(workspace), "--spec", str(experiment_spec))
    registered = _run("experiment-register-step", "--run-dir", str(workspace), "--spec", str(step_spec))
    planned = _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir))
    launched = _run("hparam-launch", "--plan-dir", str(plan_dir))

    assert initialized.returncode == 0, initialized.stderr
    assert registered.returncode == 0, registered.stderr
    assert planned.returncode == 0, planned.stderr
    assert launched.returncode == 0, launched.stderr
    step_manifest = yaml.safe_load((workspace / "steps" / payload["step"]["id"] / "step.yaml").read_text())
    assert step_manifest["step"]["inputs"] == ["reports/ranking.csv"]
    assert step_manifest["step"]["outputs"] == ["reports/final.md"]
    assert step_manifest["experiment_id"] == payload["experiment"]["id"]
    assert step_manifest["recipe_path"] == str(recipe)
    assert step_manifest["plans"] == [str(plan_dir)]
    events = [json.loads(line) for line in (workspace / "events.jsonl").read_text().splitlines()]
    assert [event["event_type"] for event in events].count("step_registered") == 1


def test_hparam_launch_rejects_unregistered_plan_copy_before_start(tmp_path: Path, monkeypatch):
    recipe = _hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    copied_plan = tmp_path / "copied-plan"
    shutil.copytree(plan_dir, copied_plan)
    started = []
    monkeypatch.setattr(
        hparam_runtime, "_start_process", lambda _execution, command: started.append(command) or "launched"
    )

    with pytest.raises(ValueError, match="not registered"):
        hparam_runtime.launch_hparam_runs(copied_plan, dry_run=False)

    assert started == []
    assert not (copied_plan / "launch_manifest.tsv").exists()


def test_hparam_launch_rejects_completed_experiment_without_writes(tmp_path: Path, monkeypatch):
    recipe = _hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    experiment_path = tmp_path / "experiment.yaml"
    experiment_manifest = yaml.safe_load(experiment_path.read_text())
    experiment_manifest["experiment"]["status"] = "completed"
    experiment_path.write_text(yaml.safe_dump(experiment_manifest, sort_keys=False))
    run_rows = _read_table(tmp_path / "run_manifest.tsv")
    run_rows[0]["status"] = "completed"
    manifests.write_rows(tmp_path / "run_manifest.tsv", run_rows)
    before = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    started = []
    monkeypatch.setattr(
        hparam_runtime, "_start_process", lambda _execution, command: started.append(command) or "launched"
    )

    with pytest.raises(ValueError, match="Experiment is completed"):
        hparam_runtime.launch_hparam_runs(plan_dir, dry_run=False)

    after = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    assert started == []
    assert after == before


def test_hparam_launch_does_not_restart_workspace_terminal_run(tmp_path: Path, monkeypatch):
    recipe = _hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    hparam_runtime.launch_hparam_runs(plan_dir, dry_run=True)
    workspace_rows = _read_table(tmp_path / "run_manifest.tsv")
    workspace_rows[0]["status"] = "failed"
    manifests.write_rows(tmp_path / "run_manifest.tsv", workspace_rows)
    started = []
    monkeypatch.setattr(
        hparam_runtime, "_start_process", lambda _execution, command: started.append(command) or "launched"
    )

    hparam_runtime.launch_hparam_runs(plan_dir, dry_run=False)

    assert started == []
    assert _read_table(plan_dir / "run_status.tsv")[0]["status"] == "failed"


@pytest.mark.parametrize(
    ("local_status_exists", "launch_exists", "expected_status"),
    [
        (False, False, "running"),
        (False, True, "finished"),
        (True, False, "running"),
        (True, True, "finished"),
    ],
)
def test_hparam_launch_does_not_restart_canonical_running_run_from_stale_mirrors(
    tmp_path: Path,
    monkeypatch,
    local_status_exists: bool,
    launch_exists: bool,
    expected_status: str,
):
    _write_runtime_rows(tmp_path, [{"run_id": "run-000", "status": "planned"}])
    workspace_rows = _read_table(tmp_path / "run_manifest.tsv")
    workspace_rows[0]["status"] = "running"
    manifests.write_rows(tmp_path / "run_manifest.tsv", workspace_rows)
    if not local_status_exists:
        (tmp_path / "run_status.tsv").unlink()
    if not launch_exists:
        (tmp_path / "launch_manifest.tsv").unlink()
    started = []
    monkeypatch.setattr(
        hparam_runtime, "_start_process", lambda _execution, command: started.append(command) or "launched"
    )

    hparam_runtime.launch_hparam_runs(tmp_path, dry_run=False)

    assert started == []
    assert _read_table(tmp_path / "launch_manifest.tsv")[0]["status"] == expected_status
    assert _read_table(tmp_path / "run_status.tsv")[0]["status"] == expected_status
    assert _read_table(tmp_path / "run_manifest.tsv")[0]["status"] == expected_status


@pytest.mark.parametrize("local_status", ["launched", "unknown_remote"])
def test_hparam_launch_ignores_stale_local_status(tmp_path: Path, monkeypatch, local_status: str):
    _write_runtime_rows(tmp_path, [{"run_id": "run-000", "status": local_status}])
    workspace_rows = _read_table(tmp_path / "run_manifest.tsv")
    workspace_rows[0]["status"] = "running"
    workspace_rows[0]["score"] = "0.9"
    manifests.write_rows(tmp_path / "run_manifest.tsv", workspace_rows)
    local_rows = _read_table(tmp_path / "run_status.tsv")
    local_rows[0]["score"] = "0.1"
    manifests.write_rows(tmp_path / "run_status.tsv", local_rows)
    (tmp_path / "launch_manifest.tsv").unlink()
    started = []
    monkeypatch.setattr(
        hparam_runtime,
        "_start_process",
        lambda _execution, command: started.append(command) or "launched",
    )

    hparam_runtime.launch_hparam_runs(tmp_path, dry_run=False)

    assert started == []
    assert _read_table(tmp_path / "run_manifest.tsv")[0]["status"] == "running"
    assert _read_table(tmp_path / "run_manifest.tsv")[0]["score"] == "0.9"
    assert _read_table(tmp_path / "run_status.tsv")[0]["status"] == "running"
    assert _read_table(tmp_path / "launch_manifest.tsv")[0]["status"] == "running"


@pytest.mark.parametrize("operation", ["launch", "monitor", "stop"])
def test_hparam_runtime_does_not_reapply_stale_launch_snapshot_fields(tmp_path: Path, monkeypatch, operation: str):
    _write_runtime_rows(tmp_path, [{"run_id": "run-000", "status": "running"}])
    launch_rows = _read_table(tmp_path / "launch_manifest.tsv")
    launch_rows[0].update({"score": "0.1", "wandb_url": "https://wandb.example/stale"})
    manifests.write_rows(tmp_path / "launch_manifest.tsv", launch_rows)
    canonical_rows = _read_table(tmp_path / "run_manifest.tsv")
    canonical_rows[0].update({"score": "0.9", "wandb_url": "https://wandb.example/current"})
    manifests.write_rows(tmp_path / "run_manifest.tsv", canonical_rows)
    monkeypatch.setattr(
        run_evidence,
        "status_row",
        lambda _root, row, previous, health=False: merge_run_row(previous, row),
    )
    monkeypatch.setattr(run_evidence, "read_pid", lambda _path, _row: 123)
    monkeypatch.setattr(hparam_runtime.os, "kill", lambda _pid, _signal: None)

    if operation == "launch":
        hparam_runtime.launch_hparam_runs(tmp_path, dry_run=False)
    elif operation == "monitor":
        hparam_runtime.monitor_hparam_runs(tmp_path)
    else:
        hparam_runtime.stop_hparam_run(tmp_path, "run-000", reason="manual stop")

    canonical = _read_table(tmp_path / "run_manifest.tsv")[0]
    assert canonical["score"] == "0.9"
    assert canonical["wandb_url"] == "https://wandb.example/current"


def test_hparam_launch_records_event_only_for_a_process_started_by_that_call(tmp_path: Path, monkeypatch):
    recipe = _hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    started = []
    monkeypatch.setattr(
        hparam_runtime, "_start_process", lambda _execution, command: started.append(command) or "launched"
    )

    hparam_runtime.launch_hparam_runs(plan_dir, dry_run=True)
    hparam_runtime.launch_hparam_runs(plan_dir, dry_run=False)
    hparam_runtime.launch_hparam_runs(plan_dir, dry_run=False)

    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert started and len(started) == 1
    assert [event["event_type"] for event in events].count("run_launched") == 1


def test_hparam_launch_mirrors_the_status_committed_by_the_canonical_owner(tmp_path: Path, monkeypatch):
    _write_runtime_rows(tmp_path, [{"run_id": "run-000", "status": "planned"}])
    real_merge = merge_run_manifest

    def merge_after_wandb_update(root, rows):
        real_merge(root, [{"step_id": "train-model", "run_id": "run-000", "status": "failed"}])
        return real_merge(root, rows)

    monkeypatch.setattr(hparam_runtime, "merge_run_manifest", merge_after_wandb_update)
    monkeypatch.setattr(hparam_runtime, "_start_process", lambda _execution, _command: "launched")

    hparam_runtime.launch_hparam_runs(tmp_path, dry_run=False)

    assert _read_table(tmp_path / "run_manifest.tsv")[0]["status"] == "failed"
    assert _read_table(tmp_path / "run_status.tsv")[0]["status"] == "failed"
    assert _read_table(tmp_path / "launch_manifest.tsv")[0]["status"] == "failed"
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert [event["event_type"] for event in events].count("run_launched") == 1


def test_hparam_launch_failure_does_not_record_launched_event(tmp_path: Path, monkeypatch):
    recipe = _hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    monkeypatch.setattr(hparam_runtime, "_start_process", lambda _execution, _command: "launch_failed")

    hparam_runtime.launch_hparam_runs(plan_dir, dry_run=False)

    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert "run_launched" not in [event["event_type"] for event in events]


def test_hparam_launch_rejects_workspace_frozen_drift_before_start(tmp_path: Path, monkeypatch):
    recipe = _hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    workspace_rows = _read_table(tmp_path / "run_manifest.tsv")
    workspace_rows[0]["config_sha256"] = "changed"
    manifests.write_rows(tmp_path / "run_manifest.tsv", workspace_rows)
    started = []
    monkeypatch.setattr(
        hparam_runtime, "_start_process", lambda _execution, command: started.append(command) or "launched"
    )

    with pytest.raises(ValueError, match="config_sha256"):
        hparam_runtime.launch_hparam_runs(plan_dir, dry_run=False)

    assert started == []
    assert not (plan_dir / "launch_manifest.tsv").exists()


@pytest.mark.parametrize("target_name", ["run_matrix.csv", "reports/run_matrix.md"])
@pytest.mark.parametrize("target_kind", ["directory", "hardlink"])
def test_hparam_launch_rejects_invalid_canonical_output_before_start(
    tmp_path: Path, monkeypatch, target_name: str, target_kind: str
):
    recipe = _hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    target = tmp_path / target_name
    target.unlink()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target_kind == "directory":
        target.mkdir()
    else:
        target.hardlink_to(tmp_path / "run_manifest.tsv")
    before = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    started = []
    monkeypatch.setattr(
        hparam_runtime,
        "_start_process",
        lambda _execution, command: started.append(command) or "launched",
    )

    with pytest.raises(ValueError, match="Managed output"):
        hparam_runtime.launch_hparam_runs(plan_dir, dry_run=False)

    assert started == []
    assert not (plan_dir / "launch_manifest.tsv").exists()
    assert not (plan_dir / "run_status.tsv").exists()
    assert {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before


def test_hparam_launch_rejects_aliased_local_mirrors_before_start(tmp_path: Path, monkeypatch):
    recipe = _hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    hparam_runtime.launch_hparam_runs(plan_dir, dry_run=True)
    launch_path = plan_dir / "launch_manifest.tsv"
    status_path = plan_dir / "run_status.tsv"
    status_path.unlink()
    status_path.hardlink_to(launch_path)
    before = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    started = []
    monkeypatch.setattr(
        hparam_runtime,
        "_start_process",
        lambda _execution, command: started.append(command) or "launched",
    )

    with pytest.raises(ValueError, match="Managed output"):
        hparam_runtime.launch_hparam_runs(plan_dir, dry_run=False)

    assert started == []
    assert {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before


def test_hparam_launch_rejects_aliased_status_report_before_start(tmp_path: Path, monkeypatch):
    recipe = _hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    status_report = tmp_path / "reports" / "status.md"
    status_report.hardlink_to(tmp_path / "README.md")
    before = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    started = []
    monkeypatch.setattr(
        hparam_runtime,
        "_start_process",
        lambda _execution, command: started.append(command) or "launched",
    )

    with pytest.raises(ValueError, match="Managed output"):
        hparam_runtime.launch_hparam_runs(plan_dir, dry_run=False)

    assert started == []
    assert {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before


@pytest.mark.parametrize("target_kind", ["run_dir_symlink", "log_hardlink", "pid_hardlink"])
def test_hparam_launch_rejects_aliased_run_outputs_before_start(tmp_path: Path, monkeypatch, target_kind: str):
    recipe = _hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    run = json.loads((plan_dir / "plan.json").read_text())["runs"][0]
    semantic_run_dir = Path(run["run_dir"])
    protected = plan_dir / "plan.json"
    if target_kind == "run_dir_symlink":
        real_run_dir = plan_dir / "real-run-dir"
        semantic_run_dir.rename(real_run_dir)
        semantic_run_dir.symlink_to(real_run_dir, target_is_directory=True)
    else:
        output = semantic_run_dir / ("stdout.log" if target_kind == "log_hardlink" else "pid")
        output.hardlink_to(protected)
    protected_before = protected.read_bytes()
    started = []
    monkeypatch.setattr(
        hparam_runtime,
        "_start_process",
        lambda _execution, command: started.append(command) or "launched",
    )

    with pytest.raises(ValueError, match="Managed output"):
        hparam_runtime.launch_hparam_runs(plan_dir, dry_run=False)

    assert started == []
    assert protected.read_bytes() == protected_before
    assert not (plan_dir / "launch_manifest.tsv").exists()
    assert not (plan_dir / "run_status.tsv").exists()


def test_hparam_ssh_launch_validates_run_outputs_remotely_before_start(tmp_path: Path, monkeypatch):
    recipe = _hparam_recipe(
        tmp_path,
        execution={
            "target": "ssh",
            "host": "unit-host",
            "workdir": str(tmp_path),
            "max_concurrent": 1,
        },
    )
    plan_dir = tmp_path / "plan"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    remote_calls = []

    def reject_remote_output(command, **kwargs):
        remote_calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 2, "", "aliased output")

    started = []
    monkeypatch.setattr(hparam_runtime.exp_io.subprocess, "run", reject_remote_output)
    monkeypatch.setattr(
        hparam_runtime,
        "_start_process",
        lambda _execution, command: started.append(command) or "launched",
    )

    with pytest.raises(ValueError, match="aliased output"):
        hparam_runtime.launch_hparam_runs(plan_dir, dry_run=False)

    assert started == []
    assert remote_calls[0][0][:2] == ["ssh", "unit-host"]
    assert remote_calls[0][1]["timeout"] == hparam_runtime.exp_io.SSH_TIMEOUT_SECONDS


def test_hparam_runtime_rejects_tampered_relative_workdir_before_start(tmp_path: Path, monkeypatch):
    recipe = _hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    plan_path = plan_dir / "plan.json"
    plan = json.loads(plan_path.read_text())
    plan["recipe"]["execution"] = {"workdir": "relative/runtime"}
    plan_path.write_text(json.dumps(plan))
    started = []
    monkeypatch.setattr(
        hparam_runtime, "_start_process", lambda _execution, command: started.append(command) or "launched"
    )

    with pytest.raises(ValueError, match="absolute path"):
        hparam_runtime.launch_hparam_runs(plan_dir, dry_run=False)

    assert started == []
    assert not (plan_dir / "launch_manifest.tsv").exists()


def test_hparam_runtime_rejects_workdir_that_differs_from_frozen_runtime_path(tmp_path: Path, monkeypatch):
    recipe = _hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    plan_path = plan_dir / "plan.json"
    plan = json.loads(plan_path.read_text())
    plan["recipe"]["execution"] = {"workdir": str(tmp_path / "other-runtime")}
    plan_path.write_text(json.dumps(plan))
    started = []
    monkeypatch.setattr(
        hparam_runtime, "_start_process", lambda _execution, command: started.append(command) or "launched"
    )

    with pytest.raises(ValueError, match="runtime_dir differs"):
        hparam_runtime.launch_hparam_runs(plan_dir, dry_run=False)

    assert started == []
    assert not (plan_dir / "launch_manifest.tsv").exists()


def test_remote_stop_failure_does_not_commit_stopped_state(tmp_path: Path, monkeypatch):
    rows = _write_runtime_rows(
        tmp_path,
        [{"run_id": "run-000", "target": "ssh", "host": "unit-host", "status": "running"}],
    )
    before_launch = (tmp_path / "launch_manifest.tsv").read_bytes()
    before_status = (tmp_path / "run_status.tsv").read_bytes()
    monkeypatch.setattr(run_evidence, "read_pid", lambda _path, _row: 123)
    monkeypatch.setattr(
        hparam_runtime.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 255),
    )

    with pytest.raises(RuntimeError, match="Failed to stop remote run"):
        hparam_runtime.stop_hparam_run(tmp_path, rows[0]["run_id"], reason="validation diverged")

    assert (tmp_path / "launch_manifest.tsv").read_bytes() == before_launch
    assert (tmp_path / "run_status.tsv").read_bytes() == before_status
    assert not (tmp_path / "events.jsonl").exists()


@pytest.mark.parametrize("failure", ["permission", "wrong_type", "ssh_error", "timeout"])
def test_remote_stop_pid_probe_failure_has_no_side_effects(tmp_path: Path, monkeypatch, failure: str):
    _write_runtime_rows(
        tmp_path,
        [{"run_id": "run-000", "target": "ssh", "host": "unit-host", "status": "running"}],
    )
    merge_run_manifest(tmp_path, [{"step_id": "train-model", "run_id": "run-000", "status": "running"}])
    before = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    calls = []

    def fake_pid_read(_row, command):
        if failure == "permission":
            return subprocess.CompletedProcess([], 1 if "os.lstat" in command else 44, "", "permission denied")
        if failure == "wrong_type":
            return subprocess.CompletedProcess([], 1 if "os.lstat" in command else 44, "", "is a directory")
        if failure == "timeout":
            return subprocess.CompletedProcess([], 124, "", "timed out")
        return subprocess.CompletedProcess([], 255, "", "connection lost")

    monkeypatch.setattr(run_evidence, "run_row_command", fake_pid_read)
    monkeypatch.setattr(hparam_runtime.subprocess, "run", lambda *_args, **_kwargs: calls.append("kill"))
    monkeypatch.setattr(hparam_runtime, "merge_run_manifest", lambda *_args: calls.append("merge"))
    monkeypatch.setattr(hparam_runtime, "write_rows", lambda *_args: calls.append("write"))
    monkeypatch.setattr(hparam_runtime, "append_event", lambda *_args: calls.append("event"))
    monkeypatch.setattr(hparam_runtime, "write_status_report", lambda *_args: calls.append("report"))

    with pytest.raises(RuntimeError, match="SSH PID read failed"):
        hparam_runtime.stop_hparam_run(tmp_path, "run-000", reason="remote state unknown")

    assert calls == []
    assert {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before


@pytest.mark.parametrize("pid_text", ["0", "-1"])
def test_hparam_stop_rejects_nonpositive_pid_before_kill(tmp_path: Path, monkeypatch, pid_text: str):
    rows = _write_runtime_rows(tmp_path, [{"run_id": "run-000", "status": "running"}])
    Path(rows[0]["pid_path"]).write_text(pid_text)
    before = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    killed = []
    monkeypatch.setattr(hparam_runtime.os, "kill", lambda pid, sig: killed.append((pid, sig)))

    with pytest.raises(RuntimeError, match="PID file is empty or invalid"):
        hparam_runtime.stop_hparam_run(tmp_path, "run-000", reason="invalid PID evidence")

    assert killed == []
    assert {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before


@pytest.mark.parametrize("failure", ["directory", "invalid_utf8", "os_error", "dangling_symlink"])
def test_hparam_stop_rejects_unreadable_local_pid_before_kill(tmp_path: Path, monkeypatch, failure: str):
    rows = _write_runtime_rows(tmp_path, [{"run_id": "run-000", "status": "running"}])
    pid_path = Path(rows[0]["pid_path"])
    if failure == "directory":
        pid_path.mkdir()
    elif failure == "invalid_utf8":
        pid_path.write_bytes(b"\xff")
    elif failure == "dangling_symlink":
        pid_path.symlink_to(tmp_path / "missing.pid")
    else:
        pid_path.write_text("123")
        original_read_text = Path.read_text

        def fail_pid_read(path: Path, *args, **kwargs):
            if path == pid_path:
                raise OSError("PID read failed")
            return original_read_text(path, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", fail_pid_read)
    before = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    killed = []
    monkeypatch.setattr(hparam_runtime.os, "kill", lambda pid, sig: killed.append((pid, sig)))

    with pytest.raises(RuntimeError, match="PID file read failed"):
        hparam_runtime.stop_hparam_run(tmp_path, "run-000", reason="unreadable PID evidence")

    assert killed == []
    assert {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before


@pytest.mark.parametrize("status", ["completed", "failed", "finished", "launch_failed", "stopped", "superseded"])
def test_hparam_stop_rejects_terminal_status_before_pid_or_mutation(tmp_path: Path, monkeypatch, status: str):
    _write_runtime_rows(
        tmp_path,
        [{"run_id": "run-000", "status": "planned" if status == "superseded" else "running"}],
    )
    merge_run_manifest(tmp_path, [{"step_id": "train-model", "run_id": "run-000", "status": status}])
    before = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    calls = []
    monkeypatch.setattr(run_evidence, "read_pid", lambda *_args: calls.append("read_pid") or 123)
    monkeypatch.setattr(hparam_runtime.os, "kill", lambda *_args: calls.append("kill"))
    monkeypatch.setattr(hparam_runtime, "merge_run_manifest", lambda *_args: calls.append("merge"))
    monkeypatch.setattr(hparam_runtime, "write_rows", lambda *_args: calls.append("write"))
    monkeypatch.setattr(hparam_runtime, "append_event", lambda *_args: calls.append("event"))
    monkeypatch.setattr(hparam_runtime, "write_status_report", lambda *_args: calls.append("report"))

    with pytest.raises(ValueError, match="already terminal"):
        hparam_runtime.stop_hparam_run(tmp_path, "run-000", reason="terminal run")

    assert calls == []
    assert {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before


def test_hparam_monitor_rejects_dangling_launch_manifest_before_canonical_write(tmp_path: Path):
    _write_runtime_rows(tmp_path, [{"run_id": "run-000", "status": "running"}])
    launch_path = tmp_path / "launch_manifest.tsv"
    missing_target = tmp_path / "missing-launch-manifest.tsv"
    launch_path.unlink()
    launch_path.symlink_to(missing_target)
    before = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}

    with pytest.raises(FileNotFoundError, match="symlink target is missing"):
        hparam_runtime.monitor_hparam_runs(tmp_path)

    assert not missing_target.exists()
    assert {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before


def test_hparam_stop_rejects_dangling_status_manifest_before_kill_or_write(tmp_path: Path, monkeypatch):
    _write_runtime_rows(tmp_path, [{"run_id": "run-000", "status": "running"}])
    status_path = tmp_path / "run_status.tsv"
    missing_target = tmp_path / "missing-run-status.tsv"
    status_path.unlink()
    status_path.symlink_to(missing_target)
    before = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    killed = []
    monkeypatch.setattr(run_evidence, "read_pid", lambda *_args: 123)
    monkeypatch.setattr(hparam_runtime.os, "kill", lambda *_args: killed.append(True))

    with pytest.raises(FileNotFoundError, match="symlink target is missing"):
        hparam_runtime.stop_hparam_run(tmp_path, "run-000", reason="dangling status")

    assert killed == []
    assert not missing_target.exists()
    assert {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before


def test_hparam_stop_commits_one_final_row_to_all_manifests(tmp_path: Path, monkeypatch):
    _write_runtime_rows(tmp_path, [{"run_id": "run-000", "status": "running"}])
    killed = []
    monkeypatch.setattr(run_evidence, "read_pid", lambda _path, _row: 123)
    monkeypatch.setattr(hparam_runtime.os, "kill", lambda pid, sig: killed.append((pid, sig)))

    hparam_runtime.stop_hparam_run(tmp_path, "run-000", reason="manual stop")

    canonical = _read_table(tmp_path / "run_manifest.tsv")[0]
    local = _read_table(tmp_path / "run_status.tsv")[0]
    launch = _read_table(tmp_path / "launch_manifest.tsv")[0]
    assert canonical == local == launch
    assert canonical["status"] == "stopped"
    assert canonical["stop_reason"] == "manual stop"
    assert killed == [(123, hparam_runtime.signal.SIGTERM)]
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert [event["event_type"] for event in events].count("run_stopped") == 1

    with pytest.raises(ValueError, match="already terminal"):
        hparam_runtime.stop_hparam_run(tmp_path, "run-000", reason="repeat stop")

    assert killed == [(123, hparam_runtime.signal.SIGTERM)]
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert [event["event_type"] for event in events].count("run_stopped") == 1


def test_hparam_stop_ignores_stale_local_terminal_status(tmp_path: Path, monkeypatch):
    rows = _write_runtime_rows(tmp_path, [{"run_id": "run-000", "status": "running"}])
    Path(rows[0]["pid_path"]).write_text("123")
    merge_run_manifest(
        tmp_path,
        [{"step_id": "train-model", "run_id": "run-000", "status": "running"}],
    )
    local_rows = _read_table(tmp_path / "run_status.tsv")
    local_rows[0]["status"] = "failed"
    manifests.write_rows(tmp_path / "run_status.tsv", local_rows)
    killed = []
    monkeypatch.setattr(hparam_runtime.os, "kill", lambda pid, sig: killed.append((pid, sig)))

    hparam_runtime.stop_hparam_run(tmp_path, "run-000", reason="manual stop")

    assert killed == [(123, hparam_runtime.signal.SIGTERM)]
    assert _read_table(tmp_path / "run_manifest.tsv")[0]["status"] == "stopped"
    assert _read_table(tmp_path / "run_status.tsv")[0]["status"] == "stopped"
    assert _read_table(tmp_path / "launch_manifest.tsv")[0]["status"] == "stopped"


@pytest.mark.parametrize("target_name", ["run_matrix.csv", "reports/run_matrix.md"])
@pytest.mark.parametrize("target_kind", ["directory", "hardlink"])
def test_hparam_stop_rejects_invalid_canonical_output_before_kill(
    tmp_path: Path, monkeypatch, target_name: str, target_kind: str
):
    _write_runtime_rows(tmp_path, [{"run_id": "run-000", "status": "running"}])
    target = tmp_path / target_name
    target.parent.mkdir(parents=True, exist_ok=True)
    if target_kind == "directory":
        target.mkdir()
    else:
        target.hardlink_to(tmp_path / "run_manifest.tsv")
    before = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    killed = []
    monkeypatch.setattr(run_evidence, "read_pid", lambda _path, _row: 123)
    monkeypatch.setattr(hparam_runtime.os, "kill", lambda pid, sig: killed.append((pid, sig)))

    with pytest.raises(ValueError, match="Managed output"):
        hparam_runtime.stop_hparam_run(tmp_path, "run-000", reason="manual stop")

    assert killed == []
    assert {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before
    assert not (tmp_path / "events.jsonl").exists()


def test_hparam_stop_rejects_aliased_local_mirrors_before_kill(tmp_path: Path, monkeypatch):
    _write_runtime_rows(tmp_path, [{"run_id": "run-000", "status": "running"}])
    launch_path = tmp_path / "launch_manifest.tsv"
    status_path = tmp_path / "run_status.tsv"
    status_path.unlink()
    status_path.hardlink_to(launch_path)
    before = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    killed = []
    monkeypatch.setattr(run_evidence, "read_pid", lambda _path, _row: 123)
    monkeypatch.setattr(hparam_runtime.os, "kill", lambda pid, sig: killed.append((pid, sig)))

    with pytest.raises(ValueError, match="Managed output"):
        hparam_runtime.stop_hparam_run(tmp_path, "run-000", reason="manual stop")

    assert killed == []
    assert {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before


def test_hparam_stop_rejects_aliased_status_report_before_kill(tmp_path: Path, monkeypatch):
    _write_runtime_rows(tmp_path, [{"run_id": "run-000", "status": "running"}])
    status_report = tmp_path / "reports" / "status.md"
    status_report.parent.mkdir(parents=True)
    status_report.hardlink_to(tmp_path / "experiment.yaml")
    before = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    killed = []
    monkeypatch.setattr(run_evidence, "read_pid", lambda _path, _row: 123)
    monkeypatch.setattr(hparam_runtime.os, "kill", lambda pid, sig: killed.append((pid, sig)))

    with pytest.raises(ValueError, match="Managed output"):
        hparam_runtime.stop_hparam_run(tmp_path, "run-000", reason="manual stop")

    assert killed == []
    assert {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before


def test_hparam_stop_mirrors_the_status_committed_by_the_canonical_owner(tmp_path: Path, monkeypatch):
    _write_runtime_rows(tmp_path, [{"run_id": "run-000", "status": "running"}])
    merge_run_manifest(tmp_path, [{"step_id": "train-model", "run_id": "run-000", "status": "running"}])
    real_merge = merge_run_manifest

    def merge_after_wandb_update(root, rows):
        real_merge(root, [{"step_id": "train-model", "run_id": "run-000", "status": "failed"}])
        return real_merge(root, rows)

    monkeypatch.setattr(hparam_runtime, "merge_run_manifest", merge_after_wandb_update)
    monkeypatch.setattr(run_evidence, "read_pid", lambda _path, _row: 123)
    monkeypatch.setattr(hparam_runtime.os, "kill", lambda _pid, _signal: None)

    hparam_runtime.stop_hparam_run(tmp_path, "run-000", reason="manual stop")

    assert _read_table(tmp_path / "run_manifest.tsv")[0]["status"] == "failed"
    assert _read_table(tmp_path / "run_status.tsv")[0]["status"] == "failed"
    assert _read_table(tmp_path / "launch_manifest.tsv")[0]["status"] == "failed"
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert [event["event_type"] for event in events].count("run_stopped") == 1


def test_hparam_stop_rejects_frozen_drift_before_kill(tmp_path: Path, monkeypatch):
    rows = _write_runtime_rows(tmp_path, [{"run_id": "run-000", "status": "running"}])
    rows[0]["version"] = "drifted-version"
    manifests.write_rows(tmp_path / "launch_manifest.tsv", rows)
    before = (tmp_path / "launch_manifest.tsv").read_bytes()
    killed = []
    monkeypatch.setattr(run_evidence, "read_pid", lambda _path, _row: 123)
    monkeypatch.setattr(hparam_runtime.os, "kill", lambda pid, sig: killed.append((pid, sig)))

    with pytest.raises(ValueError, match="version"):
        hparam_runtime.stop_hparam_run(tmp_path, "run-000", reason="stale metadata")

    assert killed == []
    assert (tmp_path / "launch_manifest.tsv").read_bytes() == before


def test_hparam_launch_dry_run_renders_ssh_conda_gpu_wandb_and_pid_paths(
    tmp_path: Path,
):
    recipe = _hparam_recipe(
        tmp_path,
        execution={
            "target": "ssh",
            "host": "baichuan3",
            "workdir": str(tmp_path / "plan"),
            "conda_env": "ywx",
            "gpu_pool": [6, 7],
            "gpus_per_run": 2,
            "max_concurrent": 1,
            "wandb_project": "sleep2vec-unit-hparam",
            "wandb_group": "unit",
        },
    )
    plan_dir = tmp_path / "plan"

    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    result = _run("hparam-launch", "--plan-dir", str(plan_dir))

    assert result.returncode == 0, result.stderr
    rows = _read_table(plan_dir / "launch_manifest.tsv")
    assert rows[0]["status"] == "planned"
    assert rows[0]["target"] == "ssh"
    assert rows[0]["host"] == "baichuan3"
    assert rows[0]["gpus"] == "6,7"
    assert "ssh baichuan3" in rows[0]["command"]
    assert "mkdir -p" in rows[0]["command"]
    assert "(nohup env " in rows[0]["command"]
    assert "conda run --no-capture-output -n ywx" in rows[0]["command"]
    assert "CUDA_VISIBLE_DEVICES=6,7" in rows[0]["command"]
    assert "WANDB_PROJECT=sleep2vec-unit-hparam" in rows[0]["command"]
    assert rows[0]["log_path"].endswith("runs/run-000--lr-1e-6/stdout.log")
    assert rows[0]["pid_path"].endswith("runs/run-000--lr-1e-6/pid")
    assert not (plan_dir / "logs").exists()
    assert not (plan_dir / "pids").exists()


def test_repeated_ssh_dry_run_does_not_observe_runtime_before_execute(tmp_path: Path, monkeypatch):
    recipe = _hparam_recipe(
        tmp_path,
        execution={
            "target": "ssh",
            "host": "offline-host",
            "workdir": str(tmp_path / "plan"),
            "max_concurrent": 1,
        },
    )
    plan_dir = tmp_path / "plan"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    remote_calls = []

    def fake_remote_command(row, command):
        remote_calls.append((row, command))
        return subprocess.CompletedProcess([], run_evidence.REMOTE_MISSING_RETURN_CODE, "", "")

    monkeypatch.setattr(run_evidence, "run_row_command", fake_remote_command)

    hparam_runtime.launch_hparam_runs(plan_dir, dry_run=True)
    hparam_runtime.launch_hparam_runs(plan_dir, dry_run=True)

    assert remote_calls == []
    assert _read_table(tmp_path / "run_manifest.tsv")[0]["status"] == "planned"
    assert _read_table(plan_dir / "launch_manifest.tsv")[0]["status"] == "planned"
    real_validate = hparam_runtime.exp_io.validate_managed_output_paths

    def validate_without_remote(root, paths, remote=None):
        if remote is None:
            return real_validate(root, paths)

    started = []
    monkeypatch.setattr(hparam_runtime.exp_io, "validate_managed_output_paths", validate_without_remote)
    monkeypatch.setattr(
        hparam_runtime, "_start_process", lambda _execution, command: started.append(command) or "launched"
    )

    hparam_runtime.launch_hparam_runs(plan_dir, dry_run=False)

    assert len(started) == 1
    assert _read_table(tmp_path / "run_manifest.tsv")[0]["status"] == "launched"


@pytest.mark.parametrize("runtime_fault", ["existing", "ancestor_symlink"])
def test_hparam_launch_rejects_unsafe_runtime_root_before_start(tmp_path: Path, monkeypatch, runtime_fault: str):
    recipe = _hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    run = json.loads((plan_dir / "plan.json").read_text())["runs"][0]
    runtime_dir = Path(run["runtime_dir"])
    if runtime_fault == "existing":
        runtime_dir.mkdir(parents=True)
    else:
        outside = tmp_path / "outside-runtime"
        outside.mkdir()
        runtime_dir.parent.symlink_to(outside, target_is_directory=True)
    started = []
    monkeypatch.setattr(
        hparam_runtime,
        "_start_process",
        lambda _execution, command: started.append(command) or "launched",
    )

    with pytest.raises(ValueError, match="Managed runtime output|Managed output"):
        hparam_runtime.launch_hparam_runs(plan_dir, dry_run=False)

    assert started == []
    assert _read_table(tmp_path / "run_manifest.tsv")[0]["status"] == "planned"


def test_hparam_ssh_launch_rejects_existing_remote_runtime_root_before_start(tmp_path: Path, monkeypatch):
    recipe = _hparam_recipe(
        tmp_path,
        execution={
            "target": "ssh",
            "host": "offline-host",
            "workdir": str(tmp_path),
            "max_concurrent": 1,
        },
    )
    plan_dir = tmp_path / "plan"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    run = json.loads((plan_dir / "plan.json").read_text())["runs"][0]
    runtime_dir = Path(run["runtime_dir"])
    real_validate = hparam_runtime.exp_io.validate_managed_output_paths

    def fake_validate(root, paths, remote=None):
        if remote and runtime_dir in paths:
            raise ValueError(f"Managed output paths must be independent regular files: {runtime_dir}")
        if not remote:
            real_validate(root, paths)

    monkeypatch.setattr(
        hparam_runtime.exp_io,
        "validate_managed_output_paths",
        fake_validate,
    )
    started = []
    monkeypatch.setattr(
        hparam_runtime,
        "_start_process",
        lambda _execution, command: started.append(command) or "launched",
    )

    with pytest.raises(ValueError, match="Managed output paths must be independent regular files"):
        hparam_runtime.launch_hparam_runs(plan_dir, dry_run=False)

    assert started == []
    assert _read_table(tmp_path / "run_manifest.tsv")[0]["status"] == "planned"


def test_hparam_launch_accepts_scalar_runtime_devices(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    base_recipe = Path(payload["base_recipe"])
    base_payload = yaml.safe_load(base_recipe.read_text())
    base_payload["runtime"]["devices"] = 2
    write_yaml(base_recipe, base_payload)
    plan_dir = tmp_path / "plan"

    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    result = _run("hparam-launch", "--plan-dir", str(plan_dir))

    assert result.returncode == 0, result.stderr
    rows = _read_table(plan_dir / "launch_manifest.tsv")
    assert rows[0]["gpus"] == "2"
    assert "(nohup env " in rows[0]["command"]
    assert "CUDA_VISIBLE_DEVICES=2" in rows[0]["command"]


def test_hparam_launch_resolves_relative_plan_dir_before_cd(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path)
    plan_dir = tmp_path / "relative_plan"

    plan = subprocess.run(
        [
            sys.executable,
            "-m",
            "agent_tools",
            "plan",
            "--recipe",
            str(recipe),
            "--output-dir",
            "relative_plan",
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        env={**os.environ, "PYTHONPATH": str(Path.cwd())},
    )
    launch = subprocess.run(
        [
            sys.executable,
            "-m",
            "agent_tools",
            "hparam-launch",
            "--plan-dir",
            "relative_plan",
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        env={**os.environ, "PYTHONPATH": str(Path.cwd())},
    )

    assert plan.returncode == 0, plan.stderr
    assert launch.returncode == 0, launch.stderr
    rows = _read_table(plan_dir / "launch_manifest.tsv")
    assert rows[0]["script"] == str(plan_dir / "runs" / "run-000--lr-1e-6" / "launch.sh")
    assert rows[0]["log_path"] == str(plan_dir / "runs" / "run-000--lr-1e-6" / "stdout.log")
    assert "relative_plan/relative_plan" not in rows[0]["command"]


def test_hparam_launch_does_not_retry_missing_pid(tmp_path: Path, monkeypatch):
    recipe = _hparam_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    payload["search"]["max_runs"] = 2
    payload["search"]["parameters"]["runtime.lr"] = [1e-6, 2e-6]
    payload["execution"] = {"max_concurrent": 1}
    recipe.write_text(yaml.safe_dump(payload))
    plan_dir = tmp_path / "plan"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    hparam_runtime.launch_hparam_runs(plan_dir, dry_run=True)
    rows = _read_table(plan_dir / "launch_manifest.tsv")
    rows[0]["status"] = "launched"
    rows[0]["pid_path"] = str(plan_dir / "missing.pid")
    rows[1]["status"] = "pending"
    manifests.write_rows(plan_dir / "launch_manifest.tsv", rows)
    manifests.write_rows(plan_dir / "run_status.tsv", rows)
    merge_run_manifest(
        tmp_path,
        [{"step_id": row["step_id"], "run_id": row["run_id"], "status": row["status"]} for row in rows],
    )
    started = []
    monkeypatch.setattr(
        hparam_runtime, "_start_process", lambda _execution, command: started.append(command) or "launched"
    )

    hparam_runtime.launch_hparam_runs(plan_dir, dry_run=False)

    assert started == []
    status = {row["run_id"]: row["status"] for row in _read_table(plan_dir / "launch_manifest.tsv")}
    assert status == {"run-000": "missing_pid", "run-001": "pending"}


def test_hparam_launch_validates_every_snapshot_before_starting(tmp_path: Path, monkeypatch):
    recipe = _hparam_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    payload["search"]["max_runs"] = 2
    payload["search"]["parameters"]["runtime.lr"] = [1e-6, 2e-6]
    recipe.write_text(yaml.safe_dump(payload))
    plan_dir = tmp_path / "plan"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    runs = json.loads((plan_dir / "plan.json").read_text())["runs"]
    Path(runs[1]["config"]).write_text("changed: true\n")
    started = []
    monkeypatch.setattr(
        hparam_runtime, "_start_process", lambda _execution, command: started.append(command) or "launched"
    )

    with pytest.raises(ValueError, match="snapshot hash changed"):
        hparam_runtime.launch_hparam_runs(plan_dir, dry_run=False)

    assert started == []
    assert not (plan_dir / "launch_manifest.tsv").exists()
    assert not (plan_dir / "run_status.tsv").exists()


def test_hparam_runtime_rejects_legacy_plan_without_side_effects(tmp_path: Path, monkeypatch):
    (tmp_path / "plan.json").write_text(json.dumps({"trials": [{"trial_id": "trial_000"}], "recipe": {}}))
    launch_path = tmp_path / "launch_manifest.tsv"
    status_path = tmp_path / "trial_status.tsv"
    launch_path.write_text("trial_id\tstatus\ntrial_000\tlaunched\n")
    status_path.write_text("trial_id\tstatus\ntrial_000\tlaunched\n")
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    started = []
    killed = []
    monkeypatch.setattr(
        hparam_runtime, "_start_process", lambda _execution, command: started.append(command) or "launched"
    )
    monkeypatch.setattr(hparam_runtime.os, "kill", lambda pid, sig: killed.append((pid, sig)))

    with pytest.raises(ValueError, match="Legacy hparam plan"):
        hparam_runtime.launch_hparam_runs(tmp_path, dry_run=False)
    with pytest.raises(ValueError, match="Legacy hparam plan"):
        hparam_runtime.monitor_hparam_runs(tmp_path)
    with pytest.raises(ValueError, match="Legacy hparam plan"):
        hparam_runtime.stop_hparam_run(tmp_path, "trial_000", reason="legacy")

    assert started == []
    assert killed == []
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == before


def test_hparam_runtime_rejects_legacy_status_rows_before_start_or_kill(tmp_path: Path, monkeypatch):
    rows = _write_runtime_rows(tmp_path, [{"run_id": "run-000", "version": "v0", "status": "launched"}])
    legacy_rows = [{**rows[0], "trial_id": "trial_000"}]
    manifests.write_rows(tmp_path / "launch_manifest.tsv", legacy_rows)
    before = (tmp_path / "launch_manifest.tsv").read_bytes()
    started = []
    killed = []
    monkeypatch.setattr(
        hparam_runtime, "_start_process", lambda _execution, command: started.append(command) or "launched"
    )
    monkeypatch.setattr(hparam_runtime.os, "kill", lambda pid, sig: killed.append((pid, sig)))

    with pytest.raises(ValueError, match="trial_id"):
        hparam_runtime.launch_hparam_runs(tmp_path, dry_run=False)
    with pytest.raises(ValueError, match="trial_id"):
        hparam_runtime.monitor_hparam_runs(tmp_path)
    with pytest.raises(ValueError, match="trial_id"):
        hparam_runtime.stop_hparam_run(tmp_path, "run-000", reason="legacy table")

    assert started == []
    assert killed == []
    assert (tmp_path / "launch_manifest.tsv").read_bytes() == before


@pytest.mark.parametrize("table", ["launch_manifest.tsv", "run_status.tsv"])
def test_hparam_runtime_rejects_header_only_removed_status_table_before_start_or_kill(
    tmp_path: Path, monkeypatch, table: str
):
    _write_runtime_rows(tmp_path, [{"run_id": "run-000", "version": "v0", "status": "launched"}])
    (tmp_path / table).write_text("trial_id\n")
    before = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    started = []
    killed = []
    monkeypatch.setattr(
        hparam_runtime, "_start_process", lambda _execution, command: started.append(command) or "launched"
    )
    monkeypatch.setattr(hparam_runtime.os, "kill", lambda pid, sig: killed.append((pid, sig)))

    with pytest.raises(ValueError, match="Historical managed table fields"):
        hparam_runtime.launch_hparam_runs(tmp_path, dry_run=False)
    with pytest.raises(ValueError, match="Historical managed table fields"):
        hparam_runtime.monitor_hparam_runs(tmp_path)
    with pytest.raises(ValueError, match="Historical managed table fields"):
        hparam_runtime.stop_hparam_run(tmp_path, "run-000", reason="legacy table")

    assert started == []
    assert killed == []
    assert {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before


def test_hparam_runtime_rejects_legacy_status_filename(tmp_path: Path, monkeypatch):
    _write_runtime_rows(tmp_path, [{"run_id": "run-000", "version": "v0", "status": "planned"}])
    legacy_status = tmp_path / "trial_status.tsv"
    legacy_status.write_text("trial_id\tstatus\ntrial_000\tfailed\n")
    current_status = (tmp_path / "run_status.tsv").read_bytes()
    started = []
    monkeypatch.setattr(
        hparam_runtime, "_start_process", lambda _execution, command: started.append(command) or "launched"
    )

    with pytest.raises(ValueError, match="Legacy hparam status"):
        hparam_runtime.launch_hparam_runs(tmp_path, dry_run=False)

    assert started == []
    assert (tmp_path / "run_status.tsv").read_bytes() == current_status


def test_hparam_doctor_rejects_invalid_execution_target(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path, execution={"target": "cluster"})

    result = _run("doctor", "--recipe", str(recipe), "--output-dir", str(tmp_path / "doctor"))

    assert result.returncode == 1
    assert "execution.target" in result.stdout


def test_hparam_doctor_rejects_deprecated_log_and_pid_dirs(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path, execution={"log_dir": "logs", "pid_dir": "pids"})

    result = _run("doctor", "--recipe", str(recipe), "--output-dir", str(tmp_path / "doctor"))

    assert result.returncode == 1
    assert "execution.log_dir" in result.stdout
    assert "execution.pid_dir" in result.stdout


def test_hparam_monitor_handles_running_finished_and_failed_rows(tmp_path: Path):
    pid_path = tmp_path / "running.pid"
    pid_path.write_text(str(os.getpid()))
    missing_pid = tmp_path / "missing.pid"
    fail_pid = tmp_path / "fail.pid"
    fail_pid.write_text("999999999")
    fail_log = tmp_path / "fail.log"
    fail_log.write_text("Traceback\nRuntimeError: boom\n")
    _write_runtime_rows(
        tmp_path,
        [
            {"run_id": "running", "version": "v1", "pid_path": str(pid_path), "status": "launched"},
            {"run_id": "missing", "version": "v2", "pid_path": str(missing_pid), "status": "launched"},
            {
                "run_id": "failed",
                "version": "v3",
                "pid_path": str(fail_pid),
                "log_path": str(fail_log),
                "status": "launched",
            },
        ],
    )

    monitor_hparam_runs(tmp_path)

    status = {row["run_id"]: row["status"] for row in _read_table(tmp_path / "run_status.tsv")}
    assert status["running"] == "running"
    assert status["missing"] == "missing_pid"
    assert status["failed"] == "failed"
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    status_events = [event for event in events if event["event_type"] == "run_status_changed"]
    assert status_events
    assert all(event["step_id"] == "train-model" for event in status_events)
    assert {event["run_id"] for event in status_events} == {"running", "missing", "failed"}


def test_hparam_monitor_does_not_overwrite_workspace_terminal_status(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    hparam_runtime.launch_hparam_runs(plan_dir, dry_run=True)
    rows = _read_table(plan_dir / "launch_manifest.tsv")
    rows[0]["status"] = "launched"
    rows[0]["pid_path"] = str(plan_dir / "missing.pid")
    manifests.write_rows(plan_dir / "launch_manifest.tsv", rows)
    manifests.write_rows(plan_dir / "run_status.tsv", rows)
    merge_run_manifest(
        tmp_path,
        [{"step_id": rows[0]["step_id"], "run_id": rows[0]["run_id"], "status": "failed"}],
    )
    event_count = len((tmp_path / "events.jsonl").read_text().splitlines())

    hparam_runtime.monitor_hparam_runs(plan_dir)

    local_rows = _read_table(plan_dir / "run_status.tsv")
    workspace_rows = _read_table(tmp_path / "run_manifest.tsv")
    assert local_rows[0]["status"] == "failed"
    assert workspace_rows[0]["status"] == "failed"
    assert len((tmp_path / "events.jsonl").read_text().splitlines()) == event_count


def test_hparam_monitor_mirrors_and_reports_the_status_committed_by_the_canonical_owner(tmp_path: Path, monkeypatch):
    _write_runtime_rows(tmp_path, [{"run_id": "run-000", "status": "running"}])
    merge_run_manifest(tmp_path, [{"step_id": "train-model", "run_id": "run-000", "status": "running"}])
    real_merge = merge_run_manifest

    def merge_after_wandb_update(root, rows):
        real_merge(root, [{"step_id": "train-model", "run_id": "run-000", "status": "failed"}])
        return real_merge(root, rows)

    monkeypatch.setattr(hparam_runtime, "merge_run_manifest", merge_after_wandb_update)
    monkeypatch.setattr(
        hparam_runtime.evidence,
        "status_row",
        lambda _root, observation, _prior, *, health: {**observation, "status": "finished"},
    )

    hparam_runtime.monitor_hparam_runs(tmp_path)

    assert _read_table(tmp_path / "run_manifest.tsv")[0]["status"] == "failed"
    assert _read_table(tmp_path / "run_status.tsv")[0]["status"] == "failed"
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    status_event = next(event for event in events if event["event_type"] == "run_status_changed")
    assert status_event["from"] == "running"
    assert status_event["to"] == "failed"


def test_hparam_monitor_without_launch_manifest_preserves_running_status(tmp_path: Path):
    _write_runtime_rows(tmp_path, [{"run_id": "run-000", "status": "running"}])
    workspace_rows = _read_table(tmp_path / "run_manifest.tsv")
    workspace_rows[0]["status"] = "running"
    manifests.write_rows(tmp_path / "run_manifest.tsv", workspace_rows)
    (tmp_path / "launch_manifest.tsv").unlink()

    hparam_runtime.monitor_hparam_runs(tmp_path)
    hparam_runtime.monitor_hparam_runs(tmp_path)

    assert _read_table(tmp_path / "run_manifest.tsv")[0]["status"] == "running"
    assert _read_table(tmp_path / "run_status.tsv")[0]["status"] == "running"
    assert not (tmp_path / "events.jsonl").exists()


@pytest.mark.parametrize("local_status", ["launched", "unknown_remote"])
def test_hparam_monitor_without_launch_manifest_ignores_stale_local_status(tmp_path: Path, local_status: str):
    _write_runtime_rows(tmp_path, [{"run_id": "run-000", "status": local_status}])
    workspace_rows = _read_table(tmp_path / "run_manifest.tsv")
    workspace_rows[0]["status"] = "running"
    manifests.write_rows(tmp_path / "run_manifest.tsv", workspace_rows)
    (tmp_path / "launch_manifest.tsv").unlink()

    hparam_runtime.monitor_hparam_runs(tmp_path)

    assert _read_table(tmp_path / "run_manifest.tsv")[0]["status"] == "running"
    assert _read_table(tmp_path / "run_status.tsv")[0]["status"] == "running"
    assert not (tmp_path / "events.jsonl").exists()


def test_hparam_monitor_rejects_aliased_status_report_before_canonical_write(tmp_path: Path):
    _write_runtime_rows(tmp_path, [{"run_id": "run-000", "status": "running"}])
    status_report = tmp_path / "reports" / "status.md"
    status_report.parent.mkdir(parents=True)
    status_report.hardlink_to(tmp_path / "experiment.yaml")
    before = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}

    with pytest.raises(ValueError, match="Managed output"):
        hparam_runtime.monitor_hparam_runs(tmp_path)

    assert {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before
    assert not (tmp_path / "events.jsonl").exists()


def test_hparam_monitor_keeps_failed_status_after_failure_evidence_disappears(tmp_path: Path):
    dead_pid = tmp_path / "dead.pid"
    dead_pid.write_text("999999999")
    fail_log = tmp_path / "fail.log"
    fail_log.write_text("Traceback\nRuntimeError: boom\n")
    _write_runtime_rows(
        tmp_path,
        [
            {
                "run_id": "run-000",
                "version": "v0",
                "pid_path": str(dead_pid),
                "log_path": str(fail_log),
                "status": "launched",
            }
        ],
    )

    hparam_runtime.monitor_hparam_runs(tmp_path)
    assert _read_table(tmp_path / "run_status.tsv")[0]["status"] == "failed"
    fail_log.write_text("")

    hparam_runtime.monitor_hparam_runs(tmp_path)

    assert _read_table(tmp_path / "run_status.tsv")[0]["status"] == "failed"


def test_hparam_monitor_never_launches_pending_runs(tmp_path: Path, monkeypatch):
    dead_pid = tmp_path / "dead.pid"
    dead_pid.write_text("999999999")
    _write_runtime_rows(
        tmp_path,
        [
            {
                "run_id": "run-000",
                "version": "v0",
                "pid_path": str(dead_pid),
                "status": "launched",
                "launched_at": "2026-01-01T00:00:00Z",
            },
            {"run_id": "run-001", "version": "v1", "status": "pending"},
        ],
    )
    started = []

    def fake_start(_execution, command):
        started.append(command)
        return "launched"

    monkeypatch.setattr(hparam_runtime, "_start_process", fake_start)

    monitor_hparam_runs(tmp_path)

    status = {row["run_id"]: row for row in _read_table(tmp_path / "run_status.tsv")}
    manifest = {row["run_id"]: row for row in _read_table(tmp_path / "launch_manifest.tsv")}
    assert started == []
    assert status["run-000"]["status"] == "finished"
    assert status["run-001"]["status"] == "pending"
    assert manifest["run-001"]["status"] == "pending"
    assert not manifest["run-001"]["launched_at"]


def test_hparam_monitor_health_is_opt_in(tmp_path: Path, monkeypatch):
    pid_path = tmp_path / "running.pid"
    pid_path.write_text("123")
    _write_runtime_rows(
        tmp_path,
        [{"run_id": "running", "version": "v1", "pid_path": str(pid_path), "status": "launched"}],
    )
    monkeypatch.setattr(run_evidence, "process_running", lambda row, pid: True)

    monitor_hparam_runs(tmp_path)

    row = _read_table(tmp_path / "run_status.tsv")[0]
    assert "health_status" not in row


def test_hparam_status_preserves_terminal_state_with_live_pid(tmp_path: Path, monkeypatch):
    pid_path = tmp_path / "stopped.pid"
    pid_path.write_text("123")
    monkeypatch.setattr(run_evidence, "process_running", lambda row, pid: True)

    row = run_evidence.status_row(tmp_path, {"status": "stopped", "pid_path": str(pid_path)})

    assert row["status"] == "stopped"


@pytest.mark.parametrize("pid_text", ["", "not-a-pid"])
def test_hparam_status_does_not_infer_terminal_from_corrupt_local_pid(tmp_path: Path, pid_text: str):
    pid_path = tmp_path / "running.pid"
    pid_path.write_text(pid_text)
    previous = {"status": "running", "pid_path": str(pid_path)}

    row = run_evidence.status_row(tmp_path, previous, previous)

    assert row["status"] == "running"


@pytest.mark.parametrize("pid_text", ["", "not-a-pid", "0", "-1"])
@pytest.mark.parametrize("status", ["planned", "pending"])
def test_hparam_launch_does_not_start_when_local_pid_is_corrupt(
    tmp_path: Path, monkeypatch, status: str, pid_text: str
):
    rows = _write_runtime_rows(tmp_path, [{"run_id": "run-000", "status": status}])
    Path(rows[0]["pid_path"]).write_text(pid_text)
    started = []
    monkeypatch.setattr(
        hparam_runtime,
        "_start_process",
        lambda _execution, command: started.append(command) or "launched",
    )

    hparam_runtime.launch_hparam_runs(tmp_path, dry_run=False)

    assert started == []
    assert _read_table(tmp_path / "run_status.tsv")[0]["status"] == "missing_pid"


@pytest.mark.parametrize("status", ["planned", "pending"])
def test_hparam_launch_recovers_after_transient_local_pid_read_error(tmp_path: Path, monkeypatch, status: str):
    rows = _write_runtime_rows(tmp_path, [{"run_id": "run-000", "status": status}])
    merge_run_manifest(tmp_path, [{"step_id": "train-model", "run_id": "run-000", "status": status}])
    pid_path = Path(rows[0]["pid_path"])
    pid_path.write_text("123")
    original_read_text = Path.read_text
    read_fails = {"value": True}

    def fail_pid_read(path: Path, *args, **kwargs):
        if path == pid_path and read_fails["value"]:
            raise OSError("temporary PID read failure")
        return original_read_text(path, *args, **kwargs)

    started = []
    monkeypatch.setattr(Path, "read_text", fail_pid_read)
    monkeypatch.setattr(
        hparam_runtime,
        "_start_process",
        lambda _execution, command: started.append(command) or "launched",
    )

    with pytest.raises(RuntimeError, match="PID file read failed"):
        hparam_runtime.launch_hparam_runs(tmp_path, dry_run=False)

    assert started == []
    assert _read_table(tmp_path / "run_status.tsv")[0]["status"] == status
    assert _read_table(tmp_path / "run_manifest.tsv")[0]["status"] == status

    read_fails["value"] = False
    pid_path.unlink()
    hparam_runtime.launch_hparam_runs(tmp_path, dry_run=False)

    assert len(started) == 1
    assert _read_table(tmp_path / "run_status.tsv")[0]["status"] == "launched"
    assert _read_table(tmp_path / "run_manifest.tsv")[0]["status"] == "launched"


@pytest.mark.parametrize("failure", ["directory", "invalid_utf8", "os_error", "dangling_symlink"])
def test_hparam_monitor_preserves_nonterminal_status_for_unreadable_local_pid(
    tmp_path: Path, monkeypatch, failure: str
):
    rows = _write_runtime_rows(tmp_path, [{"run_id": "run-000", "status": "running"}])
    pid_path = Path(rows[0]["pid_path"])
    if failure == "directory":
        pid_path.mkdir()
    elif failure == "invalid_utf8":
        pid_path.write_bytes(b"\xff")
    elif failure == "dangling_symlink":
        pid_path.symlink_to(tmp_path / "missing.pid")
    else:
        pid_path.write_text("123")
        original_read_text = Path.read_text

        def fail_pid_read(path: Path, *args, **kwargs):
            if path == pid_path:
                raise OSError("PID read failed")
            return original_read_text(path, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", fail_pid_read)

    hparam_runtime.monitor_hparam_runs(tmp_path)

    assert _read_table(tmp_path / "run_status.tsv")[0]["status"] == "running"
    assert _read_table(tmp_path / "run_manifest.tsv")[0]["status"] == "running"


def test_hparam_monitor_health_classifies_compute_active(tmp_path: Path, monkeypatch):
    pid_path = tmp_path / "running.pid"
    pid_path.write_text("123")
    _write_runtime_rows(
        tmp_path,
        [{"run_id": "running", "version": "v1", "pid_path": str(pid_path), "status": "launched"}],
    )
    monkeypatch.setattr(run_evidence, "process_running", lambda row, pid: True)
    monkeypatch.setattr(run_evidence, "gpu_summary", lambda row, pid: "123, GPU-1, 1024")
    monkeypatch.setattr(run_evidence, "proc_io", lambda row, pid: {})
    monkeypatch.setattr(run_evidence, "log_age_seconds", lambda path, row: None)
    monkeypatch.setattr(run_evidence, "read_run_progress", lambda run_dir, row: {"status": "missing"})

    monitor_hparam_runs(tmp_path, health=True)

    row = _read_table(tmp_path / "run_status.tsv")[0]
    assert row["health_status"] == "compute_active"
    assert row["gpu_summary"] == "123, GPU-1, 1024"


def test_hparam_monitor_health_classifies_data_loading_from_io_delta(tmp_path: Path, monkeypatch):
    pid_path = tmp_path / "running.pid"
    pid_path.write_text("123")
    rows = _write_runtime_rows(
        tmp_path,
        [{"run_id": "running", "version": "v1", "pid_path": str(pid_path), "status": "launched"}],
    )
    rows[0].update({"status": "running", "io_read_bytes": 100, "io_write_bytes": 50, "checkpoint_count": 0})
    manifests.write_rows(tmp_path / "run_status.tsv", rows)
    merge_run_manifest(
        tmp_path,
        [
            {
                "step_id": rows[0]["step_id"],
                "run_id": rows[0]["run_id"],
                "status": "running",
                "io_read_bytes": 100,
                "io_write_bytes": 50,
                "checkpoint_count": 0,
            }
        ],
    )
    monkeypatch.setattr(run_evidence, "process_running", lambda row, pid: True)
    monkeypatch.setattr(run_evidence, "gpu_summary", lambda row, pid: "")
    monkeypatch.setattr(run_evidence, "proc_io", lambda row, pid: {"read_bytes": 250, "write_bytes": 50})
    monkeypatch.setattr(run_evidence, "log_age_seconds", lambda path, row: None)
    monkeypatch.setattr(run_evidence, "read_run_progress", lambda run_dir, row: {"status": "missing"})

    monitor_hparam_runs(tmp_path, health=True)

    row = _read_table(tmp_path / "run_status.tsv")[0]
    assert row["health_status"] == "data_loading"
    assert row["io_read_delta_bytes"] == "150"


def test_hparam_monitor_health_classifies_stalled_and_unknown_remote(tmp_path: Path, monkeypatch):
    pid_path = tmp_path / "running.pid"
    pid_path.write_text("123")
    _write_runtime_rows(
        tmp_path,
        [
            {"run_id": "stalled", "version": "v1", "pid_path": str(pid_path), "status": "launched"},
            {
                "run_id": "remote",
                "version": "v2",
                "target": "ssh",
                "host": "baichuan3",
                "pid_path": str(pid_path),
                "status": "launched",
            },
        ],
    )

    def fake_running(row, pid):
        return None if row["run_id"] == "remote" else True

    monkeypatch.setattr(run_evidence, "process_running", fake_running)
    monkeypatch.setattr(run_evidence, "read_pid", lambda path, row: 123)
    monkeypatch.setattr(run_evidence, "gpu_summary", lambda row, pid: "")
    monkeypatch.setattr(run_evidence, "proc_io", lambda row, pid: {"read_bytes": 100, "write_bytes": 50})
    monkeypatch.setattr(run_evidence, "log_age_seconds", lambda path, row: 500)
    monkeypatch.setattr(run_evidence, "read_run_progress", lambda run_dir, row: {"status": "missing"})

    monitor_hparam_runs(tmp_path, health=True)

    status = {row["run_id"]: row["health_status"] for row in _read_table(tmp_path / "run_status.tsv")}
    assert status["stalled"] == "possibly_stalled"
    assert status["remote"] == "unknown_remote"


@pytest.mark.parametrize("failure", ["timeout", "ssh_error", "permission", "wrong_type", "missing", "ps_error"])
def test_hparam_monitor_remote_pid_probe_failure_is_unknown_until_recovery(tmp_path: Path, monkeypatch, failure: str):
    _write_runtime_rows(
        tmp_path,
        [
            {
                "run_id": "run-000",
                "target": "ssh",
                "host": "unit-host",
                "status": "running",
                "pid": "123",
            }
        ],
    )
    merge_run_manifest(tmp_path, [{"step_id": "train-model", "run_id": "run-000", "status": "running"}])
    probe = {"failure": failure}

    def fake_run(args, **kwargs):
        command = args[-1]
        if "checkpoint_dir = sys.argv[2]" in command:
            return subprocess.CompletedProcess(args, 0, '{"run_manifest": "", "checkpoints": []}', "")
        if "os.lstat" in command:
            assert "open(path" in command
            if probe["failure"] == "timeout":
                raise subprocess.TimeoutExpired(args, kwargs["timeout"])
            if probe["failure"] == "ssh_error":
                return subprocess.CompletedProcess(args, 255, "", "connection lost")
            if probe["failure"] == "permission":
                return subprocess.CompletedProcess(args, 1, "", "permission denied")
            if probe["failure"] == "wrong_type":
                return subprocess.CompletedProcess(args, 1, "", "is a directory")
            if probe["failure"] == "missing":
                return subprocess.CompletedProcess(args, 44, "", "missing")
            return subprocess.CompletedProcess(args, 0, "123\n", "")
        if command.startswith("ps "):
            if probe["failure"] == "ps_error":
                return subprocess.CompletedProcess(args, 2, "", "ps failed")
            return subprocess.CompletedProcess(args, 0, "123\n", "")
        return subprocess.CompletedProcess(args, 44, "", "missing")

    monkeypatch.setattr(run_evidence.subprocess, "run", fake_run)

    hparam_runtime.monitor_hparam_runs(tmp_path)
    first = _read_table(tmp_path / "run_status.tsv")[0]
    assert first["status"] == "unknown_remote"
    assert first["pid"] == ("" if failure == "missing" else "123")
    assert _read_table(tmp_path / "run_manifest.tsv")[0]["status"] == "unknown_remote"
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert [(event["from"], event["to"]) for event in events] == [("running", "unknown_remote")]

    hparam_runtime.monitor_hparam_runs(tmp_path)
    assert _read_table(tmp_path / "run_status.tsv")[0]["status"] == "unknown_remote"
    assert len((tmp_path / "events.jsonl").read_text().splitlines()) == 1

    probe["failure"] = ""
    hparam_runtime.monitor_hparam_runs(tmp_path)

    assert _read_table(tmp_path / "run_status.tsv")[0]["status"] == "running"
    assert _read_table(tmp_path / "run_manifest.tsv")[0]["status"] == "running"
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert [(event["from"], event["to"]) for event in events] == [
        ("running", "unknown_remote"),
        ("unknown_remote", "running"),
    ]


@pytest.mark.parametrize("uncertain_returncode", [124, 255, 1])
@pytest.mark.parametrize(
    ("recovered_log", "expected_status"), [("clean shutdown\n", "finished"), ("Traceback\n", "failed")]
)
def test_hparam_monitor_requires_explicit_remote_log_read_before_terminal_state(
    tmp_path: Path,
    monkeypatch,
    uncertain_returncode: int,
    recovered_log: str,
    expected_status: str,
):
    _write_runtime_rows(
        tmp_path,
        [
            {
                "run_id": "run-000",
                "target": "ssh",
                "host": "unit-host",
                "status": "running",
                "pid": "123",
            }
        ],
    )
    state = {"uncertain": True}

    def fake_remote_command(_row, command):
        if command.startswith("ps "):
            return subprocess.CompletedProcess([], 1, "", "")
        if "lines = file_obj.readlines()" in command:
            if state["uncertain"]:
                return subprocess.CompletedProcess([], uncertain_returncode, "", "permission or transport failure")
            return subprocess.CompletedProcess([], 0, recovered_log, "")
        if "checkpoint_dir = sys.argv[2]" in command:
            return subprocess.CompletedProcess([], 0, '{"run_manifest": "", "checkpoints": []}', "")
        if command.startswith("tail -n 8"):
            return subprocess.CompletedProcess(
                [], 0 if not state["uncertain"] else uncertain_returncode, recovered_log, ""
            )
        if "sys.stdout.write(file_obj.read())" in command:
            return subprocess.CompletedProcess([], 0, "123\n", "")
        raise AssertionError(f"Unexpected remote command: {command}")

    monkeypatch.setattr(run_evidence, "run_row_command", fake_remote_command)

    hparam_runtime.monitor_hparam_runs(tmp_path)
    hparam_runtime.monitor_hparam_runs(tmp_path)

    assert _read_table(tmp_path / "run_status.tsv")[0]["status"] == "unknown_remote"
    assert _read_table(tmp_path / "run_manifest.tsv")[0]["status"] == "unknown_remote"
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert [(event["from"], event["to"]) for event in events] == [("running", "unknown_remote")]

    state["uncertain"] = False
    hparam_runtime.monitor_hparam_runs(tmp_path)

    assert _read_table(tmp_path / "run_status.tsv")[0]["status"] == expected_status
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert [(event["from"], event["to"]) for event in events] == [
        ("running", "unknown_remote"),
        ("unknown_remote", expected_status),
    ]


@pytest.mark.parametrize(
    ("returncode", "expected"),
    [(0, 123), (run_evidence.REMOTE_MISSING_RETURN_CODE, None)],
)
def test_remote_pid_read_uses_lstat_and_open_missing_contract(monkeypatch, returncode: int, expected: int | None):
    commands = []

    def fake_run(_row, command):
        commands.append(command)
        return subprocess.CompletedProcess([], returncode, "123\n" if returncode == 0 else "", "")

    monkeypatch.setattr(run_evidence, "run_row_command", fake_run)

    assert run_evidence.read_pid("/remote/run.pid", {"target": "ssh", "host": "unit-host"}) == expected
    assert "os.lstat" in commands[0]
    assert "open(path" in commands[0]
    assert "[ -f" not in commands[0]


def test_hparam_monitor_health_requires_fresh_progress(tmp_path: Path, monkeypatch):
    pid_path = tmp_path / "running.pid"
    pid_path.write_text("123")
    rows = _write_runtime_rows(
        tmp_path,
        [{"run_id": "running", "version": "v1", "pid_path": str(pid_path), "status": "launched"}],
    )
    rows[0].update(
        {
            "status": "running",
            "progress_processed": 5,
            "progress_updated_at": "2000-01-01T00:00:00Z",
            "checkpoint_count": 0,
        }
    )
    manifests.write_rows(tmp_path / "run_status.tsv", rows)
    monkeypatch.setattr(run_evidence, "process_running", lambda row, pid: True)
    monkeypatch.setattr(run_evidence, "gpu_summary", lambda row, pid: "")
    monkeypatch.setattr(run_evidence, "proc_io", lambda row, pid: {})
    monkeypatch.setattr(run_evidence, "log_age_seconds", lambda path, row: 500)
    monkeypatch.setattr(
        run_evidence,
        "read_run_progress",
        lambda run_dir, row: {
            "status": "running",
            "processed": 5,
            "updated_at": "2000-01-01T00:00:00Z",
        },
    )

    monitor_hparam_runs(tmp_path, health=True)

    row = _read_table(tmp_path / "run_status.tsv")[0]
    assert row["health_status"] == "possibly_stalled"


def test_hparam_remote_command_timeout_returns_unknown_remote(monkeypatch):
    def fake_run(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(["ssh", "baichuan3", "ps"], 10)

    monkeypatch.setattr(run_evidence.subprocess, "run", fake_run)

    result = run_evidence.run_row_command({"target": "ssh", "host": "baichuan3"}, "ps")

    assert result.returncode == 124


def test_hparam_start_process_timeout_returns_launch_failed(monkeypatch):
    def fake_run(*_args, **kwargs):
        assert kwargs["timeout"] == hparam_runtime.LAUNCH_TIMEOUT_SECONDS
        raise subprocess.TimeoutExpired(["bash", "-lc", "cmd"], kwargs["timeout"])

    monkeypatch.setattr(hparam_runtime.subprocess, "run", fake_run)

    assert hparam_runtime._start_process({}, "cmd") == "launch_failed"
