from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading

from agent_tool_test_helpers import write_finetune_recipe, write_yaml
import pytest
import yaml

from agent_tools import decision_hparam, hparam_runtime, manifests, run_artifacts, run_evidence
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
            "gpus": "",
            "pid_path": str(managed_dir / "pid"),
            "log_path": str(managed_dir / "stdout.log"),
            "command": hparam_runtime._launch_command(
                {"workdir": str(root)},
                script,
                managed_dir / "stdout.log",
                managed_dir / "pid",
                [],
            ),
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
    (root / "recipe.resolved.yaml").write_text(
        yaml.safe_dump(
            {
                "experiment": experiment,
                "step": step,
                "execution": {"workdir": str(root)},
            },
            sort_keys=False,
        )
    )
    manifests.write_rows(
        root / "run_manifest.tsv",
        rows,
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


@pytest.mark.parametrize("operation", ["launch", "monitor", "stop"])
def test_hparam_runtime_does_not_reapply_stale_launch_snapshot_fields(tmp_path: Path, monkeypatch, operation: str):
    _write_runtime_rows(tmp_path, [{"run_id": "run-000", "status": "running"}])
    launch_rows = _read_table(tmp_path / "launch_manifest.tsv")
    launch_rows[0].update({"status": "planned", "score": "0.1", "wandb_url": "https://wandb.example/stale"})
    manifests.write_rows(tmp_path / "launch_manifest.tsv", launch_rows)
    manifests.write_rows(tmp_path / "run_status.tsv", launch_rows)
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
    started = []
    monkeypatch.setattr(
        hparam_runtime,
        "_start_process",
        lambda _execution, command: started.append(command) or "launched",
    )

    if operation == "launch":
        hparam_runtime.launch_hparam_runs(tmp_path, dry_run=False)
    elif operation == "monitor":
        hparam_runtime.monitor_hparam_runs(tmp_path)
    else:
        hparam_runtime.stop_hparam_run(tmp_path, "run-000", reason="manual stop")

    canonical = _read_table(tmp_path / "run_manifest.tsv")[0]
    assert started == []
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


def test_hparam_launch_serializes_concurrent_execute_calls(tmp_path: Path, monkeypatch):
    recipe = _hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    entered = threading.Event()
    release = threading.Event()
    started = []
    failures = []

    def start(_execution, command):
        started.append(command)
        entered.set()
        assert release.wait(timeout=5)
        return "launched"

    def launch():
        try:
            hparam_runtime.launch_hparam_runs(plan_dir, dry_run=False)
        except Exception as exc:
            failures.append(exc)

    monkeypatch.setattr(hparam_runtime, "_start_process", start)
    first = threading.Thread(target=launch)
    second = threading.Thread(target=launch)
    first.start()
    assert entered.wait(timeout=5)
    second.start()
    lock_probe = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import fcntl, sys\n"
                "with open(sys.argv[1], 'a+') as lock_file:\n"
                "    try:\n"
                "        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)\n"
                "    except BlockingIOError:\n"
                "        raise SystemExit(1)\n"
                "raise SystemExit(0)\n"
            ),
            str(tmp_path / "run_manifest.tsv.lock"),
        ],
    )
    assert lock_probe.returncode == 1
    assert len(started) == 1
    release.set()
    first.join(timeout=5)
    second.join(timeout=5)

    assert not first.is_alive()
    assert not second.is_alive()
    assert failures == []
    assert len(started) == 1


def test_hparam_launch_commits_execution_identity_before_start(tmp_path: Path, monkeypatch):
    _write_runtime_rows(tmp_path, [{"run_id": "run-000", "status": "planned"}])
    started = []

    def start_after_identity_commit(_execution, command):
        canonical = _read_table(tmp_path / "run_manifest.tsv")[0]
        assert canonical["status"] == "planned"
        assert canonical["target"] == "local"
        assert canonical["command"] == command
        assert canonical["pid_path"]
        started.append(command)
        return "launched"

    monkeypatch.setattr(hparam_runtime, "_start_process", start_after_identity_commit)

    hparam_runtime.launch_hparam_runs(tmp_path, dry_run=False)

    assert len(started) == 1
    assert _read_table(tmp_path / "run_manifest.tsv")[0]["status"] == "launched"


def test_hparam_launch_does_not_start_when_identity_precommit_fails(tmp_path: Path, monkeypatch):
    _write_runtime_rows(tmp_path, [{"run_id": "run-000", "status": "planned"}])
    started = []
    monkeypatch.setattr(
        hparam_runtime,
        "merge_run_manifest",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("identity precommit failed")),
    )
    monkeypatch.setattr(
        hparam_runtime, "_start_process", lambda _execution, command: started.append(command) or "launched"
    )

    with pytest.raises(RuntimeError, match="identity precommit failed"):
        hparam_runtime.launch_hparam_runs(tmp_path, dry_run=False)

    assert started == []


def test_hparam_launch_preserves_first_commit_when_second_start_raises(tmp_path: Path, monkeypatch):
    _write_runtime_rows(
        tmp_path,
        [{"run_id": "run-000", "status": "planned"}, {"run_id": "run-001", "status": "planned"}],
    )
    plan_path = tmp_path / "plan.json"
    plan = json.loads(plan_path.read_text())
    plan["recipe"]["execution"]["max_concurrent"] = 2
    plan_path.write_text(json.dumps(plan))
    resolved_path = tmp_path / "recipe.resolved.yaml"
    resolved = yaml.safe_load(resolved_path.read_text())
    resolved["execution"]["max_concurrent"] = 2
    resolved_path.write_text(yaml.safe_dump(resolved, sort_keys=False))
    starts = 0

    def fail_second_start(_execution, _command):
        nonlocal starts
        starts += 1
        if starts == 2:
            raise RuntimeError("second start failed")
        return "launched"

    monkeypatch.setattr(hparam_runtime, "_start_process", fail_second_start)

    with pytest.raises(RuntimeError, match="second start failed"):
        hparam_runtime.launch_hparam_runs(tmp_path, dry_run=False)

    rows = {row["run_id"]: row for row in _read_table(tmp_path / "run_manifest.tsv")}
    assert rows["run-000"]["status"] == "launched"
    assert rows["run-001"]["status"] == "planned"
    assert rows["run-001"]["target"] == "local"


def test_hparam_launch_artifact_reconciliation_never_starts_pending_runs_and_deduplicates_events(
    tmp_path: Path, monkeypatch
):
    _write_runtime_rows(
        tmp_path,
        [{"run_id": "run-000", "status": "launched"}, {"run_id": "run-001", "status": "pending"}],
    )
    real_append = hparam_runtime.append_event

    def append_then_raise(*args, **kwargs):
        real_append(*args, **kwargs)
        raise RuntimeError("event report failed")

    monkeypatch.setattr(hparam_runtime, "append_event", append_then_raise)
    monkeypatch.setattr(
        hparam_runtime,
        "_start_process",
        lambda *_args: (_ for _ in ()).throw(AssertionError("artifact reconciliation must not start a process")),
    )

    hparam_runtime.reconcile_hparam_launch_artifacts(tmp_path, {("train-model", "run-000")})
    hparam_runtime.reconcile_hparam_launch_artifacts(tmp_path, {("train-model", "run-000")})

    rows = {row["run_id"]: row for row in _read_table(tmp_path / "run_status.tsv")}
    assert rows["run-000"]["status"] == "launched"
    assert rows["run-001"]["status"] == "pending"
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert [event["event_type"] for event in events].count("run_launched") == 1


def test_hparam_launch_does_not_start_after_canonical_owner_commits_terminal_status(tmp_path: Path, monkeypatch):
    _write_runtime_rows(tmp_path, [{"run_id": "run-000", "status": "planned"}])
    real_merge = merge_run_manifest
    started = []

    def merge_after_wandb_update(root, rows, **_kwargs):
        kwargs = {"lock_held": True} if _kwargs.get("lock_held") else {}
        real_merge(root, [{"step_id": "train-model", "run_id": "run-000", "status": "failed"}], **kwargs)
        return real_merge(root, rows, **kwargs)

    monkeypatch.setattr(hparam_runtime, "merge_run_manifest", merge_after_wandb_update)
    monkeypatch.setattr(
        hparam_runtime, "_start_process", lambda _execution, command: started.append(command) or "launched"
    )

    hparam_runtime.launch_hparam_runs(tmp_path, dry_run=False)

    assert _read_table(tmp_path / "run_manifest.tsv")[0]["status"] == "failed"
    assert _read_table(tmp_path / "run_status.tsv")[0]["status"] == "failed"
    assert _read_table(tmp_path / "launch_manifest.tsv")[0]["status"] == "failed"
    assert started == []
    assert not (tmp_path / "events.jsonl").exists()


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


def test_hparam_launch_rejects_invalid_canonical_output_before_start(tmp_path: Path, monkeypatch):
    recipe = _hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    target = tmp_path / "run_matrix.csv"
    target.unlink()
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

    with pytest.raises(ValueError, match="Managed output"):
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

    with pytest.raises(ValueError, match="Managed output"):
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


def test_hparam_stop_rejects_invalid_canonical_output_before_kill(tmp_path: Path, monkeypatch):
    _write_runtime_rows(tmp_path, [{"run_id": "run-000", "status": "running"}])
    target = tmp_path / "run_matrix.csv"
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


def test_hparam_stop_mirrors_the_status_committed_by_the_canonical_owner(tmp_path: Path, monkeypatch):
    _write_runtime_rows(tmp_path, [{"run_id": "run-000", "status": "running"}])
    merge_run_manifest(tmp_path, [{"step_id": "train-model", "run_id": "run-000", "status": "running"}])
    real_merge = merge_run_manifest

    def merge_after_wandb_update(root, rows, **_kwargs):
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


@pytest.mark.parametrize(
    ("field", "changed"),
    [
        ("target", "ssh"),
        ("host", "other-host"),
        ("workdir", "/other/workdir"),
        ("gpus", "7"),
        ("pid_path", "/tmp/other.pid"),
        ("log_path", "/tmp/other.log"),
        ("command", "other-command"),
    ],
)
def test_hparam_stop_ignores_execution_identity_drift_in_projection(
    tmp_path: Path, monkeypatch, field: str, changed: str
):
    rows = _write_runtime_rows(tmp_path, [{"run_id": "run-000", "status": "running"}])
    merge_run_manifest(tmp_path, [rows[0]])
    rows[0][field] = changed
    manifests.write_rows(tmp_path / "launch_manifest.tsv", rows)
    calls = []
    monkeypatch.setattr(run_evidence, "read_pid", lambda *_args: calls.append("pid") or 123)
    monkeypatch.setattr(hparam_runtime.os, "kill", lambda *_args: calls.append("kill"))
    monkeypatch.setattr(hparam_runtime.subprocess, "run", lambda *_args, **_kwargs: calls.append("ssh"))

    hparam_runtime.stop_hparam_run(tmp_path, "run-000", reason="manual stop")

    assert calls == ["pid", "kill"]
    canonical = _read_table(tmp_path / "run_manifest.tsv")[0]
    assert _read_table(tmp_path / "launch_manifest.tsv")[0][field] == canonical[field]


@pytest.mark.parametrize(
    ("field", "changed"),
    [
        ("target", "ssh"),
        ("host", "other-host"),
        ("env", {"UNIT_CHANGED": "1"}),
        ("conda_env", "other-env"),
        ("gpu_pool", [7]),
        ("gpus_per_run", 2),
    ],
)
def test_hparam_launch_rejects_execution_drift_from_resolved_recipe_before_side_effects(
    tmp_path: Path, monkeypatch, field: str, changed
):
    recipe = _hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    plan_path = plan_dir / "plan.json"
    plan = json.loads(plan_path.read_text())
    plan["recipe"].setdefault("execution", {})[field] = changed
    plan_path.write_text(json.dumps(plan))
    before = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    calls = []
    monkeypatch.setattr(
        hparam_runtime,
        "_start_process",
        lambda *_args, **_kwargs: calls.append("start") or "launched",
    )
    real_validate = hparam_runtime.exp_io.validate_managed_output_paths

    def record_remote_probe(root, paths, remote=None):
        if remote is not None:
            calls.append("remote-probe")
        return real_validate(root, paths, remote=remote)

    monkeypatch.setattr(
        hparam_runtime.exp_io,
        "validate_managed_output_paths",
        record_remote_probe,
    )

    with pytest.raises(ValueError, match="recipe.resolved.yaml"):
        hparam_runtime.launch_hparam_runs(plan_dir, dry_run=False)

    assert calls == []
    assert {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before


def test_hparam_launch_rejects_base_runtime_drift_before_side_effects(tmp_path: Path, monkeypatch):
    recipe = _hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    plan_path = plan_dir / "plan.json"
    plan = json.loads(plan_path.read_text())
    plan["recipe"]["_base_recipe"]["runtime"]["devices"] = [7]
    plan_path.write_text(json.dumps(plan))
    before = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    calls = []
    monkeypatch.setattr(
        hparam_runtime,
        "_start_process",
        lambda *_args, **_kwargs: calls.append("start") or "launched",
    )

    with pytest.raises(ValueError, match="recipe.resolved.yaml"):
        hparam_runtime.launch_hparam_runs(plan_dir, dry_run=False)

    assert calls == []
    assert {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before


@pytest.mark.parametrize("operation", ["launch", "monitor"])
def test_hparam_runtime_ignores_uncommitted_launch_execution_identity(tmp_path: Path, monkeypatch, operation: str):
    recipe = _hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    run = json.loads((plan_dir / "plan.json").read_text())["runs"][0]
    manifests.write_rows(
        plan_dir / "launch_manifest.tsv",
        [
            {
                **run,
                "status": "launched",
                "target": "ssh",
                "host": "foreign-host",
                "workdir": "/foreign/workdir",
                "gpus": "7",
                "pid_path": "/foreign/run.pid",
                "log_path": "/foreign/run.log",
                "command": "foreign-command",
            }
        ],
    )
    calls = []
    monkeypatch.setattr(
        run_evidence,
        "status_row",
        lambda *_args, **_kwargs: calls.append("observe") or {},
    )
    monkeypatch.setattr(
        hparam_runtime,
        "_start_process",
        lambda *_args, **_kwargs: calls.append("start") or "launched",
    )

    if operation == "launch":
        hparam_runtime.launch_hparam_runs(plan_dir, dry_run=False)
    else:
        hparam_runtime.monitor_hparam_runs(plan_dir)

    canonical = _read_table(tmp_path / "run_manifest.tsv")[0]
    if operation == "launch":
        assert calls == ["start"]
        assert canonical["target"] == "local"
        assert _read_table(plan_dir / "launch_manifest.tsv")[0]["target"] == "local"
    else:
        assert calls == []
        assert canonical.get("target", "") == ""


@pytest.mark.parametrize(
    "failure",
    ["runtime_dir_file", "symlink", "dangling_symlink", "directory", "bad_encoding", "bad_json"],
)
def test_local_runtime_manifest_corruption_fails_closed(tmp_path: Path, monkeypatch, failure: str):
    rows = _write_runtime_rows(tmp_path, [{"run_id": "run-000", "status": "running"}])
    runtime_dir = Path(rows[0]["runtime_dir"])
    if failure == "runtime_dir_file":
        runtime_dir.parent.mkdir(parents=True)
        runtime_dir.write_text("not a directory")
    else:
        runtime_dir.mkdir(parents=True)
    manifest = runtime_dir / "run_manifest.json"
    if failure == "runtime_dir_file":
        pass
    elif failure == "symlink":
        foreign = tmp_path / "foreign.json"
        foreign.write_text(json.dumps({"metrics": {"val_ahi_pearson": 0.99}}))
        manifest.symlink_to(foreign)
    elif failure == "dangling_symlink":
        manifest.symlink_to(tmp_path / "missing.json")
    elif failure == "directory":
        manifest.mkdir()
    elif failure == "bad_encoding":
        manifest.write_bytes(b"\xff")
    else:
        manifest.write_text("{")
    monkeypatch.setattr(run_evidence, "process_running", lambda *_args: True)

    with pytest.raises((ValueError, UnicodeError), match="run manifest"):
        run_evidence.status_row(tmp_path, rows[0], rows[0])


@pytest.mark.parametrize(
    "failure",
    [
        "runtime_dir_file",
        "symlink",
        "dangling_symlink",
        "directory",
        "bad_encoding",
        "bad_json",
        "checkpoint_dir_symlink",
    ],
)
def test_remote_runtime_manifest_corruption_fails_closed(tmp_path: Path, monkeypatch, failure: str):
    runtime_dir = tmp_path / "remote-runtime"
    if failure == "runtime_dir_file":
        runtime_dir.write_text("not a directory")
    else:
        runtime_dir.mkdir()
    manifest = runtime_dir / "run_manifest.json"
    if failure == "runtime_dir_file":
        pass
    elif failure == "symlink":
        foreign = tmp_path / "foreign.json"
        foreign.write_text(json.dumps({"metrics": {"val_ahi_pearson": 0.99}}))
        manifest.symlink_to(foreign)
    elif failure == "dangling_symlink":
        manifest.symlink_to(tmp_path / "missing.json")
    elif failure == "directory":
        manifest.mkdir()
    elif failure == "bad_encoding":
        manifest.write_bytes(b"\xff")
    elif failure == "bad_json":
        manifest.write_text("{")
    else:
        manifest.write_text(json.dumps({"metrics": {"val_ahi_pearson": 0.7}}))
        target = tmp_path / "checkpoint-target"
        target.mkdir()
        (runtime_dir / "checkpoints").symlink_to(target, target_is_directory=True)
    row = {
        "step_id": "train-model",
        "run_id": "run-000",
        "status": "running",
        "target": "ssh",
        "host": "unit-host",
        "pid_path": "/remote/run.pid",
        "log_path": "/remote/run.log",
        "runtime_dir": str(runtime_dir),
        "checkpoint_dir": str(runtime_dir / "checkpoints"),
    }

    def fake_command(_row, command):
        if "sys.stdout.write(file_obj.read())" in command:
            return subprocess.CompletedProcess([], 0, "123\n", "")
        if command.startswith("ps "):
            return subprocess.CompletedProcess([], 0, "123\n", "")
        if "checkpoint_dir = sys.argv[2]" in command:
            assert "json.load" in command
            assert "stat.S_ISREG" in command
            assert "stat.S_ISDIR" in command
            return subprocess.run(["bash", "-lc", command], text=True, capture_output=True)
        if command.startswith("tail -n 8"):
            return subprocess.CompletedProcess([], 0, "running", "")
        raise AssertionError(command)

    monkeypatch.setattr(run_evidence, "run_row_command", fake_command)

    with pytest.raises(RuntimeError, match="runtime artifact observation failed"):
        run_evidence.status_row(tmp_path, row, row)


@pytest.mark.parametrize("state", ["missing", "regular"])
def test_remote_runtime_manifest_distinguishes_missing_and_regular_file(tmp_path: Path, monkeypatch, state: str):
    runtime_dir = tmp_path / "remote-runtime"
    runtime_dir.mkdir()
    manifest = runtime_dir / "run_manifest.json"
    if state == "regular":
        manifest.write_text(json.dumps({"metrics": {"val_ahi_pearson": 0.7}}))
    row = {
        "step_id": "train-model",
        "run_id": "run-000",
        "status": "running",
        "target": "ssh",
        "host": "unit-host",
        "pid_path": "/remote/run.pid",
        "log_path": "/remote/run.log",
        "runtime_dir": str(runtime_dir),
        "checkpoint_dir": str(runtime_dir / "checkpoints"),
    }

    def fake_command(_row, command):
        if "sys.stdout.write(file_obj.read())" in command:
            return subprocess.CompletedProcess([], 0, "123\n", "")
        if command.startswith("ps "):
            return subprocess.CompletedProcess([], 0, "123\n", "")
        if "checkpoint_dir = sys.argv[2]" in command:
            return subprocess.run(["bash", "-lc", command], text=True, capture_output=True)
        if command.startswith("tail -n 8"):
            return subprocess.CompletedProcess([], 0, "running", "")
        raise AssertionError(command)

    monkeypatch.setattr(run_evidence, "run_row_command", fake_command)

    observed = run_evidence.status_row(tmp_path, row, row)

    assert observed["run_manifest"] == (str(manifest) if state == "regular" else "")


def test_find_run_manifest_distinguishes_missing_and_valid_regular_file(tmp_path: Path):
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()
    run = {"runtime_dir": str(runtime_dir)}

    assert run_artifacts.find_run_manifest(run) is None

    manifest = runtime_dir / "run_manifest.json"
    manifest.write_text(json.dumps({"metrics": {"val_ahi_pearson": 0.7}}))

    assert run_artifacts.find_run_manifest(run) == manifest


def test_hparam_launch_binds_ssh_conda_gpu_and_pid_identity_only_after_a_launch_slot(
    tmp_path: Path,
    monkeypatch,
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
    assert "CUDA_VISIBLE_DEVICES=6,7" in rows[0]["command"]
    canonical = _read_table(tmp_path / "run_manifest.tsv")
    assert canonical[0]["target"] == ""
    assert canonical[0]["gpus"] == ""
    assert canonical[0]["command"] == ""
    status = _read_table(plan_dir / "run_status.tsv")
    assert status[0]["target"] == ""
    assert status[0]["gpus"] == ""
    assert status[0]["command"] == ""
    script = Path(rows[0]["script"]).read_text()
    assert "--wandb-project sleep2vec-unit-hparam" in script
    assert "--wandb-group unit" in script
    assert not (plan_dir / "logs").exists()
    assert not (plan_dir / "pids").exists()
    real_validate = hparam_runtime.exp_io.validate_managed_output_paths

    def validate_without_remote(root, paths, remote=None):
        if remote is None:
            return real_validate(root, paths)

    started = []
    monkeypatch.setattr(hparam_runtime.exp_io, "validate_managed_output_paths", validate_without_remote)
    monkeypatch.setattr(
        hparam_runtime,
        "_start_process",
        lambda _execution, command: started.append(command) or "launched",
    )

    hparam_runtime.launch_hparam_runs(plan_dir, dry_run=False)

    rows = _read_table(plan_dir / "launch_manifest.tsv")
    assert rows[0]["status"] == "launched"
    assert rows[0]["target"] == "ssh"
    assert rows[0]["host"] == "baichuan3"
    assert rows[0]["gpus"] == "6,7"
    assert "ssh baichuan3" in rows[0]["command"]
    assert "mkdir -p" in rows[0]["command"]
    assert "(nohup env " in rows[0]["command"]
    assert "conda run --no-capture-output -n ywx" in rows[0]["command"]
    assert "CUDA_VISIBLE_DEVICES=6,7" in rows[0]["command"]
    assert "WANDB_PROJECT=" not in rows[0]["command"]
    assert "WANDB_GROUP=" not in rows[0]["command"]
    assert "WANDB_RUN_GROUP=" not in rows[0]["command"]
    assert rows[0]["log_path"].endswith("runs/run-000--lr-1e-6/stdout.log")
    assert rows[0]["pid_path"].endswith("runs/run-000--lr-1e-6/pid")
    assert started == [rows[0]["command"]]


@pytest.mark.parametrize(
    ("execution", "runtime_devices", "expected_devices"),
    [
        ({"gpu_pool": [6, 7], "gpus_per_run": 2}, [0], "0 1"),
        ({"gpu_pool": [6, 7], "gpus_per_run": 1}, [0, 1], "0"),
        ({"gpus_per_run": 1}, [6, 7], "0"),
    ],
)
def test_hparam_plan_uses_logical_devices_for_scheduled_gpu_groups(
    tmp_path: Path,
    execution: dict,
    runtime_devices,
    expected_devices: str,
):
    recipe = _hparam_recipe(tmp_path, execution={"workdir": str(tmp_path), **execution})
    payload = yaml.safe_load(recipe.read_text())
    base_recipe = Path(payload["base_recipe"])
    base_payload = yaml.safe_load(base_recipe.read_text())
    base_payload["runtime"]["devices"] = runtime_devices
    write_yaml(base_recipe, base_payload)
    plan_dir = tmp_path / "plan"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir))

    assert result.returncode == 0, result.stderr or result.stdout
    command = json.loads((plan_dir / "plan.json").read_text())["runs"][0]["command"]
    assert f"--devices {expected_devices} --precision" in command


def test_hparam_plan_rejects_gpus_per_run_without_a_physical_pool_before_workspace_creation(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path, execution={"workdir": str(tmp_path), "gpus_per_run": 2})
    payload = yaml.safe_load(recipe.read_text())
    base_recipe = Path(payload["base_recipe"])
    base_payload = yaml.safe_load(base_recipe.read_text())
    base_payload["runtime"].pop("devices")
    write_yaml(base_recipe, base_payload)
    plan_dir = tmp_path / "plan"
    before = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}

    doctor = _run("doctor", "--recipe", str(recipe))
    planned = _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir))

    message = "execution.gpus_per_run requires a non-empty execution.gpu_pool or runtime.devices"
    assert doctor.returncode == 1
    assert message in doctor.stdout
    assert planned.returncode == 1
    assert message in planned.stdout
    assert not plan_dir.exists()
    assert {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before


@pytest.mark.parametrize("gpus_per_run", [0, False, 0.5, 1.5, "0.5", "1.5"])
def test_hparam_execution_reports_invalid_gpus_per_run(gpus_per_run):
    issues = decision_hparam._hparam_execution_issues(
        {"gpu_pool": [0, 1], "gpus_per_run": gpus_per_run},
        {},
    )

    assert len(issues) == 1
    assert issues[0].field == "execution.gpus_per_run"
    assert issues[0].status.value == "FAIL"
    assert "must be a positive integer" in issues[0].message


def test_hparam_runtime_rejects_gpus_per_run_without_a_physical_pool():
    with pytest.raises(
        ValueError,
        match="execution.gpus_per_run requires a non-empty execution.gpu_pool or runtime.devices",
    ):
        hparam_runtime._gpu_groups({"execution": {"gpus_per_run": 2}})


def test_hparam_launch_defaults_to_one_run_per_gpu_group_and_uses_the_free_group(tmp_path: Path, monkeypatch):
    recipe = _hparam_recipe(
        tmp_path,
        execution={"workdir": str(tmp_path), "gpu_pool": [0, 1], "gpus_per_run": 1},
    )
    payload = yaml.safe_load(recipe.read_text())
    payload["search"]["max_runs"] = 4
    payload["search"]["parameters"]["runtime.lr"] = [1e-6, 2e-6, 3e-6, 4e-6]
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
    plan_dir = tmp_path / "plan"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    started = []
    monkeypatch.setattr(
        hparam_runtime,
        "_start_process",
        lambda _execution, command: started.append(command) or "launched",
    )

    hparam_runtime.launch_hparam_runs(plan_dir, dry_run=False)

    rows = _read_table(plan_dir / "launch_manifest.tsv")
    assert len(started) == 2
    assert [row["gpus"] for row in rows] == ["0", "1", "", ""]
    assert [row["status"] for row in rows] == ["launched", "launched", "pending", "pending"]
    assert all(rows[index]["target"] == "" and rows[index]["command"] == "" for index in (2, 3))

    merge_run_manifest(
        tmp_path,
        [{"step_id": rows[1]["step_id"], "run_id": rows[1]["run_id"], "status": "finished"}],
    )
    started.clear()
    hparam_runtime.launch_hparam_runs(plan_dir, dry_run=False)

    rows = _read_table(plan_dir / "launch_manifest.tsv")
    assert len(started) == 1
    assert "CUDA_VISIBLE_DEVICES=1" in started[0]
    assert [row["gpus"] for row in rows] == ["0", "1", "1", ""]
    assert [row["status"] for row in rows] == ["missing_pid", "finished", "launched", "pending"]


def test_hparam_launch_counts_active_gpu_load_from_previous_plan(tmp_path: Path, monkeypatch):
    execution = {"workdir": str(tmp_path), "gpu_pool": [0, 1], "gpus_per_run": 1}
    first_recipe = _hparam_recipe(tmp_path, execution=execution)
    first_plan = tmp_path / "plan-1"
    assert _run("plan", "--recipe", str(first_recipe), "--output-dir", str(first_plan)).returncode == 0

    second_payload = yaml.safe_load(first_recipe.read_text())
    second_payload["search"]["max_runs"] = 2
    second_payload["search"]["parameters"]["runtime.lr"] = [2e-6, 3e-6]
    second_recipe = write_yaml(tmp_path / "tune-2.yaml", second_payload)
    second_plan = tmp_path / "plan-2"
    assert _run("plan", "--recipe", str(second_recipe), "--output-dir", str(second_plan)).returncode == 0
    started = []
    monkeypatch.setattr(
        hparam_runtime,
        "_start_process",
        lambda _execution, command: started.append(command) or "launched",
    )

    hparam_runtime.launch_hparam_runs(second_plan, dry_run=True)
    assert [row["gpus"] for row in _read_table(second_plan / "launch_manifest.tsv")] == ["0", "1"]
    second_keys = {
        (row["step_id"], row["run_id"]) for row in json.loads((second_plan / "plan.json").read_text())["runs"]
    }
    assert all(
        row["target"] == "" and row["gpus"] == "" and row["command"] == ""
        for row in _read_table(tmp_path / "run_manifest.tsv")
        if (row["step_id"], row["run_id"]) in second_keys
    )
    hparam_runtime.launch_hparam_runs(first_plan, dry_run=False)
    started.clear()
    hparam_runtime.launch_hparam_runs(second_plan, dry_run=False)

    rows = _read_table(second_plan / "launch_manifest.tsv")
    assert len(started) == 1
    assert "CUDA_VISIBLE_DEVICES=1" in started[0]
    assert [row["gpus"] for row in rows] == ["1", ""]
    assert [row["status"] for row in rows] == ["launched", "pending"]


def test_hparam_launch_full_previous_plan_keeps_replacement_pending(tmp_path: Path, monkeypatch):
    execution = {"workdir": str(tmp_path), "gpu_pool": [0, 1], "gpus_per_run": 1}
    first_recipe = _hparam_recipe(tmp_path, execution=execution)
    first_payload = yaml.safe_load(first_recipe.read_text())
    first_payload["search"]["max_runs"] = 2
    first_payload["search"]["parameters"]["runtime.lr"] = [1e-6, 2e-6]
    first_recipe.write_text(yaml.safe_dump(first_payload, sort_keys=False))
    first_plan = tmp_path / "plan-1"
    assert _run("plan", "--recipe", str(first_recipe), "--output-dir", str(first_plan)).returncode == 0

    second_payload = yaml.safe_load(first_recipe.read_text())
    second_payload["search"]["max_runs"] = 1
    second_payload["search"]["parameters"]["runtime.lr"] = [3e-6]
    second_recipe = write_yaml(tmp_path / "tune-2.yaml", second_payload)
    second_plan = tmp_path / "plan-2"
    assert _run("plan", "--recipe", str(second_recipe), "--output-dir", str(second_plan)).returncode == 0
    started = []
    monkeypatch.setattr(
        hparam_runtime,
        "_start_process",
        lambda _execution, command: started.append(command) or "launched",
    )

    hparam_runtime.launch_hparam_runs(first_plan, dry_run=False)
    first_rows = _read_table(first_plan / "launch_manifest.tsv")
    assert [row["gpus"] for row in first_rows] == ["0", "1"]
    started.clear()

    hparam_runtime.launch_hparam_runs(second_plan, dry_run=False)

    row = _read_table(second_plan / "launch_manifest.tsv")[0]
    assert row["status"] == "pending"
    assert row["gpus"] == ""
    assert started == []


def test_hparam_launch_keeps_cpu_only_concurrency_plan_local(tmp_path: Path, monkeypatch):
    execution = {"workdir": str(tmp_path)}
    first_recipe = _hparam_recipe(tmp_path, execution=execution)
    first_payload = yaml.safe_load(first_recipe.read_text())
    first_payload["runtime"] = {"devices": []}
    write_yaml(first_recipe, first_payload)
    first_plan = tmp_path / "plan-1"
    assert _run("plan", "--recipe", str(first_recipe), "--output-dir", str(first_plan)).returncode == 0

    second_payload = yaml.safe_load(first_recipe.read_text())
    second_payload["search"]["max_runs"] = 2
    second_payload["search"]["parameters"]["runtime.lr"] = [2e-6, 3e-6]
    second_recipe = write_yaml(tmp_path / "tune-2.yaml", second_payload)
    second_plan = tmp_path / "plan-2"
    assert _run("plan", "--recipe", str(second_recipe), "--output-dir", str(second_plan)).returncode == 0
    started = []
    monkeypatch.setattr(
        hparam_runtime,
        "_start_process",
        lambda _execution, command: started.append(command) or "launched",
    )

    hparam_runtime.launch_hparam_runs(first_plan, dry_run=False)
    started.clear()
    hparam_runtime.launch_hparam_runs(second_plan, dry_run=False)

    rows = _read_table(second_plan / "launch_manifest.tsv")
    assert len(started) == 1
    assert [row["status"] for row in rows] == ["launched", "pending"]


def test_hparam_launch_explicit_gpu_oversubscription_warns_and_balances_groups(tmp_path: Path, monkeypatch):
    recipe = _hparam_recipe(
        tmp_path,
        execution={
            "workdir": str(tmp_path),
            "gpu_pool": [0, 1],
            "gpus_per_run": 1,
            "max_concurrent": 4,
        },
    )
    payload = yaml.safe_load(recipe.read_text())
    payload["search"]["max_runs"] = 4
    payload["search"]["parameters"]["runtime.lr"] = [1e-6, 2e-6, 3e-6, 4e-6]
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
    plan_dir = tmp_path / "plan"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir))

    assert result.returncode == 0, result.stderr
    assert "Status: WARN" in result.stdout
    assert "GPU oversubscription is explicitly enabled" in result.stdout
    started = []
    monkeypatch.setattr(
        hparam_runtime,
        "_start_process",
        lambda _execution, command: started.append(command) or "launched",
    )

    hparam_runtime.launch_hparam_runs(plan_dir, dry_run=False)

    rows = _read_table(plan_dir / "launch_manifest.tsv")
    assert len(started) == 4
    assert [row["gpus"] for row in rows] == ["0", "1", "0", "1"]
    assert {row["status"] for row in rows} == {"launched"}


def test_hparam_launch_explicit_oversubscription_balances_overlapping_previous_group(tmp_path: Path, monkeypatch):
    first_recipe = _hparam_recipe(
        tmp_path,
        execution={"workdir": str(tmp_path), "gpu_pool": [0, 1], "gpus_per_run": 2},
    )
    first_plan = tmp_path / "plan-1"
    assert _run("plan", "--recipe", str(first_recipe), "--output-dir", str(first_plan)).returncode == 0

    second_payload = yaml.safe_load(first_recipe.read_text())
    second_payload["execution"] = {
        "workdir": str(tmp_path),
        "gpu_pool": [0, 1, 2],
        "gpus_per_run": 1,
        "max_concurrent": 4,
    }
    second_payload["search"]["max_runs"] = 4
    second_payload["search"]["parameters"]["runtime.lr"] = [2e-6, 3e-6, 4e-6, 5e-6]
    second_recipe = write_yaml(tmp_path / "tune-2.yaml", second_payload)
    second_plan = tmp_path / "plan-2"
    assert _run("plan", "--recipe", str(second_recipe), "--output-dir", str(second_plan)).returncode == 0
    started = []
    monkeypatch.setattr(
        hparam_runtime,
        "_start_process",
        lambda _execution, command: started.append(command) or "launched",
    )

    hparam_runtime.launch_hparam_runs(first_plan, dry_run=False)
    started.clear()
    hparam_runtime.launch_hparam_runs(second_plan, dry_run=False)

    rows = _read_table(second_plan / "launch_manifest.tsv")
    assert len(started) == 3
    assert "CUDA_VISIBLE_DEVICES=2" in started[0]
    assert [row["gpus"] for row in rows] == ["2", "0", "1", ""]
    assert [row["status"] for row in rows] == ["launched", "launched", "launched", "pending"]


@pytest.mark.parametrize(
    ("different_field", "expected_gpus", "expected_statuses"),
    [
        ("host", ["0", "1"], ["launched", "launched"]),
        ("workdir", ["1", ""], ["launched", "pending"]),
        ("local_host", ["1", ""], ["launched", "pending"]),
    ],
)
def test_hparam_launch_scopes_active_gpu_load_by_target_and_ssh_host(
    tmp_path: Path,
    monkeypatch,
    different_field: str,
    expected_gpus: list[str],
    expected_statuses: list[str],
):
    first_execution = {
        "target": "ssh",
        "host": "host-a",
        "workdir": str(tmp_path / "remote-a"),
        "gpu_pool": [0, 1],
        "gpus_per_run": 1,
    }
    if different_field == "local_host":
        first_execution["target"] = "local"
        first_execution["host"] = "local-label-a"
    second_execution = dict(first_execution)
    if different_field == "host":
        second_execution["host"] = "host-b"
    elif different_field == "workdir":
        second_execution["workdir"] = str(tmp_path / "remote-b")
    else:
        second_execution["host"] = "local-label-b"
    first_recipe = _hparam_recipe(tmp_path, execution=first_execution)
    first_plan = tmp_path / "plan-1"
    assert _run("plan", "--recipe", str(first_recipe), "--output-dir", str(first_plan)).returncode == 0

    second_payload = yaml.safe_load(first_recipe.read_text())
    second_payload["execution"] = second_execution
    second_payload["search"]["max_runs"] = 2
    second_payload["search"]["parameters"]["runtime.lr"] = [2e-6, 3e-6]
    second_recipe = write_yaml(tmp_path / "tune-2.yaml", second_payload)
    second_plan = tmp_path / "plan-2"
    assert _run("plan", "--recipe", str(second_recipe), "--output-dir", str(second_plan)).returncode == 0
    real_validate = hparam_runtime.exp_io.validate_managed_output_paths

    def validate_without_remote(root, paths, remote=None):
        if remote is None:
            return real_validate(root, paths)

    started = []
    monkeypatch.setattr(hparam_runtime.exp_io, "validate_managed_output_paths", validate_without_remote)
    monkeypatch.setattr(
        hparam_runtime,
        "_start_process",
        lambda _execution, command: started.append(command) or "launched",
    )

    hparam_runtime.launch_hparam_runs(first_plan, dry_run=False)
    started.clear()
    hparam_runtime.launch_hparam_runs(second_plan, dry_run=False)

    rows = _read_table(second_plan / "launch_manifest.tsv")
    assert len(started) == expected_statuses.count("launched")
    assert [row["gpus"] for row in rows] == expected_gpus
    assert [row["status"] for row in rows] == expected_statuses
    if different_field in {"workdir", "local_host"}:
        assert "CUDA_VISIBLE_DEVICES=1" in started[0]


def test_hparam_plan_rejects_duplicate_gpu_assignments_within_a_run(tmp_path: Path):
    recipe = _hparam_recipe(
        tmp_path,
        execution={"workdir": str(tmp_path), "gpu_pool": [0, 0], "gpus_per_run": 2},
    )

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(tmp_path / "plan"))

    assert result.returncode == 1
    assert "must not contain duplicate GPU identifiers" in result.stdout


@pytest.mark.parametrize("env_name", ["WANDB_PROJECT", "WANDB_GROUP", "WANDB_RUN_GROUP", "WANDB_MODE"])
def test_hparam_plan_rejects_wandb_environment_aliases(tmp_path: Path, env_name: str):
    recipe = _hparam_recipe(
        tmp_path,
        execution={"workdir": str(tmp_path), "env": {env_name: "unit"}},
    )

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(tmp_path / "plan"))

    assert result.returncode == 1
    assert f"execution.env.{env_name}" in result.stdout


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
        if "checkpoint_dir = sys.argv[2]" in command:
            return subprocess.CompletedProcess([], 0, json.dumps({"run_manifest": "", "checkpoints": []}), "")
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


def test_hparam_launch_accepts_scalar_runtime_devices(tmp_path: Path, monkeypatch):
    recipe = _hparam_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    base_recipe = Path(payload["base_recipe"])
    base_payload = yaml.safe_load(base_recipe.read_text())
    base_payload["runtime"]["devices"] = 2
    write_yaml(base_recipe, base_payload)
    plan_dir = tmp_path / "plan"

    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    started = []
    monkeypatch.setattr(
        hparam_runtime,
        "_start_process",
        lambda _execution, command: started.append(command) or "launched",
    )

    hparam_runtime.launch_hparam_runs(plan_dir, dry_run=False)

    rows = _read_table(plan_dir / "launch_manifest.tsv")
    assert rows[0]["gpus"] == "2"
    assert "--devices 2 --precision" in Path(rows[0]["script"]).read_text()
    assert "(nohup env " in rows[0]["command"]
    assert "CUDA_VISIBLE_DEVICES=2" in rows[0]["command"]
    assert started == [rows[0]["command"]]


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
    payload["execution"] = {"workdir": str(tmp_path), "max_concurrent": 1}
    recipe.write_text(yaml.safe_dump(payload))
    plan_dir = tmp_path / "plan"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    started = []
    monkeypatch.setattr(
        hparam_runtime, "_start_process", lambda _execution, command: started.append(command) or "launched"
    )

    hparam_runtime.launch_hparam_runs(plan_dir, dry_run=False)
    assert len(started) == 1
    started.clear()
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


def test_hparam_runtime_rewrites_legacy_projection_rows_from_canonical(tmp_path: Path, monkeypatch):
    rows = _write_runtime_rows(tmp_path, [{"run_id": "run-000", "version": "v0", "status": "launched"}])
    legacy_rows = [{**rows[0], "trial_id": "trial_000"}]
    manifests.write_rows(tmp_path / "launch_manifest.tsv", legacy_rows)
    started = []
    killed = []
    monkeypatch.setattr(
        hparam_runtime, "_start_process", lambda _execution, command: started.append(command) or "launched"
    )
    monkeypatch.setattr(hparam_runtime.os, "kill", lambda pid, sig: killed.append((pid, sig)))

    hparam_runtime.launch_hparam_runs(tmp_path, dry_run=True)

    assert started == []
    assert killed == []
    assert "trial_id" not in (tmp_path / "launch_manifest.tsv").read_text()
    assert "trial_id" not in (tmp_path / "run_status.tsv").read_text()


@pytest.mark.parametrize("table", ["launch_manifest.tsv", "run_status.tsv"])
def test_hparam_runtime_rewrites_header_only_removed_projection_table(tmp_path: Path, monkeypatch, table: str):
    _write_runtime_rows(tmp_path, [{"run_id": "run-000", "version": "v0", "status": "launched"}])
    (tmp_path / table).write_text("trial_id\n")
    started = []
    killed = []
    monkeypatch.setattr(
        hparam_runtime, "_start_process", lambda _execution, command: started.append(command) or "launched"
    )
    monkeypatch.setattr(hparam_runtime.os, "kill", lambda pid, sig: killed.append((pid, sig)))

    hparam_runtime.launch_hparam_runs(tmp_path, dry_run=True)

    assert started == []
    assert killed == []
    assert "trial_id" not in (tmp_path / table).read_text()


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

    def merge_after_wandb_update(root, rows, **_kwargs):
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


def test_hparam_monitor_without_launch_manifest_uses_canonical_execution_evidence(tmp_path: Path):
    _write_runtime_rows(tmp_path, [{"run_id": "run-000", "status": "running"}])
    workspace_rows = _read_table(tmp_path / "run_manifest.tsv")
    workspace_rows[0]["status"] = "running"
    manifests.write_rows(tmp_path / "run_manifest.tsv", workspace_rows)
    (tmp_path / "launch_manifest.tsv").unlink()

    hparam_runtime.monitor_hparam_runs(tmp_path)
    hparam_runtime.monitor_hparam_runs(tmp_path)

    assert _read_table(tmp_path / "run_manifest.tsv")[0]["status"] == "finished"
    assert _read_table(tmp_path / "run_status.tsv")[0]["status"] == "finished"
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert [event["event_type"] for event in events].count("run_status_changed") == 1


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
