from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import threading
import time

import pytest
from test_agent_tools_hparam_runtime import (
    _REAL_VALIDATED_EXECUTION_SNAPSHOT,
    _hparam_recipe,
    _is_remote_python_program,
    _process_identity,
    _read_table,
    _run,
    _set_execution_probe,
    _write_process_identity,
    _write_runtime_rows,
    write_yaml,
)
from test_agent_tools_hparam_runtime import _stub_execution_snapshot_preflight  # noqa: F401
import yaml

from agent_tools import hparam_runtime, managed_scheduler, manifests, plan_rendering, run_evidence
from agent_tools.experiment_workspace import MONITOR_EXIT_CODE_PREFIX, file_sha256, merge_run_manifest, merge_run_row


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
    assert step_manifest["plan_controller"] == "ordinary"
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
    rows = _write_runtime_rows(
        tmp_path,
        [{"run_id": "run-000", "status": "running", **_process_identity()}],
    )
    _write_process_identity(rows[0]["pid_path"])
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
        lambda _root, row, previous, *, script_commits_terminal_status, health=False: merge_run_row(previous, row),
    )
    monkeypatch.setattr(run_evidence, "stop_process_group", lambda *_args: None)
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
    plan["resolved_recipe_sha256"] = file_sha256(resolved_path)
    plan_path.write_text(json.dumps(plan))
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

    with pytest.raises(ValueError, match="Managed file is missing or aliased"):
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


def test_hparam_launch_rejects_synchronized_recipe_drift_before_side_effects(tmp_path: Path, monkeypatch):
    recipe = _hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    plan_path = plan_dir / "plan.json"
    plan = json.loads(plan_path.read_text())
    plan["recipe"].setdefault("execution", {})["max_concurrent"] = 2
    plan_path.write_text(json.dumps(plan))
    resolved_path = plan_dir / "recipe.resolved.yaml"
    resolved = yaml.safe_load(resolved_path.read_text())
    resolved.setdefault("execution", {})["max_concurrent"] = 2
    resolved_path.write_text(yaml.safe_dump(resolved, sort_keys=False))
    before = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    calls = []
    monkeypatch.setattr(
        hparam_runtime,
        "_start_process",
        lambda *_args, **_kwargs: calls.append("start") or "launched",
    )

    with pytest.raises(ValueError, match="Frozen hparam recipe SHA-256"):
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
    assert "start_new_session=True" in rows[0]["command"]
    assert "conda run --no-capture-output -n ywx" in rows[0]["command"]
    assert "CUDA_VISIBLE_DEVICES=6,7" in rows[0]["command"]
    assert "WANDB_PROJECT=" not in rows[0]["command"]
    assert "WANDB_GROUP=" not in rows[0]["command"]
    assert "WANDB_RUN_GROUP=" not in rows[0]["command"]
    assert rows[0]["log_path"].endswith("runs/run-000--lr-1e-6/stdout.log")
    assert rows[0]["pid_path"].endswith("runs/run-000--lr-1e-6/pid")
    assert started == [rows[0]["command"]]


def test_hparam_run_queue_fails_on_missing_pid_capacity_blocker_from_another_plan(tmp_path: Path, monkeypatch):
    execution = {"workdir": str(tmp_path), "gpu_pool": [0], "gpus_per_run": 1}
    first_recipe = _hparam_recipe(tmp_path, execution=execution)
    first_plan = tmp_path / "plan-1"
    assert _run("plan", "--recipe", str(first_recipe), "--output-dir", str(first_plan)).returncode == 0
    monkeypatch.setattr(hparam_runtime, "_start_process", lambda *_args: "launched")
    hparam_runtime.launch_hparam_runs(first_plan, dry_run=False)
    first_run = json.loads((first_plan / "plan.json").read_text())["runs"][0]
    merge_run_manifest(
        tmp_path,
        [{"step_id": first_run["step_id"], "run_id": first_run["run_id"], "status": "missing_pid"}],
    )

    second_payload = yaml.safe_load(first_recipe.read_text())
    second_payload["search"]["parameters"]["runtime.lr"] = [2e-6]
    second_recipe = write_yaml(tmp_path / "tune-2.yaml", second_payload)
    second_plan = tmp_path / "plan-2"
    assert _run("plan", "--recipe", str(second_recipe), "--output-dir", str(second_plan)).returncode == 0
    monkeypatch.setattr(
        hparam_runtime.evidence,
        "status_row",
        lambda _root, observation, previous, *, script_commits_terminal_status, health: {
            **previous,
            **observation,
        },
    )
    monkeypatch.setattr(hparam_runtime, "_start_process", lambda *_args: pytest.fail("blocked queue must not start"))
    monkeypatch.setattr(hparam_runtime.time, "sleep", lambda *_args: pytest.fail("blocked queue must not sleep"))

    with pytest.raises(
        managed_scheduler.MissingPidCapacityError,
        match=f"{first_run['run_id']} has status missing_pid",
    ) as exc_info:
        hparam_runtime.run_hparam_queue(second_plan, dry_run=False)

    assert exc_info.value.step_id == first_run["step_id"]
    assert exc_info.value.run_id == first_run["run_id"]


def test_hparam_launch_revalidates_verified_execution_target_before_start(tmp_path: Path, monkeypatch):
    recipe = _hparam_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    payload["search"]["max_runs"] = 2
    payload["search"]["parameters"]["runtime.lr"] = [1e-6, 2e-6]
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
    plan_dir = tmp_path / "plan"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    calls = _set_execution_probe(monkeypatch, plan_dir)
    monkeypatch.setattr(hparam_runtime, "_validated_execution_snapshot", _REAL_VALIDATED_EXECUTION_SNAPSHOT)
    monkeypatch.setattr(
        hparam_runtime,
        "_start_process",
        lambda _execution, _command: calls.append("start") or "launched",
    )

    hparam_runtime.launch_hparam_runs(plan_dir, dry_run=False)

    snapshot = json.loads((plan_dir / hparam_runtime.EXECUTION_SNAPSHOT_NAME).read_text())
    assert calls == ["identity", "parse", "start"]
    assert snapshot["python"] == json.loads((plan_dir / "plan.json").read_text())["recipe"]["execution"]["python"]
    assert (
        snapshot["runtime_commit"]
        == json.loads((plan_dir / "plan.json").read_text())["recipe"]["execution"]["runtime_commit"]
    )
    assert snapshot["runtime_hostname"] == "test-runtime"
    assert snapshot["module"] == "sleep2vec.finetune"
    assert set(snapshot["required_options"]).issubset(snapshot["supported_options"])
    assert not list(plan_dir.glob(f".{hparam_runtime.EXECUTION_SNAPSHOT_NAME}.*"))


def test_hparam_launch_rejects_pre_identity_plan_without_writes(tmp_path: Path, monkeypatch):
    recipe = _hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    plan = json.loads((plan_dir / "plan.json").read_text())
    resolved = yaml.safe_load((plan_dir / "recipe.resolved.yaml").read_text())
    for payload in (plan["recipe"], resolved):
        payload["execution"].pop("python")
        payload["execution"].pop("runtime_commit")
    (plan_dir / "plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    (plan_dir / "recipe.resolved.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False))
    plan["resolved_recipe_sha256"] = file_sha256(plan_dir / "recipe.resolved.yaml")
    (plan_dir / "plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    manifest_before = (tmp_path / "run_manifest.tsv").read_bytes()
    monkeypatch.setattr(hparam_runtime, "_validated_execution_snapshot", _REAL_VALIDATED_EXECUTION_SNAPSHOT)
    monkeypatch.setattr(
        hparam_runtime,
        "_run_execution_command",
        lambda *_args, **_kwargs: pytest.fail("legacy plan must fail before target probing"),
    )
    monkeypatch.setattr(hparam_runtime, "_start_process", lambda *_args: pytest.fail("legacy plan must not start"))

    with pytest.raises(ValueError, match="lacks execution.python or execution.runtime_commit; create a new plan"):
        hparam_runtime.launch_hparam_runs(plan_dir, dry_run=False)

    assert (tmp_path / "run_manifest.tsv").read_bytes() == manifest_before
    assert (plan_dir / hparam_runtime.EXECUTION_SNAPSHOT_NAME).exists()
    assert not (plan_dir / "launch_manifest.tsv").exists()
    assert not (plan_dir / "run_status.tsv").exists()


def test_execution_probe_uses_target_cwd_and_isolated_pythonpath(monkeypatch):
    calls = []

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(hparam_runtime.subprocess, "run", run)
    workdir = "/runtime checkout"

    hparam_runtime._run_execution_command(
        {"workdir": workdir, "conda_env": "runtime", "env": {"TOKEN": "value"}},
        ["/runtime/bin/python", "-c", "pass"],
    )

    argv, kwargs = calls[0]
    assert argv[:2] == ["bash", "-lc"]
    shell = argv[2]
    assert shell.index(f"cd {shlex.quote(workdir)}") < shell.index("conda run")
    assert "export PYTHONPATH=" in shell
    assert "${PYTHONPATH" not in shell
    assert "TOKEN=value" in shell
    assert kwargs["timeout"] == hparam_runtime.LAUNCH_TIMEOUT_SECONDS


def test_verified_launch_rechecks_snapshot_and_artifacts_immediately_before_process_start(tmp_path: Path):
    script = tmp_path / "launch.sh"
    config = tmp_path / "config.yaml"
    script.write_text("#!/usr/bin/env bash\ntrue\n")
    config.write_text("value: 1\n")
    snapshot = {
        "python": "/runtime/bin/python",
        "python_version": "3.10.0",
        "runtime_commit": "a" * 40,
        "runtime_repo_root": "/runtime/repo",
        "runtime_hostname": "runtime-host",
        "module": "sleep2vec.finetune",
        "module_origin": "/runtime/repo/sleep2vec/finetune.py",
    }
    command = hparam_runtime._launch_command(
        {"workdir": str(tmp_path), "python": "/runtime/bin/python", "runtime_commit": "a" * 40},
        script,
        tmp_path / "stdout.log",
        tmp_path / "pid",
        [],
        execution_snapshot=snapshot,
        config_path=config,
        script_sha256=file_sha256(script),
        config_sha256=file_sha256(config),
    )

    assert "Target runtime identity changed before process start" in command
    assert "Frozen run artifact changed before process start" in command
    assert str(script) in command
    assert str(config) in command
    assert command.index("Target runtime identity changed before process start") < command.index(
        "start_new_session=True"
    )


def test_launch_creates_and_stops_a_dedicated_process_group(tmp_path: Path):
    script = tmp_path / "launch.sh"
    script.write_text("#!/usr/bin/env bash\nsleep 30\n")
    pid_path = tmp_path / "pid"
    command = hparam_runtime._launch_command(
        {"workdir": str(tmp_path), "python": sys.executable},
        script,
        tmp_path / "stdout.log",
        pid_path,
        [],
    )

    result = subprocess.run(["bash", "-lc", command], text=True, capture_output=True, timeout=10)
    assert result.returncode == 0
    identity = run_evidence.read_process_identity(pid_path, {})
    assert identity is not None
    assert identity["pid"] == identity["process_group_id"]
    assert run_evidence.process_identity_running({}, identity) is True

    run_evidence.stop_process_group({}, identity)

    assert run_evidence.process_identity_running({}, identity) is False


@pytest.mark.parametrize(
    ("exit_code", "expected_status"),
    [
        pytest.param(0, "finished", id="zero"),
        pytest.param(7, "failed", id="nonzero"),
    ],
)
def test_monitor_owned_launch_uses_shell_exit_code(tmp_path: Path, exit_code: int, expected_status: str):
    script = tmp_path / "launch.sh"
    script.write_text(
        "\n".join(
            plan_rendering.hparam_script_lines(
                ["sleep 0.2", f"exit {exit_code}"],
                record_exit_code=True,
                run_cwd=tmp_path,
            )
        )
        + "\n"
    )
    log_path = tmp_path / "stdout.log"
    pid_path = tmp_path / "pid"
    command = hparam_runtime._launch_command(
        {"workdir": str(tmp_path), "python": sys.executable},
        script,
        log_path,
        pid_path,
        [],
    )

    result = subprocess.run(["bash", "-lc", command], text=True, capture_output=True, timeout=10)
    assert result.returncode == 0, result.stderr
    identity = run_evidence.read_process_identity(pid_path, {})
    assert identity is not None
    deadline = time.monotonic() + 10
    while run_evidence.process_identity_running({}, identity) is not False:
        assert time.monotonic() < deadline
        time.sleep(0.05)

    row = {
        "script": str(script),
        "pid_path": str(pid_path),
        "log_path": str(log_path),
        "status": "running",
        "terminal_status_owner": "monitor",
        **identity,
    }
    observed = run_evidence.status_row(tmp_path, row, row, script_commits_terminal_status=False)

    assert log_path.read_text().splitlines()[-1] == f"{MONITOR_EXIT_CODE_PREFIX}{exit_code}"
    assert observed["status"] == expected_status


def test_launch_timeout_remains_nonterminal_until_process_evidence_reconciles(monkeypatch):
    monkeypatch.setattr(
        hparam_runtime.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(subprocess.TimeoutExpired("launch", 60)),
    )

    assert hparam_runtime._start_process({}, "managed launch") == "launched"


@pytest.mark.parametrize(
    ("execution", "expected_status"),
    [
        pytest.param({"target": "ssh", "host": "unit-host"}, "launched", id="ssh"),
        pytest.param({"target": "local"}, "launch_failed", id="local"),
    ],
)
def test_launch_returncode_255_is_uncertain_only_over_ssh(monkeypatch, execution: dict, expected_status: str):
    monkeypatch.setattr(
        hparam_runtime.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 255, "", "connection lost"),
    )

    assert hparam_runtime._start_process(execution, "managed launch") == expected_status


def test_unresolved_launch_timeout_is_not_relaunched(tmp_path: Path, monkeypatch):
    _write_runtime_rows(tmp_path, [{"run_id": "run-000", "status": "planned"}])
    starts = []
    monkeypatch.setattr(
        hparam_runtime,
        "_start_process",
        lambda *_args: starts.append("timeout") or "launched",
    )

    hparam_runtime.launch_hparam_runs(tmp_path, dry_run=False)
    hparam_runtime.launch_hparam_runs(tmp_path, dry_run=False)

    assert starts == ["timeout"]
    assert _read_table(tmp_path / "run_manifest.tsv")[0]["status"] == "missing_pid"


def test_execution_probe_allows_untracked_experiment_artifacts(tmp_path: Path):
    repo = tmp_path / "runtime-repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    (repo / "runtime_cli.py").write_text(
        "import argparse\n"
        "if __name__ == '__main__':\n"
        "    parser = argparse.ArgumentParser()\n"
        "    parser.add_argument('--value', choices=['ok'], required=True)\n"
        "    parser.parse_args()\n"
    )
    (repo / ".gitignore").write_text("*.log\n.codex-tmp/\n")
    subprocess.run(["git", "add", "runtime_cli.py", ".gitignore"], cwd=repo, check=True)
    subprocess.run(
        ["git", "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-qm", "fixture"],
        cwd=repo,
        check=True,
    )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, text=True, capture_output=True
    ).stdout.strip()
    artifact_dir = repo / "experiment" / "plan"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "plan.json").write_text("{}\n")
    config = artifact_dir / "config.yaml"
    config.write_text("value: ok\n")
    checkpoint = artifact_dir / "model.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    (repo / "dataset.csv").write_text("value\nok\n")
    ignored_tools = repo / ".codex-tmp"
    ignored_tools.mkdir()
    (ignored_tools / "tool.py").write_text("raise RuntimeError('not importable from the runtime root')\n")
    command = f"{shlex.quote(sys.executable)} -m runtime_cli --value ok"
    script = artifact_dir / "launch.sh"
    script.write_text(f"#!/usr/bin/env bash\n{command}\n")

    snapshot = hparam_runtime._inspect_execution_target(
        {"workdir": str(repo), "python": sys.executable, "runtime_commit": commit},
        [{"run_id": "run-000", "script": str(script), "command": command}],
    )

    assert snapshot["runtime_commit"] == commit
    assert snapshot["runtime_repo_root"] == str(repo)
    assert snapshot["module_origin"] == str(repo / "runtime_cli.py")
    assert "--value" in snapshot["supported_options"]
    launch = managed_scheduler.build_launch_command(
        {"workdir": str(repo), "python": sys.executable, "runtime_commit": commit},
        script,
        artifact_dir / "stdout.log",
        artifact_dir / "pid",
        [],
        execution_snapshot=snapshot,
        config_path=config,
        script_sha256=file_sha256(script),
        config_sha256=file_sha256(config),
        checkpoint_path=checkpoint,
        checkpoint_sha256=file_sha256(checkpoint),
    )
    config.write_text("value: changed\n")

    config_result = subprocess.run(["bash", "-lc", launch], text=True, capture_output=True)

    assert config_result.returncode != 0
    assert "Frozen run artifact changed before process start" in config_result.stderr
    assert not (artifact_dir / "pid").exists()
    config.write_text("value: ok\n")
    script.write_text(script.read_text() + "# changed\n")

    script_result = subprocess.run(["bash", "-lc", launch], text=True, capture_output=True)

    assert script_result.returncode != 0
    assert "Frozen run artifact changed before process start" in script_result.stderr
    assert not (artifact_dir / "pid").exists()
    script.write_text(f"#!/usr/bin/env bash\n{command}\n")
    (artifact_dir / "checkpoint-alias.ckpt").hardlink_to(checkpoint)

    checkpoint_result = subprocess.run(["bash", "-lc", launch], text=True, capture_output=True)

    assert checkpoint_result.returncode != 0
    assert f"Frozen run artifact is not an independent file: {checkpoint}" in checkpoint_result.stderr
    assert not (artifact_dir / "pid").exists()


def test_execution_probe_rejects_runtime_module_outside_verified_repository(tmp_path: Path):
    repo = tmp_path / "runtime-repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    (repo / "README.md").write_text("fixture\n")
    subprocess.run(["git", "add", "README.md"], cwd=repo, check=True)
    subprocess.run(
        ["git", "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-qm", "fixture"],
        cwd=repo,
        check=True,
    )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, text=True, capture_output=True
    ).stdout.strip()
    script = repo / "launch.sh"
    command = f"{shlex.quote(sys.executable)} -m pytest --help"
    script.write_text(f"#!/usr/bin/env bash\n{command}\n")

    with pytest.raises(RuntimeError, match="module is outside the verified repository"):
        hparam_runtime._inspect_execution_target(
            {"workdir": str(repo), "python": sys.executable, "runtime_commit": commit},
            [{"run_id": "run-000", "script": str(script), "command": command}],
        )


def test_hparam_launch_rejects_missing_target_cli_option_before_managed_writes(tmp_path: Path, monkeypatch):
    recipe = _hparam_recipe(
        tmp_path,
        execution={"workdir": str(tmp_path), "wandb_project": "unit", "wandb_group": "runtime-preflight"},
    )
    payload = yaml.safe_load(recipe.read_text())
    base_recipe = Path(payload["base_recipe"])
    base_payload = yaml.safe_load(base_recipe.read_text())
    base_payload["runtime"]["wandb_mode"] = "online"
    write_yaml(base_recipe, base_payload)
    plan_dir = tmp_path / "plan"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    _set_execution_probe(monkeypatch, plan_dir, missing_options={"--wandb-mode"})
    monkeypatch.setattr(hparam_runtime, "_validated_execution_snapshot", _REAL_VALIDATED_EXECUTION_SNAPSHOT)
    started = []
    monkeypatch.setattr(
        hparam_runtime,
        "_start_process",
        lambda _execution, command: started.append(command) or "launched",
    )
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}

    with pytest.raises(ValueError, match=r"does not accept planned options: --wandb-mode"):
        hparam_runtime.launch_hparam_runs(plan_dir, dry_run=False)

    assert started == []
    assert all(path.read_bytes() == content for path, content in before.items())
    assert (plan_dir / hparam_runtime.EXECUTION_SNAPSHOT_NAME).exists()
    assert not (plan_dir / "launch_manifest.tsv").exists()
    assert not (plan_dir / "run_status.tsv").exists()
    row = _read_table(tmp_path / "run_manifest.tsv")[0]
    assert row["status"] == "planned"
    assert row.get("target", "") == ""


def test_hparam_launch_rejects_frozen_cli_values_before_managed_writes(tmp_path: Path, monkeypatch):
    recipe = _hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    calls = _set_execution_probe(monkeypatch, plan_dir, parse_error="invalid choice: bf16-mixed")
    monkeypatch.setattr(hparam_runtime, "_validated_execution_snapshot", _REAL_VALIDATED_EXECUTION_SNAPSHOT)
    monkeypatch.setattr(hparam_runtime, "_start_process", lambda *_args: pytest.fail("must not start"))

    with pytest.raises(ValueError, match="rejected frozen arguments.*invalid choice"):
        hparam_runtime.launch_hparam_runs(plan_dir, dry_run=False)

    assert calls == ["identity", "parse"]
    assert (plan_dir / hparam_runtime.EXECUTION_SNAPSHOT_NAME).exists()
    assert _read_table(tmp_path / "run_manifest.tsv")[0]["status"] == "planned"


def test_hparam_launch_rejects_unintended_first_runtime_commit(tmp_path: Path, monkeypatch):
    recipe = _hparam_recipe(tmp_path, execution={"workdir": str(tmp_path), "runtime_commit": "a" * 40})
    plan_dir = tmp_path / "plan"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    calls = _set_execution_probe(monkeypatch, plan_dir, commit="b" * 40)
    monkeypatch.setattr(hparam_runtime, "_validated_execution_snapshot", _REAL_VALIDATED_EXECUTION_SNAPSHOT)
    monkeypatch.setattr(hparam_runtime, "_start_process", lambda *_args: pytest.fail("must not start"))

    with pytest.raises(ValueError, match="expected a{40}, observed b{40}"):
        hparam_runtime.launch_hparam_runs(plan_dir, dry_run=False)

    assert calls == ["identity"]
    assert (plan_dir / hparam_runtime.EXECUTION_SNAPSHOT_NAME).exists()


def test_hparam_launch_rejects_execution_snapshot_drift_before_next_wave(tmp_path: Path, monkeypatch):
    recipe = _hparam_recipe(tmp_path, execution={"workdir": str(tmp_path), "max_concurrent": 1})
    payload = yaml.safe_load(recipe.read_text())
    payload["search"]["max_runs"] = 2
    payload["search"]["parameters"]["runtime.lr"] = [1e-6, 2e-6]
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
    plan_dir = tmp_path / "plan"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    monkeypatch.setattr(hparam_runtime, "_validated_execution_snapshot", _REAL_VALIDATED_EXECUTION_SNAPSHOT)
    _set_execution_probe(monkeypatch, plan_dir)
    started = []
    monkeypatch.setattr(
        hparam_runtime,
        "_start_process",
        lambda _execution, command: started.append(command) or "launched",
    )
    hparam_runtime.launch_hparam_runs(plan_dir, dry_run=False)
    first = _read_table(plan_dir / "launch_manifest.tsv")[0]
    merge_run_manifest(
        tmp_path,
        [{"step_id": first["step_id"], "run_id": first["run_id"], "status": "finished"}],
    )
    _set_execution_probe(monkeypatch, plan_dir, commit="b" * 40)
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}

    with pytest.raises(ValueError, match="Target runtime commit differs"):
        hparam_runtime.launch_hparam_runs(plan_dir, dry_run=False)

    assert len(started) == 1
    assert all(path.read_bytes() == content for path, content in before.items())


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
        if _is_remote_python_program(command, "run_evidence.runtime_artifacts"):
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
    assert "start_new_session=True" in rows[0]["command"]
    assert "CUDA_VISIBLE_DEVICES=2" in rows[0]["command"]
    assert started == [rows[0]["command"]]


def test_hparam_launch_resolves_relative_plan_dir_before_cd(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path)
    plan_dir = tmp_path / "relative_plan"
    runner = Path(__file__).with_name("agent_tools_cli_stub.py")

    plan = subprocess.run(
        [
            sys.executable,
            str(runner),
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
            str(runner),
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
    payload["execution"].update({"workdir": str(tmp_path), "max_concurrent": 1})
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


def test_hparam_launch_fail_flag_reports_owned_missing_pid_before_start(tmp_path: Path, monkeypatch):
    recipe = _hparam_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    payload["search"]["max_runs"] = 2
    payload["search"]["parameters"]["runtime.lr"] = [1e-6, 2e-6]
    payload["execution"].update({"workdir": str(tmp_path), "max_concurrent": 1})
    recipe.write_text(yaml.safe_dump(payload))
    plan_dir = tmp_path / "plan"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    runs = json.loads((plan_dir / "plan.json").read_text())["runs"]
    started = []
    monkeypatch.setattr(
        hparam_runtime,
        "_start_process",
        lambda _execution, command: started.append(command) or "launched",
    )

    hparam_runtime.launch_hparam_runs(plan_dir, dry_run=False)
    assert len(started) == 1
    started.clear()

    with pytest.raises(managed_scheduler.MissingPidCapacityError) as exc_info:
        hparam_runtime.launch_hparam_runs(
            plan_dir,
            dry_run=False,
            fail_on_missing_pid_blocker=True,
        )

    assert exc_info.value.step_id == runs[0]["step_id"]
    assert exc_info.value.run_id == runs[0]["run_id"]
    assert started == []
    expected = {runs[0]["run_id"]: "missing_pid", runs[1]["run_id"]: "pending"}
    assert {row["run_id"]: row["status"] for row in _read_table(plan_dir / "launch_manifest.tsv")} == expected
    assert {row["run_id"]: row["status"] for row in _read_table(tmp_path / "run_manifest.tsv")} == expected


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
    monkeypatch.setattr(run_evidence.os, "kill", lambda pid, sig: killed.append((pid, sig)))

    with pytest.raises(ValueError, match="Legacy hparam plan"):
        hparam_runtime.launch_hparam_runs(tmp_path, dry_run=False)
    with pytest.raises(ValueError, match="Legacy hparam plan"):
        hparam_runtime.monitor_hparam_runs(tmp_path)
    with pytest.raises(ValueError, match="Legacy hparam plan"):
        hparam_runtime.stop_hparam_run(tmp_path, "trial_000", reason="legacy")

    assert started == []
    assert killed == []
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == before
