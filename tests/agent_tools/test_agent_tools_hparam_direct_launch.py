from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import threading
import time

from agent_tool_test_helpers import SUBPROCESS_WAIT_SECONDS, FakeLauncher, frozen_artifact_launch, run_cli
import pytest
from test_agent_tools_hparam_runtime import (
    _hparam_recipe,
    _process_identity,
    _read_table,
    _set_execution_probe,
    _write_process_identity,
    _write_runtime_rows,
    write_yaml,
)
from test_agent_tools_hparam_runtime import _stub_execution_snapshot_preflight  # noqa: F401
import yaml

from agent_tools import (
    execution_snapshot,
    experiment_workspace,
    experiments,
    hparam_runtime,
    managed_scheduler,
    manifests,
    plan_rendering,
    python_programs,
    run_evidence,
)
from agent_tools.experiment_workspace import MONITOR_EXIT_CODE_PREFIX, file_sha256, merge_run_manifest, merge_run_row


def test_hparam_launch_rejects_unregistered_plan_copy_before_start(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    assert run_cli("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    copied_plan = tmp_path / "copied-plan"
    shutil.copytree(plan_dir, copied_plan)
    launcher = FakeLauncher()

    # The copy names the same workspace, but its step registers only the original plan directory.
    with pytest.raises(ValueError, match="not registered by its managed step"):
        hparam_runtime.launch_hparam_runs(copied_plan, dry_run=False, hooks=launcher.hooks())

    launcher.assert_not_started()


def test_hparam_launch_rejects_completed_experiment_without_writes(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    assert run_cli("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    experiment_path = tmp_path / "experiment.yaml"
    experiment_manifest = yaml.safe_load(experiment_path.read_text())
    experiment_manifest["experiment"]["status"] = "completed"
    experiment_path.write_text(yaml.safe_dump(experiment_manifest, sort_keys=False))
    run_rows = _read_table(tmp_path / "run_manifest.tsv")
    run_rows[0]["status"] = "completed"
    manifests.write_rows(tmp_path / "run_manifest.tsv", run_rows)
    before = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    launcher = FakeLauncher()

    with pytest.raises(ValueError, match="Experiment is completed"):
        hparam_runtime.launch_hparam_runs(plan_dir, dry_run=False, hooks=launcher.hooks())

    after = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    launcher.assert_not_started()
    assert after == before


def test_hparam_launch_does_not_restart_workspace_terminal_run(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    assert run_cli("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    hparam_runtime.launch_hparam_runs(plan_dir, dry_run=True)
    workspace_rows = _read_table(tmp_path / "run_manifest.tsv")
    workspace_rows[0]["status"] = "failed"
    manifests.write_rows(tmp_path / "run_manifest.tsv", workspace_rows)
    launcher = FakeLauncher()

    hparam_runtime.launch_hparam_runs(plan_dir, dry_run=False, hooks=launcher.hooks())

    launcher.assert_not_started()
    assert _read_table(tmp_path / "run_manifest.tsv")[0]["status"] == "failed"


@pytest.mark.parametrize("operation", ["launch", "monitor", "stop"])
def test_hparam_runtime_preserves_canonical_result_fields(tmp_path: Path, monkeypatch, operation: str):
    rows = _write_runtime_rows(
        tmp_path,
        [{"run_id": "run-000", "status": "running", **_process_identity()}],
    )
    _write_process_identity(rows[0]["pid_path"])
    canonical_rows = _read_table(tmp_path / "run_manifest.tsv")
    canonical_rows[0].update({"score": "0.9", "wandb_url": "https://wandb.example/current"})
    manifests.write_rows(tmp_path / "run_manifest.tsv", canonical_rows)
    monkeypatch.setattr(
        run_evidence,
        "status_row",
        lambda _root, row, previous, *, script_commits_terminal_status, health=False: merge_run_row(previous, row),
    )
    monkeypatch.setattr(run_evidence, "stop_process_group", lambda *_args: None)
    launcher = FakeLauncher(verify_target=False)
    # Monitor and stop take no hooks, so the public owner guards them against starting a process.
    monkeypatch.setattr(managed_scheduler, "start_process", launcher.start_process)

    if operation == "launch":
        hparam_runtime.launch_hparam_runs(tmp_path, dry_run=False, hooks=launcher.hooks())
    elif operation == "monitor":
        experiments.monitor_experiment(tmp_path)
    else:
        hparam_runtime.stop_hparam_run(tmp_path, "run-000", reason="manual stop")

    canonical = _read_table(tmp_path / "run_manifest.tsv")[0]
    launcher.assert_not_started()
    assert canonical["score"] == "0.9"
    assert canonical["wandb_url"] == "https://wandb.example/current"


def test_hparam_launch_records_event_only_for_a_process_started_by_that_call(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    assert run_cli("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    launcher = FakeLauncher()

    hparam_runtime.launch_hparam_runs(plan_dir, dry_run=True, hooks=launcher.hooks())
    hparam_runtime.launch_hparam_runs(plan_dir, dry_run=False, hooks=launcher.hooks())
    hparam_runtime.launch_hparam_runs(plan_dir, dry_run=False, hooks=launcher.hooks())

    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert launcher.starts and len(launcher.starts) == 1
    assert [event["event_type"] for event in events].count("run_launched") == 1


def test_hparam_launch_serializes_concurrent_execute_calls(tmp_path: Path, monkeypatch):
    recipe = _hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    assert run_cli("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    entered = threading.Event()
    release = threading.Event()
    second_parked = threading.Event()
    first_done = threading.Event()
    failures = []
    real_lstat = os.lstat
    real_run_lock = experiment_workspace.managed_run_lock

    def start(_execution, _command):
        entered.set()
        assert release.wait(timeout=5)
        return "launched"

    launcher = FakeLauncher(start)

    def lstat(path, *args, **kwargs):
        info = real_lstat(path, *args, **kwargs)
        if threading.current_thread() is second and os.fspath(path).endswith("run_manifest.tsv"):
            # Keep a managed read between its stat and open until the first launch commits its manifest replacement.
            second_parked.set()
            assert first_done.wait(timeout=5)
        return info

    @contextlib.contextmanager
    def run_lock(workspace):
        if threading.current_thread() is second:
            second_parked.set()
        with real_run_lock(workspace):
            yield

    def launch(done=None):
        try:
            hparam_runtime.launch_hparam_runs(plan_dir, dry_run=False, hooks=launcher.hooks())
        except Exception as exc:
            failures.append(exc)
        finally:
            if done is not None:
                done.set()

    first = threading.Thread(target=launch, args=(first_done,))
    second = threading.Thread(target=launch)
    monkeypatch.setattr(os, "lstat", lstat)
    monkeypatch.setattr(experiment_workspace, "managed_run_lock", run_lock)
    first.start()
    assert entered.wait(timeout=5)
    second.start()
    assert second_parked.wait(timeout=5)
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
    assert len(launcher.starts) == 1
    release.set()
    first.join(timeout=5)
    second.join(timeout=5)

    assert not first.is_alive()
    assert not second.is_alive()
    assert failures == []
    assert len(launcher.starts) == 1


def test_hparam_launch_commits_execution_identity_before_start(tmp_path: Path):
    _write_runtime_rows(tmp_path, [{"run_id": "run-000", "status": "planned"}])

    def start_after_identity_commit(_execution, command):
        canonical = _read_table(tmp_path / "run_manifest.tsv")[0]
        assert canonical["status"] == "planned"
        assert canonical["target"] == "local"
        assert canonical["command"] == command
        assert canonical["pid_path"]
        return "launched"

    launcher = FakeLauncher(start_after_identity_commit, verify_target=False)

    hparam_runtime.launch_hparam_runs(tmp_path, dry_run=False, hooks=launcher.hooks())

    assert len(launcher.starts) == 1
    assert _read_table(tmp_path / "run_manifest.tsv")[0]["status"] == "launched"


def test_hparam_launch_does_not_start_when_identity_precommit_fails(tmp_path: Path, monkeypatch):
    _write_runtime_rows(tmp_path, [{"run_id": "run-000", "status": "planned"}])
    launcher = FakeLauncher(verify_target=False)
    monkeypatch.setattr(
        hparam_runtime,
        "merge_run_manifest",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("identity precommit failed")),
    )

    with pytest.raises(RuntimeError, match="identity precommit failed"):
        hparam_runtime.launch_hparam_runs(tmp_path, dry_run=False, hooks=launcher.hooks())

    launcher.assert_not_started()


def test_hparam_launch_preserves_first_commit_when_second_start_raises(tmp_path: Path):
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
    launcher = FakeLauncher("launched", RuntimeError("second start failed"), verify_target=False)

    with pytest.raises(RuntimeError, match="second start failed"):
        hparam_runtime.launch_hparam_runs(tmp_path, dry_run=False, hooks=launcher.hooks())

    rows = {row["run_id"]: row for row in _read_table(tmp_path / "run_manifest.tsv")}
    assert rows["run-000"]["status"] == "launched"
    assert rows["run-001"]["status"] == "planned"
    assert rows["run-001"]["target"] == "local"


def test_hparam_launch_does_not_start_after_canonical_owner_commits_terminal_status(tmp_path: Path, monkeypatch):
    _write_runtime_rows(tmp_path, [{"run_id": "run-000", "status": "planned"}])
    real_merge = merge_run_manifest
    launcher = FakeLauncher(verify_target=False)

    def merge_after_wandb_update(root, rows, **_kwargs):
        kwargs = {"lock_held": True} if _kwargs.get("lock_held") else {}
        real_merge(root, [{"step_id": "train-model", "run_id": "run-000", "status": "failed"}], **kwargs)
        return real_merge(root, rows, **kwargs)

    monkeypatch.setattr(hparam_runtime, "merge_run_manifest", merge_after_wandb_update)

    hparam_runtime.launch_hparam_runs(tmp_path, dry_run=False, hooks=launcher.hooks())

    assert _read_table(tmp_path / "run_manifest.tsv")[0]["status"] == "failed"
    launcher.assert_not_started()
    assert not (tmp_path / "events.jsonl").exists()


def test_hparam_launch_failure_does_not_record_launched_event(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    assert run_cli("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    launcher = FakeLauncher("launch_failed")

    hparam_runtime.launch_hparam_runs(plan_dir, dry_run=False, hooks=launcher.hooks())

    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert "run_launched" not in [event["event_type"] for event in events]


def test_hparam_launch_rejects_workspace_frozen_drift_before_start(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    assert run_cli("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    workspace_rows = _read_table(tmp_path / "run_manifest.tsv")
    workspace_rows[0]["config_sha256"] = "changed"
    manifests.write_rows(tmp_path / "run_manifest.tsv", workspace_rows)
    launcher = FakeLauncher()

    with pytest.raises(ValueError, match="config_sha256"):
        hparam_runtime.launch_hparam_runs(plan_dir, dry_run=False, hooks=launcher.hooks())

    launcher.assert_not_started()


def test_hparam_launch_rejects_replaced_execution_snapshot_before_start(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    assert run_cli("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    snapshot_path = plan_dir / "execution_snapshot.json"
    snapshot = json.loads(snapshot_path.read_text())
    snapshot["runtime_hostname"] = "other-host"
    snapshot_path.write_text(json.dumps(snapshot))
    launcher = FakeLauncher()

    with pytest.raises(ValueError, match="execution snapshot changed after registration"):
        hparam_runtime.launch_hparam_runs(plan_dir, dry_run=False, hooks=launcher.hooks())

    launcher.assert_not_started()


@pytest.mark.parametrize(
    ("field", "changed"),
    [
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
    assert run_cli("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    plan_path = plan_dir / "plan.json"
    plan = json.loads(plan_path.read_text())
    plan["recipe"].setdefault("execution", {})[field] = changed
    plan_path.write_text(json.dumps(plan))
    before = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    calls = []
    launcher = FakeLauncher(lambda *_args: calls.append("start") or "launched")

    with pytest.raises(ValueError, match="recipe.resolved.yaml"):
        hparam_runtime.launch_hparam_runs(plan_dir, dry_run=False, hooks=launcher.hooks())

    assert calls == []
    assert {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before


def test_hparam_launch_rejects_synchronized_recipe_drift_before_side_effects(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    assert run_cli("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
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
    launcher = FakeLauncher(lambda *_args: calls.append("start") or "launched")

    with pytest.raises(ValueError, match="Frozen hparam recipe SHA-256"):
        hparam_runtime.launch_hparam_runs(plan_dir, dry_run=False, hooks=launcher.hooks())

    assert calls == []
    assert {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before


def test_hparam_launch_rejects_base_runtime_drift_before_side_effects(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    assert run_cli("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    plan_path = plan_dir / "plan.json"
    plan = json.loads(plan_path.read_text())
    plan["recipe"]["_base_recipe"]["runtime"]["devices"] = [7]
    plan_path.write_text(json.dumps(plan))
    before = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    calls = []
    launcher = FakeLauncher(lambda *_args: calls.append("start") or "launched")

    with pytest.raises(ValueError, match="recipe.resolved.yaml"):
        hparam_runtime.launch_hparam_runs(plan_dir, dry_run=False, hooks=launcher.hooks())

    assert calls == []
    assert {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before


def test_hparam_launch_binds_conda_gpu_and_pid_identity_only_after_a_launch_slot(
    tmp_path: Path,
    monkeypatch,
):
    recipe = _hparam_recipe(
        tmp_path,
        execution={
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

    assert run_cli("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    rows = hparam_runtime.run_hparam_queue(plan_dir)

    assert rows[0]["status"] == "planned"
    assert rows[0]["target"] == "local"
    assert rows[0]["gpus"] == "6,7"
    assert rows[0]["command"].startswith(f"cd {shlex.quote(str(plan_dir))} && ")
    assert "CUDA_VISIBLE_DEVICES=6,7" in rows[0]["command"]
    canonical = _read_table(tmp_path / "run_manifest.tsv")
    assert canonical[0]["target"] == ""
    assert canonical[0]["gpus"] == ""
    assert canonical[0]["command"] == ""
    script = Path(rows[0]["script"]).read_text()
    assert "--wandb-project sleep2vec-unit-hparam" in script
    assert "--wandb-group unit" in script
    assert not (plan_dir / "logs").exists()
    assert not (plan_dir / "pids").exists()
    launcher = FakeLauncher()
    # The fake launcher creates no PID file for the follow-up identity lookup.
    monkeypatch.setattr(run_evidence, "read_process_identity", lambda *_args: None)

    hparam_runtime.launch_hparam_runs(plan_dir, dry_run=False, hooks=launcher.hooks())

    rows = _read_table(tmp_path / "run_manifest.tsv")
    assert rows[0]["status"] == "launched"
    assert rows[0]["target"] == "local"
    assert rows[0]["gpus"] == "6,7"
    assert rows[0]["command"].startswith(f"cd {shlex.quote(str(plan_dir))} && ")
    assert "start_new_session=True" in rows[0]["command"]
    assert "conda run --no-capture-output -n ywx" in rows[0]["command"]
    assert "CUDA_VISIBLE_DEVICES=6,7" in rows[0]["command"]
    assert "WANDB_PROJECT=" not in rows[0]["command"]
    assert "WANDB_GROUP=" not in rows[0]["command"]
    assert "WANDB_RUN_GROUP=" not in rows[0]["command"]
    assert rows[0]["log_path"].endswith("runs/run-000--lr-1e-6/stdout.log")
    assert rows[0]["pid_path"].endswith("runs/run-000--lr-1e-6/pid")
    assert launcher.commands == [rows[0]["command"]]


def test_hparam_run_queue_fails_on_missing_pid_capacity_blocker_from_another_plan(tmp_path: Path, monkeypatch):
    execution = {"workdir": str(tmp_path), "gpu_pool": [0], "gpus_per_run": 1}
    first_recipe = _hparam_recipe(tmp_path, execution=execution)
    first_plan = tmp_path / "plan-1"
    assert run_cli("plan", "--recipe", str(first_recipe), "--output-dir", str(first_plan)).returncode == 0
    hparam_runtime.launch_hparam_runs(first_plan, dry_run=False, hooks=FakeLauncher().hooks())
    first_run = json.loads((first_plan / "plan.json").read_text())["runs"][0]
    merge_run_manifest(
        tmp_path,
        [{"step_id": first_run["step_id"], "run_id": first_run["run_id"], "status": "missing_pid"}],
    )

    second_payload = yaml.safe_load(first_recipe.read_text())
    second_payload["search"]["parameters"]["runtime.lr"] = [2e-6]
    second_recipe = write_yaml(tmp_path / "tune-2.yaml", second_payload)
    second_plan = tmp_path / "plan-2"
    assert run_cli("plan", "--recipe", str(second_recipe), "--output-dir", str(second_plan)).returncode == 0
    monkeypatch.setattr(
        hparam_runtime.evidence,
        "status_row",
        lambda _root, observation, previous, *, script_commits_terminal_status, health: {
            **previous,
            **observation,
        },
    )
    monkeypatch.setattr(managed_scheduler, "start_process", lambda *_args: pytest.fail("blocked queue must not start"))
    monkeypatch.setattr(hparam_runtime.time, "sleep", lambda *_args: pytest.fail("blocked queue must not sleep"))

    with pytest.raises(
        managed_scheduler.MissingPidCapacityError,
        match=f"{first_run['run_id']} has status missing_pid",
    ) as exc_info:
        hparam_runtime.run_hparam_queue(second_plan, dry_run=False)

    assert exc_info.value.step_id == first_run["step_id"]
    assert exc_info.value.run_id == first_run["run_id"]


def _launch_verification(command: str) -> list[list[str]]:
    """Return the verification programs a managed launch command runs before it starts the process."""
    return json.loads(shlex.split(command)[-1])


def test_hparam_launch_verifies_the_frozen_execution_snapshot_inside_the_launch_command(tmp_path: Path, monkeypatch):
    recipe = _hparam_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    payload["search"]["max_runs"] = 2
    payload["search"]["parameters"]["runtime.lr"] = [1e-6, 2e-6]
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
    plan_dir = tmp_path / "plan"
    assert run_cli("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    calls = _set_execution_probe(monkeypatch, plan_dir)
    snapshot_bytes = (plan_dir / execution_snapshot.EXECUTION_SNAPSHOT_NAME).read_bytes()
    launcher = FakeLauncher(lambda _execution, _command: calls.append("start") or "launched")

    hparam_runtime.launch_hparam_runs(plan_dir, dry_run=False, hooks=launcher.hooks())

    # The manager does not probe the target; the launch command checks the frozen snapshot at process start.
    assert calls == ["start"]
    assert (plan_dir / execution_snapshot.EXECUTION_SNAPSHOT_NAME).read_bytes() == snapshot_bytes
    snapshot = json.loads(snapshot_bytes)
    identity, cli_preflight = _launch_verification(launcher.commands[0])
    assert identity[2] == python_programs.source("managed_scheduler.runtime_identity")
    assert json.loads(identity[4]) == snapshot
    assert cli_preflight[2] == python_programs.source("managed_scheduler.cli_preflight")
    assert cli_preflight[3] == snapshot["module"]
    assert snapshot["python"] == json.loads((plan_dir / "plan.json").read_text())["recipe"]["execution"]["python"]
    assert (
        snapshot["runtime_commit"]
        == json.loads((plan_dir / "plan.json").read_text())["recipe"]["execution"]["runtime_commit"]
    )
    assert snapshot["runtime_hostname"] == "test-runtime"
    assert snapshot["module"] == "sleep2vec.finetune"
    assert set(snapshot["required_options"]).issubset(snapshot["supported_options"])
    assert not list(plan_dir.glob(f".{execution_snapshot.EXECUTION_SNAPSHOT_NAME}.*"))


def test_execution_probe_uses_target_cwd_and_isolated_pythonpath(monkeypatch):
    calls = []

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(subprocess, "run", run)
    workdir = "/runtime checkout"

    execution_snapshot.run_execution_command(
        {"workdir": workdir, "conda_env": "runtime", "env": {"TOKEN": "value"}},
        ["/runtime/bin/python", "-c", "pass"],
    )

    argv, kwargs = calls[0]
    assert argv[:2] == ["bash", "-lc"]
    shell = argv[-1]
    assert shell.index(f"cd {shlex.quote(workdir)}") < shell.index("conda run")
    assert "export PYTHONPATH=" in shell
    assert "${PYTHONPATH" not in shell
    assert "TOKEN=value" in shell
    assert kwargs["timeout"] == 300


def test_execution_probe_timeout_propagates_without_retry(monkeypatch):
    calls = []

    def timeout(argv, **kwargs):
        calls.append(kwargs["timeout"])
        raise subprocess.TimeoutExpired(argv, kwargs["timeout"])

    monkeypatch.setattr(subprocess, "run", timeout)

    with pytest.raises(subprocess.TimeoutExpired):
        execution_snapshot.run_execution_command(
            {"workdir": "/runtime"},
            ["/runtime/bin/python", "-c", "pass"],
        )

    assert calls == [300]


@pytest.mark.parametrize(
    ("failing_program", "message"),
    [
        ("managed_scheduler.runtime_identity", "Frozen run artifact changed before process start: config.yaml"),
        ("managed_scheduler.cli_preflight", "Frozen argv rejected in launch lock"),
    ],
)
def test_verified_launch_failure_inside_runtime_lock_is_launch_failed_with_reason(
    tmp_path: Path, monkeypatch, failing_program: str, message: str
):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("runtime\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", tracked.name], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "runtime",
        ],
        check=True,
    )
    commit = subprocess.run(
        ["git", "-C", str(tmp_path), "rev-parse", "HEAD"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    marker = tmp_path / "child-started"
    script = tmp_path / "launch.sh"
    config = tmp_path / "config.yaml"
    pid_path = tmp_path / "pid"
    script.write_text(f"#!/usr/bin/env bash\ntouch {shlex.quote(str(marker))}\n")
    config.write_text("value: 1\n")
    real_source = python_programs.source

    def embedded_source(name: str) -> str:
        if name == failing_program:
            return f"import sys; print({message!r}, file=sys.stderr); raise SystemExit(2)\n"
        if name in {"managed_scheduler.runtime_identity", "managed_scheduler.cli_preflight"}:
            return "pass\n"
        return real_source(name)

    monkeypatch.setattr(python_programs, "source", embedded_source)
    command = managed_scheduler.build_launch_command(
        {"workdir": str(tmp_path), "python": sys.executable, "runtime_commit": commit},
        script,
        tmp_path / "stdout.log",
        pid_path,
        [],
        execution_snapshot={"module": "runtime_cli", "module_origin": str(tmp_path / "runtime_cli.py")},
        config_path=config,
        script_sha256=file_sha256(script),
        config_sha256=file_sha256(config),
        planned_command=f"{shlex.quote(sys.executable)} -m runtime_cli --value old",
        run_id="run-000",
    )

    status, reason = managed_scheduler.start_process({"target": "local"}, command)

    # A verification failure is definite, never a pending retry; the verifier's own stderr is the reason.
    assert status == "launch_failed"
    assert reason.startswith("launch exit 2: ")
    assert reason.endswith(message)
    assert not pid_path.exists()
    assert not marker.exists()


def test_launch_creates_and_stops_a_dedicated_process_group(tmp_path: Path):
    script = tmp_path / "launch.sh"
    script.write_text("#!/usr/bin/env bash\nsleep 30\n")
    pid_path = tmp_path / "pid"
    command = managed_scheduler.build_launch_command(
        {"workdir": str(tmp_path), "python": sys.executable},
        script,
        tmp_path / "stdout.log",
        pid_path,
        [],
    )

    result = subprocess.run(["bash", "-lc", command], text=True, capture_output=True, timeout=SUBPROCESS_WAIT_SECONDS)
    assert result.returncode == 0
    identity = run_evidence.read_process_identity(pid_path, {})
    assert identity is not None
    assert identity["pid"] == identity["process_group_id"]
    assert run_evidence.process_identity_running({}, identity) is True

    run_evidence.stop_process_group({}, identity)

    assert run_evidence.process_identity_running({}, identity) is False


def test_detached_process_stdio_survives_transport_pipe_closure(tmp_path: Path):
    script = tmp_path / "launch.sh"
    log_path, pid_path = tmp_path / "stdout.log", tmp_path / "pid"
    worker = (
        "import os, time; "
        "assert os.read(0, 1) == b''; "
        "assert os.fstat(1).st_ino == os.fstat(2).st_ino; "
        "time.sleep(0.2); "
        "os.write(1, b'delayed stdout\\n'); "
        "os.write(2, b'delayed stderr\\n')"
    )
    script.write_text(f"#!/usr/bin/env bash\n{shlex.join([sys.executable, '-c', worker])}\n")
    command = managed_scheduler.build_launch_command(
        {"workdir": str(tmp_path), "python": sys.executable}, script, log_path, pid_path, []
    )
    launcher = subprocess.Popen(["bash", "-lc", command], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    launcher.stdout.close()
    launcher.stderr.close()
    identity = None
    try:
        assert launcher.wait(timeout=SUBPROCESS_WAIT_SECONDS) == 0
        identity = run_evidence.read_process_identity(pid_path, {})
        assert identity is not None
        assert identity["pid"] == identity["process_group_id"]
        deadline = time.monotonic() + SUBPROCESS_WAIT_SECONDS
        while run_evidence.process_identity_running({}, identity) is not False:
            assert time.monotonic() < deadline
            time.sleep(0.05)
        assert log_path.read_text() == "delayed stdout\ndelayed stderr\n"
    finally:
        if identity is not None and run_evidence.process_identity_running({}, identity) is True:
            run_evidence.stop_process_group({}, identity)
        if launcher.poll() is None:
            launcher.kill()
            launcher.wait(timeout=SUBPROCESS_WAIT_SECONDS)


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
    command = managed_scheduler.build_launch_command(
        {"workdir": str(tmp_path), "python": sys.executable},
        script,
        log_path,
        pid_path,
        [],
    )

    result = subprocess.run(["bash", "-lc", command], text=True, capture_output=True, timeout=SUBPROCESS_WAIT_SECONDS)
    assert result.returncode == 0, result.stderr
    identity = run_evidence.read_process_identity(pid_path, {})
    assert identity is not None
    deadline = time.monotonic() + SUBPROCESS_WAIT_SECONDS
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


def test_launch_waits_without_timeout_and_propagates_its_exception(monkeypatch):
    calls = []

    def timeout(*_args, **kwargs):
        calls.append(kwargs.get("timeout"))
        raise subprocess.TimeoutExpired("launch", 1)

    monkeypatch.setattr(subprocess, "run", timeout)

    with pytest.raises(subprocess.TimeoutExpired):
        managed_scheduler.start_process({"target": "local"}, "managed launch")
    assert calls == [None]


def test_launch_returncode_255_is_launch_failed(monkeypatch):
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 255, "", "launcher failed"),
    )

    assert managed_scheduler.start_process({"target": "local"}, "managed launch") == (
        "launch_failed",
        "launch exit 255: launcher failed",
    )


def test_former_retryable_exit_is_launch_failed_with_one_line_bounded_stderr_tail(monkeypatch):
    stderr = "Traceback (most recent call last):\n" + "x" * 2000 + "\n\nRuntimeError: runtime compatibility changed\n"
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 75, "", stderr),
    )

    status, reason = managed_scheduler.start_process({"target": "local"}, "managed launch")

    # Exit 75 no longer maps to a silent pending retry.
    assert status == "launch_failed"
    assert reason.startswith("launch exit 75: ")
    assert reason.endswith(" / RuntimeError: runtime compatibility changed")
    assert "\n" not in reason
    assert len(reason) == len("launch exit 75: ") + 1000


def test_pre_spawn_failure_records_launch_failed_reason_and_is_not_retried(tmp_path: Path):
    _write_runtime_rows(tmp_path, [{"run_id": "run-000", "status": "planned"}])
    reason = "launch exit 2: Frozen run artifact changed before process start: config.yaml"
    launcher = FakeLauncher(("launch_failed", reason), verify_target=False)

    hparam_runtime.launch_hparam_runs(tmp_path, dry_run=False, hooks=launcher.hooks())
    hparam_runtime.launch_hparam_runs(tmp_path, dry_run=False, hooks=launcher.hooks())

    assert len(launcher.starts) == 1
    row = _read_table(tmp_path / "run_manifest.tsv")[0]
    assert (row["status"], row["scheduler_reason"]) == ("launch_failed", reason)


def test_unconfirmed_launch_is_not_relaunched(tmp_path: Path):
    _write_runtime_rows(tmp_path, [{"run_id": "run-000", "status": "planned"}])
    launcher = FakeLauncher(verify_target=False)

    hparam_runtime.launch_hparam_runs(tmp_path, dry_run=False, hooks=launcher.hooks())
    hparam_runtime.launch_hparam_runs(tmp_path, dry_run=False, hooks=launcher.hooks())

    assert len(launcher.starts) == 1
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
    agent_tools = repo / "agent_tools"
    agent_tools.mkdir()
    (agent_tools / "__init__.py").write_text("")
    (agent_tools / "experiment_workspace.py").write_text("def commit_run_start(*args, **kwargs):\n    pass\n")
    (agent_tools / "runtime_lock.py").write_text("def runtime_lock(*args, **kwargs):\n    pass\n")
    (repo / ".gitignore").write_text("*.log\n.codex-tmp/\n")
    subprocess.run(["git", "add", "runtime_cli.py", "agent_tools", ".gitignore"], cwd=repo, check=True)
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

    snapshot = execution_snapshot.inspect_execution_target(
        {"workdir": str(repo), "python": sys.executable, "runtime_commit": commit},
        [{"run_id": "run-000", "script": str(script), "command": command}],
        plan_label="hparam",
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
        planned_command=command,
        run_id="run-000",
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
        execution_snapshot.inspect_execution_target(
            {"workdir": str(repo), "python": sys.executable, "runtime_commit": commit},
            [{"run_id": "run-000", "script": str(script), "command": command}],
            plan_label="hparam",
        )


def test_hparam_launch_accepts_runtime_commit_drift_and_records_both_commits(tmp_path: Path, monkeypatch):
    recipe = _hparam_recipe(tmp_path, execution={"workdir": str(tmp_path), "runtime_commit": "a" * 40})
    plan_dir = tmp_path / "plan"
    assert run_cli("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    run = json.loads((plan_dir / "plan.json").read_text())["runs"][0]
    snapshot_path = plan_dir / execution_snapshot.EXECUTION_SNAPSHOT_NAME
    snapshot_before = snapshot_path.read_bytes()
    calls = _set_execution_probe(monkeypatch, plan_dir, commit="b" * 40)

    def start(_execution, _command):
        Path(run["run_dir"], "pid").write_text(
            json.dumps(
                {
                    "pid": 123,
                    "process_group_id": 123,
                    "process_start_token": "proc:unit-start",
                    "runtime_commit": "b" * 40,
                }
            )
            + "\n"
        )
        return "launched"

    launcher = FakeLauncher(start)
    hparam_runtime.launch_hparam_runs(plan_dir, dry_run=False, hooks=launcher.hooks())

    assert calls == []
    assert len(launcher.starts) == 1
    assert snapshot_path.read_bytes() == snapshot_before
    row = _read_table(tmp_path / "run_manifest.tsv")[0]
    assert row["planned_runtime_commit"] == "a" * 40
    assert row["runtime_commit"] == "b" * 40


def test_hparam_launch_accepts_runtime_commit_drift_before_next_wave(tmp_path: Path, monkeypatch):
    recipe = _hparam_recipe(tmp_path, execution={"workdir": str(tmp_path), "max_concurrent": 1})
    payload = yaml.safe_load(recipe.read_text())
    payload["search"]["max_runs"] = 2
    payload["search"]["parameters"]["runtime.lr"] = [1e-6, 2e-6]
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
    plan_dir = tmp_path / "plan"
    assert run_cli("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    _set_execution_probe(monkeypatch, plan_dir)
    launcher = FakeLauncher()
    hparam_runtime.launch_hparam_runs(plan_dir, dry_run=False, hooks=launcher.hooks())
    first = _read_table(tmp_path / "run_manifest.tsv")[0]
    merge_run_manifest(
        tmp_path,
        [{"step_id": first["step_id"], "run_id": first["run_id"], "status": "finished"}],
    )
    _set_execution_probe(monkeypatch, plan_dir, commit="b" * 40)
    snapshot_path = plan_dir / execution_snapshot.EXECUTION_SNAPSHOT_NAME
    snapshot_before = snapshot_path.read_bytes()

    hparam_runtime.launch_hparam_runs(plan_dir, dry_run=False, hooks=launcher.hooks())

    assert len(launcher.starts) == 2
    assert snapshot_path.read_bytes() == snapshot_before


def test_repeated_dry_run_does_not_observe_runtime_before_execute(tmp_path: Path, monkeypatch):
    recipe = _hparam_recipe(tmp_path, execution={"workdir": str(tmp_path / "plan"), "max_concurrent": 1})
    plan_dir = tmp_path / "plan"
    assert run_cli("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    observations = []
    real_run_row_command = run_evidence.run_row_command
    real_runtime_artifacts = run_evidence.runtime_artifacts

    def record_command(command):
        observations.append(command)
        return real_run_row_command(command)

    def record_artifacts(row):
        observations.append("runtime_artifacts")
        return real_runtime_artifacts(row)

    monkeypatch.setattr(run_evidence, "run_row_command", record_command)
    monkeypatch.setattr(run_evidence, "runtime_artifacts", record_artifacts)

    hparam_runtime.launch_hparam_runs(plan_dir, dry_run=True)
    hparam_runtime.launch_hparam_runs(plan_dir, dry_run=True)

    assert observations == []
    assert _read_table(tmp_path / "run_manifest.tsv")[0]["status"] == "planned"
    launcher = FakeLauncher()

    hparam_runtime.launch_hparam_runs(plan_dir, dry_run=False, hooks=launcher.hooks())

    assert len(launcher.starts) == 1
    assert _read_table(tmp_path / "run_manifest.tsv")[0]["status"] == "launched"


def test_hparam_launch_rejects_an_existing_runtime_dir_before_start(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    assert run_cli("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    run = json.loads((plan_dir / "plan.json").read_text())["runs"][0]
    Path(run["runtime_dir"]).mkdir(parents=True)
    launcher = FakeLauncher()

    with pytest.raises(ValueError, match="Managed output"):
        hparam_runtime.launch_hparam_runs(plan_dir, dry_run=False, hooks=launcher.hooks())

    launcher.assert_not_started()
    assert _read_table(tmp_path / "run_manifest.tsv")[0]["status"] == "planned"


def test_hparam_launch_remaps_scalar_runtime_device_pool(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    base_recipe = Path(payload["base_recipe"])
    base_payload = yaml.safe_load(base_recipe.read_text())
    base_payload["runtime"]["devices"] = 2
    write_yaml(base_recipe, base_payload)
    plan_dir = tmp_path / "plan"

    assert run_cli("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    launcher = FakeLauncher()

    hparam_runtime.launch_hparam_runs(plan_dir, dry_run=False, hooks=launcher.hooks())

    rows = _read_table(tmp_path / "run_manifest.tsv")
    assert rows[0]["gpus"] == "2"
    assert "--devices 0 --precision" in Path(rows[0]["script"]).read_text()
    assert "start_new_session=True" in rows[0]["command"]
    assert "CUDA_VISIBLE_DEVICES=2" in rows[0]["command"]
    assert launcher.commands == [rows[0]["command"]]


def test_hparam_launch_resolves_relative_plan_dir_before_cd(tmp_path: Path, monkeypatch):
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
    assert plan.returncode == 0, plan.stderr
    monkeypatch.chdir(tmp_path)

    rows = hparam_runtime.run_hparam_queue("relative_plan")

    assert rows[0]["script"] == str(plan_dir / "runs" / "run-000--lr-1e-6" / "launch.sh")
    assert rows[0]["log_path"] == str(plan_dir / "runs" / "run-000--lr-1e-6" / "stdout.log")
    assert "relative_plan/relative_plan" not in rows[0]["command"]


def test_hparam_launch_does_not_retry_missing_pid(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    payload["search"]["max_runs"] = 2
    payload["search"]["parameters"]["runtime.lr"] = [1e-6, 2e-6]
    payload["execution"].update({"workdir": str(tmp_path), "max_concurrent": 1})
    recipe.write_text(yaml.safe_dump(payload))
    plan_dir = tmp_path / "plan"
    assert run_cli("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    launcher = FakeLauncher()

    hparam_runtime.launch_hparam_runs(plan_dir, dry_run=False, hooks=launcher.hooks())
    assert len(launcher.starts) == 1
    launcher.starts.clear()
    hparam_runtime.launch_hparam_runs(plan_dir, dry_run=False, hooks=launcher.hooks())

    launcher.assert_not_started()
    status = {row["run_id"]: row["status"] for row in _read_table(tmp_path / "run_manifest.tsv")}
    assert status == {"run-000": "missing_pid", "run-001": "pending"}


def test_hparam_launch_fail_flag_reports_owned_missing_pid_before_start(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    payload["search"]["max_runs"] = 2
    payload["search"]["parameters"]["runtime.lr"] = [1e-6, 2e-6]
    payload["execution"].update({"workdir": str(tmp_path), "max_concurrent": 1})
    recipe.write_text(yaml.safe_dump(payload))
    plan_dir = tmp_path / "plan"
    assert run_cli("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    runs = json.loads((plan_dir / "plan.json").read_text())["runs"]
    launcher = FakeLauncher()

    hparam_runtime.launch_hparam_runs(plan_dir, dry_run=False, hooks=launcher.hooks())
    assert len(launcher.starts) == 1
    launcher.starts.clear()

    with pytest.raises(managed_scheduler.MissingPidCapacityError) as exc_info:
        hparam_runtime.launch_hparam_runs(
            plan_dir,
            dry_run=False,
            fail_on_missing_pid_blocker=True,
            hooks=launcher.hooks(),
        )

    assert exc_info.value.step_id == runs[0]["step_id"]
    assert exc_info.value.run_id == runs[0]["run_id"]
    launcher.assert_not_started()
    expected = {runs[0]["run_id"]: "missing_pid", runs[1]["run_id"]: "pending"}
    assert {row["run_id"]: row["status"] for row in _read_table(tmp_path / "run_manifest.tsv")} == expected


def test_hparam_launch_command_verifies_each_run_frozen_config_and_script(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path, execution={"workdir": str(tmp_path), "gpu_pool": [0, 1], "gpus_per_run": 1})
    payload = yaml.safe_load(recipe.read_text())
    payload["search"]["max_runs"] = 2
    payload["search"]["parameters"]["runtime.lr"] = [1e-6, 2e-6]
    recipe.write_text(yaml.safe_dump(payload))
    plan_dir = tmp_path / "plan"
    assert run_cli("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    runs = json.loads((plan_dir / "plan.json").read_text())["runs"]
    Path(runs[1]["config"]).write_text("changed: true\n")
    launcher = FakeLauncher(frozen_artifact_launch)

    hparam_runtime.launch_hparam_runs(plan_dir, dry_run=False, hooks=launcher.hooks())
    hparam_runtime.launch_hparam_runs(plan_dir, dry_run=False, hooks=launcher.hooks())

    # Each run's own launch command re-hashes its frozen config and launch.sh right before process start, so the
    # changed config fails inside run-001's command as a definite launch_failed, not retried, while run-000 starts.
    assert len(launcher.commands) == 2
    rows = {row["run_id"]: row for row in _read_table(tmp_path / "run_manifest.tsv")}
    assert (rows[runs[1]["run_id"]]["status"], rows[runs[1]["run_id"]]["scheduler_reason"]) == (
        "launch_failed",
        f"launch exit 2: Frozen run artifact changed before process start: {runs[1]['config']}",
    )
    assert rows[runs[0]["run_id"]]["status"] != "launch_failed"
    for run, command in zip(runs, launcher.commands):
        identity = _launch_verification(command)[0]
        assert identity[2] == python_programs.source("managed_scheduler.runtime_identity")
        assert json.loads(identity[5]) == [
            {"path": run["script"], "sha256": run["script_sha256"]},
            {"path": run["config"], "sha256": run["config_sha256"]},
        ]
    assert file_sha256(runs[1]["config"]) != runs[1]["config_sha256"]


def test_execution_snapshot_preserves_dynamic_identity_and_drift(tmp_path: Path):
    command = "python -m runtime_cli"
    script = tmp_path / "launch.sh"
    script.write_text(command + "\n")
    execution = {"python": "python", "runtime_commit": "a" * 40}
    runs = [{"run_id": "run-000", "script": str(script), "command": command}]
    identity = {
        "python": "python",
        "python_version": 310,
        "runtime_commit": "a" * 40,
        "runtime_repo_root": "/runtime",
        "runtime_hostname": "host",
        "module": "external_module",
        "module_origin": "/runtime/runtime_cli.py",
        "target": {"external": True},
        "extra_evidence": [1, 2],
    }

    def run_command(_execution, argv):
        if argv[2] == python_programs.source("managed_scheduler.runtime_identity"):
            output = json.dumps(identity)
        else:
            output = "AGENT_CLI_PREFLIGHT=" + json.dumps(
                {
                    "supported_options": [2, 1],
                    "cli_options_sha256": "options-hash",
                }
            )
        return subprocess.CompletedProcess(argv, 0, stdout=output, stderr="")

    def inspect(execution, runs):
        return execution_snapshot.inspect_execution_target(execution, runs, command_runner=run_command)

    snapshot = inspect(execution, runs)
    assert snapshot["target"] == {"external": True}
    assert snapshot["python_version"] == 310
    assert snapshot["extra_evidence"] == [1, 2]
    assert snapshot["module"] == "runtime_cli"
    assert snapshot["supported_options"] == [1, 2]
    snapshot_path = tmp_path / execution_snapshot.EXECUTION_SNAPSHOT_NAME
    execution_snapshot.write_execution_snapshot_file(snapshot_path, snapshot)
    frozen_bytes = snapshot_path.read_bytes()
    assert json.loads(frozen_bytes) == snapshot
    assert execution_snapshot.validated_execution_snapshot(
        tmp_path,
        execution,
        runs,
        {},
        inspector=inspect,
    ) == (snapshot, False)
    # A Slurm submission may probe from another login node; the probing host is not runtime identity.
    identity["runtime_hostname"] = "other-login-node"
    resubmitted, written = execution_snapshot.validated_execution_snapshot(
        tmp_path, execution, runs, {}, inspector=inspect
    )
    assert (resubmitted["runtime_hostname"], written) == ("other-login-node", False)
    identity["extra_evidence"] = [1, 3]
    with pytest.raises(ValueError, match="Frozen execution snapshot changed: extra_evidence"):
        execution_snapshot.validated_execution_snapshot(tmp_path, execution, runs, {}, inspector=inspect)
    assert snapshot_path.read_bytes() == frozen_bytes
