from __future__ import annotations

from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from types import SimpleNamespace

from agent_tool_test_helpers import FakeLauncher, call_while_run_lock_holder_commits, run_cli
import pytest
from test_agent_tools_hparam_runtime import (
    _hparam_recipe,
    _process_identity,
    _read_table,
    _write_proc_stat,
    _write_process_identity,
    _write_runtime_rows,
    _write_slurm_plan,
)
from test_agent_tools_hparam_runtime import _stub_execution_snapshot_preflight  # noqa: F401
import yaml

from agent_tools import (
    cli,
    experiments,
    hparam_runtime,
    managed_scheduler,
    manifests,
    process_identity,
    run_artifacts,
    run_evidence,
    slurm,
)
from agent_tools.experiment_workspace import EXECUTION_IDENTITY_FIELDS, MONITOR_EXIT_CODE_PREFIX, merge_run_manifest


@pytest.mark.parametrize(("status", "dry_run"), [("planned", False), ("pending", True)])
def test_hparam_stop_cancels_unlaunched_run_without_runtime_probes(tmp_path: Path, monkeypatch, status, dry_run):
    recipe = _hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    result = run_cli("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir))
    assert result.returncode == 0, result.stderr
    plan = run_artifacts.read_hparam_plan(plan_dir)
    run = plan["runs"][0]
    if dry_run:
        assert hparam_runtime.launch_hparam_runs(plan_dir, dry_run=True).launch_rows[0]["target"] == "local"
    merge_run_manifest(tmp_path, [{"step_id": run["step_id"], "run_id": run["run_id"], "status": status}])
    frozen_paths = [plan_dir / "plan.json", plan_dir / "recipe.resolved.yaml", Path(run["script"]), Path(run["config"])]
    frozen = {path: path.read_bytes() for path in frozen_paths}
    before = _read_table(tmp_path / "run_manifest.tsv")[0]
    now = "2026-08-30T04:05:06Z"
    reason = "Cancel before launch: superseded experiment"
    monkeypatch.setattr(hparam_runtime, "utc_now", lambda: now)
    monkeypatch.setattr(run_evidence, "read_pid", lambda *_a, **_k: pytest.fail("no PID read"))
    monkeypatch.setattr(run_evidence, "read_process_identity", lambda *_a, **_k: pytest.fail("no identity read"))
    monkeypatch.setattr(run_evidence, "run_row_command", lambda *_a, **_k: pytest.fail("no process command"))
    monkeypatch.setattr(run_evidence, "stop_process_group", lambda *_a, **_k: pytest.fail("no signal"))
    monkeypatch.setattr(slurm, "user_jobs", lambda *_a, **_k: pytest.fail("no scheduler lookup"))
    monkeypatch.setattr(slurm, "cancel", lambda *_a, **_k: pytest.fail("no scheduler cancel"))
    launcher = FakeLauncher()
    # Stop and the queue take no hooks, so the public owner guards them against starting a process.
    monkeypatch.setattr(managed_scheduler, "start_process", launcher.start_process)

    assert hparam_runtime.stop_hparam_run(plan_dir, run["run_id"], reason=reason) == tmp_path / "run_manifest.tsv"

    canonical = _read_table(tmp_path / "run_manifest.tsv")[0]
    assert canonical["status"] == "stopped"
    assert canonical["stop_reason"] == reason
    assert canonical["stopped_at"] == now
    assert all(canonical.get(field, "") == before.get(field, "") == "" for field in EXECUTION_IDENTITY_FIELDS)
    assert canonical.get("launched_at", "") == canonical.get("stop_requested_at", "") == ""
    assert run_artifacts.read_hparam_plan(plan_dir) == plan
    snapshot = experiments.experiment_status(tmp_path)
    assert snapshot["summary"]["status_counts"] == {"stopped": 1}
    assert snapshot["summary"]["state"] == "ready_to_report"
    assert any(blocker["code"] == "failure_report_required" for blocker in snapshot["blockers"])
    assert {path: path.read_bytes() for path in frozen_paths} == frozen
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert [event["event_type"] for event in events].count("run_stopped") == 1
    stopped_event = next(event for event in events if event["event_type"] == "run_stopped")
    assert (stopped_event["step_id"], stopped_event["run_id"], stopped_event["reason"]) == (
        run["step_id"],
        run["run_id"],
        reason,
    )
    assert not any(event["event_type"] == "run_stop_requested" for event in events)
    before_repeat = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    with pytest.raises(ValueError, match="already terminal"):
        hparam_runtime.stop_hparam_run(plan_dir, run["run_id"], reason=reason)
    assert {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before_repeat
    hparam_runtime.launch_hparam_runs(plan_dir, dry_run=False, hooks=launcher.hooks())
    assert _read_table(tmp_path / "run_manifest.tsv")[0]["status"] == "stopped"
    monkeypatch.setattr(hparam_runtime, "launch_hparam_runs", lambda *_a, **_k: pytest.fail("no queue launch"))
    assert [row["status"] for row in hparam_runtime.run_hparam_queue(plan_dir, dry_run=False)] == ["stopped"]
    launcher.assert_not_started()


def test_hparam_unlaunched_stop_leaves_other_runs_launchable(tmp_path: Path, monkeypatch):
    recipe = _hparam_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    payload["search"]["max_runs"] = 2
    payload["search"]["parameters"]["runtime.lr"] = [1e-6, 2e-6]
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
    plan_dir = tmp_path / "plan"
    result = run_cli("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir))
    assert result.returncode == 0, result.stderr
    cancelled, remaining = run_artifacts.read_hparam_plan(plan_dir)["runs"]
    launcher = FakeLauncher()
    # Stop takes no hooks, so the public owner guards it against starting a process.
    monkeypatch.setattr(managed_scheduler, "start_process", launcher.start_process)
    monkeypatch.setattr(run_evidence, "read_process_identity", lambda *_a, **_k: None)
    monkeypatch.setattr(run_evidence, "stop_process_group", lambda *_a, **_k: pytest.fail("no signal"))

    hparam_runtime.stop_hparam_run(plan_dir, cancelled["run_id"], reason="cancel one candidate")
    hparam_runtime.launch_hparam_runs(plan_dir, dry_run=False, hooks=launcher.hooks())

    rows = {row["run_id"]: row for row in _read_table(tmp_path / "run_manifest.tsv")}
    assert rows[cancelled["run_id"]]["status"] == "stopped"
    assert rows[remaining["run_id"]]["status"] == "launched"
    assert len(launcher.starts) == 1
    assert remaining["script"] in launcher.commands[0]
    assert cancelled["script"] not in launcher.commands[0]


@pytest.mark.parametrize("backend", ["direct", "slurm"])
def test_hparam_metadata_stop_event_failure_keeps_terminal_commit(tmp_path: Path, monkeypatch, backend):
    if backend == "slurm":
        plan_dir, plan = _write_slurm_plan(tmp_path)
        # A launchable Slurm row is looked up by job name first; no job means a metadata-only stop.
        monkeypatch.setattr(slurm, "user_jobs", lambda *_a, **_k: [])
        monkeypatch.setattr(slurm, "cancel", lambda *_a, **_k: pytest.fail("no scheduler cancel"))
    else:
        recipe = _hparam_recipe(tmp_path)
        plan_dir = tmp_path / "plan"
        result = run_cli("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir))
        assert result.returncode == 0, result.stderr
        plan = run_artifacts.read_hparam_plan(plan_dir)
    run = plan["runs"][0]
    reason = "cancel before execution"
    stopped_at = "2026-08-30T04:05:06Z"
    monkeypatch.setattr(hparam_runtime, "utc_now", lambda: stopped_at)

    def fail_event(*_args, **_kwargs):
        raise RuntimeError("append_event failed")

    monkeypatch.setattr(hparam_runtime, "append_event", fail_event)

    with pytest.raises(RuntimeError, match="append_event failed"):
        hparam_runtime.stop_hparam_run(plan_dir, run["run_id"], reason=reason)

    canonical = next(row for row in _read_table(tmp_path / "run_manifest.tsv") if row["run_id"] == run["run_id"])
    assert canonical["status"] == "stopped"
    assert canonical["stop_reason"] == reason
    assert canonical["stopped_at"] == stopped_at
    before_retry = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    for name in ("merge_run_manifest", "append_event"):
        monkeypatch.setattr(hparam_runtime, name, lambda *_a, **_k: pytest.fail("terminal retry must not write"))

    with pytest.raises(ValueError, match="already terminal"):
        hparam_runtime.stop_hparam_run(plan_dir, run["run_id"], reason=reason)

    assert {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before_retry
    snapshot = experiments.experiment_status(tmp_path)
    assert next(row for row in snapshot["runs"] if row["run_id"] == run["run_id"])["status"] == "stopped"
    assert snapshot["summary"]["status_counts"] == (
        {"stopped": 1} if backend == "direct" else {"planned": 1, "stopped": 1}
    )
    if backend == "direct":
        assert snapshot["summary"]["state"] == "ready_to_report"
        assert any(blocker["code"] == "failure_report_required" for blocker in snapshot["blockers"])


@pytest.mark.parametrize(
    "evidence",
    [
        {"target": "local"},
        {"command": "unconfirmed launch"},
        {"pid": "123"},
        {"launched_at": "2026-08-30T04:00:00Z"},
        {"stop_requested_at": "2026-08-30T04:00:00Z"},
        {"status": "launched"},
        {"status": "missing_pid"},
    ],
)
def test_hparam_stop_does_not_metadata_cancel_uncertain_launch(tmp_path: Path, monkeypatch, evidence):
    recipe = _hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    result = run_cli("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir))
    assert result.returncode == 0, result.stderr
    rows = _read_table(tmp_path / "run_manifest.tsv")
    rows[0].update(evidence)
    manifests.write_rows(tmp_path / "run_manifest.tsv", rows)
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    monkeypatch.setattr(run_evidence, "read_process_identity", lambda *_a, **_k: pytest.fail("no identity read"))
    monkeypatch.setattr(run_evidence, "stop_process_group", lambda *_a, **_k: pytest.fail("no signal"))

    with pytest.raises(ValueError):
        hparam_runtime.stop_hparam_run(plan_dir, rows[0]["run_id"], reason="unconfirmed launch")

    assert {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before


@pytest.mark.parametrize("first_operation", ["stop", "launch"])
def test_hparam_unlaunched_stop_serializes_with_launch(tmp_path: Path, monkeypatch, first_operation):
    recipe = _hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    result = run_cli("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir))
    assert result.returncode == 0, result.stderr
    run = run_artifacts.read_hparam_plan(plan_dir)["runs"][0]
    first_ready = threading.Event()
    release_first = threading.Event()
    second_attempted = threading.Event()
    second_acquired = threading.Event()
    failures = []
    stopped = []
    real_lock = hparam_runtime.experiment_workspace.managed_run_lock
    real_merge = hparam_runtime.merge_run_manifest

    @contextmanager
    def track_lock(root):
        second = threading.current_thread().name == "second-operation"
        if second:
            second_attempted.set()
        with real_lock(root):
            if second:
                second_acquired.set()
            yield

    def merge(root, rows, **kwargs):
        if first_operation == "stop" and threading.current_thread().name == "first-operation":
            first_ready.set()
            assert release_first.wait(timeout=5)
        return real_merge(root, rows, **kwargs)

    def start(_execution, _command):
        canonical = _read_table(tmp_path / "run_manifest.tsv")[0]
        _write_process_identity(canonical["pid_path"])
        first_ready.set()
        assert release_first.wait(timeout=5)
        return "launched"

    launcher = FakeLauncher(start)

    def operation(name):
        try:
            if name == "stop":
                hparam_runtime.stop_hparam_run(plan_dir, run["run_id"], reason="cancel race")
            else:
                hparam_runtime.launch_hparam_runs(plan_dir, dry_run=False, hooks=launcher.hooks())
        except BaseException as exc:
            failures.append(exc)

    monkeypatch.setattr(hparam_runtime.experiment_workspace, "managed_run_lock", track_lock)
    monkeypatch.setattr(hparam_runtime, "merge_run_manifest", merge)
    # Stop takes no hooks, so the public owner guards it against starting a process.
    monkeypatch.setattr(managed_scheduler, "start_process", launcher.start_process)
    monkeypatch.setattr(run_evidence, "stop_process_group", lambda _row, identity: stopped.append(identity))
    first = threading.Thread(target=operation, args=(first_operation,), name="first-operation")
    second = threading.Thread(
        target=operation, args=("launch" if first_operation == "stop" else "stop",), name="second-operation"
    )
    first.start()
    try:
        assert first_ready.wait(timeout=5), failures
        second.start()
        assert second_attempted.wait(timeout=5)
        assert not second_acquired.is_set()
    finally:
        release_first.set()
        first.join(timeout=5)
        if second.ident is not None:
            second.join(timeout=5)
    assert not first.is_alive() and not second.is_alive()
    assert failures == []
    assert len(launcher.starts) == (first_operation == "launch")
    assert stopped == ([_process_identity()] if first_operation == "launch" else [])
    canonical = _read_table(tmp_path / "run_manifest.tsv")[0]
    assert canonical["status"] == "stopped"
    assert canonical["stop_reason"] == "cancel race"


def test_hparam_unlaunched_stop_survives_stale_monitor_commit(tmp_path: Path, monkeypatch):
    recipe = _hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    result = run_cli("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir))
    assert result.returncode == 0, result.stderr
    run = run_artifacts.read_hparam_plan(plan_dir)["runs"][0]
    real_merge = experiments.merge_run_manifest
    cancelled = False
    monkeypatch.setattr(hparam_runtime, "utc_now", lambda: "2026-08-30T04:05:06Z")

    def merge_after_cancel(root, rows, **kwargs):
        nonlocal cancelled
        if not cancelled:
            cancelled = True
            hparam_runtime.stop_hparam_run(plan_dir, run["run_id"], reason="cancel before launch")
            rows = [{**row, "stopped_at": "", "stop_reason": ""} for row in rows]
        return real_merge(root, rows, **kwargs)

    monkeypatch.setattr(experiments, "merge_run_manifest", merge_after_cancel)
    monkeypatch.setattr(run_evidence, "stop_process_group", lambda *_a, **_k: pytest.fail("no signal"))

    experiments.monitor_experiment(tmp_path)

    canonical = _read_table(tmp_path / "run_manifest.tsv")[0]
    assert canonical["status"] == "stopped"
    assert canonical["stop_reason"] == "cancel before launch"
    assert canonical["stopped_at"] == "2026-08-30T04:05:06Z"
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert [event["event_type"] for event in events].count("run_stopped") == 1


def test_hparam_stop_reads_workspace_state_only_under_run_lock(tmp_path: Path, monkeypatch):
    recipe = _hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    result = run_cli("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir))
    assert result.returncode == 0, result.stderr
    run = run_artifacts.read_hparam_plan(plan_dir)["runs"][0]

    manifest_path = call_while_run_lock_holder_commits(
        monkeypatch,
        tmp_path,
        lambda: hparam_runtime.stop_hparam_run(plan_dir, run["run_id"], reason="cancel during monitor commit"),
    )

    canonical = _read_table(tmp_path / "run_manifest.tsv")[0]
    assert manifest_path == tmp_path / "run_manifest.tsv"
    assert canonical["status"] == "stopped"
    assert canonical["stop_reason"] == "cancel during monitor commit"


def test_hparam_queue_reads_workspace_state_only_under_run_lock(tmp_path: Path, monkeypatch):
    recipe = _hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    result = run_cli("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir))
    assert result.returncode == 0, result.stderr
    # Terminal runs let the queue return after its first canonical poll.
    for run in run_artifacts.read_hparam_plan(plan_dir)["runs"]:
        hparam_runtime.stop_hparam_run(plan_dir, run["run_id"], reason="settle queue")

    rows = call_while_run_lock_holder_commits(
        monkeypatch, tmp_path, lambda: hparam_runtime.run_hparam_queue(plan_dir, dry_run=False)
    )

    assert rows == _read_table(tmp_path / "run_manifest.tsv")


def test_hparam_plan_reads_workspace_state_only_under_run_lock(tmp_path: Path, monkeypatch):
    recipe = _hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    result = run_cli("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir))
    assert result.returncode == 0, result.stderr

    plan = call_while_run_lock_holder_commits(
        monkeypatch, tmp_path, lambda: hparam_runtime.read_hparam_plan_under_run_lock(plan_dir)
    )

    assert plan == run_artifacts.read_hparam_plan(plan_dir)


def test_stop_signal_failure_does_not_commit_stopped_state(tmp_path: Path, monkeypatch):
    rows = _write_runtime_rows(tmp_path, [{"run_id": "run-000", "status": "running", **_process_identity()}])
    before_manifest = (tmp_path / "run_manifest.tsv").read_bytes()
    monkeypatch.setattr(run_evidence, "read_process_identity", lambda _path, _row: _process_identity())
    monkeypatch.setattr(
        run_evidence,
        "stop_process_group",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("Failed to stop process group")),
    )

    with pytest.raises(RuntimeError, match="Failed to stop process group"):
        hparam_runtime.stop_hparam_run(tmp_path, rows[0]["run_id"], reason="validation diverged")

    assert (tmp_path / "run_manifest.tsv").read_bytes() == before_manifest
    assert not (tmp_path / "events.jsonl").exists()


@pytest.mark.parametrize("pid_text", ["0", "-1"])
def test_hparam_stop_rejects_nonpositive_pid_before_kill(tmp_path: Path, monkeypatch, pid_text: str):
    rows = _write_runtime_rows(tmp_path, [{"run_id": "run-000", "status": "running"}])
    Path(rows[0]["pid_path"]).write_text(pid_text)
    before = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    killed = []
    monkeypatch.setattr(run_evidence.os, "kill", lambda pid, sig: killed.append((pid, sig)))

    with pytest.raises(RuntimeError, match="PID file is empty or invalid"):
        hparam_runtime.stop_hparam_run(tmp_path, "run-000", reason="invalid PID evidence")

    assert killed == []
    assert {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before


@pytest.mark.parametrize("failure", ["invalid_utf8", "os_error"])
def test_hparam_stop_rejects_unreadable_local_pid_before_kill(tmp_path: Path, monkeypatch, failure: str):
    rows = _write_runtime_rows(tmp_path, [{"run_id": "run-000", "status": "running"}])
    pid_path = Path(rows[0]["pid_path"])
    if failure == "invalid_utf8":
        pid_path.write_bytes(b"\xff")
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
    monkeypatch.setattr(run_evidence.os, "kill", lambda pid, sig: killed.append((pid, sig)))

    with pytest.raises(RuntimeError, match="PID file read failed"):
        hparam_runtime.stop_hparam_run(tmp_path, "run-000", reason="unreadable PID evidence")

    assert killed == []
    assert {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before


def test_hparam_stop_rejects_a_directory_pid_path_before_read_or_kill(tmp_path: Path, monkeypatch):
    rows = _write_runtime_rows(tmp_path, [{"run_id": "run-000", "status": "running"}])
    Path(rows[0]["pid_path"]).mkdir()
    calls = []
    monkeypatch.setattr(run_evidence, "read_process_identity", lambda *_args, **_kwargs: calls.append("read"))
    monkeypatch.setattr(run_evidence.os, "kill", lambda *_args: calls.append("kill"))

    with pytest.raises(ValueError, match="Managed output"):
        hparam_runtime.stop_hparam_run(tmp_path, "run-000", reason="unsafe PID topology")

    assert calls == []


@pytest.mark.parametrize("missing_field", ["pid", "process_group_id", "process_start_token"])
def test_hparam_stop_rejects_partial_canonical_process_identity_before_pid_read_or_signal(
    tmp_path: Path,
    monkeypatch,
    missing_field: str,
):
    identity = _process_identity()
    identity[missing_field] = ""
    rows = _write_runtime_rows(
        tmp_path,
        [{"run_id": "run-000", "status": "running", **identity}],
    )
    _write_process_identity(rows[0]["pid_path"])
    before = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    calls = []
    monkeypatch.setattr(run_evidence, "read_process_identity", lambda *_args, **_kwargs: calls.append("read"))
    monkeypatch.setattr(run_evidence, "stop_process_group", lambda *_args, **_kwargs: calls.append("signal"))

    with pytest.raises(ValueError, match="partial process identity"):
        hparam_runtime.stop_hparam_run(tmp_path, "run-000", reason="partial identity")

    assert calls == []
    assert {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before


@pytest.mark.parametrize(
    "process_args",
    [
        "bash {script}",
        "/bin/bash {script}",
    ],
)
def test_hparam_stop_binds_first_process_identity_to_the_frozen_script(
    tmp_path: Path,
    monkeypatch,
    process_args: str,
):
    root = tmp_path / "experiment with spaces"
    root.mkdir()
    rows = _write_runtime_rows(root, [{"run_id": "run-000", "status": "running"}])
    identity = _write_process_identity(rows[0]["pid_path"])
    commands = []

    def matched_script(command):
        commands.append(command)
        return subprocess.CompletedProcess([], 0, process_args.format(script=rows[0]["script"]) + "\n", "")

    stopped = []
    monkeypatch.setattr(run_evidence, "run_row_command", matched_script)
    monkeypatch.setattr(run_evidence, "stop_process_group", lambda _row, observed: stopped.append(observed))

    hparam_runtime.stop_hparam_run(root, "run-000", reason="matched frozen script")

    assert commands == ["ps -ww -p 123 -o args="]
    assert stopped == [identity]
    canonical = _read_table(root / "run_manifest.tsv")[0]
    assert canonical["pid"] == "123"
    assert canonical["process_group_id"] == "123"
    assert canonical["process_start_token"] == "proc:unit-start"
    assert canonical["status"] == "stopped"


@pytest.mark.parametrize(
    "process_args",
    [
        "python unrelated_job.py",
        "fakebash {script}",
        "python unrelated_job.py bash {script}",
        "conda run --no-capture-output -n exp bash {script}",
    ],
)
def test_hparam_stop_rejects_unbound_first_process_identity_before_signal_or_commit(
    tmp_path: Path,
    monkeypatch,
    process_args: str,
):
    rows = _write_runtime_rows(tmp_path, [{"run_id": "run-000", "status": "running"}])
    _write_process_identity(rows[0]["pid_path"])
    before = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    monkeypatch.setattr(
        run_evidence,
        "run_row_command",
        lambda _command: subprocess.CompletedProcess(
            [],
            0,
            process_args.format(script=rows[0]["script"]) + "\n",
            "",
        ),
    )
    calls = []
    monkeypatch.setattr(run_evidence, "stop_process_group", lambda *_args: calls.append("signal"))
    monkeypatch.setattr(hparam_runtime, "merge_run_manifest", lambda *_args: calls.append("commit"))

    with pytest.raises(run_evidence.ProcessIdentityError, match="does not match frozen script"):
        hparam_runtime.stop_hparam_run(tmp_path, "run-000", reason="unbound identity")

    assert calls == []
    assert {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before


def test_hparam_stop_uses_the_recorded_process_group_identity(tmp_path: Path, monkeypatch):
    rows = _write_runtime_rows(
        tmp_path,
        [{"run_id": "run-000", "status": "running", **_process_identity()}],
    )
    identity = _write_process_identity(rows[0]["pid_path"])
    calls = []
    monkeypatch.setattr(
        run_evidence,
        "stop_process_group",
        lambda row, observed: calls.append((row["run_id"], observed)),
    )

    hparam_runtime.stop_hparam_run(tmp_path, "run-000", reason="manual stop")

    assert calls == [("run-000", identity)]
    canonical = _read_table(tmp_path / "run_manifest.tsv")[0]
    assert canonical["status"] == "stopped"
    assert canonical["pid"] == "123"
    assert canonical["process_group_id"] == "123"
    assert canonical["process_start_token"] == "proc:unit-start"


def test_hparam_stop_rejects_a_reused_process_before_canonical_commit(tmp_path: Path, monkeypatch):
    rows = _write_runtime_rows(
        tmp_path,
        [{"run_id": "run-000", "status": "running", **_process_identity()}],
    )
    _write_process_identity(rows[0]["pid_path"])
    before = (tmp_path / "run_manifest.tsv").read_bytes()
    monkeypatch.setattr(
        run_evidence,
        "stop_process_group",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("PID 123 was reused by a different process")),
    )

    with pytest.raises(RuntimeError, match="was reused"):
        hparam_runtime.stop_hparam_run(tmp_path, "run-000", reason="stale process identity")

    assert (tmp_path / "run_manifest.tsv").read_bytes() == before


def _fake_process_probe(monkeypatch, leader: dict[str, int | str], group_running: bool | None) -> None:
    """Make the in-process probe observe ``leader`` as the live PID and report ``group_running``."""
    monkeypatch.setattr(run_evidence.os, "getpgid", lambda _pid: leader["process_group_id"])
    monkeypatch.setattr(run_evidence, "start_token", lambda _pid: leader["process_start_token"])
    monkeypatch.setattr(run_evidence, "process_group_running", lambda _pgid: group_running)


@pytest.mark.parametrize(
    ("leader_state", "live_child", "expected"),
    [
        pytest.param("Z", False, False, id="zombie-only"),
        pytest.param("X", False, False, id="dead-only"),
        pytest.param("Z", True, True, id="live-child"),
    ],
)
def test_process_group_probe_requires_a_non_zombie_member(
    tmp_path: Path, leader_state: str, live_child: bool, expected: bool
):
    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    _write_proc_stat(proc_root, 123, 123, leader_state)
    if live_child:
        _write_proc_stat(proc_root, 124, 123, "S")
    _write_proc_stat(proc_root, 200, 200, "S")

    assert process_identity.process_group_running(123, str(proc_root)) is expected


def test_process_group_probe_uses_the_live_leader_fast_path(tmp_path: Path, monkeypatch):
    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    _write_proc_stat(proc_root, 123, 123, "S")
    monkeypatch.setattr(
        process_identity.os,
        "listdir",
        lambda *_args: (_ for _ in ()).throw(AssertionError("live leader should not scan /proc")),
    )

    assert process_identity.process_group_running(123, str(proc_root)) is True


def test_process_group_probe_preserves_uncertainty_from_unreadable_proc_entries(tmp_path: Path):
    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    _write_proc_stat(proc_root, 123, 123, "Z")
    (proc_root / "124" / "stat").mkdir(parents=True)

    assert process_identity.process_group_running(123, str(proc_root)) is None


@pytest.mark.parametrize("group_running", [False, True], ids=["zombie-only", "live-child"])
def test_process_monitor_uses_group_liveness_for_a_matching_leader(monkeypatch, group_running: bool):
    identity = _process_identity()
    _fake_process_probe(monkeypatch, identity, group_running)

    assert run_evidence.process_identity_running({}, identity) is group_running


def test_process_monitor_rejects_a_reused_pid_start_token(monkeypatch):
    identity = _process_identity()
    reused = {**identity, "process_start_token": "proc:other-start"}
    _fake_process_probe(monkeypatch, reused, True)

    with pytest.raises(RuntimeError, match="was reused"):
        run_evidence.process_identity_running({}, identity)


@pytest.mark.parametrize(
    ("leader_error", "token", "group_running", "expected"),
    [
        pytest.param(None, None, True, None, id="unreadable-start-token"),
        # An uncertain group reads as unknown before the observed leader is compared with the identity.
        pytest.param(None, "proc:other-start", None, None, id="uncertain-group"),
        pytest.param(PermissionError(), "proc:unit-start", True, None, id="leader-inspection-error"),
        pytest.param(ProcessLookupError(), None, False, False, id="leader-gone"),
        pytest.param(ProcessLookupError(), None, True, True, id="leader-gone-group-alive"),
    ],
)
def test_process_monitor_reports_an_uncertain_probe_as_unknown(
    monkeypatch, leader_error: OSError | None, token: str | None, group_running: bool | None, expected: bool | None
):
    identity = _process_identity()
    monkeypatch.setattr(
        run_evidence.os,
        "getpgid",
        lambda _pid: (_ for _ in ()).throw(leader_error) if leader_error is not None else 123,
    )
    monkeypatch.setattr(run_evidence, "start_token", lambda _pid: token)
    monkeypatch.setattr(run_evidence, "process_group_running", lambda _pgid: group_running)

    assert run_evidence.process_identity_running({}, identity) is expected


@pytest.mark.parametrize("group_running", [False, True])
def test_process_monitor_treats_a_leader_that_exits_mid_probe_as_gone(monkeypatch, group_running: bool):
    # getpgid sees the leader, then its stat vanishes before the start token is read (a zombie being reaped).
    identity = _process_identity()
    calls = []

    def getpgid(_pid):
        calls.append(_pid)
        if len(calls) > 1:
            raise ProcessLookupError()
        return 123

    monkeypatch.setattr(run_evidence.os, "getpgid", getpgid)
    monkeypatch.setattr(run_evidence, "start_token", lambda _pid: None)
    monkeypatch.setattr(run_evidence, "process_group_running", lambda _pgid: group_running)

    assert run_evidence.process_identity_running({}, identity) is group_running


def test_process_group_probe_reads_members_with_undecodable_names(tmp_path: Path):
    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    _write_proc_stat(proc_root, 123, 123, "Z")
    member = proc_root / "124"
    member.mkdir()
    (member / "stat").write_bytes(b"124 (bad\xffname) S 1 123\n")

    assert process_identity.process_group_running(123, proc_root=str(proc_root)) is True


def test_process_group_probe_skips_members_that_exit_mid_scan(tmp_path: Path, monkeypatch):
    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    _write_proc_stat(proc_root, 123, 123, "Z")
    _write_proc_stat(proc_root, 124, 123, "S")
    real_open = open

    def open_exiting_member(path, *args, **kwargs):
        if str(path) == str(proc_root / "124" / "stat"):
            raise ProcessLookupError()
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(process_identity, "open", open_exiting_member, raising=False)

    assert process_identity.process_group_running(123, proc_root=str(proc_root)) is False


@pytest.mark.parametrize(
    ("process_args", "expected_status", "expected_pid"),
    [
        pytest.param("/bin/bash {script}", "running", 123, id="matching-script"),
        pytest.param("python unrelated_job.py", "missing_pid", "", id="unrelated-process"),
    ],
)
def test_status_binds_first_process_identity_only_to_the_frozen_script(
    tmp_path: Path,
    monkeypatch,
    process_args: str,
    expected_status: str,
    expected_pid: int | str,
):
    pid_path = tmp_path / "managed.pid"
    _write_process_identity(pid_path)
    script = tmp_path / "launch.sh"
    script.write_text("#!/usr/bin/env bash\ntrue\n")
    row = {
        "script": str(script),
        "pid_path": str(pid_path),
        "status": "running",
    }
    monkeypatch.setattr(run_evidence, "process_identity_running", lambda *_args: True)
    monkeypatch.setattr(
        run_evidence,
        "run_row_command",
        lambda command: (
            subprocess.CompletedProcess(
                [],
                0,
                process_args.format(script=script) + "\n",
                "",
            )
            if command == "ps -ww -p 123 -o args="
            else (_ for _ in ()).throw(AssertionError(command))
        ),
    )

    observed = run_evidence.status_row(tmp_path, row, row, script_commits_terminal_status=False)

    assert observed["status"] == expected_status
    assert observed["pid"] == expected_pid
    assert observed.get("process_group_id", "") == expected_pid
    assert observed.get("process_start_token", "") == ("proc:unit-start" if expected_pid else "")


def test_status_keeps_dead_unfrozen_process_identity_unbound(tmp_path: Path, monkeypatch):
    script = tmp_path / "launch.sh"
    script.write_text("#!/usr/bin/env bash\ntrue\n")
    row = {
        "target": "local",
        "script": str(script),
        "pid_path": str(tmp_path / "managed.pid"),
        "status": "launched",
    }
    monkeypatch.setattr(run_evidence, "read_process_identity", lambda *_args: _process_identity())
    monkeypatch.setattr(run_evidence, "process_identity_running", lambda *_args: False)
    monkeypatch.setattr(
        run_evidence,
        "_require_process_script",
        lambda *_args: (_ for _ in ()).throw(AssertionError("dead identity must remain unbound")),
    )
    monkeypatch.setattr(run_evidence, "runtime_artifacts", lambda *_args: ("", {}, []))
    monkeypatch.setattr(run_evidence, "log_tail", lambda *_args: "")

    observed = run_evidence.status_row(tmp_path, row, row, script_commits_terminal_status=False)

    assert observed["status"] == "missing_pid"
    assert observed["pid"] == ""
    assert observed.get("process_group_id", "") == ""
    assert observed.get("process_start_token", "") == ""


def test_hparam_run_queue_fails_after_dead_unfrozen_process_identity(tmp_path: Path, monkeypatch):
    rows = _write_runtime_rows(tmp_path, [{"run_id": "run-000", "status": "launched"}])
    _write_process_identity(rows[0]["pid_path"])
    monkeypatch.setattr(run_evidence, "process_identity_running", lambda *_args: False)
    monkeypatch.setattr(
        run_evidence,
        "_require_process_script",
        lambda *_args: (_ for _ in ()).throw(AssertionError("dead identity must remain unbound")),
    )
    monkeypatch.setattr(managed_scheduler, "start_process", lambda *_args: pytest.fail("no start"))
    monkeypatch.setattr(hparam_runtime.time, "sleep", lambda *_args: pytest.fail("no sleep"))

    with pytest.raises(managed_scheduler.MissingPidCapacityError, match="run-000 has status missing_pid"):
        hparam_runtime.run_hparam_queue(tmp_path, dry_run=False)

    assert _read_table(tmp_path / "run_manifest.tsv")[0]["status"] == "missing_pid"


def test_status_marks_a_zombie_only_managed_process_finished(tmp_path: Path, monkeypatch):
    pid_path = tmp_path / "managed.pid"
    identity = _write_process_identity(pid_path)
    row = {
        "script": str(tmp_path / "launch.sh"),
        "pid_path": str(pid_path),
        "status": "running",
        **identity,
    }
    _fake_process_probe(monkeypatch, identity, False)

    observed = run_evidence.status_row(tmp_path, row, row, script_commits_terminal_status=False)

    assert observed["status"] == "finished"


def test_local_stop_checks_once_more_at_the_deadline_for_a_zombie_only_group(monkeypatch):
    identity = _process_identity()
    states = iter([True, False])
    clock = iter([0.0, 1.0])
    calls = []
    monkeypatch.setattr(
        run_evidence,
        "process_identity_running",
        lambda *_args: calls.append("probe") or next(states),
    )
    monkeypatch.setattr(run_evidence.os, "killpg", lambda *_args: calls.append("signal"))
    monkeypatch.setattr(run_evidence.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(
        run_evidence.time,
        "sleep",
        lambda *_args: (_ for _ in ()).throw(AssertionError("zombie-only group should not wait")),
    )

    run_evidence.stop_process_group({}, identity, timeout=0.5)

    assert calls == ["probe", "signal", "probe"]


@pytest.mark.parametrize("script_commits_terminal_status", [False, True])
def test_status_keeps_a_reused_managed_process_identity_capacity_blocking(
    tmp_path: Path, monkeypatch, script_commits_terminal_status: bool
):
    pid_path = tmp_path / "managed.pid"
    identity = _write_process_identity(pid_path)
    reused = {**identity, "process_start_token": "proc:other-start"}
    row = {
        "script": str(tmp_path / "launch.sh"),
        "pid_path": str(pid_path),
        "status": "running",
        **identity,
    }
    _fake_process_probe(monkeypatch, reused, True)

    observed = run_evidence.status_row(
        tmp_path,
        row,
        row,
        script_commits_terminal_status=script_commits_terminal_status,
    )

    assert observed["status"] == "missing_pid"
    assert "reused by a different process" in observed["process_identity_error"]


@pytest.mark.parametrize("script_commits_terminal_status", [False, True])
def test_status_keeps_legacy_integer_only_managed_identity_capacity_blocking(
    tmp_path: Path, script_commits_terminal_status: bool
):
    pid_path = tmp_path / "managed.pid"
    pid_path.write_text("123")
    row = {
        "script": str(tmp_path / "launch.sh"),
        "pid_path": str(pid_path),
        "status": "running",
        "pid": "123",
    }

    observed = run_evidence.status_row(
        tmp_path,
        row,
        row,
        script_commits_terminal_status=script_commits_terminal_status,
    )

    assert observed["status"] == "missing_pid"
    assert "partial process identity" in observed["process_identity_error"]


def test_status_marks_lifecycle_owned_script_exit_without_terminal_commit_failed(tmp_path: Path, monkeypatch):
    pid_path = tmp_path / "managed.pid"
    identity = _write_process_identity(pid_path)
    log_path = tmp_path / "managed.log"
    log_path.write_text("training completed\n")
    row = {
        "script": str(tmp_path / "launch.sh"),
        "pid_path": str(pid_path),
        "log_path": str(log_path),
        "status": "running",
        **identity,
    }
    monkeypatch.setattr(run_evidence, "process_identity_running", lambda *_args: False)

    observed = run_evidence.status_row(tmp_path, row, row, script_commits_terminal_status=True)

    assert observed["status"] == "failed"


def test_status_fails_closed_without_monitor_owned_exit_code(tmp_path: Path, monkeypatch):
    pid_path = tmp_path / "managed.pid"
    identity = _write_process_identity(pid_path)
    log_path = tmp_path / "managed.log"
    log_path.write_text("training completed\n")
    row = {
        "script": str(tmp_path / "launch.sh"),
        "pid_path": str(pid_path),
        "log_path": str(log_path),
        "status": "running",
        "terminal_status_owner": "monitor",
        **identity,
    }
    monkeypatch.setattr(run_evidence, "process_identity_running", lambda *_args: False)

    observed = run_evidence.status_row(tmp_path, row, row, script_commits_terminal_status=False)

    assert observed["status"] == "failed"


@pytest.mark.parametrize(("exit_code", "expected"), [(0, False), (7, True)])
def test_log_failure_projection_recognizes_monitor_exit_code(tmp_path: Path, exit_code: int, expected: bool):
    log_path = tmp_path / "managed.log"
    log_path.write_text(f"{MONITOR_EXIT_CODE_PREFIX}{exit_code}\n")

    assert run_evidence.log_has_failure(log_path, require_exit_code=True) is expected


@pytest.mark.parametrize("prefix", ["0: ", "1: "])
def test_log_failure_projection_rejects_rank_labelled_exit_marker(tmp_path: Path, prefix: str):
    # Slurm runs report their exit through run_dir/exit_code, so no srun rank label is a terminal marker.
    log_path = tmp_path / "managed.log"
    log_path.write_text(f"{prefix}{MONITOR_EXIT_CODE_PREFIX}0\n")

    assert run_evidence.log_has_failure(log_path, require_exit_code=True) is True


def test_log_failure_projection_keeps_labeled_traceback_detection(tmp_path: Path):
    log_path = tmp_path / "managed.log"
    log_path.write_text("2: Traceback: CUDA out of memory\n")

    assert run_evidence.log_has_failure(log_path) is True


def test_status_marks_lifecycle_owned_running_run_with_missing_pid_as_missing_pid(tmp_path: Path):
    row = {
        "script": str(tmp_path / "launch.sh"),
        "pid_path": str(tmp_path / "missing.pid"),
        "status": "running",
        **_process_identity(),
    }

    observed = run_evidence.status_row(tmp_path, row, row, script_commits_terminal_status=True)

    assert observed["status"] == "missing_pid"


def test_status_marks_lifecycle_owned_script_without_process_identity_exit_failed(tmp_path: Path):
    row = {
        "script": str(tmp_path / "launch.sh"),
        "state": "finished",
        "status": "running",
    }

    observed = run_evidence.status_row(tmp_path, row, row, script_commits_terminal_status=True)

    assert observed["status"] == "failed"


@pytest.mark.parametrize("status", ["completed", "superseded"])
def test_hparam_stop_rejects_terminal_status_before_pid_or_mutation(tmp_path: Path, monkeypatch, status: str):
    _write_runtime_rows(
        tmp_path,
        [{"run_id": "run-000", "status": "planned" if status == "superseded" else "running"}],
    )
    merge_run_manifest(tmp_path, [{"step_id": "train-model", "run_id": "run-000", "status": status}])
    before = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    calls = []
    monkeypatch.setattr(run_evidence, "read_pid", lambda *_args: calls.append("read_pid") or 123)
    monkeypatch.setattr(run_evidence.os, "kill", lambda *_args: calls.append("kill"))
    monkeypatch.setattr(hparam_runtime, "merge_run_manifest", lambda *_args: calls.append("merge"))
    monkeypatch.setattr(hparam_runtime, "append_event", lambda *_args: calls.append("event"))

    with pytest.raises(ValueError, match="already terminal"):
        hparam_runtime.stop_hparam_run(tmp_path, "run-000", reason="terminal run")

    assert calls == []
    assert {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before


def test_hparam_stop_commits_one_final_canonical_row(tmp_path: Path, monkeypatch):
    rows = _write_runtime_rows(
        tmp_path,
        [{"run_id": "run-000", "status": "running", **_process_identity()}],
    )
    _write_process_identity(rows[0]["pid_path"])
    stopped = []
    monkeypatch.setattr(run_evidence, "stop_process_group", lambda _row, identity: stopped.append(identity))

    hparam_runtime.stop_hparam_run(tmp_path, "run-000", reason="manual stop")

    canonical = _read_table(tmp_path / "run_manifest.tsv")[0]
    assert canonical["status"] == "stopped"
    assert canonical["stop_reason"] == "manual stop"
    assert stopped == [_process_identity()]
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert [event["event_type"] for event in events].count("run_stopped") == 1

    with pytest.raises(ValueError, match="already terminal"):
        hparam_runtime.stop_hparam_run(tmp_path, "run-000", reason="repeat stop")

    assert stopped == [_process_identity()]
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert [event["event_type"] for event in events].count("run_stopped") == 1


def test_hparam_stop_serializes_process_exit_and_terminal_commit_against_monitor(tmp_path: Path, monkeypatch):
    rows = _write_runtime_rows(
        tmp_path,
        [{"run_id": "run-000", "status": "running", **_process_identity()}],
    )
    identity = _write_process_identity(rows[0]["pid_path"])
    real_merge = merge_run_manifest
    monitor_observing = threading.Event()
    release_monitor_observation = threading.Event()
    monitor_commit_ready = threading.Event()
    release_monitor_commit = threading.Event()
    monitor_failures = []

    def stale_observation(_root, observation, _prior, **_kwargs):
        # The monitor read the running row under the run lock and observes outside it while stop runs.
        if threading.current_thread().name == "competing-experiment-monitor":
            monitor_observing.set()
            assert release_monitor_observation.wait(timeout=5)
        return {**observation, "status": "finished"}

    def competing_merge(root, rows, **kwargs):
        if threading.current_thread().name == "competing-experiment-monitor":
            monitor_commit_ready.set()
            assert release_monitor_commit.wait(timeout=5)
        return real_merge(root, rows, **kwargs)

    def monitor():
        try:
            experiments.monitor_experiment(tmp_path)
        except Exception as exc:
            monitor_failures.append(exc)

    monitor_thread = threading.Thread(target=monitor, name="competing-experiment-monitor")
    stopped = []

    def stop_while_monitor_commits(_row, observed):
        stopped.append(observed)
        release_monitor_observation.set()
        assert monitor_commit_ready.wait(timeout=5)
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
        try:
            assert lock_probe.returncode == 1
        finally:
            release_monitor_commit.set()

    monkeypatch.setattr(experiments, "merge_run_manifest", competing_merge)
    monkeypatch.setattr(run_evidence, "status_row", stale_observation)
    monkeypatch.setattr(run_evidence, "stop_process_group", stop_while_monitor_commits)
    monitor_thread.start()
    assert monitor_observing.wait(timeout=5)

    hparam_runtime.stop_hparam_run(tmp_path, "run-000", reason="manual stop")
    monitor_thread.join(timeout=5)

    assert not monitor_thread.is_alive()
    assert monitor_failures == []
    assert stopped == [identity]
    canonical = _read_table(tmp_path / "run_manifest.tsv")[0]
    assert canonical["status"] == "stopped"
    assert canonical["stop_reason"] == "manual stop"
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert [event["event_type"] for event in events].count("run_stopped") == 1
    assert not any(
        event.get("event_type") == "run_status_changed" and event.get("to") in {"finished", "completed"}
        for event in events
    )


def _publish_queue_plan(tmp_path: Path, backend: str) -> Path:
    if backend == "slurm":
        plan_dir, _plan = _write_slurm_plan(tmp_path)
        return plan_dir
    recipe = _hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    result = run_cli("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir))
    assert result.returncode == 0, result.stderr
    return plan_dir


def _record_queue_runner_spawns(monkeypatch) -> list[tuple[list[str], dict]]:
    real_popen = subprocess.Popen
    spawns = []

    def popen(argv, **kwargs):
        if list(argv[1:4]) != ["-m", "agent_tools", "hparam-run-queue"]:
            return real_popen(argv, **kwargs)
        log = os.fstat(kwargs["stdout"])
        kwargs["log_identity"] = (log.st_dev, log.st_ino)
        kwargs["log_flags"] = fcntl.fcntl(kwargs["stdout"], fcntl.F_GETFL)
        spawns.append((list(argv), kwargs))
        return SimpleNamespace(pid=4242)

    monkeypatch.setattr(hparam_runtime.subprocess, "Popen", popen)
    return spawns


def test_hparam_run_queue_execute_detaches_one_runner_for_a_direct_plan(tmp_path: Path, monkeypatch, capsys):
    plan_dir = _publish_queue_plan(tmp_path, "direct")
    spawns = _record_queue_runner_spawns(monkeypatch)
    launcher = FakeLauncher()
    monkeypatch.setattr(managed_scheduler, "start_process", launcher.start_process)
    monkeypatch.setattr(hparam_runtime, "launch_hparam_runs", lambda *_a, **_k: pytest.fail("no queue launch"))
    manifest_before = (tmp_path / "run_manifest.tsv").read_bytes()
    log_path = plan_dir / "queue_runner.log"

    assert cli.main(["hparam-run-queue", "--plan-dir", str(plan_dir), "--execute", "--poll-seconds", "5"]) == 0

    assert capsys.readouterr().out.splitlines() == [
        "Mode: execute (detached queue runner started)",
        f"Queue runner: started; pid=4242; log={log_path}",
    ]
    [(argv, kwargs)] = spawns
    assert argv == [
        sys.executable,
        "-m",
        "agent_tools",
        "hparam-run-queue",
        "--plan-dir",
        str(plan_dir),
        "--execute",
        "--foreground",
        "--poll-seconds",
        "5.0",
    ]
    assert kwargs["start_new_session"] is True
    assert kwargs["stdin"] == subprocess.DEVNULL
    assert kwargs["stdout"] == kwargs["stderr"]
    log_stat = log_path.stat()
    assert kwargs["log_identity"] == (log_stat.st_dev, log_stat.st_ino)
    assert kwargs["log_flags"] & os.O_APPEND
    assert f"starting queue runner for {plan_dir}" in log_path.read_text()
    # The caller only checked the lock: the detached runner takes it, and this process launched nothing.
    assert managed_scheduler.queue_runner_liveness(plan_dir)["active"] is False
    assert (tmp_path / "run_manifest.tsv").read_bytes() == manifest_before
    launcher.assert_not_started()


def test_hparam_run_queue_refuses_a_second_runner_while_the_lock_is_held(tmp_path: Path, monkeypatch, capsys):
    plan_dir = _publish_queue_plan(tmp_path, "direct")
    spawns = _record_queue_runner_spawns(monkeypatch)
    monkeypatch.setattr(hparam_runtime, "launch_hparam_runs", lambda *_a, **_k: pytest.fail("no queue launch"))
    log_path = plan_dir / "queue_runner.log"

    with managed_scheduler.queue_runner_lock(plan_dir, record_pid=True):
        assert cli.main(["hparam-run-queue", "--plan-dir", str(plan_dir), "--execute"]) == 1
        error = capsys.readouterr().err
        assert "already active" in error
        assert f"pid {os.getpid()}" in error
        assert str(log_path) in error
        with pytest.raises(RuntimeError, match="already active"):
            hparam_runtime.run_hparam_queue(plan_dir, dry_run=False)

    assert spawns == []
    assert not log_path.exists()


def test_hparam_run_queue_foreground_holds_the_runner_lock_for_the_loop(tmp_path: Path, monkeypatch):
    plan_dir = _publish_queue_plan(tmp_path, "direct")
    plan = run_artifacts.read_hparam_plan(plan_dir)
    observed = []

    def launch(_run_dir, **_kwargs):
        observed.append(managed_scheduler.queue_runner_liveness(plan_dir))
        for run in plan["runs"]:
            hparam_runtime.stop_hparam_run(plan_dir, run["run_id"], reason="settle queue")

    monkeypatch.setattr(hparam_runtime, "launch_hparam_runs", launch)
    monkeypatch.setattr(hparam_runtime.time, "sleep", lambda *_args: pytest.fail("settled queue must not sleep"))

    rows = hparam_runtime.run_hparam_queue(plan_dir, dry_run=False)

    assert {row["status"] for row in rows} == {"stopped"}

    assert observed == [
        {
            "plan_dir": str(plan_dir),
            "active": True,
            "pid": str(os.getpid()),
            "log_path": str(plan_dir / "queue_runner.log"),
        }
    ]
    assert managed_scheduler.queue_runner_liveness(plan_dir)["active"] is False


def test_hparam_queue_runner_is_not_started_for_a_slurm_plan(tmp_path: Path, monkeypatch):
    plan_dir = _publish_queue_plan(tmp_path, "slurm")
    spawns = _record_queue_runner_spawns(monkeypatch)

    assert hparam_runtime.start_hparam_queue_runner(plan_dir) is None

    assert spawns == []
    assert not (plan_dir / "queue_runner.lock").exists()
    assert not (plan_dir / "queue_runner.log").exists()
