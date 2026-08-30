from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest
from test_agent_tools_hparam_runtime import (
    _hparam_recipe,
    _is_remote_python_program,
    _process_identity,
    _read_table,
    _run,
    _write_process_identity,
    _write_runtime_rows,
)
from test_agent_tools_hparam_runtime import _stub_execution_snapshot_preflight  # noqa: F401

from agent_tools import hparam_runtime, manifests, run_artifacts, run_evidence, transport
from agent_tools.experiment_workspace import merge_run_manifest
from agent_tools.hparam_runtime import monitor_hparam_runs


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
        run_evidence.status_row(tmp_path, rows[0], rows[0], script_commits_terminal_status=False)


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
        if _is_remote_python_program(command, "run_evidence.read_pid_text"):
            return subprocess.CompletedProcess([], 0, "123\n", "")
        if command.startswith("ps "):
            return subprocess.CompletedProcess([], 0, "123\n", "")
        if _is_remote_python_program(command, "run_evidence.runtime_artifacts"):
            assert "json.load" in command
            assert "stat.S_ISREG" in command
            assert "stat.S_ISDIR" in command
            return subprocess.run(["bash", "-lc", command], text=True, capture_output=True)
        if command.startswith("tail -n 8"):
            return subprocess.CompletedProcess([], 0, "running", "")
        raise AssertionError(command)

    monkeypatch.setattr(run_evidence, "run_row_command", fake_command)

    with pytest.raises(RuntimeError, match="runtime artifact observation failed"):
        run_evidence.status_row(tmp_path, row, row, script_commits_terminal_status=False)


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
        if _is_remote_python_program(command, "run_evidence.read_pid_text"):
            return subprocess.CompletedProcess([], 0, "123\n", "")
        if command.startswith("ps "):
            return subprocess.CompletedProcess([], 0, "123\n", "")
        if _is_remote_python_program(command, "run_evidence.runtime_artifacts"):
            return subprocess.run(["bash", "-lc", command], text=True, capture_output=True)
        if command.startswith("tail -n 8"):
            return subprocess.CompletedProcess([], 0, "running", "")
        raise AssertionError(command)

    monkeypatch.setattr(run_evidence, "run_row_command", fake_command)

    observed = run_evidence.status_row(tmp_path, row, row, script_commits_terminal_status=False)

    assert observed["run_manifest"] == (str(manifest) if state == "regular" else "")


def test_find_run_manifest_distinguishes_missing_and_valid_regular_file(tmp_path: Path):
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()
    run = {"runtime_dir": str(runtime_dir)}

    assert run_artifacts.find_run_manifest(run) is None

    manifest = runtime_dir / "run_manifest.json"
    manifest.write_text(json.dumps({"metrics": {"val_ahi_pearson": 0.7}}))

    assert run_artifacts.find_run_manifest(run) == manifest


def test_hparam_monitor_handles_running_missing_and_failed_rows(tmp_path: Path, monkeypatch):
    pid_path = tmp_path / "running.pid"
    _write_process_identity(pid_path, 101)
    missing_pid = tmp_path / "missing.pid"
    fail_pid = tmp_path / "fail.pid"
    _write_process_identity(fail_pid, 102)
    fail_log = tmp_path / "fail.log"
    fail_log.write_text("Traceback\nRuntimeError: boom\n")
    _write_runtime_rows(
        tmp_path,
        [
            {
                "run_id": "running",
                "version": "v1",
                "pid_path": str(pid_path),
                "status": "launched",
                **_process_identity(101),
            },
            {"run_id": "missing", "version": "v2", "pid_path": str(missing_pid), "status": "launched"},
            {
                "run_id": "failed",
                "version": "v3",
                "pid_path": str(fail_pid),
                "log_path": str(fail_log),
                "status": "launched",
                **_process_identity(102),
            },
        ],
    )
    monkeypatch.setattr(
        run_evidence,
        "process_identity_running",
        lambda _row, identity: identity["pid"] == 101,
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


def test_hparam_monitor_polls_until_the_current_plan_is_terminal(tmp_path: Path, monkeypatch):
    _write_runtime_rows(tmp_path, [{"run_id": "run-000", "status": "running"}])
    statuses = iter(["running", "finished"])
    observed = []
    sleeps = []

    def observe(_root, prior, *_args, **_kwargs):
        observed.append((prior["status"], prior.get("log_tail", "")))
        return {**prior, "status": next(statuses)}

    def sleep(seconds):
        sleeps.append(seconds)
        rows = _read_table(tmp_path / "run_manifest.tsv")
        rows[0]["log_tail"] = "external-update"
        manifests.write_rows(tmp_path / "run_manifest.tsv", rows)

    monkeypatch.setattr(hparam_runtime.scheduler, "observe_run", observe)
    monkeypatch.setattr(hparam_runtime.time, "sleep", sleep)
    monkeypatch.setattr(hparam_runtime, "launch_hparam_runs", lambda *_args, **_kwargs: pytest.fail("no launch"))
    monkeypatch.setattr(hparam_runtime.scheduler.slurm, "submit", lambda *_args, **_kwargs: pytest.fail("no submit"))

    out = monitor_hparam_runs(tmp_path, once=False, poll_seconds=60)

    assert out == tmp_path / "run_status.tsv"
    assert observed == [("running", ""), ("running", "external-update")]
    assert sleeps == [60]
    assert _read_table(tmp_path / "run_manifest.tsv")[0]["status"] == "finished"
    assert _read_table(tmp_path / "run_status.tsv")[0]["status"] == "finished"


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
        lambda _root, observation, _prior, *, script_commits_terminal_status, health: {
            **observation,
            "status": "finished",
        },
    )

    hparam_runtime.monitor_hparam_runs(tmp_path)

    assert _read_table(tmp_path / "run_manifest.tsv")[0]["status"] == "failed"
    assert _read_table(tmp_path / "run_status.tsv")[0]["status"] == "failed"
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    status_event = next(event for event in events if event["event_type"] == "run_status_changed")
    assert status_event["from"] == "running"
    assert status_event["to"] == "failed"


def test_hparam_monitor_marks_clean_exit_finished_without_launch_manifest(tmp_path: Path, monkeypatch):
    rows = _write_runtime_rows(
        tmp_path,
        [{"run_id": "run-000", "status": "running", **_process_identity()}],
    )
    _write_process_identity(rows[0]["pid_path"])
    monkeypatch.setattr(run_evidence, "process_identity_running", lambda *_args: False)
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


@pytest.mark.parametrize("output_name", ["run_status", "status_report"])
@pytest.mark.parametrize("alias_kind", ["symlink", "hardlink"])
def test_continuous_hparam_monitor_revalidates_output_aliases_each_round(
    tmp_path: Path,
    monkeypatch,
    output_name: str,
    alias_kind: str,
):
    _write_runtime_rows(tmp_path, [{"run_id": "run-000", "status": "running"}])
    statuses = iter(["running", "finished"])
    experiment_path = tmp_path / "experiment.yaml"
    experiment_bytes = experiment_path.read_bytes()
    alias_path = tmp_path / "run_status.tsv"
    if output_name == "status_report":
        alias_path = tmp_path / "reports" / "status.md"

    def observe(_root, prior, *_args, **_kwargs):
        return {**prior, "status": next(statuses)}

    def replace_output_with_alias(seconds):
        assert seconds == 60
        alias_path.unlink()
        if alias_kind == "symlink":
            alias_path.symlink_to(experiment_path)
        else:
            alias_path.hardlink_to(experiment_path)

    monkeypatch.setattr(hparam_runtime.scheduler, "observe_run", observe)
    monkeypatch.setattr(hparam_runtime.time, "sleep", replace_output_with_alias)

    with pytest.raises(ValueError, match="Managed output"):
        monitor_hparam_runs(tmp_path, once=False, poll_seconds=60)

    assert experiment_path.read_bytes() == experiment_bytes
    assert alias_path.read_bytes() == experiment_bytes
    assert _read_table(tmp_path / "run_manifest.tsv")[0]["status"] == "running"


@pytest.mark.parametrize("poll_seconds", [0, -1, float("nan"), float("inf")])
def test_hparam_monitor_rejects_invalid_poll_interval_before_writing(
    tmp_path: Path,
    monkeypatch,
    poll_seconds: float,
):
    _write_runtime_rows(tmp_path, [{"run_id": "run-000", "status": "running"}])
    before = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    monkeypatch.setattr(hparam_runtime.time, "sleep", lambda *_args: pytest.fail("invalid interval must not sleep"))

    with pytest.raises(ValueError, match="poll_seconds must be positive"):
        monitor_hparam_runs(tmp_path, once=False, poll_seconds=poll_seconds)

    assert {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before


def test_hparam_monitor_keeps_failed_status_after_failure_evidence_disappears(tmp_path: Path, monkeypatch):
    dead_pid = tmp_path / "dead.pid"
    _write_process_identity(dead_pid)
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
                **_process_identity(),
            }
        ],
    )
    monkeypatch.setattr(run_evidence, "process_identity_running", lambda *_args: False)

    hparam_runtime.monitor_hparam_runs(tmp_path)
    assert _read_table(tmp_path / "run_status.tsv")[0]["status"] == "failed"
    fail_log.write_text("")

    hparam_runtime.monitor_hparam_runs(tmp_path)

    assert _read_table(tmp_path / "run_status.tsv")[0]["status"] == "failed"


def test_hparam_monitor_never_launches_pending_runs(tmp_path: Path, monkeypatch):
    dead_pid = tmp_path / "dead.pid"
    _write_process_identity(dead_pid)
    _write_runtime_rows(
        tmp_path,
        [
            {
                "run_id": "run-000",
                "version": "v0",
                "pid_path": str(dead_pid),
                "status": "launched",
                "launched_at": "2026-01-01T00:00:00Z",
                **_process_identity(),
            },
            {"run_id": "run-001", "version": "v1", "status": "pending"},
        ],
    )
    started = []

    def fake_start(_execution, command):
        started.append(command)
        return "launched"

    monkeypatch.setattr(hparam_runtime, "_start_process", fake_start)
    monkeypatch.setattr(run_evidence, "process_identity_running", lambda *_args: False)
    monkeypatch.setattr(hparam_runtime.time, "sleep", lambda *_args: pytest.fail("once monitor must not sleep"))
    monkeypatch.setattr(hparam_runtime, "launch_hparam_runs", lambda *_args, **_kwargs: pytest.fail("no launch"))
    monkeypatch.setattr(hparam_runtime.scheduler.slurm, "submit", lambda *_args, **_kwargs: pytest.fail("no submit"))

    monitor_hparam_runs(tmp_path, once=True, poll_seconds=60)

    status = {row["run_id"]: row for row in _read_table(tmp_path / "run_status.tsv")}
    manifest = {row["run_id"]: row for row in _read_table(tmp_path / "launch_manifest.tsv")}
    assert started == []
    assert status["run-000"]["status"] == "finished"
    assert status["run-001"]["status"] == "pending"
    assert manifest["run-001"]["status"] == "pending"
    assert not manifest["run-001"]["launched_at"]


def test_continuous_hparam_monitor_never_launches_pending_runs(tmp_path: Path, monkeypatch):
    _write_runtime_rows(tmp_path, [{"run_id": "run-000", "status": "pending"}])

    class StopPolling(Exception):
        pass

    def stop_polling(seconds):
        assert seconds == 60
        raise StopPolling

    monkeypatch.setattr(hparam_runtime.time, "sleep", stop_polling)
    monkeypatch.setattr(hparam_runtime, "_start_process", lambda *_args, **_kwargs: pytest.fail("no start"))
    monkeypatch.setattr(hparam_runtime, "launch_hparam_runs", lambda *_args, **_kwargs: pytest.fail("no launch"))
    monkeypatch.setattr(hparam_runtime.scheduler.slurm, "submit", lambda *_args, **_kwargs: pytest.fail("no submit"))

    with pytest.raises(StopPolling):
        monitor_hparam_runs(tmp_path, once=False, poll_seconds=60)

    assert _read_table(tmp_path / "run_manifest.tsv")[0]["status"] == "pending"
    assert _read_table(tmp_path / "run_status.tsv")[0]["status"] == "pending"
    launch_row = _read_table(tmp_path / "launch_manifest.tsv")[0]
    assert launch_row["status"] == "pending"
    assert not launch_row["launched_at"]


def test_hparam_monitor_health_is_opt_in(tmp_path: Path, monkeypatch):
    pid_path = tmp_path / "running.pid"
    _write_process_identity(pid_path)
    _write_runtime_rows(
        tmp_path,
        [
            {
                "run_id": "running",
                "version": "v1",
                "pid_path": str(pid_path),
                "status": "launched",
                **_process_identity(),
            }
        ],
    )
    monkeypatch.setattr(run_evidence, "process_identity_running", lambda _row, _identity: True)

    monitor_hparam_runs(tmp_path)

    row = _read_table(tmp_path / "run_status.tsv")[0]
    assert "health_status" not in row


def test_hparam_status_preserves_terminal_state_with_live_pid(tmp_path: Path, monkeypatch):
    pid_path = tmp_path / "stopped.pid"
    pid_path.write_text("123")
    monkeypatch.setattr(run_evidence, "process_running", lambda row, pid: True)

    row = run_evidence.status_row(
        tmp_path,
        {"status": "stopped", "pid_path": str(pid_path)},
        script_commits_terminal_status=False,
    )

    assert row["status"] == "stopped"


@pytest.mark.parametrize("pid_text", ["", "not-a-pid"])
def test_hparam_status_does_not_infer_terminal_from_corrupt_local_pid(tmp_path: Path, pid_text: str):
    pid_path = tmp_path / "running.pid"
    pid_path.write_text(pid_text)
    previous = {"status": "running", "pid_path": str(pid_path)}

    row = run_evidence.status_row(tmp_path, previous, previous, script_commits_terminal_status=False)

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


def test_gpu_summary_matches_managed_process_group_children(monkeypatch):
    def fake_command(_row, command):
        if "--query-compute-apps" in command:
            return subprocess.CompletedProcess(
                [],
                0,
                "456, GPU-managed, 1024\n789, GPU-foreign, 2048\n",
                "",
            )
        if command == "ps -eo pid=,pgid=":
            return subprocess.CompletedProcess([], 0, "123 123\n456 123\n789 789\n", "")
        if "--query-gpu" in command:
            return subprocess.CompletedProcess([], 0, "0, 90, 4096\n", "")
        raise AssertionError(command)

    monkeypatch.setattr(run_evidence, "run_row_command", fake_command)

    summary = run_evidence.gpu_summary({"gpus": "0", "process_group_id": 123}, 123)

    assert "456, GPU-managed, 1024" in summary
    assert "789, GPU-foreign, 2048" not in summary


def test_gpu_summary_preserves_process_group_probe_uncertainty(monkeypatch):
    def fake_command(_row, command):
        if "--query-compute-apps" in command:
            return subprocess.CompletedProcess([], 0, "456, GPU-managed, 1024\n", "")
        if command == "ps -eo pid=,pgid=":
            return subprocess.CompletedProcess([], 1, "", "ps failed")
        raise AssertionError(command)

    monkeypatch.setattr(run_evidence, "run_row_command", fake_command)

    assert run_evidence.gpu_summary({"gpus": "0", "process_group_id": 123}, 123) is None


def test_hparam_monitor_health_classifies_compute_active(tmp_path: Path, monkeypatch):
    pid_path = tmp_path / "running.pid"
    _write_process_identity(pid_path)
    _write_runtime_rows(
        tmp_path,
        [
            {
                "run_id": "running",
                "version": "v1",
                "pid_path": str(pid_path),
                "status": "launched",
                **_process_identity(),
            }
        ],
    )
    monkeypatch.setattr(run_evidence, "process_identity_running", lambda _row, _identity: True)
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
    _write_process_identity(pid_path)
    rows = _write_runtime_rows(
        tmp_path,
        [
            {
                "run_id": "running",
                "version": "v1",
                "pid_path": str(pid_path),
                "status": "launched",
                **_process_identity(),
            }
        ],
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
    monkeypatch.setattr(run_evidence, "process_identity_running", lambda _row, _identity: True)
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
    _write_process_identity(pid_path)
    _write_runtime_rows(
        tmp_path,
        [
            {
                "run_id": "stalled",
                "version": "v1",
                "pid_path": str(pid_path),
                "status": "launched",
                **_process_identity(),
            },
            {
                "run_id": "remote",
                "version": "v2",
                "target": "ssh",
                "host": "baichuan3",
                "pid_path": str(pid_path),
                "status": "launched",
                **_process_identity(),
            },
        ],
    )
    merge_run_manifest(
        tmp_path,
        [
            {"step_id": "train-model", "run_id": "stalled", "checkpoint_count": 0},
            {"step_id": "train-model", "run_id": "remote", "checkpoint_count": 0},
        ],
    )

    def fake_running(row, _identity):
        return None if row["run_id"] == "remote" else True

    monkeypatch.setattr(
        run_evidence,
        "run_row_command",
        lambda _row, command: subprocess.CompletedProcess(command, 255, "", "unreachable test host"),
    )
    monkeypatch.setattr(run_evidence, "process_identity_running", fake_running)
    monkeypatch.setattr(run_evidence, "gpu_summary", lambda row, pid: "")
    monkeypatch.setattr(run_evidence, "proc_io", lambda row, pid: {"read_bytes": 100, "write_bytes": 50})
    monkeypatch.setattr(run_evidence, "log_age_seconds", lambda path, row: 500)
    monkeypatch.setattr(run_evidence, "read_run_progress", lambda run_dir, row: {"status": "missing"})

    monitor_hparam_runs(tmp_path, health=True)

    status = {row["run_id"]: row["health_status"] for row in _read_table(tmp_path / "run_status.tsv")}
    assert status["stalled"] == "possibly_stalled"
    assert status["remote"] == "unknown_remote"


def test_health_requires_a_comparable_checkpoint_observation():
    inputs = {
        "status": "running",
        "running_state": True,
        "gpu_summary": "",
        "io_read_delta": None,
        "io_write_delta": None,
        "progress": {"status": "missing"},
        "progress_is_fresh": False,
        "log_age_seconds": 500,
        "checkpoint_count": 0,
    }

    assert run_evidence.classify_health(**inputs, previous_checkpoint_count=None) == "health_unknown"
    assert run_evidence.classify_health(**inputs, previous_checkpoint_count=0) == "possibly_stalled"
    assert (
        run_evidence.classify_health(**{**inputs, "gpu_summary": None}, previous_checkpoint_count=0) == "health_unknown"
    )
    assert (
        run_evidence.classify_health(
            **{**inputs, "gpu_summary": "456, GPU-managed, 1024", "checkpoint_count": None},
            previous_checkpoint_count=0,
        )
        == "compute_active"
    )


def test_health_preserves_checkpoint_inventory_when_remote_probe_is_unavailable(tmp_path: Path, monkeypatch):
    identity = _process_identity()
    row = {
        "step_id": "train-model",
        "run_id": "run-000",
        "status": "running",
        "target": "ssh",
        "host": "unit-host",
        "script": "/remote/run.sh",
        "pid_path": "/remote/run.pid",
        "log_path": "/remote/run.log",
        "gpus": "0",
        "checkpoints": "epoch=1.ckpt",
        "checkpoint_count": 1,
        **identity,
    }
    monkeypatch.setattr(run_evidence, "read_process_identity", lambda *_args, **_kwargs: identity)
    monkeypatch.setattr(run_evidence, "process_identity_running", lambda *_args: True)
    monkeypatch.setattr(run_evidence, "runtime_artifacts", lambda _row: None)
    monkeypatch.setattr(run_evidence, "log_tail", lambda *_args, **_kwargs: "")
    monkeypatch.setattr(run_evidence, "gpu_summary", lambda _row, _pid: "")
    monkeypatch.setattr(run_evidence, "proc_io", lambda _row, _pid: {})
    monkeypatch.setattr(run_evidence, "log_age_seconds", lambda _path, _row: 500)
    monkeypatch.setattr(run_evidence, "read_run_progress", lambda _run_dir, _row: {"status": "missing"})

    observed = run_evidence.status_row(
        tmp_path,
        row,
        row,
        script_commits_terminal_status=False,
        health=True,
    )

    assert observed["status"] == "running"
    assert observed["checkpoints"] == "epoch=1.ckpt"
    assert observed["checkpoint_count"] == ""
    assert observed["health_status"] == "health_unknown"


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
                **_process_identity(),
            }
        ],
    )
    merge_run_manifest(tmp_path, [{"step_id": "train-model", "run_id": "run-000", "status": "running"}])
    probe = {"failure": failure}

    def fake_run(args, **kwargs):
        command = args[-1]
        if _is_remote_python_program(command, "run_evidence.runtime_artifacts"):
            return subprocess.CompletedProcess(args, 0, '{"run_manifest": "", "checkpoints": []}', "")
        if _is_remote_python_program(command, "run_evidence.read_pid_text"):
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
            return subprocess.CompletedProcess(args, 0, json.dumps(_process_identity()) + "\n", "")
        if _is_remote_python_program(command, "run_evidence.process_probe"):
            if probe["failure"] == "ps_error":
                return subprocess.CompletedProcess(args, 2, "", "ps failed")
            return subprocess.CompletedProcess(
                args,
                0,
                json.dumps({"leader": _process_identity(), "group_running": True}) + "\n",
                "",
            )
        return subprocess.CompletedProcess(args, 44, "", "missing")

    monkeypatch.setattr(run_evidence.subprocess, "run", fake_run)

    hparam_runtime.monitor_hparam_runs(tmp_path)
    first = _read_table(tmp_path / "run_status.tsv")[0]
    assert first["status"] == "unknown_remote"
    assert first["pid"] == "123"
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
def test_hparam_monitor_requires_certain_remote_process_absence_before_clean_finish(
    tmp_path: Path, monkeypatch, uncertain_returncode: int
):
    _write_runtime_rows(
        tmp_path,
        [
            {
                "run_id": "run-000",
                "target": "ssh",
                "host": "unit-host",
                "status": "running",
                **_process_identity(),
            }
        ],
    )
    state = {"uncertain": True}

    def fake_remote_command(_row, command):
        if _is_remote_python_program(command, "run_evidence.read_pid_text"):
            return subprocess.CompletedProcess([], 0, json.dumps(_process_identity()) + "\n", "")
        if _is_remote_python_program(command, "run_evidence.process_probe"):
            if state["uncertain"]:
                return subprocess.CompletedProcess([], uncertain_returncode, "", "permission or transport failure")
            return subprocess.CompletedProcess([], 0, '{"leader": null, "group_running": false}\n', "")
        if _is_remote_python_program(command, "run_evidence.runtime_artifacts"):
            return subprocess.CompletedProcess([], 0, '{"run_manifest": "", "checkpoints": []}', "")
        if _is_remote_python_program(command, "run_evidence.log_tail"):
            return subprocess.CompletedProcess([], 0, "", "")
        if command.startswith("tail -n 8"):
            return subprocess.CompletedProcess([], 0, "", "")
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

    assert _read_table(tmp_path / "run_status.tsv")[0]["status"] == "finished"
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert [(event["from"], event["to"]) for event in events] == [
        ("running", "unknown_remote"),
        ("unknown_remote", "finished"),
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
    assert commands == [transport.remote_python_program_command("run_evidence.read_pid_text", "/remote/run.pid")]


def test_hparam_monitor_health_requires_fresh_progress(tmp_path: Path, monkeypatch):
    pid_path = tmp_path / "running.pid"
    _write_process_identity(pid_path)
    rows = _write_runtime_rows(
        tmp_path,
        [
            {
                "run_id": "running",
                "version": "v1",
                "pid_path": str(pid_path),
                "status": "launched",
                **_process_identity(),
            }
        ],
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
    merge_run_manifest(
        tmp_path,
        [
            {
                "step_id": "train-model",
                "run_id": "running",
                "progress_processed": 5,
                "progress_updated_at": "2000-01-01T00:00:00Z",
                "checkpoint_count": 0,
            }
        ],
    )
    monkeypatch.setattr(run_evidence, "process_identity_running", lambda _row, _identity: True)
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
