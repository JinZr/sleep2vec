from __future__ import annotations

import json
from pathlib import Path
import subprocess

from agent_tool_test_helpers import FakeLauncher
import pytest
from test_agent_tools_hparam_runtime import _process_identity, _read_table, _write_process_identity, _write_runtime_rows
from test_agent_tools_hparam_runtime import _stub_execution_snapshot_preflight  # noqa: F401

from agent_tools import experiments, hparam_runtime, managed_scheduler, manifests, run_evidence
from agent_tools.experiment_workspace import merge_run_manifest


@pytest.mark.parametrize(
    "failure",
    ["runtime_dir_file", "symlink", "dangling_symlink", "directory", "bad_encoding", "bad_json", "non_object"],
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
    elif failure == "non_object":
        manifest.write_text("[]")
    else:
        manifest.write_text("{")
    monkeypatch.setattr(run_evidence, "process_running", lambda *_args: True)

    with pytest.raises((ValueError, UnicodeError), match="run manifest") as excinfo:
        run_evidence.status_row(tmp_path, rows[0], rows[0], script_commits_terminal_status=False)
    if failure == "non_object":
        assert str(excinfo.value) == f"Runtime run manifest is corrupt: {manifest}"


def test_find_run_manifest_distinguishes_missing_and_valid_regular_file(tmp_path: Path):
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()
    run = {"runtime_dir": str(runtime_dir)}

    assert run_evidence.find_run_manifest(run) is None

    manifest = runtime_dir / "run_manifest.json"
    manifest.write_text(json.dumps({"metrics": {"val_ahi_pearson": 0.7}}))

    assert run_evidence.find_run_manifest(run) == manifest


def test_experiment_monitor_handles_running_missing_and_failed_hparam_rows(tmp_path: Path, monkeypatch):
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

    experiments.monitor_experiment(tmp_path)

    status = {row["run_id"]: row["status"] for row in _read_table(tmp_path / "run_manifest.tsv")}
    assert status["running"] == "running"
    assert status["missing"] == "missing_pid"
    assert status["failed"] == "failed"
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    status_events = [event for event in events if event["event_type"] == "run_status_changed"]
    assert status_events
    assert all(event["step_id"] == "train-model" for event in status_events)
    assert {event["run_id"] for event in status_events} == {"running", "missing", "failed"}


def test_experiment_monitor_reports_the_status_committed_by_the_canonical_owner(tmp_path: Path, monkeypatch):
    _write_runtime_rows(tmp_path, [{"run_id": "run-000", "status": "running"}])
    merge_run_manifest(tmp_path, [{"step_id": "train-model", "run_id": "run-000", "status": "running"}])
    real_merge = merge_run_manifest

    def merge_after_wandb_update(root, rows, **_kwargs):
        real_merge(root, [{"step_id": "train-model", "run_id": "run-000", "status": "failed"}])
        return real_merge(root, rows)

    monkeypatch.setattr(experiments, "merge_run_manifest", merge_after_wandb_update)
    monkeypatch.setattr(
        run_evidence,
        "status_row",
        lambda _root, observation, _prior, *, script_commits_terminal_status, health: {
            **observation,
            "status": "finished",
        },
    )

    result = experiments.monitor_experiment(tmp_path)

    assert _read_table(tmp_path / "run_manifest.tsv")[0]["status"] == "failed"
    assert result["runs"][0]["status"] == "failed"
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    status_event = next(event for event in events if event["event_type"] == "run_status_changed")
    assert status_event["from"] == "running"
    assert status_event["to"] == "failed"


def test_experiment_monitor_marks_clean_hparam_exit_completed_once(tmp_path: Path, monkeypatch):
    rows = _write_runtime_rows(
        tmp_path,
        [{"run_id": "run-000", "status": "running", **_process_identity()}],
    )
    _write_process_identity(rows[0]["pid_path"])
    monkeypatch.setattr(run_evidence, "process_identity_running", lambda *_args: False)
    workspace_rows = _read_table(tmp_path / "run_manifest.tsv")
    workspace_rows[0]["status"] = "running"
    manifests.write_rows(tmp_path / "run_manifest.tsv", workspace_rows)

    experiments.monitor_experiment(tmp_path)
    experiments.monitor_experiment(tmp_path)

    assert _read_table(tmp_path / "run_manifest.tsv")[0]["status"] == "completed"
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert [event["event_type"] for event in events].count("run_status_changed") == 1


def test_experiment_monitor_keeps_failed_status_after_failure_evidence_disappears(tmp_path: Path, monkeypatch):
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

    experiments.monitor_experiment(tmp_path)
    assert _read_table(tmp_path / "run_manifest.tsv")[0]["status"] == "failed"
    fail_log.write_text("")

    experiments.monitor_experiment(tmp_path)

    assert _read_table(tmp_path / "run_manifest.tsv")[0]["status"] == "failed"


def test_experiment_monitor_never_launches_pending_hparam_runs(tmp_path: Path, monkeypatch):
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

    monkeypatch.setattr(managed_scheduler, "start_process", fake_start)
    monkeypatch.setattr(run_evidence, "process_identity_running", lambda *_args: False)
    monkeypatch.setattr(hparam_runtime, "launch_hparam_runs", lambda *_args, **_kwargs: pytest.fail("no launch"))
    monkeypatch.setattr(hparam_runtime.scheduler.slurm, "submit", lambda *_args, **_kwargs: pytest.fail("no submit"))

    experiments.monitor_experiment(tmp_path)

    status = {row["run_id"]: row for row in _read_table(tmp_path / "run_manifest.tsv")}
    assert started == []
    assert status["run-000"]["status"] == "completed"
    assert status["run-001"]["status"] == "pending"
    assert not status["run-001"]["launched_at"]


def test_hparam_status_preserves_terminal_state_with_live_pid(tmp_path: Path, monkeypatch):
    pid_path = tmp_path / "stopped.pid"
    pid_path.write_text("123")
    monkeypatch.setattr(run_evidence, "process_running", lambda _pid: True)

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
def test_hparam_launch_does_not_start_when_local_pid_is_corrupt(tmp_path: Path, status: str, pid_text: str):
    rows = _write_runtime_rows(tmp_path, [{"run_id": "run-000", "status": status}])
    Path(rows[0]["pid_path"]).write_text(pid_text)
    launcher = FakeLauncher(verify_target=False)

    hparam_runtime.launch_hparam_runs(tmp_path, dry_run=False, hooks=launcher.hooks())

    launcher.assert_not_started()
    assert _read_table(tmp_path / "run_manifest.tsv")[0]["status"] == "missing_pid"


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

    launcher = FakeLauncher(verify_target=False)
    monkeypatch.setattr(Path, "read_text", fail_pid_read)

    with pytest.raises(RuntimeError, match="PID file read failed"):
        hparam_runtime.launch_hparam_runs(tmp_path, dry_run=False, hooks=launcher.hooks())

    launcher.assert_not_started()
    assert _read_table(tmp_path / "run_manifest.tsv")[0]["status"] == status

    read_fails["value"] = False
    pid_path.unlink()
    hparam_runtime.launch_hparam_runs(tmp_path, dry_run=False, hooks=launcher.hooks())

    assert len(launcher.starts) == 1
    assert _read_table(tmp_path / "run_manifest.tsv")[0]["status"] == "launched"


@pytest.mark.parametrize("failure", ["directory", "invalid_utf8", "os_error", "dangling_symlink"])
def test_experiment_monitor_preserves_nonterminal_status_for_unreadable_local_pid(
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

    experiments.monitor_experiment(tmp_path)

    assert _read_table(tmp_path / "run_manifest.tsv")[0]["status"] == "running"


def test_gpu_summary_matches_managed_process_group_children(monkeypatch):
    def fake_command(command):
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
    def fake_command(command):
        if "--query-compute-apps" in command:
            return subprocess.CompletedProcess([], 0, "456, GPU-managed, 1024\n", "")
        if command == "ps -eo pid=,pgid=":
            return subprocess.CompletedProcess([], 1, "", "ps failed")
        raise AssertionError(command)

    monkeypatch.setattr(run_evidence, "run_row_command", fake_command)

    assert run_evidence.gpu_summary({"gpus": "0", "process_group_id": 123}, 123) is None


def test_experiment_monitor_health_classifies_compute_active(tmp_path: Path, monkeypatch):
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
    monkeypatch.setattr(run_evidence, "proc_io", lambda _pid: {})
    monkeypatch.setattr(run_evidence, "log_age_seconds", lambda _path: None)
    monkeypatch.setattr(run_evidence, "read_run_progress", lambda run_dir, row: {"status": "missing"})

    row = experiments.monitor_experiment(tmp_path)["runs"][0]
    assert row["health_status"] == "compute_active"
    assert row["gpu_summary"] == "123, GPU-1, 1024"


def test_experiment_monitor_health_classifies_data_loading_from_io_delta(tmp_path: Path, monkeypatch):
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
    monkeypatch.setattr(run_evidence, "proc_io", lambda _pid: {"read_bytes": 250, "write_bytes": 50})
    monkeypatch.setattr(run_evidence, "log_age_seconds", lambda _path: None)
    monkeypatch.setattr(run_evidence, "read_run_progress", lambda run_dir, row: {"status": "missing"})

    row = experiments.monitor_experiment(tmp_path)["runs"][0]
    assert row["health_status"] == "data_loading"
    assert row["io_read_delta_bytes"] == "150"


def test_experiment_monitor_health_classifies_stalled(tmp_path: Path, monkeypatch):
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
        ],
    )
    merge_run_manifest(
        tmp_path,
        [
            {"step_id": "train-model", "run_id": "stalled", "checkpoint_count": 0},
        ],
    )

    monkeypatch.setattr(run_evidence, "process_identity_running", lambda _row, _identity: True)
    monkeypatch.setattr(run_evidence, "gpu_summary", lambda row, pid: "")
    monkeypatch.setattr(run_evidence, "proc_io", lambda _pid: {"read_bytes": 100, "write_bytes": 50})
    monkeypatch.setattr(run_evidence, "log_age_seconds", lambda _path: 500)
    monkeypatch.setattr(run_evidence, "read_run_progress", lambda run_dir, row: {"status": "missing"})

    status = {row["run_id"]: row["health_status"] for row in experiments.monitor_experiment(tmp_path)["runs"]}
    assert status == {"stalled": "possibly_stalled"}


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


def test_process_identity_accepts_legacy_receipt_without_runtime_commit(tmp_path: Path):
    pid_path = tmp_path / "legacy.pid"
    legacy_identity = {"pid": 123, "process_group_id": 123, "process_start_token": "proc:unit-start"}
    pid_path.write_text(json.dumps(legacy_identity) + "\n")

    assert run_evidence.read_process_identity(pid_path) == legacy_identity


def test_process_identity_reads_receipt_with_runtime_commit(tmp_path: Path):
    pid_path = tmp_path / "current.pid"
    identity = {
        "pid": 123,
        "process_group_id": 123,
        "process_start_token": "proc:unit-start",
        "runtime_commit": "b" * 40,
    }
    pid_path.write_text(json.dumps(identity) + "\n")

    assert run_evidence.read_process_identity(pid_path) == identity


@pytest.mark.parametrize(
    "runtime_commit",
    ["", None, "a" * 39, "a" * 41, "a" * 64, "A" * 40, "g" * 40, 123],
)
def test_process_identity_rejects_malformed_runtime_commit(tmp_path: Path, runtime_commit: object):
    pid_path = tmp_path / "malformed.pid"
    pid_path.write_text(
        json.dumps(
            {
                "pid": 123,
                "process_group_id": 123,
                "process_start_token": "proc:unit-start",
                "runtime_commit": runtime_commit,
            }
        )
        + "\n"
    )

    with pytest.raises(run_evidence.ProcessIdentityError, match="invalid runtime commit"):
        run_evidence.read_process_identity(pid_path)


def test_experiment_monitor_health_requires_fresh_progress(tmp_path: Path, monkeypatch):
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
    monkeypatch.setattr(run_evidence, "proc_io", lambda _pid: {})
    monkeypatch.setattr(run_evidence, "log_age_seconds", lambda _path: 500)
    monkeypatch.setattr(
        run_evidence,
        "read_run_progress",
        lambda run_dir, row: {
            "status": "running",
            "processed": 5,
            "updated_at": "2000-01-01T00:00:00Z",
        },
    )

    row = experiments.monitor_experiment(tmp_path)["runs"][0]
    assert row["health_status"] == "possibly_stalled"


def test_run_row_command_timeout_returns_timeout_returncode(monkeypatch):
    def fake_run(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(["bash", "-lc", "ps"], 10)

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = run_evidence.run_row_command("ps")

    assert result.returncode == 124


@pytest.mark.parametrize("observed", [False, True])
def test_health_fields_distinguish_missing_observations_from_zero(tmp_path, monkeypatch, observed):
    progress = {"status": None, "processed": "003", "total": None, "updated_at": ""}
    monkeypatch.setattr(run_evidence, "read_run_progress", lambda *_args: progress)
    monkeypatch.setattr(run_evidence, "proc_io", lambda *_args: {"read_bytes": 0, "write_bytes": 0} if observed else {})
    monkeypatch.setattr(run_evidence, "log_age_seconds", lambda *_args: 0 if observed else None)
    monkeypatch.setattr(run_evidence, "gpu_summary", lambda *_args: "" if observed else None)

    result = run_evidence.health_fields(
        tmp_path,
        {},
        {"io_read_bytes": "0", "io_write_bytes": "0", "checkpoint_count": "0"},
        123,
        True,
        "running",
        [] if observed else None,
    )

    expected = 0 if observed else ""
    for field in (
        "io_read_bytes",
        "io_write_bytes",
        "io_read_delta_bytes",
        "io_write_delta_bytes",
        "log_age_seconds",
        "checkpoint_count",
    ):
        assert result[field] == expected
    assert result["health_status"] == ("healthy_running" if observed else "health_unknown")
    assert result["gpu_summary"] == ""
    assert result["progress_status"] is None
    assert result["progress_processed"] == "003"
    assert result["progress_total"] is None
    assert result["progress_updated_at"] == ""
    assert result["progress_age_seconds"] is None
