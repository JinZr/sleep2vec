from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import threading

import pytest
from test_agent_tools_hparam_runtime import (
    _embedded_process_group_running,
    _is_remote_python_program,
    _process_identity,
    _read_table,
    _write_proc_stat,
    _write_process_identity,
    _write_runtime_rows,
)
from test_agent_tools_hparam_runtime import _stub_execution_snapshot_preflight  # noqa: F401

from agent_tools import hparam_runtime, manifests, run_evidence, transport
from agent_tools.experiment_workspace import MONITOR_EXIT_CODE_PREFIX, merge_run_manifest


def test_remote_stop_failure_does_not_commit_stopped_state(tmp_path: Path, monkeypatch):
    rows = _write_runtime_rows(
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
    before_launch = (tmp_path / "launch_manifest.tsv").read_bytes()
    before_status = (tmp_path / "run_status.tsv").read_bytes()
    real_validate = hparam_runtime.exp_io.validate_managed_output_paths
    monkeypatch.setattr(
        hparam_runtime.exp_io,
        "validate_managed_output_paths",
        lambda root, paths, remote=None: None if remote else real_validate(root, paths),
    )
    monkeypatch.setattr(run_evidence, "read_process_identity", lambda _path, _row: _process_identity())
    monkeypatch.setattr(
        run_evidence,
        "stop_process_group",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("Failed to stop remote process group")),
    )

    with pytest.raises(RuntimeError, match="Failed to stop remote process group"):
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
    real_validate = hparam_runtime.exp_io.validate_managed_output_paths
    monkeypatch.setattr(
        hparam_runtime.exp_io,
        "validate_managed_output_paths",
        lambda root, paths, remote=None: None if remote else real_validate(root, paths),
    )

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


@pytest.mark.parametrize("failure", ["directory", "symlink", "dangling_symlink", "hardlink", "fifo"])
def test_hparam_stop_rejects_unsafe_local_pid_topology_before_read_or_kill(tmp_path: Path, monkeypatch, failure: str):
    rows = _write_runtime_rows(tmp_path, [{"run_id": "run-000", "status": "running"}])
    pid_path = Path(rows[0]["pid_path"])
    target = tmp_path / "outside.pid"
    if failure == "directory":
        pid_path.mkdir()
    elif failure == "symlink":
        target.write_text("123")
        pid_path.symlink_to(target)
    elif failure == "dangling_symlink":
        pid_path.symlink_to(target)
    elif failure == "hardlink":
        target.write_text("123")
        pid_path.hardlink_to(target)
    else:
        os.mkfifo(pid_path)
    calls = []
    monkeypatch.setattr(run_evidence, "read_process_identity", lambda *_args, **_kwargs: calls.append("read"))
    monkeypatch.setattr(run_evidence.os, "kill", lambda *_args: calls.append("kill"))

    with pytest.raises(ValueError, match="Managed output"):
        hparam_runtime.stop_hparam_run(tmp_path, "run-000", reason="unsafe PID topology")

    assert calls == []


def test_hparam_stop_rejects_unsafe_remote_pid_topology_before_read_or_signal(tmp_path: Path, monkeypatch):
    _write_runtime_rows(
        tmp_path,
        [{"run_id": "run-000", "target": "ssh", "host": "unit-host", "status": "running"}],
    )
    calls = []

    def reject_remote(_root, _paths, *, remote=None):
        if remote:
            calls.append(("preflight", remote))
            raise ValueError("Managed output paths must be independent regular files: /remote/run.pid")

    monkeypatch.setattr(hparam_runtime.exp_io, "validate_managed_output_paths", reject_remote)
    monkeypatch.setattr(run_evidence, "read_process_identity", lambda *_args, **_kwargs: calls.append("read"))
    monkeypatch.setattr(run_evidence, "stop_process_group", lambda *_args, **_kwargs: calls.append("signal"))

    with pytest.raises(ValueError, match="Managed output"):
        hparam_runtime.stop_hparam_run(tmp_path, "run-000", reason="unsafe remote PID topology")

    assert calls == [("preflight", "unit-host")]


@pytest.mark.parametrize(
    ("target", "host", "message"),
    [
        pytest.param("cluster", "unit-host", "target must be local or ssh", id="unknown-target"),
        pytest.param("ssh", "", "requires a non-empty host", id="missing-ssh-host"),
        pytest.param("ssh", "   ", "requires a non-empty host", id="blank-ssh-host"),
    ],
)
def test_hparam_stop_rejects_invalid_transport_before_pid_read_or_signal(
    tmp_path: Path,
    monkeypatch,
    target: str,
    host: str,
    message: str,
):
    rows = _write_runtime_rows(
        tmp_path,
        [
            {
                "run_id": "run-000",
                "target": target,
                "host": host,
                "status": "running",
                **_process_identity(),
            }
        ],
    )
    _write_process_identity(rows[0]["pid_path"])
    before = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    calls = []
    monkeypatch.setattr(run_evidence, "read_process_identity", lambda *_args, **_kwargs: calls.append("read"))
    monkeypatch.setattr(run_evidence, "stop_process_group", lambda *_args, **_kwargs: calls.append("signal"))

    with pytest.raises(ValueError, match=message):
        hparam_runtime.stop_hparam_run(tmp_path, "run-000", reason="invalid transport")

    assert calls == []
    assert {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before


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

    def matched_script(_row, command):
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
        lambda _row, _command: subprocess.CompletedProcess(
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


def test_hparam_stop_rejects_unbound_remote_first_process_identity_before_signal_or_commit(
    tmp_path: Path,
    monkeypatch,
):
    _write_runtime_rows(
        tmp_path,
        [{"run_id": "run-000", "target": "ssh", "host": "unit-host", "status": "running"}],
    )
    before = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    real_validate = hparam_runtime.exp_io.validate_managed_output_paths
    monkeypatch.setattr(
        hparam_runtime.exp_io,
        "validate_managed_output_paths",
        lambda root, paths, remote=None: None if remote else real_validate(root, paths),
    )
    commands = []

    def remote_evidence(_row, command):
        commands.append(command)
        if _is_remote_python_program(command, "run_evidence.read_pid_text"):
            return subprocess.CompletedProcess([], 0, json.dumps(_process_identity()) + "\n", "")
        if command == "ps -ww -p 123 -o args=":
            return subprocess.CompletedProcess([], 0, "python unrelated_job.py\n", "")
        raise AssertionError(command)

    monkeypatch.setattr(run_evidence, "run_row_command", remote_evidence)
    calls = []
    monkeypatch.setattr(run_evidence, "stop_process_group", lambda *_args: calls.append("signal"))
    monkeypatch.setattr(hparam_runtime, "merge_run_manifest", lambda *_args: calls.append("commit"))

    with pytest.raises(run_evidence.ProcessIdentityError, match="does not match frozen script"):
        hparam_runtime.stop_hparam_run(tmp_path, "run-000", reason="unbound remote identity")

    assert len(commands) == 2
    assert commands[1] == "ps -ww -p 123 -o args="
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


def test_remote_stop_verifies_and_signals_in_one_command(monkeypatch):
    identity = _process_identity()
    commands = []
    monkeypatch.setattr(
        run_evidence,
        "run_row_command",
        lambda _row, command: commands.append(command) or subprocess.CompletedProcess([], 0, "", ""),
    )

    run_evidence.stop_process_group({"target": "ssh", "host": "unit-host"}, identity)

    assert len(commands) == 1
    assert commands[0] == transport.remote_python_program_command(
        "run_evidence.process_stop",
        identity["pid"],
        identity["process_group_id"],
        identity["process_start_token"],
        5.0,
    )


@pytest.mark.parametrize(
    ("leader_state", "live_child", "expected"),
    [
        pytest.param("Z", False, False, id="zombie-only"),
        pytest.param("X", False, False, id="dead-only"),
        pytest.param("Z", True, True, id="live-child"),
    ],
)
def test_embedded_process_group_probe_requires_a_non_zombie_member(
    tmp_path: Path, leader_state: str, live_child: bool, expected: bool
):
    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    _write_proc_stat(proc_root, 123, 123, leader_state)
    if live_child:
        _write_proc_stat(proc_root, 124, 123, "S")
    _write_proc_stat(proc_root, 200, 200, "S")

    assert _embedded_process_group_running(proc_root, 123) is expected


def test_embedded_process_group_probe_uses_the_live_leader_fast_path(tmp_path: Path, monkeypatch):
    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    _write_proc_stat(proc_root, 123, 123, "S")
    monkeypatch.setattr(
        run_evidence.os,
        "listdir",
        lambda *_args: (_ for _ in ()).throw(AssertionError("live leader should not scan /proc")),
    )

    assert _embedded_process_group_running(proc_root, 123) is True


def test_embedded_process_group_probe_preserves_uncertainty_from_unreadable_proc_entries(tmp_path: Path):
    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    _write_proc_stat(proc_root, 123, 123, "Z")
    (proc_root / "124" / "stat").mkdir(parents=True)

    assert _embedded_process_group_running(proc_root, 123) is None


@pytest.mark.parametrize("group_running", [False, True], ids=["zombie-only", "live-child"])
def test_process_monitor_uses_group_liveness_for_a_matching_leader(monkeypatch, group_running: bool):
    identity = _process_identity()
    monkeypatch.setattr(
        run_evidence,
        "run_row_command",
        lambda *_args: subprocess.CompletedProcess(
            [],
            0,
            json.dumps({"leader": identity, "group_running": group_running}) + "\n",
            "",
        ),
    )

    assert run_evidence.process_identity_running({}, identity) is group_running


def test_process_monitor_rejects_a_reused_pid_start_token(monkeypatch):
    identity = _process_identity()
    reused = {**identity, "process_start_token": "proc:other-start"}
    monkeypatch.setattr(
        run_evidence,
        "run_row_command",
        lambda *_args: subprocess.CompletedProcess(
            [],
            0,
            json.dumps({"leader": reused, "group_running": True}) + "\n",
            "",
        ),
    )

    with pytest.raises(RuntimeError, match="was reused"):
        run_evidence.process_identity_running({}, identity)


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
        lambda _row, command: (
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


@pytest.mark.parametrize(
    ("target", "host", "expected_status"),
    [
        pytest.param("local", "", "missing_pid", id="local"),
        pytest.param("ssh", "unit-host", "unknown_remote", id="ssh"),
    ],
)
def test_status_keeps_dead_unfrozen_process_identity_unbound(
    tmp_path: Path,
    monkeypatch,
    target: str,
    host: str,
    expected_status: str,
):
    script = tmp_path / "launch.sh"
    script.write_text("#!/usr/bin/env bash\ntrue\n")
    row = {
        "target": target,
        "host": host,
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
    monkeypatch.setattr(run_evidence, "runtime_artifacts", lambda *_args: None)
    monkeypatch.setattr(run_evidence, "log_tail", lambda *_args: "")

    observed = run_evidence.status_row(tmp_path, row, row, script_commits_terminal_status=False)

    assert observed["status"] == expected_status
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
    monkeypatch.setattr(hparam_runtime, "launch_hparam_runs", lambda *_args, **_kwargs: pytest.fail("no launch"))
    monkeypatch.setattr(hparam_runtime.time, "sleep", lambda *_args: pytest.fail("no sleep"))

    with pytest.raises(RuntimeError, match="cannot advance.*missing_pid"):
        hparam_runtime.run_hparam_queue(tmp_path, dry_run=False)

    assert _read_table(tmp_path / "run_manifest.tsv")[0]["status"] == "missing_pid"
    assert _read_table(tmp_path / "run_status.tsv")[0]["status"] == "missing_pid"


def test_status_marks_a_zombie_only_managed_process_finished(tmp_path: Path, monkeypatch):
    pid_path = tmp_path / "managed.pid"
    identity = _write_process_identity(pid_path)
    row = {
        "script": str(tmp_path / "launch.sh"),
        "pid_path": str(pid_path),
        "status": "running",
        **identity,
    }
    monkeypatch.setattr(
        run_evidence,
        "run_row_command",
        lambda *_args: subprocess.CompletedProcess(
            [],
            0,
            json.dumps({"leader": identity, "group_running": False}) + "\n",
            "",
        ),
    )

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
    monkeypatch.setattr(
        run_evidence,
        "run_row_command",
        lambda *_args: subprocess.CompletedProcess(
            [],
            0,
            json.dumps({"leader": reused, "group_running": True}) + "\n",
            "",
        ),
    )

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

    assert run_evidence.log_has_failure(log_path) is expected


def test_status_preserves_remote_uncertainty_when_monitor_exit_log_is_unreadable(tmp_path: Path, monkeypatch):
    identity = _process_identity()
    row = {
        "target": "ssh",
        "host": "unit-host",
        "script": str(tmp_path / "launch.sh"),
        "pid_path": str(tmp_path / "managed.pid"),
        "log_path": str(tmp_path / "managed.log"),
        "status": "running",
        "terminal_status_owner": "monitor",
        **identity,
    }
    monkeypatch.setattr(run_evidence, "read_process_identity", lambda *_args: identity)
    monkeypatch.setattr(run_evidence, "process_identity_running", lambda *_args: False)
    monkeypatch.setattr(run_evidence, "runtime_artifacts", lambda *_args: None)
    monkeypatch.setattr(run_evidence, "log_tail", lambda *_args: "")
    monkeypatch.setattr(
        run_evidence,
        "run_row_command",
        lambda *_args: subprocess.CompletedProcess([], 255, "", "connection lost"),
    )

    observed = run_evidence.status_row(tmp_path, row, row, script_commits_terminal_status=False)

    assert observed["status"] == "unknown_remote"


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
    monkeypatch.setattr(run_evidence.os, "kill", lambda *_args: calls.append("kill"))
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
    monkeypatch.setattr(run_evidence.os, "kill", lambda *_args: killed.append(True))

    with pytest.raises(ValueError, match="Managed output"):
        hparam_runtime.stop_hparam_run(tmp_path, "run-000", reason="dangling status")

    assert killed == []
    assert not missing_target.exists()
    assert {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before


def test_hparam_stop_commits_one_final_row_to_all_manifests(tmp_path: Path, monkeypatch):
    rows = _write_runtime_rows(
        tmp_path,
        [{"run_id": "run-000", "status": "running", **_process_identity()}],
    )
    _write_process_identity(rows[0]["pid_path"])
    stopped = []
    monkeypatch.setattr(run_evidence, "stop_process_group", lambda _row, identity: stopped.append(identity))

    hparam_runtime.stop_hparam_run(tmp_path, "run-000", reason="manual stop")

    canonical = _read_table(tmp_path / "run_manifest.tsv")[0]
    local = _read_table(tmp_path / "run_status.tsv")[0]
    launch = _read_table(tmp_path / "launch_manifest.tsv")[0]
    assert canonical == local == launch
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


def test_hparam_stop_rejects_invalid_canonical_output_before_kill(tmp_path: Path, monkeypatch):
    _write_runtime_rows(tmp_path, [{"run_id": "run-000", "status": "running"}])
    target = tmp_path / "run_matrix.csv"
    target.hardlink_to(tmp_path / "run_manifest.tsv")
    before = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    killed = []
    monkeypatch.setattr(run_evidence, "read_pid", lambda _path, _row, **_kwargs: 123)
    monkeypatch.setattr(run_evidence.os, "kill", lambda pid, sig: killed.append((pid, sig)))

    with pytest.raises(ValueError, match="Managed file is missing or aliased"):
        hparam_runtime.stop_hparam_run(tmp_path, "run-000", reason="manual stop")

    assert killed == []
    assert {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before
    assert not (tmp_path / "events.jsonl").exists()


def test_hparam_stop_serializes_process_exit_and_terminal_commit_against_monitor(tmp_path: Path, monkeypatch):
    rows = _write_runtime_rows(
        tmp_path,
        [{"run_id": "run-000", "status": "running", **_process_identity()}],
    )
    identity = _write_process_identity(rows[0]["pid_path"])
    real_merge = merge_run_manifest
    monitor_commit_ready = threading.Event()
    release_monitor_commit = threading.Event()
    monitor_failures = []

    def competing_merge(root, rows, **kwargs):
        if threading.current_thread().name == "competing-hparam-monitor":
            monitor_commit_ready.set()
            assert release_monitor_commit.wait(timeout=5)
        return real_merge(root, rows, **kwargs)

    def monitor():
        try:
            hparam_runtime.monitor_hparam_runs(tmp_path)
        except Exception as exc:
            monitor_failures.append(exc)

    monitor_thread = threading.Thread(target=monitor, name="competing-hparam-monitor")
    stopped = []

    def stop_while_monitor_commits(_row, observed):
        stopped.append(observed)
        monitor_thread.start()
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

    monkeypatch.setattr(hparam_runtime, "merge_run_manifest", competing_merge)
    monkeypatch.setattr(
        hparam_runtime.scheduler,
        "observe_run",
        lambda _root, observation, _prior, **_kwargs: {**observation, "status": "finished"},
    )
    monkeypatch.setattr(run_evidence, "stop_process_group", stop_while_monitor_commits)

    hparam_runtime.stop_hparam_run(tmp_path, "run-000", reason="manual stop")
    monitor_thread.join(timeout=5)

    assert not monitor_thread.is_alive()
    assert monitor_failures == []
    assert stopped == [identity]
    canonical = _read_table(tmp_path / "run_manifest.tsv")[0]
    assert canonical["status"] == "stopped"
    assert canonical["stop_reason"] == "manual stop"
    assert _read_table(tmp_path / "run_status.tsv")[0] == canonical
    assert _read_table(tmp_path / "launch_manifest.tsv")[0] == canonical
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert [event["event_type"] for event in events].count("run_stopped") == 1
    assert not any(
        event.get("event_type") == "run_status_changed" and event.get("to") == "finished" for event in events
    )


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
    rows = _write_runtime_rows(
        tmp_path,
        [{"run_id": "run-000", "status": "running", **_process_identity()}],
    )
    _write_process_identity(rows[0]["pid_path"])
    merge_run_manifest(tmp_path, [rows[0]])
    rows[0][field] = changed
    manifests.write_rows(tmp_path / "launch_manifest.tsv", rows)
    calls = []
    monkeypatch.setattr(run_evidence, "stop_process_group", lambda *_args: calls.append("stop"))

    hparam_runtime.stop_hparam_run(tmp_path, "run-000", reason="manual stop")

    assert calls == ["stop"]
    canonical = _read_table(tmp_path / "run_manifest.tsv")[0]
    assert _read_table(tmp_path / "launch_manifest.tsv")[0][field] == canonical[field]
