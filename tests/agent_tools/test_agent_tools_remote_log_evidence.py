from __future__ import annotations

import os

import pytest

from agent_tools import managed_scheduler, run_evidence

ROW = {"target": "local", "scheduler_type": "slurm", "status": "running"}


@pytest.mark.parametrize("health", [False, True])
def test_slurm_health_reads_fresh_log_evidence_each_observation(tmp_path, monkeypatch, health):
    path = tmp_path / "log.txt"
    path.write_text("first observation\n")
    monkeypatch.setattr(run_evidence.time, "time", lambda: 1000.75)
    os.utime(path, (987, 987))
    monkeypatch.setattr(run_evidence, "runtime_artifacts", lambda _row: ("", {}, []))
    row = {**ROW, "log_path": str(path)}
    observed = managed_scheduler._slurm_artifact_observation(dict(row), health=health)
    assert observed["log_tail"] == "first observation"
    if health:
        assert observed["log_age_seconds"] == 13
        assert observed["health_status"] == "scheduler_running"
    else:
        assert "log_age_seconds" not in observed
    path.write_text("next observation\n")
    os.utime(path, (999, 999))
    observed = managed_scheduler._slurm_artifact_observation(dict(row), health=health)
    assert observed["log_tail"] == "next observation"
    if health:
        assert observed["log_age_seconds"] == 1


@pytest.mark.parametrize("age", [None, 0])
@pytest.mark.parametrize("health_error", ["", "accounting unavailable"])
def test_slurm_health_preserves_missing_ages_and_lifecycle(monkeypatch, age, health_error):
    monkeypatch.setattr(run_evidence, "runtime_artifacts", lambda *_args: ("", {}, []))
    monkeypatch.setattr(run_evidence, "log_tail", lambda *_args: "log")
    monkeypatch.setattr(run_evidence, "log_age_seconds", lambda *_args: age)
    monkeypatch.setattr(managed_scheduler, "_timestamp_age_seconds", lambda *_args: age)
    custom = {"retained": True}
    row = {"status": "queued", "custom": custom}

    observed = managed_scheduler._slurm_artifact_observation(row, health=True, health_error=health_error)

    assert observed is row
    assert observed["custom"] is custom
    assert observed["status"] == "queued"
    assert observed["health_status"] == ("health_unknown" if health_error else "scheduler_queued")
    assert observed["scheduler_health_error"] == health_error
    for field in ("scheduler_queue_age_seconds", "scheduler_allocation_age_seconds", "log_age_seconds"):
        assert observed[field] == ("" if age is None else age)
    assert observed["log_tail"] == "log"


def test_direct_health_keeps_tail_progress_io_age_order(tmp_path, monkeypatch):
    calls = []
    row = {**ROW, "scheduler_type": "", "log_path": "log.txt", "run_id": "run-000"}
    monkeypatch.setattr(run_evidence, "read_pid", lambda *_args: 123)
    monkeypatch.setattr(run_evidence, "process_running", lambda *_args: True)
    monkeypatch.setattr(run_evidence, "runtime_artifacts", lambda *_args: ("", {}, []))
    monkeypatch.setattr(run_evidence, "log_tail", lambda *_args: calls.append("tail") or "tail")
    monkeypatch.setattr(run_evidence, "read_run_progress", lambda *_args: calls.append("progress") or {})
    monkeypatch.setattr(run_evidence, "proc_io", lambda *_args: calls.append("io") or {})
    monkeypatch.setattr(run_evidence, "log_age_seconds", lambda *_args: calls.append("age") or 7)
    monkeypatch.setattr(run_evidence, "gpu_summary", lambda *_args: calls.append("gpu") or "")
    observed = run_evidence.status_row(tmp_path, row, script_commits_terminal_status=True, health=True)
    assert observed["log_tail"] == "tail"
    assert observed["log_age_seconds"] == 7
    assert calls == ["tail", "progress", "io", "age", "gpu"]
