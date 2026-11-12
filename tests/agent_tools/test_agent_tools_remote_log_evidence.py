from __future__ import annotations

import os

import pytest

from agent_tools import managed_scheduler, run_evidence


@pytest.mark.parametrize("health", [False, True])
def test_slurm_health_reads_fresh_log_evidence_each_observation(tmp_path, monkeypatch, health):
    path = tmp_path / "log.txt"
    path.write_text("first observation\n")
    monkeypatch.setattr(run_evidence.time, "time", lambda: 1000.75)
    os.utime(path, (987, 987))
    monkeypatch.setattr(run_evidence, "runtime_artifacts", lambda _row: ("", {}, []))
    row = {"status": "running", "log_path": str(path)}
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
def test_slurm_health_preserves_missing_ages_and_lifecycle(monkeypatch, age):
    monkeypatch.setattr(run_evidence, "runtime_artifacts", lambda *_args: ("", {}, []))
    monkeypatch.setattr(run_evidence, "log_tail", lambda *_args: "log")
    monkeypatch.setattr(run_evidence, "log_age_seconds", lambda *_args: age)
    custom = {"retained": True}
    row = {"status": "queued", "custom": custom}

    observed = managed_scheduler._slurm_artifact_observation(row, health=True)

    assert observed is row
    assert observed["custom"] is custom
    assert observed["status"] == "queued"
    assert observed["health_status"] == "scheduler_queued"
    assert observed["log_age_seconds"] == ("" if age is None else age)
    assert observed["log_tail"] == "log"
