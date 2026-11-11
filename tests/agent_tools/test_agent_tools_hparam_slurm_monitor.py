from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_agent_tools_hparam_runtime import _read_table, _write_slurm_plan
from test_agent_tools_hparam_runtime import _stub_execution_snapshot_preflight  # noqa: F401

from agent_tools import cli, experiments, hparam_runtime, managed_scheduler, run_artifacts, run_evidence, slurm
from agent_tools.experiment_workspace import EXECUTION_IDENTITY_FIELDS, SCHEDULER_PLAN_IDENTITY_FIELDS

_STOP_AT = "2026-08-21T03:40:00Z"


# Applies only to a non-terminal Slurm run with a job id: exit_code first, then squeue, then the stop request.
@pytest.mark.parametrize(
    ("prior_status", "exit_code", "squeue_state", "stop_requested", "status"),
    [
        ("running", "0", "COMPLETED", False, "completed"),
        ("stopping", "0", None, True, "completed"),
        ("running", "7", "FAILED", False, "failed"),
        ("running", "143", None, False, "failed"),
        ("stopping", "143", "CANCELLED", True, "stopped"),
        ("queued", None, "PENDING", False, "queued"),
        ("queued", None, "REQUEUE_HOLD", False, "queued"),
        ("stopping", None, "PENDING", True, "stopping"),
        ("queued", None, "RUNNING", False, "running"),
        ("running", None, "CONFIGURING", False, "running"),
        ("running", None, "COMPLETING", False, "running"),
        ("running", None, "SUSPENDED", False, "running"),
        ("stopping", None, "RUNNING", True, "stopping"),
        ("stopping", None, "CANCELLED", True, "stopped"),
        ("stopping", None, None, True, "stopped"),
        ("running", None, "TIMEOUT", False, "lost"),
        ("running", None, "CANCELLED", False, "lost"),
        ("running", None, None, False, "lost"),
        ("lost", None, None, False, "lost"),
        ("lost", "0", None, False, "completed"),
        ("lost", "1", "NODE_FAIL", False, "failed"),
    ],
)
def test_slurm_monitor_state_table(
    tmp_path: Path, prior_status: str, exit_code, squeue_state, stop_requested: bool, status: str
):
    run_dir = tmp_path / "runs" / "run-000--lr-1e-4"
    run_dir.mkdir(parents=True)
    if exit_code is not None:
        (run_dir / "exit_code").write_text(f"{exit_code}\n")
    row = {
        "step_id": "tune",
        "run_id": "run-000",
        "scheduler_type": "slurm",
        "scheduler_job_id": "3880",
        "status": prior_status,
        "run_dir": str(run_dir),
        "log_path": str(run_dir / "slurm.log"),
    }
    if stop_requested:
        row.update(stop_requested_at=_STOP_AT, stop_reason="validation diverged")
    jobs = {"3881": slurm.JobObservation("3881", "other/step/run-000", "RUNNING", "", "gpu-02")}
    if squeue_state is not None:
        jobs["3880"] = slurm.JobObservation("3880", "exp/tune/run-000", squeue_state, "Reason", "gpu-01")

    observed = managed_scheduler.observe_slurm_run(tmp_path, row, jobs)

    assert observed["status"] == status
    assert observed["scheduler_job_id"] == "3880"
    assert observed.get("scheduler_exit_code", "") == ("" if exit_code is None else int(exit_code))
    # The raw state is recorded whenever squeue still shows the job, including for lost runs.
    assert observed.get("scheduler_raw_state", "") == (squeue_state or "")
    assert observed.get("scheduler_node", "") == ("gpu-01" if squeue_state else "")
    if status == "stopped":
        assert observed["stopped_at"]
    else:
        assert not observed.get("stopped_at")
    if status == "lost":
        assert "no exit_code" in observed["scheduler_reason"]
    assert status in managed_scheduler.ACTIVE_STATUSES | hparam_runtime.TERMINAL_STATUSES


def test_slurm_monitor_copies_recorded_runtime_commit(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "runtime_commit").write_text("a" * 40 + "\n")
    row = {
        "step_id": "tune",
        "run_id": "run-000",
        "scheduler_type": "slurm",
        "scheduler_job_id": "3880",
        "status": "queued",
        "run_dir": str(run_dir),
    }
    jobs = {"3880": slurm.JobObservation("3880", "exp/tune/run-000", "RUNNING", "", "gpu-01")}

    observed = managed_scheduler.observe_slurm_run(tmp_path, row, jobs)

    assert observed["status"] == "running"
    assert observed["runtime_commit"] == "a" * 40


@pytest.mark.parametrize(
    ("name", "text"),
    [("exit_code", "zero\n"), ("exit_code", "256\n"), ("runtime_commit", "dirty"), ("runtime_commit", "c" * 64)],
)
def test_slurm_monitor_rejects_malformed_job_records(tmp_path: Path, name: str, text: str):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / name).write_text(text)
    row = {
        "step_id": "tune",
        "run_id": "run-000",
        "scheduler_type": "slurm",
        "scheduler_job_id": "3880",
        "status": "running",
        "run_dir": str(run_dir),
    }

    with pytest.raises(ValueError, match=name):
        managed_scheduler.observe_slurm_run(tmp_path, row, {})


@pytest.mark.parametrize(
    "row",
    [
        {"status": "completed", "scheduler_job_id": "3880"},
        {"status": "lost"},
        {"status": "planned"},
    ],
)
def test_slurm_monitor_leaves_rows_without_a_live_job_binding_alone(tmp_path: Path, monkeypatch, row: dict):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "exit_code").write_text("0\n")
    row = {"step_id": "tune", "run_id": "run-000", "scheduler_type": "slurm", "run_dir": str(run_dir), **row}
    monkeypatch.setattr(slurm, "user_jobs", lambda **_kwargs: pytest.fail("no squeue query"))

    observed = managed_scheduler.observe_slurm_run(tmp_path, row)

    assert observed["status"] == row["status"]
    assert managed_scheduler.slurm_job_snapshot([row]) == {}


def _fake_queue(monkeypatch, *, job_ids: list[str] | None = None) -> dict:
    queue: dict = {"jobs": [], "squeue": 0, "submitted": [], "cancelled": []}
    ids = iter(job_ids or ["3880", "3881", "3882"])

    def user_jobs(**_kwargs):
        queue["squeue"] += 1
        return list(queue["jobs"])

    def submit(script, *, timeout):
        queue["submitted"].append(script)
        return next(ids)

    monkeypatch.setattr(slurm, "user_jobs", user_jobs)
    monkeypatch.setattr(slurm, "submit", submit)
    monkeypatch.setattr(slurm, "cancel", lambda job_id, **_kwargs: queue["cancelled"].append(job_id))
    return queue


def _canonical(tmp_path: Path, run: dict) -> dict[str, str]:
    return next(row for row in _read_table(tmp_path / "run_manifest.tsv") if row["run_id"] == run["run_id"])


def test_hparam_monitor_uses_one_squeue_query_per_round_and_records_runtime_drift(tmp_path: Path, monkeypatch):
    plan_dir, plan = _write_slurm_plan(tmp_path, run_count=2)
    first, second = plan["runs"]
    queue = _fake_queue(monkeypatch)
    hparam_runtime.launch_hparam_runs(plan_dir, dry_run=False)
    assert queue["submitted"] == [first["scheduler_script"], second["scheduler_script"]]
    assert queue["squeue"] == 1
    queue["jobs"] = [
        slurm.JobObservation("3880", slurm.job_name(first), "COMPLETED", "", "gpu-01"),
        slurm.JobObservation("3881", slurm.job_name(second), "RUNNING", "", "gpu-02"),
    ]
    (Path(first["run_dir"]) / "exit_code").write_text("0\n")
    (Path(first["run_dir"]) / "runtime_commit").write_text("f" * 40 + "\n")
    (Path(second["run_dir"]) / "runtime_commit").write_text(plan["recipe"]["execution"]["runtime_commit"] + "\n")

    hparam_runtime.monitor_hparam_runs(plan_dir)

    assert queue["squeue"] == 2
    finished, running = _canonical(tmp_path, first), _canonical(tmp_path, second)
    assert (finished["status"], finished["scheduler_exit_code"], finished["scheduler_raw_state"]) == (
        "completed",
        "0",
        "COMPLETED",
    )
    assert (running["status"], running["scheduler_node"]) == ("running", "gpu-02")
    assert finished["runtime_commit"] == "f" * 40
    assert finished["planned_runtime_commit"] == plan["recipe"]["execution"]["runtime_commit"]
    matrix = (tmp_path / "reports" / "run_matrix.md").read_text()
    assert "different (rolling update)" in matrix and "same" in matrix
    assert [row["status"] for row in _read_table(plan_dir / "run_status.tsv")] == ["completed", "running"]
    slurm_columns = {
        "scheduler_type",
        "scheduler_script",
        "scheduler_script_sha256",
        "scheduler_job_id",
        "scheduler_raw_state",
        "scheduler_reason",
        "scheduler_node",
        "scheduler_exit_code",
    }
    assert {field for field in finished if field.startswith("scheduler_")} <= slurm_columns


def test_slurm_hparam_run_queue_execute_is_one_submit_pass(tmp_path: Path, monkeypatch, capsys):
    plan_dir, plan = _write_slurm_plan(tmp_path, run_count=2)
    queue = _fake_queue(monkeypatch)
    monkeypatch.setattr(hparam_runtime.time, "sleep", lambda _seconds: pytest.fail("a Slurm queue never polls"))
    monkeypatch.setattr(hparam_runtime.subprocess, "Popen", lambda *_a, **_k: pytest.fail("no detached runner"))

    assert cli.main(["hparam-run-queue", "--plan-dir", str(plan_dir), "--execute"]) == 0

    output = capsys.readouterr().out
    assert "Lifecycle states: queued=2" in output
    assert "Queue runner" not in output
    assert queue["submitted"] == [run["scheduler_script"] for run in plan["runs"]]
    assert not (plan_dir / managed_scheduler.QUEUE_RUNNER_LOCK_NAME).exists()

    assert cli.main(["hparam-run-queue", "--plan-dir", str(plan_dir), "--execute"]) == 0

    assert len(queue["submitted"]) == 2
    assert [row["status"] for row in _read_table(plan_dir / "run_status.tsv")] == ["queued", "queued"]


def test_slurm_stop_of_unsubmitted_run_is_a_metadata_stop(tmp_path: Path, monkeypatch):
    plan_dir, plan = _write_slurm_plan(tmp_path, run_count=2)
    cancelled_run, remaining = plan["runs"]
    queue = _fake_queue(monkeypatch)
    monkeypatch.setattr(run_evidence, "stop_process_group", lambda *_a, **_k: pytest.fail("no signal"))
    monkeypatch.setattr(hparam_runtime, "utc_now", lambda: "2026-08-30T04:05:06Z")
    before = _canonical(tmp_path, cancelled_run)

    hparam_runtime.stop_hparam_run(plan_dir, cancelled_run["run_id"], reason="cancel before submission")

    canonical = _canonical(tmp_path, cancelled_run)
    assert (queue["squeue"], queue["cancelled"]) == (1, [])
    assert canonical["status"] == "stopped"
    assert canonical["stop_reason"] == "cancel before submission"
    assert canonical["stopped_at"] == "2026-08-30T04:05:06Z"
    for field in SCHEDULER_PLAN_IDENTITY_FIELDS | {"log_path"}:
        assert canonical.get(field, "") == before.get(field, "")
    for field in EXECUTION_IDENTITY_FIELDS - {"log_path"} | {"scheduler_job_id", "launched_at", "stop_requested_at"}:
        assert canonical.get(field, "") == ""
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert [event["event_type"] for event in events].count("run_stopped") == 1
    assert "run_stop_requested" not in [event["event_type"] for event in events]
    with pytest.raises(ValueError, match="already terminal"):
        hparam_runtime.stop_hparam_run(plan_dir, cancelled_run["run_id"], reason="cancel before submission")

    hparam_runtime.launch_hparam_runs(plan_dir, dry_run=False)

    assert queue["submitted"] == [remaining["scheduler_script"]]
    assert _canonical(tmp_path, cancelled_run)["status"] == "stopped"
    assert _canonical(tmp_path, remaining)["status"] == "queued"
    assert run_artifacts.read_hparam_plan(plan_dir) == plan


def test_hparam_stop_records_intent_before_scancel_and_monitor_commits_stopped(tmp_path: Path, monkeypatch):
    plan_dir, plan = _write_slurm_plan(tmp_path)
    run = plan["runs"][0]
    queue = _fake_queue(monkeypatch)
    hparam_runtime.launch_hparam_runs(plan_dir, dry_run=False)
    monkeypatch.setattr(hparam_runtime, "utc_now", lambda: _STOP_AT)

    def cancel(job_id, **_kwargs):
        durable = _canonical(tmp_path, run)
        assert (durable["status"], durable["scheduler_job_id"]) == ("stopping", "3880")
        assert (durable["stop_requested_at"], durable["stop_reason"]) == (_STOP_AT, "validation diverged")
        queue["cancelled"].append(job_id)

    monkeypatch.setattr(slurm, "cancel", cancel)

    hparam_runtime.stop_hparam_run(plan_dir, run["run_id"], reason="validation diverged")

    canonical = _canonical(tmp_path, run)
    assert queue["cancelled"] == ["3880"]
    assert canonical["status"] == "stopping"
    assert canonical.get("stopped_at", "") == ""
    assert _read_table(plan_dir / "run_status.tsv")[0] == canonical
    with pytest.raises(ValueError, match="pending stop request"):
        hparam_runtime.stop_hparam_run(plan_dir, run["run_id"], reason="repeat request")
    hparam_runtime.stop_hparam_run(plan_dir, run["run_id"], reason="validation diverged")
    assert queue["cancelled"] == ["3880", "3880"]

    queue["jobs"] = [slurm.JobObservation("3880", slurm.job_name(run), "CANCELLED", "", "gpu-01")]
    hparam_runtime.monitor_hparam_runs(plan_dir)

    canonical = _canonical(tmp_path, run)
    assert canonical["status"] == "stopped"
    assert canonical["stop_reason"] == "validation diverged"
    events = [json.loads(line)["event_type"] for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert (events.count("run_stop_requested"), events.count("run_stopped")) == (1, 1)


def test_hparam_stop_resolves_lost_run_without_scancel(tmp_path: Path, monkeypatch):
    plan_dir, plan = _write_slurm_plan(tmp_path)
    run = plan["runs"][0]
    queue = _fake_queue(monkeypatch)
    hparam_runtime.launch_hparam_runs(plan_dir, dry_run=False)
    hparam_runtime.monitor_hparam_runs(plan_dir)
    assert _canonical(tmp_path, run)["status"] == "lost"
    snapshot = experiments.experiment_status(tmp_path)
    assert any(blocker["code"] == "lost" for blocker in snapshot["blockers"])

    hparam_runtime.stop_hparam_run(plan_dir, run["run_id"], reason="job vanished without an exit code")
    hparam_runtime.launch_hparam_runs(plan_dir, dry_run=False)

    canonical = _canonical(tmp_path, run)
    assert canonical["status"] == "stopped"
    assert canonical["stop_reason"] == "job vanished without an exit code"
    assert canonical["scheduler_job_id"] == "3880"
    assert queue["cancelled"] == []
    assert queue["submitted"] == [run["scheduler_script"]]


def test_slurm_stop_intent_merge_failure_prevents_scancel(tmp_path: Path, monkeypatch):
    plan_dir, plan = _write_slurm_plan(tmp_path)
    run = plan["runs"][0]
    queue = _fake_queue(monkeypatch)
    hparam_runtime.launch_hparam_runs(plan_dir, dry_run=False)
    before = (tmp_path / "run_manifest.tsv").read_bytes()
    monkeypatch.setattr(
        hparam_runtime,
        "merge_run_manifest",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("intent merge failed")),
    )

    with pytest.raises(RuntimeError, match="intent merge failed"):
        hparam_runtime.stop_hparam_run(plan_dir, run["run_id"], reason="validation diverged")

    assert queue["cancelled"] == []
    assert (tmp_path / "run_manifest.tsv").read_bytes() == before


@pytest.mark.parametrize("stale_status", ["queued", "running", "lost"])
def test_slurm_monitor_preserves_concurrent_stop_intent(tmp_path: Path, monkeypatch, stale_status: str):
    plan_dir, plan = _write_slurm_plan(tmp_path)
    run = plan["runs"][0]
    _fake_queue(monkeypatch)
    hparam_runtime.launch_hparam_runs(plan_dir, dry_run=False)
    monkeypatch.setattr(hparam_runtime, "utc_now", lambda: _STOP_AT)

    def observe_stale_row(_root, row, _jobs=None, *, health=False):
        assert row["status"] == "queued"
        assert row.get("stop_requested_at", "") == ""
        hparam_runtime.stop_hparam_run(plan_dir, run["run_id"], reason="validation diverged")
        return {**row, "status": stale_status, "stop_requested_at": "", "stop_reason": "", "scheduler_raw_state": "X"}

    monkeypatch.setattr(managed_scheduler, "observe_slurm_run", observe_stale_row)

    hparam_runtime.monitor_hparam_runs(plan_dir)

    canonical = _canonical(tmp_path, run)
    assert canonical["status"] == "stopping"
    assert canonical["stop_requested_at"] == _STOP_AT
    assert canonical["stop_reason"] == "validation diverged"
    assert _read_table(plan_dir / "run_status.tsv")[0] == canonical
