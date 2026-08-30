from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest
from test_agent_tools_hparam_runtime import _read_table, _write_slurm_plan
from test_agent_tools_hparam_runtime import _stub_execution_snapshot_preflight  # noqa: F401

from agent_tools import experiments, hparam_runtime, managed_scheduler, manifests, run_artifacts, run_evidence, slurm
from agent_tools.experiment_workspace import (
    EXECUTION_IDENTITY_FIELDS,
    SCHEDULER_PLAN_IDENTITY_FIELDS,
    merge_run_manifest,
)


@pytest.mark.parametrize(
    ("direct_controller", "target", "status", "dry_run"),
    [(False, "local", "planned", False), (True, "local", "pending", True), (False, "ssh", "planned", False)],
)
def test_slurm_stop_cancels_unsubmitted_run_without_scheduler_probes(
    tmp_path: Path, monkeypatch, direct_controller, target, status, dry_run
):
    execution = {"target": target}
    if target == "ssh":
        execution["host"] = "unit-host"
    real_validate = hparam_runtime.exp_io.validate_managed_output_paths
    monkeypatch.setattr(
        hparam_runtime.exp_io,
        "validate_managed_output_paths",
        lambda root, paths, remote=None: None if remote else real_validate(root, paths),
    )
    plan_dir, plan = _write_slurm_plan(tmp_path, execution=execution, direct_controller=direct_controller)
    run = plan["runs"][0]
    if dry_run:
        hparam_runtime.launch_hparam_runs(plan_dir, dry_run=True)
    update = {"step_id": run["step_id"], "run_id": run["run_id"], "status": status}
    if dry_run:
        update["execution_snapshot_sha256"] = plan["execution_snapshot"]["sha256"]
    merge_run_manifest(tmp_path, [update])
    before = next(row for row in _read_table(tmp_path / "run_manifest.tsv") if row["run_id"] == run["run_id"])
    assert before["log_path"] == run["log_path"]
    snapshot_path = plan_dir / "execution_snapshot.json"
    snapshot_bytes = snapshot_path.read_bytes()
    for name in ("active_jobs", "accounting_job", "show_job", "cancel", "controller_cluster", "submit"):
        monkeypatch.setattr(slurm, name, lambda *_a, **_k: pytest.fail("no scheduler command"))
    monkeypatch.setattr(run_evidence, "read_process_identity", lambda *_a, **_k: pytest.fail("no identity read"))
    monkeypatch.setattr(run_evidence, "stop_process_group", lambda *_a, **_k: pytest.fail("no signal"))
    monkeypatch.setattr(managed_scheduler, "observe_slurm_run", lambda *_a, **_k: pytest.fail("no scheduler probe"))

    def validate_local(root, paths, *, remote=None):
        assert remote is None
        return real_validate(root, paths)

    monkeypatch.setattr(hparam_runtime.exp_io, "validate_managed_output_paths", validate_local)
    monkeypatch.setattr(hparam_runtime, "utc_now", lambda: "2026-08-30T04:05:06Z")

    hparam_runtime.stop_hparam_run(plan_dir, run["run_id"], reason="cancel before submission")

    canonical = next(row for row in _read_table(tmp_path / "run_manifest.tsv") if row["run_id"] == run["run_id"])
    assert canonical == _read_table(plan_dir / "run_status.tsv")[0] == _read_table(plan_dir / "launch_manifest.tsv")[0]
    assert canonical["status"] == "stopped"
    assert canonical["stop_reason"] == "cancel before submission"
    assert canonical["stopped_at"] == "2026-08-30T04:05:06Z"
    for field in SCHEDULER_PLAN_IDENTITY_FIELDS | {"log_path", "execution_snapshot_sha256"}:
        assert canonical.get(field, "") == before.get(field, "")
    for field in EXECUTION_IDENTITY_FIELDS - {"log_path"} | {
        "scheduler_job_id",
        "scheduler_cluster",
        "launched_at",
        "stop_requested_at",
    }:
        assert canonical.get(field, "") == ""
    assert run_artifacts.read_hparam_plan(plan_dir) == plan
    snapshot = experiments.experiment_status(tmp_path)
    assert snapshot["summary"]["status_counts"] == {"planned": 1, "stopped": 1}
    assert next(row for row in snapshot["runs"] if row["run_id"] == run["run_id"])["status"] == "stopped"
    assert snapshot_path.read_bytes() == snapshot_bytes
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert [event["event_type"] for event in events].count("run_stopped") == 1
    stopped_event = next(event for event in events if event["event_type"] == "run_stopped")
    assert (stopped_event["step_id"], stopped_event["run_id"], stopped_event["reason"]) == (
        run["step_id"],
        run["run_id"],
        "cancel before submission",
    )
    assert not any(event["event_type"] == "run_stop_requested" for event in events)
    before_repeat = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    with pytest.raises(ValueError, match="already terminal"):
        hparam_runtime.stop_hparam_run(plan_dir, run["run_id"], reason="cancel before submission")
    assert {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before_repeat
    hparam_runtime.launch_hparam_runs(plan_dir, dry_run=False)
    assert _read_table(plan_dir / "run_status.tsv")[0]["status"] == "stopped"


def test_slurm_unsubmitted_stop_leaves_other_runs_launchable(tmp_path: Path, monkeypatch):
    plan_dir, plan = _write_slurm_plan(tmp_path, run_count=2)
    cancelled, remaining = plan["runs"]
    submitted = []

    def submit(_execution, script, token, **_kwargs):
        submitted.append((script, token))
        return slurm.JobIdentity("3880", "wuji-h20")

    monkeypatch.setattr(slurm, "submit", submit)

    hparam_runtime.stop_hparam_run(plan_dir, cancelled["run_id"], reason="cancel one candidate")
    hparam_runtime.launch_hparam_runs(plan_dir, dry_run=False)

    rows = {row["run_id"]: row for row in _read_table(tmp_path / "run_manifest.tsv")}
    assert submitted == [(remaining["scheduler_script"], remaining["scheduler_submit_token"])]
    assert rows[cancelled["run_id"]]["status"] == "stopped"
    assert rows[cancelled["run_id"]].get("target", "") == ""
    assert rows[cancelled["run_id"]].get("scheduler_job_id", "") == ""
    assert rows[remaining["run_id"]]["status"] == "queued"
    assert rows[remaining["run_id"]]["scheduler_job_id"] == "3880"
    assert run_artifacts.read_hparam_plan(plan_dir) == plan


@pytest.mark.parametrize(
    "evidence",
    [
        {"status": "submitting"},
        {"status": "submitting", "scheduler_cluster": "wuji-h20"},
        {"status": "unknown_scheduler", "scheduler_job_id": "3880"},
        {"launched_at": "2026-08-30T04:00:00Z"},
        {"stop_requested_at": "2026-08-30T04:00:00Z"},
    ],
)
def test_slurm_stop_does_not_metadata_cancel_uncertain_submission(tmp_path: Path, monkeypatch, evidence):
    plan_dir, plan = _write_slurm_plan(tmp_path)
    run = plan["runs"][0]
    rows = _read_table(tmp_path / "run_manifest.tsv")
    next(row for row in rows if row["run_id"] == run["run_id"]).update(evidence)
    manifests.write_rows(tmp_path / "run_manifest.tsv", rows)
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    monkeypatch.setattr(slurm, "active_jobs", lambda *_a, **_k: pytest.fail("no scheduler lookup"))
    monkeypatch.setattr(slurm, "cancel", lambda *_a, **_k: pytest.fail("no scheduler cancel"))

    with pytest.raises(ValueError):
        hparam_runtime.stop_hparam_run(plan_dir, run["run_id"], reason="unconfirmed submission")

    assert {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before


def test_slurm_unsubmitted_stop_survives_stale_monitor_without_duplicate_event(tmp_path: Path, monkeypatch):
    plan_dir, plan = _write_slurm_plan(tmp_path)
    run = plan["runs"][0]
    real_merge = hparam_runtime.merge_run_manifest
    cancelled = False
    monkeypatch.setattr(hparam_runtime, "utc_now", lambda: "2026-08-30T04:05:06Z")

    def merge_after_cancel(root, rows, **kwargs):
        nonlocal cancelled
        if not cancelled:
            cancelled = True
            hparam_runtime.stop_hparam_run(plan_dir, run["run_id"], reason="cancel before submission")
            rows = [{**row, "stopped_at": "", "stop_reason": ""} for row in rows]
        return real_merge(root, rows, **kwargs)

    monkeypatch.setattr(hparam_runtime, "merge_run_manifest", merge_after_cancel)
    monkeypatch.setattr(slurm, "active_jobs", lambda *_a, **_k: pytest.fail("no scheduler lookup"))
    monkeypatch.setattr(slurm, "cancel", lambda *_a, **_k: pytest.fail("no scheduler cancel"))

    hparam_runtime.monitor_hparam_runs(plan_dir)

    canonical = next(row for row in _read_table(tmp_path / "run_manifest.tsv") if row["run_id"] == run["run_id"])
    assert canonical == _read_table(plan_dir / "run_status.tsv")[0] == _read_table(plan_dir / "launch_manifest.tsv")[0]
    assert canonical["status"] == "stopped"
    assert canonical["stop_reason"] == "cancel before submission"
    assert canonical["stopped_at"] == "2026-08-30T04:05:06Z"
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert [event["event_type"] for event in events].count("run_stopped") == 1


def test_slurm_monitor_commits_terminal_sidecar_result(tmp_path: Path, monkeypatch):
    plan_dir, plan = _write_slurm_plan(tmp_path)
    run = plan["runs"][0]
    monkeypatch.setattr(
        managed_scheduler.slurm,
        "submit",
        lambda *_args, **_kwargs: slurm.JobIdentity("3880", "wuji-h20"),
    )
    hparam_runtime.launch_hparam_runs(plan_dir, dry_run=False)
    Path(run["scheduler_result_path"]).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "scheduler_job_id": "3880",
                "scheduler_cluster": "wuji-h20",
                "scheduler_submit_token": run["scheduler_submit_token"],
                "node": "h20-bj-96",
                "exit_code": 0,
            }
        )
    )
    Path(run["allocation_identity_path"]).symlink_to(tmp_path / "missing-allocation.json")
    read_calls = []
    real_read = managed_scheduler.exp_io.read_managed_output_texts_at

    def read_output_texts(root, paths, *, remote=None):
        read_calls.append([str(path) for path in paths])
        return real_read(root, paths, remote=remote)

    monkeypatch.setattr(managed_scheduler.exp_io, "read_managed_output_texts_at", read_output_texts)
    monkeypatch.setattr(managed_scheduler.slurm, "active_jobs", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        managed_scheduler.slurm,
        "show_job",
        lambda _execution, job_id, *, cluster=None, timeout=10: slurm.JobObservation(
            job_id,
            "COMPLETED",
            "",
            "h20-bj-96",
            run["scheduler_submit_token"],
            "0:0",
        ),
    )

    hparam_runtime.monitor_hparam_runs(plan_dir)

    canonical = next(row for row in _read_table(tmp_path / "run_manifest.tsv") if row["run_id"] == run["run_id"])
    assert canonical["status"] == "completed"
    assert canonical["scheduler_raw_state"] == "COMPLETED"
    assert canonical["scheduler_exit_code"] == "0"
    assert canonical["scheduler_node"] == "h20-bj-96"
    assert read_calls == [[run["scheduler_result_path"]]]


@pytest.mark.parametrize(
    ("terminal_text", "expected_error"),
    [
        (None, None),
        ("", None),
        ("{}", None),
        (" \n", "not valid JSON"),
        ("{", "not valid JSON"),
        ("[]", "must be a mapping"),
        ("null", "must be a mapping"),
    ],
    ids=["missing", "empty", "empty-mapping", "whitespace", "invalid-json", "array", "null"],
)
def test_slurm_monitor_reads_allocation_only_for_absent_terminal_mapping(
    tmp_path: Path, monkeypatch, terminal_text: str | None, expected_error: str | None
):
    plan_dir, plan = _write_slurm_plan(tmp_path)
    row = {
        **plan["runs"][0],
        "target": "local",
        "scheduler_job_id": "3880",
        "scheduler_cluster": "wuji-h20",
        "status": "queued",
    }
    if terminal_text is not None:
        Path(row["scheduler_result_path"]).write_text(terminal_text)
    Path(row["allocation_identity_path"]).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "scheduler_job_id": "3880",
                "scheduler_cluster": "wuji-h20",
                "scheduler_submit_token": row["scheduler_submit_token"],
                "node": "h20-bj-96",
            }
        )
    )
    read_calls = []
    scheduler_calls = []
    real_read = managed_scheduler.exp_io.read_managed_output_texts_at

    def read_output_texts(root, paths, *, remote=None):
        read_calls.append([str(path) for path in paths])
        return real_read(root, paths, remote=remote)

    def active_jobs(_execution, **kwargs):
        scheduler_calls.append(kwargs)
        return [slurm.JobObservation("3880", "PENDING", "", "", row["scheduler_submit_token"])]

    monkeypatch.setattr(managed_scheduler.exp_io, "read_managed_output_texts_at", read_output_texts)
    monkeypatch.setattr(slurm, "active_jobs", active_jobs)

    if expected_error:
        with pytest.raises(ValueError, match=expected_error):
            managed_scheduler.observe_slurm_run(plan_dir, {"target": "local"}, row)
        assert read_calls == [[row["scheduler_result_path"]]]
        assert scheduler_calls == []
    else:
        observed = managed_scheduler.observe_slurm_run(plan_dir, {"target": "local"}, row)
        assert read_calls == [[row["scheduler_result_path"]], [row["allocation_identity_path"]]]
        assert len(scheduler_calls) == 1
        assert observed["status"] == "queued"
        assert observed["scheduler_node"] == "h20-bj-96"
        assert observed["scheduler_job_id"] == row["scheduler_job_id"]
        assert observed["scheduler_cluster"] == row["scheduler_cluster"]


@pytest.mark.parametrize(
    ("scheduler_state", "sidecar_exit_code", "expected_status"),
    [
        ("RUNNING", 0, "running"),
        ("COMPLETING", 0, "running"),
        ("SUSPENDED", 0, "running"),
        ("STOPPED", 0, "running"),
        ("RESIZING", 0, "running"),
        ("SIGNALING", 0, "running"),
        ("STAGE_OUT", 0, "running"),
        ("COMPLETED", 7, "failed"),
        ("FAILED", 0, "failed"),
        ("CANCELLED", 0, "failed"),
    ],
)
def test_slurm_monitor_requires_scheduler_and_sidecar_terminal_evidence(
    tmp_path: Path,
    monkeypatch,
    scheduler_state: str,
    sidecar_exit_code: int,
    expected_status: str,
):
    plan_dir, plan = _write_slurm_plan(tmp_path)
    run = plan["runs"][0]
    token = run["scheduler_submit_token"]
    monkeypatch.setattr(
        managed_scheduler.slurm,
        "submit",
        lambda *_args, **_kwargs: slurm.JobIdentity("3880", "wuji-h20"),
    )
    hparam_runtime.launch_hparam_runs(plan_dir, dry_run=False)
    Path(run["scheduler_result_path"]).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "scheduler_job_id": "3880",
                "scheduler_cluster": "wuji-h20",
                "scheduler_submit_token": token,
                "node": "h20-bj-96",
                "exit_code": sidecar_exit_code,
            }
        )
    )
    monkeypatch.setattr(
        managed_scheduler.slurm,
        "active_jobs",
        lambda *_args, **_kwargs: [slurm.JobObservation("3880", scheduler_state, "", "h20-bj-96", token)],
    )

    hparam_runtime.monitor_hparam_runs(plan_dir)

    canonical = next(row for row in _read_table(tmp_path / "run_manifest.tsv") if row["run_id"] == run["run_id"])
    assert canonical["status"] == expected_status
    assert canonical["scheduler_raw_state"] == scheduler_state


def test_slurm_monitor_keeps_terminal_sidecar_unknown_without_scheduler_record(tmp_path: Path, monkeypatch):
    plan_dir, plan = _write_slurm_plan(tmp_path)
    run = plan["runs"][0]
    monkeypatch.setattr(
        managed_scheduler.slurm,
        "submit",
        lambda *_args, **_kwargs: slurm.JobIdentity("3880", "wuji-h20"),
    )
    hparam_runtime.launch_hparam_runs(plan_dir, dry_run=False)
    Path(run["scheduler_result_path"]).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "scheduler_job_id": "3880",
                "scheduler_cluster": "wuji-h20",
                "scheduler_submit_token": run["scheduler_submit_token"],
                "node": "h20-bj-96",
                "exit_code": 0,
            }
        )
    )
    monkeypatch.setattr(managed_scheduler.slurm, "active_jobs", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(managed_scheduler.slurm, "show_job", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(managed_scheduler.slurm, "accounting_job", lambda *_args, **_kwargs: None)

    hparam_runtime.monitor_hparam_runs(plan_dir)

    canonical = next(row for row in _read_table(tmp_path / "run_manifest.tsv") if row["run_id"] == run["run_id"])
    assert canonical["status"] == "unknown_scheduler"
    assert canonical["scheduler_raw_state"] == "MISSING"
    assert "before terminal scheduler state was observed" in canonical["scheduler_reason"]


@pytest.mark.parametrize("canonical_job_id,canonical_cluster", [("", ""), ("3880", ""), ("", "wuji-h20")])
@pytest.mark.parametrize("sidecar_kind", ["allocation", "terminal"])
@pytest.mark.parametrize("first_observation", ["error", "missing"])
def test_slurm_monitor_does_not_commit_sidecar_identity_without_scheduler_evidence(
    tmp_path: Path,
    monkeypatch,
    canonical_job_id: str,
    canonical_cluster: str,
    sidecar_kind: str,
    first_observation: str,
):
    plan_dir, plan = _write_slurm_plan(tmp_path)
    run = plan["runs"][0]
    merge_run_manifest(
        tmp_path,
        [
            {
                "step_id": run["step_id"],
                "run_id": run["run_id"],
                "target": "local",
                "status": "queued" if canonical_job_id else "submitting",
                "scheduler_job_id": canonical_job_id,
                "scheduler_cluster": canonical_cluster,
                "launched_at": "2026-08-21T00:00:00Z" if canonical_job_id else "",
            }
        ],
    )
    sidecar = {
        "schema_version": 1,
        "scheduler_job_id": "3880",
        "scheduler_cluster": canonical_cluster or "sidecar-only-cluster",
        "scheduler_submit_token": run["scheduler_submit_token"],
        "node": "h20-bj-96",
        "started_at": "2026-08-21T00:00:00Z",
    }
    sidecar_path = run["allocation_identity_path"]
    if sidecar_kind == "terminal":
        sidecar_path = run["scheduler_result_path"]
        sidecar.update({"ended_at": "2026-08-21T00:01:00Z", "exit_code": 0})
    Path(sidecar_path).write_text(json.dumps(sidecar))
    scheduler_calls = []

    def run_command(_execution, argv, *, timeout):
        scheduler_calls.append(argv)
        if argv[0] == "squeue":
            if observation == "error":
                return subprocess.CompletedProcess([], 1, "", "controller is unavailable")
            return subprocess.CompletedProcess([], 0, "", "")
        if argv[0] == "scontrol":
            return subprocess.CompletedProcess([], 1, "", "slurm_load_jobs error: Invalid job id specified")
        assert argv[0] == "sacct"
        return subprocess.CompletedProcess([], 1, "", "Slurm accounting storage is disabled")

    monkeypatch.setattr(slurm, "run_command", run_command)

    for observation in (first_observation, "missing"):
        hparam_runtime.monitor_hparam_runs(plan_dir)

        canonical = next(row for row in _read_table(tmp_path / "run_manifest.tsv") if row["run_id"] == run["run_id"])
        assert canonical["status"] == ("unknown_scheduler" if canonical_job_id else "submitting")
        assert canonical.get("scheduler_job_id", "") == canonical_job_id
        assert canonical.get("scheduler_cluster", "") == canonical_cluster
        if not canonical_job_id:
            assert canonical.get("launched_at", "") == ""
        assert _read_table(plan_dir / "run_status.tsv")[0] == canonical
    expected_cluster_args = [f"--clusters={canonical_cluster}"] if canonical_cluster else []
    assert all(
        [arg for arg in argv if arg.startswith("--clusters=")] == expected_cluster_args for argv in scheduler_calls
    )


@pytest.mark.parametrize("canonical_job_id", ["", "3880"])
@pytest.mark.parametrize("observation_source", ["queue", "controller", "accounting"])
def test_slurm_monitor_authenticates_sidecar_job_on_frozen_route_without_binding_sidecar_cluster(
    tmp_path: Path,
    monkeypatch,
    canonical_job_id: str,
    observation_source: str,
):
    plan_dir, plan = _write_slurm_plan(tmp_path)
    run = plan["runs"][0]
    token = run["scheduler_submit_token"]
    merge_run_manifest(
        tmp_path,
        [
            {
                "step_id": run["step_id"],
                "run_id": run["run_id"],
                "target": "local",
                "status": "queued" if canonical_job_id else "submitting",
                "scheduler_job_id": canonical_job_id,
                "launched_at": "2026-08-21T00:00:00Z" if canonical_job_id else "",
            }
        ],
    )
    Path(run["scheduler_result_path"]).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "scheduler_job_id": "3880",
                "scheduler_cluster": "sidecar-only-cluster",
                "scheduler_submit_token": token,
                "node": "h20-bj-96",
                "exit_code": 0,
            }
        )
    )
    scheduler_calls = []

    def run_command(_execution, argv, *, timeout):
        scheduler_calls.append(argv)
        if argv[0] == "squeue":
            output = f"3880|COMPLETED||h20-bj-96|{token}\n" if observation_source == "queue" else ""
            return subprocess.CompletedProcess([], 0, output, "")
        if argv[0] == "scontrol":
            if observation_source == "controller":
                return subprocess.CompletedProcess(
                    [], 0, f"JobId=3880 JobState=COMPLETED Comment={token} NodeList=h20-bj-96 ExitCode=0:0", ""
                )
            return subprocess.CompletedProcess([], 1, "", "slurm_load_jobs error: Invalid job id specified")
        assert argv[0] == "sacct"
        return subprocess.CompletedProcess([], 0, f"3880|COMPLETED|0:0|h20-bj-96|{token}\n", "")

    monkeypatch.setattr(slurm, "run_command", run_command)

    hparam_runtime.monitor_hparam_runs(plan_dir)

    canonical = next(row for row in _read_table(tmp_path / "run_manifest.tsv") if row["run_id"] == run["run_id"])
    assert canonical["status"] == "completed"
    assert canonical["scheduler_job_id"] == "3880"
    assert canonical.get("scheduler_cluster", "") == ""
    assert canonical["scheduler_exit_code"] == "0"
    assert canonical["launched_at"]
    assert _read_table(plan_dir / "run_status.tsv")[0] == canonical
    assert all(not any(arg.startswith("--clusters=") for arg in argv) for argv in scheduler_calls)
    assert scheduler_calls[0][-2:] == ["--jobs", "3880"]


@pytest.mark.parametrize(
    (
        "terminal_exit_code",
        "stop_requested",
        "canonical_cluster",
        "sidecar_cluster",
        "direct_controller",
        "expected_status",
    ),
    [
        (None, False, "wuji-h20", "wuji-h20", False, "unknown_scheduler"),
        (0, False, "wuji-h20", "wuji-h20", False, "completed"),
        (0, False, "wuji-h20", "wuji-h20", True, "completed"),
        (7, False, "wuji-h20", "wuji-h20", False, "failed"),
        (143, True, "wuji-h20", "wuji-h20", False, "failed"),
        (0, False, "wuji-h20", "", False, "unknown_scheduler"),
        (0, False, "", "wuji-h20", False, "unknown_scheduler"),
        (0, False, "", "", False, "unknown_scheduler"),
    ],
)
def test_slurm_monitor_handles_purged_job_when_accounting_is_unavailable(
    tmp_path: Path,
    monkeypatch,
    terminal_exit_code: int | None,
    stop_requested: bool,
    canonical_cluster: str,
    sidecar_cluster: str,
    direct_controller: bool,
    expected_status: str,
):
    plan_dir, plan = _write_slurm_plan(tmp_path, direct_controller=direct_controller)
    run = plan["runs"][0]
    token = run["scheduler_submit_token"]
    if canonical_cluster:
        monkeypatch.setattr(
            managed_scheduler.slurm,
            "submit",
            lambda *_args, **_kwargs: slurm.JobIdentity("3880", canonical_cluster),
        )
        hparam_runtime.launch_hparam_runs(plan_dir, dry_run=False)
    else:
        merge_run_manifest(
            tmp_path,
            [
                {
                    "step_id": run["step_id"],
                    "run_id": run["run_id"],
                    "target": "local",
                    "status": "queued",
                    "scheduler_job_id": "3880",
                    "scheduler_cluster": "",
                    "launched_at": "2026-08-21T00:00:00Z",
                }
            ],
        )

    if stop_requested:
        monkeypatch.setattr(slurm, "cancel", lambda *_args, **_kwargs: None)
        hparam_runtime.stop_hparam_run(plan_dir, run["run_id"], reason="user requested stop")
    if terminal_exit_code is not None:
        Path(run["scheduler_result_path"]).write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "scheduler_job_id": "3880",
                    "scheduler_cluster": sidecar_cluster,
                    "scheduler_submit_token": token,
                    "node": "h20-bj-96",
                    "exit_code": terminal_exit_code,
                }
            )
        )

    scheduler_calls = []

    def run_command(_execution, argv, *, timeout):
        scheduler_calls.append(argv)
        if argv[0] == "squeue":
            return subprocess.CompletedProcess([], 0, "", "")
        if argv[0] == "scontrol":
            return subprocess.CompletedProcess([], 1, "", "slurm_load_jobs error: Invalid job id specified")
        assert argv[0] == "sacct"
        return subprocess.CompletedProcess([], 1, "", "Slurm accounting storage is disabled")

    monkeypatch.setattr(managed_scheduler.slurm, "run_command", run_command)

    hparam_runtime.monitor_hparam_runs(plan_dir)
    if not canonical_cluster and terminal_exit_code is not None:
        hparam_runtime.monitor_hparam_runs(plan_dir)

    canonical = next(row for row in _read_table(tmp_path / "run_manifest.tsv") if row["run_id"] == run["run_id"])
    assert canonical["status"] == expected_status
    assert canonical["scheduler_raw_state"] == "MISSING"
    expected_commands = ["squeue", "scontrol", "sacct"] * (
        2 if not canonical_cluster and terminal_exit_code is not None else 1
    )
    assert [argv[0] for argv in scheduler_calls] == expected_commands
    if not canonical_cluster:
        assert canonical["scheduler_cluster"] == ""
    if canonical_cluster and expected_status in {"completed", "failed"}:
        cluster_flag = f"--clusters={canonical_cluster}"
        assert all((cluster_flag not in argv) == direct_controller for argv in scheduler_calls)
    if expected_status in {"completed", "failed"}:
        assert canonical["scheduler_exit_code"] == str(terminal_exit_code)
        assert "accounting storage is disabled" in canonical["scheduler_reason"]
        assert "authenticated terminal sidecar" in canonical["scheduler_reason"]
        assert canonical["status"] != "stopped"
    else:
        assert "Slurm accounting query failed: Slurm accounting storage is disabled" in canonical["scheduler_reason"]
        if terminal_exit_code not in (None, 0):
            assert f"non-zero exit code {terminal_exit_code}" in canonical["scheduler_reason"]


def test_slurm_monitor_ignores_local_host_label_for_accounting_disabled_recovery(
    tmp_path: Path,
    monkeypatch,
):
    plan_dir, plan = _write_slurm_plan(tmp_path, execution={"host": "local-label"})
    run = plan["runs"][0]
    monkeypatch.setattr(
        managed_scheduler.slurm,
        "submit",
        lambda *_args, **_kwargs: slurm.JobIdentity("3880", "wuji-h20"),
    )
    hparam_runtime.launch_hparam_runs(plan_dir, dry_run=False)
    Path(run["scheduler_result_path"]).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "scheduler_job_id": "3880",
                "scheduler_cluster": "wuji-h20",
                "scheduler_submit_token": run["scheduler_submit_token"],
                "node": "h20-bj-96",
                "exit_code": 0,
            }
        )
    )

    def run_command(_execution, argv, *, timeout):
        if argv[0] == "squeue":
            return subprocess.CompletedProcess([], 0, "", "")
        if argv[0] == "scontrol":
            return subprocess.CompletedProcess([], 1, "", "slurm_load_jobs error: Invalid job id specified")
        return subprocess.CompletedProcess([], 1, "", "Slurm accounting storage is disabled")

    monkeypatch.setattr(managed_scheduler.slurm, "run_command", run_command)

    hparam_runtime.monitor_hparam_runs(plan_dir)

    canonical = next(row for row in _read_table(tmp_path / "run_manifest.tsv") if row["run_id"] == run["run_id"])
    assert canonical["target"] == "local"
    assert canonical["host"] == "local-label"
    assert canonical["status"] == "completed"
    assert canonical["scheduler_raw_state"] == "MISSING"


def test_slurm_accounting_disabled_recovery_requires_matching_ssh_host(tmp_path: Path, monkeypatch):
    plan_dir, plan = _write_slurm_plan(tmp_path)
    run = plan["runs"][0]
    monkeypatch.setattr(
        managed_scheduler.slurm,
        "submit",
        lambda *_args, **_kwargs: slurm.JobIdentity("3880", "wuji-h20"),
    )
    hparam_runtime.launch_hparam_runs(plan_dir, dry_run=False)
    row = next(item for item in _read_table(tmp_path / "run_manifest.tsv") if item["run_id"] == run["run_id"])
    row.update({"target": "ssh", "host": "baichuan3"})
    terminal = {
        "schema_version": 1,
        "scheduler_job_id": "3880",
        "scheduler_cluster": "wuji-h20",
        "scheduler_submit_token": row["scheduler_submit_token"],
        "node": "h20-bj-96",
        "exit_code": 0,
    }

    def read_output_texts(root, paths, *, remote=None):
        assert root == plan_dir
        assert paths == [row["scheduler_result_path"]]
        assert remote == "other-host"
        return {row["scheduler_result_path"]: json.dumps(terminal)}

    monkeypatch.setattr(managed_scheduler.exp_io, "read_managed_output_texts_at", read_output_texts)
    monkeypatch.setattr(run_evidence, "runtime_artifacts", lambda _row: None)
    monkeypatch.setattr(run_evidence, "log_tail", lambda *_args, **_kwargs: "")

    def run_command(_execution, argv, *, timeout):
        if argv[0] == "squeue":
            return subprocess.CompletedProcess([], 0, "", "")
        if argv[0] == "scontrol":
            return subprocess.CompletedProcess([], 1, "", "slurm_load_jobs error: Invalid job id specified")
        return subprocess.CompletedProcess([], 1, "", "Slurm accounting storage is disabled")

    monkeypatch.setattr(managed_scheduler.slurm, "run_command", run_command)

    observed = managed_scheduler.observe_slurm_run(
        plan_dir,
        {"target": "ssh", "host": "other-host"},
        row,
    )

    assert observed["status"] == "unknown_scheduler"
    assert observed["scheduler_raw_state"] == "MISSING"


@pytest.mark.parametrize("accounting_failure", ["permission", "timeout"])
def test_slurm_monitor_keeps_purged_sidecar_unknown_for_other_accounting_failures(
    tmp_path: Path,
    monkeypatch,
    accounting_failure: str,
):
    plan_dir, plan = _write_slurm_plan(tmp_path)
    run = plan["runs"][0]
    monkeypatch.setattr(
        managed_scheduler.slurm,
        "submit",
        lambda *_args, **_kwargs: slurm.JobIdentity("3880", "wuji-h20"),
    )
    hparam_runtime.launch_hparam_runs(plan_dir, dry_run=False)
    Path(run["scheduler_result_path"]).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "scheduler_job_id": "3880",
                "scheduler_cluster": "wuji-h20",
                "scheduler_submit_token": run["scheduler_submit_token"],
                "node": "h20-bj-96",
                "exit_code": 0,
            }
        )
    )

    def run_command(_execution, argv, *, timeout):
        if argv[0] == "squeue":
            return subprocess.CompletedProcess([], 0, "", "")
        if argv[0] == "scontrol":
            return subprocess.CompletedProcess([], 1, "", "slurm_load_jobs error: Invalid job id specified")
        if accounting_failure == "timeout":
            raise subprocess.TimeoutExpired(argv, timeout)
        return subprocess.CompletedProcess([], 1, "", "sacct: error: Access denied")

    monkeypatch.setattr(managed_scheduler.slurm, "run_command", run_command)

    hparam_runtime.monitor_hparam_runs(plan_dir)

    canonical = next(row for row in _read_table(tmp_path / "run_manifest.tsv") if row["run_id"] == run["run_id"])
    assert canonical["status"] == "unknown_scheduler"
    assert canonical["scheduler_raw_state"] == "MISSING"


@pytest.mark.parametrize(
    ("identity_override", "execution"),
    [
        ({"scheduler_job_id": ""}, {"target": "local"}),
        ({"scheduler_submit_token": ""}, {"target": "local"}),
        ({"scheduler_direct_controller": ""}, {"target": "local"}),
        ({"target": ""}, {"target": "local"}),
        ({"target": "ssh", "host": ""}, {"target": "local"}),
        ({}, {"target": "local", "scheduler": {"direct_controller": True}}),
        ({"target": "ssh", "host": "baichuan3"}, {"target": "local"}),
    ],
)
def test_slurm_monitor_requires_complete_matching_canonical_identity_for_accounting_disabled_recovery(
    tmp_path: Path,
    monkeypatch,
    identity_override: dict[str, str],
    execution: dict,
):
    plan_dir, plan = _write_slurm_plan(tmp_path)
    run = plan["runs"][0]
    monkeypatch.setattr(
        managed_scheduler.slurm,
        "submit",
        lambda *_args, **_kwargs: slurm.JobIdentity("3880", "wuji-h20"),
    )
    hparam_runtime.launch_hparam_runs(plan_dir, dry_run=False)
    row = next(item for item in _read_table(tmp_path / "run_manifest.tsv") if item["run_id"] == run["run_id"])
    row.update(identity_override)
    Path(run["scheduler_result_path"]).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "scheduler_job_id": "3880",
                "scheduler_cluster": "wuji-h20",
                "scheduler_submit_token": row["scheduler_submit_token"],
                "node": "h20-bj-96",
                "exit_code": 0,
            }
        )
    )

    def run_command(_execution, argv, *, timeout):
        if argv[0] == "squeue":
            return subprocess.CompletedProcess([], 0, "", "")
        if argv[0] == "scontrol":
            return subprocess.CompletedProcess([], 1, "", "slurm_load_jobs error: Invalid job id specified")
        return subprocess.CompletedProcess([], 1, "", "Slurm accounting storage is disabled")

    monkeypatch.setattr(managed_scheduler.slurm, "run_command", run_command)
    monkeypatch.setattr(run_evidence, "runtime_artifacts", lambda _row: None)
    monkeypatch.setattr(run_evidence, "log_tail", lambda *_args, **_kwargs: "")

    observed = managed_scheduler.observe_slurm_run(plan_dir, execution, row)

    assert observed["status"] == ("unknown_scheduler" if row.get("scheduler_job_id") else "submitting")
    assert observed.get("scheduler_job_id", "") == row.get("scheduler_job_id", "")
    assert observed["scheduler_raw_state"] == "MISSING"


def test_slurm_monitor_keeps_ssh_transport_failure_unknown_when_output_mentions_disabled_accounting(
    tmp_path: Path,
    monkeypatch,
):
    plan_dir, plan = _write_slurm_plan(tmp_path)
    run = plan["runs"][0]
    monkeypatch.setattr(
        managed_scheduler.slurm,
        "submit",
        lambda *_args, **_kwargs: slurm.JobIdentity("3880", "wuji-h20"),
    )
    hparam_runtime.launch_hparam_runs(plan_dir, dry_run=False)
    row = next(item for item in _read_table(tmp_path / "run_manifest.tsv") if item["run_id"] == run["run_id"])
    row.update({"target": "ssh", "host": "baichuan3"})
    terminal = {
        "schema_version": 1,
        "scheduler_job_id": "3880",
        "scheduler_cluster": "wuji-h20",
        "scheduler_submit_token": row["scheduler_submit_token"],
        "node": "h20-bj-96",
        "exit_code": 0,
    }

    def read_output_texts(root, paths, *, remote=None):
        assert root == plan_dir
        assert paths == [row["scheduler_result_path"]]
        assert remote == "baichuan3"
        return {row["scheduler_result_path"]: json.dumps(terminal)}

    monkeypatch.setattr(managed_scheduler.exp_io, "read_managed_output_texts_at", read_output_texts)
    monkeypatch.setattr(run_evidence, "runtime_artifacts", lambda _row: None)
    monkeypatch.setattr(run_evidence, "log_tail", lambda *_args, **_kwargs: "")

    def run_command(_execution, argv, *, timeout):
        if argv[0] == "squeue":
            return subprocess.CompletedProcess([], 0, "", "")
        if argv[0] == "scontrol":
            return subprocess.CompletedProcess([], 1, "", "slurm_load_jobs error: Invalid job id specified")
        return subprocess.CompletedProcess(
            [],
            255,
            "",
            "ssh: connection closed\nSlurm accounting storage is disabled",
        )

    monkeypatch.setattr(managed_scheduler.slurm, "run_command", run_command)

    observed = managed_scheduler.observe_slurm_run(
        plan_dir,
        {"target": "ssh", "host": "baichuan3"},
        row,
    )

    assert observed["status"] == "unknown_scheduler"
    assert observed["scheduler_raw_state"] == "MISSING"
    assert "ssh: connection closed" in observed["scheduler_reason"]


def test_slurm_monitor_rejects_mismatched_terminal_sidecar_cluster_before_query(tmp_path: Path, monkeypatch):
    plan_dir, plan = _write_slurm_plan(tmp_path)
    run = plan["runs"][0]
    monkeypatch.setattr(
        managed_scheduler.slurm,
        "submit",
        lambda *_args, **_kwargs: slurm.JobIdentity("3880", "wuji-h20"),
    )
    hparam_runtime.launch_hparam_runs(plan_dir, dry_run=False)
    Path(run["scheduler_result_path"]).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "scheduler_job_id": "3880",
                "scheduler_cluster": "other-cluster",
                "scheduler_submit_token": run["scheduler_submit_token"],
                "node": "h20-bj-96",
                "exit_code": 0,
            }
        )
    )
    manifest_path = tmp_path / "run_manifest.tsv"
    before = manifest_path.read_bytes()
    monkeypatch.setattr(
        managed_scheduler.slurm,
        "run_command",
        lambda *_args, **_kwargs: pytest.fail("mismatched sidecar must fail before scheduler query"),
    )

    with pytest.raises(ValueError, match="terminal sidecar cluster differs"):
        hparam_runtime.monitor_hparam_runs(plan_dir)

    assert manifest_path.read_bytes() == before


def test_slurm_monitor_recovers_terminal_state_from_accounting_after_controller_purge(tmp_path: Path, monkeypatch):
    plan_dir, plan = _write_slurm_plan(tmp_path, direct_controller=True)
    run = plan["runs"][0]
    monkeypatch.setattr(
        managed_scheduler.slurm,
        "submit",
        lambda *_args, **_kwargs: slurm.JobIdentity("3880", "wuji-h20"),
    )
    hparam_runtime.launch_hparam_runs(plan_dir, dry_run=False)
    Path(run["scheduler_result_path"]).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "scheduler_job_id": "3880",
                "scheduler_cluster": "wuji-h20",
                "scheduler_submit_token": run["scheduler_submit_token"],
                "node": "h20-bj-96",
                "exit_code": 0,
            }
        )
    )
    scheduler_calls = []

    def run_command(_execution, argv, *, timeout):
        scheduler_calls.append(argv)
        if argv[0] in {"squeue", "scontrol"}:
            return subprocess.CompletedProcess([], 1, "", "slurm_load_jobs error: Invalid job id specified")
        assert argv[0] == "sacct"
        return subprocess.CompletedProcess(
            [],
            0,
            f"3880|COMPLETED|0:0|h20-bj-96|{run['scheduler_submit_token']}\n",
            "",
        )

    monkeypatch.setattr(managed_scheduler.slurm, "run_command", run_command)

    hparam_runtime.monitor_hparam_runs(plan_dir)

    canonical = next(row for row in _read_table(tmp_path / "run_manifest.tsv") if row["run_id"] == run["run_id"])
    assert canonical["status"] == "completed"
    assert canonical["scheduler_raw_state"] == "COMPLETED"
    assert canonical["scheduler_exit_code"] == "0"
    assert [argv[0] for argv in scheduler_calls] == ["squeue", "scontrol", "sacct"]
    assert all("--clusters=wuji-h20" not in argv for argv in scheduler_calls)
    assert "--duplicates" in scheduler_calls[-1]


@pytest.mark.parametrize("observation_source", ["controller", "accounting"])
@pytest.mark.parametrize("terminal_exit_code", [None, 0, 7])
def test_slurm_monitor_keeps_revoked_federation_sibling_unknown(
    tmp_path: Path,
    monkeypatch,
    observation_source: str,
    terminal_exit_code: int | None,
):
    plan_dir, plan = _write_slurm_plan(tmp_path)
    run = plan["runs"][0]
    token = run["scheduler_submit_token"]
    monkeypatch.setattr(
        managed_scheduler.slurm,
        "submit",
        lambda *_args, **_kwargs: slurm.JobIdentity("3880", "wuji-h20"),
    )
    hparam_runtime.launch_hparam_runs(plan_dir, dry_run=False)
    if terminal_exit_code is not None:
        Path(run["scheduler_result_path"]).write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "scheduler_job_id": "3880",
                    "scheduler_cluster": "wuji-h20",
                    "scheduler_submit_token": token,
                    "node": "h20-bj-96",
                    "exit_code": terminal_exit_code,
                }
            )
        )
    revoked = slurm.JobObservation("3880", "REVOKED", "Federated", "", token)
    accounting_calls = []
    if observation_source == "controller":
        monkeypatch.setattr(managed_scheduler.slurm, "active_jobs", lambda *_args, **_kwargs: [revoked])
        monkeypatch.setattr(
            managed_scheduler.slurm,
            "accounting_job",
            lambda *_args, **_kwargs: pytest.fail("controller REVOKED must not query accounting"),
        )
    else:
        monkeypatch.setattr(managed_scheduler.slurm, "active_jobs", lambda *_args, **_kwargs: [])
        monkeypatch.setattr(managed_scheduler.slurm, "show_job", lambda *_args, **_kwargs: None)

        def accounting_job(_execution, job_id, *, submit_token, cluster=None, timeout=10):
            accounting_calls.append((job_id, submit_token, cluster))
            return revoked

        monkeypatch.setattr(managed_scheduler.slurm, "accounting_job", accounting_job)

    hparam_runtime.monitor_hparam_runs(plan_dir)

    canonical = next(row for row in _read_table(tmp_path / "run_manifest.tsv") if row["run_id"] == run["run_id"])
    assert canonical["status"] == "unknown_scheduler"
    assert canonical["scheduler_job_id"] == "3880"
    assert canonical["scheduler_cluster"] == "wuji-h20"
    assert canonical["scheduler_raw_state"] == "REVOKED"
    assert canonical["scheduler_reason"] == (
        "Slurm reports REVOKED federation sibling state; sibling-cluster rebinding is unsupported. "
        "Scheduler reason: Federated"
    )
    assert canonical.get("scheduler_exit_code", "") == (
        str(terminal_exit_code) if terminal_exit_code is not None else ""
    )
    assert accounting_calls == ([("3880", token, "wuji-h20")] if observation_source == "accounting" else [])


def test_slurm_monitor_rejects_unauthenticated_accounting_without_terminalizing_run(tmp_path: Path, monkeypatch):
    plan_dir, plan = _write_slurm_plan(tmp_path)
    run = plan["runs"][0]
    monkeypatch.setattr(
        managed_scheduler.slurm,
        "submit",
        lambda *_args, **_kwargs: slurm.JobIdentity("3880", "wuji-h20"),
    )
    hparam_runtime.launch_hparam_runs(plan_dir, dry_run=False)
    Path(run["scheduler_result_path"]).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "scheduler_job_id": "3880",
                "scheduler_cluster": "wuji-h20",
                "scheduler_submit_token": run["scheduler_submit_token"],
                "node": "h20-bj-96",
                "exit_code": 0,
            }
        )
    )
    monkeypatch.setattr(managed_scheduler.slurm, "active_jobs", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(managed_scheduler.slurm, "show_job", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        managed_scheduler.slurm,
        "run_command",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 0, "3880|COMPLETED|0:0|h20-bj-96|other-token\n", ""),
    )
    manifest_path = tmp_path / "run_manifest.tsv"
    before = manifest_path.read_bytes()

    with pytest.raises(ValueError, match="exactly one authenticated allocation row"):
        hparam_runtime.monitor_hparam_runs(plan_dir)

    assert manifest_path.read_bytes() == before
    canonical = next(row for row in _read_table(manifest_path) if row["run_id"] == run["run_id"])
    assert canonical["status"] != "completed"


def test_slurm_monitor_fails_closed_when_job_disappears_without_sidecar(tmp_path: Path, monkeypatch):
    plan_dir, plan = _write_slurm_plan(tmp_path)
    run = plan["runs"][0]
    monkeypatch.setattr(
        managed_scheduler.slurm,
        "submit",
        lambda *_args, **_kwargs: slurm.JobIdentity("3880", "wuji-h20"),
    )
    hparam_runtime.launch_hparam_runs(plan_dir, dry_run=False)
    monkeypatch.setattr(managed_scheduler.slurm, "active_jobs", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(managed_scheduler.slurm, "show_job", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        managed_scheduler.slurm,
        "accounting_job",
        lambda _execution, job_id, **_kwargs: slurm.JobObservation(job_id, "COMPLETED", exit_code="0:0"),
    )

    hparam_runtime.monitor_hparam_runs(plan_dir)

    canonical = next(row for row in _read_table(tmp_path / "run_manifest.tsv") if row["run_id"] == run["run_id"])
    assert canonical["status"] == "unknown_scheduler"
    assert canonical["scheduler_raw_state"] == "COMPLETED"
    assert "missing the matching terminal sidecar" in canonical["scheduler_reason"]


def test_slurm_monitor_health_reports_queue_diagnostics_without_pid_or_gpu_probes(tmp_path: Path, monkeypatch):
    plan_dir, plan = _write_slurm_plan(tmp_path)
    run = plan["runs"][0]
    token = run["scheduler_submit_token"]
    monkeypatch.setattr(
        managed_scheduler.slurm,
        "submit",
        lambda *_args, **_kwargs: slurm.JobIdentity("3880", "wuji-h20"),
    )
    hparam_runtime.launch_hparam_runs(plan_dir, dry_run=False)
    monkeypatch.setattr(
        managed_scheduler.slurm,
        "active_jobs",
        lambda *_args, **_kwargs: [slurm.JobObservation("3880", "PENDING", "Resources", "", token)],
    )
    detail_calls = []
    monkeypatch.setattr(
        managed_scheduler.slurm,
        "show_job",
        lambda *_args, **_kwargs: detail_calls.append(True)
        or slurm.JobObservation(
            "3880",
            "PENDING",
            "Resources",
            "",
            token,
            details={"priority": "42", "nice": "0", "partition": "gpu", "time_limit": "01:00:00"},
        ),
    )
    monkeypatch.setattr(run_evidence, "log_age_seconds", lambda *_args: None)
    monkeypatch.setattr(
        run_evidence,
        "gpu_summary",
        lambda *_args, **_kwargs: pytest.fail("Slurm health must not probe host-global GPUs"),
    )
    monkeypatch.setattr(
        run_evidence,
        "read_process_identity",
        lambda *_args, **_kwargs: pytest.fail("Slurm health must not read PID identity"),
    )

    hparam_runtime.monitor_hparam_runs(plan_dir, health=False)
    without_health = next(row for row in _read_table(tmp_path / "run_manifest.tsv") if row["run_id"] == run["run_id"])
    assert "health_status" not in without_health
    assert detail_calls == []

    hparam_runtime.monitor_hparam_runs(plan_dir, health=True)
    canonical = next(row for row in _read_table(tmp_path / "run_manifest.tsv") if row["run_id"] == run["run_id"])
    assert detail_calls == [True]
    assert canonical["status"] == "queued"
    assert canonical["health_status"] == "scheduler_queued"
    assert canonical["scheduler_reason"] == "Resources"
    assert canonical["scheduler_priority"] == "42"
    assert canonical["scheduler_nice"] == "0"
    assert canonical["scheduler_partition"] == "gpu"
    assert int(canonical["scheduler_queue_age_seconds"]) >= 0


@pytest.mark.parametrize("health", [False, True])
def test_slurm_monitor_reuses_successful_controller_fallback(tmp_path: Path, monkeypatch, health: bool):
    plan_dir, plan = _write_slurm_plan(tmp_path)
    run = plan["runs"][0]
    token = run["scheduler_submit_token"]
    merge_run_manifest(
        tmp_path,
        [
            {
                "step_id": run["step_id"],
                "run_id": run["run_id"],
                "target": "local",
                "status": "queued",
                "scheduler_job_id": "3880",
                "scheduler_cluster": "wuji-h20",
            }
        ],
    )
    calls = []
    details = iter(
        [
            slurm.JobObservation("3880", "RUNNING", "", "node-a", token, details={"priority": "42"}),
            slurm.JobObservation("3880", "PENDING", "Resources", "node-b", token, details={"priority": "99"}),
        ]
    )
    monkeypatch.setattr(slurm, "active_jobs", lambda *_a, **_k: calls.append("squeue") or [])

    def show_job(*_args, **_kwargs):
        calls.append("scontrol")
        return next(details)

    monkeypatch.setattr(slurm, "show_job", show_job)
    monkeypatch.setattr(slurm, "accounting_job", lambda *_a, **_k: pytest.fail("controller already found the job"))

    hparam_runtime.monitor_hparam_runs(plan_dir, health=health)

    canonical = next(row for row in _read_table(tmp_path / "run_manifest.tsv") if row["run_id"] == run["run_id"])
    assert calls == ["squeue", "scontrol"]
    assert canonical["status"] == "running"
    assert canonical["scheduler_raw_state"] == "RUNNING"
    assert canonical["scheduler_node"] == "node-a"
    assert canonical["scheduler_priority"] == "42"
    if health:
        assert canonical["health_status"] == "scheduler_running"
        assert canonical["scheduler_health_error"] == ""
    else:
        assert "health_status" not in canonical


@pytest.mark.parametrize("detail_result", ["running", "missing", "error"])
def test_slurm_monitor_health_refreshes_controller_after_accounting(tmp_path: Path, monkeypatch, detail_result: str):
    plan_dir, plan = _write_slurm_plan(tmp_path)
    run = plan["runs"][0]
    token = run["scheduler_submit_token"]
    merge_run_manifest(
        tmp_path,
        [
            {
                "step_id": run["step_id"],
                "run_id": run["run_id"],
                "target": "local",
                "status": "queued",
                "scheduler_job_id": "3880",
                "scheduler_cluster": "wuji-h20",
            }
        ],
    )
    Path(run["scheduler_result_path"]).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "scheduler_job_id": "3880",
                "scheduler_cluster": "wuji-h20",
                "scheduler_submit_token": token,
                "exit_code": 0,
            }
        )
    )
    calls = []
    monkeypatch.setattr(slurm, "active_jobs", lambda *_a, **_k: calls.append("squeue") or [])

    def show_job(*_args, **_kwargs):
        calls.append("scontrol")
        if calls == ["squeue", "scontrol"] or detail_result == "missing":
            return None
        if detail_result == "error":
            raise RuntimeError("detail probe unavailable")
        return slurm.JobObservation("3880", "RUNNING", "", "node-current", token, details={"priority": "43"})

    def accounting_job(*_args, **_kwargs):
        calls.append("sacct")
        return slurm.JobObservation("3880", "COMPLETED", "", "node-accounting", token, exit_code="0:0")

    monkeypatch.setattr(slurm, "show_job", show_job)
    monkeypatch.setattr(slurm, "accounting_job", accounting_job)

    hparam_runtime.monitor_hparam_runs(plan_dir, health=True)

    canonical = next(row for row in _read_table(tmp_path / "run_manifest.tsv") if row["run_id"] == run["run_id"])
    assert calls == ["squeue", "scontrol", "sacct", "scontrol"]
    if detail_result == "running":
        assert canonical["status"] == "running"
        assert canonical["scheduler_raw_state"] == "RUNNING"
        assert canonical["scheduler_node"] == "node-current"
        assert canonical["scheduler_priority"] == "43"
        assert canonical["health_status"] == "scheduler_running"
        assert canonical["scheduler_health_error"] == ""
    else:
        assert canonical["status"] == "completed"
        assert canonical["scheduler_raw_state"] == "COMPLETED"
        assert canonical["scheduler_node"] == "node-accounting"
        assert canonical["health_status"] == "health_unknown"
        assert canonical["scheduler_health_error"] == (
            "detail probe unavailable" if detail_result == "error" else "Slurm job details are unavailable."
        )


def test_slurm_monitor_health_uses_allocation_start_and_preserves_lifecycle_on_detail_failure(
    tmp_path: Path, monkeypatch
):
    plan_dir, plan = _write_slurm_plan(tmp_path)
    run = plan["runs"][0]
    token = run["scheduler_submit_token"]
    monkeypatch.setattr(
        managed_scheduler.slurm,
        "submit",
        lambda *_args, **_kwargs: slurm.JobIdentity("3880", "wuji-h20"),
    )
    hparam_runtime.launch_hparam_runs(plan_dir, dry_run=False)
    started_at = "2026-08-21T00:00:00Z"
    Path(run["allocation_identity_path"]).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "scheduler_job_id": "3880",
                "scheduler_cluster": "wuji-h20",
                "scheduler_submit_token": token,
                "node": "h20-bj-96",
                "started_at": started_at,
            }
        )
        + "\n"
    )
    monkeypatch.setattr(
        managed_scheduler.slurm,
        "active_jobs",
        lambda *_args, **_kwargs: [slurm.JobObservation("3880", "RUNNING", "", "h20-bj-96", token)],
    )
    monkeypatch.setattr(
        managed_scheduler.slurm,
        "show_job",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("detail probe unavailable")),
    )
    monkeypatch.setattr(run_evidence, "log_age_seconds", lambda *_args: 12)

    hparam_runtime.monitor_hparam_runs(plan_dir, health=True)

    canonical = next(row for row in _read_table(tmp_path / "run_manifest.tsv") if row["run_id"] == run["run_id"])
    assert canonical["status"] == "running"
    assert canonical["health_status"] == "health_unknown"
    assert canonical["scheduler_health_error"] == "detail probe unavailable"
    assert canonical["scheduler_started_at"] == started_at
    assert canonical["scheduler_node"] == "h20-bj-96"
    assert canonical["log_age_seconds"] == "12"
    assert int(canonical["scheduler_allocation_age_seconds"]) >= 0


@pytest.mark.parametrize("submission_failure", ["timeout", "ssh255"])
def test_slurm_submission_uncertainty_reconciles_exact_submit_token(
    tmp_path: Path, monkeypatch, submission_failure: str
):
    plan_dir, plan = _write_slurm_plan(tmp_path)
    run = plan["runs"][0]

    def submit(*_args, **_kwargs):
        if submission_failure == "timeout":
            raise subprocess.TimeoutExpired("sbatch", 60)
        raise slurm.SlurmCommandError("submission", subprocess.CompletedProcess([], 255, "", "ssh: connection closed"))

    monkeypatch.setattr(slurm, "submit", submit)

    def active_jobs(_execution, *, job_id=None, submit_token=None, cluster=None, timeout=10):
        assert job_id is None
        assert submit_token == run["scheduler_submit_token"]
        assert cluster == "wuji-h20"
        return [slurm.JobObservation("3880", "PENDING", "Resources", "", submit_token)]

    monkeypatch.setattr(managed_scheduler.slurm, "active_jobs", active_jobs)

    hparam_runtime.launch_hparam_runs(plan_dir, dry_run=False)

    canonical = next(row for row in _read_table(tmp_path / "run_manifest.tsv") if row["run_id"] == run["run_id"])
    assert canonical["status"] == "queued"
    assert canonical["scheduler_job_id"] == "3880"
    assert canonical["scheduler_cluster"] == "wuji-h20"
    assert canonical["scheduler_reason"] == "Resources"


def test_slurm_submission_timeout_reconciles_revoked_without_resubmitting(tmp_path: Path, monkeypatch):
    plan_dir, plan = _write_slurm_plan(tmp_path)
    run = plan["runs"][0]
    submitted = []

    def timeout(*_args, **_kwargs):
        submitted.append(True)
        raise subprocess.TimeoutExpired("sbatch", 60)

    def active_jobs(_execution, *, job_id=None, submit_token=None, cluster=None, timeout=10):
        if job_id is None:
            assert submit_token == run["scheduler_submit_token"]
        else:
            assert job_id == "3880"
            assert submit_token is None
        assert cluster == "wuji-h20"
        return [slurm.JobObservation("3880", "REVOKED", "Sibling", "", run["scheduler_submit_token"])]

    monkeypatch.setattr(managed_scheduler.slurm, "submit", timeout)
    monkeypatch.setattr(managed_scheduler.slurm, "active_jobs", active_jobs)

    hparam_runtime.launch_hparam_runs(plan_dir, dry_run=False)
    hparam_runtime.launch_hparam_runs(plan_dir, dry_run=False)

    canonical = next(row for row in _read_table(tmp_path / "run_manifest.tsv") if row["run_id"] == run["run_id"])
    assert submitted == [True]
    assert canonical["status"] == "unknown_scheduler"
    assert canonical["scheduler_job_id"] == "3880"
    assert canonical["scheduler_cluster"] == "wuji-h20"
    assert canonical["scheduler_raw_state"] == "REVOKED"
    assert canonical["scheduler_reason"] == (
        "Slurm reports REVOKED federation sibling state; sibling-cluster rebinding is unsupported. "
        "Scheduler reason: Sibling"
    )


def test_slurm_submission_timeout_sidecar_still_waits_for_scheduler_terminal(tmp_path: Path, monkeypatch):
    plan_dir, plan = _write_slurm_plan(tmp_path)
    run = plan["runs"][0]
    token = run["scheduler_submit_token"]

    def submit(*_args, **_kwargs):
        Path(run["scheduler_result_path"]).write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "scheduler_job_id": "3880",
                    "scheduler_cluster": "wuji-h20",
                    "scheduler_submit_token": token,
                    "node": "h20-bj-96",
                    "exit_code": 0,
                }
            )
        )
        raise subprocess.TimeoutExpired("sbatch", 60)

    def active_jobs(_execution, *, job_id=None, submit_token=None, cluster=None, timeout=10):
        assert job_id == "3880"
        assert submit_token is None
        assert cluster == "wuji-h20"
        return [slurm.JobObservation("3880", "COMPLETING", "", "h20-bj-96", token)]

    monkeypatch.setattr(managed_scheduler.slurm, "submit", submit)
    monkeypatch.setattr(managed_scheduler.slurm, "active_jobs", active_jobs)

    hparam_runtime.launch_hparam_runs(plan_dir, dry_run=False)

    canonical = next(row for row in _read_table(tmp_path / "run_manifest.tsv") if row["run_id"] == run["run_id"])
    assert canonical["status"] == "running"
    assert canonical["scheduler_cluster"] == "wuji-h20"
    assert canonical["scheduler_raw_state"] == "COMPLETING"
    assert canonical["scheduler_exit_code"] == "0"


def test_slurm_submission_timeout_never_resubmits_unresolved_run(tmp_path: Path, monkeypatch):
    plan_dir, plan = _write_slurm_plan(tmp_path)
    run = plan["runs"][0]
    calls = []

    def timeout(*_args, **_kwargs):
        calls.append("submit")
        raise subprocess.TimeoutExpired("sbatch", 60)

    monkeypatch.setattr(managed_scheduler.slurm, "submit", timeout)

    def active_jobs(_execution, *, job_id=None, submit_token=None, cluster=None, timeout=10):
        assert job_id is None
        assert submit_token == run["scheduler_submit_token"]
        assert cluster == "wuji-h20"
        return []

    monkeypatch.setattr(managed_scheduler.slurm, "active_jobs", active_jobs)

    with pytest.raises(RuntimeError, match="outcome is uncertain"):
        hparam_runtime.launch_hparam_runs(plan_dir, dry_run=False)
    hparam_runtime.launch_hparam_runs(plan_dir, dry_run=False)

    canonical = next(row for row in _read_table(tmp_path / "run_manifest.tsv") if row["run_id"] == run["run_id"])
    assert calls == ["submit"]
    assert canonical["status"] == "submitting"
    assert canonical.get("scheduler_job_id", "") == ""
    assert canonical["scheduler_cluster"] == "wuji-h20"


@pytest.mark.parametrize("binding_source", ["queue", "allocation", "terminal"])
@pytest.mark.parametrize("launched_at", ["", "2026-01-01T00:00:00Z"])
def test_slurm_late_job_binding_records_launch_time_without_overwriting_existing(
    tmp_path: Path, monkeypatch, binding_source: str, launched_at: str
):
    plan_dir, plan = _write_slurm_plan(tmp_path)
    run = plan["runs"][0]
    row = {
        **run,
        "status": "submitting",
        "target": "local",
        "scheduler_job_id": "",
        "launched_at": launched_at,
    }
    token = run["scheduler_submit_token"]
    if binding_source == "allocation":
        Path(run["allocation_identity_path"]).write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "scheduler_job_id": "3880",
                    "scheduler_cluster": "wuji-h20",
                    "scheduler_submit_token": token,
                    "node": "h20-bj-96",
                    "started_at": "2026-08-21T00:00:00Z",
                }
            )
        )
    elif binding_source == "terminal":
        Path(run["scheduler_result_path"]).write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "scheduler_job_id": "3880",
                    "scheduler_cluster": "wuji-h20",
                    "scheduler_submit_token": token,
                    "node": "h20-bj-96",
                    "started_at": "2026-08-21T00:00:00Z",
                    "ended_at": "2026-08-21T00:01:00Z",
                    "exit_code": 0,
                }
            )
        )

    def active_jobs(_execution, *, job_id=None, submit_token=None, cluster=None, timeout=10):
        if binding_source == "queue":
            assert job_id is None
            assert submit_token == token
            assert cluster is None
        else:
            assert job_id == "3880"
            assert submit_token is None
            assert cluster is None
        return [slurm.JobObservation("3880", "RUNNING", "", "h20-bj-96", token)]

    monkeypatch.setattr(managed_scheduler.slurm, "active_jobs", active_jobs)
    monkeypatch.setattr(managed_scheduler, "utc_now", lambda: "2026-08-21T00:02:00Z")

    observed = managed_scheduler.observe_slurm_run(plan_dir, {"target": "local"}, row)

    assert observed["status"] == "running"
    assert observed["scheduler_job_id"] == "3880"
    assert observed.get("scheduler_cluster", "") == ""
    assert observed["launched_at"] == (launched_at or "2026-08-21T00:02:00Z")


@pytest.mark.parametrize("direct_controller", [False, True])
def test_hparam_stop_uses_scancel_for_slurm_run(tmp_path: Path, monkeypatch, direct_controller: bool):
    plan_dir, plan = _write_slurm_plan(tmp_path, direct_controller=direct_controller)
    run = plan["runs"][0]
    monkeypatch.setattr(
        managed_scheduler.slurm,
        "submit",
        lambda *_args, **_kwargs: slurm.JobIdentity("3880", "wuji-h20"),
    )
    hparam_runtime.launch_hparam_runs(plan_dir, dry_run=False)
    recipe_scheduler = plan["recipe"]["execution"]["scheduler"]
    if direct_controller:
        recipe_scheduler.pop("direct_controller")
    else:
        recipe_scheduler["direct_controller"] = True
    monkeypatch.setattr(run_artifacts, "read_hparam_plan", lambda _path: plan)
    cancelled = []
    monkeypatch.setattr(hparam_runtime, "utc_now", lambda: "2026-08-21T03:40:00Z")

    def cancel(execution, job_id, *, cluster=None):
        durable = next(row for row in _read_table(tmp_path / "run_manifest.tsv") if row["run_id"] == run["run_id"])
        assert durable["status"] == "stopping"
        assert durable["scheduler_job_id"] == "3880"
        assert durable["scheduler_cluster"] == "wuji-h20"
        assert durable["stop_requested_at"] == "2026-08-21T03:40:00Z"
        assert durable["stop_reason"] == "validation diverged"
        cancelled.append((execution, job_id, cluster))

    monkeypatch.setattr(slurm, "cancel", cancel)

    hparam_runtime.stop_hparam_run(plan_dir, run["run_id"], reason="validation diverged")

    canonical = next(row for row in _read_table(tmp_path / "run_manifest.tsv") if row["run_id"] == run["run_id"])
    execution = {"target": "local", "host": ""}
    if direct_controller:
        execution["scheduler"] = {"direct_controller": True}
    assert cancelled == [(execution, "3880", "wuji-h20")]
    assert canonical["status"] == "stopping"
    assert canonical["status"] not in hparam_runtime.TERMINAL_STATUSES
    assert canonical["stop_requested_at"] == "2026-08-21T03:40:00Z"
    assert canonical.get("stopped_at", "") == ""
    assert canonical["stop_reason"] == "validation diverged"
    assert _read_table(plan_dir / "run_status.tsv")[0] == canonical
    assert _read_table(plan_dir / "launch_manifest.tsv")[0] == canonical
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert [event["event_type"] for event in events].count("run_stop_requested") == 1
    assert [event["event_type"] for event in events].count("run_stopped") == 0

    with pytest.raises(ValueError, match="pending stop request"):
        hparam_runtime.stop_hparam_run(plan_dir, run["run_id"], reason="repeat request")

    assert cancelled == [(execution, "3880", "wuji-h20")]


@pytest.mark.parametrize(
    ("initial_status", "scheduler_state", "terminal_sidecar", "terminal_exit_code", "expected_status"),
    [
        ("queued", "PENDING", False, 0, "stopping"),
        ("running", "RUNNING", False, 0, "stopping"),
        ("running", "RUNNING", True, 143, "stopping"),
        ("running", "COMPLETING", True, 143, "stopping"),
        ("running", "SUSPENDED", True, 143, "stopping"),
        ("running", "STOPPED", True, 143, "stopping"),
        ("running", "REVOKED", False, 0, "unknown_scheduler"),
        ("running", "RESIZING", True, 143, "stopping"),
        ("running", "SIGNALING", True, 143, "stopping"),
        ("running", "STAGE_OUT", True, 143, "stopping"),
        ("queued", "CANCELLED", False, 0, "stopped"),
        ("running", "CANCELLED", False, 0, "stopped"),
        ("running", "CANCELLED", True, 143, "stopped"),
        ("unknown_scheduler", "CANCELLED", False, 0, "stopped"),
        ("running", "COMPLETED", True, 0, "completed"),
        ("running", "FAILED", True, 143, "failed"),
    ],
)
def test_slurm_stop_request_waits_for_matching_scheduler_cancellation(
    tmp_path: Path,
    monkeypatch,
    initial_status: str,
    scheduler_state: str,
    terminal_sidecar: bool,
    terminal_exit_code: int,
    expected_status: str,
):
    plan_dir, plan = _write_slurm_plan(tmp_path)
    run = plan["runs"][0]
    token = run["scheduler_submit_token"]
    monkeypatch.setattr(
        managed_scheduler.slurm,
        "submit",
        lambda *_args, **_kwargs: slurm.JobIdentity("3880", "wuji-h20"),
    )
    hparam_runtime.launch_hparam_runs(plan_dir, dry_run=False)
    if initial_status == "running":
        merge_run_manifest(
            tmp_path,
            [{"step_id": run["step_id"], "run_id": run["run_id"], "status": "running"}],
        )
    monkeypatch.setattr(slurm, "cancel", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(hparam_runtime, "utc_now", lambda: "2026-08-21T03:40:00Z")
    hparam_runtime.stop_hparam_run(plan_dir, run["run_id"], reason="validation diverged")
    if initial_status == "unknown_scheduler":
        requested = next(row for row in _read_table(tmp_path / "run_manifest.tsv") if row["run_id"] == run["run_id"])
        merge_run_manifest(tmp_path, [{**requested, "status": "unknown_scheduler"}])
    if terminal_sidecar:
        Path(run["scheduler_result_path"]).write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "scheduler_job_id": "3880",
                    "scheduler_cluster": "wuji-h20",
                    "scheduler_submit_token": token,
                    "node": "h20-bj-96",
                    "exit_code": terminal_exit_code,
                }
            )
        )
    monkeypatch.setattr(
        managed_scheduler.slurm,
        "active_jobs",
        lambda *_args, **_kwargs: [slurm.JobObservation("3880", scheduler_state, "", "h20-bj-96", token)],
    )
    monkeypatch.setattr(managed_scheduler, "utc_now", lambda: "2026-08-21T03:41:00Z")

    hparam_runtime.monitor_hparam_runs(plan_dir)

    canonical = next(row for row in _read_table(tmp_path / "run_manifest.tsv") if row["run_id"] == run["run_id"])
    assert canonical["status"] == expected_status
    assert canonical["scheduler_raw_state"] == scheduler_state
    assert canonical["stop_requested_at"] == "2026-08-21T03:40:00Z"
    assert canonical["stop_reason"] == "validation diverged"
    assert canonical.get("stopped_at", "") == ("2026-08-21T03:41:00Z" if expected_status == "stopped" else "")
    if scheduler_state == "REVOKED":
        assert canonical["scheduler_job_id"] == "3880"
        assert canonical["scheduler_cluster"] == "wuji-h20"
        assert canonical["scheduler_reason"] == (
            "Slurm reports REVOKED federation sibling state; sibling-cluster rebinding is unsupported."
        )
    events_path = tmp_path / "events.jsonl"
    events = [json.loads(line) for line in events_path.read_text().splitlines()]
    assert [event["event_type"] for event in events].count("run_stop_requested") == 1
    assert [event["event_type"] for event in events].count("run_stopped") == (1 if expected_status == "stopped" else 0)
    if expected_status == "stopped":
        hparam_runtime.monitor_hparam_runs(plan_dir)
        repeated_events = [json.loads(line) for line in events_path.read_text().splitlines()]
        assert [event["event_type"] for event in repeated_events].count("run_stop_requested") == 1
        assert [event["event_type"] for event in repeated_events].count("run_stopped") == 1


def test_slurm_stop_request_uses_accounting_after_controller_purges_cancelled_job(tmp_path: Path, monkeypatch):
    plan_dir, plan = _write_slurm_plan(tmp_path, direct_controller=True)
    run = plan["runs"][0]
    monkeypatch.setattr(
        managed_scheduler.slurm,
        "submit",
        lambda *_args, **_kwargs: slurm.JobIdentity("3880", "wuji-h20"),
    )
    hparam_runtime.launch_hparam_runs(plan_dir, dry_run=False)
    monkeypatch.setattr(slurm, "cancel", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(hparam_runtime, "utc_now", lambda: "2026-08-21T03:40:00Z")
    hparam_runtime.stop_hparam_run(plan_dir, run["run_id"], reason="validation diverged")
    scheduler_calls = []

    def run_command(_execution, argv, *, timeout):
        scheduler_calls.append(argv)
        if argv[0] in {"squeue", "scontrol"}:
            return subprocess.CompletedProcess([], 1, "", "slurm_load_jobs error: Invalid job id specified")
        assert argv[0] == "sacct"
        return subprocess.CompletedProcess(
            [],
            0,
            f"3880|CANCELLED|0:15|h20-bj-96|{run['scheduler_submit_token']}\n",
            "",
        )

    monkeypatch.setattr(managed_scheduler.slurm, "run_command", run_command)
    monkeypatch.setattr(managed_scheduler, "utc_now", lambda: "2026-08-21T03:41:00Z")

    hparam_runtime.monitor_hparam_runs(plan_dir)

    canonical = next(row for row in _read_table(tmp_path / "run_manifest.tsv") if row["run_id"] == run["run_id"])
    assert canonical["status"] == "stopped"
    assert canonical["scheduler_raw_state"] == "CANCELLED"
    assert canonical["stopped_at"] == "2026-08-21T03:41:00Z"
    assert canonical["stop_requested_at"] == "2026-08-21T03:40:00Z"
    assert canonical["stop_reason"] == "validation diverged"
    assert [argv[0] for argv in scheduler_calls] == ["squeue", "scontrol", "sacct"]
    assert all("--clusters=wuji-h20" not in argv for argv in scheduler_calls)
    assert "--duplicates" in scheduler_calls[-1]


def test_slurm_stop_request_rejects_blank_accounting_comment_without_terminalizing_run(tmp_path: Path, monkeypatch):
    plan_dir, plan = _write_slurm_plan(tmp_path)
    run = plan["runs"][0]
    monkeypatch.setattr(
        managed_scheduler.slurm,
        "submit",
        lambda *_args, **_kwargs: slurm.JobIdentity("3880", "wuji-h20"),
    )
    hparam_runtime.launch_hparam_runs(plan_dir, dry_run=False)
    monkeypatch.setattr(slurm, "cancel", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(hparam_runtime, "utc_now", lambda: "2026-08-21T03:40:00Z")
    hparam_runtime.stop_hparam_run(plan_dir, run["run_id"], reason="validation diverged")
    monkeypatch.setattr(managed_scheduler.slurm, "active_jobs", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(managed_scheduler.slurm, "show_job", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        managed_scheduler.slurm,
        "run_command",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 0, "3880|CANCELLED|0:0|h20-bj-96|\n", ""),
    )
    manifest_path = tmp_path / "run_manifest.tsv"
    before = manifest_path.read_bytes()

    with pytest.raises(ValueError, match="exactly one authenticated allocation row"):
        hparam_runtime.monitor_hparam_runs(plan_dir)

    assert manifest_path.read_bytes() == before
    canonical = next(row for row in _read_table(manifest_path) if row["run_id"] == run["run_id"])
    assert canonical["status"] == "stopping"
    assert canonical.get("stopped_at", "") == ""


def test_slurm_stop_failure_preserves_retriable_request_state(tmp_path: Path, monkeypatch):
    plan_dir, plan = _write_slurm_plan(tmp_path)
    run = plan["runs"][0]
    monkeypatch.setattr(
        managed_scheduler.slurm,
        "submit",
        lambda *_args, **_kwargs: slurm.JobIdentity("3880", "wuji-h20"),
    )
    hparam_runtime.launch_hparam_runs(plan_dir, dry_run=False)
    monkeypatch.setattr(hparam_runtime, "utc_now", lambda: "2026-08-21T03:40:00Z")
    monkeypatch.setattr(slurm, "cancel", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("cancel failed")))

    with pytest.raises(RuntimeError, match="cancel failed"):
        hparam_runtime.stop_hparam_run(plan_dir, run["run_id"], reason="validation diverged")

    canonical = next(row for row in _read_table(tmp_path / "run_manifest.tsv") if row["run_id"] == run["run_id"])
    assert canonical["status"] == "stopping"
    assert canonical["stop_requested_at"] == "2026-08-21T03:40:00Z"
    assert canonical["stop_reason"] == "validation diverged"
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert [event["event_type"] for event in events].count("run_stop_requested") == 1

    cancelled = []
    monkeypatch.setattr(slurm, "cancel", lambda *_args, **_kwargs: cancelled.append(True))

    hparam_runtime.stop_hparam_run(plan_dir, run["run_id"], reason="validation diverged")

    retried = next(row for row in _read_table(tmp_path / "run_manifest.tsv") if row["run_id"] == run["run_id"])
    assert cancelled == [True]
    assert retried["stop_requested_at"] == "2026-08-21T03:40:00Z"
    assert _read_table(plan_dir / "run_status.tsv")[0] == retried
    assert _read_table(plan_dir / "launch_manifest.tsv")[0] == retried
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert [event["event_type"] for event in events].count("run_stop_requested") == 1


def test_slurm_stop_intent_merge_failure_prevents_scancel(tmp_path: Path, monkeypatch):
    plan_dir, plan = _write_slurm_plan(tmp_path)
    run = plan["runs"][0]
    monkeypatch.setattr(
        managed_scheduler.slurm,
        "submit",
        lambda *_args, **_kwargs: slurm.JobIdentity("3880", "wuji-h20"),
    )
    hparam_runtime.launch_hparam_runs(plan_dir, dry_run=False)
    before = (tmp_path / "run_manifest.tsv").read_bytes()
    cancelled = []
    monkeypatch.setattr(slurm, "cancel", lambda *_args, **_kwargs: cancelled.append(True))
    monkeypatch.setattr(
        hparam_runtime,
        "merge_run_manifest",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("intent merge failed")),
    )

    with pytest.raises(RuntimeError, match="intent merge failed"):
        hparam_runtime.stop_hparam_run(plan_dir, run["run_id"], reason="validation diverged")

    assert cancelled == []
    assert (tmp_path / "run_manifest.tsv").read_bytes() == before


def test_slurm_stop_interruption_after_intent_commit_remains_recoverable(tmp_path: Path, monkeypatch):
    plan_dir, plan = _write_slurm_plan(tmp_path)
    run = plan["runs"][0]
    monkeypatch.setattr(
        managed_scheduler.slurm,
        "submit",
        lambda *_args, **_kwargs: slurm.JobIdentity("3880", "wuji-h20"),
    )
    hparam_runtime.launch_hparam_runs(plan_dir, dry_run=False)
    monkeypatch.setattr(hparam_runtime, "utc_now", lambda: "2026-08-21T03:40:00Z")
    monkeypatch.setattr(slurm, "cancel", lambda *_args, **_kwargs: (_ for _ in ()).throw(KeyboardInterrupt()))

    with pytest.raises(KeyboardInterrupt):
        hparam_runtime.stop_hparam_run(plan_dir, run["run_id"], reason="validation diverged")

    canonical = next(row for row in _read_table(tmp_path / "run_manifest.tsv") if row["run_id"] == run["run_id"])
    assert canonical["status"] == "stopping"
    assert canonical["stop_requested_at"] == "2026-08-21T03:40:00Z"
    assert canonical["stop_reason"] == "validation diverged"


def test_slurm_stop_post_cancel_projection_failure_keeps_canonical_intent(tmp_path: Path, monkeypatch):
    plan_dir, plan = _write_slurm_plan(tmp_path)
    run = plan["runs"][0]
    token = run["scheduler_submit_token"]
    monkeypatch.setattr(
        managed_scheduler.slurm,
        "submit",
        lambda *_args, **_kwargs: slurm.JobIdentity("3880", "wuji-h20"),
    )
    hparam_runtime.launch_hparam_runs(plan_dir, dry_run=False)
    cancelled = []
    monkeypatch.setattr(slurm, "cancel", lambda *_args, **_kwargs: cancelled.append(True))
    real_write_rows = hparam_runtime.write_rows
    monkeypatch.setattr(
        hparam_runtime,
        "write_rows",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("projection write failed")),
    )

    with pytest.raises(RuntimeError, match="projection write failed"):
        hparam_runtime.stop_hparam_run(plan_dir, run["run_id"], reason="validation diverged")

    canonical = next(row for row in _read_table(tmp_path / "run_manifest.tsv") if row["run_id"] == run["run_id"])
    assert cancelled == [True]
    assert canonical["status"] == "stopping"
    assert canonical["stop_requested_at"]
    assert canonical["stop_reason"] == "validation diverged"

    monkeypatch.setattr(hparam_runtime, "write_rows", real_write_rows)
    monkeypatch.setattr(
        managed_scheduler.slurm,
        "active_jobs",
        lambda *_args, **_kwargs: [slurm.JobObservation("3880", "CANCELLED", "", "h20-bj-96", token)],
    )

    hparam_runtime.monitor_hparam_runs(plan_dir)

    recovered = next(row for row in _read_table(tmp_path / "run_manifest.tsv") if row["run_id"] == run["run_id"])
    assert recovered["status"] == "stopped"


@pytest.mark.parametrize("stale_status", ["queued", "running", "unknown_scheduler"])
def test_slurm_monitor_preserves_concurrent_stop_intent(tmp_path: Path, monkeypatch, stale_status: str):
    plan_dir, plan = _write_slurm_plan(tmp_path)
    run = plan["runs"][0]
    monkeypatch.setattr(
        managed_scheduler.slurm,
        "submit",
        lambda *_args, **_kwargs: slurm.JobIdentity("3880", "wuji-h20"),
    )
    hparam_runtime.launch_hparam_runs(plan_dir, dry_run=False)
    merge_run_manifest(
        tmp_path,
        [{"step_id": run["step_id"], "run_id": run["run_id"], "status": "running"}],
    )
    monkeypatch.setattr(slurm, "cancel", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(hparam_runtime, "utc_now", lambda: "2026-08-21T03:40:00Z")

    def observe_stale_row(_root, _execution, row, *, health=False, monitor_context=None):
        assert row["status"] == "running"
        assert row.get("stop_requested_at", "") == ""
        hparam_runtime.stop_hparam_run(plan_dir, run["run_id"], reason="validation diverged")
        return {
            **row,
            "status": stale_status,
            "stop_requested_at": "",
            "stop_reason": "",
            "scheduler_raw_state": "RUNNING",
        }

    monkeypatch.setattr(managed_scheduler, "observe_slurm_run", observe_stale_row)

    hparam_runtime.monitor_hparam_runs(plan_dir)

    canonical = next(row for row in _read_table(tmp_path / "run_manifest.tsv") if row["run_id"] == run["run_id"])
    assert canonical["status"] == "stopping"
    assert canonical["stop_requested_at"] == "2026-08-21T03:40:00Z"
    assert canonical["stop_reason"] == "validation diverged"
    assert canonical["scheduler_raw_state"] == "RUNNING"
    assert _read_table(plan_dir / "run_status.tsv")[0] == canonical
    launch = _read_table(plan_dir / "launch_manifest.tsv")[0]
    assert launch["status"] == "stopping"
    assert launch["stop_requested_at"] == "2026-08-21T03:40:00Z"
    assert launch["stop_reason"] == "validation diverged"
