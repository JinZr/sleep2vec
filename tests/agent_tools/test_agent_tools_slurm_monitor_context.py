from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest
from test_agent_tools_hparam_runtime import _read_table, _write_slurm_plan
from test_agent_tools_hparam_runtime import _stub_execution_snapshot_preflight  # noqa: F401

from agent_tools import experiments, hparam_runtime, managed_scheduler, run_evidence, slurm
from agent_tools.experiment_workspace import merge_run_manifest


@pytest.fixture(autouse=True)
def _isolate_execution_evidence(monkeypatch):
    read_outputs = managed_scheduler.exp_io.read_managed_output_texts_at
    monkeypatch.setattr(
        managed_scheduler.exp_io,
        "read_managed_output_texts_at",
        lambda owner, paths, remote=None: (
            {str(path): None for path in paths} if remote else read_outputs(owner, paths)
        ),
    )
    monkeypatch.setattr(run_evidence, "runtime_artifacts", lambda _row: ("", {}, []))
    monkeypatch.setattr(run_evidence, "log_tail", lambda *_args: "")
    monkeypatch.setattr(run_evidence, "log_age_seconds", lambda *_args: None)
    monkeypatch.setattr(slurm, "submit", lambda *_a, **_k: pytest.fail("monitor tests must not submit"))
    monkeypatch.setattr(slurm, "cancel", lambda *_a, **_k: pytest.fail("monitor tests must not cancel"))


def _bound_rows(tmp_path: Path, count: int = 2) -> list[dict]:
    return [
        {
            "step_id": "train-model",
            "run_id": f"run-{index:03d}",
            "scheduler_type": "slurm",
            "scheduler_job_id": str(3880 + index),
            "scheduler_cluster": "cluster-a",
            "scheduler_submit_token": f"token-{index}",
            "scheduler_direct_controller": "false",
            "scheduler_result_path": str(tmp_path / f"terminal-{index}.json"),
            "allocation_identity_path": str(tmp_path / f"allocation-{index}.json"),
            "target": "local",
            "host": "",
            "status": "queued",
        }
        for index in range(count)
    ]


def _execution(row: dict) -> dict:
    execution = {"target": row["target"]}
    if row["target"] == "ssh":
        execution["host"] = row["host"]
    if row["scheduler_direct_controller"] == "true":
        execution["scheduler"] = {"direct_controller": True}
    return execution


def _queue_output(rows: list[dict], state: str = "RUNNING") -> str:
    return "".join(
        f"{row['scheduler_job_id']}|{state}||node-{row['run_id']}|{row['scheduler_submit_token']}\n" for row in rows
    )


def _seed_plan(tmp_path: Path, count: int = 2) -> tuple[Path, list[dict]]:
    plan_dir, plan = _write_slurm_plan(tmp_path, run_count=count)
    merge_run_manifest(
        tmp_path,
        [
            {
                "step_id": run["step_id"],
                "run_id": run["run_id"],
                "target": "local",
                "status": "queued",
                "scheduler_job_id": str(3880 + index),
                "scheduler_cluster": "cluster-a",
            }
            for index, run in enumerate(plan["runs"])
        ],
    )
    run_ids = {run["run_id"] for run in plan["runs"]}
    return plan_dir, [row for row in _read_table(tmp_path / "run_manifest.tsv") if row["run_id"] in run_ids]


@pytest.mark.parametrize("count", [1, 6, 12, 50])
@pytest.mark.parametrize("health", [False, True])
def test_public_hparam_monitor_shares_exact_queue_hits(tmp_path: Path, monkeypatch, count: int, health: bool):
    plan_dir, rows = _seed_plan(tmp_path, count)
    by_job = {row["scheduler_job_id"]: row for row in rows}
    calls = []

    def run_command(_execution, argv, *, timeout):
        calls.append(argv)
        if argv[0] == "squeue":
            queried = [by_job[job_id] for job_id in argv[-1].split(",")]
            return subprocess.CompletedProcess(argv, 0, _queue_output(queried), "")
        assert argv[0] == "scontrol"
        row = by_job[argv[-1]]
        output = f"JobId={argv[-1]} JobState=RUNNING Comment={row['scheduler_submit_token']} Priority=42"
        return subprocess.CompletedProcess(argv, 0, output, "")

    monkeypatch.setattr(slurm, "run_command", run_command)

    hparam_runtime.monitor_hparam_runs(plan_dir, health=health)

    queue_calls = [argv for argv in calls if argv[0] == "squeue"]
    assert len(queue_calls) == 1
    assert queue_calls[0][-2:] == ["--jobs", ",".join(sorted(by_job))]
    assert len(calls) == 1 + (count if health else 0)
    canonical = [row for row in _read_table(tmp_path / "run_manifest.tsv") if row["scheduler_job_id"]]
    assert {row["status"] for row in canonical} == {"running"}
    assert _read_table(plan_dir / "run_status.tsv") == canonical


@pytest.mark.parametrize("entrypoint", ["hparam", "experiment"])
def test_public_monitor_rebuilds_snapshot_between_calls(tmp_path: Path, monkeypatch, entrypoint: str):
    plan_dir, rows = _seed_plan(tmp_path)
    by_job = {row["scheduler_job_id"]: row for row in rows}
    calls = []
    state = "PENDING"

    def run_command(_execution, argv, *, timeout):
        calls.append(argv)
        if argv[0] == "squeue":
            return subprocess.CompletedProcess(argv, 0, _queue_output(rows, state), "")
        assert argv[0] == "scontrol"
        output = f"JobId={argv[-1]} JobState={state} Comment={by_job[argv[-1]]['scheduler_submit_token']}"
        return subprocess.CompletedProcess(argv, 0, output, "")

    monkeypatch.setattr(slurm, "run_command", run_command)
    for state in ("PENDING", "RUNNING"):
        if entrypoint == "hparam":
            hparam_runtime.monitor_hparam_runs(plan_dir)
        else:
            experiments.monitor_experiment(tmp_path)
        assert {
            row["scheduler_raw_state"] for row in _read_table(tmp_path / "run_manifest.tsv") if row["scheduler_job_id"]
        } == {state}

    assert [argv[-1] for argv in calls if argv[0] == "squeue"] == ["3880,3881", "3880,3881"]


def test_hparam_continuous_monitor_rebuilds_snapshot_each_round(tmp_path: Path, monkeypatch):
    plan_dir, rows = _seed_plan(tmp_path)
    for row in rows:
        Path(row["scheduler_result_path"]).write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    **{key: row[key] for key in ("scheduler_job_id", "scheduler_cluster", "scheduler_submit_token")},
                    "exit_code": 0,
                }
            )
        )
    calls = []

    def run_command(_execution, argv, *, timeout):
        calls.append(argv)
        assert argv[0] == "squeue"
        return subprocess.CompletedProcess(
            argv, 0, _queue_output(rows, "PENDING" if len(calls) == 1 else "COMPLETED"), ""
        )

    sleeps = []
    monkeypatch.setattr(slurm, "run_command", run_command)
    monkeypatch.setattr(hparam_runtime.time, "sleep", lambda seconds: sleeps.append(seconds))

    hparam_runtime.monitor_hparam_runs(plan_dir, once=False, poll_seconds=0.1)

    assert len(calls) == 2
    assert sleeps == [0.1]
    assert {row["status"] for row in _read_table(tmp_path / "run_manifest.tsv") if row["scheduler_job_id"]} == {
        "completed"
    }


def test_context_separates_same_job_ids_by_frozen_route(tmp_path: Path, monkeypatch):
    routes = [
        ("local", "", "cluster-a", "false"),
        ("local", "", "cluster-b", "false"),
        ("ssh", "host-a", "cluster-a", "false"),
        ("ssh", "host-a", "cluster-a", "true"),
        ("ssh", "host-b", "cluster-a", "false"),
    ]
    rows = _bound_rows(tmp_path, 2 * len(routes))
    grouped = {}
    for index, row in enumerate(rows):
        target, host, cluster, topology = routes[index // 2]
        row.update(target=target, host=host, scheduler_cluster=cluster, scheduler_direct_controller=topology)
        row["scheduler_job_id"] = str(3880 + index % 2)
        key = (target, host, "" if topology == "true" else cluster)
        grouped.setdefault(key, []).append(row)
    calls = []

    def run_command(execution, argv, *, timeout):
        assert argv[0] == "squeue"
        cluster = next((arg.split("=", 1)[1] for arg in argv if arg.startswith("--clusters=")), "")
        key = (execution["target"], execution.get("host", ""), cluster)
        calls.append((key, argv))
        return subprocess.CompletedProcess(argv, 0, _queue_output(grouped[key]), "")

    monkeypatch.setattr(slurm, "run_command", run_command)
    context = managed_scheduler.SlurmMonitorContext(rows, owner_dir=tmp_path)
    for row in rows:
        observed = managed_scheduler.observe_slurm_run(tmp_path, _execution(row), row, monitor_context=context)
        assert observed["scheduler_node"] == f"node-{row['run_id']}"
        assert observed["scheduler_cluster"] == row["scheduler_cluster"]
    assert len(calls) == len(routes)
    assert all(argv[-2:] == ["--jobs", "3880,3881"] for _key, argv in calls)


def test_context_does_not_batch_duplicate_input_job_ids(tmp_path: Path, monkeypatch):
    rows = _bound_rows(tmp_path, 1) * 2
    calls = []

    def run_command(_execution, argv, *, timeout):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, _queue_output(rows[:1]), "")

    monkeypatch.setattr(slurm, "run_command", run_command)
    context = managed_scheduler.SlurmMonitorContext(rows, owner_dir=tmp_path)
    assert calls == []
    for row in rows:
        managed_scheduler.observe_slurm_run(tmp_path, _execution(row), row, monitor_context=context)
    assert [argv[-2:] for argv in calls] == [["--jobs", "3880"], ["--jobs", "3880"]]


def test_shared_queue_hit_still_requires_exact_submit_token(tmp_path: Path, monkeypatch):
    rows = _bound_rows(tmp_path)
    calls = []

    def run_command(_execution, argv, *, timeout):
        calls.append(argv)
        output = _queue_output([{**rows[0], "scheduler_submit_token": "foreign-token"}, rows[1]])
        return subprocess.CompletedProcess(argv, 0, output, "")

    monkeypatch.setattr(slurm, "run_command", run_command)
    context = managed_scheduler.SlurmMonitorContext(rows, owner_dir=tmp_path)

    with pytest.raises(ValueError, match="comment differs from the frozen submit token"):
        managed_scheduler.observe_slurm_run(tmp_path, _execution(rows[0]), rows[0], monitor_context=context)

    assert [argv[-1] for argv in calls] == ["3880,3881"]
    assert rows[0]["scheduler_submit_token"] == "token-0"


@pytest.mark.parametrize("failure", ["cluster", "json"])
def test_context_is_lazy_until_after_sidecar_validation(tmp_path: Path, monkeypatch, failure: str):
    rows = _bound_rows(tmp_path)
    terminal = {
        "schema_version": 1,
        "scheduler_job_id": rows[0]["scheduler_job_id"],
        "scheduler_cluster": "wrong-cluster",
        "scheduler_submit_token": rows[0]["scheduler_submit_token"],
        "exit_code": 0,
    }
    Path(rows[0]["scheduler_result_path"]).write_text(json.dumps(terminal) if failure == "cluster" else "{")
    monkeypatch.setattr(slurm, "run_command", lambda *_a, **_k: pytest.fail("sidecar must fail before any query"))

    context = managed_scheduler.SlurmMonitorContext(rows, owner_dir=tmp_path)

    with pytest.raises(ValueError, match="cluster differs|not valid JSON"):
        managed_scheduler.observe_slurm_run(tmp_path, _execution(rows[0]), rows[0], monitor_context=context)


@pytest.mark.parametrize("failure", ["partial", "ssh255", "timeout", "malformed", "duplicate"])
def test_bad_batch_disables_only_its_group_and_requeries_every_job(tmp_path: Path, monkeypatch, failure: str):
    rows = _bound_rows(tmp_path, 4)
    for row in rows[2:]:
        row["scheduler_cluster"] = "cluster-b"
    by_job = {row["scheduler_job_id"]: row for row in rows}
    calls = []

    def run_command(_execution, argv, *, timeout):
        calls.append(argv)
        assert argv[0] == "squeue"
        queried = [by_job[job_id] for job_id in argv[-1].split(",")]
        if len(queried) > 1 and queried[0]["scheduler_cluster"] == "cluster-a":
            if failure == "timeout":
                raise subprocess.TimeoutExpired(argv, timeout)
            if failure == "partial":
                return subprocess.CompletedProcess(argv, 1, _queue_output(queried[:1]), "Invalid job id specified")
            if failure == "ssh255":
                return subprocess.CompletedProcess(argv, 255, _queue_output(queried[:1]), "ssh: connection closed")
            output = _queue_output(queried[:1]) + (
                "broken row\n" if failure == "malformed" else _queue_output(queried[:1])
            )
            return subprocess.CompletedProcess(argv, 0, output, "")
        return subprocess.CompletedProcess(argv, 0, _queue_output(queried), "")

    monkeypatch.setattr(slurm, "run_command", run_command)
    context = managed_scheduler.SlurmMonitorContext(rows, owner_dir=tmp_path)
    observed = [
        managed_scheduler.observe_slurm_run(tmp_path, _execution(row), row, monitor_context=context) for row in rows
    ]

    assert [argv[-1] for argv in calls] == ["3880,3881", "3880", "3881", "3882,3883"]
    assert {row["status"] for row in observed} == {"running"}


def test_failed_batch_preserves_each_exact_query_failure(tmp_path: Path, monkeypatch):
    rows = _bound_rows(tmp_path)
    calls = []

    def run_command(_execution, argv, *, timeout):
        calls.append(argv)
        assert argv[0] == "squeue"
        if "," in argv[-1]:
            return subprocess.CompletedProcess(argv, 255, _queue_output(rows[:1]), "batch connection closed")
        if argv[-1] == "3880":
            return subprocess.CompletedProcess(argv, 255, "", "exact connection closed")
        return subprocess.CompletedProcess(argv, 1, "", "Access denied")

    monkeypatch.setattr(slurm, "run_command", run_command)
    context = managed_scheduler.SlurmMonitorContext(rows, owner_dir=tmp_path)
    observed = [
        managed_scheduler.observe_slurm_run(tmp_path, _execution(row), row, monitor_context=context) for row in rows
    ]

    assert [argv[-1] for argv in calls] == ["3880,3881", "3880", "3881"]
    assert {row["status"] for row in observed} == {"unknown_scheduler"}
    assert "exact connection closed" in observed[0]["scheduler_reason"]
    assert "Access denied" in observed[1]["scheduler_reason"]
    assert all("batch connection closed" not in row["scheduler_reason"] for row in observed)


@pytest.mark.parametrize("missing_result", ["reappeared", "vanished"])
def test_successful_batch_missing_job_keeps_per_job_query(tmp_path: Path, monkeypatch, missing_result: str):
    rows = _bound_rows(tmp_path)
    calls = []

    def run_command(_execution, argv, *, timeout):
        calls.append(argv)
        if argv[0] == "squeue":
            if "," in argv[-1]:
                return subprocess.CompletedProcess(argv, 0, _queue_output(rows[:1]), "")
            assert argv[-1] == "3881"
            if missing_result == "reappeared":
                return subprocess.CompletedProcess(argv, 0, _queue_output(rows[1:]), "")
            return subprocess.CompletedProcess(argv, 1, "", "Invalid job id specified")
        if argv[0] == "scontrol":
            return subprocess.CompletedProcess(argv, 1, "", "Invalid job id specified")
        assert argv[0] == "sacct"
        return subprocess.CompletedProcess(argv, 1, "", "Slurm accounting storage is disabled")

    monkeypatch.setattr(slurm, "run_command", run_command)
    context = managed_scheduler.SlurmMonitorContext(rows, owner_dir=tmp_path)
    observed = [
        managed_scheduler.observe_slurm_run(tmp_path, _execution(row), row, monitor_context=context) for row in rows
    ]

    assert [argv[-1] for argv in calls if argv[0] == "squeue"] == ["3880,3881", "3881"]
    assert observed[0]["status"] == "running"
    assert observed[1]["status"] == ("running" if missing_result == "reappeared" else "unknown_scheduler")
    assert len(calls) == (2 if missing_result == "reappeared" else 4)


@pytest.mark.parametrize(
    "override",
    [
        {"scheduler_type": "direct"},
        {"scheduler_job_id": ""},
        {"scheduler_cluster": ""},
        {"scheduler_submit_token": ""},
        {"target": ""},
        {"target": "ssh", "host": ""},
        {"scheduler_direct_controller": ""},
        {"scheduler_direct_controller": True},
        {"scheduler_raw_state": "SUBMISSION_CLUSTER_MISMATCH"},
    ],
)
def test_context_excludes_incomplete_and_quarantined_rows(tmp_path: Path, monkeypatch, override: dict):
    rows = _bound_rows(tmp_path, 3)
    rows[2].update(override)
    calls = []

    def run_command(_execution, argv, *, timeout):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, _queue_output(rows[:2]), "")

    monkeypatch.setattr(slurm, "run_command", run_command)
    context = managed_scheduler.SlurmMonitorContext(rows, owner_dir=tmp_path)
    if override.get("scheduler_raw_state") == "SUBMISSION_CLUSTER_MISMATCH":
        observed = managed_scheduler.observe_slurm_run(tmp_path, _execution(rows[2]), rows[2], monitor_context=context)
        assert observed["scheduler_raw_state"] == "SUBMISSION_CLUSTER_MISMATCH"
        assert calls == []
    for row in rows[:2]:
        managed_scheduler.observe_slurm_run(tmp_path, _execution(row), row, monitor_context=context)
    assert [argv[-1] for argv in calls] == ["3880,3881"]


def test_context_never_promotes_legacy_sidecar_identity_across_rounds(tmp_path: Path, monkeypatch):
    rows = _bound_rows(tmp_path, 3)
    rows[2].update(scheduler_job_id="", scheduler_cluster="", status="submitting")
    Path(rows[2]["scheduler_result_path"]).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "scheduler_job_id": "3999",
                "scheduler_cluster": "sidecar-only-cluster",
                "scheduler_submit_token": rows[2]["scheduler_submit_token"],
                "exit_code": 0,
            }
        )
    )
    calls = []

    def run_command(_execution, argv, *, timeout):
        calls.append(argv)
        if argv[0] == "squeue" and "," in argv[-1]:
            return subprocess.CompletedProcess(argv, 0, _queue_output(rows[:2]), "")
        assert not any(arg.startswith("--clusters=") for arg in argv)
        if argv[0] == "squeue":
            assert argv[-1] == "3999"
            return (
                subprocess.CompletedProcess(argv, 1, "", "controller unavailable")
                if phase == "error"
                else subprocess.CompletedProcess(argv, 0, "", "")
            )
        if argv[0] == "scontrol":
            return subprocess.CompletedProcess(argv, 1, "", "Invalid job id specified")
        assert argv[0] == "sacct"
        return subprocess.CompletedProcess(argv, 1, "", "Slurm accounting storage is disabled")

    monkeypatch.setattr(slurm, "run_command", run_command)
    for phase in ("error", "missing"):
        context = managed_scheduler.SlurmMonitorContext(rows, owner_dir=tmp_path)
        rows = [
            managed_scheduler.observe_slurm_run(tmp_path, _execution(row), row, monitor_context=context) for row in rows
        ]
        assert rows[2]["scheduler_job_id"] == ""
        assert rows[2]["scheduler_cluster"] == ""
        assert rows[2]["status"] == "submitting"

    assert [argv[-1] for argv in calls if argv[0] == "squeue"] == ["3880,3881", "3999", "3880,3881", "3999"]


def test_context_does_not_consume_snapshot_on_different_execution_route(tmp_path: Path, monkeypatch):
    rows = _bound_rows(tmp_path)
    calls = []

    def run_command(execution, argv, *, timeout):
        calls.append((execution, argv))
        return subprocess.CompletedProcess(argv, 0, _queue_output(rows), "")

    monkeypatch.setattr(slurm, "run_command", run_command)
    context = managed_scheduler.SlurmMonitorContext(rows, owner_dir=tmp_path)
    managed_scheduler.observe_slurm_run(
        tmp_path, {"target": "ssh", "host": "other-host"}, rows[0], monitor_context=context
    )
    for row in rows:
        managed_scheduler.observe_slurm_run(tmp_path, _execution(row), row, monitor_context=context)
    assert [argv[-1] for _execution, argv in calls] == ["3880", "3880,3881"]


def test_remote_transport_override_rows_do_not_share_snapshot(tmp_path: Path, monkeypatch):
    rows = _bound_rows(tmp_path)
    calls = []

    def run_command(_execution, argv, *, timeout):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, _queue_output(rows), "")

    monkeypatch.setattr(slurm, "run_command", run_command)
    context = managed_scheduler.SlurmMonitorContext(rows, owner_dir=tmp_path, remote="unit-host")
    for row in rows:
        observation = {**row, "target": "ssh", "host": "unit-host"}
        managed_scheduler.observe_slurm_run(tmp_path, _execution(observation), observation, monitor_context=context)
    assert [argv[-1] for argv in calls] == ["3880", "3881"]


def test_queue_launch_observation_does_not_reuse_monitor_snapshot(tmp_path: Path, monkeypatch):
    plan_dir, rows = _seed_plan(tmp_path)
    by_job = {row["scheduler_job_id"]: row for row in rows}
    for row in rows:
        Path(row["scheduler_result_path"]).write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    **{key: row[key] for key in ("scheduler_job_id", "scheduler_cluster", "scheduler_submit_token")},
                    "exit_code": 0,
                }
            )
        )
    calls = []

    def run_command(_execution, argv, *, timeout):
        calls.append(argv)
        assert argv[0] == "squeue"
        queried = [by_job[job_id] for job_id in argv[-1].split(",")]
        state = "PENDING" if len(queried) > 1 else "COMPLETED"
        return subprocess.CompletedProcess(argv, 0, _queue_output(queried, state), "")

    monkeypatch.setattr(slurm, "run_command", run_command)
    monkeypatch.setattr(hparam_runtime.time, "sleep", lambda *_args: pytest.fail("terminal queue must not sleep"))

    hparam_runtime.run_hparam_queue(plan_dir, dry_run=False)

    assert [argv[-1] for argv in calls] == ["3880,3881", "3880", "3881"]
    assert {row["status"] for row in _read_table(tmp_path / "run_manifest.tsv") if row["scheduler_job_id"]} == {
        "completed"
    }


@pytest.mark.parametrize("scenario", ["queue", "controller", "terminal_accounting_disabled"])
@pytest.mark.parametrize("health", [False, True])
def test_context_preserves_complete_observation_for_same_seed(tmp_path: Path, monkeypatch, scenario: str, health: bool):
    rows = _bound_rows(tmp_path)
    by_job = {row["scheduler_job_id"]: row for row in rows}
    for row in rows:
        row["launched_at"] = "2026-08-21T00:00:00Z"
        if scenario == "terminal_accounting_disabled":
            Path(row["scheduler_result_path"]).write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        **{
                            key: row[key] for key in ("scheduler_job_id", "scheduler_cluster", "scheduler_submit_token")
                        },
                        "exit_code": 0,
                        "node": "terminal-node",
                        "started_at": "2026-08-21T00:01:00Z",
                    }
                )
            )
    seed = json.dumps(rows, sort_keys=True)
    monkeypatch.setattr(managed_scheduler, "utc_now", lambda: "2026-08-21T00:02:00Z")
    monkeypatch.setattr(managed_scheduler, "_timestamp_age_seconds", lambda value: 60 if value else None)

    def run_command(_execution, argv, *, timeout):
        if argv[0] == "squeue":
            queried = [by_job[job_id] for job_id in argv[-1].split(",")]
            return subprocess.CompletedProcess(argv, 0, _queue_output(queried) if scenario == "queue" else "", "")
        if argv[0] == "scontrol":
            if scenario == "terminal_accounting_disabled":
                return subprocess.CompletedProcess(argv, 1, "", "Invalid job id specified")
            row = by_job[argv[-1]]
            output = (
                f"JobId={argv[-1]} JobState=RUNNING Comment={row['scheduler_submit_token']} "
                "NodeList=controller-node Reason=None Priority=42 Nice=0 Partition=gpu "
                "Account=research QOS=normal TimeLimit=01:00:00"
            )
            return subprocess.CompletedProcess(argv, 0, output, "")
        assert argv[0] == "sacct"
        return subprocess.CompletedProcess(argv, 1, "", "Slurm accounting storage is disabled")

    monkeypatch.setattr(slurm, "run_command", run_command)
    unshared = [managed_scheduler.observe_slurm_run(tmp_path, _execution(row), row, health=health) for row in rows]
    context = managed_scheduler.SlurmMonitorContext(rows, owner_dir=tmp_path)
    shared = [
        managed_scheduler.observe_slurm_run(tmp_path, _execution(row), row, health=health, monitor_context=context)
        for row in rows
    ]

    assert shared == unshared
    assert json.dumps(rows, sort_keys=True) == seed
