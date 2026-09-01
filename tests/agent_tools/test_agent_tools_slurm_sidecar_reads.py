from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest
from test_agent_tools_hparam_runtime import _stub_execution_snapshot_preflight  # noqa: F401
from test_agent_tools_slurm_monitor_context import _bound_rows, _execution, _queue_output, _seed_plan

from agent_tools import experiment_io, experiments, hparam_runtime, managed_scheduler, run_evidence, slurm
from agent_tools.experiment_workspace import initialize_run_manifest, merge_run_manifest, read_run_manifest


@pytest.fixture
def reads(monkeypatch):
    calls = []
    reader = experiment_io.read_managed_output_texts_at

    def read(root, paths, *, remote=None):
        calls.append((remote, tuple(str(path) for path in paths)))
        return reader(root, paths)

    monkeypatch.setattr(experiment_io, "read_managed_output_texts_at", read)
    monkeypatch.setattr(run_evidence, "runtime_artifacts", lambda _row: ("", {}, []))
    monkeypatch.setattr(run_evidence, "log_tail", lambda *_args: "")
    monkeypatch.setattr(run_evidence, "log_age_seconds", lambda *_args: None)
    monkeypatch.setattr(slurm, "submit", lambda *_a, **_k: pytest.fail("monitor must not submit"))
    monkeypatch.setattr(slurm, "cancel", lambda *_a, **_k: pytest.fail("monitor must not cancel"))
    return calls


def _write_sidecar(row, field="scheduler_result_path"):
    payload = {
        "schema_version": 1,
        "scheduler_job_id": row["scheduler_job_id"] or "3999",
        "scheduler_cluster": row["scheduler_cluster"] or "sidecar-only-cluster",
        "scheduler_submit_token": row["scheduler_submit_token"],
    }
    if field == "scheduler_result_path":
        payload["exit_code"] = 0
    Path(row[field]).write_text(json.dumps(payload))


def _stub_queue(monkeypatch, rows, state="RUNNING"):
    calls = []

    def run_command(_execution, argv, *, timeout):
        calls.append(argv)
        requested = argv[-1].split(",")
        selected = [row for row in rows if row["scheduler_job_id"] in requested]
        if argv[0] == "scontrol":
            row = selected[0]
            detail = f"JobId={argv[-1]} JobState={state} Comment={row['scheduler_submit_token']}"
            return subprocess.CompletedProcess(argv, 0, detail, "")
        assert argv[0] == "squeue"
        return subprocess.CompletedProcess(argv, 0, _queue_output(selected, state), "")

    monkeypatch.setattr(slurm, "run_command", run_command)
    return calls


@pytest.mark.parametrize("field", ["allocation_identity_path", "scheduler_result_path"])
def test_observe_slurm_run_first_fills_canonical_actual_runtime_commit_from_sidecar(
    tmp_path, monkeypatch, reads, field
):
    row = _bound_rows(tmp_path, 1)[0]
    row.update(
        experiment_id="unit",
        planned_runtime_commit="a" * 40,
        runtime_commit="",
        scheduler_script=str(tmp_path / "job.sbatch"),
        scheduler_script_sha256="c" * 64,
    )
    (tmp_path / "experiment.yaml").write_text("experiment:\n  id: unit\n")
    initialize_run_manifest(tmp_path)
    merge_run_manifest(tmp_path, [row])
    row = read_run_manifest(tmp_path)[0]
    _write_sidecar(row, field)
    sidecar_path = Path(row[field])
    payload = json.loads(sidecar_path.read_text())
    if field == "allocation_identity_path":
        payload["execution_snapshot"] = {"runtime_commit": "b" * 40}
    else:
        payload["runtime_commit"] = "b" * 40
    sidecar_path.write_text(json.dumps(payload))
    _stub_queue(monkeypatch, [row])

    observed = managed_scheduler.observe_slurm_run(tmp_path, _execution(row), row)
    committed = merge_run_manifest(tmp_path, [observed])[0]

    assert committed["status"] == "running"
    assert committed["planned_runtime_commit"] == "a" * 40
    assert committed["runtime_commit"] == "b" * 40


@pytest.mark.parametrize("target", ["local", "ssh"])
@pytest.mark.parametrize("terminal", [True, False])
def test_sidecars_are_read_in_two_lazy_host_phases(tmp_path, monkeypatch, reads, target, terminal):
    rows = _bound_rows(tmp_path, 6)
    for row in rows:
        row.update(target=target, host="unit-host" if target == "ssh" else "")
        if terminal:
            _write_sidecar(row)
            Path(row["allocation_identity_path"]).symlink_to(tmp_path / "forbidden")
    _stub_queue(monkeypatch, rows)
    context = managed_scheduler.SlurmMonitorContext(rows, owner_dir=tmp_path)
    assert reads == []

    observed = [
        managed_scheduler.observe_slurm_run(tmp_path, _execution(row), row, monitor_context=context) for row in rows
    ]

    assert {row["status"] for row in observed} == {"running"}
    expected_fields = ["scheduler_result_path"] + ([] if terminal else ["allocation_identity_path"])
    assert reads == [
        ("unit-host" if target == "ssh" else None, tuple(row[field] for row in rows)) for field in expected_fields
    ]


def test_file_batches_use_host_not_scheduler_route(tmp_path, monkeypatch, reads):
    rows = _bound_rows(tmp_path, 6)
    for index, row in enumerate(rows):
        row.update(target="ssh", host="host-a" if index < 4 else "host-b")
        row["scheduler_cluster"] = f"cluster-{index}"
        row["scheduler_direct_controller"] = "true" if index % 2 else "false"
        _write_sidecar(row)
    _stub_queue(monkeypatch, rows)
    context = managed_scheduler.SlurmMonitorContext(rows, owner_dir=tmp_path)

    for row in rows:
        managed_scheduler.observe_slurm_run(tmp_path, _execution(row), row, monitor_context=context)

    assert reads == [
        ("host-a", tuple(row["scheduler_result_path"] for row in rows[:4])),
        ("host-b", tuple(row["scheduler_result_path"] for row in rows[4:])),
    ]


@pytest.mark.parametrize("bad_text", ["{", "null", "[]", " "])
def test_future_bad_terminal_does_not_block_current_allocation(tmp_path, monkeypatch, reads, bad_text):
    rows = _bound_rows(tmp_path)
    _write_sidecar(rows[0], "allocation_identity_path")
    Path(rows[1]["scheduler_result_path"]).write_text(bad_text)
    Path(rows[1]["allocation_identity_path"]).symlink_to(tmp_path / "forbidden")
    calls = _stub_queue(monkeypatch, rows)
    context = managed_scheduler.SlurmMonitorContext(rows, owner_dir=tmp_path)

    assert (
        managed_scheduler.observe_slurm_run(tmp_path, _execution(rows[0]), rows[0], monitor_context=context)["status"]
        == "running"
    )
    with pytest.raises(ValueError, match="not valid JSON|must be a mapping"):
        managed_scheduler.observe_slurm_run(tmp_path, _execution(rows[1]), rows[1], monitor_context=context)

    assert len(calls) == 1
    assert reads == [
        (None, tuple(row["scheduler_result_path"] for row in rows)),
        (None, (rows[0]["allocation_identity_path"],)),
    ]


@pytest.mark.parametrize("terminal_text", [None, "", "{}"])
def test_allocation_batch_contains_only_rows_with_absent_terminal(tmp_path, monkeypatch, reads, terminal_text):
    rows = _bound_rows(tmp_path, 3)
    _write_sidecar(rows[0])
    Path(rows[0]["allocation_identity_path"]).symlink_to(tmp_path / "forbidden")
    for row in rows[1:]:
        if terminal_text is not None:
            Path(row["scheduler_result_path"]).write_text(terminal_text)
        _write_sidecar(row, "allocation_identity_path")
    _stub_queue(monkeypatch, rows)
    context = managed_scheduler.SlurmMonitorContext(rows, owner_dir=tmp_path)

    for row in rows:
        managed_scheduler.observe_slurm_run(tmp_path, _execution(row), row, monitor_context=context)

    assert reads == [
        (None, tuple(row["scheduler_result_path"] for row in rows)),
        (None, tuple(row["allocation_identity_path"] for row in rows[1:])),
    ]


@pytest.mark.parametrize("field", ["scheduler_result_path", "allocation_identity_path"])
def test_bad_future_file_discards_batch_but_preserves_current_error_order(tmp_path, monkeypatch, reads, field):
    rows = _bound_rows(tmp_path, 3)
    Path(rows[1][field]).symlink_to(tmp_path / "forbidden")
    calls = _stub_queue(monkeypatch, rows)
    context = managed_scheduler.SlurmMonitorContext(rows, owner_dir=tmp_path)

    assert (
        managed_scheduler.observe_slurm_run(tmp_path, _execution(rows[0]), rows[0], monitor_context=context)["status"]
        == "running"
    )
    queried = len(calls)
    with pytest.raises(ValueError):
        managed_scheduler.observe_slurm_run(tmp_path, _execution(rows[1]), rows[1], monitor_context=context)
    assert len(calls) == queried
    assert (
        managed_scheduler.observe_slurm_run(tmp_path, _execution(rows[2]), rows[2], monitor_context=context)["status"]
        == "running"
    )
    batches = [paths for _host, paths in reads if len(paths) > 1 and rows[1][field] in paths]
    assert len(batches) == 1
    assert (None, (rows[0][field],)) in reads
    assert (None, (rows[1][field],)) in reads
    assert (None, (rows[2][field],)) in reads


@pytest.mark.parametrize("failure", ["timeout", "transport", "payload"])
def test_batch_protocol_failure_has_only_one_attempt_then_singletons(tmp_path, monkeypatch, reads, failure):
    rows = _bound_rows(tmp_path, 3)
    for row in rows:
        _write_sidecar(row)
    read = experiment_io.read_managed_output_texts_at
    batches = []

    def fail_batch(root, paths, *, remote=None):
        if len(paths) > 1:
            batches.append(tuple(paths))
            if failure == "timeout":
                raise subprocess.TimeoutExpired("ssh", 1)
            if failure == "transport":
                raise RuntimeError("SSH read failed")
            raise ValueError("incomplete remote payload")
        return read(root, paths, remote=remote)

    monkeypatch.setattr(experiment_io, "read_managed_output_texts_at", fail_batch)
    _stub_queue(monkeypatch, rows)
    context = managed_scheduler.SlurmMonitorContext(rows, owner_dir=tmp_path)
    for row in rows:
        managed_scheduler.observe_slurm_run(tmp_path, _execution(row), row, monitor_context=context)
    assert len(batches) == 1
    assert reads == [(None, (row["scheduler_result_path"],)) for row in rows]


def test_conflict_row_is_excluded_from_other_runs_prefetch(tmp_path, monkeypatch, reads):
    rows = _bound_rows(tmp_path, 3)
    rows[1]["scheduler_raw_state"] = "SUBMISSION_CLUSTER_MISMATCH"
    rows[1]["status"] = "unknown_scheduler"
    for row in rows:
        _write_sidecar(row)
    calls = _stub_queue(monkeypatch, rows)
    context = managed_scheduler.SlurmMonitorContext(rows, owner_dir=tmp_path)
    for row in rows:
        observed = managed_scheduler.observe_slurm_run(tmp_path, _execution(row), row, monitor_context=context)
        if row is rows[1]:
            assert observed["status"] == "unknown_scheduler"
    assert reads == [(None, (rows[0]["scheduler_result_path"], rows[2]["scheduler_result_path"]))]
    assert len(calls) == 1


def test_snapshot_does_not_cross_owner_or_execution_host(tmp_path, monkeypatch, reads):
    rows = _bound_rows(tmp_path)
    for row in rows:
        row.update(target="ssh", host="host-a")
        _write_sidecar(row)
    _stub_queue(monkeypatch, rows)
    context = managed_scheduler.SlurmMonitorContext(rows, owner_dir=tmp_path)
    managed_scheduler.observe_slurm_run(tmp_path, _execution(rows[0]), rows[0], monitor_context=context)
    managed_scheduler.observe_slurm_run(tmp_path, {"target": "ssh", "host": "host-b"}, rows[1], monitor_context=context)
    other_owner = tmp_path / "other-owner"
    other_owner.mkdir()
    other_rows = _bound_rows(other_owner)
    for row in other_rows:
        row.update(target="ssh", host="host-a")
        _write_sidecar(row)
    managed_scheduler.observe_slurm_run(other_owner, _execution(other_rows[0]), other_rows[0], monitor_context=context)
    assert reads == [
        ("host-a", tuple(row["scheduler_result_path"] for row in rows)),
        ("host-b", (rows[1]["scheduler_result_path"],)),
        ("host-a", (other_rows[0]["scheduler_result_path"],)),
    ]


def test_terminal_published_after_prefetch_is_read_next_round(tmp_path, monkeypatch, reads):
    rows = _bound_rows(tmp_path)
    _stub_queue(monkeypatch, rows, "COMPLETED")
    context = managed_scheduler.SlurmMonitorContext(rows, owner_dir=tmp_path)
    managed_scheduler.observe_slurm_run(tmp_path, _execution(rows[0]), rows[0], monitor_context=context)
    _write_sidecar(rows[1])
    assert (
        managed_scheduler.observe_slurm_run(tmp_path, _execution(rows[1]), rows[1], monitor_context=context)["status"]
        == "unknown_scheduler"
    )
    fresh = managed_scheduler.SlurmMonitorContext(rows, owner_dir=tmp_path)
    assert (
        managed_scheduler.observe_slurm_run(tmp_path, _execution(rows[1]), rows[1], monitor_context=fresh)["status"]
        == "completed"
    )


@pytest.mark.parametrize("field", ["scheduler_result_path", "allocation_identity_path"])
@pytest.mark.parametrize("missing_identity", ["job", "cluster", "both"])
def test_batched_sidecars_never_accumulate_trusted_legacy_identity(
    tmp_path, monkeypatch, reads, field, missing_identity
):
    rows = _bound_rows(tmp_path)
    legacy = rows[1]
    legacy["scheduler_direct_controller"] = ""
    if missing_identity in {"job", "both"}:
        legacy.update(scheduler_job_id="", status="submitting")
    if missing_identity in {"cluster", "both"}:
        legacy["scheduler_cluster"] = ""
    identity = legacy["scheduler_job_id"], legacy["scheduler_cluster"]
    _write_sidecar(legacy, field)

    def run_command(_execution, argv, *, timeout):
        if argv[0] == "squeue" and argv[-1] == rows[0]["scheduler_job_id"]:
            return subprocess.CompletedProcess(argv, 0, _queue_output(rows[:1]), "")
        if phase == "error":
            return subprocess.CompletedProcess(argv, 255, "", "SSH connection closed")
        if argv[0] == "squeue":
            return subprocess.CompletedProcess(argv, 0, "", "")
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
        assert (rows[1]["scheduler_job_id"], rows[1]["scheduler_cluster"]) == identity
        assert rows[1]["status"] == ("unknown_scheduler" if identity[0] else "submitting")
    terminal_batches = [paths for _host, paths in reads if rows[1]["scheduler_result_path"] in paths]
    assert len(terminal_batches) == 2
    assert all(len(paths) == 2 for paths in terminal_batches)


@pytest.mark.parametrize("entrypoint", ["hparam", "experiment"])
def test_public_monitors_make_fresh_sidecar_batches(tmp_path, monkeypatch, reads, entrypoint):
    plan_dir, rows = _seed_plan(tmp_path)
    _stub_queue(monkeypatch, rows, "COMPLETED")
    for iteration in range(2):
        if iteration:
            for row in rows:
                _write_sidecar(row)
        if entrypoint == "hparam":
            hparam_runtime.monitor_hparam_runs(plan_dir)
        else:
            experiments.monitor_experiment(tmp_path)
    assert len(reads) == 3
    assert [len(paths) for _host, paths in reads] == [2, 2, 2]


def test_hparam_monitor_preserves_relative_plan_directory(tmp_path, monkeypatch, reads):
    plan_dir, rows = _seed_plan(tmp_path)
    for row in rows:
        _write_sidecar(row)
    _stub_queue(monkeypatch, rows, "COMPLETED")
    monkeypatch.chdir(tmp_path)

    report = hparam_runtime.monitor_hparam_runs(Path(plan_dir.name))

    assert report == Path(plan_dir.name) / "run_status.tsv"
    assert len(reads) == 1
    assert reads[0][0] is None
    assert set(reads[0][1]) == {row["scheduler_result_path"] for row in rows}


@pytest.mark.parametrize("routes", [1, 2])
def test_real_subprocess_sidecar_counts_scale_with_hosts(tmp_path, routes):
    repo = Path(__file__).resolve().parents[2]
    output = tmp_path / "counts"
    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("slurm_monitor_benchmark.py")),
            "--output-dir",
            str(output),
            "--runs",
            "4",
            "--samples",
            "1",
            "--modes",
            "ordinary",
            "--monitor",
            "both",
            "--routes",
            str(routes),
        ],
        cwd=repo,
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads((output / "results.json").read_text())
    assert report["transport"] == "real_subprocess_fake_ssh_fake_slurm"
    assert len(report["samples"]) == 2
    for sample in report["samples"]:
        counts = sample["counts"]
        assert counts["ssh:sidecar_read"] == 2 * routes
        assert counts.get("ssh:sidecar_validate", 0) == 0
        assert counts["squeue"] == routes
        assert counts["ssh"] == (8 if sample["mode"] == "ordinary" else 12) + 3 * routes
        assert counts.get("ssh:run_evidence.log_tail_and_age", 0) == (0 if sample["mode"] == "ordinary" else 4)
