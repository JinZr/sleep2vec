from __future__ import annotations

from contextlib import contextmanager
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from agent_tools import (
    experiment_io,
    experiment_tracking,
    experiment_workspace,
    experiments,
    python_programs,
    run_evidence,
    transport,
)


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "agent_tools", *args], text=True, capture_output=True)


def _read_table(path: Path) -> list[dict[str, str]]:
    delimiter = "\t" if path.suffix == ".tsv" else ","
    with path.open(newline="") as file_obj:
        return list(csv.DictReader(file_obj, delimiter=delimiter))


def _is_remote_python_program(command: str, name: str) -> bool:
    return command.startswith(f"python3 -c {transport.sh(python_programs.source(name))}")


def _initialize_workspace(root: Path, *, experiment_id: str = "unit") -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "experiment.yaml").write_text(
        json.dumps(
            {
                "experiment": {
                    "id": experiment_id,
                    "title": "Unit experiment",
                    "objective": "Exercise experiment workspace contracts.",
                    "root": str(root),
                    "baseline": {"type": "none", "rationale": "Unit fixture."},
                }
            }
        )
    )


def test_experiment_monitor_preserves_wandb_running_without_pid(tmp_path: Path):
    _initialize_workspace(tmp_path)
    (tmp_path / "run_manifest.tsv").write_text(
        "experiment_id\tstep_id\trun_id\trun_name\tversion\tstate\tstatus\n"
        "unit\ttrain-model\trun-000\tlr-2e-6\trun_a\trunning\trunning\n"
    )
    stale_launch = (
        "step_id\trun_id\tversion\tstatus\tpid_path\n"
        f"train-model\trun-000\trun_a\tlaunched\t{tmp_path / 'missing.pid'}\n"
    )
    (tmp_path / "launch_manifest.tsv").write_text(stale_launch)
    (tmp_path / "run_status.tsv").write_text(stale_launch)

    result = experiments.monitor_experiment(tmp_path)

    assert result["runs"][0]["status"] == "running"
    assert result["runs"][0]["health_status"] == "running"
    rows = _read_table(tmp_path / "run_manifest.tsv")
    assert rows[0]["status"] == "running"
    report = tmp_path / "running_final.md"
    report.write_text("# Final\n")
    finalized = _run("experiment-finalize", "--run-dir", str(tmp_path), "--report", str(report))
    assert finalized.returncode == 1
    assert "unresolved runs" in finalized.stderr


@pytest.mark.parametrize(
    "extra_fields",
    [{}, {"pid": "12345"}, {"wandb_run_id": "wandb-1"}, {"pid": "12345", "wandb_run_id": "wandb-1"}],
    ids=["no-extra-fields", "bare-pid", "bare-wandb-run-id", "bare-pid-and-wandb-run-id"],
)
def test_experiment_monitor_preserves_script_owned_running_without_execution_evidence(
    tmp_path: Path, monkeypatch, extra_fields: dict[str, str]
):
    _initialize_workspace(tmp_path)
    experiment_io.write_rows_at(
        tmp_path / "run_manifest.tsv",
        [
            {
                "experiment_id": "unit",
                "step_id": "train-model",
                "run_id": "run-000",
                "run_name": "lr-2e-6",
                "version": "run_a",
                "script": "/plan/run.sh",
                "status": "running",
                **extra_fields,
            }
        ],
    )
    monkeypatch.setattr(
        run_evidence,
        "status_row",
        lambda *_args, **_kwargs: pytest.fail("script-owned runs must not use process inference"),
    )

    result = experiments.monitor_experiment(tmp_path)

    assert result["runs"][0]["status"] == "running"
    assert result["runs"][0]["health_status"] == "running"
    assert _read_table(tmp_path / "run_manifest.tsv")[0]["status"] == "running"


@pytest.mark.parametrize(
    ("terminal_status_owner", "expected_status"),
    [(None, "completed"), ("script", "failed")],
    ids=["legacy-monitor-owner", "explicit-script-owner"],
)
def test_experiment_monitor_respects_terminal_owner_for_dead_process(
    tmp_path: Path,
    monkeypatch,
    terminal_status_owner: str | None,
    expected_status: str,
):
    _initialize_workspace(tmp_path)
    pid_path = tmp_path / "hparam.pid"
    identity = {"pid": 123, "process_group_id": 123, "process_start_token": "proc:unit-start"}
    pid_path.write_text(json.dumps(identity) + "\n")
    log_path = tmp_path / "hparam.log"
    log_path.write_text("training completed\n")
    row = {
        "experiment_id": "unit",
        "step_id": "train-model",
        "run_id": "run-000",
        "run_name": "hparam",
        "version": "hparam-v1",
        "script": str(tmp_path / "launch.sh"),
        "pid_path": str(pid_path),
        "log_path": str(log_path),
        "status": "running",
        **identity,
    }
    if terminal_status_owner is not None:
        row["terminal_status_owner"] = terminal_status_owner
    experiment_io.write_rows_at(tmp_path / "run_manifest.tsv", [row])
    monkeypatch.setattr(run_evidence, "process_identity_running", lambda *_args: False)

    result = experiments.monitor_experiment(tmp_path)

    assert result["runs"][0]["status"] == expected_status
    assert _read_table(tmp_path / "run_manifest.tsv")[0]["status"] == expected_status


@pytest.mark.parametrize(
    ("canonical_owner", "incoming_owner", "expected_script_owner"),
    [
        ("script", "monitor", True),
        ("monitor", "script", False),
    ],
)
def test_experiment_monitor_uses_canonical_terminal_status_owner(
    tmp_path: Path,
    monkeypatch,
    canonical_owner: str,
    incoming_owner: str,
    expected_script_owner: bool,
):
    observed = []
    row = {
        "step_id": "train-model",
        "run_id": "run-000",
        "script": "/plan/run.sh",
        "pid_path": "/run.pid",
        "terminal_status_owner": incoming_owner,
        "status": "running",
    }
    previous = {**row, "terminal_status_owner": canonical_owner}

    def fake_status(_root, current, previous, *, script_commits_terminal_status, health):
        observed.append(script_commits_terminal_status)
        return {**previous, **current, "health_status": "running"}

    monkeypatch.setattr(run_evidence, "status_row", fake_status)

    experiment_tracking.monitor_run_row(tmp_path, row, [previous])

    assert observed == [expected_script_owner]


def test_experiment_monitor_marks_lifecycle_script_exit_without_terminal_commit_failed(tmp_path: Path):
    _initialize_workspace(tmp_path)
    experiment_io.write_rows_at(
        tmp_path / "run_manifest.tsv",
        [
            {
                "experiment_id": "unit",
                "step_id": "train-model",
                "run_id": "run-000",
                "run_name": "managed",
                "version": "managed-v1",
                "script": str(tmp_path / "run.sh"),
                "state": "finished",
                "status": "running",
            }
        ],
    )

    result = experiments.monitor_experiment(tmp_path)

    assert result["runs"][0]["status"] == "failed"
    assert _read_table(tmp_path / "run_manifest.tsv")[0]["status"] == "failed"


def test_experiment_monitor_ignores_stale_auxiliary_status_rows(tmp_path: Path):
    _initialize_workspace(tmp_path)
    (tmp_path / "run_manifest.tsv").write_text(
        "experiment_id\tstep_id\trun_id\trun_name\tversion\tstatus\n"
        "unit\ttrain-model\trun-000\tfirst\tfirst-v1\tfailed\n"
        "unit\ttrain-model\trun-001\tsecond\tsecond-v1\trunning\n"
        "unit\ttrain-model\trun-002\tthird\tthird-v1\tcompleted\n"
    )
    (tmp_path / "launch_manifest.tsv").write_text(
        "step_id\trun_id\tversion\tstatus\tpid_path\n"
        f"train-model\trun-000\tfirst-v1\tlaunched\t{tmp_path / 'first.pid'}\n"
        f"train-model\trun-001\tsecond-v1\tlaunched\t{tmp_path / 'second.pid'}\n"
        f"train-model\trun-002\tthird-v1\tlaunched\t{tmp_path / 'third.pid'}\n"
    )
    (tmp_path / "run_status.tsv").write_text(
        "step_id\trun_id\tversion\tstatus\tpid_path\n"
        f"train-model\trun-000\tfirst-v1\tlaunched\t{tmp_path / 'first.pid'}\n"
        f"train-model\trun-001\tsecond-v1\tfailed\t{tmp_path / 'second.pid'}\n"
        f"train-model\trun-002\tthird-v1\tfailed\t{tmp_path / 'third.pid'}\n"
    )

    result = experiments.monitor_experiment(tmp_path)

    assert {(row["run_id"], row["status"]) for row in result["runs"]} == {
        ("run-000", "failed"),
        ("run-001", "completed"),
        ("run-002", "completed"),
    }


@pytest.mark.parametrize("remote", [None, "unit-host"], ids=["local", "remote"])
def test_experiment_monitor_reads_one_validated_input_snapshot_then_fresh_commit(tmp_path: Path, monkeypatch, remote):
    _initialize_workspace(tmp_path)
    manifest_path = tmp_path / "run_manifest.tsv"
    experiment_io.write_rows_at(
        manifest_path,
        [{"experiment_id": "unit", "step_id": "train-model", "run_id": "run-000", "status": "running"}],
    )
    real_managed_read = experiment_io.read_managed_files_at
    real_read = experiment_io.read_text_at
    real_lock = experiment_io.blocking_file_lock
    real_commit = experiment_io.conditional_atomic_replace_text_at
    real_managed_rows = experiments._managed_rows
    snapshot_reads = []
    commit_reads = []
    commits = []
    snapshots = []
    observation_started = False
    lock_held = False

    def read_managed(root, paths, *, remote=None, **kwargs):
        if manifest_path in paths:
            snapshot_reads.append(remote)
        return real_managed_read(root, paths, **kwargs)

    def read_text(path, *, remote=None):
        if Path(path) == manifest_path:
            commit_reads.append((remote, observation_started, lock_held))
        return real_read(path)

    @contextmanager
    def held_lock(path):
        nonlocal lock_held
        with real_lock(path):
            lock_held = True
            try:
                yield
            finally:
                lock_held = False

    def managed_rows(root, *, remote=None):
        rows = real_managed_rows(root, remote=remote)
        snapshots.append((rows, [dict(row) for row in rows]))
        return rows

    def fake_status(_root, row, previous, *, script_commits_terminal_status, health):
        nonlocal observation_started
        observation_started = True
        assert row is not previous
        assert previous["status"] == "running"
        row.update(status="completed", health_status="completed")
        return row

    def commit(path, text, expected_sha256, *, remote=None, **kwargs):
        assert expected_sha256 == hashlib.sha256(Path(path).read_bytes()).hexdigest()
        commits.append(remote)
        return real_commit(path, text, expected_sha256, **kwargs)

    monkeypatch.setattr(experiment_io, "read_managed_files_at", read_managed)
    monkeypatch.setattr(experiment_io, "read_text_at", read_text)
    monkeypatch.setattr(experiment_io, "blocking_file_lock", held_lock)
    monkeypatch.setattr(experiment_io, "conditional_atomic_replace_text_at", commit)
    monkeypatch.setattr(experiments, "_managed_rows", managed_rows)
    monkeypatch.setattr(run_evidence, "status_row", fake_status)
    if remote:
        real_exists = experiment_io.path_exists_at
        real_validate = experiment_io.validate_managed_output_paths
        real_write = experiment_io.write_text_at
        monkeypatch.setattr(experiment_io, "path_exists_at", lambda path, **_kwargs: real_exists(path))
        monkeypatch.setattr(
            experiment_io,
            "validate_managed_output_paths",
            lambda root, paths, **_kwargs: real_validate(root, paths),
        )
        monkeypatch.setattr(experiment_io, "write_text_at", lambda path, text, **_kwargs: real_write(path, text))
        monkeypatch.setattr(
            experiment_workspace,
            "_write_remote_run_matrix_if_current",
            lambda root, rows, _text, _remote: bool(experiment_workspace.write_run_matrix(root, rows)),
        )
    monkeypatch.setattr(transport, "run_ssh", lambda *_args, **_kwargs: pytest.fail("Unexpected SSH in monitor test"))

    result = experiments.monitor_experiment(tmp_path, remote=remote)

    assert snapshot_reads == [remote]
    assert commit_reads == [(remote, True, remote is None)]
    assert commits == [remote]
    assert len(snapshots) == 1
    assert snapshots[0][0] == snapshots[0][1]
    assert result["runs"][0]["status"] == "completed"
    assert _read_table(manifest_path)[0]["status"] == "completed"


def test_monitor_observation_contains_only_managed_identity_and_status_fields(tmp_path: Path, monkeypatch):
    row = {
        "experiment_id": "unit",
        "step_id": "train-model",
        "run_id": "run-000",
        "run_name": "managed",
        "version": "managed-v1",
        "config": "/config.yaml",
        "status": "running",
    }
    monkeypatch.setattr(
        run_evidence,
        "status_row",
        lambda *_args, **_kwargs: {
            **row,
            "status": "completed",
            "health_status": "completed",
            "monitored_at": "now",
        },
    )

    observation = experiment_tracking.monitor_run_row(tmp_path, dict(row), [dict(row)])

    assert observation == {
        "step_id": "train-model",
        "run_id": "run-000",
        "status": "completed",
        "health_status": "completed",
        "monitored_at": "now",
    }


@pytest.mark.parametrize("canonical_target", [None, "local"])
def test_remote_monitor_uses_transport_identity_without_persisting_it(tmp_path: Path, monkeypatch, canonical_target):
    previous = {"step_id": "train-model", "run_id": "run-000", "status": "running"}
    if canonical_target is not None:
        previous.update({"target": canonical_target, "host": ""})
    observed_rows = []

    def fake_status(_root, row, _previous, *, script_commits_terminal_status, health):
        assert script_commits_terminal_status is False
        observed_rows.append(dict(row))
        return {**row, "status": "running", "health_status": "running"}

    monkeypatch.setattr(run_evidence, "status_row", fake_status)

    observation = experiment_tracking.monitor_run_row(
        tmp_path,
        dict(previous),
        [dict(previous)],
        remote="unit-host",
    )

    assert observed_rows[0]["target"] == "ssh"
    assert observed_rows[0]["host"] == "unit-host"
    assert "target" not in observation
    assert "host" not in observation


@pytest.mark.parametrize(
    ("remote", "expected_transport"),
    [(None, {"target": "local"}), ("unit-host", {"target": "ssh", "host": "unit-host"})],
    ids=["local", "ssh"],
)
@pytest.mark.parametrize(
    "controller_topology",
    [None, "false", "true"],
    ids=["legacy-default", "federation", "direct-controller"],
)
def test_experiment_monitor_routes_slurm_observation_by_transport(
    tmp_path: Path,
    monkeypatch,
    remote: str | None,
    expected_transport: dict[str, str],
    controller_topology: str | None,
):
    row = {
        "step_id": "train-model",
        "run_id": "run-000",
        "script": "/plan/run.sh",
        "scheduler_type": "slurm",
        "scheduler_submit_token": "unit-token",
        "scheduler_script": "/plan/job.sbatch",
        "scheduler_script_sha256": "a" * 64,
        "scheduler_result_path": "/plan/slurm_terminal.json",
        "allocation_identity_path": "/plan/slurm_allocation.json",
        "target": "local",
        "host": "",
        "status": "submitting",
    }
    if controller_topology is not None:
        row["scheduler_direct_controller"] = controller_topology
    observed = []

    def fake_slurm_observation(owner_dir, execution, observed_row, *, health, monitor_context=None):
        observed.append((owner_dir, execution, dict(observed_row), health))
        observed_row["log_tail"] = "observer-owned update"
        return {
            **observed_row,
            "scheduler_job_id": "3880",
            "scheduler_cluster": "wuji-h20",
            "scheduler_raw_state": "PENDING",
            "scheduler_reason": "Resources",
            "scheduler_node": "",
            "scheduler_priority": "42",
            "scheduler_queue_age_seconds": 30,
            "health_status": "scheduler_queued",
            "scheduler_observed_at": "now",
            "status": "queued",
            "monitored_at": "now",
        }

    monkeypatch.setattr(experiment_tracking.managed_scheduler, "observe_slurm_run", fake_slurm_observation)
    monkeypatch.setattr(
        run_evidence,
        "status_row",
        lambda *_args, **_kwargs: pytest.fail("Slurm runs must not use direct PID evidence"),
    )

    snapshot = dict(row)
    observation = experiment_tracking.monitor_run_row(tmp_path, row, [row], remote=remote)

    assert row == snapshot
    assert observed[0][0] == tmp_path
    expected_execution = dict(expected_transport)
    if controller_topology == "true":
        expected_execution["scheduler"] = {"direct_controller": True}
    assert observed[0][1] == expected_execution
    assert observed[0][3] is True
    assert observation["status"] == "queued"
    assert observation["scheduler_job_id"] == "3880"
    assert observation["scheduler_cluster"] == "wuji-h20"
    assert observation["scheduler_raw_state"] == "PENDING"
    assert observation["scheduler_reason"] == "Resources"
    assert observation["scheduler_priority"] == "42"
    assert observation["scheduler_queue_age_seconds"] == 30
    assert observation["health_status"] == "scheduler_queued"
    assert observation["scheduler_observed_at"] == "now"
    if remote:
        assert observed[0][2]["target"] == "ssh"
        assert observed[0][2]["host"] == "unit-host"
        assert "target" not in observation
        assert "host" not in observation


def test_experiment_monitor_rejects_invalid_controller_topology_before_scheduler_query(tmp_path: Path, monkeypatch):
    row = {
        "step_id": "train-model",
        "run_id": "run-000",
        "scheduler_type": "slurm",
        "scheduler_direct_controller": "True",
        "target": "local",
        "status": "submitting",
    }
    monkeypatch.setattr(
        experiment_tracking.managed_scheduler,
        "observe_slurm_run",
        lambda *_args, **_kwargs: pytest.fail("invalid topology must fail before querying Slurm"),
    )

    with pytest.raises(ValueError, match="scheduler_direct_controller must be true or false"):
        experiment_tracking.monitor_run_row(tmp_path, row, [dict(row)])


def test_experiment_monitor_does_not_observe_unlaunched_slurm_run(tmp_path: Path, monkeypatch):
    row = {
        "step_id": "train-model",
        "run_id": "run-000",
        "scheduler_type": "slurm",
        "scheduler_submit_token": "unit-token",
        "scheduler_script": "/plan/job.sbatch",
        "scheduler_script_sha256": "a" * 64,
        "scheduler_result_path": "/plan/slurm_terminal.json",
        "allocation_identity_path": "/plan/slurm_allocation.json",
        "status": "planned",
    }
    monkeypatch.setattr(
        experiment_tracking.managed_scheduler,
        "observe_slurm_run",
        lambda *_args, **_kwargs: pytest.fail("An unlaunched Slurm run must remain planned"),
    )

    observation = experiment_tracking.monitor_run_row(tmp_path, dict(row), [dict(row)], remote="unit-host")

    assert observation["status"] == "planned"
    assert "target" not in observation
    assert "host" not in observation


@pytest.mark.parametrize(
    "concurrent_fields",
    [
        {"status": "failed"},
        {
            "status": "stopping",
            "stop_requested_at": "2026-08-31T01:02:03Z",
            "stop_reason": "User requested stop during monitor",
        },
    ],
    ids=["terminal", "stop-intent"],
)
def test_experiment_monitor_reports_latest_committed_rows(tmp_path: Path, monkeypatch, concurrent_fields):
    _initialize_workspace(tmp_path)
    experiment_io.write_rows_at(
        tmp_path / "run_manifest.tsv",
        [{"experiment_id": "unit", "step_id": "train-model", "run_id": "run-000", "status": "running"}],
    )
    monkeypatch.setattr(
        run_evidence,
        "status_row",
        lambda _root, row, _previous, *, script_commits_terminal_status, health: {
            **row,
            "status": "completed",
            "health_status": "completed",
        },
    )
    real_merge = experiment_workspace.merge_run_manifest

    def merge_after_concurrent_update(root, rows, *, remote=None):
        real_merge(
            root,
            [
                {"step_id": "train-model", "run_id": "run-000", **concurrent_fields},
                {
                    "experiment_id": "unit",
                    "step_id": "train-model",
                    "run_id": "run-001",
                    "status": "planned",
                },
            ],
            remote=remote,
        )
        return real_merge(root, rows, remote=remote)

    monkeypatch.setattr(experiments, "merge_run_manifest", merge_after_concurrent_update)

    result = experiments.monitor_experiment(tmp_path)

    assert [(row["run_id"], row["status"]) for row in result["runs"]] == [
        ("run-000", concurrent_fields["status"]),
        ("run-001", "planned"),
    ]
    for field, value in concurrent_fields.items():
        assert result["runs"][0][field] == value
    assert _read_table(tmp_path / "run_manifest.tsv") == _read_table(tmp_path / "run_matrix.csv")
    report = (tmp_path / "reports" / "monitor.md").read_text()
    assert "train-model / run-001" in report


def test_experiment_monitor_does_not_advance_planned_run_from_local_mirror(tmp_path: Path):
    _initialize_workspace(tmp_path)
    (tmp_path / "run_manifest.tsv").write_text(
        "experiment_id\tstep_id\trun_id\tversion\tstatus\n" "unit\ttrain-model\trun-000\tmanaged-v1\tplanned\n"
    )
    (tmp_path / "run_status.tsv").write_text(
        "step_id\trun_id\tversion\tstatus\n" "train-model\trun-000\tmanaged-v1\trunning\n"
    )

    result = experiments.monitor_experiment(tmp_path)

    assert result["runs"][0]["status"] == "planned"
    assert _read_table(tmp_path / "run_manifest.tsv")[0]["status"] == "planned"


def test_experiment_monitor_does_not_regress_canonical_running_from_stale_status(tmp_path: Path, monkeypatch):
    _initialize_workspace(tmp_path)
    (tmp_path / "run_manifest.tsv").write_text(
        "experiment_id\tstep_id\trun_id\tversion\tstatus\n" "unit\ttrain-model\trun-000\tmanaged-v1\trunning\n"
    )
    (tmp_path / "run_status.tsv").write_text(
        "step_id\trun_id\tversion\tstatus\n" "train-model\trun-000\tmanaged-v1\tplanned\n"
    )

    def fake_status(_root, row, previous, *, script_commits_terminal_status, health):
        assert script_commits_terminal_status is False
        assert health is True
        assert row["status"] == "running"
        assert previous["status"] == "running"
        return {**row, "health_status": "running"}

    monkeypatch.setattr(run_evidence, "status_row", fake_status)

    result = experiments.monitor_experiment(tmp_path)

    assert result["runs"][0]["status"] == "running"
    assert _read_table(tmp_path / "run_manifest.tsv")[0]["status"] == "running"


def test_read_run_manifest_ignores_conflicting_projection_fields(tmp_path: Path):
    (tmp_path / "run_manifest.tsv").write_text(
        "experiment_id\tstep_id\trun_id\trun_name\tparameter_summary\tversion\tconfig\tstatus\tstate\n"
        "unit\ttrain-model\trun-000\tmanaged\tlr=1e-6\tmanaged-v1\t/config.yaml\tplanned\tcanonical\n"
    )
    (tmp_path / "run_status.tsv").write_text(
        "experiment_id\tstep_id\trun_id\trun_name\tparameter_summary\tversion\tconfig\tstatus\tstate\tlog_path\n"
        "other\ttrain-model\trun-000\tcorrupt\tlr=9\tdisplay-name\t/other.yaml\trunning\tauxiliary\t/run.log\n"
    )

    rows = experiment_workspace.read_run_manifest(tmp_path)

    assert rows == [
        {
            "experiment_id": "unit",
            "step_id": "train-model",
            "run_id": "run-000",
            "run_name": "managed",
            "parameter_summary": "lr=1e-6",
            "version": "managed-v1",
            "config": "/config.yaml",
            "status": "planned",
            "state": "canonical",
        }
    ]


def test_experiment_run_rows_rejects_historical_status_without_rewriting(tmp_path: Path):
    _initialize_workspace(tmp_path)
    legacy = "trial_id\tversion\tstatus\ntrial_000\tlegacy-v1\tfailed\n"
    (tmp_path / "trial_status.tsv").write_text(legacy)

    with pytest.raises(ValueError, match="read-only"):
        experiments.monitor_experiment(tmp_path)

    assert (tmp_path / "trial_status.tsv").read_text() == legacy
    assert not (tmp_path / "run_manifest.tsv").exists()


def test_experiment_monitor_keeps_duplicate_versions_scoped_by_step_and_run(tmp_path: Path, monkeypatch):
    _initialize_workspace(tmp_path)
    (tmp_path / "run_manifest.tsv").write_text(
        "experiment_id\tstep_id\trun_id\trun_name\tversion\tstatus\n"
        "unit\tstep-a\trun-000\ta0\tshared\trunning\n"
        "unit\tstep-a\trun-001\ta1\tshared\trunning\n"
        "unit\tstep-b\trun-000\tb0\tshared\trunning\n"
    )
    monkeypatch.setattr(
        run_evidence,
        "status_row",
        lambda _root, row, previous, *, script_commits_terminal_status, health: {
            **row,
            "health_status": "running",
        },
    )

    result = experiments.monitor_experiment(tmp_path)

    assert {(row["step_id"], row["run_id"]) for row in result["runs"]} == {
        ("step-a", "run-000"),
        ("step-a", "run-001"),
        ("step-b", "run-000"),
    }


def test_experiment_monitor_matches_previous_rows_by_managed_identity(tmp_path: Path, monkeypatch):
    _initialize_workspace(tmp_path)
    (tmp_path / "run_manifest.tsv").write_text(
        "experiment_id\tstep_id\trun_id\trun_name\tversion\tstatus\tmarker\n"
        "unit\tprepare-data\trun-000\tprepare\tshared\trunning\tprepare-marker\n"
        "unit\ttrain-model\trun-000\ttrain\tshared\trunning\ttrain-marker\n"
    )
    matched = []

    def fake_status(_root, row, previous, *, script_commits_terminal_status, health):
        assert script_commits_terminal_status is False
        assert health is True
        assert row["marker"] == previous["marker"]
        matched.append((row["step_id"], previous["step_id"], previous["marker"]))
        return {**row, "health_status": "running"}

    monkeypatch.setattr(run_evidence, "status_row", fake_status)

    experiments.monitor_experiment(tmp_path)

    assert matched == [
        ("prepare-data", "prepare-data", "prepare-marker"),
        ("train-model", "train-model", "train-marker"),
    ]
    monitor = (tmp_path / "reports" / "monitor.md").read_text()
    assert "prepare-data / run-000 — prepare" in monitor
    assert "train-model / run-000 — train" in monitor


@pytest.mark.parametrize("artifact_returncode", [0, 255])
def test_experiment_monitor_observes_remote_artifacts_over_ssh_and_preserves_them_on_uncertainty(
    tmp_path: Path, monkeypatch, artifact_returncode: int
):
    row = {
        "step_id": "train-model",
        "run_id": "run-000",
        "status": "running",
        "target": "ssh",
        "host": "unit-host",
        "pid_path": "/remote/run.pid",
        "log_path": "/remote/run.log",
        "runtime_dir": "/remote/runtime/run-000",
        "checkpoint_dir": "/remote/runtime/run-000/checkpoints",
    }
    previous = {
        **row,
        "run_manifest": "/remote/runtime/previous/run_manifest.json",
        "checkpoints": "previous.ckpt",
    }
    commands = []

    def fake_remote_command(_row, command):
        commands.append(command)
        if _is_remote_python_program(command, "run_evidence.read_pid_text"):
            return subprocess.CompletedProcess([], 0, "123\n", "")
        if command.startswith("ps "):
            return subprocess.CompletedProcess([], 0, "123\n", "")
        if _is_remote_python_program(command, "run_evidence.runtime_artifacts"):
            payload = json.dumps(
                {
                    "run_manifest": "/remote/runtime/run-000/run_manifest.json",
                    "checkpoints": ["epoch=01.ckpt", "last.ckpt"],
                }
            )
            return subprocess.CompletedProcess([], artifact_returncode, payload if artifact_returncode == 0 else "", "")
        if command.startswith("tail -n 8"):
            return subprocess.CompletedProcess([], 0, "still running", "")
        raise AssertionError(f"Unexpected remote command: {command}")

    monkeypatch.setattr(run_evidence, "run_row_command", fake_remote_command)
    monkeypatch.setattr(
        run_evidence,
        "health_fields",
        lambda _root, _row, _previous, _pid, _running, status, checkpoints: {
            "health_status": "health_unknown" if checkpoints is None else status,
            "checkpoint_count": "" if checkpoints is None else len(checkpoints),
        },
    )

    observation = experiment_tracking.monitor_run_row(tmp_path, row, [previous], remote="unit-host")

    if artifact_returncode == 0:
        assert observation["run_manifest"] == "/remote/runtime/run-000/run_manifest.json"
        assert observation["checkpoints"] == "epoch=01.ckpt;last.ckpt"
        assert observation["checkpoint_count"] == 2
    else:
        assert observation["run_manifest"] == previous["run_manifest"]
        assert observation["checkpoints"] == previous["checkpoints"]
        assert observation["checkpoint_count"] == ""
        assert observation["health_status"] == "health_unknown"
    assert any("/remote/runtime/run-000" in command for command in commands)
