from __future__ import annotations

from contextlib import contextmanager
import csv
import json
from pathlib import Path
import subprocess
import sys

import pytest

from agent_tools import experiment_io, experiment_tracking, experiment_workspace, experiments, run_evidence, slurm


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "agent_tools", *args], text=True, capture_output=True)


def _read_table(path: Path) -> list[dict[str, str]]:
    delimiter = "\t" if path.suffix == ".tsv" else ","
    with path.open(newline="") as file_obj:
        return list(csv.DictReader(file_obj, delimiter=delimiter))


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


def test_experiment_monitor_reads_one_validated_input_snapshot_then_fresh_commit(tmp_path: Path, monkeypatch):
    _initialize_workspace(tmp_path)
    manifest_path = tmp_path / "run_manifest.tsv"
    experiment_io.write_rows_at(
        manifest_path,
        [{"experiment_id": "unit", "step_id": "train-model", "run_id": "run-000", "status": "running"}],
    )
    real_managed_read = experiment_io.read_managed_files_at
    real_read = experiment_io.read_text_at
    real_lock = experiment_io.blocking_file_lock
    real_commit = experiment_io.atomic_replace
    real_managed_rows = experiments._managed_rows
    snapshot_reads = []
    commit_reads = []
    commits = []
    snapshots = []
    observation_started = False
    lock_held = False

    def read_managed(root, paths, **kwargs):
        if manifest_path in paths:
            snapshot_reads.append(Path(root))
        return real_managed_read(root, paths, **kwargs)

    def read_text(path):
        if Path(path) == manifest_path:
            commit_reads.append((observation_started, lock_held))
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

    def managed_rows(root):
        rows = real_managed_rows(root)
        snapshots.append((rows, [dict(row) for row in rows]))
        return rows

    def fake_status(_root, row, previous, *, script_commits_terminal_status, health):
        nonlocal observation_started
        observation_started = True
        assert row is not previous
        assert previous["status"] == "running"
        row.update(status="completed", health_status="completed")
        return row

    def commit(path, text):
        assert lock_held
        commits.append(Path(path))
        return real_commit(path, text)

    monkeypatch.setattr(experiment_io, "read_managed_files_at", read_managed)
    monkeypatch.setattr(experiment_io, "read_text_at", read_text)
    monkeypatch.setattr(experiment_io, "blocking_file_lock", held_lock)
    monkeypatch.setattr(experiment_io, "atomic_replace", commit)
    monkeypatch.setattr(experiments, "_managed_rows", managed_rows)
    monkeypatch.setattr(run_evidence, "status_row", fake_status)
    result = experiments.monitor_experiment(tmp_path)

    assert snapshot_reads == [tmp_path]
    assert commit_reads == [(True, True)]
    # The status change commits the manifest, then appends its run_status_changed event.
    assert commits == [manifest_path, tmp_path / "events.jsonl"]
    assert len(snapshots) == 1
    assert snapshots[0][0] == snapshots[0][1]
    assert result["runs"][0]["status"] == "completed"
    assert _read_table(manifest_path)[0]["status"] == "completed"
    transitions = [
        (event["from"], event["to"])
        for event in experiment_workspace.read_experiment_events(tmp_path)
        if event["event_type"] == "run_status_changed"
    ]
    assert transitions == [("running", "completed")]


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


def test_experiment_monitor_routes_slurm_observation_through_the_shared_job_snapshot(tmp_path: Path, monkeypatch):
    row = {
        "step_id": "train-model",
        "run_id": "run-000",
        "script": "/plan/run.sh",
        "scheduler_type": "slurm",
        "scheduler_script": "/plan/job.sbatch",
        "scheduler_script_sha256": "a" * 64,
        "scheduler_job_id": "3880",
        "target": "local",
        "status": "queued",
    }
    jobs = {"3880": slurm.JobObservation("3880", "exp/train-model/run-000", "PENDING", "Resources", "")}
    observed = []

    def fake_slurm_observation(owner_dir, observed_row, slurm_jobs, *, health):
        observed.append((owner_dir, dict(observed_row), slurm_jobs, health))
        observed_row["log_tail"] = "observer-owned update"
        return {
            **observed_row,
            "scheduler_raw_state": "PENDING",
            "scheduler_reason": "Resources",
            "scheduler_node": "",
            "health_status": "scheduler_queued",
            "status": "queued",
            "monitored_at": "now",
            "unrelated_observer_field": "dropped",
        }

    monkeypatch.setattr(experiment_tracking.managed_scheduler, "observe_slurm_run", fake_slurm_observation)
    monkeypatch.setattr(
        run_evidence,
        "status_row",
        lambda *_args, **_kwargs: pytest.fail("Slurm runs must not use direct PID evidence"),
    )

    snapshot = dict(row)
    observation = experiment_tracking.monitor_run_row(tmp_path, row, [row], slurm_jobs=jobs)

    assert row == snapshot
    assert observed == [(tmp_path, row, jobs, True)]
    assert observation["status"] == "queued"
    assert observation["scheduler_job_id"] == "3880"
    assert observation["scheduler_raw_state"] == "PENDING"
    assert observation["scheduler_reason"] == "Resources"
    assert observation["health_status"] == "scheduler_queued"
    assert "unrelated_observer_field" not in observation


@pytest.mark.parametrize("binding", [{}, {"scheduler_job_id": "3880"}])
def test_experiment_monitor_observes_a_slurm_run_once_it_has_execution_identity_or_a_job(
    tmp_path: Path, monkeypatch, binding: dict
):
    row = {
        "step_id": "train-model",
        "run_id": "run-000",
        "scheduler_type": "slurm",
        "scheduler_script": "/plan/job.sbatch",
        "scheduler_script_sha256": "a" * 64,
        "status": "stopping" if binding else "planned",
        **binding,
    }
    observed = []
    monkeypatch.setattr(
        experiment_tracking.managed_scheduler,
        "observe_slurm_run",
        lambda _root, observed_row, *_args, **_kwargs: observed.append(observed_row) or dict(observed_row),
    )

    observation = experiment_tracking.monitor_run_row(tmp_path, dict(row), [dict(row)])

    assert observation["status"] == row["status"]
    assert "target" not in observation
    # A stop can bind a job before any submit pass recorded the run's execution identity.
    assert len(observed) == (1 if binding else 0)


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

    def merge_after_concurrent_update(root, rows):
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
        )
        return real_merge(root, rows)

    monkeypatch.setattr(experiments, "merge_run_manifest", merge_after_concurrent_update)

    result = experiments.monitor_experiment(tmp_path)

    assert [(row["run_id"], row["status"]) for row in result["runs"]] == [
        ("run-000", concurrent_fields["status"]),
        ("run-001", "planned"),
    ]
    for field, value in concurrent_fields.items():
        assert result["runs"][0][field] == value
    assert not (tmp_path / "run_matrix.csv").exists()
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
