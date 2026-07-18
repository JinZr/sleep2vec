from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys
import types

import pytest

from agent_tools import experiment_io, experiment_tracking, experiment_workspace, experiments, run_evidence


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "agent_tools", *args], text=True, capture_output=True)


def _read_table(path: Path) -> list[dict[str, str]]:
    delimiter = "\t" if path.suffix == ".tsv" else ","
    with path.open(newline="") as file_obj:
        return list(csv.DictReader(file_obj, delimiter=delimiter))


def _experiment_spec(tmp_path: Path) -> Path:
    path = tmp_path / "experiment_spec.yaml"
    path.write_text(
        "id: unit\n"
        "title: Unit experiment\n"
        "objective: Exercise experiment workspace contracts.\n"
        "baseline:\n"
        "  type: none\n"
        "  rationale: Unit fixture.\n"
    )
    return path


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


def test_experiment_indexes_checkpoints_and_ranks_validation_metric(tmp_path: Path):
    _initialize_workspace(tmp_path)
    run_dir = tmp_path / "log-finetune" / "run_a"
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    ckpt = ckpt_dir / "epoch=02-step=20.ckpt"
    ckpt.write_text("checkpoint")
    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "best_model_path": str(ckpt_dir / "best-epoch=02-step=20.ckpt"),
                "epoch": 2,
                "metrics": {"val_auroc": 0.8},
            }
        )
    )
    (tmp_path / "metrics_manifest.tsv").write_text(
        "step_id\trun_id\tversion\tepoch\tmetric\tvalue\tmetric_scope\tsource\n"
        "train-model\trun-000\trun_a\t1\tval_auroc\t0.6\tvalidation\twandb_history\n"
        "train-model\trun-000\trun_a\t2\tval_auroc\t0.8\tvalidation\twandb_history\n"
    )
    (tmp_path / "run_manifest.tsv").write_text(
        "experiment_id\tstep_id\trun_id\trun_name\tversion\truntime_dir\tcheckpoint_dir\tstatus\n"
        f"unit\ttrain-model\trun-000\tmanaged\trun_a\t{run_dir}\t{ckpt_dir}\tcompleted\n"
    )
    hparam_ranking = tmp_path / "reports" / "ranking.csv"
    hparam_ranking.parent.mkdir(parents=True)
    hparam_ranking.write_text("step_id,run_id,rank\ntune-model,run-000,1\n")
    original_hparam_ranking = hparam_ranking.read_text()

    checkpoints = _run("experiment-index-checkpoints", "--run-dir", str(tmp_path))
    ranking = _run("experiment-rank", "--run-dir", str(tmp_path), "--metric", "val_auroc", "--mode", "max")

    assert checkpoints.returncode == 0, checkpoints.stderr
    assert ranking.returncode == 0, ranking.stderr
    checkpoint_rows = _read_table(tmp_path / "checkpoint_manifest.tsv")
    assert checkpoint_rows[0]["epoch"] == "02"
    assert hparam_ranking.read_text() == original_hparam_ranking
    ranked = _read_table(tmp_path / "reports" / "experiment_ranking.csv")
    assert ranked[0]["version"] == "run_a"
    assert ranked[0]["score"] == "0.8"
    assert ranked[0]["checkpoint_path"].endswith("epoch=02-step=20.ckpt")
    assert (tmp_path / "reports" / "experiment_ranking.md").exists()


@pytest.mark.parametrize("epoch", ["2", "2.5"])
def test_experiment_rank_does_not_fallback_to_checkpoint_from_another_epoch(epoch: str):
    metric = {"step_id": "train-model", "run_id": "run-000", "epoch": epoch}
    checkpoints = [
        {
            "step_id": "train-model",
            "run_id": "run-000",
            "epoch": "5",
            "checkpoint_path": "/runtime/checkpoints/epoch=5.ckpt",
            "is_best_by_val": "true",
            "is_last": "false",
        },
        {
            "step_id": "train-model",
            "run_id": "run-000",
            "epoch": "",
            "checkpoint_path": "/runtime/checkpoints/last.ckpt",
            "is_best_by_val": "false",
            "is_last": "true",
        },
    ]

    assert experiment_tracking._checkpoint_for_metric_row(metric, checkpoints) == ""


def test_experiment_indexes_checkpoints_from_managed_runtime_dir(tmp_path: Path):
    workspace = tmp_path / "workspace"
    _initialize_workspace(workspace)
    runtime_dir = tmp_path / "log-finetune" / "managed-v1"
    checkpoint_dir = runtime_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    checkpoint = checkpoint_dir / "epoch=03-step=30.ckpt"
    checkpoint.write_text("checkpoint")
    (workspace / "run_manifest.tsv").write_text(
        "experiment_id\tstep_id\trun_id\tversion\truntime_dir\tcheckpoint_dir\tstatus\n"
        f"unit\ttrain-model\trun-000\tmanaged-v1\t{runtime_dir}\t{checkpoint_dir}\tcompleted\n"
    )

    manifest = experiments.index_checkpoints(workspace)

    rows = _read_table(manifest)
    assert len(rows) == 1
    assert rows[0]["version"] == "managed-v1"
    assert rows[0]["checkpoint_path"] == str(checkpoint)


@pytest.mark.parametrize("empty_field", ["runtime_dir", "checkpoint_dir"])
@pytest.mark.parametrize("remote", [None, "baichuan3"])
def test_experiment_checkpoint_preflight_rejects_partial_paths_before_scan(
    tmp_path: Path, monkeypatch, empty_field: str, remote: str | None
):
    _initialize_workspace(tmp_path)
    run = {
        "experiment_id": "unit",
        "step_id": "train-model",
        "run_id": "run-000",
        "version": "managed-v1",
        "runtime_dir": "/runtime/managed-v1",
        "checkpoint_dir": "/runtime/managed-v1/checkpoints",
        "status": "completed",
    }
    run[empty_field] = ""
    experiment_io.write_rows_at(tmp_path / "run_manifest.tsv", [run])
    checkpoint_manifest = tmp_path / "checkpoint_manifest.tsv"
    checkpoint_manifest.write_text("step_id\trun_id\tcheckpoint_path\ntrain-model\trun-000\t/runtime/old.ckpt\n")
    original_manifest = checkpoint_manifest.read_bytes()
    scan_calls = []
    writes = []

    monkeypatch.setattr(
        experiment_tracking,
        "_local_checkpoint_rows",
        lambda runs: scan_calls.append(("local", runs)) or [],
    )
    monkeypatch.setattr(
        experiment_tracking,
        "_remote_checkpoint_rows",
        lambda runs, host: scan_calls.append((host, runs)) or [],
    )
    if remote:
        monkeypatch.setattr(experiment_io, "validate_managed_output_paths", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(
            experiment_io,
            "read_text_at",
            lambda path, remote=None: Path(path).read_text() if Path(path).exists() else "",
        )
        monkeypatch.setattr(
            experiment_io,
            "path_exists_at",
            lambda path, remote=None: Path(path).exists(),
        )
        monkeypatch.setattr(
            experiment_io,
            "read_rows_at",
            lambda path, remote=None, **_kwargs: _read_table(Path(path)) if Path(path).exists() else [],
        )
        monkeypatch.setattr(
            experiment_io,
            "write_rows_at",
            lambda path, rows, remote=None: writes.append((Path(path), rows)),
        )

    with pytest.raises(ValueError, match="frozen artifact paths"):
        experiments.index_checkpoints(tmp_path, remote=remote)

    assert scan_calls == []
    assert writes == []
    assert checkpoint_manifest.read_bytes() == original_manifest


@pytest.mark.parametrize("previous_owner", ["unmanaged", "non_checkpoint_run"])
def test_experiment_checkpoint_preflight_rejects_previous_rows_outside_eligible_runs(
    tmp_path: Path, monkeypatch, previous_owner: str
):
    _initialize_workspace(tmp_path)
    run = {
        "experiment_id": "unit",
        "step_id": "train-model",
        "run_id": "run-000",
        "version": "managed-v1",
        "runtime_dir": "/runtime/managed-v1",
        "checkpoint_dir": "/runtime/managed-v1/checkpoints",
        "status": "completed",
    }
    previous_run_id = "run-999"
    if previous_owner == "non_checkpoint_run":
        run["runtime_dir"] = ""
        run["checkpoint_dir"] = ""
        previous_run_id = "run-000"
    experiment_io.write_rows_at(tmp_path / "run_manifest.tsv", [run])
    checkpoint_manifest = tmp_path / "checkpoint_manifest.tsv"
    checkpoint_manifest.write_text(
        "step_id\trun_id\tcheckpoint_path\n" f"train-model\t{previous_run_id}\t/runtime/old.ckpt\n"
    )
    original_manifest = checkpoint_manifest.read_bytes()
    scan_calls = []
    monkeypatch.setattr(
        experiment_tracking,
        "_local_checkpoint_rows",
        lambda runs: scan_calls.append(runs) or [],
    )

    with pytest.raises(ValueError, match="eligible managed run"):
        experiments.index_checkpoints(tmp_path)

    assert scan_calls == []
    assert checkpoint_manifest.read_bytes() == original_manifest


@pytest.mark.parametrize("directory_name", ["runtime", "checkpoint"])
@pytest.mark.parametrize("directory_kind", ["file", "symlink"])
def test_experiment_checkpoint_scan_rejects_invalid_declared_directory_without_rewriting(
    tmp_path: Path, directory_name: str, directory_kind: str
):
    _initialize_workspace(tmp_path)
    runtime_dir = tmp_path / "runtime"
    checkpoint_dir = runtime_dir / "checkpoints"
    invalid_dir = runtime_dir if directory_name == "runtime" else checkpoint_dir
    if directory_name == "checkpoint":
        runtime_dir.mkdir()
    if directory_kind == "file":
        invalid_dir.write_text("not a directory")
    else:
        target = tmp_path / f"external-{directory_name}"
        target.mkdir()
        invalid_dir.symlink_to(target, target_is_directory=True)
    (tmp_path / "run_manifest.tsv").write_text(
        "experiment_id\tstep_id\trun_id\tversion\truntime_dir\tcheckpoint_dir\n"
        f"unit\ttrain-model\trun-000\tmanaged-v1\t{runtime_dir}\t{checkpoint_dir}\n"
    )
    checkpoint_manifest = tmp_path / "checkpoint_manifest.tsv"
    checkpoint_manifest.write_text(
        "step_id\trun_id\tcheckpoint_path\n" f"train-model\trun-000\t{checkpoint_dir / 'old.ckpt'}\n"
    )
    before = checkpoint_manifest.read_bytes()

    with pytest.raises(
        ValueError,
        match="runtime_dir is not a directory|checkpoint_dir is not a directory|not a regular managed checkpoint",
    ):
        experiments.index_checkpoints(tmp_path)

    assert checkpoint_manifest.read_bytes() == before


@pytest.mark.parametrize("missing_path", ["runtime", "checkpoint"])
def test_experiment_checkpoint_scan_skips_uncreated_directories_and_indexes_other_runs(
    tmp_path: Path, missing_path: str
):
    _initialize_workspace(tmp_path)
    missing_runtime = tmp_path / "runtime-missing"
    missing_checkpoint = missing_runtime / "checkpoints"
    if missing_path == "checkpoint":
        missing_runtime.mkdir()
    ready_runtime = tmp_path / "runtime-ready"
    ready_checkpoint = ready_runtime / "checkpoints"
    ready_checkpoint.mkdir(parents=True)
    checkpoint = ready_checkpoint / "epoch=01-step=10.ckpt"
    checkpoint.write_text("checkpoint")
    experiment_io.write_rows_at(
        tmp_path / "run_manifest.tsv",
        [
            {
                "experiment_id": "unit",
                "step_id": "train-model",
                "run_id": "run-000",
                "version": "missing-v1",
                "runtime_dir": str(missing_runtime),
                "checkpoint_dir": str(missing_checkpoint),
            },
            {
                "experiment_id": "unit",
                "step_id": "train-model",
                "run_id": "run-001",
                "version": "ready-v1",
                "runtime_dir": str(ready_runtime),
                "checkpoint_dir": str(ready_checkpoint),
            },
        ],
    )

    experiments.index_checkpoints(tmp_path)

    rows = _read_table(tmp_path / "checkpoint_manifest.tsv")
    assert [(row["run_id"], row["checkpoint_path"]) for row in rows] == [("run-001", str(checkpoint))]


def test_experiment_checkpoint_scan_preserves_existing_inventory_when_directory_disappears(tmp_path: Path):
    _initialize_workspace(tmp_path)
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()
    checkpoint_dir = runtime_dir / "checkpoints"
    experiment_io.write_rows_at(
        tmp_path / "run_manifest.tsv",
        [
            {
                "experiment_id": "unit",
                "step_id": "train-model",
                "run_id": "run-000",
                "version": "managed-v1",
                "runtime_dir": str(runtime_dir),
                "checkpoint_dir": str(checkpoint_dir),
            }
        ],
    )
    checkpoint_manifest = tmp_path / "checkpoint_manifest.tsv"
    checkpoint_manifest.write_text(
        "experiment_id\tstep_id\trun_id\tversion\tcheckpoint_path\n"
        f"unit\ttrain-model\trun-000\tmanaged-v1\t{checkpoint_dir / 'epoch=01.ckpt'}\n"
    )
    before = checkpoint_manifest.read_bytes()

    with pytest.raises(ValueError, match="existing checkpoint inventory"):
        experiments.index_checkpoints(tmp_path)

    assert checkpoint_manifest.read_bytes() == before


def test_experiment_checkpoint_preflight_rejects_foreign_experiment_before_scan(tmp_path: Path, monkeypatch):
    _initialize_workspace(tmp_path)
    experiment_io.write_rows_at(
        tmp_path / "run_manifest.tsv",
        [
            {
                "experiment_id": "unit",
                "step_id": "train-model",
                "run_id": "run-000",
                "version": "managed-v1",
                "runtime_dir": "/runtime/managed-v1",
                "checkpoint_dir": "/runtime/managed-v1/checkpoints",
                "status": "completed",
            }
        ],
    )
    checkpoint_manifest = tmp_path / "checkpoint_manifest.tsv"
    checkpoint_manifest.write_text(
        "experiment_id\tstep_id\trun_id\tversion\tcheckpoint_path\n"
        "other\ttrain-model\trun-000\tmanaged-v1\t/runtime/old.ckpt\n"
    )
    before = checkpoint_manifest.read_bytes()
    scan_calls = []
    monkeypatch.setattr(
        experiment_tracking,
        "_local_checkpoint_rows",
        lambda runs: scan_calls.append(runs) or [],
    )

    with pytest.raises(ValueError, match="Frozen run field differs.*experiment_id"):
        experiments.index_checkpoints(tmp_path)

    assert scan_calls == []
    assert checkpoint_manifest.read_bytes() == before


def test_experiment_checkpoint_preflight_rejects_foreign_metrics_before_scan(tmp_path: Path, monkeypatch):
    _initialize_workspace(tmp_path)
    experiment_io.write_rows_at(
        tmp_path / "run_manifest.tsv",
        [
            {
                "experiment_id": "unit",
                "step_id": "train-model",
                "run_id": "run-000",
                "version": "managed-v1",
                "runtime_dir": "/runtime/managed-v1",
                "checkpoint_dir": "/runtime/managed-v1/checkpoints",
                "status": "completed",
            }
        ],
    )
    (tmp_path / "metrics_manifest.tsv").write_text(
        "experiment_id\tstep_id\trun_id\tversion\tmetric\tvalue\n"
        "other\ttrain-model\trun-000\tmanaged-v1\tval_auroc\t0.99\n"
    )
    scan_calls = []
    monkeypatch.setattr(
        experiment_tracking,
        "_local_checkpoint_rows",
        lambda runs: scan_calls.append(runs) or [],
    )

    with pytest.raises(ValueError, match="Frozen run field differs.*experiment_id"):
        experiments.index_checkpoints(tmp_path)

    assert scan_calls == []


@pytest.mark.parametrize("existing_manifest", [False, True])
def test_experiment_checkpoint_index_skips_runs_without_checkpoint_paths(
    tmp_path: Path, monkeypatch, existing_manifest: bool
):
    _initialize_workspace(tmp_path)
    experiment_io.write_rows_at(
        tmp_path / "run_manifest.tsv",
        [
            {
                "experiment_id": "unit",
                "step_id": "prepare-data",
                "run_id": "run-000",
                "version": "prepare-v1",
                "runtime_dir": "",
                "checkpoint_dir": "",
                "status": "completed",
            }
        ],
    )
    checkpoint_manifest = tmp_path / "checkpoint_manifest.tsv"
    if existing_manifest:
        checkpoint_manifest.write_text("step_id\trun_id\n")
    scan_calls = []
    monkeypatch.setattr(
        experiment_tracking,
        "_local_checkpoint_rows",
        lambda runs: scan_calls.append(runs) or [],
    )

    manifest = experiments.index_checkpoints(tmp_path)

    assert scan_calls == []
    assert _read_table(manifest) == []


def test_experiment_ranking_keeps_same_run_id_from_different_steps(tmp_path: Path):
    _initialize_workspace(tmp_path)
    (tmp_path / "run_manifest.tsv").write_text(
        "experiment_id\tstep_id\trun_id\trun_name\tversion\tstatus\n"
        "unit\ttune-router\trun-000\trouter-frozen\trouter-v1\tcompleted\n"
        "unit\ttune-lr\trun-000\tlr-2e-6\tlr-v1\tcompleted\n"
    )
    (tmp_path / "metrics_manifest.tsv").write_text(
        "step_id\trun_id\tversion\tepoch\tmetric\tvalue\tmetric_scope\tsource\n"
        "tune-router\trun-000\trouter-v1\t1\tval_auroc\t0.7\tvalidation\twandb_history\n"
        "tune-lr\trun-000\tlr-v1\t1\tval_auroc\t0.8\tvalidation\twandb_history\n"
    )

    ranking = experiments.rank_experiment_candidates(tmp_path, metric="val_auroc", mode="max")

    rows = _read_table(ranking)
    assert len(rows) == 2
    assert {(row["step_id"], row["run_id"], row["run_name"]) for row in rows} == {
        ("tune-router", "run-000", "router-frozen"),
        ("tune-lr", "run-000", "lr-2e-6"),
    }


def test_experiment_wandb_sync_exports_summary_history_and_metrics(tmp_path: Path, monkeypatch):
    class FakeRun:
        id = "run123"
        name = "wandb-display-name"
        state = "finished"
        url = "https://wandb.ai/entity/project/runs/run123"
        group = "unit_group"
        created_at = "2026-01-01"
        updated_at = "2026-01-02"
        summary = {"val_auroc": 0.71, "test_auroc": 0.66, "epoch": 3}
        config = {"experiment_id": "unit", "step_id": "train-model", "run_id": "run-000"}

        def history(self, **_kwargs):
            return [{"epoch": 1, "val_auroc": 0.6}, {"epoch": 2, "val_auroc": 0.72}]

    class FakeApi:
        def runs(self, path, filters=None):
            assert path == "entity/project"
            assert filters == {"group": "unit_group"}
            unmatched = FakeRun()
            unmatched.id = "unmatched123"
            unmatched.name = "unmatched-v1"
            unmatched.config = {}
            unmatched.summary = {"val_auroc": 0.99, "epoch": 4}
            return [FakeRun(), unmatched]

    monkeypatch.setitem(sys.modules, "wandb", types.SimpleNamespace(Api=lambda: FakeApi()))
    spec = _experiment_spec(tmp_path.parent)
    assert _run("experiment-init", "--run-dir", str(tmp_path), "--spec", str(spec)).returncode == 0
    (tmp_path / "run_manifest.tsv").write_text(
        "experiment_id\tstep_id\trun_id\trun_name\tversion\tstatus\n"
        "unit\ttrain-model\trun-000\tlr-2e-6\trun_a\tplanned\n"
    )

    out = experiments.sync_wandb_runs(tmp_path, entity="entity", project="project", group="unit_group")

    assert out == tmp_path / "wandb" / "runs.tsv"
    run_rows = _read_table(out)
    assert {row["version"] for row in run_rows} == {"wandb-display-name", "unmatched-v1"}
    assert (tmp_path / "wandb" / "history" / "run123.csv").exists()
    metric_rows = _read_table(tmp_path / "metrics_manifest.tsv")
    assert {row["wandb_run_id"] for row in metric_rows} == {"run123"}
    assert {row["version"] for row in metric_rows} == {"run_a"}
    scopes = {row["metric"]: row["metric_scope"] for row in metric_rows}
    assert scopes["val_auroc"] == "validation"
    assert scopes["test_auroc"] == "test_or_external"
    managed_rows = _read_table(tmp_path / "run_manifest.tsv")
    assert len(managed_rows) == 1
    assert managed_rows[0]["step_id"] == "train-model"
    assert managed_rows[0]["run_id"] == "run-000"
    assert managed_rows[0]["run_name"] == "lr-2e-6"
    assert managed_rows[0]["version"] == "run_a"
    assert managed_rows[0]["state"] == "finished"
    assert managed_rows[0]["status"] == "completed"
    ranking = experiments.rank_experiment_candidates(tmp_path, metric="val_auroc", mode="max")
    ranked = _read_table(ranking)
    assert len(ranked) == 1
    assert ranked[0]["run_id"] == "run-000"
    assert ranked[0]["run_name"] == "lr-2e-6"
    report = tmp_path.parent / "wandb_final.md"
    report.write_text("# Final\n")
    assert _run("experiment-finalize", "--run-dir", str(tmp_path), "--report", str(report)).returncode == 0


def test_experiment_wandb_sync_replaces_updated_metric_on_repeat(tmp_path: Path, monkeypatch):
    _initialize_workspace(tmp_path)
    (tmp_path / "run_manifest.tsv").write_text(
        "experiment_id\tstep_id\trun_id\tversion\tstatus\n" "unit\ttrain-model\trun-000\tmanaged-v1\trunning\n"
    )

    class FakeRun:
        id = "wandb-1"
        name = "managed-v1"
        state = "running"
        url = "https://wandb.example/run"
        group = ""
        created_at = "2026-01-01"
        updated_at = "2026-01-02"
        config = {"experiment_id": "unit", "step_id": "train-model", "run_id": "run-000"}

        def __init__(self, value):
            self.summary = {"val_auroc": value, "epoch": 1}

        def history(self, **_kwargs):
            return []

    current = [FakeRun(0.7)]
    monkeypatch.setattr(experiment_tracking, "wandb_runs", lambda *_args: current)

    experiments.sync_wandb_runs(tmp_path, entity="entity", project="project")
    current[:] = [FakeRun(0.8)]
    experiments.sync_wandb_runs(tmp_path, entity="entity", project="project")

    rows = _read_table(tmp_path / "metrics_manifest.tsv")
    matching = [row for row in rows if row["metric"] == "val_auroc" and row["source"] == "wandb_summary"]
    assert len(matching) == 1
    assert matching[0]["value"] == "0.8"


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


def test_experiment_monitor_treats_hparam_script_with_process_identity_as_monitor_owned(tmp_path: Path, monkeypatch):
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
    experiment_io.write_rows_at(tmp_path / "run_manifest.tsv", [row])
    monkeypatch.setattr(run_evidence, "process_identity_running", lambda *_args: False)

    result = experiments.monitor_experiment(tmp_path)

    assert result["runs"][0]["status"] == "completed"
    assert _read_table(tmp_path / "run_manifest.tsv")[0]["status"] == "completed"


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


@pytest.mark.parametrize(
    ("existing_status", "wandb_state", "expected_status"),
    [("stopped", "running", "stopped"), ("completed", "failed", "failed")],
)
def test_experiment_wandb_sync_applies_directional_terminal_precedence(
    tmp_path: Path, monkeypatch, existing_status: str, wandb_state: str, expected_status: str
):
    _initialize_workspace(tmp_path)

    class FakeRun:
        id = "run123"
        name = "run_a"
        state = ""
        url = "https://wandb.ai/entity/project/runs/run123"
        group = "unit_group"
        created_at = "2026-01-01"
        updated_at = "2026-01-02"
        summary = {}
        config = {"run_id": "run-000"}

        def history(self, **_kwargs):
            return []

    class FakeApi:
        def runs(self, path, filters=None):
            return [FakeRun()]

    FakeRun.state = wandb_state
    monkeypatch.setitem(sys.modules, "wandb", types.SimpleNamespace(Api=lambda: FakeApi()))
    (tmp_path / "run_manifest.tsv").write_text(
        "experiment_id\tstep_id\trun_id\trun_name\tversion\tstatus\n"
        f"unit\ttrain-model\trun-000\tlr-2e-6\trun_a\t{existing_status}\n"
    )

    experiments.sync_wandb_runs(tmp_path, entity="entity", project="project", group="unit_group")
    synced = _read_table(tmp_path / "run_manifest.tsv")
    assert synced[0]["status"] == expected_status
    assert synced[0]["state"] == wandb_state
    assert synced[0]["step_id"] == "train-model"
    assert synced[0]["run_id"] == "run-000"


def test_wandb_observations_strip_matching_and_frozen_fields():
    managed = [
        {
            "experiment_id": "unit",
            "step_id": "train-model",
            "run_id": "run-000",
            "run_name": "managed",
            "version": "managed-v1",
            "config": "/config.yaml",
            "status": "planned",
        }
    ]

    observations = experiment_tracking.wandb_run_observations(
        managed,
        [
            {
                "version": "managed-v1",
                "experiment_id": "other",
                "run_name": "display-name",
                "config": "other.yaml",
                "status": "running",
                "state": "running",
                "wandb_run_id": "wandb-1",
            },
            {
                "step_id": "train-model",
                "run_id": "run-000",
                "experiment_id": "unit",
                "version": "different-display-name",
                "status": "failed",
                "state": "failed",
                "wandb_url": "https://wandb.example/run",
            },
        ],
    )

    assert observations == [
        {
            "step_id": "train-model",
            "run_id": "run-000",
            "status": "failed",
            "state": "failed",
            "wandb_url": "https://wandb.example/run",
        }
    ]


def test_wandb_without_experiment_id_uses_unique_version_instead_of_managed_key():
    managed = [
        {"experiment_id": "unit", "step_id": "step-a", "run_id": "run-000", "version": "version-a"},
        {"experiment_id": "unit", "step_id": "step-b", "run_id": "run-000", "version": "version-b"},
    ]

    observations = experiment_tracking.wandb_run_observations(
        managed,
        [
            {
                "step_id": "step-a",
                "run_id": "run-000",
                "version": "version-b",
                "status": "running",
            }
        ],
    )

    assert observations == [{"step_id": "step-b", "run_id": "run-000", "status": "running"}]


def test_wandb_sync_rejects_multiple_attempts_for_one_managed_run_before_writes(tmp_path: Path, monkeypatch):
    _initialize_workspace(tmp_path)
    (tmp_path / "run_manifest.tsv").write_text(
        "experiment_id\tstep_id\trun_id\trun_name\tversion\tstatus\n"
        "unit\ttrain-model\trun-000\tmanaged\tmanaged-v1\trunning\n"
    )

    class FakeRun:
        url = ""
        group = ""
        created_at = "2026-01-01"
        updated_at = "2026-01-02"
        name = "managed-v1"
        state = "running"
        config = {"experiment_id": "unit", "step_id": "train-model", "run_id": "run-000"}

        def __init__(self, wandb_run_id: str, score: float):
            self.id = wandb_run_id
            self.summary = {"val_auroc": score}

        def history(self, **_kwargs):
            return []

    monkeypatch.setattr(
        experiment_tracking,
        "wandb_runs",
        lambda *_args: [FakeRun("wandb-attempt-1", 0.7), FakeRun("wandb-attempt-2", 0.8)],
    )
    before = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}

    with pytest.raises(ValueError, match="Ambiguous W&B runs for managed run train-model / run-000"):
        experiments.sync_wandb_runs(tmp_path, entity="entity", project="project")

    assert {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before


@pytest.mark.parametrize("foreign_first", [False, True])
def test_wandb_sync_keeps_foreign_experiment_in_raw_inventory_only(tmp_path: Path, monkeypatch, foreign_first: bool):
    _initialize_workspace(tmp_path, experiment_id="experiment-a")
    (tmp_path / "run_manifest.tsv").write_text(
        "experiment_id\tstep_id\trun_id\trun_name\tversion\tstatus\n"
        "experiment-a\ttrain-model\trun-000\tmanaged\texperiment-a-v1\trunning\n"
    )

    class FakeRun:
        url = ""
        group = ""
        created_at = "2026-01-01"
        updated_at = "2026-01-02"

        def __init__(self, experiment_id: str, run_id: str, name: str, state: str, score: float):
            self.id = run_id
            self.name = name
            self.state = state
            self.summary = {"val_auroc": score, "epoch": 1}
            self.config = {
                "experiment_id": experiment_id,
                "step_id": "train-model",
                "run_id": "run-000",
            }

        def history(self, **_kwargs):
            return []

    local = FakeRun("experiment-a", "local-wandb", "experiment-a-v1", "finished", 0.7)
    foreign = FakeRun("experiment-b", "foreign-wandb", "experiment-b-v1", "failed", 0.99)
    runs = [foreign, local] if foreign_first else [local, foreign]
    monkeypatch.setattr(experiment_tracking, "wandb_runs", lambda *_args: runs)

    experiments.sync_wandb_runs(tmp_path, entity="entity", project="shared-project")

    raw_rows = _read_table(tmp_path / "wandb" / "runs.tsv")
    assert {row["wandb_run_id"] for row in raw_rows} == {"local-wandb", "foreign-wandb"}
    canonical = _read_table(tmp_path / "run_manifest.tsv")
    assert canonical[0]["status"] == "completed"
    assert canonical[0]["wandb_run_id"] == "local-wandb"
    metrics = _read_table(tmp_path / "metrics_manifest.tsv")
    assert {row["wandb_run_id"] for row in metrics} == {"local-wandb"}
    assert {row["experiment_id"] for row in metrics} == {"experiment-a"}


def test_wandb_sync_rejects_foreign_existing_metrics_before_api_or_write(tmp_path: Path, monkeypatch):
    _initialize_workspace(tmp_path)
    (tmp_path / "run_manifest.tsv").write_text(
        "experiment_id\tstep_id\trun_id\tversion\tstatus\n" "unit\ttrain-model\trun-000\tmanaged-v1\trunning\n"
    )
    (tmp_path / "metrics_manifest.tsv").write_text(
        "experiment_id\tstep_id\trun_id\tversion\tmetric\tvalue\n"
        "other\ttrain-model\trun-000\tmanaged-v1\tval_auroc\t0.99\n"
    )
    before = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    calls = []
    monkeypatch.setattr(experiment_tracking, "wandb_runs", lambda *_args: calls.append("wandb") or [])

    with pytest.raises(ValueError, match="Frozen run field differs.*experiment_id"):
        experiments.sync_wandb_runs(tmp_path, entity="entity", project="project")

    assert calls == []
    assert {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before


def test_wandb_sync_rejects_invalid_canonical_target_before_api_or_write(tmp_path: Path, monkeypatch):
    _initialize_workspace(tmp_path)
    experiment_workspace.initialize_run_manifest(tmp_path)
    (tmp_path / "run_matrix.csv").mkdir()
    before = {path.relative_to(tmp_path): path.read_bytes() if path.is_file() else None for path in tmp_path.rglob("*")}
    calls = []
    monkeypatch.setattr(experiment_tracking, "wandb_runs", lambda *_args: calls.append("wandb") or [])

    with pytest.raises(ValueError, match="Managed output"):
        experiments.sync_wandb_runs(tmp_path, entity="entity", project="project")

    assert calls == []
    assert {
        path.relative_to(tmp_path): path.read_bytes() if path.is_file() else None for path in tmp_path.rglob("*")
    } == before


def test_wandb_sync_commits_observations_against_latest_canonical_rows(tmp_path: Path, monkeypatch):
    _initialize_workspace(tmp_path)
    experiment_io.write_rows_at(
        tmp_path / "run_manifest.tsv",
        [
            {
                "experiment_id": "unit",
                "step_id": "train-model",
                "run_id": "run-000",
                "version": "managed-v1",
                "status": "running",
            }
        ],
    )

    class FakeRun:
        id = "wandb-1"
        name = "managed-v1"
        state = "finished"
        url = "https://wandb.example/run"
        group = ""
        created_at = "2026-01-01"
        updated_at = "2026-01-02"
        summary = {}
        config = {"step_id": "train-model", "run_id": "run-000"}

        def history(self, **_kwargs):
            return []

    monkeypatch.setattr(experiment_tracking, "wandb_runs", lambda *_args: [FakeRun()])
    real_merge = experiment_workspace.merge_run_manifest

    def merge_after_concurrent_update(root, rows, *, remote=None):
        real_merge(
            root,
            [
                {"step_id": "train-model", "run_id": "run-000", "status": "failed"},
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

    experiments.sync_wandb_runs(tmp_path, entity="entity", project="project")

    assert [(row["run_id"], row["status"]) for row in _read_table(tmp_path / "run_manifest.tsv")] == [
        ("run-000", "failed"),
        ("run-001", "planned"),
    ]


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
    monkeypatch.setattr(experiment_tracking, "read_run_manifest", lambda *_args, **_kwargs: [dict(previous)])
    monkeypatch.setattr(experiment_tracking.exp_io, "validate_managed_output_paths", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(experiment_tracking.exp_io, "read_rows_at", lambda *_args, **_kwargs: [])
    run_rows = experiment_tracking.experiment_run_rows(tmp_path, remote="unit-host")

    observation = experiment_tracking.monitor_run_row(
        tmp_path,
        run_rows[0],
        [dict(previous)],
        remote="unit-host",
    )

    assert observed_rows[0]["target"] == "ssh"
    assert observed_rows[0]["host"] == "unit-host"
    assert "target" not in observation
    assert "host" not in observation


def test_experiment_monitor_reports_latest_committed_rows(tmp_path: Path, monkeypatch):
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
                {"step_id": "train-model", "run_id": "run-000", "status": "failed"},
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
        ("run-000", "failed"),
        ("run-001", "planned"),
    ]
    report = (tmp_path / "reports" / "monitor.md").read_text()
    assert "train-model / run-001" in report


def test_experiment_checkpoint_scan_ignores_checkpoint_named_directories(tmp_path: Path):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "epoch=02.ckpt").mkdir()
    runs = [
        {
            "experiment_id": "unit",
            "step_id": "train-model",
            "run_id": "run-000",
            "version": "managed-v1",
            "runtime_dir": str(tmp_path),
            "checkpoint_dir": str(checkpoint_dir),
        }
    ]

    assert experiment_tracking._local_checkpoint_rows(runs) == []


def test_local_checkpoint_scan_prefers_manifest_epoch_over_best_filename(tmp_path: Path):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    mismatched_best = checkpoint_dir / "best-epoch=01.ckpt"
    matching_epoch = checkpoint_dir / "epoch=02.ckpt"
    mismatched_best.write_text("checkpoint")
    matching_epoch.write_text("checkpoint")
    (tmp_path / "run_manifest.json").write_text(json.dumps({"best_model_path": str(mismatched_best), "epoch": 2}))
    runs = [
        {
            "experiment_id": "unit",
            "step_id": "train-model",
            "run_id": "run-000",
            "version": "managed-v1",
            "runtime_dir": str(tmp_path),
            "checkpoint_dir": str(checkpoint_dir),
        }
    ]

    rows = experiment_tracking._local_checkpoint_rows(runs)

    assert {row["checkpoint_path"]: row["is_best_by_val"] for row in rows} == {
        str(mismatched_best): "false",
        str(matching_epoch): "true",
    }


def test_local_checkpoint_scan_keeps_same_epoch_best_only_fallback(tmp_path: Path):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    checkpoint = checkpoint_dir / "best-epoch=03.ckpt"
    checkpoint.write_text("checkpoint")
    (tmp_path / "run_manifest.json").write_text(json.dumps({"best_model_path": str(checkpoint), "epoch": 3}))
    runs = [
        {
            "experiment_id": "unit",
            "step_id": "train-model",
            "run_id": "run-000",
            "version": "managed-v1",
            "runtime_dir": str(tmp_path),
            "checkpoint_dir": str(checkpoint_dir),
        }
    ]

    rows = experiment_tracking._local_checkpoint_rows(runs)

    assert rows[0]["is_best_by_val"] == "true"
    assert experiment_tracking._checkpoint_for_metric_row(
        {"step_id": "train-model", "run_id": "run-000", "epoch": ""}, rows
    ) == str(checkpoint)
    assert experiment_tracking._checkpoint_for_metric_row(
        {"step_id": "train-model", "run_id": "run-000", "epoch": 3}, rows
    ) == str(checkpoint)
    assert (
        experiment_tracking._checkpoint_for_metric_row(
            {"step_id": "train-model", "run_id": "run-000", "epoch": 4}, rows
        )
        == ""
    )


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


def test_experiment_run_rows_ignores_conflicting_projection_fields(tmp_path: Path):
    (tmp_path / "run_manifest.tsv").write_text(
        "experiment_id\tstep_id\trun_id\trun_name\tparameter_summary\tversion\tconfig\tstatus\tstate\n"
        "unit\ttrain-model\trun-000\tmanaged\tlr=1e-6\tmanaged-v1\t/config.yaml\tplanned\tcanonical\n"
    )
    (tmp_path / "run_status.tsv").write_text(
        "experiment_id\tstep_id\trun_id\trun_name\tparameter_summary\tversion\tconfig\tstatus\tstate\tlog_path\n"
        "other\ttrain-model\trun-000\tcorrupt\tlr=9\tdisplay-name\t/other.yaml\trunning\tauxiliary\t/run.log\n"
    )

    rows = experiment_tracking.experiment_run_rows(tmp_path)

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


def test_candidate_rows_reject_foreign_metric_before_metric_filtering():
    managed = [
        {
            "experiment_id": "unit",
            "step_id": "train-model",
            "run_id": "run-000",
            "version": "managed-v1",
        }
    ]
    metrics = [
        {
            "experiment_id": "other",
            "step_id": "train-model",
            "run_id": "run-000",
            "version": "managed-v1",
            "metric": "unrelated_metric",
            "value": 1,
        }
    ]

    with pytest.raises(ValueError, match="Frozen run field differs.*experiment_id"):
        experiment_tracking.candidate_rows(managed, metrics, "val_auroc")


def test_experiment_rank_rejects_unmanaged_checkpoint_before_ranking(tmp_path: Path):
    _initialize_workspace(tmp_path)
    (tmp_path / "run_manifest.tsv").write_text(
        "experiment_id\tstep_id\trun_id\tversion\tstatus\n" "unit\ttrain-model\trun-000\tmanaged-v1\tcompleted\n"
    )
    (tmp_path / "metrics_manifest.tsv").write_text(
        "experiment_id\tstep_id\trun_id\tversion\tmetric\tvalue\n"
        "unit\ttrain-model\trun-000\tmanaged-v1\tval_auroc\t0.7\n"
    )
    (tmp_path / "checkpoint_manifest.tsv").write_text(
        "experiment_id\tstep_id\trun_id\tversion\tcheckpoint_path\n"
        "unit\ttrain-model\trun-999\tunknown-v1\t/runtime/unknown.ckpt\n"
    )

    with pytest.raises(ValueError, match="outside the canonical manifest"):
        experiments.rank_experiment_candidates(tmp_path, metric="val_auroc", mode="max")


@pytest.mark.parametrize("operation", ["index", "rank"])
def test_experiment_checkpoint_evidence_rejects_path_outside_frozen_directory_before_output(
    tmp_path: Path, monkeypatch, operation: str
):
    _initialize_workspace(tmp_path)
    checkpoint_dir = tmp_path / "managed" / "checkpoints"
    (tmp_path / "run_manifest.tsv").write_text(
        "experiment_id\tstep_id\trun_id\tversion\truntime_dir\tcheckpoint_dir\tstatus\n"
        f"unit\ttrain-model\trun-000\tmanaged-v1\t{checkpoint_dir.parent}\t{checkpoint_dir}\tcompleted\n"
    )
    (tmp_path / "metrics_manifest.tsv").write_text(
        "experiment_id\tstep_id\trun_id\tversion\tepoch\tmetric\tvalue\n"
        "unit\ttrain-model\trun-000\tmanaged-v1\t1\tval_auroc\t0.7\n"
    )
    checkpoint_manifest = tmp_path / "checkpoint_manifest.tsv"
    checkpoint_manifest.write_text(
        "experiment_id\tstep_id\trun_id\tversion\tepoch\tcheckpoint_path\n"
        "unit\ttrain-model\trun-000\tmanaged-v1\t1\t/foreign/epoch=1.ckpt\n"
    )
    before = checkpoint_manifest.read_bytes()
    scan_calls = []
    monkeypatch.setattr(
        experiment_tracking,
        "_local_checkpoint_rows",
        lambda runs: scan_calls.append(runs) or [],
    )

    with pytest.raises(ValueError, match="checkpoint_path.*checkpoint_dir"):
        if operation == "index":
            experiments.index_checkpoints(tmp_path)
        else:
            experiments.rank_experiment_candidates(tmp_path, metric="val_auroc", mode="max")

    assert scan_calls == []
    assert checkpoint_manifest.read_bytes() == before
    assert not (tmp_path / "reports" / "experiment_ranking.csv").exists()


def test_experiment_rank_ignores_managed_checkpoint_without_requested_metric(tmp_path: Path):
    _initialize_workspace(tmp_path)
    (tmp_path / "run_manifest.tsv").write_text(
        "experiment_id\tstep_id\trun_id\tversion\tcheckpoint_dir\tstatus\n"
        "unit\ttrain-model\trun-000\tmanaged-v1\t/runtime\tcompleted\n"
        "unit\ttrain-model\trun-001\tmanaged-v2\t/runtime\tcompleted\n"
    )
    (tmp_path / "metrics_manifest.tsv").write_text(
        "experiment_id\tstep_id\trun_id\tversion\tmetric\tvalue\n"
        "unit\ttrain-model\trun-000\tmanaged-v1\tval_auroc\t0.7\n"
    )
    (tmp_path / "checkpoint_manifest.tsv").write_text(
        "experiment_id\tstep_id\trun_id\tversion\tcheckpoint_path\n"
        "unit\ttrain-model\trun-001\tmanaged-v2\t/runtime/managed-v2.ckpt\n"
    )

    ranking = experiments.rank_experiment_candidates(tmp_path, metric="val_auroc", mode="max")

    assert [(row["run_id"], row["checkpoint_path"]) for row in _read_table(ranking)] == [("run-000", "")]


def test_wandb_observations_only_accept_allowed_fields(tmp_path: Path):
    (tmp_path / "run_manifest.tsv").write_text(
        "experiment_id\tstep_id\trun_id\trun_name\tparameter_summary\tversion\tconfig\tstatus\n"
        "unit\ttrain-model\trun-000\tmanaged\tlr=1e-6\tmanaged-v1\t/config.yaml\tplanned\n"
    )

    row = experiment_tracking.wandb_run_observations(
        _read_table(tmp_path / "run_manifest.tsv"),
        [
            {
                "experiment_id": "unit",
                "step_id": "train-model",
                "run_id": "run-000",
                "run_name": "corrupt",
                "parameter_summary": "lr=9",
                "version": "display-name",
                "config": "other.yaml",
                "status": "running",
                "state": "running",
                "wandb_run_id": "wandb-1",
                "unexpected": "ignored",
            }
        ],
    )[0]

    assert set(row) == {"step_id", "run_id", "status", "state", "wandb_run_id"}
    assert row["step_id"] == "train-model"
    assert row["run_id"] == "run-000"
    assert row["status"] == "running"
    assert row["state"] == "running"
    assert row["wandb_run_id"] == "wandb-1"
    assert "unexpected" not in row


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
    monkeypatch.setitem(
        sys.modules, "wandb", types.SimpleNamespace(Api=lambda: types.SimpleNamespace(runs=lambda *_a, **_k: []))
    )

    experiments.sync_wandb_runs(tmp_path, entity="entity", project="project")

    assert {(row["step_id"], row["run_id"]) for row in _read_table(tmp_path / "run_manifest.tsv")} == {
        ("step-a", "run-000"),
        ("step-a", "run-001"),
        ("step-b", "run-000"),
    }
    manifest_before = (tmp_path / "run_manifest.tsv").read_text()

    class FakeRun:
        id = "wandb-shared"
        name = "shared"
        state = "running"
        url = "https://wandb.ai/entity/project/runs/wandb-shared"
        group = ""
        created_at = "2026-01-01"
        updated_at = "2026-01-02"
        summary = {}
        config = {"run_id": "run-000"}

        def history(self, **_kwargs):
            return []

    monkeypatch.setitem(
        sys.modules,
        "wandb",
        types.SimpleNamespace(Api=lambda: types.SimpleNamespace(runs=lambda *_a, **_k: [FakeRun()])),
    )

    with pytest.raises(ValueError, match="Ambiguous runtime version"):
        experiments.sync_wandb_runs(tmp_path, entity="entity", project="project")
    assert (tmp_path / "run_manifest.tsv").read_text() == manifest_before

    (tmp_path / "metrics_manifest.tsv").write_text("run_id\tversion\tmetric\nrun-000\tshared\tval_auroc\n")
    with pytest.raises(ValueError, match="step_id and run_id"):
        experiments.rank_experiment_candidates(tmp_path, metric="val_auroc", mode="max")


def test_experiment_monitor_matches_previous_rows_by_managed_identity(tmp_path: Path, monkeypatch):
    _initialize_workspace(tmp_path)
    (tmp_path / "run_manifest.tsv").write_text(
        "experiment_id\tstep_id\trun_id\trun_name\tversion\tstatus\tmarker\n"
        "unit\tprepare-data\trun-000\tprepare\tshared\trunning\tprepare-marker\n"
        "unit\ttrain-model\trun-000\ttrain\tshared\trunning\ttrain-marker\n"
    )
    current = [
        {
            "step_id": "prepare-data",
            "run_id": "run-000",
            "run_name": "prepare",
            "version": "shared",
            "status": "running",
        },
        {
            "step_id": "train-model",
            "run_id": "run-000",
            "run_name": "train",
            "version": "shared",
            "status": "running",
        },
    ]
    matched = []

    def fake_status(_root, row, previous, *, script_commits_terminal_status, health):
        assert script_commits_terminal_status is False
        assert health is True
        matched.append((row["step_id"], previous["step_id"], previous["marker"]))
        return {**row, "health_status": "running"}

    monkeypatch.setattr(experiment_tracking, "experiment_run_rows", lambda _root, remote=None: current)
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
        if "sys.stdout.write(file_obj.read())" in command:
            return subprocess.CompletedProcess([], 0, "123\n", "")
        if command.startswith("ps "):
            return subprocess.CompletedProcess([], 0, "123\n", "")
        if "checkpoint_dir = sys.argv[2]" in command:
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
            "health_status": status,
            "checkpoint_count": len(checkpoints),
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
        assert observation["checkpoint_count"] == 1
    assert any("/remote/runtime/run-000" in command for command in commands)


def test_experiment_wandb_sync_remote_writes_outputs_over_ssh(monkeypatch):
    class FakeRun:
        id = "run123"
        name = "run_a"
        state = "finished"
        url = "https://wandb.ai/entity/project/runs/run123"
        group = "unit_group"
        created_at = "2026-01-01"
        updated_at = "2026-01-02"
        summary = {"val_auroc": 0.71, "epoch": 3}
        config = {"run_id": "run_a"}

        def history(self, **_kwargs):
            return [{"epoch": 1, "val_auroc": 0.6}]

    class FakeApi:
        def runs(self, path, filters=None):
            return [FakeRun()]

    calls = []
    experiment_text = json.dumps(
        {
            "experiment": {
                "id": "unit",
                "title": "Unit experiment",
                "objective": "Exercise experiment workspace contracts.",
                "root": "/wujidata/run",
                "baseline": {"type": "none", "rationale": "Unit fixture."},
            }
        }
    )
    run_manifest = "step_id\trun_id\n"

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        shell = command[-1]
        if "seen_inodes" in shell:
            return subprocess.CompletedProcess(command, 0, "", "")
        if "experiment.yaml" in shell and "sys.stdout.write" in shell:
            return subprocess.CompletedProcess(command, 0, experiment_text, "")
        if "run_manifest.tsv" in shell and "sys.stdout.write" in shell:
            return subprocess.CompletedProcess(command, 0, run_manifest, "")
        if "run_manifest.tsv" in shell and "os.lstat" in shell:
            return subprocess.CompletedProcess(command, 0, "", "")
        if "os.replace(temporary, path)" in shell:
            return subprocess.CompletedProcess(command, 0, "", "")
        if "cat >" in shell or "mkdir -p" in shell:
            return subprocess.CompletedProcess(command, 0, "", "")
        return subprocess.CompletedProcess(command, experiment_io.REMOTE_MISSING_RETURN_CODE, "", "")

    monkeypatch.setitem(sys.modules, "wandb", types.SimpleNamespace(Api=lambda: FakeApi()))
    monkeypatch.setattr("agent_tools.experiment_io.subprocess.run", fake_run)

    experiments.sync_wandb_runs("/wujidata/run", entity="entity", project="project", remote="baichuan3")

    write_targets = [command[-1] for command, kwargs in calls if "cat >" in command[-1]]
    assert any("/wujidata/run/wandb/runs.tsv" in target for target in write_targets)
    assert any("/wujidata/run/wandb/history/run123.csv" in target for target in write_targets)
    assert any("/wujidata/run/metrics_manifest.tsv" in target for target in write_targets)
    atomic_targets = [command[-1] for command, kwargs in calls if "os.replace(temporary, path)" in command[-1]]
    assert any("/wujidata/run/run_manifest.tsv" in target for target in atomic_targets)
    assert any("/wujidata/run/reports/wandb.md" in target for target in write_targets)


def test_experiment_wandb_sync_writes_blocked_report(tmp_path: Path, monkeypatch):
    _initialize_workspace(tmp_path)
    (tmp_path / "run_manifest.tsv").write_text("step_id\trun_id\n")

    class FakeApi:
        def __init__(self):
            raise RuntimeError("not logged in")

    monkeypatch.setitem(sys.modules, "wandb", types.SimpleNamespace(Api=FakeApi))

    try:
        experiments.sync_wandb_runs(tmp_path, entity="entity", project="project")
    except RuntimeError as exc:
        error = str(exc)
    else:
        error = ""

    assert "W&B sync blocked" in error
    assert (tmp_path / "reports" / "wandb_blocked.md").exists()


@pytest.mark.parametrize(("manifest_epoch", "expected_best"), [(None, "one"), (2, "two"), (3, None)])
def test_experiment_remote_checkpoint_index_respects_manifest_epoch(monkeypatch, manifest_epoch, expected_best):
    root = Path("/remote/workspace")
    checkpoint_one = "/remote/runtime/run_a/checkpoints/epoch=01-step=10.ckpt"
    checkpoint_two = "/remote/runtime/run_a/checkpoints/epoch=02-step=20.ckpt"
    run_manifest = (
        "experiment_id\tstep_id\trun_id\trun_name\tversion\truntime_dir\tcheckpoint_dir\tstatus\n"
        "unit\ttrain-model\trun-000\tmanaged\trun_a\t/remote/runtime/run_a\t"
        "/remote/runtime/run_a/checkpoints\tcompleted\n"
    )
    writes = {}

    def fake_read_text(path, remote=None):
        name = Path(path).name
        if name == "experiment.yaml":
            return json.dumps(
                {
                    "experiment": {
                        "id": "unit",
                        "title": "Unit experiment",
                        "objective": "Exercise experiment workspace contracts.",
                        "root": str(root),
                        "baseline": {"type": "none", "rationale": "Unit fixture."},
                    }
                }
            )
        if name == "run_manifest.tsv":
            return run_manifest
        if name == "run_manifest.json":
            manifest = {"best_model_path": checkpoint_one}
            if manifest_epoch is not None:
                manifest["epoch"] = manifest_epoch
            return json.dumps(manifest)
        return ""

    monkeypatch.setattr(experiment_io, "validate_managed_output_paths", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(experiment_io, "read_text_at", fake_read_text)
    monkeypatch.setattr(
        experiment_io,
        "path_exists_at",
        lambda path, remote=None: str(path)
        in {
            "/remote/workspace/run_manifest.tsv",
            "/remote/runtime/run_a",
            "/remote/runtime/run_a/checkpoints",
            "/remote/runtime/run_a/run_manifest.json",
        },
    )
    monkeypatch.setattr(experiment_io, "read_rows_at", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        experiment_io,
        "write_rows_at",
        lambda path, rows, remote=None: writes.update({Path(path).name: rows}),
    )
    monkeypatch.setattr(
        experiment_tracking.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command,
            0,
            f"{checkpoint_one}\t123.0\n{checkpoint_two}\t124.0\n",
            "",
        ),
    )

    result = experiments.index_checkpoints(root, remote="baichuan3")

    assert result == root / "checkpoint_manifest.tsv"
    assert {row["checkpoint_path"]: row["is_best_by_val"] for row in writes["checkpoint_manifest.tsv"]} == {
        checkpoint_one: str(expected_best == "one").lower(),
        checkpoint_two: str(expected_best == "two").lower(),
    }


def test_remote_checkpoint_scan_skips_confirmed_missing_run_and_indexes_other_runs(monkeypatch):
    missing = {
        "experiment_id": "unit",
        "step_id": "train-model",
        "run_id": "run-000",
        "version": "missing",
        "runtime_dir": "/remote/runtime/missing",
        "checkpoint_dir": "/remote/runtime/missing/checkpoints",
    }
    ready = {
        "experiment_id": "unit",
        "step_id": "train-model",
        "run_id": "run-001",
        "version": "ready",
        "runtime_dir": "/remote/runtime/ready",
        "checkpoint_dir": "/remote/runtime/ready/checkpoints",
    }
    checkpoint = "/remote/runtime/ready/checkpoints/epoch=01.ckpt"
    commands = []

    def fake_exists(path, *, remote=None):
        return str(path) in {ready["runtime_dir"], ready["checkpoint_dir"]}

    def fake_run(command, **kwargs):
        commands.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, f"{checkpoint}\t123.0\n", "")

    monkeypatch.setattr(experiment_io, "path_exists_at", fake_exists)
    monkeypatch.setattr(experiment_io, "validate_managed_output_paths", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(experiment_io, "read_text_at", lambda *_args, **_kwargs: "")
    monkeypatch.setattr(experiment_tracking.subprocess, "run", fake_run)

    rows = experiment_tracking._remote_checkpoint_rows([missing, ready], "unit-host")

    assert [(row["run_id"], row["checkpoint_path"]) for row in rows] == [("run-001", checkpoint)]
    command, kwargs = commands[0]
    assert missing["checkpoint_dir"] not in command[-1]
    assert ready["checkpoint_dir"] in command[-1]
    assert kwargs["timeout"] == experiment_io.SSH_TIMEOUT_SECONDS


def test_remote_checkpoint_scan_preserves_inventory_when_directory_disappears(monkeypatch):
    root = Path("/remote/workspace")
    run = {
        "experiment_id": "unit",
        "step_id": "train-model",
        "run_id": "run-000",
        "version": "managed-v1",
        "runtime_dir": "/remote/runtime/run-000",
        "checkpoint_dir": "/remote/runtime/run-000/checkpoints",
    }
    previous = {
        **run,
        "checkpoint_path": "/remote/runtime/run-000/checkpoints/epoch=01.ckpt",
    }
    monkeypatch.setattr(
        experiment_io,
        "read_rows_at",
        lambda path, **_kwargs: [previous] if Path(path).name == "checkpoint_manifest.tsv" else [],
    )
    monkeypatch.setattr(experiment_tracking, "read_run_manifest", lambda *_args, **_kwargs: [run])
    monkeypatch.setattr(experiment_io, "path_exists_at", lambda *_args, **_kwargs: False)

    with pytest.raises(RuntimeError, match="missing frozen artifact directory with existing inventory"):
        experiment_tracking.checkpoint_rows(root, remote="unit-host")


def test_remote_checkpoint_scan_propagates_path_probe_errors(monkeypatch):
    run = {
        "step_id": "train-model",
        "run_id": "run-000",
        "runtime_dir": "/remote/runtime/run-000",
        "checkpoint_dir": "/remote/runtime/run-000/checkpoints",
    }
    monkeypatch.setattr(
        experiment_io,
        "path_exists_at",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("permission denied")),
    )

    with pytest.raises(RuntimeError, match="SSH checkpoint scan failed.*permission denied"):
        experiment_tracking._remote_checkpoint_rows([run], "unit-host")


@pytest.mark.parametrize(
    "failure",
    [
        "nonzero",
        "timeout",
        "malformed",
        "invalid_mtime",
        "unmanaged",
        "manifest_transport",
        "corrupt_manifest",
        "empty_manifest",
    ],
)
def test_experiment_remote_checkpoint_scan_fails_closed_without_writing(tmp_path: Path, monkeypatch, failure: str):
    _initialize_workspace(tmp_path)
    checkpoint_manifest = tmp_path / "checkpoint_manifest.tsv"
    checkpoint_manifest.write_text("step_id\trun_id\tcheckpoint_path\ntrain-model\trun-000\t/remote/old.ckpt\n")
    original_manifest = checkpoint_manifest.read_bytes()
    calls = []
    writes = {}
    run_manifest = (
        "experiment_id\tstep_id\trun_id\tversion\truntime_dir\tcheckpoint_dir\n"
        "unit\ttrain-model\trun-000\trun_a\t/remote/runtime/run_a\t/remote/runtime/run_a/checkpoints\n"
    )

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        if "seen_inodes" in command[-1]:
            return subprocess.CompletedProcess(command, 0, "", "")
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        if failure == "malformed":
            return subprocess.CompletedProcess(
                command,
                0,
                "/remote/runtime/run_a/checkpoints/epoch=1.ckpt\t123.0\nincomplete output\n",
                "",
            )
        if failure == "invalid_mtime":
            return subprocess.CompletedProcess(
                command,
                0,
                "/remote/runtime/run_a/checkpoints/epoch=1.ckpt\tnan\n",
                "",
            )
        if failure == "unmanaged":
            return subprocess.CompletedProcess(
                command,
                0,
                "/remote/runtime/run_a/checkpoints/epoch=1.ckpt\t123.0\n"
                "/remote/runtime/other/checkpoints/epoch=2.ckpt\t124.0\n",
                "",
            )
        if failure in {"manifest_transport", "corrupt_manifest", "empty_manifest"}:
            return subprocess.CompletedProcess(
                command,
                0,
                "/remote/runtime/run_a/checkpoints/epoch=1.ckpt\t123.0\n",
                "",
            )
        return subprocess.CompletedProcess(
            command,
            1,
            "/remote/runtime/run_a/checkpoints/epoch=1.ckpt\t123.0\n",
            "",
        )

    monkeypatch.setattr("agent_tools.experiment_tracking.subprocess.run", fake_run)

    def fake_read_text(path, remote=None, **_kwargs):
        if Path(path).name == "experiment.yaml":
            return (tmp_path / "experiment.yaml").read_text()
        if Path(path).name == "run_manifest.tsv":
            return run_manifest
        if Path(path).name == "run_manifest.json":
            if failure == "manifest_transport":
                raise RuntimeError("SSH read failed")
            if failure == "corrupt_manifest":
                return "{"
        return ""

    monkeypatch.setattr(experiment_io, "read_text_at", fake_read_text)
    monkeypatch.setattr(
        experiment_io,
        "path_exists_at",
        lambda path, remote=None: str(path)
        in {
            str(tmp_path / "run_manifest.tsv"),
            "/remote/runtime/run_a",
            "/remote/runtime/run_a/checkpoints",
        }
        or (failure == "empty_manifest" and Path(path).name == "run_manifest.json"),
    )
    monkeypatch.setattr(
        experiment_io,
        "read_rows_at",
        lambda path, remote=None, **_kwargs: (
            [
                {
                    "experiment_id": "unit",
                    "step_id": "train-model",
                    "run_id": "run-000",
                    "version": "run_a",
                    "runtime_dir": "/remote/runtime/run_a",
                    "checkpoint_dir": "/remote/runtime/run_a/checkpoints",
                }
            ]
            if Path(path).name == "run_manifest.tsv"
            else []
        ),
    )
    monkeypatch.setattr(
        experiment_io,
        "write_rows_at",
        lambda path, rows, remote=None: writes.update({Path(path).name: rows}),
    )

    with pytest.raises(RuntimeError, match="SSH checkpoint scan"):
        experiments.index_checkpoints(tmp_path, remote="baichuan3")

    command, kwargs = next((command, kwargs) for command, kwargs in calls if "find " in command[-1])
    assert command[:2] == ["ssh", "baichuan3"]
    assert "/remote/runtime/run_a/checkpoints" in command[-1]
    assert kwargs["timeout"] == experiment_io.SSH_TIMEOUT_SECONDS
    assert "checkpoint_manifest.tsv" not in writes
    assert checkpoint_manifest.read_bytes() == original_manifest


def test_experiment_rank_remote_reads_and_writes_over_ssh(monkeypatch):
    calls = []
    experiment_text = json.dumps(
        {
            "experiment": {
                "id": "unit",
                "title": "Unit experiment",
                "objective": "Exercise experiment workspace contracts.",
                "root": "/wujidata/run",
                "baseline": {"type": "none", "rationale": "Unit fixture."},
            }
        }
    )
    metrics = (
        "step_id\trun_id\tversion\tepoch\tmetric\tvalue\tmetric_scope\tsource\n"
        "train-model\trun-000\trun_a\t1\tval_auroc\t0.6\tvalidation\twandb_history\n"
        "train-model\trun-000\trun_a\t2\tval_auroc\t0.8\tvalidation\twandb_history\n"
    )
    checkpoints = (
        "step_id\trun_id\tversion\tepoch\tcheckpoint_path\tis_best_by_val\tis_last\n"
        "train-model\trun-000\trun_a\t2\t/remote/run/run_a/checkpoints/epoch=2.ckpt\ttrue\tfalse\n"
    )
    run_manifest = (
        "experiment_id\tstep_id\trun_id\trun_name\tversion\tcheckpoint_dir\tstatus\n"
        "unit\ttrain-model\trun-000\tmanaged\trun_a\t/remote/run/run_a/checkpoints\tcompleted\n"
    )

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        shell = command[-1]
        if "seen_inodes" in shell:
            return subprocess.CompletedProcess(command, 0, "", "")
        if "experiment.yaml" in shell and "sys.stdout.write" in shell:
            return subprocess.CompletedProcess(command, 0, experiment_text, "")
        if "metrics_manifest.tsv" in shell and "sys.stdout.write" in shell:
            return subprocess.CompletedProcess(command, 0, metrics, "")
        if "checkpoint_manifest.tsv" in shell and "sys.stdout.write" in shell:
            return subprocess.CompletedProcess(command, 0, checkpoints, "")
        if "run_manifest.tsv" in shell and "sys.stdout.write" in shell:
            return subprocess.CompletedProcess(command, 0, run_manifest, "")
        if "run_manifest.tsv" in shell and "os.lstat" in shell:
            return subprocess.CompletedProcess(command, 0, "", "")
        if "os.lstat" in shell:
            return subprocess.CompletedProcess(command, experiment_io.REMOTE_MISSING_RETURN_CODE, "", "")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("agent_tools.experiment_io.subprocess.run", fake_run)

    experiments.rank_experiment_candidates("/wujidata/run", metric="val_auroc", mode="max", remote="baichuan3")

    write_targets = [command[-1] for command, kwargs in calls if "cat >" in command[-1]]
    assert any("/wujidata/run/reports/experiment_ranking.csv" in target for target in write_targets)
    assert any("/wujidata/run/reports/experiment_ranking.md" in target for target in write_targets)
    ranking_write = next(
        kwargs["input"]
        for command, kwargs in calls
        if "cat >" in command[-1] and "reports/experiment_ranking.csv" in command[-1]
    )
    assert "0.8" in ranking_write
    assert "/remote/run/run_a/checkpoints/epoch=2.ckpt" in ranking_write
