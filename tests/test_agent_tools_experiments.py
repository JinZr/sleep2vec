from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys
import types

from agent_tools import experiments


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "agent_tools", *args], text=True, capture_output=True)


def _read_table(path: Path) -> list[dict[str, str]]:
    delimiter = "\t" if path.suffix == ".tsv" else ","
    with path.open(newline="") as file_obj:
        return list(csv.DictReader(file_obj, delimiter=delimiter))


def test_experiment_init_creates_manifest(tmp_path: Path):
    result = _run("experiment-init", "--run-dir", str(tmp_path), "--name", "unit")

    assert result.returncode == 0, result.stderr
    rows = _read_table(tmp_path / "experiment_manifest.tsv")
    assert rows[0]["experiment_id"] == "unit"
    assert rows[0]["remote_host"] == ""
    assert (tmp_path / "reports").exists()


def test_experiment_init_remote_writes_remote_not_local(tmp_path: Path, monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("agent_tools.experiments.subprocess.run", fake_run)

    experiments.init_experiment("/wujidata/remote_run", "unit", remote="baichuan3")

    assert len(calls) == 3
    assert calls[0][0][:2] == ["ssh", "baichuan3"]
    assert "mkdir -p" in calls[0][0][2]
    assert calls[1][0] == ["ssh", "baichuan3", "cat /wujidata/remote_run/experiment_manifest.tsv"]
    assert calls[2][0][:2] == ["ssh", "baichuan3"]
    assert "cat > /wujidata/remote_run/experiment_manifest.tsv" in calls[2][0][2]
    assert calls[2][1]["input"]
    assert not (tmp_path / "reports").exists()


def test_experiment_indexes_checkpoints_and_ranks_validation_metric(tmp_path: Path):
    run_dir = tmp_path / "log-finetune" / "trial_a"
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    ckpt = ckpt_dir / "epoch=2-step=20.ckpt"
    ckpt.write_text("checkpoint")
    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "best_model_path": str(ckpt_dir / "best-epoch=2-step=20.ckpt"),
                "epoch": 2,
                "metrics": {"val_auroc": 0.8},
            }
        )
    )
    (tmp_path / "metrics_manifest.tsv").write_text(
        "trial_id\tversion\tepoch\tmetric\tvalue\tmetric_scope\tsource\n"
        "trial_a\ttrial_a\t1\tval_auroc\t0.6\tvalidation\twandb_history\n"
        "trial_a\ttrial_a\t2\tval_auroc\t0.8\tvalidation\twandb_history\n"
    )

    checkpoints = _run("experiment-index-checkpoints", "--run-dir", str(tmp_path))
    ranking = _run("experiment-rank", "--run-dir", str(tmp_path), "--metric", "val_auroc", "--mode", "max")

    assert checkpoints.returncode == 0, checkpoints.stderr
    assert ranking.returncode == 0, ranking.stderr
    checkpoint_rows = _read_table(tmp_path / "checkpoint_manifest.tsv")
    assert checkpoint_rows[0]["epoch"] == "2"
    ranked = _read_table(tmp_path / "candidate_ranking.tsv")
    assert ranked[0]["version"] == "trial_a"
    assert ranked[0]["score"] == "0.8"
    assert ranked[0]["checkpoint_path"].endswith("epoch=2-step=20.ckpt")


def test_experiment_wandb_sync_exports_summary_history_and_metrics(tmp_path: Path, monkeypatch):
    class FakeRun:
        id = "run123"
        name = "trial_a"
        state = "finished"
        url = "https://wandb.ai/entity/project/runs/run123"
        group = "unit_group"
        created_at = "2026-01-01"
        updated_at = "2026-01-02"
        summary = {"val_auroc": 0.71, "test_auroc": 0.66, "epoch": 3}
        config = {"trial_id": "trial_a"}

        def history(self, **_kwargs):
            return [{"epoch": 1, "val_auroc": 0.6}, {"epoch": 2, "val_auroc": 0.72}]

    class FakeApi:
        def runs(self, path, filters=None):
            assert path == "entity/project"
            assert filters == {"group": "unit_group"}
            return [FakeRun()]

    monkeypatch.setitem(sys.modules, "wandb", types.SimpleNamespace(Api=lambda: FakeApi()))

    out = experiments.sync_wandb_runs(tmp_path, entity="entity", project="project", group="unit_group")

    assert out == tmp_path / "wandb" / "runs.tsv"
    run_rows = _read_table(out)
    assert run_rows[0]["version"] == "trial_a"
    assert (tmp_path / "wandb" / "history" / "run123.csv").exists()
    metric_rows = _read_table(tmp_path / "metrics_manifest.tsv")
    scopes = {row["metric"]: row["metric_scope"] for row in metric_rows}
    assert scopes["val_auroc"] == "validation"
    assert scopes["test_auroc"] == "test_or_external"


def test_experiment_wandb_sync_remote_writes_outputs_over_ssh(monkeypatch):
    class FakeRun:
        id = "run123"
        name = "trial_a"
        state = "finished"
        url = "https://wandb.ai/entity/project/runs/run123"
        group = "unit_group"
        created_at = "2026-01-01"
        updated_at = "2026-01-02"
        summary = {"val_auroc": 0.71, "epoch": 3}
        config = {"trial_id": "trial_a"}

        def history(self, **_kwargs):
            return [{"epoch": 1, "val_auroc": 0.6}]

    class FakeApi:
        def runs(self, path, filters=None):
            return [FakeRun()]

    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        returncode = 0 if "cat >" in command[-1] or "mkdir -p" in command[-1] else 1
        return subprocess.CompletedProcess(command, returncode, "", "")

    monkeypatch.setitem(sys.modules, "wandb", types.SimpleNamespace(Api=lambda: FakeApi()))
    monkeypatch.setattr("agent_tools.experiments.subprocess.run", fake_run)

    experiments.sync_wandb_runs("/wujidata/run", entity="entity", project="project", remote="baichuan3")

    write_targets = [command[-1] for command, kwargs in calls if "cat >" in command[-1]]
    assert any("/wujidata/run/wandb/runs.tsv" in target for target in write_targets)
    assert any("/wujidata/run/wandb/history/run123.csv" in target for target in write_targets)
    assert any("/wujidata/run/metrics_manifest.tsv" in target for target in write_targets)
    assert any("/wujidata/run/run_manifest.tsv" in target for target in write_targets)
    assert any("/wujidata/run/reports/wandb_rank.md" in target for target in write_targets)


def test_experiment_wandb_sync_writes_blocked_report(tmp_path: Path, monkeypatch):
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


def test_experiment_remote_checkpoint_scan_uses_short_ssh_timeout(tmp_path: Path, monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(
            command,
            0,
            "/remote/run/trial_a/checkpoints/epoch=1.ckpt\t123.0\n",
            "",
        )

    monkeypatch.setattr("agent_tools.experiments.subprocess.run", fake_run)

    experiments.index_checkpoints(tmp_path, remote="baichuan3")

    command, kwargs = calls[0]
    assert command[:2] == ["ssh", "baichuan3"]
    assert kwargs["timeout"] == experiments.SSH_TIMEOUT_SECONDS


def test_experiment_rank_remote_reads_and_writes_over_ssh(monkeypatch):
    calls = []
    metrics = (
        "trial_id\tversion\tepoch\tmetric\tvalue\tmetric_scope\tsource\n"
        "trial_a\ttrial_a\t1\tval_auroc\t0.6\tvalidation\twandb_history\n"
        "trial_a\ttrial_a\t2\tval_auroc\t0.8\tvalidation\twandb_history\n"
    )
    checkpoints = (
        "trial_id\tversion\tepoch\tcheckpoint_path\tis_best_by_val\tis_last\n"
        "trial_a\ttrial_a\t2\t/remote/run/trial_a/checkpoints/epoch=2.ckpt\ttrue\tfalse\n"
    )

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        shell = command[-1]
        if shell == "cat /wujidata/run/metrics_manifest.tsv":
            return subprocess.CompletedProcess(command, 0, metrics, "")
        if shell == "cat /wujidata/run/checkpoint_manifest.tsv":
            return subprocess.CompletedProcess(command, 0, checkpoints, "")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("agent_tools.experiments.subprocess.run", fake_run)

    experiments.rank_experiment_candidates("/wujidata/run", metric="val_auroc", mode="max", remote="baichuan3")

    write_targets = [command[-1] for command, kwargs in calls if "cat >" in command[-1]]
    assert any("/wujidata/run/candidate_ranking.tsv" in target for target in write_targets)
    assert any("/wujidata/run/reports/wandb_rank.md" in target for target in write_targets)
    ranking_write = next(kwargs["input"] for command, kwargs in calls if "candidate_ranking.tsv" in command[-1])
    assert "0.8" in ranking_write
    assert "/remote/run/trial_a/checkpoints/epoch=2.ckpt" in ranking_write
