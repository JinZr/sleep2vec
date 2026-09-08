from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from agent_tools import adaptive_hparam, experiment_sources, run_evidence
from agent_tools.experiment_workspace import merge_run_manifest
from tests.agent_tools import adaptive_hparam_test_support as test_support
from tests.agent_tools.adaptive_hparam_test_support import (
    _agent_recipe,
    _mark_round_terminal,
    _read_table,
    _run,
    _write_agent_submission,
    _write_fake_manifest,
)

_stub_execution_snapshot_preflight = test_support._stub_execution_snapshot_preflight


def _write_history(workspace: Path, content: str, *, run_id: str = "canonical-id") -> Path:
    path = workspace / "wandb" / "history" / f"{run_id}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


@pytest.fixture
def terminal_workflow(tmp_path: Path):
    recipe = _agent_recipe(tmp_path)
    workflow = tmp_path / "workflow"
    initialized = _run("hparam-adaptive-init", "--recipe", str(recipe), "--output-dir", str(workflow))
    assert initialized.returncode == 0, initialized.stderr
    _write_fake_manifest(workflow, score=0.73)
    _mark_round_terminal(workflow, tmp_path)
    round_dir = workflow / "adaptive" / "rounds" / "round_000"
    run = json.loads((round_dir / "plan.json").read_text())["runs"][0]
    merge_run_manifest(
        tmp_path,
        [{"step_id": run["step_id"], "run_id": run["run_id"], "wandb_run_id": "canonical-id"}],
    )
    return workflow, round_dir, run


def test_training_history_preserves_sparse_evidence_and_logged_learning_rate_ranges(tmp_path: Path, monkeypatch):
    path = _write_history(
        tmp_path,
        "epoch,_step,trainer/global_step,train_loss_epoch,val_loss,val_ahi_pearson,val_auroc,"
        "train_loss_step,test_auroc,external_loss,lr-AdamW/pg1,lr-AdamW/pg2\n"
        "0,2,10,0.9,,,,0.4,0.99,0.01,0.0001,0.00001\n"
        "0,3,10,,0.8,0.6,0.7,,,,,\n"
        ",4,11,,,,,,,,0.00001,0.00002\n"
        "1,5,20,0.7,0.6,0.65,0.75,0.2,0.98,0.02,0.00003,\n"
        "2,6,30,nan,inf,-inf,,0.1,0.97,0.03,nan,inf\n",
    )

    def unexpected_sync(*_args, **_kwargs):
        pytest.fail("Reading existing history must not call W&B")

    monkeypatch.setattr(experiment_sources, "wandb_runs", unexpected_sync)
    history = experiment_sources.read_wandb_training_history(
        tmp_path, {"wandb_run_id": "canonical-id"}, monitor="val_ahi_pearson", objective="val_auroc"
    )

    assert history == {
        "source_path": str(path),
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "wandb_run_id": "canonical-id",
        "observations": [
            {"epoch": 0, "_step": 2.0, "trainer/global_step": 10.0, "metrics": {"train_loss_epoch": 0.9}},
            {
                "epoch": 0,
                "_step": 3.0,
                "trainer/global_step": 10.0,
                "metrics": {"val_loss": 0.8, "val_ahi_pearson": 0.6, "val_auroc": 0.7},
            },
            {
                "epoch": 1,
                "_step": 5.0,
                "trainer/global_step": 20.0,
                "metrics": {"train_loss_epoch": 0.7, "val_loss": 0.6, "val_ahi_pearson": 0.65, "val_auroc": 0.75},
            },
        ],
        "learning_rate_ranges": {
            "lr-AdamW/pg1": {"min": 0.00001, "max": 0.0001},
            "lr-AdamW/pg2": {"min": 0.00001, "max": 0.00002},
        },
    }


def test_training_history_omits_test_objective_and_unlogged_coordinates(tmp_path: Path):
    _write_history(
        tmp_path,
        "train_loss_epoch,val_loss,test_auroc,external_auroc,val_unselected_metric\n0.8,0.7,0.99,0.98,0.97\n",
    )

    history = experiment_sources.read_wandb_training_history(
        tmp_path, {"wandb_run_id": "canonical-id"}, monitor="external_auroc", objective="test_auroc"
    )

    assert history is not None
    assert history["observations"] == [{"metrics": {"train_loss_epoch": 0.8, "val_loss": 0.7}}]
    assert history["learning_rate_ranges"] == {}


def test_training_history_preserves_logged_coordinates_without_inferring_completed_epochs(tmp_path: Path):
    _write_history(
        tmp_path,
        "epoch,trainer/epoch,current_epoch,_step,trainer/global_step,val_loss\n"
        "0.5,0.75,0.25,12,10,0.8\n"
        ",,,13,,0.7\n",
    )

    history = experiment_sources.read_wandb_training_history(
        tmp_path, {"wandb_run_id": "canonical-id"}, monitor="val_loss", objective="test_auroc"
    )

    assert history is not None
    assert history["observations"] == [
        {
            "epoch": 0.5,
            "trainer/epoch": 0.75,
            "current_epoch": 0.25,
            "_step": 12.0,
            "trainer/global_step": 10.0,
            "metrics": {"val_loss": 0.8},
        },
        {"_step": 13.0, "metrics": {"val_loss": 0.7}},
    ]


@pytest.mark.parametrize("run_id", [None, "missing-id"])
def test_training_history_requires_canonical_id_and_workspace_file(tmp_path: Path, run_id: str | None):
    _write_history(tmp_path, "epoch,val_loss\n0,0.1\n", run_id="display-name")
    runtime = tmp_path / "runtime"
    _write_history(runtime, "epoch,val_loss\n0,0.2\n", run_id="missing-id")

    history = experiment_sources.read_wandb_training_history(
        tmp_path,
        {"wandb_run_id": run_id, "version": "display-name", "runtime_dir": str(runtime), "target": "ssh"},
        monitor="val_loss",
        objective="test_auroc",
    )

    assert history is None


@pytest.mark.parametrize("available", [False, True])
def test_agent_proposal_includes_history_without_changing_canonical_scores(
    tmp_path: Path, terminal_workflow, available
):
    workflow, round_dir, _run_row = terminal_workflow
    if available:
        path = _write_history(
            tmp_path,
            "epoch,train_loss_epoch,val_loss,val_ahi_pearson,test_auroc\n3,0.1,0.2,0.99,0.98\n",
        )

    input_path = adaptive_hparam.adaptive_step(workflow)
    assert input_path is not None
    row = json.loads(input_path.read_text())["input"]["digest_rows"][0]
    digest = _read_table(workflow / "adaptive" / "digests" / "round_000.csv")[0]

    assert row["val_ahi_pearson"] == "0.5"
    assert row["test_auroc"] == "0.73"
    assert "train_loss_epoch" not in row
    assert row["status"] == "finished"
    assert row["stop_reason"] == ""
    if available:
        assert row["training_history"]["source_path"] == str(path)
        assert row["training_history"]["source_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert row["training_history"]["observations"] == [
            {"epoch": 3, "metrics": {"train_loss_epoch": 0.1, "val_loss": 0.2, "val_ahi_pearson": 0.99}}
        ]
        assert json.loads(digest["training_history"]) == row["training_history"]
    else:
        assert row["training_history"] == digest["training_history"] == ""
    proposal_path = _write_agent_submission(input_path)
    assert adaptive_hparam.adaptive_step(workflow, proposal_path=proposal_path) == proposal_path
    assert not (round_dir.parent / "round_001").exists()


@pytest.mark.parametrize("drift", ["metric", "source_bytes", "removed"])
@pytest.mark.parametrize("execute", [False, True])
def test_unaccepted_proposal_rejects_history_drift_without_mutation(
    tmp_path: Path, terminal_workflow, drift: str, execute: bool
):
    workflow, _round_dir, _run_row = terminal_workflow
    history_path = _write_history(tmp_path, "epoch,val_loss\n0,0.8\n")
    input_path = adaptive_hparam.adaptive_step(workflow)
    assert input_path is not None
    proposal_path = _write_agent_submission(input_path)
    assert adaptive_hparam.adaptive_step(workflow, proposal_path=proposal_path) == proposal_path
    if drift == "metric":
        history_path.write_text("epoch,val_loss\n0,0.7\n")
    elif drift == "source_bytes":
        history_path.write_bytes(history_path.read_bytes().replace(b"\n", b"\r\n"))
    else:
        history_path.unlink()
    managed_paths = [
        tmp_path / "events.jsonl",
        tmp_path / "run_manifest.tsv",
        workflow / "adaptive" / "workflow.json",
        workflow / "adaptive" / "run_registry.tsv",
        workflow / "adaptive" / "digests" / "round_000.csv",
        workflow / "adaptive" / "incumbents.tsv",
        input_path,
        proposal_path,
    ]
    before = {path: path.read_bytes() for path in managed_paths}

    with pytest.raises(ValueError, match=r"^Agent proposal input does not match the current authoritative snapshot\.$"):
        adaptive_hparam.adaptive_step(workflow, proposal_path=proposal_path, execute=execute)

    assert {path: path.read_bytes() for path in managed_paths} == before
    assert not (workflow / "adaptive" / "proposals" / "round_001.json").exists()
    assert not (workflow / "adaptive" / "suggestions" / "round_001.yaml").exists()
    assert not (workflow / "adaptive" / "suggestions" / "round_001.md").exists()
    assert not (workflow / "adaptive" / "rounds" / "round_001").exists()


def test_ssh_digest_binds_workspace_history_by_canonical_run_id(tmp_path: Path, terminal_workflow, monkeypatch):
    _workflow, round_dir, run = terminal_workflow
    canonical = _write_history(tmp_path, "epoch,val_loss\n2,0.4\n")
    _write_history(tmp_path, "epoch,val_loss\n2,0.99\n", run_id=run["version"])
    _write_history(Path(run["runtime_dir"]), "epoch,val_loss\n2,0.88\n")
    merge_run_manifest(
        tmp_path,
        [{"step_id": run["step_id"], "run_id": run["run_id"], "target": "ssh", "host": "unit-host"}],
    )
    manifest = {
        "monitor": "val_ahi_pearson",
        "best_model_score": 0.5,
        "best_model_path": "/remote/checkpoints/best-epoch=3.ckpt",
        "metrics": {"val_ahi_pearson": 0.5, "test_auroc": 0.73},
    }
    monkeypatch.setattr(adaptive_hparam, "monitor_hparam_runs", lambda _round_dir: None)
    monkeypatch.setattr(
        run_evidence,
        "runtime_artifacts",
        lambda _row: ("/remote/run_manifest.json", manifest, ["epoch=3.ckpt"]),
    )
    monkeypatch.setattr(run_evidence, "log_has_failure", lambda *_args: False)
    monkeypatch.setattr(run_evidence, "log_tail", lambda *_args, **_kwargs: "")

    row = _read_table(adaptive_hparam.digest_hparam_run(round_dir))[0]

    history = json.loads(row["training_history"])
    assert history["source_path"] == str(canonical)
    assert history["source_sha256"] == hashlib.sha256(canonical.read_bytes()).hexdigest()
    assert history["wandb_run_id"] == "canonical-id"
    assert history["observations"] == [{"epoch": 2, "metrics": {"val_loss": 0.4}}]
    assert row["test_auroc"] == "0.73"
    assert row["val_ahi_pearson"] == "0.5"


@pytest.mark.parametrize("status", ["stopped", "failed"])
@pytest.mark.parametrize("variant", ["sleep2vec", "sleep2vec2", "sleep2expert"])
@pytest.mark.parametrize("explicit_task", [False, True])
def test_incomplete_run_history_keeps_frozen_validation_monitor(
    tmp_path: Path, monkeypatch, status: str, variant: str, explicit_task: bool
):
    monitor = "val_auroc" if explicit_task else "val_ahi_pearson"
    task = (
        {"type": "classification", "output_dim": 2, "is_seq": False, "monitor": monitor, "monitor_mod": "max"}
        if explicit_task
        else None
    )
    config_path = tmp_path / "frozen-config.yaml"
    config_path.write_text(yaml.safe_dump({"finetune": {"task": task}}))
    run = {
        "experiment_id": "unit-experiment",
        "step_id": "unit-round-000",
        "run_id": "run-001",
        "run_name": "lr-0.001",
        "version": "unit-version",
        "config": str(config_path),
    }
    canonical = {
        **run,
        "status": status,
        "stop_reason": "budget limit" if status == "stopped" else "",
        "wandb_run_id": "canonical-id",
    }
    recipe = {
        "variant": variant,
        "inputs": {"label_name": "custom_binary" if explicit_task else "ahi"},
        "evaluation_policy": {"selection_split": "test"},
    }
    monkeypatch.setattr(adaptive_hparam.artifacts, "read_hparam_plan", lambda _path: {"recipe": recipe, "runs": [run]})
    monkeypatch.setattr(adaptive_hparam, "read_run_manifest", lambda _path: [canonical])
    monkeypatch.setattr(run_evidence, "runtime_artifacts", lambda _row: None)
    _write_history(tmp_path, f"epoch,val_loss,{monitor},test_auroc\n0,0.4,0.73,0.99\n")

    row = adaptive_hparam._digest_rows(tmp_path, 0, tmp_path, {"metric": "test_auroc", "mode": "max"})[0]

    assert json.loads(row["training_history"])["observations"] == [
        {"epoch": 0, "metrics": {"val_loss": 0.4, monitor: 0.73}}
    ]
    assert row["status"] == status
    assert row["stop_reason"] == canonical["stop_reason"]
    assert row["run_manifest"] == row["checkpoint_path"] == ""
    assert monitor not in row
    assert "test_auroc" not in row
    assert "epoch" not in row
