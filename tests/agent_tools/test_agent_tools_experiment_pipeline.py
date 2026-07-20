from __future__ import annotations

import copy
import csv
import fcntl
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from agent_tools import experiment_pipeline
from agent_tools.experiment_workspace import file_sha256
from agent_tools.manifests import write_rows


def _spec(root: Path) -> dict:
    return {
        "schema_version": 1,
        "pipeline": {
            "id": "external-v1",
            "kind": "external_matrix",
            "experiment_id": "unit",
            "step": {
                "id": "external-evaluate",
                "phase": "evaluate",
                "purpose": "Run the frozen external matrix.",
            },
            "finalize": True,
        },
        "runtime": {
            "workdir": "/runtime/snapshot",
            "python": "/runtime/python",
            "runtime_commit": "a" * 40,
            "accelerator": "gpu",
            "device": "cuda",
            "precision": "32-true",
            "batch_size": 128,
            "seed": 4523,
        },
        "execution": {
            "gpu_pool": list(range(8)),
            "gpus_per_run": 1,
            "max_concurrent": 8,
            "max_attempts": 2,
        },
        "evaluation_policy": {
            "external_test_locked": False,
            "final_test_unlocked": True,
        },
        "checkpoint_policy": {
            "avg_ckpts": 1,
            "require_no_model_averaging": True,
            "forbidden_state_dict_prefixes": ["ema_model.", "running_mean_model."],
            "require_ahi_eval_threshold": True,
        },
        "checkpoint_sources": {
            "age": {
                "plan": str(root / "plans" / "train-age"),
                "selection_metric": "val_mae",
                "selection_mode": "min",
                "task": "age",
                "variant": "sleep2vec2",
                "label_name": "age",
            }
        },
        "jobs": [
            {
                "id": "age-hsp-i2-psg",
                "checkpoint_source": "age",
                "cohort": "hsp_i2",
                "modality": "psg",
                "inference_preset_path": str(root / "presets" / "hsp_i2_age.pickle"),
                "num_workers": 8,
                "task": "age",
                "variant": "sleep2vec2",
                "label_name": "age",
            }
        ],
    }


def _write_experiment(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "experiment.yaml").write_text(
        yaml.safe_dump(
            {
                "experiment": {
                    "id": "unit",
                    "title": "Unit",
                    "objective": "Exercise the external runner.",
                    "root": str(root),
                    "baseline": {"type": "none"},
                    "status": "active",
                }
            },
            sort_keys=False,
        )
    )


def _selection(tmp_path: Path) -> dict:
    config = tmp_path / "config.yaml"
    checkpoint = tmp_path / "model.ckpt"
    config.write_text("model: unit\n")
    checkpoint.write_bytes(b"checkpoint")
    return {
        "source_id": "age",
        "selection_metric": "val_mae",
        "selection_mode": "min",
        "score": 4.5,
        "config": str(config),
        "config_sha256": file_sha256(config),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "variant": "sleep2vec2",
        "label_name": "age",
    }


def _result_manifest_context(tmp_path: Path) -> tuple[dict, dict, dict, Path, dict]:
    root = tmp_path / "workspace"
    spec = _spec(root)
    config = tmp_path / "config.yaml"
    checkpoint = tmp_path / "model.ckpt"
    preset = tmp_path / "preset.pickle"
    config.write_text("model: unit\n")
    checkpoint.write_bytes(b"checkpoint")
    preset.write_bytes(b"preset")
    result_root = tmp_path / "result"
    manifest_path = result_root / "nested" / "run_manifest.json"
    manifest_path.parent.mkdir(parents=True)
    attempt = {
        "result_root": str(result_root),
        "checkpoint": str(checkpoint),
        "preset": str(preset),
        "config_sha256": file_sha256(config),
        "label_name": "age",
        "variant": "sleep2vec2",
    }
    run = {"config": str(config)}
    manifest = {
        "namespace": "sleep2vec2",
        "config_path": str(config),
        "label_name": "age",
        "eval_split": "test",
        "checkpoint": {
            "input": str(checkpoint),
            "resolved_path": str(checkpoint),
            "avg_ckpts": 1,
        },
        "runtime": {
            "inference_preset_path": str(preset),
            "batch_size": 128,
            "accelerator": "gpu",
            "precision": "32-true",
            "devices": [0],
        },
        "paths": {
            "run_dir": str(manifest_path.parent),
            "manifest_path": str(manifest_path),
        },
        "prediction_row_count": 5,
        "metrics": {"accuracy": 0.75},
    }
    manifest_path.write_text(json.dumps(manifest) + "\n")
    return spec, attempt, run, manifest_path, manifest


def test_schema_rejects_duplicate_job_ids_illegal_phase_and_missing_unlock(tmp_path: Path):
    root = tmp_path / "workspace"
    spec = _spec(root)
    experiment_pipeline._validate_spec(spec, root, unlock_final_test=True)

    duplicate = copy.deepcopy(spec)
    duplicate["jobs"].append(copy.deepcopy(duplicate["jobs"][0]))
    with pytest.raises(ValueError, match="Duplicate external job id"):
        experiment_pipeline._validate_spec(duplicate, root, unlock_final_test=True)

    illegal_phase = copy.deepcopy(spec)
    illegal_phase["pipeline"]["step"]["phase"] = "external_test"
    with pytest.raises(ValueError, match="phase must be 'evaluate'"):
        experiment_pipeline._validate_spec(illegal_phase, root, unlock_final_test=True)

    with pytest.raises(ValueError, match="requires --unlock-final-test"):
        experiment_pipeline._validate_spec(spec, root, unlock_final_test=False)


def test_dry_run_does_not_freeze_or_mutate_workspace(tmp_path: Path, monkeypatch):
    root = tmp_path / "workspace"
    root.mkdir()
    spec_path = tmp_path / "external.yaml"
    spec_path.write_text(yaml.safe_dump(_spec(root), sort_keys=False))
    before = {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}
    monkeypatch.setattr(
        experiment_pipeline,
        "_inspect_sources",
        lambda *_args, **_kwargs: [
            {
                "source_id": "age",
                "plan": str(root / "plans" / "train-age"),
                "statuses": ["completed"],
                "complete": True,
                "failed_runs": [],
                "uncertain_runs": [],
            }
        ],
    )

    result = experiment_pipeline.run_experiment_pipeline(root, spec_path)

    after = {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}
    assert result["status"] == "ready"
    assert result["dry_run"] is True
    assert before == after
    assert not (root / "pipelines").exists()


def test_pipeline_directory_alias_is_rejected_before_state_read(tmp_path: Path):
    root = tmp_path / "workspace"
    pipelines = root / "pipelines"
    pipelines.mkdir(parents=True)
    outside = tmp_path / "outside-pipeline"
    outside.mkdir()
    (pipelines / "external-v1").symlink_to(outside, target_is_directory=True)
    spec_path = tmp_path / "external.yaml"
    spec_path.write_text(yaml.safe_dump(_spec(root), sort_keys=False))

    with pytest.raises(ValueError, match="independent regular files"):
        experiment_pipeline.run_experiment_pipeline(root, spec_path)


def test_second_pipeline_runner_is_rejected_by_exclusive_lock(tmp_path: Path, monkeypatch):
    root = tmp_path / "workspace"
    pipeline_dir = root / "pipelines" / "external-v1"
    pipeline_dir.mkdir(parents=True)
    (pipeline_dir / "pipeline.json").write_text("{}\n")
    spec_path = tmp_path / "external.yaml"
    spec_path.write_text(yaml.safe_dump(_spec(root), sort_keys=False))
    monkeypatch.setattr(
        experiment_pipeline,
        "_validate_experiment",
        lambda *_args, **_kwargs: {"status": "active"},
    )
    monkeypatch.setattr(
        experiment_pipeline,
        "_validate_frozen_pipeline",
        lambda *_args, **_kwargs: pytest.fail("the second runner must not inspect mutable state"),
    )
    lock_path = root / "pipelines" / ".external-v1.runner.lock"
    with lock_path.open("a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="already active"):
            experiment_pipeline.run_experiment_pipeline(
                root,
                spec_path,
                unlock_final_test=True,
                execute=True,
                resume=True,
            )


def test_checkpoint_validation_rejects_averaging_and_requires_ahi_threshold(tmp_path: Path):
    torch = pytest.importorskip("torch")
    policy = _spec(tmp_path)["checkpoint_policy"]
    checkpoint = tmp_path / "model.ckpt"
    torch.save({"state_dict": {"model.weight": torch.tensor([1.0])}}, checkpoint)

    evidence = experiment_pipeline._validate_checkpoint_payload(checkpoint, "age", policy)
    assert evidence == {"state_dict_key_count": 1, "has_ahi_eval_threshold": False}
    with pytest.raises(ValueError, match="lacks ahi_eval_threshold"):
        experiment_pipeline._validate_checkpoint_payload(checkpoint, "ahi", policy)

    torch.save(
        {
            "state_dict": {"model.weight": torch.tensor([1.0])},
            "ahi_eval_threshold": 15.0,
        },
        checkpoint,
    )
    assert experiment_pipeline._validate_checkpoint_payload(checkpoint, "ahi", policy)["has_ahi_eval_threshold"] is True

    torch.save({"state_dict": {"ema_model.weight": torch.tensor([1.0])}}, checkpoint)
    with pytest.raises(ValueError, match="forbidden averaging state"):
        experiment_pipeline._validate_checkpoint_payload(checkpoint, "age", policy)


def test_failed_attempt_creates_exactly_one_fresh_second_attempt(tmp_path: Path, monkeypatch):
    root = tmp_path / "workspace"
    _write_experiment(root)
    pipeline_dir = root / "pipelines" / "external-v1"
    pipeline_dir.mkdir(parents=True)
    spec = _spec(root)
    selection = _selection(tmp_path)

    monkeypatch.setattr(
        experiment_pipeline,
        "preflight_plan",
        lambda **_kwargs: (None, None, SimpleNamespace(exit_code=0)),
    )

    monkeypatch.setattr(
        experiment_pipeline,
        "_materialize_attempt",
        lambda _root, _spec, job, _selection, attempt, **paths: {
            "job_id": job["id"],
            "attempt": attempt,
            "status": "planned",
            "verified": "false",
            "result_root": str(paths["result_root"]),
        },
    )
    monkeypatch.setattr(experiment_pipeline, "append_event", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(experiment_pipeline, "read_run_manifest", lambda _root: [])
    attempts = [{"job_id": "age-hsp-i2-psg", "attempt": 1, "status": "failed", "verified": "false"}]

    updated, created = experiment_pipeline._create_needed_retries(
        root,
        pipeline_dir,
        spec,
        {"age": selection},
        attempts,
    )

    assert created is True
    assert [int(row["attempt"]) for row in updated] == [1, 2]
    assert Path(updated[1]["result_root"]).name == "attempt-002"
    assert not Path(updated[1]["result_root"]).exists()

    updated[1]["status"] = "failed"
    unchanged, created_again = experiment_pipeline._create_needed_retries(
        root,
        pipeline_dir,
        spec,
        {"age": selection},
        updated,
    )
    assert created_again is False
    assert unchanged == updated
    assert experiment_pipeline._logical_job_states(spec, updated)[0]["status"] == "failed"


@pytest.mark.parametrize("status", ["missing_pid", "unknown_remote", "stopped", "superseded"])
def test_uncertain_or_human_terminal_attempt_is_blocked_and_not_retried(tmp_path: Path, monkeypatch, status: str):
    root = tmp_path / "workspace"
    spec = _spec(root)
    attempts = [{"job_id": "age-hsp-i2-psg", "attempt": 1, "status": status, "verified": "false"}]
    monkeypatch.setattr(
        experiment_pipeline,
        "_attempt_recipe",
        lambda *_args, **_kwargs: pytest.fail("uncertain attempts must not be retried"),
    )
    monkeypatch.setattr(experiment_pipeline, "read_run_manifest", lambda _root: [])

    unchanged, created = experiment_pipeline._create_needed_retries(
        root,
        root / "pipelines" / "external-v1",
        spec,
        {"age": {}},
        attempts,
    )

    assert created is False
    assert unchanged == attempts
    assert experiment_pipeline._logical_job_states(spec, attempts)[0]["status"] == "blocked"


@pytest.mark.parametrize("modality,workers", [("psg", 8), ("bcg", 16)])
def test_attempt_recipe_freezes_fp32_batch_workers_and_logical_gpu_zero(tmp_path: Path, modality: str, workers: int):
    root = tmp_path / "workspace"
    _write_experiment(root)
    pipeline_dir = root / "pipelines" / "external-v1"
    job = copy.deepcopy(_spec(root)["jobs"][0])
    job["modality"] = modality
    job["num_workers"] = workers

    recipe, _recipe_path, _plan_dir, result_root = experiment_pipeline._attempt_recipe(
        pipeline_dir,
        _spec(root),
        job,
        _selection(tmp_path),
        1,
    )

    assert recipe["runtime"] == {
        "devices": [0],
        "accelerator": "gpu",
        "device": "cuda",
        "precision": "32-true",
        "batch_size": 128,
        "num_workers": workers,
        "seed": 4523,
        "avg_ckpts": 1,
        "results_root": str(result_root),
    }
    assert recipe["execution"]["runtime_commit"] == "a" * 40
    assert recipe["artifacts"]["overwrite"] is False


def test_result_manifest_validation_accepts_exact_manifest_and_rejects_mismatch(tmp_path: Path):
    spec, attempt, run, manifest_path, manifest = _result_manifest_context(tmp_path)
    assert experiment_pipeline._validate_result_manifest(spec, attempt, run) == manifest_path

    manifest["runtime"]["devices"] = [1]
    manifest_path.write_text(json.dumps(manifest) + "\n")
    with pytest.raises(ValueError, match="logical device 0"):
        experiment_pipeline._validate_result_manifest(spec, attempt, run)


def test_result_manifest_validation_rejects_missing_and_corrupt_manifest(tmp_path: Path):
    spec, attempt, run, manifest_path, _manifest = _result_manifest_context(tmp_path)
    manifest_path.unlink()
    with pytest.raises(ValueError, match="exactly one run_manifest"):
        experiment_pipeline._validate_result_manifest(spec, attempt, run)

    manifest_path.write_text("{not-json\n")
    with pytest.raises(json.JSONDecodeError):
        experiment_pipeline._validate_result_manifest(spec, attempt, run)


def test_nan_metric_is_preserved_in_pipeline_aggregation(tmp_path: Path):
    spec, attempt, _run, manifest_path, manifest = _result_manifest_context(tmp_path)
    manifest["metrics"] = {"accuracy": 0.75, "undefined": float("nan"), "metadata": "ignored"}
    manifest_path.write_text(json.dumps(manifest, allow_nan=True) + "\n")
    root = tmp_path / "workspace"
    pipeline_dir = root / "pipelines" / "external-v1"
    attempt.update(
        {
            "step_id": "external-evaluate",
            "run_id": "run-001",
            "job_id": "age-hsp-i2-psg",
            "attempt": 1,
            "verified": "true",
            "result_manifest": str(manifest_path),
            "runtime_commit": "a" * 40,
        }
    )
    write_rows(pipeline_dir / "jobs.tsv", [attempt])
    selection = _selection(tmp_path)

    report = experiment_pipeline._aggregate_results(
        root,
        pipeline_dir,
        spec,
        {"age": selection},
        [{"job_id": "age-hsp-i2-psg", "status": "completed"}],
    )

    with (pipeline_dir / "metrics.csv").open(newline="") as file_obj:
        metric_rows = list(csv.DictReader(file_obj))
    assert {row["metric"]: row["value"] for row in metric_rows} == {
        "accuracy": "0.75",
        "undefined": "NaN",
    }
    assert "| age-hsp-i2-psg | undefined | NaN |" in report.read_text()


def test_incomplete_matrix_does_not_aggregate_or_finalize(tmp_path: Path, monkeypatch):
    root = tmp_path / "workspace"
    pipeline_dir = root / "pipelines" / "external-v1"
    pipeline_dir.mkdir(parents=True)
    (pipeline_dir / "spec.source.yaml").write_text("schema_version: 1\n")
    spec = _spec(root)
    finalize_calls = []
    monkeypatch.setattr(experiment_pipeline, "_validate_frozen_pipeline", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        experiment_pipeline,
        "_inspect_sources",
        lambda *_args, **_kwargs: [{"complete": True, "failed_runs": [], "uncertain_runs": []}],
    )
    monkeypatch.setattr(experiment_pipeline, "_update_state", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(experiment_pipeline, "_load_or_freeze_selections", lambda *_args: {"age": {}})
    monkeypatch.setattr(experiment_pipeline, "_load_or_create_initial_attempts", lambda *_args: [])
    monkeypatch.setattr(
        experiment_pipeline,
        "_run_attempts",
        lambda *_args, **_kwargs: {
            "status": "failed",
            "jobs": [{"job_id": "age-hsp-i2-psg", "status": "failed"}],
        },
    )
    monkeypatch.setattr(
        experiment_pipeline,
        "_aggregate_results",
        lambda *_args, **_kwargs: pytest.fail("an incomplete matrix must not aggregate"),
    )

    result = experiment_pipeline._execute_pipeline(
        root,
        pipeline_dir,
        spec,
        poll_seconds=0,
        finalize_callback=lambda *_args: finalize_calls.append(True),
    )

    assert result["status"] == "failed"
    assert finalize_calls == []
    assert not (pipeline_dir / "final.md").exists()
