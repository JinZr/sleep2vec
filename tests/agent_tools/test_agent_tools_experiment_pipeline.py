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
    metrics_path = manifest_path.parent / "metrics.csv"
    prediction_path = manifest_path.parent / "predictions.csv"
    metrics_path.write_text("metric,value\naccuracy,0.75\n")
    prediction_path.write_text("prediction\n0.75\n")
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
            "metrics_csv_path": str(metrics_path),
            "prediction_csv_path": str(prediction_path),
            "survival_per_disease_metrics_csv_path": str(manifest_path.parent / "survival_per_disease.csv"),
            "multilabel_per_disease_metrics_csv_path": str(manifest_path.parent / "multilabel_per_disease.csv"),
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


@pytest.mark.parametrize("field", ["workdir", "python", "runtime_commit"])
def test_schema_rejects_non_string_runtime_identity(tmp_path: Path, field: str):
    root = tmp_path / "workspace"
    spec = _spec(root)
    spec["runtime"][field] = []

    with pytest.raises(ValueError, match=rf"runtime\.{field}"):
        experiment_pipeline._validate_spec(spec, root, unlock_final_test=True)


@pytest.mark.parametrize("python_command", ["conda run -n exp python", "~/miniconda/bin/python"])
def test_schema_rejects_non_executable_runtime_python(tmp_path: Path, python_command: str):
    root = tmp_path / "workspace"
    spec = _spec(root)
    spec["runtime"]["python"] = python_command

    with pytest.raises(ValueError, match=r"runtime\.python must be a single executable"):
        experiment_pipeline._validate_spec(spec, root, unlock_final_test=True)


@pytest.mark.parametrize(
    "section,field,value,message",
    [
        (None, "schema_version", True, "schema_version"),
        (None, "schema_version", 1.0, "schema_version"),
        (None, "schema_version", 2, "schema_version"),
        ("runtime", "batch_size", True, r"runtime\.batch_size"),
        ("runtime", "batch_size", 128.0, r"runtime\.batch_size"),
        ("runtime", "batch_size", 64, r"runtime\.batch_size"),
        ("execution", "gpus_per_run", True, r"execution\.gpus_per_run"),
        ("execution", "gpus_per_run", 1.0, r"execution\.gpus_per_run"),
        ("execution", "gpus_per_run", 2, r"execution\.gpus_per_run"),
        ("execution", "max_attempts", True, r"execution\.max_attempts"),
        ("execution", "max_attempts", 2.0, r"execution\.max_attempts"),
        ("execution", "max_attempts", 3, r"execution\.max_attempts"),
        ("checkpoint_policy", "avg_ckpts", True, r"checkpoint_policy\.avg_ckpts"),
        ("checkpoint_policy", "avg_ckpts", 1.0, r"checkpoint_policy\.avg_ckpts"),
        ("checkpoint_policy", "avg_ckpts", 2, r"checkpoint_policy\.avg_ckpts"),
    ],
)
def test_schema_rejects_non_integer_or_wrong_fixed_values(
    tmp_path: Path,
    section: str | None,
    field: str,
    value: object,
    message: str,
):
    root = tmp_path / "workspace"
    spec = _spec(root)
    target = spec if section is None else spec[section]
    target[field] = value

    with pytest.raises(ValueError, match=message):
        experiment_pipeline._validate_spec(spec, root, unlock_final_test=True)


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


@pytest.mark.parametrize("other_status", ["completed", "running"])
def test_checkpoint_selection_stays_with_frozen_source_plan(tmp_path: Path, monkeypatch, other_status: str):
    root = tmp_path / "workspace"
    spec = _spec(root)
    source_plan_dir = Path(spec["checkpoint_sources"]["age"]["plan"])
    source_config = tmp_path / "source-config.yaml"
    source_config.write_text("model: source\n")
    source_checkpoint_dir = tmp_path / "source-checkpoints"
    source_checkpoint_dir.mkdir()
    source_checkpoint = source_checkpoint_dir / "source.ckpt"
    source_checkpoint.write_bytes(b"source")
    other_config = tmp_path / "other-config.yaml"
    other_config.write_text("model: other\n")
    other_checkpoint_dir = tmp_path / "other-checkpoints"
    other_checkpoint_dir.mkdir()
    other_checkpoint = other_checkpoint_dir / "other.ckpt"
    other_checkpoint.write_bytes(b"other")
    step_id = "train-age"
    source_run = {"step_id": step_id, "run_id": "run-001"}
    source_plan = {
        "recipe": {
            "task": "hparam_tune",
            "variant": "sleep2vec2",
            "experiment": {"root": str(root)},
            "step": {"id": step_id},
            "inputs": {"label_name": "age"},
        },
        "runs": [source_run],
    }
    ranking = [
        {
            "step_id": step_id,
            "run_id": "run-000",
            "run_name": "other-plan-best",
            "rank": "1",
            "score": "3.0",
            "config": str(other_config),
            "checkpoint_path": str(other_checkpoint),
        },
        {
            **source_run,
            "run_name": "source-plan-best",
            "rank": "2",
            "score": "4.5",
            "config": str(source_config),
            "checkpoint_path": str(source_checkpoint),
        },
    ]
    canonical = [
        {
            "step_id": step_id,
            "run_id": "run-000",
            "status": other_status,
            "checkpoint_dir": str(other_checkpoint_dir),
        },
        {
            **source_run,
            "status": "completed",
            "checkpoint_dir": str(source_checkpoint_dir),
        },
    ]
    monkeypatch.setattr(experiment_pipeline.artifacts, "read_hparam_plan", lambda plan_dir: source_plan)
    monkeypatch.setattr(experiment_pipeline, "select_hparam_candidates", lambda *_args: None)
    monkeypatch.setattr(experiment_pipeline, "read_run_manifest", lambda _root: canonical)
    monkeypatch.setattr(experiment_pipeline, "read_rows", lambda *_args, **_kwargs: ranking)
    monkeypatch.setattr(
        experiment_pipeline,
        "_validate_checkpoint_payload",
        lambda *_args: {"state_dict_key_count": 1, "has_ahi_eval_threshold": False},
    )

    selected = experiment_pipeline._select_checkpoint_sources(root, spec)

    assert selected[0]["plan"] == str(source_plan_dir)
    assert selected[0]["run_id"] == source_run["run_id"]
    assert selected[0]["checkpoint"] == str(source_checkpoint)
    assert selected[0]["score"] == 4.5


@pytest.mark.parametrize(
    ("initial_status", "current_status", "should_select"),
    [("running", "completed", True), ("completed", "failed", False)],
)
def test_checkpoint_selection_uses_canonical_status_after_ranking(
    tmp_path: Path,
    monkeypatch,
    initial_status: str,
    current_status: str,
    should_select: bool,
):
    root = tmp_path / "workspace"
    spec = _spec(root)
    config = tmp_path / "source-config.yaml"
    config.write_text("model: source\n")
    checkpoint_dir = tmp_path / "source-checkpoints"
    checkpoint_dir.mkdir()
    checkpoint = checkpoint_dir / "source.ckpt"
    checkpoint.write_bytes(b"source")
    source_run = {"step_id": "train-age", "run_id": "run-001"}
    source_plan = {
        "recipe": {
            "task": "hparam_tune",
            "variant": "sleep2vec2",
            "step": {"id": "train-age"},
            "inputs": {"label_name": "age"},
        },
        "runs": [source_run],
    }
    ranking = [
        {
            **source_run,
            "run_name": "source-plan-best",
            "rank": "1",
            "score": "4.5",
            "config": str(config),
            "checkpoint_path": str(checkpoint),
            "status": initial_status,
        }
    ]
    canonical = {
        **source_run,
        "status": initial_status,
        "checkpoint_dir": str(checkpoint_dir),
    }

    def select_candidates(*_args):
        canonical["status"] = current_status

    monkeypatch.setattr(experiment_pipeline.artifacts, "read_hparam_plan", lambda _plan_dir: source_plan)
    monkeypatch.setattr(experiment_pipeline, "select_hparam_candidates", select_candidates)
    monkeypatch.setattr(experiment_pipeline, "read_run_manifest", lambda _root: [dict(canonical)])
    monkeypatch.setattr(experiment_pipeline, "read_rows", lambda *_args, **_kwargs: ranking)
    monkeypatch.setattr(
        experiment_pipeline,
        "_validate_checkpoint_payload",
        lambda *_args: {"state_dict_key_count": 1, "has_ahi_eval_threshold": False},
    )

    if not should_select:
        with pytest.raises(ValueError, match="Selected checkpoint source is not successful"):
            experiment_pipeline._select_checkpoint_sources(root, spec)
        return

    selected = experiment_pipeline._select_checkpoint_sources(root, spec)

    assert selected[0]["run_id"] == source_run["run_id"]
    assert selected[0]["checkpoint"] == str(checkpoint)


def test_checkpoint_selection_rejects_hardlinked_checkpoint(tmp_path: Path, monkeypatch):
    root = tmp_path / "workspace"
    spec = _spec(root)
    selection = _selection(tmp_path)
    checkpoint = Path(selection["checkpoint"])
    (tmp_path / "checkpoint-alias.ckpt").hardlink_to(checkpoint)
    source_run = {"step_id": "train-age", "run_id": "run-001"}
    source_plan = {
        "recipe": {
            "task": "hparam_tune",
            "variant": "sleep2vec2",
            "step": {"id": "train-age"},
            "inputs": {"label_name": "age"},
        },
        "runs": [source_run],
    }
    ranking = [
        {
            **source_run,
            "run_name": "source-best",
            "rank": "1",
            "score": "4.5",
            "config": selection["config"],
            "checkpoint_path": str(checkpoint),
        }
    ]
    canonical = [{**source_run, "status": "completed", "checkpoint_dir": str(checkpoint.parent)}]
    monkeypatch.setattr(experiment_pipeline.artifacts, "read_hparam_plan", lambda _plan_dir: source_plan)
    monkeypatch.setattr(experiment_pipeline, "select_hparam_candidates", lambda *_args: None)
    monkeypatch.setattr(experiment_pipeline, "read_run_manifest", lambda _root: canonical)
    monkeypatch.setattr(experiment_pipeline, "read_rows", lambda *_args, **_kwargs: ranking)

    with pytest.raises(ValueError, match="independent regular files"):
        experiment_pipeline._select_checkpoint_sources(root, spec)


def test_frozen_checkpoint_selection_rejects_hardlinked_checkpoint(tmp_path: Path):
    root = tmp_path / "workspace"
    spec = _spec(root)
    selection = _selection(tmp_path)
    checkpoint = Path(selection["checkpoint"])
    (tmp_path / "checkpoint-alias.ckpt").hardlink_to(checkpoint)
    selection.update(
        {
            "plan": spec["checkpoint_sources"]["age"]["plan"],
            "source_task": "age",
            "source_plan_task": "hparam_tune",
            "inference_task": "infer",
        }
    )
    path = tmp_path / "checkpoints.json"
    path.write_text(json.dumps({"pipeline_id": "external-v1", "sources": [selection]}) + "\n")

    with pytest.raises(ValueError, match="independent regular files"):
        experiment_pipeline._read_frozen_selections(path, spec)


@pytest.mark.parametrize("retryable_status", ["failed", "launch_failed"])
def test_retryable_attempt_creates_exactly_one_fresh_second_attempt(tmp_path: Path, monkeypatch, retryable_status: str):
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
    attempts = [{"job_id": "age-hsp-i2-psg", "attempt": 1, "status": retryable_status, "verified": "false"}]

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
    retry_recipe = yaml.safe_load((pipeline_dir / "recipes" / "age-hsp-i2-psg" / "attempt-002.yaml").read_text())
    assert retry_recipe["execution"] == {
        "target": "local",
        "workdir": spec["runtime"]["workdir"],
        "python": spec["runtime"]["python"],
        "runtime_commit": spec["runtime"]["runtime_commit"],
    }

    updated[1]["status"] = retryable_status
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
    assert recipe["execution"] == {
        "target": "local",
        "workdir": "/runtime/snapshot",
        "python": "/runtime/python",
        "runtime_commit": "a" * 40,
    }
    assert recipe["artifacts"]["overwrite"] is False


def test_result_manifest_validation_accepts_exact_manifest_and_rejects_mismatch(tmp_path: Path):
    spec, attempt, run, manifest_path, manifest = _result_manifest_context(tmp_path)
    assert experiment_pipeline._validate_result_manifest(spec, attempt, run) == manifest_path

    manifest["runtime"]["devices"] = [1]
    manifest_path.write_text(json.dumps(manifest) + "\n")
    with pytest.raises(ValueError, match="logical device 0"):
        experiment_pipeline._validate_result_manifest(spec, attempt, run)


@pytest.mark.parametrize("avg_ckpts", [True, 1.0, 2.0, 2])
def test_result_manifest_validation_rejects_non_integer_or_wrong_avg_ckpts(tmp_path: Path, avg_ckpts: object):
    spec, attempt, run, manifest_path, manifest = _result_manifest_context(tmp_path)
    manifest["checkpoint"]["avg_ckpts"] = avg_ckpts
    manifest_path.write_text(json.dumps(manifest) + "\n")

    with pytest.raises(ValueError, match="does not prove avg_ckpts=1"):
        experiment_pipeline._validate_result_manifest(spec, attempt, run)


def test_result_manifest_validation_rejects_missing_and_corrupt_manifest(tmp_path: Path):
    spec, attempt, run, manifest_path, _manifest = _result_manifest_context(tmp_path)
    manifest_path.unlink()
    with pytest.raises(ValueError, match="exactly one run_manifest"):
        experiment_pipeline._validate_result_manifest(spec, attempt, run)

    manifest_path.write_text("{not-json\n")
    with pytest.raises(json.JSONDecodeError):
        experiment_pipeline._validate_result_manifest(spec, attempt, run)


def test_result_manifest_validation_rejects_hardlinked_manifest(tmp_path: Path):
    spec, attempt, run, manifest_path, _manifest = _result_manifest_context(tmp_path)
    (tmp_path / "manifest-alias.json").hardlink_to(manifest_path)

    with pytest.raises(ValueError, match="independent regular files"):
        experiment_pipeline._validate_result_manifest(spec, attempt, run)


@pytest.mark.parametrize("field", ["metrics_csv_path", "prediction_csv_path"])
def test_result_manifest_validation_requires_result_artifact_path(tmp_path: Path, field: str):
    spec, attempt, run, manifest_path, manifest = _result_manifest_context(tmp_path)
    manifest["paths"].pop(field)
    manifest_path.write_text(json.dumps(manifest) + "\n")

    with pytest.raises(ValueError, match=rf"paths\.{field} is required"):
        experiment_pipeline._validate_result_manifest(spec, attempt, run)


@pytest.mark.parametrize("field", ["metrics_csv_path", "prediction_csv_path"])
def test_result_manifest_validation_rejects_missing_result_artifact(tmp_path: Path, field: str):
    spec, attempt, run, _manifest_path, manifest = _result_manifest_context(tmp_path)
    Path(manifest["paths"][field]).unlink()

    with pytest.raises(ValueError, match=rf"missing or not a regular file: {field}"):
        experiment_pipeline._validate_result_manifest(spec, attempt, run)


@pytest.mark.parametrize("field", ["metrics_csv_path", "prediction_csv_path"])
def test_result_manifest_validation_rejects_hardlinked_result_artifact(tmp_path: Path, field: str):
    spec, attempt, run, _manifest_path, manifest = _result_manifest_context(tmp_path)
    (tmp_path / f"{field}-alias.csv").hardlink_to(Path(manifest["paths"][field]))

    with pytest.raises(ValueError, match="independent regular files"):
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
