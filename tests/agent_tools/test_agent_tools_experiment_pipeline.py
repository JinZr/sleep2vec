from __future__ import annotations

import copy
import csv
import fcntl
import json
from pathlib import Path
import threading
from types import SimpleNamespace

from agent_tool_test_helpers import (
    FakePipelineRuntime,
    PipelineInterrupted,
    dry_run_pipeline,
    prepare_pipeline_sources,
    run_pipeline,
    write_finetune_recipe,
)
import pytest
import yaml

from agent_tools import (
    experiment_pipeline,
    experiment_pipeline_attempts as pipeline_attempts,
    experiment_pipeline_results as pipeline_results,
    experiment_pipeline_spec as pipeline_spec,
    hparam_selection,
)
from agent_tools.experiment_workspace import commit_step_manifest, file_sha256, plan_registration_lock
from agent_tools.manifests import read_rows, write_rows


def test_freeze_attempt_recipe_writes_once_and_reuses_exact_yaml(tmp_path: Path, monkeypatch):
    recipe_path = tmp_path / "recipes" / "attempt-001.yaml"
    recipe = {"task": "infer", "runtime": {"seed": 4523}}
    writes = []

    def write(path, text):
        writes.append((path, text))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    monkeypatch.setattr(pipeline_attempts, "_atomic_write_text", write)

    pipeline_attempts._freeze_attempt_recipe(recipe, recipe_path, drift_message="recipe drifted")
    pipeline_attempts._freeze_attempt_recipe(recipe, recipe_path, drift_message="recipe drifted")

    assert writes == [(recipe_path, yaml.safe_dump(recipe, sort_keys=False))]


@pytest.mark.parametrize("existing_kind", ["changed", "directory", "symlink"])
def test_freeze_attempt_recipe_rejects_existing_drift(tmp_path: Path, existing_kind: str):
    recipe_path = tmp_path / "attempt-001.yaml"
    if existing_kind == "changed":
        recipe_path.write_text("task: changed\n")
    elif existing_kind == "directory":
        recipe_path.mkdir()
    else:
        target = tmp_path / "other.yaml"
        target.write_text("task: infer\n")
        recipe_path.symlink_to(target)

    with pytest.raises(ValueError) as exc_info:
        pipeline_attempts._freeze_attempt_recipe(
            {"task": "infer"},
            recipe_path,
            drift_message="recipe drifted",
        )

    assert str(exc_info.value) == f"recipe drifted: {recipe_path}"


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
    pipeline_spec.validate_spec(spec, root, unlock_final_test=True)

    duplicate = copy.deepcopy(spec)
    duplicate["jobs"].append(copy.deepcopy(duplicate["jobs"][0]))
    with pytest.raises(ValueError, match="Duplicate external job id"):
        pipeline_spec.validate_spec(duplicate, root, unlock_final_test=True)

    illegal_phase = copy.deepcopy(spec)
    illegal_phase["pipeline"]["step"]["phase"] = "external_test"
    with pytest.raises(ValueError, match="phase must be 'evaluate'"):
        pipeline_spec.validate_spec(illegal_phase, root, unlock_final_test=True)

    with pytest.raises(ValueError, match="requires --unlock-final-test"):
        pipeline_spec.validate_spec(spec, root, unlock_final_test=False)


def test_freeze_pipeline_rejects_step_controller_conflict_before_writing_state(tmp_path: Path, monkeypatch):
    root = tmp_path / "workspace"
    spec = _spec(root)
    spec_path = prepare_pipeline_sources(tmp_path, monkeypatch, spec)
    commit_step_manifest(
        root,
        {
            "step": spec["pipeline"]["step"],
            "experiment_id": "unit",
            "plan_controller": "ordinary",
            "recipe_path": str(root / "ordinary.yaml"),
            "plans": [str(root / "plans" / "ordinary")],
        },
    )

    with pytest.raises(ValueError, match="plan_controller differs"):
        run_pipeline(spec_path, _stop_at_registration_hooks())

    staging_dirs = list((root / "pipelines").glob(".external-v1.*.staging"))
    assert [list(path.iterdir()) for path in staging_dirs] == [[]]
    assert not (root / "pipelines" / "external-v1").exists()


@pytest.mark.parametrize("field", ["workdir", "python", "runtime_commit"])
def test_schema_rejects_non_string_runtime_identity(tmp_path: Path, field: str):
    root = tmp_path / "workspace"
    spec = _spec(root)
    spec["runtime"][field] = []

    with pytest.raises(ValueError, match=rf"runtime\.{field}"):
        pipeline_spec.validate_spec(spec, root, unlock_final_test=True)


@pytest.mark.parametrize("python_command", ["conda run -n exp python", "~/miniconda/bin/python"])
def test_schema_rejects_non_executable_runtime_python(tmp_path: Path, python_command: str):
    root = tmp_path / "workspace"
    spec = _spec(root)
    spec["runtime"]["python"] = python_command

    with pytest.raises(ValueError, match=r"runtime\.python must be a single executable"):
        pipeline_spec.validate_spec(spec, root, unlock_final_test=True)


def test_schema_rejects_sha256_runtime_commit(tmp_path: Path):
    root = tmp_path / "workspace"
    spec = _spec(root)
    spec["runtime"]["runtime_commit"] = "b" * 64

    with pytest.raises(ValueError, match="full lowercase 40-character"):
        pipeline_spec.validate_spec(spec, root, unlock_final_test=True)


def test_external_pipeline_explicitly_rejects_slurm_before_state_creation(tmp_path: Path):
    root = tmp_path / "workspace"
    root.mkdir()
    spec = _spec(root)
    spec["execution"]["scheduler"] = {"type": "slurm"}
    spec_path = tmp_path / "external.yaml"
    spec_path.write_text(yaml.safe_dump(spec, sort_keys=False))

    with pytest.raises(ValueError, match="supports only execution.scheduler.type=direct"):
        experiment_pipeline.run_experiment_pipeline(root, spec_path, unlock_final_test=True, execute=True)

    assert not (root / "pipelines").exists()


def test_external_pipeline_accepts_explicit_direct_scheduler(tmp_path: Path):
    root = tmp_path / "workspace"
    spec = _spec(root)
    spec["execution"]["scheduler"] = {"type": "direct"}

    pipeline_spec.validate_spec(spec, root, unlock_final_test=True)


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
        pipeline_spec.validate_spec(spec, root, unlock_final_test=True)


def test_dry_run_does_not_freeze_or_mutate_workspace(tmp_path: Path, monkeypatch):
    root = tmp_path / "workspace"
    spec = _spec(root)
    prepare_pipeline_sources(tmp_path, monkeypatch, spec)
    # Reading the source plan takes the workspace run lock; the lock file is not a pipeline output.
    (root / "run_manifest.tsv.lock").touch()
    before = {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}

    result = dry_run_pipeline(root, spec)

    after = {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}
    assert result["status"] == "ready"
    assert result["dry_run"] is True
    assert before == after
    assert not (root / "pipelines").exists()


@pytest.mark.parametrize("execute", [False, True])
def test_external_pipeline_rejects_ssh_source_plan_before_outputs(tmp_path: Path, monkeypatch, execute: bool):
    root = tmp_path / "workspace"
    root.mkdir()
    # Reading the source plan takes the workspace run lock; the lock file is not a pipeline output.
    (root / "run_manifest.tsv.lock").touch()
    spec_path = tmp_path / "external.yaml"
    spec_path.write_text(yaml.safe_dump(_spec(root), sort_keys=False))
    monkeypatch.setattr(
        experiment_pipeline.artifacts,
        "read_hparam_plan",
        lambda *_args, **_kwargs: {
            "recipe": {"experiment": {"root": str(root)}, "execution": {"target": "ssh", "host": "unit-host"}},
        },
    )
    before = {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}

    with pytest.raises(ValueError, match=r"checkpoint_sources\.age\.plan.*SSH execution target"):
        experiment_pipeline.run_experiment_pipeline(
            root,
            spec_path,
            unlock_final_test=execute,
            execute=execute,
        )

    after = {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}
    assert after == before
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
        experiment_pipeline.artifacts,
        "read_hparam_plan",
        lambda *_args, **_kwargs: {"recipe": {"execution": {"target": "local"}}},
    )
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


@pytest.fixture
def cross_round_source(tmp_path: Path, monkeypatch):
    root = tmp_path / "workspace"
    spec = _spec(root)
    _write_experiment(root)
    source_dir = Path(spec["checkpoint_sources"]["age"]["plan"])
    source_recipe = yaml.safe_load(write_finetune_recipe(tmp_path / "source", variant="sleep2vec2").read_text())
    config = Path(source_recipe["inputs"]["config"])
    preset = Path(spec["jobs"][0]["inference_preset_path"])
    preset.parent.mkdir(parents=True)
    preset.write_bytes(b"preset")
    registered = []
    canonical = []
    for index, plan_dir in enumerate((source_dir, source_dir.with_name("round-002")), start=1):
        plan_dir.mkdir(parents=True)
        (plan_dir / "plan.json").write_text("{}\n")
        (plan_dir / "recipe.resolved.yaml").write_text("task: hparam_tune\n")
        checkpoint_dir = plan_dir / "checkpoints"
        checkpoint_dir.mkdir()
        checkpoint = checkpoint_dir / "epoch=1.ckpt"
        checkpoint.write_bytes(f"round {index}".encode())
        runtime_dir = plan_dir / "runtime"
        runtime_dir.mkdir()
        (runtime_dir / "run_manifest.json").write_text(json.dumps({"status": "completed"}) + "\n")
        run = {
            "step_id": "train-age",
            "run_id": f"run-{index:03d}",
            "config": str(config),
            "config_sha256": file_sha256(config),
            "checkpoint_dir": str(checkpoint_dir),
            "runtime_dir": str(runtime_dir),
        }
        recipe = {
            "task": "hparam_tune",
            "variant": "sleep2vec2",
            "experiment": {"id": "unit", "root": str(root)},
            "step": {"id": "train-age"},
            "inputs": {"label_name": "age"},
            "evaluation_policy": {"selection_metric": "val_mae", "selection_mode": "min", "selection_split": "val"},
            "execution": {"target": "local"},
        }
        registered.append((plan_dir, {"recipe": recipe, "runs": [run]}))
        canonical.append(
            {
                **run,
                "status": "completed",
                "run_name": f"round-{index}",
                "rank": str(3 - index),
                "score": str(6 - index),
                "checkpoint_path": str(checkpoint),
                "checkpoint_sha256": file_sha256(checkpoint),
            }
        )
    # Attempt registration reads the run manifest file; source status reads below see the mutable canonical list.
    write_rows(
        root / "run_manifest.tsv",
        [{"experiment_id": "unit", **plan["runs"][0], "status": "completed"} for _path, plan in registered],
    )
    plans = dict(registered)
    monkeypatch.setattr(experiment_pipeline.artifacts, "read_hparam_plan", lambda path, **_kwargs: plans[Path(path)])
    monkeypatch.setattr(
        experiment_pipeline.artifacts, "iter_registered_hparam_plans", lambda *_args, **_kwargs: iter(registered)
    )
    monkeypatch.setattr(experiment_pipeline, "read_run_manifest", lambda _root: canonical)
    monkeypatch.setattr(hparam_selection, "read_run_manifest", lambda _root: canonical)
    monkeypatch.setattr(hparam_selection.tracking, "validated_hparam_ranking", lambda _step: canonical)
    monkeypatch.setattr(
        hparam_selection,
        "read_rows",
        lambda *_args, **_kwargs: hparam_selection.tracking.hparam_ranking_projection(canonical),
    )
    monkeypatch.setattr(experiment_pipeline, "select_hparam_candidates", lambda *_args: None)
    monkeypatch.setattr(experiment_pipeline, "_validate_checkpoint_payload", lambda *_args: {})
    return root, spec, registered, canonical


def _write_spec(tmp_path: Path, spec: dict) -> Path:
    spec_path = tmp_path / "external.yaml"
    spec_path.write_text(yaml.safe_dump(spec, sort_keys=False))
    return spec_path


def _as_cohort_selection(spec: dict, candidates: dict) -> None:
    spec.pop("schema_version")
    spec["pipeline"]["kind"] = "cohort_selection"
    spec["candidates"] = candidates
    job = spec["jobs"][0]
    job.pop("checkpoint_source")
    job.update({"role": "selection", "provenance": "external"})
    spec["selector"] = {
        "strategy": "target_gate",
        "gates": [{"job": job["id"], "metric": "mae", "mode": "min", "threshold": 5.0}],
        "tie_breaker": "internal_rank",
        "on_no_feasible": "no_winner",
    }


def _stop_at_registration_hooks(probed: list | None = None, **overrides):
    """Hooks that let source refresh pass and stop when attempt registration first probes its execution target."""

    def stop(_execution, runs, **_kwargs):
        if probed is not None:
            probed.append([dict(run) for run in runs])
        raise PipelineInterrupted

    effects = {"monitor_runs": lambda plan_dir, **_kwargs: plan_dir, "inspect_target": stop, **overrides}
    return experiment_pipeline.PipelineHooks(**effects)


def _run_until_registration(spec_path: Path, *, resume: bool = False, **overrides) -> list[list[dict]]:
    """Execute until the selection is frozen and registration probes its first group; return the probed runs."""
    probed: list[list[dict]] = []
    with pytest.raises(PipelineInterrupted):
        run_pipeline(spec_path, _stop_at_registration_hooks(probed, **overrides), resume=resume)
    return probed


def _frozen_selection(root: Path, spec: dict) -> tuple[Path, list[dict]]:
    cohort = spec["pipeline"]["kind"] == "cohort_selection"
    path = root / "pipelines" / spec["pipeline"]["id"] / ("candidates.json" if cohort else "checkpoints.json")
    return path, json.loads(path.read_text())["candidates" if cohort else "sources"]


@pytest.mark.parametrize("execute", [False, True])
def test_pipeline_rejects_ssh_owner_in_another_round_before_outputs(cross_round_source, tmp_path, execute):
    root, spec, registered, _canonical = cross_round_source
    registered[0][1]["recipe"]["execution"] = {"target": "local"}
    registered[1][1]["recipe"]["execution"] = {"target": "ssh", "host": "unit-host"}
    spec_path = tmp_path / "external.yaml"
    spec_path.write_text(yaml.safe_dump(spec, sort_keys=False))
    # Reading source plans takes the workspace run lock; the lock file is not a pipeline output.
    (root / "run_manifest.tsv.lock").touch()
    before = {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}

    with pytest.raises(ValueError, match="SSH execution target"):
        experiment_pipeline.run_experiment_pipeline(
            root,
            spec_path,
            unlock_final_test=execute,
            execute=execute,
        )

    after = {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}
    assert after == before
    assert not (root / "pipelines").exists()


@pytest.mark.parametrize("late_owner", ["ssh", "incompatible_variant"])
def test_first_publication_rechecks_late_owner_before_staging(cross_round_source, tmp_path, monkeypatch, late_owner):
    root, spec, registered_plans, _canonical = cross_round_source
    spec_path = _write_spec(tmp_path, spec)
    pipeline_dir = root / "pipelines" / spec["pipeline"]["id"]
    initial_scan = threading.Event()
    registration_done = threading.Event()
    errors = []
    scans = []
    original_source_plans = experiment_pipeline._source_hparam_plans
    late_plan = copy.deepcopy(registered_plans[0][1])
    late_plan["runs"][0]["run_id"] = "run-003"
    if late_owner == "ssh":
        late_plan["recipe"]["execution"] = {"target": "ssh", "host": "unit-host"}
        message = "SSH execution target"
    else:
        late_plan["recipe"]["variant"] = "sleep2expert"
        message = "variant assertion differs"

    def register_late_plan():
        try:
            assert initial_scan.wait(timeout=5)
            with plan_registration_lock(root):
                registered_plans.append((root / "plans" / "round-003", late_plan))
        except BaseException as exc:
            errors.append(exc)
        finally:
            registration_done.set()

    registrar = threading.Thread(target=register_late_plan)

    def source_plans(*args):
        scans.append(True)
        plans = original_source_plans(*args)
        if len(scans) == 1:
            initial_scan.set()
            assert registration_done.wait(timeout=5)
            assert not errors
        return plans

    monkeypatch.setattr(experiment_pipeline, "_source_hparam_plans", source_plans)
    registrar.start()
    try:
        with pytest.raises(ValueError, match=message):
            run_pipeline(spec_path, _stop_at_registration_hooks())
    finally:
        initial_scan.set()
        registrar.join(timeout=5)

    assert not registrar.is_alive()
    assert not errors
    assert len(scans) == 2
    assert not pipeline_dir.exists()
    assert not list(pipeline_dir.parent.glob("*.staging"))
    assert not list(pipeline_dir.parent.rglob("pipeline.json"))

    registered_plans.pop()
    _run_until_registration(spec_path)

    assert json.loads((pipeline_dir / "pipeline.json").read_text())["status"] == "ready"


@pytest.mark.parametrize("scope", ["all", "top_k", "external_matrix"])
def test_checkpoint_selection_includes_other_rounds_and_preserves_owner(cross_round_source, tmp_path, scope):
    root, spec, registered, canonical = cross_round_source
    if scope != "external_matrix":
        _as_cohort_selection(spec, {"kind": scope, **({"count": 1} if scope == "top_k" else {})})

    _run_until_registration(_write_spec(tmp_path, spec))

    _path, selected = _frozen_selection(root, spec)
    expected = list(reversed(canonical)) if scope == "all" else [canonical[1]]
    assert [row["run_id"] for row in selected] == [row["run_id"] for row in expected]
    owner_dirs = {plan["runs"][0]["run_id"]: str(path) for path, plan in registered}
    for row, run in zip(selected, expected, strict=True):
        assert row["plan"] == owner_dirs[run["run_id"]]
        assert row["config"] == run["config"]
        assert row["checkpoint"] == run["checkpoint_path"]
        assert row["score"] == float(run["score"])


@pytest.mark.parametrize("kind", ["external_matrix", "cohort_selection"])
@pytest.mark.parametrize("execute", [False, True])
@pytest.mark.parametrize("later_owner", ["local", "ssh", "corrupt"])
def test_bound_selection_resume_ignores_new_running_round(
    cross_round_source, tmp_path, monkeypatch, kind, execute, later_owner
):
    root, spec, registered, canonical = cross_round_source
    if kind == "cohort_selection":
        _as_cohort_selection(spec, {"kind": "all"})
    spec_path = _write_spec(tmp_path, spec)
    _run_until_registration(spec_path)
    selection_path, frozen = _frozen_selection(root, spec)
    frozen_bytes = selection_path.read_bytes()

    later_run = {**registered[0][1]["runs"][0], "run_id": "run-003"}
    later_recipe = {**registered[0][1]["recipe"], "execution": {"target": later_owner}}
    later_plan = {} if later_owner == "corrupt" else {"recipe": later_recipe, "runs": [later_run]}
    registered.append((root / "plans" / "round-003", later_plan))
    canonical.append({**later_run, "status": "running"})
    monkeypatch.setattr(
        experiment_pipeline.artifacts,
        "iter_registered_hparam_plans",
        lambda *_args, **_kwargs: pytest.fail("bound selection must not expand to newly registered rounds"),
    )
    monkeypatch.setattr(
        experiment_pipeline,
        "plan_registration_lock",
        lambda *_args: pytest.fail("bound selection must not reacquire registration lock"),
    )

    if execute:
        probed = _run_until_registration(
            spec_path,
            resume=True,
            monitor_runs=lambda *_args, **_kwargs: pytest.fail(
                "bound selection must not monitor later training rounds"
            ),
            sleep=lambda *_args: pytest.fail("bound selection must not wait for later training rounds"),
        )
        # One variant registers as one scheduler group, so the first probe covers every attempt.
        (group,) = probed
        evaluated = {run["command"].split("--ckpt-path ", 1)[1].split()[0] for run in group}
        assert evaluated == {selection["checkpoint"] for selection in frozen}
    else:
        assert experiment_pipeline.run_experiment_pipeline(root, spec_path)["status"] == "ready"
    assert selection_path.read_bytes() == frozen_bytes


@pytest.mark.parametrize("kind", ["external_matrix", "cohort_selection"])
@pytest.mark.parametrize("orphan", [False, True])
def test_selection_publication_holds_registration_lock_through_hash_commit(cross_round_source, tmp_path, kind, orphan):
    root, spec, _registered, _canonical = cross_round_source
    if kind == "cohort_selection":
        _as_cohort_selection(spec, {"kind": "all"})
    hash_field = "candidate_selection_sha256" if kind == "cohort_selection" else "checkpoint_selection_sha256"
    event_type = "pipeline_candidates_frozen" if kind == "cohort_selection" else "pipeline_checkpoints_frozen"
    spec_path = _write_spec(tmp_path, spec)
    state_path = root / "pipelines" / spec["pipeline"]["id"] / "pipeline.json"
    if orphan:
        _run_until_registration(spec_path)
        state = json.loads(state_path.read_text())
        state.pop(hash_field)
        state_path.write_text(json.dumps(state))
    attempting = threading.Event()
    registered = threading.Event()
    errors = []
    registration_views = []

    def register_later_round():
        attempting.set()
        try:
            with plan_registration_lock(root):
                events = [event.get("event_type") for event in pipeline_attempts.read_experiment_events(root)]
                registration_views.append((json.loads(state_path.read_text()), events))
                registered.set()
        except BaseException as exc:
            errors.append(exc)

    registrar = threading.Thread(target=register_later_round)

    def monitor_before_ranking(plan_dir, **_kwargs):
        # Source refresh runs under the registration lock that ranking and selection publication keep holding.
        if registrar.ident is None:
            registrar.start()
            assert attempting.wait(timeout=5)
            assert not registered.wait(timeout=0.1)
        return plan_dir

    def inspect_after_publication(*_args, **_kwargs):
        assert registered.wait(timeout=5)
        raise PipelineInterrupted

    hooks = experiment_pipeline.PipelineHooks(
        monitor_runs=monitor_before_ranking, inspect_target=inspect_after_publication
    )
    try:
        with pytest.raises(PipelineInterrupted):
            run_pipeline(spec_path, hooks, resume=orphan)
    finally:
        if registrar.ident is not None:
            registrar.join(timeout=5)

    assert not registrar.is_alive()
    assert not errors
    selection_path, _selected = _frozen_selection(root, spec)
    state, events = registration_views[0]
    assert state[hash_field] == file_sha256(selection_path)
    assert event_type in events


def test_waiting_source_poll_releases_registration_lock_before_sleep(cross_round_source, tmp_path):
    root, spec, _registered, canonical = cross_round_source
    canonical[1]["status"] = "running"
    acquired = threading.Event()

    def register_later_round():
        with plan_registration_lock(root):
            acquired.set()

    registrar = threading.Thread(target=register_later_round)

    def sleep_after_poll(_seconds):
        registrar.start()
        assert acquired.wait(timeout=5), "source polling must release registration lock while waiting"
        raise PipelineInterrupted

    try:
        with pytest.raises(PipelineInterrupted):
            run_pipeline(_write_spec(tmp_path, spec), _stop_at_registration_hooks(sleep=sleep_after_poll))
    finally:
        if registrar.ident is not None:
            registrar.join(timeout=5)
    assert not registrar.is_alive()


@pytest.mark.parametrize("wrong_owner", [False, True])
def test_frozen_cross_round_winner_requires_its_registered_owner(cross_round_source, tmp_path, wrong_owner):
    root, spec, registered, _canonical = cross_round_source
    spec_path = _write_spec(tmp_path, spec)
    _run_until_registration(spec_path)
    selection_path, selected = _frozen_selection(root, spec)
    assert selected[0]["plan"] == str(registered[1][0])
    if not wrong_owner:
        assert len(_run_until_registration(spec_path, resume=True)) == 1
        return
    # A selection manifest whose hash was never committed is re-read before it is re-derived.
    payload = json.loads(selection_path.read_text())
    payload["sources"][0]["plan"] = str(registered[0][0])
    selection_path.write_text(json.dumps(payload))
    state_path = selection_path.parent / "pipeline.json"
    state = json.loads(state_path.read_text())
    state.pop("checkpoint_selection_sha256")
    state_path.write_text(json.dumps(state))

    with pytest.raises(ValueError, match="source field drifted: age.plan"):
        run_pipeline(spec_path, _stop_at_registration_hooks(), resume=True)


@pytest.mark.parametrize("field", ["experiment_id", "experiment_root", "step_id", "selection_split"])
def test_frozen_cross_round_owner_must_match_anchor_contract(cross_round_source, tmp_path, field):
    root, spec, registered, _canonical = cross_round_source
    spec_path = _write_spec(tmp_path, spec)
    _run_until_registration(spec_path)
    owner_recipe = registered[1][1]["recipe"]
    if field == "selection_split":
        owner_recipe["evaluation_policy"]["selection_split"] = "test"
    elif field == "step_id":
        owner_recipe["step"]["id"] = "other-step"
    elif field == "experiment_id":
        owner_recipe["experiment"]["id"] = "other-experiment"
    else:
        owner_recipe["experiment"]["root"] = str(root / "other-workspace")
        # The stubbed plan read skips the plan-inside-root check, so the run lock needs the directory.
        (root / "other-workspace").mkdir()

    with pytest.raises(ValueError, match="source field drifted: age.plan"):
        run_pipeline(spec_path, _stop_at_registration_hooks(), resume=True)


def test_source_refresh_monitors_every_registered_round(cross_round_source, tmp_path):
    root, spec, registered, canonical = cross_round_source
    canonical[1]["status"] = "running"
    monitored = []

    def stop_waiting(_seconds):
        raise PipelineInterrupted

    hooks = _stop_at_registration_hooks(
        monitor_runs=lambda path, **_kwargs: monitored.append(path) or path, sleep=stop_waiting
    )
    with pytest.raises(PipelineInterrupted):
        run_pipeline(_write_spec(tmp_path, spec), hooks)

    assert monitored == [path for path, _plan in registered]


@pytest.mark.parametrize("status", ["running", "pending", "unknown"])
def test_source_readiness_waits_for_other_registered_round(cross_round_source, status):
    root, spec, _registered, canonical = cross_round_source
    canonical[1]["status"] = status

    states = dry_run_pipeline(root, spec)["source_states"]

    assert states[0]["complete"] is False
    assert states[0]["statuses"] == ["completed", status]


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
    spec_path = prepare_pipeline_sources(tmp_path, monkeypatch, spec)
    canonical = {"status": initial_status}
    resolve = experiment_pipeline.resolve_hparam_candidates

    def select_candidates(*_args):
        canonical["status"] = current_status

    def resolve_candidates(plan_dir, runs, **kwargs):
        if canonical["status"] not in {"completed", "finished"}:
            raise ValueError("No successful selected candidates remain after canonical selection filtering.")
        return resolve(plan_dir, runs, **kwargs)

    monkeypatch.setattr(experiment_pipeline, "select_hparam_candidates", select_candidates)
    monkeypatch.setattr(experiment_pipeline, "resolve_hparam_candidates", resolve_candidates)

    if not should_select:
        with pytest.raises(ValueError, match="No successful selected candidates"):
            run_pipeline(spec_path, _stop_at_registration_hooks())
        return

    _run_until_registration(spec_path)

    _path, selected = _frozen_selection(root, spec)
    assert selected[0]["run_id"] == "run-001"
    assert selected[0]["checkpoint"] == str(
        root / "plans" / "train-age" / "runs" / "run-001" / "checkpoints" / "epoch=1.ckpt"
    )


def test_checkpoint_selection_rejects_hardlinked_checkpoint(tmp_path: Path, monkeypatch):
    root = tmp_path / "workspace"
    spec = _spec(root)
    spec_path = prepare_pipeline_sources(tmp_path, monkeypatch, spec)
    checkpoint = root / "plans" / "train-age" / "runs" / "run-001" / "checkpoints" / "epoch=1.ckpt"
    (tmp_path / "checkpoint-alias.ckpt").hardlink_to(checkpoint)

    with pytest.raises(ValueError, match="independent regular files"):
        run_pipeline(spec_path, _stop_at_registration_hooks())

    assert not (root / "pipelines" / spec["pipeline"]["id"] / "checkpoints.json").exists()


def test_frozen_checkpoint_selection_preserves_unknown_fields_and_decoded_identity(tmp_path: Path, monkeypatch):
    spec = _spec(tmp_path / "workspace")
    selection = _selection(tmp_path)
    selection.update({"step_id": "train-age", "run_id": "run-001"})
    monkeypatch.setattr(
        experiment_pipeline,
        "_validate_frozen_selection_owner",
        lambda *_args: None,
    )
    selection.update(
        {
            "plan": spec["checkpoint_sources"]["age"]["plan"],
            "source_task": "age",
            "source_plan_task": "hparam_tune",
            "inference_task": "infer",
            "extra_evidence": {"notes": ["frozen"], "optional": None},
        }
    )
    path = tmp_path / "checkpoints.json"
    path.write_text(json.dumps({"pipeline_id": "external-v1", "sources": [selection]}) + "\n")
    decoded = json.loads(path.read_text())
    monkeypatch.setattr(experiment_pipeline, "read_json", lambda _path: decoded)

    selections = experiment_pipeline._read_frozen_selections(path, spec)

    selected = selections["age"]
    assert selected == selection
    assert selected is decoded["sources"][0]
    assert selected["extra_evidence"] is decoded["sources"][0]["extra_evidence"]
    selected["extra_evidence"]["notes"].append("reviewed")
    assert decoded["sources"][0]["extra_evidence"]["notes"] == ["frozen", "reviewed"]
    assert json.loads(path.read_text())["sources"][0] == selection


def test_frozen_checkpoint_selection_rejects_hardlinked_checkpoint(tmp_path: Path, monkeypatch):
    root = tmp_path / "workspace"
    spec = _spec(root)
    spec_path = prepare_pipeline_sources(tmp_path, monkeypatch, spec)
    _run_until_registration(spec_path)
    _path, (selection,) = _frozen_selection(root, spec)
    (tmp_path / "checkpoint-alias.ckpt").hardlink_to(selection["checkpoint"])

    with pytest.raises(ValueError, match="independent regular files"):
        run_pipeline(spec_path, _stop_at_registration_hooks(), resume=True)


@pytest.mark.parametrize("retryable_status", ["failed", "launch_failed"])
def test_retryable_attempt_creates_exactly_one_fresh_second_attempt(tmp_path: Path, monkeypatch, retryable_status: str):
    root = tmp_path / "workspace"
    _write_experiment(root)
    pipeline_dir = root / "pipelines" / "external-v1"
    pipeline_dir.mkdir(parents=True)
    spec = _spec(root)
    selection = _selection(tmp_path)
    selection.update({"step_id": "train-age", "run_id": "run-001"})
    monkeypatch.setattr(
        experiment_pipeline,
        "_validate_frozen_selection_owner",
        lambda *_args: None,
    )

    monkeypatch.setattr(
        pipeline_attempts,
        "preflight_plan",
        lambda **_kwargs: (None, None, SimpleNamespace(exit_code=0)),
    )

    monkeypatch.setattr(
        pipeline_attempts,
        "_materialize_attempt",
        lambda _root, _spec, job, _selection, attempt, **paths: {
            "job_id": job["id"],
            "attempt": attempt,
            "status": "planned",
            "verified": "false",
            "result_root": str(paths["result_root"]),
        },
    )
    monkeypatch.setattr(
        pipeline_attempts,
        "_prepare_attempt_registration_groups",
        lambda _root, _spec, items, **_kwargs: {item[0]["id"]: None for item in items},
    )
    monkeypatch.setattr(pipeline_attempts, "read_run_manifest", lambda _root: [])
    attempts = [{"job_id": "age-hsp-i2-psg", "attempt": 1, "status": retryable_status, "verified": "false"}]
    original_attempt = attempts[0]

    updated, created = pipeline_attempts.create_needed_retries(
        root,
        pipeline_dir,
        spec,
        {"age": selection},
        attempts,
    )

    assert created is True
    assert updated is attempts
    assert updated[0] is original_attempt
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
    unchanged, created_again = pipeline_attempts.create_needed_retries(
        root,
        pipeline_dir,
        spec,
        {"age": selection},
        updated,
    )
    assert created_again is False
    assert unchanged == updated
    assert unchanged is updated
    assert pipeline_results.logical_job_states(spec, updated)[0]["status"] == "failed"
    retry_events = [
        event
        for event in pipeline_attempts.read_experiment_events(root)
        if event.get("event_type") == "pipeline_job_retry_planned"
    ]
    assert len(retry_events) == 1
    assert retry_events[0]["attempt"] == 2


@pytest.mark.parametrize(
    "status,verified,extra,expected_status",
    [
        ("completed", "true", {}, "completed"),
        ("running", "false", {}, "running"),
        ("failed", "false", {}, "failed"),
        ("launch_failed", "false", {}, "failed"),
        ("missing_pid", "false", {}, "blocked"),
        ("unknown_remote", "false", {}, "blocked"),
        ("stopped", "false", {}, "blocked"),
        ("superseded", "false", {}, "blocked"),
        ("planned", "false", {"retry_blocker": "unsafe identity"}, "blocked"),
        ("completed", "false", {"validation_error": "manifest drift"}, "failed"),
        ("planned", "false", {"retry_preparation_error": "preflight failed"}, "failed"),
    ],
)
def test_logical_job_states_preserves_int_and_tsv_attempts(
    tmp_path: Path, status: str, verified: str, extra: dict, expected_status: str
):
    spec = _spec(tmp_path)
    attempts = [
        {
            "job_id": spec["jobs"][0]["id"],
            "attempt": 2,
            "run_id": "run-002",
            "status": status,
            "verified": verified,
            "result_manifest": "/results/manifest.json" if verified == "true" else "",
            **extra,
        },
        {
            "job_id": spec["jobs"][0]["id"],
            "attempt": 1,
            "run_id": "run-001",
            "status": "failed",
            "verified": "false",
        },
    ]
    jobs_path = tmp_path / "jobs.tsv"
    write_rows(jobs_path, attempts)
    persisted = read_rows(jobs_path)
    assert [row["attempt"] for row in persisted] == ["2", "1"]
    before_attempts = copy.deepcopy(attempts)
    before_persisted = copy.deepcopy(persisted)
    original_rows = list(attempts)
    persisted_rows = list(persisted)

    logical = pipeline_results.logical_job_states(spec, attempts)
    from_tsv = pipeline_results.logical_job_states(spec, persisted)

    assert logical == from_tsv
    assert logical[0]["status"] == expected_status
    assert logical[0]["attempt_count"] == 2
    assert logical[0]["successful_run_id"] == ("run-002" if verified == "true" else "")
    assert logical[0]["result_manifest"] == ("/results/manifest.json" if verified == "true" else "")
    assert attempts == before_attempts
    assert persisted == before_persisted
    assert all(row is original for row, original in zip(attempts, original_rows))
    assert all(row is original for row, original in zip(persisted, persisted_rows))


@pytest.mark.parametrize("field", ["run_id", "result_manifest", "retry_preparation_error"])
def test_logical_job_states_preserves_explicit_none(tmp_path: Path, field: str):
    spec = _spec(tmp_path)
    attempt = {
        "job_id": spec["jobs"][0]["id"],
        "attempt": 1,
        "run_id": "run-001",
        "status": "completed",
        "verified": "true",
        "result_manifest": "/results/manifest.json",
        "retry_preparation_error": "",
        field: None,
    }
    before = attempt.copy()

    logical = pipeline_results.logical_job_states(spec, [attempt])

    output_field = "successful_run_id" if field == "run_id" else field
    assert logical[0][output_field] is None
    assert logical[0]["status"] == "completed"
    assert attempt == before


@pytest.mark.parametrize("status", ["missing_pid", "unknown_remote", "stopped", "superseded"])
def test_uncertain_or_human_terminal_attempt_is_blocked_and_not_retried(tmp_path: Path, monkeypatch, status: str):
    root = tmp_path / "workspace"
    root.mkdir()
    spec = _spec(root)
    attempts = [{"job_id": "age-hsp-i2-psg", "attempt": 1, "status": status, "verified": "false"}]
    monkeypatch.setattr(
        pipeline_attempts,
        "_attempt_recipe",
        lambda *_args, **_kwargs: pytest.fail("uncertain attempts must not be retried"),
    )
    monkeypatch.setattr(pipeline_attempts, "read_run_manifest", lambda _root: [])

    unchanged, created = pipeline_attempts.create_needed_retries(
        root,
        root / "pipelines" / "external-v1",
        spec,
        {"age": {}},
        attempts,
    )

    assert created is False
    assert unchanged == attempts
    assert pipeline_results.logical_job_states(spec, attempts)[0]["status"] == "blocked"


@pytest.mark.parametrize("modality,workers", [("psg", 8), ("bcg", 16)])
def test_attempt_recipe_freezes_fp32_batch_workers_and_logical_gpu_zero(tmp_path: Path, modality: str, workers: int):
    root = tmp_path / "workspace"
    _write_experiment(root)
    pipeline_dir = root / "pipelines" / "external-v1"
    job = copy.deepcopy(_spec(root)["jobs"][0])
    job["modality"] = modality
    job["num_workers"] = workers

    recipe, _recipe_path, _plan_dir, result_root = pipeline_attempts._attempt_recipe(
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
    assert pipeline_results.validate_result_manifest(spec, attempt, run) == manifest_path

    manifest["runtime"]["devices"] = [1]
    manifest_path.write_text(json.dumps(manifest) + "\n")
    with pytest.raises(ValueError, match="logical device 0"):
        pipeline_results.validate_result_manifest(spec, attempt, run)


@pytest.mark.parametrize("avg_ckpts", [True, 1.0, 2.0, 2])
def test_result_manifest_validation_rejects_non_integer_or_wrong_avg_ckpts(tmp_path: Path, avg_ckpts: object):
    spec, attempt, run, manifest_path, manifest = _result_manifest_context(tmp_path)
    manifest["checkpoint"]["avg_ckpts"] = avg_ckpts
    manifest_path.write_text(json.dumps(manifest) + "\n")

    with pytest.raises(ValueError, match="does not prove avg_ckpts=1"):
        pipeline_results.validate_result_manifest(spec, attempt, run)


def test_result_manifest_validation_rejects_missing_and_corrupt_manifest(tmp_path: Path):
    spec, attempt, run, manifest_path, _manifest = _result_manifest_context(tmp_path)
    manifest_path.unlink()
    with pytest.raises(ValueError, match="exactly one run_manifest"):
        pipeline_results.validate_result_manifest(spec, attempt, run)

    manifest_path.write_text("{not-json\n")
    with pytest.raises(json.JSONDecodeError):
        pipeline_results.validate_result_manifest(spec, attempt, run)


def test_result_manifest_validation_rejects_hardlinked_manifest(tmp_path: Path):
    spec, attempt, run, manifest_path, _manifest = _result_manifest_context(tmp_path)
    (tmp_path / "manifest-alias.json").hardlink_to(manifest_path)

    with pytest.raises(ValueError, match="independent regular files"):
        pipeline_results.validate_result_manifest(spec, attempt, run)


@pytest.mark.parametrize("field", ["metrics_csv_path", "prediction_csv_path"])
def test_result_manifest_validation_requires_result_artifact_path(tmp_path: Path, field: str):
    spec, attempt, run, manifest_path, manifest = _result_manifest_context(tmp_path)
    manifest["paths"].pop(field)
    manifest_path.write_text(json.dumps(manifest) + "\n")

    with pytest.raises(ValueError, match=rf"paths\.{field} is required"):
        pipeline_results.validate_result_manifest(spec, attempt, run)


@pytest.mark.parametrize("field", ["metrics_csv_path", "prediction_csv_path"])
def test_result_manifest_validation_rejects_missing_result_artifact(tmp_path: Path, field: str):
    spec, attempt, run, _manifest_path, manifest = _result_manifest_context(tmp_path)
    Path(manifest["paths"][field]).unlink()

    with pytest.raises(ValueError, match=rf"missing or not a regular file: {field}"):
        pipeline_results.validate_result_manifest(spec, attempt, run)


@pytest.mark.parametrize("field", ["metrics_csv_path", "prediction_csv_path"])
def test_result_manifest_validation_rejects_hardlinked_result_artifact(tmp_path: Path, field: str):
    spec, attempt, run, _manifest_path, manifest = _result_manifest_context(tmp_path)
    (tmp_path / f"{field}-alias.csv").hardlink_to(Path(manifest["paths"][field]))

    with pytest.raises(ValueError, match="independent regular files"):
        pipeline_results.validate_result_manifest(spec, attempt, run)


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

    report = pipeline_results.aggregate_results(
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
    spec = _spec(root)
    spec_path = prepare_pipeline_sources(tmp_path, monkeypatch, spec)
    runtime = FakePipelineRuntime(spec, outcome=lambda _run: "failed")
    finalized = []
    monkeypatch.setattr(
        experiment_pipeline,
        "_aggregate_results",
        lambda *_args, **_kwargs: pytest.fail("an incomplete matrix must not aggregate"),
    )

    result = run_pipeline(spec_path, runtime.hooks(), finalized=finalized)

    pipeline_dir = root / "pipelines" / spec["pipeline"]["id"]
    assert result["status"] == "failed"
    assert [job["status"] for job in result["jobs"]] == ["failed"]
    assert finalized == []
    assert not (pipeline_dir / "final.md").exists()
    assert not (pipeline_dir / "results.csv").exists()
