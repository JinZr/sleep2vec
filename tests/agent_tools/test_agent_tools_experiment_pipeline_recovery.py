from __future__ import annotations

from contextlib import contextmanager
import copy
import hashlib
import json
from pathlib import Path
import shutil
import threading
from types import SimpleNamespace

from agent_tool_test_helpers import write_finetune_recipe
import pytest
import yaml

from agent_tools import experiment_pipeline, experiments, managed_scheduler, plan_contract, plans, python_programs
from agent_tools.experiment_workspace import commit_step_manifest, file_sha256, read_run_manifest
from agent_tools.manifests import write_rows


def test_attempt_materialization_enters_plan_publication_lock(tmp_path: Path, monkeypatch):
    plan_dir = tmp_path / "attempt"
    lock_active = False

    @contextmanager
    def publication_lock(out):
        nonlocal lock_active
        assert out == plan_dir
        lock_active = True
        try:
            yield
        finally:
            lock_active = False

    def materialize_locked(*_args, **_kwargs):
        assert lock_active
        return {"status": "planned"}

    monkeypatch.setattr(experiment_pipeline, "plan_publication_lock", publication_lock)
    monkeypatch.setattr(experiment_pipeline, "_materialize_attempt_locked", materialize_locked)

    result = experiment_pipeline._materialize_attempt(
        tmp_path,
        {},
        {},
        {},
        1,
        recipe_path=tmp_path / "recipe.yaml",
        plan_dir=plan_dir,
        result_root=tmp_path / "result",
    )

    assert result == {"status": "planned"}


def test_pipeline_group_registration_waits_for_ordinary_plan(tmp_path: Path, monkeypatch):
    root = tmp_path / "workspace"
    ordinary_recipe = write_finetune_recipe(root)
    ordinary_plan = root / "plans" / "ordinary"
    pipeline_dir = root / "pipelines" / "external-v1"
    pipeline_dir.mkdir(parents=True)
    spec = _spec(root)
    selections = {"age": {"variant": "sleep2vec2"}}
    ordinary_holding = threading.Event()
    release_ordinary = threading.Event()
    pipeline_preparing = threading.Event()
    original_check = plans._assert_no_incomplete_step_registration

    def pause_ordinary(recipe, out):
        if out == ordinary_plan:
            ordinary_holding.set()
            if not release_ordinary.wait(timeout=10):
                raise AssertionError("ordinary planner was not released")
        return original_check(recipe, out)

    def attempt_recipe(_pipeline_dir, _spec, job, _selection, attempt):
        base = pipeline_dir / job["id"] / f"attempt-{attempt:03d}"
        return {"name": job["id"]}, base / "recipe.yaml", base / "plan", base / "results"

    def prepare_registration(_root, _spec, items, **_kwargs):
        pipeline_preparing.set()
        return {item[0]["id"]: item[4] for item in items}

    monkeypatch.setattr(plans, "_assert_no_incomplete_step_registration", pause_ordinary)
    monkeypatch.setattr(experiment_pipeline, "_attempt_recipe", attempt_recipe)
    monkeypatch.setattr(experiment_pipeline, "_ensure_initial_preflight", lambda *_args: None)
    monkeypatch.setattr(experiment_pipeline, "_prepare_attempt_registration_groups", prepare_registration)
    monkeypatch.setattr(
        experiment_pipeline,
        "_materialize_attempt",
        lambda _root, _spec, job, _selection, attempt, **_paths: {"job_id": job["id"], "attempt": attempt},
    )
    monkeypatch.setattr(experiment_pipeline, "_write_jobs", lambda *_args: None)
    monkeypatch.setattr(experiment_pipeline, "_validate_attempt_rows", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(experiment_pipeline, "_reconcile_pipeline_jobs_planned_event", lambda *_args: None)
    ordinary_reports = []
    errors = []
    pipeline_rows = []

    def run_ordinary():
        try:
            ordinary_reports.append(plans.build_plan(recipe_path=ordinary_recipe, output_dir=ordinary_plan))
        except BaseException as exc:
            errors.append(exc)

    def run_pipeline():
        try:
            pipeline_rows.extend(
                experiment_pipeline._load_or_create_initial_attempts(root, pipeline_dir, spec, selections)
            )
        except BaseException as exc:
            errors.append(exc)

    ordinary = threading.Thread(target=run_ordinary)
    pipeline = threading.Thread(target=run_pipeline)
    ordinary.start()
    assert ordinary_holding.wait(timeout=10)
    pipeline.start()
    try:
        assert not pipeline_preparing.wait(timeout=0.5)
    finally:
        release_ordinary.set()
    ordinary.join(timeout=30)
    pipeline.join(timeout=30)

    assert not errors
    assert ordinary_reports[0].exit_code == 0
    assert pipeline_preparing.is_set()
    assert [row["job_id"] for row in pipeline_rows] == [spec["jobs"][0]["id"]]


def test_initial_jobs_projection_failure_is_recoverable(tmp_path: Path, monkeypatch):
    root = tmp_path / "workspace"
    pipeline_dir = root / "pipelines" / "external-v1"
    pipeline_dir.mkdir(parents=True)
    spec = _spec(root)
    selections = {"age": {"variant": "sleep2vec2"}}

    def attempt_recipe(_pipeline_dir, _spec, job, _selection, attempt):
        base = pipeline_dir / job["id"] / f"attempt-{attempt:03d}"
        return {"name": job["id"]}, base / "recipe.yaml", base / "plan", base / "results"

    monkeypatch.setattr(experiment_pipeline, "_attempt_recipe", attempt_recipe)
    monkeypatch.setattr(experiment_pipeline, "_ensure_initial_preflight", lambda *_args: None)
    monkeypatch.setattr(
        experiment_pipeline,
        "_prepare_attempt_registration_groups",
        lambda _root, _spec, items, **_kwargs: {item[0]["id"]: item[4] for item in items},
    )
    monkeypatch.setattr(
        experiment_pipeline,
        "_materialize_attempt",
        lambda _root, _spec, job, _selection, attempt, **_paths: {"job_id": job["id"], "attempt": attempt},
    )
    monkeypatch.setattr(
        experiment_pipeline,
        "_write_jobs",
        lambda *_args: (_ for _ in ()).throw(OSError("jobs projection interrupted")),
    )

    with pytest.raises(experiment_pipeline.PipelineRegistrationRecoveryError, match="reconciled on resume"):
        experiment_pipeline._load_or_create_initial_attempts(root, pipeline_dir, spec, selections)


def test_pipeline_jobs_planned_event_is_reconciled_after_append_failure(tmp_path: Path, monkeypatch):
    root = tmp_path / "workspace"
    root.mkdir()
    spec = _spec(root)
    original_append = experiment_pipeline.append_event
    monkeypatch.setattr(
        experiment_pipeline,
        "append_event",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("event append interrupted")),
    )

    with pytest.raises(experiment_pipeline.PipelineRegistrationRecoveryError, match="reconciled on resume"):
        experiment_pipeline._reconcile_pipeline_jobs_planned_event(root, spec)

    monkeypatch.setattr(experiment_pipeline, "append_event", original_append)
    experiment_pipeline._reconcile_pipeline_jobs_planned_event(root, spec)
    experiment_pipeline._reconcile_pipeline_jobs_planned_event(root, spec)

    events = [
        event
        for event in experiment_pipeline.read_experiment_events(root)
        if event.get("event_type") == "pipeline_jobs_planned"
    ]
    assert len(events) == 1
    assert events[0]["pipeline_id"] == spec["pipeline"]["id"]
    assert events[0]["job_count"] == len(spec["jobs"])


@pytest.mark.parametrize("failed_read", [1, 2])
def test_pipeline_jobs_planned_event_read_failure_is_recoverable(tmp_path: Path, monkeypatch, failed_read: int):
    root = tmp_path / "workspace"
    root.mkdir()
    spec = _spec(root)
    reads = 0

    def read_events(_root):
        nonlocal reads
        reads += 1
        if reads == failed_read:
            raise OSError("event read interrupted")
        return []

    monkeypatch.setattr(experiment_pipeline, "read_experiment_events", read_events)
    monkeypatch.setattr(experiment_pipeline, "append_event", lambda *_args, **_kwargs: None)

    with pytest.raises(experiment_pipeline.PipelineRegistrationRecoveryError, match="reconciled on resume"):
        experiment_pipeline._reconcile_pipeline_jobs_planned_event(root, spec)


def test_pipeline_retry_planned_event_is_reconciled_after_append_failure(tmp_path: Path, monkeypatch):
    root = tmp_path / "workspace"
    root.mkdir()
    spec = _spec(root)
    attempt = {"job_id": spec["jobs"][0]["id"], "attempt": 2}
    original_append = experiment_pipeline.append_event
    monkeypatch.setattr(
        experiment_pipeline,
        "append_event",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("event append interrupted")),
    )

    with pytest.raises(experiment_pipeline.PipelineRegistrationRecoveryError, match="reconciled on resume"):
        experiment_pipeline._reconcile_pipeline_retry_planned_event(root, spec, attempt)

    monkeypatch.setattr(experiment_pipeline, "append_event", original_append)
    experiment_pipeline._reconcile_pipeline_retry_planned_event(root, spec, attempt)
    experiment_pipeline._reconcile_pipeline_retry_planned_event(root, spec, attempt)

    events = [
        event
        for event in experiment_pipeline.read_experiment_events(root)
        if event.get("event_type") == "pipeline_job_retry_planned"
    ]
    assert len(events) == 1
    assert events[0]["pipeline_id"] == spec["pipeline"]["id"]
    assert events[0]["job_id"] == attempt["job_id"]
    assert events[0]["attempt"] == 2


def test_pipeline_registration_recovery_error_does_not_mark_pipeline_failed(tmp_path: Path, monkeypatch):
    root = tmp_path / "workspace"
    pipeline_dir = root / "pipelines" / "external-v1"
    pipeline_dir.mkdir(parents=True)
    (pipeline_dir / "pipeline.json").write_text(json.dumps({"status": "ready"}) + "\n")
    spec_path = tmp_path / "external.yaml"
    spec_path.write_text(yaml.safe_dump(_spec(root), sort_keys=False))
    monkeypatch.setattr(
        experiment_pipeline.artifacts,
        "read_hparam_plan",
        lambda *_args, **_kwargs: {"recipe": {"execution": {"target": "local"}}},
    )
    monkeypatch.setattr(experiment_pipeline, "_validate_experiment", lambda *_args, **_kwargs: {"status": "active"})
    monkeypatch.setattr(
        experiment_pipeline,
        "_validate_frozen_pipeline",
        lambda *_args, **_kwargs: {"status": "ready"},
    )
    monkeypatch.setattr(
        experiment_pipeline,
        "_execute_pipeline",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            experiment_pipeline.PipelineRegistrationRecoveryError("resume registration")
        ),
    )

    with pytest.raises(experiment_pipeline.PipelineRegistrationRecoveryError, match="resume registration"):
        experiment_pipeline.run_experiment_pipeline(
            root,
            spec_path,
            unlock_final_test=True,
            execute=True,
            resume=True,
        )

    assert json.loads((pipeline_dir / "pipeline.json").read_text())["status"] == "ready"


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


@pytest.mark.parametrize("prefixes", [["ema_model."], ["running_mean_model."]])
def test_schema_requires_both_model_averaging_prefixes(tmp_path: Path, prefixes: list[str]):
    spec = _spec(tmp_path)
    spec["checkpoint_policy"]["forbidden_state_dict_prefixes"] = prefixes

    with pytest.raises(ValueError, match="forbidden_state_dict_prefixes"):
        experiment_pipeline._validate_spec(spec, tmp_path, unlock_final_test=True)


def test_successful_source_accepts_no_test_after_fit_manifest(tmp_path: Path, monkeypatch):
    root = tmp_path / "workspace"
    spec = _spec(root)
    run = {"step_id": "train-age", "run_id": "run-000"}
    manifest_path = tmp_path / "run_manifest.json"
    manifest_path.write_text(json.dumps({"status": "skipped_test", "metrics": {"val_mae": 4.5}}) + "\n")

    monkeypatch.setattr(
        experiment_pipeline.artifacts,
        "read_hparam_plan",
        lambda _plan_dir: {"runs": [run]},
    )
    monkeypatch.setattr(
        experiment_pipeline,
        "read_run_manifest",
        lambda _root: [{**run, "status": "finished"}],
    )
    monkeypatch.setattr(experiment_pipeline.artifacts, "find_run_manifest", lambda _run: manifest_path)

    states = experiment_pipeline._inspect_sources(root, spec, refresh=False)

    assert states[0]["complete"] is True
    assert states[0]["failed_runs"] == []


@pytest.mark.parametrize("status", ["submitting", "unknown_scheduler"])
def test_slurm_source_uncertainty_blocks_external_pipeline(tmp_path: Path, monkeypatch, status: str):
    root = tmp_path / "workspace"
    spec = _spec(root)
    run = {"step_id": "train-age", "run_id": "run-000"}
    monkeypatch.setattr(experiment_pipeline.artifacts, "read_hparam_plan", lambda _plan_dir: {"runs": [run]})
    monkeypatch.setattr(experiment_pipeline, "read_run_manifest", lambda _root: [{**run, "status": status}])

    states = experiment_pipeline._inspect_sources(root, spec, refresh=False)

    assert states[0]["uncertain_runs"] == ["run-000"]
    assert experiment_pipeline._source_summary_status(states) == "blocked"


def test_retry_preflight_failure_does_not_block_independent_retry(tmp_path: Path, monkeypatch):
    root = tmp_path / "workspace"
    pipeline_dir = root / "pipelines" / "external-v1"
    pipeline_dir.mkdir(parents=True)
    spec = _spec(root)
    second_job = dict(spec["jobs"][0], id="age-hsp-i2-bcg", modality="bcg", num_workers=16)
    spec["jobs"].append(second_job)
    attempts = [{"job_id": job["id"], "attempt": 1, "status": "failed", "verified": "false"} for job in spec["jobs"]]
    order = []

    def attempt_recipe(_pipeline_dir, _spec, job, _selection, attempt):
        base = pipeline_dir / job["id"] / f"attempt-{attempt:03d}"
        return {"job": job["id"]}, base.with_suffix(".yaml"), base / "plan", base / "results"

    def retry_preflight(_pipeline_dir, job_id, _attempt, _recipe_path, _plan_dir):
        order.append(f"preflight:{job_id}")
        if job_id == "age-hsp-i2-psg":
            raise experiment_pipeline.RetryPreparationError("preflight failed")

    def materialize(_root, _spec, job, _selection, attempt, **_paths):
        order.append(f"materialize:{job['id']}")
        return {"job_id": job["id"], "attempt": attempt, "status": "planned", "verified": "false"}

    monkeypatch.setattr(experiment_pipeline, "_attempt_recipe", attempt_recipe)
    monkeypatch.setattr(experiment_pipeline, "_ensure_retry_preflight", retry_preflight)
    monkeypatch.setattr(
        experiment_pipeline,
        "_prepare_attempt_registration_groups",
        lambda _root, _spec, items, **_kwargs: {item[0]["id"]: None for item in items},
    )
    monkeypatch.setattr(experiment_pipeline, "_materialize_attempt", materialize)
    monkeypatch.setattr(experiment_pipeline, "append_event", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(experiment_pipeline, "_reconcile_pipeline_retry_planned_event", lambda *_args: None)
    monkeypatch.setattr(experiment_pipeline, "read_run_manifest", lambda _root: [])

    updated, created = experiment_pipeline._create_needed_retries(
        root,
        pipeline_dir,
        spec,
        {"age": {"variant": "sleep2vec2"}},
        attempts,
    )

    assert created is True
    assert order == [
        "preflight:age-hsp-i2-psg",
        "preflight:age-hsp-i2-bcg",
        "materialize:age-hsp-i2-bcg",
    ]
    assert updated[0]["retry_preparation_error"] == "preflight failed"
    assert [row["attempt"] for row in updated if row["job_id"] == "age-hsp-i2-bcg"] == [1, 2]
    assert [job["status"] for job in experiment_pipeline._logical_job_states(spec, updated)] == [
        "failed",
        "running",
    ]


def test_retry_registration_preflight_failure_does_not_block_independent_retry(tmp_path: Path, monkeypatch):
    root = tmp_path / "workspace"
    pipeline_dir = root / "pipelines" / "external-v1"
    pipeline_dir.mkdir(parents=True)
    spec = _spec(root)
    second_job = dict(spec["jobs"][0], id="age-hsp-i2-bcg", modality="bcg", num_workers=16)
    spec["jobs"].append(second_job)
    attempts = [{"job_id": job["id"], "attempt": 1, "status": "failed", "verified": "false"} for job in spec["jobs"]]
    order = []

    def attempt_recipe(_pipeline_dir, _spec, job, _selection, attempt):
        base = pipeline_dir / job["id"] / f"attempt-{attempt:03d}"
        return {"job": job["id"]}, base.with_suffix(".yaml"), base / "plan", base / "results"

    def prepare_registration(_root, _spec, items, **_kwargs):
        job_id = items[0][0]["id"]
        order.append(f"prepare:{job_id}")
        if job_id == "age-hsp-i2-psg":
            raise experiment_pipeline.AttemptRegistrationPreflightError("target argv rejected")
        return {job_id: None}

    def materialize(_root, _spec, job, _selection, attempt, **_paths):
        order.append(f"materialize:{job['id']}")
        return {"job_id": job["id"], "attempt": attempt, "status": "planned", "verified": "false"}

    monkeypatch.setattr(experiment_pipeline, "_attempt_recipe", attempt_recipe)
    monkeypatch.setattr(experiment_pipeline, "_ensure_retry_preflight", lambda *_args: None)
    monkeypatch.setattr(experiment_pipeline, "_prepare_attempt_registration_groups", prepare_registration)
    monkeypatch.setattr(experiment_pipeline, "_materialize_attempt", materialize)
    monkeypatch.setattr(experiment_pipeline, "append_event", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(experiment_pipeline, "_reconcile_pipeline_retry_planned_event", lambda *_args: None)
    monkeypatch.setattr(experiment_pipeline, "read_run_manifest", lambda _root: [])

    updated, created = experiment_pipeline._create_needed_retries(
        root,
        pipeline_dir,
        spec,
        {"age": {"variant": "sleep2vec2"}},
        attempts,
    )

    assert created is True
    assert order == [
        "prepare:age-hsp-i2-psg",
        "prepare:age-hsp-i2-bcg",
        "materialize:age-hsp-i2-bcg",
    ]
    assert updated[0]["retry_preparation_error"] == "target argv rejected"
    assert [row["attempt"] for row in updated if row["job_id"] == "age-hsp-i2-psg"] == [1]
    assert [row["attempt"] for row in updated if row["job_id"] == "age-hsp-i2-bcg"] == [1, 2]


def test_retry_registration_failure_is_not_recorded_as_preflight_failure(tmp_path: Path, monkeypatch):
    root = tmp_path / "workspace"
    pipeline_dir = root / "pipelines" / "external-v1"
    pipeline_dir.mkdir(parents=True)
    spec = _spec(root)
    attempts = [{"job_id": spec["jobs"][0]["id"], "attempt": 1, "status": "failed", "verified": "false"}]

    def attempt_recipe(_pipeline_dir, _spec, job, _selection, attempt):
        base = pipeline_dir / job["id"] / f"attempt-{attempt:03d}"
        return {"job": job["id"]}, base.with_suffix(".yaml"), base / "plan", base / "results"

    monkeypatch.setattr(experiment_pipeline, "_attempt_recipe", attempt_recipe)
    monkeypatch.setattr(experiment_pipeline, "_ensure_retry_preflight", lambda *_args: None)
    monkeypatch.setattr(
        experiment_pipeline,
        "_prepare_attempt_registration_groups",
        lambda _root, _spec, items, **_kwargs: {items[0][0]["id"]: None},
    )
    monkeypatch.setattr(
        experiment_pipeline,
        "_materialize_attempt",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("canonical commit failed")),
    )
    monkeypatch.setattr(experiment_pipeline, "read_run_manifest", lambda _root: [])

    with pytest.raises(RuntimeError, match="canonical commit failed"):
        experiment_pipeline._create_needed_retries(
            root,
            pipeline_dir,
            spec,
            {"age": {"variant": "sleep2vec2"}},
            attempts,
        )

    assert "retry_preparation_error" not in attempts[0]
    assert len(attempts) == 1


def test_retry_jobs_projection_failure_is_recoverable(tmp_path: Path, monkeypatch):
    root = tmp_path / "workspace"
    pipeline_dir = root / "pipelines" / "external-v1"
    pipeline_dir.mkdir(parents=True)
    spec = _spec(root)
    attempts = [{"job_id": spec["jobs"][0]["id"], "attempt": 1, "status": "failed", "verified": "false"}]

    def attempt_recipe(_pipeline_dir, _spec, job, _selection, attempt):
        base = pipeline_dir / job["id"] / f"attempt-{attempt:03d}"
        return {"job": job["id"]}, base.with_suffix(".yaml"), base / "plan", base / "results"

    monkeypatch.setattr(experiment_pipeline, "_attempt_recipe", attempt_recipe)
    monkeypatch.setattr(experiment_pipeline, "_ensure_retry_preflight", lambda *_args: None)
    monkeypatch.setattr(
        experiment_pipeline,
        "_prepare_attempt_registration_groups",
        lambda _root, _spec, items, **_kwargs: {items[0][0]["id"]: None},
    )
    monkeypatch.setattr(
        experiment_pipeline,
        "_materialize_attempt",
        lambda _root, _spec, job, _selection, attempt, **_paths: {
            "job_id": job["id"],
            "attempt": attempt,
            "status": "planned",
            "verified": "false",
        },
    )
    monkeypatch.setattr(
        experiment_pipeline,
        "_write_jobs",
        lambda *_args: (_ for _ in ()).throw(OSError("jobs projection interrupted")),
    )
    monkeypatch.setattr(experiment_pipeline, "read_run_manifest", lambda _root: [])

    with pytest.raises(experiment_pipeline.PipelineRegistrationRecoveryError, match="reconciled on resume"):
        experiment_pipeline._create_needed_retries(
            root,
            pipeline_dir,
            spec,
            {"age": {"variant": "sleep2vec2"}},
            attempts,
        )

    assert [int(row["attempt"]) for row in attempts] == [1, 2]


@pytest.mark.parametrize("failure_kind", ["topology", "target"])
def test_initial_registration_preflight_groups_variants_before_publishing_any_attempt(
    tmp_path: Path,
    monkeypatch,
    failure_kind: str,
):
    root = tmp_path / "workspace"
    root.mkdir()
    experiment = {
        "id": "unit",
        "title": "Unit",
        "objective": "Reject the external matrix before registration.",
        "root": str(root),
        "baseline": {"type": "none"},
        "status": "active",
    }
    (root / "experiment.yaml").write_text(yaml.safe_dump({"experiment": experiment}, sort_keys=False))
    (root / "run_manifest.tsv").write_text("step_id\trun_id\n")
    pipeline_dir = root / "pipelines" / "external-v1"
    pipeline_dir.mkdir(parents=True)

    spec = _spec(root)
    second = dict(spec["jobs"][0], id="age-hsp-i2-bcg", modality="bcg")
    third = dict(
        spec["jobs"][0],
        id="age-hsp-i2-ecg",
        checkpoint_source="age-root",
        modality="ecg",
        variant="sleep2vec",
    )
    spec["jobs"].extend([second, third])
    selection_fields = {
        "config": str(tmp_path / "config.yaml"),
        "checkpoint": str(tmp_path / "model.ckpt"),
        "label_name": "age",
    }
    selections = {
        "age": {**selection_fields, "variant": "sleep2vec2", "config_sha256": "2" * 64},
        "age-root": {**selection_fields, "variant": "sleep2vec", "config_sha256": "1" * 64},
    }

    def build_staged_plan(*, recipe_path, output_dir, staging_dir, run_index_offset, **_kwargs):
        recipe = yaml.safe_load(Path(recipe_path).read_text())
        job_id = recipe["name"].split("__")[1]
        run_id = f"run-{run_index_offset:03d}"
        module = "sleep2vec2.infer" if recipe["variant"] == "sleep2vec2" else "sleep2vec.infer"
        command = f"/runtime/python -m {module} --config frozen.yaml"
        semantic_script = Path(output_dir) / "runs" / f"{run_id}--{job_id}" / "launch.sh"
        physical_script = Path(staging_dir) / semantic_script.relative_to(output_dir)
        physical_script.parent.mkdir(parents=True)
        physical_script.write_text(command + "\n")
        (Path(staging_dir) / "plan.json").write_text(
            json.dumps(
                {
                    "runs": [
                        {
                            "step_id": "external-evaluate",
                            "run_id": run_id,
                            "run_name": job_id,
                            "script": str(semantic_script),
                            "command": command,
                        }
                    ]
                }
            )
            + "\n"
        )
        return SimpleNamespace(exit_code=0)

    target_calls = []
    topology_calls = []

    def reject_unsafe_group(root_path, paths, *, remote=None):
        if [Path(path) for path in paths] == [root.parent / ".workspace.plan-registration.lock"]:
            assert Path(root_path) == root.parent
            assert remote is None
            return
        assert Path(root_path) == Path("/")
        assert remote is None
        topology_calls.append([Path(path) for path in paths])
        if failure_kind == "topology" and len(paths) == 5:
            raise ValueError("frozen output topology rejected")

    def reject_second_group(_execution, runs, *, plan_label):
        assert plan_label == "pipeline"
        assert all(Path(run["script"]).is_file() for run in runs)
        target_calls.append([run["run_id"] for run in runs])
        if failure_kind == "target" and len(runs) == 2:
            raise ValueError("frozen argv rejected")
        return {"runtime_commit": "a" * 40}

    monkeypatch.setattr(experiment_pipeline, "_ensure_initial_preflight", lambda *_args: None)
    monkeypatch.setattr(experiment_pipeline, "build_plan", build_staged_plan)
    monkeypatch.setattr(experiment_pipeline.exp_io, "validate_managed_output_paths", reject_unsafe_group)
    monkeypatch.setattr(experiment_pipeline.managed_scheduler, "inspect_execution_target", reject_second_group)

    expected_error = "frozen output topology rejected" if failure_kind == "topology" else "frozen argv rejected"
    with pytest.raises(experiment_pipeline.AttemptRegistrationPreflightError, match=expected_error):
        experiment_pipeline._load_or_create_initial_attempts(root, pipeline_dir, spec, selections)

    assert [len(paths) for paths in topology_calls] == [3, 5]
    expected_target_calls = [["run-002"]] if failure_kind == "topology" else [["run-002"], ["run-000", "run-001"]]
    assert target_calls == expected_target_calls
    assert not (pipeline_dir / "jobs.tsv").exists()
    assert not (root / "steps").exists()
    assert read_run_manifest(root) == []
    assert not list((pipeline_dir / "plans").rglob("attempt-001"))
    assert not list(pipeline_dir.rglob("*.staging"))
    assert not list(pipeline_dir.rglob(managed_scheduler.EXECUTION_SNAPSHOT_NAME))


def test_registration_preflight_freezes_complete_group_and_rejects_drift(tmp_path: Path, monkeypatch):
    root = tmp_path / "workspace"
    pipeline_dir = root / "pipelines" / "external-v1"
    pipeline_dir.mkdir(parents=True)
    spec = _spec(root)
    second = dict(spec["jobs"][0], id="age-hsp-i2-bcg", modality="bcg")
    spec["jobs"].append(second)
    selection = {"variant": "sleep2vec2", "config_sha256": "2" * 64}
    attempts = []
    for job in spec["jobs"]:
        recipe_path = pipeline_dir / "recipes" / job["id"] / "attempt-001.yaml"
        plan_dir = pipeline_dir / "plans" / job["id"] / "attempt-001"
        result_root = pipeline_dir / "results" / job["id"] / "attempt-001"
        recipe_path.parent.mkdir(parents=True, exist_ok=True)
        recipe_path.write_text("task: infer\n")
        attempts.append((job, selection, 1, recipe_path, plan_dir, result_root))

    first_plan_dir = attempts[0][4]
    first_script = first_plan_dir / "runs" / "run-000--first" / "launch.sh"
    first_script.parent.mkdir(parents=True)
    first_script.write_text("/runtime/python -m sleep2vec2.infer --config first.yaml\n")
    (first_plan_dir / "plan.json").write_text(
        json.dumps(
            {
                "runs": [
                    {
                        "step_id": "external-evaluate",
                        "run_id": "run-000",
                        "script": str(first_script),
                        "command": first_script.read_text().strip(),
                    }
                ]
            }
        )
        + "\n"
    )

    stage_count = 0

    def prepare_plan(_job_id, _selection, _recipe_path, plan_dir, *, run_index_offset):
        nonlocal stage_count
        stage_count += 1
        staging_dir = plan_dir.parent / f".{plan_dir.name}.{stage_count}.staging"
        run_id = f"run-{run_index_offset:03d}"
        script = staging_dir / "runs" / f"{run_id}--pending" / "launch.sh"
        script.parent.mkdir(parents=True)
        command = "/runtime/python -m sleep2vec2.infer --config pending.yaml"
        script.write_text(command + "\n")
        (staging_dir / "plan.json").write_text(
            json.dumps(
                {
                    "runs": [
                        {
                            "step_id": "external-evaluate",
                            "run_id": run_id,
                            "script": str(plan_dir / script.relative_to(staging_dir)),
                            "command": command,
                        }
                    ]
                }
            )
            + "\n"
        )
        return staging_dir, staging_dir

    target_snapshot = {"validated_argv_sha256": "a" * 64}

    def inspect(_execution, runs, *, plan_label):
        assert plan_label == "pipeline"
        assert [run["run_id"] for run in runs] == ["run-000", "run-001"]
        assert all(Path(run["script"]).is_file() for run in runs)
        return dict(target_snapshot)

    monkeypatch.setattr(
        experiment_pipeline,
        "read_run_manifest",
        lambda _root: [{"step_id": "external-evaluate", "run_id": "run-000"}],
    )
    monkeypatch.setattr(experiment_pipeline, "next_run_index", lambda _recipe: 1)
    monkeypatch.setattr(experiment_pipeline, "_validate_new_attempt_paths", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(experiment_pipeline, "_prepare_attempt_plan", prepare_plan)
    monkeypatch.setattr(experiment_pipeline.exp_io, "validate_managed_output_paths", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(experiment_pipeline.managed_scheduler, "inspect_execution_target", inspect)

    snapshot_owner = pipeline_dir / "initial_schedulers" / "sleep2vec2"
    assert not snapshot_owner.exists()
    prepared = experiment_pipeline._prepare_attempt_registration_groups(
        root,
        spec,
        attempts,
        snapshot_owner_dirs={"sleep2vec2": snapshot_owner},
    )

    snapshot_path = snapshot_owner / managed_scheduler.EXECUTION_SNAPSHOT_NAME
    assert prepared[spec["jobs"][0]["id"]] == first_plan_dir
    assert prepared[second["id"]] != attempts[1][4]
    assert stage_count == 1
    assert json.loads(snapshot_path.read_text()) == target_snapshot
    shutil.rmtree(prepared[second["id"]])

    target_snapshot["validated_argv_sha256"] = "b" * 64
    with pytest.raises(experiment_pipeline.AttemptRegistrationPreflightError, match="snapshot changed"):
        experiment_pipeline._prepare_attempt_registration_groups(
            root,
            spec,
            attempts,
            snapshot_owner_dirs={"sleep2vec2": snapshot_owner},
        )

    assert json.loads(snapshot_path.read_text()) == {"validated_argv_sha256": "a" * 64}
    assert not list(pipeline_dir.rglob("*.staging"))


@pytest.mark.parametrize(
    "identity_error",
    [
        "PID 123 was reused by a different process.",
        "Canonical run has partial process identity; missing: process_start_token",
    ],
)
def test_unsafe_process_identity_is_blocked_and_never_retried(
    tmp_path: Path,
    monkeypatch,
    identity_error: str,
):
    root = tmp_path / "workspace"
    pipeline_dir = root / "pipelines" / "external-v1"
    pipeline_dir.mkdir(parents=True)
    attempt = {
        "experiment_id": "unit",
        "step_id": "external-evaluate",
        "run_id": "run-001",
        "job_id": "age-hsp-i2-psg",
        "attempt": 1,
        "status": "failed",
        "verified": "false",
    }
    write_rows(root / "run_manifest.tsv", [{**attempt, "process_identity_error": identity_error}])
    monkeypatch.setattr(
        experiment_pipeline,
        "_attempt_recipe",
        lambda *_args, **_kwargs: pytest.fail("unsafe process identity must not be retried"),
    )
    monkeypatch.setattr(experiment_pipeline, "append_event", lambda *_args, **_kwargs: None)

    updated, created = experiment_pipeline._create_needed_retries(
        root,
        pipeline_dir,
        _spec(root),
        {"age": {}},
        [attempt],
    )

    assert created is False
    assert updated[0]["retry_blocker"] == f"unsafe process identity: {identity_error}"
    assert experiment_pipeline._logical_job_states(_spec(root), updated)[0]["status"] == "blocked"


def test_atomic_generic_plan_freezes_single_runtime_command(tmp_path: Path, monkeypatch):
    source = tmp_path / "source"
    recipe_path = write_finetune_recipe(source, variant="sleep2vec2")
    recipe = yaml.safe_load(recipe_path.read_text())
    workspace = tmp_path / "workspace"
    recipe["task"] = "infer"
    recipe["experiment"]["root"] = str(workspace)
    recipe["step"] = {
        "id": "external-evaluate",
        "phase": "evaluate",
        "purpose": "Exercise atomic external planning.",
    }
    recipe["execution"] = {
        "target": "local",
        "workdir": "/runtime/snapshot",
        "python": "/runtime/python",
        "runtime_commit": "a" * 40,
    }
    recipe_path.write_text(yaml.safe_dump(recipe, sort_keys=False))
    config_bytes = Path(recipe["inputs"]["config"]).read_bytes()
    bound_config = {
        "_source_config_bytes": config_bytes,
        "_source_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
    }
    plan_contract.bind_plan_context(recipe)
    report = plans.DecisionReport(status=plans.DecisionStatus.PASS, issues=[], decisions={})
    command = "/runtime/python -m sleep2vec2.infer --config frozen.yaml"
    monkeypatch.setattr(plans, "preflight_plan", lambda **_kwargs: (recipe, bound_config, report))
    monkeypatch.setattr(plans, "config_summary", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(plans, "_commands_for_recipe", lambda *_args, **_kwargs: [command])
    monkeypatch.setattr(plans.get_adapter(recipe["task"]), "frozen_commands", lambda *_args, **_kwargs: [command])
    plan_dir = workspace / "plans" / "attempt-001"
    staging_dir = workspace / "plans" / ".attempt-001.staging"

    result = plans.build_plan(
        recipe_path=recipe_path,
        output_dir=plan_dir,
        staging_dir=staging_dir,
    )

    assert result.exit_code == 0
    assert plan_dir.is_dir()
    assert not staging_dir.exists()
    plan = json.loads((plan_dir / "plan.json").read_text())
    planned = plan["runs"][0]
    assert planned["command"] == command
    script_lines = Path(planned["script"]).read_text().splitlines()
    assert command in script_lines
    helper_index = script_lines.index("_agent_commit_status() {")
    running_index = script_lines.index("_agent_commit_status running")
    command_index = script_lines.index(command)
    assert script_lines[helper_index + 1].startswith("  /runtime/python -c ")
    assert any(line.startswith("/runtime/python -c ") and "a" * 40 in line for line in script_lines)
    assert helper_index < running_index < command_index
    assert plan["recipe"]["execution"] == recipe["execution"]
    canonical = read_run_manifest(workspace)[0]
    assert canonical.get("command") in (None, "")
    experiment_pipeline._validate_attempt_plan(
        {
            "step_id": planned["step_id"],
            "run_id": planned["run_id"],
            "recipe": str(recipe_path),
            "plan_dir": str(plan_dir),
        },
        canonical,
    )

    def inspect_command(_execution, probe):
        if probe[2] == python_programs.source("managed_scheduler.runtime_identity"):
            payload = {
                "python": "/runtime/python",
                "python_version": "3.12",
                "runtime_commit": "a" * 40,
                "runtime_repo_root": "/runtime/snapshot",
                "runtime_hostname": "unit-host",
                "module": "sleep2vec2.infer",
                "module_origin": "/runtime/snapshot/sleep2vec2/infer.py",
            }
            return SimpleNamespace(returncode=0, stdout=json.dumps(payload), stderr="")
        evidence = {"supported_options": ["--config"], "cli_options_sha256": "cli-digest"}
        return SimpleNamespace(
            returncode=0,
            stdout="AGENT_CLI_PREFLIGHT=" + json.dumps(evidence) + "\n",
            stderr="",
        )

    snapshot = managed_scheduler.inspect_execution_target(
        {
            "target": "local",
            "workdir": "/runtime/snapshot",
            "python": "/runtime/python",
            "runtime_commit": "a" * 40,
        },
        [planned],
        command_runner=inspect_command,
    )
    assert snapshot["module"] == "sleep2vec2.infer"
    assert snapshot["required_options"] == ["--config"]


@pytest.mark.parametrize(
    "outcome",
    ["success", "prepared_public", "staging_tamper", "tamper", "interrupt_after_commit"],
)
def test_uncommitted_attempt_plan_is_deterministically_validated(
    tmp_path: Path,
    monkeypatch,
    outcome: str,
):
    root = tmp_path / "workspace"
    root.mkdir()
    experiment = {
        "id": "unit",
        "title": "Unit",
        "objective": "Exercise crash-safe external planning.",
        "root": str(root),
        "baseline": {"type": "none"},
        "status": "active",
    }
    (root / "experiment.yaml").write_text(yaml.safe_dump({"experiment": experiment}, sort_keys=False))
    (root / "run_manifest.tsv").write_text("step_id\trun_id\n")

    source_recipe = yaml.safe_load(write_finetune_recipe(tmp_path / "source", variant="sleep2vec2").read_text())
    config = Path(source_recipe["inputs"]["config"])
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    spec = _spec(root)
    preset = Path(spec["jobs"][0]["inference_preset_path"])
    preset.parent.mkdir(parents=True)
    preset.write_bytes(b"preset")
    selection = {
        "source_id": "age",
        "config": str(config),
        "config_sha256": file_sha256(config),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "variant": "sleep2vec2",
        "label_name": "age",
    }
    pipeline_dir = root / "pipelines" / "external-v1"
    recipe, recipe_path, plan_dir, result_root = experiment_pipeline._attempt_recipe(
        pipeline_dir,
        spec,
        spec["jobs"][0],
        selection,
        1,
    )
    recipe_path.parent.mkdir(parents=True)
    recipe_path.write_text(yaml.safe_dump(recipe, sort_keys=False))
    staging_dir = plan_dir.parent / ".attempt-001.crash-window"
    commit_step_manifest(
        root,
        {
            "step": spec["pipeline"]["step"],
            "experiment_id": experiment["id"],
            "plan_controller": "pipeline",
            "recipe_path": "",
            "plans": [],
        },
    )

    report = plans.build_plan(
        recipe_path=recipe_path,
        output_dir=plan_dir,
        unlock_final_test=True,
        staging_dir=staging_dir,
        defer_commit=True,
        plan_controller="pipeline",
    )
    assert report.exit_code == 0
    plan_dir.parent.mkdir(parents=True, exist_ok=True)
    step_manifest = root / "steps" / spec["pipeline"]["step"]["id"] / "step.yaml"
    assert yaml.safe_load(step_manifest.read_text())["plans"] == []
    assert read_run_manifest(root) == []
    frozen_plan = json.loads((staging_dir / "plan.json").read_text())
    if outcome == "staging_tamper":
        semantic_launch = Path(frozen_plan["runs"][0]["script"])
        physical_launch = staging_dir / semantic_launch.relative_to(plan_dir)
        physical_launch.write_text("tampered\n")
        with pytest.raises(ValueError, match="attempt script changed"):
            experiment_pipeline._materialize_attempt(
                root,
                spec,
                spec["jobs"][0],
                selection,
                1,
                recipe_path=recipe_path,
                plan_dir=plan_dir,
                result_root=result_root,
                prepared_plan_dir=staging_dir,
            )
        assert not plan_dir.exists()
        assert read_run_manifest(root) == []
        assert yaml.safe_load(step_manifest.read_text())["plans"] == []
        return

    staging_dir.replace(plan_dir)
    frozen_identity = {
        "target": "local",
        "workdir": spec["runtime"]["workdir"],
        "python": spec["runtime"]["python"],
        "runtime_commit": spec["runtime"]["runtime_commit"],
    }
    assert recipe["execution"] == frozen_identity
    assert frozen_plan["recipe"]["execution"] == frozen_identity
    assert yaml.safe_load((plan_dir / "recipe.resolved.yaml").read_text())["execution"] == frozen_identity
    launch_path = Path(frozen_plan["runs"][0]["script"])
    launch_before = launch_path.read_bytes()
    launch_lines = launch_before.decode().splitlines()
    helper_index = launch_lines.index("_agent_commit_status() {")
    assert launch_lines[helper_index + 1].startswith(f"  {spec['runtime']['python']} -c ")

    if outcome == "tamper":
        (plan_dir / "plan.md").write_text("tampered\n")
        with pytest.raises(ValueError, match="differs from deterministic regeneration"):
            experiment_pipeline._materialize_attempt(
                root,
                spec,
                spec["jobs"][0],
                selection,
                1,
                recipe_path=recipe_path,
                plan_dir=plan_dir,
                result_root=result_root,
            )
        assert read_run_manifest(root) == []
        assert yaml.safe_load(step_manifest.read_text())["plans"] == []
        return

    if outcome == "interrupt_after_commit":
        real_merge = experiment_pipeline.merge_run_manifest

        def merge_then_interrupt(*args, **kwargs):
            real_merge(*args, **kwargs)
            raise RuntimeError("simulated interruption after canonical commit")

        monkeypatch.setattr(experiment_pipeline, "merge_run_manifest", merge_then_interrupt)
        with pytest.raises(RuntimeError, match="simulated interruption"):
            experiment_pipeline._materialize_attempt(
                root,
                spec,
                spec["jobs"][0],
                selection,
                1,
                recipe_path=recipe_path,
                plan_dir=plan_dir,
                result_root=result_root,
            )

        canonical = read_run_manifest(root)
        assert canonical[0]["pipeline_id"] == "external-v1"
        assert canonical[0]["terminal_status_owner"] == "script"
        workspace = yaml.safe_load((root / "experiment.yaml").read_text())
        workspace["experiment"].pop("status")
        (root / "experiment.yaml").write_text(yaml.safe_dump(workspace, sort_keys=False))
        snapshot = experiments.experiment_status(root)
        assert snapshot["decision"]["recommended_next"] is None
        assert snapshot["decision"]["other_legal_actions"] == []
        assert snapshot["decision"]["blocked_actions"] == ["finalize", "pipeline_advance"]

        write_rows(root / "run_manifest.tsv", [{**canonical[0], "status": "failed"}])
        terminal = experiments.experiment_status(root)
        assert terminal["decision"]["recommended_next"] is None
        assert terminal["decision"]["other_legal_actions"] == []
        assert terminal["decision"]["blocked_actions"] == ["finalize", "pipeline_advance"]
        return

    original_prepare = experiment_pipeline._prepare_attempt_plan
    if outcome == "prepared_public":
        monkeypatch.setattr(
            experiment_pipeline,
            "_prepare_attempt_plan",
            lambda *_args, **_kwargs: pytest.fail("validated public plan must not be rebuilt"),
        )
    row = experiment_pipeline._materialize_attempt(
        root,
        spec,
        spec["jobs"][0],
        selection,
        1,
        recipe_path=recipe_path,
        plan_dir=plan_dir,
        result_root=result_root,
        prepared_plan_dir=plan_dir if outcome == "prepared_public" else None,
    )

    canonical = read_run_manifest(root)
    assert len(canonical) == 1
    assert row["job_id"] == "age-hsp-i2-psg"
    assert canonical[0]["pipeline_id"] == "external-v1"
    assert canonical[0]["terminal_status_owner"] == "script"
    step_payload = yaml.safe_load(step_manifest.read_text())
    assert step_payload["plan_controller"] == "pipeline"
    assert step_payload["plans"] == [str(plan_dir.resolve())]
    assert launch_path.read_bytes() == launch_before
    assert not list(plan_dir.parent.glob(".attempt-001.*.staging"))

    ownership_fields = {"pipeline_id", "job_id", "attempt", "result_root", "terminal_status_owner"}
    write_rows(
        root / "run_manifest.tsv",
        [{key: value for key, value in canonical[0].items() if key not in ownership_fields}],
    )

    monkeypatch.setattr(experiment_pipeline, "_prepare_attempt_plan", original_prepare)
    experiment_pipeline._materialize_attempt(
        root,
        spec,
        spec["jobs"][0],
        selection,
        1,
        recipe_path=recipe_path,
        plan_dir=plan_dir,
        result_root=result_root,
    )

    repaired = read_run_manifest(root)[0]
    assert repaired["pipeline_id"] == "external-v1"
    assert repaired["job_id"] == "age-hsp-i2-psg"
    assert repaired["attempt"] == "1"
    assert repaired["result_root"] == str(result_root)
    assert repaired["terminal_status_owner"] == "script"


def test_attempt_config_drift_fails_before_plan_publication(tmp_path: Path):
    root = tmp_path / "workspace"
    root.mkdir()
    experiment = {
        "id": "unit",
        "title": "Unit",
        "objective": "Reject attempt config drift before publication.",
        "root": str(root),
        "baseline": {"type": "none"},
        "status": "active",
    }
    (root / "experiment.yaml").write_text(yaml.safe_dump({"experiment": experiment}, sort_keys=False))
    (root / "run_manifest.tsv").write_text("step_id\trun_id\n")

    source_recipe = yaml.safe_load(write_finetune_recipe(tmp_path / "source", variant="sleep2vec2").read_text())
    config = Path(source_recipe["inputs"]["config"])
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    spec = _spec(root)
    preset = Path(spec["jobs"][0]["inference_preset_path"])
    preset.parent.mkdir(parents=True)
    preset.write_bytes(b"preset")
    selection = {
        "source_id": "age",
        "config": str(config),
        "config_sha256": file_sha256(config),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "variant": "sleep2vec2",
        "label_name": "age",
    }
    pipeline_dir = root / "pipelines" / "external-v1"
    recipe, recipe_path, plan_dir, result_root = experiment_pipeline._attempt_recipe(
        pipeline_dir,
        spec,
        spec["jobs"][0],
        selection,
        1,
    )
    recipe_path.parent.mkdir(parents=True)
    recipe_path.write_text(yaml.safe_dump(recipe, sort_keys=False))
    config.write_text(config.read_text() + "\n# drifted after checkpoint selection\n")

    with pytest.raises(ValueError, match="externally bound SHA-256"):
        experiment_pipeline._materialize_attempt(
            root,
            spec,
            spec["jobs"][0],
            selection,
            1,
            recipe_path=recipe_path,
            plan_dir=plan_dir,
            result_root=result_root,
        )

    assert not plan_dir.exists()
    assert not list(plan_dir.parent.glob(f".{plan_dir.name}.*.staging"))
    assert not result_root.exists()
    assert read_run_manifest(root) == []


def test_jobs_exceeding_capacity_launch_only_available_gpu_slots(tmp_path: Path):
    root = tmp_path / "workspace"
    root.mkdir()
    experiment = {
        "id": "unit",
        "title": "Unit",
        "objective": "Exercise external scheduler capacity.",
        "root": str(root),
        "baseline": {"type": "none"},
        "status": "active",
    }
    (root / "experiment.yaml").write_text(yaml.safe_dump({"experiment": experiment}, sort_keys=False))
    owner_dir = root / "pipelines" / "external-v1"
    owner_dir.mkdir(parents=True)
    runs = []
    for index in range(9):
        run_id = f"run-{index:03d}"
        job_id = f"job-{index:02d}"
        run_dir = owner_dir / "plans" / job_id / "attempt-001" / "runs" / f"{run_id}--{job_id}"
        run_dir.mkdir(parents=True)
        config = run_dir / "config.yaml"
        script = run_dir / "launch.sh"
        artifacts_path = run_dir / "artifacts.json"
        config.write_text("model: unit\n")
        script.write_text("#!/usr/bin/env bash\ntrue\n")
        script.chmod(0o755)
        artifacts_path.write_text("{}\n")
        runs.append(
            {
                "experiment_id": "unit",
                "step_id": "external-evaluate",
                "run_id": run_id,
                "run_name": job_id,
                "version": job_id,
                "status": "planned",
                "parameter_summary": "single resolved recipe",
                "config": str(config),
                "config_sha256": file_sha256(config),
                "script": str(script),
                "script_sha256": file_sha256(script),
                "run_dir": str(run_dir),
                "artifacts": str(artifacts_path),
                "runtime_dir": "",
                "checkpoint_dir": "",
                "pipeline_id": "external-v1",
                "job_id": job_id,
                "attempt": 1,
                "result_root": str(owner_dir / "results" / job_id / "attempt-001"),
                "terminal_status_owner": "script",
            }
        )
    write_rows(root / "run_manifest.tsv", runs)
    built = []
    started = []

    def build_command(_execution, _script, _log_path, _pid_path, gpus, **_kwargs):
        command = f"gpu={','.join(str(gpu) for gpu in gpus)}"
        built.append(command)
        return command

    hooks = managed_scheduler.SchedulerHooks(
        validated_snapshot=lambda *_args, **_kwargs: (None, False),
        build_command=build_command,
        start_process=lambda _execution, command: started.append(command) or "launched",
    )
    result = managed_scheduler.launch_managed_runs(
        root,
        owner_dir,
        runs,
        {
            "target": "local",
            "workdir": str(root),
            "gpu_pool": list(range(8)),
            "gpus_per_run": 1,
            "max_concurrent": 8,
        },
        {"devices": [0]},
        dry_run=False,
        default_script_commits_terminal_status=True,
        runtime_output_fields=("result_root",),
        runtime_output_root=root,
        hooks=hooks,
    )

    assert started == [f"gpu={index}" for index in range(8)]
    assert built == started
    assert [row["status"] for row in result.committed_rows].count("launched") == 8
    assert [row["status"] for row in result.committed_rows].count("pending") == 1
    assert sorted(row["gpus"] for row in result.committed_rows if row["status"] == "launched") == [
        str(index) for index in range(8)
    ]


def test_run_attempts_waits_when_capacity_blocks_before_execution_snapshot(tmp_path: Path, monkeypatch):
    class WaitObserved(Exception):
        pass

    root = tmp_path / "workspace"
    pipeline_dir = root / "pipelines" / "external-v1"
    pipeline_dir.mkdir(parents=True)
    (pipeline_dir / "spec.source.yaml").write_text("schema_version: 1\n")
    plan_dir = pipeline_dir / "plans" / "age-hsp-i2-psg" / "attempt-001"
    attempt = {
        "step_id": "external-evaluate",
        "run_id": "run-001",
        "pipeline_id": "external-v1",
        "job_id": "age-hsp-i2-psg",
        "variant": "sleep2vec2",
        "attempt": 1,
        "status": "pending",
        "verified": "false",
        "plan_dir": str(plan_dir),
        "runtime_commit": "",
    }
    write_rows(pipeline_dir / "jobs.tsv", [attempt])

    monkeypatch.setattr(experiment_pipeline, "_validate_frozen_pipeline", lambda *_args: {})
    monkeypatch.setattr(experiment_pipeline, "_validate_attempt_rows", lambda *_args: None)
    monkeypatch.setattr(
        experiment_pipeline,
        "_planned_runs",
        lambda _rows: [{"step_id": "external-evaluate", "run_id": "run-001"}],
    )
    launches = []

    def capacity_blocked(*_args, **_kwargs):
        launches.append(True)
        return SimpleNamespace(committed_rows=[dict(attempt)])

    monkeypatch.setattr(experiment_pipeline.managed_scheduler, "launch_managed_runs", capacity_blocked)
    monkeypatch.setattr(experiment_pipeline, "read_run_manifest", lambda _root: [dict(attempt)])
    monkeypatch.setattr(experiment_pipeline.time, "sleep", lambda _seconds: (_ for _ in ()).throw(WaitObserved()))

    with pytest.raises(WaitObserved):
        experiment_pipeline._run_attempts(
            root,
            pipeline_dir,
            _spec(root),
            {"age": {}},
            [attempt],
            poll_seconds=1,
        )

    assert launches == [True]
    assert not (pipeline_dir / "execution_snapshot.json").exists()


def test_run_attempts_terminal_attempt_skips_live_snapshot_probe_and_verifies_result(tmp_path: Path, monkeypatch):
    root = tmp_path / "workspace"
    pipeline_dir = root / "pipelines" / "external-v1"
    pipeline_dir.mkdir(parents=True)
    (pipeline_dir / "spec.source.yaml").write_text("schema_version: 1\n")
    snapshot_path = pipeline_dir / managed_scheduler.EXECUTION_SNAPSHOT_NAME
    snapshot_path.write_text(json.dumps({"runtime_commit": "a" * 40}) + "\n")
    attempt = {
        "step_id": "external-evaluate",
        "run_id": "run-001",
        "pipeline_id": "external-v1",
        "job_id": "age-hsp-i2-psg",
        "variant": "sleep2vec2",
        "attempt": 1,
        "status": "completed",
        "verified": "false",
        "plan_dir": str(pipeline_dir / "plans" / "age-hsp-i2-psg" / "attempt-001"),
        "runtime_commit": "",
    }
    write_rows(pipeline_dir / "jobs.tsv", [attempt])
    validations = []
    launches = []

    monkeypatch.setattr(experiment_pipeline, "_validate_frozen_pipeline", lambda *_args: {})
    monkeypatch.setattr(
        experiment_pipeline,
        "_validate_attempt_rows",
        lambda *_args: validations.append("attempts"),
    )
    monkeypatch.setattr(
        experiment_pipeline,
        "_planned_runs",
        lambda _rows: [{"step_id": "external-evaluate", "run_id": "run-001"}],
    )
    monkeypatch.setattr(experiment_pipeline, "read_run_manifest", lambda _root: [dict(attempt)])
    monkeypatch.setattr(
        experiment_pipeline.managed_scheduler,
        "validated_execution_snapshot",
        lambda *_args, **_kwargs: pytest.fail("terminal attempts must not probe the live runtime"),
    )
    monkeypatch.setattr(
        experiment_pipeline.managed_scheduler,
        "launch_managed_runs",
        lambda *_args, **_kwargs: launches.append(True) or SimpleNamespace(committed_rows=[dict(attempt)]),
    )
    result_manifest = pipeline_dir / "result_manifest.json"
    monkeypatch.setattr(
        experiment_pipeline,
        "_validate_result_manifest",
        lambda *_args: validations.append("result") or result_manifest,
    )

    result = experiment_pipeline._run_attempts(
        root,
        pipeline_dir,
        _spec(root),
        {"age": {}},
        [attempt],
        poll_seconds=0,
    )

    persisted = experiment_pipeline.read_rows(pipeline_dir / "jobs.tsv")[0]
    assert result["status"] == "completed"
    assert launches == [True]
    assert validations == ["attempts", "result"]
    assert persisted["verified"] == "true"
    assert persisted["runtime_commit"] == "a" * 40


def test_run_attempts_result_validation_failure_is_terminal_without_changing_canonical_status(
    tmp_path: Path, monkeypatch
):
    root = tmp_path / "workspace"
    pipeline_dir = root / "pipelines" / "external-v1"
    pipeline_dir.mkdir(parents=True)
    (pipeline_dir / "spec.source.yaml").write_text("schema_version: 1\n")
    attempt = {
        "step_id": "external-evaluate",
        "run_id": "run-001",
        "pipeline_id": "external-v1",
        "job_id": "age-hsp-i2-psg",
        "variant": "sleep2vec2",
        "attempt": 1,
        "status": "completed",
        "verified": "false",
        "plan_dir": str(pipeline_dir / "plans" / "age-hsp-i2-psg" / "attempt-001"),
        "runtime_commit": "",
        "terminal_status_owner": "script",
    }
    write_rows(pipeline_dir / "jobs.tsv", [attempt])
    canonical = dict(attempt)
    validations = []

    monkeypatch.setattr(experiment_pipeline, "_validate_frozen_pipeline", lambda *_args: {})
    monkeypatch.setattr(experiment_pipeline, "_validate_attempt_rows", lambda *_args: None)
    monkeypatch.setattr(
        experiment_pipeline,
        "_planned_runs",
        lambda _rows: [{"step_id": "external-evaluate", "run_id": "run-001"}],
    )
    monkeypatch.setattr(experiment_pipeline, "read_run_manifest", lambda _root: [dict(canonical)])
    monkeypatch.setattr(experiment_pipeline.time, "sleep", lambda *_args: pytest.fail("must not poll"))
    monkeypatch.setattr(
        experiment_pipeline,
        "merge_run_manifest",
        lambda *_args, **_kwargs: pytest.fail("result verification must not change canonical lifecycle status"),
    )
    monkeypatch.setattr(
        experiment_pipeline.managed_scheduler,
        "launch_managed_runs",
        lambda *_args, **_kwargs: SimpleNamespace(committed_rows=[dict(canonical)]),
    )
    monkeypatch.setattr(
        experiment_pipeline,
        "_attempt_recipe",
        lambda *_args, **_kwargs: pytest.fail("result verification failures must not be retried"),
    )

    def reject_result(*_args):
        validations.append("result")
        raise ValueError("manifest invalid")

    monkeypatch.setattr(experiment_pipeline, "_validate_result_manifest", reject_result)

    result = experiment_pipeline._run_attempts(
        root,
        pipeline_dir,
        _spec(root),
        {"age": {}},
        [attempt],
        poll_seconds=0,
    )

    persisted = experiment_pipeline.read_rows(pipeline_dir / "jobs.tsv")[0]
    assert result["status"] == "failed"
    assert result["jobs"][0]["status"] == "failed"
    assert result["jobs"][0]["attempt_count"] == 1
    assert persisted["status"] == "completed"
    assert persisted["verified"] == "false"
    assert persisted["validation_error"] == "manifest invalid"
    assert validations == ["result"]

    monkeypatch.setattr(
        experiment_pipeline,
        "_validate_result_manifest",
        lambda *_args: pytest.fail("resume must preserve the terminal validation failure"),
    )
    resumed = experiment_pipeline._run_attempts(
        root,
        pipeline_dir,
        _spec(root),
        {"age": {}},
        [persisted],
        poll_seconds=0,
    )

    assert resumed["status"] == "failed"
    assert experiment_pipeline.read_rows(pipeline_dir / "jobs.tsv") == [persisted]


def test_run_attempts_mixed_group_validates_live_snapshot_before_launch(tmp_path: Path, monkeypatch):
    class LaunchObserved(Exception):
        pass

    root = tmp_path / "workspace"
    pipeline_dir = root / "pipelines" / "external-v1"
    pipeline_dir.mkdir(parents=True)
    (pipeline_dir / "spec.source.yaml").write_text("schema_version: 1\n")
    (pipeline_dir / managed_scheduler.EXECUTION_SNAPSHOT_NAME).write_text("{}\n")
    spec = _spec(root)
    spec["jobs"].append(
        {
            **spec["jobs"][0],
            "id": "age-hsp-i2-bcg",
            "modality": "bcg",
            "num_workers": 16,
        }
    )
    attempts = [
        {
            "step_id": "external-evaluate",
            "run_id": "run-001",
            "pipeline_id": "external-v1",
            "job_id": "age-hsp-i2-psg",
            "variant": "sleep2vec2",
            "attempt": 1,
            "status": "completed",
            "verified": "true",
            "plan_dir": str(pipeline_dir / "plans" / "age-hsp-i2-psg" / "attempt-001"),
            "runtime_commit": "a" * 40,
        },
        {
            "step_id": "external-evaluate",
            "run_id": "run-002",
            "pipeline_id": "external-v1",
            "job_id": "age-hsp-i2-bcg",
            "variant": "sleep2vec2",
            "attempt": 1,
            "status": "pending",
            "verified": "false",
            "plan_dir": str(pipeline_dir / "plans" / "age-hsp-i2-bcg" / "attempt-001"),
            "runtime_commit": "",
        },
    ]
    write_rows(pipeline_dir / "jobs.tsv", attempts)
    calls = []

    monkeypatch.setattr(experiment_pipeline, "_validate_frozen_pipeline", lambda *_args: {})
    monkeypatch.setattr(experiment_pipeline, "_validate_attempt_rows", lambda *_args: None)
    monkeypatch.setattr(
        experiment_pipeline,
        "_planned_runs",
        lambda rows: [{"step_id": row["step_id"], "run_id": row["run_id"]} for row in rows],
    )
    monkeypatch.setattr(experiment_pipeline, "read_run_manifest", lambda _root: [dict(row) for row in attempts])

    def validate_snapshot(owner_dir, _execution, runs, _canonical):
        assert owner_dir == pipeline_dir
        assert [run["run_id"] for run in runs] == ["run-001", "run-002"]
        calls.append("snapshot")

    def observe_launch(_root, owner_dir, runs, *_args, **_kwargs):
        assert owner_dir == pipeline_dir
        assert [run["run_id"] for run in runs] == ["run-001", "run-002"]
        calls.append("launch")
        raise LaunchObserved

    monkeypatch.setattr(experiment_pipeline.managed_scheduler, "validated_execution_snapshot", validate_snapshot)
    monkeypatch.setattr(experiment_pipeline.managed_scheduler, "launch_managed_runs", observe_launch)

    with pytest.raises(LaunchObserved):
        experiment_pipeline._run_attempts(
            root,
            pipeline_dir,
            spec,
            {"age": {}},
            attempts,
            poll_seconds=0,
        )

    assert calls == ["snapshot", "launch"]


def test_run_attempts_blocks_on_external_missing_pid_capacity_blocker(tmp_path: Path, monkeypatch):
    root = tmp_path / "workspace"
    pipeline_dir = root / "pipelines" / "external-v1"
    pipeline_dir.mkdir(parents=True)
    (pipeline_dir / "spec.source.yaml").write_text("schema_version: 1\n")
    spec = _spec(root)
    spec["jobs"].append(
        {
            **spec["jobs"][0],
            "id": "age-hsp-i2-bcg",
            "modality": "bcg",
            "num_workers": 16,
        }
    )
    attempts = [
        {
            "step_id": "external-evaluate",
            "run_id": "run-001",
            "pipeline_id": "external-v1",
            "job_id": "age-hsp-i2-psg",
            "variant": "sleep2vec",
            "attempt": 1,
            "status": "pending",
            "verified": "false",
            "plan_dir": str(pipeline_dir / "plans" / "age-hsp-i2-psg" / "attempt-001"),
            "runtime_commit": "",
        },
        {
            "step_id": "external-evaluate",
            "run_id": "run-002",
            "pipeline_id": "external-v1",
            "job_id": "age-hsp-i2-bcg",
            "variant": "sleep2vec2",
            "attempt": 1,
            "status": "pending",
            "verified": "false",
            "plan_dir": str(pipeline_dir / "plans" / "age-hsp-i2-bcg" / "attempt-001"),
            "runtime_commit": "",
        },
    ]
    write_rows(pipeline_dir / "jobs.tsv", attempts)
    blocker = {
        "step_id": "train-age",
        "run_id": "run-099",
        "status": "missing_pid",
        "target": "local",
        "gpus": "0",
    }

    monkeypatch.setattr(experiment_pipeline, "_validate_frozen_pipeline", lambda *_args: {})
    monkeypatch.setattr(experiment_pipeline, "_validate_attempt_rows", lambda *_args: None)
    monkeypatch.setattr(
        experiment_pipeline,
        "_planned_runs",
        lambda rows: [{"step_id": row["step_id"], "run_id": row["run_id"]} for row in rows],
    )
    monkeypatch.setattr(
        experiment_pipeline,
        "read_run_manifest",
        lambda _root: [*[dict(row) for row in attempts], dict(blocker)],
    )
    launches = []

    def blocked_launch(*_args, **kwargs):
        launches.append(kwargs["fail_on_missing_pid_blocker"])
        raise managed_scheduler.MissingPidCapacityError(blocker["step_id"], blocker["run_id"])

    monkeypatch.setattr(experiment_pipeline.managed_scheduler, "launch_managed_runs", blocked_launch)
    monkeypatch.setattr(
        experiment_pipeline,
        "_create_needed_retries",
        lambda *_args, **_kwargs: pytest.fail("a missing_pid capacity blocker must not create retries"),
    )
    monkeypatch.setattr(
        experiment_pipeline.time,
        "sleep",
        lambda *_args: pytest.fail("a missing_pid capacity blocker must not sleep"),
    )

    result = experiment_pipeline._run_attempts(
        root,
        pipeline_dir,
        spec,
        {"age": {}},
        attempts,
        poll_seconds=1,
    )

    assert result["status"] == "blocked"
    assert result["missing_pid_blocker"] == {
        "status": "missing_pid",
        "step_id": blocker["step_id"],
        "run_id": blocker["run_id"],
    }
    assert launches == [True]
    assert [row["status"] for row in experiment_pipeline.read_rows(pipeline_dir / "jobs.tsv")] == [
        "pending",
        "pending",
    ]
    assert not list(pipeline_dir.rglob(managed_scheduler.EXECUTION_SNAPSHOT_NAME))


def test_run_attempts_blocks_before_retry_when_external_run_has_missing_pid(tmp_path: Path, monkeypatch):
    root = tmp_path / "workspace"
    pipeline_dir = root / "pipelines" / "external-v1"
    pipeline_dir.mkdir(parents=True)
    (pipeline_dir / "spec.source.yaml").write_text("schema_version: 1\n")
    attempt = {
        "step_id": "external-evaluate",
        "run_id": "run-001",
        "pipeline_id": "external-v1",
        "job_id": "age-hsp-i2-psg",
        "variant": "sleep2vec2",
        "attempt": 1,
        "status": "failed",
        "verified": "false",
        "plan_dir": str(pipeline_dir / "plans" / "age-hsp-i2-psg" / "attempt-001"),
        "runtime_commit": "",
    }
    blocker = {
        "step_id": "train-age",
        "run_id": "run-099",
        "status": "missing_pid",
        "target": "local",
        "gpus": "0",
    }
    write_rows(pipeline_dir / "jobs.tsv", [attempt])

    monkeypatch.setattr(experiment_pipeline, "_validate_frozen_pipeline", lambda *_args: {})
    monkeypatch.setattr(experiment_pipeline, "_validate_attempt_rows", lambda *_args: None)
    monkeypatch.setattr(
        experiment_pipeline,
        "_planned_runs",
        lambda rows: [{"step_id": row["step_id"], "run_id": row["run_id"]} for row in rows],
    )
    monkeypatch.setattr(
        experiment_pipeline,
        "read_run_manifest",
        lambda _root: [dict(attempt), dict(blocker)],
    )
    monkeypatch.setattr(
        experiment_pipeline.managed_scheduler,
        "launch_managed_runs",
        lambda *_args, **_kwargs: SimpleNamespace(committed_rows=[dict(attempt)]),
    )
    monkeypatch.setattr(
        experiment_pipeline,
        "_create_needed_retries",
        lambda *_args, **_kwargs: pytest.fail("capacity blocker must be handled before retry creation"),
    )
    monkeypatch.setattr(
        experiment_pipeline.time,
        "sleep",
        lambda *_args: pytest.fail("capacity blocker must not sleep"),
    )

    result = experiment_pipeline._run_attempts(
        root,
        pipeline_dir,
        _spec(root),
        {"age": {}},
        [attempt],
        poll_seconds=1,
    )

    assert result["status"] == "blocked"
    assert result["missing_pid_blocker"] == {
        "status": "missing_pid",
        "step_id": blocker["step_id"],
        "run_id": blocker["run_id"],
    }
    persisted = experiment_pipeline.read_rows(pipeline_dir / "jobs.tsv")
    assert [(row["attempt"], row["status"]) for row in persisted] == [("1", "failed")]


def test_run_attempts_syncs_owned_missing_pid_and_blocks_pending_sibling(tmp_path: Path, monkeypatch):
    root = tmp_path / "workspace"
    pipeline_dir = root / "pipelines" / "external-v1"
    pipeline_dir.mkdir(parents=True)
    (pipeline_dir / "spec.source.yaml").write_text("schema_version: 1\n")
    spec = _spec(root)
    spec["jobs"].append(
        {
            **spec["jobs"][0],
            "id": "age-hsp-i2-bcg",
            "modality": "bcg",
            "num_workers": 16,
        }
    )
    attempts = [
        {
            "step_id": "external-evaluate",
            "run_id": "run-001",
            "pipeline_id": "external-v1",
            "job_id": "age-hsp-i2-psg",
            "variant": "sleep2vec2",
            "attempt": 1,
            "status": "running",
            "verified": "false",
            "plan_dir": str(pipeline_dir / "plans" / "age-hsp-i2-psg" / "attempt-001"),
            "runtime_commit": "",
        },
        {
            "step_id": "external-evaluate",
            "run_id": "run-002",
            "pipeline_id": "external-v1",
            "job_id": "age-hsp-i2-bcg",
            "variant": "sleep2vec2",
            "attempt": 1,
            "status": "pending",
            "verified": "false",
            "plan_dir": str(pipeline_dir / "plans" / "age-hsp-i2-bcg" / "attempt-001"),
            "runtime_commit": "",
        },
    ]
    write_rows(pipeline_dir / "jobs.tsv", attempts)
    canonical = [{**attempts[0], "status": "missing_pid"}, dict(attempts[1])]

    monkeypatch.setattr(experiment_pipeline, "_validate_frozen_pipeline", lambda *_args: {})
    monkeypatch.setattr(experiment_pipeline, "_validate_attempt_rows", lambda *_args: None)
    monkeypatch.setattr(
        experiment_pipeline,
        "_planned_runs",
        lambda rows: [{"step_id": row["step_id"], "run_id": row["run_id"]} for row in rows],
    )
    monkeypatch.setattr(experiment_pipeline, "read_run_manifest", lambda _root: [dict(row) for row in canonical])
    launches = []

    def blocked_launch(*_args, **kwargs):
        launches.append(kwargs["fail_on_missing_pid_blocker"])
        raise managed_scheduler.MissingPidCapacityError(canonical[0]["step_id"], canonical[0]["run_id"])

    monkeypatch.setattr(experiment_pipeline.managed_scheduler, "launch_managed_runs", blocked_launch)
    monkeypatch.setattr(
        experiment_pipeline,
        "_create_needed_retries",
        lambda *_args, **_kwargs: pytest.fail("an owned missing_pid attempt must not create retries"),
    )
    monkeypatch.setattr(
        experiment_pipeline.time,
        "sleep",
        lambda *_args: pytest.fail("an owned missing_pid attempt must not sleep"),
    )

    result = experiment_pipeline._run_attempts(
        root,
        pipeline_dir,
        spec,
        {"age": {}},
        attempts,
        poll_seconds=1,
    )

    assert result["status"] == "blocked"
    assert result["missing_pid_blocker"] == {
        "status": "missing_pid",
        "step_id": canonical[0]["step_id"],
        "run_id": canonical[0]["run_id"],
    }
    assert launches == [True]
    persisted = {row["run_id"]: row for row in experiment_pipeline.read_rows(pipeline_dir / "jobs.tsv")}
    assert persisted["run-001"]["status"] == "missing_pid"
    assert persisted["run-002"]["status"] == "pending"
    assert not list(pipeline_dir.rglob(managed_scheduler.EXECUTION_SNAPSHOT_NAME))


def test_execute_pipeline_persists_and_clears_missing_pid_blocker_on_resume(tmp_path: Path, monkeypatch):
    class ResumeObserved(Exception):
        pass

    root = tmp_path / "workspace"
    pipeline_dir = root / "pipelines" / "external-v1"
    pipeline_dir.mkdir(parents=True)
    (pipeline_dir / "spec.source.yaml").write_text("schema_version: 1\n")
    (pipeline_dir / "pipeline.json").write_text(json.dumps({"status": "running_external"}) + "\n")
    blocker = {"status": "missing_pid", "step_id": "train-age", "run_id": "run-099"}
    blocked_result = {
        "status": "blocked",
        "jobs": [{"job_id": "age-hsp-i2-psg", "status": "running"}],
        "missing_pid_blocker": blocker,
    }

    monkeypatch.setattr(
        experiment_pipeline,
        "_validate_frozen_pipeline",
        lambda *_args: json.loads((pipeline_dir / "pipeline.json").read_text()),
    )
    monkeypatch.setattr(
        experiment_pipeline,
        "_inspect_sources",
        lambda *_args, **_kwargs: [{"failed_runs": [], "uncertain_runs": [], "complete": True}],
    )
    monkeypatch.setattr(experiment_pipeline, "_load_or_freeze_selections", lambda *_args: {})
    monkeypatch.setattr(experiment_pipeline, "_load_or_create_initial_attempts", lambda *_args: [])
    monkeypatch.setattr(experiment_pipeline, "_run_attempts", lambda *_args, **_kwargs: blocked_result)

    result = experiment_pipeline._execute_pipeline(
        root,
        pipeline_dir,
        _spec(root),
        poll_seconds=1,
        finalize_callback=None,
    )

    assert result == blocked_result
    state = json.loads((pipeline_dir / "pipeline.json").read_text())
    assert state["status"] == "blocked"
    assert state["missing_pid_blocker"] == blocker

    def observe_resume(*_args, **_kwargs):
        resumed_state = json.loads((pipeline_dir / "pipeline.json").read_text())
        assert resumed_state["status"] == "running_external"
        assert resumed_state["missing_pid_blocker"] is None
        raise ResumeObserved

    monkeypatch.setattr(experiment_pipeline, "_run_attempts", observe_resume)

    with pytest.raises(ResumeObserved):
        experiment_pipeline._execute_pipeline(
            root,
            pipeline_dir,
            _spec(root),
            poll_seconds=1,
            finalize_callback=None,
        )


@pytest.mark.parametrize(
    ("field", "drifted"),
    [
        ("step_id", "foreign-step"),
        ("run_id", "run-999"),
    ],
)
def test_planned_runs_rejects_managed_key_drift(tmp_path: Path, field: str, drifted: str):
    plan_dir = tmp_path / "plan"
    plan_dir.mkdir()
    result_root = tmp_path / "results" / "attempt-001"
    expected = {
        "pipeline_id": "external-v1",
        "job_id": "age-hsp-i2-psg",
        "attempt": 1,
        "result_root": str(result_root),
        "terminal_status_owner": "script",
    }
    planned = {
        "step_id": "external-evaluate",
        "run_id": "run-001",
        field: drifted,
    }
    (plan_dir / "plan.json").write_text(json.dumps({"runs": [planned]}) + "\n")
    attempt = {
        "step_id": "external-evaluate",
        "run_id": "run-001",
        "plan_dir": str(plan_dir),
        **expected,
    }

    with pytest.raises(ValueError, match="drift"):
        experiment_pipeline._planned_runs([attempt])


def test_planned_runs_carries_frozen_checkpoint_evidence_to_scheduler(tmp_path: Path):
    plan_dir = tmp_path / "plan"
    plan_dir.mkdir()
    planned = {"step_id": "external-evaluate", "run_id": "run-001"}
    (plan_dir / "plan.json").write_text(json.dumps({"runs": [planned]}) + "\n")
    attempt = {
        **planned,
        "plan_dir": str(plan_dir),
        "pipeline_id": "external-v1",
        "job_id": "age-hsp-i2-psg",
        "attempt": 1,
        "result_root": str(tmp_path / "results" / "attempt-001"),
        "checkpoint": str(tmp_path / "model.ckpt"),
        "checkpoint_sha256": "a" * 64,
    }

    run = experiment_pipeline._planned_runs([attempt])[0]

    assert run["checkpoint"] == attempt["checkpoint"]
    assert run["checkpoint_sha256"] == attempt["checkpoint_sha256"]


def test_frozen_pipeline_rejects_external_preset_byte_drift(tmp_path: Path, monkeypatch):
    root = tmp_path / "workspace"
    spec = _spec(root)
    plan_dir = Path(spec["checkpoint_sources"]["age"]["plan"])
    plan_dir.mkdir(parents=True)
    plan_path = plan_dir / "plan.json"
    recipe_path = plan_dir / "recipe.resolved.yaml"
    plan_path.write_text("{}\n")
    recipe_path.write_text("task: hparam_tune\n")
    preset = Path(spec["jobs"][0]["inference_preset_path"])
    preset.parent.mkdir(parents=True)
    preset.write_bytes(b"frozen-preset")

    pipeline_dir = root / "pipelines" / "external-v1"
    pipeline_dir.mkdir(parents=True)
    source_text = yaml.safe_dump(spec, sort_keys=False)
    resolved_text = yaml.safe_dump(spec, sort_keys=False)
    original_spec = tmp_path / "external.yaml"
    original_spec.write_text(source_text)
    (pipeline_dir / "spec.source.yaml").write_text(source_text)
    (pipeline_dir / "spec.resolved.yaml").write_text(resolved_text)
    state = {
        "schema_version": 1,
        "pipeline_id": "external-v1",
        "experiment_id": "unit",
        "status": "waiting_for_sources",
        "spec_path": str(original_spec),
        "spec_source_sha256": experiment_pipeline._text_sha256(source_text),
        "spec_resolved_sha256": experiment_pipeline._text_sha256(resolved_text),
        "runtime_commit": "a" * 40,
        "source_plans": [
            {
                "source_id": "age",
                "plan_dir": str(plan_dir),
                "plan_path": str(plan_path),
                "plan_sha256": file_sha256(plan_path),
                "resolved_recipe_path": str(recipe_path),
                "resolved_recipe_sha256": file_sha256(recipe_path),
            }
        ],
        "external_presets": [
            {
                "job_id": "age-hsp-i2-psg",
                "path": str(preset),
                "sha256": file_sha256(preset),
            }
        ],
    }
    (pipeline_dir / "pipeline.json").write_text(json.dumps(state) + "\n")
    monkeypatch.setattr(experiment_pipeline.artifacts, "read_hparam_plan", lambda _plan_dir: {})

    preset.write_bytes(b"changed-preset")

    with pytest.raises(ValueError, match="Frozen external preset changed"):
        experiment_pipeline._validate_frozen_pipeline(pipeline_dir, source_text, spec)


@pytest.mark.parametrize("tamper", [False, True])
def test_orphan_checkpoint_selection_is_rederived_before_state_commit(tmp_path: Path, monkeypatch, tamper: bool):
    root = tmp_path / "workspace"
    pipeline_dir = root / "pipelines" / "external-v1"
    pipeline_dir.mkdir(parents=True)
    spec = _spec(root)
    config = tmp_path / "config.yaml"
    checkpoint = tmp_path / "rank-1.ckpt"
    alternate = tmp_path / "alternate.ckpt"
    config.write_text("model: unit\n")
    checkpoint.write_bytes(b"rank-1")
    alternate.write_bytes(b"alternate")
    derived = {
        "source_id": "age",
        "plan": spec["checkpoint_sources"]["age"]["plan"],
        "selection_metric": "val_mae",
        "selection_mode": "min",
        "score": 4.5,
        "config": str(config),
        "config_sha256": file_sha256(config),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "variant": "sleep2vec2",
        "label_name": "age",
        "source_task": "age",
    }
    orphan = copy.deepcopy(derived)
    if tamper:
        orphan["checkpoint"] = str(alternate)
        orphan["checkpoint_sha256"] = file_sha256(alternate)
    checkpoints_path = pipeline_dir / "checkpoints.json"
    checkpoints_path.write_text(
        json.dumps(
            {
                "pipeline_id": "external-v1",
                "created_at": "2026-07-20T00:00:00Z",
                "sources": [orphan],
            }
        )
        + "\n"
    )
    state_path = pipeline_dir / "pipeline.json"
    state_path.write_text(json.dumps({"status": "waiting_for_sources"}) + "\n")
    monkeypatch.setattr(experiment_pipeline, "_select_checkpoint_sources", lambda *_args: [derived])

    if tamper:
        with pytest.raises(ValueError, match="differs from validation-derived selection"):
            experiment_pipeline._load_or_freeze_selections(root, pipeline_dir, spec)
        assert "checkpoint_selection_sha256" not in json.loads(state_path.read_text())
    else:
        selections = experiment_pipeline._load_or_freeze_selections(root, pipeline_dir, spec)
        assert selections == {"age": derived}
        assert json.loads(state_path.read_text())["checkpoint_selection_sha256"] == file_sha256(checkpoints_path)


def test_completed_pipeline_resume_validates_and_finalizes_without_reexecution(tmp_path: Path, monkeypatch):
    root = tmp_path / "workspace"
    pipeline_dir = root / "pipelines" / "external-v1"
    pipeline_dir.mkdir(parents=True)
    (pipeline_dir / "spec.source.yaml").write_text("schema_version: 1\n")
    report = pipeline_dir / "final.md"
    report.write_text("completed\n")
    state = {
        "status": "completed",
        "final_report": str(report),
        "result_artifacts": {str(report): file_sha256(report)},
    }
    (pipeline_dir / "pipeline.json").write_text(json.dumps(state) + "\n")
    attempt = {"status": "completed"}

    monkeypatch.setattr(experiment_pipeline, "_validate_frozen_pipeline", lambda *_args: state)
    monkeypatch.setattr(
        experiment_pipeline,
        "_validate_experiment",
        lambda *_args, **_kwargs: {"status": "active"},
    )
    monkeypatch.setattr(experiment_pipeline, "_load_or_freeze_selections", lambda *_args: {"age": {}})
    monkeypatch.setattr(experiment_pipeline, "read_rows", lambda *_args, **_kwargs: [attempt])
    monkeypatch.setattr(experiment_pipeline, "_validate_attempt_rows", lambda *_args: None)
    monkeypatch.setattr(
        experiment_pipeline,
        "_logical_job_states",
        lambda *_args: [{"job_id": "age-hsp-i2-psg", "status": "completed"}],
    )
    monkeypatch.setattr(
        experiment_pipeline,
        "_inspect_sources",
        lambda *_args, **_kwargs: pytest.fail("completed pipelines must not recheck training sources"),
    )
    monkeypatch.setattr(
        experiment_pipeline,
        "_run_attempts",
        lambda *_args, **_kwargs: pytest.fail("completed pipelines must not rerun external attempts"),
    )
    finalized = []

    result = experiment_pipeline._execute_pipeline(
        root,
        pipeline_dir,
        _spec(root),
        poll_seconds=0,
        finalize_callback=lambda finalized_root, finalized_report: finalized.append((finalized_root, finalized_report)),
    )

    assert result["status"] == "completed"
    assert finalized == [(root, report)]

    finalized.clear()
    monkeypatch.setattr(
        experiment_pipeline,
        "_validate_experiment",
        lambda *_args, **_kwargs: {"status": "completed"},
    )
    result = experiment_pipeline._execute_pipeline(
        root,
        pipeline_dir,
        _spec(root),
        poll_seconds=0,
        finalize_callback=lambda finalized_root, finalized_report: finalized.append((finalized_root, finalized_report)),
    )
    assert result["status"] == "completed"
    assert finalized == []
