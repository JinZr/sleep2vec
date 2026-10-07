from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import shutil
import threading
from types import SimpleNamespace

from agent_tool_test_helpers import (
    FakePipelineRuntime,
    PipelineInterrupted,
    call_while_run_lock_holder_commits,
    complete_experiment,
    dry_run_pipeline,
    prepare_pipeline_sources,
    run_pipeline,
    write_finetune_recipe,
)
import pytest
from test_agent_tools_experiment_pipeline_cohort_selection import _spec as _cohort_spec
import yaml

from agent_tools import (
    experiment_pipeline,
    experiment_pipeline_attempts as pipeline_attempts,
    experiment_pipeline_results,
    experiment_pipeline_spec as pipeline_spec,
    experiments,
    managed_scheduler,
    plan_contract,
    plans,
    python_programs,
)
from agent_tools.experiment_workspace import commit_step_manifest, file_sha256, merge_run_manifest, read_run_manifest
from agent_tools.manifests import write_rows


def test_attempt_materialization_enters_plan_publication_lock(tmp_path: Path, monkeypatch):
    spec = _spec(tmp_path / "workspace")
    spec_path = prepare_pipeline_sources(tmp_path, monkeypatch, spec)
    pipeline_dir = tmp_path / "workspace" / "pipelines" / "external-v1"
    locked = []
    materialized = []
    real_lock = pipeline_attempts.plan_publication_lock
    real_materialize = pipeline_attempts._materialize_attempt_locked

    @contextmanager
    def publication_lock(out):
        with real_lock(out):
            locked.append(out)
            try:
                yield
            finally:
                locked.remove(out)

    def materialize_locked(*args, plan_dir, **kwargs):
        assert locked == [plan_dir]
        materialized.append(plan_dir)
        return real_materialize(*args, plan_dir=plan_dir, **kwargs)

    monkeypatch.setattr(pipeline_attempts, "plan_publication_lock", publication_lock)
    monkeypatch.setattr(pipeline_attempts, "_materialize_attempt_locked", materialize_locked)

    result = run_pipeline(spec_path, FakePipelineRuntime(spec).hooks())

    assert result["status"] == "completed"
    assert materialized == [pipeline_dir / "plans" / "age-hsp-i2-psg" / "attempt-001"]


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
    monkeypatch.setattr(pipeline_attempts, "_attempt_recipe", attempt_recipe)
    monkeypatch.setattr(pipeline_attempts, "_ensure_initial_preflight", lambda *_args: None)
    monkeypatch.setattr(pipeline_attempts, "_prepare_attempt_registration_groups", prepare_registration)
    monkeypatch.setattr(
        pipeline_attempts,
        "_materialize_attempt",
        lambda _root, _spec, job, _selection, attempt, **_paths: {"job_id": job["id"], "attempt": attempt},
    )
    monkeypatch.setattr(pipeline_attempts, "write_jobs", lambda *_args: None)
    monkeypatch.setattr(pipeline_attempts, "validate_attempt_rows", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(pipeline_attempts, "_reconcile_pipeline_jobs_planned_event", lambda *_args: None)
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
                pipeline_attempts.load_or_create_initial_attempts(root, pipeline_dir, spec, selections)
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

    monkeypatch.setattr(pipeline_attempts, "_attempt_recipe", attempt_recipe)
    monkeypatch.setattr(pipeline_attempts, "_ensure_initial_preflight", lambda *_args: None)
    monkeypatch.setattr(
        pipeline_attempts,
        "_prepare_attempt_registration_groups",
        lambda _root, _spec, items, **_kwargs: {item[0]["id"]: item[4] for item in items},
    )
    monkeypatch.setattr(
        pipeline_attempts,
        "_materialize_attempt",
        lambda _root, _spec, job, _selection, attempt, **_paths: {"job_id": job["id"], "attempt": attempt},
    )
    monkeypatch.setattr(
        pipeline_attempts,
        "write_jobs",
        lambda *_args: (_ for _ in ()).throw(OSError("jobs projection interrupted")),
    )

    with pytest.raises(pipeline_attempts.PipelineRegistrationRecoveryError, match="reconciled on resume"):
        pipeline_attempts.load_or_create_initial_attempts(root, pipeline_dir, spec, selections)


def test_registered_jobs_retry_cleans_interrupted_atomic_temp(tmp_path: Path, monkeypatch):
    jobs_path = tmp_path / "jobs.tsv"
    rows = [{"run_id": "run-001", "status": "planned"}]
    replace = experiment_pipeline_results.os.replace
    calls = 0

    def fail_once(source, destination):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("jobs projection interrupted")
        replace(source, destination)

    monkeypatch.setattr(experiment_pipeline_results.os, "replace", fail_once)

    with pytest.raises(pipeline_attempts.PipelineRegistrationRecoveryError, match="reconciled on resume"):
        pipeline_attempts._write_registered_jobs(jobs_path, rows)

    assert not list(tmp_path.glob(".jobs.tsv.*.tmp"))
    pipeline_attempts._write_registered_jobs(jobs_path, rows)
    assert experiment_pipeline.read_rows(jobs_path) == rows


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
    monkeypatch.setattr(pipeline_attempts, "append_event", experiment_pipeline.append_event)

    with pytest.raises(pipeline_attempts.PipelineRegistrationRecoveryError, match="reconciled on resume"):
        pipeline_attempts._reconcile_pipeline_jobs_planned_event(root, spec)

    monkeypatch.setattr(pipeline_attempts, "append_event", original_append)
    pipeline_attempts._reconcile_pipeline_jobs_planned_event(root, spec)
    pipeline_attempts._reconcile_pipeline_jobs_planned_event(root, spec)

    events = [
        event
        for event in pipeline_attempts.read_experiment_events(root)
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

    monkeypatch.setattr(pipeline_attempts, "read_experiment_events", read_events)
    monkeypatch.setattr(experiment_pipeline, "append_event", lambda *_args, **_kwargs: None)

    with pytest.raises(pipeline_attempts.PipelineRegistrationRecoveryError, match="reconciled on resume"):
        pipeline_attempts._reconcile_pipeline_jobs_planned_event(root, spec)


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
    monkeypatch.setattr(pipeline_attempts, "append_event", experiment_pipeline.append_event)

    with pytest.raises(pipeline_attempts.PipelineRegistrationRecoveryError, match="reconciled on resume"):
        pipeline_attempts._reconcile_pipeline_retry_planned_event(root, spec, attempt)

    monkeypatch.setattr(pipeline_attempts, "append_event", original_append)
    pipeline_attempts._reconcile_pipeline_retry_planned_event(root, spec, attempt)
    pipeline_attempts._reconcile_pipeline_retry_planned_event(root, spec, attempt)

    events = [
        event
        for event in pipeline_attempts.read_experiment_events(root)
        if event.get("event_type") == "pipeline_job_retry_planned"
    ]
    assert len(events) == 1
    assert events[0]["pipeline_id"] == spec["pipeline"]["id"]
    assert events[0]["job_id"] == attempt["job_id"]
    assert events[0]["attempt"] == 2


@pytest.mark.parametrize(
    "kind,event_type,count_field",
    [
        ("external_matrix", "pipeline_checkpoints_frozen", "source_count"),
        ("cohort_selection", "pipeline_candidates_frozen", "candidate_count"),
    ],
)
def test_selection_event_is_reconciled_after_committed_hash(
    tmp_path: Path,
    monkeypatch,
    kind: str,
    event_type: str,
    count_field: str,
):
    root = tmp_path / "workspace"
    spec = _spec(root) if kind == "external_matrix" else _cohort_spec(root)
    if kind == "cohort_selection":
        spec["candidates"]["count"] = 1
    spec_path = prepare_pipeline_sources(tmp_path, monkeypatch, spec)
    pipeline_dir = root / "pipelines" / spec["pipeline"]["id"]
    runtime = FakePipelineRuntime(spec)
    original_append = pipeline_attempts.append_event

    def interrupt_selection_event(root_path, appended_type, payload):
        if appended_type == event_type:
            raise RuntimeError("event append interrupted")
        original_append(root_path, appended_type, payload)

    monkeypatch.setattr(pipeline_attempts, "append_event", interrupt_selection_event)

    with pytest.raises(pipeline_attempts.PipelineRegistrationRecoveryError, match="reconciled on resume"):
        run_pipeline(spec_path, runtime.hooks())

    selection_path = pipeline_dir / ("candidates.json" if kind == "cohort_selection" else "checkpoints.json")
    hash_field = "candidate_selection_sha256" if kind == "cohort_selection" else "checkpoint_selection_sha256"
    state = json.loads((pipeline_dir / "pipeline.json").read_text())
    assert state[hash_field] == file_sha256(selection_path)
    assert state["status"] == "ready"
    assert not any(name == "inspect" for name, _ids in runtime.calls)

    monkeypatch.setattr(pipeline_attempts, "append_event", original_append)
    assert run_pipeline(spec_path, runtime.hooks(), resume=True)["status"] == "completed"

    def selection_events():
        events = pipeline_attempts.read_experiment_events(root)
        return [event for event in events if event.get("event_type") == event_type]

    events = selection_events()
    assert len(events) == 1
    assert events[0]["pipeline_id"] == spec["pipeline"]["id"]
    assert events[0][count_field] == 1

    # A completed pipeline keeps its selection history unchanged during resume validation.
    history = (root / "events.jsonl").read_text().splitlines(keepends=True)
    (root / "events.jsonl").write_text(
        "".join(line for line in history if json.loads(line).get("event_type") != event_type)
    )
    assert run_pipeline(spec_path, runtime.hooks(), resume=True)["status"] == "completed"
    assert selection_events() == []


def test_pipeline_registration_recovery_error_does_not_mark_pipeline_failed(tmp_path: Path, monkeypatch):
    spec = _spec(tmp_path / "workspace")
    spec_path = prepare_pipeline_sources(tmp_path, monkeypatch, spec)
    pipeline_dir = tmp_path / "workspace" / "pipelines" / "external-v1"
    original_append = pipeline_attempts.append_event

    def interrupt_jobs_event(root_path, event_type, payload):
        if event_type == "pipeline_jobs_planned":
            raise RuntimeError("event append interrupted")
        original_append(root_path, event_type, payload)

    monkeypatch.setattr(pipeline_attempts, "append_event", interrupt_jobs_event)

    with pytest.raises(pipeline_attempts.PipelineRegistrationRecoveryError, match="reconciled on resume"):
        run_pipeline(spec_path, FakePipelineRuntime(spec).hooks())

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


def _two_job_spec(root: Path) -> dict:
    spec = _spec(root)
    spec["jobs"].append({**spec["jobs"][0], "id": "age-hsp-i2-bcg", "modality": "bcg", "num_workers": 16})
    spec["jobs"][1]["inference_preset_path"] = str(root / "presets" / "hsp_i2_age_bcg.pickle")
    return spec


@pytest.mark.parametrize("prefixes", [["ema_model."], ["running_mean_model."]])
def test_schema_requires_both_model_averaging_prefixes(tmp_path: Path, prefixes: list[str]):
    spec = _spec(tmp_path)
    spec["checkpoint_policy"]["forbidden_state_dict_prefixes"] = prefixes

    with pytest.raises(ValueError, match="forbidden_state_dict_prefixes"):
        pipeline_spec.validate_spec(spec, tmp_path, unlock_final_test=True)


@pytest.mark.parametrize("failed_status", ["failed", "stopped"])
def test_mixed_terminal_source_accepts_no_test_after_fit_manifest(
    tmp_path: Path,
    monkeypatch,
    failed_status: str,
):
    root = tmp_path / "workspace"
    root.mkdir()
    spec = _spec(root)
    successful = {"step_id": "train-age", "run_id": "run-000"}
    unsuccessful = {"step_id": "train-age", "run_id": "run-001"}
    manifest_path = tmp_path / "run_manifest.json"
    manifest_path.write_text(json.dumps({"status": "skipped_test", "metrics": {"val_mae": 4.5}}) + "\n")

    monkeypatch.setattr(
        experiment_pipeline,
        "_source_hparam_plans",
        lambda _source_id, source: [(Path(source["plan"]), {"runs": [successful, unsuccessful]})],
    )
    monkeypatch.setattr(
        experiment_pipeline,
        "read_run_manifest",
        lambda _root: [
            {**successful, "status": "finished"},
            {
                **unsuccessful,
                "status": failed_status,
                **({"stop_reason": "stopped after invalid candidate"} if failed_status == "stopped" else {}),
            },
        ],
    )
    monkeypatch.setattr(
        experiment_pipeline.artifacts,
        "find_run_manifest",
        lambda run: manifest_path if run == successful else pytest.fail("failed runs have no success manifest"),
    )

    result = dry_run_pipeline(root, spec)

    states = result["source_states"]
    assert states[0]["complete"] is True
    assert states[0]["failed_runs"] == ["run-001"]
    assert result["status"] == "ready"


def test_mixed_terminal_source_rejects_stopped_run_without_reason(tmp_path: Path, monkeypatch):
    root = tmp_path / "workspace"
    root.mkdir()
    spec = _spec(root)
    runs = [
        {"step_id": "train-age", "run_id": "run-000"},
        {"step_id": "train-age", "run_id": "run-001"},
    ]
    monkeypatch.setattr(
        experiment_pipeline,
        "_source_hparam_plans",
        lambda _source_id, source: [(Path(source["plan"]), {"runs": runs})],
    )
    monkeypatch.setattr(
        experiment_pipeline,
        "read_run_manifest",
        lambda _root: [{**runs[0], "status": "finished"}, {**runs[1], "status": "stopped"}],
    )

    with pytest.raises(ValueError, match="Stopped source runs are missing required stop_reason.*run-001"):
        dry_run_pipeline(root, spec)


@pytest.mark.parametrize("status", ["planned", "running"])
def test_active_source_waits_for_terminal_status(tmp_path: Path, monkeypatch, status: str):
    root = tmp_path / "workspace"
    root.mkdir()
    spec = _spec(root)
    run = {"step_id": "train-age", "run_id": "run-000"}
    monkeypatch.setattr(
        experiment_pipeline,
        "_source_hparam_plans",
        lambda _source_id, source: [(Path(source["plan"]), {"runs": [run]})],
    )
    monkeypatch.setattr(experiment_pipeline, "read_run_manifest", lambda _root: [{**run, "status": status}])

    result = dry_run_pipeline(root, spec)

    assert result["source_states"][0]["complete"] is False
    assert result["status"] == "waiting_for_sources"


def test_all_unsuccessful_terminal_source_fails(tmp_path: Path, monkeypatch):
    root = tmp_path / "workspace"
    root.mkdir()
    spec = _spec(root)
    runs = [
        {"step_id": "train-age", "run_id": "run-000"},
        {"step_id": "train-age", "run_id": "run-001"},
    ]
    monkeypatch.setattr(
        experiment_pipeline,
        "_source_hparam_plans",
        lambda _source_id, source: [(Path(source["plan"]), {"runs": runs})],
    )
    monkeypatch.setattr(
        experiment_pipeline,
        "read_run_manifest",
        lambda _root: [
            {**runs[0], "status": "failed"},
            {**runs[1], "status": "stopped", "stop_reason": "budget exhausted"},
        ],
    )

    result = dry_run_pipeline(root, spec)

    assert result["source_states"][0]["complete"] is False
    assert result["status"] == "failed"


@pytest.mark.parametrize("status", ["submitting", "unknown_scheduler"])
def test_slurm_source_uncertainty_blocks_external_pipeline(tmp_path: Path, monkeypatch, status: str):
    root = tmp_path / "workspace"
    root.mkdir()
    spec = _spec(root)
    run = {"step_id": "train-age", "run_id": "run-000"}
    monkeypatch.setattr(
        experiment_pipeline,
        "_source_hparam_plans",
        lambda _source_id, source: [(Path(source["plan"]), {"runs": [run]})],
    )
    monkeypatch.setattr(experiment_pipeline, "read_run_manifest", lambda _root: [{**run, "status": status}])

    result = dry_run_pipeline(root, spec)

    assert result["source_states"][0]["uncertain_runs"] == ["run-000"]
    assert result["status"] == "blocked"


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
            raise pipeline_attempts.RetryPreparationError("preflight failed")

    def materialize(_root, _spec, job, _selection, attempt, **_paths):
        order.append(f"materialize:{job['id']}")
        return {"job_id": job["id"], "attempt": attempt, "status": "planned", "verified": "false"}

    monkeypatch.setattr(pipeline_attempts, "_attempt_recipe", attempt_recipe)
    monkeypatch.setattr(pipeline_attempts, "_ensure_retry_preflight", retry_preflight)
    monkeypatch.setattr(
        pipeline_attempts,
        "_prepare_attempt_registration_groups",
        lambda _root, _spec, items, **_kwargs: {item[0]["id"]: None for item in items},
    )
    monkeypatch.setattr(pipeline_attempts, "_materialize_attempt", materialize)
    monkeypatch.setattr(experiment_pipeline, "append_event", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(pipeline_attempts, "_reconcile_pipeline_retry_planned_event", lambda *_args: None)
    monkeypatch.setattr(pipeline_attempts, "read_run_manifest", lambda _root: [])

    updated, created = pipeline_attempts.create_needed_retries(
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
    assert [job["status"] for job in experiment_pipeline_results.logical_job_states(spec, updated)] == [
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
            raise pipeline_attempts.AttemptRegistrationPreflightError("target argv rejected")
        return {job_id: None}

    def materialize(_root, _spec, job, _selection, attempt, **_paths):
        order.append(f"materialize:{job['id']}")
        return {"job_id": job["id"], "attempt": attempt, "status": "planned", "verified": "false"}

    monkeypatch.setattr(pipeline_attempts, "_attempt_recipe", attempt_recipe)
    monkeypatch.setattr(pipeline_attempts, "_ensure_retry_preflight", lambda *_args: None)
    monkeypatch.setattr(pipeline_attempts, "_prepare_attempt_registration_groups", prepare_registration)
    monkeypatch.setattr(pipeline_attempts, "_materialize_attempt", materialize)
    monkeypatch.setattr(experiment_pipeline, "append_event", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(pipeline_attempts, "_reconcile_pipeline_retry_planned_event", lambda *_args: None)
    monkeypatch.setattr(pipeline_attempts, "read_run_manifest", lambda _root: [])

    updated, created = pipeline_attempts.create_needed_retries(
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

    monkeypatch.setattr(pipeline_attempts, "_attempt_recipe", attempt_recipe)
    monkeypatch.setattr(pipeline_attempts, "_ensure_retry_preflight", lambda *_args: None)
    monkeypatch.setattr(
        pipeline_attempts,
        "_prepare_attempt_registration_groups",
        lambda _root, _spec, items, **_kwargs: {items[0][0]["id"]: None},
    )
    monkeypatch.setattr(
        pipeline_attempts,
        "_materialize_attempt",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("canonical commit failed")),
    )
    monkeypatch.setattr(pipeline_attempts, "read_run_manifest", lambda _root: [])

    with pytest.raises(RuntimeError, match="canonical commit failed"):
        pipeline_attempts.create_needed_retries(
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

    monkeypatch.setattr(pipeline_attempts, "_attempt_recipe", attempt_recipe)
    monkeypatch.setattr(pipeline_attempts, "_ensure_retry_preflight", lambda *_args: None)
    monkeypatch.setattr(
        pipeline_attempts,
        "_prepare_attempt_registration_groups",
        lambda _root, _spec, items, **_kwargs: {items[0][0]["id"]: None},
    )
    monkeypatch.setattr(
        pipeline_attempts,
        "_materialize_attempt",
        lambda _root, _spec, job, _selection, attempt, **_paths: {
            "job_id": job["id"],
            "attempt": attempt,
            "status": "planned",
            "verified": "false",
        },
    )
    monkeypatch.setattr(
        pipeline_attempts,
        "write_jobs",
        lambda *_args: (_ for _ in ()).throw(OSError("jobs projection interrupted")),
    )
    monkeypatch.setattr(pipeline_attempts, "read_run_manifest", lambda _root: [])

    with pytest.raises(pipeline_attempts.PipelineRegistrationRecoveryError, match="reconciled on resume"):
        pipeline_attempts.create_needed_retries(
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
        if [Path(path) for path in paths] == [root / "run_manifest.tsv.lock"]:
            assert Path(root_path) == root
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

    monkeypatch.setattr(pipeline_attempts, "_ensure_initial_preflight", lambda *_args: None)
    monkeypatch.setattr(pipeline_attempts, "build_plan", build_staged_plan)
    monkeypatch.setattr(experiment_pipeline.exp_io, "validate_managed_output_paths", reject_unsafe_group)
    monkeypatch.setattr(experiment_pipeline.managed_scheduler, "inspect_execution_target", reject_second_group)

    expected_error = "frozen output topology rejected" if failure_kind == "topology" else "frozen argv rejected"
    with pytest.raises(pipeline_attempts.AttemptRegistrationPreflightError, match=expected_error):
        pipeline_attempts.load_or_create_initial_attempts(root, pipeline_dir, spec, selections)

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
    monkeypatch.setattr(pipeline_attempts, "read_run_manifest", experiment_pipeline.read_run_manifest)
    monkeypatch.setattr(pipeline_attempts, "next_run_index", lambda _recipe: 1)
    monkeypatch.setattr(pipeline_attempts, "_validate_new_attempt_paths", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(pipeline_attempts, "_prepare_attempt_plan", prepare_plan)
    monkeypatch.setattr(experiment_pipeline.exp_io, "validate_managed_output_paths", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(experiment_pipeline.managed_scheduler, "inspect_execution_target", inspect)

    snapshot_owner = pipeline_dir / "initial_schedulers" / "sleep2vec2"
    assert not snapshot_owner.exists()
    prepared = pipeline_attempts._prepare_attempt_registration_groups(
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
    with pytest.raises(pipeline_attempts.AttemptRegistrationPreflightError, match="snapshot changed"):
        pipeline_attempts._prepare_attempt_registration_groups(
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
        pipeline_attempts,
        "_attempt_recipe",
        lambda *_args, **_kwargs: pytest.fail("unsafe process identity must not be retried"),
    )
    monkeypatch.setattr(experiment_pipeline, "append_event", lambda *_args, **_kwargs: None)

    updated, created = pipeline_attempts.create_needed_retries(
        root,
        pipeline_dir,
        _spec(root),
        {"age": {}},
        [attempt],
    )

    assert created is False
    assert updated[0]["retry_blocker"] == f"unsafe process identity: {identity_error}"
    assert experiment_pipeline_results.logical_job_states(_spec(root), updated)[0]["status"] == "blocked"


def test_atomic_generic_plan_freezes_single_runtime_command(tmp_path: Path, monkeypatch):
    source = tmp_path / "source"
    recipe_path = write_finetune_recipe(source, variant="sleep2vec2")
    recipe = yaml.safe_load(recipe_path.read_text())
    workspace = tmp_path / "workspace"
    runtime_commit = "a" * 40
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
        "runtime_commit": runtime_commit,
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
    helper_text = "\n".join(script_lines[helper_index:running_index])
    assert "/runtime/python -c " in helper_text
    assert "record-runtime-commit" in helper_text
    assert runtime_commit in helper_text
    assert helper_index < running_index < command_index
    assert plan["recipe"]["execution"] == recipe["execution"]
    canonical = read_run_manifest(workspace)[0]
    assert canonical.get("command") in (None, "")
    pipeline_attempts._validate_attempt_plan(
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
                "runtime_commit": runtime_commit,
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
            "runtime_commit": runtime_commit,
        },
        [planned],
        command_runner=inspect_command,
    )
    assert snapshot["module"] == "sleep2vec2.infer"
    assert snapshot["required_options"] == ["--config"]
    runtime_commit = "a" * 64
    with pytest.raises(ValueError, match="invalid runtime commit"):
        managed_scheduler.inspect_execution_target(
            {
                "target": "local",
                "workdir": "/runtime/snapshot",
                "python": "/runtime/python",
                "runtime_commit": "a" * 40,
            },
            [planned],
            command_runner=inspect_command,
        )


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
    recipe, recipe_path, plan_dir, result_root = pipeline_attempts._attempt_recipe(
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
            pipeline_attempts._materialize_attempt(
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
    running_index = launch_lines.index("_agent_commit_status running")
    assert f"{spec['runtime']['python']} -c " in "\n".join(launch_lines[helper_index:running_index])

    if outcome == "tamper":
        (plan_dir / "plan.md").write_text("tampered\n")
        with pytest.raises(ValueError, match="differs from deterministic regeneration"):
            pipeline_attempts._materialize_attempt(
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
        real_merge = pipeline_attempts.merge_run_manifest

        def merge_then_interrupt(*args, **kwargs):
            real_merge(*args, **kwargs)
            raise RuntimeError("simulated interruption after canonical commit")

        monkeypatch.setattr(pipeline_attempts, "merge_run_manifest", merge_then_interrupt)
        with pytest.raises(RuntimeError, match="simulated interruption"):
            pipeline_attempts._materialize_attempt(
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

    original_prepare = pipeline_attempts._prepare_attempt_plan
    if outcome == "prepared_public":
        monkeypatch.setattr(
            pipeline_attempts,
            "_prepare_attempt_plan",
            lambda *_args, **_kwargs: pytest.fail("validated public plan must not be rebuilt"),
        )
    row = pipeline_attempts._materialize_attempt(
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

    monkeypatch.setattr(pipeline_attempts, "_prepare_attempt_plan", original_prepare)
    pipeline_attempts._materialize_attempt(
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
    recipe, recipe_path, plan_dir, result_root = pipeline_attempts._attempt_recipe(
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
        pipeline_attempts._materialize_attempt(
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


def test_run_attempts_waits_when_capacity_defers_launch(tmp_path: Path, monkeypatch):
    spec = _spec(tmp_path / "workspace")
    spec_path = prepare_pipeline_sources(tmp_path, monkeypatch, spec)
    pipeline_dir = tmp_path / "workspace" / "pipelines" / "external-v1"
    runtime = FakePipelineRuntime(spec, outcome=lambda _run: None)
    waits = []

    def wait(seconds):
        waits.append(seconds)
        raise PipelineInterrupted

    with pytest.raises(PipelineInterrupted):
        run_pipeline(spec_path, runtime.hooks(sleep=wait))

    assert [name for name, _ids in runtime.calls] == ["monitor", "inspect", "inspect", "launch"]
    assert waits == [0]
    assert [row["status"] for row in experiment_pipeline.read_rows(pipeline_dir / "jobs.tsv")] == ["planned"]
    assert json.loads((pipeline_dir / "pipeline.json").read_text())["status"] == "running_external"


@pytest.mark.parametrize(
    "reader",
    [
        "validate_experiment",
        "prepare_registration_groups",
        "validate_attempt_rows",
        "create_needed_retries",
        "run_attempts",
        "run_attempts_with_snapshot",
    ],
)
def test_pipeline_attempt_polls_read_canonical_state_only_under_run_lock(tmp_path: Path, monkeypatch, reader):
    root = tmp_path / "workspace"
    pipeline_dir = root / "pipelines" / "external-v1"
    pipeline_dir.mkdir(parents=True)
    (pipeline_dir / "spec.source.yaml").write_text("schema_version: 1\n")
    experiment = {
        "id": "unit",
        "title": "Unit",
        "objective": "Exercise pipeline polls during canonical commits.",
        "root": str(root),
        "baseline": {"type": "none"},
        "status": "active",
    }
    (root / "experiment.yaml").write_text(yaml.safe_dump({"experiment": experiment}, sort_keys=False))
    attempt = {
        "experiment_id": "unit",
        "step_id": "external-evaluate",
        "run_id": "run-001",
        "pipeline_id": "external-v1",
        "job_id": "age-hsp-i2-psg",
        "variant": "sleep2vec2",
        "attempt": "1",
        "status": "completed",
        "verified": "false",
        "plan_dir": str(pipeline_dir / "plans" / "age-hsp-i2-psg" / "attempt-001"),
        "terminal_status_owner": "script",
    }
    write_rows(root / "run_manifest.tsv", [attempt])
    write_rows(pipeline_dir / "jobs.tsv", [attempt])
    spec = _spec(root)
    if reader.startswith("run_attempts"):
        # Stub the frozen-artifact checks so the loop's canonical reads are the first managed manifest reads; an
        # existing execution snapshot moves the first one to the pre-launch snapshot check.
        if reader == "run_attempts_with_snapshot":
            (pipeline_dir / managed_scheduler.EXECUTION_SNAPSHOT_NAME).write_text("{}\n")
        monkeypatch.setattr(experiment_pipeline, "_validate_frozen_pipeline", lambda *_args: {})
        monkeypatch.setattr(pipeline_attempts, "validate_attempt_rows", lambda *_args: None)
        monkeypatch.setattr(pipeline_attempts, "planned_runs", lambda rows: [dict(row) for row in rows])
        result_manifest = pipeline_dir / "result_manifest.json"
        monkeypatch.setattr(experiment_pipeline, "_validate_result_manifest", lambda *_args: result_manifest)
    calls = {
        "validate_experiment": lambda: experiment_pipeline._validate_experiment(root, spec),
        "prepare_registration_groups": lambda: pipeline_attempts._prepare_attempt_registration_groups(
            root, spec, [], snapshot_owner_dirs={}
        ),
        "validate_attempt_rows": lambda: pipeline_attempts.validate_attempt_rows(
            root, pipeline_dir, spec, {}, [], require_all_jobs=False
        ),
        "create_needed_retries": lambda: pipeline_attempts.create_needed_retries(root, pipeline_dir, spec, {}, []),
        "run_attempts": lambda: experiment_pipeline._run_attempts(
            root,
            pipeline_dir,
            spec,
            {"age": {}},
            [attempt],
            poll_seconds=1,
            hooks=experiment_pipeline.PipelineHooks(
                launch_runs=lambda *_args, **_kwargs: SimpleNamespace(committed_rows=[dict(attempt)])
            ),
        ),
    }
    calls["run_attempts_with_snapshot"] = calls["run_attempts"]

    result = call_while_run_lock_holder_commits(monkeypatch, root, calls[reader])

    if reader == "validate_experiment":
        assert result["id"] == "unit"
    elif reader == "prepare_registration_groups":
        assert result == {}
    elif reader == "create_needed_retries":
        assert result == ([], False)
    elif reader.startswith("run_attempts"):
        assert result["status"] == "completed"
        assert experiment_pipeline.read_rows(pipeline_dir / "jobs.tsv")[0]["verified"] == "true"


@pytest.mark.parametrize(
    "canonical_runtime_commit",
    [
        pytest.param("", id="canonical-runtime-unknown"),
        pytest.param("c" * 40, id="canonical-runtime-recorded"),
    ],
)
def test_run_attempts_terminal_attempt_projects_only_canonical_runtime_commit(
    tmp_path: Path,
    monkeypatch,
    canonical_runtime_commit: str,
):
    spec = _spec(tmp_path / "workspace")
    spec_path = prepare_pipeline_sources(tmp_path, monkeypatch, spec)
    pipeline_dir = tmp_path / "workspace" / "pipelines" / "external-v1"
    runtime = FakePipelineRuntime(spec, runtime_commit=canonical_runtime_commit)

    result = run_pipeline(spec_path, runtime.hooks())

    persisted = experiment_pipeline.read_rows(pipeline_dir / "jobs.tsv")[0]
    assert result["status"] == "completed"
    assert persisted["verified"] == "true"
    # The frozen snapshot and spec name runtime "a" * 40; the attempt projects only what the run committed.
    assert persisted["runtime_commit"] == canonical_runtime_commit
    # Only the registration and pre-launch probes inspect the live runtime; the terminal attempt is not probed.
    assert runtime.calls == [
        ("monitor", [spec["checkpoint_sources"]["age"]["plan"]]),
        ("inspect", ["run-000"]),
        ("inspect", ["run-000"]),
        ("launch", ["run-000"]),
    ]


def test_run_attempts_result_validation_failure_is_terminal_without_changing_canonical_status(
    tmp_path: Path, monkeypatch
):
    root = tmp_path / "workspace"
    spec = _spec(root)
    spec_path = prepare_pipeline_sources(tmp_path, monkeypatch, spec)
    pipeline_dir = root / "pipelines" / "external-v1"
    runtime = FakePipelineRuntime(spec)

    def launch_with_wrong_split(*args, **kwargs):
        launched = runtime.launch_runs(*args, **kwargs)
        for manifest_path in pipeline_dir.rglob("infer/run_manifest.json"):
            manifest = json.loads(manifest_path.read_text())
            manifest_path.write_text(json.dumps({**manifest, "eval_split": "val"}) + "\n")
        return launched

    hooks = runtime.hooks(launch_runs=launch_with_wrong_split, sleep=lambda _seconds: pytest.fail("must not poll"))
    result = run_pipeline(spec_path, hooks)

    persisted = experiment_pipeline.read_rows(pipeline_dir / "jobs.tsv")
    assert result["status"] == "failed"
    assert result["jobs"][0]["status"] == "failed"
    assert result["jobs"][0]["attempt_count"] == 1
    assert len(persisted) == 1
    assert persisted[0]["status"] == "completed"
    assert persisted[0]["verified"] == "false"
    assert persisted[0]["validation_error"] == "Inference result manifest label or split differs from the frozen job."
    assert [row["status"] for row in read_run_manifest(root) if row["step_id"] == "external-evaluate"] == ["completed"]
    assert json.loads((pipeline_dir / "pipeline.json").read_text())["status"] == "failed"

    with pytest.raises(ValueError, match="Failed pipelines are immutable"):
        run_pipeline(spec_path, runtime.hooks(), resume=True)
    assert experiment_pipeline.read_rows(pipeline_dir / "jobs.tsv") == persisted


def test_run_attempts_mixed_group_validates_live_snapshot_before_launch(tmp_path: Path, monkeypatch):
    spec = _two_job_spec(tmp_path / "workspace")
    spec_path = prepare_pipeline_sources(tmp_path, monkeypatch, spec)
    deferred = {"age-hsp-i2-bcg"}

    def outcome(run):
        if run["job_id"] in deferred:
            deferred.remove(run["job_id"])
            return None
        return "completed"

    runtime = FakePipelineRuntime(spec, outcome=outcome)

    result = run_pipeline(spec_path, runtime.hooks())

    assert result["status"] == "completed"
    group = ["run-000", "run-001"]
    # With one attempt terminal and its sibling still launchable, the whole frozen group is re-probed before launch.
    assert runtime.calls[1:] == [
        ("inspect", group),
        ("inspect", group),
        ("launch", group),
        ("inspect", group),
        ("launch", group),
    ]


def test_run_attempts_blocks_on_external_missing_pid_capacity_blocker(tmp_path: Path, monkeypatch):
    spec = _two_job_spec(tmp_path / "workspace")
    spec_path = prepare_pipeline_sources(tmp_path, monkeypatch, spec)
    pipeline_dir = tmp_path / "workspace" / "pipelines" / "external-v1"
    blocker = {"step_id": "train-age", "run_id": "run-099"}
    runtime = FakePipelineRuntime(spec, outcome=lambda run: "failed" if run["job_id"] == "age-hsp-i2-psg" else None)
    blocked = []

    def launch_until_blocked(*args, **kwargs):
        if not any(name == "launch" for name, _ids in runtime.calls):
            return runtime.launch_runs(*args, **kwargs)
        blocked.append(([str(run["run_id"]) for run in args[2]], kwargs["fail_on_missing_pid_blocker"]))
        raise managed_scheduler.MissingPidCapacityError(blocker["step_id"], blocker["run_id"])

    hooks = runtime.hooks(
        launch_runs=launch_until_blocked,
        sleep=lambda _seconds: pytest.fail("a missing_pid capacity blocker must not sleep"),
    )
    result = run_pipeline(spec_path, hooks)

    assert result["status"] == "blocked"
    assert result["missing_pid_blocker"] == {"status": "missing_pid", **blocker}
    # The failed attempt's retry forms its own scheduler group; the blocker stops before that group launches.
    assert blocked == [(["run-000", "run-001"], True)]
    assert runtime.calls[-2:] == [("inspect", ["run-002"]), ("inspect", ["run-000", "run-001"])]
    assert [
        (row["run_id"], row["attempt"], row["status"])
        for row in experiment_pipeline.read_rows(pipeline_dir / "jobs.tsv")
    ] == [("run-000", "1", "failed"), ("run-001", "1", "planned"), ("run-002", "2", "planned")]
    state = json.loads((pipeline_dir / "pipeline.json").read_text())
    assert state["status"] == "blocked"
    assert state["missing_pid_blocker"] == {"status": "missing_pid", **blocker}


def test_run_attempts_blocks_before_retry_when_external_run_has_missing_pid(tmp_path: Path, monkeypatch):
    root = tmp_path / "workspace"
    spec = _spec(root)
    spec_path = prepare_pipeline_sources(tmp_path, monkeypatch, spec)
    pipeline_dir = root / "pipelines" / "external-v1"
    rows = read_run_manifest(root)
    blocker = {**rows[0], "run_id": "run-099", "status": "missing_pid", "target": "local", "gpus": "0"}
    write_rows(root / "run_manifest.tsv", [*rows, blocker])
    runtime = FakePipelineRuntime(spec, outcome=lambda _run: "failed")

    result = run_pipeline(
        spec_path, runtime.hooks(sleep=lambda _seconds: pytest.fail("capacity blocker must not sleep"))
    )

    assert result["status"] == "blocked"
    assert result["missing_pid_blocker"] == {"status": "missing_pid", "step_id": "train-age", "run_id": "run-099"}
    persisted = experiment_pipeline.read_rows(pipeline_dir / "jobs.tsv")
    assert [(row["attempt"], row["status"]) for row in persisted] == [("1", "failed")]
    assert [name for name, _ids in runtime.calls].count("launch") == 1


def test_run_attempts_syncs_owned_missing_pid_and_blocks_pending_sibling(tmp_path: Path, monkeypatch):
    spec = _two_job_spec(tmp_path / "workspace")
    spec_path = prepare_pipeline_sources(tmp_path, monkeypatch, spec)
    pipeline_dir = tmp_path / "workspace" / "pipelines" / "external-v1"
    launches = []

    def lose_owned_pid(root, _owner_dir, runs, *_args, **kwargs):
        launches.append(kwargs["fail_on_missing_pid_blocker"])
        with managed_scheduler.managed_run_lock(root):
            row = next(row for row in read_run_manifest(root) if row["run_id"] == runs[0]["run_id"])
            merge_run_manifest(root, [{**row, "status": "missing_pid"}], lock_held=True)
        raise managed_scheduler.MissingPidCapacityError(row["step_id"], row["run_id"])

    hooks = FakePipelineRuntime(spec).hooks(
        launch_runs=lose_owned_pid,
        sleep=lambda _seconds: pytest.fail("an owned missing_pid attempt must not sleep"),
    )
    result = run_pipeline(spec_path, hooks)

    assert result["status"] == "blocked"
    assert result["missing_pid_blocker"] == {
        "status": "missing_pid",
        "step_id": "external-evaluate",
        "run_id": "run-000",
    }
    assert launches == [True]
    persisted = {row["run_id"]: row for row in experiment_pipeline.read_rows(pipeline_dir / "jobs.tsv")}
    assert set(persisted) == {"run-000", "run-001"}
    assert persisted["run-000"]["status"] == "missing_pid"
    assert persisted["run-001"]["status"] == "planned"


def test_execute_pipeline_persists_and_clears_missing_pid_blocker_on_resume(tmp_path: Path, monkeypatch):
    class ResumeObserved(Exception):
        pass

    spec = _spec(tmp_path / "workspace")
    spec_path = prepare_pipeline_sources(tmp_path, monkeypatch, spec)
    pipeline_dir = tmp_path / "workspace" / "pipelines" / "external-v1"
    runtime = FakePipelineRuntime(spec)
    blocker = {"status": "missing_pid", "step_id": "train-age", "run_id": "run-099"}

    def blocked_launch(*_args, **_kwargs):
        raise managed_scheduler.MissingPidCapacityError(blocker["step_id"], blocker["run_id"])

    result = run_pipeline(spec_path, runtime.hooks(launch_runs=blocked_launch))

    assert result["status"] == "blocked"
    assert result["missing_pid_blocker"] == blocker
    state = json.loads((pipeline_dir / "pipeline.json").read_text())
    assert state["status"] == "blocked"
    assert state["missing_pid_blocker"] == blocker
    assert state["logical_jobs"] == result["jobs"]

    def observe_resume(*_args, **_kwargs):
        resumed_state = json.loads((pipeline_dir / "pipeline.json").read_text())
        assert resumed_state["status"] == "running_external"
        assert resumed_state["missing_pid_blocker"] is None
        raise ResumeObserved

    with pytest.raises(ResumeObserved):
        run_pipeline(spec_path, runtime.hooks(launch_runs=observe_resume), resume=True)


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
        pipeline_attempts.planned_runs([attempt])


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

    run = pipeline_attempts.planned_runs([attempt])[0]

    assert run["checkpoint"] == attempt["checkpoint"]
    assert run["checkpoint_sha256"] == attempt["checkpoint_sha256"]


def test_frozen_pipeline_rejects_external_preset_byte_drift(tmp_path: Path, monkeypatch):
    spec = _spec(tmp_path / "workspace")
    spec_path = prepare_pipeline_sources(tmp_path, monkeypatch, spec)
    runtime = FakePipelineRuntime(spec)
    assert run_pipeline(spec_path, runtime.hooks())["status"] == "completed"
    calls = list(runtime.calls)

    Path(spec["jobs"][0]["inference_preset_path"]).write_bytes(b"changed-preset")

    with pytest.raises(ValueError, match="Frozen external preset changed"):
        run_pipeline(spec_path, runtime.hooks(), resume=True)
    assert runtime.calls == calls


@pytest.mark.parametrize("tamper", [False, True])
def test_orphan_checkpoint_selection_is_rederived_before_state_commit(tmp_path: Path, monkeypatch, tamper: bool):
    root = tmp_path / "workspace"
    spec = _spec(root)
    spec_path = prepare_pipeline_sources(tmp_path, monkeypatch, spec)
    pipeline_dir = root / "pipelines" / "external-v1"
    runtime = FakePipelineRuntime(spec)

    def interrupt_launch(*_args, **_kwargs):
        raise PipelineInterrupted

    with pytest.raises(PipelineInterrupted):
        run_pipeline(spec_path, runtime.hooks(launch_runs=interrupt_launch))

    # Leave checkpoints.json behind without its committed hash, as a runner killed between the two writes would.
    state_path = pipeline_dir / "pipeline.json"
    state = json.loads(state_path.read_text())
    del state["checkpoint_selection_sha256"]
    state_path.write_text(json.dumps(state) + "\n")
    checkpoints_path = pipeline_dir / "checkpoints.json"
    if tamper:
        payload = json.loads(checkpoints_path.read_text())
        payload["sources"][0]["score"] = 4.4
        checkpoints_path.write_text(json.dumps(payload) + "\n")

    if tamper:
        with pytest.raises(ValueError, match="differs from validation-derived selection"):
            run_pipeline(spec_path, runtime.hooks(), resume=True)
        assert "checkpoint_selection_sha256" not in json.loads(state_path.read_text())
    else:
        result = run_pipeline(spec_path, runtime.hooks(), resume=True)
        assert result["status"] == "completed"
        assert json.loads(state_path.read_text())["checkpoint_selection_sha256"] == file_sha256(checkpoints_path)
        events = [
            event
            for event in pipeline_attempts.read_experiment_events(root)
            if event.get("event_type") == "pipeline_checkpoints_frozen"
        ]
        assert len(events) == 1


def test_completed_pipeline_resume_validates_and_finalizes_without_reexecution(tmp_path: Path, monkeypatch):
    root = tmp_path / "workspace"
    spec = _spec(root)
    spec_path = prepare_pipeline_sources(tmp_path, monkeypatch, spec)
    runtime = FakePipelineRuntime(spec)
    finalized = []
    first = run_pipeline(spec_path, runtime.hooks(), finalized=finalized)
    report = Path(first["report"])
    calls = list(runtime.calls)
    finalized.clear()

    result = run_pipeline(spec_path, runtime.hooks(), resume=True, finalized=finalized)

    assert result["status"] == "completed"
    assert finalized == [(root, report)]
    # Completed pipelines neither recheck training sources nor probe, launch or poll external attempts.
    assert runtime.calls == calls

    finalized.clear()
    complete_experiment(root)
    result = run_pipeline(spec_path, runtime.hooks(), resume=True, finalized=finalized)
    assert result["status"] == "completed"
    assert finalized == []
    assert runtime.calls == calls


def test_completed_event_append_failure_resumes_before_finalization(tmp_path: Path, monkeypatch):
    root = tmp_path / "workspace"
    spec = _spec(root)
    spec_path = prepare_pipeline_sources(tmp_path, monkeypatch, spec)
    pipeline_dir = root / "pipelines" / "external-v1"
    runtime = FakePipelineRuntime(spec)
    # Completion events and finalization calls are recorded in one ordered history.
    order: list = []
    original_append = pipeline_attempts.append_event
    interrupted = {"value": True}

    def append(root_path, event_type, payload):
        if event_type == "pipeline_completed":
            if interrupted["value"]:
                raise RuntimeError("event append interrupted")
            order.append("event")
        original_append(root_path, event_type, payload)

    monkeypatch.setattr(pipeline_attempts, "append_event", append)

    with pytest.raises(pipeline_attempts.PipelineRegistrationRecoveryError, match="reconciled on resume"):
        run_pipeline(spec_path, runtime.hooks(), finalized=order)

    state = json.loads((pipeline_dir / "pipeline.json").read_text())
    assert state["status"] == "completed"
    assert state["result_artifacts"][str(pipeline_dir / "final.md")] == file_sha256(pipeline_dir / "final.md")
    assert order == []
    calls = list(runtime.calls)

    interrupted["value"] = False
    result = run_pipeline(spec_path, runtime.hooks(), resume=True, finalized=order)

    assert result["status"] == "completed"
    assert order == ["event", (root, pipeline_dir / "final.md")]
    assert runtime.calls == calls
    events_before = (root / "events.jsonl").read_bytes()

    complete_experiment(root)
    order.clear()
    result = run_pipeline(spec_path, runtime.hooks(), resume=True, finalized=order)

    assert result["status"] == "completed"
    assert order == []
    assert (root / "events.jsonl").read_bytes() == events_before
