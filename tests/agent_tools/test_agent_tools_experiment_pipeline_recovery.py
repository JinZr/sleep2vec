from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import shlex
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
    experiment_workspace,
    experiments,
    managed_scheduler,
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

    # Observation only: no public hook runs inside the publication lock, so these wrappers record its nesting.
    monkeypatch.setattr(pipeline_attempts, "plan_publication_lock", publication_lock)
    monkeypatch.setattr(pipeline_attempts, "_materialize_attempt_locked", materialize_locked)

    result = run_pipeline(spec_path, FakePipelineRuntime(spec).hooks())

    assert result["status"] == "completed"
    assert materialized == [pipeline_dir / "plans" / "age-hsp-i2-psg" / "attempt-001"]


def test_pipeline_group_registration_waits_for_ordinary_plan(tmp_path: Path, monkeypatch):
    root = tmp_path / "workspace"
    spec = _spec(root)
    spec_path = prepare_pipeline_sources(tmp_path, monkeypatch, spec)
    runtime = FakePipelineRuntime(spec)

    def interrupt_registration(*_args, **_kwargs):
        raise PipelineInterrupted

    # Freeze the checkpoint selection first, so the resumed runner's next registration lock is its group registration.
    with pytest.raises(PipelineInterrupted):
        run_pipeline(spec_path, runtime.hooks(inspect_target=interrupt_registration))

    ordinary_recipe = write_finetune_recipe(root)
    recipe = yaml.safe_load(ordinary_recipe.read_text())
    experiment = yaml.safe_load((root / "experiment.yaml").read_text())["experiment"]
    recipe["experiment"] = {field: experiment[field] for field in ("id", "title", "objective", "root", "baseline")}
    ordinary_recipe.write_text(yaml.safe_dump(recipe, sort_keys=False))
    ordinary_plan = root / "plans" / "ordinary"
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

    def prepare_registration(*args, **kwargs):
        pipeline_preparing.set()
        return runtime.inspect_target(*args, **kwargs)

    # No public seam holds an ordinary planner inside the registration lock; the pause keeps its real check.
    monkeypatch.setattr(plans, "_assert_no_incomplete_step_registration", pause_ordinary)
    ordinary_reports = []
    errors = []
    pipeline_results = []

    def run_ordinary():
        try:
            ordinary_reports.append(plans.build_plan(recipe_path=ordinary_recipe, output_dir=ordinary_plan))
        except BaseException as exc:
            errors.append(exc)

    def resume_pipeline():
        try:
            pipeline_results.append(
                run_pipeline(spec_path, runtime.hooks(inspect_target=prepare_registration), resume=True)
            )
        except BaseException as exc:
            errors.append(exc)

    ordinary = threading.Thread(target=run_ordinary)
    pipeline = threading.Thread(target=resume_pipeline)
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
    assert pipeline_results[0]["status"] == "completed"
    assert [job["job_id"] for job in pipeline_results[0]["jobs"]] == [spec["jobs"][0]["id"]]


def test_initial_jobs_projection_failure_is_recoverable(tmp_path: Path, monkeypatch):
    root = tmp_path / "workspace"
    spec = _spec(root)
    spec_path = prepare_pipeline_sources(tmp_path, monkeypatch, spec)
    pipeline_dir = root / "pipelines" / "external-v1"
    jobs_path = pipeline_dir / "jobs.tsv"
    runtime = FakePipelineRuntime(spec)

    def block_jobs_projection(*args, **kwargs):
        # The registration probe precedes the canonical commit; a directory at jobs.tsv then makes the atomic
        # projection replace after that commit fail.
        jobs_path.mkdir()
        return runtime.inspect_target(*args, **kwargs)

    with pytest.raises(pipeline_attempts.PipelineRegistrationRecoveryError, match="reconciled on resume"):
        run_pipeline(spec_path, runtime.hooks(inspect_target=block_jobs_projection))

    def evaluation_runs():
        return [row["run_id"] for row in read_run_manifest(root) if row["step_id"] == "external-evaluate"]

    assert evaluation_runs() == ["run-000"]
    assert json.loads((pipeline_dir / "pipeline.json").read_text())["status"] == "ready"
    assert not list(pipeline_dir.glob(".jobs.tsv.*.tmp"))

    jobs_path.rmdir()
    result = run_pipeline(spec_path, runtime.hooks(), resume=True)

    assert result["status"] == "completed"
    assert [row["run_id"] for row in experiment_pipeline.read_rows(jobs_path)] == ["run-000"]
    assert evaluation_runs() == ["run-000"]


def test_registered_jobs_retry_cleans_interrupted_atomic_temp(tmp_path: Path, monkeypatch):
    root = tmp_path / "workspace"
    spec = _spec(root)
    spec_path = prepare_pipeline_sources(tmp_path, monkeypatch, spec)
    pipeline_dir = root / "pipelines" / "external-v1"
    jobs_path = pipeline_dir / "jobs.tsv"
    runtime = FakePipelineRuntime(spec)
    replace = experiment_pipeline_results.os.replace
    interrupted = []

    def interrupt_first_jobs_replace(source, destination, **kwargs):
        if Path(destination) == jobs_path and not interrupted:
            interrupted.append(Path(source))
            raise OSError("jobs projection interrupted")
        replace(source, destination, **kwargs)

    # A directory at jobs.tsv fails this replace for good (see the projection tests); this interrupts it once.
    monkeypatch.setattr(experiment_pipeline_results.os, "replace", interrupt_first_jobs_replace)

    with pytest.raises(
        pipeline_attempts.PipelineRegistrationRecoveryError, match="jobs projection must be reconciled on resume"
    ):
        run_pipeline(spec_path, runtime.hooks())

    assert [path.parent for path in interrupted] == [pipeline_dir]
    assert not jobs_path.exists()
    assert not list(pipeline_dir.glob(".jobs.tsv.*.tmp"))
    result = run_pipeline(spec_path, runtime.hooks(), resume=True)
    assert result["status"] == "completed"
    assert [(row["run_id"], row["attempt"]) for row in experiment_pipeline.read_rows(jobs_path)] == [("run-000", "1")]


def test_pipeline_jobs_planned_event_is_reconciled_after_append_failure(tmp_path: Path, monkeypatch):
    root = tmp_path / "workspace"
    spec = _spec(root)
    spec_path = prepare_pipeline_sources(tmp_path, monkeypatch, spec)
    runtime = FakePipelineRuntime(spec)
    real_append = pipeline_attempts.append_event
    interrupted = []

    def interrupt_jobs_event(root_path, event_type, payload):
        if event_type == "pipeline_jobs_planned" and not interrupted:
            interrupted.append(payload)
            raise RuntimeError("event append interrupted")
        real_append(root_path, event_type, payload)

    # A directory or corrupt events.jsonl fails the reconcile's read with a fatal ValueError; interrupt this append.
    monkeypatch.setattr(pipeline_attempts, "append_event", interrupt_jobs_event)

    def jobs_planned_events():
        events = pipeline_attempts.read_experiment_events(root)
        return [event for event in events if event.get("event_type") == "pipeline_jobs_planned"]

    def interrupt_launch(*_args, **_kwargs):
        raise PipelineInterrupted

    with pytest.raises(
        pipeline_attempts.PipelineRegistrationRecoveryError,
        match="pipeline_jobs_planned event must be reconciled on resume",
    ):
        run_pipeline(spec_path, runtime.hooks())
    assert len(interrupted) == 1
    assert jobs_planned_events() == []

    # Each resumed registration reconciles the event before launch: first by appending it, then by matching it.
    with pytest.raises(PipelineInterrupted):
        run_pipeline(spec_path, runtime.hooks(launch_runs=interrupt_launch), resume=True)
    assert len(jobs_planned_events()) == 1
    assert run_pipeline(spec_path, runtime.hooks(), resume=True)["status"] == "completed"

    events = jobs_planned_events()
    assert len(events) == 1
    assert events[0]["pipeline_id"] == spec["pipeline"]["id"]
    assert events[0]["job_count"] == len(spec["jobs"])


@pytest.mark.parametrize("failed_read", [1, 2])
def test_pipeline_jobs_planned_event_read_failure_is_recoverable(tmp_path: Path, monkeypatch, failed_read: int):
    root = tmp_path / "workspace"
    spec = _spec(root)
    spec_path = prepare_pipeline_sources(tmp_path, monkeypatch, spec)
    pipeline_dir = root / "pipelines" / "external-v1"
    jobs_path = pipeline_dir / "jobs.tsv"
    runtime = FakePipelineRuntime(spec)
    real_read = pipeline_attempts.read_experiment_events
    reads = []

    def read_events(root_path):
        # After the jobs projection, the registration's event reconcile reads once before its append and once after.
        if jobs_path.exists() and len(reads) < failed_read:
            reads.append(root_path)
            if len(reads) == failed_read:
                raise OSError("event read interrupted")
        return real_read(root_path)

    # A directory or corrupt events.jsonl fails every read with a fatal ValueError; interrupt the chosen read instead.
    monkeypatch.setattr(pipeline_attempts, "read_experiment_events", read_events)

    def jobs_planned_events():
        return [event for event in real_read(root) if event.get("event_type") == "pipeline_jobs_planned"]

    with pytest.raises(
        pipeline_attempts.PipelineRegistrationRecoveryError,
        match="pipeline_jobs_planned event must be reconciled on resume",
    ):
        run_pipeline(spec_path, runtime.hooks())

    assert len(reads) == failed_read
    # Only the read after the append leaves the event committed; resume must match it rather than append it again.
    assert len(jobs_planned_events()) == failed_read - 1
    assert json.loads((pipeline_dir / "pipeline.json").read_text())["status"] == "ready"
    assert run_pipeline(spec_path, runtime.hooks(), resume=True)["status"] == "completed"
    assert len(jobs_planned_events()) == 1


def test_pipeline_retry_planned_event_is_reconciled_after_append_failure(tmp_path: Path, monkeypatch):
    root = tmp_path / "workspace"
    spec = _spec(root)
    spec_path = prepare_pipeline_sources(tmp_path, monkeypatch, spec)
    pipeline_dir = root / "pipelines" / "external-v1"
    runtime = FakePipelineRuntime(spec, outcome=lambda run: "failed" if run["attempt"] == 1 else "completed")
    real_append = pipeline_attempts.append_event
    interrupted = []

    def interrupt_retry_event(root_path, event_type, payload):
        if event_type == "pipeline_job_retry_planned" and not interrupted:
            interrupted.append(payload)
            raise RuntimeError("event append interrupted")
        real_append(root_path, event_type, payload)

    # A directory or corrupt events.jsonl fails the reconcile's read with a fatal ValueError; interrupt this append.
    monkeypatch.setattr(pipeline_attempts, "append_event", interrupt_retry_event)

    def retry_planned_events():
        events = pipeline_attempts.read_experiment_events(root)
        return [event for event in events if event.get("event_type") == "pipeline_job_retry_planned"]

    def interrupt_poll(_seconds):
        raise PipelineInterrupted

    with pytest.raises(
        pipeline_attempts.PipelineRegistrationRecoveryError,
        match="pipeline_job_retry_planned event must be reconciled on resume",
    ):
        run_pipeline(spec_path, runtime.hooks())
    assert len(interrupted) == 1
    assert retry_planned_events() == []
    assert [row["attempt"] for row in experiment_pipeline.read_rows(pipeline_dir / "jobs.tsv")] == ["1", "2"]

    # Each resumed poll reconciles the event: first by appending it while the retry waits, then by matching it.
    waiting = FakePipelineRuntime(spec, outcome=lambda _run: None)
    with pytest.raises(PipelineInterrupted):
        run_pipeline(spec_path, waiting.hooks(sleep=interrupt_poll), resume=True)
    assert len(retry_planned_events()) == 1
    assert run_pipeline(spec_path, runtime.hooks(), resume=True)["status"] == "completed"

    events = retry_planned_events()
    assert len(events) == 1
    assert events[0]["pipeline_id"] == spec["pipeline"]["id"]
    assert events[0]["job_id"] == spec["jobs"][0]["id"]
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
    interrupted = []

    def interrupt_selection_event(root_path, appended_type, payload):
        if appended_type == event_type and not interrupted:
            interrupted.append(payload)
            raise RuntimeError("event append interrupted")
        original_append(root_path, appended_type, payload)

    # A directory or corrupt events.jsonl fails the reconcile's read with a fatal ValueError; interrupt this append.
    monkeypatch.setattr(pipeline_attempts, "append_event", interrupt_selection_event)

    with pytest.raises(
        pipeline_attempts.PipelineRegistrationRecoveryError,
        match=f"{event_type} event must be reconciled on resume",
    ):
        run_pipeline(spec_path, runtime.hooks())

    assert len(interrupted) == 1
    selection_path = pipeline_dir / ("candidates.json" if kind == "cohort_selection" else "checkpoints.json")
    hash_field = "candidate_selection_sha256" if kind == "cohort_selection" else "checkpoint_selection_sha256"
    state = json.loads((pipeline_dir / "pipeline.json").read_text())
    assert state[hash_field] == file_sha256(selection_path)
    assert state["status"] == "ready"
    assert not any(name == "inspect" for name, _ids in runtime.calls)

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

    # A directory or corrupt events.jsonl fails the reconcile's read with a fatal ValueError; interrupt this append.
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
    spec = _spec(root)
    prepare_pipeline_sources(tmp_path, monkeypatch, spec, scores=(4.5, 4.6))
    unsuccessful, successful = read_run_manifest(root)
    # A failure manifest makes any success inspection of the unsuccessful run fail loudly.
    (Path(unsuccessful["runtime_dir"]) / "run_manifest.json").write_text(json.dumps({"status": "failed"}) + "\n")
    (Path(successful["runtime_dir"]) / "run_manifest.json").write_text(
        json.dumps({"status": "skipped_test", "metrics": {"val_mae": 4.5}}) + "\n"
    )
    write_rows(
        root / "run_manifest.tsv",
        [
            {
                **unsuccessful,
                "status": failed_status,
                **({"stop_reason": "stopped after invalid candidate"} if failed_status == "stopped" else {}),
            },
            {**successful, "status": "finished"},
        ],
    )

    result = dry_run_pipeline(root, spec)

    states = result["source_states"]
    assert states[0]["complete"] is True
    assert states[0]["failed_runs"] == ["run-001"]
    assert result["status"] == "ready"


def test_mixed_terminal_source_rejects_stopped_run_without_reason(tmp_path: Path, monkeypatch):
    root = tmp_path / "workspace"
    spec = _spec(root)
    prepare_pipeline_sources(tmp_path, monkeypatch, spec, scores=(4.5, 4.6))
    stopped, finished = read_run_manifest(root)
    write_rows(root / "run_manifest.tsv", [{**stopped, "status": "stopped"}, {**finished, "status": "finished"}])

    with pytest.raises(ValueError, match="Stopped source runs are missing required stop_reason.*run-001"):
        dry_run_pipeline(root, spec)


@pytest.mark.parametrize("status", ["planned", "running"])
def test_active_source_waits_for_terminal_status(tmp_path: Path, monkeypatch, status: str):
    root = tmp_path / "workspace"
    spec = _spec(root)
    prepare_pipeline_sources(tmp_path, monkeypatch, spec)
    (run,) = read_run_manifest(root)
    write_rows(root / "run_manifest.tsv", [{**run, "status": status}])

    result = dry_run_pipeline(root, spec)

    assert result["source_states"][0]["complete"] is False
    assert result["status"] == "waiting_for_sources"


def test_all_unsuccessful_terminal_source_fails(tmp_path: Path, monkeypatch):
    root = tmp_path / "workspace"
    spec = _spec(root)
    prepare_pipeline_sources(tmp_path, monkeypatch, spec, scores=(4.5, 4.6))
    failed, stopped = read_run_manifest(root)
    write_rows(
        root / "run_manifest.tsv",
        [{**failed, "status": "failed"}, {**stopped, "status": "stopped", "stop_reason": "budget exhausted"}],
    )

    result = dry_run_pipeline(root, spec)

    assert result["source_states"][0]["complete"] is False
    assert result["status"] == "failed"


@pytest.mark.parametrize("status", ["submitting", "unknown_scheduler"])
def test_slurm_source_uncertainty_blocks_external_pipeline(tmp_path: Path, monkeypatch, status: str):
    root = tmp_path / "workspace"
    spec = _spec(root)
    prepare_pipeline_sources(tmp_path, monkeypatch, spec)
    (run,) = read_run_manifest(root)
    write_rows(root / "run_manifest.tsv", [{**run, "status": status}])

    result = dry_run_pipeline(root, spec)

    assert result["source_states"][0]["uncertain_runs"] == ["run-001"]
    assert result["status"] == "blocked"


def test_retry_preflight_failure_does_not_block_independent_retry(tmp_path: Path, monkeypatch):
    root = tmp_path / "workspace"
    spec = _two_job_spec(root)
    spec_path = prepare_pipeline_sources(tmp_path, monkeypatch, spec)
    pipeline_dir = root / "pipelines" / "external-v1"
    # Another step already registered psg's retry plan directory, so its real retry preflight fails.
    commit_step_manifest(
        root,
        {
            "step": {"id": "foreign-evaluate", "phase": "evaluate", "purpose": "Own the psg retry output."},
            "experiment_id": "unit",
            "plan_controller": "ordinary",
            "recipe_path": "",
            "plans": [str(pipeline_dir / "plans" / "age-hsp-i2-psg" / "attempt-002")],
        },
    )
    runtime = FakePipelineRuntime(spec, outcome=lambda run: "failed" if run["attempt"] == 1 else "completed")
    retry_states = []

    def launch(root_path, owner_dir, runs, *args, **kwargs):
        if [run["job_id"] for run in runs] == ["age-hsp-i2-bcg"] and runs[0]["attempt"] == 2 and not retry_states:
            rows = experiment_pipeline.read_rows(pipeline_dir / "jobs.tsv")
            retry_states.append([job["status"] for job in experiment_pipeline_results.logical_job_states(spec, rows)])
        return runtime.launch_runs(root_path, owner_dir, runs, *args, **kwargs)

    result = run_pipeline(spec_path, runtime.hooks(launch_runs=launch))

    assert retry_states == [["failed", "running"]]
    psg, bcg = result["jobs"]
    assert result["status"] == "failed"
    assert (psg["status"], psg["attempt_count"]) == ("failed", 1)
    assert psg["retry_preparation_error"] == "Retry preflight failed for external job age-hsp-i2-psg."
    assert (bcg["status"], bcg["attempt_count"]) == ("completed", 2)
    rows = experiment_pipeline.read_rows(pipeline_dir / "jobs.tsv")
    assert [row["attempt"] for row in rows if row["job_id"] == "age-hsp-i2-bcg"] == ["1", "2"]
    retry_events = [
        (event["event_type"], event["job_id"])
        for event in pipeline_attempts.read_experiment_events(root)
        if event.get("event_type") in {"pipeline_job_retry_preflight_failed", "pipeline_job_retry_planned"}
    ]
    assert retry_events == [
        ("pipeline_job_retry_preflight_failed", "age-hsp-i2-psg"),
        ("pipeline_job_retry_planned", "age-hsp-i2-bcg"),
    ]
    assert not (pipeline_dir / "preflight_retries" / "age-hsp-i2-psg").exists()
    assert (pipeline_dir / "preflight_retries" / "age-hsp-i2-bcg" / "attempt-002.json").is_file()


def test_retry_registration_preflight_failure_does_not_block_independent_retry(tmp_path: Path, monkeypatch):
    root = tmp_path / "workspace"
    spec = _two_job_spec(root)
    spec_path = prepare_pipeline_sources(tmp_path, monkeypatch, spec)
    pipeline_dir = root / "pipelines" / "external-v1"
    stale_output = pipeline_dir / "results" / "age-hsp-i2-psg" / "attempt-002"
    runtime = FakePipelineRuntime(spec, outcome=lambda run: "failed" if run["attempt"] == 1 else "completed")
    recorded_before_bcg = []

    def launch(*args, **kwargs):
        launched = runtime.launch_runs(*args, **kwargs)
        if not stale_output.exists():
            # Output left where psg's retry must write makes its registration preflight fail for real.
            stale_output.mkdir(parents=True)
            (stale_output / "metrics.csv").write_text("stale\n")
        return launched

    def inspect(execution, runs, **kwargs):
        if [run["run_id"] for run in runs] == ["run-002"] and not recorded_before_bcg:
            rows = experiment_pipeline.read_rows(pipeline_dir / "jobs.tsv")
            recorded_before_bcg.extend(row.get("retry_preparation_error", "") for row in rows)
        return runtime.inspect_target(execution, runs, **kwargs)

    result = run_pipeline(spec_path, runtime.hooks(inspect_target=inspect, launch_runs=launch))

    expected_error = (
        "External attempt registration preflight failed: Managed attempt output must be a new empty directory: "
        f"{stale_output}"
    )
    assert recorded_before_bcg == [expected_error, ""]
    psg, bcg = result["jobs"]
    assert result["status"] == "failed"
    assert (psg["status"], psg["retry_preparation_error"]) == ("failed", expected_error)
    assert bcg["status"] == "completed"
    rows = experiment_pipeline.read_rows(pipeline_dir / "jobs.tsv")
    assert [row["attempt"] for row in rows if row["job_id"] == "age-hsp-i2-psg"] == ["1"]
    assert [row["attempt"] for row in rows if row["job_id"] == "age-hsp-i2-bcg"] == ["1", "2"]
    assert not (pipeline_dir / "plans" / "age-hsp-i2-psg" / "attempt-002").exists()
    assert not list(pipeline_dir.rglob("*.staging"))


def test_retry_registration_failure_is_not_recorded_as_preflight_failure(tmp_path: Path, monkeypatch):
    root = tmp_path / "workspace"
    spec = _spec(root)
    spec_path = prepare_pipeline_sources(tmp_path, monkeypatch, spec)
    pipeline_dir = root / "pipelines" / "external-v1"
    retry_plan_dir = pipeline_dir / "plans" / "age-hsp-i2-psg" / "attempt-002"
    runtime = FakePipelineRuntime(spec, outcome=lambda _run: "failed")

    def inspect(execution, runs, **kwargs):
        snapshot = runtime.inspect_target(execution, runs, **kwargs)
        if [run["run_id"] for run in runs] == ["run-001"]:
            # An incomplete plan directory appearing after the registration preflight fails the canonical commit.
            retry_plan_dir.mkdir()
        return snapshot

    with pytest.raises(ValueError, match="External attempt plan is incomplete"):
        run_pipeline(spec_path, runtime.hooks(inspect_target=inspect))

    rows = experiment_pipeline.read_rows(pipeline_dir / "jobs.tsv")
    assert "retry_preparation_error" not in rows[0]
    assert len(rows) == 1
    assert [row["attempt"] for row in read_run_manifest(root) if row["step_id"] == "external-evaluate"] == ["1"]
    assert not list(pipeline_dir.rglob("*.staging"))


def test_retry_jobs_projection_failure_is_recoverable(tmp_path: Path, monkeypatch):
    root = tmp_path / "workspace"
    spec = _spec(root)
    spec_path = prepare_pipeline_sources(tmp_path, monkeypatch, spec)
    pipeline_dir = root / "pipelines" / "external-v1"
    jobs_path = pipeline_dir / "jobs.tsv"
    runtime = FakePipelineRuntime(spec, outcome=lambda run: "failed" if run["attempt"] == 1 else "completed")
    previous_projection = []

    def block_retry_projection(execution, runs, **kwargs):
        if [run["run_id"] for run in runs] == ["run-001"] and not previous_projection:
            # The retry's registration probe precedes its canonical commit; a directory at jobs.tsv then makes the
            # atomic projection replace after that commit fail.
            previous_projection.append(jobs_path.read_bytes())
            jobs_path.unlink()
            jobs_path.mkdir()
        return runtime.inspect_target(execution, runs, **kwargs)

    with pytest.raises(pipeline_attempts.PipelineRegistrationRecoveryError, match="reconciled on resume"):
        run_pipeline(spec_path, runtime.hooks(inspect_target=block_retry_projection))

    def evaluation_attempts():
        return [row["attempt"] for row in read_run_manifest(root) if row["step_id"] == "external-evaluate"]

    assert evaluation_attempts() == ["1", "2"]
    assert json.loads((pipeline_dir / "pipeline.json").read_text())["status"] == "running_external"
    assert not list(pipeline_dir.glob(".jobs.tsv.*.tmp"))

    # A failed atomic replace leaves the previous projection in place.
    jobs_path.rmdir()
    jobs_path.write_bytes(previous_projection[0])
    result = run_pipeline(spec_path, runtime.hooks(), resume=True)

    assert result["status"] == "completed"
    assert [row["attempt"] for row in experiment_pipeline.read_rows(jobs_path)] == ["1", "2"]
    assert evaluation_attempts() == ["1", "2"]


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
    preset = Path(spec["jobs"][0]["inference_preset_path"])
    preset.parent.mkdir(parents=True)
    preset.write_bytes(b"preset")
    config = Path(yaml.safe_load(write_finetune_recipe(tmp_path / "source").read_text())["inputs"]["config"])
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    selection_fields = {
        "config": str(config),
        "config_sha256": file_sha256(config),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "label_name": "age",
    }
    selections = {
        "age": {**selection_fields, "variant": "sleep2vec2"},
        "age-root": {**selection_fields, "variant": "sleep2vec"},
    }
    bcg_result_root = pipeline_dir / "results" / "age-hsp-i2-bcg" / "attempt-001"
    if failure_kind == "topology":
        # A leftover empty result root passes the attempt path check, but not its group's output topology.
        bcg_result_root.mkdir(parents=True)
        expected_error = f"Managed output paths must be independent regular files: {bcg_result_root}"
    else:
        expected_error = "frozen argv rejected"
    runtime = FakePipelineRuntime(spec)
    target_calls = []
    topology_calls = []
    real_validate_paths = experiment_pipeline.exp_io.validate_managed_output_paths

    def observe_topology(root_path, paths, *, remote=None):
        if Path(root_path) == Path("/"):
            topology_calls.append([Path(path) for path in paths])
        return real_validate_paths(root_path, paths, remote=remote)

    def reject_second_group(execution, runs, *, plan_label):
        assert plan_label == "pipeline"
        assert all(Path(run["script"]).is_file() for run in runs)
        target_calls.append([run["run_id"] for run in runs])
        if failure_kind == "target" and len(runs) == 2:
            raise ValueError("frozen argv rejected")
        return runtime.inspect_target(execution, runs, plan_label=plan_label)

    # Observation only: each variant group's output topology still gets its real validation.
    monkeypatch.setattr(experiment_pipeline.exp_io, "validate_managed_output_paths", observe_topology)

    with pytest.raises(pipeline_attempts.AttemptRegistrationPreflightError) as excinfo:
        pipeline_attempts.load_or_create_initial_attempts(
            root, pipeline_dir, spec, selections, inspect_target=reject_second_group
        )

    assert str(excinfo.value) == f"External attempt registration preflight failed: {expected_error}"
    assert [len(paths) for paths in topology_calls] == [3, 5]
    expected_target_calls = [["run-002"]] if failure_kind == "topology" else [["run-002"], ["run-000", "run-001"]]
    assert target_calls == expected_target_calls
    assert not (pipeline_dir / "jobs.tsv").exists()
    assert not (root / "steps").exists()
    assert read_run_manifest(root) == []
    assert not list((pipeline_dir / "plans").rglob("attempt-001"))
    assert not list(pipeline_dir.rglob("*.staging"))
    assert not list(pipeline_dir.rglob(managed_scheduler.EXECUTION_SNAPSHOT_NAME))


@pytest.mark.parametrize("drift", [False, True])
def test_registration_preflight_freezes_complete_group_and_rejects_drift(tmp_path: Path, monkeypatch, drift: bool):
    root = tmp_path / "workspace"
    spec = _two_job_spec(root)
    spec_path = prepare_pipeline_sources(tmp_path, monkeypatch, spec)
    pipeline_dir = root / "pipelines" / "external-v1"
    jobs_path = pipeline_dir / "jobs.tsv"
    snapshot_path = pipeline_dir / managed_scheduler.EXECUTION_SNAPSHOT_NAME
    runtime = FakePipelineRuntime(spec)
    target_snapshot = {"validated_argv_sha256": "a" * 64}

    def block_jobs_projection(_execution, _runs, *, plan_label):
        assert plan_label == "pipeline"
        # The registration probe precedes psg's canonical commit; a directory at jobs.tsv then fails its projection,
        # so bcg is left pending with psg's registration complete.
        jobs_path.mkdir()
        return dict(target_snapshot)

    with pytest.raises(pipeline_attempts.PipelineRegistrationRecoveryError, match="reconciled on resume"):
        run_pipeline(spec_path, runtime.hooks(inspect_target=block_jobs_projection))

    def evaluation_runs():
        return [row["run_id"] for row in read_run_manifest(root) if row["step_id"] == "external-evaluate"]

    assert evaluation_runs() == ["run-000"]
    assert json.loads(snapshot_path.read_text()) == target_snapshot
    jobs_path.rmdir()
    registration_probes = []

    def inspect(_execution, runs, *, plan_label="managed"):
        if plan_label == "pipeline":
            assert all(Path(run["script"]).is_file() for run in runs)
            scripts = [Path(run["script"]) for run in runs]
            registration_probes.append(
                ([run["run_id"] for run in runs], scripts, list(pipeline_dir.rglob("*.staging")))
            )
        return dict(target_snapshot)

    if drift:
        target_snapshot["validated_argv_sha256"] = "b" * 64
        with pytest.raises(pipeline_attempts.AttemptRegistrationPreflightError) as excinfo:
            run_pipeline(spec_path, runtime.hooks(inspect_target=inspect), resume=True)
        assert str(excinfo.value) == (
            "External attempt registration preflight failed: "
            f"Frozen pipeline execution snapshot changed: {snapshot_path}"
        )
        assert evaluation_runs() == ["run-000"]
        assert not jobs_path.exists()
    else:
        result = run_pipeline(spec_path, runtime.hooks(inspect_target=inspect), resume=True)
        assert result["status"] == "completed"
        assert [row["run_id"] for row in experiment_pipeline.read_rows(jobs_path)] == ["run-000", "run-001"]
        assert evaluation_runs() == ["run-000", "run-001"]

    # The resumed registration probes the complete group: psg from its canonical plan, bcg from its only staging.
    ((run_ids, (psg_script, bcg_script), staging_dirs),) = registration_probes
    assert run_ids == ["run-000", "run-001"]
    assert psg_script.is_relative_to(pipeline_dir / "plans" / "age-hsp-i2-psg" / "attempt-001")
    assert staging_dirs == [bcg_script.parents[2]]
    assert staging_dirs[0].parent == pipeline_dir / "plans" / "age-hsp-i2-bcg"
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
    spec = _spec(root)
    spec_path = prepare_pipeline_sources(tmp_path, monkeypatch, spec)
    pipeline_dir = root / "pipelines" / "external-v1"
    launched = []

    def launch_with_unsafe_identity(root_path, _owner_dir, runs, *_args, **_kwargs):
        launched.append([run["run_id"] for run in runs])
        canonical = {
            row["run_id"]: row for row in read_run_manifest(root_path) if row["step_id"] == "external-evaluate"
        }
        # The failed run's process observation could not prove it was the launched process.
        updates = [
            {**canonical[run["run_id"]], "status": "failed", "process_identity_error": identity_error} for run in runs
        ]
        return SimpleNamespace(committed_rows=merge_run_manifest(root_path, updates))

    result = run_pipeline(spec_path, FakePipelineRuntime(spec).hooks(launch_runs=launch_with_unsafe_identity))

    assert launched == [["run-000"]]
    assert result["status"] == "blocked"
    rows = experiment_pipeline.read_rows(pipeline_dir / "jobs.tsv")
    assert [row["attempt"] for row in rows] == ["1"]
    assert rows[0]["retry_blocker"] == f"unsafe process identity: {identity_error}"
    assert experiment_pipeline_results.logical_job_states(spec, rows)[0]["status"] == "blocked"
    assert not (pipeline_dir / "recipes" / "age-hsp-i2-psg" / "attempt-002.yaml").exists()
    blocked_events = [
        (event["job_id"], event["attempt"], event["reason"])
        for event in pipeline_attempts.read_experiment_events(root)
        if event.get("event_type") == "pipeline_job_retry_blocked"
    ]
    assert blocked_events == [("age-hsp-i2-psg", 1, "unsafe_process_identity")]


def test_atomic_generic_plan_freezes_single_runtime_command(tmp_path: Path, monkeypatch):
    root = tmp_path / "workspace"
    spec = _spec(root)
    spec_path = prepare_pipeline_sources(tmp_path, monkeypatch, spec)
    pipeline_dir = root / "pipelines" / "external-v1"
    runtime_commit = spec["runtime"]["runtime_commit"]

    def interrupt_launch(*_args, **_kwargs):
        raise PipelineInterrupted

    # Launch is reached only after the pipeline validated the published plan against its canonical attempt row.
    with pytest.raises(PipelineInterrupted):
        run_pipeline(spec_path, FakePipelineRuntime(spec).hooks(launch_runs=interrupt_launch))

    plan_dir = pipeline_dir / "plans" / "age-hsp-i2-psg" / "attempt-001"
    assert plan_dir.is_dir()
    assert not list(pipeline_dir.rglob("*.staging"))
    plan = json.loads((plan_dir / "plan.json").read_text())
    (planned,) = plan["runs"]
    command = planned["command"]
    assert plan["commands"] == [command]
    assert command.startswith("/runtime/python -m sleep2vec2.infer ")
    script_lines = Path(planned["script"]).read_text().splitlines()
    assert script_lines.count(command) == 1
    helper_index = script_lines.index("_agent_commit_status() {")
    running_index = script_lines.index("_agent_commit_status running")
    command_index = script_lines.index(command)
    helper_text = "\n".join(script_lines[helper_index:running_index])
    assert "/runtime/python -c " in helper_text
    assert "record-runtime-commit" in helper_text
    assert runtime_commit in helper_text
    assert helper_index < running_index < command_index
    assert plan["recipe"]["execution"] == {
        "target": "local",
        "workdir": "/runtime/snapshot",
        "python": "/runtime/python",
        "runtime_commit": runtime_commit,
    }
    (canonical,) = [row for row in read_run_manifest(root) if row["step_id"] == "external-evaluate"]
    assert canonical["status"] == "planned"
    assert canonical.get("command") in (None, "")
    planned_options = sorted(token for token in shlex.split(command) if token.startswith("--"))
    assert "--config" in planned_options

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
        evidence = {"supported_options": planned_options, "cli_options_sha256": "cli-digest"}
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
    assert snapshot["required_options"] == planned_options
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
    spec = _spec(root)
    spec_path = prepare_pipeline_sources(tmp_path, monkeypatch, spec)
    pipeline_dir = root / "pipelines" / "external-v1"
    jobs_path = pipeline_dir / "jobs.tsv"
    plan_dir = pipeline_dir / "plans" / "age-hsp-i2-psg" / "attempt-001"
    result_root = pipeline_dir / "results" / "age-hsp-i2-psg" / "attempt-001"
    step_manifest = root / "steps" / spec["pipeline"]["step"]["id"] / "step.yaml"
    runtime = FakePipelineRuntime(spec)
    probes = []

    def evaluation_rows():
        return [row for row in read_run_manifest(root) if row["step_id"] == "external-evaluate"]

    def registration_probe(fault):
        def inspect(execution, runs, *, plan_label="managed"):
            if plan_label == "pipeline":
                # Registration probes the plan before the pipeline publishes or registers it.
                assert yaml.safe_load(step_manifest.read_text())["plans"] == []
                assert evaluation_rows() == []
                probes.append(([Path(run["script"]) for run in runs], list(pipeline_dir.rglob("*.staging"))))
                fault()
            return runtime.inspect_target(execution, runs, plan_label=plan_label)

        return inspect

    def block_jobs_projection():
        # A directory at jobs.tsv makes the projection after the canonical commit fail; resume reconciles it.
        jobs_path.mkdir()

    def interrupt():
        raise PipelineInterrupted

    if outcome == "staging_tamper":

        def tamper_staged_launch():
            (staged_launch,), (staging_dir,) = probes[-1]
            assert staged_launch.parents[2] == staging_dir
            staged_launch.write_text("tampered\n")

        with pytest.raises(ValueError, match="attempt script changed"):
            run_pipeline(spec_path, runtime.hooks(inspect_target=registration_probe(tamper_staged_launch)))
        assert not plan_dir.exists()
        assert evaluation_rows() == []
        assert yaml.safe_load(step_manifest.read_text())["plans"] == []
        assert not list(pipeline_dir.rglob("*.staging"))
        return

    if outcome in {"success", "tamper"}:
        with pytest.raises(PipelineInterrupted):
            run_pipeline(spec_path, runtime.hooks(inspect_target=registration_probe(interrupt)))
        (staging_dir,) = plan_dir.parent.glob(".attempt-001.*.staging")
        # A crash right after publication leaves the staged plan public without its canonical registration.
        with plans.plan_publication_lock(plan_dir):
            plans.publish_staged_plan_locked(staging_dir, plan_dir, out_preexisted=False)
        assert yaml.safe_load(step_manifest.read_text())["plans"] == []
        assert evaluation_rows() == []
    else:
        with pytest.raises(pipeline_attempts.PipelineRegistrationRecoveryError, match="reconciled on resume"):
            run_pipeline(spec_path, runtime.hooks(inspect_target=registration_probe(block_jobs_projection)))

    frozen_plan = json.loads((plan_dir / "plan.json").read_text())
    frozen_identity = {
        "target": "local",
        "workdir": spec["runtime"]["workdir"],
        "python": spec["runtime"]["python"],
        "runtime_commit": spec["runtime"]["runtime_commit"],
    }
    recipe = yaml.safe_load((pipeline_dir / "recipes" / "age-hsp-i2-psg" / "attempt-001.yaml").read_text())
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
        with pytest.raises(
            pipeline_attempts.AttemptRegistrationPreflightError, match="differs from deterministic regeneration"
        ):
            run_pipeline(spec_path, runtime.hooks(), resume=True)
        assert evaluation_rows() == []
        assert yaml.safe_load(step_manifest.read_text())["plans"] == []
        assert not list(pipeline_dir.rglob("*.staging"))
        return

    if outcome == "interrupt_after_commit":
        (canonical,) = evaluation_rows()
        assert canonical["pipeline_id"] == "external-v1"
        assert canonical["terminal_status_owner"] == "script"
        # experiment-status reads registered plans only, and the harness's stubbed source runs have no registered
        # step, so it judges the interrupted registration's canonical row on its own.
        write_rows(root / "run_manifest.tsv", [canonical])
        workspace = yaml.safe_load((root / "experiment.yaml").read_text())
        workspace["experiment"].pop("status")
        (root / "experiment.yaml").write_text(yaml.safe_dump(workspace, sort_keys=False))
        snapshot = experiments.experiment_status(root)
        assert snapshot["decision"]["recommended_next"] is None
        assert snapshot["decision"]["other_legal_actions"] == []
        assert snapshot["decision"]["blocked_actions"] == ["finalize", "pipeline_advance"]

        write_rows(root / "run_manifest.tsv", [{**canonical, "status": "failed"}])
        terminal = experiments.experiment_status(root)
        assert terminal["decision"]["recommended_next"] is None
        assert terminal["decision"]["other_legal_actions"] == []
        assert terminal["decision"]["blocked_actions"] == ["finalize", "pipeline_advance"]
        return

    if outcome == "success":
        probes.clear()
        with pytest.raises(pipeline_attempts.PipelineRegistrationRecoveryError, match="reconciled on resume"):
            run_pipeline(
                spec_path, runtime.hooks(inspect_target=registration_probe(block_jobs_projection)), resume=True
            )
        # Deterministic regeneration validated the public plan; registration probed and committed it in place.
        assert probes == [([launch_path], [])]

    (canonical,) = evaluation_rows()
    assert canonical["job_id"] == "age-hsp-i2-psg"
    assert canonical["pipeline_id"] == "external-v1"
    assert canonical["terminal_status_owner"] == "script"
    step_payload = yaml.safe_load(step_manifest.read_text())
    assert step_payload["plan_controller"] == "pipeline"
    assert step_payload["plans"] == [str(plan_dir.resolve())]
    assert launch_path.read_bytes() == launch_before
    assert not list(plan_dir.parent.glob(".attempt-001.*.staging"))

    ownership_fields = {"pipeline_id", "job_id", "attempt", "result_root", "terminal_status_owner"}
    write_rows(
        root / "run_manifest.tsv",
        [
            (
                {key: value for key, value in row.items() if key not in ownership_fields}
                if row["step_id"] == "external-evaluate"
                else row
            )
            for row in read_run_manifest(root)
        ],
    )
    jobs_path.rmdir()
    probes.clear()
    result = run_pipeline(spec_path, runtime.hooks(inspect_target=registration_probe(interrupt)), resume=True)

    assert result["status"] == "completed"
    # A registered plan is reused as validated: nothing is pending, so nothing is regenerated or probed again.
    assert probes == []
    assert [row["job_id"] for row in experiment_pipeline.read_rows(jobs_path)] == ["age-hsp-i2-psg"]
    (repaired,) = evaluation_rows()
    assert repaired["pipeline_id"] == "external-v1"
    assert repaired["job_id"] == "age-hsp-i2-psg"
    assert repaired["attempt"] == "1"
    assert repaired["result_root"] == str(result_root)
    assert repaired["terminal_status_owner"] == "script"


def test_attempt_config_drift_fails_before_plan_publication(tmp_path: Path, monkeypatch):
    root = tmp_path / "workspace"
    spec = _spec(root)
    spec_path = prepare_pipeline_sources(tmp_path, monkeypatch, spec)
    pipeline_dir = root / "pipelines" / "external-v1"
    plan_dir = pipeline_dir / "plans" / "age-hsp-i2-psg" / "attempt-002"
    result_root = pipeline_dir / "results" / "age-hsp-i2-psg" / "attempt-002"
    runtime = FakePipelineRuntime(spec, outcome=lambda _run: "failed")
    drifted = []

    def launch_then_drift_config(*args, **kwargs):
        launched = runtime.launch_runs(*args, **kwargs)
        if not drifted:
            # The selected source config drifts after checkpoint selection, before the failed job's retry is planned.
            (row,) = experiment_pipeline.read_rows(pipeline_dir / "jobs.tsv")
            config = Path(row["config"])
            config.write_text(config.read_text() + "\n# drifted after checkpoint selection\n")
            drifted.append(config)
        return launched

    result = run_pipeline(spec_path, runtime.hooks(launch_runs=launch_then_drift_config))

    assert len(drifted) == 1
    (job,) = result["jobs"]
    assert result["status"] == "failed"
    assert (job["status"], job["retry_preparation_error"]) == (
        "failed",
        "External attempt registration preflight failed: Source config does not match the externally bound SHA-256.",
    )
    assert not plan_dir.exists()
    assert not list(plan_dir.parent.glob(f".{plan_dir.name}.*.staging"))
    assert not result_root.exists()
    assert [row["attempt"] for row in read_run_manifest(root) if row["step_id"] == "external-evaluate"] == ["1"]


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
        with experiment_workspace.managed_run_lock(root):
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

    # A directory or corrupt events.jsonl fails the reconcile's read with a fatal ValueError; interrupt this append.
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
