from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
from typing import Any, Callable

import yaml

from . import (
    experiment_io as exp_io,
    experiment_pipeline_results as pipeline_results,
    managed_scheduler,
    run_artifacts as artifacts,
)
from .experiment_workspace import (
    SUCCESS_STATUSES,
    TERMINAL_STATUSES,
    append_event,
    canonical_local_experiment_root,
    commit_step_manifest,
    file_sha256,
    managed_run_key,
    merge_run_manifest,
    next_run_index,
    read_managed_yaml_mapping,
    read_run_manifest,
)
from .hparam_runtime import monitor_hparam_runs
from .hparam_selection import select_hparam_candidates
from .manifests import read_json, read_rows, utc_now
from .plans import build_plan, preflight_plan

SCHEMA_VERSION = 1
PIPELINE_KIND = "external_matrix"
SOURCE_MANIFEST_SUCCESS_STATUSES = SUCCESS_STATUSES | {"skipped_test"}
ACTIVE_STATUSES = {"launched", "running"}
UNCERTAIN_STATUSES = pipeline_results.UNCERTAIN_STATUSES
SOURCE_UNCERTAIN_STATUSES = UNCERTAIN_STATUSES | {"submitting", "unknown_scheduler"}
RETRYABLE_STATUSES = pipeline_results.RETRYABLE_STATUSES


class RetryPreparationError(RuntimeError):
    pass


class AttemptRegistrationPreflightError(RuntimeError):
    pass


_TOP_LEVEL_FIELDS = {
    "schema_version",
    "pipeline",
    "runtime",
    "execution",
    "evaluation_policy",
    "checkpoint_policy",
    "checkpoint_sources",
    "jobs",
}
_PIPELINE_FIELDS = {"id", "kind", "experiment_id", "step", "finalize"}
_STEP_FIELDS = {"id", "phase", "purpose"}
_RUNTIME_FIELDS = {
    "workdir",
    "python",
    "runtime_commit",
    "accelerator",
    "device",
    "precision",
    "batch_size",
    "seed",
}
_EXECUTION_FIELDS = {"gpu_pool", "gpus_per_run", "max_concurrent", "max_attempts", "scheduler"}
_EVALUATION_FIELDS = {"external_test_locked", "final_test_unlocked"}
_CHECKPOINT_POLICY_FIELDS = {
    "avg_ckpts",
    "require_no_model_averaging",
    "forbidden_state_dict_prefixes",
    "require_ahi_eval_threshold",
}
_CHECKPOINT_SOURCE_FIELDS = {
    "plan",
    "selection_metric",
    "selection_mode",
    "task",
    "variant",
    "label_name",
}
_JOB_FIELDS = {
    "id",
    "checkpoint_source",
    "cohort",
    "modality",
    "inference_preset_path",
    "num_workers",
    "task",
    "variant",
    "label_name",
}


def run_experiment_pipeline(
    run_dir: str | Path,
    spec_path: str | Path,
    *,
    unlock_final_test: bool = False,
    execute: bool = False,
    resume: bool = False,
    poll_seconds: float = 60,
    finalize_callback: Callable[[str | Path, str | Path], Path] | None = None,
) -> dict[str, Any]:
    if poll_seconds < 0:
        raise ValueError("poll_seconds must be non-negative.")
    root = canonical_local_experiment_root(run_dir, Path.cwd())
    spec_file = Path(spec_path).expanduser()
    if not spec_file.is_absolute():
        spec_file = (Path.cwd() / spec_file).resolve()
    if spec_file.is_symlink() or not spec_file.is_file():
        raise ValueError(f"External pipeline spec must be a regular file: {spec_file}")
    source_text = spec_file.read_text()
    spec = read_managed_yaml_mapping(source_text, source=f"External pipeline spec {spec_file}")
    _validate_spec(spec, root, unlock_final_test=unlock_final_test if execute else None)
    pipeline_id = str(spec["pipeline"]["id"])
    pipeline_dir = root / "pipelines" / pipeline_id
    lock_path = pipeline_dir.parent / f".{pipeline_id}.runner.lock"
    exp_io.validate_managed_output_paths(root, [pipeline_dir / "pipeline.json", lock_path])
    for source_id, source in spec["checkpoint_sources"].items():
        source_plan = artifacts.read_hparam_plan(Path(source["plan"]))
        source_recipe = source_plan.get("recipe") if isinstance(source_plan.get("recipe"), dict) else {}
        source_execution = source_recipe.get("execution") if isinstance(source_recipe.get("execution"), dict) else {}
        # Schema v1 reads source artifacts on the manager and has no SSH staging boundary.
        if str(source_execution.get("target") or "local") == "ssh":
            raise ValueError(
                f"checkpoint_sources.{source_id}.plan uses an SSH execution target; "
                "schema v1 external pipelines require local source plans."
            )

    if not execute:
        if resume:
            raise ValueError("--resume is only valid with --execute.")
        if pipeline_dir.exists():
            _validate_frozen_pipeline(pipeline_dir, source_text, spec)
        sources = _inspect_sources(root, spec, refresh=False)
        return {
            "status": _source_summary_status(sources),
            "dry_run": True,
            "pipeline_id": pipeline_id,
            "pipeline_dir": str(pipeline_dir),
            "source_states": sources,
            "job_count": len(spec["jobs"]),
        }

    if not unlock_final_test:
        raise ValueError("External pipeline execution requires --unlock-final-test.")
    pipeline_dir.parent.mkdir(parents=True, exist_ok=True)
    existed = pipeline_dir.exists()
    experiment = _validate_experiment(root, spec, allow_completed=existed)
    if existed and not resume:
        raise ValueError("Pipeline state already exists; continue only with --resume --execute.")
    if not existed and resume:
        raise ValueError("Cannot resume because pipeline state does not exist.")
    with lock_path.open("a+") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Pipeline runner is already active: {pipeline_id}") from exc
        if existed:
            state = _validate_frozen_pipeline(pipeline_dir, source_text, spec)
            if experiment.get("status") == "completed" and state.get("status") != "completed":
                raise ValueError("A completed experiment cannot resume an incomplete pipeline.")
            if state.get("status") == "failed":
                raise ValueError("Failed pipelines are immutable; create a new pipeline revision.")
        else:
            staging_dir = pipeline_dir.parent / f".{pipeline_id}.{os.getpid()}.{time.time_ns()}.staging"
            staging_dir.mkdir()
            _freeze_pipeline(root, staging_dir, spec_file, source_text, spec)
            os.replace(staging_dir, pipeline_dir)
        try:
            return _execute_pipeline(
                root,
                pipeline_dir,
                spec,
                poll_seconds=poll_seconds,
                finalize_callback=finalize_callback,
            )
        except Exception as exc:
            state = read_json(pipeline_dir / "pipeline.json")
            if state.get("status") not in {"blocked", "completed"}:
                _update_state(pipeline_dir, status="failed", error=f"{type(exc).__name__}: {exc}")
            raise


def _validate_spec(spec: dict[str, Any], root: Path, *, unlock_final_test: bool | None) -> None:
    _reject_unknown_fields(spec, _TOP_LEVEL_FIELDS, "spec")
    if type(spec.get("schema_version")) is not int or spec["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}.")
    pipeline = _mapping(spec, "pipeline")
    _reject_unknown_fields(pipeline, _PIPELINE_FIELDS, "pipeline")
    pipeline_id = _required_slug(pipeline, "id", "pipeline")
    if pipeline.get("kind") != PIPELINE_KIND:
        raise ValueError(f"pipeline.kind must be {PIPELINE_KIND!r}.")
    _required_slug(pipeline, "experiment_id", "pipeline")
    step = _mapping(pipeline, "step")
    _reject_unknown_fields(step, _STEP_FIELDS, "pipeline.step")
    _required_slug(step, "id", "pipeline.step")
    if step.get("phase") != "evaluate":
        raise ValueError("pipeline.step.phase must be 'evaluate'.")
    if not str(step.get("purpose") or "").strip():
        raise ValueError("pipeline.step.purpose is required.")
    if pipeline.get("finalize") is not True:
        raise ValueError("pipeline.finalize must be true in schema v1.")

    runtime = _mapping(spec, "runtime")
    _reject_unknown_fields(runtime, _RUNTIME_FIELDS, "runtime")
    for field in ("workdir", "runtime_commit"):
        value = runtime.get(field)
        if not isinstance(value, str) or not value.strip() or value == "ASK_USER":
            raise ValueError(f"runtime.{field} must be an explicit non-empty string.")
    python_command = runtime.get("python")
    if (
        not isinstance(python_command, str)
        or not python_command.strip()
        or python_command == "ASK_USER"
        or python_command.startswith("~")
        or re.search(r"\s", python_command) is not None
    ):
        raise ValueError(
            "runtime.python must be a single executable name or path without whitespace, arguments, or ~ shorthand."
        )
    for field in ("accelerator", "device", "precision"):
        if runtime.get(field) in (None, ""):
            raise ValueError(f"runtime.{field} is required.")
    if not Path(runtime["workdir"]).is_absolute():
        raise ValueError("runtime.workdir must be absolute.")
    if not re.fullmatch(r"[0-9a-f]{40}", runtime["runtime_commit"]):
        raise ValueError("runtime.runtime_commit must be a lowercase 40-character Git commit SHA.")
    if runtime.get("accelerator") != "gpu" or runtime.get("device") != "cuda":
        raise ValueError("Schema v1 external evaluation requires GPU/CUDA runtime.")
    if str(runtime.get("precision")) not in {"32", "32-true"}:
        raise ValueError("Schema v1 external evaluation requires FP32 precision.")
    if type(runtime.get("batch_size")) is not int or runtime["batch_size"] != 128:
        raise ValueError("Schema v1 external evaluation requires runtime.batch_size=128.")
    if isinstance(runtime.get("seed"), bool) or not isinstance(runtime.get("seed"), int):
        raise ValueError("runtime.seed must be an integer.")

    execution = _mapping(spec, "execution")
    _reject_unknown_fields(execution, _EXECUTION_FIELDS, "execution")
    if "scheduler" in execution:
        scheduler = _mapping(execution, "scheduler")
        _reject_unknown_fields(scheduler, {"type"}, "execution.scheduler")
        if scheduler.get("type") != "direct":
            raise ValueError("Schema v1 external evaluation supports only execution.scheduler.type=direct.")
    gpu_pool = execution.get("gpu_pool")
    if (
        not isinstance(gpu_pool, list)
        or not gpu_pool
        or any(isinstance(item, bool) or not isinstance(item, int) for item in gpu_pool)
    ):
        raise ValueError("execution.gpu_pool must be a non-empty list of GPU integers.")
    if len(gpu_pool) != len(set(gpu_pool)):
        raise ValueError("execution.gpu_pool contains duplicate GPUs.")
    if type(execution.get("gpus_per_run")) is not int or execution["gpus_per_run"] != 1:
        raise ValueError("Schema v1 requires execution.gpus_per_run=1.")
    max_concurrent = execution.get("max_concurrent")
    if (
        isinstance(max_concurrent, bool)
        or not isinstance(max_concurrent, int)
        or not 1 <= max_concurrent <= len(gpu_pool)
    ):
        raise ValueError("execution.max_concurrent must be between 1 and the GPU pool size.")
    if type(execution.get("max_attempts")) is not int or execution["max_attempts"] != 2:
        raise ValueError("Schema v1 requires execution.max_attempts=2.")

    evaluation = _mapping(spec, "evaluation_policy")
    _reject_unknown_fields(evaluation, _EVALUATION_FIELDS, "evaluation_policy")
    if evaluation.get("external_test_locked") is not False or evaluation.get("final_test_unlocked") is not True:
        raise ValueError("External pipeline spec must explicitly unlock final test evaluation.")
    if unlock_final_test is False:
        raise ValueError("External pipeline execution also requires --unlock-final-test.")

    checkpoint_policy = _mapping(spec, "checkpoint_policy")
    _reject_unknown_fields(checkpoint_policy, _CHECKPOINT_POLICY_FIELDS, "checkpoint_policy")
    if type(checkpoint_policy.get("avg_ckpts")) is not int or checkpoint_policy["avg_ckpts"] != 1:
        raise ValueError("checkpoint_policy.avg_ckpts must be 1.")
    if checkpoint_policy.get("require_no_model_averaging") is not True:
        raise ValueError("checkpoint_policy.require_no_model_averaging must be true.")
    prefixes = checkpoint_policy.get("forbidden_state_dict_prefixes")
    if (
        not isinstance(prefixes, list)
        or not prefixes
        or any(not isinstance(item, str) or not item for item in prefixes)
    ):
        raise ValueError("checkpoint_policy.forbidden_state_dict_prefixes must be a non-empty string list.")
    if not {"ema_model.", "running_mean_model."}.issubset(prefixes):
        raise ValueError("checkpoint_policy.forbidden_state_dict_prefixes must include EMA and running-mean keys.")
    if checkpoint_policy.get("require_ahi_eval_threshold") is not True:
        raise ValueError("checkpoint_policy.require_ahi_eval_threshold must be true.")

    sources = _mapping(spec, "checkpoint_sources")
    if not sources:
        raise ValueError("checkpoint_sources must not be empty.")
    for source_id, source in sources.items():
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", str(source_id)):
            raise ValueError(f"Invalid checkpoint source id: {source_id}")
        if not isinstance(source, dict):
            raise ValueError(f"checkpoint_sources.{source_id} must be a mapping.")
        _reject_unknown_fields(source, _CHECKPOINT_SOURCE_FIELDS, f"checkpoint_sources.{source_id}")
        plan = Path(str(source.get("plan") or ""))
        if not plan.is_absolute():
            raise ValueError(f"checkpoint_sources.{source_id}.plan must be absolute.")
        try:
            plan.resolve().relative_to(root)
        except ValueError as exc:
            raise ValueError(f"checkpoint_sources.{source_id}.plan must be inside the experiment root.") from exc
        if not str(source.get("selection_metric") or ""):
            raise ValueError(f"checkpoint_sources.{source_id}.selection_metric is required.")
        if source.get("selection_mode") not in {"min", "max"}:
            raise ValueError(f"checkpoint_sources.{source_id}.selection_mode must be min or max.")

    jobs = spec.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("jobs must be a non-empty list.")
    seen = set()
    for index, job in enumerate(jobs):
        if not isinstance(job, dict):
            raise ValueError(f"jobs[{index}] must be a mapping.")
        _reject_unknown_fields(job, _JOB_FIELDS, f"jobs[{index}]")
        job_id = _required_slug(job, "id", f"jobs[{index}]")
        if job_id in seen:
            raise ValueError(f"Duplicate external job id: {job_id}")
        seen.add(job_id)
        source_id = str(job.get("checkpoint_source") or "")
        if source_id not in sources:
            raise ValueError(f"jobs[{index}].checkpoint_source is unknown: {source_id}")
        for field in ("cohort", "modality"):
            if not str(job.get(field) or "").strip():
                raise ValueError(f"jobs[{index}].{field} is required.")
        preset = Path(str(job.get("inference_preset_path") or ""))
        if not preset.is_absolute():
            raise ValueError(f"jobs[{index}].inference_preset_path must be absolute.")
        workers = job.get("num_workers")
        if isinstance(workers, bool) or not isinstance(workers, int) or workers < 0:
            raise ValueError(f"jobs[{index}].num_workers must be a non-negative integer.")
        expected_workers = {"psg": 8, "bcg": 16}.get(str(job["modality"]).lower())
        if expected_workers is not None and workers != expected_workers:
            raise ValueError(f"jobs[{index}].num_workers must be {expected_workers} for {job['modality']} inference.")
    if not pipeline_id:
        raise AssertionError("validated pipeline id is empty")


def _validate_experiment(root: Path, spec: dict[str, Any], *, allow_completed: bool = False) -> dict[str, Any]:
    path = root / "experiment.yaml"
    manifest = read_managed_yaml_mapping(path.read_text(), source=f"Managed experiment manifest {path}")
    experiment = manifest.get("experiment")
    if not isinstance(experiment, dict):
        raise ValueError("experiment.yaml is missing experiment metadata.")
    if experiment.get("id") != spec["pipeline"]["experiment_id"]:
        raise ValueError("Pipeline experiment id differs from experiment.yaml.")
    if canonical_local_experiment_root(experiment.get("root"), Path.cwd()) != root:
        raise ValueError("Pipeline run directory differs from experiment.yaml root.")
    if experiment.get("status") == "completed" and not allow_completed:
        raise ValueError("Experiment is already completed.")
    read_run_manifest(root)
    return experiment


def _freeze_pipeline(root: Path, pipeline_dir: Path, spec_file: Path, source_text: str, spec: dict[str, Any]) -> None:
    experiment = _validate_experiment(root, spec)
    sources = _source_plan_snapshots(root, spec)
    resolved_text = yaml.safe_dump(spec, sort_keys=False)
    state = {
        "schema_version": SCHEMA_VERSION,
        "pipeline_id": spec["pipeline"]["id"],
        "experiment_id": experiment["id"],
        "status": "waiting_for_sources",
        "spec_path": str(spec_file),
        "spec_source_sha256": _text_sha256(source_text),
        "spec_resolved_sha256": _text_sha256(resolved_text),
        "source_plans": sources,
        "external_presets": _preset_snapshots(spec),
        "runtime_commit": spec["runtime"]["runtime_commit"],
        "created_at": utc_now(),
        "updated_at": utc_now(),
    }
    # Reserve controller ownership first so an interrupted freeze remains an unmaterialized pipeline blocker.
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
    _atomic_write_text(pipeline_dir / "spec.source.yaml", source_text)
    _atomic_write_text(pipeline_dir / "spec.resolved.yaml", resolved_text)
    _write_state(pipeline_dir, state)
    append_event(root, "pipeline_frozen", {"pipeline_id": state["pipeline_id"], "spec": str(spec_file)})


def _validate_frozen_pipeline(pipeline_dir: Path, source_text: str, spec: dict[str, Any]) -> dict[str, Any]:
    state_path = pipeline_dir / "pipeline.json"
    if not state_path.is_file() or state_path.is_symlink():
        raise ValueError(f"Frozen pipeline state is missing or aliased: {state_path}")
    state = read_json(state_path)
    source_path = pipeline_dir / "spec.source.yaml"
    resolved_path = pipeline_dir / "spec.resolved.yaml"
    if (
        source_path.is_symlink()
        or resolved_path.is_symlink()
        or not source_path.is_file()
        or not resolved_path.is_file()
    ):
        raise ValueError("Frozen pipeline spec artifacts are missing or aliased.")
    resolved_text = yaml.safe_dump(spec, sort_keys=False)
    expected = {
        "pipeline_id": spec["pipeline"]["id"],
        "experiment_id": spec["pipeline"]["experiment_id"],
        "spec_source_sha256": _text_sha256(source_text),
        "spec_resolved_sha256": _text_sha256(resolved_text),
        "runtime_commit": spec["runtime"]["runtime_commit"],
    }
    for field, value in expected.items():
        if state.get(field) != value:
            raise ValueError(f"Frozen pipeline field drifted: {field}")
    original_spec = Path(str(state.get("spec_path") or ""))
    if (
        not original_spec.is_absolute()
        or original_spec.is_symlink()
        or not original_spec.is_file()
        or file_sha256(original_spec) != state["spec_source_sha256"]
    ):
        raise ValueError("Original external pipeline spec changed or disappeared.")
    if file_sha256(source_path) != state["spec_source_sha256"]:
        raise ValueError("Frozen source spec changed.")
    if file_sha256(resolved_path) != state["spec_resolved_sha256"]:
        raise ValueError("Frozen resolved spec changed.")
    snapshots = state.get("source_plans")
    if not isinstance(snapshots, list):
        raise ValueError("Frozen source plan snapshots are malformed.")
    snapshots_by_id = {str(snapshot.get("source_id")): snapshot for snapshot in snapshots if isinstance(snapshot, dict)}
    if len(snapshots_by_id) != len(snapshots) or set(snapshots_by_id) != set(spec["checkpoint_sources"]):
        raise ValueError("Frozen source plan identities differ from the pipeline spec.")
    for source_id, source in spec["checkpoint_sources"].items():
        snapshot = snapshots_by_id[source_id]
        if str(snapshot.get("plan_dir") or "") != str(source["plan"]):
            raise ValueError(f"Frozen source plan path drifted: {source_id}")
        plan_path = Path(str(snapshot["plan_path"]))
        resolved_recipe_path = Path(str(snapshot["resolved_recipe_path"]))
        if file_sha256(plan_path) != snapshot["plan_sha256"]:
            raise ValueError(f"Source plan changed after pipeline freeze: {plan_path}")
        if file_sha256(resolved_recipe_path) != snapshot["resolved_recipe_sha256"]:
            raise ValueError(f"Source plan recipe changed after pipeline freeze: {resolved_recipe_path}")
        artifacts.read_hparam_plan(Path(str(snapshot["plan_dir"])))
    preset_snapshots = state.get("external_presets")
    if not isinstance(preset_snapshots, list):
        raise ValueError("Frozen external preset snapshots are malformed.")
    presets_by_job = {
        str(snapshot.get("job_id")): snapshot for snapshot in preset_snapshots if isinstance(snapshot, dict)
    }
    if len(presets_by_job) != len(preset_snapshots) or set(presets_by_job) != {str(job["id"]) for job in spec["jobs"]}:
        raise ValueError("Frozen external preset identities differ from the pipeline spec.")
    for job in spec["jobs"]:
        snapshot = presets_by_job[job["id"]]
        preset = Path(str(job["inference_preset_path"]))
        if str(snapshot.get("path") or "") != str(preset):
            raise ValueError(f"Frozen external preset path drifted: {job['id']}")
        if preset.is_symlink() or not preset.is_file() or file_sha256(preset) != snapshot.get("sha256"):
            raise ValueError(f"Frozen external preset changed: {preset}")
    checkpoints_path = pipeline_dir / "checkpoints.json"
    if checkpoints_path.exists():
        expected_hash = state.get("checkpoint_selection_sha256")
        if isinstance(expected_hash, str) and file_sha256(checkpoints_path) != expected_hash:
            raise ValueError("Frozen checkpoint selection manifest changed or was not committed.")
        _read_frozen_selections(checkpoints_path, spec)
    elif state.get("checkpoint_selection_sha256") not in (None, ""):
        raise ValueError("Frozen checkpoint selection manifest is missing.")
    return state


def _source_plan_snapshots(root: Path, spec: dict[str, Any]) -> list[dict[str, Any]]:
    snapshots = []
    for source_id, source in spec["checkpoint_sources"].items():
        plan_dir = Path(source["plan"])
        plan = artifacts.read_hparam_plan(plan_dir)
        recipe = plan["recipe"]
        if canonical_local_experiment_root(recipe["experiment"]["root"], Path.cwd()) != root:
            raise ValueError(f"Source plan belongs to another experiment: {plan_dir}")
        _assert_source_semantics(source_id, source, recipe)
        plan_path = plan_dir / "plan.json"
        resolved_recipe_path = plan_dir / "recipe.resolved.yaml"
        snapshots.append(
            {
                "source_id": source_id,
                "plan_dir": str(plan_dir),
                "plan_path": str(plan_path),
                "plan_sha256": file_sha256(plan_path),
                "resolved_recipe_path": str(resolved_recipe_path),
                "resolved_recipe_sha256": file_sha256(resolved_recipe_path),
            }
        )
    return snapshots


def _preset_snapshots(spec: dict[str, Any]) -> list[dict[str, str]]:
    snapshots = []
    for job in spec["jobs"]:
        preset = Path(str(job["inference_preset_path"]))
        if preset.is_symlink() or not preset.is_file():
            raise ValueError(f"External preset is missing or aliased: {preset}")
        snapshots.append({"job_id": job["id"], "path": str(preset), "sha256": file_sha256(preset)})
    return snapshots


def _inspect_sources(root: Path, spec: dict[str, Any], *, refresh: bool) -> list[dict[str, Any]]:
    canonical = {managed_run_key(row): row for row in read_run_manifest(root)}
    states = []
    for source_id, source in spec["checkpoint_sources"].items():
        plan_dir = Path(source["plan"])
        plan = artifacts.read_hparam_plan(plan_dir)
        if refresh:
            monitor_hparam_runs(plan_dir, once=True, health=True)
            canonical = {managed_run_key(row): row for row in read_run_manifest(root)}
        rows = [canonical[managed_run_key(run)] for run in plan["runs"]]
        statuses = [str(row.get("status") or "") for row in rows]
        uncertain = [row["run_id"] for row in rows if row.get("status") in SOURCE_UNCERTAIN_STATUSES]
        failed = [row["run_id"] for row in rows if row.get("status") in TERMINAL_STATUSES - SUCCESS_STATUSES]
        complete = bool(rows) and all(status in SUCCESS_STATUSES for status in statuses)
        if complete:
            for run in plan["runs"]:
                manifest_path = artifacts.find_run_manifest(run)
                if manifest_path is None or manifest_path.is_symlink() or not manifest_path.is_file():
                    raise ValueError(f"Successful source run lacks a valid run_manifest.json: {run['run_id']}")
                manifest = read_json(manifest_path)
                if not isinstance(manifest, dict) or not manifest:
                    raise ValueError(f"Successful source run manifest is invalid: {manifest_path}")
                manifest_status = manifest.get("status")
                if manifest_status not in (None, "", *SOURCE_MANIFEST_SUCCESS_STATUSES):
                    raise ValueError(f"Successful source run manifest reports failure: {manifest_path}")
        states.append(
            {
                "source_id": source_id,
                "plan": str(plan_dir),
                "statuses": statuses,
                "complete": complete,
                "failed_runs": failed,
                "uncertain_runs": uncertain,
            }
        )
    return states


def _source_summary_status(states: list[dict[str, Any]]) -> str:
    if any(state["failed_runs"] for state in states):
        return "failed"
    if any(state["uncertain_runs"] for state in states):
        return "blocked"
    if all(state["complete"] for state in states):
        return "ready"
    return "waiting_for_sources"


def _execute_pipeline(
    root: Path,
    pipeline_dir: Path,
    spec: dict[str, Any],
    *,
    poll_seconds: float,
    finalize_callback: Callable[[str | Path, str | Path], Path] | None,
) -> dict[str, Any]:
    state = _validate_frozen_pipeline(pipeline_dir, (pipeline_dir / "spec.source.yaml").read_text(), spec)
    if state.get("status") == "completed":
        return _finalize_completed_pipeline(root, pipeline_dir, spec, finalize_callback)
    while True:
        _validate_frozen_pipeline(pipeline_dir, (pipeline_dir / "spec.source.yaml").read_text(), spec)
        sources = _inspect_sources(root, spec, refresh=True)
        state_status = _source_summary_status(sources)
        _update_state(pipeline_dir, status=state_status, source_states=sources)
        if state_status in {"blocked", "failed"}:
            raise RuntimeError("External pipeline source plans are failed or have uncertain execution identity.")
        if state_status == "ready":
            break
        time.sleep(poll_seconds)

    selections = _load_or_freeze_selections(root, pipeline_dir, spec)
    attempts = _load_or_create_initial_attempts(root, pipeline_dir, spec, selections)
    _update_state(pipeline_dir, status="running_external", missing_pid_blocker=None)
    result = _run_attempts(root, pipeline_dir, spec, selections, attempts, poll_seconds=poll_seconds)
    if result["status"] != "completed":
        _update_state(
            pipeline_dir,
            status=result["status"],
            logical_jobs=result["jobs"],
            missing_pid_blocker=result.get("missing_pid_blocker"),
        )
        return result
    _validate_frozen_pipeline(pipeline_dir, (pipeline_dir / "spec.source.yaml").read_text(), spec)
    report = _aggregate_results(root, pipeline_dir, spec, selections, result["jobs"])
    result_paths = [
        pipeline_dir / "results.csv",
        pipeline_dir / "metrics.csv",
        pipeline_dir / "summary.md",
        report,
    ]
    _update_state(
        pipeline_dir,
        status="completed",
        completed_at=utc_now(),
        final_report=str(report),
        result_artifacts={str(path): file_sha256(path) for path in result_paths},
        logical_jobs=result["jobs"],
    )
    append_event(root, "pipeline_completed", {"pipeline_id": spec["pipeline"]["id"], "report": str(report)})
    if spec["pipeline"]["finalize"]:
        if finalize_callback is None:
            raise RuntimeError("Pipeline finalization callback is unavailable.")
        finalize_callback(root, report)
    return {
        "status": "completed",
        "pipeline_id": spec["pipeline"]["id"],
        "pipeline_dir": str(pipeline_dir),
        "report": str(report),
        "jobs": result["jobs"],
    }


def _finalize_completed_pipeline(
    root: Path,
    pipeline_dir: Path,
    spec: dict[str, Any],
    finalize_callback: Callable[[str | Path, str | Path], Path] | None,
) -> dict[str, Any]:
    state = read_json(pipeline_dir / "pipeline.json")
    report = Path(str(state.get("final_report") or ""))
    artifacts_by_path = state.get("result_artifacts")
    if not isinstance(artifacts_by_path, dict) or str(report) not in artifacts_by_path:
        raise ValueError("Completed pipeline result artifacts are not frozen.")
    for raw_path, expected_hash in artifacts_by_path.items():
        path = Path(str(raw_path))
        try:
            path.relative_to(pipeline_dir)
        except ValueError as exc:
            raise ValueError(f"Completed pipeline artifact is outside its pipeline directory: {path}") from exc
        if path.is_symlink() or not path.is_file() or file_sha256(path) != expected_hash:
            raise ValueError(f"Completed pipeline artifact changed: {path}")
    selections = _load_or_freeze_selections(root, pipeline_dir, spec)
    attempts = read_rows(pipeline_dir / "jobs.tsv", require_managed_identity=True)
    _validate_attempt_rows(root, pipeline_dir, spec, selections, attempts)
    logical = _logical_job_states(spec, attempts)
    if any(job["status"] != "completed" for job in logical) or any(
        row.get("status") in ACTIVE_STATUSES | UNCERTAIN_STATUSES for row in attempts
    ):
        raise ValueError("Completed pipeline no longer has a terminal verified external matrix.")
    experiment = _validate_experiment(root, spec, allow_completed=True)
    if experiment.get("status") != "completed":
        if finalize_callback is None:
            raise RuntimeError("Pipeline finalization callback is unavailable.")
        finalize_callback(root, report)
    return {
        "status": "completed",
        "pipeline_id": spec["pipeline"]["id"],
        "pipeline_dir": str(pipeline_dir),
        "report": str(report),
        "jobs": logical,
    }


def _load_or_freeze_selections(root: Path, pipeline_dir: Path, spec: dict[str, Any]) -> dict[str, dict[str, Any]]:
    path = pipeline_dir / "checkpoints.json"
    if path.exists():
        selections = _read_frozen_selections(path, spec)
        state = read_json(pipeline_dir / "pipeline.json")
        if state.get("checkpoint_selection_sha256") in (None, ""):
            derived = _select_checkpoint_sources(root, spec)
            if list(selections.values()) != derived:
                raise ValueError("Uncommitted checkpoint selection differs from validation-derived selection.")
            _update_state(
                pipeline_dir,
                checkpoint_selection_sha256=file_sha256(path),
                checkpoint_selected_at=read_json(path).get("created_at"),
            )
        return selections

    frozen = _select_checkpoint_sources(root, spec)
    payload = {
        "pipeline_id": spec["pipeline"]["id"],
        "created_at": utc_now(),
        "sources": frozen,
    }
    _atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _update_state(
        pipeline_dir,
        checkpoint_selection_sha256=file_sha256(path),
        checkpoint_selected_at=payload["created_at"],
    )
    append_event(
        root,
        "pipeline_checkpoints_frozen",
        {"pipeline_id": spec["pipeline"]["id"], "source_count": len(frozen)},
    )
    return {item["source_id"]: item for item in frozen}


def _select_checkpoint_sources(root: Path, spec: dict[str, Any]) -> list[dict[str, Any]]:
    policy = spec["checkpoint_policy"]
    frozen = []
    for source_id, source in spec["checkpoint_sources"].items():
        plan_dir = Path(source["plan"])
        plan = artifacts.read_hparam_plan(plan_dir)
        recipe = plan["recipe"]
        step_id = str(recipe["step"]["id"])
        select_hparam_candidates(plan_dir, source["selection_metric"], source["selection_mode"])
        ranking = read_rows(root / "reports" / "ranking.csv", require_managed_identity=True)
        source_run_keys = {managed_run_key(run) for run in plan["runs"]}
        candidates = [row for row in ranking if managed_run_key(row) in source_run_keys]
        if not candidates:
            raise ValueError(f"No ranked checkpoint selection is owned by source plan {source_id}.")
        row = min(candidates, key=lambda candidate: int(candidate["rank"]))
        key = (str(row["step_id"]), str(row["run_id"]))
        canonical = {managed_run_key(item): item for item in read_run_manifest(root)}
        canonical_row = canonical.get(key)
        if canonical_row is None or canonical_row.get("status") not in SUCCESS_STATUSES:
            raise ValueError(f"Selected checkpoint source is not successful: {source_id}")
        config = Path(str(row.get("config") or ""))
        checkpoint = Path(str(row.get("checkpoint_path") or ""))
        if config.is_symlink() or not config.is_file():
            raise ValueError(f"Selected config is not a regular file: {config}")
        if checkpoint.is_symlink() or not checkpoint.is_file():
            raise ValueError(f"Selected checkpoint is not a regular file: {checkpoint}")
        checkpoint_dir = Path(str(canonical_row.get("checkpoint_dir") or ""))
        if checkpoint.parent != checkpoint_dir or checkpoint_dir.is_symlink():
            raise ValueError(f"Selected checkpoint is outside its frozen checkpoint directory: {checkpoint}")
        exp_io.validate_managed_output_paths(checkpoint_dir, [checkpoint])
        config_payload = read_managed_yaml_mapping(config.read_text(), source=f"Selected config {config}")
        averaging_paths = _mapping_key_paths(config_payload, "model_averaging")
        if policy["require_no_model_averaging"] and averaging_paths:
            raise ValueError(f"Selected config contains model_averaging: {', '.join(averaging_paths)}")
        label_name = str((recipe.get("inputs") or {}).get("label_name") or "")
        checkpoint_evidence = _validate_checkpoint_payload(checkpoint, label_name, policy)
        score = artifacts.float_or_none(row.get("score"))
        if score is None or not math.isfinite(score):
            raise ValueError(f"Selected validation score is not finite for source {source_id}.")
        selection = {
            "source_id": source_id,
            "plan": str(plan_dir),
            "step_id": step_id,
            "run_id": str(row["run_id"]),
            "run_name": str(row.get("run_name") or ""),
            "selection_metric": source["selection_metric"],
            "selection_mode": source["selection_mode"],
            "score": score,
            "config": str(config),
            "config_sha256": file_sha256(config),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": file_sha256(checkpoint),
            "variant": str(recipe.get("variant") or ""),
            "label_name": label_name,
            "source_task": label_name,
            "source_plan_task": str(recipe.get("task") or ""),
            "inference_task": "infer",
            **checkpoint_evidence,
        }
        _assert_job_semantic_assertions(spec, source_id, selection)
        frozen.append(selection)
    return frozen


def _read_frozen_selections(path: Path, spec: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Frozen checkpoint selection manifest is missing or aliased: {path}")
    payload = read_json(path)
    if not isinstance(payload, dict) or payload.get("pipeline_id") != spec["pipeline"]["id"]:
        raise ValueError(f"Frozen checkpoint selection manifest has the wrong pipeline id: {path}")
    selections = payload.get("sources")
    if not isinstance(selections, list) or any(not isinstance(item, dict) for item in selections):
        raise ValueError(f"Frozen checkpoint selections are malformed: {path}")
    by_id = {str(item.get("source_id") or ""): item for item in selections}
    if len(by_id) != len(selections) or set(by_id) != set(spec["checkpoint_sources"]):
        raise ValueError("Frozen checkpoint source identities differ from the pipeline spec.")
    for source_id, source in spec["checkpoint_sources"].items():
        selection = by_id[source_id]
        expected = {
            "plan": str(source["plan"]),
            "selection_metric": source["selection_metric"],
            "selection_mode": source["selection_mode"],
        }
        for field, value in expected.items():
            if selection.get(field) != value:
                raise ValueError(f"Frozen checkpoint source field drifted: {source_id}.{field}")
        score = artifacts.float_or_none(selection.get("score"))
        if score is None or not math.isfinite(score):
            raise ValueError(f"Frozen checkpoint score is not finite: {source_id}")
        for path_field, hash_field in (("config", "config_sha256"), ("checkpoint", "checkpoint_sha256")):
            selected_path = Path(str(selection.get(path_field) or ""))
            if path_field == "checkpoint":
                exp_io.validate_managed_output_paths(selected_path.parent, [selected_path])
            if (
                selected_path.is_symlink()
                or not selected_path.is_file()
                or file_sha256(selected_path) != selection.get(hash_field)
            ):
                raise ValueError(f"Frozen selected {path_field} changed: {selected_path}")
        _assert_job_semantic_assertions(spec, source_id, selection)
    return by_id


def _validate_checkpoint_payload(checkpoint: Path, label_name: str, policy: dict[str, Any]) -> dict[str, Any]:
    try:
        import torch

        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise ValueError(f"Cannot inspect selected checkpoint {checkpoint}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Selected checkpoint is not a mapping: {checkpoint}")
    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError(f"Selected checkpoint lacks state_dict: {checkpoint}")
    prefixes = tuple(policy["forbidden_state_dict_prefixes"])
    forbidden = sorted(str(key) for key in state_dict if str(key).startswith(prefixes))
    if forbidden:
        raise ValueError(f"Selected checkpoint contains forbidden averaging state: {forbidden[0]}")
    has_ahi_threshold = "ahi_eval_threshold" in payload
    if label_name == "ahi" and policy["require_ahi_eval_threshold"] and not has_ahi_threshold:
        raise ValueError(f"AHI checkpoint lacks ahi_eval_threshold: {checkpoint}")
    return {
        "state_dict_key_count": len(state_dict),
        "has_ahi_eval_threshold": has_ahi_threshold,
    }


def _assert_job_semantic_assertions(spec: dict[str, Any], source_id: str, selection: dict[str, Any]) -> None:
    expected = {
        "task": selection["source_task"],
        "variant": selection["variant"],
        "label_name": selection["label_name"],
    }
    for job in spec["jobs"]:
        if job["checkpoint_source"] != source_id:
            continue
        for field, value in expected.items():
            assertion = job.get(field)
            if assertion not in (None, "") and assertion != value:
                raise ValueError(f"External job {job['id']} {field} assertion differs from its source plan.")


def _load_or_create_initial_attempts(
    root: Path,
    pipeline_dir: Path,
    spec: dict[str, Any],
    selections: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    jobs_path = pipeline_dir / "jobs.tsv"
    existing = read_rows(jobs_path, require_managed_identity=True)
    if existing:
        _validate_attempt_rows(root, pipeline_dir, spec, selections, existing, require_all_jobs=False)

    recipes = []
    for job in spec["jobs"]:
        selection = selections[job["checkpoint_source"]]
        attempt = 1
        recipe, recipe_path, plan_dir, result_root = _attempt_recipe(pipeline_dir, spec, job, selection, attempt)
        recipe_text = yaml.safe_dump(recipe, sort_keys=False)
        if recipe_path.exists():
            if recipe_path.is_symlink() or not recipe_path.is_file() or recipe_path.read_text() != recipe_text:
                raise ValueError(f"External job recipe changed during resume: {recipe_path}")
        else:
            _atomic_write_text(recipe_path, recipe_text)
        recipes.append((job, selection, attempt, recipe_path, plan_dir, result_root))

    _ensure_initial_preflight(pipeline_dir, spec, recipes)

    attempt_rows = list(existing)
    existing_jobs = {str(row["job_id"]) for row in attempt_rows}
    pending = [item for item in recipes if item[0]["id"] not in existing_jobs]
    initial_variants = {str(selection["variant"]) for _job, selection, *_paths in recipes}
    snapshot_owner_dirs = {
        variant: pipeline_dir if len(initial_variants) == 1 else pipeline_dir / "initial_schedulers" / variant
        for variant in initial_variants
    }
    prepared = _prepare_attempt_registration_groups(
        root,
        spec,
        recipes,
        snapshot_owner_dirs=snapshot_owner_dirs,
    )
    try:
        for job, selection, attempt, recipe_path, plan_dir, result_root in pending:
            row = _materialize_attempt(
                root,
                spec,
                job,
                selection,
                attempt,
                recipe_path=recipe_path,
                plan_dir=plan_dir,
                result_root=result_root,
                prepared_staging_dir=prepared[job["id"]],
            )
            attempt_rows.append(row)
            _write_jobs(jobs_path, attempt_rows)
    finally:
        for staging_dir in prepared.values():
            if staging_dir is not None and staging_dir.exists() and not staging_dir.is_symlink():
                shutil.rmtree(staging_dir)
    _validate_attempt_rows(root, pipeline_dir, spec, selections, attempt_rows)
    if len(existing) != len(spec["jobs"]):
        append_event(
            root,
            "pipeline_jobs_planned",
            {"pipeline_id": spec["pipeline"]["id"], "job_count": len(spec["jobs"])},
        )
    return attempt_rows


def _ensure_initial_preflight(
    pipeline_dir: Path,
    spec: dict[str, Any],
    recipes: list[tuple[dict[str, Any], dict[str, Any], int, Path, Path, Path]],
) -> None:
    path = pipeline_dir / "preflight.json"
    expected = {
        "pipeline_id": spec["pipeline"]["id"],
        "jobs": [
            {"job_id": job["id"], "recipe": str(recipe_path), "recipe_sha256": file_sha256(recipe_path)}
            for job, _selection, _attempt, recipe_path, _plan_dir, _result_root in recipes
        ],
    }
    if path.exists():
        if path.is_symlink() or read_json(path) != expected:
            raise ValueError("External matrix preflight evidence changed.")
        return
    if any(plan_dir.exists() and any(plan_dir.iterdir()) for *_prefix, plan_dir, _result_root in recipes):
        raise ValueError("External attempt plans exist without committed matrix preflight evidence.")
    blocked = []
    for job, _selection, _attempt, recipe_path, plan_dir, _result_root in recipes:
        _recipe, _config, report = preflight_plan(
            recipe_path=recipe_path,
            output_dir=plan_dir,
            unlock_final_test=True,
        )
        if report.exit_code != 0:
            blocked.append(f"{job['id']}: {report.status.value}")
    if blocked:
        raise RuntimeError("External matrix preflight failed before launch: " + "; ".join(blocked))
    _atomic_write_text(path, json.dumps(expected, indent=2, sort_keys=True) + "\n")


def _prepare_attempt_plan(
    job_id: str,
    selection: dict[str, Any],
    recipe_path: Path,
    plan_dir: Path,
    *,
    run_index_offset: int | None = None,
) -> tuple[Path, Path | None]:
    staging_dir = plan_dir.parent / f".{plan_dir.name}.{os.getpid()}.{time.time_ns()}.staging"
    try:
        report = build_plan(
            recipe_path=recipe_path,
            output_dir=plan_dir,
            unlock_final_test=True,
            source_config_sha256=selection["config_sha256"],
            staging_dir=staging_dir,
            defer_commit=True,
            plan_controller="pipeline",
            run_index_offset=run_index_offset,
        )
        if report.exit_code != 0:
            raise RuntimeError(f"External job plan unexpectedly failed after preflight: {job_id}")
        if not plan_dir.exists():
            return staging_dir, staging_dir
        if artifacts.plan_tree_sha256(plan_dir) != artifacts.plan_tree_sha256(staging_dir):
            raise ValueError(f"Uncommitted external attempt plan differs from deterministic regeneration: {plan_dir}")
        shutil.rmtree(staging_dir)
        return plan_dir, None
    except BaseException:
        if staging_dir.exists() and not staging_dir.is_symlink():
            shutil.rmtree(staging_dir)
        raise


def _materialize_attempt(
    root: Path,
    spec: dict[str, Any],
    job: dict[str, Any],
    selection: dict[str, Any],
    attempt: int,
    *,
    recipe_path: Path,
    plan_dir: Path,
    result_root: Path,
    prepared_staging_dir: Path | None = None,
) -> dict[str, Any]:
    _validate_new_attempt_paths(plan_dir, result_root, allow_existing_plan=True)
    plan_path = plan_dir / "plan.json"
    if plan_dir.exists() and not plan_path.exists():
        raise ValueError(f"External attempt plan is incomplete: {plan_dir}")

    plan = read_json(plan_path) if plan_path.exists() else None
    runs = plan.get("runs") if isinstance(plan, dict) else None
    run = dict(runs[0]) if isinstance(runs, list) and len(runs) == 1 and isinstance(runs[0], dict) else None
    canonical_by_key = {managed_run_key(row): row for row in read_run_manifest(root)}
    canonical = canonical_by_key.get(managed_run_key(run)) if run is not None else None
    if canonical is None:
        if prepared_staging_dir is not None:
            if plan_dir.exists():
                raise ValueError(f"External attempt plan appeared after registration preflight: {plan_dir}")
            plan_dir.parent.mkdir(parents=True, exist_ok=True)
            prepared_staging_dir.replace(plan_dir)
        else:
            _physical_plan_dir, staging_dir = _prepare_attempt_plan(job["id"], selection, recipe_path, plan_dir)
            if staging_dir is not None:
                plan_dir.parent.mkdir(parents=True, exist_ok=True)
                staging_dir.replace(plan_dir)
        plan = read_json(plan_path)
    runs = plan.get("runs") if isinstance(plan, dict) else None
    if not isinstance(runs, list) or len(runs) != 1 or not isinstance(runs[0], dict):
        raise ValueError(f"External job plan must contain exactly one managed run: {plan_dir}")
    run = dict(runs[0])
    base_run = {field: value for field, value in run.items() if field != "command"}
    _validate_physical_attempt_plan(spec, job, selection, recipe_path, plan_dir, base_run)
    enrichment = {
        "step_id": run["step_id"],
        "run_id": run["run_id"],
        "pipeline_id": spec["pipeline"]["id"],
        "job_id": job["id"],
        "attempt": attempt,
        "result_root": str(result_root),
        "terminal_status_owner": "script",
    }
    canonical_by_key = {managed_run_key(row): row for row in read_run_manifest(root)}
    canonical = canonical_by_key.get(managed_run_key(run))
    if canonical is not None:
        _validate_attempt_plan(
            {"step_id": run["step_id"], "run_id": run["run_id"], "recipe": str(recipe_path), "plan_dir": str(plan_dir)},
            canonical,
        )
    plan_recipe = plan.get("recipe") if isinstance(plan.get("recipe"), dict) else {}
    # Publish pipeline ownership before its row so a crash cannot expose the attempt as an ordinary launch candidate.
    commit_step_manifest(
        root,
        {
            "step": plan_recipe["step"],
            "experiment_id": plan_recipe["experiment"]["id"],
            "plan_controller": "pipeline",
            "recipe_path": plan_recipe.get("_recipe_path", ""),
            "plans": [str(plan_dir.resolve())],
        },
    )
    if canonical is None:
        update = {**base_run, "parameter_summary": "single resolved recipe", **enrichment}
    else:
        update = enrichment
    committed = merge_run_manifest(root, [update])
    canonical = {managed_run_key(row): row for row in committed}[managed_run_key(run)]
    projection = _attempt_projection(job, selection, canonical, recipe_path=recipe_path, plan_dir=plan_dir)
    _validate_attempt_plan(projection, canonical)
    return projection


def _prepare_attempt_registration_groups(
    root: Path,
    spec: dict[str, Any],
    attempts: list[tuple[dict[str, Any], dict[str, Any], int, Path, Path, Path]],
    *,
    snapshot_owner_dirs: dict[str, Path],
) -> dict[str, Path | None]:
    prepared: dict[str, Path | None] = {}
    groups: dict[str, list[dict[str, Any]]] = {}
    group_paths: dict[str, list[Path]] = {}
    canonical_keys = {managed_run_key(row) for row in read_run_manifest(root)}
    pending = []
    for item in attempts:
        job, _selection, _attempt, _recipe_path, plan_dir, _result_root = item
        plan_path = plan_dir / "plan.json"
        plan = read_json(plan_path) if plan_path.exists() else None
        runs = plan.get("runs") if isinstance(plan, dict) else None
        run = runs[0] if isinstance(runs, list) and len(runs) == 1 and isinstance(runs[0], dict) else None
        if run is not None and managed_run_key(run) in canonical_keys:
            prepared[job["id"]] = None
        else:
            pending.append(item)

    if not pending:
        return prepared
    pending_job_ids = {item[0]["id"] for item in pending}
    pending_variants = {str(item[1]["variant"]) for item in pending}
    first_recipe_path = pending[0][3]
    first_recipe = read_managed_yaml_mapping(
        first_recipe_path.read_text(), source=f"External attempt recipe {first_recipe_path}"
    )
    first_run_index = next_run_index(first_recipe)
    pending_run_indices = {item[0]["id"]: first_run_index + run_offset for run_offset, item in enumerate(pending)}
    try:
        for job, selection, _attempt, recipe_path, plan_dir, result_root in attempts:
            variant = str(selection["variant"])
            if variant not in pending_variants:
                continue
            if job["id"] in pending_job_ids:
                _validate_new_attempt_paths(plan_dir, result_root, allow_existing_plan=True)
                physical_plan_dir, staging_dir = _prepare_attempt_plan(
                    job["id"],
                    selection,
                    recipe_path,
                    plan_dir,
                    run_index_offset=pending_run_indices[job["id"]],
                )
                prepared[job["id"]] = staging_dir
                group_paths.setdefault(variant, []).extend([plan_dir / "plan.json", result_root])
            else:
                # A resumed registration must compare the complete scheduler group with its frozen snapshot.
                physical_plan_dir = plan_dir

            plan = read_json(physical_plan_dir / "plan.json")
            runs = plan.get("runs") if isinstance(plan, dict) else None
            if not isinstance(runs, list) or len(runs) != 1 or not isinstance(runs[0], dict):
                raise ValueError(f"External job plan must contain exactly one managed run: {plan_dir}")
            run = dict(runs[0])
            try:
                script_relative = Path(str(run["script"])).relative_to(plan_dir)
            except (KeyError, ValueError) as exc:
                raise ValueError(f"External attempt script is outside its plan: {plan_dir}") from exc
            run["script"] = str(physical_plan_dir / script_relative)
            groups.setdefault(variant, []).append(run)

        execution = _pipeline_execution(spec)
        remote = str(execution["host"]) if execution.get("target", "local") == "ssh" else None
        snapshots = {}
        for variant in sorted(groups):
            snapshot_path = snapshot_owner_dirs[variant] / managed_scheduler.EXECUTION_SNAPSHOT_NAME
            exp_io.validate_managed_output_paths(
                Path("/"),
                [*group_paths[variant], snapshot_path],
                remote=remote,
            )
            snapshot = managed_scheduler.inspect_execution_target(execution, groups[variant], plan_label="pipeline")
            if snapshot_path.exists() and read_json(snapshot_path) != snapshot:
                raise ValueError(f"Frozen pipeline execution snapshot changed: {snapshot_path}")
            snapshots[variant] = (snapshot_path, snapshot)
        for snapshot_path, snapshot in snapshots.values():
            if not snapshot_path.exists():
                snapshot_text = json.dumps(snapshot, indent=2, sort_keys=True) + "\n"
                exp_io.conditional_atomic_replace_text_at(snapshot_path, snapshot_text, None)
            if read_json(snapshot_path) != snapshot:
                raise ValueError(f"Frozen pipeline execution snapshot changed: {snapshot_path}")
        return prepared
    except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as exc:
        for staging_dir in prepared.values():
            if staging_dir is not None and staging_dir.exists() and not staging_dir.is_symlink():
                shutil.rmtree(staging_dir)
        raise AttemptRegistrationPreflightError(f"External attempt registration preflight failed: {exc}") from exc


def _validate_physical_attempt_plan(
    spec: dict[str, Any],
    job: dict[str, Any],
    selection: dict[str, Any],
    recipe_path: Path,
    plan_dir: Path,
    run: dict[str, Any],
) -> None:
    expected = {
        "experiment_id": spec["pipeline"]["experiment_id"],
        "step_id": spec["pipeline"]["step"]["id"],
        "status": "planned",
    }
    for field, value in expected.items():
        if run.get(field) != value:
            raise ValueError(f"External attempt plan field differs from its pipeline: {field}")
    run_dir = plan_dir / "runs" / f"{run['run_id']}--{run['run_name']}"
    expected_paths = {
        "run_dir": run_dir,
        "config": run_dir / "config.yaml",
        "script": run_dir / "launch.sh",
        "artifacts": run_dir / "artifacts.json",
    }
    for field, value in expected_paths.items():
        if Path(str(run.get(field) or "")) != value:
            raise ValueError(f"External attempt plan path differs from its managed directory: {field}")
    if file_sha256(run["config"]) != selection["config_sha256"]:
        raise ValueError(f"External attempt config differs from its selected source: {job['id']}")
    _validate_attempt_plan(
        {"step_id": run["step_id"], "run_id": run["run_id"], "recipe": str(recipe_path), "plan_dir": str(plan_dir)},
        run,
    )


def _attempt_recipe(
    pipeline_dir: Path,
    spec: dict[str, Any],
    job: dict[str, Any],
    selection: dict[str, Any],
    attempt: int,
) -> tuple[dict[str, Any], Path, Path, Path]:
    attempt_name = f"attempt-{attempt:03d}"
    recipe_path = pipeline_dir / "recipes" / job["id"] / f"{attempt_name}.yaml"
    plan_dir = pipeline_dir / "plans" / job["id"] / attempt_name
    result_root = pipeline_dir / "results" / job["id"] / attempt_name
    runtime = spec["runtime"]
    experiment_path = pipeline_dir.parent.parent / "experiment.yaml"
    experiment_manifest = read_managed_yaml_mapping(
        experiment_path.read_text(), source=f"Managed experiment manifest {experiment_path}"
    )["experiment"]
    experiment = {field: experiment_manifest[field] for field in ("id", "title", "objective", "root", "baseline")}
    recipe = {
        "name": f"{spec['pipeline']['id']}__{job['id']}__{attempt_name}",
        "task": "infer",
        "variant": selection["variant"],
        "experiment": experiment,
        "step": spec["pipeline"]["step"],
        "inputs": {
            "config": selection["config"],
            "ckpt_path": selection["checkpoint"],
            "label_name": selection["label_name"],
            "eval_split": "test",
            "inference_preset_path": job["inference_preset_path"],
        },
        "runtime": {
            "devices": [0],
            "accelerator": runtime["accelerator"],
            "device": runtime["device"],
            "precision": runtime["precision"],
            "batch_size": runtime["batch_size"],
            "num_workers": job["num_workers"],
            "seed": runtime["seed"],
            "avg_ckpts": spec["checkpoint_policy"]["avg_ckpts"],
            "results_root": str(result_root),
        },
        "artifacts": {"overwrite": False},
        "evaluation_policy": {"external_test_locked": False, "final_test_unlocked": True},
        "execution": {
            "target": "local",
            "workdir": runtime["workdir"],
            "python": runtime["python"],
            "runtime_commit": runtime["runtime_commit"],
        },
        "decisions": {
            "task": {"value": "infer", "source": "explicit_recipe"},
            "label_name": {"value": selection["label_name"], "source": "explicit_recipe"},
            "ckpt_path": {"value": selection["checkpoint"], "source": "explicit_recipe"},
            "external_test_locked": {"value": False, "source": "explicit_recipe"},
            "final_eval_unlock": {"value": True, "source": "explicit_recipe"},
            "overwrite_policy": {"value": False, "source": "explicit_recipe"},
        },
    }
    return recipe, recipe_path, plan_dir, result_root


def _validate_new_attempt_paths(plan_dir: Path, result_root: Path, *, allow_existing_plan: bool = False) -> None:
    for path in (plan_dir, result_root):
        if path.is_symlink():
            raise ValueError(f"Managed attempt output must not be a symlink: {path}")
        allow_nonempty = allow_existing_plan and path == plan_dir
        if path.exists() and (not path.is_dir() or (any(path.iterdir()) and not allow_nonempty)):
            raise ValueError(f"Managed attempt output must be a new empty directory: {path}")


def _attempt_projection(
    job: dict[str, Any],
    selection: dict[str, Any],
    run: dict[str, Any],
    *,
    recipe_path: Path,
    plan_dir: Path,
) -> dict[str, Any]:
    return {
        "step_id": run["step_id"],
        "run_id": run["run_id"],
        "pipeline_id": run["pipeline_id"],
        "job_id": job["id"],
        "attempt": run["attempt"],
        "status": run["status"],
        "verified": "",
        "checkpoint_source": job["checkpoint_source"],
        "checkpoint": selection["checkpoint"],
        "checkpoint_sha256": selection["checkpoint_sha256"],
        "config": selection["config"],
        "config_sha256": selection["config_sha256"],
        "label_name": selection["label_name"],
        "variant": selection["variant"],
        "cohort": job["cohort"],
        "modality": job["modality"],
        "preset": job["inference_preset_path"],
        "num_workers": job["num_workers"],
        "result_root": run["result_root"],
        "result_manifest": "",
        "recipe": str(recipe_path),
        "plan_dir": str(plan_dir),
        "runtime_commit": "",
    }


def _validate_attempt_rows(
    root: Path,
    pipeline_dir: Path,
    spec: dict[str, Any],
    selections: dict[str, dict[str, Any]],
    rows: list[dict[str, Any]],
    *,
    require_all_jobs: bool = True,
) -> None:
    canonical = {managed_run_key(row): row for row in read_run_manifest(root)}
    jobs = {job["id"]: job for job in spec["jobs"]}
    seen_attempts = set()
    for row in rows:
        key = managed_run_key(row)
        run = canonical.get(key)
        if run is None:
            raise ValueError(f"Pipeline attempt is missing from run_manifest.tsv: {key[0]} / {key[1]}")
        job_id = str(row.get("job_id") or "")
        if job_id not in jobs or run.get("job_id") != job_id:
            raise ValueError(f"Pipeline attempt job identity drifted: {key[0]} / {key[1]}")
        try:
            attempt = int(row["attempt"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Pipeline attempt number is invalid: {job_id}") from exc
        attempt_key = (job_id, attempt)
        if attempt_key in seen_attempts or not 1 <= attempt <= spec["execution"]["max_attempts"]:
            raise ValueError(f"Pipeline attempt identity is invalid or duplicated: {job_id} / {attempt}")
        seen_attempts.add(attempt_key)
        job = jobs[job_id]
        selection = selections[job["checkpoint_source"]]
        expected = {
            "pipeline_id": spec["pipeline"]["id"],
            "checkpoint_source": job["checkpoint_source"],
            "checkpoint": selection["checkpoint"],
            "checkpoint_sha256": selection["checkpoint_sha256"],
            "config": selection["config"],
            "config_sha256": selection["config_sha256"],
            "label_name": selection["label_name"],
            "variant": selection["variant"],
            "cohort": job["cohort"],
            "modality": job["modality"],
            "preset": job["inference_preset_path"],
            "num_workers": job["num_workers"],
            "result_root": str(pipeline_dir / "results" / job_id / f"attempt-{attempt:03d}"),
            "recipe": str(pipeline_dir / "recipes" / job_id / f"attempt-{attempt:03d}.yaml"),
            "plan_dir": str(pipeline_dir / "plans" / job_id / f"attempt-{attempt:03d}"),
        }
        for field, value in expected.items():
            if str(row.get(field) or "") != str(value):
                raise ValueError(f"Pipeline attempt field drifted: {job_id}.{field}")
        for field in ("pipeline_id", "attempt", "result_root", "terminal_status_owner"):
            expected_value = "script" if field == "terminal_status_owner" else row.get(field)
            if str(run.get(field) or "") != str(expected_value or ""):
                raise ValueError(f"Pipeline attempt field drifted: {field}")
        _validate_attempt_plan(row, run)
        if str(row.get("verified") or "").lower() == "true":
            if run.get("status") not in SUCCESS_STATUSES:
                raise ValueError(f"Verified pipeline attempt is not canonically successful: {job_id}")
            manifest_path = _validate_result_manifest(spec, row, run)
            if str(row.get("result_manifest") or "") != str(manifest_path):
                raise ValueError(f"Verified pipeline result manifest drifted: {job_id}")

    attempts_by_job = {
        job_id: sorted(attempt for candidate, attempt in seen_attempts if candidate == job_id) for job_id in jobs
    }
    allowed_sequences = ([1], [1, 2]) if require_all_jobs else ([], [1], [1, 2])
    if any(attempts not in allowed_sequences for attempts in attempts_by_job.values()):
        raise ValueError("Pipeline attempt sequence is incomplete or non-contiguous.")
    verified_counts = {
        job_id: sum(str(row.get("verified") or "").lower() == "true" for row in rows if row.get("job_id") == job_id)
        for job_id in jobs
    }
    if any(count > 1 for count in verified_counts.values()):
        raise ValueError("Pipeline job has multiple verified successful attempts.")


def _validate_attempt_plan(row: dict[str, Any], canonical_run: dict[str, Any]) -> None:
    recipe_path = Path(str(row["recipe"]))
    plan_dir = Path(str(row["plan_dir"]))
    plan_path = plan_dir / "plan.json"
    resolved_recipe_path = plan_dir / "recipe.resolved.yaml"
    if recipe_path.is_symlink() or not recipe_path.is_file():
        raise ValueError(f"Pipeline attempt recipe is missing or aliased: {recipe_path}")
    if plan_dir.is_symlink() or not plan_dir.is_dir():
        raise ValueError(f"Pipeline attempt plan directory is missing or aliased: {plan_dir}")
    for path in (plan_path, resolved_recipe_path):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Pipeline attempt plan artifact is missing or aliased: {path}")
    plan = read_json(plan_path)
    planned_runs = plan.get("runs") if isinstance(plan, dict) else None
    if not isinstance(planned_runs, list) or len(planned_runs) != 1:
        raise ValueError(f"Pipeline attempt plan must contain exactly one run: {plan_path}")
    planned = planned_runs[0]
    if not isinstance(planned, dict) or managed_run_key(planned) != managed_run_key(row):
        raise ValueError(f"Pipeline attempt plan run identity drifted: {plan_path}")
    commands = plan.get("commands")
    if not isinstance(commands, list) or len(commands) != 1 or planned.get("command") != commands[0]:
        raise ValueError(f"Pipeline attempt command drifted: {plan_path}")
    for field in (
        "experiment_id",
        "step_id",
        "run_id",
        "run_name",
        "version",
        "config",
        "config_sha256",
        "script",
        "script_sha256",
        "run_dir",
        "artifacts",
    ):
        if str(planned.get(field) or "") != str(canonical_run.get(field) or ""):
            raise ValueError(f"Pipeline attempt plan field drifted: {field}")
    for path_field, hash_field in (("config", "config_sha256"), ("script", "script_sha256")):
        path = Path(str(canonical_run.get(path_field) or ""))
        if path.is_symlink() or not path.is_file() or file_sha256(path) != canonical_run.get(hash_field):
            raise ValueError(f"Pipeline attempt {path_field} changed: {path}")
        try:
            path.relative_to(plan_dir)
        except ValueError as exc:
            raise ValueError(f"Pipeline attempt {path_field} is outside its plan: {path}") from exc
    if commands[0] not in Path(str(canonical_run["script"])).read_text().splitlines():
        raise ValueError(f"Pipeline attempt command drifted: {plan_path}")
    source_recipe = read_managed_yaml_mapping(recipe_path.read_text(), source=f"Pipeline recipe {recipe_path}")
    resolved_recipe = read_managed_yaml_mapping(
        resolved_recipe_path.read_text(), source=f"Pipeline resolved recipe {resolved_recipe_path}"
    )
    plan_recipe = plan.get("recipe")
    frozen_plan_recipe = (
        {key: value for key, value in plan_recipe.items() if key != "_recipe_path"}
        if isinstance(plan_recipe, dict)
        else None
    )
    authored_recipe = {
        key: value for key, value in resolved_recipe.items() if key not in {"_plan_context", "input_snapshots"}
    }
    if source_recipe != authored_recipe or frozen_plan_recipe != resolved_recipe:
        raise ValueError(f"Pipeline attempt recipe drifted: {recipe_path}")


def _write_jobs(path: Path, rows: list[dict[str, Any]]) -> None:
    _write_rows_atomic(path, rows)


def _run_attempts(
    root: Path,
    pipeline_dir: Path,
    spec: dict[str, Any],
    selections: dict[str, dict[str, Any]],
    attempts: list[dict[str, Any]],
    *,
    poll_seconds: float,
) -> dict[str, Any]:
    jobs_path = pipeline_dir / "jobs.tsv"
    execution = _pipeline_execution(spec)
    runtime = {"devices": [0]}
    while True:
        _validate_frozen_pipeline(pipeline_dir, (pipeline_dir / "spec.source.yaml").read_text(), spec)
        attempts = read_rows(jobs_path, require_managed_identity=True)
        _validate_attempt_rows(root, pipeline_dir, spec, selections, attempts)
        groups: list[tuple[Path, list[dict[str, Any]]]] = []
        initial_rows = [row for row in attempts if int(row["attempt"]) == 1]
        initial_variants = {str(row["variant"]) for row in initial_rows}
        for variant in sorted(initial_variants):
            variant_rows = [row for row in initial_rows if row["variant"] == variant]
            owner_dir = pipeline_dir if len(initial_variants) == 1 else pipeline_dir / "initial_schedulers" / variant
            groups.append((owner_dir, _planned_runs(variant_rows)))
        for row in attempts:
            if int(row["attempt"]) == 1:
                continue
            groups.append((pipeline_dir / "retry_schedulers" / row["job_id"], _planned_runs([row])))

        missing_pid_blocker = None
        for owner_dir, runs in groups:
            owner_dir.mkdir(parents=True, exist_ok=True)
            snapshot_path = owner_dir / managed_scheduler.EXECUTION_SNAPSHOT_NAME
            if snapshot_path.exists():
                canonical = {managed_run_key(row): row for row in read_run_manifest(root)}
                if any(
                    (canonical[managed_run_key(run)].get("status") or "planned")
                    in managed_scheduler.LAUNCHABLE_STATUSES
                    for run in runs
                ):
                    managed_scheduler.validated_execution_snapshot(owner_dir, execution, runs, canonical)
            try:
                managed_scheduler.launch_managed_runs(
                    root,
                    owner_dir,
                    runs,
                    execution,
                    runtime,
                    dry_run=False,
                    fail_on_missing_pid_blocker=True,
                    default_script_commits_terminal_status=True,
                    runtime_output_fields=("result_root",),
                    runtime_output_root=root,
                )
            except managed_scheduler.MissingPidCapacityError as exc:
                missing_pid_blocker = exc
                break

        canonical = {managed_run_key(row): row for row in read_run_manifest(root)}
        if any(
            canonical[managed_run_key(row)].get("status") in SUCCESS_STATUSES
            and str(row.get("verified") or "").lower() != "true"
            for row in attempts
        ):
            _validate_frozen_pipeline(pipeline_dir, (pipeline_dir / "spec.source.yaml").read_text(), spec)
        changed = False
        for row in attempts:
            key = managed_run_key(row)
            run = canonical[key]
            status = str(run.get("status") or "")
            if row.get("status") != status:
                row["status"] = status
                changed = True
            if int(row["attempt"]) == 1:
                owner_dir = (
                    pipeline_dir if len(initial_variants) == 1 else pipeline_dir / "initial_schedulers" / row["variant"]
                )
            else:
                owner_dir = pipeline_dir / "retry_schedulers" / row["job_id"]
            snapshot_path = owner_dir / managed_scheduler.EXECUTION_SNAPSHOT_NAME
            snapshot = read_json(snapshot_path) if snapshot_path.exists() else {}
            runtime_commit = str(snapshot.get("runtime_commit") or "")
            if row.get("runtime_commit") != runtime_commit:
                row["runtime_commit"] = runtime_commit
                changed = True
            if (
                status in SUCCESS_STATUSES
                and str(row.get("verified") or "").lower() != "true"
                and row.get("validation_error") in (None, "")
            ):
                try:
                    manifest_path = _validate_result_manifest(spec, row, run)
                except ValueError as exc:
                    row["verified"] = "false"
                    row["validation_error"] = str(exc)
                else:
                    row["verified"] = "true"
                    row["result_manifest"] = str(manifest_path)
                    row["validation_error"] = ""
                changed = True
        if changed:
            _write_jobs(jobs_path, attempts)

        logical = _logical_job_states(spec, attempts)
        if missing_pid_blocker is None and any(job["status"] == "running" for job in logical):
            expected_keys = {managed_run_key(row) for row in attempts}
            capacity = managed_scheduler.capacity_state(
                execution,
                runtime,
                {key: canonical[key] for key in expected_keys},
                canonical,
                expected_keys=expected_keys,
            )
            if capacity.external_missing_pid:
                missing_pid_blocker = managed_scheduler.MissingPidCapacityError(
                    *sorted(capacity.external_missing_pid)[0]
                )
        if missing_pid_blocker is not None:
            return {
                "status": "blocked",
                "jobs": logical,
                "missing_pid_blocker": {
                    "status": "missing_pid",
                    "step_id": missing_pid_blocker.step_id,
                    "run_id": missing_pid_blocker.run_id,
                },
            }

        try:
            attempts, retry_created = _create_needed_retries(root, pipeline_dir, spec, selections, attempts)
        except RetryPreparationError as exc:
            independent_active = any(row.get("status") in ACTIVE_STATUSES | {"planned", "pending"} for row in attempts)
            if not independent_active:
                raise
            _update_state(pipeline_dir, status="running_external", retry_preparation_error=str(exc))
            time.sleep(poll_seconds)
            continue
        if retry_created:
            _write_jobs(jobs_path, attempts)
            continue

        logical = _logical_job_states(spec, attempts)
        if all(job["status"] == "completed" for job in logical):
            return {"status": "completed", "jobs": logical}

        pending_or_active = any(row.get("status") in ACTIVE_STATUSES | {"planned", "pending"} for row in attempts)
        if not pending_or_active:
            final_status = "blocked" if any(job["status"] == "blocked" for job in logical) else "failed"
            return {"status": final_status, "jobs": logical}
        time.sleep(poll_seconds)


def _pipeline_execution(spec: dict[str, Any]) -> dict[str, Any]:
    return {
        "target": "local",
        "workdir": spec["runtime"]["workdir"],
        "python": spec["runtime"]["python"],
        "runtime_commit": spec["runtime"]["runtime_commit"],
        "gpu_pool": spec["execution"]["gpu_pool"],
        "gpus_per_run": spec["execution"]["gpus_per_run"],
        "max_concurrent": spec["execution"]["max_concurrent"],
    }


def _planned_runs(attempt_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    runs = []
    for row in attempt_rows:
        plan = read_json(Path(row["plan_dir"]) / "plan.json")
        planned = plan.get("runs")
        if not isinstance(planned, list) or len(planned) != 1:
            raise ValueError(f"Attempt plan must contain exactly one run: {row['plan_dir']}")
        run = dict(planned[0])
        if managed_run_key(run) != managed_run_key(row):
            raise ValueError(f"Attempt plan run identity drifted: {row['plan_dir']}")
        run.update(
            {
                "pipeline_id": row["pipeline_id"],
                "job_id": row["job_id"],
                "attempt": int(row["attempt"]),
                "result_root": row["result_root"],
                "terminal_status_owner": "script",
                "checkpoint": row["checkpoint"],
                "checkpoint_sha256": row["checkpoint_sha256"],
            }
        )
        runs.append(run)
    return runs


def _create_needed_retries(
    root: Path,
    pipeline_dir: Path,
    spec: dict[str, Any],
    selections: dict[str, dict[str, Any]],
    attempts: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], bool]:
    created = False
    state_changed = False
    canonical = {managed_run_key(row): row for row in read_run_manifest(root)}
    by_job: dict[str, list[dict[str, Any]]] = {}
    for row in attempts:
        by_job.setdefault(str(row["job_id"]), []).append(row)
    jobs = {job["id"]: job for job in spec["jobs"]}
    candidates = []
    for job_id, rows in by_job.items():
        if any(str(row.get("verified") or "").lower() == "true" for row in rows):
            continue
        latest = max(rows, key=lambda row: int(row["attempt"]))
        status = str(latest.get("status") or "")
        process_identity_error = str(canonical.get(managed_run_key(latest), {}).get("process_identity_error") or "")
        if process_identity_error:
            if latest.get("retry_blocker") in (None, ""):
                latest["retry_blocker"] = f"unsafe process identity: {process_identity_error}"
                append_event(
                    root,
                    "pipeline_job_retry_blocked",
                    {
                        "pipeline_id": spec["pipeline"]["id"],
                        "job_id": job_id,
                        "attempt": int(latest["attempt"]),
                        "reason": "unsafe_process_identity",
                    },
                )
                state_changed = True
            continue
        if (
            status not in RETRYABLE_STATUSES
            or int(latest["attempt"]) >= spec["execution"]["max_attempts"]
            or latest.get("retry_preparation_error") not in (None, "")
            or latest.get("retry_blocker") not in (None, "")
        ):
            continue
        job = jobs[job_id]
        selection = selections[job["checkpoint_source"]]
        attempt = int(latest["attempt"]) + 1
        recipe, recipe_path, plan_dir, result_root = _attempt_recipe(pipeline_dir, spec, job, selection, attempt)
        recipe_text = yaml.safe_dump(recipe, sort_keys=False)
        if recipe_path.exists():
            if recipe_path.is_symlink() or not recipe_path.is_file() or recipe_path.read_text() != recipe_text:
                raise ValueError(f"Retry recipe changed during resume: {recipe_path}")
        else:
            _atomic_write_text(recipe_path, recipe_text)
        candidates.append((latest, job, selection, attempt, recipe_path, plan_dir, result_root))

    ready = []
    retry_preflight_changed = False
    for latest, job, selection, attempt, recipe_path, plan_dir, result_root in candidates:
        try:
            _ensure_retry_preflight(pipeline_dir, job["id"], attempt, recipe_path, plan_dir)
        except RetryPreparationError as exc:
            latest["retry_preparation_error"] = str(exc)
            append_event(
                root,
                "pipeline_job_retry_preflight_failed",
                {"pipeline_id": spec["pipeline"]["id"], "job_id": job["id"], "attempt": attempt},
            )
            retry_preflight_changed = True
        else:
            ready.append((latest, job, selection, attempt, recipe_path, plan_dir, result_root))

    if retry_preflight_changed or state_changed:
        _write_jobs(pipeline_dir / "jobs.tsv", attempts)

    for latest, job, selection, attempt, recipe_path, plan_dir, result_root in ready:
        staging_dir = None
        try:
            prepared = _prepare_attempt_registration_groups(
                root,
                spec,
                [(job, selection, attempt, recipe_path, plan_dir, result_root)],
                snapshot_owner_dirs={str(selection["variant"]): pipeline_dir / "retry_schedulers" / job["id"]},
            )
            staging_dir = prepared[job["id"]]
            retry_row = _materialize_attempt(
                root,
                spec,
                job,
                selection,
                attempt,
                recipe_path=recipe_path,
                plan_dir=plan_dir,
                result_root=result_root,
                prepared_staging_dir=staging_dir,
            )
        except RuntimeError as exc:
            latest["retry_preparation_error"] = str(exc)
            _write_jobs(pipeline_dir / "jobs.tsv", attempts)
            continue
        finally:
            if staging_dir is not None and staging_dir.exists() and not staging_dir.is_symlink():
                shutil.rmtree(staging_dir)
        attempts.append(retry_row)
        _write_jobs(pipeline_dir / "jobs.tsv", attempts)
        append_event(
            root,
            "pipeline_job_retry_planned",
            {"pipeline_id": spec["pipeline"]["id"], "job_id": job["id"], "attempt": attempt},
        )
        created = True
    return attempts, created


def _ensure_retry_preflight(
    pipeline_dir: Path,
    job_id: str,
    attempt: int,
    recipe_path: Path,
    plan_dir: Path,
) -> None:
    path = pipeline_dir / "preflight_retries" / job_id / f"attempt-{attempt:03d}.json"
    expected = {
        "job_id": job_id,
        "attempt": attempt,
        "recipe": str(recipe_path),
        "recipe_sha256": file_sha256(recipe_path),
    }
    if path.exists():
        if path.is_symlink() or read_json(path) != expected:
            raise ValueError(f"Retry preflight evidence changed: {job_id}")
        return
    if plan_dir.exists() and any(plan_dir.iterdir()):
        raise ValueError(f"Retry plan exists without committed preflight evidence: {plan_dir}")
    _recipe, _config, report = preflight_plan(
        recipe_path=recipe_path,
        output_dir=plan_dir,
        unlock_final_test=True,
    )
    if report.exit_code != 0:
        raise RetryPreparationError(f"Retry preflight failed for external job {job_id}.")
    _atomic_write_text(path, json.dumps(expected, indent=2, sort_keys=True) + "\n")


_logical_job_states = pipeline_results.logical_job_states


_read_result_manifest = pipeline_results.read_result_manifest
_validate_result_manifest = pipeline_results.validate_result_manifest


_build_result_rows = pipeline_results.build_result_rows
_write_result_summary = pipeline_results.write_result_summary
_aggregate_results = pipeline_results.aggregate_results
_render_scalar = pipeline_results.render_scalar
_summary_markdown = pipeline_results.summary_markdown


def _mapping_key_paths(payload: Any, target: str, prefix: str = "") -> list[str]:
    paths = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if key == target:
                paths.append(path)
            paths.extend(_mapping_key_paths(value, target, path))
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            paths.extend(_mapping_key_paths(value, target, f"{prefix}[{index}]"))
    return paths


def _mapping(payload: dict[str, Any], field: str) -> dict[str, Any]:
    value = payload.get(field)
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a mapping.")
    return value


def _reject_unknown_fields(payload: dict[str, Any], allowed: set[str], label: str) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"Unknown {label} field(s): {', '.join(unknown)}")


def _required_slug(payload: dict[str, Any], field: str, label: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", value):
        raise ValueError(f"{label}.{field} must use lowercase letters, digits, hyphens, and underscores.")
    return value


def _assert_source_semantics(source_id: str, source: dict[str, Any], recipe: dict[str, Any]) -> None:
    evaluation = recipe.get("evaluation_policy") if isinstance(recipe.get("evaluation_policy"), dict) else {}
    if evaluation.get("selection_metric") != source["selection_metric"]:
        raise ValueError(f"Checkpoint source {source_id} selection metric differs from its plan.")
    if evaluation.get("selection_mode") != source["selection_mode"]:
        raise ValueError(f"Checkpoint source {source_id} selection mode differs from its plan.")
    derived = {
        "task": (recipe.get("inputs") or {}).get("label_name"),
        "variant": recipe.get("variant"),
        "label_name": (recipe.get("inputs") or {}).get("label_name"),
    }
    for field in ("task", "variant", "label_name"):
        assertion = source.get(field)
        if assertion not in (None, "") and assertion != derived[field]:
            raise ValueError(f"Checkpoint source {source_id} {field} assertion differs from its plan.")


def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


_atomic_write_text = pipeline_results.atomic_write_text
_write_rows_atomic = pipeline_results.write_rows_atomic


def _write_state(pipeline_dir: Path, state: dict[str, Any]) -> None:
    _atomic_write_text(
        pipeline_dir / "pipeline.json", json.dumps(state, indent=2, sort_keys=True, allow_nan=True) + "\n"
    )


def _update_state(pipeline_dir: Path, **updates: Any) -> dict[str, Any]:
    state = read_json(pipeline_dir / "pipeline.json")
    state.update(updates)
    state["updated_at"] = utc_now()
    _write_state(pipeline_dir, state)
    return state
