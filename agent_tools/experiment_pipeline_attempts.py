"""Prepare, materialize, and validate managed pipeline attempts and retries.

Layer 2 kernel used by ``experiment_pipeline``. Creates initial attempt rows, reconciles
planned-attempt events, freezes attempt recipes, runs registration preflight, materializes
and registers attempt plans, validates attempt rows against frozen pipeline state, and
prepares retries. Launch, terminal reduction, source freezing, and finalization stay in
``experiment_pipeline``.

Registration and retry failures are distinct recoverable errors rather than one
generic exception, so an interrupted attempt is reconciled on the next run
instead of leaving the pipeline half-registered.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any, cast

import yaml

from . import (
    experiment_io as exp_io,
    experiment_pipeline_cohort_selection as cohort_selection,
    experiment_pipeline_results as pipeline_results,
    managed_scheduler,
    run_artifacts as artifacts,
)
from .experiment_pipeline_spec import FrozenCheckpointCandidate, JobSpec, PipelineSpec
from .experiment_workspace import (
    SUCCESS_STATUSES,
    append_event,
    commit_step_manifest,
    event_matches,
    file_sha256,
    managed_run_key,
    merge_run_manifest,
    next_run_index,
    plan_registration_lock,
    plan_registration_rows_state,
    read_experiment_events,
    read_managed_yaml_mapping,
    read_run_manifest,
    validate_step_registration,
)
from .manifests import read_json, read_rows
from .models import JsonValue
from .plans import build_plan, plan_publication_lock, preflight_plan, publish_staged_plan_locked

RETRYABLE_STATUSES = pipeline_results.RETRYABLE_STATUSES


class RetryPreparationError(RuntimeError):
    pass


class AttemptRegistrationPreflightError(RuntimeError):
    pass


class PipelineRegistrationRecoveryError(RuntimeError):
    pass


def pipeline_execution(spec: PipelineSpec) -> dict[str, JsonValue]:
    return {
        "target": "local",
        "workdir": spec["runtime"]["workdir"],
        "python": spec["runtime"]["python"],
        "runtime_commit": spec["runtime"]["runtime_commit"],
        "gpu_pool": list(spec["execution"]["gpu_pool"]),
        "gpus_per_run": spec["execution"]["gpus_per_run"],
        "max_concurrent": spec["execution"]["max_concurrent"],
    }


def _freeze_attempt_recipe(recipe: dict[str, Any], recipe_path: Path, *, drift_message: str) -> None:
    recipe_text = yaml.safe_dump(recipe, sort_keys=False)
    # Existing attempt recipes are immutable resume evidence; reuse only the same serialized YAML.
    if recipe_path.exists():
        if recipe_path.is_symlink() or not recipe_path.is_file() or recipe_path.read_text() != recipe_text:
            raise ValueError(f"{drift_message}: {recipe_path}")
        return
    _atomic_write_text(recipe_path, recipe_text)


def load_or_create_initial_attempts(
    root: Path,
    pipeline_dir: Path,
    spec: PipelineSpec,
    selections: Mapping[str, FrozenCheckpointCandidate],
    *,
    inspect_target: Callable[..., managed_scheduler.ExecutionSnapshot] | None = None,
) -> list[dict[str, Any]]:
    jobs_path = pipeline_dir / "jobs.tsv"
    existing = read_rows(jobs_path, require_managed_identity=True)
    if existing:
        validate_attempt_rows(root, pipeline_dir, spec, selections, existing, require_all_jobs=False)

    recipes = []
    for job in spec["jobs"]:
        selection = cohort_selection.candidate_for_job(job, selections)
        attempt = 1
        recipe, recipe_path, plan_dir, result_root = _attempt_recipe(pipeline_dir, spec, job, selection, attempt)
        _freeze_attempt_recipe(
            recipe,
            recipe_path,
            drift_message="External job recipe changed during resume",
        )
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
    with plan_registration_lock(root):
        prepared = _prepare_attempt_registration_groups(
            root,
            spec,
            recipes,
            snapshot_owner_dirs=snapshot_owner_dirs,
            inspect_target=inspect_target,
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
                    prepared_plan_dir=prepared[job["id"]],
                )
                attempt_rows.append(row)
                _write_registered_jobs(jobs_path, attempt_rows)
        finally:
            plan_dirs = {job["id"]: plan_dir for job, _selection, _attempt, _recipe, plan_dir, _result in recipes}
            for job_id, physical_plan_dir in prepared.items():
                if (
                    physical_plan_dir != plan_dirs[job_id]
                    and physical_plan_dir.exists()
                    and not physical_plan_dir.is_symlink()
                ):
                    shutil.rmtree(physical_plan_dir)
        validate_attempt_rows(root, pipeline_dir, spec, selections, attempt_rows)
        _reconcile_pipeline_jobs_planned_event(root, spec)
    return attempt_rows


def _reconcile_pipeline_jobs_planned_event(root: Path, spec: PipelineSpec) -> None:
    payload = {"pipeline_id": spec["pipeline"]["id"], "job_count": len(spec["jobs"])}
    identity_fields: tuple[str, ...] = ("pipeline_id",)
    if spec.get("_execution_stage"):
        payload["phase"] = spec["_execution_stage"]
        identity_fields = ("pipeline_id", "phase")
    reconcile_pipeline_event(root, "pipeline_jobs_planned", payload, identity_fields=identity_fields)


def _reconcile_pipeline_retry_planned_event(root: Path, spec: PipelineSpec, attempt: dict[str, Any]) -> None:
    payload = {
        "pipeline_id": spec["pipeline"]["id"],
        "job_id": str(attempt["job_id"]),
        "attempt": int(attempt["attempt"]),
    }
    identity_fields: tuple[str, ...] = ("pipeline_id", "job_id", "attempt")
    if spec.get("_execution_stage"):
        payload["phase"] = spec["_execution_stage"]
        identity_fields = ("pipeline_id", "phase", "job_id", "attempt")
    reconcile_pipeline_event(
        root,
        "pipeline_job_retry_planned",
        payload,
        identity_fields=identity_fields,
    )


def reconcile_pipeline_event(
    root: Path,
    event_type: str,
    payload: dict[str, Any],
    *,
    identity_fields: tuple[str, ...],
) -> None:
    def related_events() -> list[dict[str, Any]]:
        return [
            event
            for event in read_experiment_events(root)
            if event.get("event_type") == event_type
            and all(event.get(field) == payload[field] for field in identity_fields)
        ]

    try:
        related = related_events()
    except (OSError, RuntimeError) as exc:
        raise PipelineRegistrationRecoveryError(f"Pipeline {event_type} event must be reconciled on resume.") from exc
    exact = [event for event in related if event_matches(event, event_type, payload)]
    if len(related) != len(exact) or len(exact) > 1:
        raise ValueError(f"Pipeline {event_type} event conflicts with canonical registration.")
    if exact:
        return
    try:
        append_event(root, event_type, payload)
        related = related_events()
    except (OSError, RuntimeError) as exc:
        raise PipelineRegistrationRecoveryError(f"Pipeline {event_type} event must be reconciled on resume.") from exc
    exact = [event for event in related if event_matches(event, event_type, payload)]
    if len(related) != 1 or len(exact) != 1:
        raise ValueError(f"Pipeline {event_type} event was not committed exactly once.")


def _ensure_initial_preflight(
    pipeline_dir: Path,
    spec: PipelineSpec,
    recipes: list[tuple[JobSpec, FrozenCheckpointCandidate, int, Path, Path, Path]],
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
    selection: FrozenCheckpointCandidate,
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
    spec: PipelineSpec,
    job: JobSpec,
    selection: FrozenCheckpointCandidate,
    attempt: int,
    *,
    recipe_path: Path,
    plan_dir: Path,
    result_root: Path,
    prepared_plan_dir: Path | None = None,
) -> dict[str, Any]:
    # The plan envelope and its canonical ownership must become visible under one publication lock.
    with plan_publication_lock(plan_dir):
        return _materialize_attempt_locked(
            root,
            spec,
            job,
            selection,
            attempt,
            recipe_path=recipe_path,
            plan_dir=plan_dir,
            result_root=result_root,
            prepared_plan_dir=prepared_plan_dir,
        )


def _materialize_attempt_locked(
    root: Path,
    spec: PipelineSpec,
    job: JobSpec,
    selection: FrozenCheckpointCandidate,
    attempt: int,
    *,
    recipe_path: Path,
    plan_dir: Path,
    result_root: Path,
    prepared_plan_dir: Path | None = None,
) -> dict[str, Any]:
    _validate_new_attempt_paths(plan_dir, result_root, allow_existing_plan=True)
    plan_path = plan_dir / "plan.json"
    if plan_dir.exists() and not plan_path.exists():
        raise ValueError(f"External attempt plan is incomplete: {plan_dir}")
    physical_plan_dir = prepared_plan_dir
    staging_dir = None
    if physical_plan_dir is None:
        physical_plan_dir, staging_dir = _prepare_attempt_plan(job["id"], selection, recipe_path, plan_dir)
    elif physical_plan_dir != plan_dir:
        staging_dir = physical_plan_dir
        if plan_dir.exists():
            raise ValueError(f"External attempt plan appeared after registration preflight: {plan_dir}")

    plan = read_json(physical_plan_dir / "plan.json")
    runs = plan.get("runs") if isinstance(plan, dict) else None
    if not isinstance(runs, list) or len(runs) != 1 or not isinstance(runs[0], dict):
        raise ValueError(f"External job plan must contain exactly one managed run: {physical_plan_dir}")
    run = dict(runs[0])
    base_run = {field: value for field, value in run.items() if field != "command"}
    enrichment = {
        "step_id": run["step_id"],
        "run_id": run["run_id"],
        "pipeline_id": spec["pipeline"]["id"],
        "job_id": job["id"],
        "attempt": attempt,
        "result_root": str(result_root),
        "terminal_status_owner": "script",
    }
    registration_row = {**base_run, "parameter_summary": "single resolved recipe", **enrichment}
    registration_state = plan_registration_rows_state(
        root,
        [registration_row],
        source="Canonical pipeline attempt",
    )
    if staging_dir is not None and registration_state == "present":
        raise ValueError(f"Canonical external attempt exists before plan publication: {plan_dir}")
    plan_recipe = plan["recipe"] if isinstance(plan.get("recipe"), dict) else {}
    step_payload = {
        "step": plan_recipe["step"],
        "experiment_id": plan_recipe["experiment"]["id"],
        "plan_controller": "pipeline",
        "recipe_path": plan_recipe.get("_recipe_path", ""),
        "plans": [str(plan_dir.resolve())],
    }
    validate_step_registration(root, step_payload)
    _validate_physical_attempt_plan(
        spec,
        job,
        selection,
        recipe_path,
        plan_dir,
        physical_plan_dir,
        base_run,
    )
    if staging_dir is not None:
        publish_staged_plan_locked(staging_dir, plan_dir, out_preexisted=False)

    # Pipeline jobs commit their own status under the run lock. Callers hold only the registration and publication
    # locks, and merge_run_manifest below takes the run lock itself, so only this read holds it.
    with managed_scheduler.managed_run_lock(root):
        canonical_by_key = {managed_run_key(row): row for row in read_run_manifest(root)}
    canonical = canonical_by_key.get(managed_run_key(run))
    if canonical is not None:
        _validate_attempt_plan(
            {"step_id": run["step_id"], "run_id": run["run_id"], "recipe": str(recipe_path), "plan_dir": str(plan_dir)},
            canonical,
        )
    # Publish pipeline ownership before its row so a crash cannot expose the attempt as an ordinary launch candidate.
    commit_step_manifest(root, step_payload)
    update = registration_row if canonical is None else enrichment
    committed = merge_run_manifest(root, [update])
    canonical = {managed_run_key(row): row for row in committed}[managed_run_key(run)]
    projection = _attempt_projection(job, selection, canonical, recipe_path=recipe_path, plan_dir=plan_dir)
    _validate_attempt_plan(projection, canonical)
    return projection


def _prepare_attempt_registration_groups(
    root: Path,
    spec: PipelineSpec,
    attempts: list[tuple[JobSpec, FrozenCheckpointCandidate, int, Path, Path, Path]],
    *,
    snapshot_owner_dirs: dict[str, Path],
    inspect_target: Callable[..., managed_scheduler.ExecutionSnapshot] | None = None,
) -> dict[str, Path]:
    prepared: dict[str, Path] = {}
    groups: dict[str, list[dict[str, Any]]] = {}
    group_paths: dict[str, list[Path]] = {}
    # Callers hold only the registration lock; the execution-target probe below must not hold the run lock.
    with managed_scheduler.managed_run_lock(root):
        canonical_keys = {managed_run_key(row) for row in read_run_manifest(root)}
    pending = []
    for item in attempts:
        job, _selection, _attempt, _recipe_path, plan_dir, _result_root = item
        plan_path = plan_dir / "plan.json"
        plan = read_json(plan_path) if plan_path.exists() else None
        runs = plan.get("runs") if isinstance(plan, dict) else None
        run = runs[0] if isinstance(runs, list) and len(runs) == 1 and isinstance(runs[0], dict) else None
        if run is not None and managed_run_key(run) in canonical_keys:
            prepared[job["id"]] = plan_dir
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
                physical_plan_dir, _ = _prepare_attempt_plan(
                    job["id"],
                    selection,
                    recipe_path,
                    plan_dir,
                    run_index_offset=pending_run_indices[job["id"]],
                )
                prepared[job["id"]] = physical_plan_dir
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

        execution = pipeline_execution(spec)
        remote = str(execution["host"]) if execution.get("target", "local") == "ssh" else None
        snapshots: dict[str, tuple[Path, managed_scheduler.ExecutionSnapshot]] = {}
        for variant in sorted(groups):
            snapshot_path = snapshot_owner_dirs[variant] / managed_scheduler.EXECUTION_SNAPSHOT_NAME
            exp_io.validate_managed_output_paths(
                Path("/"),
                [*group_paths[variant], snapshot_path],
                remote=remote,
            )
            inspect = inspect_target or managed_scheduler.inspect_execution_target
            snapshot = inspect(execution, groups[variant], plan_label="pipeline")
            if snapshot_path.exists() and read_json(snapshot_path) != snapshot:
                raise ValueError(f"Frozen pipeline execution snapshot changed: {snapshot_path}")
            snapshots[variant] = (snapshot_path, snapshot)
        for snapshot_path, snapshot in snapshots.values():
            if not snapshot_path.exists():
                snapshot_text = json.dumps(snapshot, indent=2, sort_keys=True) + "\n"
                exp_io.conditional_atomic_replace_text_at(
                    snapshot_path,
                    snapshot_text,
                    None,
                    managed_root=root,
                )
            if read_json(snapshot_path) != snapshot:
                raise ValueError(f"Frozen pipeline execution snapshot changed: {snapshot_path}")
        return prepared
    except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as exc:
        plan_dirs = {job["id"]: plan_dir for job, _selection, _attempt, _recipe, plan_dir, _result in attempts}
        for job_id, physical_plan_dir in prepared.items():
            if (
                physical_plan_dir != plan_dirs[job_id]
                and physical_plan_dir.exists()
                and not physical_plan_dir.is_symlink()
            ):
                shutil.rmtree(physical_plan_dir)
        raise AttemptRegistrationPreflightError(f"External attempt registration preflight failed: {exc}") from exc


def _validate_physical_attempt_plan(
    spec: PipelineSpec,
    job: JobSpec,
    selection: FrozenCheckpointCandidate,
    recipe_path: Path,
    plan_dir: Path,
    physical_plan_dir: Path,
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
    for field, expected_path in expected_paths.items():
        if Path(str(run.get(field) or "")) != expected_path:
            raise ValueError(f"External attempt plan path differs from its managed directory: {field}")
    physical_config = artifacts._physical_plan_path(Path(run["config"]), plan_dir, physical_plan_dir)
    if file_sha256(physical_config) != selection["config_sha256"]:
        raise ValueError(f"External attempt config differs from its selected source: {job['id']}")
    _validate_attempt_plan(
        {"step_id": run["step_id"], "run_id": run["run_id"], "recipe": str(recipe_path), "plan_dir": str(plan_dir)},
        run,
        physical_plan_dir=physical_plan_dir,
    )


def _attempt_recipe(
    pipeline_dir: Path,
    spec: PipelineSpec,
    job: JobSpec,
    selection: FrozenCheckpointCandidate,
    attempt: int,
) -> tuple[dict[str, Any], Path, Path, Path]:
    attempt_name = f"attempt-{attempt:03d}"
    recipe_path = pipeline_dir / "recipes" / job["id"] / f"{attempt_name}.yaml"
    plan_dir = pipeline_dir / "plans" / job["id"] / attempt_name
    result_root = pipeline_dir / "results" / job["id"] / attempt_name
    runtime = spec["runtime"]
    controller_dir = Path(str(spec.get("_controller_dir") or pipeline_dir))
    experiment_path = controller_dir.parent.parent / "experiment.yaml"
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
    job: JobSpec,
    selection: FrozenCheckpointCandidate,
    run: dict[str, Any],
    *,
    recipe_path: Path,
    plan_dir: Path,
) -> dict[str, Any]:
    projection = {
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
    for field in ("candidate_id", "job_template_id", "role", "provenance"):
        if field in job:
            projection[field] = job[field]
    return projection


def validate_attempt_rows(
    root: Path,
    pipeline_dir: Path,
    spec: PipelineSpec,
    selections: Mapping[str, FrozenCheckpointCandidate],
    rows: list[dict[str, Any]],
    *,
    require_all_jobs: bool = True,
) -> None:
    # Pipeline jobs commit their own status under the run lock; no caller holds it here.
    with managed_scheduler.managed_run_lock(root):
        canonical = {managed_run_key(row): row for row in read_run_manifest(root)}
    jobs = {job["id"]: job for job in spec["jobs"]}
    seen_attempts = set()
    for row in rows:
        # Canonical and registered attempt readers require both managed identity fields.
        key = cast(tuple[str, str], managed_run_key(row))
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
        selection = cohort_selection.candidate_for_job(job, selections)
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
        for field in ("candidate_id", "job_template_id", "role", "provenance"):
            if field in job:
                expected[field] = job[field]
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


def _validate_attempt_plan(
    row: dict[str, Any], canonical_run: dict[str, Any], *, physical_plan_dir: Path | None = None
) -> None:
    recipe_path = Path(str(row["recipe"]))
    plan_dir = Path(str(row["plan_dir"]))
    physical_plan_dir = plan_dir if physical_plan_dir is None else physical_plan_dir
    plan_path = physical_plan_dir / "plan.json"
    resolved_recipe_path = physical_plan_dir / "recipe.resolved.yaml"
    if recipe_path.is_symlink() or not recipe_path.is_file():
        raise ValueError(f"Pipeline attempt recipe is missing or aliased: {recipe_path}")
    if physical_plan_dir.is_symlink() or not physical_plan_dir.is_dir():
        raise ValueError(f"Pipeline attempt plan directory is missing or aliased: {physical_plan_dir}")
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
        physical_path = artifacts._physical_plan_path(path, plan_dir, physical_plan_dir)
        if (
            physical_path.is_symlink()
            or not physical_path.is_file()
            or file_sha256(physical_path) != canonical_run.get(hash_field)
        ):
            raise ValueError(f"Pipeline attempt {path_field} changed: {physical_path}")
        try:
            path.relative_to(plan_dir)
        except ValueError as exc:
            raise ValueError(f"Pipeline attempt {path_field} is outside its plan: {path}") from exc
    physical_script = artifacts._physical_plan_path(Path(str(canonical_run["script"])), plan_dir, physical_plan_dir)
    if commands[0] not in physical_script.read_text().splitlines():
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


def write_jobs(path: Path, rows: list[dict[str, Any]]) -> None:
    _write_rows_atomic(path, rows)


def _write_registered_jobs(path: Path, rows: list[dict[str, Any]]) -> None:
    try:
        write_jobs(path, rows)
    except (OSError, RuntimeError) as exc:
        raise PipelineRegistrationRecoveryError("Pipeline jobs projection must be reconciled on resume.") from exc


def planned_runs(attempt_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
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


def create_needed_retries(
    root: Path,
    pipeline_dir: Path,
    spec: PipelineSpec,
    selections: Mapping[str, FrozenCheckpointCandidate],
    attempts: list[dict[str, Any]],
    *,
    inspect_target: Callable[..., managed_scheduler.ExecutionSnapshot] | None = None,
) -> tuple[list[dict[str, Any]], bool]:
    created = False
    state_changed = False
    for row in attempts:
        if int(row["attempt"]) > 1:
            _reconcile_pipeline_retry_planned_event(root, spec, row)
    with managed_scheduler.managed_run_lock(root):
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
        selection = cohort_selection.candidate_for_job(job, selections)
        attempt = int(latest["attempt"]) + 1
        recipe, recipe_path, plan_dir, result_root = _attempt_recipe(pipeline_dir, spec, job, selection, attempt)
        _freeze_attempt_recipe(recipe, recipe_path, drift_message="Retry recipe changed during resume")
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
        write_jobs(pipeline_dir / "jobs.tsv", attempts)

    for latest, job, selection, attempt, recipe_path, plan_dir, result_root in ready:
        physical_plan_dir = None
        try:
            with plan_registration_lock(root):
                try:
                    prepared = _prepare_attempt_registration_groups(
                        root,
                        spec,
                        [(job, selection, attempt, recipe_path, plan_dir, result_root)],
                        snapshot_owner_dirs={str(selection["variant"]): pipeline_dir / "retry_schedulers" / job["id"]},
                        inspect_target=inspect_target,
                    )
                    physical_plan_dir = prepared[job["id"]]
                except AttemptRegistrationPreflightError as exc:
                    latest["retry_preparation_error"] = str(exc)
                    write_jobs(pipeline_dir / "jobs.tsv", attempts)
                    continue
                retry_row = _materialize_attempt(
                    root,
                    spec,
                    job,
                    selection,
                    attempt,
                    recipe_path=recipe_path,
                    plan_dir=plan_dir,
                    result_root=result_root,
                    prepared_plan_dir=physical_plan_dir,
                )
                attempts.append(retry_row)
                _write_registered_jobs(pipeline_dir / "jobs.tsv", attempts)
                _reconcile_pipeline_retry_planned_event(root, spec, retry_row)
        finally:
            if (
                physical_plan_dir is not None
                and physical_plan_dir != plan_dir
                and physical_plan_dir.exists()
                and not physical_plan_dir.is_symlink()
            ):
                shutil.rmtree(physical_plan_dir)
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


_validate_result_manifest = pipeline_results.validate_result_manifest
_atomic_write_text = pipeline_results.atomic_write_text
_write_rows_atomic = pipeline_results.write_rows_atomic
