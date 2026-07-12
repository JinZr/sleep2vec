from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .experiment_workspace import (
    experiment_metadata_issues,
    experiment_root,
    managed_run_key,
    managed_run_parameters,
    merge_step_manifest,
    read_managed_yaml_mapping,
    read_run_manifest,
    read_step_manifest,
    validate_managed_run_rows,
    verify_run_snapshot,
)
from .manifests import read_json
from .models import REPO_ROOT

RUN_METADATA_FIELDS = ("experiment_id", "run_name", "version")


def read_hparam_plan(run_dir: Path) -> dict[str, Any]:
    plan_path = run_dir / "plan.json"
    if not plan_path.exists():
        raise FileNotFoundError(f"Missing hparam plan: {plan_path}")
    plan = read_json(plan_path)
    if "trials" in plan:
        raise ValueError(f"Legacy hparam plan is read-only and cannot be managed: {plan_path}")
    legacy_status = run_dir / "trial_status.tsv"
    if legacy_status.exists():
        raise ValueError(f"Legacy hparam status is read-only and cannot be managed: {legacy_status}")
    runs = plan.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError(f"Hparam plan must define a non-empty runs list: {plan_path}")
    validate_run_rows(runs, source=str(plan_path), require_artifact_paths=True)
    recipe = plan.get("recipe") if isinstance(plan.get("recipe"), dict) else {}
    metadata_issues = experiment_metadata_issues(recipe)
    if metadata_issues:
        raise ValueError(
            "Invalid hparam workspace binding: " + "; ".join(issue["message"] for issue in metadata_issues)
        )
    workspace = experiment_root(recipe)
    if workspace is None:
        raise ValueError("Invalid hparam workspace binding: experiment.root is required.")
    try:
        run_dir.resolve().relative_to(workspace.resolve())
    except ValueError as exc:
        raise ValueError(f"Hparam plan must be inside experiment.root: {workspace}") from exc
    experiment_manifest_path = workspace / "experiment.yaml"
    step_id = str(recipe["step"]["id"])
    if not experiment_manifest_path.exists():
        raise ValueError(f"Hparam plan is not bound to an initialized experiment workspace: {workspace}")
    experiment_manifest = read_managed_yaml_mapping(
        experiment_manifest_path.read_text(),
        source=f"Managed experiment manifest {experiment_manifest_path}",
    )
    existing_experiment = experiment_manifest.get("experiment") if isinstance(experiment_manifest, dict) else None
    expected_experiment = recipe["experiment"]
    if not isinstance(existing_experiment, dict) or any(
        existing_experiment.get(field) != expected_experiment.get(field)
        for field in ("id", "title", "objective", "root", "baseline")
    ):
        raise ValueError(f"Hparam plan experiment metadata differs from the managed workspace: {workspace}")
    step_manifest = read_step_manifest(workspace, step_id)
    experiment_id = str(expected_experiment["id"])
    expected_step_manifest = merge_step_manifest(
        step_manifest,
        {
            "step": recipe["step"],
            "experiment_id": experiment_id,
            "recipe_path": recipe.get("_recipe_path", ""),
            "plans": [str(run_dir.resolve())],
        },
    )
    if expected_step_manifest != step_manifest:
        raise ValueError(f"Hparam plan is not registered by its managed step: {run_dir}")
    for run in runs:
        if str(run["experiment_id"]) != experiment_id or str(run["step_id"]) != step_id:
            raise ValueError("Managed run identity does not match the hparam recipe workspace binding.")
    workspace_rows = read_run_manifest(workspace)
    workspace_by_key = {managed_run_key(row): row for row in workspace_rows}
    missing_keys = [managed_run_key(run) for run in runs if managed_run_key(run) not in workspace_by_key]
    if missing_keys:
        missing = ", ".join(f"{step} / {run_id}" for step, run_id in missing_keys)
        raise ValueError(f"Workspace run_manifest.tsv is missing plan runs: {missing}")
    for run in runs:
        workspace_row = workspace_by_key[managed_run_key(run)]
        if workspace_row.get("status") in (None, ""):
            raise ValueError(f"Workspace run manifest is missing status: {run['step_id']} / {run['run_id']}")
        for field in (
            "experiment_id",
            "step_id",
            "run_id",
            "run_name",
            "parameter_summary",
            "version",
            "config",
            "config_sha256",
            "script",
            "script_sha256",
            "run_dir",
            "artifacts",
            "runtime_dir",
            "checkpoint_dir",
        ):
            if str(workspace_row.get(field) or "") != str(run.get(field) or ""):
                raise ValueError(
                    f"Workspace run manifest differs from plan field {field}: {run['step_id']} / {run['run_id']}"
                )
        plan_parameters = managed_run_parameters(run)
        workspace_parameters = managed_run_parameters(workspace_row)
        if set(plan_parameters) != set(workspace_parameters):
            raise ValueError(f"Workspace run parameters differ from plan: {run['step_id']} / {run['run_id']}")
        for field, value in plan_parameters.items():
            expected_value = "" if value is None else str(value)
            actual_value = "" if workspace_parameters[field] is None else str(workspace_parameters[field])
            if actual_value != expected_value:
                raise ValueError(
                    f"Workspace run manifest differs from plan field {field}: {run['step_id']} / {run['run_id']}"
                )
    search = recipe.get("search") if isinstance(recipe.get("search"), dict) else {}
    execution = recipe.get("execution") if isinstance(recipe.get("execution"), dict) else {}
    adaptive = recipe.get("adaptive") if isinstance(recipe.get("adaptive"), dict) else {}
    workdir = execution.get("workdir")
    if workdir not in (None, "") and not Path(str(workdir)).is_absolute():
        raise ValueError("execution.workdir must be an absolute path when set.")
    run_cwd = Path(str(workdir or REPO_ROOT))
    for run in runs:
        expected_runtime_dir = run_cwd / "log-finetune" / str(run["version"])
        expected_checkpoint_dir = expected_runtime_dir / "checkpoints"
        if str(run["runtime_dir"]) != str(expected_runtime_dir):
            raise ValueError(f"Managed run runtime_dir differs from execution.workdir: {run['run_id']}")
        if str(run["checkpoint_dir"]) != str(expected_checkpoint_dir):
            raise ValueError(f"Managed run checkpoint_dir differs from execution.workdir: {run['run_id']}")
    legacy_fields = [
        name
        for name, present in (
            ("search.max_trials", "max_trials" in search),
            ("execution.gpus_per_trial", "gpus_per_trial" in execution),
            ("adaptive.max_trials_total", "max_trials_total" in adaptive),
        )
        if present
    ]
    if legacy_fields:
        raise ValueError(f"Legacy hparam fields are read-only and unsupported: {', '.join(legacy_fields)}")
    for run in runs:
        verify_run_snapshot(run)
    return plan


def validate_run_rows(
    rows: list[dict[str, Any]],
    *,
    source: str,
    require_artifact_paths: bool = False,
) -> None:
    validate_managed_run_rows(rows, source=source, cardinality="one_per_run")
    versions = set()
    for index, row in enumerate(rows):
        missing = [field for field in RUN_METADATA_FIELDS if row.get(field) in (None, "")]
        if require_artifact_paths:
            missing.extend(
                field
                for field in (
                    "run_dir",
                    "runtime_dir",
                    "checkpoint_dir",
                    "config",
                    "config_sha256",
                    "script",
                    "script_sha256",
                    "artifacts",
                )
                if row.get(field) in (None, "")
            )
        if missing:
            raise ValueError(f"Managed run row {index} in {source} is missing: {', '.join(missing)}")
        if require_artifact_paths:
            relative_paths = [
                field
                for field in ("run_dir", "runtime_dir", "checkpoint_dir", "config", "script", "artifacts")
                if not Path(str(row[field])).is_absolute()
            ]
            if relative_paths:
                raise ValueError(
                    f"Managed run row {index} in {source} has non-absolute paths: {', '.join(relative_paths)}"
                )
        version = str(row["version"])
        if version in versions:
            raise ValueError(f"Duplicate managed run version in {source}: {version}")
        versions.add(version)


def find_run_manifest(run: dict[str, Any]) -> Path | None:
    if not run.get("runtime_dir"):
        return None
    path = Path(str(run["runtime_dir"])) / "run_manifest.json"
    return path if path.exists() else None


def metric_value(manifest: dict[str, Any], metric: str) -> float | str:
    metrics = manifest.get("metrics") if isinstance(manifest.get("metrics"), dict) else {}
    if metric in metrics:
        return metrics[metric]
    if manifest.get("monitor") == metric and manifest.get("best_model_score") is not None:
        return manifest["best_model_score"]
    return ""


def fixed_checkpoint_path(manifest: dict[str, Any], checkpoint_dir: Path) -> str:
    raw = manifest.get("best_model_path") or manifest.get("checkpoint_path") or ""
    if raw:
        path = Path(str(raw))
        if path.name.startswith("best-epoch="):
            fixed = checkpoint_dir / path.name.removeprefix("best-")
            if fixed.exists():
                return str(fixed)
            matched = checkpoint_for_epoch_in_dir(checkpoint_dir, epoch_number_from_checkpoint_name(fixed.name))
            if matched:
                return str(matched)
            return ""
        if path.name.startswith("epoch="):
            fixed = checkpoint_dir / path.name
            return str(fixed) if fixed.exists() else ""
        matched = checkpoint_for_epoch_in_dir(checkpoint_dir, epoch_number(manifest.get("epoch")))
        if matched:
            return str(matched)
        return ""
    checkpoints = sorted(checkpoint_dir.glob("epoch=*.ckpt"))
    if checkpoints:
        return str(checkpoints[-1])
    return ""


def checkpoint_names(run: dict[str, Any]) -> list[str]:
    if not run.get("checkpoint_dir"):
        return []
    ckpt_dir = Path(str(run["checkpoint_dir"]))
    if not ckpt_dir.exists():
        return []
    return [path.name for path in sorted(ckpt_dir.glob("*.ckpt"))]


def checkpoint_for_epoch_in_dir(ckpt_dir: Path, epoch: int | None) -> Path | None:
    if epoch is None:
        return None
    for path in sorted(ckpt_dir.glob("epoch=*.ckpt")):
        if not path.name.startswith("best-") and epoch_number_from_checkpoint_name(path.name) == epoch:
            return path
    return None


def epoch_from_checkpoint_name(name: str) -> str:
    if not name.startswith("epoch="):
        return ""
    return name.split("=", 1)[1].split("-", 1)[0].split(".", 1)[0]


def epoch_number(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(str(value)))
    except ValueError:
        return None


def epoch_number_from_checkpoint_name(name: str) -> int | None:
    return epoch_number(epoch_from_checkpoint_name(name))


def float_or_none(value: Any) -> float | None:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    return score if math.isfinite(score) else None


def sortable_score(value: Any, reverse: bool) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return -math.inf if reverse else math.inf
    return score
