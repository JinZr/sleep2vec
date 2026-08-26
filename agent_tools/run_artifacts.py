from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
from typing import Any, Iterator

import yaml

from . import decision_rules as task_rules, experiment_io as exp_io, plan_contract
from .adapters import get_adapter
from .experiment_workspace import (
    SCHEDULER_PLAN_IDENTITY_FIELDS,
    experiment_metadata_issues,
    experiment_root,
    file_sha256,
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
from .recipes import merge_recipe_layers

RUN_METADATA_FIELDS = ("experiment_id", "run_name", "version")
REGISTERED_PLAN_IDENTITY_FIELDS = (
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
    "runtime_dir",
    "checkpoint_dir",
    *sorted(SCHEDULER_PLAN_IDENTITY_FIELDS),
)


def _json_object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f"duplicate JSON key: {key}")
        payload[key] = value
    return payload


def _resolved_recipe_view(recipe: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in recipe.items() if key != "_recipe_path"}


def _read_plan_documents(
    plan_dir: Path,
    *,
    workspace: Path | None = None,
    remote: str | None = None,
    strict_control_bundle: bool = False,
    require_resolved_sha256: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    plan_path = plan_dir / "plan.json"
    resolved_recipe_path = plan_dir / "recipe.resolved.yaml"
    if strict_control_bundle:
        if workspace is None:
            raise ValueError("Strict registered-plan reads require a workspace root.")
        files = exp_io.read_managed_files_at(
            workspace,
            [plan_path, resolved_recipe_path],
            remote=remote,
        )
        try:
            plan = json.loads(
                files[str(plan_path)]["text"],
                object_pairs_hook=_json_object_without_duplicate_keys,
            )
        except ValueError as exc:
            raise ValueError(f"Registered plan manifest is corrupt: {plan_path}: {exc}") from exc
        resolved_recipe_text = files[str(resolved_recipe_path)]["text"]
        resolved_recipe_sha256 = files[str(resolved_recipe_path)]["sha256"]
    else:
        if not plan_path.exists():
            raise FileNotFoundError(f"Missing hparam plan: {plan_path}")
        plan = read_json(plan_path)
        if isinstance(plan, dict) and "trials" in plan:
            raise ValueError(f"Legacy hparam plan is read-only and cannot be managed: {plan_path}")
        if not resolved_recipe_path.exists():
            raise FileNotFoundError(f"Missing frozen hparam recipe: {resolved_recipe_path}")
        resolved_recipe_bytes = resolved_recipe_path.read_bytes()
        resolved_recipe_text = resolved_recipe_bytes.decode()
        resolved_recipe_sha256 = hashlib.sha256(resolved_recipe_bytes).hexdigest()
    if not isinstance(plan, dict):
        raise ValueError(f"Registered plan manifest must be a mapping: {plan_path}")
    if "trials" in plan:
        raise ValueError(f"Legacy hparam plan is read-only and cannot be managed: {plan_path}")
    expected_resolved_sha256 = plan.get("resolved_recipe_sha256")
    if require_resolved_sha256 and expected_resolved_sha256 != resolved_recipe_sha256:
        raise ValueError(f"Frozen hparam recipe SHA-256 is missing or changed: {resolved_recipe_path}")
    if (
        strict_control_bundle
        and expected_resolved_sha256 not in (None, "")
        and expected_resolved_sha256 != resolved_recipe_sha256
    ):
        raise ValueError(f"Frozen registered recipe SHA-256 changed: {resolved_recipe_path}")
    resolved_recipe = read_managed_yaml_mapping(
        resolved_recipe_text,
        source=f"Frozen registered recipe {resolved_recipe_path}",
    )
    return plan, resolved_recipe


def is_registered_blocked_plan(
    plan_dir: str | Path,
    *,
    workspace: str | Path,
    remote: str | None = None,
) -> bool:
    plan_dir = Path(plan_dir)
    workspace = Path(workspace)
    # Reject corrupt registrations before probing paths outside the canonical workspace.
    if not workspace.is_absolute() or not plan_dir.is_absolute() or ".." in workspace.parts or ".." in plan_dir.parts:
        raise ValueError(f"Registered plan must use an absolute canonical workspace path: {plan_dir}")
    try:
        plan_dir.relative_to(workspace)
    except ValueError as exc:
        raise ValueError(f"Registered plan is outside its managed workspace: {plan_dir}") from exc
    plan_path = plan_dir / "plan.json"
    resolved_recipe_path = plan_dir / "recipe.resolved.yaml"
    blocked_only_paths = [
        plan_dir / "questions.json",
        plan_dir / "questions.md",
        plan_dir / "plan.blocked.md",
        plan_dir / "plan.draft.json",
    ]
    # Existence checks may follow local aliases, so validate every possible control file's ancestry first.
    try:
        exp_io.validate_managed_output_paths(
            workspace,
            [plan_path, resolved_recipe_path, *blocked_only_paths],
            remote=remote,
        )
    except ValueError as exc:
        raise ValueError(f"Registered plan control bundle is missing or aliased: {plan_dir}") from exc
    plan_exists = exp_io.path_exists_at(plan_path, remote=remote)
    blocked_entries = [path for path in blocked_only_paths if exp_io.path_exists_at(path, remote=remote)]
    if plan_exists:
        if blocked_entries:
            raise ValueError(f"Registered plan contains both PASS and blocked planning artifacts: {plan_dir}")
        return False
    blocked_path = plan_dir / "plan.blocked.md"
    if not exp_io.path_exists_at(blocked_path, remote=remote) or exp_io.path_exists_at(
        resolved_recipe_path, remote=remote
    ):
        return False
    blocked_files = [plan_dir / "questions.json", plan_dir / "questions.md", blocked_path]
    draft_path = plan_dir / "plan.draft.json"
    if draft_path in blocked_entries:
        blocked_files.append(draft_path)
    exp_io.read_managed_files_at(
        workspace,
        blocked_files,
        remote=remote,
        exact_directory_entries=True,
    )
    return True


def read_registered_plan(
    plan_dir: str | Path,
    *,
    workspace: str | Path,
    workspace_experiment: dict[str, Any],
    step_manifest: dict[str, Any],
    workspace_rows: list[dict[str, Any]],
    expected_recipe_path: str | None,
    remote: str | None = None,
    run_index_offset: int = 0,
) -> dict[str, Any]:
    plan_dir = Path(plan_dir)
    workspace = Path(workspace)
    registered_paths = [str(path) for path in step_manifest.get("plans") or []]
    if str(plan_dir) not in registered_paths:
        raise ValueError(f"Plan is not registered by its managed step: {plan_dir}")

    plan_path = plan_dir / "plan.json"
    resolved_recipe_path = plan_dir / "recipe.resolved.yaml"
    plan, resolved_recipe = _read_plan_documents(
        plan_dir,
        workspace=workspace,
        remote=remote,
        strict_control_bundle=True,
    )
    legacy_status = plan_dir / "trial_status.tsv"
    if exp_io.path_exists_at(legacy_status, remote=remote):
        raise ValueError(f"Legacy hparam status is read-only and cannot be managed: {legacy_status}")
    if plan.get("status") != "PASS":
        raise ValueError(f"Registered plan must have status PASS: {plan_path}")

    recipe = plan.get("recipe") if isinstance(plan.get("recipe"), dict) else None
    if recipe is None:
        raise ValueError(f"Registered plan is missing its recipe: {plan_path}")
    task = recipe.get("task")
    adapter = get_adapter(task if isinstance(task, str) else None)
    allowed_internal_fields = {"_plan_context", "_recipe_path"}
    if adapter is not None and adapter.base_task is not None:
        allowed_internal_fields.update({"_base_recipe", "_local_recipe"})
    unexpected_internal_fields = sorted(
        str(field) for field in recipe if str(field).startswith("_") and field not in allowed_internal_fields
    )
    if unexpected_internal_fields:
        raise ValueError(
            f"Registered plan recipe has unsupported internal fields: {', '.join(unexpected_internal_fields)}"
        )
    recipe_path = recipe.get("_recipe_path")
    if not isinstance(recipe_path, str) or not Path(recipe_path).is_absolute():
        raise ValueError(f"Registered plan recipe path must be absolute: {plan_path}")
    if adapter is not None and adapter.materializes_plan and plan.get("resolved_recipe_sha256") in (None, ""):
        raise ValueError(f"Frozen hparam recipe SHA-256 is missing or changed: {resolved_recipe_path}")
    frozen_recipe = _resolved_recipe_view(recipe)
    if frozen_recipe != resolved_recipe:
        raise ValueError(f"Registered plan recipe differs from recipe.resolved.yaml: {resolved_recipe_path}")

    structure_issues = []
    if adapter is not None and adapter.base_task is not None:
        base_recipe = recipe.get("_base_recipe")
        local_recipe = recipe.get("_local_recipe")
        if not isinstance(base_recipe, dict) or not isinstance(local_recipe, dict):
            raise ValueError(f"Registered composite plan is missing frozen base/local recipe layers: {plan_path}")
        for layer_name, layer_task, layer in (
            ("base", adapter.base_task, base_recipe),
            ("local", adapter.task, local_recipe),
        ):
            unexpected_layer_fields = sorted(
                str(field) for field in layer if str(field).startswith("_") and field != "_recipe_path"
            )
            if unexpected_layer_fields:
                raise ValueError(
                    f"Registered {layer_name} recipe has unsupported internal fields: "
                    + ", ".join(unexpected_layer_fields)
                )
            if layer.get("task") != layer_task:
                raise ValueError(f"Registered {layer_name} recipe task differs from its adapter owner: {plan_path}")
            structure_issues.extend(task_rules.recipe_structure_issues(layer_task, layer, source_layer=layer_name))
            layer_metadata_issues = experiment_metadata_issues(
                layer,
                require_values=False,
                source_layer=layer_name,
            )
            if layer_metadata_issues:
                messages = "; ".join(issue["message"] for issue in layer_metadata_issues)
                raise ValueError(f"Invalid registered {layer_name} recipe binding: {messages}")
        effective_recipe = {key: value for key, value in recipe.items() if key not in {"_base_recipe", "_local_recipe"}}
        effective_overlay = _mapping_overlay(merge_recipe_layers(base_recipe, local_recipe), effective_recipe)
        effective_overlay.update({"task": task, "variant": recipe.get("variant")})
        structure_issues.extend(task_rules.recipe_structure_issues(task, effective_overlay, source_layer="effective"))
        for field in ("task", "variant", "adaptive"):
            if recipe.get(field) != local_recipe.get(field):
                raise ValueError(f"Registered composite recipe field differs from its local layer: {field}")
    else:
        structure_issues.extend(task_rules.recipe_structure_issues(task, recipe, source_layer="effective"))
    if structure_issues:
        messages = "; ".join(f"{issue.field}: {issue.message}" for issue in structure_issues)
        raise ValueError(f"Invalid registered plan recipe: {messages}")
    assert adapter is not None

    metadata_issues = experiment_metadata_issues(recipe)
    if metadata_issues:
        raise ValueError("Invalid registered plan binding: " + "; ".join(issue["message"] for issue in metadata_issues))
    experiment = recipe["experiment"]
    step = recipe["step"]
    expected_experiment = {
        field: workspace_experiment.get(field) for field in ("id", "title", "objective", "root", "baseline")
    }
    if experiment != expected_experiment:
        raise ValueError(f"Registered plan experiment metadata differs from the managed workspace: {plan_dir}")
    if str(experiment.get("id") or "") != str(step_manifest.get("experiment_id") or ""):
        raise ValueError(f"Registered plan belongs to a different experiment: {plan_dir}")
    managed_step = {field: step_manifest["step"][field] for field in ("id", "phase", "purpose")}
    if step != managed_step:
        raise ValueError(f"Registered plan step metadata differs from its managed step: {plan_dir}")

    if adapter.materializes_plan:
        contract = _compile_registered_plan_contract(
            adapter,
            recipe,
            plan_dir,
            run_index_offset=run_index_offset,
            config_bytes=b"",
        )
    else:
        layout = plan_contract.generic_run_contract(recipe, plan_dir, run_index_offset, adapter)
        contract = None

    plan_controller = step_manifest["plan_controller"]
    adaptive_enabled = isinstance(recipe.get("adaptive"), dict) and recipe["adaptive"].get("enabled") is True
    if plan_controller == "unassigned" or (plan_controller == "adaptive") != adaptive_enabled:
        raise ValueError(f"Registered plan controller differs from its frozen recipe: {plan_dir}")
    runs = plan.get("runs")
    if not isinstance(runs, list) or not runs or any(not isinstance(run, dict) for run in runs):
        raise ValueError(f"Registered plan must define a non-empty runs list of mappings: {plan_path}")
    validate_run_rows(
        runs,
        source=str(plan_path),
        require_artifact_paths=True,
        allow_empty_runtime_paths=task not in {"finetune", "hparam_tune"},
    )
    expected_runs = contract["runs"] if contract is not None else [layout]
    _validate_plan_contract_runs(runs, expected_runs, plan_path)
    plan_keys = [managed_run_key(run) for run in runs]
    if len(plan_keys) != len(set(plan_keys)):
        raise ValueError(f"Registered plan contains duplicate managed run keys: {plan_path}")
    canonical_by_key = {managed_run_key(row): row for row in workspace_rows}
    for run in runs:
        key = managed_run_key(run)
        canonical = canonical_by_key.get(key)
        if canonical is None:
            raise ValueError(f"Workspace run_manifest.tsv is missing registered plan run: {key[0]} / {key[1]}")
        if canonical.get("status") in (None, ""):
            raise ValueError(f"Workspace run manifest is missing status: {key[0]} / {key[1]}")
        pipeline_fields = ("pipeline_id", "job_id", "attempt", "result_root")
        pipeline_values = [_text_value(canonical.get(field)) for field in pipeline_fields]
        if plan_controller == "pipeline":
            if task == "hparam_tune" or not all(pipeline_values) or canonical.get("terminal_status_owner") != "script":
                raise ValueError(f"Workspace pipeline run identity is incomplete: {key[0]} / {key[1]}")
        elif any(pipeline_values):
            raise ValueError(f"Workspace run pipeline identity conflicts with its managed step: {key[0]} / {key[1]}")
        identity_fields = list(REGISTERED_PLAN_IDENTITY_FIELDS)
        if task == "hparam_tune":
            identity_fields.extend(("parameter_summary", "terminal_status_owner"))
        else:
            identity_fields.append("input_snapshots")
            if _text_value(canonical.get("parameter_summary")) != "single resolved recipe":
                raise ValueError(
                    f"Workspace run manifest differs from plan field parameter_summary: {key[0]} / {key[1]}"
                )
            if plan_controller != "pipeline":
                identity_fields.append("terminal_status_owner")
        if run.get("scheduler_type") == "slurm":
            identity_fields.append("log_path")
        for field in identity_fields:
            if _text_value(canonical.get(field)) != _text_value(run.get(field)):
                raise ValueError(f"Workspace run manifest differs from plan field {field}: {key[0]} / {key[1]}")
        _validate_registered_run_parameters(recipe, run, canonical)

    bundle_paths = []
    for run, expected_run in zip(runs, expected_runs):
        for path_field in ("config", "script", "scheduler_script"):
            path = expected_run.get(path_field)
            if path not in (None, ""):
                bundle_paths.append(Path(str(path)))
        bundle_paths.append(Path(str(expected_run["artifacts"])))
    source_config = plan_dir / "config.source.yaml"
    if adapter.materializes_plan:
        bundle_paths.append(source_config)

    launch_script = plan_dir / ("run_all.sh" if adapter.materializes_plan else "run.sh")
    bundle_paths.append(launch_script)
    final_path, expected_final_command = plan_contract.validate_final_eval_contract(
        plan, recipe, plan_dir, contract or {}
    )
    if final_path is not None:
        bundle_paths.append(final_path)

    final_script = plan_dir / "final_external_test.sh"
    final_script_present = exp_io.path_exists_at(final_script, remote=remote)
    if (expected_final_command is not None) != final_script_present:
        requirement = "missing" if expected_final_command is not None else "unexpected"
        raise ValueError(f"Registered plan has {requirement} final external-test script: {final_script}")
    if expected_final_command is not None:
        bundle_paths.append(final_script)

    bundle = exp_io.read_managed_files_at(
        workspace,
        list(dict.fromkeys(bundle_paths)),
        remote=remote,
    )
    if adapter.materializes_plan:
        contract = _compile_registered_plan_contract(
            adapter,
            recipe,
            plan_dir,
            run_index_offset=run_index_offset,
            config_bytes=bundle[str(source_config)]["text"].encode(),
        )
        expected_runs = contract["runs"]
        _validate_plan_contract_runs(runs, expected_runs, plan_path)
        if bundle[str(launch_script)]["text"] != contract["launch_script_text"]:
            raise ValueError(f"Registered plan launch script differs from its frozen recipe: {launch_script}")
        for run, run_files in zip(runs, contract["run_files"]):
            if bundle[run["config"]]["text"].encode() != run_files["config_bytes"]:
                raise ValueError(f"Registered plan config differs from its frozen recipe: {run['run_id']}")
            if bundle[run["script"]]["text"] != run_files["script_text"]:
                raise ValueError(f"Registered plan script differs from its frozen recipe: {run['run_id']}")
            scheduler_text = run_files.get("scheduler_script_text")
            if scheduler_text is not None and bundle[run["scheduler_script"]]["text"] != scheduler_text:
                raise ValueError(f"Registered Slurm script differs from its frozen recipe: {run['run_id']}")
        if expected_final_command is not None and bundle[str(final_script)]["text"] != contract["final_script_text"]:
            raise ValueError(f"Registered final external-test script differs from its frozen recipe: {final_script}")
    else:
        config_bytes = bundle[layout["config"]]["text"].encode()
        contract = _compile_registered_plan_contract(
            adapter,
            recipe,
            plan_dir,
            run_index_offset=run_index_offset,
            config_bytes=config_bytes,
        )
        expected_runs = contract["runs"]
        _validate_plan_contract_runs(runs, expected_runs, plan_path)
        commands = plan.get("commands")
        if (
            not isinstance(commands, list)
            or not commands
            or any(not isinstance(command, str) or not command for command in commands)
        ):
            raise ValueError(f"Registered plan commands are invalid: {plan_path}")
        if len(runs) != 1:
            raise ValueError(f"Generic registered plan must contain exactly one run: {plan_path}")
        if commands != contract["commands"]:
            raise ValueError(f"Registered plan commands differ from its frozen recipe: {plan_path}")
        if bundle[str(launch_script)]["text"] != contract["script_text"]:
            raise ValueError(f"Registered plan run.sh differs from its frozen recipe: {launch_script}")
        if bundle[runs[0]["script"]]["text"] != contract["script_text"]:
            raise ValueError(f"Registered plan launch script differs from its frozen recipe: {runs[0]['script']}")

    for run, expected_run in zip(runs, expected_runs):
        for path_field, hash_field in (
            ("config", "config_sha256"),
            ("script", "script_sha256"),
            ("scheduler_script", "scheduler_script_sha256"),
        ):
            path = expected_run.get(path_field)
            expected = expected_run.get(hash_field, run.get(hash_field))
            if path not in (None, "") and expected not in (None, "") and bundle[str(path)]["sha256"] != expected:
                raise ValueError(f"Registered plan frozen file SHA-256 changed: {path}")
    if final_path is not None and bundle[str(final_path)]["sha256"] != contract["final_eval_config_sha256"]:
        raise ValueError(f"Registered plan frozen file SHA-256 changed: {final_path}")

    if expected_recipe_path is not None and recipe.get("_recipe_path", "") != expected_recipe_path:
        raise ValueError(f"Registered plan recipe path differs from its managed step: {plan_dir}")
    selection = None
    if task == "hparam_tune":
        evaluation = recipe.get("evaluation_policy") if isinstance(recipe.get("evaluation_policy"), dict) else {}
        selection = {
            "metric": str(evaluation.get("selection_metric") or ""),
            "mode": str(evaluation.get("selection_mode") or ""),
            "split": str(evaluation.get("selection_split") or ""),
        }
    return {
        "path": str(plan_dir),
        "task": task,
        "run_keys": plan_keys,
        "launch_script": str(launch_script),
        "selection": selection,
    }


def _compile_registered_plan_contract(
    adapter: Any,
    recipe: dict[str, Any],
    plan_dir: Path,
    *,
    run_index_offset: int,
    config_bytes: bytes,
) -> dict[str, Any]:
    try:
        return adapter.compile_plan_contract(
            recipe,
            plan_dir,
            run_index_offset=run_index_offset,
            config_bytes=config_bytes,
        )
    except (AttributeError, KeyError, IndexError, TypeError, yaml.YAMLError) as exc:
        raise ValueError(f"Registered plan frozen config is corrupt: {exc}") from exc


def _validate_plan_contract_runs(
    runs: list[dict[str, Any]],
    expected_runs: list[dict[str, Any]],
    plan_path: Path,
) -> None:
    if len(runs) != len(expected_runs):
        raise ValueError(f"Registered plan differs from canonical expected runs: {plan_path}")
    for run, expected in zip(runs, expected_runs):
        for field, value in expected.items():
            if run.get(field) != value:
                raise ValueError(
                    f"Registered plan differs from canonical expected runs field {field}: "
                    f"{run.get('step_id')} / {run.get('run_id')}"
                )
        if managed_run_parameters(run) != managed_run_parameters(expected):
            raise ValueError(
                "Workspace run parameters differ from plan and canonical expected runs: "
                f"{run.get('step_id')} / {run.get('run_id')}"
            )


def _text_value(value: Any) -> str:
    return "" if value is None else str(value)


def _mapping_overlay(base: dict[str, Any], effective: dict[str, Any]) -> dict[str, Any]:
    overlay = {}
    for field, value in effective.items():
        base_value = base.get(field)
        if isinstance(value, dict) and isinstance(base_value, dict):
            nested = _mapping_overlay(base_value, value)
            if nested:
                overlay[field] = nested
        elif field not in base or value != base_value:
            overlay[field] = value
    return overlay


def _validate_registered_run_parameters(
    recipe: dict[str, Any],
    plan_run: dict[str, Any],
    canonical_run: dict[str, Any],
) -> None:
    plan_parameters = managed_run_parameters(plan_run)
    canonical_parameters = managed_run_parameters(canonical_run)
    declared_parameters = set()
    search = recipe.get("search") if isinstance(recipe.get("search"), dict) else {}
    parameters = search.get("parameters")
    if isinstance(parameters, dict):
        declared_parameters = {str(field) for field in parameters}
    plan_parameter_keys = set(plan_parameters)
    canonical_parameter_keys = set(canonical_parameters)
    missing_parameters = plan_parameter_keys - canonical_parameter_keys
    missing_declared = (declared_parameters - plan_parameter_keys) | (declared_parameters - canonical_parameter_keys)
    nonempty_extra_parameters = {
        field
        for field in canonical_parameter_keys - plan_parameter_keys
        if canonical_parameters[field] not in (None, "")
    }
    key = managed_run_key(plan_run)
    if missing_parameters or missing_declared or nonempty_extra_parameters:
        raise ValueError(f"Workspace run parameters differ from plan: {key[0]} / {key[1]}")
    for field, value in plan_parameters.items():
        if _text_value(canonical_parameters.get(field)) != _text_value(value):
            raise ValueError(f"Workspace run manifest differs from plan field {field}: {key[0]} / {key[1]}")


def read_hparam_plan(
    run_dir: Path,
    *,
    semantic_dir: Path | None = None,
    require_workspace_state: bool = True,
    require_adaptive_commit: bool = True,
) -> dict[str, Any]:
    physical_dir = run_dir
    plan_dir = semantic_dir or physical_dir
    plan_path = physical_dir / "plan.json"
    resolved_recipe_path = physical_dir / "recipe.resolved.yaml"
    plan, resolved_recipe = _read_plan_documents(physical_dir, require_resolved_sha256=True)
    legacy_status = physical_dir / "trial_status.tsv"
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
        plan_dir.resolve().relative_to(workspace.resolve())
    except ValueError as exc:
        raise ValueError(f"Hparam plan must be inside experiment.root: {workspace}") from exc
    step_id = str(recipe["step"]["id"])
    expected_experiment = recipe["experiment"]
    experiment_id = str(expected_experiment["id"])
    for run in runs:
        if str(run["experiment_id"]) != experiment_id or str(run["step_id"]) != step_id:
            raise ValueError("Managed run identity does not match the hparam recipe workspace binding.")
    if require_workspace_state:
        experiment_manifest_path = workspace / "experiment.yaml"
        if not experiment_manifest_path.exists():
            raise ValueError(f"Hparam plan is not bound to an initialized experiment workspace: {workspace}")
        experiment_manifest = read_managed_yaml_mapping(
            experiment_manifest_path.read_text(),
            source=f"Managed experiment manifest {experiment_manifest_path}",
        )
        existing_experiment = experiment_manifest.get("experiment") if isinstance(experiment_manifest, dict) else None
        if not isinstance(existing_experiment, dict) or any(
            existing_experiment.get(field) != expected_experiment.get(field)
            for field in ("id", "title", "objective", "root", "baseline")
        ):
            raise ValueError(f"Hparam plan experiment metadata differs from the managed workspace: {workspace}")
        step_manifest = read_step_manifest(workspace, step_id)
        expected_step_manifest = merge_step_manifest(
            step_manifest,
            {
                "step": recipe["step"],
                "experiment_id": experiment_id,
                "plan_controller": (
                    "adaptive"
                    if isinstance(recipe.get("adaptive"), dict) and recipe["adaptive"].get("enabled") is True
                    else "ordinary"
                ),
                "recipe_path": recipe.get("_recipe_path", ""),
                "plans": [str(run_dir.resolve())],
            },
        )
        if expected_step_manifest != step_manifest:
            raise ValueError(f"Hparam plan is not registered by its managed step: {run_dir}")
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
                "terminal_status_owner",
                *sorted(SCHEDULER_PLAN_IDENTITY_FIELDS),
            ):
                if str(workspace_row.get(field) or "") != str(run.get(field) or ""):
                    raise ValueError(
                        f"Workspace run manifest differs from plan field {field}: {run['step_id']} / {run['run_id']}"
                    )
            if run.get("scheduler_type") == "slurm" and str(workspace_row.get("log_path") or "") != str(
                run.get("log_path") or ""
            ):
                raise ValueError(
                    f"Workspace run manifest differs from plan field log_path: {run['step_id']} / {run['run_id']}"
                )
            _validate_registered_run_parameters(recipe, run, workspace_row)
    search = recipe.get("search") if isinstance(recipe.get("search"), dict) else {}
    execution = recipe.get("execution") if isinstance(recipe.get("execution"), dict) else {}
    adaptive = recipe.get("adaptive") if isinstance(recipe.get("adaptive"), dict) else {}
    workdir = execution.get("workdir")
    if workdir not in (None, "") and not Path(str(workdir)).is_absolute():
        raise ValueError("execution.workdir must be an absolute path when set.")
    run_cwd = Path(str(workdir or plan_contract.frozen_plan_context(recipe)["repo_root"]))
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
    frozen_recipe = _resolved_recipe_view(recipe)
    if frozen_recipe != resolved_recipe:
        raise ValueError(f"Hparam plan recipe differs from recipe.resolved.yaml: {resolved_recipe_path}")
    for run in runs:
        physical_run = dict(run)
        for field in ("config", "script", "scheduler_script"):
            if run.get(field) not in (None, ""):
                physical_run[field] = str(_physical_plan_path(Path(str(run[field])), plan_dir, physical_dir))
        verify_run_snapshot(physical_run)
    typed_plan = recipe.get("task") == "hparam_tune"
    pass_plan = plan.get("status") == "PASS"
    if typed_plan != pass_plan:
        raise ValueError(f"Hparam plan has an incomplete static contract: {plan_path}")
    if typed_plan:
        _validate_local_hparam_plan_contract(plan, recipe, physical_dir, plan_dir, runs)
        _validate_hparam_execution_snapshot(plan, physical_dir, plan_dir)
    if require_adaptive_commit:
        _validate_adaptive_workflow_commit(plan_dir, recipe)
    return plan


def _validate_local_hparam_plan_contract(
    plan: dict[str, Any],
    recipe: dict[str, Any],
    physical_dir: Path,
    plan_dir: Path,
    runs: list[dict[str, Any]],
) -> None:
    plan_path = physical_dir / "plan.json"
    resolved_plan_dir = plan_dir.resolve()
    first_run_id = str(runs[0].get("run_id") or "")
    match = re.fullmatch(r"run-(\d+)", first_run_id)
    if match is None:
        raise ValueError(f"Hparam plan has an invalid first run id: {first_run_id}")
    adapter = get_adapter("hparam_tune")
    assert adapter is not None
    source_config = physical_dir / "config.source.yaml"
    if not source_config.is_file():
        raise FileNotFoundError(f"Missing frozen hparam source config: {source_config}")
    contract = adapter.compile_plan_contract(
        recipe,
        resolved_plan_dir,
        run_index_offset=int(match.group(1)),
        config_bytes=source_config.read_bytes(),
    )
    _validate_plan_contract_runs(runs, contract["runs"], plan_path)
    launch_script = physical_dir / "run_all.sh"
    if not launch_script.is_file() or launch_script.read_text() != contract["launch_script_text"]:
        raise ValueError(f"Hparam plan launch script differs from its frozen recipe: {launch_script}")
    for run, run_files in zip(runs, contract["run_files"]):
        config_path = _physical_plan_path(Path(run["config"]), resolved_plan_dir, physical_dir)
        script_path = _physical_plan_path(Path(run["script"]), resolved_plan_dir, physical_dir)
        if config_path.read_bytes() != run_files["config_bytes"]:
            raise ValueError(f"Hparam plan config differs from its frozen recipe: {run['run_id']}")
        if script_path.read_text() != run_files["script_text"]:
            raise ValueError(f"Hparam plan script differs from its frozen recipe: {run['run_id']}")
        scheduler_text = run_files.get("scheduler_script_text")
        scheduler_path = _physical_plan_path(
            Path(str(run.get("scheduler_script") or "")), resolved_plan_dir, physical_dir
        )
        if scheduler_text is not None and scheduler_path.read_text() != scheduler_text:
            raise ValueError(f"Hparam Slurm script differs from its frozen recipe: {run['run_id']}")
    final_path, final_command = plan_contract.validate_final_eval_contract(plan, recipe, resolved_plan_dir, contract)
    if final_path is not None:
        physical_final_path = _physical_plan_path(final_path, resolved_plan_dir, physical_dir)
        if not physical_final_path.is_file() or file_sha256(physical_final_path) != plan["final_eval_config"]["sha256"]:
            raise ValueError(f"Frozen final evaluation config changed after planning: {final_path}")
    final_script = physical_dir / "final_external_test.sh"
    if (final_command is not None) != final_script.is_file():
        requirement = "missing" if final_command is not None else "unexpected"
        raise ValueError(f"Hparam plan has {requirement} final external-test script: {final_script}")
    if final_command is not None and final_script.read_text() != contract["final_script_text"]:
        raise ValueError(f"Hparam final external-test script differs from its frozen recipe: {final_script}")


def _physical_plan_path(path: Path, semantic_dir: Path, physical_dir: Path) -> Path:
    try:
        relative = path.relative_to(semantic_dir)
    except ValueError:
        return path
    return physical_dir / relative


def _validate_hparam_execution_snapshot(
    plan: dict[str, Any],
    physical_dir: Path,
    plan_dir: Path,
) -> None:
    binding = plan.get("execution_snapshot")
    if binding is None:
        return
    expected_path = plan_dir.resolve() / "execution_snapshot.json"
    if not isinstance(binding, dict) or set(binding) != {"path", "sha256"}:
        raise ValueError(f"Hparam execution snapshot binding is invalid: {physical_dir / 'plan.json'}")
    if binding["path"] != str(expected_path) or not re.fullmatch(r"[0-9a-f]{64}", str(binding["sha256"])):
        raise ValueError(f"Hparam execution snapshot binding is invalid: {physical_dir / 'plan.json'}")
    snapshot_path = physical_dir / "execution_snapshot.json"
    if snapshot_path.is_symlink() or not snapshot_path.is_file() or snapshot_path.stat().st_nlink != 1:
        raise ValueError(f"Hparam execution snapshot is not an independent regular file: {snapshot_path}")
    if file_sha256(snapshot_path) != binding["sha256"]:
        raise ValueError(f"Hparam execution snapshot changed after planning: {snapshot_path}")
    snapshot = read_json(snapshot_path)
    if not isinstance(snapshot, dict):
        raise ValueError(f"Hparam execution snapshot must be a mapping: {snapshot_path}")


def plan_tree_sha256(root: Path) -> str:
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"Managed plan is missing or aliased: {root}")
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        info = os.lstat(path)
        relative = path.relative_to(root).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(stat.S_IMODE(info.st_mode).to_bytes(4, "big"))
        if stat.S_ISDIR(info.st_mode):
            digest.update(b"directory")
            continue
        if not stat.S_ISREG(info.st_mode):
            raise ValueError(f"Managed plan contains an unsafe artifact: {path}")
        digest.update(b"file")
        digest.update(info.st_size.to_bytes(8, "big"))
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def _validate_adaptive_workflow_commit(run_dir: Path, recipe: dict[str, Any]) -> None:
    adaptive = recipe.get("adaptive") if isinstance(recipe.get("adaptive"), dict) else {}
    if adaptive.get("enabled") is not True:
        return
    if run_dir.is_symlink():
        raise ValueError(f"Adaptive plan directory must not be a symlink: {run_dir}")
    run_dir = run_dir.resolve()
    if run_dir.parent.name != "rounds" or run_dir.parent.parent.name != "adaptive":
        return
    workflow_root = run_dir.parent.parent.parent
    workflow_path = workflow_root / "adaptive" / "workflow.json"
    if workflow_path.is_symlink() or not workflow_path.is_file():
        raise FileNotFoundError(f"Adaptive workflow initialization is not committed: {workflow_path}")
    workflow = read_json(workflow_path)
    if not isinstance(workflow, dict) or str(workflow.get("root") or "") != str(workflow_root):
        raise ValueError(f"Adaptive workflow commit marker differs from the plan root: {workflow_path}")


def iter_registered_hparam_plans(
    workspace: Path,
    step_id: str,
    *,
    selection_metric: Any,
    selection_mode: Any,
    selection_split: Any,
) -> Iterator[tuple[Path, dict[str, Any]]]:
    step_manifest = read_step_manifest(workspace, step_id)
    for registered_plan_dir in step_manifest["plans"]:
        registered_root = Path(str(registered_plan_dir))
        registered_plan_path = registered_root / "plan.json"
        resolved_recipe_path = registered_root / "recipe.resolved.yaml"
        if is_registered_blocked_plan(registered_root, workspace=workspace):
            continue
        if not registered_plan_path.exists():
            raise FileNotFoundError(f"Registered plan is missing plan.json: {registered_plan_path}")
        registered_plan = read_json(registered_plan_path)
        registered_recipe = registered_plan.get("recipe") if isinstance(registered_plan.get("recipe"), dict) else {}
        resolved_recipe = read_managed_yaml_mapping(
            resolved_recipe_path.read_text(),
            source=f"Frozen registered recipe {resolved_recipe_path}",
        )
        if registered_recipe.get("task") != resolved_recipe.get("task"):
            raise ValueError(f"Registered plan task differs from recipe.resolved.yaml: {registered_root}")
        if resolved_recipe.get("task") != "hparam_tune":
            continue
        registered_plan = read_hparam_plan(registered_root)
        registered_recipe = registered_plan.get("recipe") if isinstance(registered_plan.get("recipe"), dict) else {}
        registered_step = registered_recipe.get("step") if isinstance(registered_recipe.get("step"), dict) else {}
        if str(registered_step.get("id") or "") != step_id:
            raise ValueError(f"Registered hparam plan belongs to a different step: {registered_root}")
        registered_evaluation = (
            registered_recipe.get("evaluation_policy")
            if isinstance(registered_recipe.get("evaluation_policy"), dict)
            else {}
        )
        if registered_evaluation.get("selection_metric") != selection_metric:
            raise ValueError("Existing ranking selection metric differs from the current recipe.")
        if registered_evaluation.get("selection_mode") != selection_mode:
            raise ValueError("Existing ranking selection mode differs from the current recipe.")
        # The invoking split governs evidence interpretation, so registered aggregation must be homogeneous.
        if registered_evaluation.get("selection_split") != selection_split:
            raise ValueError("Existing ranking selection split differs from the current recipe.")
        yield registered_root, registered_plan


def validate_run_rows(
    rows: list[dict[str, Any]],
    *,
    source: str,
    require_artifact_paths: bool = False,
    allow_empty_runtime_paths: bool = False,
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
                    "config",
                    "config_sha256",
                    "script",
                    "script_sha256",
                    "artifacts",
                )
                if row.get(field) in (None, "")
            )
            if allow_empty_runtime_paths:
                missing.extend(field for field in ("runtime_dir", "checkpoint_dir") if field not in row)
                if bool(row.get("runtime_dir")) != bool(row.get("checkpoint_dir")):
                    raise ValueError(f"Managed run row {index} in {source} has partial runtime artifact paths.")
            else:
                missing.extend(field for field in ("runtime_dir", "checkpoint_dir") if row.get(field) in (None, ""))
        if missing:
            raise ValueError(f"Managed run row {index} in {source} is missing: {', '.join(missing)}")
        if require_artifact_paths:
            relative_paths = [
                field
                for field in ("run_dir", "runtime_dir", "checkpoint_dir", "config", "script", "artifacts")
                if row.get(field) not in (None, "") and not Path(str(row[field])).is_absolute()
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
    runtime_dir = Path(str(run["runtime_dir"]))
    path = runtime_dir / "run_manifest.json"
    if runtime_dir.is_symlink() or path.is_symlink():
        raise ValueError(f"Runtime run manifest is not an independent regular file: {path}")
    if runtime_dir.exists() and not runtime_dir.is_dir():
        raise ValueError(f"Runtime run manifest parent is not a directory: {runtime_dir}")
    if not path.exists():
        return None
    if not path.is_file() or path.stat().st_nlink != 1:
        raise ValueError(f"Runtime run manifest is not an independent regular file: {path}")
    try:
        manifest = read_json(path)
    except (OSError, UnicodeError, ValueError) as exc:
        raise ValueError(f"Runtime run manifest is corrupt: {path}") from exc
    if not isinstance(manifest, dict):
        raise ValueError(f"Runtime run manifest is corrupt: {path}")
    return path


def metric_value(manifest: dict[str, Any], metric: str) -> float | str:
    metrics = manifest.get("metrics") if isinstance(manifest.get("metrics"), dict) else {}
    if metric in metrics:
        return metrics[metric]
    if manifest.get("monitor") == metric and manifest.get("best_model_score") is not None:
        return manifest["best_model_score"]
    return ""


def fixed_checkpoint_path(manifest: dict[str, Any], checkpoint_dir: Path) -> str:
    if checkpoint_dir.is_symlink() or not checkpoint_dir.is_dir():
        return ""
    resolved_dir = checkpoint_dir.resolve()
    raw_epoch = manifest.get("epoch")
    manifest_epoch = epoch_number(raw_epoch)
    if raw_epoch not in (None, "") and manifest_epoch is None:
        return ""
    raw = manifest.get("best_model_path") or manifest.get("checkpoint_path") or ""
    if raw:
        path = Path(str(raw))
        if path.name.startswith("best-epoch="):
            fixed = checkpoint_dir / path.name.removeprefix("best-")
            if manifest_epoch is not None and epoch_number_from_checkpoint_name(fixed.name) != manifest_epoch:
                matched = checkpoint_for_epoch_in_dir(checkpoint_dir, manifest_epoch)
                return str(matched) if matched else ""
            # Lexical containment is insufficient when the checkpoint entry itself is an alias.
            if not fixed.is_symlink() and fixed.is_file() and fixed.resolve().parent == resolved_dir:
                return str(fixed)
            matched = checkpoint_for_epoch_in_dir(checkpoint_dir, epoch_number_from_checkpoint_name(fixed.name))
            if matched:
                return str(matched)
            best = checkpoint_dir / path.name
            if (
                manifest_epoch is not None
                and not best.is_symlink()
                and best.is_file()
                and best.resolve().parent == resolved_dir
            ):
                return str(best)
            return ""
        if path.name.startswith("epoch="):
            fixed = checkpoint_dir / path.name
            if manifest_epoch is not None and epoch_number_from_checkpoint_name(fixed.name) != manifest_epoch:
                matched = checkpoint_for_epoch_in_dir(checkpoint_dir, manifest_epoch)
                return str(matched) if matched else ""
            return (
                str(fixed)
                if not fixed.is_symlink() and fixed.is_file() and fixed.resolve().parent == resolved_dir
                else ""
            )
        matched = checkpoint_for_epoch_in_dir(checkpoint_dir, manifest_epoch)
        if matched:
            return str(matched)
        return ""
    if manifest_epoch is not None:
        matched = checkpoint_for_epoch_in_dir(checkpoint_dir, manifest_epoch)
        return str(matched) if matched else ""
    return ""


def fixed_checkpoint_path_from_names(
    manifest: dict[str, Any], checkpoint_dir: str | Path, checkpoint_names: list[str]
) -> str:
    if checkpoint_dir in (None, ""):
        return ""
    checkpoint_dir = Path(str(checkpoint_dir))
    names = {str(name) for name in checkpoint_names}
    raw_epoch = manifest.get("epoch")
    manifest_epoch = epoch_number(raw_epoch)
    if raw_epoch not in (None, "") and manifest_epoch is None:
        return ""
    raw = manifest.get("best_model_path") or manifest.get("checkpoint_path") or ""
    if raw:
        raw_name = Path(str(raw)).name
        name = raw_name
        if raw_name.startswith("best-epoch="):
            name = name.removeprefix("best-")
        if (
            name.startswith("epoch=")
            and name in names
            and (manifest_epoch is None or epoch_number_from_checkpoint_name(name) == manifest_epoch)
        ):
            return str(checkpoint_dir / name)
        if (
            manifest_epoch is not None
            and raw_name.startswith("best-epoch=")
            and raw_name in names
            and epoch_number_from_checkpoint_name(name) == manifest_epoch
        ):
            return str(checkpoint_dir / raw_name)
        if manifest_epoch is None:
            return ""
        for candidate in sorted(names):
            if candidate.startswith("epoch=") and epoch_number_from_checkpoint_name(candidate) == manifest_epoch:
                return str(checkpoint_dir / candidate)
        return ""
    if manifest_epoch is not None:
        for candidate in sorted(names):
            if candidate.startswith("epoch=") and epoch_number_from_checkpoint_name(candidate) == manifest_epoch:
                return str(checkpoint_dir / candidate)
        return ""
    return ""


def checkpoint_names(run: dict[str, Any]) -> list[str]:
    if not run.get("checkpoint_dir"):
        return []
    ckpt_dir = Path(str(run["checkpoint_dir"]))
    if ckpt_dir.is_symlink() or not ckpt_dir.is_dir():
        return []
    # Match remote evidence collection: only physical checkpoint files belong to the runtime inventory.
    return [path.name for path in sorted(ckpt_dir.glob("*.ckpt")) if not path.is_symlink() and path.is_file()]


def checkpoint_for_epoch_in_dir(ckpt_dir: Path, epoch: int | None) -> Path | None:
    if epoch is None or ckpt_dir.is_symlink() or not ckpt_dir.is_dir():
        return None
    resolved_dir = ckpt_dir.resolve()
    for path in sorted(ckpt_dir.glob("epoch=*.ckpt")):
        if (
            not path.name.startswith("best-")
            and not path.is_symlink()
            and path.is_file()
            and path.resolve().parent == resolved_dir
            and epoch_number_from_checkpoint_name(path.name) == epoch
        ):
            return path
    return None


def epoch_from_checkpoint_name(name: str) -> str:
    if not name.startswith("epoch="):
        return ""
    return name.split("=", 1)[1].split("-", 1)[0].split(".", 1)[0]


def epoch_number(value: Any) -> int | None:
    if isinstance(value, bool) or value in (None, ""):
        return None
    try:
        number = float(str(value))
    except ValueError:
        return None
    if not math.isfinite(number) or not number.is_integer():
        return None
    return int(number)


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
    # NaN is unordered, so every non-finite score must use the same worst-value sentinel.
    return score if math.isfinite(score) else (-math.inf if reverse else math.inf)


def assign_ranks(
    rows: list[dict[str, Any]],
    *,
    key: str,
    reverse: bool,
    top_k: int | None = None,
    rank_metric: str | None = None,
) -> list[dict[str, Any]]:
    """Sort rows by their metric value and write 1-based ranks in place.

    Sorting is stable (ties keep their input order). top_k truncates before
    ranks are assigned, so ranks stay contiguous from 1. The input list is
    not reordered; the returned list is the sorted (and truncated) view."""
    ranked = sorted(rows, key=lambda row: sortable_score(row.get(key), reverse), reverse=reverse)
    if top_k is not None:
        ranked = ranked[:top_k]
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
        if rank_metric is not None:
            row["rank_metric"] = rank_metric
    return ranked
