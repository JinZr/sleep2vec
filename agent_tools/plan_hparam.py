from __future__ import annotations

import copy
import hashlib
from importlib import import_module
from itertools import product
from pathlib import Path
import subprocess
import sys
from tempfile import NamedTemporaryFile
from typing import Any

import yaml

from . import plan_rendering as rendering, transport
from .decision_models import DecisionIssue, DecisionStatus
from .decision_paths import path_context, path_validation, validate_input_path
from .experiment_workspace import (
    append_event,
    experiment_root,
    file_sha256,
    merge_run_manifest,
    next_run_index,
    parameter_summary,
    run_identity,
)
from .manifests import write_json, write_text
from .models import REPO_ROOT, coerce_list, resolve_repo_path
from .repo import repo_summary

FROZEN_FINAL_EVAL_CONFIG_NAME = "config.final_eval.yaml"
_FINAL_EVAL_CONFIG_SNAPSHOT = "_final_eval_config_snapshot"


def final_test_unlocked(evaluation: dict, unlock_final_test: bool = False) -> bool:
    return unlock_final_test or (
        evaluation.get("external_test_locked") is False and evaluation.get("final_test_unlocked") is True
    )


def has_resolved_ckpt_path(recipe: dict) -> bool:
    ckpt_path = resolved_ckpt_path(recipe)
    return ckpt_path not in (None, "", "ASK_USER") and not str(ckpt_path).startswith("<")


def final_script_allowed(
    recipe: dict,
    evaluation: dict,
    unlock_final_test: bool,
) -> bool:
    return unlock_final_test or (final_test_unlocked(evaluation) and has_resolved_ckpt_path(recipe))


def final_test_checkpoint_issues(
    recipe: dict,
    *,
    unlock_final_test: bool,
) -> list[DecisionIssue]:
    evaluation = recipe.get("evaluation_policy") if isinstance(recipe.get("evaluation_policy"), dict) else {}
    if not unlock_final_test and not final_script_allowed(recipe, evaluation, unlock_final_test):
        return []
    issues: list[DecisionIssue] = []
    ckpt_path = resolved_ckpt_path(recipe)
    if ckpt_path in (None, "", "ASK_USER") or str(ckpt_path).startswith("<"):
        return [
            DecisionIssue(
                DecisionStatus.NEEDS_USER_INPUT,
                "ckpt_path",
                "Final external-test evaluation requires an explicit checkpoint path.",
                "Which checkpoint path should be used for final external-test evaluation?",
                {"ckpt_path": ckpt_path},
            )
        ]
    ckpt_issue = validate_input_path(recipe, "ckpt_path", ckpt_path, configured=False)
    if ckpt_issue is not None:
        issues.append(ckpt_issue)
        if ckpt_issue.status == DecisionStatus.FAIL:
            return issues
    final_config = resolved_final_eval_config_path(recipe, None)
    if has_yaml_search_overrides(recipe) and not has_explicit_final_eval_config(recipe):
        issues.append(
            DecisionIssue(
                DecisionStatus.NEEDS_USER_INPUT,
                "final_eval_config_path",
                "Final external-test evaluation for YAML-overridden hparam runs requires an explicit config path.",
                "Which selected run config should be used for final external-test evaluation?",
                {"final_eval_config_path": final_config},
            )
        )
        return issues
    if has_explicit_final_eval_config(recipe):
        config_issue = validate_input_path(recipe, "final_eval_config_path", final_config, configured=False)
        if config_issue is not None:
            issues.append(config_issue)
            if config_issue.status == DecisionStatus.FAIL:
                return issues
        try:
            config_bytes = read_final_eval_config_bytes(recipe, final_config)
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            issues.append(
                DecisionIssue(
                    DecisionStatus.FAIL,
                    "final_eval_config_path",
                    f"Final evaluation config cannot be frozen as exact bytes: {exc}",
                    None,
                    {
                        "final_eval_config_path": str(final_config),
                        "preflight_before_workspace": True,
                    },
                )
            )
            return issues
        recipe[_FINAL_EVAL_CONFIG_SNAPSHOT] = {
            "source_path": str(final_config),
            "bytes": config_bytes,
            "sha256": hashlib.sha256(config_bytes).hexdigest(),
        }
        try:
            validate_final_eval_config_bytes(recipe, config_bytes)
        except Exception as exc:
            issues.append(
                DecisionIssue(
                    DecisionStatus.FAIL,
                    "final_eval_config_path",
                    f"Final evaluation config is invalid for variant={recipe.get('variant')}: {exc}",
                    None,
                    {
                        "final_eval_config_path": str(final_config),
                        "preflight_before_workspace": True,
                    },
                )
            )
            return issues
        drift_issue = final_eval_config_drift_issue(recipe)
        if drift_issue is not None:
            issues.append(drift_issue)
    return issues


def final_eval_config_snapshot(recipe: dict) -> dict[str, Any] | None:
    snapshot = recipe.get(_FINAL_EVAL_CONFIG_SNAPSHOT)
    return snapshot if isinstance(snapshot, dict) else None


def validate_final_eval_config_bytes(recipe: dict, config_bytes: bytes) -> None:
    config_module = import_module(rendering.variant_module(recipe, "config"))
    with NamedTemporaryFile(suffix=".yaml") as snapshot:
        snapshot.write(config_bytes)
        snapshot.flush()
        bundle = config_module.load_finetune_config(Path(snapshot.name))
        config_module.validate_model_config(bundle.model)


def read_final_eval_config_bytes(recipe: dict, config_path: Any) -> bytes:
    context = path_context(recipe, config_path)
    validation = path_validation(recipe, context)
    if context == "remote":
        if validation == "remote":
            validation = "ssh"
        if validation != "ssh":
            raise ValueError(
                "remote final_eval_config_path requires execution.path_validation=ssh; "
                f"{validation} cannot capture exact bytes"
            )
        execution = recipe.get("execution") if isinstance(recipe.get("execution"), dict) else {}
        host = execution.get("host")
        if not host:
            raise ValueError("execution.host is required to freeze a remote final_eval_config_path")
        result = transport.run_ssh(str(host), f"cat -- {transport.sh(config_path)}", text=False)
        if result.returncode != 0:
            stderr = result.stderr.decode(errors="replace") if isinstance(result.stderr, bytes) else str(result.stderr)
            raise ValueError(f"remote config read failed on {host}: {stderr.strip() or 'unknown SSH error'}")
        if not isinstance(result.stdout, bytes):
            raise ValueError("remote config read did not return exact bytes")
        return result.stdout
    if context != "local":
        raise ValueError(f"unsupported final_eval_config_path context: {context}")
    resolved = resolve_repo_path(config_path)
    if resolved is None or not resolved.is_file():
        raise ValueError(f"config is not a readable local file: {config_path}")
    return resolved.read_bytes()


def final_eval_config_drift_issue(recipe: dict) -> DecisionIssue | None:
    snapshot = final_eval_config_snapshot(recipe)
    if snapshot is None:
        return None
    try:
        current_bytes = read_final_eval_config_bytes(recipe, snapshot.get("source_path"))
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        return DecisionIssue(
            DecisionStatus.FAIL,
            "final_eval_config_path",
            f"Final evaluation config could not be re-read before planning: {exc}",
            None,
            {"preflight_before_workspace": True},
        )
    if current_bytes == snapshot.get("bytes"):
        return None
    return DecisionIssue(
        DecisionStatus.FAIL,
        "final_eval_config_path",
        "Final evaluation config changed while plan preflight was validating it.",
        None,
        {
            "final_eval_config_path": snapshot.get("source_path"),
            "preflight_before_workspace": True,
        },
    )


def hparam_yaml_override_issues(recipe: dict, *, config_bytes: bytes) -> list[DecisionIssue]:
    inputs = recipe.get("inputs") if isinstance(recipe.get("inputs"), dict) else {}
    evaluation = recipe.get("evaluation_policy") if isinstance(recipe.get("evaluation_policy"), dict) else {}
    selection_metric = evaluation.get("selection_metric")
    if "selection_metric" in evaluation and selection_metric in (None, ""):
        return [
            DecisionIssue(
                DecisionStatus.FAIL,
                "selection_metric",
                "selection_metric must be a non-empty value for hparam planning.",
                None,
                {"selection_metric": selection_metric, "preflight_before_workspace": True},
            )
        ]
    config_path = inputs.get("config")
    if not config_path:
        return []
    try:
        base_config = yaml.safe_load(config_bytes)
        if not isinstance(base_config, dict):
            raise ValueError(f"YAML must be a mapping: {config_path}")
        base_data = base_config.get("data") if isinstance(base_config.get("data"), dict) else {}
        data_backend = inputs.get("data_backend")
        if data_backend in (None, ""):
            data_backend = base_data.get("backend") or "npz"
        selection_mode = evaluation.get("selection_mode")
        for field, decision_value in {
            "data_backend": data_backend,
            "selection_metric": selection_metric,
            "selection_mode": selection_mode,
        }.items():
            if decision_value == "ASK_USER":
                return [
                    DecisionIssue(
                        DecisionStatus.FAIL,
                        field,
                        f"{field} must be resolved before hparam YAML overrides.",
                        None,
                        {"decision": decision_value, "preflight_before_workspace": True},
                    )
                ]
        for combo in hparam_combos(recipe):
            run_config = copy.deepcopy(base_config)
            apply_search_overrides(run_config, combo)
            data = run_config.get("data") if isinstance(run_config.get("data"), dict) else {}
            finetune = run_config.get("finetune") if isinstance(run_config.get("finetune"), dict) else {}
            task = finetune.get("task") if isinstance(finetune.get("task"), dict) else {}
            for field, (decision_value, config_value, config_field) in {
                "data_backend": (
                    data_backend,
                    data.get("backend") or "npz",
                    "data.backend",
                ),
                "selection_metric": (
                    selection_metric,
                    task.get("monitor"),
                    "finetune.task.monitor",
                ),
                "selection_mode": (
                    selection_mode,
                    task.get("monitor_mod"),
                    "finetune.task.monitor_mod",
                ),
            }.items():
                if decision_value not in (None, "") and decision_value != config_value:
                    return [
                        DecisionIssue(
                            DecisionStatus.FAIL,
                            field,
                            f"{field} decision differs from config {config_field} after hparam YAML overrides.",
                            None,
                            {
                                "decision": decision_value,
                                "config": config_value,
                                "parameters": combo,
                                "preflight_before_workspace": True,
                            },
                        )
                    ]
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        return [
            DecisionIssue(
                DecisionStatus.FAIL,
                "hparam_search_space",
                str(exc),
                None,
                {},
            )
        ]
    return []


def resolved_ckpt_path(recipe: dict) -> Any:
    inputs = recipe.get("inputs") if isinstance(recipe.get("inputs"), dict) else {}
    return inputs.get("ckpt_path")


def resolved_final_eval_config_path(recipe: dict, fallback: Any) -> Any:
    inputs = recipe.get("inputs") if isinstance(recipe.get("inputs"), dict) else {}
    return inputs.get("final_eval_config_path", fallback)


def has_explicit_final_eval_config(recipe: dict) -> bool:
    value = resolved_final_eval_config_path(recipe, None)
    return value not in (None, "", "ASK_USER") and not str(value).startswith("<")


def has_yaml_search_overrides(recipe: dict) -> bool:
    search = recipe.get("search") if isinstance(recipe.get("search"), dict) else {}
    parameters = search.get("parameters") if isinstance(search.get("parameters"), dict) else {}
    if any(isinstance(key, str) and key.startswith("yaml:/") for key in parameters):
        return True
    configurations = search.get("configurations") if isinstance(search.get("configurations"), list) else []
    try:
        max_runs = int(search.get("max_runs"))
    except (TypeError, ValueError):
        max_runs = None  # malformed budgets are the contract layer's report, not ours
    if max_runs is not None:
        # Points beyond max_runs never execute (same prefix truncation as
        # hparam_combos), so their keys must not force config requirements.
        configurations = configurations[:max_runs]
    return any(
        isinstance(point, dict) and any(isinstance(key, str) and key.startswith("yaml:/") for key in point)
        for point in configurations
    )


def hparam_combos(recipe: dict) -> list[dict[str, Any]]:
    search = recipe.get("search") or {}
    configurations = search.get("configurations") or []
    if configurations:
        combos = [dict(point) for point in configurations]
    else:
        params = search.get("parameters") or {}
        keys = list(params)
        combos = [dict(zip(keys, values)) for values in product(*(params[key] for key in keys))]
    max_runs = int(search.get("max_runs")) if search.get("max_runs") not in (None, "") else len(combos)
    return combos[:max_runs]


def apply_search_overrides(config: dict[str, Any], combo: dict[str, Any]) -> dict[str, Any]:
    runtime: dict[str, Any] = {}
    for key, value in combo.items():
        if key.startswith("runtime."):
            runtime[key.split(".", 1)[1]] = value
        elif key.startswith("yaml:/"):
            set_json_pointer(config, key.removeprefix("yaml:"), value)
    return runtime


def set_json_pointer(config: Any, pointer: str, value: Any) -> None:
    parts = json_pointer_parts(pointer)
    if not parts:
        raise ValueError("YAML override pointer must not target the document root.")
    parent = config
    for part in parts[:-1]:
        parent = json_pointer_child(parent, part)
    last = parts[-1]
    if isinstance(parent, dict):
        if last not in parent:
            raise KeyError(f"YAML override path does not exist: {pointer}")
        parent[last] = value
        return
    if isinstance(parent, list):
        index = int(last)
        if index < 0 or index >= len(parent):
            raise IndexError(f"YAML override list index is out of range: {pointer}")
        parent[index] = value
        return
    raise TypeError(f"YAML override parent is not indexable: {pointer}")


def json_pointer_child(parent: Any, part: str) -> Any:
    if isinstance(parent, dict):
        if part not in parent:
            raise KeyError(f"YAML override path component does not exist: {part}")
        return parent[part]
    if isinstance(parent, list):
        index = int(part)
        if index < 0 or index >= len(parent):
            raise IndexError(f"YAML override list index is out of range: {part}")
        return parent[index]
    raise TypeError(f"YAML override parent is not indexable: {part}")


def json_pointer_parts(pointer: str) -> list[str]:
    if not pointer.startswith("/"):
        raise ValueError(f"YAML override must be a JSON Pointer: {pointer}")
    return [part.replace("~1", "/").replace("~0", "~") for part in pointer.split("/")[1:]]


def freeze_hparam_execution(recipe: dict) -> dict:
    recipe = copy.deepcopy(recipe)
    execution = dict(recipe.get("execution")) if isinstance(recipe.get("execution"), dict) else {}
    manager_runtime = (
        str(execution.get("target", "local") or "local") == "local"
        and execution.get("workdir") in (None, "", str(REPO_ROOT))
        and execution.get("conda_env") in (None, "")
    )
    if execution.get("python") in (None, "", "ASK_USER"):
        if not manager_runtime:
            raise ValueError("execution.python must be explicit when the target runtime is not local REPO_ROOT.")
        execution["python"] = sys.executable
    if execution.get("runtime_commit") in (None, "", "ASK_USER"):
        if not manager_runtime:
            raise ValueError(
                "execution.runtime_commit must be explicit when the target runtime is not local REPO_ROOT."
            )
        repository = repo_summary().get("git") or {}
        if not repository.get("available") or not repository.get("commit"):
            raise ValueError("Cannot freeze the target runtime commit because the manager repository is unavailable.")
        execution["runtime_commit"] = repository["commit"]
    execution["runtime_commit"] = str(execution["runtime_commit"]).lower()
    recipe["execution"] = execution
    return recipe


def write_hparam_plan(
    recipe: dict,
    out: Path,
    *,
    unlock_final_test: bool,
    source_config_bytes: bytes,
    source_config_sha256: str,
) -> None:
    out = out.expanduser()
    if not out.is_absolute():
        out = out.resolve()
    recipe = freeze_hparam_execution(recipe)
    execution = recipe["execution"]
    run_cwd = Path(str(execution.get("workdir") or REPO_ROOT))
    if not run_cwd.is_absolute():
        raise ValueError("execution.workdir must be an absolute path when set.")
    inputs = recipe.get("inputs") if isinstance(recipe.get("inputs"), dict) else {}
    run_inputs = {key: value for key, value in inputs.items() if key != "ckpt_path"}
    runtime_defaults = recipe.get("runtime") if isinstance(recipe.get("runtime"), dict) else {}
    artifacts = recipe.get("artifacts") if isinstance(recipe.get("artifacts"), dict) else {}
    evaluation = recipe.get("evaluation_policy") or {}
    final_allowed = final_script_allowed(recipe, evaluation, unlock_final_test)
    frozen_final_eval_config = out / FROZEN_FINAL_EVAL_CONFIG_NAME
    final_config_snapshot = final_eval_config_snapshot(recipe)
    if final_allowed and has_explicit_final_eval_config(recipe):
        if final_config_snapshot is None:
            raise ValueError("Explicit final evaluation config requires bound config bytes.")
        final_config_bytes = final_config_snapshot.get("bytes")
        final_config_sha256 = final_config_snapshot.get("sha256")
        if not isinstance(final_config_bytes, bytes) or not isinstance(final_config_sha256, str):
            raise ValueError("Bound final evaluation config is incomplete.")
        if hashlib.sha256(final_config_bytes).hexdigest() != final_config_sha256:
            raise ValueError("Final evaluation config bytes do not match their bound SHA-256.")
    source_config_path = inputs.get("config")
    if not source_config_path:
        raise FileNotFoundError("Config path is required.")
    if hashlib.sha256(source_config_bytes).hexdigest() != source_config_sha256:
        raise ValueError("Hparam source config does not match the bound SHA-256.")
    base_config = yaml.safe_load(source_config_bytes)
    if not isinstance(base_config, dict):
        raise ValueError(f"YAML must be a mapping: {source_config_path}")
    frozen_source_config = out / "config.source.yaml"
    out.mkdir(parents=True, exist_ok=True)
    frozen_source_config.write_bytes(source_config_bytes)
    if final_allowed and has_explicit_final_eval_config(recipe):
        frozen_final_eval_config.write_bytes(final_config_bytes)
    elif frozen_final_eval_config.exists():
        frozen_final_eval_config.unlink()
    combos = hparam_combos(recipe)
    runs = []
    test_after_fit = evaluation.get("test_after_fit")
    run_index_offset = next_run_index(recipe)
    for idx, combo in enumerate(combos):
        identity = run_identity(recipe, run_index_offset + idx, combo)
        run_id = identity["run_id"]
        run_name = identity["run_name"]
        run_dir = out / "runs" / f"{run_id}--{run_name}"
        run_dir.mkdir(parents=True, exist_ok=True)
        cfg_copy = run_dir / "config.yaml"
        run_config = copy.deepcopy(base_config)
        runtime_overrides = apply_search_overrides(run_config, combo)
        with cfg_copy.open("w") as file_obj:
            yaml.safe_dump(run_config, file_obj)
        version = identity["version"]
        runtime = {**runtime_defaults, **runtime_overrides}
        if execution.get("gpu_pool") or "gpus_per_run" in execution:
            gpus_per_run = (
                int(execution["gpus_per_run"])
                if "gpus_per_run" in execution
                else len(coerce_list(runtime_defaults.get("devices"))) or 1
            )
            runtime["devices"] = list(range(gpus_per_run))
        command_parts = [
            execution["python"],
            "-m",
            rendering.variant_module(recipe, "finetune"),
            "--config",
            cfg_copy,
            "--label-name",
            inputs.get("label_name"),
            "--version-name",
            version,
            "--results-csv-path",
            plan_output_path(out, artifacts.get("results_csv_path"), "results/agent_hparam_results.csv"),
            *rendering.runtime_cli_args(runtime, variant=str(recipe.get("variant"))),
            *rendering.finetune_input_cli_args(
                run_inputs,
                variant=str(recipe.get("variant")),
            ),
        ]
        if recipe.get("variant") != "sex_age_baseline":
            rendering.append_option(command_parts, "--wandb-project", execution.get("wandb_project"))
            rendering.append_option(command_parts, "--wandb-group", execution.get("wandb_group"))
        if test_after_fit is not True:
            command_parts.append("--no-test-after-fit")
        command = rendering.render_command(command_parts)
        script_path = run_dir / "launch.sh"
        write_text(
            script_path,
            "\n".join(
                rendering.hparam_script_lines(
                    [command],
                    test_after_fit=test_after_fit is True,
                    run_cwd=run_cwd,
                )
            )
            + "\n",
            executable=True,
        )
        run = {
            "experiment_id": (recipe.get("experiment") or {}).get("id"),
            "step_id": (recipe.get("step") or {}).get("id"),
            "run_id": run_id,
            "run_name": run_name,
            "parameter_summary": parameter_summary(combo),
            "version": version,
            "run_dir": str(run_dir),
            "config": str(cfg_copy),
            "script": str(script_path),
            "command": command,
            "config_sha256": file_sha256(cfg_copy),
            "script_sha256": file_sha256(script_path),
            **combo,
        }
        runtime_dir = run_cwd / "log-finetune" / version
        checkpoint_dir = runtime_dir / "checkpoints"
        artifacts_path = run_dir / "artifacts.json"
        run["artifacts"] = str(artifacts_path)
        run["runtime_dir"] = str(runtime_dir)
        run["checkpoint_dir"] = str(checkpoint_dir)
        runs.append(run)
        write_json(
            run_dir / "run.json",
            {
                "status": "planned",
                **run,
            },
        )
        write_json(
            artifacts_path,
            {
                "runtime_dir": str(runtime_dir),
                "checkpoint_dir": str(checkpoint_dir),
                "external_artifacts": True,
            },
        )
    write_text(
        out / "run_all.sh",
        "\n".join(
            rendering.hparam_script_lines(
                [
                    rendering.render_command(
                        [sys.executable, "-m", "agent_tools", "hparam-run-queue", "--plan-dir", out, "--execute"]
                    )
                ],
                test_after_fit=test_after_fit is True,
                run_cwd=REPO_ROOT,
            )
        )
        + "\n",
        executable=True,
    )
    write_text(
        out / "validation.sh",
        "\n".join(
            rendering.script_lines([rendering.render_command(["python", "-m", "agent_tools", "skills", "--validate"])])
        )
        + "\n",
        executable=True,
    )
    plan_recipe = {key: value for key, value in recipe.items() if key != _FINAL_EVAL_CONFIG_SNAPSHOT}
    plan_payload = {"status": "PASS", "runs": runs, "recipe": plan_recipe}
    if final_allowed and has_explicit_final_eval_config(recipe):
        plan_payload["final_eval_config"] = {
            "path": str(frozen_final_eval_config),
            "sha256": file_sha256(frozen_final_eval_config),
            "source_path": final_config_snapshot["source_path"],
        }
    write_json(out / "plan.json", plan_payload)
    root = experiment_root(recipe)
    if root is None:
        raise ValueError("experiment.root is required.")
    manifest_rows = []
    parameter_keys = {key for combo in combos for key in combo}
    for run in runs:
        row = {
            "experiment_id": run["experiment_id"],
            "step_id": run["step_id"],
            "run_id": run["run_id"],
            "run_name": run["run_name"],
            "parameter_summary": run["parameter_summary"],
            "version": run["version"],
            "status": "planned",
            "config": run["config"],
            "config_sha256": run["config_sha256"],
            "script": run["script"],
            "script_sha256": run["script_sha256"],
            "run_dir": run["run_dir"],
            "artifacts": run["artifacts"],
            "runtime_dir": run["runtime_dir"],
            "checkpoint_dir": run["checkpoint_dir"],
        }
        row.update({key: run.get(key) for key in parameter_keys})
        manifest_rows.append(row)
    merge_run_manifest(root, manifest_rows)
    append_event(
        root,
        "plan_created",
        {"step_id": (recipe.get("step") or {}).get("id"), "plan_dir": str(out), "run_count": len(runs)},
    )
    resolved_recipe = {
        key: value for key, value in recipe.items() if key not in {"_recipe_path", _FINAL_EVAL_CONFIG_SNAPSHOT}
    }
    (out / "recipe.resolved.yaml").write_text(yaml.safe_dump(resolved_recipe, sort_keys=False))
    final_script_path = out / "final_external_test.sh"
    final_unlocked = final_test_unlocked(evaluation, unlock_final_test)
    test_after_fit_message = (
        "Run commands evaluate the configured test split because test_after_fit is explicitly unlocked."
        if test_after_fit is True
        else "Run commands do not evaluate the external test split."
    )
    plan_lines = [
        "# Hyper-Parameter Plan",
        "",
        "Status: PASS",
        "",
        test_after_fit_message,
    ]
    if final_allowed:
        ckpt_path = resolved_ckpt_path(recipe)
        final_config_path = frozen_final_eval_config if has_explicit_final_eval_config(recipe) else frozen_source_config
        final_command = rendering.render_command(
            [
                execution["python"],
                "-m",
                rendering.variant_module(recipe, "infer"),
                "--config",
                final_config_path,
                "--ckpt-path",
                ckpt_path,
                "--label-name",
                inputs.get("label_name"),
                "--eval-split",
                "test",
                *rendering.infer_runtime_cli_args(runtime_defaults),
                *rendering.infer_input_cli_args(
                    inputs,
                    variant=str(recipe.get("variant")),
                ),
            ]
        )
        write_text(
            final_script_path,
            "\n".join(
                rendering.hparam_script_lines(
                    [final_command],
                    final_external_test=True,
                    run_cwd=run_cwd,
                )
            )
            + "\n",
            executable=True,
        )
        plan_lines.append("Final external-test script generated because final test was explicitly unlocked.")
    else:
        if final_script_path.exists():
            final_script_path.unlink()
        if final_unlocked:
            plan_lines.append("Final external-test script not generated; explicit checkpoint path is required.")
        else:
            plan_lines.append("Final external-test script not generated; explicit unlock is required.")
    write_text(out / "plan.md", "\n".join(plan_lines) + "\n")


def plan_output_path(out: Path, raw: Any, default: str) -> Path:
    path = Path(str(raw or default)).expanduser()
    return path if path.is_absolute() else out / path
