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

from . import (
    experiment_io as exp_io,
    managed_scheduler,
    plan_context,
    plan_contract,
    plan_rendering as rendering,
    slurm,
    transport,
)
from .decision_models import DecisionIssue, DecisionStatus
from .decision_paths import (
    inference_checkpoint_averaging_issue,
    multilabel_sidecar_issue,
    path_context,
    path_issues,
    path_validation,
    survival_sidecar_issue,
    validate_input_path,
)
from .experiment_workspace import (
    SCHEDULER_PLAN_IDENTITY_FIELDS,
    append_event,
    ensure_experiment_workspace,
    experiment_root,
    file_sha256,
    managed_run_parameters,
    merge_run_manifest,
    next_run_index,
    parameter_summary,
    plan_registration_rows_state,
    run_identity,
)
from .manifests import read_json, write_json, write_text
from .models import REPO_ROOT, coerce_list, resolve_repo_path
from .repo import repo_summary

FROZEN_FINAL_EVAL_CONFIG_NAME = plan_contract.FROZEN_FINAL_EVAL_CONFIG_NAME
_FINAL_EVAL_CONFIG_SNAPSHOT = "_final_eval_config_snapshot"


class HparamRegistrationPreflightError(ValueError):
    pass


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
    config_summary: dict | None,
    *,
    unlock_final_test: bool,
) -> list[DecisionIssue]:
    evaluation = recipe.get("evaluation_policy") if isinstance(recipe.get("evaluation_policy"), dict) else {}
    if not unlock_final_test and not final_script_allowed(recipe, evaluation, unlock_final_test):
        return []
    issues: list[DecisionIssue] = []
    runtime = recipe.get("runtime") if isinstance(recipe.get("runtime"), dict) else {}
    avg_ckpts = runtime.get("avg_ckpts", 1)
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
    averaging_issue = inference_checkpoint_averaging_issue(recipe, ckpt_path)
    if averaging_issue is not None:
        issues.append(averaging_issue)
        if averaging_issue.status == DecisionStatus.FAIL:
            return issues
    if not (avg_ckpts > 1 and ckpt_path in ("best", "last")):
        ckpt_issue = validate_input_path(
            recipe,
            "ckpt_path",
            ckpt_path,
            configured=False,
            require_file=True,
        )
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
        config_issue = validate_input_path(
            recipe,
            "final_eval_config_path",
            final_config,
            configured=False,
            relative_to_workdir=False,
        )
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
        config_summary = plan_context.load_config_summary_for_recipe(recipe, config_bytes=config_bytes)
        for message in (config_summary or {}).get("blocking_issues", []):
            issues.append(
                DecisionIssue(
                    DecisionStatus.FAIL,
                    "final_eval_config_path",
                    f"Final evaluation config is not runnable: {message}",
                    None,
                    {"final_eval_config_path": str(final_config), "preflight_before_workspace": True},
                )
            )
        drift_issue = final_eval_config_drift_issue(recipe)
        if drift_issue is not None:
            issues.append(drift_issue)
    issues.extend(
        path_issues(
            "infer",
            recipe,
            config_summary,
            preset_path_recipe_field="inference_preset_path",
            validates_dataset_paths=True,
            uses_finetune_config=True,
        )
    )
    survival_issue = survival_sidecar_issue(
        "infer",
        recipe,
        config_summary,
        preset_path_recipe_field="inference_preset_path",
        uses_finetune_config=True,
    )
    if survival_issue is not None:
        issues.append(survival_issue)
    multilabel_issue = multilabel_sidecar_issue(
        "infer",
        recipe,
        config_summary,
        preset_path_recipe_field="inference_preset_path",
        uses_finetune_config=True,
    )
    if multilabel_issue is not None:
        issues.append(multilabel_issue)
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
    selection_split = evaluation.get("selection_split")
    search = recipe.get("search") if isinstance(recipe.get("search"), dict) else {}
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
            if search.get("profile") == "finetune_balanced":
                validate_final_eval_config_bytes(recipe, yaml.safe_dump(run_config, sort_keys=False).encode())
            data = run_config.get("data") if isinstance(run_config.get("data"), dict) else {}
            finetune = run_config.get("finetune") if isinstance(run_config.get("finetune"), dict) else {}
            task = finetune.get("task") if isinstance(finetune.get("task"), dict) else {}
            config_contract = {
                "data_backend": (
                    data_backend,
                    data.get("backend") or "npz",
                    "data.backend",
                ),
            }
            if selection_split != "test":
                config_contract.update(
                    {
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
                    }
                )
            for field, (decision_value, config_value, config_field) in config_contract.items():
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
                {"preflight_before_workspace": True},
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


def compile_hparam_run_contracts(
    recipe: dict[str, Any],
    out: Path,
    run_index_offset: int,
    *,
    source_config_bytes: bytes | None = None,
) -> list[dict[str, Any]]:
    plan_context = plan_contract.frozen_plan_context(recipe)
    execution = recipe.get("execution") if isinstance(recipe.get("execution"), dict) else {}
    if execution.get("python") in (None, "", "ASK_USER") or execution.get("runtime_commit") in (
        None,
        "",
        "ASK_USER",
    ):
        raise ValueError("Frozen hparam plan lacks execution.python or execution.runtime_commit; create a new plan.")
    scheduler = execution.get("scheduler") if isinstance(execution.get("scheduler"), dict) else {}
    scheduler_type = str(scheduler.get("type") or "direct")
    if scheduler_type not in {"direct", "slurm"}:
        raise ValueError("execution.scheduler.type must be direct or slurm.")
    slurm_resources = (
        slurm.normalize_resources(scheduler, execution.get("gpus_per_run", 1)) if scheduler_type == "slurm" else None
    )
    run_cwd = Path(str(execution.get("workdir") or plan_context["repo_root"]))
    if not run_cwd.is_absolute():
        raise ValueError("execution.workdir must be an absolute path when set.")
    inputs = recipe.get("inputs") if isinstance(recipe.get("inputs"), dict) else {}
    base_config = None
    if source_config_bytes is not None:
        source_sha256 = hashlib.sha256(source_config_bytes).hexdigest()
        source_path = plan_contract.resolve_frozen_repo_path(recipe, inputs.get("config"))
        source_snapshot = plan_contract.frozen_input_snapshot(recipe, "inputs.config")
        if (
            source_path is None
            or source_snapshot["path"] != str(source_path)
            or source_snapshot["sha256"] != source_sha256
        ):
            raise ValueError("Frozen hparam source config differs from its recipe digest.")
        base_config = yaml.safe_load(source_config_bytes)
        if not isinstance(base_config, dict):
            raise ValueError("Frozen hparam source config must be a mapping.")
    run_inputs = {key: value for key, value in inputs.items() if key != "ckpt_path"}
    runtime_defaults = recipe.get("runtime") if isinstance(recipe.get("runtime"), dict) else {}
    artifacts = recipe.get("artifacts") if isinstance(recipe.get("artifacts"), dict) else {}
    evaluation = recipe.get("evaluation_policy") if isinstance(recipe.get("evaluation_policy"), dict) else {}
    test_after_fit = evaluation["test_after_fit"]
    selection_split = str(evaluation.get("selection_split") or "")
    contracts = []
    for layout in hparam_run_layouts(recipe, out, run_index_offset):
        identity = layout["identity"]
        combo = layout["parameters"]
        run_dir = layout["run_dir"]
        runtime_overrides = {key.split(".", 1)[1]: value for key, value in combo.items() if key.startswith("runtime.")}
        runtime = {**runtime_defaults, **runtime_overrides}
        if scheduler_type == "slurm":
            runtime["devices"] = list(range(slurm_resources["gpus_per_run"]))
        elif execution.get("gpu_pool") or "gpus_per_run" in execution:
            gpus_per_run = (
                int(execution["gpus_per_run"])
                if "gpus_per_run" in execution
                else len(coerce_list(runtime_defaults.get("devices"))) or 1
            )
            runtime["devices"] = list(range(gpus_per_run))
        run_module = rendering.variant_module(recipe, "finetune")
        command_parts = [
            execution["python"],
            "-m",
            run_module,
            "--config",
            run_dir / "config.yaml",
            "--label-name",
            inputs.get("label_name"),
            "--version-name",
            identity["version"],
            "--results-csv-path",
            plan_output_path(out, artifacts.get("results_csv_path"), "results/agent_hparam_results.csv"),
            *rendering.runtime_cli_args(runtime, variant=str(recipe.get("variant"))),
            *rendering.finetune_input_cli_args(run_inputs, variant=str(recipe.get("variant"))),
        ]
        if recipe.get("variant") != "sex_age_baseline":
            rendering.append_option(command_parts, "--wandb-project", execution.get("wandb_project"))
            rendering.append_option(command_parts, "--wandb-group", execution.get("wandb_group"))
        command_parts.append("--test-after-fit" if test_after_fit else "--no-test-after-fit")
        if selection_split == "test":
            command_parts.append("--test-all-checkpoints-after-fit")
        runtime_dir = run_cwd / "log-finetune" / identity["version"]
        row = {
            "experiment_id": (recipe.get("experiment") or {}).get("id"),
            "step_id": (recipe.get("step") or {}).get("id"),
            **identity,
            "parameter_summary": parameter_summary(combo),
            "run_dir": str(run_dir),
            "config": str(run_dir / "config.yaml"),
            "script": str(run_dir / "launch.sh"),
            "command": rendering.render_command(command_parts),
            "terminal_status_owner": "monitor",
            "scheduler_type": scheduler_type,
            "artifacts": str(run_dir / "artifacts.json"),
            "runtime_dir": str(runtime_dir),
            "checkpoint_dir": str(runtime_dir / "checkpoints"),
            **combo,
        }
        if scheduler_type == "slurm":
            row.update(
                {
                    "scheduler_direct_controller": str(slurm_resources["direct_controller"]).lower(),
                    "scheduler_script": str(run_dir / "job.sbatch"),
                    "scheduler_result_path": str(run_dir / "slurm_terminal.json"),
                    "allocation_identity_path": str(run_dir / "allocation_identity.json"),
                    "log_path": str(run_dir / "slurm.log"),
                    "terminal_status_owner": "scheduler_sidecar",
                }
            )
        contract = {"row": row}
        if base_config is not None:
            run_config = copy.deepcopy(base_config)
            apply_search_overrides(run_config, combo)
            config_bytes = yaml.safe_dump(run_config).encode()
            script_commands = [row["command"]]
            if selection_split == "test":
                script_commands.insert(
                    0,
                    rendering.render_command(["export", f"_SLEEP2VEC_FROZEN_CHECKPOINT_DIR={row['checkpoint_dir']}"]),
                )
            script_text = (
                "\n".join(
                    rendering.hparam_script_lines(
                        script_commands,
                        test_after_fit=test_after_fit,
                        selection_split=selection_split,
                        record_exit_code=True,
                        run_cwd=run_cwd,
                    )
                )
                + "\n"
            )
            row.update(
                {
                    "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
                    "script_sha256": hashlib.sha256(script_text.encode()).hexdigest(),
                }
            )
            contract.update({"config_bytes": config_bytes, "script_text": script_text})
            if scheduler_type == "slurm":
                token = slurm.submit_token(row, slurm_resources, execution["runtime_commit"])
                scheduler_script_text = slurm.render_batch_script(
                    run=row,
                    execution={**execution, "workdir": str(run_cwd)},
                    resources=slurm_resources,
                    token=token,
                    result_path=row["scheduler_result_path"],
                    allocation_identity_path=row["allocation_identity_path"],
                    execution_snapshot_path=out / "execution_snapshot.json",
                    log_path=row["log_path"],
                    module=run_module,
                )
                row.update(
                    {
                        "scheduler_submit_token": token,
                        "scheduler_script_sha256": hashlib.sha256(scheduler_script_text.encode()).hexdigest(),
                    }
                )
                contract["scheduler_script_text"] = scheduler_script_text
        contracts.append(contract)
    return contracts


def hparam_run_layouts(recipe: dict[str, Any], out: Path, run_index_offset: int) -> list[dict[str, Any]]:
    layouts = []
    for index, combo in enumerate(hparam_combos(recipe), start=run_index_offset):
        identity = run_identity(recipe, index, combo)
        layouts.append(
            {
                "identity": identity,
                "parameters": combo,
                "run_dir": out / "runs" / f"{identity['run_id']}--{identity['run_name']}",
            }
        )
    return layouts


def compile_hparam_final_command(recipe: dict[str, Any], out: Path) -> str | None:
    evaluation = recipe.get("evaluation_policy") if isinstance(recipe.get("evaluation_policy"), dict) else {}
    if not final_script_allowed(recipe, evaluation, False):
        return None
    execution = recipe.get("execution") if isinstance(recipe.get("execution"), dict) else {}
    inputs = recipe.get("inputs") if isinstance(recipe.get("inputs"), dict) else {}
    runtime = recipe.get("runtime") if isinstance(recipe.get("runtime"), dict) else {}
    config_path = (
        out / FROZEN_FINAL_EVAL_CONFIG_NAME if has_explicit_final_eval_config(recipe) else out / "config.source.yaml"
    )
    return rendering.render_command(
        [
            execution["python"],
            "-m",
            rendering.variant_module(recipe, "infer"),
            "--config",
            config_path,
            "--ckpt-path",
            resolved_ckpt_path(recipe),
            "--label-name",
            inputs.get("label_name"),
            "--eval-split",
            "test",
            *rendering.infer_runtime_cli_args(runtime),
            *rendering.infer_input_cli_args(inputs, variant=str(recipe.get("variant"))),
        ]
    )


def render_hparam_final_script(recipe: dict[str, Any], command: str) -> str:
    plan_context = plan_contract.frozen_plan_context(recipe)
    execution = recipe.get("execution") if isinstance(recipe.get("execution"), dict) else {}
    evaluation = recipe.get("evaluation_policy") if isinstance(recipe.get("evaluation_policy"), dict) else {}
    return (
        "\n".join(
            rendering.hparam_script_lines(
                [command],
                selection_split=str(evaluation.get("selection_split") or ""),
                final_external_test=True,
                run_cwd=Path(str(execution.get("workdir") or plan_context["repo_root"])),
            )
        )
        + "\n"
    )


def apply_search_overrides(config: dict[str, Any], combo: dict[str, Any]) -> None:
    for key, value in combo.items():
        if key.startswith("yaml:/"):
            set_json_pointer(config, key.removeprefix("yaml:"), value)


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
    scheduler = execution.get("scheduler")
    if scheduler is None:
        scheduler = {"type": "direct"}
    if not isinstance(scheduler, dict):
        raise ValueError("execution.scheduler must be a mapping.")
    execution["scheduler"] = scheduler
    if scheduler.get("type") == "slurm":
        execution.setdefault("gpus_per_run", 1)
    recipe["execution"] = execution
    return recipe


def compile_hparam_run_all_script(recipe: dict[str, Any], out: Path) -> str:
    plan_context = plan_contract.frozen_plan_context(recipe)
    evaluation = recipe.get("evaluation_policy") if isinstance(recipe.get("evaluation_policy"), dict) else {}
    return (
        "\n".join(
            rendering.hparam_script_lines(
                [
                    rendering.render_command(
                        [
                            plan_context["python"],
                            "-m",
                            "agent_tools",
                            "hparam-run-queue",
                            "--plan-dir",
                            out,
                            "--execute",
                        ]
                    )
                ],
                test_after_fit=evaluation["test_after_fit"],
                selection_split=str(evaluation.get("selection_split") or ""),
                run_cwd=Path(plan_context["repo_root"]),
            )
        )
        + "\n"
    )


def write_hparam_plan(
    recipe: dict,
    out: Path,
    *,
    write_out: Path | None = None,
    unlock_final_test: bool,
    source_config_bytes: bytes,
    source_config_sha256: str,
    profile_audit: dict[str, Any] | None = None,
    run_index_offset: int | None = None,
) -> None:
    out = out.expanduser()
    if not out.is_absolute():
        out = out.resolve()
    physical_out = (write_out or out).expanduser()
    if not physical_out.is_absolute():
        physical_out = physical_out.resolve()
    recipe = freeze_hparam_execution(recipe)
    if unlock_final_test:
        evaluation = dict(recipe.get("evaluation_policy") or {})
        evaluation.update({"external_test_locked": False, "final_test_unlocked": True})
        recipe["evaluation_policy"] = evaluation
    execution = recipe["execution"]
    scheduler = execution.get("scheduler") or {}
    if not isinstance(scheduler, dict):
        raise ValueError("execution.scheduler must be a mapping.")
    scheduler_type = str(scheduler.get("type") or "direct")
    if scheduler_type not in {"direct", "slurm"}:
        raise ValueError("execution.scheduler.type must be direct or slurm.")
    inputs = recipe.get("inputs") if isinstance(recipe.get("inputs"), dict) else {}
    evaluation = recipe.get("evaluation_policy") or {}
    final_allowed = final_script_allowed(recipe, evaluation, False)
    frozen_final_eval_config = out / FROZEN_FINAL_EVAL_CONFIG_NAME
    write_frozen_final_eval_config = physical_out / FROZEN_FINAL_EVAL_CONFIG_NAME
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
    if not inputs.get("config"):
        raise FileNotFoundError("Config path is required.")
    if hashlib.sha256(source_config_bytes).hexdigest() != source_config_sha256:
        raise ValueError("Hparam source config does not match the bound SHA-256.")
    source_config_path = resolve_repo_path(inputs.get("config"))
    if source_config_path is None:
        raise ValueError("Hparam source config path is required.")
    plan_contract.bind_frozen_input_snapshot(
        recipe,
        "inputs.config",
        source_config_path,
        source_config_sha256,
    )
    if final_allowed and has_explicit_final_eval_config(recipe):
        plan_contract.bind_frozen_input_snapshot(
            recipe,
            "inputs.final_eval_config_path",
            final_config_snapshot["source_path"],
            final_config_sha256,
        )
    else:
        snapshots = recipe.get("input_snapshots")
        if isinstance(snapshots, list):
            recipe["input_snapshots"] = [
                snapshot
                for snapshot in snapshots
                if not isinstance(snapshot, dict) or snapshot.get("field") != "inputs.final_eval_config_path"
            ]
    write_frozen_source_config = physical_out / "config.source.yaml"
    physical_out.mkdir(parents=True, exist_ok=True)
    write_frozen_source_config.write_bytes(source_config_bytes)
    if final_allowed and has_explicit_final_eval_config(recipe):
        write_frozen_final_eval_config.write_bytes(final_config_bytes)
    elif write_frozen_final_eval_config.exists():
        write_frozen_final_eval_config.unlink()
    runs = []
    test_after_fit = evaluation["test_after_fit"]
    selection_split = str(evaluation.get("selection_split") or "")
    run_index_offset = next_run_index(recipe) if run_index_offset is None else run_index_offset
    for contract in compile_hparam_run_contracts(
        recipe,
        out,
        run_index_offset,
        source_config_bytes=source_config_bytes,
    ):
        run = dict(contract["row"])
        run_dir = Path(run["run_dir"])
        write_run_dir = physical_out / run_dir.relative_to(out)
        write_run_dir.mkdir(parents=True, exist_ok=True)
        write_cfg_copy = write_run_dir / "config.yaml"
        write_cfg_copy.write_bytes(contract["config_bytes"])
        runtime_dir = Path(run["runtime_dir"])
        checkpoint_dir = Path(run["checkpoint_dir"])
        write_script_path = write_run_dir / "launch.sh"
        write_text(write_script_path, contract["script_text"], executable=True)
        if scheduler_type == "slurm":
            write_scheduler_script = write_run_dir / "job.sbatch"
            write_text(write_scheduler_script, contract["scheduler_script_text"], executable=True)
        write_artifacts_path = write_run_dir / "artifacts.json"
        runs.append(run)
        write_json(
            write_run_dir / "run.json",
            {
                "status": "planned",
                **run,
            },
        )
        write_json(
            write_artifacts_path,
            {
                "runtime_dir": str(runtime_dir),
                "checkpoint_dir": str(checkpoint_dir),
                "external_artifacts": True,
            },
        )
    write_text(physical_out / "run_all.sh", compile_hparam_run_all_script(recipe, out), executable=True)
    write_text(
        physical_out / "validation.sh",
        "\n".join(
            rendering.script_lines([rendering.render_command(["python", "-m", "agent_tools", "skills", "--validate"])])
        )
        + "\n",
        executable=True,
    )
    resolved_recipe = {
        key: value for key, value in recipe.items() if key not in {"_recipe_path", _FINAL_EVAL_CONFIG_SNAPSHOT}
    }
    (physical_out / "recipe.resolved.yaml").write_text(yaml.safe_dump(resolved_recipe, sort_keys=False))
    final_script_path = physical_out / "final_external_test.sh"
    final_unlocked = final_test_unlocked(evaluation, False)
    plan_lines = [
        "# Hyper-Parameter Plan",
        "",
        "Status: PASS",
        "",
        (
            "Run commands evaluate the configured test split after fit."
            if test_after_fit
            else "Run commands do not evaluate the configured test split."
        ),
        f"Candidate selection uses the frozen {selection_split} split metric.",
    ]
    if profile_audit is not None:
        plan_lines.extend(
            [
                (
                    "Selection reports the best observed candidate within frozen search domain, "
                    f"metric {evaluation.get('selection_metric')}, split {selection_split}, and budget "
                    f"{profile_audit['budget']}."
                ),
                "",
                "## Automatic Search Profile",
                "",
                f"Profile: {profile_audit['id']}",
                f"Candidate count: {profile_audit['candidate_count']}",
                "",
                "| Searched family | Canonical keys | Covered levels |",
                "|---|---|---:|",
            ]
        )
        for family in profile_audit["searched_families"]:
            plan_lines.append(f"| {family['id']} | {', '.join(family['keys'])} | {family['covered_levels']} |")
    if final_allowed:
        final_command = compile_hparam_final_command(recipe, out)
        assert final_command is not None
        final_script_text = render_hparam_final_script(recipe, final_command)
        write_text(final_script_path, final_script_text, executable=True)
        plan_lines.append("Final external-test script generated because final test was explicitly unlocked.")
    else:
        if final_script_path.exists():
            final_script_path.unlink()
        if final_unlocked:
            plan_lines.append("Final external-test script not generated; explicit checkpoint path is required.")
        else:
            plan_lines.append("Final external-test script not generated; explicit unlock is required.")
    write_text(physical_out / "plan.md", "\n".join(plan_lines) + "\n")

    plan_recipe = {key: value for key, value in recipe.items() if key != _FINAL_EVAL_CONFIG_SNAPSHOT}
    plan_payload = {
        "status": "PASS",
        "runs": runs,
        "recipe": plan_recipe,
        "resolved_recipe_sha256": file_sha256(physical_out / "recipe.resolved.yaml"),
    }
    if final_allowed and has_explicit_final_eval_config(recipe):
        plan_payload["final_eval_config"] = {
            "path": str(frozen_final_eval_config),
            "sha256": file_sha256(write_frozen_final_eval_config),
            "source_path": final_config_snapshot["source_path"],
        }
    # plan.json is the terminal physical-plan manifest and is written only after
    # every frozen file in the bundle is complete.
    write_json(physical_out / "plan.json", plan_payload)


def render_hparam_preflight_card(
    recipe: dict[str, Any],
    snapshot: dict[str, Any],
    run_configs: list[tuple[dict[str, Any], bytes]],
) -> str:
    variant = str(recipe["variant"])
    config_module = rendering.variant_module(recipe, "config")
    loader = (
        f"{config_module}.load_config(validate_sidecars=True)"
        if variant == "sex_age_baseline"
        else f"{config_module}.load_finetune_config"
    )
    routes: dict[tuple[str, str, str, str, tuple[str, ...]], list[str]] = {}
    for run, config_bytes in run_configs:
        summary = plan_context.load_config_summary_for_recipe(recipe, config_bytes=config_bytes)
        model = (summary or {}).get("model") or {}
        architecture = model.get("backbone") or model.get("name")
        if architecture in (None, ""):
            raise ValueError(f"Generated hparam config lacks architecture provenance: {run['run_id']}")
        details = []
        if model.get("hidden_size") not in (None, ""):
            details.append(f"hidden_size={model['hidden_size']}")
        if model.get("backbone_depth") not in (None, ""):
            details.append(f"layers={model['backbone_depth']}")
        if isinstance(model.get("features"), list) and model["features"]:
            details.append(f"features={', '.join(str(feature) for feature in model['features'])}")
        architecture_text = str(architecture)
        if details:
            architecture_text += f" ({', '.join(details)})"
        channels = []
        for channel in model.get("channels") or []:
            if channel.get("name") in (None, ""):
                continue
            channel_details = [
                f"{field}={channel[field]}"
                for field in ("input_dim", "tokenizer", "out_dim")
                if channel.get(field) not in (None, "")
            ]
            channel_text = str(channel["name"])
            if channel_details:
                channel_text += f" ({', '.join(channel_details)})"
            channels.append(channel_text)
        channel_signature = tuple(channels)
        route = (variant, str(snapshot["module"]), loader, architecture_text, channel_signature)
        routes.setdefault(route, []).append(str(run["run_id"]))

    target = str(snapshot["target"])
    if snapshot.get("host"):
        target += f":{snapshot['host']}"
    lines = [
        "## Hparam Registration Preflight Provenance",
        "",
        f"- Execution target: `{target}`",
        f"- Target Python: `{snapshot['python']}` (frozen command: `{snapshot['python_command']}`)",
        f"- Runtime commit: `{snapshot['runtime_commit']}`",
        f"- Module origin: `{snapshot['module_origin']}`",
        f"- Validated run count: {len(run_configs)}",
        f"- Validated argv count: {len(run_configs)}",
        f"- Validated argv SHA-256: `{snapshot['validated_argv_sha256']}`",
        "",
        "| Variant | Python module | Canonical config loader | Architecture | Channels | Run IDs |",
        "|---|---|---|---|---|---|",
    ]
    for route, run_ids in routes.items():
        route_variant, module, config_loader, architecture, channels = route
        lines.append(
            f"| {route_variant} | {module} | {config_loader} | {architecture} | "
            f"{', '.join(channels) if channels else 'none'} | {', '.join(run_ids)} |"
        )
    return "\n".join(lines)


def preflight_hparam_plan(physical_out: str | Path, *, semantic_out: str | Path) -> str:
    from . import run_artifacts as artifacts

    physical_dir = Path(physical_out)
    plan_dir = Path(semantic_out)
    plan = artifacts.read_hparam_plan(
        physical_dir,
        semantic_dir=plan_dir,
        require_workspace_state=False,
        require_adaptive_commit=False,
    )
    recipe = plan.get("recipe") if isinstance(plan.get("recipe"), dict) else {}
    _hparam_registration_state(plan)
    validate_hparam_output_paths(plan_dir, plan)
    execution = recipe.get("execution") if isinstance(recipe.get("execution"), dict) else {}
    validation_runs = []
    for run in plan["runs"]:
        validation_run = dict(run)
        validation_run["script"] = str(artifacts._physical_plan_path(Path(str(run["script"])), plan_dir, physical_dir))
        validation_runs.append(validation_run)
    snapshot = _inspect_hparam_execution_target(execution, validation_runs)
    snapshot_path = physical_dir / managed_scheduler.EXECUTION_SNAPSHOT_NAME
    managed_scheduler.write_execution_snapshot_file(snapshot_path, snapshot)
    card = render_hparam_preflight_card(
        recipe,
        snapshot,
        [
            (
                run,
                artifacts._physical_plan_path(Path(str(run["config"])), plan_dir, physical_dir).read_bytes(),
            )
            for run in plan["runs"]
        ],
    )
    plan_markdown = physical_dir / "plan.md"
    write_text(plan_markdown, f"{plan_markdown.read_text().rstrip()}\n\n{card}\n")
    plan_payload = read_json(physical_dir / "plan.json")
    plan_payload["execution_snapshot"] = {
        "path": str(plan_dir / managed_scheduler.EXECUTION_SNAPSHOT_NAME),
        "sha256": file_sha256(snapshot_path),
    }
    # Keep plan.json as the terminal physical-plan manifest after adding the frozen target evidence.
    write_json(physical_dir / "plan.json", plan_payload)
    artifacts.read_hparam_plan(
        physical_dir,
        semantic_dir=plan_dir,
        require_workspace_state=False,
        require_adaptive_commit=False,
    )
    managed_scheduler.validated_execution_snapshot(
        physical_dir,
        execution,
        validation_runs,
        {},
        inspector=_inspect_hparam_execution_target,
        plan_label="hparam",
    )
    _hparam_registration_state(plan)
    validate_hparam_output_paths(plan_dir, plan)
    return card


def _inspect_hparam_execution_target(execution: dict[str, Any], runs: list[dict[str, Any]]) -> dict[str, Any]:
    return managed_scheduler.inspect_execution_target(execution, runs, plan_label="hparam")


def commit_hparam_plan(
    out: str | Path,
    *,
    emit_event: bool = True,
    preflight_validated: bool = False,
) -> dict[str, Any]:
    from . import run_artifacts as artifacts

    plan_dir = Path(out).expanduser()
    if not plan_dir.is_absolute():
        plan_dir = plan_dir.resolve()
    try:
        plan = artifacts.read_hparam_plan(
            plan_dir,
            require_workspace_state=False,
            require_adaptive_commit=False,
        )
        if "execution_snapshot" not in plan:
            raise ValueError(f"Hparam plan lacks registration preflight evidence: {plan_dir}")
        recipe = plan.get("recipe") if isinstance(plan.get("recipe"), dict) else {}
        ensure_experiment_workspace(
            recipe,
            plan_dir,
            validate_only=True,
            allow_published_plan=True,
        )
        root, manifest_rows = _hparam_registration_state(plan)
        if not preflight_validated:
            validate_hparam_output_paths(plan_dir, plan)
            execution = recipe.get("execution") if isinstance(recipe.get("execution"), dict) else {}
            managed_scheduler.validated_execution_snapshot(
                plan_dir,
                execution,
                plan["runs"],
                {},
                inspector=_inspect_hparam_execution_target,
                plan_label="hparam",
            )
    except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as exc:
        raise HparamRegistrationPreflightError(str(exc)) from exc
    ensure_experiment_workspace(recipe, plan_dir, allow_published_plan=True)
    merge_run_manifest(root, manifest_rows)
    if emit_event:
        append_event(
            root,
            "plan_created",
            {
                "step_id": (recipe.get("step") or {}).get("id"),
                "plan_dir": str(plan_dir),
                "run_count": len(manifest_rows),
            },
        )
    artifacts.read_hparam_plan(plan_dir, require_adaptive_commit=False)
    return plan


def validate_hparam_output_paths(
    out: str | Path,
    plan: dict[str, Any],
    *,
    runs: list[dict[str, Any]] | None = None,
) -> None:
    plan_dir = Path(out)
    recipe = plan.get("recipe") if isinstance(plan.get("recipe"), dict) else {}
    declared_artifacts = recipe.get("artifacts") if isinstance(recipe.get("artifacts"), dict) else {}
    paths = [plan_dir / "plan.json"]
    for run in plan["runs"] if runs is None else runs:
        paths.extend([Path(str(run["runtime_dir"])), Path(str(run["checkpoint_dir"]))])
    paths.append(
        plan_output_path(plan_dir, declared_artifacts.get("results_csv_path"), "results/agent_hparam_results.csv")
    )
    execution = recipe.get("execution") if isinstance(recipe.get("execution"), dict) else {}
    remote = str(execution["host"]) if execution.get("target", "local") == "ssh" else None
    exp_io.validate_managed_output_paths(Path("/"), paths, remote=remote)


def _hparam_registration_state(plan: dict[str, Any]) -> tuple[Path, list[dict[str, Any]]]:
    recipe = plan.get("recipe") if isinstance(plan.get("recipe"), dict) else {}
    root = experiment_root(recipe)
    if root is None:
        raise ValueError("experiment.root is required.")
    manifest_rows = hparam_manifest_rows(plan)
    plan_registration_rows_state(root, manifest_rows, source="Canonical hparam plan")
    return root, manifest_rows


def hparam_manifest_rows(plan: dict[str, Any]) -> list[dict[str, Any]]:
    runs = plan.get("runs") if isinstance(plan.get("runs"), list) else []
    rows = []
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
        if "terminal_status_owner" in run:
            row["terminal_status_owner"] = run["terminal_status_owner"]
        for field in sorted(SCHEDULER_PLAN_IDENTITY_FIELDS | {"log_path"}):
            if field in run:
                row[field] = run[field]
        row.update(managed_run_parameters(run))
        rows.append(row)
    return rows


def plan_output_path(out: Path, raw: Any, default: str) -> Path:
    path = Path(str(raw or default)).expanduser()
    return path if path.is_absolute() else out / path
