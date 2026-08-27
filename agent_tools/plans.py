from __future__ import annotations

import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any

import yaml

from . import (
    decision_rules as task_rules,
    experiment_io as exp_io,
    plan_context as context,
    plan_contract,
    plan_rendering as rendering,
    repo as repo_tools,
    run_artifacts as artifacts,
    schema_map,
)
from .adapters import SUPPORTED_TASKS, composite_adapter, get_adapter
from .adapters.base import PlanRegistrationPreflightError, TaskAdapter
from .configs import config_summary
from .decision_models import USER_DECISIONS_FILENAME
from .decisions import (
    DecisionIssue,
    DecisionReport,
    DecisionStatus,
    consultation_contract_issues,
    decision_entry_contract_issues,
    evaluate_consultation_gates,
    merge_status,
    user_decision_template,
)
from .experiment_workspace import (
    append_event,
    canonical_local_experiment_root,
    ensure_experiment_workspace,
    experiment_metadata_issues,
    experiment_root,
    file_sha256,
    merge_run_manifest,
    next_run_index,
    read_run_manifest,
    validate_plan_output,
)
from .manifests import read_json, write_json, write_text
from .markdown import questions_markdown, questions_payload
from .models import REPO_ROOT, resolve_repo_path
from .recipes import load_consultation_policy, load_recipe_with_base, load_user_decisions


def _resolve_write_targets(task: str | None) -> dict[str, tuple[str, str]]:
    adapter = get_adapter(task)
    adapter_targets = dict(adapter.decision_recipe_targets) if adapter is not None else {}
    return schema_map.merged_write_targets(adapter_targets)


def _recipe_contract_issues(recipe: dict, user_decisions: dict, policy: dict) -> list[DecisionIssue]:
    has_layers = isinstance(recipe.get("_base_recipe"), dict) and isinstance(recipe.get("_local_recipe"), dict)
    task_owner = recipe["_local_recipe"] if has_layers else recipe
    recipe_task = task_owner.get("task")
    recipe_decisions = task_owner.get("decisions") if isinstance(task_owner.get("decisions"), dict) else {}
    effective_task = recipe_task
    if effective_task in (None, "", "ASK_USER"):
        effective_task = _decision_value(recipe_decisions.get("task"))
    if effective_task in (None, "", "ASK_USER"):
        effective_task = _decision_value(user_decisions.get("task"))
    # ASK_USER is an unresolved sentinel, not a task scope for contract validation.
    if effective_task == "ASK_USER":
        effective_task = None
    issues: list[DecisionIssue] = []
    if has_layers:
        base_recipe = recipe["_base_recipe"]
        local_recipe = recipe["_local_recipe"]
        local_task = local_recipe.get("task")
        if local_task in (None, "", "ASK_USER"):
            local_task = effective_task
        if local_task in (None, "", "ASK_USER"):
            issues.append(
                DecisionIssue(
                    DecisionStatus.NEEDS_USER_INPUT,
                    "task",
                    "Task is missing from the hparam recipe owner.",
                    "Which task should this recipe use?",
                    {"source_layer": "local", "preflight_before_workspace": True},
                )
            )
        owner = composite_adapter()
        local_contract_task = owner.task if local_task in (None, "", "ASK_USER") else str(local_task)
        sources = [
            (base_recipe, owner.base_task, "base"),
            (local_recipe, local_contract_task, "local"),
        ]
    else:
        sources = [(recipe, str(effective_task or ""), "effective")]

    if has_layers and base_recipe.get("task") not in (None, "", "ASK_USER", owner.base_task):
        issues.append(
            _recipe_contract_issue(
                "task",
                "Hparam base recipe must use task=finetune.",
                base_recipe.get("task"),
                "base",
            )
        )
    for source_recipe, task, source_layer in sources:
        issues.extend(_source_recipe_contract_issues(source_recipe, task, policy, source_layer))
    issues.extend(
        consultation_contract_issues(
            str(effective_task) if effective_task not in (None, "") else None,
            {"decisions": user_decisions},
            policy,
            source_layer="user",
        )
    )
    return issues


def _source_recipe_contract_issues(
    recipe: dict,
    task: str,
    policy: dict,
    source_layer: str,
) -> list[DecisionIssue]:
    if not task:
        return []
    issues = task_rules.recipe_structure_issues(task, recipe, source_layer=source_layer)
    if task not in SUPPORTED_TASKS:
        return issues
    for issue in experiment_metadata_issues(recipe, require_values=False, source_layer=source_layer):
        issues.append(
            DecisionIssue(
                DecisionStatus(issue["status"]),
                issue["field"],
                issue["message"],
                issue.get("question"),
                issue.get("evidence", {}),
            )
        )
    issues.extend(decision_entry_contract_issues(task, recipe, policy, source_layer=source_layer))
    return issues


def _recipe_contract_issue(field: str, message: str, value: Any, source_layer: str) -> DecisionIssue:
    return DecisionIssue(
        DecisionStatus.FAIL,
        field,
        message,
        None,
        {"value": value, "source_layer": source_layer, "preflight_before_workspace": True},
    )


def _decision_value(raw: Any) -> Any:
    return raw.get("value") if isinstance(raw, dict) else raw


def _materialize_decisions(
    recipe: dict,
    decisions: dict,
    *,
    user_supplied: bool = False,
) -> list[DecisionIssue]:
    decision_values = {field: raw.get("value") if isinstance(raw, dict) else raw for field, raw in decisions.items()}
    issues: list[DecisionIssue] = []

    if "task" in decision_values:
        task = decision_values["task"]
        task_owner = recipe.get("_local_recipe") if isinstance(recipe.get("_local_recipe"), dict) else recipe
        recipe_task = task_owner.get("task")
        if task not in (None, "", "ASK_USER"):
            if recipe_task in (None, "", "ASK_USER"):
                recipe["task"] = task
                task_owner["task"] = task
            elif task != recipe_task:
                issues.append(
                    DecisionIssue(
                        DecisionStatus.FAIL,
                        "task",
                        "Explicit task decision conflicts with the recipe task.",
                        None,
                        {"recipe": recipe_task, "user": task, "preflight_before_workspace": True},
                    )
                )

    if user_supplied and "train_val_test_policy" in decision_values:
        selection_split = decision_values["train_val_test_policy"]
        if selection_split not in (None, "", "ASK_USER") and selection_split not in ("train", "val", "test"):
            issues.append(
                DecisionIssue(
                    DecisionStatus.FAIL,
                    "train_val_test_policy",
                    "Explicit train_val_test_policy must be train, val, or test.",
                    None,
                    {"value": selection_split, "preflight_before_workspace": True},
                )
            )

    canonical_fields = _resolve_write_targets(recipe.get("task"))
    if decision_values.get("train_val_test_policy") not in ("train", "val", "test"):
        canonical_fields.pop("train_val_test_policy", None)

    for field, (section, key) in canonical_fields.items():
        if field not in decision_values:
            continue
        value = decision_values[field]
        if value == "ASK_USER":
            continue
        if value in (None, "") and not (field == "pretrained_backbone_path" and value is None):
            issues.append(
                DecisionIssue(
                    DecisionStatus.NEEDS_USER_INPUT,
                    field,
                    f"{field} decision is unresolved.",
                    f"What value should {field} use?",
                    {"value": value, "preflight_before_workspace": True},
                )
            )
            continue
        target = recipe.get(section) if isinstance(recipe.get(section), dict) else {}
        recipe[section] = {**target, key: value}
    if user_supplied:
        recipe_decisions = recipe.get("decisions") if isinstance(recipe.get("decisions"), dict) else {}
        recipe["decisions"] = {**recipe_decisions, **decisions}
    return issues


def _materialize_task_defaults(recipe: dict, policy: dict, user_decisions: dict) -> None:
    task_defaults = (policy.get("task_defaults") or {}).get(recipe.get("task"), {})
    decisions = recipe.get("decisions") if isinstance(recipe.get("decisions"), dict) else {}
    local_recipe = recipe.get("_local_recipe") if isinstance(recipe.get("_local_recipe"), dict) else recipe
    local_decisions = local_recipe.get("decisions") if isinstance(local_recipe.get("decisions"), dict) else {}
    targets = _resolve_write_targets(recipe.get("task"))
    for field, decision in task_defaults.items():
        section, key = targets[field]
        target = recipe.get(section) if isinstance(recipe.get(section), dict) else {}
        local_target = local_recipe.get(section) if isinstance(local_recipe.get(section), dict) else {}
        resolved_decision = decisions.get(field)
        if (
            field in user_decisions
            or (isinstance(resolved_decision, dict) and resolved_decision.get("source") == "explicit_user")
            or field in local_decisions
            or key in local_target
        ):
            continue
        recipe[section] = {**target, key: decision["value"]}
        decisions = {**decisions, field: decision}
        recipe["decisions"] = decisions


def evaluate_recipe(
    recipe_path: str | Path,
    user_decisions_path: str | Path | None = None,
) -> tuple[dict, dict | None, DecisionReport]:
    recipe = load_recipe_with_base(recipe_path)
    source_recipe = copy.deepcopy(recipe)
    source = resolve_repo_path(recipe_path)
    if source is not None:
        recipe["_recipe_path"] = str(source.resolve())
    policy = load_consultation_policy()
    user_decisions = load_user_decisions(user_decisions_path)
    contract_issues = _recipe_contract_issues(recipe, user_decisions, policy)
    if contract_issues:
        return (
            recipe,
            None,
            DecisionReport(
                status=merge_status(contract_issues),
                issues=contract_issues,
                decisions={},
            ),
        )
    recipe_decisions = recipe.get("decisions") if isinstance(recipe.get("decisions"), dict) else {}
    local_recipe = recipe.get("_local_recipe") if isinstance(recipe.get("_local_recipe"), dict) else None
    if local_recipe is not None:
        local_decisions = local_recipe.get("decisions") if isinstance(local_recipe.get("decisions"), dict) else {}
        recipe_decisions = dict(recipe_decisions)
        if "task" in local_decisions:
            recipe_decisions["task"] = local_decisions["task"]
        else:
            recipe_decisions.pop("task", None)
        recipe["decisions"] = recipe_decisions
    materialization_issues = _materialize_decisions(recipe, recipe_decisions)
    materialization_issues.extend(_materialize_decisions(recipe, user_decisions, user_supplied=True))
    _materialize_task_defaults(recipe, policy, user_decisions)

    inputs = recipe.get("inputs") if isinstance(recipe.get("inputs"), dict) else {}
    source_config = inputs.get("config")
    source_config_path = resolve_repo_path(source_config)
    source_config_bytes = None
    if source_config_path is not None and source_config_path.is_file():
        source_config_bytes = source_config_path.read_bytes()

    config_error = None
    try:
        cfg = context.load_config_summary_for_recipe(recipe, config_bytes=source_config_bytes)
    except Exception as exc:
        if "config" not in user_decisions:
            raise
        cfg = None
        config_error = str(exc)
    config_changed_during_validation = False
    if source_config_bytes is not None:
        try:
            config_changed_during_validation = (
                source_config_path is None or source_config_path.read_bytes() != source_config_bytes
            )
        except OSError:
            config_changed_during_validation = True
        if cfg is not None and not config_changed_during_validation:
            cfg = dict(cfg)
            cfg["_source_config_bytes"] = source_config_bytes
            cfg["_source_config_sha256"] = hashlib.sha256(source_config_bytes).hexdigest()
    consultation_cfg = dict(cfg) if cfg is not None else None
    if consultation_cfg is not None:
        consultation_cfg.pop("_source_config_bytes", None)
    recipe_adapter = get_adapter(recipe.get("task"))
    binding_issues = (
        recipe_adapter.bind_effective_recipe(recipe, consultation_cfg, source_recipe=source_recipe)
        if recipe_adapter is not None
        else []
    )
    report = evaluate_consultation_gates(
        recipe.get("task"),
        recipe,
        consultation_cfg,
        {"user_decisions": user_decisions},
        policy,
    )
    report = _append_issues(report, materialization_issues)
    report = _append_issues(report, binding_issues)
    if config_changed_during_validation:
        report = _append_issues(
            report,
            [
                DecisionIssue(
                    DecisionStatus.FAIL,
                    "config",
                    "Config changed while consultation gates were validating it.",
                    None,
                    {"config": str(source_config), "preflight_before_workspace": True},
                )
            ],
        )
    if (
        recipe_adapter is not None
        and recipe_adapter.enforces_required_channels
        and cfg is not None
        and cfg.get("is_finetune") is True
        and not cfg.get("blocking_issues")
    ):
        required_channels = user_decisions.get("required_channels", recipe_decisions.get("required_channels"))
        required_channels_value = _decision_value(required_channels)
        config_required_channels = (cfg.get("preset_build") or {}).get("required_channels")
        if (
            required_channels is not None
            and required_channels_value not in (None, "", "ASK_USER")
            and config_required_channels is not None
            and required_channels_value != config_required_channels
        ):
            report = _append_issues(
                report,
                [
                    DecisionIssue(
                        DecisionStatus.FAIL,
                        "required_channels",
                        "required_channels decision differs from config preset_build.required_channels.",
                        None,
                        {
                            "decision": required_channels_value,
                            "config": config_required_channels,
                            "preflight_before_workspace": True,
                        },
                    )
                ],
            )
    override_issues = None
    accepts_config = cfg is not None and (
        (cfg.get("is_finetune") is True and not cfg.get("blocking_issues"))
        or (
            recipe_adapter is not None
            and recipe_adapter.accepts_pretrain_config
            and (cfg.get("is_pretrain") is True or cfg.get("is_finetune") is True)
        )
    )
    if accepts_config:
        override_issues = recipe_adapter.config_override_issues(recipe, cfg) if recipe_adapter is not None else None
    if (
        override_issues is None
        and cfg is not None
        and cfg.get("is_finetune") is True
        and not cfg.get("blocking_issues")
    ):
        inputs = recipe.get("inputs") if isinstance(recipe.get("inputs"), dict) else {}
        evaluation = recipe.get("evaluation_policy") if isinstance(recipe.get("evaluation_policy"), dict) else {}
        decision_values = {
            "data_backend": inputs.get("data_backend"),
            "selection_metric": evaluation.get("selection_metric"),
            "selection_mode": evaluation.get("selection_mode"),
        }
        config_contracts = {}
        for field, decision_value in decision_values.items():
            spec = schema_map.CONFIG_FIELDS[field]
            node = cfg
            for part in spec.summary_path[:-1]:
                node = node.get(part, {})
            config_value = node.get(spec.summary_path[-1])
            config_contracts[field] = (decision_value, config_value, spec.display_path)
        contract_issues = []
        for field, (decision_value, config_value, config_field) in config_contracts.items():
            if decision_value in (None, "", "ASK_USER"):
                continue
            if decision_value != config_value:
                contract_issues.append(
                    DecisionIssue(
                        DecisionStatus.FAIL,
                        field,
                        f"{field} decision differs from config {config_field}.",
                        None,
                        {"decision": decision_value, "config": config_value},
                    )
                )
        report = _append_issues(report, contract_issues)
    raw_config_decision = user_decisions.get("config")
    selected_config_value = (
        raw_config_decision.get("value") if isinstance(raw_config_decision, dict) else raw_config_decision
    )
    selected_config = (
        recipe_adapter is not None
        and (recipe_adapter.uses_finetune_config or recipe_adapter.accepts_pretrain_config)
        and "config" in user_decisions
    )
    if selected_config and selected_config_value in (None, "", "ASK_USER"):
        report = _append_issues(
            report,
            [
                DecisionIssue(
                    DecisionStatus.NEEDS_USER_INPUT,
                    "config",
                    "Explicit config decision is unresolved.",
                    "Which config should this task use?",
                    {
                        "config": selected_config_value,
                        "preflight_before_workspace": True,
                    },
                )
            ],
        )
    elif selected_config:
        blocking_config_issues = (
            cfg.get("blocking_issues", []) if cfg is not None and not recipe_adapter.accepts_pretrain_config else []
        )
        config_kind_valid = cfg is not None and (
            cfg.get("is_finetune") is True
            or (recipe_adapter.accepts_pretrain_config and cfg.get("is_pretrain") is True)
        )
        if config_error or not config_kind_valid or blocking_config_issues:
            message = config_error or (
                "Selected config must be a readable pretrain or finetune model config without blocking issues."
                if recipe_adapter.accepts_pretrain_config
                else "Selected config must be a readable finetune model config without blocking issues."
            )
            report = _append_issues(
                report,
                [
                    DecisionIssue(
                        DecisionStatus.FAIL,
                        "config",
                        message,
                        None,
                        {
                            "config": selected_config_value,
                            "preflight_before_workspace": True,
                        },
                    )
                ],
            )
    report = _append_issues(report, context.index_summary_issues(recipe, cfg))
    if override_issues:
        report = _append_issues(report, override_issues)
    return recipe, cfg, report


def write_questions(output_dir: str | Path, report: DecisionReport) -> None:
    out = Path(output_dir)
    write_json(out / "questions.json", {"questions": questions_payload(report)})
    write_text(out / "questions.md", questions_markdown(report))


def _plan_publication_lock(out: Path):
    lock_root = Path(tempfile.gettempdir()).resolve() / "agent-tools-plan-locks"
    lock_root.mkdir(mode=0o700, exist_ok=True)
    lock_name = hashlib.sha256(str(out).encode()).hexdigest() + ".lock"
    lock_path = lock_root / lock_name
    exp_io.validate_managed_output_paths(Path(lock_path.anchor), [lock_path])
    return exp_io.blocking_file_lock(lock_path)


def write_user_decision_template(
    output_dir: str | Path,
    recipe: dict,
    report: DecisionReport,
) -> tuple[Path, bool] | None:
    payload = user_decision_template(recipe.get("task"), report, load_consultation_policy())
    if not payload:
        return None
    target = Path(output_dir) / USER_DECISIONS_FILENAME
    text = yaml.safe_dump(payload, sort_keys=False)
    target.parent.mkdir(parents=True, exist_ok=True)
    # This file is human-editable, so exclusive creation preserves decisions saved during publication.
    try:
        with target.open("x") as file_obj:
            file_obj.write(text)
    except FileExistsError:
        return target, False
    return target, True


def prepare_doctor_report(output_dir: str | Path | None, recipe: dict, report: DecisionReport) -> DecisionReport:
    adapter = get_adapter(recipe.get("task"))
    return adapter.prepare_doctor_report(recipe, report) if adapter is not None else report


def write_doctor_outputs(
    output_dir: str | Path | None,
    recipe: dict,
    report: DecisionReport,
) -> tuple[Path, bool] | None:
    if output_dir is None or _has_output_artifact_issue(report):
        return None
    out = Path(output_dir).expanduser()
    if not out.is_absolute():
        out = Path.cwd() / out
    out = Path(os.path.normpath(out))
    with _plan_publication_lock(out):
        locked_report = _guard_existing_outputs(
            report,
            plan_contract.blocked_plan_control_paths(out),
            True,
            root=out,
        )
        locked_report = _guard_existing_outputs(
            locked_report,
            plan_contract.pass_plan_control_paths(out),
            _overwrite_policy(recipe),
            root=out,
            require_fresh="PASS plan artifacts already exist; doctor output requires a fresh --output-dir.",
        )
        if _has_output_artifact_issue(locked_report):
            raise ValueError(locked_report.blocking_issues()[-1].message)
        if report.blocking_issues():
            write_questions(out, report)
        return write_user_decision_template(out, recipe, report)


def build_context(
    *,
    task: str,
    config: str | Path | None,
    output_dir: str | Path,
    label_name: str | None = None,
    variant: str | None = None,
    user_decisions_path: str | Path | None = None,
) -> DecisionReport:
    recipe = {
        "name": Path(str(output_dir)).name,
        "task": task,
        "variant": variant,
        "inputs": {"config": str(config) if config else None, "label_name": label_name},
        "evaluation_policy": {},
        "artifacts": {"output_dir": str(output_dir)},
    }
    source_recipe = copy.deepcopy(recipe)
    policy = load_consultation_policy()
    user_decisions = load_user_decisions(user_decisions_path)
    contract_issues = consultation_contract_issues(
        task,
        {"decisions": user_decisions},
        policy,
        source_layer="user",
    )
    if contract_issues:
        return DecisionReport(status=merge_status(contract_issues), issues=contract_issues, decisions={})
    recipe_decisions = recipe.get("decisions") if isinstance(recipe.get("decisions"), dict) else {}
    materialization_issues = _materialize_decisions(recipe, recipe_decisions)
    materialization_issues.extend(_materialize_decisions(recipe, user_decisions, user_supplied=True))
    _materialize_task_defaults(recipe, policy, user_decisions)
    effective_config = (recipe.get("inputs") or {}).get("config")
    cfg = config_summary(effective_config, variant=variant) if effective_config else None
    recipe_adapter = get_adapter(task)
    binding_issues = (
        recipe_adapter.bind_effective_recipe(recipe, cfg, source_recipe=source_recipe)
        if recipe_adapter is not None
        else []
    )
    report = evaluate_consultation_gates(
        task,
        recipe,
        cfg,
        {"label_name": label_name, "user_decisions": user_decisions},
        policy,
        require_experiment=True,
    )
    report = _append_issues(report, materialization_issues)
    report = _append_issues(report, binding_issues)
    out = Path(output_dir)
    if report.exit_code == 0:
        workspace_issue = validate_plan_output(recipe, out)
        if workspace_issue:
            report = _append_issues(
                report,
                [DecisionIssue(DecisionStatus.FAIL, "experiment.root", workspace_issue, None, {})],
            )
    index_payload = context.context_index_summary(recipe, cfg)
    report = _append_issues(
        report,
        context.index_summary_issues(recipe, cfg, index_payload=index_payload),
    )
    commands = _commands_for_recipe(recipe, cfg) if report.exit_code == 0 else []
    if report.exit_code == 0 and not commands:
        report = _unsupported_command_report(report, task)
    report = _guard_existing_outputs(
        report,
        context.planned_context_paths(out, report),
        _overwrite_policy(recipe),
        root=out,
    )
    if _has_output_artifact_issue(report):
        return report
    skill, relevant_docs = context.skill_context(task)
    payload = {
        "task": task,
        "status": report.status.value,
        "can_generate_commands": report.exit_code == 0,
        "consultation_required": any(issue.status == DecisionStatus.NEEDS_USER_INPUT for issue in report.issues),
        "questions": questions_payload(report),
        "repo": repo_tools.repo_summary(),
        "skill": skill,
        "owners": skill.get("owners", []),
        "relevant_docs": relevant_docs,
        "inputs": recipe["inputs"],
        "config_summary": cfg,
        "index_summary": index_payload,
        "preset_summary": context.context_preset_summary(recipe, cfg),
        "expected_artifacts": context.expected_context_artifacts(recipe, cfg, out, report),
        "recommended_commands": commands if report.exit_code == 0 else [],
        "validation_commands": context.validation_commands(recipe),
        "warnings": [issue.message for issue in report.issues if issue.status == DecisionStatus.WARN],
        "blocking_issues": [issue.message for issue in report.blocking_issues()],
    }
    write_json(out / "context.json", payload)
    write_text(out / "context.md", context.context_markdown(payload))
    if report.blocking_issues():
        write_questions(out, report)
        write_text(out / "commands.blocked.sh", rendering.blocked_script(), executable=True)
    elif report.exit_code == 0:
        write_text(
            out / "commands.sh",
            "\n".join(
                rendering.script_lines(
                    _commands_for_recipe(recipe, cfg),
                    run_cwd=REPO_ROOT,
                )
            )
            + "\n",
            executable=True,
        )
        write_text(
            out / "validation.sh",
            "\n".join(rendering.script_lines(context.validation_commands(recipe), run_cwd=REPO_ROOT)) + "\n",
            executable=True,
        )
    return report


def _validate_bound_recipe(
    recipe: dict[str, Any],
    cfg: dict[str, Any] | None,
    report: DecisionReport,
    out: Path,
    *,
    expected_recipe: dict[str, Any] | None,
    expected_base_recipe: dict[str, Any] | None,
    registered_recipe_path: str | Path | None,
    source_config_sha256: str | None,
) -> tuple[bytes | None, str | None]:
    if expected_recipe is not None:
        recipe_source = recipe.get("_local_recipe") if isinstance(recipe.get("_local_recipe"), dict) else recipe
        actual_recipe = {key: value for key, value in recipe_source.items() if not str(key).startswith("_")}
        actual_recipe_json = json.dumps(actual_recipe, sort_keys=True, separators=(",", ":"), allow_nan=False)
        expected_recipe_json = json.dumps(expected_recipe, sort_keys=True, separators=(",", ":"), allow_nan=False)
        if actual_recipe_json != expected_recipe_json:
            raise ValueError("Plan recipe does not match the bound adaptive recipe.")
        base_source = recipe.get("_base_recipe") if isinstance(recipe.get("_base_recipe"), dict) else None
        actual_base_recipe = (
            {key: value for key, value in base_source.items() if not str(key).startswith("_")}
            if base_source is not None
            else None
        )
        actual_base_json = json.dumps(actual_base_recipe, sort_keys=True, separators=(",", ":"), allow_nan=False)
        expected_base_json = json.dumps(expected_base_recipe, sort_keys=True, separators=(",", ":"), allow_nan=False)
        if actual_base_json != expected_base_json:
            raise ValueError("Plan base recipe does not match the bound adaptive recipe.")
    if registered_recipe_path is not None:
        frozen_recipe_path = Path(registered_recipe_path).expanduser()
        if not frozen_recipe_path.is_absolute():
            frozen_recipe_path = (Path.cwd() / frozen_recipe_path).resolve()
        else:
            frozen_recipe_path = frozen_recipe_path.resolve()
        try:
            frozen_recipe_path.relative_to(out)
        except ValueError as exc:
            raise ValueError(
                f"Registered plan recipe must be inside the final plan directory: {frozen_recipe_path}"
            ) from exc
        recipe["_recipe_path"] = str(frozen_recipe_path)
        if isinstance(recipe.get("_local_recipe"), dict):
            recipe["_local_recipe"]["_recipe_path"] = str(frozen_recipe_path)
    validated_config_bytes = cfg.get("_source_config_bytes") if isinstance(cfg, dict) else None
    validated_config_sha256 = cfg.get("_source_config_sha256") if isinstance(cfg, dict) else None
    if report.exit_code == 0:
        if not isinstance(validated_config_bytes, bytes) or not isinstance(validated_config_sha256, str):
            raise ValueError("Successful plan preflight did not bind the source config bytes.")
        if hashlib.sha256(validated_config_bytes).hexdigest() != validated_config_sha256:
            raise ValueError("Validated source config bytes do not match their SHA-256.")
        if source_config_sha256 is not None and source_config_sha256 != validated_config_sha256:
            raise ValueError("Source config does not match the externally bound SHA-256.")
    return validated_config_bytes, validated_config_sha256


def _materialize_adapter_plan(
    *,
    plan_adapter: TaskAdapter,
    recipe: dict[str, Any],
    report: DecisionReport,
    out: Path,
    write_out: Path,
    output_identity: tuple[int, int] | None,
    generated_staging: bool,
    staging_dir: str | Path | None,
    defer_commit: bool,
    validate_only: bool,
    unlock_final_test: bool,
    validated_config_bytes: bytes,
    validated_config_sha256: str,
) -> DecisionReport:
    try:
        plan_adapter.write_plan(
            recipe,
            out,
            write_out=write_out,
            unlock_final_test=unlock_final_test,
            source_config_bytes=validated_config_bytes,
            source_config_sha256=validated_config_sha256,
        )
        preflight_summary = plan_adapter.precommit_plan(out, write_out=write_out)
        if preflight_summary:
            report = _append_issues(
                report,
                [DecisionIssue(DecisionStatus.PASS, "execution.preflight", preflight_summary)],
            )
    except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as exc:
        if write_out.exists() and not write_out.is_symlink():
            shutil.rmtree(write_out)
        if isinstance(exc, OSError) and not isinstance(exc, subprocess.TimeoutExpired):
            raise
        return _append_issues(
            report,
            [
                DecisionIssue(
                    DecisionStatus.FAIL,
                    "execution.preflight",
                    str(exc),
                    None,
                    {"preflight_before_workspace": True},
                )
            ],
        )
    if validate_only:
        shutil.rmtree(write_out)
        return report
    if defer_commit:
        return report
    with _plan_publication_lock(out):
        report = _guard_pass_plan_publication(
            report,
            recipe,
            out,
            unlock_final_test=unlock_final_test,
        )
        if _has_output_artifact_issue(report):
            if write_out.exists() and not write_out.is_symlink():
                shutil.rmtree(write_out)
            return report
        current_output_identity = None
        if os.path.lexists(out):
            output_stat = out.lstat()
            current_output_identity = (output_stat.st_dev, output_stat.st_ino)
        if current_output_identity != output_identity:
            shutil.rmtree(write_out)
            raise ValueError(f"Atomic plan output changed during preflight: {out}")
        out_preexisted = current_output_identity is not None
        if staging_dir is not None or generated_staging:
            try:
                _publish_materialized_plan(write_out, out, out_preexisted=out_preexisted)
            except BaseException:
                if write_out.exists() and not write_out.is_symlink():
                    shutil.rmtree(write_out)
                raise
        try:
            plan_adapter.commit_plan(out, preflight_validated=True)
        except PlanRegistrationPreflightError as exc:
            if not out_preexisted and out.exists() and not out.is_symlink():
                shutil.rmtree(out)
            return _append_issues(
                report,
                [
                    DecisionIssue(
                        DecisionStatus.FAIL,
                        "execution.preflight",
                        str(exc),
                        None,
                        {"preflight_before_workspace": True},
                    )
                ],
            )
    return report


def _materialize_single_run_plan(
    *,
    task: str,
    recipe: dict[str, Any],
    report: DecisionReport,
    out: Path,
    write_out: Path,
    output_identity: tuple[int, int] | None,
    generated_staging: bool,
    staging_dir: str | Path | None,
    defer_commit: bool,
    plan_controller: str | None,
    run_index_offset: int | None,
    unlock_final_test: bool,
    validated_config_bytes: bytes,
) -> DecisionReport:
    if staging_dir is None:
        ensure_experiment_workspace(recipe, out, register_step=False, plan_controller=plan_controller)
    root = experiment_root(recipe)
    if root is None:
        raise ValueError("experiment.root is required.")
    run_adapter = get_adapter(task)
    assert run_adapter is not None
    run_index = next_run_index(recipe) if run_index_offset is None else run_index_offset
    run = plan_contract.generic_run_contract(recipe, out, run_index, run_adapter)
    run_id = run["run_id"]
    run_name = run["run_name"]
    write_run_dir = write_out / "runs" / f"{run_id}--{run_name}"
    write_run_dir.mkdir(parents=True, exist_ok=True)
    write_config_path = write_run_dir / "config.yaml"
    write_config_path.write_bytes(validated_config_bytes)
    contract = run_adapter.compile_plan_contract(
        recipe,
        out,
        run_index_offset=run_index,
        config_bytes=validated_config_bytes,
    )
    run = contract["runs"][0]
    commands = contract["commands"]
    run.update({"status": "planned", "config_sha256": file_sha256(write_config_path)})
    write_text(write_out / "plan.md", context.plan_markdown(report, commands))
    write_text(write_out / "run.sh", contract["script_text"], executable=True)
    write_launch_path = write_run_dir / "launch.sh"
    write_text(write_launch_path, (write_out / "run.sh").read_text(), executable=True)
    run["script_sha256"] = file_sha256(write_launch_path)
    artifact_payload = {
        "declared": recipe.get("artifacts") or {},
        "runtime_dir": run["runtime_dir"],
        "checkpoint_dir": run["checkpoint_dir"],
        "external_artifacts": True,
    }
    write_json(
        write_run_dir / "artifacts.json",
        artifact_payload,
    )
    planned_run = {**run, "command": commands[0]} if len(commands) == 1 else dict(run)
    write_json(write_run_dir / "run.json", {**planned_run, "commands": commands})
    write_json(
        write_out / "plan.json",
        {"status": report.status.value, "commands": commands, "runs": [planned_run], "recipe": recipe},
    )
    resolved_recipe = {key: value for key, value in recipe.items() if key != "_recipe_path"}
    (write_out / "recipe.resolved.yaml").write_text(yaml.safe_dump(resolved_recipe, sort_keys=False))
    if defer_commit:
        return report
    # Generic plans stay staged until the same locked publication gate as materialized hparam plans.
    with _plan_publication_lock(out):
        report = _guard_pass_plan_publication(
            report,
            recipe,
            out,
            unlock_final_test=unlock_final_test,
        )
        if _has_output_artifact_issue(report):
            if write_out != out and write_out.exists() and not write_out.is_symlink():
                shutil.rmtree(write_out)
            return report
        current_output_identity = None
        if os.path.lexists(out):
            output_stat = out.lstat()
            current_output_identity = (output_stat.st_dev, output_stat.st_ino)
        if current_output_identity != output_identity:
            shutil.rmtree(write_out)
            raise ValueError(f"Atomic plan output changed during preflight: {out}")
        out_preexisted = current_output_identity is not None
        if staging_dir is not None or generated_staging:
            try:
                _publish_materialized_plan(write_out, out, out_preexisted=out_preexisted)
            except BaseException:
                if write_out.exists() and not write_out.is_symlink():
                    shutil.rmtree(write_out)
                raise
        ensure_experiment_workspace(
            recipe,
            out,
            plan_controller=plan_controller,
            allow_published_plan=staging_dir is not None or generated_staging,
        )
        manifest_row = {
            **run,
            "parameter_summary": "single resolved recipe",
        }
        merge_run_manifest(
            root,
            [manifest_row],
        )
        append_event(
            root,
            "plan_created",
            {"step_id": (recipe.get("step") or {}).get("id"), "plan_dir": str(out), "run_count": 1},
        )
    return report


def build_plan(
    *,
    recipe_path: str | Path,
    output_dir: str | Path,
    user_decisions_path: str | Path | None = None,
    allow_unresolved: bool = False,
    unlock_final_test: bool = False,
    source_config_sha256: str | None = None,
    expected_recipe: dict[str, Any] | None = None,
    expected_base_recipe: dict[str, Any] | None = None,
    staging_dir: str | Path | None = None,
    defer_commit: bool = False,
    registered_recipe_path: str | Path | None = None,
    allow_adaptive_workflow: bool = False,
    plan_controller: str | None = None,
    run_index_offset: int | None = None,
    validate_only: bool = False,
) -> DecisionReport:
    out = canonical_local_experiment_root(output_dir, Path.cwd())
    recipe, cfg, report = preflight_plan(
        recipe_path=recipe_path,
        output_dir=out,
        user_decisions_path=user_decisions_path,
        allow_unresolved=allow_unresolved,
        unlock_final_test=unlock_final_test,
        allow_existing_output_artifacts=defer_commit,
        allow_adaptive_workflow=allow_adaptive_workflow,
    )
    validated_config_bytes, validated_config_sha256 = _validate_bound_recipe(
        recipe,
        cfg,
        report,
        out,
        expected_recipe=expected_recipe,
        expected_base_recipe=expected_base_recipe,
        registered_recipe_path=registered_recipe_path,
        source_config_sha256=source_config_sha256,
    )
    task = recipe.get("task")
    plan_adapter = get_adapter(task)
    if _has_output_artifact_issue(report):
        return report
    if validate_only and not (plan_adapter is not None and plan_adapter.materializes_plan):
        report = _append_issues(
            report,
            [
                DecisionIssue(
                    DecisionStatus.FAIL,
                    "validate_only",
                    "plan --validate-only currently supports only materialized hyper-parameter recipes.",
                    None,
                    {"preflight_before_workspace": True},
                )
            ],
        )
    if report.exit_code != 0:
        if validate_only:
            return report
        preflight_failed_before_workspace = bool(experiment_metadata_issues(recipe)) or any(
            issue.field in {"experiment", "step", "execution.workdir"}
            or issue.field.startswith("experiment.")
            or issue.field.startswith("step.")
            or issue.evidence.get("preflight_before_workspace") is True
            for issue in report.blocking_issues()
        )
        if preflight_failed_before_workspace:
            return report
        with _plan_publication_lock(out):
            report = _guard_blocked_plan_publication(
                report,
                recipe,
                out,
                allow_unresolved=allow_unresolved,
                unlock_final_test=unlock_final_test,
            )
            if _has_output_artifact_issue(report):
                return report
            ensure_experiment_workspace(recipe, out, plan_controller=plan_controller)
            write_questions(out, report)
            template = write_user_decision_template(out, recipe, report)
            template_path = template[0] if template is not None else None
            write_text(
                out / "plan.blocked.md",
                context.blocked_plan_markdown(report, allow_unresolved, user_decisions_path=template_path),
            )
            if allow_unresolved and report.exit_code == 2:
                write_json(
                    out / "plan.draft.json",
                    {"status": report.status.value, "recipe": recipe, "questions": questions_payload(report)},
                )
        return report

    root = experiment_root(recipe)
    if root is None:
        raise ValueError("experiment.root is required.")
    recipe["experiment"]["root"] = str(root)
    ensure_experiment_workspace(
        recipe,
        out,
        plan_controller=plan_controller,
        validate_only=True,
    )
    input_snapshots = []
    if plan_adapter is not None:
        input_paths = plan_adapter.frozen_input_paths(recipe)
        try:
            input_snapshots = [
                {"field": field, "path": str(path), "sha256": file_sha256(path)} for field, path in input_paths
            ]
            report = _append_issues(report, plan_adapter.configured_input_issues(recipe, cfg))
            current_snapshots = [
                {"field": field, "path": str(path), "sha256": file_sha256(path)} for field, path in input_paths
            ]
        except OSError as exc:
            report = _append_issues(
                report,
                [
                    DecisionIssue(
                        DecisionStatus.FAIL,
                        "inputs",
                        f"Failed to bind planned input files: {exc}",
                        None,
                        {"preflight_before_workspace": True},
                    )
                ],
            )
        else:
            if input_snapshots != current_snapshots:
                report = _append_issues(
                    report,
                    [
                        DecisionIssue(
                            DecisionStatus.FAIL,
                            "inputs",
                            "An input file changed while the final plan snapshot was validating it.",
                            None,
                            {"preflight_before_workspace": True},
                        )
                    ],
                )
        if report.exit_code != 0:
            return report
    source_config_path = resolve_repo_path((recipe.get("inputs") or {}).get("config"))
    if source_config_path is None:
        raise ValueError("Successful plan preflight did not bind the source config path.")
    recipe["input_snapshots"] = input_snapshots
    plan_contract.bind_frozen_input_snapshot(
        recipe,
        "inputs.config",
        source_config_path,
        validated_config_sha256,
    )

    write_out = out
    generated_staging = False
    output_identity = None
    if plan_adapter is not None and os.path.lexists(out):
        output_stat = out.lstat()
        output_identity = (output_stat.st_dev, output_stat.st_ino)
    if defer_commit and staging_dir is None:
        raise ValueError("Deferred plan commit requires a staging directory.")
    if staging_dir is not None:
        if out.exists() and not defer_commit:
            raise ValueError(f"Atomic plan output already exists: {out}")
        write_out = canonical_local_experiment_root(staging_dir, Path.cwd())
        root = experiment_root(recipe)
        if root is None:
            raise ValueError("experiment.root is required.")
        try:
            write_out.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"Atomic plan staging directory must be inside experiment.root: {write_out}") from exc
        if write_out.is_symlink() or write_out.exists():
            raise ValueError(f"Atomic plan staging directory must not exist: {write_out}")
        write_out.mkdir(parents=True)
    elif plan_adapter is not None:
        staging_parent = out.parent
        if os.path.lexists(out) and out.lstat().st_dev != out.parent.lstat().st_dev:
            # A plan may itself be a mount point, so its parent is not always the destination filesystem.
            staging_parent = out
        while not os.path.lexists(staging_parent):
            staging_parent = staging_parent.parent
        write_out = Path(tempfile.mkdtemp(prefix=f".{out.name}.", suffix=".staging", dir=staging_parent))
        generated_staging = True

    if plan_adapter is not None and plan_adapter.materializes_plan:
        return _materialize_adapter_plan(
            plan_adapter=plan_adapter,
            recipe=recipe,
            report=report,
            out=out,
            write_out=write_out,
            output_identity=output_identity,
            generated_staging=generated_staging,
            staging_dir=staging_dir,
            defer_commit=defer_commit,
            validate_only=validate_only,
            unlock_final_test=unlock_final_test,
            validated_config_bytes=validated_config_bytes,
            validated_config_sha256=validated_config_sha256,
        )
    else:
        return _materialize_single_run_plan(
            task=task,
            recipe=recipe,
            report=report,
            out=out,
            write_out=write_out,
            output_identity=output_identity,
            generated_staging=generated_staging,
            staging_dir=staging_dir,
            defer_commit=defer_commit,
            plan_controller=plan_controller,
            run_index_offset=run_index_offset,
            unlock_final_test=unlock_final_test,
            validated_config_bytes=validated_config_bytes,
        )


def _publish_materialized_plan(write_out: Path, out: Path, *, out_preexisted: bool) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    if not out_preexisted:
        write_out.replace(out)
        return
    if out.is_symlink() or not out.is_dir():
        raise ValueError(f"Atomic plan output is not a directory: {out}")

    backup_parent = out if write_out.parent == out else out.parent
    backup = Path(tempfile.mkdtemp(prefix=f".{out.name}.", suffix=".backup", dir=backup_parent))
    source_names = {path.name for path in write_out.iterdir()}
    replaced_names = set(source_names)
    for optional_name in ("final_external_test.sh", "config.final_eval.yaml"):
        if optional_name not in source_names:
            replaced_names.add(optional_name)
    old_order = ["plan.json", *sorted(replaced_names - {"plan.json"})]
    new_order = [*sorted(source_names - {"plan.json"}), "plan.json"]
    moved_old = []
    moved_new = []
    try:
        # Hide the old manifest while plan-owned top-level entries change; restore it last on failure.
        for name in old_order:
            current = out / name
            if os.path.lexists(current):
                current.replace(backup / name)
                moved_old.append(name)
        for name in new_order:
            (write_out / name).replace(out / name)
            moved_new.append(name)
    except BaseException:
        for name in reversed(moved_new):
            current = out / name
            if os.path.lexists(current):
                current.replace(write_out / name)
        for name in reversed(moved_old):
            (backup / name).replace(out / name)
        shutil.rmtree(backup)
        raise
    shutil.rmtree(backup)
    write_out.rmdir()


def preflight_plan(
    *,
    recipe_path: str | Path,
    output_dir: str | Path,
    user_decisions_path: str | Path | None = None,
    allow_unresolved: bool = False,
    unlock_final_test: bool = False,
    allow_existing_output_artifacts: bool = False,
    allow_adaptive_workflow: bool = False,
) -> tuple[dict, dict | None, DecisionReport]:
    recipe, cfg, report = evaluate_recipe(recipe_path, user_decisions_path)
    adaptive = recipe.get("adaptive") if isinstance(recipe.get("adaptive"), dict) else {}
    if adaptive.get("enabled") is True and not allow_adaptive_workflow:
        report = _append_issues(
            report,
            [
                DecisionIssue(
                    DecisionStatus.FAIL,
                    "adaptive.enabled",
                    "Adaptive recipes must be initialized with hparam-adaptive-init, not plan.",
                    None,
                    {"preflight_before_workspace": True},
                )
            ],
        )
    out = canonical_local_experiment_root(output_dir, Path.cwd())
    metadata_unresolved = bool(experiment_metadata_issues(recipe)) or any(
        issue.field in {"experiment", "step"}
        or issue.field.startswith("experiment.")
        or issue.field.startswith("step.")
        for issue in report.blocking_issues()
    )
    if not metadata_unresolved:
        workspace_issue = validate_plan_output(recipe, out)
        if workspace_issue:
            report = _append_issues(
                report,
                [DecisionIssue(DecisionStatus.FAIL, "experiment.root", workspace_issue, None, {})],
            )
    inputs = recipe.get("inputs") if isinstance(recipe.get("inputs"), dict) else {}
    source_config = inputs.get("config")
    if source_config not in (None, "", "ASK_USER"):
        source_path = resolve_repo_path(source_config)
        try:
            config_is_freezable = source_path is not None and source_path.is_file()
            if config_is_freezable:
                source_path.read_text()
        except (OSError, UnicodeError):
            config_is_freezable = False
        if not config_is_freezable:
            blocking_config_issue = next(
                (issue for issue in report.blocking_issues() if issue.field == "config"),
                None,
            )
            if blocking_config_issue is not None and blocking_config_issue.status == DecisionStatus.FAIL:
                blocking_config_issue.message = f"Config cannot be frozen from a local file: {source_config}"
                blocking_config_issue.evidence["preflight_before_workspace"] = True
            else:
                report = _append_issues(
                    report,
                    [
                        DecisionIssue(
                            DecisionStatus.FAIL,
                            "config",
                            f"Config cannot be frozen from a local file: {source_config}",
                            None,
                            {"config": str(source_config), "preflight_before_workspace": True},
                        )
                    ],
                )
    preflight_adapter = get_adapter(recipe.get("task"))
    if preflight_adapter is not None and (
        report.exit_code == 0 or (report.exit_code == 2 and preflight_adapter.preflight_on_unresolved)
    ):
        adapter_preflight = preflight_adapter.preflight_issues(
            recipe,
            cfg,
            unlock_final_test=unlock_final_test,
            output_dir=out,
        )
        if adapter_preflight:
            report = _append_issues(report, adapter_preflight)
    if report.exit_code == 0 and not (preflight_adapter is not None and preflight_adapter.materializes_plan):
        commands = _commands_for_recipe(recipe, cfg)
        if not commands:
            report = _unsupported_command_report(report, str(recipe.get("task")))
    successful_plan = report.exit_code == 0
    if successful_plan:
        plan_contract.bind_plan_context(recipe)
    if successful_plan:
        report = _guard_pass_plan_publication(
            report,
            recipe,
            out,
            unlock_final_test=unlock_final_test,
            allow_existing=allow_existing_output_artifacts,
        )
        root = experiment_root(recipe)
        if root is not None:
            report = _guard_existing_outputs(
                report,
                [root / "run_matrix.csv", root / "reports" / "run_matrix.md", root / "events.jsonl"],
                _overwrite_policy(recipe),
                root=root,
                allow_existing=True,
            )
    else:
        report = _guard_blocked_plan_publication(
            report,
            recipe,
            out,
            allow_unresolved=allow_unresolved,
            unlock_final_test=unlock_final_test,
            allow_existing=allow_existing_output_artifacts,
        )
    return recipe, cfg, report


def collect_runs(root: str | Path, metric: str | None, output: str | Path) -> None:
    rows: list[dict[str, Any]] = []
    root_path = Path(root)
    output_path = Path(output).expanduser()
    if not output_path.is_absolute():
        output_path = Path.cwd() / output_path
    canonical_manifest = root_path / "run_manifest.tsv"
    if output_path.resolve() == canonical_manifest.resolve() or (
        output_path.exists() and canonical_manifest.exists() and output_path.samefile(canonical_manifest)
    ):
        raise ValueError("collect-runs output cannot overwrite canonical run_manifest.tsv.")
    # The report may live outside the experiment, so validate its complete absolute topology from the filesystem root.
    exp_io.validate_managed_output_paths(Path(output_path.anchor), [output_path])
    managed_rows = read_run_manifest(root_path)
    for managed in managed_rows:
        runtime_dir = Path(managed["runtime_dir"]) if managed.get("runtime_dir") else None
        manifest = artifacts.find_run_manifest(managed)
        data = read_json(manifest) if manifest is not None else {}
        wandb_summary = _wandb_summary_for_run(runtime_dir) if runtime_dir is not None else {}
        row = {
            "kind": "managed_run",
            **managed,
            "best checkpoint": data.get("best_model_path"),
            "best monitor": data.get("monitor"),
            "monitor mode": data.get("monitor_mode"),
            "epoch": data.get("epoch"),
            "timestamps": data.get("finished_at_utc") or data.get("created_at_utc"),
        }
        if metric:
            row[metric] = (data.get("metrics") or {}).get(metric, wandb_summary.get(metric))
        for key, value in wandb_summary.items():
            row[f"wandb.{key}"] = value
        rows.append(row)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row}) if rows else ["version"]
    with output_path.open("w", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _wandb_summary_for_run(run_dir: Path) -> dict[str, Any]:
    candidates = sorted(run_dir.glob("wandb/*/files/wandb-summary.json"))
    if not candidates:
        return {}
    try:
        data = yaml.safe_load(candidates[-1].read_text())
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _commands_for_recipe(recipe: dict, cfg: dict | None = None) -> list[str]:
    task = recipe.get("task")
    adapter = get_adapter(task)
    if adapter is not None:
        return adapter.commands(recipe, cfg)
    return []


def _append_issues(report: DecisionReport, issues: list[DecisionIssue]) -> DecisionReport:
    all_issues = [*report.issues, *issues]
    return DecisionReport(status=merge_status(all_issues), issues=all_issues, decisions=report.decisions)


def _unsupported_command_report(report: DecisionReport, task: str | None) -> DecisionReport:
    return _append_issues(
        report,
        [
            DecisionIssue(
                DecisionStatus.FAIL,
                "task",
                f"No command renderer is implemented for task: {task}.",
                None,
                {"task": task},
            )
        ],
    )


def _has_output_artifact_issue(report: DecisionReport) -> bool:
    return any(issue.field == "output_artifacts" for issue in report.issues)


def _overwrite_policy(recipe: dict) -> Any:
    section, key = _resolve_write_targets(recipe.get("task"))["overwrite_policy"]
    owner = recipe.get(section) if isinstance(recipe.get(section), dict) else {}
    return owner.get(key)


def _guard_existing_outputs(
    report: DecisionReport,
    paths: list[Path],
    overwrite_policy: Any,
    *,
    root: Path,
    allow_existing: bool = False,
    require_fresh: str | None = None,
) -> DecisionReport:
    try:
        exp_io.validate_managed_output_paths(root, paths)
    except ValueError as exc:
        return _append_issues(
            report,
            [
                DecisionIssue(
                    DecisionStatus.FAIL,
                    "output_artifacts",
                    f"Output artifacts are unsafe: {exc}",
                    None,
                    {"paths": [str(path) for path in paths]},
                )
            ],
        )
    if allow_existing:
        return report
    existing = sorted(str(path) for path in paths if path.exists())
    if not existing:
        return report
    if require_fresh is not None:
        status = DecisionStatus.FAIL
        message = require_fresh
        question = None
    elif overwrite_policy is True:
        return report
    elif overwrite_policy is False:
        status = DecisionStatus.FAIL
        message = "Output artifacts already exist and overwrite_policy=false."
        question = None
    else:
        status = DecisionStatus.NEEDS_USER_INPUT
        message = "Output artifacts already exist and overwrite policy is not explicit."
        question = "Is overwriting existing agent-generated output files allowed for this task?"
    evidence = {"existing_paths": existing}
    if require_fresh is None:
        evidence["user_decision_field"] = "overwrite_policy"
    return _append_issues(
        report,
        [
            DecisionIssue(
                status,
                "output_artifacts",
                message,
                question,
                evidence,
            )
        ],
    )


def _guard_pass_plan_publication(
    report: DecisionReport,
    recipe: dict,
    out: Path,
    *,
    unlock_final_test: bool,
    allow_existing: bool = False,
) -> DecisionReport:
    planned_paths = _planned_plan_paths(recipe, out, report, False, unlock_final_test)
    report = _guard_existing_outputs(
        report,
        planned_paths,
        _overwrite_policy(recipe),
        root=out,
        allow_existing=allow_existing,
    )
    # Blocked bundles are human-editable evidence; reusing one would mix PASS and blocked envelopes.
    return _guard_existing_outputs(
        report,
        plan_contract.blocked_plan_control_paths(out),
        _overwrite_policy(recipe),
        root=out,
        require_fresh="Blocked plan artifacts already exist; retry with a fresh --output-dir.",
    )


def _guard_blocked_plan_publication(
    report: DecisionReport,
    recipe: dict,
    out: Path,
    *,
    allow_unresolved: bool,
    unlock_final_test: bool,
    allow_existing: bool = False,
) -> DecisionReport:
    planned_paths = _planned_plan_paths(recipe, out, report, allow_unresolved, unlock_final_test)
    report = _guard_existing_outputs(
        report,
        planned_paths,
        _overwrite_policy(recipe),
        root=out,
        allow_existing=allow_existing,
    )
    report = _guard_existing_outputs(
        report,
        plan_contract.pass_plan_control_paths(out),
        _overwrite_policy(recipe),
        root=out,
        require_fresh="PASS plan artifacts already exist; retry with a fresh --output-dir.",
    )
    if _has_output_artifact_issue(report) or not out.is_dir():
        return report
    allowed_names = {path.name for path in plan_contract.blocked_plan_control_paths(out)}
    unexpected = sorted(str(path) for path in out.iterdir() if path.name not in allowed_names)
    if not unexpected:
        return report
    # Blocked-plan readers enforce an exact envelope, so foreign entries must fail before step registration.
    return _append_issues(
        report,
        [
            DecisionIssue(
                DecisionStatus.FAIL,
                "output_artifacts",
                "Blocked plan output contains unexpected entries; retry with a fresh --output-dir.",
                None,
                {"unexpected_paths": unexpected},
            )
        ],
    )


def _planned_plan_paths(
    recipe: dict,
    out: Path,
    report: DecisionReport,
    allow_unresolved: bool,
    unlock_final_test: bool,
) -> list[Path]:
    adapter = get_adapter(recipe.get("task"))
    if adapter is not None:
        adapter_paths = adapter.planned_plan_paths(
            recipe, out, report, allow_unresolved=allow_unresolved, unlock_final_test=unlock_final_test
        )
        if adapter_paths is not None:
            return adapter_paths
    if report.exit_code != 0:
        return plan_contract.blocked_plan_control_paths(out)
    assert adapter is not None
    run = plan_contract.generic_run_contract(recipe, out, next_run_index(recipe), adapter)
    run_dir = Path(run["run_dir"])
    return [
        out / "plan.json",
        out / "plan.md",
        out / "run.sh",
        out / "recipe.resolved.yaml",
        run_dir / "run.json",
        run_dir / "config.yaml",
        run_dir / "launch.sh",
        run_dir / "artifacts.json",
    ]
