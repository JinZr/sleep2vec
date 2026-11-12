"""Recipe evaluation, doctor reports, and frozen plan publication.

Adds authored/base/local layer orchestration and policy-dependent decision
validation on top of the structural contract in ``decision_rules``, then stages
and publishes a plan under the publication lock.

``doctor`` runs the same single validation pass (``preflight_plan`` with
``output_dir=None``) and reports it without publishing runnable commands.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
import copy
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from typing import Any

import yaml

from . import (
    configs,
    decision_rules as task_rules,
    execution_snapshot,
    experiment_io as exp_io,
    plan_context as context,
    plan_contract,
    plan_hparam,
    run_artifacts as artifacts,
    schema_map,
)
from .adapters.base import TaskAdapter
from .adapters.registry import SUPPORTED_TASKS, composite_adapter, get_adapter
from .decision_models import USER_DECISIONS_FILENAME
from .decisions import (
    DecisionIssue,
    DecisionReport,
    DecisionStatus,
    consultation_contract_issues,
    decision_entry_contract_issues,
    evaluate_consultation_gates,
    experiment_consultation_issues,
    merge_status,
    resolved_user_decisions,
    user_decision_template,
)
from .experiment_workspace import (
    append_event,
    canonical_local_experiment_root,
    ensure_experiment_workspace,
    event_matches,
    experiment_metadata_issues,
    experiment_root,
    file_sha256,
    merge_run_manifest,
    next_run_index,
    plan_registration_lock,
    plan_registration_rows_state,
    read_experiment_events,
    read_registered_steps,
    read_step_manifest,
    validate_existing_experiment_manifest,
    validate_plan_output,
)
from .manifests import read_json, write_json, write_text
from .markdown import questions_markdown, questions_payload
from .models import (
    YAML_SAFE_LOADER,
    BoundFinalEvalConfigSnapshot,
    ConfigSummaryInput,
    JsonValue,
    json_ready,
    resolve_repo_path,
    validate_json_value,
)
from .recipes import load_consultation_policy, load_recipe_with_base, load_user_decisions


def _resolve_write_targets(task: str | None) -> dict[str, tuple[str, str]]:
    adapter = get_adapter(task)
    adapter_targets = dict(adapter.decision_recipe_targets) if adapter is not None else {}
    return schema_map.merged_write_targets(adapter_targets)


def _recipe_contract_issues(recipe: dict, user_decisions: dict, policy: dict) -> list[DecisionIssue]:
    has_layers = isinstance(recipe.get("_base_recipe"), dict) and isinstance(recipe.get("_local_recipe"), dict)
    task_owner = recipe["_local_recipe"] if has_layers else recipe
    recipe_task = task_owner.get("task")
    recipe_decisions_value = task_owner.get("decisions")
    recipe_decisions = recipe_decisions_value if isinstance(recipe_decisions_value, dict) else {}
    effective_task = recipe_task
    if effective_task in (None, "", "ASK_USER"):
        task_decision = user_decisions.get("task") if "task" in user_decisions else recipe_decisions.get("task")
        effective_task = _decision_value(task_decision)
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
    task: str | None,
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
        task_owner_value = recipe.get("_local_recipe")
        task_owner = task_owner_value if isinstance(task_owner_value, dict) else recipe
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

    canonical_fields = _resolve_write_targets(recipe.get("task"))
    if decision_values.get("train_val_test_policy") != "val":
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
        target_value = recipe.get(section)
        target = target_value if isinstance(target_value, dict) else {}
        if field == "train_val_test_policy" and not user_supplied:
            existing_value = target.get(key)
            if existing_value not in (None, "", "ASK_USER") and existing_value != value:
                issues.append(
                    DecisionIssue(
                        DecisionStatus.FAIL,
                        field,
                        "Authored train_val_test_policy conflicts with evaluation_policy.selection_split.",
                        None,
                        {
                            "decision": value,
                            "selection_split": existing_value,
                            "preflight_before_workspace": True,
                        },
                    )
                )
                continue
        if field == "hparam_search_space":
            # A search-space decision replaces the authored search shape.
            target = dict(target)
            target.pop("configurations", None)
        recipe[section] = {**target, key: value}
    if user_supplied:
        recipe_decisions_value = recipe.get("decisions")
        recipe_decisions = recipe_decisions_value if isinstance(recipe_decisions_value, dict) else {}
        recipe["decisions"] = {**recipe_decisions, **decisions}
        evaluation_policy = recipe.get("evaluation_policy")
        selection_split = evaluation_policy.get("selection_split") if isinstance(evaluation_policy, dict) else None
        if selection_split not in (None, "", "ASK_USER", "val") and not any(
            issue.field == "train_val_test_policy" for issue in issues
        ):
            issues.append(
                DecisionIssue(
                    DecisionStatus.FAIL,
                    "evaluation_policy.selection_split",
                    "evaluation_policy.selection_split must be val; the test split is never a selection split.",
                    None,
                    {"value": selection_split, "preflight_before_workspace": True},
                )
            )
    return issues


def _materialize_task_defaults(recipe: dict, policy: dict, user_decisions: dict) -> None:
    task_defaults = (policy.get("task_defaults") or {}).get(recipe.get("task"), {})
    decisions_value = recipe.get("decisions")
    decisions = decisions_value if isinstance(decisions_value, dict) else {}
    local_recipe_value = recipe.get("_local_recipe")
    local_recipe = local_recipe_value if isinstance(local_recipe_value, dict) else recipe
    local_decisions_value = local_recipe.get("decisions")
    local_decisions = local_decisions_value if isinstance(local_decisions_value, dict) else {}
    targets = _resolve_write_targets(recipe.get("task"))
    for field, decision in task_defaults.items():
        section, key = targets[field]
        target_value = recipe.get(section)
        target = target_value if isinstance(target_value, dict) else {}
        local_target_value = local_recipe.get(section)
        local_target = local_target_value if isinstance(local_target_value, dict) else {}
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


def _normalize_runtime_commit(recipe: dict[str, Any]) -> None:
    execution = recipe.get("execution")
    if not isinstance(execution, dict):
        return
    runtime_commit = execution.get("runtime_commit")
    if isinstance(runtime_commit, str) and runtime_commit not in ("", "ASK_USER"):
        execution["runtime_commit"] = runtime_commit.lower()


def evaluate_recipe(
    recipe_path: str | Path,
    user_decisions_path: str | Path | None = None,
    *,
    check_existing_experiment: bool = False,
) -> tuple[dict, ConfigSummaryInput | None, DecisionReport]:
    """Evaluate recipe decisions and return (recipe, config summary, consultation report).

    Early recipe-contract blockers return before decisions or defaults are
    materialized. Later stages apply authored/user decisions and task defaults,
    but a blocked result need not be fully resolved. The config summary may be
    None; when available, it can include source bytes and their hash for later
    freezing. Inspect the report before using either result to publish a plan.

    Reads recipe/config/policy inputs and any evidence required by their gates;
    does not publish a workspace or launch runs. check_existing_experiment also
    checks the existing workspace metadata for compatibility. Consultation
    blockers and handled config-loading failures are returned in the report;
    unhandled loading, validation and I/O errors propagate."""
    return _evaluate_loaded_recipe(
        load_recipe_with_base(recipe_path),
        recipe_path,
        user_decisions_path,
        check_existing_experiment=check_existing_experiment,
    )


def _evaluate_loaded_recipe(
    recipe: dict,
    recipe_path: str | Path,
    user_decisions_path: str | Path | None,
    *,
    check_existing_experiment: bool,
) -> tuple[dict, ConfigSummaryInput | None, DecisionReport]:
    # Structure -> decisions -> metadata/workspace identity -> config and gates, each checked once.
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
                decisions=resolved_user_decisions(user_decisions),
            ),
        )
    recipe_decisions_value = recipe.get("decisions")
    recipe_decisions = recipe_decisions_value if isinstance(recipe_decisions_value, dict) else {}
    local_recipe_value = recipe.get("_local_recipe")
    local_recipe = local_recipe_value if isinstance(local_recipe_value, dict) else None
    if local_recipe is not None:
        local_decisions_value = local_recipe.get("decisions")
        local_decisions = local_decisions_value if isinstance(local_decisions_value, dict) else {}
        recipe_decisions = dict(recipe_decisions)
        if "task" in local_decisions:
            recipe_decisions["task"] = local_decisions["task"]
        else:
            recipe_decisions.pop("task", None)
        recipe["decisions"] = recipe_decisions
    authored_decisions = {field: raw for field, raw in recipe_decisions.items() if field not in user_decisions}
    materialization_issues = _materialize_decisions(recipe, authored_decisions)
    materialization_issues.extend(_materialize_decisions(recipe, user_decisions, user_supplied=True))
    _materialize_task_defaults(recipe, policy, user_decisions)

    try:
        validate_json_value(recipe, "Recipe", finite=False)
    except ValueError as exc:
        sources = f"recipe {recipe_path}"
        if user_decisions_path is not None:
            sources += f" and user decisions {user_decisions_path}"
        raise ValueError(
            f"Cannot freeze {sources} as JSON: {exc} Quote YAML dates/timestamps if a string was intended."
        ) from exc

    recipe_adapter = get_adapter(recipe.get("task"))
    input_issues = list(materialization_issues)
    if recipe_adapter is not None:
        input_issues.extend(recipe_adapter.recipe_input_issues(recipe))
    if any(issue.status == DecisionStatus.FAIL for issue in input_issues):
        for issue in input_issues:
            issue.evidence["preflight_before_workspace"] = True
        return (
            recipe,
            None,
            DecisionReport(
                status=merge_status(input_issues),
                issues=input_issues,
                decisions=resolved_user_decisions(user_decisions),
            ),
        )

    metadata_issues = experiment_consultation_issues(recipe.get("task"), recipe)
    if metadata_issues:
        for issue in metadata_issues:
            issue.evidence["preflight_before_workspace"] = True
        issues = [*materialization_issues, *metadata_issues]
        return (
            recipe,
            None,
            DecisionReport(
                status=merge_status(issues),
                issues=issues,
                decisions=resolved_user_decisions(user_decisions),
            ),
        )

    if check_existing_experiment and not materialization_issues:
        root = experiment_root(recipe)
        assert root is not None
        manifest_path = root / "experiment.yaml"
        existing_text = exp_io.read_managed_output_texts_at(root, [manifest_path])[str(manifest_path)]
        if existing_text is not None:
            experiment = {
                key: json_ready(value) for key, value in recipe["experiment"].items() if not str(key).startswith("_")
            }
            experiment["root"] = str(root)
            try:
                validate_existing_experiment_manifest(existing_text, experiment, root)
            except ValueError as exc:
                return (
                    recipe,
                    None,
                    DecisionReport(
                        status=DecisionStatus.FAIL,
                        issues=[
                            DecisionIssue(
                                DecisionStatus.FAIL,
                                "experiment",
                                str(exc),
                                None,
                                {"preflight_before_workspace": True},
                            )
                        ],
                        decisions=resolved_user_decisions(user_decisions),
                    ),
                )

    cfg, report = _evaluate_config_consultation(
        recipe,
        source_recipe=source_recipe,
        recipe_adapter=recipe_adapter,
        policy=policy,
        user_decisions=user_decisions,
        recipe_decisions=recipe_decisions,
        materialization_issues=materialization_issues,
    )
    return recipe, cfg, report


def _evaluate_config_consultation(
    recipe: dict[str, Any],
    *,
    source_recipe: dict[str, Any],
    recipe_adapter: TaskAdapter | None,
    policy: dict[str, Any],
    user_decisions: dict[str, Any],
    recipe_decisions: dict[str, Any],
    materialization_issues: list[DecisionIssue],
) -> tuple[ConfigSummaryInput | None, DecisionReport]:
    inputs_value = recipe.get("inputs")
    inputs = inputs_value if isinstance(inputs_value, dict) else {}
    source_config = inputs.get("config")
    source_config_path = resolve_repo_path(source_config)
    source_config_bytes = None
    if source_config_path is not None and source_config_path.is_file():
        # The only read of the source config: validation, hashing and freezing all use these bytes.
        source_config_bytes = source_config_path.read_bytes()

    cfg: ConfigSummaryInput | None
    config_error = None
    validated_sidecar_keys: dict[str, set[str]] = {}
    try:
        cfg = configs.load_config_summary_for_recipe(
            recipe,
            config_bytes=source_config_bytes,
            validated_sidecar_keys=validated_sidecar_keys,
        )
    except Exception as exc:
        if "config" not in user_decisions:
            raise
        cfg = None
        config_error = str(exc)
    if source_config_bytes is not None and cfg is not None:
        bound_cfg = dict(cfg)
        bound_cfg["_source_config_bytes"] = source_config_bytes
        bound_cfg["_source_config_sha256"] = hashlib.sha256(source_config_bytes).hexdigest()
        cfg = bound_cfg
    consultation_cfg = dict(cfg) if cfg is not None else None
    if consultation_cfg is not None:
        consultation_cfg.pop("_source_config_bytes", None)
    binding_issues = (
        recipe_adapter.bind_effective_recipe(recipe, consultation_cfg, source_recipe=source_recipe)
        if recipe_adapter is not None
        else []
    )
    _normalize_runtime_commit(recipe)
    report = evaluate_consultation_gates(
        recipe.get("task"),
        recipe,
        consultation_cfg,
        {"user_decisions": user_decisions},
        policy,
    )
    report = _append_issues(report, materialization_issues)
    report = _append_issues(report, binding_issues)
    report = _evaluate_required_channels_decision(report, cfg, recipe_adapter, user_decisions, recipe_decisions)
    report, override_issues = _evaluate_config_override_decisions(report, recipe, cfg, recipe_adapter)
    report = _evaluate_selected_config_decision(report, cfg, recipe_adapter, user_decisions, config_error)
    report = _append_issues(
        report, context.index_summary_issues(recipe, cfg, validated_sidecar_keys=validated_sidecar_keys)
    )
    if override_issues:
        report = _append_issues(report, override_issues)
    return cfg, report


def _evaluate_required_channels_decision(
    report: DecisionReport,
    cfg: ConfigSummaryInput | None,
    recipe_adapter: TaskAdapter | None,
    user_decisions: dict[str, Any],
    recipe_decisions: dict[str, Any],
) -> DecisionReport:
    if (
        recipe_adapter is not None
        and recipe_adapter.enforces_required_channels
        and cfg is not None
        and cfg.get("is_finetune") is True
        and not cfg.get("blocking_issues")
    ):
        required_channels = user_decisions.get("required_channels", recipe_decisions.get("required_channels"))
        required_channels_value = _decision_value(required_channels)
        preset_build_summary: Any = cfg.get("preset_build") or {}
        config_required_channels = preset_build_summary.get("required_channels")
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
    return report


def _evaluate_config_override_decisions(
    report: DecisionReport,
    recipe: dict[str, Any],
    cfg: ConfigSummaryInput | None,
    recipe_adapter: TaskAdapter | None,
) -> tuple[DecisionReport, list[DecisionIssue] | None]:
    override_issues = None
    accepts_config = cfg is not None and cfg.get("is_finetune") is True and not cfg.get("blocking_issues")
    if accepts_config and recipe_adapter is not None:
        override_issues = recipe_adapter.config_override_issues(recipe, cfg)
    if override_issues is None and accepts_config:
        inputs_value = recipe.get("inputs")
        inputs = inputs_value if isinstance(inputs_value, dict) else {}
        evaluation_value = recipe.get("evaluation_policy")
        evaluation = evaluation_value if isinstance(evaluation_value, dict) else {}
        decision_values = {
            "data_backend": inputs.get("data_backend"),
            "selection_metric": evaluation.get("selection_metric"),
            "selection_mode": evaluation.get("selection_mode"),
        }
        config_contracts = {}
        for field, decision_value in decision_values.items():
            spec = schema_map.CONFIG_FIELDS[field]
            node: Any = cfg
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
    return report, override_issues


def _evaluate_selected_config_decision(
    report: DecisionReport,
    cfg: ConfigSummaryInput | None,
    recipe_adapter: TaskAdapter | None,
    user_decisions: dict[str, Any],
    config_error: str | None,
) -> DecisionReport:
    raw_config_decision = user_decisions.get("config")
    selected_config_value = (
        raw_config_decision.get("value") if isinstance(raw_config_decision, dict) else raw_config_decision
    )
    selects_config = recipe_adapter is not None and recipe_adapter.uses_finetune_config and "config" in user_decisions
    if selects_config and selected_config_value in (None, "", "ASK_USER"):
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
    elif selects_config:
        blocking_config_issues = cfg.get("blocking_issues", []) if cfg is not None else []
        config_kind_valid = cfg is not None and cfg.get("is_finetune") is True
        if config_error or not config_kind_valid or blocking_config_issues:
            message = (
                config_error or "Selected config must be a readable finetune model config without blocking issues."
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
    return report


def _write_questions(output_dir: str | Path, report: DecisionReport) -> None:
    """Write questions.json and questions.md from an existing consultation report.

    Creates the output directory as needed and overwrites these files in order,
    without checking report status, guarding existing artifacts, or taking a
    publication lock. Bundle publishers own those checks and synchronization.
    Serialization and I/O errors propagate; earlier writes are not rolled back.
    """
    out = Path(output_dir)
    write_json(out / "questions.json", {"questions": questions_payload(report)})
    write_text(out / "questions.md", questions_markdown(report))


_LOCAL_PLAN_LOCKS: dict[Path, threading.Lock] = {}
_LOCAL_PLAN_LOCKS_GUARD = threading.Lock()


@contextmanager
def plan_publication_lock(out: Path) -> Iterator[None]:
    """Block until the publication lock for the canonical output path is held.

    Coordinates threads and cooperating processes for the same output path,
    using a per-user lock file under /tmp rather than inside the plan directory.
    Controllers must keep publication and the associated ownership registration
    inside this context; acquiring it does not publish or register anything.
    The lock is released on context exit, including when the body raises.
    Path-validation and lock I/O errors propagate.
    """
    canonical_out = canonical_local_experiment_root(out, Path.cwd())
    lock_root = Path("/tmp").resolve() / f"agent-tools-plan-locks-{os.getuid()}"
    lock_root.mkdir(mode=0o700, exist_ok=True)
    lock_name = hashlib.sha256(str(canonical_out).encode()).hexdigest() + ".lock"
    lock_path = lock_root / lock_name
    exp_io.validate_managed_output_paths(Path(lock_path.anchor), [lock_path])
    with _LOCAL_PLAN_LOCKS_GUARD:
        local_lock = _LOCAL_PLAN_LOCKS.setdefault(lock_path, threading.Lock())
    with local_lock, exp_io.blocking_file_lock(lock_path):
        yield


def _plan_publication_temporary_paths(out: Path) -> list[Path]:
    if out.is_symlink() or not out.is_dir():
        return []
    prefix = f".{out.name}."
    return sorted(
        path for path in out.iterdir() if path.name.startswith(prefix) and path.name.endswith((".staging", ".backup"))
    )


def write_user_decision_template(
    output_dir: str | Path,
    recipe: dict,
    report: DecisionReport,
    *,
    preserve_existing: bool,
) -> tuple[Path, bool] | None:
    """Publish decisions.yaml from an evaluated recipe and consultation report.

    Returns None without writing when the report is not NEEDS_USER_INPUT or has
    no template-eligible decisions. Otherwise returns (path, created), where
    created is True only when this call published a new file. Uses the local
    recipe's task for layered recipes and does not reevaluate consultation.

    Never overwrites an existing file. preserve_existing=False rejects one;
    True preserves its bytes only if it is an independent regular file whose
    decisions cover the generated template and whose present resolved values
    agree with the report. Missing fields or conflicting concrete values require
    a fresh output directory and raise ValueError.

    Does not acquire a publication lock; callers coordinating a doctor or blocked
    plan bundle hold plan_publication_lock around their writes. Creates parent
    directories and publishes complete template bytes before validating the
    target. Errors propagate and may leave directories or a newly published
    template; this function does not roll back the surrounding output bundle.
    """
    task_owner_value = recipe.get("_local_recipe")
    task_owner = task_owner_value if isinstance(task_owner_value, dict) else recipe
    payload = user_decision_template(task_owner.get("task"), report, load_consultation_policy())
    if not payload:
        return None
    target = Path(output_dir) / USER_DECISIONS_FILENAME
    text = yaml.safe_dump(payload, sort_keys=False)
    target.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
    temporary_path = Path(temporary_name)
    created = False
    try:
        with os.fdopen(file_descriptor, "w") as file_obj:
            file_obj.write(text)
            os.fchmod(file_obj.fileno(), 0o644)
            file_obj.flush()
            os.fsync(file_obj.fileno())
        # Exact blocked bundles cannot retain a CAS lock; link complete bytes without clobbering human edits.
        os.link(temporary_path, target)
        created = True
    except FileExistsError:
        if not preserve_existing:
            raise ValueError(
                f"User decisions appeared during blocked plan publication; retry with a fresh --output-dir: {target}"
            ) from None
        try:
            existing_text = exp_io.read_managed_output_texts_at(target.parent, [target])[str(target)]
        except ValueError as exc:
            raise ValueError(f"Managed output paths must be independent regular files: {target}") from exc
        try:
            existing_payload = yaml.load(existing_text, Loader=YAML_SAFE_LOADER) if existing_text is not None else None
        except (TypeError, yaml.YAMLError):
            existing_payload = None
        existing_decisions = existing_payload.get("decisions") if isinstance(existing_payload, dict) else None
        if not isinstance(existing_decisions, dict) or not set(payload["decisions"]).issubset(existing_decisions):
            raise ValueError(
                f"Existing user decisions file is missing newly requested decisions; "
                f"retry with a fresh --output-dir: {target}"
            ) from None
        resolved_values = {
            field: decision.value
            for field, decision in report.decisions.items()
            if decision.value not in (None, "", "ASK_USER")
        }
        conflicting_fields = [
            field
            for field, resolved_value in resolved_values.items()
            if field in existing_decisions and _decision_value(existing_decisions[field]) != resolved_value
        ]
        if conflicting_fields:
            raise ValueError(
                f"Existing user decisions file conflicts with current resolved decisions "
                f"({', '.join(sorted(conflicting_fields))}); retry with a fresh --output-dir: {target}"
            ) from None
    finally:
        temporary_path.unlink(missing_ok=True)
    exp_io.validate_managed_output_paths(target.parent, [target])
    return target, created


def doctor_runtime_diagnostics_supported(recipe: dict[str, Any]) -> bool:
    adapter = get_adapter(recipe.get("task"))
    return bool(adapter is not None and adapter.supports_doctor_runtime_diagnostics)


def doctor_runtime_card(recipe: dict[str, Any]) -> str:
    """Read-only target runtime diagnostic for doctor, for tasks that declare
    ``supports_doctor_runtime_diagnostics``."""
    execution = recipe.get("execution")
    if not isinstance(execution, dict):
        execution = {}
    python = str(execution.get("python") or sys.executable)
    program = (
        "import importlib.metadata, json, socket, sys; "
        "pl_version = next((dist.version for dist in importlib.metadata.distributions() "
        "if str(dist.metadata.get('Name', '')).lower() == 'pytorch-lightning'), 'unavailable'); "
        "print(json.dumps({'host': socket.gethostname(), 'python': sys.executable, "
        "'python_version': sys.version.split()[0], 'pytorch_lightning_version': pl_version}, sort_keys=True))"
    )
    try:
        result = execution_snapshot.run_execution_command(execution, [python, "-c", program])
    except subprocess.TimeoutExpired:
        return "Doctor runtime unavailable: diagnostic probe timed out"
    except (OSError, ValueError):
        return "Doctor runtime unavailable: diagnostic probe could not start"
    if result.returncode != 0:
        return f"Doctor runtime unavailable: diagnostic probe exited with code {result.returncode}"
    try:
        payload = json.loads(result.stdout)
    except (json.JSONDecodeError, TypeError):
        return "Doctor runtime unavailable: malformed diagnostic output"
    fields = ("host", "python", "python_version", "pytorch_lightning_version")
    if not isinstance(payload, dict) or any(
        not isinstance(payload.get(field), str) or not payload[field] for field in fields
    ):
        return "Doctor runtime unavailable: incomplete diagnostic output"
    return (
        f"Doctor runtime: host={payload['host']}, python={payload['python']}, "
        f"python_version={payload['python_version']}, "
        f"pytorch-lightning={payload['pytorch_lightning_version']}"
    )


def write_doctor_outputs(
    output_dir: str | Path | None,
    recipe: dict,
    report: DecisionReport,
) -> tuple[Path, bool] | None:
    """Write doctor questions and a decision template from a prepared report.

    Returns None without writing if output_dir is None or the report already
    contains an output_artifacts issue. Otherwise canonicalizes the output path,
    acquires plan_publication_lock, and rejects conflicting plan artifacts or
    unsafe doctor output paths with ValueError before writing doctor files.

    Blocking issues cause questions.json and questions.md to be written before
    write_user_decision_template is called with preserve_existing=True. Returns
    its (path, created) result or None when no template is eligible; None does
    not imply that no question files were written. When a template is eligible,
    existing decisions must pass that helper's compatibility checks. Otherwise
    existing decisions are left untouched without checking their content.

    Does not evaluate the recipe, run diagnostics, publish a runnable plan, or
    register workspace ownership. Errors propagate without rolling back earlier
    writes, so question files may already have changed when template publication
    fails. The caller retains the consultation report and its exit status.
    """
    if output_dir is None or _has_output_artifact_issue(report):
        return None
    out = canonical_local_experiment_root(output_dir, Path.cwd())
    with plan_publication_lock(out):
        locked_report = _guard_existing_outputs(
            report,
            [
                *plan_contract.blocked_plan_marker_paths(out),
                *plan_contract.pass_plan_artifact_paths(out),
                *_plan_publication_temporary_paths(out),
            ],
            root=out,
            require_fresh="Plan artifacts already exist; doctor output requires a fresh --output-dir.",
        )
        locked_report = _guard_existing_outputs(
            locked_report,
            plan_contract.doctor_control_paths(out),
            root=out,
            allow_existing=True,
        )
        if _has_output_artifact_issue(locked_report):
            raise ValueError(locked_report.blocking_issues()[-1].message)
        if report.blocking_issues():
            _write_questions(out, report)
        return write_user_decision_template(out, recipe, report, preserve_existing=True)


def _validate_bound_recipe(
    cfg: ConfigSummaryInput | None,
    report: DecisionReport,
) -> tuple[bytes, str, BoundFinalEvalConfigSnapshot | None] | None:
    validated_config_bytes = cfg.get("_source_config_bytes") if isinstance(cfg, dict) else None
    validated_config_sha256 = cfg.get("_source_config_sha256") if isinstance(cfg, dict) else None
    final_eval_config: BoundFinalEvalConfigSnapshot | None = (
        cfg.get("_final_eval_config_snapshot") if isinstance(cfg, dict) else None
    )
    if report.exit_code == 0:
        if not isinstance(validated_config_bytes, bytes) or not isinstance(validated_config_sha256, str):
            raise ValueError("Successful plan preflight did not bind the source config bytes.")
        return validated_config_bytes, validated_config_sha256, final_eval_config
    return None


def _materialize_hparam_plan(
    *,
    recipe: dict[str, Any],
    report: DecisionReport,
    out: Path,
    write_out: Path,
    generated_staging: bool,
    run_index_offset: int | None,
    validate_only: bool,
    unlock_final_test: bool,
    validated_config_bytes: bytes,
    validated_config_sha256: str,
    final_eval_config: BoundFinalEvalConfigSnapshot | None,
) -> DecisionReport:
    try:
        plan_hparam.write_hparam_plan(
            recipe,
            out,
            write_out=write_out,
            unlock_final_test=unlock_final_test,
            source_config_bytes=validated_config_bytes,
            source_config_sha256=validated_config_sha256,
            final_eval_config=final_eval_config,
            run_index_offset=run_index_offset,
        )
        try:
            preflight_summary = artifacts.preflight_hparam_plan(write_out, semantic_out=out)
        except OSError as exc:
            raise RuntimeError(f"Target execution preflight failed: {exc}") from exc
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

    registration_rows = plan_hparam.hparam_manifest_rows(read_json(write_out / "plan.json"))
    staged_tree_sha256 = artifacts.plan_tree_sha256(write_out)
    plan_tree_entries = frozenset(path.name for path in write_out.iterdir()) if out == experiment_root(recipe) else None
    with plan_publication_lock(out):
        try:
            registration_state = _plan_registration_state(
                recipe,
                out,
                registration_rows,
                expected_tree_sha256=staged_tree_sha256,
                expected_tree_entries=plan_tree_entries,
            )
        except BaseException:
            if write_out.exists() and not write_out.is_symlink():
                shutil.rmtree(write_out)
            raise
        if registration_state == "complete":
            shutil.rmtree(write_out)
            return _registered_plan_immutable_report(report, out)
        out_preexisted = os.path.lexists(out)
        if registration_state == "unregistered" and generated_staging:
            try:
                publish_staged_plan_locked(
                    write_out,
                    out,
                    out_preexisted=out_preexisted,
                    replace_unowned_plan=_is_unowned_published_plan(recipe, out),
                )
            except BaseException:
                if write_out.exists() and not write_out.is_symlink():
                    shutil.rmtree(write_out)
                raise
        elif registration_state != "unregistered":
            shutil.rmtree(write_out)
        try:
            artifacts.commit_hparam_plan(out)
        except artifacts.HparamRegistrationPreflightError as exc:
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
    run_adapter: TaskAdapter | None,
    recipe: dict[str, Any],
    report: DecisionReport,
    out: Path,
    write_out: Path,
    generated_staging: bool,
    run_index_offset: int | None,
    validated_config_bytes: bytes,
) -> DecisionReport:
    root = experiment_root(recipe)
    if root is None:
        raise ValueError("experiment.root is required.")
    assert run_adapter is not None
    run_index = next_run_index(recipe) if run_index_offset is None else run_index_offset
    run: plan_contract.GenericRunContract | dict[str, Any] = plan_contract.generic_run_contract(
        recipe, out, run_index, run_adapter
    )
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
    write_text(write_out / "run.sh", contract.get("launch_script_text", contract["script_text"]), executable=True)
    write_launch_path = write_run_dir / "launch.sh"
    write_text(write_launch_path, contract["script_text"], executable=True)
    run["script_sha256"] = file_sha256(write_launch_path)
    if "scheduler_script_text" in contract:
        write_text(write_run_dir / "job.sbatch", contract["scheduler_script_text"], executable=True)
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
    plan = {"status": report.status.value, "commands": commands, "runs": [planned_run], "recipe": recipe}
    write_json(write_out / "plan.json", plan)
    resolved_recipe = {key: value for key, value in recipe.items() if key != "_recipe_path"}
    (write_out / "recipe.resolved.yaml").write_text(yaml.safe_dump(resolved_recipe, sort_keys=False))
    manifest_rows = run_adapter.registration_rows(plan)
    staged_tree_sha256 = artifacts.plan_tree_sha256(write_out)
    plan_tree_entries = frozenset(path.name for path in write_out.iterdir()) if out == root else None
    # Generic plans stay staged until the same locked publication step as materialized hparam plans.
    with plan_publication_lock(out):
        try:
            registration_state = _plan_registration_state(
                recipe,
                out,
                manifest_rows,
                expected_tree_sha256=staged_tree_sha256,
                expected_tree_entries=plan_tree_entries,
            )
        except BaseException:
            if write_out != out and write_out.exists() and not write_out.is_symlink():
                shutil.rmtree(write_out)
            raise
        if registration_state == "complete":
            if write_out != out:
                shutil.rmtree(write_out)
            return _registered_plan_immutable_report(report, out)
        out_preexisted = os.path.lexists(out)
        if registration_state == "unregistered" and generated_staging:
            try:
                publish_staged_plan_locked(
                    write_out,
                    out,
                    out_preexisted=out_preexisted,
                    replace_unowned_plan=_is_unowned_published_plan(recipe, out),
                )
            except BaseException:
                if write_out.exists() and not write_out.is_symlink():
                    shutil.rmtree(write_out)
                raise
        elif registration_state != "unregistered" and write_out != out:
            shutil.rmtree(write_out)
        ensure_experiment_workspace(recipe, out, allow_published_plan=generated_staging)
        merge_run_manifest(
            root,
            manifest_rows,
        )
        append_event(
            root,
            "plan_created",
            {"step_id": (recipe.get("step") or {}).get("id"), "plan_dir": str(out), "run_count": 1},
        )
    return report


@dataclass(frozen=True)
class PlanBuildRequest:
    """Caller arguments of one ``build_plan`` call, as ``_build_plan`` consumes them."""

    recipe_path: str | Path
    output_dir: str | Path
    user_decisions_path: str | Path | None
    allow_unresolved: bool
    unlock_final_test: bool
    validate_only: bool


def build_plan(
    *,
    recipe_path: str | Path,
    output_dir: str | Path,
    user_decisions_path: str | Path | None = None,
    allow_unresolved: bool = False,
    unlock_final_test: bool = False,
    validate_only: bool = False,
) -> DecisionReport:
    """Validate and publish a frozen plan, returning its DecisionReport without launching runs.

    Normally registers the plan and run rows in the experiment workspace. A
    blocked report can publish questions/decisions and, with allow_unresolved,
    a non-runnable draft; early blockers can return without publishing anything.

    validate_only supports materialized hparam plans: it checks a scratch build
    without publishing a runnable plan, but can create scratch and lock files.

    Consultation and handled precommit failures return a non-passing report;
    other exceptions propagate. A failure does not promise rollback of every
    workspace artifact. See doc/agent_contracts/ for the publication contracts."""
    recipe: dict | None = None
    build_lock: AbstractContextManager
    if validate_only:
        # Validate-only leaves the recipe load to preflight_plan, under the publication lock.
        build_lock = plan_publication_lock(Path(output_dir))
    else:
        recipe = load_recipe_with_base(recipe_path)
        root = experiment_root(recipe)
        build_lock = plan_registration_lock(root) if root is not None else nullcontext()
    request = PlanBuildRequest(
        recipe_path=recipe_path,
        output_dir=output_dir,
        user_decisions_path=user_decisions_path,
        allow_unresolved=allow_unresolved,
        unlock_final_test=unlock_final_test,
        validate_only=validate_only,
    )
    with build_lock:
        return _build_plan(request=request, loaded_recipe=recipe)


def _prepare_plan_staging(*, out: Path, plan_adapter: TaskAdapter | None) -> tuple[Path, bool]:
    if plan_adapter is None:
        return out, False
    staging_parent = out.parent
    if os.path.lexists(out) and out.lstat().st_dev != out.parent.lstat().st_dev:
        # A plan may itself be a mount point, so its parent is not always the destination filesystem.
        staging_parent = out
    while not os.path.lexists(staging_parent):
        staging_parent = staging_parent.parent
    return Path(tempfile.mkdtemp(prefix=f".{out.name}.", suffix=".staging", dir=staging_parent)), True


def _build_plan(*, request: PlanBuildRequest, loaded_recipe: dict | None) -> DecisionReport:
    out = canonical_local_experiment_root(request.output_dir, Path.cwd())
    preflight_recipe, cfg, report = preflight_plan(
        recipe_path=request.recipe_path,
        output_dir=out,
        user_decisions_path=request.user_decisions_path,
        unlock_final_test=request.unlock_final_test,
        recipe=loaded_recipe,
    )
    bound_config = _validate_bound_recipe(cfg, report)
    plan_adapter = get_adapter(preflight_recipe.get("task"))
    if _has_output_artifact_issue(report):
        return report
    if request.validate_only and not (plan_adapter is not None and plan_adapter.materializes_plan):
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
    if report.exit_code != 0 or bound_config is None:
        if request.validate_only:
            return report
        preflight_failed_before_workspace = any(
            issue.field in {"experiment", "step", "execution.workdir"}
            or issue.field.startswith("experiment.")
            or issue.field.startswith("step.")
            or issue.evidence.get("preflight_before_workspace") is True
            for issue in report.blocking_issues()
        )
        if preflight_failed_before_workspace:
            return report
        with plan_publication_lock(out):
            workspace, _step_dir = ensure_experiment_workspace(preflight_recipe, out, register_step=False)
            _write_questions(out, report)
            template = write_user_decision_template(out, preflight_recipe, report, preserve_existing=False)
            template_path = template[0] if template is not None else None
            if template_path is not None:
                report.published_user_decisions_path = str(template_path)
            write_text(
                out / "plan.blocked.md",
                context.blocked_plan_markdown(report, request.allow_unresolved, user_decisions_path=template_path),
            )
            if request.allow_unresolved and report.exit_code == 2:
                write_json(
                    out / "plan.draft.json",
                    {"status": report.status.value, "recipe": preflight_recipe, "questions": questions_payload(report)},
                )
            if not artifacts.is_registered_blocked_plan(out, workspace=workspace):
                raise ValueError(f"Blocked plan publication did not produce a complete control bundle: {out}")
            # step.yaml is canonical ownership; expose the plan only after its blocked bundle is complete.
            ensure_experiment_workspace(preflight_recipe, out)
        return report

    # A bound config exists only when evaluate_recipe reached the config consultation past its freeze
    # check, so this recipe is the JSON document the writers below take.
    recipe: dict[str, JsonValue] = preflight_recipe
    validated_config_bytes, validated_config_sha256, final_eval_config = bound_config
    root = experiment_root(recipe)
    if root is None:
        raise ValueError("experiment.root is required.")
    # experiment_root resolved a root, so experiment is a mapping.
    experiment = recipe.get("experiment")
    if isinstance(experiment, dict):
        experiment["root"] = str(root)
    input_snapshots: list[JsonValue] = []
    if plan_adapter is not None:
        try:
            input_snapshots = [
                {"field": field, "path": str(path), "sha256": file_sha256(path)}
                for field, path in plan_adapter.frozen_input_paths(recipe)
            ]
        except OSError as exc:
            return _append_issues(
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
    # Preflight already resolved inputs.config as a path, so a passing recipe holds it as a string.
    inputs = recipe.get("inputs")
    source_config = inputs.get("config") if isinstance(inputs, dict) else None
    source_config_path = resolve_repo_path(source_config if isinstance(source_config, str) else None)
    if source_config_path is None:
        raise ValueError("Successful plan preflight did not bind the source config path.")
    recipe["input_snapshots"] = input_snapshots
    plan_contract.bind_frozen_input_snapshot(
        recipe,
        "inputs.config",
        source_config_path,
        validated_config_sha256,
    )

    ensure_experiment_workspace(recipe, out, validate_only=True)
    if not request.validate_only:
        _assert_no_incomplete_step_registration(recipe, out)
    run_index_offset = _registered_plan_run_index(recipe, out)

    write_out, generated_staging = _prepare_plan_staging(out=out, plan_adapter=plan_adapter)

    if plan_adapter is not None and plan_adapter.materializes_plan:
        return _materialize_hparam_plan(
            recipe=recipe,
            report=report,
            out=out,
            write_out=write_out,
            generated_staging=generated_staging,
            run_index_offset=run_index_offset,
            validate_only=request.validate_only,
            unlock_final_test=request.unlock_final_test,
            validated_config_bytes=validated_config_bytes,
            validated_config_sha256=validated_config_sha256,
            final_eval_config=final_eval_config,
        )
    return _materialize_single_run_plan(
        run_adapter=plan_adapter,
        recipe=recipe,
        report=report,
        out=out,
        write_out=write_out,
        generated_staging=generated_staging,
        run_index_offset=run_index_offset,
        validated_config_bytes=validated_config_bytes,
    )


def _validate_published_pass_envelope(out: Path) -> None:
    blocked_paths = plan_contract.blocked_plan_control_paths(out)
    exp_io.validate_managed_output_paths(Path(out.anchor), blocked_paths)
    blocked = sorted(str(path) for path in blocked_paths if os.path.lexists(path))
    if blocked:
        raise ValueError(f"Published PASS plan contains blocked planning artifacts: {', '.join(blocked)}")


def publish_staged_plan_locked(
    write_out: Path,
    out: Path,
    *,
    out_preexisted: bool,
    replace_unowned_plan: bool = False,
) -> None:
    """Move a staged plan into its published location under a caller-held lock.

    The caller must hold plan_publication_lock(out), prepare and validate the
    staged files in write_out for their intended location at out, and determine
    out_preexisted under that lock. This function checks the published envelope,
    but does not register workspace ownership or run rows; the caller owns those
    steps and their failure handling.

    A new destination receives the entire staging directory. For an existing
    destination, plan-level files are replaced and run directories are appended
    without collisions, with plan.json published last. replace_unowned_plan
    permits replacing the existing runs tree when staging contains runs; the
    caller must first establish that the old plan is unowned. Successful
    publication consumes write_out.

    If envelope validation fails after publishing a new directory, that directory
    remains in place. Failures during the existing-directory replacement attempt
    to restore moved entries before propagating; rollback and final cleanup can
    also fail. This is not an all-or-nothing workspace transaction. Validation
    and filesystem errors propagate.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    if not out_preexisted:
        write_out.replace(out)
        # A post-publish conflict may contain user bytes; leave it unregistered instead of deleting it via rollback.
        _validate_published_pass_envelope(out)
        return
    if out.is_symlink() or not out.is_dir():
        raise ValueError(f"Atomic plan output is not a directory: {out}")

    staged_runs = write_out / "runs"
    source_names = {path.name for path in write_out.iterdir()}
    run_names = []
    if "runs" in source_names:
        if staged_runs.is_symlink() or not staged_runs.is_dir():
            raise ValueError(f"Staged plan runs path is not a physical directory: {staged_runs}")
        run_names = sorted(path.name for path in staged_runs.iterdir())
        destination_runs = out / "runs"
        if os.path.lexists(destination_runs):
            if destination_runs.is_symlink() or not destination_runs.is_dir():
                raise ValueError(f"Published plan runs path is not a physical directory: {destination_runs}")
            collisions = [name for name in run_names if os.path.lexists(destination_runs / name)]
            if collisions and not replace_unowned_plan:
                raise ValueError(f"Published run directories are immutable: {', '.join(collisions)}")

    backup_parent = out if write_out.parent == out else out.parent
    backup = Path(tempfile.mkdtemp(prefix=f".{out.name}.", suffix=".backup", dir=backup_parent))
    plan_source_names = source_names - {"runs"}
    replaced_names = set(plan_source_names)
    if replace_unowned_plan and "runs" in source_names:
        replaced_names.add("runs")
    for optional_name in ("final_external_test.sh", "config.final_eval.yaml"):
        if optional_name not in plan_source_names:
            replaced_names.add(optional_name)
    old_order = ["plan.json", *sorted(replaced_names - {"plan.json"})]
    new_order = [*sorted(plan_source_names - {"plan.json"}), "plan.json"]
    moved_old = []
    moved_new = []
    moved_run_names = []
    moved_runs_dir = False
    try:
        # Hide the old manifest while plan-owned top-level entries change; restore it last on failure.
        for name in old_order:
            current = out / name
            if os.path.lexists(current):
                current.replace(backup / name)
                moved_old.append(name)
        for name in new_order[:-1]:
            (write_out / name).replace(out / name)
            moved_new.append(name)
        # Committed run directories are append-only; republication replaces only plan-level files.
        if "runs" in source_names:
            destination_runs = out / "runs"
            if os.path.lexists(destination_runs):
                for name in run_names:
                    (staged_runs / name).replace(destination_runs / name)
                    moved_run_names.append(name)
                staged_runs.rmdir()
            else:
                staged_runs.replace(destination_runs)
                moved_runs_dir = True
        (write_out / "plan.json").replace(out / "plan.json")
        moved_new.append("plan.json")
        _validate_published_pass_envelope(out)
    except BaseException:
        for name in reversed(moved_new):
            current = out / name
            if os.path.lexists(current):
                current.replace(write_out / name)
        if moved_runs_dir:
            (out / "runs").replace(staged_runs)
        elif moved_run_names:
            staged_runs.mkdir(exist_ok=True)
            for name in reversed(moved_run_names):
                (out / "runs" / name).replace(staged_runs / name)
        for name in reversed(moved_old):
            (backup / name).replace(out / name)
        shutil.rmtree(backup)
        raise
    shutil.rmtree(backup)
    write_out.rmdir()


def preflight_plan(
    *,
    recipe_path: str | Path,
    output_dir: str | Path | None,
    user_decisions_path: str | Path | None = None,
    unlock_final_test: bool = False,
    recipe: dict | None = None,
) -> tuple[dict, ConfigSummaryInput | None, DecisionReport]:
    """Run plan validation once, without publishing anything.

    Returns (recipe, optional config summary, DecisionReport), retaining
    evaluate_recipe's early-blocked materialization limits and adding
    freezeability, task preflight and command checks. With an output_dir it also
    checks the experiment workspace and the plan output; output_dir=None is
    doctor's publication-disabled path through the same validation. recipe is
    the recipe already loaded from recipe_path (build_plan loads it once to pick
    its lock); None loads it here. Reads the inputs and existing artifacts needed
    for these checks; launches no runs. Expected blockers are reported, while
    unhandled loading/I/O errors propagate.

    unlock_final_test is checked against the frozen evaluation policy, not a
    substitute for it. build_plan runs this under its registration lock and
    publishes from this single validation pass.

    The recipe keeps evaluate_recipe's broad type: contract-blocked early
    returns precede its freeze check, so only a passing report establishes the
    JSON shape, and _build_plan narrows the recipe to dict[str, JsonValue] on
    its pass path."""
    recipe, cfg, report = _evaluate_loaded_recipe(
        load_recipe_with_base(recipe_path) if recipe is None else recipe,
        recipe_path,
        user_decisions_path,
        check_existing_experiment=True,
    )
    # Static blockers, including NEEDS_USER_INPUT, stop before the config, task and output checks.
    if cfg is None and any(
        issue.evidence.get("preflight_before_workspace") is True for issue in report.blocking_issues()
    ):
        return recipe, cfg, report
    out = None if output_dir is None else canonical_local_experiment_root(output_dir, Path.cwd())
    metadata_unresolved = any(
        issue.field in {"experiment", "step"}
        or issue.field.startswith("experiment.")
        or issue.field.startswith("step.")
        for issue in report.blocking_issues()
    )
    if out is not None and not metadata_unresolved:
        workspace_issue = validate_plan_output(
            recipe,
            out,
            allow_published_plan=_is_unowned_published_plan(recipe, out),
        )
        if workspace_issue:
            report = _append_issues(
                report,
                [DecisionIssue(DecisionStatus.FAIL, "experiment.root", workspace_issue, None, {})],
            )
    inputs_value = recipe.get("inputs")
    inputs = inputs_value if isinstance(inputs_value, dict) else {}
    source_config = inputs.get("config")
    source_path = resolve_repo_path(source_config) if source_config not in (None, "", "ASK_USER") else None
    # A stat, not a read: the config consultation already read and bound the only frozen bytes.
    if source_config not in (None, "", "ASK_USER") and not (source_path is not None and source_path.is_file()):
        blocking_config_issue = next(
            (issue for issue in report.blocking_issues() if issue.field == "config"),
            None,
        )
        if blocking_config_issue is not None and blocking_config_issue.status == DecisionStatus.FAIL:
            # Keep the specific config failure (e.g. a missing file); it already blocks before the workspace.
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
    if preflight_adapter is not None and report.exit_code == 0:
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
    if report.exit_code == 0:
        plan_contract.bind_plan_context(recipe)
    if out is None:
        return recipe, cfg, report
    if report.exit_code == 0:
        report = _guard_pass_plan_publication(
            report,
            recipe,
            out,
            unlock_final_test=unlock_final_test,
        )
        root = experiment_root(recipe)
        if root is not None:
            report = _guard_existing_outputs(
                report,
                [root / "events.jsonl"],
                root=root,
                allow_existing=True,
            )
    else:
        report = _guard_blocked_plan_publication(
            report,
            recipe,
            out,
            unlock_final_test=unlock_final_test,
        )
    return recipe, cfg, report


def _commands_for_recipe(recipe: dict, cfg: ConfigSummaryInput | None = None) -> list[str]:
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


def _registered_plan_owners(recipe: dict[str, Any], out: Path) -> list[dict[str, Any]]:
    root = experiment_root(recipe)
    experiment_value = recipe.get("experiment")
    experiment = experiment_value if isinstance(experiment_value, dict) else {}
    experiment_id = str(experiment.get("id") or "")
    if root is None or not experiment_id or not exp_io.path_exists_at(root / "steps"):
        return []
    return [
        manifest
        for manifest in read_registered_steps(root, experiment_id=experiment_id)
        if str(out) in manifest["plans"]
    ]


def _is_unowned_published_plan(recipe: dict[str, Any], out: Path) -> bool:
    return exp_io.path_exists_at(out / "plan.json") and not _registered_plan_owners(recipe, out)


def _registered_plan_run_index(recipe: dict[str, Any], out: Path) -> int | None:
    owners = _registered_plan_owners(recipe, out)
    step_id = str((recipe.get("step") or {}).get("id") or "")
    if len(owners) != 1 or str((owners[0].get("step") or {}).get("id") or "") != step_id:
        return None
    root = experiment_root(recipe)
    assert root is not None
    plan_path = out / "plan.json"
    snapshot = exp_io.read_managed_files_at(root, [plan_path])[str(plan_path)]
    plan = json.loads(snapshot["text"])
    runs = plan.get("runs") if isinstance(plan, dict) else None
    first_run_id = str(runs[0].get("run_id") or "") if isinstance(runs, list) and runs else ""
    match = re.fullmatch(r"run-(\d+)", first_run_id)
    if match is None:
        raise ValueError(f"Registered plan has an invalid first run id: {plan_path}")
    return int(match.group(1))


def _assert_no_incomplete_step_registration(recipe: dict[str, Any], out: Path) -> None:
    root = experiment_root(recipe)
    if root is None:
        raise ValueError("experiment.root is required.")
    if not exp_io.path_exists_at(root / "steps"):
        return
    step_id = str((recipe.get("step") or {}).get("id") or "")
    step_manifest = read_step_manifest(root, step_id, allow_missing=True)
    if step_manifest is None:
        return
    for registered_path in step_manifest["plans"]:
        plan_dir = Path(registered_path)
        if plan_dir == out or artifacts.is_registered_blocked_plan(plan_dir, workspace=root):
            continue
        plan_path = plan_dir / "plan.json"
        snapshot = exp_io.read_managed_files_at(root, [plan_path])[str(plan_path)]
        plan = json.loads(snapshot["text"])
        frozen_recipe = plan.get("recipe") if isinstance(plan, dict) else None
        runs = plan.get("runs") if isinstance(plan, dict) else None
        if not isinstance(frozen_recipe, dict) or not isinstance(runs, list):
            raise ValueError(f"Registered plan is incomplete: {plan_dir}")
        adapter = get_adapter(frozen_recipe.get("task"))
        expected_rows = adapter.registration_rows(plan) if adapter is not None else runs
        state = _plan_registration_state(
            frozen_recipe,
            plan_dir,
            expected_rows,
            expected_tree_sha256=None,
            expected_tree_entries=None,
        )
        if state != "complete":
            raise ValueError(
                f"Step has an interrupted registered plan; recover it before creating another plan: {plan_dir}"
            )


def _plan_registration_state(
    recipe: dict[str, Any],
    out: Path,
    expected_rows: list[dict[str, Any]],
    *,
    expected_tree_sha256: str | None,
    expected_tree_entries: frozenset[str] | None,
) -> str:
    root = experiment_root(recipe)
    if root is None:
        raise ValueError("experiment.root is required.")
    row_state = plan_registration_rows_state(root, expected_rows, source="Canonical plan")
    event_payload = {
        "step_id": (recipe.get("step") or {}).get("id"),
        "plan_dir": str(out),
        "run_count": len(expected_rows),
    }
    related_events = [
        event
        for event in read_experiment_events(root)
        if event.get("event_type") == "plan_created" and event.get("plan_dir") == str(out)
    ]
    owners = _registered_plan_owners(recipe, out)
    if not owners:
        if related_events:
            raise ValueError(f"Plan-created event exists before plan ownership registration: {out}")
        if row_state == "present":
            raise ValueError(f"Canonical run registration exists before plan ownership registration: {out}")
        if exp_io.path_exists_at(out / "plan.json"):
            ensure_experiment_workspace(
                recipe,
                out,
                validate_only=True,
                allow_published_plan=True,
            )
            tree_matches = (
                expected_tree_sha256 is not None
                and artifacts.plan_tree_sha256(out, top_level_entries=expected_tree_entries) == expected_tree_sha256
            )
            if tree_matches:
                return "owner_missing"
            raise ValueError(f"Published plan differs from deterministic regeneration: {out}")
        return "unregistered"
    step_id = str((recipe.get("step") or {}).get("id") or "")
    if len(owners) != 1 or str((owners[0].get("step") or {}).get("id") or "") != step_id:
        raise ValueError(f"Plan directory has conflicting canonical owners: {out}")
    ensure_experiment_workspace(
        recipe,
        out,
        validate_only=True,
        allow_published_plan=True,
    )
    if (
        expected_tree_sha256 is not None
        and artifacts.plan_tree_sha256(out, top_level_entries=expected_tree_entries) != expected_tree_sha256
    ):
        raise ValueError(f"Registered plan differs from deterministic regeneration: {out}")

    exact_events = [event for event in related_events if event_matches(event, "plan_created", event_payload)]
    if len(related_events) != len(exact_events) or len(exact_events) > 1:
        raise ValueError(f"Plan-created event conflicts with canonical registration: {out}")
    if row_state == "missing":
        if exact_events:
            raise ValueError(f"Plan-created event exists before canonical run registration: {out}")
        return "rows_missing"
    return "complete" if exact_events else "event_missing"


def _registered_plan_immutable_report(report: DecisionReport, out: Path) -> DecisionReport:
    return _append_issues(
        report,
        [
            DecisionIssue(
                DecisionStatus.FAIL,
                "output_artifacts",
                "Registered plan directories are immutable; retry with a fresh --output-dir.",
                None,
                {"existing_paths": [str(out)]},
            )
        ],
    )


def _guard_existing_outputs(
    report: DecisionReport,
    paths: list[Path],
    *,
    root: Path | None,
    allow_existing: bool = False,
    require_fresh: str | None = None,
) -> DecisionReport:
    """Fail when a planned output already exists; outputs are single-use.

    root=None marks outputs outside any managed workspace, which skip the managed-path check.
    """
    existing = (
        sorted(str(path) for path in paths if os.path.lexists(path))
        if require_fresh is not None and not allow_existing
        else []
    )
    if not existing:
        try:
            if root is not None:
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
    return _append_issues(
        report,
        [
            DecisionIssue(
                DecisionStatus.FAIL,
                "output_artifacts",
                require_fresh or "Output artifacts already exist; retry with a fresh --output-dir.",
                None,
                {"existing_paths": existing},
            )
        ],
    )


def _guard_pass_plan_publication(
    report: DecisionReport,
    recipe: dict,
    out: Path,
    *,
    unlock_final_test: bool,
) -> DecisionReport:
    owners = _registered_plan_owners(recipe, out)
    step_id = str((recipe.get("step") or {}).get("id") or "")
    owned_by_current_step = len(owners) == 1 and str((owners[0].get("step") or {}).get("id") or "") == step_id
    recovering_unowned_plan = _is_unowned_published_plan(recipe, out)
    if owners and not owned_by_current_step:
        report = _registered_plan_immutable_report(report, out)
    planned_paths = _planned_plan_paths(recipe, out, report, unlock_final_test)
    report = _guard_existing_outputs(
        report,
        planned_paths,
        root=out,
        allow_existing=owned_by_current_step or recovering_unowned_plan,
    )
    adapter = get_adapter(recipe.get("task"))
    assert adapter is not None
    report = _guard_existing_outputs(
        report,
        adapter.external_output_paths(recipe),
        root=None,
        allow_existing=owned_by_current_step or recovering_unowned_plan,
        require_fresh="Task output files already exist; these outputs are single-use, so choose fresh output paths.",
    )
    # Blocked bundles are human-editable evidence; reusing one would mix PASS and blocked envelopes.
    return _guard_existing_outputs(
        report,
        plan_contract.blocked_plan_control_paths(out),
        root=out,
        require_fresh="Blocked plan artifacts already exist; retry with a fresh --output-dir.",
    )


def _guard_blocked_plan_publication(
    report: DecisionReport,
    recipe: dict,
    out: Path,
    *,
    unlock_final_test: bool,
) -> DecisionReport:
    planned_paths = _planned_plan_paths(recipe, out, report, unlock_final_test)
    blocked_paths = plan_contract.blocked_plan_control_paths(out)
    report = _guard_existing_outputs(
        report,
        blocked_paths,
        root=out,
        require_fresh="Blocked plan artifacts already exist; retry with a fresh --output-dir.",
    )
    non_control_paths = [path for path in planned_paths if path not in blocked_paths]
    if non_control_paths:
        report = _guard_existing_outputs(
            report,
            non_control_paths,
            root=out,
        )
    root_resident = out == experiment_root(recipe)
    pass_plan_paths = (
        plan_contract.pass_plan_artifact_paths(out) if root_resident else plan_contract.pass_plan_control_paths(out)
    )
    if root_resident:
        pass_plan_paths.extend(_plan_publication_temporary_paths(out))
    report = _guard_existing_outputs(
        report,
        pass_plan_paths,
        root=out,
        require_fresh="PASS plan artifacts already exist; retry with a fresh --output-dir.",
    )
    if _has_output_artifact_issue(report) or not out.is_dir() or root_resident:
        return report
    allowed_names = {path.name for path in plan_contract.blocked_plan_control_paths(out)}
    unexpected = sorted(str(path) for path in out.iterdir() if path.name not in allowed_names)
    if not unexpected:
        return report
    # Blocked-plan readers enforce an exact nested envelope, so reject foreign entries before registration.
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
    unlock_final_test: bool,
) -> list[Path]:
    adapter = get_adapter(recipe.get("task"))
    if adapter is not None and adapter.materializes_plan:
        return plan_hparam.hparam_planned_plan_paths(
            recipe, out, blocked=report.exit_code != 0, unlock_final_test=unlock_final_test
        )
    if report.exit_code != 0:
        return plan_contract.blocked_plan_control_paths(out)
    assert adapter is not None
    run = plan_contract.generic_run_contract(recipe, out, next_run_index(recipe), adapter)
    run_dir = Path(run["run_dir"])
    paths = [
        out / "plan.json",
        out / "plan.md",
        out / "run.sh",
        out / "recipe.resolved.yaml",
        run_dir / "run.json",
        run_dir / "config.yaml",
        run_dir / "launch.sh",
        run_dir / "artifacts.json",
    ]
    if run.get("scheduler_script"):
        paths.append(Path(run["scheduler_script"]))
    return paths
