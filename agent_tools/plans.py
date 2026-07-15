from __future__ import annotations

import copy
import csv
from pathlib import Path
from typing import Any

import yaml

from . import (
    decision_hparam as hparam_rules,
    decision_paths as path_rules,
    decision_rules as task_rules,
    experiment_io as exp_io,
    plan_context as context,
    plan_hparam as hparam,
    plan_rendering as rendering,
    repo as repo_tools,
    run_artifacts as artifacts,
)
from .adapters import get_adapter
from .configs import config_summary
from .decisions import (
    DecisionIssue,
    DecisionReport,
    DecisionStatus,
    consultation_contract_issues,
    evaluate_consultation_gates,
    merge_status,
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
    run_identity,
    safe_artifact_name,
    validate_plan_output,
)
from .manifests import read_json, write_json, write_text
from .markdown import questions_markdown, questions_payload
from .models import CONFIG_FINETUNE_SECTION, REPO_ROOT, resolve_repo_path
from .recipes import load_consultation_policy, load_recipe_with_base, load_user_decisions, recipe_name

_COMMON_RECIPE_FIELDS = {"decisions", "experiment", "name", "step", "task", "variant"}
_TASK_RECIPE_FIELDS = {
    "finetune": _COMMON_RECIPE_FIELDS | {"artifacts", "evaluation_policy", "execution", "inputs", "runtime"},
    "hparam_tune": _COMMON_RECIPE_FIELDS
    | {"adaptive", "artifacts", "base_recipe", "evaluation_policy", "execution", "inputs", "runtime", "search"},
}
_ARTIFACT_FIELDS = {
    "finetune": {"overwrite", "results_csv_path", "version_name"},
    "hparam_tune": {"overwrite", "results_csv_path"},
}


def _recipe_fields_for_task(task: str) -> set[str] | None:
    adapter = get_adapter(task)
    if adapter is not None:
        return _COMMON_RECIPE_FIELDS | adapter.recipe_extra_fields
    return _TASK_RECIPE_FIELDS.get(task)


def _artifact_fields_for_task(task: str) -> set[str]:
    adapter = get_adapter(task)
    if adapter is not None:
        return set(adapter.artifact_fields)
    return _ARTIFACT_FIELDS.get(task, set())


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
        local_contract_task = "hparam_tune" if local_task in (None, "", "ASK_USER") else str(local_task)
        sources = [
            (base_recipe, hparam_rules.HPARAM_BASE_TASK, "base"),
            (local_recipe, local_contract_task, "local"),
        ]
    else:
        sources = [(recipe, str(effective_task or ""), "effective")]

    if has_layers and base_recipe.get("task") not in (None, "", "ASK_USER", hparam_rules.HPARAM_BASE_TASK):
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
    allowed_top_level = _recipe_fields_for_task(task)
    if allowed_top_level is None:
        return []
    issues = [
        _recipe_contract_issue(
            str(field),
            f"Unknown recipe field for task={task or 'unresolved'}: {field}.",
            recipe[field],
            source_layer,
        )
        for field in sorted(set(recipe) - allowed_top_level)
        if not str(field).startswith("_")
    ]
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
    issues.extend(consultation_contract_issues(task or None, recipe, policy, source_layer=source_layer))
    if task == "hparam_tune":
        issues.extend(hparam_rules.hparam_recipe_contract_issues(recipe, source_layer=source_layer))
    else:
        issues.extend(task_rules.task_recipe_contract_issues(task, recipe, source_layer=source_layer))
        issues.extend(path_rules.execution_contract_issues(recipe, source_layer=source_layer))
    issues.extend(_artifact_contract_issues(task, recipe, source_layer))
    return issues


def _artifact_contract_issues(task: str, recipe: dict, source_layer: str) -> list[DecisionIssue]:
    if "artifacts" not in recipe:
        return []
    artifacts_value = recipe["artifacts"]
    if not isinstance(artifacts_value, dict):
        return [
            _recipe_contract_issue(
                "artifacts",
                "artifacts must be a mapping.",
                artifacts_value,
                source_layer,
            )
        ]
    allowed_fields = _artifact_fields_for_task(task)
    return [
        _recipe_contract_issue(
            f"artifacts.{field}",
            f"Unknown artifacts field for task={task}: {field}.",
            artifacts_value[field],
            source_layer,
        )
        for field in sorted(set(artifacts_value) - allowed_fields)
    ]


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

    canonical_fields = {
        "label_name": ("inputs", "label_name"),
        "data_backend": ("inputs", "data_backend"),
        "external_test_locked": ("evaluation_policy", "external_test_locked"),
        "selection_metric": ("evaluation_policy", "selection_metric"),
        "selection_mode": ("evaluation_policy", "selection_mode"),
        "pretrained_backbone_path": ("inputs", "pretrained_backbone_path"),
        "config": ("inputs", "config"),
        "ckpt_path": ("inputs", "ckpt_path"),
        "eval_split": ("inputs", "eval_split"),
        "final_eval_config_path": ("inputs", "final_eval_config_path"),
        "min_channels": ("preset", "min_channels"),
        "hparam_search_space": ("search", "parameters"),
        "hparam_budget": ("search", "max_runs"),
        "final_eval_unlock": ("evaluation_policy", "final_test_unlocked"),
        "test_after_fit": ("evaluation_policy", "test_after_fit"),
    }
    canonical_fields["overwrite_policy"] = ("artifacts", "overwrite")
    adapter = get_adapter(recipe.get("task"))
    if adapter is not None:
        canonical_fields.update(adapter.decision_recipe_targets)
    if decision_values.get("train_val_test_policy") in ("train", "val", "test"):
        canonical_fields["train_val_test_policy"] = ("evaluation_policy", "selection_split")

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


def evaluate_recipe(
    recipe_path: str | Path,
    user_decisions_path: str | Path | None = None,
) -> tuple[dict, dict | None, DecisionReport]:
    recipe = load_recipe_with_base(recipe_path)
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

    config_error = None
    try:
        cfg = context.load_config_summary_for_recipe(recipe)
    except Exception as exc:
        if "config" not in user_decisions:
            raise
        cfg = None
        config_error = str(exc)
    report = evaluate_consultation_gates(
        recipe.get("task"),
        recipe,
        cfg,
        {"user_decisions": user_decisions},
        policy,
    )
    report = _append_issues(report, materialization_issues)
    if (
        recipe.get("task") in {"finetune", "hparam_tune"}
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
    if (
        recipe.get("task") != "hparam_tune"
        and cfg is not None
        and cfg.get("is_finetune") is True
        and not cfg.get("blocking_issues")
    ):
        inputs = recipe.get("inputs") if isinstance(recipe.get("inputs"), dict) else {}
        evaluation = recipe.get("evaluation_policy") if isinstance(recipe.get("evaluation_policy"), dict) else {}
        finetune_task = cfg.get(CONFIG_FINETUNE_SECTION, {}).get("task", {})
        config_contracts = {
            "data_backend": (
                inputs.get("data_backend"),
                cfg.get("data_backend"),
                "data.backend",
            ),
            "selection_metric": (
                evaluation.get("selection_metric"),
                finetune_task.get("monitor"),
                "finetune.task.monitor",
            ),
            "selection_mode": (
                evaluation.get("selection_mode"),
                finetune_task.get("monitor_mod"),
                "finetune.task.monitor_mod",
            ),
        }
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
        recipe.get("task") in {"finetune", "infer", "evaluate", "hparam_tune"} and "config" in user_decisions
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
        blocking_config_issues = cfg.get("blocking_issues", []) if cfg is not None else []
        if config_error or cfg is None or cfg.get("is_finetune") is not True or blocking_config_issues:
            message = config_error or "Selected config must be a readable finetune config without blocking issues."
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
    if (
        recipe.get("task") == "hparam_tune"
        and cfg is not None
        and cfg.get("is_finetune") is True
        and not cfg.get("blocking_issues")
    ):
        report = _append_issues(report, hparam.hparam_yaml_override_issues(recipe))
    return recipe, cfg, report


def write_questions(output_dir: str | Path, report: DecisionReport) -> None:
    out = Path(output_dir)
    write_json(out / "questions.json", {"questions": questions_payload(report)})
    write_text(out / "questions.md", questions_markdown(report))


def prepare_doctor_report(output_dir: str | Path | None, recipe: dict, report: DecisionReport) -> DecisionReport:
    return report


def write_doctor_outputs(output_dir: str | Path | None, recipe: dict, report: DecisionReport) -> None:
    if output_dir is None or _has_output_artifact_issue(report):
        return
    if report.blocking_issues():
        write_questions(output_dir, report)


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
    effective_config = (recipe.get("inputs") or {}).get("config")
    cfg = config_summary(effective_config, variant=variant) if effective_config else None
    report = evaluate_consultation_gates(
        task,
        recipe,
        cfg,
        {"label_name": label_name, "user_decisions": user_decisions},
        policy,
        require_experiment=True,
    )
    report = _append_issues(report, materialization_issues)
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


def build_plan(
    *,
    recipe_path: str | Path,
    output_dir: str | Path,
    user_decisions_path: str | Path | None = None,
    allow_unresolved: bool = False,
    unlock_final_test: bool = False,
) -> DecisionReport:
    out = canonical_local_experiment_root(output_dir, Path.cwd())
    recipe, cfg, report = preflight_plan(
        recipe_path=recipe_path,
        output_dir=out,
        user_decisions_path=user_decisions_path,
        allow_unresolved=allow_unresolved,
        unlock_final_test=unlock_final_test,
    )
    if _has_output_artifact_issue(report):
        return report
    if report.exit_code != 0:
        preflight_failed_before_workspace = bool(experiment_metadata_issues(recipe)) or any(
            issue.field in {"experiment", "step", "execution.workdir"}
            or issue.field.startswith("experiment.")
            or issue.field.startswith("step.")
            or issue.evidence.get("preflight_before_workspace") is True
            for issue in report.blocking_issues()
        )
        if preflight_failed_before_workspace:
            return report
        ensure_experiment_workspace(recipe, out)
        write_questions(out, report)
        write_text(out / "plan.blocked.md", context.blocked_plan_markdown(report, allow_unresolved))
        if allow_unresolved and report.exit_code == 2:
            write_json(
                out / "plan.draft.json",
                {"status": report.status.value, "recipe": recipe, "questions": questions_payload(report)},
            )
        return report

    ensure_experiment_workspace(recipe, out)

    task = recipe.get("task")
    if task == "hparam_tune":
        hparam.write_hparam_plan(recipe, out, unlock_final_test=unlock_final_test)
    else:
        root = experiment_root(recipe)
        if root is None:
            raise ValueError("experiment.root is required.")
        declared_name = safe_artifact_name((recipe.get("artifacts") or {}).get("version_name") or recipe_name(recipe))
        identity = run_identity(recipe, next_run_index(recipe), {}, run_name=declared_name)
        run_id = identity["run_id"]
        run_name = identity["run_name"]
        version = identity["version"]
        run_dir = out / "runs" / f"{run_id}--{run_name}"
        run_dir.mkdir(parents=True, exist_ok=True)
        config_path = run_dir / "config.yaml"
        source_config = (recipe.get("inputs") or {}).get("config")
        source_path = resolve_repo_path(source_config)
        if source_path is None or not source_path.exists():
            raise ValueError(f"Cannot freeze missing config: {source_config}")
        config_path.write_text(source_path.read_text())
        runtime_recipe = copy.deepcopy(recipe)
        runtime_recipe.setdefault("inputs", {})["config"] = str(config_path)
        runtime_recipe.setdefault("artifacts", {})["version_name"] = version
        runtime_cfg = config_summary(config_path, variant=recipe.get("variant"))
        commands = _commands_for_recipe(runtime_recipe, runtime_cfg)
        runtime_dir = REPO_ROOT / "log-finetune" / version if task == "finetune" else None
        checkpoint_dir = runtime_dir / "checkpoints" if runtime_dir is not None else None
        artifacts_path = run_dir / "artifacts.json"
        run = {
            "experiment_id": (recipe.get("experiment") or {}).get("id"),
            "step_id": (recipe.get("step") or {}).get("id"),
            "run_id": run_id,
            "run_name": run_name,
            "version": version,
            "status": "planned",
            "config": str(config_path),
            "config_sha256": file_sha256(config_path),
            "script": str(run_dir / "launch.sh"),
            "run_dir": str(run_dir),
            "artifacts": str(artifacts_path),
            "runtime_dir": str(runtime_dir) if runtime_dir is not None else "",
            "checkpoint_dir": str(checkpoint_dir) if checkpoint_dir is not None else "",
        }
        write_text(out / "plan.md", context.plan_markdown(report, commands))
        write_text(
            out / "run.sh",
            "\n".join(
                rendering.script_lines(
                    commands,
                    run_cwd=REPO_ROOT,
                    experiment_root=root,
                    step_id=run["step_id"],
                    run_id=run_id,
                )
            )
            + "\n",
            executable=True,
        )
        launch_path = run_dir / "launch.sh"
        write_text(launch_path, (out / "run.sh").read_text(), executable=True)
        run["script_sha256"] = file_sha256(launch_path)
        artifact_payload = {
            "declared": recipe.get("artifacts") or {},
            "runtime_dir": run["runtime_dir"],
            "checkpoint_dir": run["checkpoint_dir"],
            "external_artifacts": True,
        }
        write_json(
            artifacts_path,
            artifact_payload,
        )
        write_json(run_dir / "run.json", {**run, "commands": commands})
        write_json(
            out / "plan.json",
            {"status": report.status.value, "commands": commands, "runs": [run], "recipe": recipe},
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
        resolved_recipe = {key: value for key, value in recipe.items() if not str(key).startswith("_")}
        (out / "recipe.resolved.yaml").write_text(yaml.safe_dump(resolved_recipe, sort_keys=False))
    return report


def preflight_plan(
    *,
    recipe_path: str | Path,
    output_dir: str | Path,
    user_decisions_path: str | Path | None = None,
    allow_unresolved: bool = False,
    unlock_final_test: bool = False,
) -> tuple[dict, dict | None, DecisionReport]:
    recipe, cfg, report = evaluate_recipe(recipe_path, user_decisions_path)
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
    if report.exit_code == 0 and recipe.get("task") == "hparam_tune":
        report = _append_issues(
            report,
            hparam.final_test_checkpoint_issues(recipe, unlock_final_test=unlock_final_test),
        )
    if report.exit_code == 0 and recipe.get("task") != "hparam_tune":
        commands = _commands_for_recipe(recipe, cfg)
        if not commands:
            report = _unsupported_command_report(report, str(recipe.get("task")))
    successful_plan = report.exit_code == 0
    report = _guard_existing_outputs(
        report,
        _planned_plan_paths(recipe, out, report, allow_unresolved, unlock_final_test),
        _overwrite_policy(recipe),
        root=out,
    )
    if successful_plan:
        root = experiment_root(recipe)
        if root is not None:
            report = _guard_existing_outputs(
                report,
                [root / "run_matrix.csv", root / "reports" / "run_matrix.md", root / "events.jsonl"],
                _overwrite_policy(recipe),
                root=root,
                allow_existing=True,
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
    inputs = recipe.get("inputs") if isinstance(recipe.get("inputs"), dict) else {}
    runtime = recipe.get("runtime") if isinstance(recipe.get("runtime"), dict) else {}
    artifacts = recipe.get("artifacts") if isinstance(recipe.get("artifacts"), dict) else {}
    evaluation = recipe.get("evaluation_policy") if isinstance(recipe.get("evaluation_policy"), dict) else {}
    if task == "finetune":
        test_after_fit = evaluation.get("test_after_fit")
        pieces = [
            "python",
            "-m",
            rendering.variant_module(recipe, "finetune"),
            "--config",
            inputs.get("config"),
            "--label-name",
            inputs.get("label_name"),
            "--version-name",
            artifacts.get("version_name", recipe_name(recipe)),
            "--results-csv-path",
            artifacts.get("results_csv_path", "results/agent_results.csv"),
            *rendering.runtime_cli_args(runtime, variant=str(recipe.get("variant"))),
            *rendering.finetune_input_cli_args(
                inputs,
                variant=str(recipe.get("variant")),
            ),
        ]
        if test_after_fit is False or evaluation.get("external_test_locked") is True:
            pieces.append("--no-test-after-fit")
        return [rendering.render_command(pieces)]
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
    section, key = "artifacts", "overwrite"
    adapter = get_adapter(recipe.get("task"))
    if adapter is not None:
        section, key = adapter.decision_recipe_targets.get("overwrite_policy", (section, key))
    owner = recipe.get(section) if isinstance(recipe.get(section), dict) else {}
    return owner.get(key)


def _guard_existing_outputs(
    report: DecisionReport,
    paths: list[Path],
    overwrite_policy: Any,
    *,
    root: Path,
    allow_existing: bool = False,
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
    if overwrite_policy is True:
        return report
    if overwrite_policy is False:
        status = DecisionStatus.FAIL
        message = "Output artifacts already exist and overwrite_policy=false."
        question = None
    else:
        status = DecisionStatus.NEEDS_USER_INPUT
        message = "Output artifacts already exist and overwrite policy is not explicit."
        question = "Is overwriting existing agent-generated output files allowed for this task?"
    return _append_issues(
        report,
        [
            DecisionIssue(
                status,
                "output_artifacts",
                message,
                question,
                {"existing_paths": existing},
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
    if report.exit_code != 0:
        paths = [out / "questions.json", out / "questions.md", out / "plan.blocked.md"]
        if recipe.get("task") == "hparam_tune":
            evaluation = recipe.get("evaluation_policy") or {}
            if hparam.final_test_unlocked(evaluation, unlock_final_test):
                paths.append(out / "final_external_test.sh")
        if allow_unresolved and report.exit_code == 2:
            paths.append(out / "plan.draft.json")
        return paths
    if recipe.get("task") != "hparam_tune":
        declared_name = safe_artifact_name((recipe.get("artifacts") or {}).get("version_name") or recipe_name(recipe))
        identity = run_identity(recipe, next_run_index(recipe), {}, run_name=declared_name)
        run_dir = out / "runs" / f"{identity['run_id']}--{identity['run_name']}"
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

    paths = [
        out / "plan.json",
        out / "plan.md",
        out / "run_all.sh",
        out / "validation.sh",
        out / "recipe.resolved.yaml",
    ]
    offset = next_run_index(recipe)
    for idx, combo in enumerate(hparam.hparam_combos(recipe)):
        identity = run_identity(recipe, offset + idx, combo)
        run_dir = out / "runs" / f"{identity['run_id']}--{identity['run_name']}"
        paths.extend([run_dir / "launch.sh", run_dir / "config.yaml", run_dir / "run.json", run_dir / "artifacts.json"])
    evaluation = recipe.get("evaluation_policy") or {}
    paths.append(out / "final_external_test.sh")
    return paths
