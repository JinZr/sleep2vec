from __future__ import annotations

import copy
import csv
from pathlib import Path
from typing import Any

import yaml

from . import (
    experiment_io as exp_io,
    plan_context as context,
    plan_hparam as hparam,
    plan_rendering as rendering,
    repo as repo_tools,
)
from .configs import config_summary
from .decisions import DecisionIssue, DecisionReport, DecisionStatus, evaluate_consultation_gates, merge_status
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
from .manifests import write_json, write_text
from .markdown import questions_markdown, questions_payload
from .models import REPO_ROOT, resolve_repo_path
from .recipes import load_policy_files, load_recipe_with_base, load_user_decisions, recipe_name


def evaluate_recipe(
    recipe_path: str | Path,
    user_decisions_path: str | Path | None = None,
) -> tuple[dict, dict | None, DecisionReport]:
    recipe = load_recipe_with_base(recipe_path)
    source = resolve_repo_path(recipe_path)
    if source is not None:
        recipe["_recipe_path"] = str(source.resolve())
    policy, defaults = load_policy_files()
    user_decisions = load_user_decisions(user_decisions_path)
    cfg = context.load_config_summary_for_recipe(recipe)
    report = evaluate_consultation_gates(
        recipe.get("task"),
        recipe,
        cfg,
        {"user_decisions": user_decisions},
        policy,
        defaults,
    )
    config_decision = report.decisions.get("config")
    selected_config = (
        recipe.get("task") in {"finetune", "infer", "evaluate", "hparam_tune"}
        and "config" in user_decisions
        and config_decision is not None
    )
    if selected_config and config_decision.value in (None, "", "ASK_USER"):
        report = _append_issues(
            report,
            [
                DecisionIssue(
                    DecisionStatus.NEEDS_USER_INPUT,
                    "config",
                    "Explicit config decision is unresolved.",
                    "Which config should this task use?",
                    {
                        "config": config_decision.value,
                        "preflight_before_workspace": True,
                    },
                )
            ],
        )
    elif selected_config:
        inputs = recipe.get("inputs") if isinstance(recipe.get("inputs"), dict) else {}
        recipe["inputs"] = {**inputs, "config": config_decision.value}
        if recipe.get("task") == "hparam_tune" and isinstance(recipe.get("_base_recipe"), dict):
            base_inputs = (
                recipe["_base_recipe"].get("inputs") if isinstance(recipe["_base_recipe"].get("inputs"), dict) else {}
            )
            recipe["_base_recipe"]["inputs"] = {**base_inputs, "config": config_decision.value}
        config_error = None
        try:
            cfg = context.load_config_summary_for_recipe(recipe)
        except Exception as exc:
            cfg = None
            config_error = str(exc)
        report = evaluate_consultation_gates(
            recipe.get("task"),
            recipe,
            cfg,
            {"user_decisions": user_decisions},
            policy,
            defaults,
        )
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
                            "config": config_decision.value,
                            "preflight_before_workspace": True,
                        },
                    )
                ],
            )
    report = _append_issues(report, context.index_summary_issues(recipe, cfg, report.decisions))
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
    cfg = config_summary(config, variant=variant) if config else None
    policy, defaults = load_policy_files()
    user_decisions = load_user_decisions(user_decisions_path)
    report = evaluate_consultation_gates(
        task,
        recipe,
        cfg,
        {"label_name": label_name, "user_decisions": user_decisions},
        policy,
        defaults,
        require_experiment=True,
    )
    out = Path(output_dir)
    if report.exit_code == 0:
        workspace_issue = validate_plan_output(recipe, out)
        if workspace_issue:
            report = _append_issues(
                report,
                [DecisionIssue(DecisionStatus.FAIL, "experiment.root", workspace_issue, None, {})],
            )
    index_payload = context.context_index_summary(recipe, cfg, decisions=report.decisions)
    report = _append_issues(
        report,
        context.index_summary_issues(recipe, cfg, report.decisions, index_payload=index_payload),
    )
    commands = _commands_for_recipe(recipe, cfg, report.decisions) if report.exit_code == 0 else []
    if report.exit_code == 0 and not commands:
        report = _unsupported_command_report(report, task)
    report = _guard_existing_outputs(
        report,
        context.planned_context_paths(out, report),
        report.decisions.get("overwrite_policy"),
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
            "\n".join(rendering.script_lines(_commands_for_recipe(recipe, cfg, report.decisions))) + "\n",
            executable=True,
        )
        write_text(
            out / "validation.sh",
            "\n".join(rendering.script_lines(context.validation_commands(recipe))) + "\n",
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
        hparam.write_hparam_plan(recipe, cfg, out, unlock_final_test=unlock_final_test, report=report)
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
        if task in {"finetune", "infer", "evaluate"}:
            source_config = rendering.decision_value(report.decisions, "config", source_config)
        source_path = resolve_repo_path(source_config)
        if source_path is None or not source_path.exists():
            raise ValueError(f"Cannot freeze missing config: {source_config}")
        config_path.write_text(source_path.read_text())
        runtime_recipe = copy.deepcopy(recipe)
        runtime_recipe.setdefault("inputs", {})["config"] = str(config_path)
        runtime_recipe.setdefault("artifacts", {})["version_name"] = version
        runtime_decisions = {key: value for key, value in report.decisions.items() if key != "config"}
        runtime_cfg = config_summary(config_path, variant=recipe.get("variant"))
        commands = _commands_for_recipe(runtime_recipe, runtime_cfg, runtime_decisions)
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
            "\n".join(rendering.script_lines(commands, run_cwd=REPO_ROOT if task == "finetune" else None)) + "\n",
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
    if report.exit_code == 0 and recipe.get("task") == "hparam_tune":
        report = _append_issues(report, hparam.hparam_yaml_override_issues(recipe))
    if report.exit_code == 0 and recipe.get("task") == "hparam_tune":
        report = _append_issues(
            report,
            hparam.final_test_checkpoint_issues(report, recipe, unlock_final_test=unlock_final_test),
        )
    if report.exit_code == 0 and recipe.get("task") != "hparam_tune":
        commands = _commands_for_recipe(recipe, cfg, report.decisions)
        if not commands:
            report = _unsupported_command_report(report, str(recipe.get("task")))
    successful_plan = report.exit_code == 0
    report = _guard_existing_outputs(
        report,
        _planned_plan_paths(recipe, out, report, allow_unresolved, unlock_final_test),
        report.decisions.get("overwrite_policy"),
        root=out,
    )
    if successful_plan:
        root = experiment_root(recipe)
        if root is not None:
            report = _guard_existing_outputs(
                report,
                [root / "run_matrix.csv", root / "reports" / "run_matrix.md", root / "events.jsonl"],
                report.decisions.get("overwrite_policy"),
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
        manifest = runtime_dir / "run_manifest.json" if runtime_dir is not None else None
        data = yaml.safe_load(manifest.read_text()) if manifest is not None and manifest.exists() else {}
        if not isinstance(data, dict):
            raise ValueError(f"Run manifest must contain a mapping: {manifest}")
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


def _commands_for_recipe(recipe: dict, cfg: dict | None = None, decisions: dict | None = None) -> list[str]:
    task = recipe.get("task")
    inputs = recipe.get("inputs") if isinstance(recipe.get("inputs"), dict) else {}
    runtime = recipe.get("runtime") if isinstance(recipe.get("runtime"), dict) else {}
    artifacts = recipe.get("artifacts") if isinstance(recipe.get("artifacts"), dict) else {}
    evaluation = recipe.get("evaluation_policy") if isinstance(recipe.get("evaluation_policy"), dict) else {}
    if task == "sleep2stat":
        config = inputs.get("config")
        run_dir = rendering.sleep2stat_config_run_dir(cfg)
        if not run_dir:
            return []
        commands = [
            rendering.render_command(["python", "-m", "sleep2stat", "validate-config", "--config", config]),
        ]
        if rendering.sleep2stat_has_yasa_stage(cfg):
            commands.append(
                rendering.render_command(
                    ["python", "-m", "sleep2stat", "validate-config", "--config", config]
                    + rendering.sleep2stat_record_check_args(recipe)
                )
            )
        commands.append(
            rendering.render_command(
                [
                    "python",
                    "-m",
                    "sleep2stat",
                    "run",
                    "--config",
                    config,
                    *rendering.sleep2stat_runtime_args(recipe),
                ]
            )
        )
        if runtime.get("summarize_after_run", True) and not runtime.get("dry_run"):
            commands.append(rendering.render_command(["python", "-m", "sleep2stat", "summarize", "--run-dir", run_dir]))
        if runtime.get("plot_cohort_after_run") is True and not runtime.get("dry_run"):
            plot_cmd = [
                "python",
                "-m",
                "sleep2stat",
                "plot-cohort",
                "--run-dir",
                run_dir,
                "--group-column",
                runtime.get("plot_group_column", "source"),
            ]
            plot_stage_source = runtime.get("plot_stage_source")
            if plot_stage_source not in (None, ""):
                plot_cmd.extend(["--stage-source", str(plot_stage_source)])
            rendering.append_list_option(plot_cmd, "--adjust-covariates", runtime.get("plot_adjust_covariates"))
            commands.append(rendering.render_command(plot_cmd))
        return commands
    if task == "finetune":
        test_after_fit = rendering.decision_value(decisions, "test_after_fit", evaluation.get("test_after_fit"))
        pieces = [
            "python",
            "-m",
            rendering.variant_module(recipe, "finetune"),
            "--config",
            rendering.decision_value(decisions, "config", inputs.get("config")),
            "--label-name",
            rendering.decision_value(decisions, "label_name", inputs.get("label_name")),
            "--version-name",
            artifacts.get("version_name", recipe_name(recipe)),
            "--results-csv-path",
            artifacts.get("results_csv_path", "results/agent_results.csv"),
            *rendering.runtime_cli_args(runtime),
            *rendering.finetune_input_cli_args(
                inputs,
                decisions,
                ckpt_from_decisions=True,
                variant=str(recipe.get("variant")),
            ),
        ]
        if test_after_fit is False:
            pieces.append("--no-test-after-fit")
        return [rendering.with_env(rendering.render_command(pieces), rendering.runtime_env_vars(runtime))]
    if task in {"infer", "evaluate"}:
        return [
            rendering.render_command(
                [
                    "python",
                    "-m",
                    rendering.variant_module(recipe, "infer"),
                    "--config",
                    rendering.decision_value(decisions, "config", inputs.get("config")),
                    "--ckpt-path",
                    rendering.decision_value(decisions, "ckpt_path", inputs.get("ckpt_path")),
                    "--label-name",
                    rendering.decision_value(decisions, "label_name", inputs.get("label_name")),
                    "--eval-split",
                    rendering.decision_value(decisions, "eval_split", inputs.get("eval_split")),
                    *rendering.infer_runtime_cli_args(runtime),
                    *rendering.infer_input_cli_args(inputs, decisions, variant=str(recipe.get("variant"))),
                ]
            )
        ]
    if task == "preset_prepare":
        preset = recipe.get("preset") if isinstance(recipe.get("preset"), dict) else {}
        return [
            rendering.render_command(
                [
                    "python",
                    "preprocess/save_dataset_presets.py",
                    "--config",
                    inputs.get("config"),
                    "--index",
                    *rendering.list_value(inputs.get("index")),
                    "--dataset-name",
                    inputs.get("dataset_name"),
                    "--n-tokens",
                    preset.get("n_tokens"),
                    "--split",
                    *rendering.list_value(preset.get("split")),
                    *rendering.preset_cli_args(preset),
                ]
            )
        ]
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


def _guard_existing_outputs(
    report: DecisionReport,
    paths: list[Path],
    overwrite_decision: Any,
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
    if overwrite_decision is not None and overwrite_decision.value is True:
        return report
    if overwrite_decision is not None and overwrite_decision.value is False:
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
            if hparam.final_test_unlocked(evaluation, report.decisions, unlock_final_test):
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
