from __future__ import annotations

import copy
import csv
from itertools import product
from pathlib import Path
import shlex
from typing import Any

import yaml

from .configs import config_summary, load_yaml
from .decisions import (
    DecisionIssue,
    DecisionReport,
    DecisionStatus,
    _validate_input_path,
    evaluate_consultation_gates,
    merge_status,
)
from .index_csv import index_summary
from .manifests import write_json, write_text
from .markdown import questions_markdown, questions_payload
from .models import REPO_ROOT, module_for_variant, resolve_repo_path
from .presets import preset_summary
from .recipes import load_policy_files, load_recipe_with_base, load_user_decisions, recipe_name
from .repo import repo_summary
from .skills import list_skills


def evaluate_recipe(
    recipe_path: str | Path,
    user_decisions_path: str | Path | None = None,
) -> tuple[dict, dict | None, DecisionReport]:
    recipe = load_recipe_with_base(recipe_path)
    cfg = _load_config_summary_for_recipe(recipe)
    policy, defaults = load_policy_files()
    user_decisions = load_user_decisions(user_decisions_path)
    report = evaluate_consultation_gates(
        recipe.get("task"),
        recipe,
        cfg,
        {"user_decisions": user_decisions},
        policy,
        defaults,
    )
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
    cfg = config_summary(config) if config else None
    policy, defaults = load_policy_files()
    user_decisions = load_user_decisions(user_decisions_path)
    report = evaluate_consultation_gates(
        task,
        recipe,
        cfg,
        {"label_name": label_name, "user_decisions": user_decisions},
        policy,
        defaults,
    )
    out = Path(output_dir)
    commands = _commands_for_recipe(recipe, cfg, report.decisions) if report.exit_code == 0 else []
    if report.exit_code == 0 and not commands:
        report = _unsupported_command_report(report, task)
    report = _guard_existing_outputs(
        report,
        _planned_context_paths(out, report),
        report.decisions.get("overwrite_policy"),
    )
    if _has_output_artifact_issue(report):
        return report
    skill, relevant_docs = _skill_context(task)
    payload = {
        "task": task,
        "status": report.status.value,
        "can_generate_commands": report.exit_code == 0,
        "consultation_required": any(issue.status == DecisionStatus.NEEDS_USER_INPUT for issue in report.issues),
        "questions": questions_payload(report),
        "repo": repo_summary(),
        "skill": skill,
        "owners": skill.get("owners", []),
        "relevant_docs": relevant_docs,
        "inputs": recipe["inputs"],
        "config_summary": cfg,
        "index_summary": _context_index_summary(recipe, cfg),
        "preset_summary": _context_preset_summary(recipe, cfg),
        "expected_artifacts": _expected_context_artifacts(recipe, cfg, out, report),
        "recommended_commands": commands if report.exit_code == 0 else [],
        "validation_commands": _validation_commands(recipe),
        "warnings": [issue.message for issue in report.issues if issue.status == DecisionStatus.WARN],
        "blocking_issues": [issue.message for issue in report.blocking_issues()],
    }
    write_json(out / "context.json", payload)
    write_text(out / "context.md", _context_markdown(payload))
    if report.blocking_issues():
        write_questions(out, report)
        write_text(out / "commands.blocked.sh", _blocked_script(), executable=True)
    elif report.exit_code == 0:
        write_text(
            out / "commands.sh",
            "\n".join(_script_lines(_commands_for_recipe(recipe, cfg, report.decisions))) + "\n",
            executable=True,
        )
        write_text(
            out / "validation.sh",
            "\n".join(_script_lines(_validation_commands(recipe))) + "\n",
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
    recipe, cfg, report = evaluate_recipe(recipe_path, user_decisions_path)
    out = Path(output_dir)
    if report.exit_code == 0 and recipe.get("task") == "hparam_tune":
        report = _apply_hparam_yaml_override_gate(report, recipe)
    if report.exit_code == 0 and recipe.get("task") == "hparam_tune":
        report = _apply_final_test_checkpoint_gate(report, recipe, unlock_final_test=unlock_final_test)
    commands: list[str] | None = None
    if report.exit_code == 0 and recipe.get("task") != "hparam_tune":
        commands = _commands_for_recipe(recipe, cfg, report.decisions)
        if not commands:
            report = _unsupported_command_report(report, str(recipe.get("task")))
    report = _guard_existing_outputs(
        report,
        _planned_plan_paths(recipe, out, report, allow_unresolved, unlock_final_test),
        report.decisions.get("overwrite_policy"),
    )
    if _has_output_artifact_issue(report):
        return report
    if report.exit_code == 2:
        write_questions(out, report)
        write_text(out / "plan.blocked.md", _blocked_plan_markdown(report, allow_unresolved))
        if allow_unresolved:
            write_json(
                out / "plan.draft.json",
                {"status": report.status.value, "recipe": recipe, "questions": questions_payload(report)},
            )
        return report
    if report.exit_code == 1:
        write_questions(out, report)
        write_text(out / "plan.blocked.md", _blocked_plan_markdown(report, allow_unresolved))
        return report

    task = recipe.get("task")
    if task == "hparam_tune":
        _write_hparam_plan(recipe, cfg, out, unlock_final_test=unlock_final_test, report=report)
    else:
        commands = commands if commands is not None else _commands_for_recipe(recipe, cfg, report.decisions)
        write_json(out / "plan.json", {"status": report.status.value, "commands": commands, "recipe": recipe})
        write_text(out / "plan.md", _plan_markdown(report, commands))
        write_text(out / "run.sh", "\n".join(_script_lines(commands)) + "\n", executable=True)
    return report


def collect_runs(root: str | Path, metric: str | None, output: str | Path) -> None:
    rows: list[dict[str, Any]] = []
    root_path = Path(root)
    for manifest in root_path.glob("**/run_manifest.json"):
        try:
            data = yaml.safe_load(manifest.read_text())
        except Exception:
            continue
        wandb_summary = _wandb_summary_for_run(manifest.parent)
        row = {
            "kind": "run_manifest",
            "version": data.get("version"),
            "config": data.get("config_path"),
            "best checkpoint": data.get("best_model_path"),
            "best monitor": data.get("monitor"),
            "monitor mode": data.get("monitor_mode"),
            "epoch": data.get("epoch"),
            "status": data.get("status"),
            "command": data.get("command"),
            "timestamps": data.get("finished_at_utc") or data.get("created_at_utc"),
        }
        if metric:
            row[metric] = (data.get("metrics") or {}).get(metric, wandb_summary.get(metric))
        for key, value in wandb_summary.items():
            row[f"wandb.{key}"] = value
        rows.append(row)
    for status_path in [*root_path.glob("**/trial_status.tsv"), *root_path.glob("**/launch_manifest.tsv")]:
        for row in _read_table_rows(status_path):
            rows.append({"kind": status_path.stem, **row})
    for threshold_path in root_path.glob("**/threshold_summary.csv"):
        for row in _read_table_rows(threshold_path):
            rows.append({"kind": "threshold_summary", **row})
    for overview_path in root_path.glob("**/overview.csv"):
        for row in _read_table_rows(overview_path):
            rows.append({"kind": "inference_overview", **row})
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row}) if rows else ["version"]
    with output_path.open("w", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_table_rows(path: Path) -> list[dict[str, str]]:
    delimiter = "\t" if path.suffix == ".tsv" else ","
    try:
        with path.open(newline="") as file_obj:
            return list(csv.DictReader(file_obj, delimiter=delimiter))
    except Exception:
        return []


def _wandb_summary_for_run(run_dir: Path) -> dict[str, Any]:
    candidates = sorted(run_dir.glob("wandb/*/files/wandb-summary.json"))
    if not candidates:
        return {}
    try:
        data = yaml.safe_load(candidates[-1].read_text())
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _load_config_summary_for_recipe(recipe: dict) -> dict | None:
    inputs = recipe.get("inputs") if isinstance(recipe.get("inputs"), dict) else {}
    config = inputs.get("config")
    if not config:
        return None
    resolved = resolve_repo_path(config)
    if resolved is None or not resolved.exists():
        return None
    return config_summary(config)


def _variant_module(recipe: dict, entrypoint: str) -> str:
    return module_for_variant(str(recipe.get("variant")), entrypoint)


def _render_command(parts: list[Any]) -> str:
    missing = [idx for idx, part in enumerate(parts) if part is None]
    if missing:
        raise ValueError(f"Cannot render command with missing token(s) at positions: {missing}")
    return " ".join(shlex.quote(str(part)) for part in parts)


def _append_option(args: list[Any], flag: str, value: Any) -> None:
    if value in (None, "", "ASK_USER"):
        return
    args.extend([flag, value])


def _append_list_option(args: list[Any], flag: str, values: Any) -> None:
    if values in (None, "", "ASK_USER"):
        return
    if isinstance(values, (list, tuple)):
        if not values:
            return
        args.extend([flag, *values])
    else:
        args.extend([flag, values])


def _append_bool_option(args: list[Any], value: Any, true_flag: str, false_flag: str | None = None) -> None:
    if value is True:
        args.append(true_flag)
    elif value is False and false_flag:
        args.append(false_flag)


def _runtime_cli_args(runtime: dict[str, Any]) -> list[Any]:
    args: list[Any] = [
        "--devices",
        *[str(item) for item in _list_value(runtime.get("devices", [0])) or [0]],
        "--precision",
        runtime.get("precision", "bf16-mixed"),
        "--epochs",
        runtime.get("epochs", 30),
        "--batch-size",
        runtime.get("batch_size", 12),
        "--num-workers",
        runtime.get("num_workers", 8),
        "--lr",
        runtime.get("lr", 1e-6),
        "--weight-decay",
        runtime.get("weight_decay", 1e-5),
    ]
    for key, flag in [
        ("device", "--device"),
        ("warmup_steps", "--warmup-steps"),
        ("gradient_clip_val", "--gradient-clip-val"),
        ("accumulate_grad_batches", "--accumulate-grad-batches"),
        ("patience", "--patience"),
        ("check_val_every_n_epoch", "--check-val-every-n-epoch"),
        ("ckpt_every_n_epochs", "--ckpt-every-n-epochs"),
    ]:
        _append_option(args, flag, runtime.get(key))
    return args


def _runtime_env_vars(runtime: dict[str, Any]) -> dict[str, Any]:
    env: dict[str, Any] = {}
    if runtime.get("wandb_mode") not in (None, "", "ASK_USER"):
        env["WANDB_MODE"] = runtime["wandb_mode"]
    return env


def _with_env(command: str, env: dict[str, Any]) -> str:
    if not env:
        return command
    prefix = " ".join(f"{key}={shlex.quote(str(value))}" for key, value in sorted(env.items()))
    return f"{prefix} {command}"


def _infer_runtime_cli_args(runtime: dict[str, Any]) -> list[Any]:
    args: list[Any] = [
        "--devices",
        *[str(item) for item in _list_value(runtime.get("devices", [0])) or [0]],
        "--precision",
        runtime.get("precision", "bf16-mixed"),
        "--batch-size",
        runtime.get("batch_size", 12),
        "--num-workers",
        runtime.get("num_workers", 8),
        "--lr",
        runtime.get("lr", 1e-6),
        "--weight-decay",
        runtime.get("weight_decay", 1e-5),
    ]
    for key, flag in [
        ("accelerator", "--accelerator"),
        ("device", "--device"),
        ("avg_ckpts", "--avg-ckpts"),
        ("avg_ckpt_dir", "--avg-ckpt-dir"),
        ("seed", "--seed"),
        ("wandb_mode", "--wandb-mode"),
    ]:
        _append_option(args, flag, runtime.get(key))
    return args


def _finetune_input_cli_args(
    inputs: dict[str, Any],
    decisions: dict | None,
    *,
    ckpt_from_decisions: bool,
) -> list[Any]:
    args: list[Any] = []
    _append_option(
        args,
        "--pretrained-backbone-path",
        _decision_value(decisions, "pretrained_backbone_path", inputs.get("pretrained_backbone_path")),
    )
    if ckpt_from_decisions:
        ckpt_path = _decision_value(decisions, "ckpt_path", inputs.get("ckpt_path"))
    else:
        ckpt_path = inputs.get("ckpt_path")
    _append_option(args, "--ckpt-path", ckpt_path)
    return args


def _infer_input_cli_args(inputs: dict[str, Any], decisions: dict | None) -> list[Any]:
    args: list[Any] = []
    _append_option(
        args,
        "--pretrained-backbone-path",
        _decision_value(decisions, "pretrained_backbone_path", inputs.get("pretrained_backbone_path")),
    )
    _append_option(args, "--inference-preset-path", inputs.get("inference_preset_path"))
    _append_list_option(args, "--override-dataset-names", inputs.get("override_dataset_names"))
    return args


def _list_value(value: Any) -> list[Any]:
    if value in (None, "", "ASK_USER"):
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _preset_cli_args(preset: dict[str, Any]) -> list[Any]:
    args: list[Any] = []
    _append_option(args, "--output-template", preset.get("output_template"))
    _append_option(args, "--stride-tokens", preset.get("stride_tokens"))
    _append_bool_option(args, preset.get("include_overlap_eval_splits"), "--include-overlap-eval-splits")
    _append_list_option(args, "--meta-data-names", preset.get("meta_data_names"))
    _append_bool_option(args, preset.get("include_no_metadata"), "--include-no-metadata")
    _append_list_option(args, "--channels", preset.get("channels"))
    _append_option(args, "--batch-size", preset.get("batch_size"))
    _append_bool_option(args, preset.get("shuffle"), "--shuffle", "--no-shuffle")
    _append_option(args, "--mask-rate", preset.get("mask_rate"))
    _append_bool_option(
        args,
        preset.get("allow_missing_channels"),
        "--allow-missing-channels",
        "--no-allow-missing-channels",
    )
    _append_option(args, "--min-channels", preset.get("min_channels"))
    _append_bool_option(args, preset.get("overwrite"), "--overwrite")
    _append_option(args, "--num-workers", preset.get("num_workers"))
    _append_bool_option(args, preset.get("dry_run"), "--dry-run")
    _append_option(args, "--manifest-output", preset.get("manifest_output"))
    _append_bool_option(
        args,
        preset.get("write_sidecar_manifest"),
        "--write-sidecar-manifest",
        "--no-write-sidecar-manifest",
    )
    return args


def _sleep2stat_config_run_dir(cfg: dict | None) -> str | None:
    if not cfg or not cfg.get("is_sleep2stat"):
        return None
    value = ((cfg.get("sleep2stat") or {}).get("run") or {}).get("output_dir")
    return str(value) if value not in (None, "") else None


def _sleep2stat_runtime_args(recipe: dict[str, Any]) -> list[Any]:
    runtime = recipe.get("runtime") if isinstance(recipe.get("runtime"), dict) else {}
    inputs = recipe.get("inputs") if isinstance(recipe.get("inputs"), dict) else {}
    args: list[Any] = []
    _append_list_option(args, "--split", inputs.get("split"))
    _append_option(args, "--device", runtime.get("device"))
    _append_option(args, "--num-workers", runtime.get("num_workers"))
    _append_option(args, "--batch-size", runtime.get("batch_size"))
    _append_option(args, "--limit-records", runtime.get("limit_records"))
    _append_bool_option(args, runtime.get("dry_run"), "--dry-run")
    return args


def _decision_value(decisions: dict | None, field: str, fallback: Any = None) -> Any:
    decision = decisions.get(field) if decisions else None
    if decision is not None and decision.value not in (None, ""):
        return decision.value
    return fallback


def _commands_for_recipe(recipe: dict, cfg: dict | None = None, decisions: dict | None = None) -> list[str]:
    task = recipe.get("task")
    inputs = recipe.get("inputs") if isinstance(recipe.get("inputs"), dict) else {}
    runtime = recipe.get("runtime") if isinstance(recipe.get("runtime"), dict) else {}
    artifacts = recipe.get("artifacts") if isinstance(recipe.get("artifacts"), dict) else {}
    evaluation = recipe.get("evaluation_policy") if isinstance(recipe.get("evaluation_policy"), dict) else {}
    if task == "sleep2stat":
        config = inputs.get("config")
        run_dir = _sleep2stat_config_run_dir(cfg)
        if not run_dir:
            return []
        commands = [
            _render_command(["python", "-m", "sleep2stat", "validate-config", "--config", config]),
            _render_command(
                ["python", "-m", "sleep2stat", "run", "--config", config, *_sleep2stat_runtime_args(recipe)]
            ),
        ]
        if runtime.get("summarize_after_run", True):
            commands.append(_render_command(["python", "-m", "sleep2stat", "summarize", "--run-dir", run_dir]))
        if runtime.get("plot_cohort_after_run") is True:
            plot_cmd = [
                "python",
                "-m",
                "sleep2stat",
                "plot-cohort",
                "--run-dir",
                run_dir,
                "--group-column",
                runtime.get("plot_group_column", "source"),
                "--stage-source",
                runtime.get("plot_stage_source", "auto"),
            ]
            _append_list_option(plot_cmd, "--adjust-covariates", runtime.get("plot_adjust_covariates"))
            commands.append(_render_command(plot_cmd))
        return commands
    if task == "finetune":
        test_after_fit = _decision_value(decisions, "test_after_fit", evaluation.get("test_after_fit"))
        pieces = [
            "python",
            "-m",
            _variant_module(recipe, "finetune"),
            "--config",
            _decision_value(decisions, "config", inputs.get("config")),
            "--label-name",
            _decision_value(decisions, "label_name", inputs.get("label_name")),
            "--version-name",
            artifacts.get("version_name", recipe_name(recipe)),
            "--results-csv-path",
            artifacts.get("results_csv_path", "results/agent_results.csv"),
            *_runtime_cli_args(runtime),
            *_finetune_input_cli_args(inputs, decisions, ckpt_from_decisions=True),
        ]
        if test_after_fit is False:
            pieces.append("--no-test-after-fit")
        return [_with_env(_render_command(pieces), _runtime_env_vars(runtime))]
    if task in {"infer", "evaluate"}:
        return [
            _render_command(
                [
                    "python",
                    "-m",
                    _variant_module(recipe, "infer"),
                    "--config",
                    _decision_value(decisions, "config", inputs.get("config")),
                    "--ckpt-path",
                    _decision_value(decisions, "ckpt_path", inputs.get("ckpt_path")),
                    "--label-name",
                    _decision_value(decisions, "label_name", inputs.get("label_name")),
                    "--eval-split",
                    _decision_value(decisions, "eval_split", inputs.get("eval_split")),
                    *_infer_runtime_cli_args(runtime),
                    *_infer_input_cli_args(inputs, decisions),
                ]
            )
        ]
    if task == "preset_prepare":
        preset = recipe.get("preset") if isinstance(recipe.get("preset"), dict) else {}
        return [
            _render_command(
                [
                    "python",
                    "preprocess/save_dataset_presets.py",
                    "--config",
                    inputs.get("config"),
                    "--index",
                    *_list_value(inputs.get("index")),
                    "--dataset-name",
                    inputs.get("dataset_name"),
                    "--n-tokens",
                    preset.get("n_tokens"),
                    "--split",
                    *_list_value(preset.get("split")),
                    *_preset_cli_args(preset),
                ]
            )
        ]
    return []


def _validation_commands(recipe: dict) -> list[str]:
    inputs = recipe.get("inputs") if isinstance(recipe.get("inputs"), dict) else {}
    commands = []
    if recipe.get("task") == "sleep2stat":
        if inputs.get("config"):
            commands.append(
                _render_command(["python", "-m", "sleep2stat", "validate-config", "--config", inputs["config"]])
            )
            commands.append(_render_command(["python", "utils/check_configs.py", inputs["config"]]))
        commands.append(_render_command(["python", "-m", "agent_tools", "skills", "--validate"]))
        return commands
    if inputs.get("config"):
        commands.append(_render_command(["python", "utils/check_configs.py", inputs["config"]]))
    commands.append(_render_command(["python", "-m", "agent_tools", "skills", "--validate"]))
    return commands


def _skill_context(task: str) -> tuple[dict[str, Any], list[str]]:
    for skill in list_skills():
        if task in skill.get("task_types", []):
            return (
                {"name": skill.get("name"), "path": skill.get("path"), "owners": skill.get("owners", [])},
                skill.get("relevant_index", []),
            )
    return {"name": None, "path": None, "owners": []}, []


def _context_index_summary(recipe: dict, cfg: dict | None) -> dict | None:
    inputs = recipe.get("inputs") if isinstance(recipe.get("inputs"), dict) else {}
    paths = _list_value(inputs.get("index"))
    config = inputs.get("config")
    if cfg and cfg.get("is_sleep2stat"):
        data = (cfg.get("sleep2stat") or {}).get("data") or {}
        paths = _list_value(data.get("index"))
        config = None
    if not paths and cfg:
        data = cfg.get("data") or {}
        paths = _list_value(data.get("finetune_data_index"))
    if not paths:
        return None
    try:
        return index_summary(paths, config=config)
    except Exception as exc:
        return {"blocking_issues": [f"Failed to summarize index: {exc}"]}


def _context_preset_summary(recipe: dict, cfg: dict | None) -> dict | None:
    inputs = recipe.get("inputs") if isinstance(recipe.get("inputs"), dict) else {}
    preset_path = inputs.get("preset_path") or inputs.get("inference_preset_path")
    if preset_path in (None, "") and cfg:
        preset_path = (cfg.get("data") or {}).get("finetune_preset_path")
    if preset_path in (None, ""):
        return None
    try:
        return preset_summary(preset_path)
    except Exception as exc:
        return {"blocking_issues": [f"Failed to summarize preset: {exc}"]}


def _expected_context_artifacts(
    recipe: dict, cfg: dict | None, out: Path, report: DecisionReport
) -> list[dict[str, str]]:
    artifacts = recipe.get("artifacts") if isinstance(recipe.get("artifacts"), dict) else {}
    expected = [
        {"name": name, "path": str(path)}
        for name, path in artifacts.items()
        if path not in (None, "") and isinstance(path, (str, Path))
    ]
    if recipe.get("task") == "sleep2stat":
        expected.extend(_sleep2stat_expected_artifacts(cfg))
    expected.extend({"name": path.name, "path": str(path)} for path in _planned_context_paths(out, report))
    return expected


def _sleep2stat_expected_artifacts(cfg: dict | None) -> list[dict[str, str]]:
    run_dir = _sleep2stat_config_run_dir(cfg)
    if not run_dir:
        return []
    sleep2stat = cfg.get("sleep2stat") if cfg else {}
    outputs = (sleep2stat or {}).get("outputs") or {}
    compression = outputs.get("compression", "gzip")
    global_tables = outputs.get("global_tables") or {}
    event_suffix = ".csv.gz" if compression == "gzip" else ".csv"
    expected = [
        {"name": "sleep2stat config snapshot", "path": f"{run_dir}/config.yaml"},
        {"name": "sleep2stat CLI args", "path": f"{run_dir}/cli_args.yaml"},
        {"name": "sleep2stat run manifest", "path": f"{run_dir}/run_manifest.json"},
        {"name": "sleep2stat record manifest", "path": f"{run_dir}/record_manifest.csv"},
        {"name": "sleep2stat progress", "path": f"{run_dir}/status/progress.json"},
        {"name": "sleep2stat failures", "path": f"{run_dir}/status/failures.csv"},
        {"name": "sleep2stat model summary", "path": f"{run_dir}/tables/model_summary.csv"},
        {"name": "sleep2stat analyzer summary", "path": f"{run_dir}/tables/analyzer_summary.csv"},
        {"name": "sleep2stat per-record success marker", "path": f"{run_dir}/per_record/<record_id>/_SUCCESS.json"},
        {"name": "sleep2stat per-record events", "path": f"{run_dir}/per_record/<record_id>/events{event_suffix}"},
        {"name": "sleep2stat per-record night stats", "path": f"{run_dir}/per_record/<record_id>/night_stats.json"},
        {
            "name": "sleep2stat per-record result manifest",
            "path": f"{run_dir}/per_record/<record_id>/result_manifest.csv",
        },
        {"name": "sleep2stat optional per-record arrays", "path": f"{run_dir}/per_record/<record_id>/arrays.npz"},
    ]
    if global_tables.get("night_stats", True):
        expected.append({"name": "sleep2stat night stats", "path": f"{run_dir}/tables/night_stats.csv"})
    for table in ("epoch_alignment", "second_alignment", "event_alignment"):
        if global_tables.get(table, False):
            expected.append({"name": f"sleep2stat {table}", "path": f"{run_dir}/tables/{table}{event_suffix}"})
    return expected


def _script_lines(commands: list[str]) -> list[str]:
    return [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Agent policy status: PASS",
        "# This script was generated only after consultation gates passed.",
        "# High-impact decisions were resolved by explicit recipe/config/user inputs.",
        "",
        *commands,
    ]


def _blocked_script() -> str:
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            'echo "This command plan is blocked because user input is required."',
            'echo "See questions.md."',
            "exit 2",
            "",
        ]
    )


def _context_markdown(payload: dict) -> str:
    lines = [f"# Agent Context: {payload['task']}", "", f"Status: {payload['status']}", ""]
    if payload["consultation_required"]:
        lines.extend(
            [
                "## User input required before continuing",
                "",
                "The agent must ask the user these questions before generating runnable commands.",
                "",
            ]
        )
        for question in payload["questions"]:
            lines.append(f"- {question['field']}: {question.get('question') or question.get('message')}")
    else:
        lines.extend(["## Command plan", "", *[f"- `{cmd}`" for cmd in payload["recommended_commands"]]])
    return "\n".join(lines) + "\n"


def _blocked_plan_markdown(report: DecisionReport, allow_unresolved: bool) -> str:
    lines = ["# Agent Plan Blocked", "", f"Status: {report.status.value}", ""]
    if allow_unresolved:
        lines.append("A draft plan may be written, but executable commands are not generated.")
        lines.append("")
    lines.append("## Questions")
    for issue in report.blocking_issues():
        lines.append(f"- {issue.field}: {issue.question or issue.message}")
    return "\n".join(lines) + "\n"


def _plan_markdown(report: DecisionReport, commands: list[str]) -> str:
    lines = ["# Agent Plan", "", f"Status: {report.status.value}", "", "## Commands", ""]
    lines.extend(f"```bash\n{command}\n```" for command in commands)
    return "\n\n".join(lines) + "\n"


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
) -> DecisionReport:
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


def _planned_context_paths(out: Path, report: DecisionReport) -> list[Path]:
    paths = [out / "context.json", out / "context.md"]
    if report.blocking_issues():
        paths.extend([out / "questions.json", out / "questions.md", out / "commands.blocked.sh"])
    elif report.exit_code == 0:
        paths.extend([out / "commands.sh", out / "validation.sh"])
    return paths


def _planned_plan_paths(
    recipe: dict,
    out: Path,
    report: DecisionReport,
    allow_unresolved: bool,
    unlock_final_test: bool,
) -> list[Path]:
    if report.exit_code != 0:
        paths = [out / "questions.json", out / "questions.md", out / "plan.blocked.md"]
        if allow_unresolved and report.exit_code == 2:
            paths.append(out / "plan.draft.json")
        return paths
    if recipe.get("task") != "hparam_tune":
        return [out / "plan.json", out / "plan.md", out / "run.sh"]

    paths = [out / "plan.json", out / "plan.md", out / "run_all.sh", out / "validation.sh", out / "trials.csv"]
    generated_dir = Path((recipe.get("artifacts") or {}).get("generated_config_dir", out / "configs"))
    if not generated_dir.is_absolute():
        generated_dir = REPO_ROOT / generated_dir
    for idx, _combo in enumerate(_hparam_combos(recipe)):
        paths.extend([out / f"trial_{idx:03d}.sh", generated_dir / f"trial_{idx:03d}.yaml"])
    evaluation = recipe.get("evaluation_policy") or {}
    final_allowed = unlock_final_test or (
        evaluation.get("external_test_locked") is False and evaluation.get("final_test_unlocked") is True
    )
    if final_allowed and report.exit_code == 0:
        paths.append(out / "final_external_test.sh")
    return paths


def _apply_final_test_checkpoint_gate(
    report: DecisionReport,
    recipe: dict,
    *,
    unlock_final_test: bool,
) -> DecisionReport:
    evaluation = recipe.get("evaluation_policy") if isinstance(recipe.get("evaluation_policy"), dict) else {}
    final_allowed = unlock_final_test or (
        evaluation.get("external_test_locked") is False and evaluation.get("final_test_unlocked") is True
    )
    if not final_allowed:
        return report
    ckpt_path = _resolved_ckpt_path(recipe, report)
    if ckpt_path in (None, "", "ASK_USER") or str(ckpt_path).startswith("<"):
        return _append_issues(
            report,
            [
                DecisionIssue(
                    DecisionStatus.NEEDS_USER_INPUT,
                    "ckpt_path",
                    "Final external-test evaluation requires an explicit checkpoint path.",
                    "Which checkpoint path should be used for final external-test evaluation?",
                    {"ckpt_path": ckpt_path},
                )
            ],
        )
    ckpt_issue = _validate_input_path(recipe, "ckpt_path", ckpt_path, configured=False)
    if ckpt_issue is not None:
        report = _append_issues(report, [ckpt_issue])
        if ckpt_issue.status == DecisionStatus.FAIL:
            return report
    if _has_yaml_search_overrides(recipe):
        final_config = _resolved_final_eval_config_path(recipe, report, None)
        if final_config in (None, "", "ASK_USER") or str(final_config).startswith("<"):
            return _append_issues(
                report,
                [
                    DecisionIssue(
                        DecisionStatus.NEEDS_USER_INPUT,
                        "final_eval_config_path",
                        (
                            "Final external-test evaluation for YAML-overridden hparam trials "
                            "requires an explicit config path."
                        ),
                        "Which selected trial config should be used for final external-test evaluation?",
                        {"final_eval_config_path": final_config},
                    )
                ],
            )
        config_issue = _validate_input_path(recipe, "final_eval_config_path", final_config, configured=False)
        if config_issue is not None:
            report = _append_issues(report, [config_issue])
            if config_issue.status == DecisionStatus.FAIL:
                return report
    return report


def _apply_hparam_yaml_override_gate(report: DecisionReport, recipe: dict) -> DecisionReport:
    base_recipe = recipe.get("_base_recipe") or {}
    base_cfg_path = (base_recipe.get("inputs") or {}).get("config") or (recipe.get("inputs") or {}).get("config")
    if not base_cfg_path:
        return report
    try:
        base_config = load_yaml(base_cfg_path)
        for combo in _hparam_combos(recipe):
            trial_config = copy.deepcopy(base_config)
            _apply_search_overrides(trial_config, combo)
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        return _append_issues(
            report,
            [
                DecisionIssue(
                    DecisionStatus.FAIL,
                    "hparam_search_space",
                    str(exc),
                    None,
                    {},
                )
            ],
        )
    return report


def _resolved_ckpt_path(recipe: dict, report: DecisionReport | None = None) -> Any:
    if report is not None:
        resolved = _decision_value(report.decisions, "ckpt_path")
        if resolved not in (None, ""):
            return resolved
    inputs = recipe.get("inputs") if isinstance(recipe.get("inputs"), dict) else {}
    return inputs.get("ckpt_path")


def _resolved_final_eval_config_path(recipe: dict, report: DecisionReport, fallback: Any) -> Any:
    inputs = recipe.get("inputs") if isinstance(recipe.get("inputs"), dict) else {}
    return _decision_value(report.decisions, "final_eval_config_path", inputs.get("final_eval_config_path", fallback))


def _has_yaml_search_overrides(recipe: dict) -> bool:
    search = recipe.get("search") if isinstance(recipe.get("search"), dict) else {}
    parameters = search.get("parameters") if isinstance(search.get("parameters"), dict) else {}
    return any(isinstance(key, str) and key.startswith("yaml:/") for key in parameters)


def _hparam_combos(recipe: dict) -> list[dict[str, Any]]:
    search = recipe.get("search") or {}
    params = search.get("parameters") or {}
    keys = list(params)
    combos = [dict(zip(keys, values)) for values in product(*(params[key] for key in keys))]
    max_trials = int(search.get("max_trials")) if search.get("max_trials") not in (None, "") else len(combos)
    return combos[:max_trials]


def _apply_search_overrides(config: dict[str, Any], combo: dict[str, Any]) -> dict[str, Any]:
    runtime: dict[str, Any] = {}
    for key, value in combo.items():
        if key.startswith("runtime."):
            runtime[key.split(".", 1)[1]] = value
        elif key.startswith("yaml:/"):
            _set_json_pointer(config, key.removeprefix("yaml:"), value)
    return runtime


def _set_json_pointer(config: Any, pointer: str, value: Any) -> None:
    parts = _json_pointer_parts(pointer)
    if not parts:
        raise ValueError("YAML override pointer must not target the document root.")
    parent = config
    for part in parts[:-1]:
        parent = _json_pointer_child(parent, part)
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


def _json_pointer_child(parent: Any, part: str) -> Any:
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


def _json_pointer_parts(pointer: str) -> list[str]:
    if not pointer.startswith("/"):
        raise ValueError(f"YAML override must be a JSON Pointer: {pointer}")
    return [part.replace("~1", "/").replace("~0", "~") for part in pointer.split("/")[1:]]


def _write_hparam_plan(
    recipe: dict,
    cfg: dict | None,
    out: Path,
    *,
    unlock_final_test: bool,
    report: DecisionReport,
) -> None:
    out = out.expanduser()
    if not out.is_absolute():
        out = out.resolve()
    base_recipe = recipe.get("_base_recipe") or {}
    base_cfg_path = (base_recipe.get("inputs") or {}).get("config") or (recipe.get("inputs") or {}).get("config")
    base_config = load_yaml(base_cfg_path) if base_cfg_path else {}
    combos = _hparam_combos(recipe)
    generated_dir = Path((recipe.get("artifacts") or {}).get("generated_config_dir", out / "configs"))
    if not generated_dir.is_absolute():
        generated_dir = REPO_ROOT / generated_dir
    generated_dir.mkdir(parents=True, exist_ok=True)
    trials = []
    scripts = []
    base_inputs = base_recipe.get("inputs") or {}
    base_runtime = base_recipe.get("runtime") or {}
    base_artifacts = base_recipe.get("artifacts") or {}
    evaluation = recipe.get("evaluation_policy") or {}
    for idx, combo in enumerate(combos):
        trial_id = f"trial_{idx:03d}"
        cfg_copy = generated_dir / f"{trial_id}.yaml"
        trial_config = copy.deepcopy(base_config)
        runtime_overrides = _apply_search_overrides(trial_config, combo)
        with cfg_copy.open("w") as file_obj:
            yaml.safe_dump(trial_config, file_obj)
        version = f"{recipe_name(recipe)}-{trial_id}"
        runtime = {**base_runtime, **runtime_overrides}
        command = _with_env(
            _render_command(
                [
                    "python",
                    "-m",
                    _variant_module(recipe, "finetune"),
                    "--config",
                    cfg_copy,
                    "--label-name",
                    _decision_value(report.decisions, "label_name", base_inputs.get("label_name")),
                    "--version-name",
                    version,
                    "--results-csv-path",
                    _plan_output_path(out, base_artifacts.get("results_csv_path"), "results/agent_hparam_results.csv"),
                    *_runtime_cli_args(runtime),
                    *_finetune_input_cli_args(base_inputs, report.decisions, ckpt_from_decisions=False),
                    "--no-test-after-fit",
                ]
            ),
            _runtime_env_vars(runtime),
        )
        script_name = f"{trial_id}.sh"
        write_text(out / script_name, "\n".join(_hparam_script_lines([command])) + "\n", executable=True)
        scripts.append(script_name)
        trials.append(
            {
                "trial_id": trial_id,
                "version": version,
                "config": str(cfg_copy),
                "script": script_name,
                "command": command,
                **combo,
            }
        )
    with (out / "trials.csv").open("w", newline="") as file_obj:
        fieldnames = sorted({key for row in trials for key in row})
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(trials)
    write_text(
        out / "run_all.sh",
        "\n".join(
            _hparam_script_lines(
                [
                    'cd "$(dirname "${BASH_SOURCE[0]}")"',
                    *[_render_command(["bash", script]) for script in scripts],
                ]
            )
        )
        + "\n",
        executable=True,
    )
    write_text(
        out / "validation.sh",
        "\n".join(_script_lines([_render_command(["python", "-m", "agent_tools", "skills", "--validate"])])) + "\n",
        executable=True,
    )
    write_json(out / "plan.json", {"status": "PASS", "trials": trials, "recipe": recipe})
    final_allowed = unlock_final_test or (
        evaluation.get("external_test_locked") is False and evaluation.get("final_test_unlocked") is True
    )
    plan_lines = [
        "# Hyper-Parameter Plan",
        "",
        "Status: PASS",
        "",
        "Trial commands do not evaluate the external test split.",
    ]
    if final_allowed:
        ckpt_path = _resolved_ckpt_path(recipe, report)
        final_config_path = _resolved_final_eval_config_path(recipe, report, base_cfg_path)
        final_command = _render_command(
            [
                "python",
                "-m",
                _variant_module(recipe, "infer"),
                "--config",
                final_config_path,
                "--ckpt-path",
                ckpt_path,
                "--label-name",
                _decision_value(report.decisions, "label_name", base_inputs.get("label_name")),
                "--eval-split",
                "test",
                *_infer_runtime_cli_args(base_runtime),
                *_infer_input_cli_args(base_inputs, report.decisions),
            ]
        )
        write_text(
            out / "final_external_test.sh",
            "\n".join(_hparam_script_lines([final_command])) + "\n",
            executable=True,
        )
        plan_lines.append("Final external-test script generated because final test was explicitly unlocked.")
    else:
        plan_lines.append("Final external-test script not generated; explicit unlock is required.")
    write_text(out / "plan.md", "\n".join(plan_lines) + "\n")


def _hparam_script_lines(commands: list[str]) -> list[str]:
    return [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        f"cd {shlex.quote(str(REPO_ROOT))}",
        f"export PYTHONPATH={shlex.quote(str(REPO_ROOT))}${{PYTHONPATH:+:$PYTHONPATH}}",
        "",
        "# Agent policy status: PASS",
        "# This script was generated only after consultation gates passed.",
        "# High-impact decisions were resolved by explicit recipe/config/user inputs.",
        "# External test policy:",
        "# - Trial commands do not evaluate the external test split.",
        "# - Model selection is based on validation metrics only.",
        "# - Final test evaluation requires explicit unlock.",
        "",
        *commands,
    ]


def _plan_output_path(out: Path, raw: Any, default: str) -> Path:
    path = Path(str(raw or default)).expanduser()
    return path if path.is_absolute() else out / path
