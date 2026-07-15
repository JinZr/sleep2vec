from __future__ import annotations

from pathlib import Path
from typing import Any

from . import plan_rendering as rendering
from .configs import config_summary, load_yaml
from .decision_models import DecisionIssue, DecisionReport, DecisionStatus
from .decision_paths import path_context, path_validation
from .index_csv import index_summary
from .models import coerce_list, resolve_repo_path
from .presets import preset_summary
from .skills import list_skills


def load_config_summary_for_recipe(recipe: dict) -> dict | None:
    inputs = recipe.get("inputs") if isinstance(recipe.get("inputs"), dict) else {}
    config = inputs.get("config")
    if not config:
        return None
    resolved = resolve_repo_path(config)
    if resolved is None or not resolved.exists():
        return None
    try:
        config_data = load_yaml(config)
    except Exception:
        config_data = {}
    return config_summary(
        config,
        variant=recipe.get("variant"),
        validate_survival_local_paths=not skips_local_path_validation(
            recipe,
            survival_validation_paths(config_data),
        ),
    )


def skips_local_path_validation(recipe: dict, raw_paths: list[Any] | None = None) -> bool:
    for raw_path in raw_paths or [""]:
        context = path_context(recipe, raw_path)
        if context == "remote" and path_validation(recipe, context) in {"defer", "ssh", "remote"}:
            return True
    return False


def survival_validation_paths(config_data: dict | None) -> list[Any]:
    if not isinstance(config_data, dict):
        return []
    data = config_data.get("data") if isinstance(config_data.get("data"), dict) else {}
    finetune = config_data.get("finetune") if isinstance(config_data.get("finetune"), dict) else {}
    survival = finetune.get("survival") if isinstance(finetune.get("survival"), dict) else {}
    multilabel = finetune.get("multilabel") if isinstance(finetune.get("multilabel"), dict) else {}
    paths = [data.get("finetune_data_index"), data.get("finetune_preset_path")]
    paths.extend(data.get(field) for field in ("kaldi_data_root", "kaldi_manifest"))
    paths.extend(
        survival.get(field)
        for field in ("disease_columns_index", "event_time_index", "is_event_index", "has_label_index")
    )
    paths.extend(multilabel.get(field) for field in ("disease_columns_index", "label_index", "has_label_index"))
    return [path for path in paths if path not in (None, "")]


def validation_commands(recipe: dict) -> list[str]:
    inputs = recipe.get("inputs") if isinstance(recipe.get("inputs"), dict) else {}
    commands = []
    if recipe.get("task") == "sleep2stat":
        if inputs.get("config"):
            commands.append(
                rendering.render_command(
                    ["python", "-m", "sleep2stat", "validate-config", "--config", inputs["config"]]
                )
            )
            commands.append(rendering.render_command(["python", "utils/check_configs.py", inputs["config"]]))
        commands.append(rendering.render_command(["python", "-m", "agent_tools", "skills", "--validate"]))
        return commands
    if inputs.get("config"):
        commands.append(rendering.render_command(["python", "utils/check_configs.py", inputs["config"]]))
    commands.append(rendering.render_command(["python", "-m", "agent_tools", "skills", "--validate"]))
    return commands


def skill_context(task: str) -> tuple[dict[str, Any], list[str]]:
    for skill in list_skills():
        if task in skill.get("task_types", []):
            return (
                {"name": skill.get("name"), "path": skill.get("path"), "owners": skill.get("owners", [])},
                skill.get("relevant_index", []),
            )
    return {"name": None, "path": None, "owners": []}, []


def context_index_summary(recipe: dict, cfg: dict | None) -> dict | None:
    paths, config, split_values = index_summary_inputs(recipe, cfg)
    data = (cfg or {}).get("data") or {}
    uses_kaldi_manifest = bool(
        cfg and cfg.get("variant_guess") == "sex_age_baseline" and data.get("backend") == "kaldi"
    )
    preset_path = effective_preset_path(recipe, cfg)
    finetune = (cfg or {}).get("finetune") or {}
    task_type = (finetune.get("task") or {}).get("type")
    label_sidecars_valid = False
    if task_type == "survival":
        label_sidecars_valid = (finetune.get("survival") or {}).get("valid") is True
    elif task_type == "multilabel_classification":
        label_sidecars_valid = (finetune.get("multilabel") or {}).get("valid") is True
    uses_sex_age_preset = bool(
        cfg
        and cfg.get("variant_guess") == "sex_age_baseline"
        and data.get("backend") == "npz"
        and preset_path not in (None, "", "ASK_USER")
        and label_sidecars_valid
    )
    if not paths:
        if not uses_kaldi_manifest and not uses_sex_age_preset:
            return None
        path_values = (
            [data.get("kaldi_data_root"), data.get("kaldi_manifest")] if uses_kaldi_manifest else [preset_path]
        )
        if skips_local_path_validation(recipe, path_values):
            return None
    elif skips_local_path_validation(recipe, paths):
        return None
    try:
        return index_summary(paths, config=config, split_values=split_values, preset_path=preset_path)
    except Exception as exc:
        return {"blocking_issues": [f"Failed to summarize index: {exc}"]}


def index_summary_inputs(recipe: dict, cfg: dict | None) -> tuple[list[Any], Any, list[Any]]:
    task = recipe.get("task")
    inputs = recipe.get("inputs") if isinstance(recipe.get("inputs"), dict) else {}
    config = inputs.get("config")
    if cfg and cfg.get("is_sleep2stat"):
        data = (cfg.get("sleep2stat") or {}).get("data") or {}
        return coerce_list(data.get("index")), None, []
    if task == "preset_prepare":
        preset = recipe.get("preset") if isinstance(recipe.get("preset"), dict) else {}
        return coerce_list(inputs.get("index")), config, coerce_list(preset.get("split"))
    if task in {"finetune", "hparam_tune", "infer", "evaluate"}:
        if task in {"infer", "evaluate"}:
            split_values = coerce_list(inputs.get("eval_split"))
        else:
            split_values = finetune_loaded_split_values(recipe)
        if effective_preset_path(recipe, cfg) not in (None, ""):
            return [], config, split_values
        data = (cfg or {}).get("data") or {}
        return coerce_list(data.get("finetune_data_index")), config, split_values

    paths = coerce_list(inputs.get("index"))
    if not paths and cfg:
        data = cfg.get("data") or {}
        paths = coerce_list(data.get("finetune_data_index"))
    return paths, config, []


def finetune_loaded_split_values(recipe: dict) -> list[str]:
    task = recipe.get("task")
    runtime = recipe.get("runtime") if isinstance(recipe.get("runtime"), dict) else {}
    evaluation = recipe.get("evaluation_policy") if isinstance(recipe.get("evaluation_policy"), dict) else {}

    splits: list[str] = []
    if loads_train_val(runtime.get("epochs", 30)):
        splits.extend(["train", "val"])

    test_after_fit = evaluation.get("test_after_fit")
    if task == "hparam_tune":
        if test_after_fit is True:
            splits.append("test")
    elif test_after_fit is not False:
        splits.append("test")
    return splits


def loads_train_val(epochs: Any) -> bool:
    try:
        return int(epochs) > 0
    except (TypeError, ValueError):
        return True


def index_summary_issues(
    recipe: dict,
    cfg: dict | None,
    *,
    index_payload: dict | None = None,
) -> list[DecisionIssue]:
    index_payload = context_index_summary(recipe, cfg) if index_payload is None else index_payload
    blocking = (index_payload or {}).get("blocking_issues") or []
    return [
        DecisionIssue(
            DecisionStatus.FAIL,
            "data_input",
            issue,
            None,
            {"index_summary": index_payload},
        )
        for issue in blocking
    ]


def context_preset_summary(recipe: dict, cfg: dict | None) -> dict | None:
    preset_path = effective_preset_path(recipe, cfg)
    if preset_path in (None, ""):
        return None
    try:
        return preset_summary(preset_path)
    except Exception as exc:
        return {"blocking_issues": [f"Failed to summarize preset: {exc}"]}


def effective_preset_path(recipe: dict, cfg: dict | None) -> Any:
    task = recipe.get("task")
    inputs = recipe.get("inputs") if isinstance(recipe.get("inputs"), dict) else {}
    if task in {"infer", "evaluate"}:
        preset_path = inputs.get("inference_preset_path")
        if preset_path not in (None, "", "ASK_USER"):
            return preset_path
    if task in {"finetune", "hparam_tune", "infer", "evaluate"} and cfg:
        preset_path = (cfg.get("data") or {}).get("finetune_preset_path")
        if preset_path not in (None, "", "ASK_USER"):
            return preset_path
    return None


def expected_context_artifacts(
    recipe: dict, cfg: dict | None, out: Path, report: DecisionReport
) -> list[dict[str, str]]:
    artifacts = recipe.get("artifacts") if isinstance(recipe.get("artifacts"), dict) else {}
    expected = [
        {"name": name, "path": str(path)}
        for name, path in artifacts.items()
        if path not in (None, "") and isinstance(path, (str, Path))
    ]
    if recipe.get("task") == "sleep2stat":
        expected.extend(sleep2stat_expected_artifacts(cfg))
    expected.extend({"name": path.name, "path": str(path)} for path in planned_context_paths(out, report))
    return expected


def sleep2stat_expected_artifacts(cfg: dict | None) -> list[dict[str, str]]:
    run_dir = rendering.sleep2stat_config_run_dir(cfg)
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
        {"name": "sleep2stat model summary", "path": f"{run_dir}/tables/model_summary.csv"},
        {"name": "sleep2stat analyzer summary", "path": f"{run_dir}/tables/analyzer_summary.csv"},
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


def context_markdown(payload: dict) -> str:
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


def blocked_plan_markdown(report: DecisionReport, allow_unresolved: bool) -> str:
    lines = ["# Agent Plan Blocked", "", f"Status: {report.status.value}", ""]
    if allow_unresolved:
        lines.append("A draft plan may be written, but executable commands are not generated.")
        lines.append("")
    lines.append("## Questions")
    for issue in report.blocking_issues():
        lines.append(f"- {issue.field}: {issue.question or issue.message}")
    return "\n".join(lines) + "\n"


def plan_markdown(report: DecisionReport, commands: list[str]) -> str:
    lines = ["# Agent Plan", "", f"Status: {report.status.value}", "", "## Commands", ""]
    lines.extend(f"```bash\n{command}\n```" for command in commands)
    return "\n\n".join(lines) + "\n"


def planned_context_paths(out: Path, report: DecisionReport) -> list[Path]:
    paths = [out / "context.json", out / "context.md"]
    if report.blocking_issues():
        paths.extend([out / "questions.json", out / "questions.md", out / "commands.blocked.sh"])
    elif report.exit_code == 0:
        paths.extend([out / "commands.sh", out / "validation.sh"])
    return paths
