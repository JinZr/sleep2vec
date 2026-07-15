from __future__ import annotations

from pathlib import Path
from typing import Any

from ..decision_models import DecisionIssue, DecisionStatus, ResolvedDecision, needs_issue
from ..decision_paths import path_context, path_validation, validate_input_path
from ..models import REPO_ROOT, coerce_list, repo_relative, resolve_repo_path
from ..plan_rendering import append_bool_option, append_list_option, append_option, render_command
from .base import TaskAdapter


def sleep2stat_config_run_dir(cfg: dict | None) -> str | None:
    if not cfg or not cfg.get("is_sleep2stat"):
        return None
    value = ((cfg.get("sleep2stat") or {}).get("run") or {}).get("output_dir")
    return str(value) if value not in (None, "") else None


def sleep2stat_runtime_args(recipe: dict[str, Any]) -> list[Any]:
    runtime = recipe.get("runtime") if isinstance(recipe.get("runtime"), dict) else {}
    inputs = recipe.get("inputs") if isinstance(recipe.get("inputs"), dict) else {}
    args: list[Any] = []
    append_list_option(args, "--split", inputs.get("split"))
    append_option(args, "--device", runtime.get("device"))
    append_option(args, "--num-workers", runtime.get("num_workers"))
    append_option(args, "--batch-size", runtime.get("batch_size"))
    append_option(args, "--limit-records", runtime.get("limit_records"))
    append_bool_option(args, runtime.get("dry_run"), "--dry-run")
    return args


def sleep2stat_record_check_args(recipe: dict[str, Any]) -> list[Any]:
    runtime = recipe.get("runtime") if isinstance(recipe.get("runtime"), dict) else {}
    inputs = recipe.get("inputs") if isinstance(recipe.get("inputs"), dict) else {}
    args: list[Any] = ["--check-records"]
    append_list_option(args, "--split", inputs.get("split"))
    append_option(args, "--limit-records", runtime.get("limit_records"))
    return args


def sleep2stat_has_yasa_stage(cfg: dict | None) -> bool:
    sleep2stat = (cfg or {}).get("sleep2stat") or {}
    for analyzer in sleep2stat.get("analyzers", []):
        if analyzer.get("enabled") is not False and analyzer.get("type") == "yasa_stage":
            return True
    return False


def _as_list(value: Any) -> list[Any]:
    if value in (None, "", []):
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def sleep2stat_existing_run_dir_issue(recipe: dict, raw_path: Any) -> DecisionIssue | None:
    context = path_context(recipe, raw_path)
    validation = path_validation(recipe, context)
    if context != "local" or validation in {"remote", "ssh"}:
        return None
    path = Path(str(raw_path)).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.exists() or (path.is_dir() and not any(path.iterdir())):
        return None
    return DecisionIssue(
        DecisionStatus.NEEDS_USER_INPUT,
        "sleep2stat.run.output_dir",
        "sleep2stat run.output_dir already exists and is not empty; sleep2stat run directories are single-use.",
        "Use a fresh run.output_dir or manually clear the existing directory before generating commands.",
        {"path": str(raw_path), "resolved_path": str(path)},
    )


def _looks_like_placeholder_path(value: str | Path | None) -> bool:
    if value in (None, ""):
        return True
    text = str(value).strip()
    lowered = text.lower()
    return (
        lowered in {"ask_user", "none", "null", "todo", "tbd", "placeholder"}
        or text.startswith("/path/to")
        or text.startswith("<")
        or "ASK_USER" in text
    )


def sleep2stat_config_summary(config_path: str | Path) -> dict[str, Any]:
    from sleep2stat.config import SUPPORTED_ANALYZER_TYPES, SUPPORTED_REDUCER_TYPES, load_config

    resolved = resolve_repo_path(config_path)
    if resolved is None:
        raise FileNotFoundError("Config path is required.")
    supported = {
        "supported_analyzer_types": sorted(SUPPORTED_ANALYZER_TYPES),
        "supported_reducer_types": sorted(SUPPORTED_REDUCER_TYPES),
    }
    try:
        cfg = load_config(resolved)
    except Exception as exc:
        return {
            "config_path": repo_relative(resolved),
            "is_sleep2stat": True,
            "data_backend": None,
            "sleep2stat": supported,
            "warnings": [],
            "blocking_issues": [str(exc)],
            "agent_risk_issues": [],
        }

    analyzers = []
    reducers = []
    agent_risk_issues = []
    for item in cfg.analyzers:
        analyzer = {
            "name": item.name,
            "type": item.type,
            "enabled": item.enabled,
            "namespace": item.namespace,
            "label_name": item.label_name,
            "config": str(item.config) if item.config else None,
            "ckpt_path": str(item.ckpt_path) if item.ckpt_path else None,
            "input_channels": list(item.input_channels),
            "stage_source": item.stage_source,
            "event_source": item.event_source,
        }
        analyzers.append(analyzer)
        if item.enabled and item.type == "sleep2vec_downstream":
            if _looks_like_placeholder_path(item.config):
                agent_risk_issues.append(
                    f"Analyzer {item.name} downstream config is missing or placeholder: {item.config}"
                )
            if _looks_like_placeholder_path(item.ckpt_path):
                agent_risk_issues.append(f"Analyzer {item.name} ckpt_path is missing or placeholder: {item.ckpt_path}")
    for item in cfg.reducers:
        reducers.append(
            {
                "name": item.name,
                "type": item.type,
                "enabled": item.enabled,
                "source": item.source,
                "left": item.left,
                "right": item.right,
                "age_prediction": item.age_prediction,
                "sex_prediction": item.sex_prediction,
                "metadata_age_column": item.metadata_age_column,
                "metadata_sex_column": item.metadata_sex_column,
                "options": dict(item.options),
            }
        )

    return {
        "config_path": repo_relative(resolved),
        "is_sleep2stat": True,
        "data_backend": cfg.data.backend,
        "sleep2stat": {
            "run": {
                "name": cfg.run.name,
                "output_dir": str(cfg.run.output_dir),
            },
            "data": {
                "backend": cfg.data.backend,
                "index": str(cfg.data.index) if cfg.data.index else None,
                "kaldi_data_root": str(cfg.data.kaldi_data_root) if cfg.data.kaldi_data_root else None,
                "kaldi_manifest": str(cfg.data.kaldi_manifest) if cfg.data.kaldi_manifest else None,
                "split": list(cfg.data.split),
                "metadata_columns": list(cfg.data.metadata_columns),
                "token_sec": cfg.data.token_sec,
                "max_tokens": cfg.data.max_tokens,
            },
            "analyzers": analyzers,
            "reducers": reducers,
            **supported,
            "outputs": {
                "write_global_tables": cfg.outputs.write_global_tables,
                "write_per_record": cfg.outputs.write_per_record,
                "compression": cfg.outputs.compression,
                "global_tables": dict(cfg.outputs.global_tables),
            },
        },
        "warnings": [],
        "blocking_issues": [],
        "agent_risk_issues": agent_risk_issues,
    }


SLEEP2STAT_RUNTIME_FIELDS = frozenset(
    {
        "batch_size",
        "device",
        "dry_run",
        "limit_records",
        "num_workers",
        "plot_adjust_covariates",
        "plot_cohort_after_run",
        "plot_group_column",
        "plot_stage_source",
        "summarize_after_run",
    }
)


class Sleep2statAdapter(TaskAdapter):
    task = "sleep2stat"
    requires_variant = False

    recipe_extra_fields = frozenset({"artifacts", "evaluation_policy", "execution", "inputs", "runtime"})
    artifact_fields = frozenset({"overwrite", "run_dir"})
    contract_sections = {
        "inputs": frozenset({"config", "split"}),
        "evaluation_policy": frozenset({"external_test_locked"}),
    }
    extra_decision_fields = frozenset({"config", "external_test_locked"})

    def runtime_fields(self, variant: Any) -> frozenset[str]:
        return SLEEP2STAT_RUNTIME_FIELDS

    def matches_config_data(self, data: dict[str, Any]) -> bool:
        return {"run", "data", "signals", "analyzers", "reducers", "outputs"}.issubset(set(data))

    def config_summary(self, config_path: str | Path) -> dict[str, Any]:
        return sleep2stat_config_summary(config_path)

    def task_issues(
        self,
        recipe: dict[str, Any],
        config_summary: dict[str, Any] | None,
        decisions: dict[str, ResolvedDecision],
        high_impact: dict[str, dict[str, Any]],
    ) -> list[DecisionIssue]:
        issues: list[DecisionIssue] = []
        evaluation = recipe.get("evaluation_policy") if isinstance(recipe.get("evaluation_policy"), dict) else {}
        inputs = recipe.get("inputs") if isinstance(recipe.get("inputs"), dict) else {}

        if not inputs.get("config"):
            issues.append(needs_issue("config", "sleep2stat requires inputs.config.", high_impact))
            return issues
        if not config_summary or not config_summary.get("is_sleep2stat"):
            issues.append(
                DecisionIssue(
                    DecisionStatus.FAIL,
                    "config",
                    "task=sleep2stat requires a sleep2stat config.",
                    None,
                    {"config_summary": config_summary},
                )
            )
            return issues
        for message in config_summary.get("blocking_issues", []):
            issues.append(
                DecisionIssue(
                    DecisionStatus.NEEDS_USER_INPUT,
                    "sleep2stat_config",
                    message,
                    "Please fix the sleep2stat config before the agent generates commands.",
                    {"config_path": config_summary.get("config_path")},
                )
            )
        sleep2stat = config_summary.get("sleep2stat") or {}
        cfg_run = sleep2stat.get("run") or {}
        cfg_data = sleep2stat.get("data") or {}
        recipe_run_dir = (recipe.get("artifacts") if isinstance(recipe.get("artifacts"), dict) else {}).get("run_dir")
        config_run_dir = cfg_run.get("output_dir")
        if recipe_run_dir and config_run_dir and str(recipe_run_dir) != str(config_run_dir):
            issues.append(
                DecisionIssue(
                    DecisionStatus.NEEDS_USER_INPUT,
                    "artifacts.run_dir",
                    (
                        "Recipe artifacts.run_dir differs from sleep2stat config run.output_dir. "
                        "The sleep2stat CLI uses config run.output_dir, so commands would target the wrong directory."
                    ),
                    (
                        "Should artifacts.run_dir be changed to match config run.output_dir, or should the "
                        "sleep2stat config run.output_dir be changed?"
                    ),
                    {"recipe": recipe_run_dir, "config": config_run_dir},
                )
            )
        if config_run_dir:
            existing_run_dir_issue = sleep2stat_existing_run_dir_issue(recipe, config_run_dir)
            if existing_run_dir_issue is not None:
                issues.append(existing_run_dir_issue)
        effective_split = _as_list(inputs.get("split") or cfg_data.get("split"))
        if not effective_split:
            issues.append(
                DecisionIssue(
                    DecisionStatus.NEEDS_USER_INPUT,
                    "sleep2stat_split_policy",
                    "sleep2stat split is not explicit in recipe or config.",
                    "Which split(s) should sleep2stat process?",
                    {"recipe_split": inputs.get("split"), "config_split": cfg_data.get("split")},
                )
            )
        external_test_locked = evaluation.get("external_test_locked")
        if "test" in {str(value) for value in effective_split} and external_test_locked is not True:
            issues.append(
                DecisionIssue(
                    DecisionStatus.NEEDS_USER_INPUT,
                    "external_test_locked",
                    "sleep2stat is configured for test split, but external_test_locked is not explicitly true.",
                    "Is this test split external/locked, and should outputs be descriptive-only?",
                    {"effective_split": effective_split, "external_test_locked": external_test_locked},
                )
            )
        for message in config_summary.get("agent_risk_issues", []):
            issues.append(
                DecisionIssue(
                    DecisionStatus.NEEDS_USER_INPUT,
                    "sleep2stat_config",
                    message,
                    "Please provide a concrete path, adjust path context, or disable the analyzer.",
                    {"config_path": config_summary.get("config_path")},
                )
            )
        return issues

    def configured_input_issues(
        self, recipe: dict[str, Any], config_summary: dict[str, Any] | None
    ) -> list[DecisionIssue]:
        issues: list[DecisionIssue] = []
        if not config_summary or not config_summary.get("is_sleep2stat"):
            return issues
        sleep2stat = config_summary.get("sleep2stat") or {}
        data = sleep2stat.get("data") or {}
        for data_field in ("index", "kaldi_data_root", "kaldi_manifest"):
            value = data.get(data_field)
            if value:
                issue = validate_input_path(recipe, f"sleep2stat.data.{data_field}", value, configured=True)
                if issue is not None:
                    issues.append(issue)
        for analyzer in sleep2stat.get("analyzers", []):
            if analyzer.get("enabled") is False:
                continue
            for analyzer_field in ("config", "ckpt_path"):
                value = analyzer.get(analyzer_field)
                if not value or _looks_like_placeholder_path(value):
                    continue
                issue = validate_input_path(
                    recipe,
                    f"sleep2stat.analyzer.{analyzer.get('name')}.{analyzer_field}",
                    value,
                    configured=True,
                )
                if issue is not None:
                    issues.append(issue)
        return issues

    def commands(self, recipe: dict[str, Any], config_summary: dict[str, Any] | None) -> list[str]:
        inputs = recipe.get("inputs") if isinstance(recipe.get("inputs"), dict) else {}
        runtime = recipe.get("runtime") if isinstance(recipe.get("runtime"), dict) else {}
        config = inputs.get("config")
        run_dir = sleep2stat_config_run_dir(config_summary)
        if not run_dir:
            return []
        commands = [
            render_command(["python", "-m", "sleep2stat", "validate-config", "--config", config]),
        ]
        if sleep2stat_has_yasa_stage(config_summary):
            commands.append(
                render_command(
                    ["python", "-m", "sleep2stat", "validate-config", "--config", config]
                    + sleep2stat_record_check_args(recipe)
                )
            )
        commands.append(
            render_command(
                [
                    "python",
                    "-m",
                    "sleep2stat",
                    "run",
                    "--config",
                    config,
                    *sleep2stat_runtime_args(recipe),
                ]
            )
        )
        if runtime.get("summarize_after_run", True) and not runtime.get("dry_run"):
            commands.append(render_command(["python", "-m", "sleep2stat", "summarize", "--run-dir", run_dir]))
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
            append_list_option(plot_cmd, "--adjust-covariates", runtime.get("plot_adjust_covariates"))
            commands.append(render_command(plot_cmd))
        return commands

    def validation_commands(self, recipe: dict[str, Any]) -> list[str] | None:
        inputs = recipe.get("inputs") if isinstance(recipe.get("inputs"), dict) else {}
        commands = []
        if inputs.get("config"):
            commands.append(
                render_command(["python", "-m", "sleep2stat", "validate-config", "--config", inputs["config"]])
            )
            commands.append(render_command(["python", "utils/check_configs.py", inputs["config"]]))
        commands.append(render_command(["python", "-m", "agent_tools", "skills", "--validate"]))
        return commands

    def expected_artifacts(self, recipe: dict[str, Any], config_summary: dict[str, Any] | None) -> list[dict[str, str]]:
        cfg = config_summary
        run_dir = sleep2stat_config_run_dir(cfg)
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

    def index_summary_inputs_override(
        self, recipe: dict[str, Any], config_summary: dict[str, Any] | None
    ) -> tuple[list[Any], Any, list[Any]] | None:
        if config_summary and config_summary.get("is_sleep2stat"):
            data = (config_summary.get("sleep2stat") or {}).get("data") or {}
            return coerce_list(data.get("index")), None, []
        return None


SLEEP2STAT_ADAPTER = Sleep2statAdapter()
