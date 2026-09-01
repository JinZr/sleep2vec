from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from .adaptive_hparam import (
    AdaptivePreflightError,
    adaptive_loop,
    adaptive_step,
    digest_hparam_run,
    init_adaptive_workflow,
    suggest_next_round,
)
from .configs import config_summary
from .domain.presets import preset_summary
from .experiment_tracking import format_experiment_status
from .experiments import (
    append_experiment_note,
    experiment_status,
    finalize_experiment,
    index_checkpoints,
    init_experiment,
    launch_infer_run,
    launch_preset_run,
    monitor_experiment,
    rank_experiment_candidates,
    register_experiment_step,
    run_experiment_pipeline,
    stop_infer_run,
    stop_preset_run,
    sync_wandb_runs,
)
from .hparam import (
    ensemble_hparam_outputs,
    export_hparam_logits,
    generate_external_eval,
    launch_hparam_runs,
    monitor_hparam_runs,
    run_hparam_queue,
    scan_hparam_checkpoints,
    select_hparam_candidates,
    stop_hparam_run,
    threshold_hparam_outputs,
)
from .index_csv import index_summary
from .manifests import read_rows
from .markdown import report_text
from .models import json_ready
from .plans import (
    build_context,
    build_plan,
    collect_runs,
    doctor_runtime_card,
    doctor_runtime_diagnostics_supported,
    evaluate_recipe,
    prepare_doctor_report,
    write_doctor_outputs,
)
from .progress import format_progress, read_progress
from .repo import repo_summary
from .runtime_sync import sync_runtime
from .skills import list_skills, validate_skills


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 2
    return args.func(args)


def _command(sub: argparse._SubParsersAction, name: str, summary: str) -> argparse.ArgumentParser:
    """Register a subcommand whose one-line summary serves as both its entry in
    ``agent_tools --help`` and its description in ``agent_tools <name> --help``.

    Every subcommand goes through here so the help contract cannot regress: the
    CLI contract test rejects a parser or an argument without help text.
    """
    return sub.add_parser(name, help=summary, description=summary)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent_tools",
        description="Agent-facing experiment control for sleep2vec: inspect the repository, "
        "gate high-impact decisions through consultation, publish frozen run plans, "
        "and manage their launch, monitoring, selection, and finalization.",
    )
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    skills = _command(sub, "skills", "List or validate the checked-in agent skill playbooks.")
    group = skills.add_mutually_exclusive_group(required=True)
    group.add_argument("--list", action="store_true", help="Print each skill's name, task types, and path.")
    group.add_argument(
        "--validate",
        action="store_true",
        help="Validate every skill manifest; exits non-zero and lists the issues on failure.",
    )
    skills.set_defaults(func=_cmd_skills)

    repo = _command(sub, "repo-summary", "Summarize repository entrypoints, variants, and tracked configs.")
    repo.add_argument("--json", action="store_true", help="Emit JSON instead of the human-readable rendering.")
    repo.set_defaults(func=_cmd_repo_summary)

    runtime_sync = _command(
        sub,
        "runtime-sync",
        "Inspect or fast-forward one existing runtime checkout to origin/main in place. "
        "Dry run unless --execute is given.",
    )
    runtime_sync.add_argument("--workdir", required=True, help="Existing Git checkout to inspect or update in place.")
    runtime_sync.add_argument("--host", help="SSH host that owns the checkout; omit for a local checkout.")
    runtime_sync.add_argument(
        "--python",
        default="python3",
        help="Python interpreter used for the self-contained sync program on --host; defaults to python3.",
    )
    runtime_sync.add_argument(
        "--execute",
        action="store_true",
        help="Fetch origin/main and apply a clean fast-forward update without cloning or resetting.",
    )
    runtime_sync.set_defaults(func=_cmd_runtime_sync)

    config = _command(sub, "config-summary", "Summarize a resolved training or inference YAML config.")
    config.add_argument("--config", required=True, help="Path to the YAML config to summarize.")
    config.add_argument("--json", action="store_true", help="Emit JSON instead of the human-readable rendering.")
    config.set_defaults(func=_cmd_config_summary)

    index = _command(
        sub,
        "index-summary",
        "Summarize dataset index CSVs, optionally sampling rows for path and NPZ checks.",
    )
    index.add_argument("--index", nargs="+", required=True, help="One or more index CSV paths to summarize.")
    index.add_argument("--config", help="Config whose data settings scope the summary.")
    index.add_argument("--label-name", help="Label column to report the class distribution for.")
    index.add_argument(
        "--sample-path-check",
        type=int,
        default=0,
        help="Number of sampled rows whose referenced files are checked for existence (0 disables).",
    )
    index.add_argument(
        "--sample-npz-check",
        type=int,
        default=0,
        help="Number of sampled rows whose NPZ payloads are opened and checked (0 disables).",
    )
    index.add_argument("--json", action="store_true", help="Emit JSON instead of the human-readable rendering.")
    index.set_defaults(func=_cmd_index_summary)

    preset = _command(sub, "preset-summary", "Summarize a dataset preset: channels, splits, and record counts.")
    preset.add_argument("--preset", required=True, help="Path to the preset file to summarize.")
    preset.add_argument("--json", action="store_true", help="Emit JSON instead of the human-readable rendering.")
    preset.set_defaults(func=_cmd_preset_summary)

    doctor = _command(
        sub,
        "doctor",
        "Run consultation, runtime, and task diagnostics for a recipe. Reports blocking questions "
        "instead of guessing them; does not publish runnable commands.",
    )
    doctor.add_argument("--recipe", required=True, help="Path to the task recipe YAML to diagnose.")
    doctor.add_argument(
        "--user-decisions",
        help="User decisions YAML answering previously reported blocking questions.",
    )
    doctor.add_argument(
        "--output-dir",
        help="Directory to publish the doctor report and a decisions.yaml template into.",
    )
    doctor.set_defaults(func=_cmd_doctor)

    context = _command(
        sub,
        "context",
        "Build a diagnostic-only context bundle for a task. Does not authorize runnable commands.",
    )
    context.add_argument(
        "--task",
        required=True,
        help="Task to build context for (sleep2stat, preset_prepare, finetune, hparam_tune, infer, evaluate).",
    )
    context.add_argument("--config", help="Config to summarize into the bundle.")
    context.add_argument("--label-name", help="Label column the bundle reports on.")
    context.add_argument("--variant", help="Model variant (sleep2vec, sleep2vec2, sleep2expert, sex_age_baseline).")
    context.add_argument("--user-decisions", help="User decisions YAML to fold into the bundle.")
    context.add_argument("--output-dir", required=True, help="Directory to write the context bundle into.")
    context.set_defaults(func=_cmd_context)

    plan = _command(
        sub,
        "plan",
        "Validate a recipe through consultation and publish a frozen, runnable plan bundle.",
    )
    plan.add_argument("--recipe", required=True, help="Path to the task recipe YAML to plan.")
    plan.add_argument(
        "--output-dir",
        required=True,
        help="Directory to publish the plan bundle into; retrying a blocked plan requires a fresh one.",
    )
    plan.add_argument(
        "--user-decisions",
        help="User decisions YAML answering previously reported blocking questions.",
    )
    plan.add_argument(
        "--allow-unresolved",
        action="store_true",
        help="Publish the plan while non-blocking consultation choices remain unresolved.",
    )
    plan.add_argument(
        "--unlock-final-test",
        action="store_true",
        help="Authorize external/final test access for this plan.",
    )
    plan.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the recipe and report without publishing plan artifacts.",
    )
    plan.set_defaults(func=_cmd_plan)

    collect = _command(sub, "collect-runs", "Collect finished run metrics under a root directory into one CSV.")
    collect.add_argument("--root", required=True, help="Directory tree to scan for run manifests.")
    collect.add_argument("--metric", help="Metric to collect; omit to collect every reported metric.")
    collect.add_argument("--output", required=True, help="CSV path to write the collected runs to.")
    collect.set_defaults(func=_cmd_collect_runs)

    launch = _command(
        sub,
        "hparam-launch",
        "Launch a registered hyper-parameter plan's runs. Dry run unless --execute is given.",
    )
    launch.add_argument("--plan-dir", required=True, help="Registered plan directory to launch from.")
    launch_mode = launch.add_mutually_exclusive_group()
    launch_mode.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Report the launch without changing state (default).",
    )
    launch_mode.add_argument("--execute", action="store_true", help="Actually launch the runs; enables state changes.")
    launch.set_defaults(func=_cmd_hparam_launch)

    infer_launch = _command(
        sub,
        "infer-launch",
        "Launch a registered inference plan's run. Dry run unless --execute is given.",
    )
    infer_launch.add_argument("--plan-dir", required=True, help="Registered plan directory to launch from.")
    infer_launch.add_argument("--execute", action="store_true", help="Actually launch the run; enables state changes.")
    infer_launch.set_defaults(func=_cmd_infer_launch)

    infer_stop = _command(sub, "infer-stop", "Stop a managed inference run and record why it was stopped.")
    infer_stop.add_argument("--plan-dir", required=True, help="Registered plan directory owning the run.")
    infer_stop.add_argument("--reason", required=True, help="Reason recorded in the run manifest; required.")
    infer_stop.set_defaults(func=_cmd_infer_stop)

    preset_launch = _command(
        sub,
        "preset-launch",
        "Launch a registered preset-preparation plan's run. Dry run unless --execute is given.",
    )
    preset_launch.add_argument("--plan-dir", required=True, help="Registered plan directory to launch from.")
    preset_launch.add_argument("--execute", action="store_true", help="Actually launch the run; enables state changes.")
    preset_launch.set_defaults(func=_cmd_preset_launch)

    preset_stop = _command(sub, "preset-stop", "Stop a managed preset-preparation run and record why.")
    preset_stop.add_argument("--plan-dir", required=True, help="Registered plan directory owning the run.")
    preset_stop.add_argument("--reason", required=True, help="Reason recorded in the run manifest; required.")
    preset_stop.set_defaults(func=_cmd_preset_stop)

    run_queue = _command(
        sub,
        "hparam-run-queue",
        "Drain a hyper-parameter plan's run queue within its GPU capacity, polling until every run "
        "reaches a terminal state. Dry run unless --execute is given.",
    )
    run_queue.add_argument("--plan-dir", required=True, help="Registered plan directory to drain.")
    run_queue_mode = run_queue.add_mutually_exclusive_group()
    run_queue_mode.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Report the queue without changing state (default).",
    )
    run_queue_mode.add_argument(
        "--execute",
        action="store_true",
        help="Actually launch queued runs; enables state changes.",
    )
    run_queue.add_argument("--poll-seconds", type=float, default=60, help="Seconds to wait between queue polls.")
    run_queue.set_defaults(func=_cmd_hparam_run_queue)

    monitor = _command(
        sub,
        "hparam-monitor",
        "Poll a hyper-parameter run directory and report lifecycle states. Never launches runs.",
    )
    monitor.add_argument("--run-dir", required=True, help="Run directory holding run_manifest.tsv.")
    monitor.add_argument("--once", action="store_true", help="Poll a single round and exit instead of looping.")
    monitor.add_argument("--health", action="store_true", help="Include scheduler and process health probes.")
    monitor.add_argument(
        "--include-log-tail",
        action="store_true",
        help="Print recorded raw log tails; they may contain sensitive data.",
    )
    monitor.add_argument("--poll-seconds", type=float, default=60, help="Seconds to wait between polls.")
    monitor.set_defaults(func=_cmd_hparam_monitor)

    progress = _command(sub, "progress", "Report training progress for a run directory.")
    progress.add_argument("--run-dir", required=True, help="Run directory to read progress from.")
    progress.add_argument("--remote", help="SSH host to read the run directory from instead of the local filesystem.")
    progress.add_argument("--json", action="store_true", help="Emit JSON instead of the human-readable rendering.")
    progress.set_defaults(func=_cmd_progress)

    experiment_init = _command(sub, "experiment-init", "Initialize a managed experiment workspace from a spec.")
    experiment_init.add_argument("--run-dir", required=True, help="Experiment workspace root to initialize.")
    experiment_init.add_argument(
        "--spec",
        required=True,
        help="YAML spec declaring the experiment id, title, objective, root, and baseline.",
    )
    experiment_init.add_argument("--remote", help="SSH host owning the workspace instead of the local filesystem.")
    experiment_init.set_defaults(func=_cmd_experiment_init)

    experiment_note = _command(
        sub,
        "experiment-note",
        "Append one evidence-backed entry to the experiment research log. Append-only; entries are never rewritten.",
    )
    experiment_note.add_argument("--run-dir", required=True, help="Experiment workspace root owning the log.")
    experiment_note.add_argument(
        "--entry",
        required=True,
        help="Existing local YAML file path; inline text is not accepted.",
    )
    experiment_note.add_argument("--remote", help="SSH host owning the workspace instead of the local filesystem.")
    experiment_note.set_defaults(func=_cmd_experiment_note)

    experiment_step = _command(
        sub,
        "experiment-register-step",
        "Register a pipeline step and its plans in the experiment workspace.",
    )
    experiment_step.add_argument("--run-dir", required=True, help="Experiment workspace root to register into.")
    experiment_step.add_argument(
        "--spec",
        required=True,
        help="YAML spec declaring the step id, phase, purpose, and the plans it owns.",
    )
    experiment_step.add_argument("--remote", help="SSH host owning the workspace instead of the local filesystem.")
    experiment_step.set_defaults(func=_cmd_experiment_register_step)

    experiment_finalize = _command(
        sub,
        "experiment-finalize",
        "Finalize an experiment by binding its final report. Requires no active runs and a non-empty report.",
    )
    experiment_finalize.add_argument("--run-dir", required=True, help="Experiment workspace root to finalize.")
    experiment_finalize.add_argument(
        "--report",
        required=True,
        help="Non-empty final report whose path and SHA-256 are bound to the completed experiment.",
    )
    experiment_finalize.add_argument("--remote", help="SSH host owning the workspace instead of the local filesystem.")
    experiment_finalize.set_defaults(func=_cmd_experiment_finalize)

    experiment_run = _command(
        sub,
        "experiment-run",
        "Run the managed validation-to-external-test pipeline for an experiment step. "
        "Dry run unless --execute is given; --execute is an explicit launching action.",
    )
    experiment_run.add_argument("--run-dir", required=True, help="Experiment workspace root to run.")
    experiment_run.add_argument("--spec", required=True, help="YAML spec declaring the pipeline step to run.")
    experiment_run.add_argument(
        "--unlock-final-test",
        action="store_true",
        help="Authorize the pipeline's external/final test stage.",
    )
    experiment_run_mode = experiment_run.add_mutually_exclusive_group()
    experiment_run_mode.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Report the pipeline without changing state (default).",
    )
    experiment_run_mode.add_argument(
        "--execute",
        action="store_true",
        help="Actually run the pipeline; enables launching and state changes.",
    )
    experiment_run.add_argument(
        "--resume",
        action="store_true",
        help="Resume an interrupted pipeline instead of starting a new attempt.",
    )
    experiment_run.add_argument("--poll-seconds", type=float, default=60, help="Seconds to wait between polls.")
    experiment_run.set_defaults(func=_cmd_experiment_run)

    experiment_wandb = _command(sub, "experiment-wandb-sync", "Sync W&B run history into the experiment workspace.")
    experiment_wandb.add_argument("--run-dir", required=True, help="Experiment workspace root to sync into.")
    experiment_wandb.add_argument("--entity", required=True, help="W&B entity that owns the runs.")
    experiment_wandb.add_argument("--project", required=True, help="W&B project to sync runs from.")
    experiment_wandb.add_argument("--group", help="Restrict the sync to one W&B group.")
    experiment_wandb.add_argument("--remote", help="SSH host owning the workspace instead of the local filesystem.")
    experiment_wandb.set_defaults(func=_cmd_experiment_wandb_sync)

    experiment_checkpoints = _command(
        sub,
        "experiment-index-checkpoints",
        "Index the checkpoints reachable from an experiment workspace.",
    )
    experiment_checkpoints.add_argument("--run-dir", required=True, help="Experiment workspace root to index.")
    experiment_checkpoints.add_argument(
        "--remote",
        help="SSH host owning the workspace instead of the local filesystem.",
    )
    experiment_checkpoints.set_defaults(func=_cmd_experiment_index_checkpoints)

    experiment_monitor = _command(
        sub,
        "experiment-monitor",
        "Refresh an experiment's status report from its manifests. Never launches runs.",
    )
    experiment_monitor.add_argument("--run-dir", required=True, help="Experiment workspace root to monitor.")
    experiment_monitor.add_argument("--remote", help="SSH host owning the workspace instead of the local filesystem.")
    experiment_monitor.add_argument(
        "--json",
        action="store_true",
        help="Emit the monitor result as JSON instead of the written report path.",
    )
    experiment_monitor.set_defaults(func=_cmd_experiment_monitor)

    experiment_status_parser = _command(
        sub,
        "experiment-status",
        "Print a projected status snapshot for an experiment workspace.",
    )
    experiment_status_parser.add_argument("--run-dir", required=True, help="Experiment workspace root to report on.")
    experiment_status_parser.add_argument(
        "--remote",
        help="SSH host owning the workspace instead of the local filesystem.",
    )
    experiment_status_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of the human-readable rendering.",
    )
    experiment_status_parser.set_defaults(func=_cmd_experiment_status)

    experiment_rank = _command(
        sub,
        "experiment-rank",
        "Rank an experiment's candidates by a metric and publish the ranking.",
    )
    experiment_rank.add_argument("--run-dir", required=True, help="Experiment workspace root to rank.")
    experiment_rank.add_argument("--metric", required=True, help="Metric to rank candidates by.")
    experiment_rank.add_argument(
        "--mode",
        choices=["max", "min"],
        required=True,
        help="Ranking direction: max ranks the highest metric first, min the lowest.",
    )
    experiment_rank.add_argument("--remote", help="SSH host owning the workspace instead of the local filesystem.")
    experiment_rank.set_defaults(func=_cmd_experiment_rank)

    stop = _command(sub, "hparam-stop", "Stop one hyper-parameter run and record why it was stopped.")
    stop.add_argument("--run-dir", required=True, help="Run directory holding run_manifest.tsv.")
    stop.add_argument("--run-id", required=True, help="Managed run id to stop (e.g. run-000).")
    stop.add_argument("--reason", required=True, help="Reason recorded in the run manifest; required.")
    stop.set_defaults(func=_cmd_hparam_stop)

    select = _command(
        sub,
        "hparam-select",
        "Rank hyper-parameter candidates on the frozen selection split and publish the selection report.",
    )
    select.add_argument("--run-dir", required=True, help="Run directory holding the completed runs.")
    select.add_argument("--metric", help="Selection metric; omit to use the plan's frozen metric.")
    select.add_argument(
        "--mode",
        choices=["max", "min"],
        help="Selection direction; omit to use the plan's frozen direction.",
    )
    select.set_defaults(func=_cmd_hparam_select)

    external = _command(
        sub,
        "hparam-external-eval",
        "Generate the external-evaluation script for selected hyper-parameter candidates.",
    )
    external.add_argument("--run-dir", required=True, help="Run directory holding the completed runs.")
    external.add_argument("--selected", required=True, help="Selection report identifying the ranked candidates.")
    external.add_argument(
        "--unlock-final-test",
        action="store_true",
        help="Authorize external/final test access for this evaluation.",
    )
    external.add_argument("--kaldi-data-root", help="Kaldi data root for the evaluation split.")
    external.add_argument("--kaldi-manifest", help="Kaldi manifest for the evaluation split.")
    external.add_argument("--finetune-data-index", help="Index CSV for the evaluation split.")
    external.add_argument("--eval-split", default="test", help="Dataset split to evaluate on.")
    external.add_argument("--top-k", type=int, default=1, help="Number of top-ranked candidates to evaluate.")
    external.add_argument(
        "--all-candidates",
        action="store_true",
        help="Evaluate every ranked candidate instead of the top-k.",
    )
    external.set_defaults(func=_cmd_hparam_external_eval)

    export_logits = _command(
        sub,
        "hparam-export-logits",
        "Export validation and test logits for selected candidates, for thresholding and ensembling.",
    )
    export_logits.add_argument("--run-dir", required=True, help="Run directory holding the completed runs.")
    export_logits.add_argument("--selected", required=True, help="Selection report identifying the ranked candidates.")
    export_logits.add_argument(
        "--unlock-final-test",
        action="store_true",
        help="Authorize external/final test access for the test-split export.",
    )
    export_logits.add_argument(
        "--skip-test",
        action="store_true",
        help="Export validation logits only, leaving the test split untouched.",
    )
    export_logits.add_argument("--label-name", help="Label column the exported logits are scored against.")
    export_logits.add_argument("--val-split", default="val", help="Validation split name to export.")
    export_logits.add_argument("--test-split", default="test", help="Test split name to export.")
    export_logits.add_argument("--val-kaldi-data-root", help="Kaldi data root for the validation split.")
    export_logits.add_argument("--val-kaldi-manifest", help="Kaldi manifest for the validation split.")
    export_logits.add_argument("--val-finetune-data-index", help="Index CSV for the validation split.")
    export_logits.add_argument("--test-kaldi-data-root", help="Kaldi data root for the test split.")
    export_logits.add_argument("--test-kaldi-manifest", help="Kaldi manifest for the test split.")
    export_logits.add_argument("--test-finetune-data-index", help="Index CSV for the test split.")
    export_logits.add_argument("--batch-size", type=int, default=12, help="Inference batch size.")
    export_logits.add_argument("--num-workers", type=int, default=8, help="Dataloader worker processes.")
    export_logits.add_argument("--devices", type=int, nargs="+", help="Device indices to run the export on.")
    export_logits.add_argument(
        "--accelerator",
        default="gpu",
        choices=["cpu", "gpu", "auto"],
        help="Lightning accelerator to run the export on.",
    )
    export_logits.add_argument("--device", default="cuda", help="Torch device string for the export.")
    export_logits.add_argument("--precision", default="bf16-mixed", help="Trainer precision for the export.")
    export_logits.add_argument("--seed", type=int, default=4523, help="Random seed for the export.")
    export_logits.add_argument("--top-k", type=int, default=1, help="Number of top-ranked candidates to export.")
    export_logits.add_argument(
        "--all-candidates",
        action="store_true",
        help="Export every ranked candidate instead of the top-k.",
    )
    export_logits.add_argument(
        "--execute",
        action="store_true",
        help="Run the export now instead of only writing logits_export.sh.",
    )
    export_logits.set_defaults(func=_cmd_hparam_export_logits)

    threshold = _command(sub, "hparam-threshold", "Tune decision thresholds from exported validation logits.")
    threshold.add_argument("--run-dir", required=True, help="Run directory holding the exported logits.")
    threshold.add_argument("--selected", required=True, help="Selection report identifying the ranked candidates.")
    threshold.set_defaults(func=_cmd_hparam_threshold)

    ensemble = _command(sub, "hparam-ensemble", "Score ensembles of candidates from their exported logits.")
    ensemble.add_argument("--run-dir", required=True, help="Run directory holding the exported logits.")
    ensemble.add_argument("--candidates", required=True, help="Candidate list to build ensembles from.")
    ensemble.add_argument(
        "--search-combinations",
        action="store_true",
        help="Search candidate subsets instead of scoring the single given combination.",
    )
    ensemble.add_argument("--max-size", type=int, help="Largest ensemble size to search.")
    ensemble.add_argument("--metric", default="exploratory_test_auroc", help="Metric to score ensembles by.")
    ensemble.add_argument(
        "--mode",
        choices=["max", "min"],
        default="max",
        help="Scoring direction: max keeps the highest metric, min the lowest.",
    )
    ensemble.add_argument("--top-k", type=int, help="Number of top-scoring ensembles to report.")
    ensemble.set_defaults(func=_cmd_hparam_ensemble)

    checkpoint_scan = _command(
        sub,
        "hparam-checkpoint-scan",
        "Rank every saved checkpoint of a hyper-parameter run by a metric.",
    )
    checkpoint_scan.add_argument("--run-dir", required=True, help="Run directory holding the saved checkpoints.")
    checkpoint_scan.add_argument("--metric", required=True, help="Metric to rank checkpoints by.")
    checkpoint_scan.add_argument(
        "--mode",
        choices=["max", "min"],
        required=True,
        help="Ranking direction: max ranks the highest metric first, min the lowest.",
    )
    checkpoint_scan.add_argument("--top-k", type=int, help="Number of top-ranked checkpoints to report.")
    checkpoint_scan.set_defaults(func=_cmd_hparam_checkpoint_scan)

    digest = _command(sub, "hparam-digest", "Publish a digest of a completed hyper-parameter round.")
    digest.add_argument("--run-dir", required=True, help="Run directory holding the completed round.")
    digest.set_defaults(func=_cmd_hparam_digest)

    suggest = _command(
        sub,
        "hparam-suggest",
        "Propose the next adaptive round from the workflow's published digests.",
    )
    suggest.add_argument("--workflow-dir", required=True, help="Adaptive workflow root holding the round digests.")
    suggest.set_defaults(func=_cmd_hparam_suggest)

    adaptive_init = _command(
        sub,
        "hparam-adaptive-init",
        "Initialize an adaptive hyper-parameter workflow from a recipe.",
    )
    adaptive_init.add_argument("--recipe", required=True, help="Path to the adaptive hparam recipe YAML.")
    adaptive_init.add_argument("--output-dir", required=True, help="Directory to create the workflow root in.")
    adaptive_init.set_defaults(func=_cmd_hparam_adaptive_init)

    adaptive_step_cmd = _command(
        sub,
        "hparam-adaptive-step",
        "Advance an adaptive workflow by one round. Validates a proposal unless --execute is given.",
    )
    adaptive_step_cmd.add_argument("--workflow-dir", required=True, help="Adaptive workflow root to advance.")
    adaptive_step_cmd.add_argument(
        "--proposal",
        help="External proposal YAML to validate and register; omit to use the workflow's own suggestion.",
    )
    adaptive_step_cmd.add_argument(
        "--execute",
        action="store_true",
        help="Register and launch the round; enables state changes.",
    )
    adaptive_step_cmd.set_defaults(func=_cmd_hparam_adaptive_step)

    adaptive_loop_cmd = _command(
        sub,
        "hparam-adaptive-loop",
        "Advance an adaptive workflow round after round until its budget is exhausted.",
    )
    adaptive_loop_cmd.add_argument("--workflow-dir", required=True, help="Adaptive workflow root to advance.")
    adaptive_loop_cmd.add_argument(
        "--execute",
        action="store_true",
        help="Register and launch each round; enables state changes.",
    )
    adaptive_loop_cmd.set_defaults(func=_cmd_hparam_adaptive_loop)
    return parser


def _emit(payload: Any, *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(json_ready(payload), indent=2, sort_keys=True))
    else:
        print(payload)


def _cmd_skills(args: argparse.Namespace) -> int:
    if args.list:
        for item in list_skills():
            print(f"{item['name']}\t{','.join(item['task_types'])}\t{item['path']}")
        return 0
    result = validate_skills()
    if result["ok"]:
        print("Skills validation: OK")
        return 0
    print("Skills validation: FAIL")
    for issue in result["issues"]:
        print(f"- {issue}")
    return 1


def _cmd_repo_summary(args: argparse.Namespace) -> int:
    _emit(repo_summary(), as_json=args.json)
    return 0


def _cmd_runtime_sync(args: argparse.Namespace) -> int:
    _emit(
        sync_runtime(args.workdir, host=args.host, remote_python=args.python, execute=args.execute),
        as_json=True,
    )
    return 0


def _cmd_config_summary(args: argparse.Namespace) -> int:
    _emit(config_summary(args.config), as_json=args.json)
    return 0


def _cmd_index_summary(args: argparse.Namespace) -> int:
    _emit(
        index_summary(
            args.index,
            config=args.config,
            label_name=args.label_name,
            sample_path_check=args.sample_path_check,
            sample_npz_check=args.sample_npz_check,
        ),
        as_json=args.json,
    )
    return 0


def _cmd_preset_summary(args: argparse.Namespace) -> int:
    _emit(preset_summary(args.preset), as_json=args.json)
    return 0


def _cmd_doctor(args: argparse.Namespace) -> int:
    output_dir = args.output_dir or "-"
    print(
        f"Doctor started: pid={os.getpid()} recipe={args.recipe} output_dir={output_dir}",
        file=sys.stderr,
        flush=True,
    )
    print("Doctor phase: consultation", file=sys.stderr, flush=True)
    recipe, _cfg, report = evaluate_recipe(args.recipe, args.user_decisions)
    if not report.blocking_issues() and doctor_runtime_diagnostics_supported(recipe):
        print("Doctor phase: runtime diagnostics", file=sys.stderr, flush=True)
        runtime_card = doctor_runtime_card(recipe)
        if runtime_card is not None:
            print(runtime_card, file=sys.stderr, flush=True)
    print("Doctor phase: task diagnostics", file=sys.stderr, flush=True)
    report = prepare_doctor_report(args.output_dir, recipe, report)
    print(report_text(report), flush=True)
    print("Doctor phase: publish outputs", file=sys.stderr, flush=True)
    template = write_doctor_outputs(args.output_dir, recipe, report)
    if template is not None:
        path, created = template
        action = "Wrote" if created else "Preserved existing"
        print(f"{action} user decisions file: {path}")
        print(f"Fill it and rerun with --user-decisions {path}.")
    print(f"Doctor finished: exit_code={report.exit_code}", file=sys.stderr, flush=True)
    return report.exit_code


def _cmd_context(args: argparse.Namespace) -> int:
    report = build_context(
        task=args.task,
        config=args.config,
        output_dir=args.output_dir,
        label_name=args.label_name,
        variant=args.variant,
        user_decisions_path=args.user_decisions,
    )
    print(report_text(report))
    return report.exit_code


def _cmd_plan(args: argparse.Namespace) -> int:
    report = build_plan(
        recipe_path=args.recipe,
        output_dir=args.output_dir,
        user_decisions_path=args.user_decisions,
        allow_unresolved=args.allow_unresolved,
        unlock_final_test=args.unlock_final_test,
        validate_only=args.validate_only,
    )
    print(report_text(report))
    if report.exit_code == 2 and report.published_user_decisions_path is not None:
        decisions_path = Path(report.published_user_decisions_path)
        print(f"User decisions file: {decisions_path}")
        print(f"Fill it and rerun with --user-decisions {decisions_path}.")
        print("The retry must use a fresh --output-dir.")
    return report.exit_code


def _cmd_collect_runs(args: argparse.Namespace) -> int:
    collect_runs(args.root, args.metric, args.output)
    return 0


def _cmd_hparam_launch(args: argparse.Namespace) -> int:
    manifest = launch_hparam_runs(args.plan_dir, dry_run=not args.execute)
    rows = read_rows(manifest, require_managed_identity=True)
    counts = Counter(str(row.get("status") or "unknown") for row in rows)
    mode = "execute (state changes enabled)" if args.execute else "dry-run (no launch attempted)"
    states = ", ".join(f"{status}={count}" for status, count in sorted(counts.items())) or "none"
    print(f"Mode: {mode}")
    print(f"Lifecycle states: {states}")
    print(f"Wrote {manifest}")
    return 0


def _cmd_infer_launch(args: argparse.Namespace) -> int:
    result = launch_infer_run(args.plan_dir, dry_run=not args.execute)
    row = result.launch_rows[0]
    mode = "execute (state changes enabled)" if args.execute else "dry-run (no launch attempted)"
    print(f"Mode: {mode}")
    print(f"Lifecycle state: {row['status']}")
    if row.get("command"):
        print(f"Submission command: {row['command']}")
    return 0


def _cmd_infer_stop(args: argparse.Namespace) -> int:
    manifest = stop_infer_run(args.plan_dir, reason=args.reason)
    print(f"Wrote {manifest}")
    return 0


def _cmd_preset_launch(args: argparse.Namespace) -> int:
    result = launch_preset_run(args.plan_dir, dry_run=not args.execute)
    row = result.launch_rows[0]
    mode = "execute (state changes enabled)" if args.execute else "dry-run (no launch attempted)"
    print(f"Mode: {mode}")
    print(f"Lifecycle state: {row['status']}")
    if row.get("command"):
        print(f"Launch command: {row['command']}")
    return 0


def _cmd_preset_stop(args: argparse.Namespace) -> int:
    manifest = stop_preset_run(args.plan_dir, reason=args.reason)
    print(f"Wrote {manifest}")
    return 0


def _cmd_hparam_run_queue(args: argparse.Namespace) -> int:
    status = run_hparam_queue(
        args.plan_dir,
        dry_run=not args.execute,
        poll_seconds=args.poll_seconds,
    )
    rows = read_rows(status, require_managed_identity=True)
    counts = Counter(str(row.get("status") or "unknown") for row in rows)
    mode = "execute (state changes enabled)" if args.execute else "dry-run (no launch attempted)"
    states = ", ".join(f"{run_status}={count}" for run_status, count in sorted(counts.items())) or "none"
    print(f"Mode: {mode}")
    print(f"Lifecycle states: {states}")
    print(f"Wrote {status}")
    return 0


def _cmd_hparam_monitor(args: argparse.Namespace) -> int:
    status = monitor_hparam_runs(
        args.run_dir,
        once=args.once,
        health=args.health,
        poll_seconds=args.poll_seconds,
    )
    rows = read_rows(status, require_managed_identity=True)
    counts = Counter(str(row.get("status") or "unknown") for row in rows)
    states = ", ".join(f"{status}={count}" for status, count in sorted(counts.items())) or "none"
    print(f"Lifecycle states: {states}")
    evidence_rows = [
        row
        for row in rows
        if row.get("status") in {"failed", "missing_pid", "unknown_remote", "unknown_scheduler"}
        or row.get("process_identity_error")
        or row.get("scheduler_health_error")
    ]
    if evidence_rows:
        print("Failure evidence:")
    for row in evidence_rows:
        details = [f"status={row.get('status') or 'unknown'}"]
        for label, field in (
            ("scheduler reason", "scheduler_reason"),
            ("process identity error", "process_identity_error"),
            ("scheduler health error", "scheduler_health_error"),
            ("log", "log_path"),
        ):
            if row.get(field):
                details.append(f"{label}={row[field]}")
        print(f"- {row['step_id']} / {row['run_id']}: {'; '.join(details)}")
        if args.include_log_tail:
            for line in str(row.get("log_tail") or "").splitlines():
                print(f"  {line}")
    print(f"Wrote {status}")
    return 0


def _cmd_progress(args: argparse.Namespace) -> int:
    data = read_progress(args.run_dir, remote=args.remote)
    if args.json:
        _emit(data, as_json=True)
    else:
        print(format_progress(data), end="")
    return 0


def _cmd_experiment_init(args: argparse.Namespace) -> int:
    manifest = init_experiment(args.run_dir, args.spec, remote=args.remote)
    print(f"Wrote {manifest}")
    return 0


def _cmd_experiment_note(args: argparse.Namespace) -> int:
    if not Path(args.entry).is_file():
        print(
            "error: --entry must be an existing local YAML file path; inline text and stdin are not accepted.",
            file=sys.stderr,
        )
        return 2
    result = append_experiment_note(args.run_dir, args.entry, remote=args.remote)
    status = "appended" if result["appended"] else "already present"
    print(f"Research log {result['path']}: {result['entry_id']} {status}")
    return 0


def _cmd_experiment_register_step(args: argparse.Namespace) -> int:
    path = register_experiment_step(args.run_dir, args.spec, remote=args.remote)
    print(f"Wrote {path}")
    return 0


def _cmd_experiment_finalize(args: argparse.Namespace) -> int:
    path = finalize_experiment(args.run_dir, args.report, remote=args.remote)
    print(f"Wrote {path}")
    return 0


def _cmd_experiment_run(args: argparse.Namespace) -> int:
    result = run_experiment_pipeline(
        args.run_dir,
        args.spec,
        unlock_final_test=args.unlock_final_test,
        execute=args.execute,
        resume=args.resume,
        poll_seconds=args.poll_seconds,
    )
    _emit(result, as_json=True)
    return 1 if args.execute and result.get("status") in {"blocked", "failed"} else 0


def _cmd_experiment_wandb_sync(args: argparse.Namespace) -> int:
    try:
        output = sync_wandb_runs(
            args.run_dir,
            entity=args.entity,
            project=args.project,
            group=args.group,
            remote=args.remote,
        )
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"Wrote {output}")
    return 0


def _cmd_experiment_index_checkpoints(args: argparse.Namespace) -> int:
    manifest = index_checkpoints(args.run_dir, remote=args.remote)
    print(f"Wrote {manifest}")
    return 0


def _cmd_experiment_monitor(args: argparse.Namespace) -> int:
    result = monitor_experiment(args.run_dir, remote=args.remote)
    if args.json:
        _emit(result, as_json=True)
    else:
        print(f"Wrote {result['report']}")
    return 0


def _cmd_experiment_status(args: argparse.Namespace) -> int:
    try:
        snapshot = experiment_status(args.run_dir, remote=args.remote)
    except (OSError, UnicodeError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if args.json:
        _emit(snapshot, as_json=True)
    else:
        print(format_experiment_status(snapshot), end="")
    return 0


def _cmd_experiment_rank(args: argparse.Namespace) -> int:
    ranking = rank_experiment_candidates(args.run_dir, metric=args.metric, mode=args.mode, remote=args.remote)
    print(f"Wrote {ranking}")
    return 0


def _cmd_hparam_stop(args: argparse.Namespace) -> int:
    status = stop_hparam_run(args.run_dir, args.run_id, reason=args.reason)
    print(f"Wrote {status}")
    return 0


def _cmd_hparam_select(args: argparse.Namespace) -> int:
    ranking = select_hparam_candidates(args.run_dir, args.metric, args.mode)
    print(f"Wrote {ranking}")
    return 0


def _cmd_hparam_external_eval(args: argparse.Namespace) -> int:
    script = generate_external_eval(
        args.run_dir,
        args.selected,
        unlock_final_test=args.unlock_final_test,
        kaldi_data_root=args.kaldi_data_root,
        kaldi_manifest=args.kaldi_manifest,
        finetune_data_index=args.finetune_data_index,
        eval_split=args.eval_split,
        top_k=args.top_k,
        all_candidates=args.all_candidates,
    )
    print(f"Wrote {script}")
    return 0


def _cmd_hparam_export_logits(args: argparse.Namespace) -> int:
    manifest = export_hparam_logits(
        args.run_dir,
        args.selected,
        unlock_final_test=args.unlock_final_test,
        val_split=args.val_split,
        test_split=args.test_split,
        skip_test=args.skip_test,
        label_name=args.label_name,
        val_kaldi_data_root=args.val_kaldi_data_root,
        val_kaldi_manifest=args.val_kaldi_manifest,
        val_finetune_data_index=args.val_finetune_data_index,
        test_kaldi_data_root=args.test_kaldi_data_root,
        test_kaldi_manifest=args.test_kaldi_manifest,
        test_finetune_data_index=args.test_finetune_data_index,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        devices=args.devices,
        accelerator=args.accelerator,
        device=args.device,
        precision=args.precision,
        seed=args.seed,
        top_k=args.top_k,
        all_candidates=args.all_candidates,
        execute=args.execute,
    )
    print(f"Wrote {manifest}")
    if not args.execute:
        print(f"Wrote {manifest.parent / 'logits_export.sh'}")
    return 0


def _cmd_hparam_threshold(args: argparse.Namespace) -> int:
    summary = threshold_hparam_outputs(args.run_dir, args.selected)
    print(f"Wrote {summary}")
    return 0


def _cmd_hparam_ensemble(args: argparse.Namespace) -> int:
    summary = ensemble_hparam_outputs(
        args.run_dir,
        args.candidates,
        search_combinations=args.search_combinations,
        max_size=args.max_size,
        metric=args.metric,
        mode=args.mode,
        top_k=args.top_k,
    )
    print(f"Wrote {summary}")
    return 0


def _cmd_hparam_checkpoint_scan(args: argparse.Namespace) -> int:
    ranking = scan_hparam_checkpoints(args.run_dir, args.metric, args.mode, top_k=args.top_k)
    print(f"Wrote {ranking}")
    return 0


def _cmd_hparam_digest(args: argparse.Namespace) -> int:
    digest = digest_hparam_run(args.run_dir)
    print(f"Wrote {digest}")
    return 0


def _cmd_hparam_suggest(args: argparse.Namespace) -> int:
    suggestion = suggest_next_round(args.workflow_dir)
    print(f"Wrote {suggestion}")
    return 0


def _cmd_hparam_adaptive_init(args: argparse.Namespace) -> int:
    try:
        root = init_adaptive_workflow(args.recipe, args.output_dir)
    except AdaptivePreflightError as exc:
        print(report_text(exc.report))
        return exc.report.exit_code
    print(f"Wrote {root}")
    return 0


def _cmd_hparam_adaptive_step(args: argparse.Namespace) -> int:
    result = adaptive_step(args.workflow_dir, proposal_path=args.proposal, execute=args.execute)
    if result is None:
        print("waiting_for_round_terminal")
    elif args.proposal and not args.execute:
        print(f"Validated {result}")
    else:
        print(f"Wrote {result}")
    return 0


def _cmd_hparam_adaptive_loop(args: argparse.Namespace) -> int:
    result = adaptive_loop(args.workflow_dir, execute=args.execute)
    print(f"Wrote {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
