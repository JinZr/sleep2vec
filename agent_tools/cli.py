"""The ``agent_tools`` command-line surface: 23 subcommands over the kernel.

Mixed bridge -- it forwards the domain commands (``config-summary``,
``index-summary``, ``preset-summary``) while the rest stay task-agnostic.

``test_agent_tools_cli_contract.py`` freezes the subcommand names, their
kernel/domain/mixed ownership split, the argument contracts, and the
task/variant routing matrix. Register subcommands through ``_command()``, which sets a
command's summary and description from one string: every subcommand carries a
``help=`` and a ``description=``, and every option carries a ``help=``, because
this CLI is the agent-facing documentation of its own capabilities.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import yaml

from .configs import config_summary
from .domain.index_csv import index_summary
from .domain.presets import preset_summary
from .experiment_tracking import format_experiment_status
from .experiments import (
    append_experiment_note,
    experiment_status,
    finalize_experiment,
    init_experiment,
    launch_infer_run,
    launch_preset_run,
    monitor_experiment,
    stop_infer_run,
    stop_preset_run,
)
from .hparam_runtime import launch_hparam_runs, monitor_hparam_runs, run_hparam_queue, stop_hparam_run
from .hparam_selection import select_hparam_candidates
from .manifests import read_rows
from .markdown import report_text
from .models import json_ready
from .plans import (
    build_plan,
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
    try:
        return args.func(args)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError, yaml.YAMLError) as exc:
        # Expected refusals (validation, missing artifacts, scheduler failures) end as one
        # stderr line instead of a traceback, even under --json; programming errors such as KeyError still raise.
        # Progress lines a command already wrote (doctor's phases) may precede it; the error line comes last.
        # Subprocess stderr can be multiline, so line breaks are folded to keep the single-line contract.
        message = " | ".join(line.strip() for line in str(exc).splitlines() if line.strip())
        print(f"error: {message}", file=sys.stderr)
        return 1


def _command(sub: argparse._SubParsersAction, name: str, summary: str) -> argparse.ArgumentParser:
    """Register a subcommand whose one-line summary serves as both its entry in
    ``agent_tools --help`` and its description in ``agent_tools <name> --help``.

    Every subcommand goes through here so the help contract cannot regress: the
    CLI contract test rejects a parser or an argument without help text.
    """
    return sub.add_parser(name, help=summary, description=summary)


# Shared declarations exist only for options whose help text is identical on every
# command that takes them. Per-command help (--run-dir, --plan-dir, --metric, ...) is
# part of the help contract and stays inline at each command.


def _add_json(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of the human-readable rendering.")


def _add_reason(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--reason", required=True, help="Reason recorded in the run manifest; required.")


def _add_dry_run_or_execute(parser: argparse.ArgumentParser, *, subject: str, execute_help: str) -> None:
    """Add the mutually exclusive ``--dry-run`` (the default) / ``--execute`` pair of a launching command."""
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help=f"Report the {subject} without changing state (default).",
    )
    mode.add_argument("--execute", action="store_true", help=execute_help)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent_tools",
        description="Agent-facing experiment control for sleep2vec: inspect the repository, "
        "gate high-impact decisions through consultation, publish frozen run plans, "
        "and manage their launch, monitoring, selection, and finalization.",
    )
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    _add_inspection_commands(sub)
    _add_consultation_commands(sub)
    _add_launch_commands(sub)
    _add_experiment_commands(sub)
    _add_selection_commands(sub)
    return parser


def _add_inspection_commands(sub: argparse._SubParsersAction) -> None:
    """Register the repository, config, index, and preset inspection commands."""
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
    _add_json(repo)
    repo.set_defaults(func=_cmd_repo_summary)

    runtime_sync = _command(
        sub,
        "runtime-sync",
        "Inspect or fast-forward one existing runtime checkout to origin/main in place. "
        "Dry run unless --execute is given. Never launches work.",
    )
    runtime_sync.add_argument("--workdir", required=True, help="Existing Git checkout to inspect or update in place.")
    runtime_sync.add_argument(
        "--execute",
        action="store_true",
        help="Fetch origin/main and apply a clean fast-forward update without cloning or resetting.",
    )
    runtime_sync.set_defaults(func=_cmd_runtime_sync)

    config = _command(sub, "config-summary", "Summarize a resolved training or inference YAML config.")
    config.add_argument("--config", required=True, help="Path to the YAML config to summarize.")
    _add_json(config)
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
    _add_json(index)
    index.set_defaults(func=_cmd_index_summary)

    preset = _command(sub, "preset-summary", "Summarize a dataset preset: channels, splits, and record counts.")
    preset.add_argument("--preset", required=True, help="Path to the preset file to summarize.")
    _add_json(preset)
    preset.set_defaults(func=_cmd_preset_summary)


def _add_consultation_commands(sub: argparse._SubParsersAction) -> None:
    """Register the consultation and planning commands."""
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


def _add_launch_commands(sub: argparse._SubParsersAction) -> None:
    """Register the launch, stop, queue, and monitor commands for registered plans."""
    launch = _command(
        sub,
        "hparam-launch",
        "Launch a registered hyper-parameter plan's runs. Dry run unless --execute is given.",
    )
    launch.add_argument("--plan-dir", required=True, help="Registered plan directory to launch from.")
    _add_dry_run_or_execute(launch, subject="launch", execute_help="Actually launch the runs; enables state changes.")
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
    _add_reason(infer_stop)
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
    _add_reason(preset_stop)
    preset_stop.set_defaults(func=_cmd_preset_stop)

    run_queue = _command(
        sub,
        "hparam-run-queue",
        "Drain a hyper-parameter plan's run queue within its GPU capacity, polling until every run "
        "reaches a terminal state. Dry run unless --execute is given.",
    )
    run_queue.add_argument("--plan-dir", required=True, help="Registered plan directory to drain.")
    _add_dry_run_or_execute(
        run_queue, subject="queue", execute_help="Actually launch queued runs; enables state changes."
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


def _add_experiment_commands(sub: argparse._SubParsersAction) -> None:
    """Register progress and the experiment lifecycle commands."""
    progress = _command(sub, "progress", "Report training progress for a run directory.")
    progress.add_argument("--run-dir", required=True, help="Run directory to read progress from.")
    _add_json(progress)
    progress.set_defaults(func=_cmd_progress)

    experiment_init = _command(sub, "experiment-init", "Initialize a managed experiment workspace from a spec.")
    experiment_init.add_argument("--run-dir", required=True, help="Experiment workspace root to initialize.")
    experiment_init.add_argument(
        "--spec",
        required=True,
        help="YAML spec declaring the experiment id, title, objective, root, and baseline.",
    )
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
    experiment_note.set_defaults(func=_cmd_experiment_note)

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
    experiment_finalize.set_defaults(func=_cmd_experiment_finalize)

    experiment_monitor = _command(
        sub,
        "experiment-monitor",
        "Refresh an experiment's status report from its manifests. Never launches runs.",
    )
    experiment_monitor.add_argument("--run-dir", required=True, help="Experiment workspace root to monitor.")
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
    _add_json(experiment_status_parser)
    experiment_status_parser.set_defaults(func=_cmd_experiment_status)


def _add_selection_commands(sub: argparse._SubParsersAction) -> None:
    """Register hparam-stop and hparam-select."""
    stop = _command(sub, "hparam-stop", "Stop one hyper-parameter run and record why it was stopped.")
    stop.add_argument("--run-dir", required=True, help="Run directory holding run_manifest.tsv.")
    stop.add_argument("--run-id", required=True, help="Managed run id to stop (e.g. run-000).")
    _add_reason(stop)
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
        sync_runtime(args.workdir, execute=args.execute),
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
        if row.get("status") in {"failed", "missing_pid", "unknown_scheduler"}
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
    data = read_progress(args.run_dir)
    if args.json:
        _emit(data, as_json=True)
    else:
        print(format_progress(data), end="")
    return 0


def _cmd_experiment_init(args: argparse.Namespace) -> int:
    manifest = init_experiment(args.run_dir, args.spec)
    print(f"Wrote {manifest}")
    return 0


def _cmd_experiment_note(args: argparse.Namespace) -> int:
    if not Path(args.entry).is_file():
        raise ValueError("--entry must be an existing local YAML file path; inline text and stdin are not accepted.")
    result = append_experiment_note(args.run_dir, args.entry)
    status = "appended" if result["appended"] else "already present"
    print(f"Research log {result['path']}: {result['entry_id']} {status}")
    return 0


def _cmd_experiment_finalize(args: argparse.Namespace) -> int:
    path = finalize_experiment(args.run_dir, args.report)
    print(f"Wrote {path}")
    return 0


def _cmd_experiment_monitor(args: argparse.Namespace) -> int:
    result = monitor_experiment(args.run_dir)
    if args.json:
        _emit(result, as_json=True)
    else:
        print(f"Wrote {result['report']}")
    return 0


def _cmd_experiment_status(args: argparse.Namespace) -> int:
    snapshot = experiment_status(args.run_dir)
    if args.json:
        _emit(snapshot, as_json=True)
    else:
        print(format_experiment_status(snapshot), end="")
    return 0


def _cmd_hparam_stop(args: argparse.Namespace) -> int:
    status = stop_hparam_run(args.run_dir, args.run_id, reason=args.reason)
    print(f"Wrote {status}")
    return 0


def _cmd_hparam_select(args: argparse.Namespace) -> int:
    ranking = select_hparam_candidates(args.run_dir, args.metric, args.mode)
    print(f"Wrote {ranking}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
