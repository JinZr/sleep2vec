from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from .configs import config_summary
from .decisions import DecisionStatus
from .hparam import (
    ensemble_hparam_outputs,
    generate_external_eval,
    launch_hparam_trials,
    monitor_hparam_trials,
    select_hparam_candidates,
    stop_hparam_trial,
    threshold_hparam_outputs,
)
from .index_csv import index_summary
from .manifests import write_json
from .markdown import report_text
from .models import json_ready
from .plans import build_context, build_plan, collect_runs, evaluate_recipe, prepare_doctor_report, write_doctor_outputs
from .presets import preset_summary
from .repo import repo_summary
from .skills import list_skills, validate_skills


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 2
    return args.func(args)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agent_tools")
    sub = parser.add_subparsers(dest="command")

    skills = sub.add_parser("skills")
    group = skills.add_mutually_exclusive_group(required=True)
    group.add_argument("--list", action="store_true")
    group.add_argument("--validate", action="store_true")
    skills.set_defaults(func=_cmd_skills)

    repo = sub.add_parser("repo-summary")
    repo.add_argument("--json", action="store_true")
    repo.set_defaults(func=_cmd_repo_summary)

    config = sub.add_parser("config-summary")
    config.add_argument("--config", required=True)
    config.add_argument("--json", action="store_true")
    config.set_defaults(func=_cmd_config_summary)

    index = sub.add_parser("index-summary")
    index.add_argument("--index", nargs="+", required=True)
    index.add_argument("--config")
    index.add_argument("--sample-path-check", type=int, default=0)
    index.add_argument("--sample-npz-check", type=int, default=0)
    index.add_argument("--json", action="store_true")
    index.set_defaults(func=_cmd_index_summary)

    preset = sub.add_parser("preset-summary")
    preset.add_argument("--preset", required=True)
    preset.add_argument("--json", action="store_true")
    preset.set_defaults(func=_cmd_preset_summary)

    doctor = sub.add_parser("doctor")
    doctor.add_argument("--recipe", required=True)
    doctor.add_argument("--user-decisions")
    doctor.add_argument("--output-dir")
    doctor.set_defaults(func=_cmd_doctor)

    context = sub.add_parser("context")
    context.add_argument("--task", required=True)
    context.add_argument("--config")
    context.add_argument("--label-name")
    context.add_argument("--variant")
    context.add_argument("--user-decisions")
    context.add_argument("--output-dir", required=True)
    context.set_defaults(func=_cmd_context)

    plan = sub.add_parser("plan")
    plan.add_argument("--recipe", required=True)
    plan.add_argument("--output-dir", required=True)
    plan.add_argument("--user-decisions")
    plan.add_argument("--allow-unresolved", action="store_true")
    plan.add_argument("--unlock-final-test", action="store_true")
    plan.set_defaults(func=_cmd_plan)

    collect = sub.add_parser("collect-runs")
    collect.add_argument("--root", default="log-finetune")
    collect.add_argument("--metric")
    collect.add_argument("--output", required=True)
    collect.set_defaults(func=_cmd_collect_runs)

    launch = sub.add_parser("hparam-launch")
    launch.add_argument("--plan-dir", required=True)
    launch.add_argument("--dry-run", action="store_true", default=True)
    launch.add_argument("--execute", action="store_true")
    launch.set_defaults(func=_cmd_hparam_launch)

    monitor = sub.add_parser("hparam-monitor")
    monitor.add_argument("--run-dir", required=True)
    monitor.add_argument("--once", action="store_true")
    monitor.set_defaults(func=_cmd_hparam_monitor)

    stop = sub.add_parser("hparam-stop")
    stop.add_argument("--run-dir", required=True)
    stop.add_argument("--trial-id", required=True)
    stop.set_defaults(func=_cmd_hparam_stop)

    select = sub.add_parser("hparam-select")
    select.add_argument("--run-dir", required=True)
    select.add_argument("--metric", required=True)
    select.add_argument("--mode", choices=["max", "min"], required=True)
    select.set_defaults(func=_cmd_hparam_select)

    external = sub.add_parser("hparam-external-eval")
    external.add_argument("--run-dir", required=True)
    external.add_argument("--selected", required=True)
    external.add_argument("--unlock-final-test", action="store_true")
    external.add_argument("--kaldi-data-root")
    external.add_argument("--kaldi-manifest")
    external.add_argument("--finetune-data-index")
    external.add_argument("--eval-split", default="test")
    external.set_defaults(func=_cmd_hparam_external_eval)

    threshold = sub.add_parser("hparam-threshold")
    threshold.add_argument("--run-dir", required=True)
    threshold.add_argument("--selected", required=True)
    threshold.set_defaults(func=_cmd_hparam_threshold)

    ensemble = sub.add_parser("hparam-ensemble")
    ensemble.add_argument("--run-dir", required=True)
    ensemble.add_argument("--candidates", required=True)
    ensemble.set_defaults(func=_cmd_hparam_ensemble)
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


def _cmd_config_summary(args: argparse.Namespace) -> int:
    _emit(config_summary(args.config), as_json=args.json)
    return 0


def _cmd_index_summary(args: argparse.Namespace) -> int:
    _emit(
        index_summary(
            args.index,
            config=args.config,
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
    recipe, _cfg, report = evaluate_recipe(args.recipe, args.user_decisions)
    report = prepare_doctor_report(args.output_dir, recipe, report)
    print(report_text(report))
    write_doctor_outputs(args.output_dir, recipe, report)
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
    print(f"Status: {report.status.value}")
    return report.exit_code


def _cmd_plan(args: argparse.Namespace) -> int:
    report = build_plan(
        recipe_path=args.recipe,
        output_dir=args.output_dir,
        user_decisions_path=args.user_decisions,
        allow_unresolved=args.allow_unresolved,
        unlock_final_test=args.unlock_final_test,
    )
    print(report_text(report))
    return report.exit_code


def _cmd_collect_runs(args: argparse.Namespace) -> int:
    collect_runs(args.root, args.metric, args.output)
    return 0


def _cmd_hparam_launch(args: argparse.Namespace) -> int:
    manifest = launch_hparam_trials(args.plan_dir, dry_run=not args.execute)
    print(f"Wrote {manifest}")
    return 0


def _cmd_hparam_monitor(args: argparse.Namespace) -> int:
    status = monitor_hparam_trials(args.run_dir, once=args.once)
    print(f"Wrote {status}")
    return 0


def _cmd_hparam_stop(args: argparse.Namespace) -> int:
    status = stop_hparam_trial(args.run_dir, args.trial_id)
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
    )
    print(f"Wrote {script}")
    return 0


def _cmd_hparam_threshold(args: argparse.Namespace) -> int:
    summary = threshold_hparam_outputs(args.run_dir, args.selected)
    print(f"Wrote {summary}")
    return 0


def _cmd_hparam_ensemble(args: argparse.Namespace) -> int:
    summary = ensemble_hparam_outputs(args.run_dir, args.candidates)
    print(f"Wrote {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
