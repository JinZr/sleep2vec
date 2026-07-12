from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import sys
from typing import Any

from .adaptive_hparam import adaptive_loop, adaptive_step, digest_hparam_run, init_adaptive_workflow, suggest_next_round
from .configs import config_summary
from .experiments import (
    finalize_experiment,
    index_checkpoints,
    init_experiment,
    monitor_experiment,
    rank_experiment_candidates,
    register_experiment_step,
    sync_wandb_runs,
)
from .hparam import (
    ensemble_hparam_outputs,
    export_hparam_logits,
    generate_external_eval,
    launch_hparam_runs,
    monitor_hparam_runs,
    scan_hparam_checkpoints,
    select_hparam_candidates,
    stop_hparam_run,
    threshold_hparam_outputs,
)
from .index_csv import index_summary
from .manifests import write_text
from .markdown import report_text
from .models import REPO_ROOT, json_ready
from .plans import build_context, build_plan, collect_runs, evaluate_recipe, prepare_doctor_report, write_doctor_outputs
from .presets import preset_summary
from .progress import format_progress, read_progress
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
    index.add_argument("--label-name")
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
    collect.add_argument("--root", required=True)
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
    monitor.add_argument("--health", action="store_true")
    monitor.set_defaults(func=_cmd_hparam_monitor)

    progress = sub.add_parser("progress")
    progress.add_argument("--run-dir", required=True)
    progress.add_argument("--remote")
    progress.add_argument("--json", action="store_true")
    progress.set_defaults(func=_cmd_progress)

    experiment_init = sub.add_parser("experiment-init")
    experiment_init.add_argument("--run-dir", required=True)
    experiment_init.add_argument("--spec", required=True)
    experiment_init.add_argument("--remote")
    experiment_init.set_defaults(func=_cmd_experiment_init)

    experiment_step = sub.add_parser("experiment-register-step")
    experiment_step.add_argument("--run-dir", required=True)
    experiment_step.add_argument("--spec", required=True)
    experiment_step.add_argument("--remote")
    experiment_step.set_defaults(func=_cmd_experiment_register_step)

    experiment_finalize = sub.add_parser("experiment-finalize")
    experiment_finalize.add_argument("--run-dir", required=True)
    experiment_finalize.add_argument("--report", required=True)
    experiment_finalize.add_argument("--remote")
    experiment_finalize.set_defaults(func=_cmd_experiment_finalize)

    experiment_wandb = sub.add_parser("experiment-wandb-sync")
    experiment_wandb.add_argument("--run-dir", required=True)
    experiment_wandb.add_argument("--entity", required=True)
    experiment_wandb.add_argument("--project", required=True)
    experiment_wandb.add_argument("--group")
    experiment_wandb.add_argument("--remote")
    experiment_wandb.set_defaults(func=_cmd_experiment_wandb_sync)

    experiment_checkpoints = sub.add_parser("experiment-index-checkpoints")
    experiment_checkpoints.add_argument("--run-dir", required=True)
    experiment_checkpoints.add_argument("--remote")
    experiment_checkpoints.set_defaults(func=_cmd_experiment_index_checkpoints)

    experiment_monitor = sub.add_parser("experiment-monitor")
    experiment_monitor.add_argument("--run-dir", required=True)
    experiment_monitor.add_argument("--remote")
    experiment_monitor.add_argument("--json", action="store_true")
    experiment_monitor.set_defaults(func=_cmd_experiment_monitor)

    experiment_rank = sub.add_parser("experiment-rank")
    experiment_rank.add_argument("--run-dir", required=True)
    experiment_rank.add_argument("--metric", required=True)
    experiment_rank.add_argument("--mode", choices=["max", "min"], required=True)
    experiment_rank.add_argument("--remote")
    experiment_rank.set_defaults(func=_cmd_experiment_rank)

    stop = sub.add_parser("hparam-stop")
    stop.add_argument("--run-dir", required=True)
    stop.add_argument("--run-id", required=True)
    stop.add_argument("--reason", required=True)
    stop.set_defaults(func=_cmd_hparam_stop)

    select = sub.add_parser("hparam-select")
    select.add_argument("--run-dir", required=True)
    select.add_argument("--metric")
    select.add_argument("--mode", choices=["max", "min"])
    select.set_defaults(func=_cmd_hparam_select)

    external = sub.add_parser("hparam-external-eval")
    external.add_argument("--run-dir", required=True)
    external.add_argument("--selected", required=True)
    external.add_argument("--unlock-final-test", action="store_true")
    external.add_argument("--kaldi-data-root")
    external.add_argument("--kaldi-manifest")
    external.add_argument("--finetune-data-index")
    external.add_argument("--eval-split", default="test")
    external.add_argument("--top-k", type=int, default=1)
    external.add_argument("--all-candidates", action="store_true")
    external.set_defaults(func=_cmd_hparam_external_eval)

    export_logits = sub.add_parser("hparam-export-logits")
    export_logits.add_argument("--run-dir", required=True)
    export_logits.add_argument("--selected", required=True)
    export_logits.add_argument("--unlock-final-test", action="store_true")
    export_logits.add_argument("--skip-test", action="store_true")
    export_logits.add_argument("--label-name")
    export_logits.add_argument("--val-split", default="val")
    export_logits.add_argument("--test-split", default="test")
    export_logits.add_argument("--val-kaldi-data-root")
    export_logits.add_argument("--val-kaldi-manifest")
    export_logits.add_argument("--val-finetune-data-index")
    export_logits.add_argument("--test-kaldi-data-root")
    export_logits.add_argument("--test-kaldi-manifest")
    export_logits.add_argument("--test-finetune-data-index")
    export_logits.add_argument("--batch-size", type=int, default=12)
    export_logits.add_argument("--num-workers", type=int, default=8)
    export_logits.add_argument("--devices", type=int, nargs="+")
    export_logits.add_argument("--accelerator", default="gpu", choices=["cpu", "gpu", "auto"])
    export_logits.add_argument("--device", default="cuda")
    export_logits.add_argument("--precision", default="bf16-mixed")
    export_logits.add_argument("--seed", type=int, default=4523)
    export_logits.add_argument("--top-k", type=int, default=1)
    export_logits.add_argument("--all-candidates", action="store_true")
    export_logits.add_argument("--execute", action="store_true")
    export_logits.set_defaults(func=_cmd_hparam_export_logits)

    threshold = sub.add_parser("hparam-threshold")
    threshold.add_argument("--run-dir", required=True)
    threshold.add_argument("--selected", required=True)
    threshold.set_defaults(func=_cmd_hparam_threshold)

    ensemble = sub.add_parser("hparam-ensemble")
    ensemble.add_argument("--run-dir", required=True)
    ensemble.add_argument("--candidates", required=True)
    ensemble.add_argument("--search-combinations", action="store_true")
    ensemble.add_argument("--max-size", type=int)
    ensemble.add_argument("--metric", default="exploratory_test_auroc")
    ensemble.add_argument("--mode", choices=["max", "min"], default="max")
    ensemble.add_argument("--top-k", type=int)
    ensemble.set_defaults(func=_cmd_hparam_ensemble)

    checkpoint_scan = sub.add_parser("hparam-checkpoint-scan")
    checkpoint_scan.add_argument("--run-dir", required=True)
    checkpoint_scan.add_argument("--metric", required=True)
    checkpoint_scan.add_argument("--mode", choices=["max", "min"], required=True)
    checkpoint_scan.add_argument("--top-k", type=int)
    checkpoint_scan.set_defaults(func=_cmd_hparam_checkpoint_scan)

    digest = sub.add_parser("hparam-digest")
    digest.add_argument("--run-dir", required=True)
    digest.set_defaults(func=_cmd_hparam_digest)

    suggest = sub.add_parser("hparam-suggest")
    suggest.add_argument("--workflow-dir", required=True)
    suggest.set_defaults(func=_cmd_hparam_suggest)

    adaptive_init = sub.add_parser("hparam-adaptive-init")
    adaptive_init.add_argument("--recipe", required=True)
    adaptive_init.add_argument("--output-dir", required=True)
    adaptive_init.set_defaults(func=_cmd_hparam_adaptive_init)

    adaptive_step_cmd = sub.add_parser("hparam-adaptive-step")
    adaptive_step_cmd.add_argument("--workflow-dir", required=True)
    adaptive_step_cmd.add_argument("--execute", action="store_true")
    adaptive_step_cmd.set_defaults(func=_cmd_hparam_adaptive_step)

    adaptive_loop_cmd = sub.add_parser("hparam-adaptive-loop")
    adaptive_loop_cmd.add_argument("--workflow-dir", required=True)
    adaptive_loop_cmd.add_argument("--execute", action="store_true")
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
    manifest = launch_hparam_runs(args.plan_dir, dry_run=not args.execute)
    print(f"Wrote {manifest}")
    return 0


def _cmd_hparam_monitor(args: argparse.Namespace) -> int:
    status = monitor_hparam_runs(args.run_dir, once=args.once, health=args.health)
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


def _cmd_experiment_register_step(args: argparse.Namespace) -> int:
    path = register_experiment_step(args.run_dir, args.spec, remote=args.remote)
    print(f"Wrote {path}")
    return 0


def _cmd_experiment_finalize(args: argparse.Namespace) -> int:
    path = finalize_experiment(args.run_dir, args.report, remote=args.remote)
    print(f"Wrote {path}")
    return 0


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
        script = manifest.parent / "logits_export.sh"
        write_text(script, "\n".join(_logits_export_script_lines(args)) + "\n", executable=True)
        print(f"Wrote {script}")
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
    root = init_adaptive_workflow(args.recipe, args.output_dir)
    print(f"Wrote {root}")
    return 0


def _cmd_hparam_adaptive_step(args: argparse.Namespace) -> int:
    suggestion = adaptive_step(args.workflow_dir, execute=args.execute)
    print(f"Wrote {suggestion}")
    return 0


def _cmd_hparam_adaptive_loop(args: argparse.Namespace) -> int:
    result = adaptive_loop(args.workflow_dir, execute=args.execute)
    print(f"Wrote {result}")
    return 0


def _logits_export_script_lines(args: argparse.Namespace) -> list[str]:
    command = [
        "python",
        "-m",
        "agent_tools",
        "hparam-export-logits",
        "--run-dir",
        str(Path(args.run_dir).expanduser().resolve()),
        "--selected",
        str(Path(args.selected).expanduser().resolve()),
        "--val-split",
        args.val_split,
        "--test-split",
        args.test_split,
        "--batch-size",
        args.batch_size,
        "--num-workers",
        args.num_workers,
        "--accelerator",
        args.accelerator,
        "--device",
        args.device,
        "--precision",
        args.precision,
        "--seed",
        args.seed,
        "--top-k",
        args.top_k,
        "--execute",
    ]
    if args.unlock_final_test:
        command.append("--unlock-final-test")
    if args.skip_test:
        command.append("--skip-test")
    if args.label_name:
        command.extend(["--label-name", args.label_name])
    for flag, value in (
        ("--val-kaldi-data-root", args.val_kaldi_data_root),
        ("--val-kaldi-manifest", args.val_kaldi_manifest),
        ("--val-finetune-data-index", args.val_finetune_data_index),
        ("--test-kaldi-data-root", args.test_kaldi_data_root),
        ("--test-kaldi-manifest", args.test_kaldi_manifest),
        ("--test-finetune-data-index", args.test_finetune_data_index),
    ):
        if value:
            command.extend([flag, value])
    if args.devices:
        command.append("--devices")
        command.extend(args.devices)
    if args.all_candidates:
        command.append("--all-candidates")
    repo_root = shlex.quote(str(REPO_ROOT))
    return [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        f"cd {repo_root}",
        f"export PYTHONPATH={repo_root}${{PYTHONPATH:+:$PYTHONPATH}}",
        "",
        " ".join(shlex.quote(str(part)) for part in command),
    ]


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
