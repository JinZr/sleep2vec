from __future__ import annotations

import argparse

from agent_tools import cli

SUBCOMMANDS = {
    "skills",
    "repo-summary",
    "config-summary",
    "index-summary",
    "preset-summary",
    "doctor",
    "context",
    "plan",
    "collect-runs",
    "hparam-launch",
    "hparam-monitor",
    "progress",
    "experiment-init",
    "experiment-register-step",
    "experiment-finalize",
    "experiment-wandb-sync",
    "experiment-index-checkpoints",
    "experiment-monitor",
    "experiment-rank",
    "hparam-stop",
    "hparam-select",
    "hparam-external-eval",
    "hparam-export-logits",
    "hparam-threshold",
    "hparam-ensemble",
    "hparam-checkpoint-scan",
    "hparam-digest",
    "hparam-suggest",
    "hparam-adaptive-init",
    "hparam-adaptive-step",
    "hparam-adaptive-loop",
}


def _parser_contract() -> tuple[argparse.ArgumentParser, dict[str, argparse.ArgumentParser]]:
    parser = cli._build_parser()
    subparsers = next(action for action in parser._actions if isinstance(action, argparse._SubParsersAction))
    return parser, subparsers.choices


def _actions(parser: argparse.ArgumentParser) -> dict[str, argparse.Action]:
    return {action.dest: action for action in parser._actions if action.option_strings}


def test_cli_has_exactly_31_subcommands():
    _parser, subcommands = _parser_contract()

    assert set(subcommands) == SUBCOMMANDS
    assert len(subcommands) == 31


def test_plan_cli_contract():
    parser, subcommands = _parser_contract()
    actions = _actions(subcommands["plan"])
    args = parser.parse_args(["plan", "--recipe", "recipe.yaml", "--output-dir", "plan-dir"])

    assert {name for name, action in actions.items() if action.required} == {"recipe", "output_dir"}
    assert args.user_decisions is None
    assert args.allow_unresolved is False
    assert args.unlock_final_test is False


def test_collect_runs_cli_contract():
    parser, subcommands = _parser_contract()
    actions = _actions(subcommands["collect-runs"])
    args = parser.parse_args(["collect-runs", "--root", "workspace", "--output", "runs.csv"])

    assert {name for name, action in actions.items() if action.required} == {"root", "output"}
    assert args.root == "workspace"


def test_hparam_launch_cli_contract():
    parser, subcommands = _parser_contract()
    actions = _actions(subcommands["hparam-launch"])
    args = parser.parse_args(["hparam-launch", "--plan-dir", "plan-dir"])

    assert {name for name, action in actions.items() if action.required} == {"plan_dir"}
    assert args.dry_run is True
    assert args.execute is False


def test_hparam_export_logits_cli_contract():
    parser, subcommands = _parser_contract()
    actions = _actions(subcommands["hparam-export-logits"])
    args = parser.parse_args(["hparam-export-logits", "--run-dir", "run-dir", "--selected", "selected.csv"])

    assert {name for name, action in actions.items() if action.required} == {"run_dir", "selected"}
    assert actions["accelerator"].choices == ["cpu", "gpu", "auto"]
    assert args.unlock_final_test is False
    assert args.skip_test is False
    assert args.label_name is None
    assert args.val_split == "val"
    assert args.test_split == "test"
    assert args.batch_size == 12
    assert args.num_workers == 8
    assert args.devices is None
    assert args.accelerator == "gpu"
    assert args.device == "cuda"
    assert args.precision == "bf16-mixed"
    assert args.seed == 4523
    assert args.top_k == 1
    assert args.all_candidates is False
    assert args.execute is False


def test_experiment_rank_cli_contract():
    parser, subcommands = _parser_contract()
    actions = _actions(subcommands["experiment-rank"])
    args = parser.parse_args(["experiment-rank", "--run-dir", "run-dir", "--metric", "val_auroc", "--mode", "max"])

    assert {name for name, action in actions.items() if action.required} == {"run_dir", "metric", "mode"}
    assert actions["mode"].choices == ["max", "min"]
    assert args.remote is None
