from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from agent_tools import cli, models, plans
from agent_tools.decisions import evaluate_consultation_gates
from agent_tools.recipes import load_consultation_policy

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
    "hparam-run-queue",
    "hparam-monitor",
    "progress",
    "experiment-init",
    "experiment-register-step",
    "experiment-finalize",
    "experiment-run",
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

RUNNABLE_TASK_VARIANT_MATRIX = [
    ("sleep2stat", None, "sleep2stat"),
    ("preset_prepare", "sleep2vec", "preprocess/save_dataset_presets.py"),
    ("preset_prepare", "sleep2vec2", "sleep2vec2/preprocess/save_dataset_presets.py"),
    ("preset_prepare", "sleep2expert", "sleep2expert/preprocess/save_dataset_presets.py"),
    *[
        (task, variant, f"{variant}.{'finetune' if task in {'finetune', 'hparam_tune'} else 'infer'}")
        for task in ("finetune", "hparam_tune", "infer", "evaluate")
        for variant in models.SUPPORTED_VARIANTS
    ],
]

REJECTED_TASK_VARIANT_MATRIX = [
    *[(task, None) for task in ("preset_prepare", "finetune", "hparam_tune", "infer", "evaluate")],
    *[(task, "unsupported") for task in ("preset_prepare", "finetune", "hparam_tune", "infer", "evaluate")],
    *[("sleep2stat", variant) for variant in models.SUPPORTED_VARIANTS],
    ("preset_prepare", "sex_age_baseline"),
]


def _parser_contract() -> tuple[argparse.ArgumentParser, dict[str, argparse.ArgumentParser]]:
    parser = cli._build_parser()
    subparsers = next(action for action in parser._actions if isinstance(action, argparse._SubParsersAction))
    return parser, subparsers.choices


def _actions(parser: argparse.ArgumentParser) -> dict[str, argparse.Action]:
    return {action.dest: action for action in parser._actions if action.option_strings}


def test_cli_has_exactly_33_subcommands():
    _parser, subcommands = _parser_contract()

    assert set(subcommands) == SUBCOMMANDS
    assert len(subcommands) == 33


def test_experiment_run_cli_contract():
    parser, subcommands = _parser_contract()
    actions = _actions(subcommands["experiment-run"])
    args = parser.parse_args(["experiment-run", "--run-dir", "experiment", "--spec", "matrix.yaml"])

    assert {name for name, action in actions.items() if action.required} == {"run_dir", "spec"}
    assert args.dry_run is True
    assert args.execute is False
    assert args.resume is False
    assert args.unlock_final_test is False
    assert args.poll_seconds == 60

    with pytest.raises(SystemExit):
        parser.parse_args(
            ["experiment-run", "--run-dir", "experiment", "--spec", "matrix.yaml", "--dry-run", "--execute"]
        )


@pytest.mark.parametrize(("status", "exit_code"), [("completed", 0), ("failed", 1), ("blocked", 1)])
def test_experiment_run_execute_exit_code_reflects_terminal_status(monkeypatch, status: str, exit_code: int):
    parser, _subcommands = _parser_contract()
    args = parser.parse_args(["experiment-run", "--run-dir", "experiment", "--spec", "matrix.yaml", "--execute"])
    monkeypatch.setattr(cli, "run_experiment_pipeline", lambda *_args, **_kwargs: {"status": status})
    monkeypatch.setattr(cli, "_emit", lambda *_args, **_kwargs: None)

    assert cli._cmd_experiment_run(args) == exit_code


def test_hparam_adaptive_step_cli_contract():
    parser, subcommands = _parser_contract()
    actions = _actions(subcommands["hparam-adaptive-step"])
    args = parser.parse_args(["hparam-adaptive-step", "--workflow-dir", "workflow"])

    assert {name for name, action in actions.items() if action.required} == {"workflow_dir"}
    assert args.proposal is None
    assert args.execute is False


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

    with pytest.raises(SystemExit):
        parser.parse_args(["hparam-launch", "--plan-dir", "plan-dir", "--dry-run", "--execute"])


def test_hparam_run_queue_cli_contract():
    parser, subcommands = _parser_contract()
    actions = _actions(subcommands["hparam-run-queue"])
    args = parser.parse_args(["hparam-run-queue", "--plan-dir", "plan-dir"])

    assert {name for name, action in actions.items() if action.required} == {"plan_dir"}
    assert args.dry_run is True
    assert args.execute is False
    assert args.poll_seconds == 60

    with pytest.raises(SystemExit):
        parser.parse_args(["hparam-run-queue", "--plan-dir", "plan-dir", "--dry-run", "--execute"])


def test_hparam_export_logits_cli_delegates_writes_to_postprocess(tmp_path: Path, monkeypatch, capsys):
    manifest = tmp_path / "logits_export_manifest.tsv"
    calls = []
    monkeypatch.setattr(
        cli,
        "export_hparam_logits",
        lambda *args, **kwargs: calls.append((args, kwargs)) or manifest,
    )

    result = cli.main(["hparam-export-logits", "--run-dir", str(tmp_path), "--selected", "selected.csv", "--skip-test"])

    assert result == 0
    assert calls[0][0] == (str(tmp_path), "selected.csv")
    assert calls[0][1]["skip_test"] is True
    assert calls[0][1]["execute"] is False
    assert capsys.readouterr().out.splitlines() == [
        f"Wrote {manifest}",
        f"Wrote {tmp_path / 'logits_export.sh'}",
    ]
    assert not (tmp_path / "logits_export.sh").exists()


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


@pytest.mark.parametrize(("task", "variant", "target"), RUNNABLE_TASK_VARIANT_MATRIX)
def test_runnable_task_variant_contract_matrix(task: str, variant: str | None, target: str):
    recipe = {
        "name": "contract-matrix",
        "task": task,
        "variant": variant,
        "inputs": {
            "config": "config.yaml",
            "index": ["index.csv"],
            "dataset_name": "unit",
            "label_name": "label",
            "ckpt_path": "model.ckpt",
            "eval_split": "test",
        },
        "preset": {"n_tokens": 1, "split": ["train"]},
        "evaluation_policy": {"test_after_fit": False},
    }
    if task == "sleep2stat":
        commands = plans._commands_for_recipe(
            recipe,
            {"is_sleep2stat": True, "sleep2stat": {"run": {"output_dir": "runs/unit"}}},
        )
        assert any("python -m sleep2stat run" in command for command in commands)
        assert models.task_requires_variant(task) is False
        return
    if task == "preset_prepare":
        assert target in plans._commands_for_recipe(recipe)[0]
    elif task == "hparam_tune":
        # Hparam plans compile finetune scripts separately, but use the same variant namespace resolver.
        assert models.module_for_variant(str(variant), "finetune") == target
    else:
        assert f"python -m {target}" in plans._commands_for_recipe(recipe)[0]
    module_path = Path(target) if target.endswith(".py") else Path(target.replace(".", "/") + ".py")
    assert (models.REPO_ROOT / module_path).is_file()
    assert models.task_requires_variant(task) is True


@pytest.mark.parametrize(("task", "variant"), REJECTED_TASK_VARIANT_MATRIX)
def test_rejected_task_variant_contract_matrix(tmp_path: Path, task: str, variant: str | None):
    policy = load_consultation_policy()
    recipe = {
        "name": "rejected-contract-matrix",
        "task": task,
        "variant": variant,
        "experiment": {
            "id": "contract-matrix",
            "title": "Contract matrix",
            "objective": "Validate finite task and variant routing.",
            "root": str(tmp_path),
            "baseline": {"type": "none"},
        },
        "step": {"id": "contract-step", "phase": "train", "purpose": "Validate routing."},
        "decisions": {"task": {"value": task, "source": "explicit_recipe"}},
    }

    report = evaluate_consultation_gates(task, recipe, None, {}, policy)

    assert report.exit_code != 0
    assert any(issue.field == "variant" for issue in report.blocking_issues())
