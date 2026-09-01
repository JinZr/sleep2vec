from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import subprocess
import sys

import pytest

from agent_tools import cli, managed_scheduler, models, plans
from agent_tools.decisions import evaluate_consultation_gates
from agent_tools.manifests import write_rows
from agent_tools.recipes import load_consultation_policy

SUBCOMMAND_GROUPS = {
    "Kernel": {
        "repo-summary",
        "runtime-sync",
        "collect-runs",
        "hparam-launch",
        "infer-launch",
        "infer-stop",
        "preset-launch",
        "preset-stop",
        "hparam-run-queue",
        "hparam-monitor",
        "progress",
        "experiment-init",
        "experiment-note",
        "experiment-register-step",
        "experiment-finalize",
        "experiment-run",
        "experiment-wandb-sync",
        "experiment-index-checkpoints",
        "experiment-monitor",
        "experiment-status",
        "experiment-rank",
        "hparam-stop",
        "hparam-select",
        "hparam-checkpoint-scan",
        "hparam-digest",
        "hparam-suggest",
        "hparam-adaptive-init",
        "hparam-adaptive-step",
        "hparam-adaptive-loop",
    },
    "Domain": {
        "config-summary",
        "index-summary",
        "preset-summary",
        "hparam-external-eval",
        "hparam-export-logits",
        "hparam-threshold",
        "hparam-ensemble",
    },
    "Mixed": {"skills", "doctor", "context", "plan"},
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


def _subcommand_help(parser: argparse.ArgumentParser, name: str) -> str | None:
    """The one-line summary argparse lists for ``name`` in ``agent_tools --help``."""
    subparsers = next(action for action in parser._actions if isinstance(action, argparse._SubParsersAction))
    return next((choice.help for choice in subparsers._choices_actions if choice.dest == name), None)


def test_cli_has_exactly_40_subcommands():
    _parser, subcommands = _parser_contract()

    assert set(subcommands) == set.union(*SUBCOMMAND_GROUPS.values())
    assert len(subcommands) == 40


def test_every_subcommand_documents_itself():
    # agent_tools is driven by agents that discover the CLI through --help, so a
    # command with no summary is a contract break, not a cosmetic gap. The
    # listing entry (agent_tools --help) and the description (agent_tools <cmd>
    # --help) must both be present -- cli._command sets them from one string.
    parser, subcommands = _parser_contract()

    missing_summary = sorted(name for name in subcommands if not (_subcommand_help(parser, name) or "").strip())
    missing_description = sorted(name for name, sub in subcommands.items() if not (sub.description or "").strip())

    assert missing_summary == [], f"subcommands missing a --help summary: {missing_summary}"
    assert missing_description == [], f"subcommands missing a description: {missing_description}"
    assert (parser.description or "").strip(), "agent_tools itself must describe what it is for"


def test_every_subcommand_option_documents_itself():
    # Same contract one level down: an agent reading `agent_tools <cmd> --help`
    # must learn what each flag does without opening cli.py.
    _parser, subcommands = _parser_contract()

    missing = sorted(
        f"{name} {'/'.join(action.option_strings)}"
        for name, sub in subcommands.items()
        for action in _actions(sub).values()
        if not (action.help or "").strip()
    )

    assert missing == [], f"options missing help text: {missing}"


def test_subcommand_help_guard_rejects_an_undocumented_command():
    # The guard above only protects the CLI if it actually fails on a bare
    # add_parser, which is the shape a new command regresses into.
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("documented", help="Does a thing.", description="Does a thing.")
    sub.add_parser("undocumented")

    subcommands = sub.choices
    assert _subcommand_help(parser, "undocumented") is None
    assert [name for name, choice in subcommands.items() if not (choice.description or "").strip()] == ["undocumented"]


def test_subcommand_option_help_guard_rejects_an_undocumented_option():
    parser = argparse.ArgumentParser()
    parser.add_argument("--documented", help="Does a thing.")
    parser.add_argument("--undocumented")

    missing = [
        "/".join(action.option_strings) for action in _actions(parser).values() if not (action.help or "").strip()
    ]

    assert missing == ["--undocumented"]


def _assert_cli_architecture_contract(document: str):
    section = re.search(r"^## CLI command triage \((\d+) subcommands\)\n(.*?)(?=^## |\Z)", document, re.M | re.S)
    assert section is not None, "Missing CLI command triage section"
    groups = re.findall(r"^- \*\*(\w+) \((\d+)\)\*\*: (.*?)(?=^- |\Z)", section[2], re.M | re.S)
    assert len(groups) == len(SUBCOMMAND_GROUPS)
    assert {name for name, _count, _body in groups} == set(SUBCOMMAND_GROUPS)
    documented_commands = []
    for name, count, body in groups:
        commands = [command.strip() for command in body.partition(" — ")[0].strip().removesuffix(".").split(",")]
        assert int(count) == len(commands) == len(set(commands)), name
        assert set(commands) == SUBCOMMAND_GROUPS[name], name
        documented_commands.extend(commands)
    _parser, subcommands = _parser_contract()
    assert int(section[1]) == len(documented_commands) == len(subcommands)
    assert set(documented_commands) == set(subcommands)


def test_architecture_cli_triage_matches_parser_and_ownership():
    document = (Path(cli.__file__).parent / "ARCHITECTURE.md").read_text(encoding="utf-8")

    _assert_cli_architecture_contract(document)


@pytest.mark.parametrize(
    ("original", "replacement"),
    [
        ("40 subcommands", "38 subcommands"),
        ("Kernel (29)", "Kernel (27)"),
        ("infer-launch, ", ""),
        ("infer-launch, ", "unknown-command, "),
        ("infer-launch, infer-stop", "infer-launch, infer-launch"),
        ("**Domain (7)**", "**Kernel (7)**"),
        ("**Mixed (4)**", "**Other (4)**"),
        ("skills, doctor, context, plan", "skills, doctor, context, infer-stop"),
    ],
)
def test_architecture_cli_triage_guard_rejects_drift(original: str, replacement: str):
    document = (Path(cli.__file__).parent / "ARCHITECTURE.md").read_text(encoding="utf-8")
    assert original in document

    with pytest.raises(AssertionError):
        _assert_cli_architecture_contract(document.replace(original, replacement, 1))


def test_architecture_cli_triage_guard_rejects_omission_with_matching_counts():
    document = (Path(cli.__file__).parent / "ARCHITECTURE.md").read_text(encoding="utf-8")
    document = document.replace("40 subcommands", "39 subcommands").replace("Kernel (29)", "Kernel (28)")
    document = document.replace("infer-launch, ", "", 1)

    with pytest.raises(AssertionError):
        _assert_cli_architecture_contract(document)


def test_experiment_status_cli_contract():
    parser, subcommands = _parser_contract()
    actions = _actions(subcommands["experiment-status"])
    args = parser.parse_args(["experiment-status", "--run-dir", "experiment"])

    assert {name for name, action in actions.items() if action.required} == {"run_dir"}
    assert set(actions) - {"help"} == {"run_dir", "remote", "json"}
    assert args.remote is None
    assert args.json is False


def test_runtime_sync_cli_defaults_to_dry_run_and_documents_in_place_fast_forward(monkeypatch):
    parser, subcommands = _parser_contract()
    runtime_sync = subcommands["runtime-sync"]
    actions = _actions(runtime_sync)
    args = parser.parse_args(["runtime-sync", "--workdir", "runtime"])

    assert {name for name, action in actions.items() if action.required} == {"workdir"}
    assert set(actions) - {"help"} == {"workdir", "host", "python", "execute"}
    assert args.workdir == "runtime"
    assert args.host is None
    assert args.python == "python3"
    assert args.execute is False
    assert "Dry run unless --execute is given" in (_subcommand_help(parser, "runtime-sync") or "")
    assert "Never launches work" in (_subcommand_help(parser, "runtime-sync") or "")
    assert "fast-forward" in runtime_sync.format_help()
    assert "without cloning or resetting" in actions["execute"].help

    calls = []
    monkeypatch.setattr(
        cli,
        "sync_runtime",
        lambda workdir, *, host, remote_python, execute: calls.append((workdir, host, remote_python, execute))
        or {"status": "update_available", "executed": execute},
    )

    assert cli.main(["runtime-sync", "--workdir", "runtime"]) == 0
    assert calls == [("runtime", None, "python3", False)]


def test_experiment_note_cli_contract():
    parser, subcommands = _parser_contract()
    actions = _actions(subcommands["experiment-note"])
    args = parser.parse_args(["experiment-note", "--run-dir", "experiment", "--entry", "entry.yaml"])

    assert {name for name, action in actions.items() if action.required} == {"run_dir", "entry"}
    assert args.remote is None
    assert "local YAML file path" in actions["entry"].help
    assert "inline text is not accepted" in actions["entry"].help


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
    assert args.validate_only is False

    validated = parser.parse_args(["plan", "--recipe", "recipe.yaml", "--output-dir", "plan-dir", "--validate-only"])
    assert validated.validate_only is True


def test_doctor_reports_pid_phases_and_runtime_diagnostics(monkeypatch, capsys):
    report = plans.DecisionReport(status=plans.DecisionStatus.PASS, issues=[], decisions={})
    calls = []
    monkeypatch.setattr(
        cli,
        "evaluate_recipe",
        lambda *_args: calls.append("consultation") or ({"task": "hparam_tune"}, None, report),
    )
    monkeypatch.setattr(
        cli,
        "doctor_runtime_card",
        lambda _recipe: calls.append("runtime") or "Doctor runtime: python=/target/python",
    )
    monkeypatch.setattr(
        cli,
        "prepare_doctor_report",
        lambda *_args: calls.append("task diagnostics") or report,
    )
    monkeypatch.setattr(cli, "write_doctor_outputs", lambda *_args: calls.append("publish"))

    assert cli.main(["doctor", "--recipe", "recipe.yaml", "--output-dir", "doctor-out"]) == 0
    captured = capsys.readouterr()
    assert calls == ["consultation", "runtime", "task diagnostics", "publish"]
    assert captured.err.splitlines() == [
        f"Doctor started: pid={os.getpid()} recipe=recipe.yaml output_dir=doctor-out",
        "Doctor phase: consultation",
        "Doctor phase: runtime diagnostics",
        "Doctor runtime: python=/target/python",
        "Doctor phase: task diagnostics",
        "Doctor phase: publish outputs",
        "Doctor finished: exit_code=0",
    ]
    assert "Status: PASS" in captured.out


def test_doctor_skips_runtime_diagnostics_when_consultation_blocks(monkeypatch, capsys):
    issue = plans.DecisionIssue(plans.DecisionStatus.FAIL, "recipe", "blocked", None, {})
    report = plans.DecisionReport(status=plans.DecisionStatus.FAIL, issues=[issue], decisions={})
    monkeypatch.setattr(cli, "evaluate_recipe", lambda *_args: ({"task": "hparam_tune"}, None, report))
    monkeypatch.setattr(cli, "doctor_runtime_card", lambda _recipe: pytest.fail("unexpected runtime diagnostics"))
    monkeypatch.setattr(
        cli,
        "prepare_doctor_report",
        lambda *_args: report,
    )
    monkeypatch.setattr(cli, "write_doctor_outputs", lambda *_args: None)

    assert cli.main(["doctor", "--recipe", "recipe.yaml"]) == 1
    captured = capsys.readouterr()
    assert "Doctor phase: runtime diagnostics" not in captured.err
    assert "Doctor phase: task diagnostics" in captured.err
    assert "Doctor phase: publish outputs" in captured.err
    assert "Doctor finished: exit_code=1" in captured.err


def test_doctor_skips_hparam_runtime_probe_for_other_tasks(monkeypatch, capsys):
    report = plans.DecisionReport(status=plans.DecisionStatus.PASS, issues=[], decisions={})
    monkeypatch.setattr(cli, "evaluate_recipe", lambda *_args: ({"task": "finetune"}, None, report))
    monkeypatch.setattr(cli, "doctor_runtime_card", lambda _recipe: pytest.fail("unexpected runtime diagnostics"))
    monkeypatch.setattr(cli, "prepare_doctor_report", lambda *_args: report)
    monkeypatch.setattr(cli, "write_doctor_outputs", lambda *_args: None)

    assert cli.main(["doctor", "--recipe", "recipe.yaml"]) == 0
    captured = capsys.readouterr()
    assert "Doctor phase: runtime diagnostics" not in captured.err
    assert "Doctor phase: task diagnostics" in captured.err


def test_doctor_runtime_card_probes_target_versions_without_importing_lightning(monkeypatch):
    calls = []

    def run(execution, command):
        calls.append((execution, command))
        return subprocess.CompletedProcess(
            command,
            0,
            '{"host": "runtime-host", "python": "/opt/python", "python_version": "3.10.0", '
            '"pytorch_lightning_version": "2.6.1"}\n',
            "",
        )

    monkeypatch.setattr(managed_scheduler, "run_execution_command", run)
    recipe = {
        "task": "hparam_tune",
        "execution": {
            "target": "ssh",
            "host": "baichuan3",
            "workdir": "/runtime/repo",
            "python": "/opt/python",
        },
    }

    card = plans.doctor_runtime_card(recipe)

    assert card == (
        "Doctor runtime: transport=ssh:baichuan3, host=runtime-host, python=/opt/python, "
        "python_version=3.10.0, pytorch-lightning=2.6.1"
    )
    assert calls[0][0] is recipe["execution"]
    assert calls[0][1][:2] == ["/opt/python", "-c"]
    assert "import pytorch_lightning" not in calls[0][1][2]


def test_doctor_runtime_card_does_not_echo_timed_out_command(monkeypatch):
    def timeout(_execution, _command):
        raise subprocess.TimeoutExpired(["env", "SECRET_TOKEN=do-not-print"], 30)

    monkeypatch.setattr(managed_scheduler, "run_execution_command", timeout)
    recipe = {"task": "hparam_tune", "execution": {"python": "/opt/python"}}

    card = plans.doctor_runtime_card(recipe)

    assert card == "Doctor runtime unavailable: diagnostic probe timed out"
    assert "do-not-print" not in card


def test_doctor_runtime_card_uses_manager_python_by_default(monkeypatch):
    calls = []

    def run(_execution, command):
        calls.append(command)
        return subprocess.CompletedProcess(
            command,
            0,
            '{"host": "runtime-host", "python": "/opt/python", "python_version": "3.10.0", '
            '"pytorch_lightning_version": "2.6.1"}\n',
            "",
        )

    monkeypatch.setattr(managed_scheduler, "run_execution_command", run)

    plans.doctor_runtime_card({"task": "hparam_tune"})

    assert calls[0][0] == sys.executable


def test_doctor_runtime_card_handles_rejected_probe(monkeypatch):
    def reject(_execution, _command):
        raise ValueError("private fixture detail")

    monkeypatch.setattr(managed_scheduler, "run_execution_command", reject)

    card = plans.doctor_runtime_card({"task": "hparam_tune"})

    assert card == "Doctor runtime unavailable: diagnostic probe could not start"
    assert "private fixture detail" not in card


def test_doctor_runtime_card_does_not_echo_failed_probe_output(monkeypatch):
    result = subprocess.CompletedProcess(
        ["ssh", "runtime-host"],
        23,
        "SECRET_TOKEN=do-not-print\n",
        "private target path: /secret/path\n",
    )
    monkeypatch.setattr(managed_scheduler, "run_execution_command", lambda *_args: result)
    recipe = {"task": "hparam_tune", "execution": {"python": "/opt/python"}}

    card = plans.doctor_runtime_card(recipe)

    assert card == "Doctor runtime unavailable: diagnostic probe exited with code 23"
    assert "do-not-print" not in card
    assert "/secret/path" not in card


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


def test_hparam_launch_reports_dry_run_and_lifecycle_counts(tmp_path: Path, monkeypatch, capsys):
    manifest = tmp_path / "launch_manifest.tsv"
    write_rows(
        manifest,
        [
            {"step_id": "tune", "run_id": "run-001", "status": "planned"},
            {"step_id": "tune", "run_id": "run-002", "status": "planned"},
        ],
    )
    monkeypatch.setattr(cli, "launch_hparam_runs", lambda *_args, **_kwargs: manifest)

    assert cli.main(["hparam-launch", "--plan-dir", "plan-dir"]) == 0
    assert capsys.readouterr().out.splitlines() == [
        "Mode: dry-run (no launch attempted)",
        "Lifecycle states: planned=2",
        f"Wrote {manifest}",
    ]


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


def test_hparam_run_queue_reports_execute_and_lifecycle_counts(tmp_path: Path, monkeypatch, capsys):
    status = tmp_path / "run_status.tsv"
    write_rows(
        status,
        [
            {"step_id": "tune", "run_id": "run-001", "status": "completed"},
            {"step_id": "tune", "run_id": "run-002", "status": "failed"},
        ],
    )
    monkeypatch.setattr(cli, "run_hparam_queue", lambda *_args, **_kwargs: status)

    assert cli.main(["hparam-run-queue", "--plan-dir", "plan-dir", "--execute"]) == 0
    assert capsys.readouterr().out.splitlines() == [
        "Mode: execute (state changes enabled)",
        "Lifecycle states: completed=1, failed=1",
        f"Wrote {status}",
    ]


def test_hparam_monitor_cli_contract(tmp_path: Path, monkeypatch):
    parser, subcommands = _parser_contract()
    actions = _actions(subcommands["hparam-monitor"])
    defaults = parser.parse_args(["hparam-monitor", "--run-dir", "run-dir"])
    args = parser.parse_args(
        [
            "hparam-monitor",
            "--run-dir",
            "run-dir",
            "--once",
            "--health",
            "--include-log-tail",
            "--poll-seconds",
            "17",
        ]
    )
    status = tmp_path / "run_status.tsv"
    calls = []

    def monitor(run_dir, *, once, health, poll_seconds):
        calls.append((run_dir, once, health, poll_seconds))
        return status

    monkeypatch.setattr(cli, "monitor_hparam_runs", monitor)

    assert {name for name, action in actions.items() if action.required} == {"run_dir"}
    assert defaults.once is False
    assert defaults.health is False
    assert defaults.include_log_tail is False
    assert defaults.poll_seconds == 60
    assert args.include_log_tail is True
    assert cli._cmd_hparam_monitor(defaults) == 0
    assert cli._cmd_hparam_monitor(args) == 0
    assert calls == [
        ("run-dir", False, False, 60),
        ("run-dir", True, True, 17),
    ]


def test_hparam_monitor_requires_opt_in_for_raw_log_tail(tmp_path: Path, monkeypatch, capsys):
    status = tmp_path / "run_status.tsv"
    write_rows(
        status,
        [
            {
                "step_id": "tune",
                "run_id": "run-001",
                "status": "failed",
                "scheduler_reason": "NonZeroExitCode",
                "scheduler_health_error": "accounting unavailable",
                "log_path": "/logs/run-001.log",
                "log_tail": "rank 2: Traceback\nrank 2: CUDA out of memory",
            },
            {"step_id": "tune", "run_id": "run-002", "status": "completed"},
        ],
    )
    monkeypatch.setattr(cli, "monitor_hparam_runs", lambda *_args, **_kwargs: status)

    assert cli.main(["hparam-monitor", "--run-dir", "run-dir", "--once", "--health"]) == 0
    assert capsys.readouterr().out.splitlines() == [
        "Lifecycle states: completed=1, failed=1",
        "Failure evidence:",
        "- tune / run-001: status=failed; scheduler reason=NonZeroExitCode; "
        "scheduler health error=accounting unavailable; log=/logs/run-001.log",
        f"Wrote {status}",
    ]

    assert cli.main(["hparam-monitor", "--run-dir", "run-dir", "--once", "--health", "--include-log-tail"]) == 0
    assert capsys.readouterr().out.splitlines() == [
        "Lifecycle states: completed=1, failed=1",
        "Failure evidence:",
        "- tune / run-001: status=failed; scheduler reason=NonZeroExitCode; "
        "scheduler health error=accounting unavailable; log=/logs/run-001.log",
        "  rank 2: Traceback",
        "  rank 2: CUDA out of memory",
        f"Wrote {status}",
    ]


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
