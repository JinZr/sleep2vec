from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import subprocess
import sys

import pytest

from agent_tools import cli, execution_snapshot, hparam_runtime, managed_scheduler, models, plans
from agent_tools.adapters.registry import get_adapter
from agent_tools.decisions import evaluate_consultation_gates
from agent_tools.manifests import write_rows
from agent_tools.recipes import load_consultation_policy

SUBCOMMAND_GROUPS = {
    "Kernel": {
        "repo-summary",
        "runtime-sync",
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
        "experiment-finalize",
        "experiment-monitor",
        "experiment-status",
        "hparam-stop",
        "hparam-select",
    },
    "Domain": {
        "config-summary",
        "index-summary",
        "preset-summary",
    },
    "Mixed": {"skills", "doctor", "plan"},
}

RUNNABLE_TASK_VARIANT_MATRIX = [
    ("sleep2stat", None, "sleep2stat"),
    ("preset_prepare", "sleep2vec", "preprocess/save_dataset_presets.py"),
    ("preset_prepare", "sleep2vec2", "sleep2vec2/preprocess/save_dataset_presets.py"),
    ("preset_prepare", "sleep2expert", "sleep2expert/preprocess/save_dataset_presets.py"),
    *[
        (task, variant, f"{variant}.{'finetune' if task in {'finetune', 'hparam_tune'} else 'infer'}")
        for task in ("finetune", "hparam_tune", "infer")
        for variant in models.SUPPORTED_VARIANTS
    ],
]

REJECTED_TASK_VARIANT_MATRIX = [
    *[(task, None) for task in ("preset_prepare", "finetune", "hparam_tune", "infer")],
    *[(task, "unsupported") for task in ("preset_prepare", "finetune", "hparam_tune", "infer")],
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


def test_cli_has_exactly_23_subcommands():
    _parser, subcommands = _parser_contract()

    assert set(subcommands) == set.union(*SUBCOMMAND_GROUPS.values())
    assert len(subcommands) == 23


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
        ("23 subcommands", "21 subcommands"),
        ("Kernel (17)", "Kernel (15)"),
        ("infer-launch, ", ""),
        ("infer-launch, ", "unknown-command, "),
        ("infer-launch, infer-stop", "infer-launch, infer-launch"),
        ("**Domain (3)**", "**Kernel (3)**"),
        ("**Mixed (3)**", "**Other (3)**"),
        ("skills, doctor, plan", "skills, doctor, infer-stop"),
    ],
)
def test_architecture_cli_triage_guard_rejects_drift(original: str, replacement: str):
    document = (Path(cli.__file__).parent / "ARCHITECTURE.md").read_text(encoding="utf-8")
    assert original in document

    with pytest.raises(AssertionError):
        _assert_cli_architecture_contract(document.replace(original, replacement, 1))


def test_architecture_cli_triage_guard_rejects_omission_with_matching_counts():
    document = (Path(cli.__file__).parent / "ARCHITECTURE.md").read_text(encoding="utf-8")
    document = document.replace("23 subcommands", "22 subcommands").replace("Kernel (17)", "Kernel (16)")
    document = document.replace("infer-launch, ", "", 1)

    with pytest.raises(AssertionError):
        _assert_cli_architecture_contract(document)


def test_experiment_status_cli_contract():
    parser, subcommands = _parser_contract()
    actions = _actions(subcommands["experiment-status"])
    args = parser.parse_args(["experiment-status", "--run-dir", "experiment"])

    assert {name for name, action in actions.items() if action.required} == {"run_dir"}
    assert set(actions) - {"help"} == {"run_dir", "json"}
    assert args.json is False


@pytest.mark.parametrize("command", ["experiment-status", "experiment-monitor", "experiment-note"])
def test_experiment_commands_reject_the_removed_remote_option(command: str):
    parser, _subcommands = _parser_contract()
    argv = [command, "--run-dir", "experiment", "--remote", "host"]
    if command == "experiment-note":
        argv += ["--entry", "entry.yaml"]

    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(argv)

    assert excinfo.value.code == 2


def test_cli_reports_expected_errors_as_one_line_and_exit_one(tmp_path: Path, capsys):
    assert cli.main(["hparam-monitor", "--run-dir", str(tmp_path)]) == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == f"error: Missing hparam plan: {tmp_path / 'plan.json'}\n"


def test_cli_reports_malformed_yaml_as_an_expected_error(tmp_path: Path, capsys):
    config = tmp_path / "broken.yaml"
    config.write_text("model: [unclosed\n")

    assert cli.main(["config-summary", "--config", str(config)]) == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("error: ")
    assert captured.err.count("\n") == 1


def test_cli_folds_multiline_errors_into_one_line(monkeypatch, capsys):
    def child_failure():
        raise RuntimeError("child command failed:\nTraceback (most recent call last):\n  boom\n")

    monkeypatch.setattr(cli, "repo_summary", child_failure)

    assert cli.main(["repo-summary", "--json"]) == 1
    assert capsys.readouterr().err == "error: child command failed: | Traceback (most recent call last): | boom\n"


def test_cli_reports_failed_subprocesses_as_expected_errors(monkeypatch, capsys):
    def failed_command():
        raise subprocess.CalledProcessError(1, ["git", "fetch", "origin"])

    monkeypatch.setattr(cli, "repo_summary", failed_command)

    assert cli.main(["repo-summary", "--json"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "error: Command '['git', 'fetch', 'origin']' returned non-zero exit status 1.\n"


def test_doctor_refusal_ends_stderr_with_the_error_line(tmp_path: Path, capsys):
    missing = tmp_path / "missing.yaml"

    assert cli.main(["doctor", "--recipe", str(missing), "--output-dir", str(tmp_path / "doctor")]) == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    lines = captured.err.splitlines()
    assert lines[0].startswith("Doctor started:")
    assert lines[-1].startswith("error: ") and str(missing) in lines[-1]
    assert sum(line.startswith("error: ") for line in lines) == 1


def test_cli_lets_programming_errors_raise(monkeypatch):
    def broken_summary():
        raise KeyError("missing field")

    monkeypatch.setattr(cli, "repo_summary", broken_summary)

    with pytest.raises(KeyError, match="missing field"):
        cli.main(["repo-summary", "--json"])


def test_runtime_sync_cli_defaults_to_dry_run_and_documents_in_place_fast_forward(monkeypatch):
    parser, subcommands = _parser_contract()
    runtime_sync = subcommands["runtime-sync"]
    actions = _actions(runtime_sync)
    args = parser.parse_args(["runtime-sync", "--workdir", "runtime"])

    assert {name for name, action in actions.items() if action.required} == {"workdir"}
    assert set(actions) - {"help"} == {"workdir", "execute"}
    assert args.workdir == "runtime"
    assert args.execute is False
    assert "Dry run unless --execute is given" in (_subcommand_help(parser, "runtime-sync") or "")
    assert "Never launches work" in (_subcommand_help(parser, "runtime-sync") or "")
    assert "fast-forward" in runtime_sync.format_help()
    assert "without cloning or resetting" in actions["execute"].help

    calls = []
    monkeypatch.setattr(
        cli,
        "sync_runtime",
        lambda workdir, *, execute: calls.append((workdir, execute))
        or {"status": "update_available", "executed": execute},
    )

    assert cli.main(["runtime-sync", "--workdir", "runtime"]) == 0
    assert calls == [("runtime", False)]


def test_experiment_note_cli_contract():
    parser, subcommands = _parser_contract()
    actions = _actions(subcommands["experiment-note"])
    args = parser.parse_args(["experiment-note", "--run-dir", "experiment", "--entry", "entry.yaml"])

    assert {name for name, action in actions.items() if action.required} == {"run_dir", "entry"}
    assert args.run_dir == "experiment"
    assert "local YAML file path" in actions["entry"].help
    assert "inline text is not accepted" in actions["entry"].help


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

    monkeypatch.setattr(execution_snapshot, "run_execution_command", run)
    recipe = {
        "task": "hparam_tune",
        "execution": {
            "workdir": "/runtime/repo",
            "python": "/opt/python",
        },
    }

    card = plans.doctor_runtime_card(recipe)

    assert card == (
        "Doctor runtime: host=runtime-host, python=/opt/python, " "python_version=3.10.0, pytorch-lightning=2.6.1"
    )
    assert calls[0][0] is recipe["execution"]
    assert calls[0][1][:2] == ["/opt/python", "-c"]
    assert "import pytorch_lightning" not in calls[0][1][2]


def test_doctor_runtime_card_does_not_echo_timed_out_command(monkeypatch):
    def timeout(_execution, _command):
        raise subprocess.TimeoutExpired(["env", "SECRET_TOKEN=do-not-print"], 30)

    monkeypatch.setattr(execution_snapshot, "run_execution_command", timeout)
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

    monkeypatch.setattr(execution_snapshot, "run_execution_command", run)

    plans.doctor_runtime_card({"task": "hparam_tune"})

    assert calls[0][0] == sys.executable


def test_doctor_runtime_card_handles_rejected_probe(monkeypatch):
    def reject(_execution, _command):
        raise ValueError("private fixture detail")

    monkeypatch.setattr(execution_snapshot, "run_execution_command", reject)

    card = plans.doctor_runtime_card({"task": "hparam_tune"})

    assert card == "Doctor runtime unavailable: diagnostic probe could not start"
    assert "private fixture detail" not in card


def test_doctor_runtime_card_does_not_echo_failed_probe_output(monkeypatch):
    result = subprocess.CompletedProcess(
        ["/opt/python", "-c", "probe"],
        23,
        "SECRET_TOKEN=do-not-print\n",
        "private target path: /secret/path\n",
    )
    monkeypatch.setattr(execution_snapshot, "run_execution_command", lambda *_args: result)
    recipe = {"task": "hparam_tune", "execution": {"python": "/opt/python"}}

    card = plans.doctor_runtime_card(recipe)

    assert card == "Doctor runtime unavailable: diagnostic probe exited with code 23"
    assert "do-not-print" not in card
    assert "/secret/path" not in card


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
    assert set(actions) - {"help"} == {"plan_dir", "dry_run", "execute", "foreground", "poll_seconds"}
    assert args.dry_run is True
    assert args.execute is False
    assert args.foreground is False
    assert args.poll_seconds == 60
    assert parser.parse_args(["hparam-run-queue", "--plan-dir", "plan-dir", "--execute", "--foreground"]).foreground
    summary = _subcommand_help(parser, "hparam-run-queue")
    assert summary is not None
    assert "Dry run unless --execute" in summary
    assert "--execute starts one detached runner per direct plan" in summary
    assert "--foreground runs the queue in this process" in summary

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
    monkeypatch.setattr(cli, "start_hparam_queue_runner", lambda *_args, **_kwargs: pytest.fail("no detached runner"))

    assert cli.main(["hparam-run-queue", "--plan-dir", "plan-dir", "--execute", "--foreground"]) == 0
    assert capsys.readouterr().out.splitlines() == [
        "Mode: execute (state changes enabled)",
        "Lifecycle states: completed=1, failed=1",
        f"Wrote {status}",
    ]


def test_hparam_run_queue_execute_reports_the_detached_runner(tmp_path: Path, monkeypatch, capsys):
    log_path = tmp_path / "queue_runner.log"
    calls = []

    def start(plan_dir, *, poll_seconds):
        calls.append((plan_dir, poll_seconds))
        return hparam_runtime.QueueRunnerStart(pid=4242, log_path=log_path)

    monkeypatch.setattr(cli, "start_hparam_queue_runner", start)
    monkeypatch.setattr(cli, "run_hparam_queue", lambda *_args, **_kwargs: pytest.fail("no foreground queue"))

    assert cli.main(["hparam-run-queue", "--plan-dir", "plan-dir", "--execute", "--poll-seconds", "5"]) == 0
    assert calls == [("plan-dir", 5.0)]
    assert capsys.readouterr().out.splitlines() == [
        "Mode: execute (detached queue runner started)",
        f"Queue runner: started; pid=4242; log={log_path}",
    ]


@pytest.mark.parametrize("execute", [False, True], ids=["dry-run", "slurm-execute"])
def test_hparam_run_queue_runs_in_process_without_a_detached_runner(tmp_path: Path, monkeypatch, capsys, execute: bool):
    status = tmp_path / "run_status.tsv"
    write_rows(status, [{"step_id": "tune", "run_id": "run-001", "status": "planned"}])
    queue_calls = []
    start_calls = []
    monkeypatch.setattr(
        cli, "run_hparam_queue", lambda plan_dir, *, dry_run, poll_seconds: queue_calls.append(dry_run) or status
    )
    # start_hparam_queue_runner returns None for a Slurm plan, whose queue keeps running in this process.
    monkeypatch.setattr(cli, "start_hparam_queue_runner", lambda *_args, **_kwargs: start_calls.append("start") or None)

    assert cli.main(["hparam-run-queue", "--plan-dir", "plan-dir", *(["--execute"] if execute else [])]) == 0

    assert queue_calls == [not execute]
    assert start_calls == (["start"] if execute else [])
    assert capsys.readouterr().out.splitlines()[0] == (
        "Mode: execute (state changes enabled)" if execute else "Mode: dry-run (no launch attempted)"
    )


def test_hparam_monitor_cli_contract(tmp_path: Path, monkeypatch):
    parser, subcommands = _parser_contract()
    actions = _actions(subcommands["hparam-monitor"])
    default_argv = ["hparam-monitor", "--run-dir", "run-dir"]
    argv = [
        "hparam-monitor",
        "--run-dir",
        "run-dir",
        "--once",
        "--health",
        "--include-log-tail",
        "--poll-seconds",
        "17",
    ]
    defaults = parser.parse_args(default_argv)
    args = parser.parse_args(argv)
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
    assert cli.main(default_argv) == 0
    assert cli.main(argv) == 0
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
        "Queue runner: not active; log=run-dir/queue_runner.log",
        "Failure evidence:",
        "- tune / run-001: status=failed; scheduler reason=NonZeroExitCode; "
        "scheduler health error=accounting unavailable; log=/logs/run-001.log",
        f"Wrote {status}",
    ]

    assert cli.main(["hparam-monitor", "--run-dir", "run-dir", "--once", "--health", "--include-log-tail"]) == 0
    assert capsys.readouterr().out.splitlines() == [
        "Lifecycle states: completed=1, failed=1",
        "Queue runner: not active; log=run-dir/queue_runner.log",
        "Failure evidence:",
        "- tune / run-001: status=failed; scheduler reason=NonZeroExitCode; "
        "scheduler health error=accounting unavailable; log=/logs/run-001.log",
        "  rank 2: Traceback",
        "  rank 2: CUDA out of memory",
        f"Wrote {status}",
    ]


@pytest.mark.parametrize("runner", ["active", "not-active", "slurm"])
def test_hparam_monitor_reports_queue_runner_liveness_from_the_lock_probe(
    tmp_path: Path, monkeypatch, capsys, runner: str
):
    status = tmp_path / "run_status.tsv"
    row = {"step_id": "tune", "run_id": "run-001", "status": "running"}
    if runner == "slurm":
        row["scheduler_type"] = "slurm"
    write_rows(status, [row])
    monkeypatch.setattr(cli, "monitor_hparam_runs", lambda *_args, **_kwargs: status)
    monkeypatch.setattr(cli, "start_hparam_queue_runner", lambda *_args, **_kwargs: pytest.fail("no runner start"))
    log_path = tmp_path / "queue_runner.log"

    if runner == "active":
        with managed_scheduler.queue_runner_lock(tmp_path, record_pid=True):
            assert cli.main(["hparam-monitor", "--run-dir", str(tmp_path), "--once"]) == 0
    else:
        assert cli.main(["hparam-monitor", "--run-dir", str(tmp_path), "--once"]) == 0

    runner_lines = [line for line in capsys.readouterr().out.splitlines() if line.startswith("Queue runner")]
    assert (
        runner_lines
        == {
            "active": [f"Queue runner: active; pid={os.getpid()}; log={log_path}"],
            "not-active": [f"Queue runner: not active; log={log_path}"],
            "slurm": [],
        }[runner]
    )
    # The probe never creates the lock file or a runner log.
    assert (tmp_path / "queue_runner.lock").exists() is (runner == "active")
    assert not log_path.exists()


@pytest.mark.parametrize("command", ["preset-launch", "preset-stop"])
def test_preset_cli_required_arguments_and_launch_default(command):
    parser, _subcommands = _parser_contract()
    args = [command, "--plan-dir", "/tmp/plan"]
    if command == "preset-stop":
        with pytest.raises(SystemExit):
            parser.parse_args(args)
        args += ["--reason", "authorized stop"]
    parsed = parser.parse_args(args)
    assert parsed.plan_dir == "/tmp/plan"
    if command == "preset-launch":
        assert parsed.execute is False


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
    adapter = get_adapter(task)
    if task == "sleep2stat":
        commands = adapter.commands(
            recipe,
            {"is_sleep2stat": True, "sleep2stat": {"run": {"output_dir": "runs/unit"}}},
        )
        assert any("python -m sleep2stat run" in command for command in commands)
        assert models.task_requires_variant(task) is False
        return
    if task == "preset_prepare":
        assert target in adapter.commands(recipe, None)[0]
    elif task == "hparam_tune":
        # Hparam plans compile finetune scripts separately, but use the same variant namespace resolver.
        assert models.module_for_variant(str(variant), "finetune") == target
    else:
        assert f"python -m {target}" in adapter.commands(recipe, None)[0]
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
