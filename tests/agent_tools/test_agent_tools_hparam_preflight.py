from __future__ import annotations

import errno
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

from agent_tool_test_helpers import write_finetune_recipe, write_yaml
import pytest
import yaml

from agent_tools import cli, managed_scheduler, plan_hparam, plans, run_artifacts
from agent_tools.experiment_workspace import (
    ensure_experiment_workspace,
    file_sha256,
    merge_run_manifest,
    read_run_manifest,
)
from agent_tools.models import REPO_ROOT

_RUNTIME_COMMIT = subprocess.run(
    ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True, text=True, capture_output=True
).stdout.strip()
_PREFLIGHT_CARD_HEADING = "## Hparam Registration Preflight Provenance"


def _recipe(tmp_path: Path, *, task: str = "hparam_tune") -> tuple[Path, Path]:
    base = write_finetune_recipe(tmp_path / "source")
    if task != "hparam_tune":
        recipe_path = base
    else:
        recipe_path = write_yaml(
            tmp_path / "tune.yaml",
            {
                "name": "preflight_hparam",
                "task": "hparam_tune",
                "variant": "sleep2vec",
                "base_recipe": str(base),
                "search": {"method": "grid", "max_runs": 1, "parameters": {"runtime.lr": [1e-6]}},
                "execution": {
                    "workdir": str(tmp_path / "runtime"),
                    "python": sys.executable,
                    "runtime_commit": _RUNTIME_COMMIT,
                },
                "evaluation_policy": {
                    "selection_metric": "val_ahi_pearson",
                    "selection_mode": "max",
                    "selection_split": "val",
                    "final_eval_split": "test",
                    "external_test_locked": True,
                    "test_after_fit": False,
                    "final_test_unlocked": False,
                    "require_manual_unlock_for_final_test": True,
                },
                "decisions": {
                    "task": {"value": "hparam_tune", "source": "explicit_recipe"},
                    "label_name": {"value": "ahi", "source": "explicit_recipe"},
                    "external_test_locked": {"value": True, "source": "explicit_recipe"},
                    "train_val_test_policy": {"value": "val", "source": "explicit_recipe"},
                    "overwrite_policy": {"value": False, "source": "explicit_recipe"},
                    "final_eval_unlock": {"value": False, "source": "explicit_recipe"},
                },
            },
        )
    effective, _summary, _report = plans.evaluate_recipe(recipe_path)
    return recipe_path, Path(effective["experiment"]["root"])


def _snapshot(execution: dict, runs: list[dict]) -> dict:
    planned_argv = []
    modules = set()
    for run in runs:
        tokens = shlex.split(run["command"])
        module_index = tokens.index("-m") + 1
        modules.add(tokens[module_index])
        planned_argv.append({"run_id": run["run_id"], "args": tokens[module_index + 1 :]})
    module = modules.pop()
    return {
        "target": str(execution.get("target", "local") or "local"),
        "host": str(execution.get("host") or ""),
        "workdir": str(execution.get("workdir") or REPO_ROOT),
        "conda_env": str(execution.get("conda_env") or ""),
        "python_command": execution["python"],
        "expected_runtime_commit": execution["runtime_commit"],
        "execution_env_sha256": hashlib.sha256(b"{}").hexdigest(),
        "python": execution["python"],
        "python_version": "3.10.0",
        "runtime_commit": execution["runtime_commit"],
        "runtime_repo_root": str(REPO_ROOT),
        "runtime_hostname": "unit-host",
        "module": module,
        "module_origin": str(REPO_ROOT / Path(*module.split(".")).with_suffix(".py")),
        "required_options": [],
        "supported_options": [],
        "cli_options_sha256": hashlib.sha256(b"[]").hexdigest(),
        "validated_argv_sha256": hashlib.sha256(
            json.dumps(planned_argv, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }


def _workspace_files(workspace: Path) -> dict[Path, bytes]:
    return {path.relative_to(workspace): path.read_bytes() for path in workspace.rglob("*") if path.is_file()}


def _staging_dirs(workspace: Path, plan_dir: Path) -> list[Path]:
    return list(workspace.parent.rglob(f".{plan_dir.name}.*.staging"))


def _preflight_card(text: str) -> str:
    return text[text.index(_PREFLIGHT_CARD_HEADING) :].strip()


@pytest.mark.parametrize(
    ("execution", "expected_remote"),
    [
        ({"target": "local"}, None),
        ({"target": "ssh", "host": "frozen-host"}, "frozen-host"),
    ],
)
def test_hparam_output_inventory_routes_to_execution_host(
    tmp_path: Path,
    monkeypatch,
    execution: dict,
    expected_remote: str | None,
):
    plan_dir = tmp_path / "plan"
    results_path = tmp_path / "shared-results.csv"
    run_000_runtime = tmp_path / "runtime" / "run-000"
    run_001_runtime = tmp_path / "runtime" / "run-001"
    plan = {
        "recipe": {
            "execution": execution,
            "artifacts": {"results_csv_path": str(results_path)},
        },
        "runs": [
            {
                "run_id": "run-000",
                "runtime_dir": str(run_000_runtime),
                "checkpoint_dir": str(run_000_runtime / "checkpoints"),
            },
            {
                "run_id": "run-001",
                "runtime_dir": str(run_001_runtime),
                "checkpoint_dir": str(run_001_runtime / "checkpoints"),
            },
        ],
    }
    observed = []

    def validate(root, paths, *, remote=None):
        observed.append((Path(root), [Path(path) for path in paths], remote))

    monkeypatch.setattr(plan_hparam.exp_io, "validate_managed_output_paths", validate)

    plan_hparam.validate_hparam_output_paths(plan_dir, plan)

    expected = [
        plan_dir / "plan.json",
        run_000_runtime,
        run_000_runtime / "checkpoints",
        run_001_runtime,
        run_001_runtime / "checkpoints",
        results_path,
    ]
    assert observed == [(Path("/"), expected, expected_remote)]


def test_hparam_output_inventory_rejects_duplicate_run_paths(tmp_path: Path):
    plan_dir = tmp_path / "plan"
    shared_runtime = tmp_path / "runtime" / "shared"
    plan = {
        "recipe": {"execution": {"target": "local"}},
        "runs": [
            {"runtime_dir": str(shared_runtime), "checkpoint_dir": str(shared_runtime / "checkpoints-0")},
            {"runtime_dir": str(shared_runtime), "checkpoint_dir": str(shared_runtime / "checkpoints-1")},
        ],
    }

    with pytest.raises(ValueError, match="independent regular files"):
        plan_hparam.validate_hparam_output_paths(plan_dir, plan)


def test_hparam_plan_preflights_before_registration(tmp_path: Path, monkeypatch):
    recipe, workspace = _recipe(tmp_path)
    plan_dir = workspace / "plans" / "tune"
    calls = []

    def inspect(execution, runs, *, plan_label):
        calls.append((execution, runs, plan_label))
        assert all(Path(run["script"]).name == "launch.sh" for run in runs)
        assert all(str(plan_dir) in run["command"] for run in runs)
        return _snapshot(execution, runs)

    monkeypatch.setattr(managed_scheduler, "inspect_execution_target", inspect)

    report = plans.build_plan(recipe_path=recipe, output_dir=plan_dir)

    assert report.exit_code == 0
    assert len(calls) == 2
    assert (plan_dir / "execution_snapshot.json").is_file()
    plan = run_artifacts.read_hparam_plan(plan_dir)
    assert plan["execution_snapshot"]["sha256"] == file_sha256(plan_dir / "execution_snapshot.json")
    assert len(read_run_manifest(workspace)) == 1


@pytest.mark.parametrize("output_type", ["symlink", "hardlink", "directory", "fifo"])
def test_hparam_plan_rejects_unsafe_results_before_registration(tmp_path: Path, monkeypatch, output_type: str):
    recipe, workspace = _recipe(tmp_path)
    plan_dir = workspace / "plans" / "tune"
    results_path = tmp_path / "unsafe-results" / "results.csv"
    results_path.parent.mkdir()
    if output_type == "directory":
        results_path.mkdir()
    elif output_type == "fifo":
        os.mkfifo(results_path)
    else:
        source = tmp_path / "results-source.csv"
        source.write_text("metric\n1\n")
        if output_type == "symlink":
            results_path.symlink_to(source)
        else:
            os.link(source, results_path)
    payload = yaml.safe_load(recipe.read_text())
    payload["artifacts"] = {**payload.get("artifacts", {}), "results_csv_path": str(results_path)}
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
    before = _workspace_files(workspace)
    monkeypatch.setattr(
        managed_scheduler,
        "inspect_execution_target",
        lambda *_args, **_kwargs: pytest.fail("unsafe output topology must fail before target inspection"),
    )

    report = plans.build_plan(recipe_path=recipe, output_dir=plan_dir)

    assert report.exit_code == 1
    assert "Managed output paths must be independent regular files" in report.blocking_issues()[0].message
    assert _workspace_files(workspace) == before
    assert read_run_manifest(workspace) == []
    assert not plan_dir.exists()
    assert _staging_dirs(workspace, plan_dir) == []


def test_hparam_plan_rejects_workdir_ancestor_symlink_before_registration(tmp_path: Path, monkeypatch):
    recipe, workspace = _recipe(tmp_path)
    plan_dir = workspace / "plans" / "tune"
    real_parent = tmp_path / "real-runtime-parent"
    real_parent.mkdir()
    alias_parent = tmp_path / "runtime-alias"
    alias_parent.symlink_to(real_parent, target_is_directory=True)
    payload = yaml.safe_load(recipe.read_text())
    payload["execution"]["workdir"] = str(alias_parent / "runtime")
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
    before = _workspace_files(workspace)
    monkeypatch.setattr(
        managed_scheduler,
        "inspect_execution_target",
        lambda *_args, **_kwargs: pytest.fail("unsafe output topology must fail before target inspection"),
    )

    report = plans.build_plan(recipe_path=recipe, output_dir=plan_dir)

    assert report.exit_code == 1
    assert str(alias_parent) in report.blocking_issues()[0].message
    assert _workspace_files(workspace) == before
    assert read_run_manifest(workspace) == []
    assert not plan_dir.exists()
    assert _staging_dirs(workspace, plan_dir) == []


@pytest.mark.parametrize(
    "remote_error",
    [
        RuntimeError("SSH output path validation failed on frozen-host: transport failed"),
        ValueError("Managed output paths must be independent regular files: /remote/alias"),
    ],
)
def test_hparam_plan_fails_closed_on_remote_output_preflight(
    tmp_path: Path,
    monkeypatch,
    remote_error: Exception,
):
    recipe, workspace = _recipe(tmp_path)
    plan_dir = workspace / "plans" / "tune"
    payload = yaml.safe_load(recipe.read_text())
    payload["execution"].update({"target": "ssh", "host": "frozen-host"})
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
    before = _workspace_files(workspace)
    real_validate = plan_hparam.exp_io.validate_managed_output_paths

    def validate(root, paths, *, remote=None):
        if remote is not None:
            assert remote == "frozen-host"
            raise remote_error
        return real_validate(root, paths)

    monkeypatch.setattr(plan_hparam.exp_io, "validate_managed_output_paths", validate)
    monkeypatch.setattr(
        managed_scheduler,
        "inspect_execution_target",
        lambda *_args, **_kwargs: pytest.fail("remote topology must fail before target inspection"),
    )

    report = plans.build_plan(recipe_path=recipe, output_dir=plan_dir)

    assert report.exit_code == 1
    assert str(remote_error) in report.blocking_issues()[0].message
    assert _workspace_files(workspace) == before
    assert read_run_manifest(workspace) == []
    assert not plan_dir.exists()
    assert _staging_dirs(workspace, plan_dir) == []


def test_hparam_preflight_does_not_probe_storage_capacity(tmp_path: Path, monkeypatch):
    recipe, workspace = _recipe(tmp_path)
    plan_dir = workspace / "plans" / "tune"
    monkeypatch.setattr(os, "statvfs", lambda *_args, **_kwargs: pytest.fail("statvfs must not be called"))
    monkeypatch.setattr(
        shutil,
        "disk_usage",
        lambda *_args, **_kwargs: pytest.fail("disk_usage must not be called"),
    )
    real_run = subprocess.run

    def reject_df(command, *args, **kwargs):
        argv = shlex.split(command) if isinstance(command, str) else command
        if argv and Path(str(argv[0])).name == "df":
            pytest.fail("df must not be called")
        return real_run(command, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", reject_df)
    monkeypatch.setattr(
        managed_scheduler,
        "inspect_execution_target",
        lambda execution, runs, **_kwargs: _snapshot(execution, runs),
    )

    report = plans.build_plan(recipe_path=recipe, output_dir=plan_dir)

    assert report.exit_code == 0
    assert (plan_dir / "plan.json").is_file()
    assert len(read_run_manifest(workspace)) == 1


def test_hparam_validate_only_uses_same_provenance_card_without_writes(tmp_path: Path, monkeypatch, capsys):
    recipe, workspace = _recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    payload["search"] = {
        "method": "grid",
        "max_runs": 3,
        "configurations": [
            {"runtime.lr": 1e-6},
            {
                "runtime.lr": 2e-6,
                "yaml:/model/backbone/name": "hf_bert",
                "yaml:/model/channels/0/input_dim": 16,
            },
            {"runtime.lr": 3e-6},
        ],
    }
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
    plan_dir = workspace / "plans" / "tune"
    calls = []
    before = _workspace_files(workspace)

    def inspect(execution, runs, *, plan_label):
        calls.append(plan_label)
        return _snapshot(execution, runs)

    monkeypatch.setattr(managed_scheduler, "inspect_execution_target", inspect)

    exit_code = cli.main(["plan", "--recipe", str(recipe), "--output-dir", str(plan_dir), "--validate-only"])
    validate_only_card = _preflight_card(capsys.readouterr().out)

    assert exit_code == 0
    assert calls == ["hparam", "hparam"]
    assert _workspace_files(workspace) == before
    assert not plan_dir.exists()
    assert _staging_dirs(workspace, plan_dir) == []

    exit_code = cli.main(["plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)])
    plan_stdout_card = _preflight_card(capsys.readouterr().out)

    assert exit_code == 0
    assert calls == ["hparam", "hparam", "hparam", "hparam"]
    assert plan_stdout_card == validate_only_card
    plan_markdown = (plan_dir / "plan.md").read_text()
    assert plan_markdown.count(_PREFLIGHT_CARD_HEADING) == 1
    assert _preflight_card(plan_markdown) == validate_only_card
    route_lines = [line for line in validate_only_card.splitlines() if line.startswith("| sleep2vec |")]
    assert len(route_lines) == 2
    assert "roformer (hidden_size=8)" in route_lines[0]
    assert "ppg (input_dim=8, tokenizer=linear, out_dim=8)" in route_lines[0]
    assert route_lines[0].endswith("| run-000, run-002 |")
    assert "hf_bert (hidden_size=8)" in route_lines[1]
    assert "ppg (input_dim=16, tokenizer=linear, out_dim=8)" in route_lines[1]
    assert route_lines[1].endswith("| run-001 |")

    plan = run_artifacts.read_hparam_plan(plan_dir)
    snapshot = json.loads((plan_dir / "execution_snapshot.json").read_text())
    planned_argv = []
    modules = set()
    for run in plan["runs"]:
        tokens = shlex.split(run["command"])
        module_index = tokens.index("-m") + 1
        modules.add(tokens[module_index])
        planned_argv.append({"run_id": run["run_id"], "args": tokens[module_index + 1 :]})
    expected_argv_sha256 = hashlib.sha256(
        json.dumps(planned_argv, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    assert modules == {"sleep2vec.finetune"}
    assert snapshot["module"] == "sleep2vec.finetune"
    assert snapshot["runtime_commit"] == snapshot["expected_runtime_commit"] == _RUNTIME_COMMIT
    assert snapshot["validated_argv_sha256"] == expected_argv_sha256
    assert snapshot["python"] in validate_only_card
    assert snapshot["module_origin"] in validate_only_card
    assert snapshot["runtime_commit"] in validate_only_card
    assert snapshot["validated_argv_sha256"] in validate_only_card
    assert "Validated run count: 3" in validate_only_card
    assert "Validated argv count: 3" in validate_only_card
    raw_plan = json.loads((plan_dir / "plan.json").read_text())
    assert raw_plan["execution_snapshot"]["sha256"] == file_sha256(plan_dir / "execution_snapshot.json")
    assert "preflight" not in raw_plan
    assert "route_card" not in raw_plan


def test_hparam_validate_only_needs_user_input_without_writes(tmp_path: Path, monkeypatch):
    recipe, workspace = _recipe(tmp_path)
    plan_dir = workspace / "plans" / "tune"
    effective, config, _report = plans.evaluate_recipe(recipe)
    unresolved = plans.DecisionReport(
        status=plans.DecisionStatus.NEEDS_USER_INPUT,
        issues=[
            plans.DecisionIssue(
                plans.DecisionStatus.NEEDS_USER_INPUT,
                "selection_metric",
                "Selection metric is unresolved.",
                "Which selection metric should be used?",
                {},
            )
        ],
        decisions={},
    )
    monkeypatch.setattr(plans, "preflight_plan", lambda **_kwargs: (effective, config, unresolved))
    before = _workspace_files(workspace)

    report = plans.build_plan(recipe_path=recipe, output_dir=plan_dir, validate_only=True)

    assert report.exit_code == 2
    assert _workspace_files(workspace) == before
    assert not plan_dir.exists()
    assert _staging_dirs(workspace, plan_dir) == []


def test_hparam_preflight_failure_leaves_no_plan_or_workspace(tmp_path: Path, monkeypatch):
    recipe, workspace = _recipe(tmp_path)
    plan_dir = workspace / "plans" / "tune"
    before = _workspace_files(workspace)
    monkeypatch.setattr(
        managed_scheduler,
        "inspect_execution_target",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("target argv rejected")),
    )

    report = plans.build_plan(recipe_path=recipe, output_dir=plan_dir)

    assert report.exit_code == 1
    assert "target argv rejected" in report.blocking_issues()[0].message
    assert _workspace_files(workspace) == before
    assert not plan_dir.exists()


def test_hparam_validate_only_failure_leaves_no_staging_or_canonical_state(tmp_path: Path, monkeypatch):
    recipe, workspace = _recipe(tmp_path)
    plan_dir = workspace / "plans" / "tune"
    before = _workspace_files(workspace)
    monkeypatch.setattr(
        managed_scheduler,
        "inspect_execution_target",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("target argv rejected")),
    )

    report = plans.build_plan(recipe_path=recipe, output_dir=plan_dir, validate_only=True)

    assert report.exit_code == 1
    assert "target argv rejected" in report.blocking_issues()[0].message
    assert _workspace_files(workspace) == before
    assert not plan_dir.exists()
    assert _staging_dirs(workspace, plan_dir) == []


def test_hparam_registration_recheck_rejects_target_drift_without_writes(tmp_path: Path, monkeypatch):
    recipe, workspace = _recipe(tmp_path)
    plan_dir = workspace / "plans" / "tune"
    before = _workspace_files(workspace)
    calls = 0

    def inspect(execution, runs, *, plan_label):
        nonlocal calls
        calls += 1
        snapshot = _snapshot(execution, runs)
        if calls == 2:
            snapshot["runtime_hostname"] = "changed-host"
        return snapshot

    monkeypatch.setattr(managed_scheduler, "inspect_execution_target", inspect)

    report = plans.build_plan(recipe_path=recipe, output_dir=plan_dir)

    assert report.exit_code == 1
    assert "Frozen execution snapshot changed" in report.blocking_issues()[0].message
    assert _workspace_files(workspace) == before
    assert not plan_dir.exists()


def test_hparam_registration_rejects_snapshot_drift_without_writes(tmp_path: Path, monkeypatch):
    recipe, workspace = _recipe(tmp_path)
    plan_dir = workspace / "plans" / "tune"
    before = _workspace_files(workspace)
    monkeypatch.setattr(
        managed_scheduler,
        "inspect_execution_target",
        lambda execution, runs, **_kwargs: _snapshot(execution, runs),
    )
    commit = plan_hparam.commit_hparam_plan

    def tamper_snapshot(out, **kwargs):
        (Path(out) / "execution_snapshot.json").write_text("{}\n")
        return commit(out, **kwargs)

    monkeypatch.setattr(plan_hparam, "commit_hparam_plan", tamper_snapshot)

    report = plans.build_plan(recipe_path=recipe, output_dir=plan_dir)

    assert report.exit_code == 1
    assert "execution snapshot changed" in report.blocking_issues()[0].message.lower()
    assert _workspace_files(workspace) == before
    assert not plan_dir.exists()


def test_hparam_registration_rejects_partial_canonical_rows_before_workspace_writes(tmp_path: Path, monkeypatch):
    recipe, workspace = _recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    payload["search"] = {
        "method": "grid",
        "max_runs": 2,
        "parameters": {"runtime.lr": [1e-6, 2e-6]},
    }
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
    plan_dir = workspace / "plans" / "tune"
    staging_dir = workspace / "plans" / ".tune.staging"
    effective, _config, preflight = plans.evaluate_recipe(recipe)
    assert preflight.exit_code == 0
    ensure_experiment_workspace(effective, plan_dir, register_step=False)
    events_path = workspace / "events.jsonl"
    events_before = events_path.read_bytes() if events_path.exists() else None
    calls = 0

    def inject_partial_registration(execution, runs, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            run = dict(runs[0])
            run["script"] = str(plan_dir / Path(run["script"]).relative_to(staging_dir))
            row = plan_hparam._hparam_manifest_rows({"runs": [run]})[0]
            merge_run_manifest(workspace, [row])
        return _snapshot(execution, runs)

    monkeypatch.setattr(managed_scheduler, "inspect_execution_target", inject_partial_registration)

    report = plans.build_plan(
        recipe_path=recipe,
        output_dir=plan_dir,
        staging_dir=staging_dir,
    )

    assert report.exit_code == 1
    assert "registration is partial" in report.blocking_issues()[0].message
    assert calls == 2
    assert not plan_dir.exists()
    assert not staging_dir.exists()
    assert not (workspace / "steps" / "unit-finetune" / "step.yaml").exists()
    assert (events_path.read_bytes() if events_path.exists() else None) == events_before
    assert [row["run_id"] for row in read_run_manifest(workspace)] == ["run-000"]


def test_hparam_registration_drift_preserves_existing_overwrite_destination(tmp_path: Path, monkeypatch):
    recipe, workspace = _recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    payload["artifacts"] = {**payload.get("artifacts", {}), "overwrite": True}
    payload["decisions"]["overwrite_policy"] = {"value": True, "source": "explicit_recipe"}
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
    plan_dir = workspace / "plans" / "tune"
    plan_dir.mkdir(parents=True)
    stale = plan_dir / "plan.json"
    sentinel = plan_dir / "unrelated.txt"
    stale.write_text('{"old": true}\n')
    sentinel.write_text("preserve me\n")
    before = _workspace_files(workspace)
    calls = 0

    def inspect(execution, runs, *, plan_label):
        nonlocal calls
        calls += 1
        snapshot = _snapshot(execution, runs)
        if calls == 2:
            snapshot["runtime_hostname"] = "changed-host"
        return snapshot

    monkeypatch.setattr(managed_scheduler, "inspect_execution_target", inspect)

    report = plans.build_plan(recipe_path=recipe, output_dir=plan_dir)

    assert report.exit_code == 1
    assert "Frozen execution snapshot changed" in report.blocking_issues()[0].message
    assert _workspace_files(workspace) == before
    assert stale.read_text() == '{"old": true}\n'
    assert sentinel.read_text() == "preserve me\n"


def test_hparam_staging_uses_destination_filesystem_ancestor(tmp_path: Path, monkeypatch):
    recipe, workspace = _recipe(tmp_path)
    plan_dir = workspace / "plans" / "tune"
    real_replace = Path.replace

    def reject_parent_filesystem_staging(path, target):
        if path.name.endswith(".staging") and path.parent == workspace.parent:
            raise OSError(errno.EXDEV, "cross-device link")
        return real_replace(path, target)

    monkeypatch.setattr(Path, "replace", reject_parent_filesystem_staging)
    monkeypatch.setattr(
        managed_scheduler,
        "inspect_execution_target",
        lambda execution, runs, **_kwargs: _snapshot(execution, runs),
    )

    report = plans.build_plan(recipe_path=recipe, output_dir=plan_dir)

    assert report.exit_code == 0
    assert plan_dir.is_dir()
    assert _staging_dirs(workspace, plan_dir) == []


def test_hparam_plan_directory_may_equal_experiment_root(tmp_path: Path, monkeypatch):
    recipe, _workspace = _recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    workspace = tmp_path / "workspace"
    payload["experiment"]["root"] = str(workspace)
    recipe = tmp_path / "root-plan.yaml"
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
    monkeypatch.setattr(
        managed_scheduler,
        "inspect_execution_target",
        lambda execution, runs, **_kwargs: _snapshot(execution, runs),
    )

    report = plans.build_plan(recipe_path=recipe, output_dir=workspace)

    assert report.exit_code == 0
    assert (workspace / "plan.json").is_file()
    assert (workspace / "experiment.yaml").is_file()
    assert (workspace / "run_manifest.tsv").is_file()
    assert run_artifacts.read_hparam_plan(workspace)["runs"][0]["run_id"] == "run-000"
    assert read_run_manifest(workspace)[0]["status"] == "planned"


def test_hparam_overwrite_publish_failure_restores_existing_plan(tmp_path: Path, monkeypatch):
    recipe, workspace = _recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    payload["artifacts"] = {**payload.get("artifacts", {}), "overwrite": True}
    payload["decisions"]["overwrite_policy"] = {"value": True, "source": "explicit_recipe"}
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
    plan_dir = workspace / "plans" / "tune"
    plan_dir.mkdir(parents=True)
    (plan_dir / "plan.json").write_text('{"old": true}\n')
    (plan_dir / "unrelated.txt").write_text("preserve me\n")
    before = _workspace_files(workspace)
    real_replace = Path.replace
    failed = False

    def fail_during_publish(path, target):
        nonlocal failed
        target = Path(target)
        if not failed and path.parent.name.endswith(".staging") and target == plan_dir / "plan.md":
            failed = True
            raise OSError(errno.EIO, "injected publish failure")
        return real_replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_during_publish)
    monkeypatch.setattr(
        managed_scheduler,
        "inspect_execution_target",
        lambda execution, runs, **_kwargs: _snapshot(execution, runs),
    )

    with pytest.raises(OSError, match="injected publish failure"):
        plans.build_plan(recipe_path=recipe, output_dir=plan_dir)

    assert failed is True
    assert _workspace_files(workspace) == before
    assert _staging_dirs(workspace, plan_dir) == []
    assert not list(workspace.parent.rglob(f".{plan_dir.name}.*.backup"))


def test_staged_plan_publish_rejects_existing_run_directory(tmp_path: Path):
    plan_dir = tmp_path / "plan"
    staging_dir = tmp_path / ".plan.staging"
    existing_run = plan_dir / "runs" / "run-000--unit"
    staged_run = staging_dir / "runs" / "run-000--unit"
    existing_run.mkdir(parents=True)
    staged_run.mkdir(parents=True)
    (plan_dir / "plan.json").write_text('{"old": true}\n')
    (existing_run / "run.json").write_text('{"old": true}\n')
    (staging_dir / "plan.json").write_text('{"new": true}\n')
    (staged_run / "run.json").write_text('{"new": true}\n')
    before = _workspace_files(tmp_path)

    with pytest.raises(ValueError, match="Published run directories are immutable: run-000--unit"):
        plans.publish_staged_plan_locked(staging_dir, plan_dir, out_preexisted=True)

    assert _workspace_files(tmp_path) == before
    assert not list(tmp_path.rglob(".plan.*.backup"))


def test_staged_plan_publish_failure_restores_appended_runs(tmp_path: Path, monkeypatch):
    plan_dir = tmp_path / "plan"
    staging_dir = tmp_path / ".plan.staging"
    existing_run = plan_dir / "runs" / "run-000--unit"
    existing_run.mkdir(parents=True)
    (plan_dir / "plan.json").write_text('{"old": true}\n')
    (plan_dir / "plan.md").write_text("old plan\n")
    (existing_run / "run.json").write_text('{"old": true}\n')
    for run_name in ("run-001--unit", "run-002--unit"):
        run_dir = staging_dir / "runs" / run_name
        run_dir.mkdir(parents=True)
        (run_dir / "run.json").write_text(f'{{"run": "{run_name}"}}\n')
    (staging_dir / "plan.json").write_text('{"new": true}\n')
    (staging_dir / "plan.md").write_text("new plan\n")
    before = _workspace_files(tmp_path)
    real_replace = Path.replace

    def fail_second_run(path, target):
        if path == staging_dir / "runs" / "run-002--unit":
            raise OSError(errno.EIO, "injected run publish failure")
        return real_replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_second_run)

    with pytest.raises(OSError, match="injected run publish failure"):
        plans.publish_staged_plan_locked(staging_dir, plan_dir, out_preexisted=True)

    assert _workspace_files(tmp_path) == before
    assert not list(tmp_path.rglob(".plan.*.backup"))


def test_hparam_plan_rejects_destination_appearing_during_preflight(tmp_path: Path, monkeypatch):
    recipe, workspace = _recipe(tmp_path)
    plan_dir = workspace / "plans" / "tune"
    preflight = plan_hparam.preflight_hparam_plan

    def create_destination(physical_out, *, semantic_out):
        snapshot = preflight(physical_out, semantic_out=semantic_out)
        plan_dir.mkdir(parents=True)
        (plan_dir / "other-session.txt").write_text("do not replace\n")
        return snapshot

    monkeypatch.setattr(plan_hparam, "preflight_hparam_plan", create_destination)
    monkeypatch.setattr(
        managed_scheduler,
        "inspect_execution_target",
        lambda execution, runs, **_kwargs: _snapshot(execution, runs),
    )

    with pytest.raises(ValueError, match="output changed during preflight"):
        plans.build_plan(recipe_path=recipe, output_dir=plan_dir)

    assert (plan_dir / "other-session.txt").read_text() == "do not replace\n"
    assert not (plan_dir / "plan.json").exists()
    assert read_run_manifest(workspace) == []
    assert _staging_dirs(workspace, plan_dir) == []


def test_hparam_plan_rechecks_blocked_artifacts_created_after_preflight(tmp_path: Path, monkeypatch):
    recipe, workspace = _recipe(tmp_path)
    plan_dir = workspace / "plans" / "tune"
    effective, _cfg, _report = plans.evaluate_recipe(recipe)
    ensure_experiment_workspace(effective, plan_dir, register_step=False)
    competitor = plan_dir / "decisions.yaml"
    validate_bound_recipe = plans._validate_bound_recipe

    def inject_competitor(*args, **kwargs):
        result = validate_bound_recipe(*args, **kwargs)
        plan_dir.mkdir(parents=True)
        competitor.write_text("user competitor\n")
        return result

    monkeypatch.setattr(plans, "_validate_bound_recipe", inject_competitor)
    monkeypatch.setattr(
        managed_scheduler,
        "inspect_execution_target",
        lambda execution, runs, **_kwargs: _snapshot(execution, runs),
    )

    report = plans.build_plan(recipe_path=recipe, output_dir=plan_dir)

    assert report.exit_code == 1
    assert competitor.read_text() == "user competitor\n"
    assert not (plan_dir / "plan.json").exists()
    assert not (plan_dir / "runs").exists()
    assert read_run_manifest(workspace) == []
    assert _staging_dirs(workspace, plan_dir) == []


@pytest.mark.parametrize("competitor_kind", ["file", "symlink"])
def test_hparam_plan_rolls_back_blocked_artifact_created_after_final_guard(
    tmp_path: Path,
    monkeypatch,
    competitor_kind: str,
):
    recipe, workspace = _recipe(tmp_path)
    plan_dir = workspace / "plans" / "tune"
    effective, _cfg, _report = plans.evaluate_recipe(recipe)
    ensure_experiment_workspace(effective, plan_dir, register_step=False)
    plan_dir.mkdir(parents=True)
    competitor = plan_dir / "decisions.yaml"
    guard_pass = plans._guard_pass_plan_publication
    guard_calls = 0

    def inject_after_final_guard(*args, **kwargs):
        nonlocal guard_calls
        guarded = guard_pass(*args, **kwargs)
        guard_calls += 1
        if guard_calls == 2:
            if competitor_kind == "symlink":
                target = tmp_path / "user-decisions.yaml"
                target.write_text("user competitor\n")
                competitor.symlink_to(target)
            else:
                competitor.write_text("user competitor\n")
        return guarded

    monkeypatch.setattr(plans, "_guard_pass_plan_publication", inject_after_final_guard)
    monkeypatch.setattr(
        managed_scheduler,
        "inspect_execution_target",
        lambda execution, runs, **_kwargs: _snapshot(execution, runs),
    )

    with pytest.raises(ValueError, match="planning artifacts|Managed output paths"):
        plans.build_plan(recipe_path=recipe, output_dir=plan_dir)

    assert competitor.read_text() == "user competitor\n"
    assert competitor.is_symlink() is (competitor_kind == "symlink")
    assert not (plan_dir / "plan.json").exists()
    assert read_run_manifest(workspace) == []
    assert _staging_dirs(workspace, plan_dir) == []


def test_validate_only_rejects_non_hparam_without_writes(tmp_path: Path):
    recipe, workspace = _recipe(tmp_path, task="finetune")
    plan_dir = workspace / "plans" / "finetune"
    before = _workspace_files(workspace)

    report = plans.build_plan(recipe_path=recipe, output_dir=plan_dir, validate_only=True)

    assert report.exit_code == 1
    assert any(issue.field == "validate_only" for issue in report.blocking_issues())
    assert _workspace_files(workspace) == before
    assert not plan_dir.exists()
