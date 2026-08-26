from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

from agent_tool_test_helpers import write_finetune_recipe, write_yaml
import pytest
import yaml

from agent_tools import managed_scheduler, plan_hparam, plans, run_artifacts
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
                    "workdir": str(REPO_ROOT),
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
    return list(workspace.parent.glob(f".{plan_dir.name}.*.staging"))


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


def test_hparam_validate_only_uses_same_preflight_without_writes(tmp_path: Path, monkeypatch):
    recipe, workspace = _recipe(tmp_path)
    plan_dir = workspace / "plans" / "tune"
    calls = []
    before = _workspace_files(workspace)

    def inspect(execution, runs, *, plan_label):
        calls.append(plan_label)
        return _snapshot(execution, runs)

    monkeypatch.setattr(managed_scheduler, "inspect_execution_target", inspect)

    report = plans.build_plan(recipe_path=recipe, output_dir=plan_dir, validate_only=True)

    assert report.exit_code == 0
    assert calls == ["hparam", "hparam"]
    assert _workspace_files(workspace) == before
    assert not plan_dir.exists()
    assert _staging_dirs(workspace, plan_dir) == []


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


def test_validate_only_rejects_non_hparam_without_writes(tmp_path: Path):
    recipe, workspace = _recipe(tmp_path, task="finetune")
    plan_dir = workspace / "plans" / "finetune"
    before = _workspace_files(workspace)

    report = plans.build_plan(recipe_path=recipe, output_dir=plan_dir, validate_only=True)

    assert report.exit_code == 1
    assert any(issue.field == "validate_only" for issue in report.blocking_issues())
    assert _workspace_files(workspace) == before
    assert not plan_dir.exists()
