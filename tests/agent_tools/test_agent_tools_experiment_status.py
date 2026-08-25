from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

from agent_tool_test_helpers import write_finetune_recipe, write_yaml
import pytest
import yaml

from agent_tools import (
    cli,
    decision_paths,
    decisions,
    experiment_io,
    experiment_tracking,
    experiments,
    plan_context,
    plans,
    recipes,
)
from agent_tools.adapters import all_adapters
from agent_tools.manifests import write_rows
from agent_tools.models import REPO_ROOT


def _init_workspace(root: Path) -> None:
    root.mkdir()
    experiment = {
        "id": "status-unit",
        "title": "Status unit experiment",
        "objective": "Exercise the read-only status contract.",
        "root": str(root),
        "baseline": "unit baseline",
    }
    (root / "experiment.yaml").write_text(yaml.safe_dump({"experiment": experiment}, sort_keys=False))
    (root / "run_manifest.tsv").write_text("step_id\trun_id\n")


def _add_plan(
    root: Path,
    *,
    step_id: str,
    status: str = "planned",
    task: str = "finetune",
    adaptive: bool = False,
    pipeline: bool = False,
    host: str | None = None,
) -> tuple[Path, dict[str, str]]:
    experiment = yaml.safe_load((root / "experiment.yaml").read_text())["experiment"]
    step = {"id": step_id, "phase": "train", "purpose": f"Run {step_id}."}
    plan_dir = root / "plans" / step_id
    run_dir = plan_dir / "runs" / "run-000--default"
    run_dir.mkdir(parents=True)
    config_path = run_dir / "config.yaml"
    script_path = run_dir / "launch.sh"
    artifacts_path = run_dir / "artifacts.json"
    recipe = {"task": task, "experiment": experiment, "step": step}
    if task != "sleep2stat":
        recipe["variant"] = "sleep2vec"
    if task == "hparam_tune":
        base_recipe = {
            "task": "finetune",
            "variant": "sleep2vec",
            "experiment": experiment,
            "step": step,
        }
        local_recipe = {
            "task": "hparam_tune",
            "variant": "sleep2vec",
            "base_recipe": "base.yaml",
            "experiment": experiment,
            "step": step,
        }
        recipe.update(
            {
                "base_recipe": "base.yaml",
                "_base_recipe": base_recipe,
                "_local_recipe": local_recipe,
            }
        )
    command = "python -m sleep2stat" if task == "sleep2stat" else "python -m sleep2vec.finetune"
    config_path.write_text("model: unit\n")
    script_path.write_text(f"#!/bin/sh\n{command}\n")
    artifacts_path.write_text("{}\n")
    config_sha256 = _sha256(config_path)
    script_sha256 = _sha256(script_path)
    run = {
        "experiment_id": experiment["id"],
        "step_id": step_id,
        "run_id": "run-000",
        "run_name": "default",
        "version": f"status-unit-{step_id}-run-000-default",
        "status": "planned",
        "config": str(config_path),
        "config_sha256": config_sha256,
        "script": str(script_path),
        "script_sha256": script_sha256,
        "run_dir": str(run_dir),
        "artifacts": str(artifacts_path),
        "runtime_dir": str(root / "runtime" / step_id),
        "checkpoint_dir": str(root / "runtime" / step_id / "checkpoints"),
    }
    if task == "hparam_tune":
        run["parameter_summary"] = "single resolved recipe"
    if host is not None:
        run["host"] = host
    if adaptive:
        recipe["adaptive"] = {"enabled": True, "suggest": {"strategy": "best_neighborhood"}}
        if task == "hparam_tune":
            recipe["_local_recipe"]["adaptive"] = recipe["adaptive"]
    resolved_text = yaml.safe_dump(recipe, sort_keys=False)
    resolved_path = plan_dir / "recipe.resolved.yaml"
    resolved_path.write_text(resolved_text)
    plan_run = {**run, "command": command}
    plan = {"status": "PASS", "recipe": recipe, "runs": [plan_run]}
    if task == "hparam_tune":
        plan["resolved_recipe_sha256"] = _sha256(resolved_path)
        (plan_dir / "run_all.sh").write_text(f"#!/bin/sh\n{command}\n")
    else:
        plan["commands"] = [command]
        (plan_dir / "run.sh").write_text(script_path.read_text())
    (plan_dir / "plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")

    step_dir = root / "steps" / step_id
    step_dir.mkdir(parents=True)
    (step_dir / "step.yaml").write_text(
        yaml.safe_dump(
            {
                "step": step,
                "experiment_id": experiment["id"],
                "recipe_path": "",
                "plans": [str(plan_dir)],
            },
            sort_keys=False,
        )
    )
    canonical = {
        **run,
        "status": status,
        "parameter_summary": run.get("parameter_summary", "single resolved recipe"),
    }
    if pipeline:
        canonical.update(
            {
                "pipeline_id": "pipeline-unit",
                "job_id": "job-unit",
                "attempt": "1",
                "result_root": str(root / "pipeline-results" / "job-unit" / "attempt-001"),
                "terminal_status_owner": "script",
            }
        )
    existing = _read_manifest_rows(root)
    write_rows(root / "run_manifest.tsv", [*existing, canonical])
    return plan_dir, canonical


def _read_manifest_rows(root: Path) -> list[dict[str, str]]:
    text = (root / "run_manifest.tsv").read_text().splitlines()
    if len(text) <= 1:
        return []
    import csv

    return list(csv.DictReader(text, delimiter="\t"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _workspace_files(root: Path) -> dict[str, bytes]:
    return {str(path.relative_to(root)): path.read_bytes() for path in sorted(root.rglob("*")) if path.is_file()}


def _write_public_hparam_recipe(root: Path, parameters: dict) -> Path:
    base_recipe = write_finetune_recipe(root)
    return write_yaml(
        root / "tune.yaml",
        {
            "name": "status_public_hparam",
            "task": "hparam_tune",
            "variant": "sleep2vec",
            "base_recipe": str(base_recipe),
            "inputs": {},
            "search": {"method": "grid", "max_runs": 1, "parameters": parameters},
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
                "train_val_test_policy": {"value": "select on val", "source": "explicit_recipe"},
                "overwrite_policy": {"value": False, "source": "explicit_recipe"},
                "final_eval_unlock": {"value": False, "source": "explicit_recipe"},
            },
        },
    )


def test_experiment_status_accepts_public_generic_plan_without_runtime_directories(tmp_path):
    root = tmp_path / "experiment"
    payload = yaml.safe_load((REPO_ROOT / "recipes/examples/tiny_fixture_sleep2stat.yaml").read_text())
    recipe = write_yaml(root / "sleep2stat.yaml", payload)
    plan_dir = root / "plans" / "sleep2stat"

    assert plans.build_plan(recipe_path=recipe, output_dir=plan_dir).exit_code == 0
    planned = json.loads((plan_dir / "plan.json").read_text())["runs"][0]
    assert planned["runtime_dir"] == planned["checkpoint_dir"] == ""

    snapshot = experiments.experiment_status(root)

    assert snapshot["summary"]["state"] == "ready_to_launch"
    assert snapshot["decision"]["recommended_next"]["argv"] == ["bash", str(plan_dir / "run.sh")]


def test_experiment_status_accepts_public_finetune_plan(tmp_path):
    root = tmp_path / "experiment"
    recipe = write_finetune_recipe(root)
    plan_dir = root / "plans" / "train"

    assert plans.build_plan(recipe_path=recipe, output_dir=plan_dir).exit_code == 0

    snapshot = experiments.experiment_status(root)

    assert snapshot["summary"]["state"] == "ready_to_launch"
    assert snapshot["decision"]["recommended_next"]["argv"] == ["bash", str(plan_dir / "run.sh")]


def test_experiment_status_rejects_supported_task_relabel_with_foreign_command(tmp_path):
    root = tmp_path / "experiment"
    _init_workspace(root)
    plan_dir, _canonical = _add_plan(root, step_id="train")

    plan = json.loads((plan_dir / "plan.json").read_text())
    plan["recipe"]["task"] = "infer"
    (plan_dir / "plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    resolved = yaml.safe_load((plan_dir / "recipe.resolved.yaml").read_text())
    resolved["task"] = "infer"
    (plan_dir / "recipe.resolved.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False))

    with pytest.raises(ValueError, match="task-owned entrypoint"):
        experiments.experiment_status(root)


@pytest.mark.parametrize(
    ("task", "variant", "field"),
    [
        ("preset_prepare", "sex_age_baseline", "variant"),
        ("sleep2stat", "sleep2vec", "variant"),
    ],
)
def test_experiment_status_rejects_invalid_frozen_recipe_structure(tmp_path, capsys, task, variant, field):
    root = tmp_path / "experiment"
    _init_workspace(root)
    plan_dir, _canonical = _add_plan(root, step_id="step")
    plan = json.loads((plan_dir / "plan.json").read_text())
    plan["recipe"].update({"task": task, "variant": variant})
    (plan_dir / "plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    resolved = yaml.safe_load((plan_dir / "recipe.resolved.yaml").read_text())
    resolved.update({"task": task, "variant": variant})
    (plan_dir / "recipe.resolved.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False))

    assert cli.main(["experiment-status", "--run-dir", str(root), "--json"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert field in captured.err
    assert "Traceback" not in captured.err


def test_experiment_status_rejects_unknown_frozen_internal_recipe_field(tmp_path):
    root = tmp_path / "experiment"
    _init_workspace(root)
    plan_dir, _canonical = _add_plan(root, step_id="train")
    plan = json.loads((plan_dir / "plan.json").read_text())
    plan["recipe"]["_private"] = True
    (plan_dir / "plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    resolved = yaml.safe_load((plan_dir / "recipe.resolved.yaml").read_text())
    resolved["_private"] = True
    (plan_dir / "recipe.resolved.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False))

    with pytest.raises(ValueError, match="unsupported internal fields"):
        experiments.experiment_status(root)


@pytest.mark.parametrize(
    ("layer", "field", "value"),
    [("_base_recipe", "adaptive", {}), ("_local_recipe", "unknown", True)],
)
def test_experiment_status_rejects_layered_hparam_structure_drift(tmp_path, layer, field, value):
    root = tmp_path / "experiment"
    _init_workspace(root)
    plan_dir, _canonical = _add_plan(root, step_id="tune", task="hparam_tune")
    plan = json.loads((plan_dir / "plan.json").read_text())
    plan["recipe"][layer][field] = value
    resolved = yaml.safe_load((plan_dir / "recipe.resolved.yaml").read_text())
    resolved[layer][field] = value
    resolved_path = plan_dir / "recipe.resolved.yaml"
    resolved_path.write_text(yaml.safe_dump(resolved, sort_keys=False))
    plan["resolved_recipe_sha256"] = _sha256(resolved_path)
    (plan_dir / "plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="Invalid registered plan recipe"):
        experiments.experiment_status(root)


@pytest.mark.parametrize("section", [None, "execution"])
def test_experiment_status_rejects_layered_hparam_effective_structure_drift(tmp_path, section):
    root = tmp_path / "experiment"
    _init_workspace(root)
    plan_dir, _canonical = _add_plan(root, step_id="tune", task="hparam_tune")
    plan = json.loads((plan_dir / "plan.json").read_text())
    resolved = yaml.safe_load((plan_dir / "recipe.resolved.yaml").read_text())
    if section is None:
        plan["recipe"]["unknown"] = True
        resolved["unknown"] = True
    else:
        plan["recipe"].setdefault(section, {})["unknown"] = True
        resolved.setdefault(section, {})["unknown"] = True
    resolved_path = plan_dir / "recipe.resolved.yaml"
    resolved_path.write_text(yaml.safe_dump(resolved, sort_keys=False))
    plan["resolved_recipe_sha256"] = _sha256(resolved_path)
    (plan_dir / "plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="Invalid registered plan recipe"):
        experiments.experiment_status(root)


def test_experiment_status_rejects_finetune_plan_without_runtime_directories(tmp_path):
    root = tmp_path / "experiment"
    _init_workspace(root)
    plan_dir, canonical = _add_plan(root, step_id="train", task="finetune")
    plan = json.loads((plan_dir / "plan.json").read_text())
    for row in (plan["runs"][0], canonical):
        row["runtime_dir"] = ""
        row["checkpoint_dir"] = ""
    (plan_dir / "plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    write_rows(root / "run_manifest.tsv", [canonical])

    with pytest.raises(ValueError, match="runtime_dir, checkpoint_dir"):
        experiments.experiment_status(root)


def test_experiment_status_rejects_unsupported_registered_plan_task(tmp_path, capsys):
    root = tmp_path / "experiment"
    _init_workspace(root)
    plan_dir, canonical = _add_plan(root, step_id="train", task="finetune")
    plan = json.loads((plan_dir / "plan.json").read_text())
    plan["recipe"]["task"] = "unsupported_task"
    for row in (plan["runs"][0], canonical):
        row["runtime_dir"] = ""
        row["checkpoint_dir"] = ""
    (plan_dir / "plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    resolved = yaml.safe_load((plan_dir / "recipe.resolved.yaml").read_text())
    resolved["task"] = "unsupported_task"
    (plan_dir / "recipe.resolved.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False))
    write_rows(root / "run_manifest.tsv", [canonical])

    with pytest.raises(ValueError, match="(?i)unsupported.*task"):
        experiments.experiment_status(root)
    assert cli.main(["experiment-status", "--run-dir", str(root), "--json"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "run-plan" not in captured.err


@pytest.mark.parametrize("parameters", [{"runtime.lr": [1e-6]}, {"yaml:/data/finetune_preset_path": [None]}])
def test_experiment_status_accepts_public_layered_hparam_plan(tmp_path, parameters):
    root = tmp_path / "experiment"
    recipe = _write_public_hparam_recipe(root, parameters)
    plan_dir = root / "plans" / "tune"

    assert plans.build_plan(recipe_path=recipe, output_dir=plan_dir).exit_code == 0
    plan = json.loads((plan_dir / "plan.json").read_text())
    resolved = yaml.safe_load((plan_dir / "recipe.resolved.yaml").read_text())
    assert "_base_recipe" in plan["recipe"] and "_local_recipe" in resolved
    parameter = next(iter(parameters))
    if parameters[parameter] == [None]:
        assert plan["runs"][0][parameter] is None
        assert _read_manifest_rows(root)[0][parameter] == ""

    snapshot = experiments.experiment_status(root)

    assert snapshot["summary"]["state"] == "ready_to_launch"


def test_experiment_status_requires_hparam_resolved_recipe_digest(tmp_path, capsys):
    root = tmp_path / "experiment"
    _init_workspace(root)
    plan_dir, _canonical = _add_plan(root, step_id="tune", task="hparam_tune")
    plan = json.loads((plan_dir / "plan.json").read_text())
    del plan["resolved_recipe_sha256"]
    (plan_dir / "plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="hparam recipe SHA-256"):
        experiments.experiment_status(root)
    assert cli.main(["experiment-status", "--run-dir", str(root), "--json"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "hparam-run-queue" not in captured.err


def test_experiment_status_rejects_missing_declared_blank_hparam_key(tmp_path):
    root = tmp_path / "experiment"
    parameter = "yaml:/data/finetune_preset_path"
    recipe = _write_public_hparam_recipe(root, {parameter: [None]})
    plan_dir = root / "plans" / "tune"
    assert plans.build_plan(recipe_path=recipe, output_dir=plan_dir).exit_code == 0
    plan = json.loads((plan_dir / "plan.json").read_text())
    del plan["runs"][0][parameter]
    (plan_dir / "plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="Workspace run parameters differ from plan"):
        experiments.experiment_status(root)


def test_experiment_status_allows_unrelated_blank_parameter_columns(tmp_path):
    root = tmp_path / "experiment"
    parameter = "yaml:/data/finetune_preset_path"
    recipe = _write_public_hparam_recipe(root, {parameter: [None]})
    plan_dir = root / "plans" / "tune"
    assert plans.build_plan(recipe_path=recipe, output_dir=plan_dir).exit_code == 0
    _add_plan(root, step_id="analyze", task="sleep2stat")

    generic_row = next(row for row in _read_manifest_rows(root) if row["step_id"] == "analyze")
    assert generic_row[parameter] == ""

    snapshot = experiments.experiment_status(root)

    assert snapshot["summary"]["state"] == "ready_to_launch"
    assert snapshot["decision"]["manual_choice_required"] is True


@pytest.mark.parametrize(
    ("drift", "error"),
    [
        ("missing_sha256", "final_eval_config must define"),
        ("file_drift", "frozen file SHA-256 changed"),
    ],
)
def test_experiment_status_requires_final_eval_config_integrity(tmp_path, drift, error):
    root = tmp_path / "experiment"
    _init_workspace(root)
    plan_dir, _row = _add_plan(root, step_id="tune", task="hparam_tune")
    final_config = plan_dir / "final_eval_config.frozen.yaml"
    final_config.write_text("model: unit\n")
    plan = json.loads((plan_dir / "plan.json").read_text())
    plan["final_eval_config"] = {
        "path": str(final_config),
        "sha256": _sha256(final_config),
        "source_path": str(final_config),
    }
    (plan_dir / "plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    assert experiments.experiment_status(root)["summary"]["state"] == "ready_to_launch"

    if drift == "missing_sha256":
        del plan["final_eval_config"]["sha256"]
        (plan_dir / "plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    else:
        final_config.write_text("model: changed\n")

    with pytest.raises(ValueError, match=error):
        experiments.experiment_status(root)


@pytest.mark.parametrize(
    "drift",
    ["partial_runtime", "hparam_parameter_summary", "input_snapshots", "command"],
)
def test_experiment_status_rejects_incomplete_registered_plan_identity(tmp_path, drift):
    root = tmp_path / "experiment"
    _init_workspace(root)
    task = "hparam_tune" if drift in {"hparam_parameter_summary", "command"} else "sleep2stat"
    plan_dir, canonical = _add_plan(root, step_id="train", task=task)
    plan = json.loads((plan_dir / "plan.json").read_text())
    if drift == "partial_runtime":
        plan["runs"][0]["runtime_dir"] = ""
        canonical["runtime_dir"] = ""
    elif drift == "hparam_parameter_summary":
        del plan["runs"][0]["parameter_summary"]
    elif drift == "command":
        del plan["runs"][0]["command"]
    else:
        canonical["input_snapshots"] = [{"field": "inputs.config", "path": canonical["config"], "sha256": "0" * 64}]
    (plan_dir / "plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    write_rows(root / "run_manifest.tsv", [canonical])

    with pytest.raises(ValueError):
        experiments.experiment_status(root)


def test_experiment_status_snapshot_is_deterministic_and_keeps_recorded_evidence(tmp_path):
    root = tmp_path / "experiment"
    _init_workspace(root)
    _plan, canonical = _add_plan(root, step_id="train", status="unknown_scheduler", task="hparam_tune")
    canonical.update(
        {
            "scheduler_raw_state": "MISSING",
            "scheduler_reason": "Scheduler identity is incomplete.",
            "scheduler_observed_at": "2026-08-25T01:02:03Z",
            "checkpoint_count": "50",
            "run_manifest": str(root / "runtime" / "train" / "run_manifest.json"),
        }
    )
    write_rows(root / "run_manifest.tsv", [canonical])

    first = experiments.experiment_status(root)
    second = experiments.experiment_status(root)

    assert first == second
    assert set(first) == {
        "experiment",
        "lifecycle_source",
        "live_observation",
        "summary",
        "steps",
        "runs",
        "blockers",
        "decision",
    }
    assert "schema_version" not in first
    assert "generated_at" not in first
    assert first["summary"] == {"state": "blocked", "run_count": 1, "status_counts": {"unknown_scheduler": 1}}
    assert first["runs"][0]["scheduler"]["observed_at"] == "2026-08-25T01:02:03Z"
    assert first["runs"][0]["process"]["pid"] is None
    assert first["runs"][0]["evidence"]["checkpoint_count"] == "50"
    assert first["decision"]["recommended_next"]["id"] == "experiment-monitor"
    assert first["decision"]["blocked_actions"] == ["adaptive_advance", "finalize", "resubmit"]


def test_experiment_status_snapshot_is_independent_of_input_order():
    root = Path("/experiment")
    rows = [
        {"step_id": step_id, "run_id": "run-000", "run_name": "default", "status": "planned"}
        for step_id in ("second", "first")
    ]
    registered_steps = [
        {
            "manifest": {"step": {"id": row["step_id"], "phase": "train", "purpose": "Run the fixture."}},
            "plans": [
                {
                    "path": str(root / "plans" / row["step_id"]),
                    "task": "finetune",
                    "adaptive": False,
                    "pipeline": False,
                    "run_keys": [(row["step_id"], row["run_id"])],
                    "launch_script": str(root / "plans" / row["step_id"] / "run.sh"),
                }
            ],
        }
        for row in rows
    ]
    experiment = {"id": "status-unit", "title": "Status unit"}

    first = experiment_tracking.experiment_status_snapshot(experiment, registered_steps, rows, root=root)
    second = experiment_tracking.experiment_status_snapshot(
        experiment,
        list(reversed(registered_steps)),
        list(reversed(rows)),
        root=root,
    )

    assert first == second


def test_experiment_status_rejects_contradictory_scheduler_identity(tmp_path):
    root = tmp_path / "experiment"
    _init_workspace(root)
    _plan, canonical = _add_plan(root, step_id="train")
    canonical["scheduler_job_id"] = "123"
    write_rows(root / "run_manifest.tsv", [canonical])

    with pytest.raises(ValueError, match="Direct managed run cannot define Slurm scheduler identity"):
        experiments.experiment_status(root)


@pytest.mark.parametrize("status", ["planned", "pending"])
@pytest.mark.parametrize("field", ["pid", "process_group_id", "process_start_token"])
def test_experiment_status_rejects_process_identity_on_launchable_direct_runs(tmp_path, status, field):
    root = tmp_path / "experiment"
    _init_workspace(root)
    _plan, canonical = _add_plan(root, step_id="train", status=status)
    canonical[field] = "42"
    write_rows(root / "run_manifest.tsv", [canonical])

    with pytest.raises(ValueError, match="Launchable direct managed run cannot define PID process identity"):
        experiments.experiment_status(root)


@pytest.mark.parametrize("status", ["unknown_scheduler", "unknown_remote", "missing_pid", "submitting"])
def test_experiment_status_uncertain_states_recommend_only_monitor(tmp_path, status):
    root = tmp_path / "experiment"
    _init_workspace(root)
    _add_plan(root, step_id="train", status=status)

    snapshot = experiments.experiment_status(root)

    assert snapshot["summary"]["state"] == "blocked"
    assert snapshot["decision"]["recommended_next"]["id"] == "experiment-monitor"
    assert snapshot["decision"]["other_legal_actions"] == []
    assert snapshot["decision"]["blocked_actions"] == ["adaptive_advance", "finalize", "resubmit"]


@pytest.mark.parametrize("status", ["queued", "launched", "running", "stopping"])
def test_experiment_status_active_states_recommend_monitor(tmp_path, status):
    root = tmp_path / "experiment"
    _init_workspace(root)
    _add_plan(root, step_id="train", status=status)

    snapshot = experiments.experiment_status(root)

    assert snapshot["summary"]["state"] == "in_progress"
    assert snapshot["decision"]["recommended_next"]["id"] == "experiment-monitor"
    assert "finalize" in snapshot["decision"]["blocked_actions"]
    assert "launch" in snapshot["decision"]["blocked_actions"]


def test_experiment_status_preserves_active_blockers_with_uncertain_runs(tmp_path):
    root = tmp_path / "experiment"
    _init_workspace(root)
    _add_plan(root, step_id="uncertain", status="unknown_scheduler")
    _add_plan(root, step_id="active", status="running")

    snapshot = experiments.experiment_status(root)
    blockers_by_step = {run["step_id"]: run["blockers"] for run in snapshot["runs"]}

    assert snapshot["summary"]["state"] == "blocked"
    assert snapshot["decision"]["recommended_next"]["id"] == "experiment-monitor"
    assert snapshot["decision"]["blocked_actions"] == ["adaptive_advance", "finalize", "launch", "resubmit"]
    assert blockers_by_step == {"active": ["active_runs"], "uncertain": ["unknown_scheduler"]}


def test_experiment_status_empty_workspace_does_not_guess_a_command(tmp_path):
    root = tmp_path / "experiment"
    _init_workspace(root)

    snapshot = experiments.experiment_status(root)

    assert snapshot["summary"] == {"state": "empty", "run_count": 0, "status_counts": {}}
    assert snapshot["blockers"][0]["code"] == "no_managed_runs"
    assert snapshot["decision"]["manual_choice_required"] is True
    assert snapshot["decision"]["recommended_next"] is None


def test_experiment_status_launch_actions_and_manual_choice(tmp_path):
    root = tmp_path / "experiment"
    _init_workspace(root)
    hparam_plan, _row = _add_plan(root, step_id="tune", task="hparam_tune")

    one = experiments.experiment_status(root)

    assert one["summary"]["state"] == "ready_to_launch"
    assert one["decision"]["recommended_next"]["argv"] == [
        "python",
        "-m",
        "agent_tools",
        "hparam-run-queue",
        "--plan-dir",
        str(hparam_plan),
        "--execute",
    ]

    generic_plan, _row = _add_plan(root, step_id="train")
    multiple = experiments.experiment_status(root)

    assert multiple["decision"]["manual_choice_required"] is True
    assert multiple["decision"]["recommended_next"] is None
    assert [action["id"] for action in multiple["decision"]["other_legal_actions"]] == [
        "run-plan",
        "hparam-run-queue",
    ]
    assert multiple["decision"]["other_legal_actions"][0]["argv"] == ["bash", str(generic_plan / "run.sh")]


@pytest.mark.parametrize(
    ("adaptive", "pipeline", "code"),
    [(True, False, "adaptive_phase_deferred"), (False, True, "pipeline_phase_deferred")],
)
def test_experiment_status_does_not_infer_adaptive_or_pipeline_launch(tmp_path, adaptive, pipeline, code):
    root = tmp_path / "experiment"
    _init_workspace(root)
    _add_plan(
        root,
        step_id="train",
        task="hparam_tune" if adaptive else "finetune",
        adaptive=adaptive,
        pipeline=pipeline,
    )

    snapshot = experiments.experiment_status(root)

    assert snapshot["summary"]["state"] == "blocked"
    assert snapshot["decision"]["recommended_next"] is None
    assert snapshot["blockers"][0]["code"] == code
    assert "v1" not in snapshot["blockers"][0]["message"]


def test_experiment_status_scopes_deferred_plans_away_from_ordinary_launch(tmp_path):
    root = tmp_path / "experiment"
    _init_workspace(root)
    ordinary, _row = _add_plan(root, step_id="ordinary")
    _add_plan(root, step_id="adaptive", task="hparam_tune", status="completed", adaptive=True)
    _add_plan(root, step_id="pipeline", status="failed", pipeline=True)

    snapshot = experiments.experiment_status(root)

    assert snapshot["summary"]["state"] == "ready_to_launch"
    assert snapshot["decision"]["recommended_next"]["argv"] == ["bash", str(ordinary / "run.sh")]
    assert snapshot["decision"]["blocked_actions"] == ["adaptive_advance", "finalize", "pipeline_advance"]
    assert {(blocker["code"], blocker["step_id"]) for blocker in snapshot["blockers"]} == {
        ("adaptive_phase_deferred", "adaptive"),
        ("pipeline_phase_deferred", "pipeline"),
    }


def test_experiment_status_blocks_finalize_for_unmaterialized_registered_step(tmp_path):
    root = tmp_path / "experiment"
    _init_workspace(root)
    _add_plan(root, step_id="ordinary", status="completed")
    step_dir = root / "steps" / "evaluate"
    step_dir.mkdir(parents=True)
    (step_dir / "step.yaml").write_text(
        yaml.safe_dump(
            {
                "step": {"id": "evaluate", "phase": "evaluate", "purpose": "Run external evaluation."},
                "experiment_id": "status-unit",
                "recipe_path": "",
                "plans": [],
            },
            sort_keys=False,
        )
    )

    snapshot = experiments.experiment_status(root)

    assert snapshot["summary"]["state"] == "blocked"
    assert snapshot["decision"]["other_legal_actions"] == []
    assert snapshot["decision"]["blocked_actions"] == ["finalize"]
    assert snapshot["blockers"] == [
        {
            "code": "unmaterialized_step",
            "step_id": "evaluate",
            "run_ids": [],
            "message": "The registered step has no materialized plan or canonical runs; experiment-status cannot "
            "prove controller completion.",
            "blocked_actions": ["finalize"],
        }
    ]

    manifest = yaml.safe_load((root / "experiment.yaml").read_text())
    manifest["experiment"].update({"status": "completed", "completed_at": "2026-08-25T02:00:00Z"})
    (root / "experiment.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))
    with pytest.raises(ValueError, match="unmaterialized registered steps"):
        experiments.experiment_status(root)


def test_experiment_status_keeps_multiple_ordinary_candidates_with_deferred_plan(tmp_path):
    root = tmp_path / "experiment"
    _init_workspace(root)
    first, _row = _add_plan(root, step_id="first")
    second, _row = _add_plan(root, step_id="second")
    _add_plan(root, step_id="pipeline", status="completed", pipeline=True)

    snapshot = experiments.experiment_status(root)

    assert snapshot["summary"]["state"] == "ready_to_launch"
    assert snapshot["decision"]["manual_choice_required"] is True
    assert snapshot["decision"]["recommended_next"] is None
    assert [action["argv"] for action in snapshot["decision"]["other_legal_actions"]] == [
        ["bash", str(first / "run.sh")],
        ["bash", str(second / "run.sh")],
    ]
    assert snapshot["decision"]["blocked_actions"] == ["finalize", "pipeline_advance"]


@pytest.mark.parametrize(
    ("status", "state"),
    [("running", "in_progress"), ("unknown_scheduler", "blocked")],
)
def test_experiment_status_prioritizes_active_deferred_plan_over_ordinary_launch(tmp_path, status, state):
    root = tmp_path / "experiment"
    _init_workspace(root)
    _add_plan(root, step_id="ordinary")
    _add_plan(root, step_id="adaptive", task="hparam_tune", status=status, adaptive=True)

    snapshot = experiments.experiment_status(root)

    assert snapshot["summary"]["state"] == state
    assert snapshot["decision"]["recommended_next"]["id"] == "experiment-monitor"
    assert snapshot["decision"]["other_legal_actions"] == []
    assert "adaptive_phase_deferred" in {blocker["code"] for blocker in snapshot["blockers"]}


def test_experiment_status_attributes_blockers_by_full_managed_run_key(tmp_path):
    root = tmp_path / "experiment"
    _init_workspace(root)
    _add_plan(root, step_id="first", status="unknown_scheduler")
    _add_plan(root, step_id="second", status="unknown_scheduler")
    _add_plan(root, step_id="terminal", status="failed")

    snapshot = experiments.experiment_status(root)
    blockers_by_step = {run["step_id"]: run["blockers"] for run in snapshot["runs"]}

    assert blockers_by_step["first"] == ["unknown_scheduler"]
    assert blockers_by_step["second"] == ["unknown_scheduler"]
    assert blockers_by_step["terminal"] == []


def test_experiment_status_defers_terminal_pipeline_finalization(tmp_path):
    root = tmp_path / "experiment"
    _init_workspace(root)
    _add_plan(root, step_id="evaluate", status="failed", pipeline=True)

    snapshot = experiments.experiment_status(root)

    assert snapshot["summary"]["state"] == "blocked"
    assert snapshot["decision"]["recommended_next"] is None
    assert snapshot["decision"]["other_legal_actions"] == []
    assert snapshot["decision"]["blocked_actions"] == ["finalize", "pipeline_advance"]
    assert snapshot["blockers"][0]["code"] == "pipeline_phase_deferred"

    manifest = yaml.safe_load((root / "experiment.yaml").read_text())
    manifest["experiment"].update({"status": "completed", "completed_at": "2026-08-25T02:00:00Z"})
    (root / "experiment.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))
    with pytest.raises(ValueError, match="cannot be verified for adaptive or pipeline plans") as exc_info:
        experiments.experiment_status(root)
    assert "v1" not in str(exc_info.value)


def test_experiment_status_rejects_completed_pipeline_with_successful_attempt(tmp_path):
    root = tmp_path / "experiment"
    _init_workspace(root)
    _add_plan(root, step_id="evaluate", status="completed", pipeline=True)
    manifest = yaml.safe_load((root / "experiment.yaml").read_text())
    manifest["experiment"].update({"status": "completed", "completed_at": "2026-08-25T02:00:00Z"})
    (root / "experiment.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))

    with pytest.raises(ValueError, match="cannot be verified for adaptive or pipeline plans"):
        experiments.experiment_status(root)


def test_experiment_status_defers_terminal_adaptive_finalization(tmp_path):
    root = tmp_path / "experiment"
    _init_workspace(root)
    _add_plan(root, step_id="tune", status="completed", task="hparam_tune", adaptive=True)

    snapshot = experiments.experiment_status(root)

    assert snapshot["summary"]["state"] == "blocked"
    assert snapshot["decision"]["recommended_next"] is None
    assert snapshot["decision"]["other_legal_actions"] == []
    assert snapshot["decision"]["blocked_actions"] == ["adaptive_advance", "finalize"]
    assert snapshot["blockers"][0]["code"] == "adaptive_phase_deferred"

    manifest = yaml.safe_load((root / "experiment.yaml").read_text())
    manifest["experiment"].update({"status": "completed", "completed_at": "2026-08-25T02:00:00Z"})
    (root / "experiment.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))
    with pytest.raises(ValueError, match="cannot be verified for adaptive or pipeline plans"):
        experiments.experiment_status(root)


def test_experiment_status_rejects_completed_mixed_ordinary_and_deferred_plans(tmp_path):
    root = tmp_path / "experiment"
    _init_workspace(root)
    _add_plan(root, step_id="ordinary", status="completed")
    _add_plan(root, step_id="adaptive", task="hparam_tune", status="completed", adaptive=True)
    manifest = yaml.safe_load((root / "experiment.yaml").read_text())
    manifest["experiment"].update({"status": "completed", "completed_at": "2026-08-25T02:00:00Z"})
    (root / "experiment.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))

    with pytest.raises(ValueError, match="cannot be verified for adaptive or pipeline plans"):
        experiments.experiment_status(root)


def test_experiment_status_terminal_and_completed_contract(tmp_path):
    root = tmp_path / "experiment"
    _init_workspace(root)
    _add_plan(root, step_id="train", status="failed")

    ready = experiments.experiment_status(root)

    assert ready["summary"]["state"] == "ready_to_finalize"
    finalize = ready["decision"]["other_legal_actions"][0]
    assert finalize["id"] == "experiment-finalize"
    assert finalize["required_inputs"] == ["report_path"]

    manifest = yaml.safe_load((root / "experiment.yaml").read_text())
    manifest["experiment"].update({"status": "completed", "completed_at": "2026-08-25T02:00:00Z"})
    (root / "experiment.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))
    completed = experiments.experiment_status(root)
    assert completed["summary"]["state"] == "completed"
    assert completed["decision"]["recommended_next"] is None

    rows = _read_manifest_rows(root)
    rows[0]["status"] = "running"
    write_rows(root / "run_manifest.tsv", rows)
    with pytest.raises(ValueError, match="Completed experiment metadata conflicts"):
        experiments.experiment_status(root)


def test_experiment_status_is_zero_write_and_ignores_projections(tmp_path, monkeypatch):
    root = tmp_path / "experiment"
    _init_workspace(root)
    plan_dir, _row = _add_plan(
        root,
        step_id="tune",
        task="hparam_tune",
        status="unknown_scheduler",
        adaptive=True,
    )
    _add_plan(root, step_id="evaluate", pipeline=True)
    before = _workspace_files(root)

    def unexpected(*_args, **_kwargs):
        raise AssertionError("experiment-status attempted a write or live observation")

    for name in (
        "write_text_at",
        "write_rows_at",
        "conditional_atomic_replace_text_at",
        "append_event_at",
        "mkdir_experiment_dirs",
    ):
        monkeypatch.setattr(experiment_io, name, unexpected)
    monkeypatch.setattr(experiments, "merge_run_manifest", unexpected)
    monkeypatch.setattr(experiment_tracking, "monitor_run_row", unexpected)
    monkeypatch.setattr(plans, "evaluate_recipe", unexpected)
    monkeypatch.setattr(decisions, "evaluate_consultation_gates", unexpected)
    monkeypatch.setattr(recipes, "load_consultation_policy", unexpected)
    monkeypatch.setattr(decision_paths, "path_issues", unexpected)
    monkeypatch.setattr(plan_context, "load_config_summary_for_recipe", unexpected)
    for adapter_type in {type(adapter) for adapter in all_adapters()}:
        for name in ("task_issues", "preflight_issues", "configured_input_issues"):
            monkeypatch.setattr(adapter_type, name, unexpected)

    baseline = experiments.experiment_status(root)
    assert _workspace_files(root) == before

    (root / "run_status.tsv").write_text("not\ta\tvalid\tprojection\n")
    (plan_dir / "launch_manifest.tsv").write_text("status\ncompleted\n")
    (root / "reports").mkdir()
    (root / "reports" / "monitor.md").write_text("completed\n")
    (root / "events.jsonl").write_text("not-json\n")
    (root / "wandb").mkdir()
    (root / "wandb" / "runs.tsv").write_text("status\ncompleted\n")
    pipeline_root = root / "pipelines" / "pipeline-unit"
    pipeline_root.mkdir(parents=True)
    (pipeline_root / "pipeline.json").write_text("{\n")
    (pipeline_root / "jobs.tsv").write_text("status\ncompleted\n")
    (root / "adaptive").mkdir()
    (root / "adaptive" / "workflow.json").write_text('{"completed": true}\n')

    assert experiments.experiment_status(root) == baseline


def test_experiment_status_contract_error_is_zero_write(tmp_path, monkeypatch, capsys):
    root = tmp_path / "experiment"
    _init_workspace(root)
    plan_dir, _row = _add_plan(root, step_id="train")
    plan = json.loads((plan_dir / "plan.json").read_text())
    plan["recipe"]["unknown"] = True
    (plan_dir / "plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    resolved = yaml.safe_load((plan_dir / "recipe.resolved.yaml").read_text())
    resolved["unknown"] = True
    (plan_dir / "recipe.resolved.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False))
    before = _workspace_files(root)

    def unexpected(*_args, **_kwargs):
        raise AssertionError("experiment-status attempted a write")

    for name in (
        "write_text_at",
        "write_rows_at",
        "conditional_atomic_replace_text_at",
        "append_event_at",
        "mkdir_experiment_dirs",
    ):
        monkeypatch.setattr(experiment_io, name, unexpected)
    monkeypatch.setattr(experiments, "merge_run_manifest", unexpected)

    assert cli.main(["experiment-status", "--run-dir", str(root), "--json"]) == 1
    assert _workspace_files(root) == before
    assert "Traceback" not in capsys.readouterr().err


def test_experiment_status_human_output_quotes_advisory_argv(tmp_path):
    root = tmp_path / "experiment with spaces"
    _init_workspace(root)
    _add_plan(root, step_id="train")

    rendered = experiment_tracking.format_experiment_status(experiments.experiment_status(root))

    assert "recorded evidence, not live" in rendered
    assert "Next legal action" in rendered
    assert "'" in rendered
    assert "execution host:" not in rendered
    assert "Advisory only; this output does not authorize execution." in rendered


def test_experiment_status_renders_remote_launch_execution_host():
    root = Path("/remote/experiment")
    row = {
        "step_id": "train",
        "run_id": "run-000",
        "run_name": "default",
        "status": "planned",
    }
    registered_steps = [
        {
            "manifest": {"step": {"id": "train", "phase": "train", "purpose": "Train."}},
            "plans": [
                {
                    "path": str(root / "plans" / "train"),
                    "task": "finetune",
                    "adaptive": False,
                    "pipeline": False,
                    "run_keys": [("train", "run-000")],
                    "launch_script": str(root / "plans" / "train" / "run.sh"),
                }
            ],
        }
    ]

    snapshot = experiment_tracking.experiment_status_snapshot(
        {"id": "status-unit", "title": "Remote status"},
        registered_steps,
        [row],
        root=root,
        remote="baichuan3",
    )
    action = snapshot["decision"]["recommended_next"]

    assert action["execution_host"] == "baichuan3"
    rendered = experiment_tracking.format_experiment_status(snapshot)
    assert "execution host: `baichuan3`" in rendered
    assert f"bash {root / 'plans' / 'train' / 'run.sh'}" in rendered


def test_experiment_status_keeps_local_hparam_queue_on_controller(tmp_path):
    root = tmp_path / "experiment"
    _init_workspace(root)
    plan_dir, _row = _add_plan(
        root,
        step_id="tune",
        task="hparam_tune",
        status="pending",
        host="baichuan3",
    )

    snapshot = experiments.experiment_status(root)
    action = snapshot["decision"]["recommended_next"]

    assert action["id"] == "hparam-run-queue"
    assert action["execution_host"] is None
    assert action["argv"] == [
        "python",
        "-m",
        "agent_tools",
        "hparam-run-queue",
        "--plan-dir",
        str(plan_dir),
        "--execute",
    ]
    assert "execution host:" not in experiment_tracking.format_experiment_status(snapshot)


def test_experiment_status_keeps_remote_finalize_on_controller():
    root = Path("/remote/experiment")
    row = {"step_id": "train", "run_id": "run-000", "run_name": "default", "status": "completed"}
    registered_steps = [
        {
            "manifest": {"step": {"id": "train", "phase": "train", "purpose": "Train."}},
            "plans": [
                {
                    "path": str(root / "plans" / "train"),
                    "adaptive": False,
                    "pipeline": False,
                    "run_keys": [("train", "run-000")],
                }
            ],
        }
    ]

    snapshot = experiment_tracking.experiment_status_snapshot(
        {"id": "status-unit", "title": "Remote status"},
        registered_steps,
        [row],
        root=root,
        remote="baichuan3",
    )
    action = snapshot["decision"]["other_legal_actions"][0]

    assert action["execution_host"] is None
    assert action["argv"][-2:] == ["--remote", "baichuan3"]


def test_experiment_status_cli_converts_remote_timeout_to_exit_one(monkeypatch, capsys):
    def timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(["ssh", "baichuan3"], 10)

    monkeypatch.setattr(cli, "experiment_status", timeout)

    assert cli.main(["experiment-status", "--run-dir", "/remote/experiment", "--remote", "baichuan3"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("error:")
    assert "Traceback" not in captured.err


def test_experiment_status_cli_converts_malformed_yaml_to_exit_one(tmp_path, capsys):
    root = tmp_path / "experiment"
    root.mkdir()
    (root / "experiment.yaml").write_text("experiment: [\n")
    (root / "run_manifest.tsv").write_text("step_id\trun_id\n")

    assert cli.main(["experiment-status", "--run-dir", str(root)]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "invalid YAML" in captured.err
    assert "Traceback" not in captured.err


def test_experiment_status_cli_returns_one_for_contract_errors(tmp_path, capsys):
    root = tmp_path / "experiment"
    _init_workspace(root)
    _add_plan(root, step_id="train")
    rows = _read_manifest_rows(root)
    rows[0]["status"] = "invented"
    write_rows(root / "run_manifest.tsv", rows)

    assert cli.main(["experiment-status", "--run-dir", str(root), "--json"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "unsupported status" in captured.err


def test_experiment_status_cli_returns_one_for_non_mapping_plan_run(tmp_path, capsys):
    root = tmp_path / "experiment"
    _init_workspace(root)
    plan_dir, _canonical = _add_plan(root, step_id="train")
    plan = json.loads((plan_dir / "plan.json").read_text())
    plan["runs"] = [None]
    (plan_dir / "plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")

    assert cli.main(["experiment-status", "--run-dir", str(root), "--json"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("error:")
    assert "Traceback" not in captured.err


@pytest.mark.parametrize("adaptive", [None, False, "invalid", ["invalid"]])
def test_experiment_status_cli_returns_one_for_non_mapping_adaptive(tmp_path, capsys, adaptive):
    root = tmp_path / "experiment"
    _init_workspace(root)
    plan_dir, _canonical = _add_plan(root, step_id="train", task="hparam_tune")
    plan = json.loads((plan_dir / "plan.json").read_text())
    plan["recipe"]["adaptive"] = adaptive
    plan["recipe"]["_local_recipe"]["adaptive"] = adaptive
    resolved = yaml.safe_load((plan_dir / "recipe.resolved.yaml").read_text())
    resolved["adaptive"] = adaptive
    resolved["_local_recipe"]["adaptive"] = adaptive
    resolved_path = plan_dir / "recipe.resolved.yaml"
    resolved_path.write_text(yaml.safe_dump(resolved, sort_keys=False))
    plan["resolved_recipe_sha256"] = _sha256(resolved_path)
    (plan_dir / "plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")

    assert cli.main(["experiment-status", "--run-dir", str(root), "--json"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "adaptive must be a mapping" in captured.err
    assert "Traceback" not in captured.err


@pytest.mark.parametrize(
    "drift",
    ["config", "run_script", "resolved_recipe", "plan_alias", "plan_escape", "canonical_run"],
)
def test_experiment_status_rejects_registered_plan_drift(tmp_path, drift):
    root = tmp_path / "experiment"
    _init_workspace(root)
    plan_dir, canonical = _add_plan(root, step_id="train")

    if drift == "config":
        Path(canonical["config"]).write_text("model: changed\n")
    elif drift == "run_script":
        (plan_dir / "run.sh").write_text("#!/bin/sh\nexit 1\n")
    elif drift == "resolved_recipe":
        (plan_dir / "recipe.resolved.yaml").write_text("task: changed\n")
    elif drift == "plan_alias":
        plan_path = plan_dir / "plan.json"
        target = plan_dir / "plan.real.json"
        plan_path.rename(target)
        plan_path.symlink_to(target.name)
    elif drift == "plan_escape":
        step_path = root / "steps" / "train" / "step.yaml"
        step_manifest = yaml.safe_load(step_path.read_text())
        step_manifest["plans"] = [str(tmp_path / "outside")]
        step_path.write_text(yaml.safe_dump(step_manifest, sort_keys=False))
    else:
        (root / "run_manifest.tsv").write_text("step_id\trun_id\n")

    with pytest.raises(ValueError):
        experiments.experiment_status(root)


@pytest.mark.parametrize("field", ["title", "objective", "baseline"])
def test_experiment_status_rejects_coherent_registered_plan_experiment_drift(tmp_path, field):
    root = tmp_path / "experiment"
    _init_workspace(root)
    plan_dir, _canonical = _add_plan(root, step_id="train")

    plan_path = plan_dir / "plan.json"
    plan = json.loads(plan_path.read_text())
    plan["recipe"]["experiment"][field] = f"foreign {field}"
    resolved_path = plan_dir / "recipe.resolved.yaml"
    resolved = yaml.safe_load(resolved_path.read_text())
    resolved["experiment"][field] = f"foreign {field}"
    resolved_path.write_text(yaml.safe_dump(resolved, sort_keys=False))
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="experiment metadata differs from the managed workspace"):
        experiments.experiment_status(root)


def test_experiment_status_rejects_registered_plan_recipe_path_drift(tmp_path):
    root = tmp_path / "experiment"
    _init_workspace(root)
    plan_dir, _canonical = _add_plan(root, step_id="train")

    step_path = root / "steps" / "train" / "step.yaml"
    step_manifest = yaml.safe_load(step_path.read_text())
    step_manifest["recipe_path"] = str(root / "registered-recipe.yaml")
    step_path.write_text(yaml.safe_dump(step_manifest, sort_keys=False))
    plan_path = plan_dir / "plan.json"
    plan = json.loads(plan_path.read_text())
    plan["recipe"]["_recipe_path"] = str(root / "foreign-recipe.yaml")
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="recipe path differs from its managed step"):
        experiments.experiment_status(root)


def test_experiment_status_rejects_plan_registered_by_multiple_steps(tmp_path):
    root = tmp_path / "experiment"
    _init_workspace(root)
    plan_dir, _canonical = _add_plan(root, step_id="train")
    second_step = root / "steps" / "evaluate"
    second_step.mkdir(parents=True)
    (second_step / "step.yaml").write_text(
        yaml.safe_dump(
            {
                "step": {"id": "evaluate", "phase": "evaluate", "purpose": "Evaluate."},
                "experiment_id": "status-unit",
                "recipe_path": "",
                "plans": [str(plan_dir)],
            },
            sort_keys=False,
        )
    )

    with pytest.raises(ValueError, match="more than one managed step"):
        experiments.experiment_status(root)


def test_experiment_status_routes_all_registered_reads_to_remote(monkeypatch):
    root = Path("/remote/experiment")
    experiment = {
        "id": "status-unit",
        "title": "Status unit experiment",
        "objective": "Remote status.",
        "root": str(root),
        "baseline": "unit baseline",
    }
    step = {"id": "train", "phase": "train", "purpose": "Train."}
    manifest = {
        "step": step,
        "experiment_id": "status-unit",
        "recipe_path": "",
        "plans": [str(root / "plans" / "train")],
    }
    row = {
        "experiment_id": "status-unit",
        "step_id": "train",
        "run_id": "run-000",
        "run_name": "default",
        "status": "unknown_remote",
    }
    calls = []

    def managed_workspace(candidate, *, remote, allow_completed):
        calls.append(("workspace", candidate, remote, allow_completed))
        return experiment, [row]

    def registered_steps(candidate, *, experiment_id, remote):
        calls.append(("steps", candidate, experiment_id, remote))
        return [manifest]

    def registered_plan(candidate, *, workspace, workspace_experiment, step_manifest, workspace_rows, remote):
        calls.append(("plan", candidate, workspace, workspace_experiment, step_manifest, workspace_rows, remote))
        candidate_path = Path(candidate)
        return {
            "path": str(candidate),
            "task": "finetune",
            "adaptive": False,
            "pipeline": False,
            "run_keys": [("train", "run-000")],
            "launch_script": str(candidate_path / "run.sh"),
        }

    monkeypatch.setattr(experiments, "_managed_workspace", managed_workspace)
    monkeypatch.setattr(experiments, "read_registered_steps", registered_steps)
    monkeypatch.setattr(experiments.artifacts, "read_registered_plan", registered_plan)

    def unexpected_write(*_args, **_kwargs):
        raise AssertionError("remote experiment-status attempted a write")

    for name in (
        "write_text_at",
        "write_rows_at",
        "conditional_atomic_replace_text_at",
        "append_event_at",
        "mkdir_experiment_dirs",
    ):
        monkeypatch.setattr(experiment_io, name, unexpected_write)
    monkeypatch.setattr(experiments, "merge_run_manifest", unexpected_write)

    snapshot = experiments.experiment_status(root, remote="baichuan3")

    assert calls[0] == ("workspace", root, "baichuan3", True)
    assert calls[1] == ("steps", root, "status-unit", "baichuan3")
    assert calls[2][3] == experiment
    assert calls[2][-1] == "baichuan3"
    assert snapshot["experiment"]["remote"] == "baichuan3"
    action = snapshot["decision"]["recommended_next"]
    assert action["execution_host"] is None
    assert action["argv"][-2:] == ["--remote", "baichuan3"]
