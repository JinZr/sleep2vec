from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from agent_tool_test_helpers import write_finetune_recipe, write_yaml
import pytest
import yaml

from agent_tools import (
    cli,
    decision_paths,
    decisions,
    experiment_io,
    experiment_pipeline,
    experiment_tracking,
    experiments,
    plan_context,
    plan_contract,
    plan_hparam,
    plans,
    recipes,
)
from agent_tools.adapters import all_adapters, finetune as finetune_adapter, get_adapter
from agent_tools.experiment_workspace import FROZEN_RUN_FIELDS, managed_run_parameters
from agent_tools.manifests import read_rows, write_rows
from agent_tools.models import REPO_ROOT

_RUNTIME_COMMIT = subprocess.run(
    ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True, text=True, capture_output=True
).stdout.strip()


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
    recipe_path = root / f"{step_id}.yaml"
    recipe = {
        "name": "default",
        "task": task,
        "experiment": experiment,
        "step": step,
        "inputs": {"config": str(recipe_path), "label_name": "unit"},
        "runtime": {},
        "artifacts": {},
        "execution": {},
        "evaluation_policy": {"test_after_fit": False},
        "_recipe_path": str(recipe_path),
    }
    if task == "sleep2stat":
        recipe["inputs"] = {"config": str(recipe_path)}
        recipe["evaluation_policy"] = {"external_test_locked": True}
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
                "search": {"method": "grid", "max_runs": 1, "parameters": {}},
                "execution": {
                    "python": "python",
                    "runtime_commit": "0" * 40,
                    "workdir": str(root),
                    "scheduler": {"type": "direct"},
                },
                "evaluation_policy": {
                    "selection_metric": "val_loss",
                    "selection_mode": "min",
                    "selection_split": "val",
                    "final_eval_split": "test",
                    "external_test_locked": True,
                    "test_after_fit": False,
                    "final_test_unlocked": False,
                    "require_manual_unlock_for_final_test": True,
                },
                "_base_recipe": base_recipe,
                "_local_recipe": local_recipe,
            }
        )
    if adaptive:
        recipe["adaptive"] = {"enabled": True, "suggest": {"strategy": "best_neighborhood"}}
        if task == "hparam_tune":
            recipe["_local_recipe"]["adaptive"] = recipe["adaptive"]
    if task == "sleep2stat":
        config_bytes = yaml.safe_dump(
            {"run": {"output_dir": str(root / "analysis")}, "analyzers": [], "reducers": []},
            sort_keys=False,
        ).encode()
    else:
        config_bytes = b"model: unit\n"
    recipe["input_snapshots"] = [
        {
            "field": "inputs.config",
            "path": str(recipe_path),
            "sha256": hashlib.sha256(config_bytes).hexdigest(),
        }
    ]
    plan_contract.bind_plan_context(recipe)
    adapter = get_adapter(task)
    assert adapter is not None
    contract = adapter.compile_plan_contract(
        recipe,
        plan_dir,
        run_index_offset=0,
        config_bytes=config_bytes,
    )
    run = dict(contract["runs"][0])
    run_dir = Path(run["run_dir"])
    run_dir.mkdir(parents=True)
    config_path = Path(run["config"])
    script_path = Path(run["script"])
    artifacts_path = Path(run["artifacts"])
    if task == "hparam_tune":
        run_files = contract["run_files"][0]
        config_path.write_bytes(run_files["config_bytes"])
        script_path.write_text(run_files["script_text"])
        (plan_dir / "config.source.yaml").write_bytes(config_bytes)
        commands = [run["command"]]
    else:
        config_path.write_bytes(config_bytes)
        script_path.write_text(contract["script_text"])
        commands = contract["commands"]
    artifacts_path.write_text("{}\n")
    run.update({"config_sha256": _sha256(config_path), "script_sha256": _sha256(script_path)})
    if task != "hparam_tune":
        run["status"] = "planned"
    if host is not None:
        run["host"] = host
    resolved_recipe = {key: value for key, value in recipe.items() if key != "_recipe_path"}
    resolved_text = yaml.safe_dump(resolved_recipe, sort_keys=False)
    resolved_path = plan_dir / "recipe.resolved.yaml"
    resolved_path.write_text(resolved_text)
    plan_run = dict(run)
    if task != "hparam_tune" and len(commands) == 1:
        plan_run["command"] = commands[0]
    plan = {"status": "PASS", "recipe": recipe, "runs": [plan_run]}
    if task == "hparam_tune":
        plan["resolved_recipe_sha256"] = _sha256(resolved_path)
        (plan_dir / "run_all.sh").write_text(contract["launch_script_text"])
    else:
        plan["commands"] = commands
        (plan_dir / "run.sh").write_text(script_path.read_text())
    (plan_dir / "plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")

    plan_controller = "ordinary"
    if adaptive:
        plan_controller = "adaptive"
    if pipeline:
        plan_controller = "pipeline"
    step_dir = root / "steps" / step_id
    step_dir.mkdir(parents=True)
    (step_dir / "step.yaml").write_text(
        yaml.safe_dump(
            {
                "step": step,
                "experiment_id": experiment["id"],
                "plan_controller": plan_controller,
                "recipe_path": str(recipe_path),
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


def _record_hparam_selection(
    root: Path, *, step_id: str = "tune", write_report: bool = False, score: str = "0.25"
) -> Path:
    rows = _read_manifest_rows(root)
    winner = next(row for row in rows if row["step_id"] == step_id)
    winner["target"] = winner.get("target") or "local"
    checkpoint = Path(winner["checkpoint_dir"]) / "epoch=1.ckpt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"unit checkpoint\n")
    winner.update(
        {
            "selection_task": "hparam_tune",
            "metric": "val_loss",
            "selection_mode": "min",
            "selection_split": "val",
            "score": score,
            "rank": "1",
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "run_manifest": str(Path(winner["runtime_dir"]) / "run_manifest.json"),
        }
    )
    write_rows(root / "run_manifest.tsv", rows)
    experiment = yaml.safe_load((root / "experiment.yaml").read_text())["experiment"]
    registered = experiments._registered_plan_steps(
        root,
        experiment,
        _read_manifest_rows(root),
        remote=None,
        require_registered_rows=True,
    )
    lifecycle = experiment_tracking.hparam_selection_lifecycle(
        registered,
        _read_manifest_rows(root),
        root=root,
    )
    report_path = root / "reports" / "hparam_selection.md"
    if write_report:
        report_path.parent.mkdir(exist_ok=True)
        report_text = lifecycle["expected_report"]
        report_path.write_text(report_text)
        write_rows(
            root / "reports" / "ranking.csv",
            [
                {
                    "step_id": row["step_id"],
                    "run_id": row["run_id"],
                    "run_name": row["run_name"],
                    "parameter_summary": row["parameter_summary"],
                    "version": row["version"],
                    "config": row["config"],
                    "metric": row["metric"],
                    "score": row["score"],
                    "rank": row["rank"],
                    "checkpoint_path": row["checkpoint_path"],
                    "checkpoint_sha256": row["checkpoint_sha256"],
                    "run_manifest": row.get("run_manifest", ""),
                    "status": row["status"],
                    **managed_run_parameters(row),
                }
                for row in rows
                if row["step_id"] == step_id and row.get("rank") not in (None, "")
            ],
        )
        rows = _read_manifest_rows(root)
        digest = hashlib.sha256(report_text.encode()).hexdigest()
        for row in rows:
            if row["step_id"] == step_id:
                row.update({"selection_report": str(report_path), "selection_report_sha256": digest})
        write_rows(root / "run_manifest.tsv", rows)
    return report_path


def _write_public_hparam_recipe(
    root: Path,
    parameters: dict,
    *,
    selection_metric: str = "val_ahi_pearson",
    selection_mode: str = "max",
    max_runs: int = 1,
) -> Path:
    base_recipe = write_finetune_recipe(root)
    base_payload = yaml.safe_load(base_recipe.read_text())
    base_payload["evaluation_policy"].update({"selection_metric": selection_metric, "selection_mode": selection_mode})
    base_recipe.write_text(yaml.safe_dump(base_payload, sort_keys=False))
    config_path = Path(base_payload["inputs"]["config"])
    config_payload = yaml.safe_load(config_path.read_text())
    config_payload["finetune"]["task"].update({"monitor": selection_metric, "monitor_mod": selection_mode})
    config_path.write_text(yaml.safe_dump(config_payload, sort_keys=False))
    return write_yaml(
        root / "tune.yaml",
        {
            "name": "status_public_hparam",
            "task": "hparam_tune",
            "variant": "sleep2vec",
            "base_recipe": str(base_recipe),
            "inputs": {},
            "search": {"method": "grid", "max_runs": max_runs, "parameters": parameters},
            "execution": {"workdir": str(root), "python": sys.executable, "runtime_commit": _RUNTIME_COMMIT},
            "evaluation_policy": {
                "selection_metric": selection_metric,
                "selection_mode": selection_mode,
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


def test_experiment_status_recompiles_relative_generic_plan_without_controller_host_defaults(tmp_path, monkeypatch):
    root = tmp_path / "experiment"
    recipe = write_finetune_recipe(root)
    recipe_payload = yaml.safe_load(recipe.read_text())
    recipe_payload["inputs"]["config"] = os.path.relpath(root / "config.yaml", REPO_ROOT)
    recipe.write_text(yaml.safe_dump(recipe_payload, sort_keys=False))
    plan_dir = root / "plans" / "train"
    assert plans.build_plan(recipe_path=recipe, output_dir=plan_dir).exit_code == 0
    before = _workspace_files(root)

    monkeypatch.setattr(plan_contract, "REPO_ROOT", Path("/controller/repo"))
    monkeypatch.setattr(plan_hparam, "REPO_ROOT", Path("/controller/repo"))
    monkeypatch.setattr(finetune_adapter, "REPO_ROOT", Path("/controller/repo"))
    monkeypatch.setattr(plan_contract.sys, "executable", "/controller/bin/python")

    snapshot = experiments.experiment_status(root)

    assert snapshot["summary"]["state"] == "ready_to_launch"
    assert _workspace_files(root) == before


def test_experiment_status_recompiles_relative_hparam_plan_without_controller_host_defaults(tmp_path, monkeypatch):
    root = tmp_path / "experiment"
    recipe = _write_public_hparam_recipe(root, {"runtime.lr": [1e-6]})
    recipe_payload = yaml.safe_load(recipe.read_text())
    base_recipe = Path(recipe_payload["base_recipe"])
    base_payload = yaml.safe_load(base_recipe.read_text())
    base_payload["inputs"]["config"] = os.path.relpath(root / "config.yaml", REPO_ROOT)
    base_recipe.write_text(yaml.safe_dump(base_payload, sort_keys=False))
    plan_dir = root / "plans" / "tune"
    assert plans.build_plan(recipe_path=recipe, output_dir=plan_dir).exit_code == 0
    before = _workspace_files(root)

    monkeypatch.setattr(plan_contract, "REPO_ROOT", Path("/controller/repo"))
    monkeypatch.setattr(plan_hparam, "REPO_ROOT", Path("/controller/repo"))
    monkeypatch.setattr(plan_contract.sys, "executable", "/controller/bin/python")

    snapshot = experiments.experiment_status(root)

    assert snapshot["summary"]["state"] == "ready_to_launch"
    assert _workspace_files(root) == before


@pytest.mark.parametrize(
    "context",
    [
        None,
        {"home": "/creator/home", "python": "/creator/python", "repo_root": "/creator/repo", "extra": "x"},
        {"home": "/creator/home", "python": "python", "repo_root": "/creator/repo"},
        {"home": "/creator/home", "python": "/creator/python", "repo_root": "relative/repo"},
    ],
)
def test_experiment_status_rejects_invalid_frozen_plan_context(tmp_path, context):
    root = tmp_path / "experiment"
    _init_workspace(root)
    plan_dir, _canonical = _add_plan(root, step_id="train")
    plan_path = plan_dir / "plan.json"
    plan = json.loads(plan_path.read_text())
    resolved_path = plan_dir / "recipe.resolved.yaml"
    resolved = yaml.safe_load(resolved_path.read_text())
    if context is None:
        plan["recipe"].pop("_plan_context")
        resolved.pop("_plan_context")
    else:
        plan["recipe"]["_plan_context"] = context
        resolved["_plan_context"] = context
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    resolved_path.write_text(yaml.safe_dump(resolved, sort_keys=False))

    with pytest.raises(ValueError, match="exact absolute _plan_context"):
        experiments.experiment_status(root)


def test_experiment_status_rejects_tampered_hparam_run_all_script(tmp_path):
    root = tmp_path / "experiment"
    _init_workspace(root)
    plan_dir, _canonical = _add_plan(root, step_id="tune", task="hparam_tune")
    (plan_dir / "run_all.sh").write_text("#!/usr/bin/env bash\nexit 0\n")

    with pytest.raises(ValueError, match="launch script differs from its frozen recipe"):
        experiments.experiment_status(root)


@pytest.mark.parametrize("allow_unresolved", [False, True])
def test_experiment_status_skips_registered_blocked_plan_after_successful_retry(tmp_path, allow_unresolved):
    root = tmp_path / "experiment"
    recipe = write_finetune_recipe(root, include_label=False)
    blocked_dir = root / "plans" / "blocked"

    blocked = plans.build_plan(
        recipe_path=recipe,
        output_dir=blocked_dir,
        allow_unresolved=allow_unresolved,
    )

    assert blocked.exit_code == 2
    blocked_path = blocked_dir / "plan.blocked.md"
    assert blocked_path.exists()
    assert not (blocked_dir / "plan.json").exists()
    assert not (blocked_dir / "recipe.resolved.yaml").exists()
    assert (blocked_dir / "plan.draft.json").exists() is allow_unresolved
    decisions_path = write_yaml(
        root / "decisions.yaml",
        {"decisions": {"label_name": {"value": "ahi", "source": "explicit_user"}}},
    )
    retry_dir = root / "plans" / "retry"
    retry = plans.build_plan(
        recipe_path=recipe,
        output_dir=retry_dir,
        user_decisions_path=decisions_path,
    )
    assert retry.exit_code == 0
    before = _workspace_files(root)

    snapshot = experiments.experiment_status(root)

    assert snapshot["summary"]["state"] == "ready_to_launch"
    assert snapshot["steps"][0]["plans"] == [str(retry_dir)]
    assert snapshot["decision"]["recommended_next"]["argv"] == ["bash", str(retry_dir / "run.sh")]
    assert _workspace_files(root) == before


def test_experiment_status_skips_registered_blocked_plan_after_successful_plan(tmp_path):
    root = tmp_path / "experiment"
    recipe = write_finetune_recipe(root)
    successful_dir = root / "plans" / "successful"
    assert plans.build_plan(recipe_path=recipe, output_dir=successful_dir).exit_code == 0

    payload = yaml.safe_load(recipe.read_text())
    del payload["inputs"]["label_name"]
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
    blocked_dir = root / "plans" / "blocked"
    assert plans.build_plan(recipe_path=recipe, output_dir=blocked_dir).exit_code == 2
    before = _workspace_files(root)

    snapshot = experiments.experiment_status(root)

    assert snapshot["summary"]["state"] == "ready_to_launch"
    assert snapshot["steps"][0]["plans"] == [str(successful_dir)]
    assert snapshot["decision"]["recommended_next"]["argv"] == ["bash", str(successful_dir / "run.sh")]
    assert _workspace_files(root) == before


def test_experiment_status_rejects_recipe_path_drift_on_ordinary_retry(tmp_path):
    root = tmp_path / "experiment"
    recipe = write_finetune_recipe(root, include_label=False)
    blocked_dir = root / "plans" / "blocked"
    assert plans.build_plan(recipe_path=recipe, output_dir=blocked_dir).exit_code == 2

    decisions_path = write_yaml(
        root / "decisions.yaml",
        {"decisions": {"label_name": {"value": "ahi", "source": "explicit_user"}}},
    )
    retry_dir = root / "plans" / "retry"
    assert plans.build_plan(recipe_path=recipe, output_dir=retry_dir, user_decisions_path=decisions_path).exit_code == 0
    plan_path = retry_dir / "plan.json"
    plan = json.loads(plan_path.read_text())
    plan["recipe"]["_recipe_path"] = str(root / "foreign-recipe.yaml")
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="recipe path differs from its managed step"):
        experiments.experiment_status(root)


@pytest.mark.parametrize("missing_from", ["step", "plan"])
def test_experiment_status_rejects_missing_registered_recipe_provenance(tmp_path, missing_from):
    root = tmp_path / "experiment"
    _init_workspace(root)
    plan_dir, _canonical = _add_plan(root, step_id="train")
    if missing_from == "step":
        step_path = root / "steps" / "train" / "step.yaml"
        step = yaml.safe_load(step_path.read_text())
        step["recipe_path"] = ""
        step_path.write_text(yaml.safe_dump(step, sort_keys=False))
        error = "recipe path differs from its managed step"
    else:
        plan_path = plan_dir / "plan.json"
        plan = json.loads(plan_path.read_text())
        del plan["recipe"]["_recipe_path"]
        plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
        error = "recipe path must be absolute"
    before = _workspace_files(root)

    with pytest.raises(ValueError, match=error):
        experiments.experiment_status(root)
    assert _workspace_files(root) == before


def test_experiment_status_allows_later_ordinary_plan_from_new_recipe(tmp_path):
    root = tmp_path / "experiment"
    first_recipe = write_finetune_recipe(root)
    second_recipe = root / "second-recipe.yaml"
    second_payload = yaml.safe_load(first_recipe.read_text())
    second_payload["artifacts"]["version_name"] = "unit-second"
    second_recipe.write_text(yaml.safe_dump(second_payload, sort_keys=False))
    first_plan = root / "plans" / "first"
    second_plan = root / "plans" / "second"

    assert plans.build_plan(recipe_path=first_recipe, output_dir=first_plan).exit_code == 0
    assert plans.build_plan(recipe_path=second_recipe, output_dir=second_plan).exit_code == 0

    snapshot = experiments.experiment_status(root)

    assert snapshot["summary"]["state"] == "ready_to_launch"
    assert snapshot["steps"][0]["plans"] == [str(first_plan), str(second_plan)]
    assert snapshot["decision"]["manual_choice_required"] is True
    assert len(snapshot["decision"]["other_legal_actions"]) == 2


@pytest.mark.parametrize(("adaptive", "pipeline"), [(True, False), (False, True)])
def test_experiment_status_allows_controller_owned_recipe_path(tmp_path, adaptive, pipeline):
    root = tmp_path / "experiment"
    _init_workspace(root)
    plan_dir, _canonical = _add_plan(
        root,
        step_id="train",
        task="hparam_tune" if adaptive else "finetune",
        adaptive=adaptive,
        pipeline=pipeline,
    )
    plan_path = plan_dir / "plan.json"
    plan = json.loads(plan_path.read_text())
    plan["recipe"]["_recipe_path"] = str(root / "controller-recipe.yaml")
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")

    snapshot = experiments.experiment_status(root)

    assert snapshot["summary"]["state"] == "blocked"


def test_experiment_status_rejects_blocked_artifacts_beside_pass_plan(tmp_path):
    root = tmp_path / "experiment"
    recipe = write_finetune_recipe(root)
    plan_dir = root / "plans" / "train"
    assert plans.build_plan(recipe_path=recipe, output_dir=plan_dir).exit_code == 0
    (plan_dir / "plan.blocked.md").write_text("blocked\n")
    before = _workspace_files(root)

    with pytest.raises(ValueError, match="both PASS and blocked"):
        experiments.experiment_status(root)
    assert _workspace_files(root) == before


@pytest.mark.parametrize(
    "mutation",
    ["missing_questions_json", "missing_questions_md", "launch_script", "config", "runs"],
)
def test_experiment_status_rejects_partial_registered_blocked_plan(tmp_path, mutation):
    root = tmp_path / "experiment"
    recipe = write_finetune_recipe(root, include_label=False)
    blocked_dir = root / "plans" / "blocked"
    assert plans.build_plan(recipe_path=recipe, output_dir=blocked_dir).exit_code == 2

    if mutation.startswith("missing_questions"):
        suffix = ".json" if mutation.endswith("json") else ".md"
        (blocked_dir / f"questions{suffix}").unlink()
        error = "Managed file is missing"
    elif mutation == "runs":
        (blocked_dir / "runs").mkdir()
        error = "directory entries differ"
    else:
        name = "run.sh" if mutation == "launch_script" else "config.yaml"
        (blocked_dir / name).write_text("partial artifact\n")
        error = "directory entries differ"
    before = _workspace_files(root)

    with pytest.raises(ValueError, match=error):
        experiments.experiment_status(root)
    assert _workspace_files(root) == before


@pytest.mark.parametrize("alias_kind", ["symlink", "hardlink"])
def test_experiment_status_rejects_aliased_registered_blocked_plan(tmp_path, alias_kind):
    root = tmp_path / "experiment"
    recipe = write_finetune_recipe(root, include_label=False)
    blocked_dir = root / "plans" / "blocked"
    assert plans.build_plan(recipe_path=recipe, output_dir=blocked_dir).exit_code == 2
    blocked_path = blocked_dir / "plan.blocked.md"
    contents = blocked_path.read_bytes()
    blocked_path.unlink()
    outside = root / "outside-blocked.md"
    outside.write_bytes(contents)
    if alias_kind == "symlink":
        blocked_path.symlink_to(outside)
    else:
        blocked_path.hardlink_to(outside)

    with pytest.raises(ValueError, match="missing or aliased"):
        experiments.experiment_status(root)


def test_experiment_status_preserves_registered_step_io_metadata(tmp_path):
    root = tmp_path / "experiment"
    recipe = write_finetune_recipe(root)
    recipe_payload = yaml.safe_load(recipe.read_text())
    step_spec = write_yaml(
        root / "step-spec.yaml",
        {
            **recipe_payload["step"],
            "inputs": ["config.yaml"],
            "outputs": ["reports/ranking.csv"],
        },
    )
    experiments.register_experiment_step(root, step_spec)
    plan_dir = root / "plans" / "train"
    assert plans.build_plan(recipe_path=recipe, output_dir=plan_dir).exit_code == 0
    before = _workspace_files(root)

    snapshot = experiments.experiment_status(root)

    assert snapshot["summary"]["state"] == "ready_to_launch"
    assert _workspace_files(root) == before
    step_path = root / "steps" / recipe_payload["step"]["id"] / "step.yaml"
    step_manifest = yaml.safe_load(step_path.read_text())
    assert step_manifest["step"]["inputs"] == ["config.yaml"]
    assert step_manifest["step"]["outputs"] == ["reports/ranking.csv"]

    step_manifest["step"]["purpose"] = "Different purpose."
    step_path.write_text(yaml.safe_dump(step_manifest, sort_keys=False))
    with pytest.raises(ValueError, match="step metadata differs"):
        experiments.experiment_status(root)


def test_experiment_status_rejects_unknown_registered_step_metadata(tmp_path):
    root = tmp_path / "experiment"
    _init_workspace(root)
    _add_plan(root, step_id="train")
    step_path = root / "steps" / "train" / "step.yaml"
    step_manifest = yaml.safe_load(step_path.read_text())
    step_manifest["step"]["purpsoe"] = "typo"
    step_path.write_text(yaml.safe_dump(step_manifest, sort_keys=False))

    with pytest.raises(ValueError, match="Unknown step field: purpsoe"):
        experiments.experiment_status(root)


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

    with pytest.raises(ValueError, match="Invalid registered plan recipe|frozen recipe"):
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


@pytest.mark.parametrize(
    ("layer", "binding"),
    [("_base_recipe", "experiment"), ("_local_recipe", "step")],
)
def test_experiment_status_rejects_layered_hparam_binding_drift(tmp_path, layer, binding):
    root = tmp_path / "experiment"
    _init_workspace(root)
    plan_dir, _canonical = _add_plan(root, step_id="tune", task="hparam_tune")
    plan_path = plan_dir / "plan.json"
    plan = json.loads(plan_path.read_text())
    plan["recipe"][layer][binding] = "invalid"
    resolved_path = plan_dir / "recipe.resolved.yaml"
    resolved = yaml.safe_load(resolved_path.read_text())
    resolved[layer][binding] = "invalid"
    resolved_path.write_text(yaml.safe_dump(resolved, sort_keys=False))
    plan["resolved_recipe_sha256"] = _sha256(resolved_path)
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    before = _workspace_files(root)

    with pytest.raises(ValueError, match=f"Invalid registered .* recipe binding: {binding} must be a mapping"):
        experiments.experiment_status(root)
    assert _workspace_files(root) == before


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


@pytest.mark.parametrize("search_kind", ["grid", "configurations"])
def test_experiment_status_rejects_hparam_parameter_drift_shared_by_plan_and_canonical_rows(tmp_path, search_kind):
    root = tmp_path / "experiment"
    recipe = _write_public_hparam_recipe(root, {"runtime.lr": [1e-6, 2e-6]})
    if search_kind == "configurations":
        payload = yaml.safe_load(recipe.read_text())
        payload["search"] = {
            "method": "grid",
            "max_runs": 2,
            "configurations": [{"runtime.lr": 1e-6}, {"runtime.lr": 2e-6}],
        }
        recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
    else:
        payload = yaml.safe_load(recipe.read_text())
        payload["search"]["max_runs"] = 2
        recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
    plan_dir = root / "plans" / "tune"
    assert plans.build_plan(recipe_path=recipe, output_dir=plan_dir).exit_code == 0

    plan_path = plan_dir / "plan.json"
    plan = json.loads(plan_path.read_text())
    plan["runs"][0]["runtime.lr"] = 9e-6
    plan["runs"][0]["parameter_summary"] = "runtime.lr=9e-06"
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    rows = _read_manifest_rows(root)
    rows[0]["runtime.lr"] = "9e-06"
    rows[0]["parameter_summary"] = "runtime.lr=9e-06"
    write_rows(root / "run_manifest.tsv", rows)
    before = _workspace_files(root)

    with pytest.raises(ValueError, match="parameter|canonical"):
        experiments.experiment_status(root)
    assert _workspace_files(root) == before


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
        ("missing_member", "missing final_eval_config"),
        ("missing_bundle", "missing final_eval_config"),
        ("missing_sha256", "final_eval_config must define"),
        ("file_drift", "frozen file SHA-256 changed"),
        ("coherent_file_drift", "frozen recipe digest"),
        ("extra_script_command", "final external-test script differs"),
    ],
)
def test_experiment_status_requires_final_eval_config_integrity(tmp_path, drift, error):
    root = tmp_path / "experiment"
    recipe = _write_public_hparam_recipe(root, {"yaml:/finetune/task/output_dim": [31]})
    payload = yaml.safe_load(recipe.read_text())
    base_recipe = yaml.safe_load(Path(payload["base_recipe"]).read_text())
    final_config_path = root / "selected-final-config.yaml"
    final_config = yaml.safe_load(Path(base_recipe["inputs"]["config"]).read_text())
    final_config["model"]["head"].update({"channel_agg": {"name": "mean"}, "temporal_agg": {"name": "mean"}})
    final_config_path.write_text(yaml.safe_dump(final_config, sort_keys=False))
    checkpoint = root / "selected.ckpt"
    checkpoint.write_text("checkpoint\n")
    payload["inputs"].update({"ckpt_path": str(checkpoint), "final_eval_config_path": str(final_config_path)})
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
    plan_dir = root / "plans" / "tune"
    assert plans.build_plan(recipe_path=recipe, output_dir=plan_dir, unlock_final_test=True).exit_code == 0
    assert experiments.experiment_status(root)["summary"]["state"] == "ready_to_launch"

    plan_path = plan_dir / "plan.json"
    plan = json.loads(plan_path.read_text())
    frozen_config = Path(plan["final_eval_config"]["path"])
    if drift in {"missing_member", "missing_bundle"}:
        del plan["final_eval_config"]
        if drift == "missing_bundle":
            frozen_config.unlink()
            (plan_dir / "final_external_test.sh").unlink()
        plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    elif drift == "missing_sha256":
        del plan["final_eval_config"]["sha256"]
        plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    elif drift == "file_drift":
        frozen_config.write_text("model: changed\n")
    elif drift == "coherent_file_drift":
        frozen_config.write_text("model: changed\n")
        plan["final_eval_config"]["sha256"] = _sha256(frozen_config)
        plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    else:
        final_script = plan_dir / "final_external_test.sh"
        final_script.write_text(final_script.read_text() + "echo injected\n")

    before = _workspace_files(root)

    with pytest.raises(ValueError, match=error):
        experiments.experiment_status(root)
    assert _workspace_files(root) == before


def test_experiment_status_rejects_coherent_registered_command_drift(tmp_path):
    root = tmp_path / "experiment"
    recipe = _write_public_hparam_recipe(root, {"runtime.lr": [1e-6]})
    plan_dir = root / "plans" / "tune"
    assert plans.build_plan(recipe_path=recipe, output_dir=plan_dir).exit_code == 0

    plan_path = plan_dir / "plan.json"
    plan = json.loads(plan_path.read_text())
    run = plan["runs"][0]
    command = run["command"]
    tokens = command.split()
    lr_index = tokens.index("--lr") + 1
    changed_command = command.replace(f"--lr {tokens[lr_index]}", "--lr 9e-06", 1)
    assert changed_command != command
    script_path = Path(run["script"])
    script_path.write_text(script_path.read_text().replace(command, changed_command, 1))
    run["command"] = changed_command
    run["script_sha256"] = _sha256(script_path)
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    rows = _read_manifest_rows(root)
    rows[0]["script_sha256"] = run["script_sha256"]
    write_rows(root / "run_manifest.tsv", rows)
    before = _workspace_files(root)

    with pytest.raises(ValueError, match="command"):
        experiments.experiment_status(root)
    assert _workspace_files(root) == before


def test_experiment_status_rejects_coherent_generic_command_drift(tmp_path):
    root = tmp_path / "experiment"
    recipe = write_finetune_recipe(root)
    plan_dir = root / "plans" / "train"
    assert plans.build_plan(recipe_path=recipe, output_dir=plan_dir).exit_code == 0

    plan_path = plan_dir / "plan.json"
    plan = json.loads(plan_path.read_text())
    old_command = plan["commands"][0]
    tokens = old_command.split()
    label_index = tokens.index("--label-name") + 1
    changed_command = old_command.replace(f"--label-name {tokens[label_index]}", "--label-name changed", 1)
    assert changed_command != old_command
    run = plan["runs"][0]
    run["command"] = changed_command
    plan["commands"] = [changed_command]
    for script_path in (plan_dir / "run.sh", Path(run["script"])):
        script_path.write_text(script_path.read_text().replace(old_command, changed_command, 1))
    run["script_sha256"] = _sha256(Path(run["script"]))
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    rows = _read_manifest_rows(root)
    rows[0]["script_sha256"] = run["script_sha256"]
    write_rows(root / "run_manifest.tsv", rows)
    before = _workspace_files(root)

    with pytest.raises(ValueError, match="commands differ from its frozen recipe"):
        experiments.experiment_status(root)
    assert _workspace_files(root) == before


@pytest.mark.parametrize("task", ["finetune", "hparam_tune"])
def test_experiment_status_rejects_extra_frozen_script_command(tmp_path, task):
    root = tmp_path / "experiment"
    if task == "hparam_tune":
        recipe = _write_public_hparam_recipe(root, {"runtime.lr": [1e-6]})
        plan_dir = root / "plans" / "tune"
    else:
        recipe = write_finetune_recipe(root)
        plan_dir = root / "plans" / "train"
    assert plans.build_plan(recipe_path=recipe, output_dir=plan_dir).exit_code == 0

    plan_path = plan_dir / "plan.json"
    plan = json.loads(plan_path.read_text())
    run = plan["runs"][0]
    script_path = Path(run["script"])
    script_path.write_text(script_path.read_text() + "echo injected\n")
    if task == "finetune":
        (plan_dir / "run.sh").write_text(script_path.read_text())
    run["script_sha256"] = _sha256(script_path)
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    rows = _read_manifest_rows(root)
    rows[0]["script_sha256"] = run["script_sha256"]
    write_rows(root / "run_manifest.tsv", rows)

    with pytest.raises(
        ValueError,
        match="script differs from its frozen recipe|run.sh differs from its frozen recipe|script_sha256",
    ):
        experiments.experiment_status(root)


def test_experiment_status_rejects_missing_hparam_source_config(tmp_path):
    root = tmp_path / "experiment"
    recipe = _write_public_hparam_recipe(root, {"runtime.lr": [1e-6]})
    plan_dir = root / "plans" / "tune"
    assert plans.build_plan(recipe_path=recipe, output_dir=plan_dir).exit_code == 0
    (plan_dir / "config.source.yaml").unlink()
    before = _workspace_files(root)

    with pytest.raises(ValueError, match="config.source.yaml"):
        experiments.experiment_status(root)
    assert _workspace_files(root) == before


def test_experiment_status_rejects_coherent_hparam_config_drift(tmp_path):
    root = tmp_path / "experiment"
    recipe = _write_public_hparam_recipe(root, {"yaml:/finetune/task/output_dim": [31]})
    plan_dir = root / "plans" / "tune"
    assert plans.build_plan(recipe_path=recipe, output_dir=plan_dir).exit_code == 0
    plan_path = plan_dir / "plan.json"
    plan = json.loads(plan_path.read_text())
    run = plan["runs"][0]
    config_path = Path(run["config"])
    config = yaml.safe_load(config_path.read_text())
    config["finetune"]["task"]["output_dim"] = 32
    config_path.write_text(yaml.safe_dump(config))
    run["config_sha256"] = _sha256(config_path)
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    rows = _read_manifest_rows(root)
    rows[0]["config_sha256"] = run["config_sha256"]
    write_rows(root / "run_manifest.tsv", rows)

    with pytest.raises(
        ValueError, match="config differs from its frozen recipe|canonical expected runs field config_sha256"
    ):
        experiments.experiment_status(root)


def test_experiment_status_rejects_coherent_generic_config_drift(tmp_path):
    root = tmp_path / "experiment"
    recipe = write_finetune_recipe(root)
    plan_dir = root / "plans" / "train"
    assert plans.build_plan(recipe_path=recipe, output_dir=plan_dir).exit_code == 0
    plan_path = plan_dir / "plan.json"
    plan = json.loads(plan_path.read_text())
    run = plan["runs"][0]
    config_path = Path(run["config"])
    config_path.write_text(config_path.read_text() + "\n")
    run["config_sha256"] = _sha256(config_path)
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    rows = _read_manifest_rows(root)
    rows[0]["config_sha256"] = run["config_sha256"]
    write_rows(root / "run_manifest.tsv", rows)

    with pytest.raises(ValueError, match="Frozen generic config differs from its recipe digest"):
        experiments.experiment_status(root)


def test_experiment_status_rejects_coherent_slurm_script_drift(tmp_path):
    root = tmp_path / "experiment"
    recipe = _write_public_hparam_recipe(root, {"runtime.lr": [1e-6]})
    payload = yaml.safe_load(recipe.read_text())
    payload.setdefault("execution", {}).update(
        {
            "gpus_per_run": 1,
            "scheduler": {
                "type": "slurm",
                "partition": "gpu",
                "cpus_per_task": 8,
                "memory": "64G",
                "walltime": "01:00:00",
                "nice": 0,
            },
        }
    )
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
    plan_dir = root / "plans" / "tune"
    assert plans.build_plan(recipe_path=recipe, output_dir=plan_dir).exit_code == 0
    plan_path = plan_dir / "plan.json"
    plan = json.loads(plan_path.read_text())
    run = plan["runs"][0]
    script_path = Path(run["scheduler_script"])
    script_path.write_text(script_path.read_text() + "echo injected\n")
    run["scheduler_script_sha256"] = _sha256(script_path)
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    rows = _read_manifest_rows(root)
    rows[0]["scheduler_script_sha256"] = run["scheduler_script_sha256"]
    write_rows(root / "run_manifest.tsv", rows)

    with pytest.raises(ValueError, match="Slurm script differs from its frozen recipe|scheduler_script_sha256"):
        experiments.experiment_status(root)


def test_experiment_status_rejects_hparam_run_omission_shared_by_plan_and_canonical_rows(tmp_path):
    root = tmp_path / "experiment"
    recipe = _write_public_hparam_recipe(root, {"runtime.lr": [1e-6, 2e-6]})
    payload = yaml.safe_load(recipe.read_text())
    payload["search"]["max_runs"] = 2
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
    plan_dir = root / "plans" / "tune"
    assert plans.build_plan(recipe_path=recipe, output_dir=plan_dir).exit_code == 0

    plan_path = plan_dir / "plan.json"
    plan = json.loads(plan_path.read_text())
    plan["runs"] = plan["runs"][:1]
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    write_rows(root / "run_manifest.tsv", _read_manifest_rows(root)[:1])
    before = _workspace_files(root)

    with pytest.raises(ValueError, match="canonical expected runs"):
        experiments.experiment_status(root)
    assert _workspace_files(root) == before


@pytest.mark.parametrize(
    "drift",
    ["partial_runtime", "hparam_parameter_summary", "input_snapshots", "command"],
)
def test_experiment_status_rejects_incomplete_registered_plan_identity(tmp_path, drift):
    root = tmp_path / "experiment"
    _init_workspace(root)
    if drift in {"hparam_parameter_summary", "command"}:
        task = "hparam_tune"
    elif drift == "partial_runtime":
        task = "finetune"
    else:
        task = "sleep2stat"
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
    assert first["decision"]["blocked_actions"] == ["adaptive_advance", "finalize", "launch", "resubmit"]


def test_experiment_status_snapshot_is_independent_of_input_order():
    root = Path("/experiment")
    rows = [
        {"step_id": step_id, "run_id": "run-000", "run_name": "default", "status": "planned"}
        for step_id in ("second", "first")
    ]
    registered_steps = [
        {
            "manifest": {
                "step": {"id": row["step_id"], "phase": "train", "purpose": "Run the fixture."},
                "plan_controller": "ordinary",
            },
            "plans": [
                {
                    "path": str(root / "plans" / row["step_id"]),
                    "task": "finetune",
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
def test_experiment_status_uncertain_states_block_planned_launch_and_recommend_only_monitor(tmp_path, status):
    root = tmp_path / "experiment"
    _init_workspace(root)
    _add_plan(root, step_id="train", status=status)
    _add_plan(root, step_id="planned", status="planned")

    snapshot = experiments.experiment_status(root)

    assert snapshot["summary"]["state"] == "blocked"
    assert snapshot["decision"]["recommended_next"]["id"] == "experiment-monitor"
    assert snapshot["decision"]["other_legal_actions"] == []
    assert snapshot["decision"]["blocked_actions"] == ["adaptive_advance", "finalize", "launch", "resubmit"]


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


def test_experiment_status_rejects_coherent_adaptive_recipe_downgrade(tmp_path):
    root = tmp_path / "experiment"
    _init_workspace(root)
    plan_dir, _canonical = _add_plan(root, step_id="tune", task="hparam_tune", adaptive=True)
    plan_path = plan_dir / "plan.json"
    plan = json.loads(plan_path.read_text())
    plan["recipe"].pop("adaptive")
    plan["recipe"]["_local_recipe"].pop("adaptive")
    resolved_path = plan_dir / "recipe.resolved.yaml"
    resolved = yaml.safe_load(resolved_path.read_text())
    resolved.pop("adaptive")
    resolved["_local_recipe"].pop("adaptive")
    resolved_path.write_text(yaml.safe_dump(resolved, sort_keys=False))
    plan["resolved_recipe_sha256"] = _sha256(resolved_path)
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    before = _workspace_files(root)

    with pytest.raises(ValueError, match="controller differs from its frozen recipe"):
        experiments.experiment_status(root)

    assert _workspace_files(root) == before


def test_experiment_status_rejects_pipeline_identity_downgrade(tmp_path):
    root = tmp_path / "experiment"
    _init_workspace(root)
    _plan_dir, canonical = _add_plan(root, step_id="evaluate", pipeline=True)
    for field in ("pipeline_id", "job_id", "attempt", "result_root"):
        canonical.pop(field)
    write_rows(root / "run_manifest.tsv", [canonical])
    before = _workspace_files(root)

    with pytest.raises(ValueError, match="pipeline run identity is incomplete"):
        experiments.experiment_status(root)

    assert _workspace_files(root) == before


def test_experiment_status_rejects_missing_step_controller_without_writing(tmp_path):
    root = tmp_path / "experiment"
    _init_workspace(root)
    _add_plan(root, step_id="train")
    step_path = root / "steps" / "train" / "step.yaml"
    step_manifest = yaml.safe_load(step_path.read_text())
    step_manifest.pop("plan_controller")
    step_path.write_text(yaml.safe_dump(step_manifest, sort_keys=False))
    before = _workspace_files(root)

    with pytest.raises(ValueError, match="incomplete canonical envelope"):
        experiments.experiment_status(root)

    assert _workspace_files(root) == before


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
                "plan_controller": "unassigned",
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


def test_experiment_finalize_rejects_unmaterialized_step(tmp_path):
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
                "plan_controller": "unassigned",
                "recipe_path": "",
                "plans": [],
            },
            sort_keys=False,
        )
    )
    report = tmp_path / "final.md"
    report.write_text("# Final\n")
    before = _workspace_files(root)

    with pytest.raises(ValueError, match="incomplete canonical steps: unmaterialized_step"):
        experiments.finalize_experiment(root, report)

    assert _workspace_files(root) == before


@pytest.mark.parametrize("registration_mutation", ["empty", "missing"])
def test_experiment_finalize_rejects_canonical_runs_without_registered_plan(tmp_path, registration_mutation):
    root = tmp_path / "experiment"
    _init_workspace(root)
    _add_plan(root, step_id="tune", task="hparam_tune", status="completed")
    step_path = root / "steps" / "tune" / "step.yaml"
    if registration_mutation == "empty":
        step_manifest = yaml.safe_load(step_path.read_text())
        step_manifest["plans"] = []
        step_path.write_text(yaml.safe_dump(step_manifest, sort_keys=False))
    else:
        step_path.unlink()
    report = tmp_path / "final.md"
    report.write_text("# Final\n")
    before = _workspace_files(root)

    with pytest.raises(
        ValueError, match="plans differ from canonical run keys|unregistered steps|Managed file is missing"
    ):
        experiments.finalize_experiment(root, report)

    assert _workspace_files(root) == before
    assert not (root / "reports" / "final.md").exists()


@pytest.mark.parametrize("modern_evidence", ["managed_parameter", "selection_metadata"])
def test_experiment_finalize_does_not_downgrade_managed_hparam_evidence_to_legacy(tmp_path, modern_evidence):
    root = tmp_path / "experiment"
    recipe = _write_public_hparam_recipe(root, {"runtime.lr": [1e-6]})
    plan_dir = root / "plans" / "tune"
    assert plans.build_plan(recipe_path=recipe, output_dir=plan_dir).exit_code == 0
    rows = _read_manifest_rows(root)
    row = rows[0]
    row["status"] = "completed"
    legacy_run_identity_fields = {"experiment_id", "step_id", "run_id", "run_name", "version"}
    for field in FROZEN_RUN_FIELDS - legacy_run_identity_fields:
        row[field] = ""
    for field in managed_run_parameters(row):
        row[field] = ""
    if modern_evidence == "managed_parameter":
        row["runtime.lr"] = "1e-6"
    else:
        row.update(
            {
                "selection_task": "hparam_tune",
                "selection_mode": "min",
                "selection_split": "val",
                "selection_report": str(root / "reports" / "hparam_selection.md"),
                "selection_report_sha256": "a" * 64,
            }
        )
    write_rows(root / "run_manifest.tsv", rows)
    step_path = root / "steps" / "unit-hparam-tune" / "step.yaml"
    step_path.unlink()
    (step_path.parent / ".step.yaml.cas.lock").unlink()
    step_path.parent.rmdir()
    report = tmp_path / "final.md"
    report.write_text("# Final\n")
    before = _workspace_files(root)

    with pytest.raises(ValueError, match="unregistered steps"):
        experiments.finalize_experiment(root, report)

    assert _workspace_files(root) == before
    assert not (root / "reports" / "final.md").exists()


@pytest.mark.parametrize("controller", ["adaptive", "pipeline"])
def test_experiment_finalize_preserves_controller_verified_finalization(tmp_path, controller):
    root = tmp_path / "experiment"
    _init_workspace(root)
    _add_plan(
        root,
        step_id="tune" if controller == "adaptive" else "evaluate",
        task="hparam_tune" if controller == "adaptive" else "finetune",
        status="completed",
        adaptive=controller == "adaptive",
        pipeline=controller == "pipeline",
    )
    report = tmp_path / "controller-report.md"
    report.write_text("# Controller-verified report\n")

    target = experiments.finalize_experiment(root, report)

    assert target.read_text() == report.read_text()


def test_pipeline_facade_callback_can_finalize_controller_verified_report(tmp_path, monkeypatch):
    root = tmp_path / "experiment"
    _init_workspace(root)
    _add_plan(root, step_id="evaluate", status="completed", pipeline=True)
    report = tmp_path / "pipeline-report.md"
    report.write_text("# Pipeline report\n")
    spec = tmp_path / "pipeline.yaml"
    spec.write_text("pipeline: unit\n")

    def controller(_root, _spec, **kwargs):
        target = kwargs["finalize_callback"](root, report)
        return {"status": "completed", "final_report": str(target)}

    monkeypatch.setattr(experiment_pipeline, "run_experiment_pipeline", controller)

    result = experiments.run_experiment_pipeline(root, spec, execute=True)

    assert result == {"status": "completed", "final_report": str(root / "reports" / "final.md")}
    assert (root / "reports" / "final.md").read_text() == report.read_text()


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


def test_experiment_status_advances_ordinary_hparam_selection_and_report(tmp_path):
    root = tmp_path / "experiment"
    _init_workspace(root)
    plan_dir, _row = _add_plan(root, step_id="tune", task="hparam_tune", status="completed")
    before = _workspace_files(root)

    pending = experiments.experiment_status(root)

    assert _workspace_files(root) == before
    assert pending["summary"]["state"] == "ready_to_select"
    assert pending["decision"]["recommended_next"]["argv"] == [
        "python",
        "-m",
        "agent_tools",
        "hparam-select",
        "--run-dir",
        str(plan_dir),
    ]
    assert pending["decision"]["blocked_actions"] == ["finalize"]

    report_path = _record_hparam_selection(root)
    selected_before = _workspace_files(root)
    selected = experiments.experiment_status(root)
    assert _workspace_files(root) == selected_before
    assert selected["summary"]["state"] == "ready_to_report"
    assert selected["decision"]["recommended_next"]["id"] == "hparam-select"

    _record_hparam_selection(root, write_report=True)
    ready_before = _workspace_files(root)
    ready = experiments.experiment_status(root)
    assert _workspace_files(root) == ready_before
    assert ready["summary"]["state"] == "ready_to_finalize"
    assert ready["decision"]["recommended_next"]["argv"][-2:] == ["--report", str(report_path)]


def test_experiment_status_requires_combined_report_for_mixed_ordinary_steps(tmp_path):
    root = tmp_path / "experiment"
    _init_workspace(root)
    _add_plan(root, step_id="tune", task="hparam_tune", status="completed")
    _add_plan(root, step_id="prepare", status="completed")
    _record_hparam_selection(root, write_report=True)

    snapshot = experiments.experiment_status(root)

    assert snapshot["summary"]["state"] == "ready_to_report"
    assert snapshot["decision"]["manual_choice_required"] is True
    assert snapshot["decision"]["other_legal_actions"][0]["required_inputs"] == ["report_path"]
    assert "combined_report_required" in {blocker["code"] for blocker in snapshot["blockers"]}


def test_pipeline_step_prevents_hparam_selection_report_from_becoming_experiment_final(tmp_path):
    root = tmp_path / "experiment"
    _init_workspace(root)
    _add_plan(root, step_id="tune", task="hparam_tune", status="completed")
    _add_plan(root, step_id="evaluate", status="completed", pipeline=True)
    selection_report = _record_hparam_selection(root, write_report=True)
    combined = tmp_path / "pipeline-combined.md"
    combined.write_text("# Pipeline combined report\n")

    with pytest.raises(ValueError, match="cannot replace the required combined experiment report"):
        experiments.finalize_experiment(root, selection_report)

    target = experiments.finalize_experiment(root, combined)

    assert target.read_text() == combined.read_text()


def test_experiment_status_all_failed_hparam_requires_failure_report_not_selection(tmp_path):
    root = tmp_path / "experiment"
    _init_workspace(root)
    _add_plan(root, step_id="tune", task="hparam_tune", status="failed")

    snapshot = experiments.experiment_status(root)

    assert snapshot["summary"]["state"] == "ready_to_report"
    assert snapshot["decision"]["recommended_next"] is None
    assert snapshot["decision"]["other_legal_actions"][0]["required_inputs"] == ["report_path"]
    assert "failure_report_required" in {blocker["code"] for blocker in snapshot["blockers"]}


def test_experiment_status_rejects_selection_metadata_on_non_hparam_run(tmp_path):
    root = tmp_path / "experiment"
    _init_workspace(root)
    _add_plan(root, step_id="train", status="completed")
    rows = _read_manifest_rows(root)
    rows[0].update(
        {
            "selection_task": "hparam_tune",
            "selection_mode": "max",
            "selection_split": "val",
            "selection_report": str(root / "reports" / "hparam_selection.md"),
            "selection_report_sha256": "0" * 64,
        }
    )
    write_rows(root / "run_manifest.tsv", rows)
    report = tmp_path / "final.md"
    report.write_text("# Final report\n")
    before = _workspace_files(root)

    with pytest.raises(ValueError, match="not owned by a registered hparam plan"):
        experiments.experiment_status(root)
    with pytest.raises(ValueError, match="not owned by a registered hparam plan"):
        experiments.finalize_experiment(root, report)

    assert _workspace_files(root) == before


def test_hparam_selection_lifecycle_rejects_misowned_metadata_in_mixed_step(tmp_path):
    registered_steps = [
        {
            "manifest": {
                "step": {"id": "mixed", "phase": "train", "purpose": "mixed plan types"},
                "plan_controller": "ordinary",
            },
            "plans": [
                {
                    "task": "hparam_tune",
                    "run_keys": [("mixed", "run-000")],
                    "path": str(tmp_path / "hparam"),
                    "selection": {"metric": "val_loss", "mode": "min", "split": "val"},
                },
                {
                    "task": "finetune",
                    "run_keys": [("mixed", "run-001")],
                    "path": str(tmp_path / "finetune"),
                },
            ],
        }
    ]
    rows = [
        {"step_id": "mixed", "run_id": "run-000", "status": "planned"},
        {
            "step_id": "mixed",
            "run_id": "run-001",
            "status": "completed",
            "selection_task": "hparam_tune",
        },
    ]

    with pytest.raises(ValueError, match="mixed / run-001"):
        experiment_tracking.hparam_selection_lifecycle(registered_steps, rows, root=tmp_path)


def test_experiment_status_rejects_all_failed_hparam_with_stale_checkpoint_rank(tmp_path):
    root = tmp_path / "experiment"
    _init_workspace(root)
    _add_plan(root, step_id="tune", task="hparam_tune", status="failed")
    rows = _read_manifest_rows(root)
    rows[0]["checkpoint_rank"] = "1"
    write_rows(root / "run_manifest.tsv", rows)
    report = tmp_path / "failure.md"
    report.write_text("# Failure report\n")
    before = _workspace_files(root)

    with pytest.raises(ValueError, match="stale for all-failed step"):
        experiments.experiment_status(root)
    with pytest.raises(ValueError, match="stale for all-failed step"):
        experiments.finalize_experiment(root, report)

    assert _workspace_files(root) == before


def test_experiment_status_keeps_completed_legacy_hparam_selection_readable(tmp_path):
    root = tmp_path / "experiment"
    _init_workspace(root)
    _add_plan(root, step_id="tune", task="hparam_tune", status="completed")
    rows = _read_manifest_rows(root)
    rows[0].update({"metric": "val_loss", "score": "0.25", "rank": "1", "checkpoint_path": "/legacy.ckpt"})
    write_rows(root / "run_manifest.tsv", rows)

    active = experiments.experiment_status(root)
    assert active["summary"]["state"] == "ready_to_select"

    manifest = yaml.safe_load((root / "experiment.yaml").read_text())
    manifest["experiment"].update({"status": "completed", "completed_at": "2026-08-26T00:00:00Z"})
    (root / "experiment.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))
    completed = experiments.experiment_status(root)
    assert completed["summary"]["state"] == "completed"


@pytest.mark.parametrize("field", ["selection_task", "selection_report_sha256"])
def test_experiment_status_rejects_partially_materialized_selection_metadata(tmp_path, field):
    root = tmp_path / "experiment"
    _init_workspace(root)
    _add_plan(root, step_id="tune", task="hparam_tune", status="completed")
    rows = _read_manifest_rows(root)
    rows[0][field] = "hparam_tune" if field == "selection_task" else "0" * 64
    write_rows(root / "run_manifest.tsv", rows)

    with pytest.raises(ValueError, match="partially materialized"):
        experiments.experiment_status(root)


def test_experiment_finalize_requires_selection_and_uses_verified_selection_report(tmp_path):
    root = tmp_path / "experiment"
    _init_workspace(root)
    _add_plan(root, step_id="tune", task="hparam_tune", status="completed")
    arbitrary = tmp_path / "arbitrary.md"
    arbitrary.write_text("# Arbitrary\n")

    with pytest.raises(ValueError, match="must be selected"):
        experiments.finalize_experiment(root, arbitrary)

    selection_report = _record_hparam_selection(root, write_report=True)
    with pytest.raises(ValueError, match="must finalize from"):
        experiments.finalize_experiment(root, arbitrary)

    target = experiments.finalize_experiment(root, selection_report)
    assert target.read_text() == selection_report.read_text()
    completed = yaml.safe_load((root / "experiment.yaml").read_text())["experiment"]
    assert completed["final_report"] == str(target)
    assert completed["final_report_sha256"] == _sha256(target)
    assert completed["selection_report_sha256"] == _sha256(selection_report)


def test_experiment_finalize_rehashes_selected_checkpoints_before_writing(tmp_path):
    root = tmp_path / "experiment"
    _init_workspace(root)
    _add_plan(root, step_id="tune", task="hparam_tune", status="completed")
    selection_report = _record_hparam_selection(root, write_report=True)
    checkpoint = Path(_read_manifest_rows(root)[0]["checkpoint_path"])
    checkpoint.write_bytes(b"tampered checkpoint\n")
    before = _workspace_files(root)

    assert experiments.experiment_status(root)["summary"]["state"] == "ready_to_finalize"
    with pytest.raises(ValueError, match="Frozen checkpoint SHA-256 differs"):
        experiments.finalize_experiment(root, selection_report)

    assert _workspace_files(root) == before
    assert not (root / "reports" / "final.md").exists()


@pytest.mark.parametrize(
    ("target", "host", "remote", "expected_host"),
    [("local", "", "controller", "controller"), ("ssh", "worker", "controller", "worker")],
)
def test_hparam_checkpoint_rehash_uses_execution_evidence_host(
    monkeypatch, target: str, host: str, remote: str, expected_host: str
):
    row = {
        "step_id": "tune",
        "run_id": "run-000",
        "target": target,
        "host": host,
        "checkpoint_path": "/data/epoch=1.ckpt",
        "checkpoint_sha256": "a" * 64,
    }
    validated = []
    hashed = []
    monkeypatch.setattr(
        experiments.tracking,
        "validate_checkpoint_evidence_rows",
        lambda rows, ranked, *, remote: validated.append((rows, ranked, remote)),
    )

    def checkpoint_sha256(evidence_row, checkpoint_path):
        hashed.append((evidence_row, checkpoint_path))
        return "a" * 64

    monkeypatch.setattr(experiments.evidence, "checkpoint_file_sha256", checkpoint_sha256)

    experiments._validate_hparam_checkpoints([row], [{"ranked": [row]}], remote=remote)

    assert validated == [([row], [row], remote)]
    assert hashed[0][0]["target"] == "ssh"
    assert hashed[0][0]["host"] == expected_host
    assert hashed[0][1] == row["checkpoint_path"]


def test_experiment_finalize_rejects_selection_report_copy_for_pure_hparam_experiment(tmp_path):
    root = tmp_path / "experiment"
    _init_workspace(root)
    _add_plan(root, step_id="tune", task="hparam_tune", status="completed")
    selection_report = _record_hparam_selection(root, write_report=True)
    copied_report = root / "selection-copy.md"
    copied_report.write_bytes(selection_report.read_bytes())
    before = _workspace_files(root)

    with pytest.raises(ValueError, match="must finalize from"):
        experiments.finalize_experiment(root, copied_report)

    assert _workspace_files(root) == before
    assert not (root / "reports" / "final.md").exists()


@pytest.mark.parametrize("mutation", ["delete", "tamper"])
def test_experiment_status_validates_bound_final_report(tmp_path, mutation):
    root = tmp_path / "experiment"
    _init_workspace(root)
    _add_plan(root, step_id="train", status="completed")
    report = tmp_path / "report.md"
    report.write_text("# Final report\n")
    target = experiments.finalize_experiment(root, report)
    if mutation == "delete":
        target.unlink()
    else:
        target.write_text("# Tampered final report\n")

    with pytest.raises(ValueError, match="final report|Managed file is missing"):
        experiments.experiment_status(root)


def test_experiment_status_rejects_incomplete_terminal_report_binding(tmp_path):
    root = tmp_path / "experiment"
    _init_workspace(root)
    _add_plan(root, step_id="train", status="completed")
    report = tmp_path / "report.md"
    report.write_text("# Final report\n")
    experiments.finalize_experiment(root, report)
    manifest_path = root / "experiment.yaml"
    manifest = yaml.safe_load(manifest_path.read_text())
    manifest["experiment"].pop("final_report_sha256")
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False))

    with pytest.raises(ValueError, match="incomplete or unexpected terminal fields"):
        experiments.experiment_status(root)


def test_experiment_status_requires_terminal_bindings_for_modern_hparam_selection(tmp_path):
    root = tmp_path / "experiment"
    _init_workspace(root)
    _add_plan(root, step_id="tune", task="hparam_tune", status="completed")
    selection_report = _record_hparam_selection(root, write_report=True)
    final = experiments.finalize_experiment(root, selection_report)
    manifest_path = root / "experiment.yaml"
    manifest = yaml.safe_load(manifest_path.read_text())
    for field in ("final_report", "final_report_sha256", "selection_report_sha256"):
        manifest["experiment"].pop(field)
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False))
    final.write_text("# Tampered final report\n")

    with pytest.raises(ValueError, match="missing terminal report bindings"):
        experiments.experiment_status(root)


def test_experiment_status_rejects_modern_completion_downgraded_to_legacy_selection(tmp_path):
    root = tmp_path / "experiment"
    _init_workspace(root)
    _add_plan(root, step_id="tune", task="hparam_tune", status="completed")
    selection_report = _record_hparam_selection(root, write_report=True)
    experiments.finalize_experiment(root, selection_report)

    rows = _read_manifest_rows(root)
    for row in rows:
        for field in (
            "selection_task",
            "metric",
            "selection_mode",
            "selection_split",
            "score",
            "rank",
            "checkpoint_path",
            "checkpoint_sha256",
            "selection_report",
            "selection_report_sha256",
        ):
            row[field] = ""
    write_rows(root / "run_manifest.tsv", rows)
    manifest_path = root / "experiment.yaml"
    manifest = yaml.safe_load(manifest_path.read_text())
    manifest["experiment"].pop("selection_report_sha256")
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False))

    with pytest.raises(ValueError, match="incomplete hparam selection evidence"):
        experiments.experiment_status(root)


def test_experiment_status_detects_selection_commit_after_terminal_binding(tmp_path):
    root = tmp_path / "experiment"
    _init_workspace(root)
    _add_plan(root, step_id="tune", task="hparam_tune", status="completed")
    selection_report = _record_hparam_selection(root, write_report=True)
    final = experiments.finalize_experiment(root, selection_report)
    final_bytes = final.read_bytes()

    _record_hparam_selection(root, write_report=True, score="0.5")

    with pytest.raises(ValueError, match="selection report differs from its terminal binding"):
        experiments.experiment_status(root)
    assert final.read_bytes() == final_bytes


def test_experiment_finalize_rejects_selection_report_changed_after_snapshot(tmp_path, monkeypatch):
    root = tmp_path / "experiment"
    _init_workspace(root)
    _add_plan(root, step_id="tune", task="hparam_tune", status="completed")
    selection_report = _record_hparam_selection(root, write_report=True)
    manifest = root / "experiment.yaml"
    manifest_before = manifest.read_bytes()
    original_reader = experiments._hparam_selection_report

    def read_then_tamper(*args, **kwargs):
        payload = original_reader(*args, **kwargs)
        selection_report.write_text("# Tampered after verification\n")
        return payload

    monkeypatch.setattr(experiments, "_hparam_selection_report", read_then_tamper)

    with pytest.raises(ValueError, match="selection report changed during finalization"):
        experiments.finalize_experiment(root, selection_report)

    assert manifest.read_bytes() == manifest_before
    assert not (root / "events.jsonl").exists()
    assert not (root / "reports" / "final.md").exists()
    assert selection_report.read_text() == "# Tampered after verification\n"


def test_experiment_finalize_rechecks_selection_report_before_terminal_commit(tmp_path, monkeypatch):
    root = tmp_path / "experiment"
    _init_workspace(root)
    _add_plan(root, step_id="tune", task="hparam_tune", status="completed")
    selection_report = _record_hparam_selection(root, write_report=True)
    manifest = root / "experiment.yaml"
    manifest_before = manifest.read_bytes()
    original_replace = experiment_io.conditional_atomic_replace_text_at

    def publish_then_tamper(path, text, expected_sha256, *, remote=None):
        committed = original_replace(path, text, expected_sha256, remote=remote)
        if Path(path) == root / "reports" / "final.md" and committed:
            selection_report.write_text("# Tampered before terminal commit\n")
        return committed

    monkeypatch.setattr(experiment_io, "conditional_atomic_replace_text_at", publish_then_tamper)

    with pytest.raises(ValueError, match="selection report changed during finalization"):
        experiments.finalize_experiment(root, selection_report)

    assert manifest.read_bytes() == manifest_before
    assert (root / "reports" / "final.md").exists()
    assert yaml.safe_load(manifest.read_text())["experiment"].get("status") is None


def test_experiment_finalize_rechecks_ranking_before_terminal_commit(tmp_path, monkeypatch):
    root = tmp_path / "experiment"
    _init_workspace(root)
    _add_plan(root, step_id="tune", task="hparam_tune", status="completed")
    selection_report = _record_hparam_selection(root, write_report=True)
    ranking = root / "reports" / "ranking.csv"
    manifest = root / "experiment.yaml"
    manifest_before = manifest.read_bytes()
    original_replace = experiment_io.conditional_atomic_replace_text_at

    def publish_then_tamper(path, text, expected_sha256, *, remote=None):
        committed = original_replace(path, text, expected_sha256, remote=remote)
        if Path(path) == root / "reports" / "final.md" and committed:
            ranking.write_text(ranking.read_text().replace("0.25", "999", 1))
        return committed

    monkeypatch.setattr(experiment_io, "conditional_atomic_replace_text_at", publish_then_tamper)

    with pytest.raises(ValueError, match="ranking changed during finalization"):
        experiments.finalize_experiment(root, selection_report)

    assert manifest.read_bytes() == manifest_before
    assert (root / "reports" / "final.md").exists()
    assert yaml.safe_load(manifest.read_text())["experiment"].get("status") is None


def test_experiment_finalize_rechecks_checkpoint_before_terminal_commit(tmp_path, monkeypatch):
    root = tmp_path / "experiment"
    _init_workspace(root)
    _add_plan(root, step_id="tune", task="hparam_tune", status="completed")
    selection_report = _record_hparam_selection(root, write_report=True)
    checkpoint = Path(_read_manifest_rows(root)[0]["checkpoint_path"])
    manifest = root / "experiment.yaml"
    manifest_before = manifest.read_bytes()
    original_replace = experiment_io.conditional_atomic_replace_text_at

    def publish_then_tamper(path, text, expected_sha256, *, remote=None):
        committed = original_replace(path, text, expected_sha256, remote=remote)
        if Path(path) == root / "reports" / "final.md" and committed:
            checkpoint.write_bytes(b"tampered before terminal commit\n")
        return committed

    monkeypatch.setattr(experiment_io, "conditional_atomic_replace_text_at", publish_then_tamper)

    with pytest.raises(ValueError, match="Frozen checkpoint SHA-256 differs"):
        experiments.finalize_experiment(root, selection_report)

    assert manifest.read_bytes() == manifest_before
    assert (root / "reports" / "final.md").exists()
    assert yaml.safe_load(manifest.read_text())["experiment"].get("status") is None


def test_experiment_finalize_allows_combined_or_failure_reports(tmp_path):
    mixed = tmp_path / "mixed"
    _init_workspace(mixed)
    _add_plan(mixed, step_id="tune", task="hparam_tune", status="completed")
    _add_plan(mixed, step_id="prepare", status="completed")
    selection_report = _record_hparam_selection(mixed, write_report=True)
    with pytest.raises(ValueError, match="cannot replace"):
        experiments.finalize_experiment(mixed, selection_report)
    combined = tmp_path / "combined.md"
    combined.write_text("# Combined report\n\nSelection and preparation summary.\n")
    assert experiments.finalize_experiment(mixed, combined).read_text() == combined.read_text()

    failed = tmp_path / "failed"
    _init_workspace(failed)
    _add_plan(failed, step_id="tune", task="hparam_tune", status="failed")
    stale_selection = failed / "reports" / "hparam_selection.md"
    stale_selection.parent.mkdir()
    stale_selection.write_text("# Stale selection\n")
    with pytest.raises(ValueError, match="cannot replace the required hparam failure report"):
        experiments.finalize_experiment(failed, stale_selection)
    failure_report = tmp_path / "failure.md"
    failure_report.write_text("# Failure report\n\nNo candidate completed successfully.\n")
    assert experiments.finalize_experiment(failed, failure_report).read_text() == failure_report.read_text()


def test_experiment_finalize_rejects_selection_report_dotdot_alias_for_mixed_experiment(tmp_path):
    root = tmp_path / "mixed"
    _init_workspace(root)
    _add_plan(root, step_id="tune", task="hparam_tune", status="completed")
    _add_plan(root, step_id="prepare", status="completed")
    selection_report = _record_hparam_selection(root, write_report=True)
    aliased_report = selection_report.parent / ".." / "reports" / selection_report.name
    before = _workspace_files(root)

    with pytest.raises(ValueError, match="cannot replace the required combined experiment report"):
        experiments.finalize_experiment(root, aliased_report)

    assert _workspace_files(root) == before
    assert not (root / "reports" / "final.md").exists()


def test_experiment_finalize_rejects_selection_report_copy_as_failure_report(tmp_path):
    root = tmp_path / "failed"
    _init_workspace(root)
    _add_plan(root, step_id="tune", task="hparam_tune", status="failed")
    selection_report = root / "reports" / "hparam_selection.md"
    selection_report.parent.mkdir()
    selection_report.write_text("# Stale selection report\n")
    same_content = root / "failure.md"
    same_content.write_bytes(selection_report.read_bytes())
    before = _workspace_files(root)

    with pytest.raises(ValueError, match="cannot replace the required hparam failure report"):
        experiments.finalize_experiment(root, same_content)

    assert _workspace_files(root) == before
    assert not (root / "reports" / "final.md").exists()


def test_experiment_status_requires_combined_report_when_one_hparam_step_failed(tmp_path):
    root = tmp_path / "experiment"
    _init_workspace(root)
    _add_plan(root, step_id="selected", task="hparam_tune", status="completed")
    _add_plan(root, step_id="failed", task="hparam_tune", status="failed")
    selection_report = _record_hparam_selection(root, step_id="selected", write_report=True)

    snapshot = experiments.experiment_status(root)

    assert snapshot["summary"]["state"] == "ready_to_report"
    assert "combined_report_required" in {blocker["code"] for blocker in snapshot["blockers"]}
    with pytest.raises(ValueError, match="cannot replace"):
        experiments.finalize_experiment(root, selection_report)
    combined = tmp_path / "combined.md"
    combined.write_text("# Combined report\n\nOne hparam step failed.\n")
    assert experiments.finalize_experiment(root, combined).read_text() == combined.read_text()


def test_experiment_finalize_rejects_stale_selection_after_success_becomes_failed(tmp_path):
    root = tmp_path / "experiment"
    _init_workspace(root)
    _add_plan(root, step_id="tune", task="hparam_tune", status="completed")
    selection_report = _record_hparam_selection(root, write_report=True)
    rows = _read_manifest_rows(root)
    rows[0]["status"] = "failed"
    write_rows(root / "run_manifest.tsv", rows)
    before = _workspace_files(root)

    with pytest.raises(ValueError, match="stale for all-failed step"):
        experiments.experiment_status(root)
    with pytest.raises(ValueError, match="stale for all-failed step"):
        experiments.finalize_experiment(root, selection_report)

    assert _workspace_files(root) == before


@pytest.mark.parametrize("mutation", ["missing", "tampered", "duplicate"])
def test_experiment_status_rejects_missing_or_drifted_hparam_ranking(tmp_path, mutation):
    root = tmp_path / "experiment"
    _init_workspace(root)
    _add_plan(root, step_id="tune", task="hparam_tune", status="completed")
    selection_report = _record_hparam_selection(root, write_report=True)
    ranking = root / "reports" / "ranking.csv"
    if mutation == "missing":
        ranking.unlink()
    elif mutation == "tampered":
        ranking.write_text(ranking.read_text().replace("0.25", "999", 1))
    else:
        lines = ranking.read_text().splitlines()
        ranking.write_text("\n".join([*lines, lines[1]]) + "\n")
    before = _workspace_files(root)

    snapshot = experiments.experiment_status(root)

    assert snapshot["summary"]["state"] == "ready_to_report"
    assert snapshot["decision"]["recommended_next"]["id"] == "hparam-select"
    with pytest.raises(ValueError, match="selection report is missing or differs"):
        experiments.finalize_experiment(root, selection_report)
    assert _workspace_files(root) == before


@pytest.mark.parametrize(
    ("field", "mutation"),
    [
        ("run_name", "tamper"),
        ("parameter_summary", "tamper"),
        ("version", "tamper"),
        ("config", "tamper"),
        ("runtime.lr", "tamper"),
        ("run_manifest", "tamper"),
        ("status", "tamper"),
        ("run_manifest", "remove"),
        ("status", "remove"),
        ("checkpoint_rank", "empty"),
        ("config", "remove"),
        ("runtime.lr", "remove"),
        ("unexpected", "add"),
    ],
)
def test_experiment_status_rejects_hparam_ranking_candidate_provenance_drift(tmp_path, field, mutation):
    root = tmp_path / "experiment"
    recipe = _write_public_hparam_recipe(
        root,
        {"runtime.lr": [1e-6]},
        selection_metric="val_loss",
        selection_mode="min",
    )
    plan_dir = root / "plans" / "tune"
    assert plans.build_plan(recipe_path=recipe, output_dir=plan_dir).exit_code == 0
    rows = _read_manifest_rows(root)
    for row in rows:
        row["status"] = "completed"
    write_rows(root / "run_manifest.tsv", rows)
    selection_report = _record_hparam_selection(root, step_id="unit-hparam-tune", write_report=True)
    ranking = root / "reports" / "ranking.csv"
    ranking_rows = read_rows(ranking, require_managed_identity=True)
    if mutation == "remove":
        for row in ranking_rows:
            row.pop(field)
    elif mutation == "empty":
        ranking_rows[0][field] = ""
    else:
        ranking_rows[0][field] = "tampered"
    write_rows(ranking, ranking_rows)

    snapshot = experiments.experiment_status(root)

    assert snapshot["summary"]["state"] == "ready_to_report"
    assert snapshot["decision"]["recommended_next"]["id"] == "hparam-select"
    with pytest.raises(ValueError, match="selection report is missing or differs"):
        experiments.finalize_experiment(root, selection_report)


@pytest.mark.parametrize(
    ("selection_metric", "selection_mode", "scores"),
    [
        ("val_ahi_pearson", "max", ("0.1", "0.9")),
        ("val_loss", "min", ("0.9", "0.1")),
    ],
)
def test_experiment_status_rejects_hparam_ranks_opposed_to_selection_mode(
    tmp_path, selection_metric, selection_mode, scores
):
    root = tmp_path / "experiment"
    recipe = _write_public_hparam_recipe(
        root,
        {"runtime.lr": [1e-6, 2e-6]},
        selection_metric=selection_metric,
        selection_mode=selection_mode,
        max_runs=2,
    )
    plan_dir = root / "plans" / "tune"
    assert plans.build_plan(recipe_path=recipe, output_dir=plan_dir).exit_code == 0
    rows = _read_manifest_rows(root)
    for rank, (row, score) in enumerate(zip(rows, scores), start=1):
        row.update(
            {
                "status": "completed",
                "selection_task": "hparam_tune",
                "metric": selection_metric,
                "selection_mode": selection_mode,
                "selection_split": "val",
                "score": score,
                "rank": str(rank),
                "checkpoint_path": str(Path(row["checkpoint_dir"]) / f"epoch={rank}.ckpt"),
                "checkpoint_sha256": "a" * 64,
            }
        )
    write_rows(root / "run_manifest.tsv", rows)
    report = tmp_path / "report.md"
    report.write_text("# Report\n")
    before = _workspace_files(root)

    with pytest.raises(ValueError, match="ranks disagree with selection mode"):
        experiments.experiment_status(root)
    with pytest.raises(ValueError, match="ranks disagree with selection mode"):
        experiments.finalize_experiment(root, report)
    assert _workspace_files(root) == before


def test_experiment_status_keeps_final_report_blocker_experiment_wide(tmp_path):
    root = tmp_path / "experiment"
    _init_workspace(root)
    _add_plan(root, step_id="prepare", status="completed")
    _add_plan(root, step_id="evaluate", status="failed")

    snapshot = experiments.experiment_status(root)

    blocker = next(item for item in snapshot["blockers"] if item["code"] == "final_report_required")
    assert blocker["step_id"] is None
    assert blocker["run_ids"] == []
    assert all("final_report_required" not in run["blockers"] for run in snapshot["runs"])


def test_experiment_status_blocks_finalize_for_stopped_run_without_reason(tmp_path):
    root = tmp_path / "experiment"
    _init_workspace(root)
    _add_plan(root, step_id="train", status="stopped")
    before = _workspace_files(root)

    blocked = experiments.experiment_status(root)

    assert _workspace_files(root) == before
    assert blocked["summary"]["state"] == "blocked"
    assert blocked["decision"]["recommended_next"] is None
    assert blocked["decision"]["other_legal_actions"] == []
    assert blocked["decision"]["blocked_actions"] == ["finalize"]
    assert blocked["blockers"][0]["code"] == "missing_stop_reason"
    assert blocked["blockers"][0]["run_ids"] == ["run-000"]
    assert blocked["runs"][0]["blockers"] == ["missing_stop_reason"]

    rows = _read_manifest_rows(root)
    rows[0]["stop_reason"] = "manual stop after invalid labels"
    write_rows(root / "run_manifest.tsv", rows)
    ready = experiments.experiment_status(root)
    assert ready["summary"]["state"] == "ready_to_finalize"
    assert ready["decision"]["other_legal_actions"][0]["id"] == "experiment-finalize"


def test_experiment_status_rejects_completed_stopped_run_without_reason(tmp_path):
    root = tmp_path / "experiment"
    _init_workspace(root)
    _add_plan(root, step_id="train", status="stopped")
    manifest = yaml.safe_load((root / "experiment.yaml").read_text())
    manifest["experiment"].update({"status": "completed", "completed_at": "2026-08-25T02:00:00Z"})
    (root / "experiment.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))

    with pytest.raises(ValueError, match="stopped runs missing stop_reason"):
        experiments.experiment_status(root)

    rows = _read_manifest_rows(root)
    rows[0]["stop_reason"] = "manual stop after invalid labels"
    write_rows(root / "run_manifest.tsv", rows)
    assert experiments.experiment_status(root)["summary"]["state"] == "completed"


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

    (root / "experiment_manifest.tsv").write_text("not\ta\tvalid\tprojection\n")
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


def test_experiment_status_does_not_use_following_read_for_experiment_manifest(tmp_path, monkeypatch):
    root = tmp_path / "experiment"
    _init_workspace(root)
    _add_plan(root, step_id="train")
    manifest = root / "experiment.yaml"
    outside = tmp_path / "outside-experiment.yaml"
    outside.write_bytes(manifest.read_bytes())
    real_read = experiment_io.read_text_at
    followed = False

    def swap_then_read(path, *, remote=None):
        nonlocal followed
        if Path(path) == manifest:
            manifest.unlink()
            manifest.symlink_to(outside)
            followed = True
        return real_read(path, remote=remote)

    monkeypatch.setattr(experiment_io, "read_text_at", swap_then_read)

    snapshot = experiments.experiment_status(root)

    assert snapshot["experiment"]["id"] == "status-unit"
    assert not followed


def test_experiment_status_human_output_quotes_advisory_argv(tmp_path):
    root = tmp_path / "experiment with spaces"
    _init_workspace(root)
    _add_plan(root, step_id="train")

    rendered = experiment_tracking.format_experiment_status(experiments.experiment_status(root))

    assert "recorded evidence, not live" in rendered
    assert "| Run | Canonical | Scheduler | Process | Checkpoints | Runtime manifest | Blocker |" in rendered
    assert "Test evidence" not in rendered
    assert "Next legal action" in rendered
    assert "'" in rendered
    assert "execution host:" not in rendered
    assert "Advisory only; this output does not authorize execution." in rendered


def test_experiment_status_human_output_scopes_same_code_blockers(tmp_path):
    root = tmp_path / "experiment"
    _init_workspace(root)
    for step_id in ("first", "second"):
        step_dir = root / "steps" / step_id
        step_dir.mkdir(parents=True)
        (step_dir / "step.yaml").write_text(
            yaml.safe_dump(
                {
                    "step": {"id": step_id, "phase": "evaluate", "purpose": f"Run {step_id}."},
                    "experiment_id": "status-unit",
                    "plan_controller": "unassigned",
                    "recipe_path": "",
                    "plans": [],
                },
                sort_keys=False,
            )
        )

    rendered = experiment_tracking.format_experiment_status(experiments.experiment_status(root))

    assert "`unmaterialized_step` [step=first]" in rendered
    assert "`unmaterialized_step` [step=second]" in rendered


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
            "manifest": {
                "step": {"id": "train", "phase": "train", "purpose": "Train."},
                "plan_controller": "ordinary",
            },
            "plans": [
                {
                    "path": str(root / "plans" / "train"),
                    "task": "finetune",
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
            "manifest": {
                "step": {"id": "train", "phase": "train", "purpose": "Train."},
                "plan_controller": "ordinary",
            },
            "plans": [
                {
                    "path": str(root / "plans" / "train"),
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


def test_experiment_status_rejects_unknown_experiment_envelope_field(tmp_path):
    root = tmp_path / "experiment"
    _init_workspace(root)
    manifest_path = root / "experiment.yaml"
    manifest = yaml.safe_load(manifest_path.read_text())
    manifest["duplicate_owner"] = {"status": "completed"}
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False))

    with pytest.raises(ValueError, match="only the experiment owner mapping"):
        experiments.experiment_status(root)


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


@pytest.mark.parametrize("binding", ["experiment", "step"])
def test_experiment_status_cli_returns_one_for_non_mapping_plan_binding(tmp_path, capsys, binding):
    root = tmp_path / "experiment"
    _init_workspace(root)
    plan_dir, _canonical = _add_plan(root, step_id="train")
    plan_path = plan_dir / "plan.json"
    plan = json.loads(plan_path.read_text())
    plan["recipe"][binding] = "invalid"
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    resolved_path = plan_dir / "recipe.resolved.yaml"
    resolved = yaml.safe_load(resolved_path.read_text())
    resolved[binding] = "invalid"
    resolved_path.write_text(yaml.safe_dump(resolved, sort_keys=False))
    before = _workspace_files(root)

    assert cli.main(["experiment-status", "--run-dir", str(root), "--json"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert f"{binding} must be a mapping" in captured.err
    assert "Traceback" not in captured.err
    assert _workspace_files(root) == before


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
    ("parameter", "mutation"),
    [
        ("yaml:/finetune/task/output_dim", "missing_key"),
        ("yaml:/data/data_channel_names/0", "missing_index"),
        ("yaml:/finetune/task/output_dim", "wrong_parent_type"),
        ("yaml:/finetune/task/output_dim", "malformed_yaml"),
    ],
)
def test_experiment_status_cli_converts_corrupt_frozen_hparam_config_errors(
    tmp_path,
    capsys,
    parameter,
    mutation,
):
    root = tmp_path / "experiment"
    recipe = _write_public_hparam_recipe(root, {parameter: [31 if parameter.endswith("output_dim") else "ppg"]})
    plan_dir = root / "plans" / "tune"
    assert plans.build_plan(recipe_path=recipe, output_dir=plan_dir).exit_code == 0

    source_config = plan_dir / "config.source.yaml"
    if mutation == "malformed_yaml":
        source_config.write_text("[unclosed")
    else:
        config = yaml.safe_load(source_config.read_text())
        if mutation == "missing_key":
            del config["finetune"]["task"]
        elif mutation == "missing_index":
            config["data"]["data_channel_names"] = []
        else:
            config["finetune"] = 1
        source_config.write_text(yaml.safe_dump(config, sort_keys=False))

    plan_path = plan_dir / "plan.json"
    plan = json.loads(plan_path.read_text())
    source_sha256 = _sha256(source_config)
    for snapshot in plan["recipe"]["input_snapshots"]:
        if snapshot["field"] == "inputs.config":
            snapshot["sha256"] = source_sha256
    resolved_path = plan_dir / "recipe.resolved.yaml"
    resolved = yaml.safe_load(resolved_path.read_text())
    resolved["input_snapshots"] = plan["recipe"]["input_snapshots"]
    resolved_path.write_text(yaml.safe_dump(resolved, sort_keys=False))
    plan["resolved_recipe_sha256"] = _sha256(resolved_path)
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")

    before = _workspace_files(root)
    assert cli.main(["experiment-status", "--run-dir", str(root), "--json"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Registered plan frozen config is corrupt" in captured.err
    assert "Traceback" not in captured.err
    assert _workspace_files(root) == before


@pytest.mark.parametrize("mutation", ["malformed_yaml", "invalid_analyzer"])
def test_experiment_status_cli_converts_corrupt_sleep2stat_config_error(tmp_path, capsys, mutation):
    root = tmp_path / "experiment"
    _init_workspace(root)
    plan_dir, canonical = _add_plan(root, step_id="analyze", task="sleep2stat")
    config_path = Path(canonical["config"])
    if mutation == "malformed_yaml":
        config_path.write_text("[unclosed")
    else:
        config_path.write_text(
            yaml.safe_dump(
                {"run": {"output_dir": str(root / "analysis")}, "analyzers": ["invalid"], "reducers": []},
                sort_keys=False,
            )
        )
    config_sha256 = _sha256(config_path)

    plan_path = plan_dir / "plan.json"
    plan = json.loads(plan_path.read_text())
    plan["recipe"]["input_snapshots"][0]["sha256"] = config_sha256
    plan["runs"][0]["config_sha256"] = config_sha256
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    resolved_path = plan_dir / "recipe.resolved.yaml"
    resolved = yaml.safe_load(resolved_path.read_text())
    resolved["input_snapshots"] = plan["recipe"]["input_snapshots"]
    resolved_path.write_text(yaml.safe_dump(resolved, sort_keys=False))
    canonical["config_sha256"] = config_sha256
    write_rows(root / "run_manifest.tsv", [canonical])

    before = _workspace_files(root)
    assert cli.main(["experiment-status", "--run-dir", str(root), "--json"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Registered plan frozen config is corrupt" in captured.err
    assert "Traceback" not in captured.err
    assert _workspace_files(root) == before


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


def test_experiment_status_rejects_duplicate_plan_json_keys(tmp_path, capsys):
    root = tmp_path / "experiment"
    _init_workspace(root)
    plan_dir, _canonical = _add_plan(root, step_id="train")
    plan_path = plan_dir / "plan.json"
    source = plan_path.read_text()
    plan_path.write_text('{\n  "status": "FAIL",\n' + source.lstrip()[1:])
    before = _workspace_files(root)

    assert cli.main(["experiment-status", "--run-dir", str(root), "--json"]) == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "duplicate JSON key: status" in captured.err
    assert "Traceback" not in captured.err
    assert _workspace_files(root) == before


def test_experiment_status_rejects_plan_escape_before_external_probe(tmp_path, monkeypatch):
    root = tmp_path / "experiment"
    _init_workspace(root)
    _add_plan(root, step_id="train")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "plan.blocked.md").write_text("blocked\n")
    step_path = root / "steps" / "train" / "step.yaml"
    step_manifest = yaml.safe_load(step_path.read_text())
    step_manifest["plans"] = [str(outside)]
    step_path.write_text(yaml.safe_dump(step_manifest, sort_keys=False))
    path_exists_at = experiment_io.path_exists_at

    def reject_external_probe(path, *, remote=None):
        try:
            Path(path).relative_to(root)
        except ValueError as exc:
            raise AssertionError(f"status probed outside the canonical workspace: {path}") from exc
        return path_exists_at(path, remote=remote)

    monkeypatch.setattr(experiment_io, "path_exists_at", reject_external_probe)

    with pytest.raises(ValueError, match="outside its managed workspace"):
        experiments.experiment_status(root)


def test_experiment_status_rejects_aliased_plan_directory_before_file_probe(tmp_path, monkeypatch):
    root = tmp_path / "experiment"
    _init_workspace(root)
    _add_plan(root, step_id="train")
    outside = tmp_path / "outside"
    outside.mkdir()
    alias = root / "plans" / "alias"
    alias.symlink_to(outside, target_is_directory=True)
    step_path = root / "steps" / "train" / "step.yaml"
    step_manifest = yaml.safe_load(step_path.read_text())
    step_manifest["plans"] = [str(alias)]
    step_path.write_text(yaml.safe_dump(step_manifest, sort_keys=False))
    path_exists_at = experiment_io.path_exists_at

    def reject_aliased_file_probe(path, *, remote=None):
        if Path(path).is_relative_to(alias):
            raise AssertionError(f"status probed through an aliased plan directory: {path}")
        return path_exists_at(path, remote=remote)

    monkeypatch.setattr(experiment_io, "path_exists_at", reject_aliased_file_probe)

    with pytest.raises(ValueError, match="missing or aliased"):
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
                "plan_controller": "ordinary",
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
        "plan_controller": "ordinary",
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

    def managed_workspace(candidate, *, remote, allow_completed, validate_experiment_index):
        calls.append(("workspace", candidate, remote, allow_completed, validate_experiment_index))
        return experiment, [row]

    def registered_steps(candidate, *, experiment_id, remote):
        calls.append(("steps", candidate, experiment_id, remote))
        return [manifest]

    def registered_blocked_plan(candidate, *, workspace, remote):
        calls.append(("blocked", candidate, workspace, remote))
        return False

    def registered_plan(
        candidate,
        *,
        workspace,
        workspace_experiment,
        step_manifest,
        workspace_rows,
        expected_recipe_path,
        remote,
        run_index_offset,
    ):
        calls.append(
            (
                "plan",
                candidate,
                workspace,
                workspace_experiment,
                step_manifest,
                workspace_rows,
                expected_recipe_path,
                remote,
                run_index_offset,
            )
        )
        candidate_path = Path(candidate)
        return {
            "path": str(candidate),
            "task": "finetune",
            "run_keys": [("train", "run-000")],
            "launch_script": str(candidate_path / "run.sh"),
        }

    monkeypatch.setattr(experiments, "_managed_workspace", managed_workspace)
    monkeypatch.setattr(experiments, "read_registered_steps", registered_steps)
    monkeypatch.setattr(experiments.artifacts, "is_registered_blocked_plan", registered_blocked_plan)
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

    assert calls[0] == ("workspace", root, "baichuan3", True, False)
    assert calls[1] == ("steps", root, "status-unit", "baichuan3")
    assert calls[2] == ("blocked", str(root / "plans" / "train"), root, "baichuan3")
    assert calls[3][3] == experiment
    assert calls[3][-3:] == ("", "baichuan3", 0)
    assert snapshot["experiment"]["remote"] == "baichuan3"
    action = snapshot["decision"]["recommended_next"]
    assert action["execution_host"] is None
    assert action["argv"][-2:] == ["--remote", "baichuan3"]
