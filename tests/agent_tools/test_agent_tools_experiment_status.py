from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from agent_tool_test_helpers import run_execution_preflight_fixture, write_finetune_recipe, write_yaml
import pytest
import yaml

from agent_tools import experiment_tracking, experiments, managed_scheduler, plan_contract, plan_hparam, plans
from agent_tools.adapters import finetune as finetune_adapter, get_adapter
from agent_tools.experiment_workspace import managed_run_parameters
from agent_tools.manifests import write_rows
from agent_tools.models import REPO_ROOT

_RUNTIME_COMMIT = subprocess.run(
    ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True, text=True, capture_output=True
).stdout.strip()


@pytest.fixture(autouse=True)
def _stub_execution_target(monkeypatch):
    monkeypatch.setattr(managed_scheduler, "run_execution_command", run_execution_preflight_fixture)


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
                "train_val_test_policy": {"value": "val", "source": "explicit_recipe"},
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
