from __future__ import annotations

import ast
import csv
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from agent_tool_test_helpers import config_payload, run_execution_preflight_fixture, write_finetune_recipe, write_yaml
import pytest
import yaml

from agent_tools import (
    decision_hparam,
    hparam_runtime,
    managed_scheduler,
    manifests,
    plan_hparam,
    python_programs,
    run_evidence,
    transport,
)
from agent_tools.experiment_workspace import MONITOR_EXIT_CODE_PREFIX, file_sha256
from agent_tools.models import REPO_ROOT

_REAL_VALIDATED_EXECUTION_SNAPSHOT = hparam_runtime._validated_execution_snapshot
_RUNTIME_COMMIT = subprocess.run(
    ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True, text=True, capture_output=True
).stdout.strip()


@pytest.fixture(autouse=True)
def _stub_execution_snapshot_preflight(monkeypatch, request):
    def validated_snapshot(run_dir, _execution, _runs, _workspace_by_key):
        snapshot_path = Path(run_dir) / hparam_runtime.EXECUTION_SNAPSHOT_NAME
        if snapshot_path.exists():
            return json.loads(snapshot_path.read_text()), False
        return None, False

    monkeypatch.setattr(hparam_runtime, "_validated_execution_snapshot", validated_snapshot)
    monkeypatch.setattr(managed_scheduler.slurm, "controller_cluster", lambda *_args, **_kwargs: "wuji-h20")
    if not request.node.name.startswith("test_execution_probe_"):
        monkeypatch.setattr(managed_scheduler, "run_execution_command", run_execution_preflight_fixture)


def _run(*args: str) -> subprocess.CompletedProcess:
    runner = Path(__file__).with_name("agent_tools_cli_stub.py")
    return subprocess.run([sys.executable, str(runner), *args], text=True, capture_output=True)


def _hparam_recipe(tmp_path: Path, *, execution: dict | None = None, variant: str = "sleep2vec") -> Path:
    base = write_finetune_recipe(tmp_path, variant=variant)
    execution_payload = dict(execution) if execution is not None else {"workdir": str(tmp_path)}
    manager_runtime = (
        str(execution_payload.get("target", "local") or "local") == "local"
        and execution_payload.get("workdir") in (None, "", str(REPO_ROOT))
        and execution_payload.get("conda_env") in (None, "")
    )
    if not manager_runtime:
        execution_payload.setdefault("python", sys.executable)
        execution_payload.setdefault("runtime_commit", _RUNTIME_COMMIT)
    return write_yaml(
        tmp_path / "tune.yaml",
        {
            "name": "unit_hparam",
            "task": "hparam_tune",
            "variant": variant,
            "base_recipe": str(base),
            "search": {
                "method": "grid",
                "max_runs": 1,
                "parameters": {"runtime.lr": [1e-6]},
            },
            "execution": execution_payload,
            "evaluation_policy": {
                "selection_metric": "val_ahi_pearson",
                "selection_mode": "max",
                "selection_split": "val",
                "external_test_locked": True,
                "test_after_fit": False,
                "final_eval_split": "test",
                "final_test_unlocked": False,
                "require_manual_unlock_for_final_test": True,
            },
            "decisions": {
                "task": {"value": "hparam_tune", "source": "explicit_recipe"},
                "label_name": {"value": "ahi", "source": "explicit_recipe"},
                "external_test_locked": {"value": True, "source": "explicit_recipe"},
                "train_val_test_policy": {
                    "value": "select on val",
                    "source": "explicit_recipe",
                },
                "overwrite_policy": {"value": False, "source": "explicit_recipe"},
                "final_eval_unlock": {"value": False, "source": "explicit_recipe"},
            },
        },
    )


def _write_slurm_plan(
    tmp_path: Path,
    *,
    run_count: int = 1,
    execution: dict | None = None,
    direct_controller: bool = False,
) -> tuple[Path, dict]:
    recipe_path = _hparam_recipe(tmp_path)
    if run_count > 1:
        payload = yaml.safe_load(recipe_path.read_text())
        payload["search"]["max_runs"] = run_count
        payload["search"]["parameters"]["runtime.lr"] = [1e-6 * (index + 1) for index in range(run_count)]
        recipe_path.write_text(yaml.safe_dump(payload, sort_keys=False))
    source_plan_dir = tmp_path / "source-plan"
    result = _run("plan", "--recipe", str(recipe_path), "--output-dir", str(source_plan_dir))
    assert result.returncode == 0, result.stderr
    source_plan = json.loads((source_plan_dir / "plan.json").read_text())
    recipe = source_plan["recipe"]
    recipe["execution"].update(
        {
            "gpus_per_run": 1,
            "scheduler": {
                "type": "slurm",
                "partition": "gpu",
                "cpus_per_task": 8,
                "memory": "64G",
                "walltime": "01:00:00",
            },
        }
    )
    if direct_controller:
        recipe["execution"]["scheduler"]["direct_controller"] = True
    recipe["execution"].update(execution or {})
    source_config = (source_plan_dir / "config.source.yaml").read_bytes()
    plan_dir = tmp_path / "slurm-plan"
    plan_hparam.write_hparam_plan(
        recipe,
        plan_dir,
        unlock_final_test=False,
        source_config_bytes=source_config,
        source_config_sha256=hashlib.sha256(source_config).hexdigest(),
    )
    plan_hparam.preflight_hparam_plan(plan_dir, semantic_out=plan_dir)
    plan_hparam.commit_hparam_plan(plan_dir)
    return plan_dir, json.loads((plan_dir / "plan.json").read_text())


def _read_table(path: Path) -> list[dict[str, str]]:
    delimiter = "\t" if path.suffix == ".tsv" else ","
    with path.open(newline="") as file_obj:
        return list(csv.DictReader(file_obj, delimiter=delimiter))


def _process_identity(pid: int = 123) -> dict[str, int | str]:
    return {"pid": pid, "process_group_id": pid, "process_start_token": "proc:unit-start"}


def _write_process_identity(path: str | Path, pid: int = 123) -> dict[str, int | str]:
    identity = _process_identity(pid)
    Path(path).write_text(json.dumps(identity) + "\n")
    return identity


def _write_proc_stat(proc_root: Path, pid: int, pgid: int, state: str) -> None:
    process_dir = proc_root / str(pid)
    process_dir.mkdir(parents=True)
    (process_dir / "stat").write_text(f"{pid} (unit) {state} 1 {pgid}\n")


def _embedded_process_group_running(proc_root: Path, pgid: int) -> bool | None:
    tree = ast.parse(python_programs.source("run_evidence.process_probe"))
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "process_group_running"
    )
    namespace = {"os": os}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "run_evidence.process_probe", "exec"), namespace)
    return namespace["process_group_running"](pgid, str(proc_root))


def _is_remote_python_program(command: str, name: str) -> bool:
    return command.startswith(f"python3 -c {transport.sh(python_programs.source(name))}")


def _set_execution_probe(
    monkeypatch,
    plan_dir: Path,
    *,
    commit: str | None = None,
    missing_options: set[str] | None = None,
    parse_error: str | None = None,
) -> list[str]:
    plan = json.loads((plan_dir / "plan.json").read_text())
    frozen = json.loads((plan_dir / hparam_runtime.EXECUTION_SNAPSHOT_NAME).read_text())
    command = shlex.split(plan["runs"][0]["command"])
    module = command[command.index("-m") + 1]
    options = set(frozen["supported_options"]) - set(missing_options or set())
    runtime_commit = commit or frozen["runtime_commit"]
    calls = []

    def run_probe(_execution, probe_command):
        if len(probe_command) > 2 and probe_command[2] == python_programs.source("managed_scheduler.runtime_identity"):
            calls.append("identity")
            return subprocess.CompletedProcess(
                probe_command,
                0,
                json.dumps(
                    {
                        "python": frozen["python"],
                        "python_version": frozen["python_version"],
                        "runtime_commit": runtime_commit,
                        "runtime_repo_root": frozen["runtime_repo_root"],
                        "runtime_hostname": frozen["runtime_hostname"],
                        "module": module,
                        "module_origin": frozen["module_origin"],
                    }
                ),
                "",
            )
        calls.append("parse")
        evidence = json.dumps(
            {"supported_options": sorted(options), "cli_options_sha256": frozen["cli_options_sha256"]}
        )
        return subprocess.CompletedProcess(
            probe_command,
            2 if parse_error else 0,
            "" if parse_error else f"AGENT_CLI_PREFLIGHT={evidence}\n",
            parse_error or "",
        )

    monkeypatch.setattr(hparam_runtime, "_run_execution_command", run_probe)
    return calls


def _write_runtime_rows(root: Path, specs: list[dict]) -> list[dict]:
    experiment = {
        "id": "unit-experiment",
        "title": "Unit experiment",
        "objective": "Exercise hparam runtime state transitions.",
        "root": str(root),
        "baseline": {"type": "none", "rationale": "Unit fixture."},
    }
    step = {"id": "train-model", "phase": "train", "purpose": "Exercise managed runs."}
    (root / "experiment.yaml").write_text(yaml.safe_dump({"experiment": experiment}, sort_keys=False))
    step_dir = root / "steps" / step["id"]
    step_dir.mkdir(parents=True, exist_ok=True)
    (step_dir / "step.yaml").write_text(
        yaml.safe_dump(
            {
                "step": step,
                "experiment_id": experiment["id"],
                "plan_controller": "ordinary",
                "recipe_path": "",
                "plans": [str(root.resolve())],
            },
            sort_keys=False,
        )
    )
    runs = []
    rows = []
    for index, spec in enumerate(specs):
        run_id = str(spec["run_id"])
        managed_dir = root / "runs" / run_id
        managed_dir.mkdir(parents=True, exist_ok=True)
        config = managed_dir / "config.yaml"
        script = managed_dir / "launch.sh"
        artifacts_path = managed_dir / "artifacts.json"
        config.write_text(yaml.safe_dump(config_payload(root / "index.csv")))
        script.write_text("#!/usr/bin/env bash\ntrue\n")
        artifacts_path.write_text("{}\n")
        version = str(spec.get("version") or f"version-{index}")
        runtime_dir = root / "log-finetune" / version
        run = {
            "experiment_id": "unit-experiment",
            "step_id": "train-model",
            "run_id": run_id,
            "run_name": run_id,
            "version": version,
            "run_dir": str(managed_dir),
            "runtime_dir": str(runtime_dir),
            "checkpoint_dir": str(runtime_dir / "checkpoints"),
            "config": str(config),
            "config_sha256": file_sha256(config),
            "script": str(script),
            "script_sha256": file_sha256(script),
            "artifacts": str(artifacts_path),
        }
        runs.append(run)
        row = {
            **run,
            "target": "local",
            "host": "",
            "workdir": str(root),
            "gpus": "",
            "pid_path": str(managed_dir / "pid"),
            "log_path": str(managed_dir / "stdout.log"),
            "command": hparam_runtime._launch_command(
                {"workdir": str(root)},
                script,
                managed_dir / "stdout.log",
                managed_dir / "pid",
                [],
            ),
            "status": "planned",
            "launched_at": "",
            **spec,
        }
        rows.append(row)
    resolved_recipe = {
        "variant": "sleep2vec",
        "experiment": experiment,
        "step": step,
        "execution": {"workdir": str(root)},
    }
    resolved_path = root / "recipe.resolved.yaml"
    resolved_path.write_text(yaml.safe_dump(resolved_recipe, sort_keys=False))
    (root / "plan.json").write_text(
        json.dumps(
            {
                "runs": runs,
                "recipe": resolved_recipe,
                "resolved_recipe_sha256": file_sha256(resolved_path),
            }
        )
    )
    manifests.write_rows(
        root / "run_manifest.tsv",
        rows,
    )
    (root / "run_manifest.tsv.lock").touch()
    manifests.write_rows(root / "launch_manifest.tsv", rows)
    manifests.write_rows(root / "run_status.tsv", rows)
    return rows


def test_hparam_launch_rejects_plan_without_workspace_binding_before_start(tmp_path: Path, monkeypatch):
    recipe = _hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"
    assert _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir)).returncode == 0
    plan = json.loads((plan_dir / "plan.json").read_text())
    plan["recipe"].pop("experiment")
    (plan_dir / "plan.json").write_text(json.dumps(plan))
    started = []
    monkeypatch.setattr(
        hparam_runtime, "_start_process", lambda _execution, command: started.append(command) or "launched"
    )

    with pytest.raises(ValueError, match="workspace binding"):
        hparam_runtime.launch_hparam_runs(plan_dir, dry_run=False)

    assert started == []
    assert not (plan_dir / "launch_manifest.tsv").exists()


def test_hparam_plan_canonicalizes_relative_workspace_root_consistently(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    payload["experiment"]["root"] = os.path.relpath(tmp_path, Path.cwd())
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
    plan_dir = tmp_path / "plan"

    plan_result = _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir))
    launch_result = _run("hparam-launch", "--plan-dir", str(plan_dir))

    assert plan_result.returncode == 0, plan_result.stderr
    assert launch_result.returncode == 0, launch_result.stderr
    plan = json.loads((plan_dir / "plan.json").read_text())
    manifest = yaml.safe_load((tmp_path / "experiment.yaml").read_text())
    assert plan["recipe"]["experiment"]["root"] == str(tmp_path)
    assert manifest["experiment"]["root"] == str(tmp_path)


def test_hparam_plan_records_monitor_owned_exit_status_contract(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path)
    plan_dir = tmp_path / "plan"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir))

    assert result.returncode == 0, result.stderr
    run = json.loads((plan_dir / "plan.json").read_text())["runs"][0]
    canonical = _read_table(tmp_path / "run_manifest.tsv")[0]
    script = Path(run["script"]).read_text()
    assert run["terminal_status_owner"] == "monitor"
    assert run["scheduler_type"] == "direct"
    assert json.loads((plan_dir / "plan.json").read_text())["recipe"]["execution"]["scheduler"] == {"type": "direct"}
    assert canonical["terminal_status_owner"] == "monitor"
    assert canonical["scheduler_type"] == "direct"
    assert "trap _agent_tools_record_exit EXIT" in script
    assert MONITOR_EXIT_CODE_PREFIX in script


def test_hparam_plan_rejects_duplicate_gpu_assignments_within_a_run(tmp_path: Path):
    recipe = _hparam_recipe(
        tmp_path,
        execution={"workdir": str(tmp_path), "gpu_pool": [0, 0], "gpus_per_run": 2},
    )

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(tmp_path / "plan"))

    assert result.returncode == 1
    assert "must not contain duplicate GPU identifiers" in result.stdout


@pytest.mark.parametrize("env_name", ["PYTHONPATH", "WANDB_PROJECT", "WANDB_GROUP", "WANDB_RUN_GROUP", "WANDB_MODE"])
def test_hparam_plan_rejects_environment_semantic_aliases(tmp_path: Path, env_name: str):
    recipe = _hparam_recipe(
        tmp_path,
        execution={"workdir": str(tmp_path), "env": {env_name: "unit"}},
    )

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(tmp_path / "plan"))

    assert result.returncode == 1
    assert f"execution.env.{env_name}" in result.stdout


def test_direct_hparam_allows_slurm_named_environment_variable():
    issues = decision_hparam._hparam_execution_issues(
        {"scheduler": {"type": "direct"}, "env": {"SLURM_JOB_ID": "outer-allocation"}},
        {},
    )

    assert not [issue for issue in issues if issue.field == "execution.env.SLURM_JOB_ID"]


def test_hparam_runtime_rewrites_legacy_projection_rows_from_canonical(tmp_path: Path, monkeypatch):
    rows = _write_runtime_rows(tmp_path, [{"run_id": "run-000", "version": "v0", "status": "launched"}])
    legacy_rows = [{**rows[0], "trial_id": "trial_000"}]
    manifests.write_rows(tmp_path / "launch_manifest.tsv", legacy_rows)
    started = []
    killed = []
    monkeypatch.setattr(
        hparam_runtime, "_start_process", lambda _execution, command: started.append(command) or "launched"
    )
    monkeypatch.setattr(run_evidence.os, "kill", lambda pid, sig: killed.append((pid, sig)))

    hparam_runtime.launch_hparam_runs(tmp_path, dry_run=True)

    assert started == []
    assert killed == []
    assert "trial_id" not in (tmp_path / "launch_manifest.tsv").read_text()
    assert "trial_id" not in (tmp_path / "run_status.tsv").read_text()


@pytest.mark.parametrize("table", ["launch_manifest.tsv", "run_status.tsv"])
def test_hparam_runtime_rewrites_header_only_removed_projection_table(tmp_path: Path, monkeypatch, table: str):
    _write_runtime_rows(tmp_path, [{"run_id": "run-000", "version": "v0", "status": "launched"}])
    (tmp_path / table).write_text("trial_id\n")
    started = []
    killed = []
    monkeypatch.setattr(
        hparam_runtime, "_start_process", lambda _execution, command: started.append(command) or "launched"
    )
    monkeypatch.setattr(run_evidence.os, "kill", lambda pid, sig: killed.append((pid, sig)))

    hparam_runtime.launch_hparam_runs(tmp_path, dry_run=True)

    assert started == []
    assert killed == []
    assert "trial_id" not in (tmp_path / table).read_text()


def test_hparam_runtime_rejects_legacy_status_filename(tmp_path: Path, monkeypatch):
    _write_runtime_rows(tmp_path, [{"run_id": "run-000", "version": "v0", "status": "planned"}])
    legacy_status = tmp_path / "trial_status.tsv"
    legacy_status.write_text("trial_id\tstatus\ntrial_000\tfailed\n")
    current_status = (tmp_path / "run_status.tsv").read_bytes()
    started = []
    monkeypatch.setattr(
        hparam_runtime, "_start_process", lambda _execution, command: started.append(command) or "launched"
    )

    with pytest.raises(ValueError, match="Legacy hparam status"):
        hparam_runtime.launch_hparam_runs(tmp_path, dry_run=False)

    assert started == []
    assert (tmp_path / "run_status.tsv").read_bytes() == current_status


def test_hparam_doctor_rejects_invalid_execution_target(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path, execution={"target": "cluster"})

    result = _run("doctor", "--recipe", str(recipe), "--output-dir", str(tmp_path / "doctor"))

    assert result.returncode == 1
    assert "execution.target" in result.stdout


def test_hparam_doctor_rejects_deprecated_log_and_pid_dirs(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path, execution={"log_dir": "logs", "pid_dir": "pids"})

    result = _run("doctor", "--recipe", str(recipe), "--output-dir", str(tmp_path / "doctor"))

    assert result.returncode == 1
    assert "execution.log_dir" in result.stdout
    assert "execution.pid_dir" in result.stdout
