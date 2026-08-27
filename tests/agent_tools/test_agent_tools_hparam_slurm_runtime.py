from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess

import pytest
from test_agent_tools_hparam_runtime import (
    _hparam_recipe,
    _read_table,
    _run,
    _write_runtime_rows,
    _write_slurm_plan,
    write_yaml,
)
from test_agent_tools_hparam_runtime import _stub_execution_snapshot_preflight  # noqa: F401
import yaml

from agent_tools import (
    decision_hparam,
    hparam_runtime,
    managed_scheduler,
    plan_contract,
    plan_hparam,
    plan_rendering,
    plans,
    run_artifacts,
    slurm,
)
from agent_tools.experiment_workspace import MONITOR_EXIT_CODE_PREFIX, file_sha256, next_run_index, run_identity


@pytest.mark.parametrize("direct_controller", [False, True])
def test_slurm_plan_freezes_controller_topology(tmp_path: Path, direct_controller: bool):
    plan_dir, plan = _write_slurm_plan(tmp_path, direct_controller=direct_controller)
    run = plan["runs"][0]
    canonical = next(row for row in _read_table(tmp_path / "run_manifest.tsv") if row["run_id"] == run["run_id"])
    expected = str(direct_controller).lower()

    assert run["scheduler_direct_controller"] == expected
    assert canonical["scheduler_direct_controller"] == expected


@pytest.mark.parametrize("direct_controller", [False, True])
def test_hparam_monitor_uses_canonical_slurm_controller_topology(
    tmp_path: Path,
    monkeypatch,
    direct_controller: bool,
):
    plan_dir, plan = _write_slurm_plan(tmp_path, direct_controller=direct_controller)
    monkeypatch.setattr(
        managed_scheduler.slurm,
        "submit",
        lambda *_args, **_kwargs: slurm.JobIdentity("3880", "wuji-h20"),
    )
    hparam_runtime.launch_hparam_runs(plan_dir, dry_run=False)
    recipe_scheduler = plan["recipe"]["execution"]["scheduler"]
    if direct_controller:
        recipe_scheduler.pop("direct_controller")
    else:
        recipe_scheduler["direct_controller"] = True
    monkeypatch.setattr(run_artifacts, "read_hparam_plan", lambda _path: plan)
    observed_executions = []

    def observe_slurm_run(_root, execution, row, *, health=False):
        observed_executions.append(execution)
        return row

    monkeypatch.setattr(managed_scheduler, "observe_slurm_run", observe_slurm_run)

    hparam_runtime.monitor_hparam_runs(plan_dir)

    execution = {"target": "local"}
    if direct_controller:
        execution["scheduler"] = {"direct_controller": True}
    assert observed_executions == [execution]


@pytest.mark.parametrize(
    ("variant", "gpus_per_run"),
    [("sleep2vec", 1), ("sleep2vec", 2), ("sleep2vec2", 2), ("sleep2expert", 2)],
)
def test_public_hparam_recipe_plans_slurm_leaf_jobs(tmp_path: Path, variant: str, gpus_per_run: int):
    recipe = _hparam_recipe(tmp_path, variant=variant)
    payload = yaml.safe_load(recipe.read_text())
    configured_scheduler = {
        "type": "slurm",
        "partition": "gpu",
        "cpus_per_task": 8,
        "memory": "64G",
        "walltime": "01:00:00",
        "nice": 0,
    }
    payload["execution"].update(
        {
            "gpus_per_run": gpus_per_run,
            "scheduler": configured_scheduler,
        }
    )
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
    plan_dir = tmp_path / "plan"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir))
    preview = _run("hparam-launch", "--plan-dir", str(plan_dir))

    assert result.returncode == 0, result.stderr
    assert "Slurm priority is cluster-managed and cannot be guaranteed" in result.stdout
    assert preview.returncode == 0, preview.stderr
    plan = json.loads((plan_dir / "plan.json").read_text())
    run = plan["runs"][0]
    canonical = _read_table(tmp_path / "run_manifest.tsv")[0]
    assert plan["recipe"]["execution"]["scheduler"] == configured_scheduler
    assert run["scheduler_type"] == "slurm"
    assert canonical["scheduler_type"] == "slurm"
    assert Path(run["scheduler_script"]).name == "job.sbatch"
    scheduler_script = Path(run["scheduler_script"]).read_text()
    assert "#SBATCH --nice=0" in scheduler_script
    assert f"#SBATCH --ntasks={gpus_per_run}" in scheduler_script
    assert f"#SBATCH --ntasks-per-node={gpus_per_run}" in scheduler_script
    assert f"#SBATCH --gres=gpu:{gpus_per_run}" in scheduler_script
    assert f"--gpus-per-run {gpus_per_run}" in scheduler_script
    command = shlex.split(run["command"])
    devices_index = command.index("--devices")
    assert command[devices_index + 1 : devices_index + 1 + gpus_per_run] == [
        str(device) for device in range(gpus_per_run)
    ]
    assert command[devices_index + 1 + gpus_per_run].startswith("--")
    assert "sbatch --parsable" in _read_table(plan_dir / "launch_manifest.tsv")[0]["command"]


def test_slurm_doctor_reports_live_capabilities_without_mutating_scheduler(tmp_path: Path, monkeypatch):
    recipe_path = _hparam_recipe(tmp_path)
    payload = yaml.safe_load(recipe_path.read_text())
    configured_scheduler = {
        "type": "slurm",
        "partition": "gpu",
        "cpus_per_task": 8,
        "memory": "64G",
        "walltime": "01:00:00",
        "nice": 0,
    }
    payload["execution"].update({"gpus_per_run": 1, "scheduler": configured_scheduler})
    recipe_path.write_text(yaml.safe_dump(payload, sort_keys=False))
    recipe, _cfg, report = plans.evaluate_recipe(recipe_path)
    calls = []

    def capabilities(execution, partition):
        calls.append((execution.get("target", "local"), partition))
        return {
            "slurm_version": "slurm 20.11.9",
            "priority_type": "priority/basic",
            "scheduler_type": "sched/backfill",
            "accounting_storage_type": "accounting_storage/none",
            "preempt_type": "preempt/none",
            "multifactor_priority": False,
            "backfill_enabled": True,
            "accounting_enabled": False,
            "preemption_enabled": False,
            "partition": "gpu",
            "partition_state": "UP",
            "partition_max_time": "2-00:00:00",
            "reservation_count": 0,
        }

    monkeypatch.setattr(slurm, "cluster_scheduling_capabilities", capabilities)

    doctor = plans.prepare_doctor_report(None, recipe, report)

    capability_issue = next(issue for issue in doctor.issues if issue.field == "execution.scheduler.capabilities")
    assert capability_issue.status.value == "WARN"
    assert "priority/basic" in capability_issue.message
    assert "no setting can guarantee first priority" in capability_issue.message
    assert capability_issue.evidence["backfill_enabled"] is True
    assert calls == [("local", "gpu")]
    assert recipe["execution"]["scheduler"] == configured_scheduler


def test_slurm_doctor_reports_unavailable_capabilities_as_warning(tmp_path: Path, monkeypatch):
    recipe_path = _hparam_recipe(tmp_path)
    payload = yaml.safe_load(recipe_path.read_text())
    configured_scheduler = {
        "type": "slurm",
        "partition": "gpu",
        "cpus_per_task": 8,
        "memory": "64G",
        "walltime": "01:00:00",
    }
    payload["execution"].update({"gpus_per_run": 1, "scheduler": configured_scheduler})
    recipe_path.write_text(yaml.safe_dump(payload, sort_keys=False))
    recipe, _cfg, report = plans.evaluate_recipe(recipe_path)

    def unavailable(_execution, _partition):
        raise subprocess.TimeoutExpired("scontrol", 10)

    monkeypatch.setattr(slurm, "cluster_scheduling_capabilities", unavailable)

    doctor = plans.prepare_doctor_report(None, recipe, report)

    capability_issue = next(issue for issue in doctor.issues if issue.field == "execution.scheduler.capabilities")
    assert capability_issue.status.value == "WARN"
    assert "inspection was unavailable" in capability_issue.message
    assert recipe["execution"]["scheduler"] == configured_scheduler


@pytest.mark.parametrize(
    "artifact_name", ["job.sbatch", "slurm_terminal.json", "allocation_identity.json", "slurm.log"]
)
def test_slurm_plan_rejects_existing_single_use_artifacts_before_writing(tmp_path: Path, artifact_name: str):
    recipe = _hparam_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    payload["execution"].update(
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
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
    plan_dir = tmp_path / "plan"
    resolved, _cfg, report = plans.evaluate_recipe(recipe)
    assert report.exit_code == 0
    identity = run_identity(resolved, next_run_index(resolved), plan_hparam.hparam_combos(resolved)[0])
    run_dir = plan_dir / "runs" / f"{identity['run_id']}--{identity['run_name']}"
    run_dir.mkdir(parents=True)
    artifact = run_dir / artifact_name
    artifact.write_text("stale\n")
    before = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir))

    assert result.returncode == 1
    assert "Output artifacts already exist" in result.stdout
    assert {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before


def test_slurm_plan_rejects_scheduler_artifact_symlink_before_writing(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    payload["execution"].update(
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
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
    plan_dir = tmp_path / "plan"
    resolved, _cfg, report = plans.evaluate_recipe(recipe)
    assert report.exit_code == 0
    identity = run_identity(resolved, next_run_index(resolved), plan_hparam.hparam_combos(resolved)[0])
    run_dir = plan_dir / "runs" / f"{identity['run_id']}--{identity['run_name']}"
    run_dir.mkdir(parents=True)
    target = tmp_path / "outside.sbatch"
    target.write_text("outside\n")
    artifact = run_dir / "job.sbatch"
    artifact.symlink_to(target)

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir))

    assert result.returncode == 1
    assert "Output artifacts are unsafe" in result.stdout
    assert artifact.is_symlink()
    assert target.read_text() == "outside\n"
    assert not (plan_dir / "plan.json").exists()


@pytest.mark.parametrize(
    ("section", "field", "value", "message"),
    [
        ("execution", "gpu_pool", [0], r"execution\.gpu_pool is not used"),
        ("execution", "max_concurrent", 2, r"execution\.max_concurrent is not used"),
        ("execution", "conda_env", "exp", r"explicit execution\.python path"),
        ("runtime", "devices", [7], r"derive logical runtime\.devices"),
    ],
)
def test_public_slurm_hparam_recipe_rejects_direct_scheduler_controls(
    tmp_path: Path,
    section: str,
    field: str,
    value,
    message: str,
):
    recipe = _hparam_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    payload["execution"]["scheduler"] = {
        "type": "slurm",
        "partition": "gpu",
        "cpus_per_task": 8,
        "memory": "64G",
        "walltime": "01:00:00",
    }
    payload.setdefault(section, {})[field] = value
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(tmp_path / "plan"))

    assert result.returncode == 1
    assert re.search(message, result.stdout)
    assert not (tmp_path / "plan").exists()


def test_public_slurm_hparam_recipe_rejects_unknown_scheduler_fields(tmp_path: Path):
    recipe = _hparam_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    payload["execution"]["scheduler"] = {
        "type": "slurm",
        "partition": "gpu",
        "cpus_per_task": 8,
        "memory": "64G",
        "walltime": "01:00:00",
        "sbatch_args": ["--exclusive"],
    }
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(tmp_path / "plan"))

    assert result.returncode == 1
    assert "Unknown hparam execution.scheduler field: sbatch_args" in result.stdout
    assert not (tmp_path / "plan").exists()


@pytest.mark.parametrize(
    ("scheduler", "message"),
    [
        ({}, "execution.scheduler.type must be direct or slurm"),
        (
            {"type": "slurm", "partition": "gpu", "cpus_per_task": 8, "memory": "64G"},
            "Slurm scheduler resources are missing: walltime",
        ),
    ],
)
def test_public_hparam_recipe_requires_complete_scheduler_contract(tmp_path: Path, scheduler: dict, message: str):
    recipe = _hparam_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    payload["execution"]["scheduler"] = scheduler
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(tmp_path / "plan"))

    assert result.returncode == 1
    assert message in result.stdout
    assert not (tmp_path / "plan").exists()


def test_hparam_plan_freezes_one_slurm_job_per_run_before_registration(tmp_path: Path):
    recipe_path = _hparam_recipe(tmp_path)
    direct_plan_dir = tmp_path / "direct-plan"
    assert _run("plan", "--recipe", str(recipe_path), "--output-dir", str(direct_plan_dir)).returncode == 0
    direct_plan = json.loads((direct_plan_dir / "plan.json").read_text())
    recipe = direct_plan["recipe"]
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
    source_config = (direct_plan_dir / "config.source.yaml").read_bytes()
    slurm_plan_dir = tmp_path / "slurm-plan"

    plan_hparam.write_hparam_plan(
        recipe,
        slurm_plan_dir,
        unlock_final_test=False,
        source_config_bytes=source_config,
        source_config_sha256=hashlib.sha256(source_config).hexdigest(),
    )
    plan_hparam.preflight_hparam_plan(slurm_plan_dir, semantic_out=slurm_plan_dir)
    plan_hparam.commit_hparam_plan(slurm_plan_dir)

    run = json.loads((slurm_plan_dir / "plan.json").read_text())["runs"][0]
    canonical = next(row for row in _read_table(tmp_path / "run_manifest.tsv") if row["run_id"] == run["run_id"])
    batch_script = Path(run["scheduler_script"]).read_text()
    assert run["scheduler_type"] == "slurm"
    assert run["terminal_status_owner"] == "scheduler_sidecar"
    assert run["scheduler_script_sha256"] == file_sha256(run["scheduler_script"])
    assert canonical["scheduler_submit_token"] == run["scheduler_submit_token"]
    assert canonical["scheduler_script_sha256"] == run["scheduler_script_sha256"]
    assert canonical["scheduler_result_path"] == run["scheduler_result_path"]
    assert canonical["allocation_identity_path"] == run["allocation_identity_path"]
    assert canonical["log_path"] == run["log_path"]
    assert "#SBATCH --gres=gpu:1" in batch_script
    assert "--devices 0" in Path(run["script"]).read_text()
    assert "hparam-run-queue" not in batch_script

    Path(run["scheduler_script"]).write_text(batch_script + "# changed\n")
    with pytest.raises(ValueError, match="snapshot hash changed"):
        run_artifacts.read_hparam_plan(slurm_plan_dir)


def test_hparam_reader_uses_frozen_context_for_implicit_workdir(tmp_path: Path, monkeypatch):
    recipe = _hparam_recipe(tmp_path, execution={})
    payload = yaml.safe_load(recipe.read_text())
    payload["search"]["parameters"]["runtime.lr"] = [9.87654321e-7]
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
    plan_dir = tmp_path / "plan"
    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir))
    assert result.returncode == 0, result.stderr

    monkeypatch.setattr(plan_contract, "REPO_ROOT", Path("/controller/repo"))
    monkeypatch.setattr(plan_hparam, "REPO_ROOT", Path("/controller/repo"))

    plan = run_artifacts.read_hparam_plan(plan_dir)

    assert plan["recipe"]["execution"].get("workdir") in (None, "")


def test_slurm_launch_submits_each_logical_gpu_zero_run_independently(tmp_path: Path, monkeypatch):
    plan_dir, plan = _write_slurm_plan(tmp_path, run_count=2)
    calls = []

    def submit(execution, script, token, *, execution_snapshot_sha256, timeout):
        calls.append((execution, script, token, execution_snapshot_sha256, timeout))
        return slurm.JobIdentity(str(3880 + len(calls)), "wuji-h20")

    monkeypatch.setattr(managed_scheduler.slurm, "submit", submit)

    hparam_runtime.launch_hparam_runs(plan_dir, dry_run=False)

    rows = [
        row
        for row in _read_table(tmp_path / "run_manifest.tsv")
        if row["run_id"] in {run["run_id"] for run in plan["runs"]}
    ]
    assert len(calls) == 2
    expected_snapshot_sha256 = file_sha256(plan_dir / hparam_runtime.EXECUTION_SNAPSHOT_NAME)
    assert all(call[3] == expected_snapshot_sha256 for call in calls)
    assert [row["status"] for row in rows] == ["queued", "queued"]
    assert [row["scheduler_job_id"] for row in rows] == ["3881", "3882"]
    assert all(row["execution_snapshot_sha256"] == expected_snapshot_sha256 for row in rows)
    assert all(row["gpus"] == "" for row in rows)
    assert all(row["pid"] == "" and row["process_group_id"] == "" for row in rows)
    assert all("--devices 0" in Path(run["script"]).read_text() for run in plan["runs"])


@pytest.mark.parametrize("field", ["runtime_dir", "checkpoint_dir"])
def test_slurm_launch_rejects_existing_local_runtime_output_before_submission(tmp_path: Path, monkeypatch, field: str):
    plan_dir, plan = _write_slurm_plan(tmp_path)
    snapshot = (plan_dir / hparam_runtime.EXECUTION_SNAPSHOT_NAME).read_bytes()
    run = plan["runs"][0]
    Path(run[field]).mkdir(parents=True)
    submitted = []
    monkeypatch.setattr(managed_scheduler.slurm, "submit", lambda *_args, **_kwargs: submitted.append(True))

    with pytest.raises(ValueError, match="Managed output paths must be independent regular files"):
        hparam_runtime.launch_hparam_runs(plan_dir, dry_run=False)

    canonical = next(row for row in _read_table(tmp_path / "run_manifest.tsv") if row["run_id"] == run["run_id"])
    assert submitted == []
    assert canonical["status"] in {"planned", "pending"}
    assert (plan_dir / hparam_runtime.EXECUTION_SNAPSHOT_NAME).read_bytes() == snapshot


@pytest.mark.parametrize("scheduler_kind", ["direct", "slurm"])
@pytest.mark.parametrize("topology_drift", ["results_symlink", "execution_ancestor_symlink"])
def test_hparam_launch_rejects_topology_drift_before_launch(
    tmp_path: Path, monkeypatch, scheduler_kind: str, topology_drift: str
):
    if topology_drift == "execution_ancestor_symlink":
        execution_parent = tmp_path / "execution-parent"
        workdir = execution_parent / "repo"
        workdir.mkdir(parents=True)
        execution = {"workdir": str(workdir)}
        if scheduler_kind == "slurm":
            execution.update(
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
        recipe = _hparam_recipe(tmp_path, execution=execution)
        plan_dir = tmp_path / "plan"
        result = _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir))
        assert result.returncode == 0, result.stderr
        plan = json.loads((plan_dir / "plan.json").read_text())
        real_execution_parent = tmp_path / "execution-real"
        execution_parent.rename(real_execution_parent)
        execution_parent.symlink_to(real_execution_parent, target_is_directory=True)
    elif scheduler_kind == "slurm":
        plan_dir, plan = _write_slurm_plan(tmp_path)
    else:
        recipe = _hparam_recipe(tmp_path)
        plan_dir = tmp_path / "plan"
        result = _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir))
        assert result.returncode == 0, result.stderr
        plan = json.loads((plan_dir / "plan.json").read_text())

    if topology_drift == "results_symlink":
        artifacts = plan["recipe"].get("artifacts") or {}
        results_path = plan_hparam.plan_output_path(
            plan_dir,
            artifacts.get("results_csv_path"),
            "results/agent_hparam_results.csv",
        )
        results_path.parent.mkdir(parents=True, exist_ok=True)
        results_path.symlink_to(tmp_path / "run_manifest.tsv")

    manifest_before = (tmp_path / "run_manifest.tsv").read_bytes()
    events_before = (tmp_path / "events.jsonl").read_bytes()
    launch_attempts = []
    monkeypatch.setattr(
        hparam_runtime,
        "_start_process",
        lambda *_args, **_kwargs: launch_attempts.append("start"),
    )
    monkeypatch.setattr(
        managed_scheduler.slurm,
        "submit",
        lambda *_args, **_kwargs: launch_attempts.append("sbatch"),
    )

    with pytest.raises(ValueError, match="Managed output paths must be independent regular files"):
        hparam_runtime.launch_hparam_runs(plan_dir, dry_run=False)

    canonical = next(
        row for row in _read_table(tmp_path / "run_manifest.tsv") if row["run_id"] == plan["runs"][0]["run_id"]
    )
    assert launch_attempts == []
    assert canonical["status"] == "planned"
    assert (tmp_path / "run_manifest.tsv").read_bytes() == manifest_before
    assert (tmp_path / "events.jsonl").read_bytes() == events_before
    assert not (plan_dir / "launch_manifest.tsv").exists()
    assert not (plan_dir / "run_status.tsv").exists()


def test_hparam_launch_ignores_existing_runtime_dir_for_nonlaunchable_run(tmp_path: Path, monkeypatch):
    rows = _write_runtime_rows(
        tmp_path,
        [
            {"run_id": "run-000", "status": "launched"},
            {"run_id": "run-001", "status": "planned"},
        ],
    )
    Path(rows[0]["runtime_dir"]).mkdir(parents=True)
    calls = []
    monkeypatch.setattr(
        managed_scheduler,
        "launch_managed_runs",
        lambda *_args, **_kwargs: calls.append(True),
    )

    hparam_runtime.launch_hparam_runs(tmp_path, dry_run=False)

    assert calls == [True]


def test_slurm_ssh_launch_rejects_unsafe_remote_checkpoint_dir_before_submission(tmp_path: Path, monkeypatch):
    real_validate = hparam_runtime.exp_io.validate_managed_output_paths

    def validate_without_remote(root, paths, remote=None):
        if remote is None:
            return real_validate(root, paths)

    monkeypatch.setattr(hparam_runtime.exp_io, "validate_managed_output_paths", validate_without_remote)
    plan_dir, plan = _write_slurm_plan(
        tmp_path,
        execution={"target": "ssh", "host": "offline-host", "workdir": str(tmp_path)},
    )
    run = plan["runs"][0]
    snapshot = (plan_dir / hparam_runtime.EXECUTION_SNAPSHOT_NAME).read_bytes()
    checkpoint_dir = Path(run["checkpoint_dir"])
    remote_probes = []

    def reject_remote_checkpoint(root, paths, remote=None):
        if remote:
            remote_probes.append((Path(root), list(paths), remote))
            if checkpoint_dir in paths:
                raise ValueError(f"Managed output paths must be independent regular files: {checkpoint_dir}")
            return None
        return real_validate(root, paths)

    monkeypatch.setattr(hparam_runtime.exp_io, "validate_managed_output_paths", reject_remote_checkpoint)
    submitted = []
    monkeypatch.setattr(managed_scheduler.slurm, "submit", lambda *_args, **_kwargs: submitted.append(True))

    with pytest.raises(ValueError, match="Managed output paths must be independent regular files"):
        hparam_runtime.launch_hparam_runs(plan_dir, dry_run=False)

    canonical = next(row for row in _read_table(tmp_path / "run_manifest.tsv") if row["run_id"] == run["run_id"])
    assert remote_probes == [
        (
            Path("/"),
            [
                plan_dir / "plan.json",
                Path(run["runtime_dir"]),
                checkpoint_dir,
                plan_hparam.plan_output_path(
                    plan_dir,
                    (plan["recipe"].get("artifacts") or {}).get("results_csv_path"),
                    "results/agent_hparam_results.csv",
                ),
            ],
            "offline-host",
        )
    ]
    assert submitted == []
    assert canonical["status"] in {"planned", "pending"}
    assert (plan_dir / hparam_runtime.EXECUTION_SNAPSHOT_NAME).read_bytes() == snapshot


@pytest.mark.parametrize(
    ("execution", "runtime_devices", "expected_devices"),
    [
        ({"gpu_pool": [6, 7], "gpus_per_run": 2}, [0], "0 1"),
        ({"gpu_pool": [6, 7], "gpus_per_run": 1}, [0, 1], "0"),
        ({"gpus_per_run": 1}, [6, 7], "0"),
    ],
)
def test_hparam_plan_uses_logical_devices_for_scheduled_gpu_groups(
    tmp_path: Path,
    execution: dict,
    runtime_devices,
    expected_devices: str,
):
    recipe = _hparam_recipe(tmp_path, execution={"workdir": str(tmp_path), **execution})
    payload = yaml.safe_load(recipe.read_text())
    base_recipe = Path(payload["base_recipe"])
    base_payload = yaml.safe_load(base_recipe.read_text())
    base_payload["runtime"]["devices"] = runtime_devices
    write_yaml(base_recipe, base_payload)
    plan_dir = tmp_path / "plan"

    result = _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir))

    assert result.returncode == 0, result.stderr or result.stdout
    command = json.loads((plan_dir / "plan.json").read_text())["runs"][0]["command"]
    assert f"--devices {expected_devices} --precision" in command


def test_slurm_sex_age_baseline_rejects_multi_gpu_without_changing_direct_execution():
    scheduler = {
        "type": "slurm",
        "partition": "gpu",
        "cpus_per_task": 8,
        "memory": "64G",
        "walltime": "01:00:00",
    }

    slurm_issues = decision_hparam._hparam_execution_issues(
        {"gpus_per_run": 2, "scheduler": scheduler},
        {},
        variant="sex_age_baseline",
    )
    direct_issues = decision_hparam._hparam_execution_issues(
        {"gpu_pool": [0, 1], "gpus_per_run": 2},
        {},
        variant="sex_age_baseline",
    )

    failures = [issue for issue in slurm_issues if issue.status.value == "FAIL"]
    assert len(failures) == 1
    assert failures[0].field == "execution.gpus_per_run"
    assert "does not support multi-GPU Slurm execution" in failures[0].message
    assert not [issue for issue in direct_issues if issue.status.value == "FAIL"]


@pytest.mark.parametrize("gpus_per_run", [1.0, "1"])
def test_slurm_hparam_rejects_coerced_gpu_count_before_plan_write(tmp_path: Path, gpus_per_run):
    recipe = _hparam_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    payload["execution"].update(
        {
            "gpus_per_run": gpus_per_run,
            "scheduler": {
                "type": "slurm",
                "partition": "gpu",
                "cpus_per_task": 8,
                "memory": "64G",
                "walltime": "01:00:00",
            },
        }
    )
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
    plan_dir = tmp_path / "plan"

    doctor = _run("doctor", "--recipe", str(recipe))
    planned = _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir))

    for result in (doctor, planned):
        assert result.returncode == 1
        assert "execution.gpus_per_run must be a positive integer" in result.stdout
    assert not (plan_dir / "plan.json").exists()
    assert not (plan_dir / "recipe.resolved.yaml").exists()
    assert not (plan_dir / "run_all.sh").exists()


@pytest.mark.parametrize(("slurm_procid", "marker_count"), [("0", 1), ("1", 0)])
def test_hparam_exit_marker_is_written_only_by_slurm_rank_zero(tmp_path: Path, slurm_procid: str, marker_count: int):
    script = tmp_path / "launch.sh"
    script.write_text(
        "\n".join(
            plan_rendering.hparam_script_lines(
                ["exit 7"],
                record_exit_code=True,
                run_cwd=tmp_path,
            )
        )
        + "\n"
    )

    result = subprocess.run(
        ["bash", str(script)],
        text=True,
        capture_output=True,
        env={**os.environ, "SLURM_PROCID": slurm_procid},
    )

    assert result.returncode == 7
    assert result.stdout.count(MONITOR_EXIT_CODE_PREFIX) == marker_count


@pytest.mark.parametrize(
    "env_name",
    [
        "SLURM_JOB_ID",
        "SLURM_CLUSTER_NAME",
        "SLURM_ARRAY_TASK_ID",
        "CUDA_VISIBLE_DEVICES",
        "RANK",
        "LOCAL_RANK",
        "WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
    ],
)
def test_slurm_hparam_rejects_scheduler_owned_environment_before_plan_write(tmp_path: Path, env_name: str):
    recipe = _hparam_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    payload["execution"].update(
        {
            "gpus_per_run": 1,
            "env": {env_name: "unit"},
            "scheduler": {
                "type": "slurm",
                "partition": "gpu",
                "cpus_per_task": 8,
                "memory": "64G",
                "walltime": "01:00:00",
            },
        }
    )
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))
    plan_dir = tmp_path / "plan"

    doctor = _run("doctor", "--recipe", str(recipe))
    planned = _run("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir))

    for result in (doctor, planned):
        assert result.returncode == 1
        assert f"execution.env.{env_name}" in result.stdout
        assert "owned by Slurm" in result.stdout
    assert not (plan_dir / "plan.json").exists()
    assert not (plan_dir / "recipe.resolved.yaml").exists()
    assert not (plan_dir / "run_all.sh").exists()
