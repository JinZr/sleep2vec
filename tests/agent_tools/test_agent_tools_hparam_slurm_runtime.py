from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess

from agent_tool_test_helpers import (
    read_table,
    run_cli,
    tree_files,
    write_local_hparam_recipe,
    write_slurm_plan,
    write_yaml,
)
import pytest
import yaml

from agent_tools import (
    execution_snapshot,
    hparam_runtime,
    managed_scheduler,
    plan_hparam,
    plan_rendering,
    plans,
    run_artifacts,
)
from agent_tools.experiment_workspace import (
    MONITOR_EXIT_CODE_PREFIX,
    file_sha256,
    merge_run_manifest,
    next_run_index,
    run_identity,
)

pytestmark = pytest.mark.usefixtures("stub_execution_target")


@pytest.mark.parametrize(
    ("variant", "gpus_per_run"),
    [("sleep2vec", 1), ("sleep2vec", 2), ("sleep2vec2", 2), ("sleep2expert", 2)],
)
def test_public_hparam_recipe_plans_slurm_leaf_jobs(tmp_path: Path, variant: str, gpus_per_run: int):
    recipe = write_local_hparam_recipe(tmp_path, variant=variant)
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

    result = run_cli("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir))
    preview = run_cli("hparam-run-queue", "--plan-dir", str(plan_dir))

    assert result.returncode == 0, result.stderr
    assert "execution.scheduler.priority" not in result.stdout
    assert preview.returncode == 0, preview.stderr
    plan = json.loads((plan_dir / "plan.json").read_text())
    run = plan["runs"][0]
    canonical = read_table(tmp_path / "run_manifest.tsv")[0]
    assert plan["recipe"]["execution"]["scheduler"] == configured_scheduler
    assert run["scheduler_type"] == "slurm"
    assert canonical["scheduler_type"] == "slurm"
    assert Path(run["scheduler_script"]).name == "job.sbatch"
    scheduler_script = Path(run["scheduler_script"]).read_text()
    assert "#SBATCH --nice=0" in scheduler_script
    assert f"#SBATCH --ntasks={gpus_per_run}" in scheduler_script
    assert f"#SBATCH --ntasks-per-node={gpus_per_run}" in scheduler_script
    assert f"#SBATCH --gres=gpu:{gpus_per_run}" in scheduler_script
    assert f"srun --nodes=1 --ntasks={gpus_per_run} --ntasks-per-node={gpus_per_run}" in scheduler_script
    assert f"#SBATCH --job-name={run['experiment_id']}/{run['step_id']}/{run['run_id']}" in scheduler_script
    command = shlex.split(run["command"])
    devices_index = command.index("--devices")
    assert command[devices_index + 1 : devices_index + 1 + gpus_per_run] == [
        str(device) for device in range(gpus_per_run)
    ]
    assert command[devices_index + 1 + gpus_per_run].startswith("--")
    # The one-line sbatch submission is printed in full by the dry-run preview.
    assert f"- {run['step_id']} / {run['run_id']}: status=planned; command=" in preview.stdout
    assert "sbatch --parsable" in preview.stdout


@pytest.mark.parametrize("artifact_name", ["job.sbatch", "slurm.log"])
def test_slurm_plan_rejects_existing_single_use_artifacts_before_writing(tmp_path: Path, artifact_name: str):
    recipe = write_local_hparam_recipe(tmp_path)
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
    before = tree_files(tmp_path)

    result = run_cli("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir))

    assert result.returncode == 1
    assert "Output artifacts already exist" in result.stdout
    assert tree_files(tmp_path) == before


def test_slurm_plan_rejects_existing_scheduler_artifact_before_writing(tmp_path: Path):
    recipe = write_local_hparam_recipe(tmp_path)
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

    result = run_cli("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir))

    assert result.returncode == 1
    # A pre-existing artifact, even a symlink out of the plan, fails before anything is written.
    assert "Output artifacts already exist; retry with a fresh --output-dir." in result.stdout
    assert "job.sbatch" in result.stdout
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
    recipe = write_local_hparam_recipe(tmp_path)
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

    result = run_cli("plan", "--recipe", str(recipe), "--output-dir", str(tmp_path / "plan"))

    assert result.returncode == 1
    assert re.search(message, result.stdout)
    assert not (tmp_path / "plan").exists()


def test_public_slurm_hparam_recipe_rejects_unrecognized_scheduler_fields(tmp_path: Path):
    recipe = write_local_hparam_recipe(tmp_path)
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

    result = run_cli("plan", "--recipe", str(recipe), "--output-dir", str(tmp_path / "plan"))

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
    recipe = write_local_hparam_recipe(tmp_path)
    payload = yaml.safe_load(recipe.read_text())
    payload["execution"]["scheduler"] = scheduler
    recipe.write_text(yaml.safe_dump(payload, sort_keys=False))

    result = run_cli("plan", "--recipe", str(recipe), "--output-dir", str(tmp_path / "plan"))

    assert result.returncode == 1
    assert message in result.stdout
    assert not (tmp_path / "plan").exists()


def test_hparam_plan_freezes_one_slurm_job_per_run_before_registration(tmp_path: Path, monkeypatch):
    recipe_path = write_local_hparam_recipe(tmp_path)
    direct_plan_dir = tmp_path / "direct-plan"
    assert run_cli("plan", "--recipe", str(recipe_path), "--output-dir", str(direct_plan_dir)).returncode == 0
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
        final_eval_config=None,
    )
    run_artifacts.preflight_hparam_plan(slurm_plan_dir, semantic_out=slurm_plan_dir)
    run_artifacts.commit_hparam_plan(slurm_plan_dir)

    run = json.loads((slurm_plan_dir / "plan.json").read_text())["runs"][0]
    canonical = next(row for row in read_table(tmp_path / "run_manifest.tsv") if row["run_id"] == run["run_id"])
    batch_script = Path(run["scheduler_script"]).read_text()
    assert run["scheduler_type"] == "slurm"
    assert run["terminal_status_owner"] == "monitor"
    assert run["scheduler_script_sha256"] == file_sha256(run["scheduler_script"])
    slurm_columns = {
        "scheduler_type",
        "scheduler_script",
        "scheduler_script_sha256",
        "scheduler_job_id",
        "scheduler_raw_state",
        "scheduler_reason",
        "scheduler_node",
        "scheduler_exit_code",
    }
    assert {field for field in [*run, *canonical] if field.startswith("scheduler_")} <= slurm_columns
    assert canonical["scheduler_script_sha256"] == run["scheduler_script_sha256"]
    assert canonical["log_path"] == run["log_path"]
    assert "#SBATCH --gres=gpu:1" in batch_script
    assert f"{run['config_sha256']}  {run['config']}" in batch_script
    assert f"{run['script_sha256']}  {run['script']}" in batch_script
    assert "--devices 0" in Path(run["script"]).read_text()
    assert "hparam-run-queue" not in batch_script

    Path(run["scheduler_script"]).write_text(batch_script + "# changed\n")
    monkeypatch.setattr(managed_scheduler.slurm, "user_jobs", lambda **_kwargs: pytest.fail("no squeue"))
    monkeypatch.setattr(managed_scheduler.slurm, "submit", lambda *_args, **_kwargs: pytest.fail("no sbatch"))
    # Submission re-hashes job.sbatch; inside the job, sha256sum --check then covers config.yaml and launch.sh.
    with pytest.raises(ValueError, match="Frozen Slurm script changed before submission"):
        hparam_runtime.launch_hparam_runs(slurm_plan_dir, dry_run=False)


def test_slurm_launch_submits_each_logical_gpu_zero_run_independently(tmp_path: Path, monkeypatch):
    plan_dir, plan = write_slurm_plan(tmp_path, run_count=2)
    calls = []

    def submit(script, *, timeout):
        calls.append((script, timeout))
        return str(3880 + len(calls))

    monkeypatch.setattr(managed_scheduler.slurm, "user_jobs", lambda **_kwargs: [])
    monkeypatch.setattr(managed_scheduler.slurm, "submit", submit)

    hparam_runtime.launch_hparam_runs(plan_dir, dry_run=False)

    rows = [
        row
        for row in read_table(tmp_path / "run_manifest.tsv")
        if row["run_id"] in {run["run_id"] for run in plan["runs"]}
    ]
    assert calls == [(run["scheduler_script"], managed_scheduler.LAUNCH_TIMEOUT_SECONDS) for run in plan["runs"]]
    assert (plan_dir / execution_snapshot.EXECUTION_SNAPSHOT_NAME).is_file()
    assert [row["status"] for row in rows] == ["queued", "queued"]
    assert [row["scheduler_job_id"] for row in rows] == ["3881", "3882"]
    assert all("execution_snapshot_sha256" not in row for row in rows)
    assert all(row["gpus"] == "" for row in rows)
    assert all(row["pid"] == "" and row["process_group_id"] == "" for row in rows)
    assert all("--devices 0" in Path(run["script"]).read_text() for run in plan["runs"])


@pytest.mark.parametrize("field", ["runtime_dir", "checkpoint_dir"])
def test_slurm_launch_rejects_existing_local_runtime_output_before_submission(tmp_path: Path, monkeypatch, field: str):
    plan_dir, plan = write_slurm_plan(tmp_path)
    snapshot = (plan_dir / execution_snapshot.EXECUTION_SNAPSHOT_NAME).read_bytes()
    run = plan["runs"][0]
    Path(run[field]).mkdir(parents=True)
    submitted = []
    monkeypatch.setattr(managed_scheduler.slurm, "submit", lambda *_args, **_kwargs: submitted.append(True))

    with pytest.raises(ValueError, match="Managed output paths must be independent regular files"):
        hparam_runtime.launch_hparam_runs(plan_dir, dry_run=False)

    canonical = next(row for row in read_table(tmp_path / "run_manifest.tsv") if row["run_id"] == run["run_id"])
    assert submitted == []
    assert canonical["status"] in {"planned", "pending"}
    assert (plan_dir / execution_snapshot.EXECUTION_SNAPSHOT_NAME).read_bytes() == snapshot


def test_hparam_launch_ignores_existing_runtime_dir_for_nonlaunchable_run(tmp_path: Path, monkeypatch):
    recipe_path = write_local_hparam_recipe(tmp_path)
    recipe = yaml.safe_load(recipe_path.read_text())
    recipe["search"] = {"method": "grid", "max_runs": 2, "parameters": {"runtime.lr": [1e-6, 2e-6]}}
    recipe_path.write_text(yaml.safe_dump(recipe))
    plan_dir = tmp_path / "plan"
    report = plans.build_plan(recipe_path=recipe_path, output_dir=plan_dir)
    assert report.exit_code == 0
    rows = run_artifacts.read_hparam_plan(plan_dir)["runs"]
    merge_run_manifest(tmp_path, [{"step_id": rows[0]["step_id"], "run_id": rows[0]["run_id"], "status": "launched"}])
    Path(rows[0]["runtime_dir"]).mkdir(parents=True)
    calls = []
    monkeypatch.setattr(
        managed_scheduler,
        "launch_managed_runs",
        lambda *_args, **_kwargs: calls.append(True),
    )

    hparam_runtime.launch_hparam_runs(plan_dir, dry_run=False)

    assert calls == [True]


@pytest.mark.parametrize(
    ("execution", "runtime_devices", "expected_devices"),
    [
        ({"gpu_pool": [6, 7], "gpus_per_run": 2}, [0], "0 1"),
        ({"gpu_pool": [6, 7], "gpus_per_run": 1}, [0, 1], "0"),
        ({"gpus_per_run": 1}, [6, 7], "0"),
        ({}, [2], "0"),
        ({}, [2, 3], "0 1"),
    ],
)
def test_hparam_plan_uses_logical_devices_for_scheduled_gpu_groups(
    tmp_path: Path,
    execution: dict,
    runtime_devices,
    expected_devices: str,
):
    recipe = write_local_hparam_recipe(tmp_path, execution={"workdir": str(tmp_path), **execution})
    payload = yaml.safe_load(recipe.read_text())
    base_recipe = Path(payload["base_recipe"])
    base_payload = yaml.safe_load(base_recipe.read_text())
    base_payload["runtime"]["devices"] = runtime_devices
    write_yaml(base_recipe, base_payload)
    plan_dir = tmp_path / "plan"

    result = run_cli("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir))

    assert result.returncode == 0, result.stderr or result.stdout
    command = json.loads((plan_dir / "plan.json").read_text())["runs"][0]["command"]
    assert f"--devices {expected_devices} --precision" in command


@pytest.mark.parametrize("gpus_per_run", [1.0, "1"])
def test_slurm_hparam_rejects_coerced_gpu_count_before_plan_write(tmp_path: Path, gpus_per_run):
    recipe = write_local_hparam_recipe(tmp_path)
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

    doctor = run_cli("doctor", "--recipe", str(recipe))
    planned = run_cli("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir))

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
        env={
            **os.environ,
            "SLURM_JOB_ID": "3880",
            "SLURM_STEP_ID": "2",
            "SLURM_PROCID": slurm_procid,
            "SLURM_LOCALID": slurm_procid,
            "SLURM_NODEID": "0",
            "SLURM_NTASKS": "2",
            "SLURMD_NODENAME": "gpu-01",
            "CUDA_VISIBLE_DEVICES": "0,1",
            "MASTER_ADDR": "secret-master",
            "WANDB_API_KEY": "secret-wandb",
        },
    )

    assert result.returncode == 7
    assert result.stdout.count(MONITOR_EXIT_CODE_PREFIX) == marker_count
    start_lines = [line for line in result.stdout.splitlines() if line.startswith("AGENT_TOOLS_SLURM_TASK_START ")]
    assert len(start_lines) == 1
    assert re.fullmatch(
        rf"AGENT_TOOLS_SLURM_TASK_START job_id=3880 step_id=2 procid={slurm_procid} "
        rf"localid={slurm_procid} nodeid=0 ntasks=2 node=gpu-01 pid=[1-9][0-9]* "
        rf"cuda_visible_devices=0,1",
        start_lines[0],
    )
    assert "secret-master" not in result.stdout
    assert "secret-wandb" not in result.stdout


def test_hparam_script_omits_slurm_task_start_record_for_direct_run(tmp_path: Path):
    script = tmp_path / "launch.sh"
    script.write_text("\n".join(plan_rendering.hparam_script_lines(["true"], run_cwd=tmp_path)) + "\n")

    result = subprocess.run(["bash", str(script)], text=True, capture_output=True)

    assert result.returncode == 0
    assert "AGENT_TOOLS_SLURM_TASK_START" not in result.stdout


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
    recipe = write_local_hparam_recipe(tmp_path)
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

    doctor = run_cli("doctor", "--recipe", str(recipe))
    planned = run_cli("plan", "--recipe", str(recipe), "--output-dir", str(plan_dir))

    for result in (doctor, planned):
        assert result.returncode == 1
        assert f"execution.env.{env_name}" in result.stdout
        assert "owned by Slurm" in result.stdout
    assert not (plan_dir / "plan.json").exists()
    assert not (plan_dir / "recipe.resolved.yaml").exists()
    assert not (plan_dir / "run_all.sh").exists()
