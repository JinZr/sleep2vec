from __future__ import annotations

import functools
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

from agent_tool_test_helpers import (
    call_while_run_lock_holder_commits,
    config_payload,
    run_execution_preflight_fixture,
    write_survival_sidecars,
    write_yaml,
)
import pytest
import yaml

from agent_tools import (
    cli,
    execution_snapshot,
    experiment_io,
    experiment_workspace,
    experiments,
    managed_scheduler,
    plans,
    run_artifacts,
    slurm,
)
from agent_tools.adapters.registry import get_adapter
from agent_tools.models import REPO_ROOT


@pytest.fixture(autouse=True)
def _no_external_scheduler(tmp_path: Path, monkeypatch):
    commands = {"ssh", "sbatch", "squeue", "sacct", "scontrol", "scancel", "srun"}
    guarded_bin = tmp_path / "guarded-bin"
    guarded_bin.mkdir()
    attempted = tmp_path / "unexpected-scheduler-command"
    for name in commands:
        executable = guarded_bin / name
        executable.write_text(f"#!/bin/sh\nprintf '%s\\n' \"$0\" >> {shlex.quote(str(attempted))}\nexit 97\n")
        executable.chmod(0o755)
    monkeypatch.setenv("PATH", str(guarded_bin) + os.pathsep + os.environ.get("PATH", ""))
    original_popen = subprocess.Popen

    def guarded_popen(args, *positional, **kwargs):
        executable = kwargs.get("executable")
        if executable is None and isinstance(args, (list, tuple)) and args:
            executable = args[0]
        if executable is not None and Path(str(executable)).name in commands:
            resolved = shutil.which(str(executable), path=(kwargs.get("env") or os.environ).get("PATH"))
            assert resolved == str(guarded_bin / Path(str(executable)).name), "Genuine SSH/Slurm command is forbidden"
        return original_popen(args, *positional, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", guarded_popen)
    yield
    assert not attempted.exists(), "Planning must not invoke SSH or Slurm, including from a child shell"


@pytest.fixture
def _runtime_probe(monkeypatch):
    monkeypatch.setattr(execution_snapshot, "run_execution_command", run_execution_preflight_fixture)


@pytest.fixture(scope="module")
def _runtime_commit():
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True, text=True, capture_output=True
    ).stdout.strip()


def _infer_slurm_recipe(tmp_path: Path, *, variant: str, task: str, runtime_commit: str) -> Path:
    workspace = tmp_path / "workspace"
    inputs_dir = workspace / "inputs"
    inputs_dir.mkdir(parents=True)
    workdir = tmp_path / "runtime"
    workdir.mkdir()
    if variant == "sex_age_baseline":
        index = inputs_dir / "index.csv"
        index.write_text("path,split,duration,eid,age,sex\nx.npz,train,60,001,50,0\ny.npz,val,60,002,60,1\n")
        config_values = yaml.safe_load((REPO_ROOT / "configs" / "sex_age_baseline" / "cox.yaml").read_text())
        config_values["data"]["finetune_data_index"] = str(index)
        config_values["finetune"]["task"]["output_dim"] = 2
        config_values["finetune"]["survival"].update(write_survival_sidecars(inputs_dir))
        config = write_yaml(inputs_dir / "config.yaml", config_values)
        label = "incident_cox"
    else:
        index = inputs_dir / "index.csv"
        index.write_text("path,split,duration,ppg_mask,ah_event_mask,stage_mask\nx.npz,val,60,1,1,1\n")
        config = write_yaml(inputs_dir / "config.yaml", config_payload(index))
        label = "ahi"
    checkpoint = inputs_dir / "model.ckpt"
    checkpoint.write_text("unit checkpoint input, never loaded by a model\n")
    return write_yaml(
        workspace / "recipe.yaml",
        {
            "name": "unit_managed_infer",
            "task": task,
            "variant": variant,
            "step": {"id": "unit-infer", "phase": "evaluate", "purpose": "Exercise ordinary managed inference."},
            "inputs": {
                "config": str(config),
                "ckpt_path": str(checkpoint),
                "label_name": label,
                "eval_split": "val",
            },
            "runtime": {"batch_size": 2, "num_workers": 0},
            "execution": {
                "workdir": str(workdir),
                "python": sys.executable,
                "runtime_commit": runtime_commit,
                "gpus_per_run": 1,
                "scheduler": {
                    "type": "slurm",
                    "partition": "unit-gpu",
                    "cpus_per_task": 1,
                    "memory": "1G",
                    "walltime": "00:01:00",
                },
            },
            "artifacts": {"overwrite": False},
            "evaluation_policy": {"external_test_locked": True, "final_test_unlocked": False},
            "decisions": {
                "task": {"value": task, "source": "explicit_recipe"},
                "label_name": {"value": label, "source": "explicit_recipe"},
                "ckpt_path": {"value": str(checkpoint), "source": "explicit_recipe"},
                "eval_split": {"value": "val", "source": "explicit_recipe"},
                "final_eval_unlock": {"value": False, "source": "explicit_recipe"},
                "overwrite_policy": {"value": False, "source": "explicit_recipe"},
            },
        },
    )


def _read_registered_infer_plan(plan_dir: Path):
    workspace = plan_dir.parent
    rows = experiment_workspace.read_run_manifest(workspace)
    (row,) = rows
    experiment_path = workspace / "experiment.yaml"
    experiment = experiment_workspace.read_managed_yaml_mapping(experiment_path.read_text(), source=experiment_path)[
        "experiment"
    ]
    step = experiment_workspace.read_step_manifest(workspace, row["step_id"])
    assert step is not None
    return run_artifacts.read_registered_plan(
        plan_dir,
        workspace=workspace,
        workspace_experiment=experiment,
        step_manifest=step,
        workspace_rows=rows,
        expected_recipe_path=str(workspace / "recipe.yaml"),
    )


def _build_infer_slurm_plan(tmp_path: Path, runtime_commit: str):
    recipe_path = _infer_slurm_recipe(tmp_path, variant="sleep2vec", task="infer", runtime_commit=runtime_commit)
    plan_dir = recipe_path.parent / "plan"
    report = plans.build_plan(recipe_path=recipe_path, output_dir=plan_dir)
    assert report.exit_code == 0, [(issue.field, issue.message) for issue in report.blocking_issues()]
    return plan_dir, _read_registered_infer_plan(plan_dir)


class _FakeSlurm:
    """Stand-in for the sbatch, squeue and scancel clients behind slurm.run_command."""

    def __init__(self, monkeypatch, plan_dir: Path, plan: dict):
        (self.run,) = plan["runs"]
        self.workspace = plan_dir.parent
        self.name = slurm.job_name(self.run)
        self.calls: list[list[str]] = []
        self.jobs: list[str] = []
        self.squeue_returncode = 0
        self.sbatch = lambda argv: subprocess.CompletedProcess(argv, 0, "3880\n", "")
        monkeypatch.setattr(slurm, "run_command", self)

    def __call__(self, argv, *, timeout):
        self.calls.append(argv)
        if argv[0] == "bash":
            (row,) = experiment_workspace.read_run_manifest(self.workspace)
            # Launch evidence is durable before sbatch runs, so a lost receipt is never a definite failure.
            assert row["status"] in {"planned", "pending"}
            assert not row.get("scheduler_job_id")
            assert row["command"] == slurm.submission_command(self.run["scheduler_script"])
            assert shlex.split(row["command"]) == ["env", "-u", "SLURM_CLUSTERS", *argv]
            assert argv[-3:] == ["sbatch", "--parsable", self.run["scheduler_script"]]
            return self.sbatch(argv)
        if argv[0] == "squeue":
            assert argv == ["squeue", "--me", "--states=all", "--noheader", "--format=%i|%j|%T|%r|%N"]
            stdout = "".join(f"{line}\n" for line in self.jobs)
            return subprocess.CompletedProcess(argv, self.squeue_returncode, stdout, "slurm_load_jobs error")
        assert argv[0] == "scancel", argv
        return subprocess.CompletedProcess(argv, 0, "", "")

    def count(self, client: str) -> int:
        return sum(argv[0] == client for argv in self.calls)


def _row(plan_dir: Path) -> dict[str, str]:
    (row,) = experiment_workspace.read_run_manifest(plan_dir.parent)
    return row


@pytest.mark.parametrize("summary", [None, "runtime.lr=0.001"])
def test_generic_registration_preserves_existing_frozen_parameter_summary(summary):
    run = {"run_id": "run-000", "command": "frozen model command"}
    if summary is not None:
        run["parameter_summary"] = summary
    before = dict(run)

    (row,) = get_adapter("finetune").registration_rows({"runs": [run]})

    assert row == {"run_id": "run-000", "parameter_summary": summary or "single resolved recipe"}
    assert run == before


@pytest.mark.parametrize("variant", ["sleep2vec", "sleep2vec2", "sleep2expert", "sex_age_baseline"])
def test_infer_slurm_round_trip_preserves_unsubmitted_identity(
    tmp_path: Path, variant: str, _runtime_commit, _runtime_probe
):
    recipe_path = _infer_slurm_recipe(tmp_path, variant=variant, task="infer", runtime_commit=_runtime_commit)
    workspace = recipe_path.parent
    plan_dir = workspace / "plan"

    result = plans.build_plan(recipe_path=recipe_path, output_dir=plan_dir)

    assert result.exit_code == 0, [(issue.field, issue.message) for issue in result.blocking_issues()]
    plan = json.loads((plan_dir / "plan.json").read_text())
    (run,) = plan["runs"]
    (row,) = experiment_workspace.read_run_manifest(workspace)
    assert row["status"] == "planned"
    assert row["run_id"] == "run-000"
    assert row["scheduler_type"] == "slurm"
    assert row["terminal_status_owner"] == "monitor"
    for field in ("target", "workdir", "command", "scheduler_job_id"):
        assert row.get(field, "") == ""
    assert not experiment_workspace.has_managed_launch_evidence(row)
    for field in (
        "config",
        "config_sha256",
        "script",
        "script_sha256",
        "scheduler_script",
        "scheduler_script_sha256",
        "log_path",
    ):
        assert row[field] == run[field]
        assert row[field]
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
    assert {field for field in [*run, *row] if field.startswith("scheduler_")} <= slurm_columns

    validated = _read_registered_infer_plan(plan_dir)

    assert validated["recipe"] == plan["recipe"]
    assert validated["runs"] == plan["runs"]
    frozen_recipe = validated["recipe"]
    assert frozen_recipe["runtime"]["devices"] == [0]
    adapter = get_adapter("infer")
    assert adapter is not None
    contract = adapter.compile_plan_contract(
        frozen_recipe, plan_dir, run_index_offset=0, config_bytes=Path(run["config"]).read_bytes()
    )
    assert contract == adapter.compile_plan_contract(
        frozen_recipe, plan_dir, run_index_offset=0, config_bytes=Path(run["config"]).read_bytes()
    )
    manager_text = (plan_dir / "run.sh").read_text()
    worker_text = Path(run["script"]).read_text()
    scheduler_text = Path(run["scheduler_script"]).read_text()
    assert manager_text == contract["launch_script_text"]
    assert worker_text == contract["script_text"]
    assert scheduler_text == contract["scheduler_script_text"]
    assert manager_text != worker_text
    assert "infer-launch" in manager_text
    # Every Slurm task runs the worker; the monitor owns status, and no allocation guard re-authenticates it.
    assert "_agent_commit_status" not in worker_text
    assert "allocation" not in worker_text
    assert f"#SBATCH --job-name={slurm.job_name(run)}" in scheduler_text
    assert f"--label bash {run['script']}" in scheduler_text
    assert f"{run['config_sha256']}  {run['config']}" in scheduler_text
    assert f"{run['script_sha256']}  {run['script']}" in scheduler_text

    (command,) = plan["commands"]
    assert command == run["command"]
    assert command == json.loads((Path(run["run_dir"]) / "run.json").read_text())["command"]
    assert worker_text.splitlines().count(command) == 1
    assert shlex.split(command)[:3] == [sys.executable, "-m", f"{variant}.infer"]
    for path_field, digest_field in (
        ("config", "config_sha256"),
        ("script", "script_sha256"),
        ("scheduler_script", "scheduler_script_sha256"),
    ):
        assert hashlib.sha256(Path(run[path_field]).read_bytes()).hexdigest() == run[digest_field]


@pytest.mark.parametrize("variant", ["sleep2vec", "sex_age_baseline"])
def test_infer_slurm_derives_multiple_logical_devices(tmp_path: Path, variant, _runtime_commit, _runtime_probe):
    recipe_path = _infer_slurm_recipe(tmp_path, variant=variant, task="infer", runtime_commit=_runtime_commit)
    recipe = yaml.safe_load(recipe_path.read_text())
    recipe["execution"]["gpus_per_run"] = 2
    recipe_path.write_text(yaml.safe_dump(recipe))
    plan_dir = recipe_path.parent / "plan"

    assert plans.build_plan(recipe_path=recipe_path, output_dir=plan_dir).exit_code == 0

    validated = _read_registered_infer_plan(plan_dir)
    assert validated["recipe"]["runtime"]["devices"] == [0, 1]
    (run,) = validated["runs"]
    scheduler_text = Path(run["scheduler_script"]).read_text()
    assert "#SBATCH --nodes=1\n" in scheduler_text
    assert "#SBATCH --ntasks=2\n" in scheduler_text
    assert "#SBATCH --gres=gpu:2\n" in scheduler_text


@pytest.mark.parametrize(
    ("variant", "section", "field", "value", "expected_field"),
    [
        ("sleep2vec", "runtime", "devices", [2], "runtime.devices"),
        ("sleep2vec", "runtime", "accelerator", "cpu", "runtime.accelerator"),
        ("sleep2vec", "execution", "workdir", None, "execution.workdir"),
        ("sleep2vec", "execution", "python", None, "execution.python"),
        ("sleep2vec", "execution", "runtime_commit", None, "execution.runtime_commit"),
    ],
)
def test_infer_slurm_rejects_invalid_execution_contract(
    tmp_path: Path, variant, section, field, value, expected_field, _runtime_commit, _runtime_probe
):
    recipe_path = _infer_slurm_recipe(tmp_path, variant=variant, task="infer", runtime_commit=_runtime_commit)
    recipe = yaml.safe_load(recipe_path.read_text())
    if value is None:
        recipe[section].pop(field)
    else:
        recipe[section][field] = value
    recipe_path.write_text(yaml.safe_dump(recipe))
    plan_dir = recipe_path.parent / "plan"

    result = plans.build_plan(recipe_path=recipe_path, output_dir=plan_dir)

    assert result.exit_code == 1
    assert expected_field in {issue.field for issue in result.issues}
    assert not (plan_dir / "plan.json").exists()
    assert experiment_workspace.read_run_manifest(recipe_path.parent) == []


@pytest.mark.parametrize("artifact", ["manager", "worker", "scheduler"])
def test_infer_slurm_reader_rejects_changed_script(tmp_path: Path, artifact: str, _runtime_commit, _runtime_probe):
    recipe_path = _infer_slurm_recipe(tmp_path, variant="sleep2vec", task="infer", runtime_commit=_runtime_commit)
    plan_dir = recipe_path.parent / "plan"
    assert plans.build_plan(recipe_path=recipe_path, output_dir=plan_dir).exit_code == 0
    (run,) = _read_registered_infer_plan(plan_dir)["runs"]
    path = {
        "manager": plan_dir / "run.sh",
        "worker": Path(run["script"]),
        "scheduler": Path(run["scheduler_script"]),
    }[artifact]
    path.write_text(path.read_text() + "# unregistered edit\n")
    manifest_path = recipe_path.parent / "run_manifest.tsv"
    before = manifest_path.read_bytes()

    with pytest.raises(ValueError, match="differs|changed"):
        _read_registered_infer_plan(plan_dir)

    assert manifest_path.read_bytes() == before


def test_infer_slurm_reader_recompiles_coherently_edited_scheduler(tmp_path: Path, _runtime_commit, _runtime_probe):
    recipe_path = _infer_slurm_recipe(tmp_path, variant="sleep2vec", task="infer", runtime_commit=_runtime_commit)
    plan_dir = recipe_path.parent / "plan"
    assert plans.build_plan(recipe_path=recipe_path, output_dir=plan_dir).exit_code == 0
    plan_path = plan_dir / "plan.json"
    plan = json.loads(plan_path.read_text())
    (run,) = plan["runs"]
    scheduler_path = Path(run["scheduler_script"])
    scheduler_path.write_text(scheduler_path.read_text() + "# coherent but unauthorized edit\n")
    edited_digest = hashlib.sha256(scheduler_path.read_bytes()).hexdigest()
    run["scheduler_script_sha256"] = edited_digest
    plan_path.write_text(json.dumps(plan))
    run_path = Path(run["run_dir"]) / "run.json"
    run_payload = json.loads(run_path.read_text())
    run_payload["scheduler_script_sha256"] = edited_digest
    run_path.write_text(json.dumps(run_payload))
    rows = experiment_workspace.read_run_manifest(recipe_path.parent)
    rows[0]["scheduler_script_sha256"] = edited_digest
    experiment_io.write_rows_at(recipe_path.parent / "run_manifest.tsv", rows)

    with pytest.raises(ValueError, match="scheduler_script_sha256|Slurm script"):
        _read_registered_infer_plan(plan_dir)


def test_infer_slurm_reader_rejects_run_payload_drift(tmp_path: Path, _runtime_commit, _runtime_probe):
    plan_dir, plan = _build_infer_slurm_plan(tmp_path, _runtime_commit)
    (run,) = plan["runs"]
    run_path = Path(run["run_dir"]) / "run.json"
    run_payload = json.loads(run_path.read_text())
    run_payload["unexpected"] = True
    run_path.write_text(json.dumps(run_payload, indent=2, sort_keys=True) + "\n")
    manifest_path = plan_dir.parent / "run_manifest.tsv"
    before = manifest_path.read_bytes()

    with pytest.raises(ValueError, match="run payload differs from its frozen plan"):
        _read_registered_infer_plan(plan_dir)

    assert manifest_path.read_bytes() == before


def test_infer_launch_dry_run_then_one_submit_pass(tmp_path: Path, monkeypatch, _runtime_commit, _runtime_probe):
    plan_dir, plan = _build_infer_slurm_plan(tmp_path, _runtime_commit)
    fake = _FakeSlurm(monkeypatch, plan_dir, plan)
    manifest = plan_dir.parent / "run_manifest.tsv"
    before = manifest.read_bytes()

    preview = experiments.launch_infer_run(plan_dir)
    assert cli.main(["infer-launch", "--plan-dir", str(plan_dir)]) == 0

    assert manifest.read_bytes() == before
    assert fake.calls == []
    assert not (plan_dir / "execution_snapshot.json").exists()
    assert preview.launch_rows[0]["target"] == "local"
    assert not experiment_workspace.has_managed_launch_evidence(preview.committed_rows[0])

    launched = experiments.launch_infer_run(plan_dir, dry_run=False)

    row = launched.committed_rows[0]
    assert row["status"] == "queued"
    assert row["scheduler_job_id"] == "3880"
    assert row["launched_at"]
    assert row["planned_runtime_commit"] == _runtime_commit
    assert launched.started_keys == frozenset({("unit-infer", "run-000")})
    assert (plan_dir / "execution_snapshot.json").is_file()
    assert (fake.count("squeue"), fake.count("bash")) == (1, 1)
    events = experiment_workspace.read_experiment_events(plan_dir.parent)
    launched_events = [event for event in events if event["event_type"] == "run_launched"]
    assert [event["scheduler_job_id"] for event in launched_events] == ["3880"]

    repeated = experiments.launch_infer_run(plan_dir, dry_run=False)

    # A submitted run is the monitor's; another pass neither queries nor submits it.
    assert not repeated.started_keys
    assert repeated.committed_rows[0]["status"] == "queued"
    assert (fake.count("squeue"), fake.count("bash")) == (1, 1)


@pytest.mark.parametrize(("state", "status"), [("PENDING", "queued"), ("RUNNING", "running")])
def test_infer_launch_adopts_the_one_job_named_for_the_run(
    tmp_path: Path, monkeypatch, state: str, status: str, _runtime_commit, _runtime_probe
):
    plan_dir, plan = _build_infer_slurm_plan(tmp_path, _runtime_commit)
    fake = _FakeSlurm(monkeypatch, plan_dir, plan)
    fake.jobs = ["3700|another/step/run-000|RUNNING|None|gpu-02", f"3890|{fake.name}|{state}|None|gpu-01"]

    launched = experiments.launch_infer_run(plan_dir, dry_run=False)

    row = launched.committed_rows[0]
    assert fake.count("bash") == 0
    assert row["status"] == status
    assert row["scheduler_job_id"] == "3890"
    assert row["scheduler_raw_state"] == state
    assert row["scheduler_node"] == "gpu-01"
    assert launched.started_keys == frozenset({("unit-infer", "run-000")})


def test_infer_launch_raises_on_duplicate_job_name_without_submitting(
    tmp_path: Path, monkeypatch, _runtime_commit, _runtime_probe
):
    plan_dir, plan = _build_infer_slurm_plan(tmp_path, _runtime_commit)
    fake = _FakeSlurm(monkeypatch, plan_dir, plan)
    fake.jobs = [f"3890|{fake.name}|PENDING|Priority|", f"3891|{fake.name}|RUNNING|None|gpu-01"]

    with pytest.raises(RuntimeError, match="Several Slurm jobs are named"):
        experiments.launch_infer_run(plan_dir, dry_run=False)

    row = _row(plan_dir)
    assert fake.count("bash") == 0
    assert row["status"] == "planned"
    assert not row.get("scheduler_job_id")
    # The run carries launch evidence, so the failure is never recorded as definitely unsubmitted.
    assert experiment_workspace.has_managed_launch_evidence(row)


@pytest.mark.parametrize("artifact", ["slurm.log", "exit_code"])
def test_infer_launch_marks_run_lost_when_job_output_exists_without_a_job(
    tmp_path: Path, monkeypatch, artifact: str, _runtime_commit, _runtime_probe
):
    plan_dir, plan = _build_infer_slurm_plan(tmp_path, _runtime_commit)
    fake = _FakeSlurm(monkeypatch, plan_dir, plan)
    (Path(fake.run["run_dir"]) / artifact).write_text("0\n")

    launched = experiments.launch_infer_run(plan_dir, dry_run=False)

    row = launched.committed_rows[0]
    assert fake.count("bash") == 0
    assert row["status"] == "lost"
    assert row["scheduler_reason"] == "job ran earlier; id unknown"
    assert not row.get("scheduler_job_id")
    assert not launched.started_keys

    experiments.launch_infer_run(plan_dir, dry_run=False)
    experiments.monitor_experiment(plan_dir.parent)
    snapshot = experiments.experiment_status(plan_dir.parent)

    assert fake.count("bash") == 0
    assert _row(plan_dir)["status"] == "lost"
    assert snapshot["summary"]["state"] == "blocked"
    assert any(blocker["code"] == "lost" for blocker in snapshot["blockers"])

    experiments.stop_infer_run(plan_dir, "run-000", reason="job output exists but its id aged out")

    row = _row(plan_dir)
    assert row["status"] == "stopped"
    assert row["stop_reason"] == "job output exists but its id aged out"
    assert fake.count("scancel") == 0
    events = experiment_workspace.read_experiment_events(plan_dir.parent)
    assert sum(event["event_type"] == "run_stopped" for event in events) == 1


def test_infer_launch_records_sbatch_failure_as_launch_failed(
    tmp_path: Path, monkeypatch, _runtime_commit, _runtime_probe
):
    plan_dir, plan = _build_infer_slurm_plan(tmp_path, _runtime_commit)
    fake = _FakeSlurm(monkeypatch, plan_dir, plan)
    fake.sbatch = lambda argv: subprocess.CompletedProcess(argv, 1, "", "sbatch: error: invalid partition")

    launched = experiments.launch_infer_run(plan_dir, dry_run=False)

    row = launched.committed_rows[0]
    assert row["status"] == "launch_failed"
    assert "invalid partition" in row["scheduler_reason"]
    assert not row.get("scheduler_job_id")
    assert not launched.started_keys

    experiments.launch_infer_run(plan_dir, dry_run=False)

    assert fake.count("bash") == 1


def test_infer_launch_sbatch_timeout_raises_and_next_pass_adopts_by_name(
    tmp_path: Path, monkeypatch, _runtime_commit, _runtime_probe
):
    plan_dir, plan = _build_infer_slurm_plan(tmp_path, _runtime_commit)
    fake = _FakeSlurm(monkeypatch, plan_dir, plan)

    def timed_out(argv):
        raise subprocess.TimeoutExpired(argv, managed_scheduler.LAUNCH_TIMEOUT_SECONDS)

    fake.sbatch = timed_out
    with pytest.raises(subprocess.TimeoutExpired):
        experiments.launch_infer_run(plan_dir, dry_run=False)

    row = _row(plan_dir)
    assert row["status"] == "planned"
    assert not row.get("scheduler_job_id")
    assert experiment_workspace.has_managed_launch_evidence(row)

    fake.sbatch = lambda _argv: pytest.fail("the job already exists; it must be adopted, not resubmitted")
    fake.jobs = [f"3880|{fake.name}|PENDING|Priority|"]
    launched = experiments.launch_infer_run(plan_dir, dry_run=False)

    assert launched.committed_rows[0]["status"] == "queued"
    assert launched.committed_rows[0]["scheduler_job_id"] == "3880"
    assert fake.count("bash") == 1


def test_infer_launch_squeue_failure_raises_before_sbatch(tmp_path: Path, monkeypatch, _runtime_commit, _runtime_probe):
    plan_dir, plan = _build_infer_slurm_plan(tmp_path, _runtime_commit)
    fake = _FakeSlurm(monkeypatch, plan_dir, plan)
    fake.squeue_returncode = 1

    with pytest.raises(slurm.SlurmCommandError, match="queue query"):
        experiments.launch_infer_run(plan_dir, dry_run=False)

    assert fake.count("bash") == 0
    assert _row(plan_dir)["status"] == "planned"


def test_infer_launch_guard_failure_records_definitely_unsubmitted_failure(
    tmp_path: Path, monkeypatch, _runtime_commit, _runtime_probe
):
    plan_dir, plan = _build_infer_slurm_plan(tmp_path, _runtime_commit)
    fake = _FakeSlurm(monkeypatch, plan_dir, plan)

    def failed_runtime_probe(*_args, **_kwargs):
        raise ValueError("runtime interpreter changed")

    monkeypatch.setattr(execution_snapshot, "run_execution_command", failed_runtime_probe)
    with pytest.raises(ValueError, match="runtime interpreter changed"):
        experiments.launch_infer_run(plan_dir, dry_run=False)

    row = _row(plan_dir)
    assert row["status"] == "launch_failed"
    assert "Pre-submission guard failed" in row["scheduler_reason"]
    assert not experiment_workspace.has_managed_launch_evidence(row)
    assert fake.calls == []


@pytest.mark.parametrize(("operation", "expected_status"), [("launch", "queued"), ("stop", "stopped")])
def test_infer_lifecycle_reads_workspace_state_only_under_run_lock(
    tmp_path: Path, monkeypatch, operation, expected_status, _runtime_commit, _runtime_probe
):
    plan_dir, plan = _build_infer_slurm_plan(tmp_path, _runtime_commit)
    _FakeSlurm(monkeypatch, plan_dir, plan)
    call = (
        functools.partial(experiments.launch_infer_run, plan_dir, dry_run=False)
        if operation == "launch"
        else functools.partial(experiments.stop_infer_run, plan_dir, "run-000", reason="cancel during monitor commit")
    )

    call_while_run_lock_holder_commits(monkeypatch, plan_dir.parent, call)

    assert _row(plan_dir)["status"] == expected_status


def test_infer_launch_untrusted_plan_does_not_invent_failure_state(tmp_path: Path, _runtime_commit, _runtime_probe):
    plan_dir, plan = _build_infer_slurm_plan(tmp_path, _runtime_commit)
    script = Path(plan["runs"][0]["script"])
    script.write_text(script.read_text() + "# edited after registration\n")
    manifest = plan_dir.parent / "run_manifest.tsv"
    before = manifest.read_bytes()

    with pytest.raises(ValueError, match="differs|changed"):
        experiments.launch_infer_run(plan_dir, dry_run=False)

    assert manifest.read_bytes() == before


def test_run_stop_cli_stops_unsubmitted_infer_plan_without_cancelling(
    tmp_path: Path, monkeypatch, capsys, _runtime_commit, _runtime_probe
):
    plan_dir, plan = _build_infer_slurm_plan(tmp_path, _runtime_commit)
    fake = _FakeSlurm(monkeypatch, plan_dir, plan)
    manifest = plan_dir.parent / "run_manifest.tsv"
    before = manifest.read_bytes()
    with pytest.raises(ValueError, match="non-empty reason"):
        experiments.stop_infer_run(plan_dir, "run-000", reason=" ")
    # A single-run plan still needs --run-id, and it must name that run.
    with pytest.raises(SystemExit):
        cli.main(["run-stop", "--plan-dir", str(plan_dir), "--reason", "cancel unsubmitted unit run"])
    assert cli.main(["run-stop", "--plan-dir", str(plan_dir), "--run-id", "run-001", "--reason", "wrong run"]) == 1
    assert "Unknown run_id: run-001" in capsys.readouterr().err
    assert manifest.read_bytes() == before

    argv = ["run-stop", "--plan-dir", str(plan_dir), "--run-id", "run-000", "--reason", "cancel unsubmitted unit run"]
    assert cli.main(argv) == 0

    row = _row(plan_dir)
    assert row["status"] == "stopped"
    assert row["stop_reason"] == "cancel unsubmitted unit run"
    assert not experiment_workspace.has_managed_launch_evidence(row)
    assert not row.get("stop_requested_at")
    assert (fake.count("squeue"), fake.count("scancel")) == (1, 0)
    events = experiment_workspace.read_experiment_events(plan_dir.parent)
    assert sum(event["event_type"] == "run_stopped" for event in events) == 1
    assert all(event["event_type"] != "run_stop_requested" for event in events)


def test_infer_stop_cancels_a_job_named_for_a_launchable_run(
    tmp_path: Path, monkeypatch, _runtime_commit, _runtime_probe
):
    plan_dir, plan = _build_infer_slurm_plan(tmp_path, _runtime_commit)
    fake = _FakeSlurm(monkeypatch, plan_dir, plan)
    fake.jobs = [f"3880|{fake.name}|PENDING|Priority|"]

    experiments.stop_infer_run(plan_dir, "run-000", reason="cancel a job whose receipt was lost")

    row = _row(plan_dir)
    assert row["status"] == "stopping"
    assert row["scheduler_job_id"] == "3880"
    assert row["stop_requested_at"]
    assert fake.calls[-1] == ["scancel", "3880"]
    events = experiment_workspace.read_experiment_events(plan_dir.parent)
    assert sum(event["event_type"] == "run_stop_requested" for event in events) == 1

    fake.jobs = []
    monitored = experiments.monitor_experiment(plan_dir.parent)

    assert monitored["runs"][0]["status"] == "stopped"
    assert fake.count("bash") == 0


def test_infer_queued_stop_reuses_intent_and_converges_when_job_leaves(
    tmp_path: Path, monkeypatch, _runtime_commit, _runtime_probe
):
    plan_dir, plan = _build_infer_slurm_plan(tmp_path, _runtime_commit)
    fake = _FakeSlurm(monkeypatch, plan_dir, plan)
    experiments.launch_infer_run(plan_dir, dry_run=False)
    reason = "cancel queued unit run"
    real_call = fake.__call__

    def cancel_after_durable_intent(argv, *, timeout):
        if argv[0] == "scancel":
            row = _row(plan_dir)
            assert row["status"] == "stopping"
            assert row["stop_requested_at"]
            assert row["stop_reason"] == reason
            events = experiment_workspace.read_experiment_events(plan_dir.parent)
            assert sum(event["event_type"] == "run_stop_requested" for event in events) == 1
        return real_call(argv, timeout=timeout)

    monkeypatch.setattr(slurm, "run_command", cancel_after_durable_intent)
    assert experiments.stop_infer_run(plan_dir, "run-000", reason=reason) == plan_dir.parent / "run_manifest.tsv"
    intent = _row(plan_dir)["stop_requested_at"]
    experiments.stop_infer_run(plan_dir, "run-000", reason=reason)
    assert _row(plan_dir)["stop_requested_at"] == intent
    assert fake.count("scancel") == 2
    with pytest.raises(ValueError, match="different reason"):
        experiments.stop_infer_run(plan_dir, "run-000", reason="changed reason")
    assert fake.count("scancel") == 2

    fake.jobs = [f"3880|{fake.name}|CANCELLED|None|gpu-01"]
    monitored = experiments.monitor_experiment(plan_dir.parent)

    assert monitored["runs"][0]["status"] == "stopped"
    assert monitored["runs"][0]["scheduler_raw_state"] == "CANCELLED"
    assert monitored["runs"][0]["stop_reason"] == reason
    assert fake.count("squeue") == 2


def test_infer_external_cancel_is_lost_until_exit_code_or_reasoned_stop(
    tmp_path: Path, monkeypatch, _runtime_commit, _runtime_probe
):
    plan_dir, plan = _build_infer_slurm_plan(tmp_path, _runtime_commit)
    fake = _FakeSlurm(monkeypatch, plan_dir, plan)
    experiments.launch_infer_run(plan_dir, dry_run=False)
    fake.jobs = [f"3880|{fake.name}|CANCELLED|None|gpu-01"]

    monitored = experiments.monitor_experiment(plan_dir.parent)

    row = monitored["runs"][0]
    assert row["status"] == "lost"
    assert row["scheduler_raw_state"] == "CANCELLED"
    assert not row.get("stop_requested_at")

    fake.jobs = []
    (Path(fake.run["run_dir"]) / "exit_code").write_text("143\n")
    monitored = experiments.monitor_experiment(plan_dir.parent)

    # lost is not sticky: a later exit code decides the run.
    assert monitored["runs"][0]["status"] == "failed"
    assert monitored["runs"][0]["scheduler_exit_code"] == "143"
    assert fake.count("bash") == 1


@pytest.mark.parametrize("boundary", ["intent", "event"])
def test_infer_stop_failure_preserves_cancel_order_and_committed_intent(
    tmp_path: Path, monkeypatch, boundary: str, _runtime_commit, _runtime_probe
):
    plan_dir, plan = _build_infer_slurm_plan(tmp_path, _runtime_commit)
    fake = _FakeSlurm(monkeypatch, plan_dir, plan)
    experiments.launch_infer_run(plan_dir, dry_run=False)
    manifest = plan_dir.parent / "run_manifest.tsv"
    before = manifest.read_bytes()

    def fail(*_args, **_kwargs):
        raise OSError(f"{boundary} write failed")

    owner_field = {"intent": "merge_run_manifest", "event": "append_event"}[boundary]
    monkeypatch.setattr(experiments, owner_field, fail)

    with pytest.raises(OSError, match=f"{boundary} write failed"):
        experiments.stop_infer_run(plan_dir, "run-000", reason="cancel unit run")

    assert fake.count("scancel") == (1 if boundary == "report" else 0)
    row = _row(plan_dir)
    if boundary == "intent":
        assert manifest.read_bytes() == before
        assert row["status"] == "queued"
        assert not row.get("stop_requested_at")
    else:
        assert row["status"] == "stopping"
        assert row["stop_requested_at"]
        assert row["stop_reason"] == "cancel unit run"
