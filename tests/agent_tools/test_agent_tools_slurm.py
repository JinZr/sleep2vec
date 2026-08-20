from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from agent_tools import slurm


@pytest.mark.parametrize(
    ("stdout", "job_id", "cluster"),
    [("3880\n", "3880", ""), ("3880;wuji-h20\n", "3880", "wuji-h20")],
)
def test_parse_sbatch_output(stdout: str, job_id: str, cluster: str):
    assert slurm.parse_sbatch_output(stdout) == slurm.JobIdentity(job_id, cluster)


@pytest.mark.parametrize("stdout", ["", "0", "-1", "3880\n3881\n", "warning\n3880\n", "3880;bad cluster"])
def test_parse_sbatch_output_rejects_ambiguous_or_malformed_values(stdout: str):
    with pytest.raises(ValueError):
        slurm.parse_sbatch_output(stdout)


def test_submit_quotes_remote_arguments_and_parses_identity(monkeypatch):
    calls = []

    def fake_run_shell(host, command, *, timeout):
        calls.append((host, command, timeout))
        return subprocess.CompletedProcess([], 0, "3880;wuji-h20\n", "")

    monkeypatch.setattr(slurm.transport, "run_shell", fake_run_shell)

    identity = slurm.submit(
        {"target": "ssh", "host": "baichuan3"},
        "/shared/run dir/job.sbatch",
        "token;$(touch nope)",
    )

    assert identity == slurm.JobIdentity("3880", "wuji-h20")
    assert calls == [
        (
            "baichuan3",
            "sbatch --parsable '--comment=token;$(touch nope)' '/shared/run dir/job.sbatch'",
            10,
        )
    ]


def test_submit_preserves_nonzero_result(monkeypatch):
    monkeypatch.setattr(
        slurm.transport,
        "run_shell",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 1, "", "invalid partition"),
    )

    with pytest.raises(slurm.SlurmCommandError) as exc_info:
        slurm.submit({"target": "local"}, "/shared/job.sbatch", "token")

    assert exc_info.value.returncode == 1
    assert "invalid partition" in str(exc_info.value)


def test_active_jobs_filters_exact_submit_token(monkeypatch):
    output = "\n".join(
        [
            "3880|PENDING|Resources|(null)|token-a",
            "3881|RUNNING|None|h20-bj-96|token-b",
        ]
    )
    monkeypatch.setattr(
        slurm.transport,
        "run_shell",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 0, output, ""),
    )

    jobs = slurm.active_jobs({"target": "local"}, submit_token="token-b")

    assert jobs == [slurm.JobObservation("3881", "RUNNING", "", "h20-bj-96", "token-b")]


def test_show_job_parses_exact_job_and_missing_job(monkeypatch):
    results = iter(
        [
            subprocess.CompletedProcess(
                [],
                0,
                "JobId=3880 JobState=COMPLETED Reason=None NodeList=h20-bj-96 Comment=token ExitCode=0:0\n",
                "",
            ),
            subprocess.CompletedProcess([], 1, "", "slurm_load_jobs error: Invalid job id specified"),
        ]
    )
    monkeypatch.setattr(slurm.transport, "run_shell", lambda *_args, **_kwargs: next(results))

    assert slurm.show_job({"target": "local"}, "3880") == slurm.JobObservation(
        "3880", "COMPLETED", "", "h20-bj-96", "token", "0:0"
    )
    assert slurm.show_job({"target": "local"}, "3880") is None


def test_cancel_uses_exact_numeric_job_id(monkeypatch):
    calls = []

    def fake_run_shell(host, command, *, timeout):
        calls.append((host, command))
        return subprocess.CompletedProcess([], 0, "", "")

    monkeypatch.setattr(slurm.transport, "run_shell", fake_run_shell)

    slurm.cancel({"target": "local"}, "3880")

    assert calls == [(None, "scancel 3880")]
    with pytest.raises(ValueError):
        slurm.cancel({"target": "local"}, "3880_2")


@pytest.mark.parametrize(
    ("state", "category"),
    [
        ("PENDING", "queued"),
        ("CONFIGURING", "queued"),
        ("RUNNING", "running"),
        ("COMPLETING", "running"),
        ("COMPLETED+", "completed"),
        ("CANCELLED", "cancelled"),
        ("OUT_OF_MEMORY", "failed"),
        ("TIMEOUT", "failed"),
        ("REQUEUED", "unknown"),
    ],
)
def test_state_category(state: str, category: str):
    assert slurm.state_category(state) == category


def test_parse_exit_code():
    assert slurm.parse_exit_code("0:0") == (0, 0)
    assert slurm.parse_exit_code("7:9") == (7, 9)
    with pytest.raises(ValueError):
        slurm.parse_exit_code("0")


def test_sidecar_identity_requires_frozen_token_and_job_id():
    payload = {
        "schema_version": 1,
        "scheduler_job_id": "3880",
        "scheduler_cluster": "wuji-h20",
        "scheduler_submit_token": "agent-tools-unit",
        "exit_code": 0,
    }

    assert slurm.sidecar_identity(payload, "agent-tools-unit", expected_job_id="3880") == slurm.JobIdentity(
        "3880", "wuji-h20"
    )
    assert slurm.terminal_exit_code(payload) == 0
    with pytest.raises(ValueError, match="submit token"):
        slurm.sidecar_identity(payload, "different")
    with pytest.raises(ValueError, match="job id"):
        slurm.sidecar_identity(payload, "agent-tools-unit", expected_job_id="3881")


def test_normalize_resources_freezes_supported_slurm_fields():
    assert slurm.normalize_resources(
        {
            "type": "slurm",
            "partition": "gpu",
            "cpus_per_task": 8,
            "memory": "64G",
            "walltime": "1-00:00:00",
            "nice": 100,
            "nodelist": "h20-bj-[94,96]",
        },
        2,
    ) == {
        "partition": "gpu",
        "cpus_per_task": 8,
        "memory": "64G",
        "walltime": "1-00:00:00",
        "nice": 100,
        "nodelist": "h20-bj-[94,96]",
        "gpus_per_run": 2,
    }


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("cpus_per_task", True, "positive integer"),
        ("memory", "64 GB", "positive Slurm size"),
        ("walltime", "01:60:00", "HH:MM:SS"),
        ("nice", -1, "0 to 10000"),
        ("nodelist", "node;touch", "node-list expression"),
    ],
)
def test_normalize_resources_rejects_unsafe_values(field: str, value, message: str):
    scheduler = {
        "type": "slurm",
        "partition": "gpu",
        "cpus_per_task": 8,
        "memory": "64G",
        "walltime": "01:00:00",
        field: value,
    }
    with pytest.raises(ValueError, match=message):
        slurm.normalize_resources(scheduler, 1)


def test_render_batch_script_is_one_frozen_leaf_job(tmp_path: Path):
    run = {
        "experiment_id": "unit experiment",
        "step_id": "tune",
        "run_id": "run-000",
        "run_dir": str(tmp_path),
        "config": str(tmp_path / "config.yaml"),
        "config_sha256": "a" * 64,
        "script": str(tmp_path / "launch.sh"),
        "script_sha256": "b" * 64,
        "command": "/opt/python -m sleep2vec.finetune --config config.yaml",
    }
    resources = slurm.normalize_resources(
        {
            "type": "slurm",
            "partition": "gpu",
            "cpus_per_task": 8,
            "memory": "64G",
            "walltime": "01:00:00",
        },
        1,
    )
    token = slurm.submit_token(run, resources, "c" * 40)

    script = slurm.render_batch_script(
        run=run,
        execution={"python": "/opt/python", "runtime_commit": "c" * 40, "workdir": "/shared/repo"},
        resources=resources,
        token=token,
        result_path=tmp_path / "slurm_terminal.json",
        allocation_identity_path=tmp_path / "allocation_identity.json",
        log_path=tmp_path / "slurm.log",
        module="sleep2vec.finetune",
    )

    assert "#SBATCH --nodes=1" in script
    assert "#SBATCH --ntasks=1" in script
    assert "#SBATCH --gres=gpu:1" in script
    assert "#SBATCH --no-requeue" in script
    assert f"#SBATCH --comment={token}" in script
    assert "agent_tools.slurm run-frozen-job" in script
    assert "hparam-run-queue" not in script
    assert "CUDA_VISIBLE_DEVICES" not in script
    assert "start_new_session" not in script


def test_run_frozen_job_writes_allocation_and_terminal_sidecars(tmp_path: Path, monkeypatch):
    config = tmp_path / "config.yaml"
    config.write_text("task: unit\n")
    script = tmp_path / "launch.sh"
    script.write_text("#!/usr/bin/env bash\necho leaf-run\n")
    script.chmod(0o755)
    monkeypatch.setenv("SLURM_JOB_ID", "3880")
    monkeypatch.setenv("SLURM_CLUSTER_NAME", "wuji-h20")

    from agent_tools import managed_scheduler

    monkeypatch.setattr(
        managed_scheduler,
        "inspect_execution_target",
        lambda *_args, **_kwargs: {"module": "sleep2vec.finetune", "runtime_commit": "c" * 40},
    )
    result_path = tmp_path / "slurm_terminal.json"
    allocation_path = tmp_path / "allocation_identity.json"
    log_path = tmp_path / "slurm.log"

    exit_code = slurm.run_frozen_job(
        run_id="run-000",
        command="/opt/python -m sleep2vec.finetune --config config.yaml",
        script=str(script),
        script_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),
        config=str(config),
        config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
        result_path=str(result_path),
        allocation_identity_path=str(allocation_path),
        log_path=str(log_path),
        submit_token="agent-tools-unit",
        workdir=str(tmp_path),
        python="/opt/python",
        runtime_commit="c" * 40,
        module="sleep2vec.finetune",
    )

    assert exit_code == 0
    assert json.loads(allocation_path.read_text())["scheduler_job_id"] == "3880"
    terminal = json.loads(result_path.read_text())
    assert terminal["scheduler_cluster"] == "wuji-h20"
    assert terminal["exit_code"] == 0
    assert "leaf-run" in log_path.read_text()
