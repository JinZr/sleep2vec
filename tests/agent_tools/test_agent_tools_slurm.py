from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess

import pytest

from agent_tools import slurm


def _frozen_job_inputs(tmp_path: Path, *, script_text: str = "#!/usr/bin/env bash\ntrue\n"):
    config = tmp_path / "config.yaml"
    config.write_text("task: unit\n")
    script = tmp_path / "launch.sh"
    script.write_text(script_text)
    script.chmod(0o755)
    snapshot = {
        "python": "/opt/python",
        "python_version": "3.10.0",
        "runtime_commit": "c" * 40,
        "runtime_repo_root": str(tmp_path),
        "module": "sleep2vec.finetune",
        "module_origin": str(tmp_path / "sleep2vec" / "finetune.py"),
    }
    snapshot_path = tmp_path / "execution_snapshot.json"
    snapshot_path.write_text(json.dumps(snapshot))
    return (
        {
            "run_id": "run-000",
            "command": "/opt/python -m sleep2vec.finetune --config config.yaml",
            "script": str(script),
            "script_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
            "config": str(config),
            "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
            "result_path": str(tmp_path / "slurm_terminal.json"),
            "allocation_identity_path": str(tmp_path / "allocation_identity.json"),
            "execution_snapshot_path": str(snapshot_path),
            "execution_snapshot_sha256": hashlib.sha256(snapshot_path.read_bytes()).hexdigest(),
            "log_path": str(tmp_path / "slurm.log"),
            "submit_token": "agent-tools-unit",
            "workdir": str(tmp_path),
            "python": "/opt/python",
            "runtime_commit": "c" * 40,
            "module": "sleep2vec.finetune",
        },
        snapshot,
    )


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
        execution_snapshot_sha256="d" * 64,
    )

    assert identity == slurm.JobIdentity("3880", "wuji-h20")
    assert calls == [
        (
            "baichuan3",
            slurm.submission_command(
                "/shared/run dir/job.sbatch",
                "token;$(touch nope)",
                "d" * 64,
            ),
            10,
        )
    ]
    assert calls[0][1] == slurm.submission_command(
        "/shared/run dir/job.sbatch",
        "token;$(touch nope)",
        "d" * 64,
    )


def test_submit_preserves_nonzero_result(monkeypatch):
    monkeypatch.setattr(
        slurm.transport,
        "run_shell",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 1, "", "invalid partition"),
    )

    with pytest.raises(slurm.SlurmCommandError) as exc_info:
        slurm.submit(
            {"target": "local"},
            "/shared/job.sbatch",
            "token",
            execution_snapshot_sha256="d" * 64,
        )

    assert exc_info.value.returncode == 1
    assert "invalid partition" in str(exc_info.value)


def test_submit_rejects_malformed_execution_snapshot_digest_before_transport(monkeypatch):
    calls = []
    monkeypatch.setattr(slurm.transport, "run_shell", lambda *args, **kwargs: calls.append((args, kwargs)))

    with pytest.raises(ValueError, match="SHA-256"):
        slurm.submit(
            {"target": "local"},
            "/shared/job.sbatch",
            "token",
            execution_snapshot_sha256="not-a-digest",
        )

    assert calls == []


def test_submit_strips_ambient_sbatch_environment_on_submission_host(tmp_path: Path, monkeypatch):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    capture_env = tmp_path / "env.txt"
    capture_argv = tmp_path / "argv.txt"
    fake_sbatch = fake_bin / "sbatch"
    fake_sbatch.write_text(
        "#!/usr/bin/env bash\n"
        'env | sort > "$CAPTURE_ENV"\n'
        'printf "%s\\n" "$@" > "$CAPTURE_ARGV"\n'
        'printf "3880;wuji-h20\\n"\n'
    )
    fake_sbatch.chmod(0o755)
    calls = []

    def fake_run_shell(host, command, *, timeout):
        calls.append((host, command, timeout))
        env = {
            **os.environ,
            "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
            "CAPTURE_ENV": str(capture_env),
            "CAPTURE_ARGV": str(capture_argv),
            "SBATCH_PARTITION": "wrong-partition",
            "SBATCH_FUTURE_OPTION": "wrong-future-value",
            "KEEP_ME": "kept",
        }
        return subprocess.run(["bash", "-c", command], text=True, capture_output=True, timeout=timeout, env=env)

    monkeypatch.setattr(slurm.transport, "run_shell", fake_run_shell)
    digest = "d" * 64

    identity = slurm.submit(
        {"target": "ssh", "host": "baichuan3"},
        "/shared/job.sbatch",
        "token",
        execution_snapshot_sha256=digest,
    )

    assert identity == slurm.JobIdentity("3880", "wuji-h20")
    assert calls == [("baichuan3", slurm.submission_command("/shared/job.sbatch", "token", digest), 10)]
    environment = capture_env.read_text().splitlines()
    assert not any(line.startswith("SBATCH_") for line in environment)
    assert "KEEP_ME=kept" in environment
    assert capture_argv.read_text().splitlines() == [
        "--parsable",
        "--comment=token",
        "/shared/job.sbatch",
        digest,
    ]


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
                "JobId=3880 JobState=COMPLETED Reason=None NodeList=h20-bj-96 Comment=token ExitCode=0:0 "
                "Priority=42 Nice=0 Partition=gpu Account=(null) QOS=N/A Reservation=(null) "
                "SubmitTime=2026-08-21T01:00:00 EligibleTime=2026-08-21T01:00:01 "
                "StartTime=2026-08-21T01:05:00 TimeLimit=01:00:00 ReqNodeList=(null) Features=(null) "
                "ReqTRES=cpu=8,mem=64G,node=1,billing=8,gres/gpu=1 TresPerNode=gres:gpu:1\n",
                "",
            ),
            subprocess.CompletedProcess([], 1, "", "slurm_load_jobs error: Invalid job id specified"),
        ]
    )
    monkeypatch.setattr(slurm.transport, "run_shell", lambda *_args, **_kwargs: next(results))

    assert slurm.show_job({"target": "local"}, "3880") == slurm.JobObservation(
        "3880",
        "COMPLETED",
        "",
        "h20-bj-96",
        "token",
        "0:0",
        {
            "priority": "42",
            "nice": "0",
            "partition": "gpu",
            "account": "",
            "qos": "",
            "reservation": "",
            "submit_time": "2026-08-21T01:00:00",
            "eligible_time": "2026-08-21T01:00:01",
            "start_time": "2026-08-21T01:05:00",
            "time_limit": "01:00:00",
            "requested_nodes": "",
            "features": "",
            "requested_tres": "cpu=8,mem=64G,node=1,billing=8,gres/gpu=1",
            "tres_per_node": "gres:gpu:1",
        },
    )
    assert slurm.show_job({"target": "local"}, "3880") is None


def test_accounting_job_queries_exact_allocation_on_bound_cluster(monkeypatch):
    calls = []

    def fake_run_command(execution, argv, *, timeout):
        calls.append((execution, argv, timeout))
        output = "3880|COMPLETED|0:0|h20-bj-96\n3880.batch|COMPLETED|0:0|h20-bj-96\n"
        return subprocess.CompletedProcess([], 0, output, "")

    monkeypatch.setattr(slurm, "run_command", fake_run_command)

    assert slurm.accounting_job({"target": "local"}, "3880", cluster="wuji-h20") == slurm.JobObservation(
        "3880", "COMPLETED", node_list="h20-bj-96", exit_code="0:0"
    )
    assert calls == [
        (
            {"target": "local"},
            [
                "sacct",
                "--noheader",
                "--parsable2",
                "--allocations",
                "--clusters=wuji-h20",
                "--jobs",
                "3880",
                "--format=JobIDRaw,State%64,ExitCode,NodeList",
            ],
            10,
        )
    ]


def test_follow_up_commands_route_to_bound_cluster(monkeypatch):
    calls = []
    results = iter(
        [
            subprocess.CompletedProcess([], 0, "3880|RUNNING|None|h20-bj-96|token\n", ""),
            subprocess.CompletedProcess(
                [],
                0,
                "JobId=3880 JobState=RUNNING Reason=None NodeList=h20-bj-96 Comment=token ExitCode=0:0\n",
                "",
            ),
            subprocess.CompletedProcess([], 0, "", ""),
        ]
    )

    def fake_run_command(execution, argv, *, timeout):
        calls.append((execution, argv, timeout))
        return next(results)

    monkeypatch.setattr(slurm, "run_command", fake_run_command)

    assert slurm.active_jobs({"target": "local"}, job_id="3880", cluster="wuji-h20") == [
        slurm.JobObservation("3880", "RUNNING", "", "h20-bj-96", "token")
    ]
    assert slurm.show_job({"target": "local"}, "3880", cluster="wuji-h20") is not None
    slurm.cancel({"target": "local"}, "3880", cluster="wuji-h20")

    assert [argv for _execution, argv, _timeout in calls] == [
        ["squeue", "--noheader", "--format=%i|%T|%R|%N|%k", "--clusters=wuji-h20", "--jobs", "3880"],
        ["scontrol", "--clusters=wuji-h20", "show", "job", "--oneliner", "3880"],
        ["scancel", "--clusters=wuji-h20", "3880"],
    ]


def test_follow_up_commands_reject_invalid_cluster_before_execution(monkeypatch):
    monkeypatch.setattr(slurm, "run_command", lambda *_args, **_kwargs: pytest.fail("must not execute"))

    with pytest.raises(ValueError, match="cluster name is invalid"):
        slurm.active_jobs({"target": "local"}, cluster="wuji;scancel 3880")


def test_cluster_scheduling_capabilities_use_read_only_scontrol_queries(monkeypatch):
    outputs = {
        "scontrol --version": "slurm 20.11.9\n",
        "scontrol show config": """
PriorityType              = priority/basic
SchedulerType             = sched/backfill
AccountingStorageType     = accounting_storage/none
PreemptType               = preempt/none
""",
        "scontrol show partition -o": (
            "PartitionName=gpu State=UP MaxTime=2-00:00:00 TotalNodes=8\n"
            "PartitionName=cpu State=UP MaxTime=INFINITE TotalNodes=16\n"
        ),
        "scontrol show reservation -o": "No reservations in the system\n",
    }
    calls = []

    def fake_run_shell(host, command, *, timeout):
        calls.append((host, command, timeout))
        return subprocess.CompletedProcess([], 0, outputs[command], "")

    monkeypatch.setattr(slurm.transport, "run_shell", fake_run_shell)

    capabilities = slurm.cluster_scheduling_capabilities({"target": "local"}, "gpu")

    assert capabilities == {
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
    assert [command for _host, command, _timeout in calls] == list(outputs)
    assert not any("sprio" in command or "sacctmgr" in command for _host, command, _timeout in calls)


def test_parse_cluster_scheduling_capabilities_detects_multifactor_accounting():
    capabilities = slurm.parse_cluster_scheduling_capabilities(
        version_output="slurm 24.05.2\n",
        config_output="""
PriorityType = priority/multifactor
SchedulerType = sched/backfill
AccountingStorageType = accounting_storage/slurmdbd
PreemptType = preempt/qos
""",
        partition_output="PartitionName=gpu State=UP MaxTime=INFINITE\n",
        reservation_output="ReservationName=urgent State=ACTIVE\n",
        partition="gpu",
    )

    assert capabilities["multifactor_priority"] is True
    assert capabilities["accounting_enabled"] is True
    assert capabilities["preemption_enabled"] is True
    assert capabilities["reservation_count"] == 1


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
        ("EXPEDITING", "queued"),
        ("POWER_UP_NODE", "queued"),
        ("REQUEUED", "queued"),
        ("REQUEUE_FED", "queued"),
        ("REQUEUE_HOLD", "queued"),
        ("RESV_DEL_HOLD", "queued"),
        ("REVOKED", "queued"),
        ("SPECIAL_EXIT", "queued"),
        ("RUNNING", "running"),
        ("COMPLETING", "running"),
        ("SUSPENDED", "running"),
        ("RESIZING", "running"),
        ("SIGNALING", "running"),
        ("STAGE_OUT", "running"),
        ("STOPPED", "running"),
        ("UPDATE_DB", "running"),
        ("COMPLETED+", "completed"),
        ("CANCELLED", "cancelled"),
        ("FAILED", "failed"),
        ("NODE_FAIL", "failed"),
        ("BOOT_FAIL", "failed"),
        ("DEADLINE", "failed"),
        ("PREEMPTED", "failed"),
        ("OUT_OF_MEMORY", "failed"),
        ("TIMEOUT", "failed"),
        ("LAUNCH_FAILED", "failed"),
        ("RECONFIG_FAIL", "failed"),
        ("FUTURE_STATE", "unknown"),
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
    log_path = tmp_path / "run%j" / "slurm%A-%a.log"

    script = slurm.render_batch_script(
        run=run,
        execution={"python": "/opt/python", "runtime_commit": "c" * 40, "workdir": "/shared/repo"},
        resources=resources,
        token=token,
        result_path=tmp_path / "slurm_terminal.json",
        allocation_identity_path=tmp_path / "allocation_identity.json",
        execution_snapshot_path=tmp_path / "execution_snapshot.json",
        log_path=log_path,
        module="sleep2vec.finetune",
    )

    assert "#SBATCH --nodes=1" in script
    assert "#SBATCH --ntasks=1" in script
    assert "#SBATCH --gres=gpu:1" in script
    assert "#SBATCH --no-requeue" in script
    assert f"#SBATCH --comment={token}" in script
    assert f"#SBATCH --output={str(log_path).replace('%', '%%')}" in script
    assert f"#SBATCH --error={str(log_path).replace('%', '%%')}" in script
    assert "agent_tools.slurm run-frozen-job" in script
    assert f"--execution-snapshot-path {tmp_path / 'execution_snapshot.json'}" in script
    assert '--execution-snapshot-sha256 "${1:-}"' in script
    assert f"--log-path {log_path}" in script
    assert "export PYTHONPATH=/shared/repo" in script
    assert "${PYTHONPATH" not in script
    assert "hparam-run-queue" not in script
    assert "CUDA_VISIBLE_DEVICES" not in script
    assert "start_new_session" not in script


def test_run_frozen_job_writes_allocation_and_terminal_sidecars(tmp_path: Path, monkeypatch):
    kwargs, snapshot = _frozen_job_inputs(tmp_path, script_text="#!/usr/bin/env bash\necho leaf-run\n")
    monkeypatch.setenv("SLURM_JOB_ID", "3880")
    monkeypatch.setenv("SLURM_CLUSTER_NAME", "wuji-h20")

    from agent_tools import managed_scheduler

    monkeypatch.setattr(managed_scheduler, "inspect_execution_target", lambda *_args, **_kwargs: snapshot)
    result_path = Path(kwargs["result_path"])
    allocation_path = Path(kwargs["allocation_identity_path"])
    log_path = Path(kwargs["log_path"])

    exit_code = slurm.run_frozen_job(**kwargs)

    assert exit_code == 0
    assert json.loads(allocation_path.read_text())["scheduler_job_id"] == "3880"
    terminal = json.loads(result_path.read_text())
    assert terminal["scheduler_cluster"] == "wuji-h20"
    assert terminal["exit_code"] == 0
    assert "leaf-run" in log_path.read_text()


@pytest.mark.parametrize(("field", "observed"), [("python", "/other/python"), ("python_version", "3.11.0")])
def test_run_frozen_job_rejects_allocation_interpreter_drift_before_spawn(
    tmp_path: Path, monkeypatch, field: str, observed: str
):
    kwargs, frozen_snapshot = _frozen_job_inputs(tmp_path)
    monkeypatch.setenv("SLURM_JOB_ID", "3880")

    observed_snapshot = {**frozen_snapshot, field: observed}
    from agent_tools import managed_scheduler

    monkeypatch.setattr(managed_scheduler, "inspect_execution_target", lambda *_args, **_kwargs: observed_snapshot)
    spawned = []
    monkeypatch.setattr(slurm.subprocess, "Popen", lambda *args, **kwargs: spawned.append((args, kwargs)))
    result_path = Path(kwargs["result_path"])
    allocation_path = Path(kwargs["allocation_identity_path"])
    log_path = Path(kwargs["log_path"])

    exit_code = slurm.run_frozen_job(**kwargs)

    assert exit_code == 2
    assert spawned == []
    assert not allocation_path.exists()
    assert json.loads(result_path.read_text())["exit_code"] == 2
    assert field in log_path.read_text()


def test_run_frozen_job_rejects_changed_execution_snapshot_digest_before_spawn(tmp_path: Path, monkeypatch):
    kwargs, snapshot = _frozen_job_inputs(tmp_path)
    monkeypatch.setenv("SLURM_JOB_ID", "3880")
    snapshot_path = Path(kwargs["execution_snapshot_path"])
    snapshot_path.write_text(json.dumps(snapshot, indent=2))
    from agent_tools import managed_scheduler

    monkeypatch.setattr(managed_scheduler, "inspect_execution_target", lambda *_args, **_kwargs: snapshot)
    spawned = []
    monkeypatch.setattr(slurm.subprocess, "Popen", lambda *args, **kwargs: spawned.append((args, kwargs)))
    result_path = Path(kwargs["result_path"])
    allocation_path = Path(kwargs["allocation_identity_path"])
    log_path = Path(kwargs["log_path"])

    exit_code = slurm.run_frozen_job(**kwargs)

    assert exit_code == 2
    assert spawned == []
    assert not allocation_path.exists()
    assert json.loads(result_path.read_text())["exit_code"] == 2
    assert "execution snapshot changed" in log_path.read_text()


@pytest.mark.parametrize("signum", [slurm.signal.SIGTERM, slurm.signal.SIGINT])
def test_run_frozen_job_does_not_spawn_after_signal_during_verification(tmp_path: Path, monkeypatch, signum: int):
    kwargs, snapshot = _frozen_job_inputs(tmp_path)
    monkeypatch.setenv("SLURM_JOB_ID", "3880")
    handlers = {}

    def capture_signal(current_signum, handler):
        previous = handlers.get(current_signum, slurm.signal.SIG_DFL)
        handlers[current_signum] = handler
        return previous

    monkeypatch.setattr(slurm.signal, "signal", capture_signal)
    from agent_tools import managed_scheduler

    def inspect(*_args, **_kwargs):
        handlers[signum](signum, None)
        return snapshot

    monkeypatch.setattr(managed_scheduler, "inspect_execution_target", inspect)
    spawned = []
    monkeypatch.setattr(slurm.subprocess, "Popen", lambda *args, **kwargs: spawned.append((args, kwargs)))
    result_path = Path(kwargs["result_path"])

    exit_code = slurm.run_frozen_job(**kwargs)

    assert spawned == []
    assert exit_code == 128 + signum
    assert json.loads(result_path.read_text())["exit_code"] == 128 + signum
