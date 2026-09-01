from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

import pytest

from agent_tools import managed_scheduler, python_programs, slurm


def _frozen_job_inputs(tmp_path: Path, *, script_text: str = "#!/usr/bin/env bash\ntrue\n"):
    (tmp_path / ".git").mkdir()
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
            "gpus_per_run": 1,
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


@pytest.mark.parametrize("host", [None, "scheduler"], ids=["local", "ssh"])
def test_submit_strips_ambient_sbatch_environment_on_submission_host(tmp_path: Path, monkeypatch, host: str | None):
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
            "SLURM_CLUSTERS": "wrong-cluster",
            "SLURM_CONF": "/etc/selected slurm.conf",
            "KEEP_ME": "kept",
        }
        return subprocess.run(["bash", "-c", command], text=True, capture_output=True, timeout=timeout, env=env)

    monkeypatch.setattr(slurm.transport, "run_shell", fake_run_shell)
    digest = "d" * 64

    identity = slurm.submit(
        {"target": "ssh", "host": host} if host else {"target": "local"},
        "/shared/job.sbatch",
        "token",
        execution_snapshot_sha256=digest,
    )

    assert identity == slurm.JobIdentity("3880", "wuji-h20")
    assert calls == [(host, slurm.submission_command("/shared/job.sbatch", "token", digest), 10)]
    environment = capture_env.read_text().splitlines()
    assert not any(line.startswith("SBATCH_") for line in environment)
    assert not any(line.startswith("SLURM_CLUSTERS=") for line in environment)
    assert "SLURM_CONF=/etc/selected slurm.conf" in environment
    assert f"PATH={fake_bin}{os.pathsep}{os.environ['PATH']}" in environment
    assert "KEEP_ME=kept" in environment
    assert capture_argv.read_text().splitlines() == [
        "--parsable",
        "--comment=token",
        "/shared/job.sbatch",
        digest,
    ]


@pytest.mark.parametrize("host", [None, "scheduler"], ids=["local", "ssh"])
def test_submit_real_transport_matches_canonical_command_and_preserves_parent_environment(
    tmp_path: Path, monkeypatch, host: str | None
):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    capture_env = tmp_path / "env.txt"
    capture_argv = tmp_path / "argv.txt"
    capture_ssh = tmp_path / "ssh.txt"
    fake_sbatch = fake_bin / "sbatch"
    fake_sbatch.write_text(
        "#!/bin/bash\n" 'env > "$CAPTURE_ENV"\n' 'printf "%s\\n" "$@" > "$CAPTURE_ARGV"\n' 'printf "3880;wuji-h20\\n"\n'
    )
    fake_sbatch.chmod(0o755)
    fake_ssh = fake_bin / "ssh"
    fake_ssh.write_text("#!/bin/bash\n" 'printf "%s\\n" "$@" > "$CAPTURE_SSH"\n' 'exec /bin/bash -c "$2"\n')
    fake_ssh.chmod(0o755)
    child_path = f"{fake_bin}{os.pathsep}{os.environ['PATH']}"
    bash_env = tmp_path / "bash-env.sh"
    bash_env.write_text(f"export PATH={shlex.quote(child_path)}\n")
    for name, value in {
        "PATH": child_path,
        "BASH_ENV": str(bash_env),
        "CAPTURE_ENV": str(capture_env),
        "CAPTURE_ARGV": str(capture_argv),
        "CAPTURE_SSH": str(capture_ssh),
        "SBATCH_PARTITION": "wrong-partition",
        "SBATCH_FUTURE_OPTION": "wrong-future-value",
        "SLURM_CLUSTERS": "wrong-cluster",
        "SLURM_CONF": "/etc/selected slurm.conf",
        "KEEP_ME": "kept",
    }.items():
        monkeypatch.setenv(name, value)
    parent_environment = dict(os.environ)
    calls = []
    real_run = subprocess.run

    def record_run(argv, **kwargs):
        calls.append(argv)
        return real_run(argv, **kwargs)

    monkeypatch.setattr(slurm.transport.subprocess, "run", record_run)
    execution = {"target": "ssh", "host": host} if host else {"target": "local"}
    script = "/shared/run dir/job.sbatch"
    token = "token;$(touch nope)"
    digest = "d" * 64
    canonical = managed_scheduler._slurm_execution_identity(
        execution,
        {"scheduler_script": script, "scheduler_submit_token": token, "log_path": str(tmp_path / "slurm.log")},
        digest,
    )["command"]

    identity = slurm.submit(execution, script, token, execution_snapshot_sha256=digest)

    assert identity == slurm.JobIdentity("3880", "wuji-h20")
    inner = slurm.submission_command(script, token, digest)
    if host:
        assert canonical == shlex.join(["ssh", host, inner])
        assert calls == [["ssh", host, inner]]
        assert capture_ssh.read_text().splitlines() == [host, inner]
    else:
        assert canonical == inner
        assert calls == [["bash", "-lc", canonical]]
        assert not capture_ssh.exists()
    environment = capture_env.read_text().splitlines()
    assert not any(line.startswith(("SBATCH_", "SLURM_CLUSTERS=")) for line in environment)
    assert "SLURM_CONF=/etc/selected slurm.conf" in environment
    assert f"PATH={child_path}" in environment
    assert "KEEP_ME=kept" in environment
    assert capture_argv.read_text().splitlines() == ["--parsable", f"--comment={token}", script, digest]
    assert dict(os.environ) == parent_environment


@pytest.mark.parametrize("host", [None, "scheduler"], ids=["local", "ssh"])
@pytest.mark.parametrize("client", ["scontrol", "squeue", "sacct", "scancel"])
def test_slurm_clients_strip_only_cluster_routing_environment(
    tmp_path: Path, monkeypatch, host: str | None, client: str
):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    capture_env = tmp_path / "env.txt"
    capture_argv = tmp_path / "argv.txt"
    executable = fake_bin / client
    executable.write_text(
        "#!/usr/bin/env bash\n" 'env | sort > "$CAPTURE_ENV"\n' 'printf "%s\\n" "$@" > "$CAPTURE_ARGV"\n'
    )
    executable.chmod(0o755)
    calls = []

    def fake_run_shell(observed_host, command, *, timeout):
        calls.append((observed_host, command, timeout))
        env = {
            **os.environ,
            "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
            "CAPTURE_ENV": str(capture_env),
            "CAPTURE_ARGV": str(capture_argv),
            "SLURM_CLUSTERS": "wrong-cluster",
            "SLURM_CONF": "/etc/selected slurm.conf",
            "SBATCH_PARTITION": "unchanged-for-non-submission",
            "KEEP_ME": "kept",
        }
        return subprocess.run(["bash", "-c", command], text=True, capture_output=True, timeout=timeout, env=env)

    monkeypatch.setattr(slurm.transport, "run_shell", fake_run_shell)
    execution = {"target": "ssh", "host": host} if host else {"target": "local"}
    arguments = ["argument with spaces", "token;$(touch nope)"]

    result = slurm.run_command(execution, [client, *arguments], timeout=7)

    assert result.returncode == 0
    assert calls == [(host, "env -u SLURM_CLUSTERS " + shlex.join([client, *arguments]), 7)]
    environment = capture_env.read_text().splitlines()
    assert not any(line.startswith("SLURM_CLUSTERS=") for line in environment)
    assert "SLURM_CONF=/etc/selected slurm.conf" in environment
    assert f"PATH={fake_bin}{os.pathsep}{os.environ['PATH']}" in environment
    assert "SBATCH_PARTITION=unchanged-for-non-submission" in environment
    assert "KEEP_ME=kept" in environment
    assert capture_argv.read_text().splitlines() == arguments


@pytest.mark.parametrize("host", [None, "scheduler"], ids=["local", "ssh"])
def test_controller_cluster_queries_submission_controller(monkeypatch, host: str | None):
    calls = []

    def fake_run_shell(observed_host, command, *, timeout):
        calls.append((observed_host, command, timeout))
        return subprocess.CompletedProcess([], 0, "Other = ignored\n  ClusterName = wuji-h20\n", "")

    monkeypatch.setattr(slurm.transport, "run_shell", fake_run_shell)
    execution = {"target": "ssh", "host": host} if host else {"target": "local"}

    assert slurm.controller_cluster(execution, timeout=7) == "wuji-h20"
    assert calls == [(host, "env -u SLURM_CLUSTERS scontrol show config", 7)]


@pytest.mark.parametrize(
    "output",
    [
        "",
        "Other = ignored\n",
        "ClusterName\n",
        "ClusterName = \n",
        "ClusterName = wuji-h20\nClusterName = wuji-h20\n",
        "ClusterName = wuji-h20\nClusterName = other\n",
    ],
)
def test_controller_cluster_rejects_missing_empty_or_duplicate_identity(monkeypatch, output: str):
    monkeypatch.setattr(slurm, "run_command", lambda *_args, **_kwargs: subprocess.CompletedProcess([], 0, output, ""))

    with pytest.raises(ValueError, match="exactly one non-empty ClusterName"):
        slurm.controller_cluster({"target": "local"})


@pytest.mark.parametrize("cluster", ["bad cluster", "wuji;scancel 3880", "$(touch nope)"])
def test_controller_cluster_rejects_invalid_identity(monkeypatch, cluster: str):
    monkeypatch.setattr(
        slurm, "run_command", lambda *_args, **_kwargs: subprocess.CompletedProcess([], 0, f"ClusterName={cluster}", "")
    )

    with pytest.raises(ValueError, match="cluster name is invalid"):
        slurm.controller_cluster({"target": "local"})


@pytest.mark.parametrize("returncode", [1, 255])
def test_controller_cluster_preserves_command_failure(monkeypatch, returncode: int):
    monkeypatch.setattr(
        slurm,
        "run_command",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], returncode, "ClusterName=wuji-h20\n", "query failed"),
    )

    with pytest.raises(slurm.SlurmCommandError, match="controller cluster query") as exc_info:
        slurm.controller_cluster({"target": "local"})

    assert exc_info.value.returncode == returncode


def test_controller_cluster_preserves_timeout(monkeypatch):
    def timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired("scontrol", 7)

    monkeypatch.setattr(slurm, "run_command", timeout)

    with pytest.raises(subprocess.TimeoutExpired):
        slurm.controller_cluster({"target": "local"}, timeout=7)


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


@pytest.mark.parametrize("host", [None, "scheduler"], ids=["local", "ssh"])
@pytest.mark.parametrize("direct_controller", [False, True])
def test_active_jobs_batch_uses_real_transport_and_exact_ids(
    tmp_path: Path, monkeypatch, host: str | None, direct_controller: bool
):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    captured = tmp_path / "query.json"
    executable = fake_bin / "squeue"
    executable.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        f"with open({str(captured)!r}, 'w') as stream:\n"
        "    json.dump({'argv': sys.argv[1:], 'env': {key: os.environ.get(key) for key in "
        "['SLURM_CLUSTERS', 'SLURM_CONF', 'SBATCH_PARTITION', 'KEEP_ME']}}, stream)\n"
        "print('3880|RUNNING|None|node-a|token-a')\n"
        "print('3881|PENDING|Resources|(null)|token-b')\n"
        "print('9999|RUNNING|None|other-node|other-token')\n"
    )
    executable.chmod(0o755)
    fake_ssh = fake_bin / "ssh"
    fake_ssh.write_text('#!/bin/bash\n[ "$1" = scheduler ] || exit 99\nexec /bin/bash -c "$2"\n')
    fake_ssh.chmod(0o755)
    child_path = f"{fake_bin}{os.pathsep}{os.environ['PATH']}"
    bash_env = tmp_path / "bash-env.sh"
    bash_env.write_text(f"export PATH={shlex.quote(child_path)}\n")
    for key, value in {
        "PATH": child_path,
        "BASH_ENV": str(bash_env),
        "SLURM_CLUSTERS": "wrong-cluster",
        "SLURM_CONF": "/etc/selected slurm.conf",
        "SBATCH_PARTITION": "preserved",
        "KEEP_ME": "kept",
    }.items():
        monkeypatch.setenv(key, value)
    parent_environment = dict(os.environ)
    execution = {"target": "ssh", "host": host} if host else {"target": "local"}
    execution["scheduler"] = {"direct_controller": direct_controller}

    jobs = slurm.active_jobs(execution, job_id=("3880", "3881"), cluster="wuji-h20")

    assert jobs == [
        slurm.JobObservation("3880", "RUNNING", "", "node-a", "token-a"),
        slurm.JobObservation("3881", "PENDING", "Resources", "", "token-b"),
    ]
    payload = json.loads(captured.read_text())
    assert payload["argv"] == [
        "--noheader",
        "--format=%i|%T|%R|%N|%k",
        *([] if direct_controller else ["--clusters=wuji-h20"]),
        "--jobs",
        "3880,3881",
    ]
    assert payload["env"] == {
        "SLURM_CLUSTERS": None,
        "SLURM_CONF": "/etc/selected slurm.conf",
        "SBATCH_PARTITION": "preserved",
        "KEEP_ME": "kept",
    }
    assert dict(os.environ) == parent_environment


@pytest.mark.parametrize("returncode", [1, 255])
def test_active_jobs_batch_rejects_partial_output_on_missing_job_error(monkeypatch, returncode: int):
    monkeypatch.setattr(
        slurm,
        "run_command",
        lambda *_a, **_k: subprocess.CompletedProcess(
            [], returncode, "3880|RUNNING|None|node-a|token-a\n", "Invalid job id specified"
        ),
    )
    with pytest.raises(slurm.SlurmCommandError, match="active-job query"):
        slurm.active_jobs({"target": "local"}, job_id=("3880", "3881"))


@pytest.mark.parametrize("job_ids", [(), ("3880", "bad;job")])
def test_active_jobs_batch_validates_ids_before_transport(monkeypatch, job_ids):
    monkeypatch.setattr(slurm, "run_command", lambda *_a, **_k: pytest.fail("invalid IDs must not run a command"))
    with pytest.raises(ValueError):
        slurm.active_jobs({"target": "local"}, job_id=job_ids)


def test_active_jobs_batch_does_not_return_partially_parsed_records(monkeypatch):
    monkeypatch.setattr(
        slurm,
        "run_command",
        lambda *_a, **_k: subprocess.CompletedProcess([], 0, "3880|RUNNING|None|node-a|token-a\n3881|RUN", ""),
    )
    with pytest.raises(ValueError, match="Invalid squeue output"):
        slurm.active_jobs({"target": "local"}, job_id=("3880", "3881"))


@pytest.mark.parametrize(
    "diagnostic",
    [
        "slurm_load_jobs error: Invalid job id specified",
        "squeue: error: Unknown JobId 3880",
    ],
)
def test_active_jobs_returns_empty_when_bound_job_is_missing(monkeypatch, diagnostic: str):
    monkeypatch.setattr(
        slurm,
        "run_command",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 1, "", diagnostic),
    )

    assert slurm.active_jobs({"target": "local"}, job_id="3880", cluster="wuji-h20") == []


@pytest.mark.parametrize(
    ("query", "diagnostic"),
    [
        ({"job_id": "3880"}, "squeue: error: Access denied"),
        ({"submit_token": "token"}, "slurm_load_jobs error: Invalid job id specified"),
    ],
)
def test_active_jobs_preserves_other_query_failures(monkeypatch, query: dict, diagnostic: str):
    monkeypatch.setattr(
        slurm,
        "run_command",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 1, "", diagnostic),
    )

    with pytest.raises(slurm.SlurmCommandError, match="active-job query"):
        slurm.active_jobs({"target": "local"}, **query)


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
        output = "3880|COMPLETED|0:0|h20-bj-96|agent-tools-unit\n3880.batch|COMPLETED|0:0|h20-bj-96|\n"
        return subprocess.CompletedProcess([], 0, output, "")

    monkeypatch.setattr(slurm, "run_command", fake_run_command)

    assert slurm.accounting_job(
        {"target": "ssh", "host": "scheduler"},
        "3880",
        submit_token="agent-tools-unit",
        cluster="wuji-h20",
    ) == slurm.JobObservation("3880", "COMPLETED", node_list="h20-bj-96", comment="agent-tools-unit", exit_code="0:0")
    assert calls == [
        (
            {"target": "ssh", "host": "scheduler"},
            [
                "sacct",
                "--duplicates",
                "--noheader",
                "--parsable2",
                "--allocations",
                "--clusters=wuji-h20",
                "--jobs",
                "3880",
                "--format=JobIDRaw,State%64,ExitCode,NodeList,Comment%64",
            ],
            10,
        )
    ]


def test_accounting_job_authenticates_space_padded_comment(monkeypatch):
    submit_token = "agent-tools-unit"
    padded_comment = submit_token.rjust(64)
    monkeypatch.setattr(
        slurm,
        "run_command",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            [], 0, f"3880|COMPLETED|0:0|h20-bj-96|{padded_comment}\n", ""
        ),
    )

    assert slurm.accounting_job({"target": "local"}, "3880", submit_token=submit_token) == slurm.JobObservation(
        "3880", "COMPLETED", node_list="h20-bj-96", comment=submit_token, exit_code="0:0"
    )


@pytest.mark.parametrize("comment", ["", "other-token"])
def test_accounting_job_rejects_unique_unauthenticated_comment(monkeypatch, comment: str):
    monkeypatch.setattr(
        slurm,
        "run_command",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 0, f"3880|COMPLETED|0:0|h20-bj-96|{comment}\n", ""),
    )

    with pytest.raises(ValueError, match="exactly one authenticated allocation row"):
        slurm.accounting_job({"target": "local"}, "3880", submit_token="agent-tools-unit")


def test_accounting_job_selects_unique_submit_token_from_duplicate_records(monkeypatch):
    monkeypatch.setattr(
        slurm,
        "run_command",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            [],
            0,
            "3880|FAILED|1:0|old-node|other-token\n"
            "3880|COMPLETED|0:0|h20-bj-96|agent-tools-unit\n"
            "3880|CANCELLED|0:15|older-node|\n",
            "",
        ),
    )

    assert slurm.accounting_job({"target": "local"}, "3880", submit_token="agent-tools-unit") == slurm.JobObservation(
        "3880", "COMPLETED", node_list="h20-bj-96", comment="agent-tools-unit", exit_code="0:0"
    )


@pytest.mark.parametrize(
    "output",
    [
        "3880|COMPLETED|0:0|node-a|agent-tools-unit\n" "3880|FAILED|1:0|node-b|agent-tools-unit\n",
        "3880|COMPLETED|0:0|node-a|other-token\n3880|FAILED|1:0|node-b|\n",
    ],
)
def test_accounting_job_rejects_duplicate_records_without_one_token_match(monkeypatch, output: str):
    monkeypatch.setattr(
        slurm,
        "run_command",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 0, output, ""),
    )

    with pytest.raises(ValueError, match="exactly one authenticated allocation row"):
        slurm.accounting_job({"target": "local"}, "3880", submit_token="agent-tools-unit")


@pytest.mark.parametrize(
    "execution",
    [{"target": "local"}, {"target": "ssh", "host": "scheduler"}],
    ids=["local", "ssh"],
)
def test_follow_up_commands_route_to_bound_cluster(monkeypatch, execution: dict):
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
            subprocess.CompletedProcess([], 0, "3880|COMPLETED|0:0|h20-bj-96|token\n", ""),
            subprocess.CompletedProcess([], 0, "", ""),
        ]
    )

    def fake_run_command(execution, argv, *, timeout):
        calls.append((execution, argv, timeout))
        return next(results)

    monkeypatch.setattr(slurm, "run_command", fake_run_command)

    assert slurm.active_jobs(execution, job_id="3880", cluster="wuji-h20") == [
        slurm.JobObservation("3880", "RUNNING", "", "h20-bj-96", "token")
    ]
    assert slurm.show_job(execution, "3880", cluster="wuji-h20") is not None
    assert slurm.accounting_job(execution, "3880", submit_token="token", cluster="wuji-h20") is not None
    slurm.cancel(execution, "3880", cluster="wuji-h20")

    assert [argv for _execution, argv, _timeout in calls] == [
        ["squeue", "--noheader", "--format=%i|%T|%R|%N|%k", "--clusters=wuji-h20", "--jobs", "3880"],
        ["scontrol", "--clusters=wuji-h20", "show", "job", "--oneliner", "3880"],
        [
            "sacct",
            "--duplicates",
            "--noheader",
            "--parsable2",
            "--allocations",
            "--clusters=wuji-h20",
            "--jobs",
            "3880",
            "--format=JobIDRaw,State%64,ExitCode,NodeList,Comment%64",
        ],
        ["scancel", "--clusters=wuji-h20", "3880"],
    ]


@pytest.mark.parametrize(
    "execution",
    [
        {"target": "local", "scheduler": {"direct_controller": True}},
        {"target": "ssh", "host": "scheduler", "scheduler": {"direct_controller": True}},
    ],
    ids=["local", "ssh"],
)
def test_follow_up_commands_do_not_route_direct_controller_through_federation(monkeypatch, execution: dict):
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
            subprocess.CompletedProcess([], 0, "3880|COMPLETED|0:0|h20-bj-96|token\n", ""),
            subprocess.CompletedProcess([], 0, "", ""),
        ]
    )

    def fake_run_command(execution, argv, *, timeout):
        calls.append((execution, argv, timeout))
        return next(results)

    monkeypatch.setattr(slurm, "run_command", fake_run_command)

    assert slurm.active_jobs(execution, job_id="3880", cluster="wuji-h20")
    assert slurm.show_job(execution, "3880", cluster="wuji-h20") is not None
    assert slurm.accounting_job(execution, "3880", submit_token="token", cluster="wuji-h20") is not None
    slurm.cancel(execution, "3880", cluster="wuji-h20")

    assert all(not any(arg.startswith("--clusters=") for arg in argv) for _execution, argv, _timeout in calls)


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
        return subprocess.CompletedProcess([], 0, outputs[command.removeprefix("env -u SLURM_CLUSTERS ")], "")

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
    assert [command for _host, command, _timeout in calls] == [
        "env -u SLURM_CLUSTERS " + command for command in outputs
    ]
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

    assert calls == [(None, "env -u SLURM_CLUSTERS scancel 3880")]
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
        ("REVOKED", "unknown"),
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
            "direct_controller": True,
        },
        2,
    ) == {
        "partition": "gpu",
        "cpus_per_task": 8,
        "memory": "64G",
        "walltime": "1-00:00:00",
        "nice": 100,
        "nodelist": "h20-bj-[94,96]",
        "direct_controller": True,
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
        ("direct_controller", "true", "boolean"),
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


@pytest.mark.parametrize("whitespace", [" ", "\t"])
@pytest.mark.parametrize("gpus_per_run", [1, 2, 4])
def test_render_batch_script_is_one_frozen_leaf_job(tmp_path: Path, whitespace: str, gpus_per_run: int):
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
        gpus_per_run,
    )
    token = slurm.submit_token(run, resources, "c" * 40)
    log_path = tmp_path / f"run{whitespace}dir%j" / "slurm%A-%a.log"

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
    assert f"#SBATCH --ntasks={gpus_per_run}" in script
    assert f"#SBATCH --ntasks-per-node={gpus_per_run}" in script
    assert f"#SBATCH --gres=gpu:{gpus_per_run}" in script
    assert "#SBATCH --no-requeue" in script
    assert f"#SBATCH --comment={token}" in script
    escaped_log_path = str(log_path).replace("%", "%%")
    output_directive = next(line for line in script.splitlines() if line.startswith("#SBATCH --output="))
    error_directive = next(line for line in script.splitlines() if line.startswith("#SBATCH --error="))
    assert shlex.split(output_directive) == ["#SBATCH", f"--output={escaped_log_path}"]
    assert shlex.split(error_directive) == ["#SBATCH", f"--error={escaped_log_path}"]
    assert "agent_tools.slurm" in script
    assert f"--execution-snapshot-path {tmp_path / 'execution_snapshot.json'}" in script
    assert f"--gpus-per-run {gpus_per_run}" in script
    assert '--execution-snapshot-sha256 "${1:-}"' in script
    assert f"exec /opt/python -c {shlex.quote(python_programs.source('slurm.worker_bootstrap'))}" in script
    assert f"--log-path {shlex.quote(str(log_path))}" in script
    assert "export PYTHONPATH=/shared/repo" in script
    assert "${PYTHONPATH" not in script
    assert "hparam-run-queue" not in script
    assert "CUDA_VISIBLE_DEVICES" not in script
    assert "start_new_session" not in script


def test_slurm_bootstrap_writes_terminal_sidecar_when_checkout_import_fails(tmp_path: Path):
    runtime = tmp_path / "runtime"
    package = runtime / "agent_tools"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "slurm.py").write_text("raise ImportError('broken rolling checkout')\n")
    result_path = tmp_path / "slurm_terminal.json"
    env = {
        **os.environ,
        "PYTHONPATH": str(runtime),
        "SLURM_JOB_ID": "3880",
        "SLURM_CLUSTER_NAME": "wuji-h20",
    }

    process = subprocess.run(
        [
            sys.executable,
            "-c",
            python_programs.source("slurm.worker_bootstrap"),
            "run-frozen-job",
            "--result-path",
            str(result_path),
            "--submit-token",
            "agent-tools-unit",
        ],
        env=env,
        cwd=runtime,
        text=True,
        capture_output=True,
        timeout=10,
    )

    assert process.returncode != 0
    terminal = json.loads(result_path.read_text())
    assert slurm.sidecar_identity(terminal, "agent-tools-unit", expected_job_id="3880") == slurm.JobIdentity(
        "3880", "wuji-h20"
    )
    assert slurm.terminal_exit_code(terminal) == process.returncode
    assert terminal["runtime_commit"] == ""


def test_slurm_bootstrap_forwards_signal_and_writes_terminal_sidecar(tmp_path: Path):
    runtime = tmp_path / "runtime"
    package = runtime / "agent_tools"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    marker = tmp_path / "worker-started"
    (package / "slurm.py").write_text(
        "from pathlib import Path\n"
        "import os\n"
        "import time\n"
        "Path(os.environ['WORKER_MARKER']).write_text('ready')\n"
        "while True:\n"
        "    time.sleep(1)\n"
    )
    result_path = tmp_path / "slurm_terminal.json"
    env = {
        **os.environ,
        "PYTHONPATH": str(runtime),
        "SLURM_JOB_ID": "3880",
        "SLURM_CLUSTER_NAME": "wuji-h20",
        "WORKER_MARKER": str(marker),
    }
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            python_programs.source("slurm.worker_bootstrap"),
            "run-frozen-job",
            "--result-path",
            str(result_path),
            "--submit-token",
            "agent-tools-unit",
        ],
        env=env,
        cwd=runtime,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        deadline = time.monotonic() + 10
        while not marker.exists():
            if process.poll() is not None:
                stdout, stderr = process.communicate()
                pytest.fail(f"Slurm bootstrap exited before signal: {stdout}{stderr}")
            assert time.monotonic() < deadline, "timed out waiting for Slurm worker"
            time.sleep(0.01)
        process.send_signal(slurm.signal.SIGTERM)
        stdout, stderr = process.communicate(timeout=10)
        assert process.returncode == 128 + slurm.signal.SIGTERM, stdout + stderr
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)

    terminal = json.loads(result_path.read_text())
    assert slurm.sidecar_identity(terminal, "agent-tools-unit", expected_job_id="3880") == slurm.JobIdentity(
        "3880", "wuji-h20"
    )
    assert slurm.terminal_exit_code(terminal) == 128 + slurm.signal.SIGTERM


def test_run_frozen_job_writes_allocation_and_terminal_sidecars(tmp_path: Path, monkeypatch):
    kwargs, snapshot = _frozen_job_inputs(tmp_path, script_text="#!/usr/bin/env bash\necho leaf-run\n")
    monkeypatch.setenv("SLURM_JOB_ID", "3880")
    monkeypatch.setenv("SLURM_CLUSTER_NAME", "wuji-h20")
    monkeypatch.setenv("SLURM_NTASKS", "1")
    for env_name in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        monkeypatch.setenv(env_name, "ambient")

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_srun = fake_bin / "srun"
    fake_srun.write_text('#!/usr/bin/env bash\nfor arg in "$@"; do command="$arg"; done\nexec "$command"\n')
    fake_srun.chmod(0o755)
    monkeypatch.setenv("PATH", f"{fake_bin}{os.pathsep}{os.environ['PATH']}")

    from agent_tools import managed_scheduler

    monkeypatch.setattr(managed_scheduler, "inspect_execution_target", lambda *_args, **_kwargs: snapshot)
    result_path = Path(kwargs["result_path"])
    allocation_path = Path(kwargs["allocation_identity_path"])
    log_path = Path(kwargs["log_path"])

    real_popen = slurm.subprocess.Popen
    spawned = []

    def record_popen(*args, **popen_kwargs):
        spawned.append((args, popen_kwargs))
        return real_popen(*args, **popen_kwargs)

    monkeypatch.setattr(slurm.subprocess, "Popen", record_popen)
    exit_code = slurm.run_frozen_job(**kwargs)

    assert exit_code == 0
    assert json.loads(allocation_path.read_text())["scheduler_job_id"] == "3880"
    terminal = json.loads(result_path.read_text())
    assert terminal["scheduler_cluster"] == "wuji-h20"
    assert terminal["exit_code"] == 0
    assert "leaf-run" in log_path.read_text()
    assert len(spawned) == 1
    assert spawned[0][0][0] == [
        "srun",
        "--nodes=1",
        "--ntasks=1",
        "--ntasks-per-node=1",
        "--kill-on-bad-exit=1",
        "--quit-on-interrupt",
        "--label",
        kwargs["script"],
    ]
    for env_name in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        assert env_name not in spawned[0][1]["env"]


def test_run_frozen_job_records_allocation_runtime_commit_drift_without_blocking(tmp_path: Path, monkeypatch):
    kwargs, frozen_snapshot = _frozen_job_inputs(tmp_path)
    planned_commit = "a" * 40
    actual_commit = "b" * 40
    kwargs["runtime_commit"] = planned_commit
    frozen_snapshot["runtime_commit"] = planned_commit
    snapshot_path = Path(kwargs["execution_snapshot_path"])
    snapshot_path.write_text(json.dumps(frozen_snapshot))
    kwargs["execution_snapshot_sha256"] = hashlib.sha256(snapshot_path.read_bytes()).hexdigest()
    actual_snapshot = {**frozen_snapshot, "expected_runtime_commit": planned_commit, "runtime_commit": actual_commit}
    monkeypatch.setenv("SLURM_JOB_ID", "3880")
    monkeypatch.setenv("SLURM_CLUSTER_NAME", "wuji-h20")
    monkeypatch.setenv("SLURM_NTASKS", "1")

    events = []

    @contextmanager
    def observed_lock(checkout):
        assert checkout == kwargs["workdir"]
        events.append("lock-enter")
        yield
        events.append("lock-exit")

    def inspect_execution_target(execution, *_args, **_kwargs):
        assert events == ["lock-enter"]
        events.append("inspect")
        assert execution["runtime_commit"] == planned_commit
        return actual_snapshot

    class CompletedChild:
        def wait(self):
            assert events[-1] == "lock-exit"
            events.append("wait")
            return 0

    def popen(*_args, **_kwargs):
        assert events[-1] == "inspect"
        events.append("popen")
        return CompletedChild()

    monkeypatch.setattr(slurm, "runtime_lock", observed_lock)
    monkeypatch.setattr(managed_scheduler, "inspect_execution_target", inspect_execution_target)
    monkeypatch.setattr(slurm.subprocess, "Popen", popen)

    assert slurm.run_frozen_job(**kwargs) == 0

    assert events == ["lock-enter", "inspect", "popen", "lock-exit", "wait"]
    allocation = json.loads(Path(kwargs["allocation_identity_path"]).read_text())
    terminal = json.loads(Path(kwargs["result_path"]).read_text())
    assert allocation["execution_snapshot"]["runtime_commit"] == actual_commit
    assert terminal["runtime_commit"] == actual_commit


@pytest.mark.parametrize("artifact_key", ["script", "config"])
def test_run_frozen_job_rechecks_frozen_artifacts_inside_runtime_lock_before_spawn(
    tmp_path: Path, monkeypatch, artifact_key: str
):
    kwargs, snapshot = _frozen_job_inputs(tmp_path)
    monkeypatch.setenv("SLURM_JOB_ID", "3880")
    monkeypatch.setenv("SLURM_NTASKS", "1")
    artifact = Path(kwargs[artifact_key])

    @contextmanager
    def drifting_lock(checkout):
        assert checkout == kwargs["workdir"]
        artifact.write_text("drifted\n")
        yield

    spawned = []
    monkeypatch.setattr(slurm, "runtime_lock", drifting_lock)
    monkeypatch.setattr(managed_scheduler, "inspect_execution_target", lambda *_args, **_kwargs: snapshot)
    monkeypatch.setattr(slurm.subprocess, "Popen", lambda *args, **kwargs: spawned.append((args, kwargs)))

    assert slurm.run_frozen_job(**kwargs) == 2

    assert spawned == []
    assert not Path(kwargs["allocation_identity_path"]).exists()
    assert json.loads(Path(kwargs["result_path"]).read_text())["exit_code"] == 2
    assert f"Frozen run artifact changed before process start: {artifact}" in Path(kwargs["log_path"]).read_text()


@pytest.mark.parametrize(("field", "observed"), [("python", "/other/python"), ("python_version", "3.11.0")])
def test_run_frozen_job_rejects_allocation_interpreter_drift_before_spawn(
    tmp_path: Path, monkeypatch, field: str, observed: str
):
    kwargs, frozen_snapshot = _frozen_job_inputs(tmp_path)
    monkeypatch.setenv("SLURM_JOB_ID", "3880")
    monkeypatch.setenv("SLURM_NTASKS", "1")

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
    monkeypatch.setenv("SLURM_NTASKS", "1")
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
    monkeypatch.setenv("SLURM_NTASKS", "1")
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


@pytest.mark.parametrize(("gpus_per_run", "slurm_ntasks"), [(0, "1"), (2, "1"), (2, "")])
def test_run_frozen_job_rejects_invalid_allocation_task_count_before_spawn(
    tmp_path: Path, monkeypatch, gpus_per_run: int, slurm_ntasks: str
):
    kwargs, snapshot = _frozen_job_inputs(tmp_path)
    kwargs["gpus_per_run"] = gpus_per_run
    monkeypatch.setenv("SLURM_JOB_ID", "3880")
    if slurm_ntasks:
        monkeypatch.setenv("SLURM_NTASKS", slurm_ntasks)
    else:
        monkeypatch.delenv("SLURM_NTASKS", raising=False)

    from agent_tools import managed_scheduler

    inspected = []
    monkeypatch.setattr(
        managed_scheduler,
        "inspect_execution_target",
        lambda *_args, **_kwargs: inspected.append(True) or snapshot,
    )
    spawned = []
    monkeypatch.setattr(slurm.subprocess, "Popen", lambda *args, **popen_kwargs: spawned.append((args, popen_kwargs)))

    exit_code = slurm.run_frozen_job(**kwargs)

    assert exit_code == 2
    assert inspected == []
    assert spawned == []
    assert not Path(kwargs["allocation_identity_path"]).exists()
    assert json.loads(Path(kwargs["result_path"]).read_text())["exit_code"] == 2
    assert "gpus_per_run" in Path(kwargs["log_path"]).read_text()


def test_run_frozen_job_records_aggregate_srun_failure(tmp_path: Path, monkeypatch):
    kwargs, snapshot = _frozen_job_inputs(tmp_path)
    kwargs["gpus_per_run"] = 2
    monkeypatch.setenv("SLURM_JOB_ID", "3880")
    monkeypatch.setenv("SLURM_NTASKS", "2")

    from agent_tools import managed_scheduler

    monkeypatch.setattr(managed_scheduler, "inspect_execution_target", lambda *_args, **_kwargs: snapshot)
    spawned = []

    class FailedStep:
        def wait(self):
            return 7

        def poll(self):
            return 7

    def fake_popen(*args, **popen_kwargs):
        spawned.append((args, popen_kwargs))
        return FailedStep()

    monkeypatch.setattr(slurm.subprocess, "Popen", fake_popen)
    created_sidecars = []
    real_atomic_create_json = slurm._atomic_create_json

    def record_sidecar(path, payload):
        created_sidecars.append(Path(path))
        real_atomic_create_json(path, payload)

    monkeypatch.setattr(slurm, "_atomic_create_json", record_sidecar)

    exit_code = slurm.run_frozen_job(**kwargs)

    assert exit_code == 7
    assert len(spawned) == 1
    assert spawned[0][0][0] == [
        "srun",
        "--nodes=1",
        "--ntasks=2",
        "--ntasks-per-node=2",
        "--kill-on-bad-exit=1",
        "--quit-on-interrupt",
        "--label",
        kwargs["script"],
    ]
    assert created_sidecars == [Path(kwargs["allocation_identity_path"]), Path(kwargs["result_path"])]
    assert json.loads(Path(kwargs["result_path"]).read_text())["exit_code"] == 7


@pytest.mark.parametrize("signum", [slurm.signal.SIGTERM, slurm.signal.SIGINT])
def test_run_frozen_job_forwards_signal_to_active_srun_and_records_terminal_sidecar(
    tmp_path: Path, monkeypatch, signum: int
):
    kwargs, snapshot = _frozen_job_inputs(tmp_path)
    monkeypatch.setenv("SLURM_JOB_ID", "3880")
    monkeypatch.setenv("SLURM_NTASKS", "1")
    handlers = {}

    def capture_signal(current_signum, handler):
        previous = handlers.get(current_signum, slurm.signal.SIG_DFL)
        handlers[current_signum] = handler
        return previous

    monkeypatch.setattr(slurm.signal, "signal", capture_signal)
    from agent_tools import managed_scheduler

    monkeypatch.setattr(managed_scheduler, "inspect_execution_target", lambda *_args, **_kwargs: snapshot)
    sent_signals = []

    class ActiveStep:
        def wait(self):
            handlers[signum](signum, None)
            return -signum

        def poll(self):
            return None

        def send_signal(self, child_signum):
            sent_signals.append(child_signum)

    spawned = []

    def fake_popen(*args, **popen_kwargs):
        spawned.append((args, popen_kwargs))
        return ActiveStep()

    monkeypatch.setattr(slurm.subprocess, "Popen", fake_popen)
    created_sidecars = []
    real_atomic_create_json = slurm._atomic_create_json

    def record_sidecar(path, payload):
        created_sidecars.append(Path(path))
        real_atomic_create_json(path, payload)

    monkeypatch.setattr(slurm, "_atomic_create_json", record_sidecar)

    exit_code = slurm.run_frozen_job(**kwargs)

    assert len(spawned) == 1
    assert "--quit-on-interrupt" in spawned[0][0][0]
    assert sent_signals == [signum]
    assert exit_code == 128 + signum
    assert created_sidecars == [Path(kwargs["allocation_identity_path"]), Path(kwargs["result_path"])]
    assert json.loads(Path(kwargs["result_path"]).read_text())["exit_code"] == 128 + signum


@pytest.mark.parametrize("signum", [slurm.signal.SIGTERM, slurm.signal.SIGINT])
def test_run_frozen_job_forwards_signal_received_before_popen_returns_exactly_once(
    tmp_path: Path, monkeypatch, signum: int
):
    kwargs, snapshot = _frozen_job_inputs(tmp_path)
    monkeypatch.setenv("SLURM_JOB_ID", "3880")
    monkeypatch.setenv("SLURM_NTASKS", "1")
    handlers = {}

    def capture_signal(current_signum, handler):
        previous = handlers.get(current_signum, slurm.signal.SIG_DFL)
        handlers[current_signum] = handler
        return previous

    monkeypatch.setattr(slurm.signal, "signal", capture_signal)
    from agent_tools import managed_scheduler

    monkeypatch.setattr(managed_scheduler, "inspect_execution_target", lambda *_args, **_kwargs: snapshot)
    sent_signals = []

    class ActiveStep:
        def wait(self):
            return -signum

        def poll(self):
            return None

        def send_signal(self, child_signum):
            sent_signals.append(child_signum)

    child = ActiveStep()

    def signal_before_return(*_args, **_kwargs):
        handlers[signum](signum, None)
        return child

    monkeypatch.setattr(slurm.subprocess, "Popen", signal_before_return)
    created_sidecars = []
    real_atomic_create_json = slurm._atomic_create_json

    def record_sidecar(path, payload):
        created_sidecars.append(Path(path))
        real_atomic_create_json(path, payload)

    monkeypatch.setattr(slurm, "_atomic_create_json", record_sidecar)

    exit_code = slurm.run_frozen_job(**kwargs)

    assert sent_signals == [signum]
    assert exit_code == 128 + signum
    assert created_sidecars == [Path(kwargs["allocation_identity_path"]), Path(kwargs["result_path"])]
    assert json.loads(Path(kwargs["result_path"]).read_text())["exit_code"] == 128 + signum
