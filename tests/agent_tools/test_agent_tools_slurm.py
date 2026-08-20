from __future__ import annotations

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
