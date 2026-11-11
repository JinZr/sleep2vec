from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import time

from agent_tool_test_helpers import SUBPROCESS_WAIT_SECONDS
import pytest

from agent_tools import managed_scheduler, slurm


@pytest.mark.parametrize(("stdout", "job_id"), [("3880\n", "3880"), ("3880;wuji-h20\n", "3880")])
def test_parse_sbatch_output(stdout: str, job_id: str):
    assert slurm.parse_sbatch_output(stdout) == job_id


@pytest.mark.parametrize("stdout", ["", "0", "-1", "3880\n3881\n", "warning\n3880\n", "3880;bad cluster"])
def test_parse_sbatch_output_rejects_ambiguous_or_malformed_values(stdout: str):
    with pytest.raises(ValueError):
        slurm.parse_sbatch_output(stdout)


def test_submit_quotes_arguments_and_parses_job_id(monkeypatch):
    calls = []

    def fake_run_shell(command, *, timeout):
        calls.append((command, timeout))
        return subprocess.CompletedProcess([], 0, "3880\n", "")

    monkeypatch.setattr(slurm.transport, "run_shell", fake_run_shell)

    assert slurm.submit("/shared/run dir/job.sbatch") == "3880"
    assert calls == [(slurm.submission_command("/shared/run dir/job.sbatch"), 10)]


def test_submit_preserves_nonzero_result(monkeypatch):
    monkeypatch.setattr(
        slurm.transport,
        "run_shell",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 1, "", "invalid partition"),
    )

    with pytest.raises(slurm.SlurmCommandError) as exc_info:
        slurm.submit("/shared/job.sbatch")

    assert exc_info.value.returncode == 1
    assert "invalid partition" in str(exc_info.value)


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
        'printf "3880\\n"\n'
    )
    fake_sbatch.chmod(0o755)
    calls = []

    def fake_run_shell(command, *, timeout):
        calls.append((command, timeout))
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

    assert slurm.submit("/shared/job.sbatch") == "3880"
    assert calls == [(slurm.submission_command("/shared/job.sbatch"), 10)]
    environment = capture_env.read_text().splitlines()
    assert not any(line.startswith("SBATCH_") for line in environment)
    assert not any(line.startswith("SLURM_CLUSTERS=") for line in environment)
    assert "SLURM_CONF=/etc/selected slurm.conf" in environment
    assert f"PATH={fake_bin}{os.pathsep}{os.environ['PATH']}" in environment
    assert "KEEP_ME=kept" in environment
    assert capture_argv.read_text().splitlines() == ["--parsable", "/shared/job.sbatch"]


def test_submit_real_transport_matches_canonical_command_and_preserves_parent_environment(tmp_path: Path, monkeypatch):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    capture_env = tmp_path / "env.txt"
    capture_argv = tmp_path / "argv.txt"
    fake_sbatch = fake_bin / "sbatch"
    fake_sbatch.write_text(
        "#!/bin/bash\n" 'env > "$CAPTURE_ENV"\n' 'printf "%s\\n" "$@" > "$CAPTURE_ARGV"\n' 'printf "3880\\n"\n'
    )
    fake_sbatch.chmod(0o755)
    child_path = f"{fake_bin}{os.pathsep}{os.environ['PATH']}"
    bash_env = tmp_path / "bash-env.sh"
    bash_env.write_text(f"export PATH={shlex.quote(child_path)}\n")
    for name, value in {
        "PATH": child_path,
        "BASH_ENV": str(bash_env),
        "CAPTURE_ENV": str(capture_env),
        "CAPTURE_ARGV": str(capture_argv),
        "SBATCH_PARTITION": "wrong-partition",
        "SLURM_CLUSTERS": "wrong-cluster",
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
    script = "/shared/run dir/job.sbatch"
    canonical = managed_scheduler._slurm_execution_identity(
        {"target": "local"}, {"scheduler_script": script, "log_path": str(tmp_path / "slurm.log")}
    )["command"]

    assert slurm.submit(script) == "3880"
    assert canonical == slurm.submission_command(script)
    assert calls == [["bash", "-lc", canonical]]
    environment = capture_env.read_text().splitlines()
    assert not any(line.startswith(("SBATCH_", "SLURM_CLUSTERS=")) for line in environment)
    assert "KEEP_ME=kept" in environment
    assert capture_argv.read_text().splitlines() == ["--parsable", script]
    assert dict(os.environ) == parent_environment


@pytest.mark.parametrize("client", ["squeue", "scancel"])
def test_slurm_clients_strip_only_cluster_routing_environment(tmp_path: Path, monkeypatch, client: str):
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

    def fake_run_shell(command, *, timeout):
        calls.append((command, timeout))
        env = {
            **os.environ,
            "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
            "CAPTURE_ENV": str(capture_env),
            "CAPTURE_ARGV": str(capture_argv),
            "SLURM_CLUSTERS": "wrong-cluster",
            "SLURM_CONF": "/etc/selected slurm.conf",
            "SBATCH_PARTITION": "unchanged-for-non-submission",
        }
        return subprocess.run(["bash", "-c", command], text=True, capture_output=True, timeout=timeout, env=env)

    monkeypatch.setattr(slurm.transport, "run_shell", fake_run_shell)
    arguments = ["argument with spaces", "token;$(touch nope)"]

    result = slurm.run_command([client, *arguments], timeout=7)

    assert result.returncode == 0
    assert calls == [("env -u SLURM_CLUSTERS " + shlex.join([client, *arguments]), 7)]
    environment = capture_env.read_text().splitlines()
    assert not any(line.startswith("SLURM_CLUSTERS=") for line in environment)
    assert "SLURM_CONF=/etc/selected slurm.conf" in environment
    assert "SBATCH_PARTITION=unchanged-for-non-submission" in environment
    assert capture_argv.read_text().splitlines() == arguments


def test_user_jobs_runs_one_all_states_query_and_parses_names(monkeypatch):
    calls = []

    def fake_run_shell(command, *, timeout):
        calls.append((command, timeout))
        return subprocess.CompletedProcess(
            [],
            0,
            "3880|exp/tune/run-001|PENDING|Priority|(null)\n"
            "3881|exp/tune/run-002|RUNNING|None|h20-bj-94\n"
            "3882|other|job|name|COMPLETED+|None|h20-bj-96\n"
            "3883_[1-4]|array|PENDING|Resources|\n",
            "",
        )

    monkeypatch.setattr(slurm.transport, "run_shell", fake_run_shell)

    jobs = slurm.user_jobs(timeout=5)

    assert calls == [
        ("env -u SLURM_CLUSTERS squeue --me --states=all --noheader '--format=%i|%j|%T|%r|%N'", 5),
    ]
    assert jobs == [
        slurm.JobObservation("3880", "exp/tune/run-001", "PENDING", "Priority", ""),
        slurm.JobObservation("3881", "exp/tune/run-002", "RUNNING", "", "h20-bj-94"),
        slurm.JobObservation("3882", "other|job|name", "COMPLETED", "", "h20-bj-96"),
        slurm.JobObservation("3883_[1-4]", "array", "PENDING", "Resources", ""),
    ]


@pytest.mark.parametrize("returncode", [1, 255])
def test_user_jobs_raises_on_query_failure(monkeypatch, returncode: int):
    monkeypatch.setattr(
        slurm.transport,
        "run_shell",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], returncode, "3880|a|RUNNING||n\n", "timeout"),
    )

    with pytest.raises(slurm.SlurmCommandError, match="queue query"):
        slurm.user_jobs()


@pytest.mark.parametrize("line", ["3880", "3880|name|RUNNING|reason"])
def test_parse_squeue_line_rejects_short_rows(line: str):
    with pytest.raises(ValueError, match="Invalid squeue output row"):
        slurm.parse_squeue_line(line)


def test_cancel_uses_exact_numeric_job_id(monkeypatch):
    calls = []

    def fake_run_shell(command, *, timeout):
        calls.append(command)
        return subprocess.CompletedProcess([], 0, "", "")

    monkeypatch.setattr(slurm.transport, "run_shell", fake_run_shell)

    slurm.cancel("3880")

    assert calls == ["env -u SLURM_CLUSTERS scancel 3880"]
    with pytest.raises(ValueError):
        slurm.cancel("3880_2")


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
        ("controller", True, "Unknown execution.scheduler field: controller"),
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


def _frozen_run(tmp_path: Path) -> dict[str, str]:
    run_dir = tmp_path / "runs" / "run-000--lr-1e-4"
    return {
        "experiment_id": "unit-exp",
        "step_id": "tune",
        "run_id": "run-000",
        "run_dir": str(run_dir),
        "config": str(run_dir / "config.yaml"),
        "config_sha256": "a" * 64,
        "script": str(run_dir / "launch.sh"),
        "script_sha256": "b" * 64,
        "log_path": str(run_dir / "slurm.log"),
    }


@pytest.mark.parametrize("gpus_per_run", [1, 2, 4])
@pytest.mark.parametrize("nodelist", ["", "h20-bj-94"])
def test_render_batch_script_is_one_frozen_leaf_job(tmp_path: Path, gpus_per_run: int, nodelist: str):
    run = _frozen_run(tmp_path)
    resources = slurm.normalize_resources(
        {
            "type": "slurm",
            "partition": "gpu",
            "cpus_per_task": 8,
            "memory": "64G",
            "walltime": "01:00:00",
            "nice": 5,
            "nodelist": nodelist,
        },
        gpus_per_run,
    )

    script = slurm.render_batch_script(
        run=run,
        execution={"workdir": "/shared/repo", "env": {"WANDB_MODE": "offline"}},
        resources=resources,
    )
    lines = script.splitlines()

    directives = [line for line in lines if line.startswith("#SBATCH")]
    assert directives == [
        "#SBATCH --job-name=unit-exp/tune/run-000",
        "#SBATCH --partition=gpu",
        "#SBATCH --nodes=1",
        f"#SBATCH --ntasks={gpus_per_run}",
        f"#SBATCH --ntasks-per-node={gpus_per_run}",
        f"#SBATCH --gres=gpu:{gpus_per_run}",
        "#SBATCH --cpus-per-task=8",
        "#SBATCH --mem=64G",
        "#SBATCH --time=01:00:00",
        "#SBATCH --nice=5",
        *([f"#SBATCH --nodelist={nodelist}"] if nodelist else []),
        "#SBATCH --no-requeue",
        f"#SBATCH --output={run['log_path']}",
    ]
    assert slurm.job_name(run) == "unit-exp/tune/run-000"
    # Explicit TERM/INT traps: an EXIT trap would see $?=0 after SIGTERM and record success.
    assert "trap 'finish 143' TERM" in lines
    assert "trap 'finish 130' INT" in lines
    assert not any("EXIT" in line for line in lines)
    assert 'mv -f "$run_dir/exit_code.tmp" "$run_dir/exit_code"' in script
    # The job records the checkout it runs and never blocks on drift from the planned commit.
    assert 'git rev-parse HEAD > "$run_dir/runtime_commit" || finish 3' in lines
    assert sum("git " in line for line in lines) == 1
    assert "git diff" not in script and "--quiet" not in script
    assert "sha256sum --check --strict <<'EOF' || finish 3" in lines
    assert f"{'a' * 64}  {run['config']}" in lines
    assert f"{'b' * 64}  {run['script']}" in lines
    assert lines[-2:] == [
        f"srun --nodes=1 --ntasks={gpus_per_run} --ntasks-per-node={gpus_per_run} --kill-on-bad-exit=1 --label "
        f"bash {run['script']}",
        "finish $?",
    ]
    assert "cd /shared/repo || finish 3" in lines
    assert "export PYTHONPATH=/shared/repo" in lines
    assert "export WANDB_MODE=offline" in lines
    assert "unset RANK LOCAL_RANK WORLD_SIZE MASTER_ADDR MASTER_PORT" in lines
    assert "CUDA_VISIBLE_DEVICES" not in script


def test_render_batch_script_escapes_output_path(tmp_path: Path):
    run = {**_frozen_run(tmp_path), "log_path": str(tmp_path / "run dir%j" / "slurm.log")}
    resources = slurm.normalize_resources(
        {"type": "slurm", "partition": "gpu", "cpus_per_task": 8, "memory": "64G", "walltime": "01:00:00"}, 1
    )

    script = slurm.render_batch_script(run=run, execution={"workdir": "/shared/repo"}, resources=resources)

    output_directive = next(line for line in script.splitlines() if line.startswith("#SBATCH --output="))
    assert shlex.split(output_directive) == ["#SBATCH", f"--output={run['log_path'].replace('%', '%%')}"]


def _executable_job(tmp_path: Path, srun_body: str) -> tuple[Path, dict[str, str], dict[str, str]]:
    if shutil.which("sha256sum") is None:
        pytest.skip("sha256sum is unavailable on this host")
    run = _frozen_run(tmp_path)
    run_dir = Path(run["run_dir"])
    run_dir.mkdir(parents=True)
    Path(run["config"]).write_text("model: {}\n")
    Path(run["script"]).write_text("#!/usr/bin/env bash\ntrue\n")
    run["config_sha256"] = hashlib.sha256(Path(run["config"]).read_bytes()).hexdigest()
    run["script_sha256"] = hashlib.sha256(Path(run["script"]).read_bytes()).hexdigest()
    workdir = tmp_path / "repo"
    workdir.mkdir()
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    (fake_bin / "srun").write_text(f"#!/usr/bin/env bash\n{srun_body}\n")
    (fake_bin / "git").write_text("#!/usr/bin/env bash\necho " + "c" * 40 + "\n")
    for executable in fake_bin.iterdir():
        executable.chmod(0o755)
    resources = slurm.normalize_resources(
        {"type": "slurm", "partition": "gpu", "cpus_per_task": 8, "memory": "64G", "walltime": "01:00:00"}, 2
    )
    job = run_dir / "job.sbatch"
    job.write_text(slurm.render_batch_script(run=run, execution={"workdir": str(workdir)}, resources=resources))
    env = {**os.environ, "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}", "SLURM_JOB_ID": "3880"}
    return job, run, env


@pytest.mark.parametrize("srun_exit", [0, 7])
def test_batch_script_records_srun_exit_code_and_runtime_commit(tmp_path: Path, srun_exit: int):
    job, run, env = _executable_job(tmp_path, f'echo "srun $*"; exit {srun_exit}')

    result = subprocess.run(
        ["bash", str(job)], env=env, capture_output=True, text=True, timeout=SUBPROCESS_WAIT_SECONDS
    )

    run_dir = Path(run["run_dir"])
    assert result.returncode == srun_exit
    assert (run_dir / "exit_code").read_text() == f"{srun_exit}\n"
    assert not (run_dir / "exit_code.tmp").exists()
    assert (run_dir / "runtime_commit").read_text() == "c" * 40 + "\n"
    assert "--ntasks=2" in result.stdout


def test_batch_script_rejects_changed_frozen_file_before_srun(tmp_path: Path):
    job, run, env = _executable_job(tmp_path, 'echo "srun must not run"; exit 0')
    Path(run["config"]).write_text("model: {changed: true}\n")

    result = subprocess.run(
        ["bash", str(job)], env=env, capture_output=True, text=True, timeout=SUBPROCESS_WAIT_SECONDS
    )

    assert result.returncode == 3
    assert (Path(run["run_dir"]) / "exit_code").read_text() == "3\n"
    assert "srun must not run" not in result.stdout


@pytest.mark.parametrize(("signum", "exit_code"), [(signal.SIGTERM, 143), (signal.SIGINT, 130)])
def test_batch_script_records_signal_exit_code(tmp_path: Path, signum: int, exit_code: int):
    started = tmp_path / "srun-started"
    job, run, env = _executable_job(tmp_path, f"touch {shlex.quote(str(started))}; exec sleep 60")
    process = subprocess.Popen(["bash", str(job)], env=env, start_new_session=True)
    deadline = time.monotonic() + SUBPROCESS_WAIT_SECONDS
    while not started.exists():
        assert process.poll() is None and time.monotonic() < deadline
        time.sleep(0.05)

    # Slurm signals every process of a cancelled job, including the batch shell and srun.
    os.killpg(process.pid, signum)

    assert process.wait(timeout=SUBPROCESS_WAIT_SECONDS) == exit_code
    assert (Path(run["run_dir"]) / "exit_code").read_text() == f"{exit_code}\n"
