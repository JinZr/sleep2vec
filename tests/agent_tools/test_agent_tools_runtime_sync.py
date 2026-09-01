from __future__ import annotations

from contextlib import contextmanager
import fcntl
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import pytest

from agent_tools import experiment_workspace, managed_scheduler, python_programs, run_evidence, runtime_sync, transport
from agent_tools.experiment_workspace import file_sha256
from agent_tools.runtime_sync import sync_runtime


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(["git", "-C", str(repo), *args], check=True, text=True, capture_output=True)
    return result.stdout.strip()


def _commit(repo: Path, message: str, content: str) -> str:
    (repo / "tracked.txt").write_text(content)
    _git(repo, "add", "tracked.txt")
    _git(
        repo,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "-c",
        "core.hooksPath=/dev/null",
        "commit",
        "-m",
        message,
    )
    return _git(repo, "rev-parse", "HEAD")


@pytest.fixture
def rolling_runtime(tmp_path: Path) -> tuple[Path, Path, str]:
    source = tmp_path / "source"
    origin = tmp_path / "origin.git"
    runtime = tmp_path / "runtime"
    subprocess.run(["git", "init", "-q", "-b", "main", str(source)], check=True)
    first = _commit(source, "Initial", "one\n")
    subprocess.run(["git", "init", "-q", "--bare", str(origin)], check=True)
    _git(source, "remote", "add", "origin", str(origin))
    _git(source, "push", "-u", "origin", "main")
    subprocess.run(["git", "clone", "-q", "--branch", "main", str(origin), str(runtime)], check=True)
    return source, runtime, first


def test_runtime_sync_fast_forwards_the_existing_checkout_without_copying(
    rolling_runtime: tuple[Path, Path, str],
) -> None:
    source, runtime, first = rolling_runtime
    second = _commit(source, "Advance", "two\n")
    _git(source, "push", "origin", "main")
    siblings = sorted(path.name for path in runtime.parent.iterdir())

    preview = sync_runtime(runtime)

    assert preview == {
        "status": "update_available",
        "executed": False,
        "host": "",
        "workdir": str(runtime),
        "before_commit": first,
        "upstream_commit": second,
        "after_commit": first,
    }
    assert _git(runtime, "rev-parse", "HEAD") == first

    updated = sync_runtime(runtime, execute=True)

    assert updated["status"] == "fast_forwarded"
    assert updated["before_commit"] == first
    assert updated["after_commit"] == updated["upstream_commit"] == second
    assert (runtime / "tracked.txt").read_text() == "two\n"
    assert sorted(path.name for path in runtime.parent.iterdir()) == siblings


@pytest.mark.parametrize("execute", [False, True], ids=["dry-run", "execute"])
def test_runtime_sync_reports_unchanged_checkout(rolling_runtime: tuple[Path, Path, str], execute: bool) -> None:
    _source, runtime, first = rolling_runtime

    result = sync_runtime(runtime, execute=execute)

    assert result == {
        "status": "unchanged",
        "executed": execute,
        "host": "",
        "workdir": str(runtime),
        "before_commit": first,
        "upstream_commit": first,
        "after_commit": first,
    }


def test_runtime_sync_rejects_tracked_or_importable_untracked_code(
    rolling_runtime: tuple[Path, Path, str],
) -> None:
    _source, runtime, _first = rolling_runtime
    (runtime / "tracked.txt").write_text("dirty\n")

    with pytest.raises(RuntimeError, match="tracked worktree changes"):
        sync_runtime(runtime, execute=True)

    _git(runtime, "checkout", "--", "tracked.txt")
    (runtime / "local_module.py").write_text("VALUE = 1\n")

    with pytest.raises(RuntimeError, match="untracked or ignored importable code"):
        sync_runtime(runtime, execute=True)


def test_runtime_sync_rejects_ignored_importable_code(
    rolling_runtime: tuple[Path, Path, str],
) -> None:
    _source, runtime, _first = rolling_runtime
    (runtime / ".git" / "info" / "exclude").write_text("ignored_module.py\n")
    (runtime / "ignored_module.py").write_text("VALUE = 1\n")

    with pytest.raises(RuntimeError, match="untracked or ignored importable code"):
        sync_runtime(runtime, execute=True)


def test_runtime_sync_fast_forwards_detached_head_without_moving_main(
    rolling_runtime: tuple[Path, Path, str],
) -> None:
    source, runtime, first = rolling_runtime
    second = _commit(source, "Advance", "two\n")
    _git(source, "push", "origin", "main")
    _git(runtime, "checkout", "--detach", first)

    updated = sync_runtime(runtime, execute=True)

    assert updated["status"] == "fast_forwarded"
    assert updated["before_commit"] == first
    assert updated["after_commit"] == updated["upstream_commit"] == second
    assert _git(runtime, "rev-parse", "--abbrev-ref", "HEAD") == "HEAD"
    assert _git(runtime, "rev-parse", "refs/heads/main") == first


def test_runtime_sync_rejects_diverged_head_without_rewriting_it(
    rolling_runtime: tuple[Path, Path, str],
) -> None:
    source, runtime, _first = rolling_runtime
    upstream = _commit(source, "Upstream", "upstream\n")
    _git(source, "push", "origin", "main")
    local = _commit(runtime, "Local", "local\n")

    with pytest.raises(RuntimeError, match="has diverged"):
        sync_runtime(runtime, execute=True)

    assert upstream != local
    assert _git(runtime, "rev-parse", "HEAD") == local


def test_direct_launch_holds_runtime_sync_through_verification_head_capture_and_popen(
    rolling_runtime: tuple[Path, Path, str], tmp_path: Path, monkeypatch
) -> None:
    source, runtime, first = rolling_runtime
    second = _commit(source, "Advance", "two\n")
    _git(source, "push", "origin", "main")
    script = tmp_path / "launch.sh"
    config = tmp_path / "config.yaml"
    log_path = tmp_path / "stdout.log"
    pid_path = tmp_path / "pid.json"
    verification_started = tmp_path / "verification-started"
    verification_release = tmp_path / "verification-release"
    before_popen = tmp_path / "before-popen"
    popen_release = tmp_path / "popen-release"
    script.write_text("#!/usr/bin/env bash\nsleep 30\n")
    config.write_text("task: unit\n")

    real_source = python_programs.source
    launcher_source = real_source("managed_scheduler.process_launch")
    popen_line = "        process = subprocess.Popen(\n"
    assert launcher_source.count(popen_line) == 1
    launcher_source = launcher_source.replace(
        popen_line,
        (
            f'        Path({str(before_popen)!r}).write_text("ready")\n'
            f"        while not Path({str(popen_release)!r}).exists():\n"
            '            __import__("time").sleep(0.01)\n' + popen_line
        ),
    )
    verification_source = (
        "from pathlib import Path\n"
        "import time\n"
        f'Path({str(verification_started)!r}).write_text("ready")\n'
        f"while not Path({str(verification_release)!r}).exists():\n"
        "    time.sleep(0.01)\n"
    )

    def embedded_source(name: str) -> str:
        if name == "managed_scheduler.process_launch":
            return launcher_source
        if name == "managed_scheduler.runtime_identity":
            return verification_source
        if name == "managed_scheduler.cli_preflight":
            return "pass\n"
        return real_source(name)

    monkeypatch.setattr(python_programs, "source", embedded_source)
    command = managed_scheduler.build_launch_command(
        {"workdir": str(runtime), "python": sys.executable, "runtime_commit": first},
        script,
        log_path,
        pid_path,
        [],
        execution_snapshot={"module": "runtime_cli", "module_origin": str(runtime / "runtime_cli.py")},
        config_path=config,
        script_sha256=file_sha256(script),
        config_sha256=file_sha256(config),
        planned_command=f"{sys.executable} -m runtime_cli --value ok",
        run_id="run-000",
    )

    sync_blocked = threading.Event()
    sync_acquired = threading.Event()
    sync_results = []
    sync_errors = []
    real_runtime_lock = runtime_sync.runtime_lock

    @contextmanager
    def observed_runtime_lock(checkout):
        descriptor = os.open(runtime / ".agent-tools-runtime.lock", os.O_RDWR)
        try:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                sync_blocked.set()
            else:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
                raise AssertionError("runtime-sync did not encounter the direct launch lock")
        finally:
            os.close(descriptor)
        with real_runtime_lock(checkout):
            sync_acquired.set()
            yield

    monkeypatch.setattr(runtime_sync, "runtime_lock", observed_runtime_lock)

    def update_runtime() -> None:
        try:
            sync_results.append(sync_runtime(runtime, execute=True))
        except BaseException as exc:
            sync_errors.append(exc)

    launcher = None
    sync_thread = None
    identity = None

    def wait_for(path: Path) -> None:
        deadline = time.monotonic() + 10
        while not path.exists():
            if launcher is not None and launcher.poll() is not None:
                stdout, stderr = launcher.communicate()
                pytest.fail(f"direct launcher exited before {path.name}: {stdout}{stderr}")
            assert time.monotonic() < deadline, f"timed out waiting for {path.name}"
            time.sleep(0.01)

    try:
        launcher = subprocess.Popen(["bash", "-lc", command], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        wait_for(verification_started)
        sync_thread = threading.Thread(target=update_runtime)
        sync_thread.start()
        assert sync_blocked.wait(timeout=5)
        assert not sync_acquired.is_set()
        assert _git(runtime, "rev-parse", "HEAD") == first

        verification_release.touch()
        wait_for(before_popen)
        assert not sync_acquired.is_set()
        assert _git(runtime, "rev-parse", "HEAD") == first

        popen_release.touch()
        stdout, stderr = launcher.communicate(timeout=10)
        assert launcher.returncode == 0, stdout + stderr
        sync_thread.join(timeout=10)
        assert not sync_thread.is_alive()
        assert sync_errors == []
        assert len(sync_results) == 1
        assert sync_results[0]["before_commit"] == first
        assert sync_results[0]["after_commit"] == second
        assert _git(runtime, "rev-parse", "HEAD") == second
        identity = run_evidence.read_process_identity(pid_path, {})
        assert identity is not None
        assert identity["runtime_commit"] == first
    finally:
        verification_release.touch(exist_ok=True)
        popen_release.touch(exist_ok=True)
        if launcher is not None and launcher.poll() is None:
            launcher.terminate()
            try:
                launcher.wait(timeout=10)
            except subprocess.TimeoutExpired:
                launcher.kill()
                launcher.wait(timeout=10)
        if sync_thread is not None:
            sync_thread.join(timeout=10)
        if identity is None and pid_path.exists():
            identity = run_evidence.read_process_identity(pid_path, {})
        if identity is not None and run_evidence.process_identity_running({}, identity) is True:
            run_evidence.stop_process_group({}, identity)


@pytest.mark.parametrize("inherited", [True, False], ids=["inherited", "read-head"])
def test_commit_status_releases_runtime_lock_before_manifest_commit(monkeypatch, inherited: bool) -> None:
    planned_commit = "a" * 40
    actual_commit = "b" * 40
    events = []
    committed = []
    lock_held = False

    @contextmanager
    def runtime_lock(_workdir):
        nonlocal lock_held
        assert not lock_held
        lock_held = True
        events.append("lock-enter")
        try:
            yield
        finally:
            lock_held = False
            events.append("lock-exit")

    def check_output(command, *, text):
        assert lock_held
        assert command == ["git", "rev-parse", "HEAD"]
        assert text is True
        events.append("head")
        return actual_commit + "\n"

    def commit_run_start(root, step_id, run_id, *, planned_runtime_commit, runtime_commit):
        assert not lock_held
        events.append("commit")
        committed.append((root, step_id, run_id, planned_runtime_commit, runtime_commit))
        return [{"step_id": step_id, "run_id": run_id, "status": "running"}]

    runtime_lock_module = importlib.import_module("agent_tools.runtime_lock")
    monkeypatch.setattr(runtime_lock_module, "runtime_lock", runtime_lock)
    monkeypatch.setattr(experiment_workspace, "commit_run_start", commit_run_start)
    monkeypatch.setattr(
        experiment_workspace,
        "merge_run_manifest",
        lambda *_args, **_kwargs: pytest.fail("running provenance must use commit_run_start"),
    )
    monkeypatch.setattr(subprocess, "check_output", check_output)
    monkeypatch.setattr(
        sys,
        "argv",
        ["commit-status", "/experiment", "train", "run-000", "running", "record-runtime-commit", planned_commit],
    )
    if inherited:
        monkeypatch.setenv("AGENT_TOOLS_PROCESS_RUNTIME_COMMIT", actual_commit)
    else:
        monkeypatch.delenv("AGENT_TOOLS_PROCESS_RUNTIME_COMMIT", raising=False)

    exec(compile(python_programs.source("plan_rendering.commit_status"), "commit_status", "exec"), {})

    assert committed == [("/experiment", "train", "run-000", planned_commit, actual_commit)]
    assert events == (["commit"] if inherited else ["lock-enter", "head", "lock-exit", "commit"])


@pytest.mark.parametrize("execute", [False, True], ids=["dry-run", "execute"])
def test_remote_runtime_sync_quotes_the_checkout_and_selected_python(monkeypatch, execute: bool) -> None:
    calls = []
    upstream_commit = "a" * 40 if execute else "b" * 40
    payload = {
        "status": "unchanged" if execute else "update_available",
        "executed": execute,
        "host": "",
        "workdir": "/remote/runtime dir",
        "before_commit": "a" * 40,
        "upstream_commit": upstream_commit,
        "after_commit": "a" * 40,
    }

    def run_shell(host, command, *, timeout):
        calls.append((host, command, timeout))
        return subprocess.CompletedProcess([], 0, json.dumps(payload), "")

    monkeypatch.setattr(runtime_sync.transport, "run_shell", run_shell)

    result = sync_runtime(
        "/remote/runtime dir",
        host="baichuan3",
        remote_python="/opt/conda env/bin/python",
        execute=execute,
    )

    argv = [
        "/opt/conda env/bin/python",
        "-c",
        python_programs.source("runtime_sync.sync"),
        "/remote/runtime dir",
        "1" if execute else "0",
    ]
    expected = " ".join(transport.sh(part) for part in argv)
    assert calls == [("baichuan3", expected, 120)]
    assert result == {**payload, "host": "baichuan3"}


@pytest.mark.parametrize(
    ("payload", "execute"),
    [
        pytest.param({}, False, id="empty"),
        pytest.param(
            {
                "status": "unchanged",
                "executed": False,
                "host": "",
                "workdir": "/remote/runtime",
                "before_commit": "a" * 40,
                "upstream_commit": "a" * 40,
                "after_commit": "a" * 40,
                "unexpected": True,
            },
            False,
            id="extra-field",
        ),
        pytest.param(
            {
                "status": "unchanged",
                "executed": True,
                "host": "",
                "workdir": "/remote/runtime",
                "before_commit": "a" * 40,
                "upstream_commit": "a" * 40,
                "after_commit": "a" * 40,
            },
            False,
            id="execute-mismatch",
        ),
        pytest.param(
            {
                "status": "moved",
                "executed": False,
                "host": "",
                "workdir": "/remote/runtime",
                "before_commit": "a" * 40,
                "upstream_commit": "a" * 40,
                "after_commit": "a" * 40,
            },
            False,
            id="invalid-status",
        ),
        pytest.param(
            {
                "status": "unchanged",
                "executed": False,
                "host": "",
                "workdir": "",
                "before_commit": "a" * 40,
                "upstream_commit": "a" * 40,
                "after_commit": "a" * 40,
            },
            False,
            id="blank-workdir",
        ),
        pytest.param(
            {
                "status": "unchanged",
                "executed": False,
                "host": "",
                "workdir": "relative/runtime",
                "before_commit": "a" * 40,
                "upstream_commit": "a" * 40,
                "after_commit": "a" * 40,
            },
            False,
            id="relative-workdir",
        ),
        pytest.param(
            {
                "status": "unchanged",
                "executed": False,
                "host": "",
                "workdir": "/remote/runtime",
                "before_commit": "A" * 40,
                "upstream_commit": "a" * 40,
                "after_commit": "a" * 40,
            },
            False,
            id="invalid-sha",
        ),
        pytest.param(
            {
                "status": "unchanged",
                "executed": False,
                "host": "",
                "workdir": "/remote/runtime",
                "before_commit": "a" * 40,
                "upstream_commit": "b" * 40,
                "after_commit": "a" * 40,
            },
            False,
            id="unchanged-relation",
        ),
        pytest.param(
            {
                "status": "update_available",
                "executed": False,
                "host": "",
                "workdir": "/remote/runtime",
                "before_commit": "a" * 40,
                "upstream_commit": "b" * 40,
                "after_commit": "b" * 40,
            },
            False,
            id="update-relation",
        ),
        pytest.param(
            {
                "status": "fast_forwarded",
                "executed": True,
                "host": "",
                "workdir": "/remote/runtime",
                "before_commit": "a" * 40,
                "upstream_commit": "b" * 40,
                "after_commit": "a" * 40,
            },
            True,
            id="fast-forward-relation",
        ),
    ],
)
def test_remote_runtime_sync_rejects_untrusted_evidence(monkeypatch, payload: dict, execute: bool) -> None:
    monkeypatch.setattr(
        runtime_sync.transport,
        "run_shell",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 0, json.dumps(payload), ""),
    )

    with pytest.raises(RuntimeError, match="evidence"):
        sync_runtime("/remote/runtime", host="unit-host", execute=execute)


def test_remote_runtime_sync_bootstraps_checkout_without_agent_tools(
    rolling_runtime: tuple[Path, Path, str], monkeypatch
) -> None:
    source, runtime, first = rolling_runtime
    second = _commit(source, "Advance", "two\n")
    _git(source, "push", "origin", "main")
    calls = []

    def run_shell(host, command, *, timeout):
        calls.append((host, command, timeout))
        return subprocess.run(["bash", "-lc", command], text=True, capture_output=True, timeout=timeout)

    assert not (runtime / "agent_tools").exists()
    monkeypatch.setattr(runtime_sync.transport, "run_shell", run_shell)

    result = sync_runtime(runtime, host="unit-host", remote_python=sys.executable, execute=True)

    assert len(calls) == 1
    assert calls[0][0] == "unit-host"
    assert calls[0][2] == 120
    assert result["status"] == "fast_forwarded"
    assert result["host"] == "unit-host"
    assert result["before_commit"] == first
    assert result["after_commit"] == result["upstream_commit"] == second
    assert _git(runtime, "rev-parse", "HEAD") == second
    assert not (runtime / "agent_tools").exists()
