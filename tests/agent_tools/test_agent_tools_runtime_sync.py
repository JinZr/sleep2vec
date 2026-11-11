from __future__ import annotations

from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

from agent_tool_test_helpers import SUBPROCESS_WAIT_SECONDS
import pytest

from agent_tools import experiment_workspace, managed_scheduler, python_programs, run_evidence, runtime_sync
from agent_tools.experiment_workspace import file_sha256
from agent_tools.runtime_sync import sync_runtime


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(["git", "-C", str(repo), *args], check=True, text=True, capture_output=True)
    return result.stdout.strip()


def _commit_staged(repo: Path, message: str) -> str:
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


def _commit(repo: Path, message: str, content: str) -> str:
    (repo / "tracked.txt").write_text(content)
    _git(repo, "add", "tracked.txt")
    return _commit_staged(repo, message)


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


def test_runtime_sync_keeps_using_the_checkout_selected_before_lock(
    rolling_runtime: tuple[Path, Path, str], tmp_path: Path, monkeypatch
) -> None:
    _source, runtime, _first = rolling_runtime
    other = tmp_path / "other-runtime"
    subprocess.run(["git", "init", "-q", "-b", "main", str(other)], check=True)
    _commit(other, "Other runtime", "other\n")
    alias = tmp_path / "runtime-alias"
    alias.symlink_to(runtime, target_is_directory=True)
    expected = {"workdir": str(runtime)}

    @contextmanager
    def retargeting_lock(checkout):
        assert checkout == str(runtime)
        alias.unlink()
        alias.symlink_to(other, target_is_directory=True)
        yield

    def observed_sync(checkout: str, *, execute: bool):
        assert checkout == str(runtime)
        assert execute is True
        return expected

    monkeypatch.setattr(runtime_sync, "runtime_lock", retargeting_lock)
    monkeypatch.setattr(runtime_sync, "_sync_local", observed_sync)

    assert sync_runtime(alias, execute=True) is expected


@pytest.mark.parametrize("execute", [False, True], ids=["dry-run", "execute"])
def test_runtime_sync_reports_unchanged_checkout(rolling_runtime: tuple[Path, Path, str], execute: bool) -> None:
    _source, runtime, first = rolling_runtime

    result = sync_runtime(runtime, execute=execute)

    assert result == {
        "status": "unchanged",
        "executed": execute,
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


@pytest.mark.parametrize(
    "index_flag",
    ["--assume-unchanged", "--skip-worktree"],
    ids=["assume-unchanged", "skip-worktree"],
)
def test_runtime_sync_rejects_index_hidden_tracked_changes(
    rolling_runtime: tuple[Path, Path, str], index_flag: str
) -> None:
    _source, runtime, first = rolling_runtime
    _git(runtime, "update-index", index_flag, "tracked.txt")
    (runtime / "tracked.txt").write_text("hidden change\n")

    with pytest.raises(RuntimeError, match="index-hidden tracked worktree changes"):
        sync_runtime(runtime, execute=True)

    assert _git(runtime, "rev-parse", "HEAD") == first


def test_runtime_sync_rejects_ignored_importable_code(
    rolling_runtime: tuple[Path, Path, str],
) -> None:
    _source, runtime, _first = rolling_runtime
    (runtime / ".git" / "info" / "exclude").write_text("ignored_module.py\n")
    (runtime / "ignored_module.py").write_text("VALUE = 1\n")

    with pytest.raises(RuntimeError, match="untracked or ignored importable code"):
        sync_runtime(runtime, execute=True)


def test_runtime_sync_scans_importable_code_from_repository_root(
    rolling_runtime: tuple[Path, Path, str],
) -> None:
    _source, runtime, _first = rolling_runtime
    subdirectory = runtime / "nested"
    subdirectory.mkdir()
    (runtime / "root_module.py").write_text("VALUE = 1\n")

    with pytest.raises(RuntimeError, match="untracked or ignored importable code"):
        sync_runtime(subdirectory, execute=True)


def test_local_runtime_sync_reports_updated_head_when_upstream_deletes_supplied_subdirectory(
    rolling_runtime: tuple[Path, Path, str],
) -> None:
    source, runtime, _first = rolling_runtime
    (source / "nested").mkdir()
    (source / "nested" / "module.py").write_text("VALUE = 1\n")
    _git(source, "add", "nested/module.py")
    _commit(source, "Add nested runtime", "one\n")
    _git(source, "push", "origin", "main")
    sync_runtime(runtime, execute=True)
    subdirectory = runtime / "nested"
    _git(source, "rm", "nested/module.py")
    final_commit = _commit(source, "Delete nested runtime", "two\n")
    _git(source, "push", "origin", "main")

    result = sync_runtime(subdirectory, execute=True)

    assert result["status"] == "fast_forwarded"
    assert result["after_commit"] == final_commit
    assert _git(runtime, "rev-parse", "HEAD") == final_commit
    assert not subdirectory.exists()


@pytest.mark.parametrize("ignored", [False, True], ids=["untracked", "ignored"])
@pytest.mark.parametrize("initialized", [False, True], ids=["namespace-package", "regular-package"])
def test_runtime_sync_rejects_symlinked_package_directories(
    rolling_runtime: tuple[Path, Path, str],
    tmp_path: Path,
    ignored: bool,
    initialized: bool,
) -> None:
    _source, runtime, _first = rolling_runtime
    package = tmp_path / "outside-package"
    package.mkdir()
    (package / "module.py").write_text("VALUE = 1\n")
    if initialized:
        (package / "__init__.py").write_text("VALUE = 1\n")
    (runtime / "plugin").symlink_to(package, target_is_directory=True)
    if ignored:
        (runtime / ".git" / "info" / "exclude").write_text("plugin\n")

    with pytest.raises(RuntimeError, match="untracked or ignored importable code"):
        sync_runtime(runtime, execute=True)


def test_runtime_sync_allows_untracked_non_code_files(rolling_runtime: tuple[Path, Path, str]) -> None:
    _source, runtime, first = rolling_runtime
    (runtime / "notes.txt").write_text("research notes\n")

    result = sync_runtime(runtime, execute=True)

    assert result["after_commit"] == first


@pytest.mark.parametrize("ignored", [False, True], ids=["untracked", "ignored"])
def test_runtime_sync_rejects_sourceless_bytecode(rolling_runtime: tuple[Path, Path, str], ignored: bool) -> None:
    _source, runtime, _first = rolling_runtime
    if ignored:
        (runtime / ".git" / "info" / "exclude").write_text("orphan.pyc\n")
    (runtime / "orphan.pyc").write_bytes(b"sourceless bytecode")

    with pytest.raises(RuntimeError, match="untracked or ignored importable code"):
        sync_runtime(runtime, execute=True)


def test_runtime_sync_allows_non_sourceless_bytecode(
    rolling_runtime: tuple[Path, Path, str],
) -> None:
    _source, runtime, _first = rolling_runtime
    (runtime / "tracked_module.py").write_text("VALUE = 1\n")
    _git(runtime, "add", "tracked_module.py")
    _commit_staged(runtime, "Track module")
    (runtime / "tracked_module.pyc").write_bytes(b"legacy cache with source")
    cache_dir = runtime / "__pycache__"
    cache_dir.mkdir()
    (cache_dir / "orphan.cpython-310.pyc").write_bytes(b"normal cache layout")

    assert sync_runtime(runtime)["status"] == "update_available"


def test_runtime_sync_rejects_update_that_deletes_legacy_bytecode_source(
    rolling_runtime: tuple[Path, Path, str],
) -> None:
    source, runtime, _first = rolling_runtime
    (source / "legacy_module.py").write_text("VALUE = 1\n")
    _git(source, "add", "legacy_module.py")
    _commit_staged(source, "Add runtime module")
    _git(source, "push", "origin", "main")
    sync_runtime(runtime, execute=True)
    before = _git(runtime, "rev-parse", "HEAD")
    (runtime / ".git" / "info" / "exclude").write_text("legacy_module.pyc\n")
    (runtime / "legacy_module.pyc").write_bytes(b"legacy cache with source")
    _git(source, "rm", "legacy_module.py")
    _commit_staged(source, "Remove runtime module")
    _git(source, "push", "origin", "main")

    with pytest.raises(RuntimeError, match="sourceless bytecode"):
        sync_runtime(runtime, execute=True)

    assert _git(runtime, "rev-parse", "HEAD") == before
    assert (runtime / "legacy_module.py").is_file()


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


def test_runtime_sync_ignores_ambient_repository_selection_variables(
    rolling_runtime: tuple[Path, Path, str], tmp_path: Path, monkeypatch
) -> None:
    source, runtime, _first = rolling_runtime
    second = _commit(source, "Advance runtime", "two\n")
    _git(source, "push", "origin", "main")
    decoy = tmp_path / "decoy"
    subprocess.run(["git", "init", "-q", str(decoy)], check=True)
    _commit(decoy, "Initialize decoy", "decoy\n")
    monkeypatch.setenv("GIT_DIR", str(decoy / ".git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(decoy))
    monkeypatch.setenv("GIT_INDEX_FILE", str(decoy / ".git" / "index"))

    result = sync_runtime(runtime, execute=True)

    assert result["status"] == "fast_forwarded"
    assert result["after_commit"] == second
    assert (runtime / "tracked.txt").read_text() == "two\n"
    assert (decoy / "tracked.txt").read_text() == "decoy\n"


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
    _source, runtime, first = rolling_runtime
    script = tmp_path / "launch.sh"
    config = tmp_path / "config.yaml"
    log_path = tmp_path / "stdout.log"
    pid_path = tmp_path / "pid.json"
    verification_started = tmp_path / "verification-started"
    verification_release = tmp_path / "verification-release"
    before_popen = tmp_path / "before-popen"
    popen_release = tmp_path / "popen-release"
    script.write_text("#!/usr/bin/env bash\nsleep 120\n")
    config.write_text("task: unit\n")

    real_source = python_programs.source
    launcher_source = real_source("managed_scheduler.process_launch")
    popen_line = "            process = subprocess.Popen(\n"
    assert launcher_source.count(popen_line) == 1
    launcher_source = launcher_source.replace(
        popen_line,
        (
            f'            Path({str(before_popen)!r}).write_text("ready")\n'
            f"            while not Path({str(popen_release)!r}).exists():\n"
            '                __import__("time").sleep(0.01)\n' + popen_line
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
    sync_result = {
        "status": "unchanged",
        "executed": True,
        "workdir": str(runtime),
        "before_commit": first,
        "upstream_commit": first,
        "after_commit": first,
    }

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
            yield

    def observed_sync_local(checkout: str, *, execute: bool):
        assert checkout == str(runtime)
        assert execute is True
        sync_acquired.set()
        return sync_result

    monkeypatch.setattr(runtime_sync, "runtime_lock", observed_runtime_lock)
    monkeypatch.setattr(runtime_sync, "_sync_local", observed_sync_local)

    def update_runtime() -> None:
        try:
            sync_results.append(sync_runtime(runtime, execute=True))
        except BaseException as exc:
            sync_errors.append(exc)

    launcher = None
    sync_thread = None
    identity = None

    def wait_for(path: Path) -> None:
        deadline = time.monotonic() + SUBPROCESS_WAIT_SECONDS
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
        stdout, stderr = launcher.communicate(timeout=SUBPROCESS_WAIT_SECONDS)
        assert launcher.returncode == 0, stdout + stderr
        assert sync_acquired.wait(timeout=5)
        sync_thread.join(timeout=5)
        assert not sync_thread.is_alive()
        assert sync_errors == []
        assert sync_results == [sync_result]
        assert _git(runtime, "rev-parse", "HEAD") == first
        identity = run_evidence.read_process_identity(pid_path, {})
        assert identity is not None
        assert identity["runtime_commit"] == first
    finally:
        verification_release.touch(exist_ok=True)
        popen_release.touch(exist_ok=True)
        if launcher is not None and launcher.poll() is None:
            launcher.terminate()
            try:
                launcher.wait(timeout=SUBPROCESS_WAIT_SECONDS)
            except subprocess.TimeoutExpired:
                launcher.kill()
                launcher.wait(timeout=SUBPROCESS_WAIT_SECONDS)
        if sync_thread is not None:
            sync_thread.join(timeout=SUBPROCESS_WAIT_SECONDS)
        if identity is None and pid_path.exists():
            identity = run_evidence.read_process_identity(pid_path, {})
        if identity is not None and run_evidence.process_identity_running({}, identity) is True:
            run_evidence.stop_process_group({}, identity)


@pytest.mark.parametrize("inherited", [True, False], ids=["inherited", "read-head"])
def test_commit_status_records_running_provenance(monkeypatch, inherited: bool) -> None:
    planned_commit = "a" * 40
    actual_commit = "b" * 40
    committed = []

    def check_output(command, *, env, text):
        assert command == ["git", "rev-parse", "HEAD"]
        assert "GIT_DIR" not in env
        assert text is True
        return actual_commit + "\n"

    def commit_run_start(root, step_id, run_id, *, planned_runtime_commit, runtime_commit):
        committed.append((root, step_id, run_id, planned_runtime_commit, runtime_commit))
        return [{"step_id": step_id, "run_id": run_id, "status": "running"}]

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


def test_direct_process_launcher_ignores_authored_repository_selection_variables(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    decoy = tmp_path / "decoy"
    subprocess.run(["git", "init", "-q", "-b", "main", str(runtime)], check=True)
    runtime_commit = _commit(runtime, "Initialize runtime", "runtime\n")
    subprocess.run(["git", "init", "-q", "-b", "main", str(decoy)], check=True)
    decoy_commit = _commit(decoy, "Initialize decoy", "decoy\n")
    assert runtime_commit != decoy_commit
    script = tmp_path / "launch.sh"
    pid_path = tmp_path / "pid.json"
    script.write_text("#!/usr/bin/env bash\nsleep 30\n")
    command = managed_scheduler.build_launch_command(
        {
            "workdir": str(runtime),
            "python": sys.executable,
            "runtime_commit": runtime_commit,
            "env": {
                "GIT_DIR": str(decoy / ".git"),
                "GIT_WORK_TREE": str(decoy),
                "GIT_INDEX_FILE": str(decoy / ".git" / "index"),
            },
        },
        script,
        tmp_path / "stdout.log",
        pid_path,
        [],
    )
    identity = None
    try:
        result = subprocess.run(
            ["bash", "-lc", command], text=True, capture_output=True, timeout=SUBPROCESS_WAIT_SECONDS
        )
        assert result.returncode == 0, result.stderr
        identity = run_evidence.read_process_identity(pid_path, {})
        assert identity is not None
        assert identity["runtime_commit"] == runtime_commit
    finally:
        if identity is not None and run_evidence.process_identity_running({}, identity) is True:
            run_evidence.stop_process_group({}, identity)
