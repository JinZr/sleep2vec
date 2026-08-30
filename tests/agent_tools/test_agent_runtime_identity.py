from __future__ import annotations

import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
from pathlib import Path
import socket
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest

from agent_tools import python_programs

GIT_COMMANDS = [
    ["git", "rev-parse", "HEAD"],
    ["git", "rev-parse", "--show-toplevel"],
    ["git", "status", "--porcelain", "--untracked-files=no"],
    ["git", "ls-files", "--others", "--exclude-standard", "--", "*.py", "*.pyi", "*.so"],
    ["git", "ls-files", "--others", "--ignored", "--exclude-standard", "--", "*.py", "*.pyi", "*.so"],
]


@pytest.fixture
def identity_probe(tmp_path, monkeypatch):
    program = compile(python_programs.source("managed_scheduler.runtime_identity"), "runtime_identity", "exec")
    payload = {
        "python": sys.executable,
        "python_version": sys.version.split()[0],
        "runtime_commit": "a" * 40,
        "runtime_repo_root": str(tmp_path),
        "runtime_hostname": socket.gethostname(),
        "module": "runtime_cli",
        "module_origin": str(tmp_path / "runtime_cli.py"),
    }
    results = [
        subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")
        for command, stdout in zip(GIT_COMMANDS, ("a" * 40 + "\n", str(tmp_path) + "\n", "", "", ""))
    ]
    monkeypatch.setattr(subprocess, "run", lambda command, **_kwargs: results[GIT_COMMANDS.index(command)])
    monkeypatch.setattr(importlib.util, "find_spec", lambda _module: SimpleNamespace(origin=payload["module_origin"]))
    monkeypatch.setattr(
        sys,
        "argv",
        ["-c", "runtime_cli", "{}", json.dumps([{"path": str(tmp_path / "artifact"), "sha256": "0" * 64}])],
    )
    return program, results, payload


def test_runtime_identity_uses_five_exact_queries_with_three_ordered_workers(
    tmp_path, monkeypatch, capsys, identity_probe
):
    program, results, payload = identity_probe
    artifact = tmp_path / "artifact"
    artifact.write_bytes(b"frozen artifact")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "-c",
            "runtime_cli",
            json.dumps(payload),
            json.dumps([{"path": str(artifact), "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest()}]),
        ],
    )
    barrier = threading.Barrier(3, timeout=5)
    finished = [threading.Event() for _ in GIT_COMMANDS]
    lock = threading.Lock()
    active = 0
    peak_active = 0
    calls = []
    completion_order = []
    executor_sizes = []

    def create_executor(*, max_workers):
        executor_sizes.append(max_workers)
        return ThreadPoolExecutor(max_workers=max_workers)

    def run_git(command, **kwargs):
        nonlocal active, peak_active
        index = GIT_COMMANDS.index(command)
        with lock:
            calls.append((command, kwargs))
            active += 1
            peak_active = max(peak_active, active)
        try:
            if index < 3:
                barrier.wait()
            if index == 0:
                assert finished[1].wait(5)
            elif index == 1:
                assert finished[4].wait(5)
            with lock:
                completion_order.append(index)
        finally:
            with lock:
                active -= 1
        finished[index].set()
        return results[index]

    monkeypatch.setattr(concurrent.futures, "ThreadPoolExecutor", create_executor)
    monkeypatch.setattr(subprocess, "run", run_git)

    exec(program, {})

    assert executor_sizes == [3]
    assert peak_active == 3
    assert active == 0
    assert completion_order == [2, 3, 4, 1, 0]
    assert sorted(calls, key=lambda call: GIT_COMMANDS.index(call[0])) == [
        (command, {"text": True, "capture_output": True}) for command in GIT_COMMANDS
    ]
    output = capsys.readouterr()
    assert output.err == ""
    assert json.loads(output.out) == payload


@pytest.mark.parametrize("failed_query", range(5))
def test_runtime_identity_git_failure_precedes_runtime_reads(monkeypatch, capsys, identity_probe, failed_query):
    program, results, _payload = identity_probe
    results[failed_query].returncode = 1
    results[failed_query].stderr = f"git failure {failed_query}"
    results[2].stdout = " M runtime_cli.py\n"
    results[3].stdout = "new_module.py\n"
    monkeypatch.setattr(importlib.util, "find_spec", lambda _module: pytest.fail("must not resolve module"))
    monkeypatch.setattr(Path, "read_bytes", lambda _path: pytest.fail("must not read frozen artifact"))

    with pytest.raises(SystemExit) as error:
        exec(program, {})

    assert error.value.code == 2
    output = capsys.readouterr()
    assert output.out == ""
    assert output.err == f"git failure {failed_query}\n"


@pytest.mark.parametrize("first_diagnostic", range(5))
def test_runtime_identity_git_stderr_priority_follows_query_order(
    monkeypatch, capsys, identity_probe, first_diagnostic
):
    program, results, _payload = identity_probe
    results[-1].returncode = 1
    for index in range(first_diagnostic, 5):
        results[index].stderr = f"git diagnostic {index}"
    monkeypatch.setattr(importlib.util, "find_spec", lambda _module: pytest.fail("must not resolve module"))
    monkeypatch.setattr(Path, "read_bytes", lambda _path: pytest.fail("must not read frozen artifact"))

    with pytest.raises(SystemExit) as error:
        exec(program, {})

    assert error.value.code == 2
    output = capsys.readouterr()
    assert output.out == ""
    assert output.err == f"git diagnostic {first_diagnostic}\n"


@pytest.mark.parametrize("failed_query", range(5))
def test_runtime_identity_git_oserror_precedes_runtime_reads(monkeypatch, capsys, identity_probe, failed_query):
    program, results, _payload = identity_probe

    def run_git(command, **_kwargs):
        index = GIT_COMMANDS.index(command)
        if index == failed_query:
            raise OSError(f"git unavailable {failed_query}")
        return results[index]

    monkeypatch.setattr(subprocess, "run", run_git)
    monkeypatch.setattr(importlib.util, "find_spec", lambda _module: pytest.fail("must not resolve module"))
    monkeypatch.setattr(Path, "read_bytes", lambda _path: pytest.fail("must not read frozen artifact"))

    with pytest.raises(OSError, match=f"git unavailable {failed_query}"):
        exec(program, {})

    output = capsys.readouterr()
    assert output.out == output.err == ""


@pytest.mark.parametrize(
    ("tracked", "untracked", "ignored", "diagnostic"),
    [
        (
            " M runtime_cli.py\n",
            "new_module.py\n",
            "ignored_module.py\n",
            "Target runtime has tracked worktree changes; launch requires a clean commit.",
        ),
        ("", "new_module.py\n", "", "Target runtime has untracked or ignored Python code"),
        ("", "", "ignored_module.py\n", "Target runtime has untracked or ignored Python code"),
    ],
)
def test_runtime_identity_worktree_checks_precede_runtime_reads(
    monkeypatch, capsys, identity_probe, tracked, untracked, ignored, diagnostic
):
    program, results, _payload = identity_probe
    results[2].stdout, results[3].stdout, results[4].stdout = tracked, untracked, ignored
    monkeypatch.setattr(importlib.util, "find_spec", lambda _module: pytest.fail("must not resolve module"))
    monkeypatch.setattr(Path, "read_bytes", lambda _path: pytest.fail("must not read frozen artifact"))

    with pytest.raises(SystemExit) as error:
        exec(program, {})

    assert error.value.code == 2
    output = capsys.readouterr()
    assert output.out == ""
    assert output.err.startswith(diagnostic)


def test_runtime_identity_drift_precedes_artifact_reads(monkeypatch, capsys, identity_probe):
    program, _results, payload = identity_probe
    expected = {**payload, "runtime_commit": "b" * 40, "python": "another-python"}
    monkeypatch.setattr(sys, "argv", [*sys.argv[:2], json.dumps(expected), sys.argv[3]])
    monkeypatch.setattr(Path, "read_bytes", lambda _path: pytest.fail("must not read frozen artifact"))

    with pytest.raises(SystemExit) as error:
        exec(program, {})

    assert error.value.code == 2
    output = capsys.readouterr()
    assert output.out == ""
    assert output.err == "Target runtime identity changed before process start: python, runtime_commit\n"


@pytest.fixture
def runtime_repo(tmp_path):
    repo = tmp_path / "runtime"
    repo.mkdir()
    (repo / "runtime_cli.py").write_text("pass\n")
    (repo / ".gitignore").write_text("ignored_repo/\n")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "add", "runtime_cli.py", ".gitignore"], cwd=repo, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "-c",
            "core.hooksPath=/dev/null",
            "commit",
            "-qm",
            "Initialize runtime fixture",
        ],
        cwd=repo,
        check=True,
    )
    return repo


def test_runtime_identity_rejects_ignored_nested_repository_code(runtime_repo):
    nested = runtime_repo / "ignored_repo"
    nested.mkdir()
    (nested / "nested_module.py").write_text("pass\n")
    subprocess.run(["git", "init", "-q", str(nested)], check=True)

    result = subprocess.run(
        [sys.executable, "-c", python_programs.source("managed_scheduler.runtime_identity"), "runtime_cli"],
        cwd=runtime_repo,
        text=True,
        capture_output=True,
        timeout=10,
    )

    assert result.returncode == 2
    assert result.stdout == ""
    assert "Target runtime has untracked or ignored Python code" in result.stderr


def test_runtime_identity_allows_only_untracked_code_inside_tracked_submodule(runtime_repo):
    nested = runtime_repo / "submodule"
    nested.mkdir()
    (nested / "README.md").write_text("tracked submodule fixture\n")
    subprocess.run(["git", "init", "-q", str(nested)], check=True)
    subprocess.run(["git", "add", "README.md"], cwd=nested, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "-c",
            "core.hooksPath=/dev/null",
            "commit",
            "-qm",
            "Initialize submodule fixture",
        ],
        cwd=nested,
        check=True,
    )
    (runtime_repo / ".gitmodules").write_text('[submodule "submodule"]\n\tpath = submodule\n\turl = ./submodule\n')
    subprocess.run(["git", "add", "submodule", ".gitmodules"], cwd=runtime_repo, check=True, capture_output=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "-c",
            "core.hooksPath=/dev/null",
            "commit",
            "-qm",
            "Track submodule fixture",
        ],
        cwd=runtime_repo,
        check=True,
    )
    (nested / "untracked_module.py").write_text("pass\n")

    result = subprocess.run(
        [sys.executable, "-c", python_programs.source("managed_scheduler.runtime_identity"), "runtime_cli"],
        cwd=runtime_repo,
        text=True,
        capture_output=True,
        timeout=10,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["module_origin"] == str(runtime_repo / "runtime_cli.py")
