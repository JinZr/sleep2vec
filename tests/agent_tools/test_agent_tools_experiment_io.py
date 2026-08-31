from contextlib import contextmanager
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from agent_tools import experiment_io, manifests, python_programs


@pytest.fixture
def masked_ssh(monkeypatch):
    children = []

    def rewritten_exit(_host, command, **kwargs):
        kwargs.pop("check", None)
        argv = (
            [sys.executable, *shlex.split(command)[1:]] if command.startswith("python3 ") else ["bash", "-c", command]
        )
        child = subprocess.run(argv, capture_output=True, timeout=5, **kwargs)
        children.append(child)
        return subprocess.CompletedProcess(command, 0, child.stdout, child.stderr)

    monkeypatch.setattr(experiment_io.transport, "run_ssh", rewritten_exit)
    return children


@pytest.mark.parametrize(
    ("operation", "case", "child_returncode", "expected"),
    [
        ("path", "file", 0, True),
        ("path", "alias", 0, True),
        ("path", "missing", 44, False),
        ("path", "bad_parent", 1, None),
        ("directory", "empty_directory", 0, False),
        ("directory", "directory", 0, True),
        ("directory", "missing", 44, False),
        ("directory", "file", 1, None),
        ("read", "file", 0, "first\r\n第二行\r\n"),
        ("read", "empty_file", 0, ""),
        ("read", "missing", 44, ""),
        ("read", "empty_directory", 1, None),
        ("read", "invalid_utf8", 1, None),
        ("validate", "missing", 0, None),
        ("validate", "file", 0, None),
        ("validate", "alias", 2, None),
    ],
)
def test_remote_reads_require_actual_program_results_when_exit_is_masked(
    tmp_path, masked_ssh, operation, case, child_returncode, expected
):
    target = tmp_path / "target"
    if case in {"file", "bad_parent"}:
        target.write_bytes("first\r\n第二行\r\n".encode())
        if case == "bad_parent":
            target /= "child"
    elif case == "empty_file":
        target.write_bytes(b"")
    elif case == "invalid_utf8":
        target.write_bytes(b"\xff")
    elif case in {"empty_directory", "directory"}:
        target.mkdir()
        if case == "directory":
            (target / "entry").write_text("present")
    elif case == "alias":
        target.symlink_to(tmp_path / "absent")

    operations = {
        "path": lambda: experiment_io.path_exists_at(target, remote="host"),
        "directory": lambda: experiment_io.remote_dir_nonempty(target, "host"),
        "read": lambda: experiment_io.read_text_at(target, remote="host"),
        "validate": lambda: experiment_io.validate_managed_output_paths(tmp_path, [target], remote="host"),
    }
    if child_returncode in {1, 2}:
        with pytest.raises(RuntimeError, match="no valid result"):
            operations[operation]()
    else:
        assert operations[operation]() == expected
    assert len(masked_ssh) == 1
    assert masked_ssh[0].returncode == child_returncode


@pytest.mark.parametrize(
    ("operation", "returncode"),
    [
        ("path", 0),
        ("path", 44),
        ("directory", 0),
        ("directory", 44),
        ("read", 0),
        ("read", 44),
        ("validate", 0),
        ("cas", 0),
        ("cas", 45),
    ],
)
@pytest.mark.parametrize("stdout", ["", "tru", "[]", "true\ntrailing"])
def test_remote_operations_reject_missing_or_malformed_results(monkeypatch, operation, returncode, stdout):
    calls = []

    def incomplete_result(_host, command, **_kwargs):
        calls.append(command)
        if operation == "cas":
            return subprocess.CompletedProcess(command, returncode, stdout.encode(), b"")
        return subprocess.CompletedProcess(command, returncode, stdout, "")

    monkeypatch.setattr(experiment_io.transport, "run_ssh", incomplete_result)
    operations = {
        "path": lambda: experiment_io.path_exists_at("/remote/target", remote="host"),
        "directory": lambda: experiment_io.remote_dir_nonempty(Path("/remote/target"), "host"),
        "read": lambda: experiment_io.read_text_at("/remote/target", remote="host"),
        "validate": lambda: experiment_io.validate_managed_output_paths("/remote", ["/remote/target"], remote="host"),
        "cas": lambda: experiment_io.conditional_atomic_replace_text_at(
            "/remote/target", "new", None, managed_root="/remote", remote="host"
        ),
    }
    with pytest.raises(RuntimeError, match="no valid result"):
        operations[operation]()
    assert len(calls) == 1


@pytest.mark.parametrize("operation", ["path", "directory", "read", "validate"])
def test_remote_reads_never_accept_success_output_after_ssh_failure(monkeypatch, operation):
    stdout = '"contents"\n' if operation == "read" else "true\n"
    monkeypatch.setattr(
        experiment_io.transport,
        "run_ssh",
        lambda _host, command, **_kwargs: subprocess.CompletedProcess(command, 255, stdout, "connection lost"),
    )
    operations = {
        "path": lambda: experiment_io.path_exists_at("/remote/target", remote="host"),
        "directory": lambda: experiment_io.remote_dir_nonempty(Path("/remote/target"), "host"),
        "read": lambda: experiment_io.read_text_at("/remote/target", remote="host"),
        "validate": lambda: experiment_io.validate_managed_output_paths("/remote", ["/remote/target"], remote="host"),
    }
    with pytest.raises(RuntimeError, match="connection lost"):
        operations[operation]()


@pytest.mark.parametrize("mode", ["create", "replace", "conflict", "missing_root", "append"])
def test_remote_cas_and_append_use_actual_publication_results(tmp_path, masked_ssh, mode):
    root = tmp_path / "workspace"
    if mode != "missing_root":
        root.mkdir()
    target = root / "state.tsv"
    if mode in {"replace", "conflict"}:
        target.write_bytes(b"old\r\n")
    expected = hashlib.sha256(b"old\r\n").hexdigest() if mode == "replace" else None
    if mode == "append":
        experiment_io.append_event_at(root, "first", {}, remote="host")
        assert json.loads((root / "events.jsonl").read_text())["event_type"] == "first"
    elif mode == "missing_root":
        with pytest.raises(RuntimeError, match="outcome may be unknown"):
            experiment_io.conditional_atomic_replace_text_at(
                target, "new\r\n", expected, managed_root=root, remote="host"
            )
        assert not root.exists()
    else:
        committed = experiment_io.conditional_atomic_replace_text_at(
            target, "new\r\n", expected, managed_root=root, remote="host"
        )
        assert committed is (mode != "conflict")
        assert target.read_bytes() == (b"old\r\n" if mode == "conflict" else b"new\r\n")
    assert len(masked_ssh) == 1
    assert masked_ssh[0].returncode == (45 if mode == "conflict" else 1 if mode == "missing_root" else 0)


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_remote_create_conflict_requires_successful_temporary_cleanup(tmp_path, monkeypatch, masked_ssh, cleanup_fails):
    target = tmp_path / "state.tsv"
    original_source = python_programs.source
    source = original_source("experiment_io.conditional_atomic_replace_text")
    marker = "def rename_noreplace_at(parent_descriptor, source_name, target_name):\n"
    assert source.count(marker) == 1
    injection = f"""    competitor = os.open(
        target_name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644, dir_fd=parent_descriptor
    )
    with os.fdopen(competitor, "wb") as file_obj:
        file_obj.write(b"competitor\\n")
    if {cleanup_fails!r}:
        def fail_cleanup(*args, **kwargs):
            raise OSError(errno.EIO, "injected temporary cleanup failure")
        os.unlink = fail_cleanup
"""
    source = source.replace(marker, marker + injection)
    monkeypatch.setattr(
        python_programs,
        "source",
        lambda name: source if name == "experiment_io.conditional_atomic_replace_text" else original_source(name),
    )

    if cleanup_fails:
        with pytest.raises(RuntimeError, match="(?s)outcome may be unknown.*injected temporary cleanup failure"):
            experiment_io.conditional_atomic_replace_text_at(
                target, "new\n", None, managed_root=tmp_path, remote="host"
            )
    else:
        assert not experiment_io.conditional_atomic_replace_text_at(
            target, "new\n", None, managed_root=tmp_path, remote="host"
        )

    assert len(masked_ssh) == 1
    child = masked_ssh[0]
    assert child.returncode == (1 if cleanup_fails else 45)
    assert child.stdout == (b"" if cleanup_fails else b"false\n")
    assert target.read_bytes() == b"competitor\n"
    staged = [path for path in tmp_path.glob(".state.tsv.*") if not path.name.endswith(".lock")]
    assert len(staged) == int(cleanup_fails)
    if staged:
        assert staged[0].read_bytes() == b"new\n"


@pytest.mark.parametrize(
    "conflict",
    [
        "dependency_parent",
        "dependency_missing",
        "dependency_hash",
        "guard_parent",
        "guard_missing",
        "guard_hash",
        "target_missing",
        "target_hash",
    ],
)
def test_remote_cas_conflict_receipts_survive_successful_scope_cleanup(tmp_path, masked_ssh, conflict):
    target = tmp_path / "state.tsv"
    if conflict != "target_missing":
        target.write_bytes(b"old\n")
    expected = hashlib.sha256(b"old\n").hexdigest()
    kwargs = {}
    if conflict.startswith(("dependency_", "guard_")):
        kind, fault = conflict.split("_")
        other = tmp_path / kind
        if fault == "parent":
            other /= "missing.tsv"
        elif fault == "hash":
            other.write_bytes(b"changed\n")
        kwargs = {f"{kind}_path": other, f"expected_{kind}_sha256": expected}
    elif conflict == "target_hash":
        expected = hashlib.sha256(b"stale\n").hexdigest()

    assert not experiment_io.conditional_atomic_replace_text_at(
        target, "new\n", expected, managed_root=tmp_path, remote="host", **kwargs
    )

    assert len(masked_ssh) == 1
    assert masked_ssh[0].returncode == 45
    assert masked_ssh[0].stdout == b"false\n"
    assert target.exists() is (conflict != "target_missing")
    if target.exists():
        assert target.read_bytes() == b"old\n"


@pytest.mark.parametrize("operation", ["cas", "append", "write", "mkdir"])
@pytest.mark.parametrize("failure", ["lost", "truncated", "ssh255", "timeout"])
def test_remote_writes_do_not_retry_after_losing_actual_success(tmp_path, monkeypatch, masked_ssh, operation, failure):
    run_once = experiment_io.transport.run_ssh

    def lose_result(host, command, **kwargs):
        result = run_once(host, command, **kwargs)
        assert masked_ssh[-1].returncode == 0
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, 5)
        if failure == "ssh255":
            return subprocess.CompletedProcess(command, 255, result.stdout, result.stderr)
        result.stdout = result.stdout[:0] if failure == "lost" else result.stdout[:-2]
        return result

    monkeypatch.setattr(experiment_io.transport, "run_ssh", lose_result)
    target = tmp_path / "state.tsv"
    operations = {
        "cas": lambda: experiment_io.conditional_atomic_replace_text_at(
            target, "new\r\n", None, managed_root=tmp_path, remote="host"
        ),
        "append": lambda: experiment_io.append_event_at(tmp_path, "first", {}, remote="host"),
        "write": lambda: experiment_io.write_text_at(target, "new\r\n", remote="host"),
        "mkdir": lambda: experiment_io.mkdir_experiment_dirs(tmp_path, remote="host"),
    }
    with pytest.raises(subprocess.TimeoutExpired if failure == "timeout" else RuntimeError):
        operations[operation]()
    assert len(masked_ssh) == 1
    if operation == "append":
        assert len((tmp_path / "events.jsonl").read_text().splitlines()) == 1
    elif operation == "mkdir":
        assert (tmp_path / "reports").is_dir()
        assert (tmp_path / "wandb" / "history").is_dir()
    else:
        assert target.read_bytes() == b"new\r\n"


@pytest.mark.parametrize("operation", ["append", "write", "mkdir"])
def test_remote_writes_reject_actual_child_failure_hidden_by_zero(tmp_path, masked_ssh, operation):
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("keep")
    operations = {
        "append": lambda: experiment_io.append_event_at(blocker, "first", {}, remote="host"),
        "write": lambda: experiment_io.write_text_at(blocker / "state.tsv", "new", remote="host"),
        "mkdir": lambda: experiment_io.mkdir_experiment_dirs(blocker, remote="host"),
    }
    with pytest.raises(RuntimeError, match="outcome may be unknown"):
        operations[operation]()
    assert len(masked_ssh) == 1
    assert masked_ssh[0].returncode != 0
    assert blocker.read_text() == "keep"


def test_local_event_append_uses_managed_writer(tmp_path: Path, monkeypatch):
    calls = []

    def record(path, text, *, managed_root):
        calls.append((path, text, managed_root))

    monkeypatch.setattr(experiment_io, "append_managed_text_at", record)

    experiment_io.append_event_at(tmp_path, "step_registered", {"step_id": "train"})

    assert len(calls) == 1
    path, text, managed_root = calls[0]
    assert path == tmp_path / "events.jsonl"
    assert managed_root == tmp_path
    row = json.loads(text)
    assert isinstance(row.pop("time"), str)
    assert row == {
        "event_type": "step_registered",
        "step_id": "train",
    }


def test_remote_event_append_uses_managed_cas_program(monkeypatch):
    calls = []

    def fake_run(host, command, **kwargs):
        calls.append((host, command, kwargs))
        return subprocess.CompletedProcess(command, 0, b"true\n", b"")

    monkeypatch.setattr(experiment_io.transport, "run_ssh", fake_run)

    experiment_io.append_event_at(Path("/remote/workspace"), "step_registered", {"step_id": "train"}, remote="host")

    host, remote_command, kwargs = calls[0]
    assert host == "host"
    assert "append_mode" in remote_command
    assert 'f".{target_name}.cas.lock"' in remote_command
    assert "cat >>" not in remote_command
    assert json.loads(kwargs["input"])["event_type"] == "step_registered"


def test_embedded_remote_event_append_uses_shared_cas_lock(tmp_path: Path):
    path = tmp_path / "events.jsonl"
    command = [
        sys.executable,
        "-c",
        python_programs.source("experiment_io.conditional_atomic_replace_text"),
        str(tmp_path),
        str(path),
        "",
        "",
        "",
        "",
        "",
        "append",
    ]

    for event_type in ("first", "second"):
        result = subprocess.run(
            command,
            input=(json.dumps({"event_type": event_type}) + "\n").encode(),
            capture_output=True,
            timeout=5,
        )
        assert result.returncode == 0, result.stderr.decode()

    assert [json.loads(line)["event_type"] for line in path.read_text().splitlines()] == ["first", "second"]
    assert (tmp_path / ".events.jsonl.cas.lock").is_file()


def test_remote_event_append_does_not_retry_unknown_outcome(monkeypatch):
    calls = 0

    def fail_after_unknown_commit(_host, command, **_kwargs):
        nonlocal calls
        calls += 1
        return subprocess.CompletedProcess(command, 255, b"", b"connection lost")

    monkeypatch.setattr(experiment_io.transport, "run_ssh", fail_after_unknown_commit)

    with pytest.raises(RuntimeError, match="outcome may be unknown"):
        experiment_io.append_event_at(Path("/remote/workspace"), "step_registered", {}, remote="host")

    assert calls == 1


@pytest.mark.parametrize(
    ("returncode", "expected"),
    [(0, True), (experiment_io.REMOTE_MISSING_RETURN_CODE, False)],
)
def test_remote_path_probe_distinguishes_existing_from_missing(monkeypatch, returncode, expected):
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, returncode, "true\n" if expected else "false\n", "")

    monkeypatch.setattr(experiment_io.subprocess, "run", fake_run)

    assert experiment_io.path_exists_at("/remote/path", remote="host") is expected
    command, kwargs = calls[0]
    assert "os.lstat" in command[-1]
    assert "[ -e" not in command[-1]
    assert kwargs["timeout"] == experiment_io.SSH_TIMEOUT_SECONDS
    assert kwargs["text"] is True
    assert "check" not in kwargs
    assert "input" not in kwargs


@pytest.mark.parametrize("returncode", [1, 255])
def test_remote_path_probe_fails_closed_on_nonmissing_error(monkeypatch, returncode):
    monkeypatch.setattr(
        experiment_io.subprocess,
        "run",
        lambda command, **_kwargs: subprocess.CompletedProcess(command, returncode, "", "permission denied"),
    )

    with pytest.raises(RuntimeError, match="SSH path probe failed"):
        experiment_io.path_exists_at("/remote/path", remote="host")


@pytest.mark.parametrize(
    ("returncode", "stdout", "expected"),
    [(0, '"contents"\n', "contents"), (experiment_io.REMOTE_MISSING_RETURN_CODE, "null\n", "")],
)
def test_remote_read_distinguishes_contents_from_missing(monkeypatch, returncode, stdout, expected):
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, returncode, stdout, "")

    monkeypatch.setattr(experiment_io.subprocess, "run", fake_run)

    assert experiment_io.read_text_at("/remote/file", remote="host") == expected
    command, kwargs = calls[0]
    assert "os.lstat" in command[-1]
    assert "open(path" in command[-1]
    assert "[ -f" not in command[-1]
    assert kwargs["timeout"] == experiment_io.SSH_TIMEOUT_SECONDS


def test_remote_read_preserves_exact_line_endings(monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, json.dumps("a\r\nb\r\n").encode(), b"")

    monkeypatch.setattr(experiment_io.subprocess, "run", fake_run)

    assert experiment_io.read_text_at("/remote/file", remote="host") == "a\r\nb\r\n"
    assert "text" not in calls[0][1]


def test_local_managed_control_reads_reject_aliases_and_non_directory_entries(tmp_path):
    root = tmp_path / "workspace"
    steps = root / "steps"
    step = steps / "train"
    step.mkdir(parents=True)
    manifest = step / "step.yaml"
    manifest.write_text("step: train\n")

    assert experiment_io.list_managed_subdirectories_at(root, steps) == ["train"]
    assert experiment_io.read_managed_files_at(root, [manifest])[str(manifest)]["text"] == "step: train\n"

    manifest.rename(step / "step.real.yaml")
    manifest.symlink_to("step.real.yaml")
    with pytest.raises(ValueError, match="missing or aliased"):
        experiment_io.read_managed_files_at(root, [manifest])

    (steps / "unexpected.txt").write_text("unexpected\n")
    with pytest.raises(ValueError, match="non-directory entry"):
        experiment_io.list_managed_subdirectories_at(root, steps)


def test_managed_control_read_can_snapshot_invalid_utf8_for_repair(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    manifest = root / "state.tsv"
    manifest.write_bytes(b"\xff")

    with pytest.raises(ValueError, match="not valid UTF-8"):
        experiment_io.read_managed_files_at(root, [manifest])

    assert experiment_io.read_managed_files_at(root, [manifest], allow_invalid_utf8=True)[str(manifest)] == {
        "text": None,
        "sha256": hashlib.sha256(b"\xff").hexdigest(),
    }


def test_managed_control_reads_reject_dot_dot_before_resolution(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()

    with pytest.raises(ValueError, match="must not contain '..'"):
        experiment_io.read_managed_files_at(root, [root / "missing" / ".." / "file"])


def test_remote_managed_control_reads_use_one_read_only_probe(monkeypatch):
    calls = []
    payload = {"/remote/workspace/step.yaml": {"text": "step: train\n", "sha256": "a" * 64}}

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, json.dumps(payload), "")

    monkeypatch.setattr(experiment_io.subprocess, "run", fake_run)

    result = experiment_io.read_managed_files_at(
        "/remote/workspace",
        ["/remote/workspace/step.yaml"],
        remote="host",
    )

    assert result == payload
    assert len(calls) == 1
    assert 'open(target, "rb")' in calls[0][0][-1]
    assert "mkdir" not in calls[0][0][-1]


def test_remote_managed_control_read_requests_invalid_utf8_snapshot(monkeypatch):
    path = "/remote/workspace/state.tsv"
    payload = {path: {"text": None, "sha256": "a" * 64}}
    program_calls = []

    def fake_program_command(name, request):
        program_calls.append((name, json.loads(request)))
        return ["remote-program"]

    monkeypatch.setattr(experiment_io.transport, "remote_python_program_command", fake_program_command)
    monkeypatch.setattr(
        experiment_io.transport,
        "run_ssh",
        lambda _remote, command, **_kwargs: subprocess.CompletedProcess(command, 0, json.dumps(payload), ""),
    )

    assert (
        experiment_io.read_managed_files_at(
            "/remote/workspace",
            [path],
            remote="host",
            allow_invalid_utf8=True,
        )
        == payload
    )
    assert program_calls == [
        (
            "experiment_io.read_managed_files",
            ["/remote/workspace", [path], False, True],
        )
    ]


def test_local_managed_control_bundle_requires_exact_directory_entries(tmp_path):
    root = tmp_path / "workspace"
    bundle = root / "bundle"
    bundle.mkdir(parents=True)
    paths = [bundle / "questions.json", bundle / "questions.md", bundle / "plan.blocked.md"]
    for path in paths:
        path.write_text(f"{path.name}\n")

    experiment_io.read_managed_files_at(root, paths, exact_directory_entries=True)
    (bundle / "run.sh").write_text("partial launch\n")

    with pytest.raises(ValueError, match="directory entries differ"):
        experiment_io.read_managed_files_at(root, paths, exact_directory_entries=True)


def test_remote_exact_managed_control_bundle_uses_one_read_only_probe(monkeypatch):
    calls = []
    paths = [
        "/remote/workspace/plan/questions.json",
        "/remote/workspace/plan/questions.md",
        "/remote/workspace/plan/plan.blocked.md",
    ]
    payload = {path: {"text": "blocked\n", "sha256": "a" * 64} for path in paths}

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, json.dumps(payload), "")

    monkeypatch.setattr(experiment_io.subprocess, "run", fake_run)

    result = experiment_io.read_managed_files_at(
        "/remote/workspace",
        paths,
        remote="host",
        exact_directory_entries=True,
    )

    assert result == payload
    assert len(calls) == 1
    assert "os.listdir(parent)" in calls[0][0][-1]
    assert "mkdir" not in calls[0][0][-1]


def test_remote_managed_control_read_fails_closed_on_transport_error(monkeypatch):
    monkeypatch.setattr(
        experiment_io.subprocess,
        "run",
        lambda command, **_kwargs: subprocess.CompletedProcess(command, 255, "", "transport failed"),
    )

    with pytest.raises(RuntimeError, match="SSH managed-file read failed"):
        experiment_io.read_managed_files_at(
            "/remote/workspace",
            ["/remote/workspace/step.yaml"],
            remote="host",
        )


def _run_managed_output_reader(implementation, root, paths, *, prefix=""):
    if implementation == "local":
        source = """import json, sys
from agent_tools import experiment_io
root, paths = json.loads(sys.argv[1])
print(json.dumps(experiment_io.read_managed_output_texts_at(root, paths)))
"""
    else:
        source = python_programs.source("experiment_io.read_managed_output_texts")
    return subprocess.run(
        [sys.executable, "-c", prefix + source, json.dumps([str(root), [str(path) for path in paths]])],
        cwd=Path(__file__).resolve().parents[2],
        text=True,
        capture_output=True,
        timeout=5,
    )


@pytest.mark.parametrize("implementation", ["local", "embedded-remote"])
def test_managed_output_reader_preserves_requested_keys_empty_text_and_missing(tmp_path, implementation):
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "terminal.json").write_bytes("状态\r\nready\r\n".encode())
    empty = root / "empty.json"
    empty.write_text("")
    paths = [str(root) + "/./terminal.json", str(empty), str(root / "missing.json"), str(root / "absent" / "file")]

    result = _run_managed_output_reader(implementation, root, paths)

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == dict(zip(paths, ["状态\r\nready\r\n", "", None, None]))
    assert not (root / "absent").exists()


@pytest.mark.parametrize("implementation", ["local", "embedded-remote"])
@pytest.mark.parametrize("missing", ["root", "parent", "leaf"])
def test_managed_output_reader_only_missing_opens_return_none(tmp_path, implementation, missing):
    root = tmp_path / "workspace"
    parent = root / "outputs"
    if missing != "root":
        root.mkdir()
    if missing == "leaf":
        parent.mkdir()
    paths = [parent / "terminal.json", parent / "allocation.json"]

    result = _run_managed_output_reader(implementation, root, paths)

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {str(path): None for path in paths}
    assert not paths[0].exists()


@pytest.mark.parametrize("implementation", ["local", "embedded-remote"])
@pytest.mark.parametrize(
    "topology",
    [
        "ancestor_symlink",
        "root_symlink",
        "parent_symlink",
        "leaf_symlink",
        "dangling_symlink",
        "hardlink",
        "fifo",
        "root_file",
        "parent_file",
        "leaf_directory",
    ],
)
def test_managed_output_reader_rejects_unsafe_objects(tmp_path, implementation, topology):
    root = tmp_path / "workspace"
    parent = root / "outputs"
    outside = tmp_path / "outside"
    outside.mkdir()
    secret = outside / "terminal.json"
    secret.write_text("outside\n")
    if topology == "root_file":
        root.write_text("not a directory\n")
    else:
        root.mkdir()
        if topology == "parent_file":
            parent.write_text("not a directory\n")
        elif topology == "parent_symlink":
            parent.symlink_to(outside, target_is_directory=True)
        else:
            parent.mkdir()
    target = parent / "terminal.json"
    if topology == "ancestor_symlink":
        alias = tmp_path / "alias"
        alias.symlink_to(tmp_path, target_is_directory=True)
        target.write_text("inside\n")
        root = alias / root.name
        target = root / "outputs" / target.name
    elif topology == "root_symlink":
        moved = tmp_path / "workspace-original"
        target.write_text("inside\n")
        root.rename(moved)
        root.symlink_to(moved, target_is_directory=True)
    elif topology in {"leaf_symlink", "dangling_symlink"}:
        target.symlink_to(secret if topology == "leaf_symlink" else outside / "missing.json")
    elif topology == "hardlink":
        target.hardlink_to(secret)
    elif topology == "fifo":
        os.mkfifo(target)
    elif topology == "leaf_directory":
        target.mkdir()

    result = _run_managed_output_reader(implementation, root, [target])

    assert result.returncode != 0
    assert result.stdout == ""
    assert secret.read_text() == "outside\n"


@pytest.mark.parametrize("implementation", ["local", "embedded-remote"])
@pytest.mark.parametrize("location", ["root", "target"])
@pytest.mark.parametrize("component", ["absent", "alias"])
def test_managed_output_reader_rejects_raw_parent_components(tmp_path, implementation, location, component):
    root = tmp_path / "workspace"
    root.mkdir()
    if component == "alias":
        (root / component).symlink_to(tmp_path, target_is_directory=True)
    if location == "root":
        root = root / component / ".."
        target = root / "terminal.json"
    else:
        target = root / component / ".." / "terminal.json"

    result = _run_managed_output_reader(implementation, root, [target])

    assert result.returncode != 0
    assert "must not contain '..'" in result.stderr
    assert result.stdout == ""


@pytest.mark.parametrize("implementation", ["local", "embedded-remote"])
@pytest.mark.parametrize("alias", [False, True])
def test_managed_output_reader_rejects_duplicate_requested_targets(tmp_path, implementation, alias):
    target = str(tmp_path / "terminal.json")
    duplicate = str(tmp_path) + "/./terminal.json" if alias else target

    result = _run_managed_output_reader(implementation, tmp_path, [target, duplicate])

    assert result.returncode != 0
    assert "must be unique" in result.stderr
    assert result.stdout == ""


@pytest.mark.parametrize("implementation", ["local", "embedded-remote"])
def test_managed_output_reader_rejects_invalid_utf8_without_partial_output(tmp_path, implementation):
    good = tmp_path / "allocation.json"
    bad = tmp_path / "terminal.json"
    good.write_text("{}")
    bad.write_bytes(b"\xff")

    result = _run_managed_output_reader(implementation, tmp_path, [good, bad])

    assert result.returncode != 0
    assert result.stdout == ""
    assert "utf-8" in result.stderr


@pytest.mark.parametrize("implementation", ["local", "embedded-remote"])
def test_managed_output_reader_does_not_hide_permission_errors(tmp_path, implementation):
    if os.geteuid() == 0:
        pytest.skip("Root bypasses the file permission used by this test")
    target = tmp_path / "terminal.json"
    target.write_text("{}")
    target.chmod(0)
    try:
        result = _run_managed_output_reader(implementation, tmp_path, [target])
    finally:
        target.chmod(0o600)

    assert result.returncode != 0
    assert result.stdout == ""
    assert "PermissionError" in result.stderr


@pytest.mark.parametrize("implementation", ["local", "embedded-remote"])
@pytest.mark.parametrize("error_number", [errno.EIO, errno.ENOENT])
def test_managed_output_reader_does_not_treat_descriptor_read_errors_as_missing(tmp_path, implementation, error_number):
    target = tmp_path / "terminal.json"
    target.write_text("{}")
    prefix = f"""from contextlib import contextmanager
import os
from types import SimpleNamespace
_original_fdopen = os.fdopen
def _failed_read():
    raise OSError({error_number}, "injected descriptor read failure")
@contextmanager
def _failed_fdopen(*args, **kwargs):
    with _original_fdopen(*args, **kwargs):
        yield SimpleNamespace(read=_failed_read)
os.fdopen = _failed_fdopen
"""

    result = _run_managed_output_reader(implementation, tmp_path, [target], prefix=prefix)

    assert result.returncode != 0
    assert result.stdout == ""
    assert "injected descriptor read failure" in result.stderr


@pytest.mark.parametrize("implementation", ["local", "embedded-remote"])
@pytest.mark.parametrize("replaced", ["parent", "leaf"])
def test_managed_output_reader_reads_opened_descriptor_after_path_replacement(tmp_path, implementation, replaced):
    root = tmp_path / "workspace"
    parent = root / "outputs"
    parent.mkdir(parents=True)
    target = parent / "terminal.json"
    target.write_text("original\n")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / target.name).write_text("outside\n")
    prefix = f"""import json, os, sys
from pathlib import Path
_race_root, _race_paths = json.loads(sys.argv[1])
_race_target = Path(_race_paths[0])
_race_outside = Path(_race_root).parent / "outside"
_original_open = os.open
_replaced = False
def _replace_after_open(name, flags, *args, **kwargs):
    global _replaced
    descriptor = _original_open(name, flags, *args, **kwargs)
    watched = _race_target.parent.name if {replaced!r} == "parent" else _race_target.name
    if not _replaced and name == watched:
        _replaced = True
        path = _race_target.parent if {replaced!r} == "parent" else _race_target
        path.rename(path.with_name(path.name + "-original"))
        path.symlink_to(_race_outside if {replaced!r} == "parent" else _race_outside / path.name)
    return descriptor
os.open = _replace_after_open
"""

    result = _run_managed_output_reader(implementation, root, [target], prefix=prefix)

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {str(target): "original\n"}
    assert (parent if replaced == "parent" else target).is_symlink()
    assert (outside / target.name).read_text() == "outside\n"


def test_remote_managed_output_reader_uses_one_complete_read_operation(monkeypatch):
    paths = ["/remote/root/./terminal.json", "/remote/root/empty.json", "/remote/root/missing.json"]
    expected = dict(zip(paths, ["{}\r\n", "", None]))
    calls = []

    def complete_read(host, command, **kwargs):
        calls.append((host, shlex.split(command), kwargs))
        return subprocess.CompletedProcess(command, 0, json.dumps(expected), "")

    monkeypatch.setattr(experiment_io.transport, "run_ssh", complete_read)

    assert experiment_io.read_managed_output_texts_at("/remote/root", paths, remote="host") == expected
    assert len(calls) == 1
    host, command, kwargs = calls[0]
    assert host == "host"
    assert command[:2] == ["python3", "-c"]
    assert command[2] == python_programs.source("experiment_io.read_managed_output_texts")
    assert json.loads(command[3]) == ["/remote/root", paths]
    assert kwargs == {"text": True}


@pytest.mark.parametrize(
    "stdout",
    [
        "",
        "{",
        "null",
        "[]",
        "{}",
        '{"/remote/root/a": null}',
        '{"/remote/root/a": null, "/remote/root/b": "", "/remote/root/extra": null}',
        '{"/remote/root/a": null, "/remote/root/a": "", "/remote/root/b": ""}',
        '{"/remote/root/a": true, "/remote/root/b": ""}',
        '{"/remote/root/a": 1, "/remote/root/b": ""}',
        '{"/remote/root/a": [], "/remote/root/b": ""}',
        '{"/remote/root/a": {}, "/remote/root/b": ""}',
    ],
)
def test_remote_managed_output_reader_requires_positive_complete_output(monkeypatch, stdout):
    monkeypatch.setattr(
        experiment_io.transport,
        "run_ssh",
        lambda _host, command, **_kwargs: subprocess.CompletedProcess(command, 0, stdout, ""),
    )

    with pytest.raises(ValueError):
        experiment_io.read_managed_output_texts_at("/remote/root", ["/remote/root/a", "/remote/root/b"], remote="host")


@pytest.mark.parametrize("returncode", [1, 2, 255])
def test_remote_managed_output_reader_rejects_failed_transport_with_complete_stdout(monkeypatch, returncode):
    path = "/remote/root/a"
    monkeypatch.setattr(
        experiment_io.transport,
        "run_ssh",
        lambda _host, command, **_kwargs: subprocess.CompletedProcess(
            command, returncode, json.dumps({path: None}), "failed"
        ),
    )

    with pytest.raises(ValueError if returncode == 2 else RuntimeError, match="failed"):
        experiment_io.read_managed_output_texts_at("/remote/root", [path], remote="host")


def test_remote_managed_output_reader_rejects_real_child_failure_hidden_by_outer_zero(tmp_path, monkeypatch):
    good = tmp_path / "allocation.json"
    bad = tmp_path / "terminal.json"
    good.write_text("{}")
    bad.write_bytes(b"\xff")
    children = []

    def rewritten_exit(_host, command, **kwargs):
        child = subprocess.run([sys.executable, *shlex.split(command)[1:]], capture_output=True, timeout=5, **kwargs)
        children.append(child)
        return subprocess.CompletedProcess(command, 0, child.stdout, child.stderr)

    monkeypatch.setattr(experiment_io.transport, "run_ssh", rewritten_exit)

    with pytest.raises(ValueError):
        experiment_io.read_managed_output_texts_at(tmp_path, [good, bad], remote="host")
    assert len(children) == 1
    assert children[0].returncode != 0
    assert children[0].stdout == ""


def test_local_conditional_replace_requires_expected_digest(tmp_path: Path):
    path = tmp_path / "state.tsv"
    path.write_bytes(b"old\r\n")
    path.chmod(0o640)

    assert not experiment_io.conditional_atomic_replace_text_at(path, "new\n", "wrong", managed_root=tmp_path)
    assert path.read_bytes() == b"old\r\n"
    assert experiment_io.conditional_atomic_replace_text_at(
        path,
        "new\n",
        hashlib.sha256(b"old\r\n").hexdigest(),
        managed_root=tmp_path,
    )
    assert path.read_bytes() == b"new\n"
    assert path.stat().st_mode & 0o777 == 0o640


def test_conditional_replace_requires_managed_root(tmp_path: Path):
    with pytest.raises(TypeError, match="managed_root"):
        experiment_io.conditional_atomic_replace_text_at(tmp_path / "state.tsv", "new\n", None)


def test_local_conditional_replace_requires_current_dependency(tmp_path: Path):
    path = tmp_path / "state.tsv"
    dependency = tmp_path / "run_manifest.tsv"
    path.write_text("old\n")
    dependency.write_text("current\n")
    target_sha256 = hashlib.sha256(b"old\n").hexdigest()
    dependency_sha256 = hashlib.sha256(b"current\n").hexdigest()

    dependency.write_text("changed\n")

    assert not experiment_io.conditional_atomic_replace_text_at(
        path,
        "new\n",
        target_sha256,
        managed_root=tmp_path,
        dependency_path=dependency,
        expected_dependency_sha256=dependency_sha256,
    )
    assert path.read_text() == "old\n"


def test_local_conditional_replace_deduplicates_colliding_lock_names(tmp_path: Path, monkeypatch):
    path = tmp_path / "state.tsv"
    dependency = tmp_path / ".state.tsv.cas"
    path.write_text("old\n")
    dependency.write_text("dependency\n")
    locks = []
    original_lock = experiment_io._blocking_file_lock_at

    @contextmanager
    def record_lock(parent_descriptor, name):
        locks.append((os.fstat(parent_descriptor).st_ino, name))
        with original_lock(parent_descriptor, name):
            yield

    monkeypatch.setattr(experiment_io, "_blocking_file_lock_at", record_lock)

    assert experiment_io.conditional_atomic_replace_text_at(
        path,
        "new\n",
        hashlib.sha256(b"old\n").hexdigest(),
        managed_root=tmp_path,
        dependency_path=dependency,
        expected_dependency_sha256=hashlib.sha256(b"dependency\n").hexdigest(),
    )
    assert len(locks) == 1


def test_local_conditional_replace_requires_current_guard(tmp_path: Path):
    path = tmp_path / "state.tsv"
    dependency = tmp_path / "run_manifest.tsv"
    guard = tmp_path / "experiment.yaml"
    path.write_text("old\n")
    dependency.write_text("current\n")
    guard.write_text("active\n")

    guard.write_text("completed\n")

    assert not experiment_io.conditional_atomic_replace_text_at(
        path,
        "new\n",
        hashlib.sha256(b"old\n").hexdigest(),
        managed_root=tmp_path,
        dependency_path=dependency,
        expected_dependency_sha256=hashlib.sha256(b"current\n").hexdigest(),
        guard_path=guard,
        expected_guard_sha256=hashlib.sha256(b"active\n").hexdigest(),
    )
    assert path.read_text() == "old\n"


def test_blocking_file_lock_reopens_descriptor_after_transient_eio(tmp_path: Path, monkeypatch):
    lock_path = tmp_path / "state.lock"
    attempts = 0
    delays = []
    opened_lock_files = []
    real_flock = fcntl.flock
    real_open = Path.open

    def tracked_open(candidate, *args, **kwargs):
        file_obj = real_open(candidate, *args, **kwargs)
        if candidate == lock_path:
            opened_lock_files.append(file_obj)
        return file_obj

    def flaky_flock(file_descriptor, operation):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError(errno.EIO, os.strerror(errno.EIO))
        real_flock(file_descriptor, operation)

    monkeypatch.setattr(experiment_io, "fcntl", SimpleNamespace(LOCK_EX=fcntl.LOCK_EX, flock=flaky_flock))
    monkeypatch.setattr(experiment_io, "time", SimpleNamespace(sleep=delays.append))
    monkeypatch.setattr(Path, "open", tracked_open)

    with experiment_io.blocking_file_lock(lock_path):
        pass

    assert attempts == 2
    assert delays == [0.1]
    assert len(opened_lock_files) == 2
    assert all(file_obj.closed for file_obj in opened_lock_files)


def test_local_conditional_replace_retries_transient_flock_eio(tmp_path: Path, monkeypatch):
    path = tmp_path / "state.tsv"
    path.write_text("old\n")
    attempts = 0
    delays = []
    real_flock = fcntl.flock

    def flaky_flock(file_descriptor, operation):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError(errno.EIO, os.strerror(errno.EIO))
        real_flock(file_descriptor, operation)

    monkeypatch.setattr(experiment_io, "fcntl", SimpleNamespace(LOCK_EX=fcntl.LOCK_EX, flock=flaky_flock))
    monkeypatch.setattr(experiment_io, "time", SimpleNamespace(sleep=delays.append))

    assert experiment_io.conditional_atomic_replace_text_at(
        path,
        "new\n",
        hashlib.sha256(b"old\n").hexdigest(),
        managed_root=tmp_path,
    )
    assert attempts == 2
    assert delays == [0.1]
    assert path.read_bytes() == b"new\n"


def test_local_conditional_replace_fails_closed_after_persistent_flock_eio(tmp_path: Path, monkeypatch):
    path = tmp_path / "state.tsv"
    path.write_text("old\n")
    attempts = 0
    delays = []

    def failing_flock(_file_descriptor, _operation):
        nonlocal attempts
        attempts += 1
        raise OSError(errno.EIO, os.strerror(errno.EIO))

    monkeypatch.setattr(experiment_io, "fcntl", SimpleNamespace(LOCK_EX=fcntl.LOCK_EX, flock=failing_flock))
    monkeypatch.setattr(experiment_io, "time", SimpleNamespace(sleep=delays.append))

    with pytest.raises(OSError) as error:
        experiment_io.conditional_atomic_replace_text_at(
            path,
            "new\n",
            hashlib.sha256(b"old\n").hexdigest(),
            managed_root=tmp_path,
        )

    assert error.value.errno == errno.EIO
    assert attempts == 4
    assert delays == [0.1, 0.2, 0.4]
    assert path.read_bytes() == b"old\n"


def test_local_conditional_replace_does_not_retry_other_flock_errors(tmp_path: Path, monkeypatch):
    path = tmp_path / "state.tsv"
    path.write_text("old\n")
    attempts = 0

    def failing_flock(_file_descriptor, _operation):
        nonlocal attempts
        attempts += 1
        raise OSError(errno.EPERM, os.strerror(errno.EPERM))

    monkeypatch.setattr(experiment_io, "fcntl", SimpleNamespace(LOCK_EX=fcntl.LOCK_EX, flock=failing_flock))

    with pytest.raises(OSError) as error:
        experiment_io.conditional_atomic_replace_text_at(
            path,
            "new\n",
            hashlib.sha256(b"old\n").hexdigest(),
            managed_root=tmp_path,
        )

    assert error.value.errno == errno.EPERM
    assert attempts == 1
    assert path.read_bytes() == b"old\n"


def test_blocking_file_lock_does_not_retry_post_lock_eio(tmp_path: Path, monkeypatch):
    attempts = 0
    real_flock = fcntl.flock

    def tracked_flock(file_descriptor, operation):
        nonlocal attempts
        attempts += 1
        real_flock(file_descriptor, operation)

    monkeypatch.setattr(experiment_io, "fcntl", SimpleNamespace(LOCK_EX=fcntl.LOCK_EX, flock=tracked_flock))

    with pytest.raises(OSError) as error:
        with experiment_io.blocking_file_lock(tmp_path / "state.lock"):
            raise OSError(errno.EIO, os.strerror(errno.EIO))

    assert error.value.errno == errno.EIO
    assert attempts == 1


def test_local_conditional_create_never_replaces_an_existing_file(tmp_path: Path):
    path = tmp_path / "state.tsv"

    assert experiment_io.conditional_atomic_replace_text_at(path, "first\n", None, managed_root=tmp_path)
    assert path.read_text() == "first\n"
    assert not experiment_io.conditional_atomic_replace_text_at(path, "second\n", None, managed_root=tmp_path)
    assert path.read_text() == "first\n"


def test_local_conditional_create_preserves_a_dangling_symlink(tmp_path: Path):
    path = tmp_path / "state.tsv"
    missing = tmp_path / "missing.tsv"
    path.symlink_to(missing)

    assert not experiment_io.conditional_atomic_replace_text_at(path, "new\n", None, managed_root=tmp_path)
    assert path.is_symlink()
    assert path.readlink() == missing


@pytest.mark.parametrize("alias_kind", ["symlink", "hardlink"])
def test_local_conditional_replace_rejects_leaf_alias(tmp_path: Path, alias_kind: str):
    path = tmp_path / "state.tsv"
    outside = tmp_path / "outside.tsv"
    outside.write_text("old\n")
    if alias_kind == "symlink":
        path.symlink_to(outside)
    else:
        path.hardlink_to(outside)

    with pytest.raises((OSError, ValueError)):
        experiment_io.conditional_atomic_replace_text_at(
            path,
            "new\n",
            hashlib.sha256(b"old\n").hexdigest(),
            managed_root=tmp_path,
        )

    assert outside.read_text() == "old\n"


@pytest.mark.parametrize("fifo_role", ["target", "dependency", "guard"])
def test_local_conditional_replace_rejects_fifo_without_blocking(tmp_path: Path, fifo_role: str):
    target = tmp_path / "state.tsv"
    dependency = tmp_path / "run_manifest.tsv"
    guard = tmp_path / "experiment.yaml"
    target.write_text("old\n")
    dependency.write_text("dependency\n")
    guard.write_text("guard\n")
    fifo = {"target": target, "dependency": dependency, "guard": guard}[fifo_role]
    fifo.unlink()
    os.mkfifo(fifo)

    with pytest.raises(ValueError, match="missing or aliased"):
        experiment_io.conditional_atomic_replace_text_at(
            target,
            "new\n",
            hashlib.sha256(b"old\n").hexdigest(),
            managed_root=tmp_path,
            dependency_path=dependency,
            expected_dependency_sha256=hashlib.sha256(b"dependency\n").hexdigest(),
            guard_path=guard,
            expected_guard_sha256=hashlib.sha256(b"guard\n").hexdigest(),
        )

    assert fifo.is_fifo()
    if fifo_role != "target":
        assert target.read_bytes() == b"old\n"


def test_local_conditional_create_preserves_a_publish_time_competitor(tmp_path: Path, monkeypatch):
    path = tmp_path / "state.tsv"
    original_open_temporary = experiment_io._open_temporary_at

    def create_competitor(parent_descriptor, target_name):
        descriptor, temporary = original_open_temporary(parent_descriptor, target_name)
        path.write_text("competitor\n")
        return descriptor, temporary

    monkeypatch.setattr(experiment_io, "_open_temporary_at", create_competitor)

    assert not experiment_io.conditional_atomic_replace_text_at(path, "new\n", None, managed_root=tmp_path)
    assert path.read_text() == "competitor\n"


def test_local_conditional_create_does_not_leave_a_temporary_hardlink(tmp_path: Path):
    path = tmp_path / "state.tsv"

    assert experiment_io.conditional_atomic_replace_text_at(path, "first\n", None, managed_root=tmp_path)
    assert path.stat().st_nlink == 1
    assert not [entry for entry in tmp_path.glob(f".{path.name}.*") if entry.name != f".{path.name}.cas.lock"]
    experiment_io.validate_managed_output_paths(tmp_path, [path])


def test_local_conditional_replace_rejects_public_parent_alias_drift(tmp_path: Path, monkeypatch):
    root = tmp_path / "workspace"
    parent = root / "adaptive"
    moved_parent = root / "adaptive-original"
    outside = tmp_path / "outside"
    target = parent / "run_registry.tsv"
    outside_target = outside / target.name
    parent.mkdir(parents=True)
    outside.mkdir()
    target.write_text("old\n")
    outside_target.write_text("old\n")
    original_open_temporary = experiment_io._open_temporary_at

    def swap_parent_after_open(parent_descriptor, target_name):
        parent.rename(moved_parent)
        parent.symlink_to(outside, target_is_directory=True)
        return original_open_temporary(parent_descriptor, target_name)

    monkeypatch.setattr(experiment_io, "_open_temporary_at", swap_parent_after_open)

    with pytest.raises(ValueError, match="path changed during publication"):
        experiment_io.conditional_atomic_replace_text_at(
            target,
            "new\n",
            hashlib.sha256(b"old\n").hexdigest(),
            managed_root=root,
        )

    assert (moved_parent / target.name).read_text() == "old\n"
    assert outside_target.read_text() == "old\n"


@pytest.mark.parametrize("target_exists", [False, True])
def test_local_conditional_write_reports_unknown_when_public_parent_moves_during_rename(
    tmp_path: Path,
    monkeypatch,
    target_exists: bool,
):
    root = tmp_path / "workspace"
    parent = root / "adaptive"
    moved_parent = root / "adaptive-original"
    outside = tmp_path / "outside"
    target = parent / "run_registry.tsv"
    outside_target = outside / target.name
    parent.mkdir(parents=True)
    outside.mkdir()
    outside_target.write_text("outside\n")
    expected = None
    if target_exists:
        target.write_text("old\n")
        expected = hashlib.sha256(b"old\n").hexdigest()

    def move_public_parent():
        parent.rename(moved_parent)
        parent.symlink_to(outside, target_is_directory=True)

    rename_owner = experiment_io.os if target_exists else experiment_io
    rename_name = "replace" if target_exists else "_rename_noreplace_at"
    original_rename = getattr(rename_owner, rename_name)

    def rename_after_parent_moves(*args, **kwargs):
        move_public_parent()
        return original_rename(*args, **kwargs)

    monkeypatch.setattr(rename_owner, rename_name, rename_after_parent_moves)

    with pytest.raises(RuntimeError, match="publication outcome is unknown"):
        experiment_io.conditional_atomic_replace_text_at(
            target,
            "new\n",
            expected,
            managed_root=root,
        )

    assert outside_target.read_text() == "outside\n"
    assert (moved_parent / target.name).read_text() == "new\n"


@pytest.mark.parametrize("mode", ["create", "replace", "append"])
def test_embedded_conditional_write_reports_unknown_when_public_parent_moves_during_rename(
    tmp_path: Path,
    mode: str,
):
    root = tmp_path / "workspace"
    parent = root / "adaptive"
    moved_parent = root / "adaptive-moved"
    outside = tmp_path / "outside"
    target = parent / "run_registry.tsv"
    outside_target = outside / target.name
    parent.mkdir(parents=True)
    outside.mkdir()
    outside_target.write_text("outside\n")
    expected = ""
    rename_name = "rename_noreplace_at"
    extra_args = []
    if mode != "create":
        target.write_text("old\n")
        rename_name = "os.replace"
    if mode == "replace":
        expected = hashlib.sha256(b"old\n").hexdigest()
    elif mode == "append":
        extra_args = ["append"]

    marker = (
        "                    # Supported writers share this lock; this binds the namespace "
        "for the immediately following rename."
    )
    source = python_programs.source("experiment_io.conditional_atomic_replace_text")
    assert source.count(marker) == 1
    injection = f"""                    original_rename = {rename_name}

                    def drift_during_rename(*args, **kwargs):
                        path.parent.rename(path.parent.with_name(path.parent.name + "-moved"))
                        path.parent.symlink_to(managed_root.parent / "outside", target_is_directory=True)
                        return original_rename(*args, **kwargs)

                    {rename_name} = drift_during_rename
"""
    source = source.replace(marker, injection + marker)

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            source,
            str(root),
            str(target),
            expected,
            "",
            "",
            "",
            "",
        ]
        + extra_args,
        input=b"new\n",
        capture_output=True,
        timeout=5,
    )

    assert result.returncode != 0
    assert b"publication outcome is unknown" in result.stderr
    assert outside_target.read_text() == "outside\n"
    assert (moved_parent / target.name).read_text() == ("old\nnew\n" if mode == "append" else "new\n")


def test_remote_conditional_replace_reports_conflict_and_writes_exact_bytes(monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, experiment_io.REMOTE_CONFLICT_RETURN_CODE, b"false\n", b"")

    monkeypatch.setattr(experiment_io.subprocess, "run", fake_run)

    assert not experiment_io.conditional_atomic_replace_text_at(
        "/remote/state.tsv",
        "new\r\n",
        hashlib.sha256(b"old\r\n").hexdigest(),
        managed_root="/remote",
        remote="host",
    )
    command, kwargs = calls[0]
    assert "fcntl.flock" in command[-1]
    assert "os.fchmod" in command[-1]
    assert "os.replace" in command[-1]
    assert kwargs["input"] == b"new\r\n"


def test_remote_conditional_create_uses_atomic_no_replace_without_hardlink(monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, b"true\n", b"")

    monkeypatch.setattr(experiment_io.subprocess, "run", fake_run)

    assert experiment_io.conditional_atomic_replace_text_at(
        "/remote/state.tsv",
        "new\n",
        None,
        managed_root="/remote",
        remote="host",
    )
    command, kwargs = calls[0]
    assert "expect_missing = not expected" in command[-1]
    assert "dir_fd=target_parent" in command[-1]
    assert "renameat2" in command[-1]
    assert "renameatx_np" in command[-1]
    assert "renamex_np" not in command[-1]
    assert "-100" not in command[-1]
    assert "errno.EEXIST" in command[-1]
    assert "os.link(" not in command[-1]
    assert kwargs["input"] == b"new\n"


def test_remote_conditional_replace_locks_and_checks_dependency(monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, b"true\n", b"")

    monkeypatch.setattr(experiment_io.subprocess, "run", fake_run)

    assert experiment_io.conditional_atomic_replace_text_at(
        "/remote/experiment.yaml",
        "experiment: {}\n",
        "a" * 64,
        managed_root="/remote",
        remote="host",
        dependency_path="/remote/run_manifest.tsv",
        expected_dependency_sha256="b" * 64,
        guard_path="/remote/reports/final.md",
        expected_guard_sha256="c" * 64,
    )
    command, _kwargs = calls[0]
    remote_command = command[-1]
    assert "/remote/experiment.yaml" in remote_command
    assert "/remote/run_manifest.tsv" in remote_command
    assert "/remote/reports/final.md" in remote_command
    assert 'dependency_name + ".lock"' in remote_command
    assert "hashlib.sha256(dependency_current).hexdigest()" in remote_command
    assert "hashlib.sha256(guard_current).hexdigest()" in remote_command


def test_remote_conditional_replace_deduplicates_target_dependency_lock(tmp_path: Path):
    path = tmp_path / "state.tsv"
    path.write_text("old\n")
    digest = hashlib.sha256(b"old\n").hexdigest()

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            python_programs.source("experiment_io.conditional_atomic_replace_text"),
            str(tmp_path),
            str(path),
            digest,
            str(path),
            digest,
            "",
            "",
        ],
        input=b"new\n",
        capture_output=True,
        timeout=5,
    )

    assert result.returncode == 0, result.stderr.decode()
    assert path.read_bytes() == b"new\n"
    assert (tmp_path / "state.tsv.lock").is_file()
    assert (tmp_path / ".state.tsv.cas.lock").is_file()


def test_local_and_embedded_remote_conditional_replace_share_a_cas_lock(tmp_path: Path, monkeypatch):
    path = tmp_path / "state.tsv"
    path.write_text("old\n")
    digest = hashlib.sha256(b"old\n").hexdigest()
    local_ready = threading.Event()
    release_local = threading.Event()
    original_open_temporary = experiment_io._open_temporary_at

    def pause_local_writer(parent_descriptor, target_name):
        descriptor, temporary = original_open_temporary(parent_descriptor, target_name)
        local_ready.set()
        assert release_local.wait(timeout=5)
        return descriptor, temporary

    monkeypatch.setattr(experiment_io, "_open_temporary_at", pause_local_writer)
    local_result = []
    local_thread = threading.Thread(
        target=lambda: local_result.append(
            experiment_io.conditional_atomic_replace_text_at(path, "local\n", digest, managed_root=tmp_path)
        )
    )
    local_thread.start()
    assert local_ready.wait(timeout=5)

    remote = subprocess.Popen(
        [
            sys.executable,
            "-c",
            python_programs.source("experiment_io.conditional_atomic_replace_text"),
            str(tmp_path),
            str(path),
            digest,
            "",
            "",
            "",
            "",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert remote.stdin is not None
    remote.stdin.write(b"remote\n")
    remote.stdin.close()
    remote_lock_held = False
    deadline = time.monotonic() + 5
    while remote.poll() is None and time.monotonic() < deadline:
        try:
            lock_descriptor = os.open(tmp_path / "state.tsv.lock", os.O_RDWR)
        except FileNotFoundError:
            time.sleep(0.01)
            continue
        try:
            fcntl.flock(lock_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            remote_lock_held = True
            break
        else:
            fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
        finally:
            os.close(lock_descriptor)
        time.sleep(0.01)
    release_local.set()

    local_thread.join(timeout=5)
    assert not local_thread.is_alive()
    remote.wait(timeout=5)
    assert remote_lock_held, "Embedded remote writer bypassed the local CAS lock"
    assert local_result == [True]
    assert remote.returncode == experiment_io.REMOTE_CONFLICT_RETURN_CODE
    assert path.read_bytes() == b"local\n"


def test_remote_conditional_replace_rejects_fifo_without_blocking(tmp_path: Path):
    path = tmp_path / "state.tsv"
    os.mkfifo(path)

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            python_programs.source("experiment_io.conditional_atomic_replace_text"),
            str(tmp_path),
            str(path),
            hashlib.sha256(b"old\n").hexdigest(),
            "",
            "",
            "",
            "",
        ],
        input=b"new\n",
        capture_output=True,
        timeout=5,
    )

    assert result.returncode != 0
    assert b"missing or aliased" in result.stderr
    assert path.is_fifo()


@pytest.mark.parametrize("returncode", [1, 255])
def test_remote_read_fails_closed_on_nonmissing_error(monkeypatch, returncode):
    monkeypatch.setattr(
        experiment_io.subprocess,
        "run",
        lambda command, **_kwargs: subprocess.CompletedProcess(command, returncode, "partial", "read failed"),
    )

    with pytest.raises(RuntimeError, match="SSH read failed"):
        experiment_io.read_text_at("/remote/file", remote="host")


@pytest.mark.parametrize(
    ("returncode", "stdout", "expected"),
    [
        (0, "false\n", False),
        (0, "true\n", True),
        (experiment_io.REMOTE_MISSING_RETURN_CODE, "false\n", False),
    ],
)
def test_remote_directory_probe_distinguishes_empty_from_missing(monkeypatch, returncode, stdout, expected):
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, returncode, stdout, "")

    monkeypatch.setattr(experiment_io.subprocess, "run", fake_run)

    assert experiment_io.remote_dir_nonempty(Path("/remote/root"), "host") is expected
    command, kwargs = calls[0]
    assert "os.lstat" in command[-1]
    assert "os.listdir" in command[-1]
    assert "find " not in command[-1]
    assert kwargs["timeout"] == experiment_io.SSH_TIMEOUT_SECONDS


@pytest.mark.parametrize("returncode", [1, 255])
def test_remote_directory_probe_fails_closed_on_nonmissing_error(monkeypatch, returncode):
    monkeypatch.setattr(
        experiment_io.subprocess,
        "run",
        lambda command, **_kwargs: subprocess.CompletedProcess(command, returncode, "partial", "not a directory"),
    )

    with pytest.raises(RuntimeError, match="SSH directory probe failed"):
        experiment_io.remote_dir_nonempty(Path("/remote/root"), "host")


@pytest.mark.parametrize(
    "operation",
    [
        lambda: experiment_io.path_exists_at("/remote/path", remote="host"),
        lambda: experiment_io.read_text_at("/remote/file", remote="host"),
        lambda: experiment_io.remote_dir_nonempty(Path("/remote/root"), "host"),
    ],
)
def test_remote_authoritative_reads_propagate_timeout(monkeypatch, operation):
    def timeout(command, **_kwargs):
        raise subprocess.TimeoutExpired(command, experiment_io.SSH_TIMEOUT_SECONDS)

    monkeypatch.setattr(experiment_io.subprocess, "run", timeout)

    with pytest.raises(subprocess.TimeoutExpired):
        operation()


@pytest.mark.parametrize("header", ["trial_id\n", "run_id\n", "step_id\tstep_id\trun_id\n"])
def test_managed_table_reader_rejects_removed_or_malformed_header_only_tables(tmp_path: Path, header: str):
    path = tmp_path / "run_status.tsv"
    path.write_text(header)

    with pytest.raises(ValueError):
        experiment_io.read_rows_at(path, require_managed_identity=True)


def test_managed_table_reader_accepts_current_header_only_table(tmp_path: Path):
    path = tmp_path / "run_status.tsv"
    path.write_text("step_id\trun_id\n")

    assert experiment_io.read_rows_at(path, require_managed_identity=True) == []


@pytest.mark.parametrize(
    "reader",
    [
        pytest.param(lambda path: manifests.read_rows(path, require_managed_identity=True), id="read_rows"),
        pytest.param(lambda path: experiment_io.read_rows_at(path, require_managed_identity=True), id="read_rows_at"),
    ],
)
@pytest.mark.parametrize(
    ("header", "message"),
    [
        pytest.param(
            "trial_id\tstep_id\trun_id\n",
            "Historical managed table fields are read-only; Historical trial_id fields are unsupported: {path}",
            id="trial_id",
        ),
        pytest.param(
            "step_id\trun_id\tparam.lr\n",
            "Historical parameter fields are read-only: {path}",
            id="param_prefix",
        ),
        pytest.param(
            "experiment_id\n",
            "Managed table header must define step_id and run_id; missing step_id, run_id: {path}",
            id="missing_identity",
        ),
    ],
)
def test_managed_header_contract_messages_are_identical_across_readers(
    tmp_path: Path, reader, header: str, message: str
):
    path = tmp_path / "run_status.tsv"
    path.write_text(header)

    with pytest.raises(ValueError) as excinfo:
        reader(path)

    assert str(excinfo.value) == message.format(path=path)


@pytest.mark.parametrize(
    "contents",
    [
        "experiment_id\texperiment_id\texperiment_root\nunit\tunit\t/root\n",
        "experiment_id\texperiment_root\nunit\t/root\textra\n",
    ],
)
def test_strict_table_reader_rejects_duplicate_header_and_non_rectangular_rows(tmp_path: Path, contents: str):
    path = tmp_path / "experiment_manifest.tsv"
    path.write_text(contents)

    with pytest.raises(ValueError):
        experiment_io.read_rows_at(path, strict=True)


def test_strict_table_reader_does_not_require_managed_run_identity(tmp_path: Path):
    path = tmp_path / "experiment_manifest.tsv"
    path.write_text("experiment_id\texperiment_root\nunit\t/root\n")

    assert experiment_io.read_rows_at(path, strict=True) == [{"experiment_id": "unit", "experiment_root": "/root"}]


@pytest.mark.parametrize(
    "target_kind",
    ["symlink", "dangling_symlink", "hardlink", "directory", "fifo", "ancestor_symlink"],
)
def test_managed_output_preflight_rejects_unsafe_topology(tmp_path: Path, target_kind: str):
    canonical = tmp_path / "run_manifest.tsv"
    canonical.write_text("step_id\trun_id\n")
    output = tmp_path / "reports" / "final.md"
    if target_kind == "ancestor_symlink":
        outside = tmp_path / "outside"
        outside.mkdir()
        output.parent.symlink_to(outside, target_is_directory=True)
    else:
        output.parent.mkdir()
    if target_kind == "symlink":
        output.symlink_to(canonical)
    elif target_kind == "dangling_symlink":
        output.symlink_to(tmp_path / "missing.tsv")
    elif target_kind == "hardlink":
        os.link(canonical, output)
    elif target_kind == "directory":
        output.mkdir()
    elif target_kind == "fifo":
        os.mkfifo(output)

    with pytest.raises(ValueError, match="independent regular files"):
        experiment_io.validate_managed_output_paths(tmp_path, [output])


def test_managed_output_preflight_rejects_symlink_above_workspace_root(tmp_path: Path):
    outside = tmp_path / "outside"
    outside.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(outside, target_is_directory=True)
    root = alias / "workspace"

    with pytest.raises(ValueError, match="independent regular files"):
        experiment_io.validate_managed_output_paths(root, [root / "result.tsv"])


def test_remote_managed_output_preflight_fails_closed(monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 2, "", "aliased output")

    monkeypatch.setattr(experiment_io.subprocess, "run", fake_run)

    with pytest.raises(ValueError, match="aliased output"):
        experiment_io.validate_managed_output_paths("/remote/root", ["/remote/root/reports/final.md"], remote="host")

    assert calls[0][1]["timeout"] == experiment_io.SSH_TIMEOUT_SECONDS


@pytest.mark.parametrize("returncode", [1, 255])
def test_remote_managed_output_preflight_propagates_transport_failure(monkeypatch, returncode):
    monkeypatch.setattr(
        experiment_io.subprocess,
        "run",
        lambda command, **_kwargs: subprocess.CompletedProcess(command, returncode, "", "transport failed"),
    )

    with pytest.raises(RuntimeError, match="SSH output path validation failed"):
        experiment_io.validate_managed_output_paths("/remote/root", ["/remote/root/final.md"], remote="host")


def test_remote_managed_output_preflight_propagates_timeout(monkeypatch):
    def timeout(command, **_kwargs):
        raise subprocess.TimeoutExpired(command, experiment_io.SSH_TIMEOUT_SECONDS)

    monkeypatch.setattr(experiment_io.subprocess, "run", timeout)

    with pytest.raises(subprocess.TimeoutExpired):
        experiment_io.validate_managed_output_paths("/remote/root", ["/remote/root/final.md"], remote="host")
