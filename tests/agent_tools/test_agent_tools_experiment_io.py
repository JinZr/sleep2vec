import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from agent_tools import experiment_io, manifests


@pytest.mark.parametrize(
    ("returncode", "expected"),
    [(0, True), (experiment_io.REMOTE_MISSING_RETURN_CODE, False)],
)
def test_remote_path_probe_distinguishes_existing_from_missing(monkeypatch, returncode, expected):
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, returncode, "", "")

    monkeypatch.setattr(experiment_io.subprocess, "run", fake_run)

    assert experiment_io.path_exists_at("/remote/path", remote="host") is expected
    command, kwargs = calls[0]
    assert "os.lstat" in command[-1]
    assert "[ -e" not in command[-1]
    assert kwargs["timeout"] == experiment_io.SSH_TIMEOUT_SECONDS


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
    [(0, "contents", "contents"), (experiment_io.REMOTE_MISSING_RETURN_CODE, "", "")],
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
    monkeypatch.setattr(
        experiment_io.subprocess,
        "run",
        lambda command, **_kwargs: subprocess.CompletedProcess(command, 0, b"a\r\nb\r\n", b""),
    )

    assert experiment_io.read_text_at("/remote/file", remote="host") == "a\r\nb\r\n"


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


def test_local_conditional_replace_requires_expected_digest(tmp_path: Path):
    path = tmp_path / "state.tsv"
    path.write_bytes(b"old\r\n")
    path.chmod(0o640)

    assert not experiment_io.conditional_atomic_replace_text_at(path, "new\n", "wrong")
    assert path.read_bytes() == b"old\r\n"
    assert experiment_io.conditional_atomic_replace_text_at(
        path,
        "new\n",
        hashlib.sha256(b"old\r\n").hexdigest(),
    )
    assert path.read_bytes() == b"new\n"
    assert path.stat().st_mode & 0o777 == 0o640


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
        dependency_path=dependency,
        expected_dependency_sha256=dependency_sha256,
    )
    assert path.read_text() == "old\n"


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

    assert experiment_io.conditional_atomic_replace_text_at(path, "first\n", None)
    assert path.read_text() == "first\n"
    assert not experiment_io.conditional_atomic_replace_text_at(path, "second\n", None)
    assert path.read_text() == "first\n"


def test_local_conditional_create_preserves_a_dangling_symlink(tmp_path: Path):
    path = tmp_path / "state.tsv"
    missing = tmp_path / "missing.tsv"
    path.symlink_to(missing)

    assert not experiment_io.conditional_atomic_replace_text_at(path, "new\n", None)
    assert path.is_symlink()
    assert path.readlink() == missing


def test_local_conditional_create_preserves_a_publish_time_competitor(tmp_path: Path, monkeypatch):
    path = tmp_path / "state.tsv"
    original_mkstemp = experiment_io.tempfile.mkstemp

    def create_competitor(*args, **kwargs):
        descriptor, temporary = original_mkstemp(*args, **kwargs)
        path.write_text("competitor\n")
        return descriptor, temporary

    monkeypatch.setattr(experiment_io.tempfile, "mkstemp", create_competitor)

    assert not experiment_io.conditional_atomic_replace_text_at(path, "new\n", None)
    assert path.read_text() == "competitor\n"


def test_local_conditional_create_avoids_trash_retained_hardlinks(tmp_path: Path, monkeypatch):
    path = tmp_path / "state.tsv"
    trash = tmp_path / ".trash"
    original_unlink = Path.unlink

    def retain_in_trash(candidate: Path, missing_ok: bool = False):
        if candidate.parent == tmp_path and candidate.name.startswith(f".{path.name}."):
            trash.mkdir(exist_ok=True)
            os.link(candidate, trash / candidate.name)
        original_unlink(candidate, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", retain_in_trash)

    assert experiment_io.conditional_atomic_replace_text_at(path, "first\n", None)
    assert path.stat().st_nlink == 1
    assert not trash.exists()
    experiment_io.validate_managed_output_paths(tmp_path, [path])


def test_remote_conditional_replace_reports_conflict_and_writes_exact_bytes(monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, experiment_io.REMOTE_CONFLICT_RETURN_CODE, b"", b"")

    monkeypatch.setattr(experiment_io.subprocess, "run", fake_run)

    assert not experiment_io.conditional_atomic_replace_text_at(
        "/remote/state.tsv",
        "new\r\n",
        hashlib.sha256(b"old\r\n").hexdigest(),
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
        return subprocess.CompletedProcess(command, 0, b"", b"")

    monkeypatch.setattr(experiment_io.subprocess, "run", fake_run)

    assert experiment_io.conditional_atomic_replace_text_at("/remote/state.tsv", "new\n", None, remote="host")
    command, kwargs = calls[0]
    assert "expect_missing = not expected" in command[-1]
    assert "os.path.lexists(path)" in command[-1]
    assert "renameat2" in command[-1]
    assert "renamex_np" in command[-1]
    assert "errno.EEXIST" in command[-1]
    assert "os.link(" not in command[-1]
    assert kwargs["input"] == b"new\n"


def test_remote_conditional_replace_locks_and_checks_dependency(monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, b"", b"")

    monkeypatch.setattr(experiment_io.subprocess, "run", fake_run)

    assert experiment_io.conditional_atomic_replace_text_at(
        "/remote/experiment.yaml",
        "experiment: {}\n",
        "a" * 64,
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
    assert 'dependency_path + ".lock"' in remote_command
    assert "hashlib.sha256(dependency_current).hexdigest()" in remote_command
    assert "hashlib.sha256(guard_current).hexdigest()" in remote_command


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
        (0, "", False),
        (0, "nonempty\n", True),
        (experiment_io.REMOTE_MISSING_RETURN_CODE, "", False),
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
