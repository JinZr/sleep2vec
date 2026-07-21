import hashlib
import os
from pathlib import Path
import subprocess

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
