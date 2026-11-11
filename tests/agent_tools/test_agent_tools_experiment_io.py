from contextlib import contextmanager
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from agent_tools import experiment_io, manifests


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


def _run_managed_output_reader(root, paths, *, prefix=""):
    source = """import json, sys
from agent_tools import experiment_io
root, paths = json.loads(sys.argv[1])
print(json.dumps(experiment_io.read_managed_output_texts_at(root, paths)))
"""
    return subprocess.run(
        [sys.executable, "-c", prefix + source, json.dumps([str(root), [str(path) for path in paths]])],
        cwd=Path(__file__).resolve().parents[2],
        text=True,
        capture_output=True,
        timeout=5,
    )


def test_managed_output_reader_preserves_requested_keys_empty_text_and_missing(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "terminal.json").write_bytes("状态\r\nready\r\n".encode())
    empty = root / "empty.json"
    empty.write_text("")
    paths = [str(root) + "/./terminal.json", str(empty), str(root / "missing.json"), str(root / "absent" / "file")]

    result = _run_managed_output_reader(root, paths)

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == dict(zip(paths, ["状态\r\nready\r\n", "", None, None]))
    assert not (root / "absent").exists()


@pytest.mark.parametrize("missing", ["root", "parent", "leaf"])
def test_managed_output_reader_only_missing_opens_return_none(tmp_path, missing):
    root = tmp_path / "workspace"
    parent = root / "outputs"
    if missing != "root":
        root.mkdir()
    if missing == "leaf":
        parent.mkdir()
    paths = [parent / "terminal.json", parent / "allocation.json"]

    result = _run_managed_output_reader(root, paths)

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {str(path): None for path in paths}
    assert not paths[0].exists()


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
def test_managed_output_reader_rejects_unsafe_objects(tmp_path, topology):
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

    result = _run_managed_output_reader(root, [target])

    assert result.returncode != 0
    assert result.stdout == ""
    assert secret.read_text() == "outside\n"


@pytest.mark.parametrize("location", ["root", "target"])
@pytest.mark.parametrize("component", ["absent", "alias"])
def test_managed_output_reader_rejects_raw_parent_components(tmp_path, location, component):
    root = tmp_path / "workspace"
    root.mkdir()
    if component == "alias":
        (root / component).symlink_to(tmp_path, target_is_directory=True)
    if location == "root":
        root = root / component / ".."
        target = root / "terminal.json"
    else:
        target = root / component / ".." / "terminal.json"

    result = _run_managed_output_reader(root, [target])

    assert result.returncode != 0
    assert "must not contain '..'" in result.stderr
    assert result.stdout == ""


@pytest.mark.parametrize("alias", [False, True])
def test_managed_output_reader_rejects_duplicate_requested_targets(tmp_path, alias):
    target = str(tmp_path / "terminal.json")
    duplicate = str(tmp_path) + "/./terminal.json" if alias else target

    result = _run_managed_output_reader(tmp_path, [target, duplicate])

    assert result.returncode != 0
    assert "must be unique" in result.stderr
    assert result.stdout == ""


def test_managed_output_reader_rejects_invalid_utf8_without_partial_output(tmp_path):
    good = tmp_path / "allocation.json"
    bad = tmp_path / "terminal.json"
    good.write_text("{}")
    bad.write_bytes(b"\xff")

    result = _run_managed_output_reader(tmp_path, [good, bad])

    assert result.returncode != 0
    assert result.stdout == ""
    assert "utf-8" in result.stderr


def test_managed_output_reader_does_not_hide_permission_errors(tmp_path):
    if os.geteuid() == 0:
        pytest.skip("Root bypasses the file permission used by this test")
    target = tmp_path / "terminal.json"
    target.write_text("{}")
    target.chmod(0)
    try:
        result = _run_managed_output_reader(tmp_path, [target])
    finally:
        target.chmod(0o600)

    assert result.returncode != 0
    assert result.stdout == ""
    assert "PermissionError" in result.stderr


@pytest.mark.parametrize("error_number", [errno.EIO, errno.ENOENT])
def test_managed_output_reader_does_not_treat_descriptor_read_errors_as_missing(tmp_path, error_number):
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

    result = _run_managed_output_reader(tmp_path, [target], prefix=prefix)

    assert result.returncode != 0
    assert result.stdout == ""
    assert "injected descriptor read failure" in result.stderr


@pytest.mark.parametrize("replaced", ["parent", "leaf"])
def test_managed_output_reader_reads_opened_descriptor_after_path_replacement(tmp_path, replaced):
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

    result = _run_managed_output_reader(root, [target], prefix=prefix)

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {str(target): "original\n"}
    assert (parent if replaced == "parent" else target).is_symlink()
    assert (outside / target.name).read_text() == "outside\n"


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
    parent = root / "reports"
    moved_parent = root / "reports-original"
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
    parent = root / "reports"
    moved_parent = root / "reports-original"
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


@pytest.mark.parametrize("document", ["[]", '"plan"', "1", "null"])
def test_json_reader_rejects_a_document_that_is_not_an_object(tmp_path: Path, document: str):
    path = tmp_path / "plan.json"
    path.write_text(document + "\n")

    with pytest.raises(ValueError) as excinfo:
        manifests.read_json(path)

    assert str(excinfo.value) == f"JSON must be a mapping: {path}"


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
