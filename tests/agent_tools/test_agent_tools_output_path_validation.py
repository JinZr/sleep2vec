import errno
import json
import os
from pathlib import Path
import stat
import sys

import pytest

from agent_tools import experiment_io, python_programs


@pytest.fixture(params=["local", "embedded-remote"])
def validate_outputs(request, monkeypatch):
    source = compile(
        python_programs.source("experiment_io.validate_managed_output_paths"),
        "experiment_io.validate_managed_output_paths",
        "exec",
    )

    def validate(root, paths):
        if request.param == "local":
            experiment_io.validate_managed_output_paths(root, paths)
        else:
            with monkeypatch.context() as patch:
                patch.setattr(sys, "argv", ["validate", json.dumps([str(root), *map(str, paths)])])
                exec(source, {"__name__": "__main__"})

    return validate


@pytest.fixture
def descriptors(monkeypatch):
    opened = set()
    leaf_opens = []
    real_open, real_dup, real_close = os.open, os.dup, os.close

    def open_file(path, flags, *args, **kwargs):
        descriptor = real_open(path, flags, *args, **kwargs)
        opened.add(descriptor)
        if Path(path).name in {"first.tsv", "second.tsv"}:
            leaf_opens.append((Path(path).name, flags, descriptor))
        return descriptor

    def duplicate(descriptor):
        copied = real_dup(descriptor)
        opened.add(copied)
        return copied

    def close(descriptor):
        real_close(descriptor)
        opened.discard(descriptor)

    monkeypatch.setattr(os, "open", open_file)
    monkeypatch.setattr(os, "dup", duplicate)
    monkeypatch.setattr(os, "close", close)
    yield leaf_opens
    assert not opened, "Collision validation leaked descriptors"


def _collide_once(monkeypatch, first, second, change=lambda: None):
    original = first.stat()
    real_lstat = os.lstat
    observed = False

    def lstat(path, *args, **kwargs):
        nonlocal observed
        info = real_lstat(path, *args, **kwargs)
        if Path(path) == second and not observed:
            observed = True
            change()
            values = list(info)
            # Model a freed inode being reused, without relying on the filesystem allocator.
            values[1:3] = [original.st_ino, original.st_dev]
            return os.stat_result(values)
        return info

    monkeypatch.setattr(os, "lstat", lstat)


def test_recycled_inode_uses_fresh_opened_objects(tmp_path, monkeypatch, validate_outputs, descriptors):
    first, second, replacement = (tmp_path / name for name in ("first.tsv", "second.tsv", "replacement"))
    first.write_text("old\n")
    second.write_text("independent\n")
    replacement.write_text("new\n")
    _collide_once(monkeypatch, first, second, lambda: os.replace(replacement, first))
    monkeypatch.setattr(os, "read", lambda *_args: pytest.fail("Metadata validation read file contents"))
    monkeypatch.setattr(os, "fdopen", lambda *_args, **_kwargs: pytest.fail("Metadata validation opened file contents"))

    validate_outputs(tmp_path, [first, second])

    assert {name for name, _flags, _descriptor in descriptors} == {first.name, second.name}
    assert all(flags & os.O_NOFOLLOW and flags & os.O_NONBLOCK for _name, flags, _descriptor in descriptors)


def test_collision_uses_metadata_only_descriptors_when_supported(tmp_path, monkeypatch, validate_outputs, descriptors):
    first, second = tmp_path / "first.tsv", tmp_path / "second.tsv"
    first.write_text("first\n")
    second.write_text("second\n")
    _collide_once(monkeypatch, first, second)
    real_open = os.open
    native_metadata_flag = getattr(os, "O_PATH", None)
    metadata_flag = native_metadata_flag or (1 << 29)
    monkeypatch.setattr(os, "O_PATH", metadata_flag, raising=False)
    requested = []

    def require_metadata_access(path, flags, *args, **kwargs):
        if not flags & metadata_flag:
            raise PermissionError(errno.EACCES, "File data access is denied")
        if Path(path).name in {first.name, second.name}:
            requested.append(flags)
        if native_metadata_flag is None:
            # Exercise selection on hosts without O_PATH; real permission behavior is tested on Linux.
            flags &= ~metadata_flag
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", require_metadata_access)
    monkeypatch.setattr(os, "read", lambda *_args: pytest.fail("Metadata validation read file contents"))
    monkeypatch.setattr(os, "fdopen", lambda *_args, **_kwargs: pytest.fail("Metadata validation opened file contents"))

    validate_outputs(tmp_path, [first, second])

    assert len(requested) == 2
    assert all(flags & os.O_NOFOLLOW and flags & os.O_CLOEXEC for flags in requested)


@pytest.mark.parametrize("mode", [0o200, 0])
@pytest.mark.parametrize("directory_access", ["readable", "search_only_root", "search_only_parent"])
@pytest.mark.parametrize("metadata_only", [True, False], ids=["metadata-descriptors", "no-metadata-descriptors"])
def test_collision_handles_real_unreadable_outputs(
    tmp_path, monkeypatch, validate_outputs, descriptors, mode, directory_access, metadata_only
):
    if metadata_only and not hasattr(os, "O_PATH"):
        pytest.skip("This platform does not expose metadata-only O_PATH descriptors")
    if os.geteuid() == 0:
        pytest.skip("Read permission denial must be exercised without root bypass")
    if not metadata_only:
        monkeypatch.delattr(os, "O_PATH", raising=False)
    root = tmp_path / "workspace"
    parent = root / "outputs"
    parent.mkdir(parents=True)
    first, second = parent / "first.tsv", parent / "second.tsv"
    first.write_text("first\n")
    second.write_text("second\n")
    _collide_once(monkeypatch, first, second)
    first.chmod(mode)
    second.chmod(mode)
    restricted_directory = None
    if directory_access != "readable":
        restricted_directory = root if directory_access == "search_only_root" else parent
        restricted_directory.chmod(0o111)
    try:
        with pytest.raises(PermissionError):
            descriptor = os.open(first, os.O_RDONLY)
            os.close(descriptor)

        if restricted_directory:
            with pytest.raises(PermissionError):
                descriptor = os.open(restricted_directory, os.O_RDONLY | os.O_DIRECTORY)
                os.close(descriptor)

        if metadata_only:
            validate_outputs(root, [first, second])
        else:
            with pytest.raises(PermissionError):
                validate_outputs(root, [first, second])
    finally:
        root.chmod(0o700)
        parent.chmod(0o700)
        first.chmod(0o600)
        second.chmod(0o600)

    if metadata_only:
        assert len(descriptors) == 2
        assert all(flags & os.O_PATH for _name, flags, _descriptor in descriptors)
    else:
        assert not descriptors


@pytest.mark.skipif(not hasattr(os, "O_PATH"), reason="This platform does not expose metadata-only O_PATH descriptors")
def test_metadata_descriptor_for_symlink_is_rejected_by_fstat(tmp_path, monkeypatch, validate_outputs, descriptors):
    first, second, target = (tmp_path / name for name in ("first.tsv", "second.tsv", "target.tsv"))
    first.write_text("first\n")
    second.write_text("second\n")
    target.write_text("untouched\n")

    def change():
        first.unlink()
        first.symlink_to(target)

    _collide_once(monkeypatch, first, second, change)
    real_open = os.open
    symlink_opened = False

    def record_symlink(path, flags, *args, **kwargs):
        nonlocal symlink_opened
        descriptor = real_open(path, flags, *args, **kwargs)
        if Path(path).name == first.name:
            assert flags & os.O_PATH and flags & os.O_NOFOLLOW
            symlink_opened = stat.S_ISLNK(os.fstat(descriptor).st_mode)
        return descriptor

    monkeypatch.setattr(os, "open", record_symlink)

    with pytest.raises((ValueError, SystemExit), match="independent regular files|2"):
        validate_outputs(tmp_path, [first, second])

    assert symlink_opened, "O_PATH success must not be mistaken for a regular file"


@pytest.mark.parametrize("replace_alias", [False, True])
def test_current_alias_is_rejected_even_after_replacement(
    tmp_path, monkeypatch, validate_outputs, descriptors, replace_alias
):
    first, second, replacement = (tmp_path / name for name in ("first.tsv", "second.tsv", "replacement"))
    first.write_text("old\n")
    second.hardlink_to(first)
    replacement.write_text("new\n")
    real_lstat, real_fstat = os.lstat, os.fstat

    def one_link(info):
        values = list(info)
        values[3] = 1
        return os.stat_result(values)

    # Case-insensitive aliases can share an inode while reporting st_nlink == 1.
    monkeypatch.setattr(os, "lstat", lambda *args, **kwargs: one_link(real_lstat(*args, **kwargs)))
    monkeypatch.setattr(os, "fstat", lambda descriptor: one_link(real_fstat(descriptor)))

    def replace():
        if replace_alias:
            os.replace(replacement, first)
            second.unlink()
            second.hardlink_to(first)

    _collide_once(monkeypatch, first, second, replace)

    with pytest.raises((ValueError, SystemExit), match="independent regular files|2"):
        validate_outputs(tmp_path, [first, second])

    assert {name for name, _flags, _descriptor in descriptors} == {first.name, second.name}


def test_stale_collision_does_not_discard_an_earlier_alias_candidate(
    tmp_path, monkeypatch, validate_outputs, descriptors
):
    first, second, third = (tmp_path / name for name in ("first.tsv", "second.tsv", "third.tsv"))
    first.write_text("shared\n")
    second.write_text("independent\n")
    third.hardlink_to(first)
    real_lstat, real_fstat = os.lstat, os.fstat

    def one_link(info):
        values = list(info)
        values[3] = 1
        return os.stat_result(values)

    monkeypatch.setattr(os, "lstat", lambda *args, **kwargs: one_link(real_lstat(*args, **kwargs)))
    monkeypatch.setattr(os, "fstat", lambda descriptor: one_link(real_fstat(descriptor)))
    _collide_once(monkeypatch, first, second)

    with pytest.raises((ValueError, SystemExit), match="independent regular files|2"):
        validate_outputs(tmp_path, [first, second, third])


@pytest.mark.parametrize("alias_of", ["first.tsv", "second.tsv"])
def test_collision_refresh_retains_fresh_identities_for_later_aliases(
    tmp_path, monkeypatch, validate_outputs, descriptors, alias_of
):
    first, second, third, replacement = (
        tmp_path / name for name in ("first.tsv", "second.tsv", "third.tsv", "replacement")
    )
    first.write_text("old\n")
    second.write_text("independent\n")
    replacement.write_text("new\n")
    real_lstat, real_fstat = os.lstat, os.fstat

    def one_link(info):
        values = list(info)
        values[3] = 1
        return os.stat_result(values)

    monkeypatch.setattr(os, "lstat", lambda *args, **kwargs: one_link(real_lstat(*args, **kwargs)))
    monkeypatch.setattr(os, "fstat", lambda descriptor: one_link(real_fstat(descriptor)))

    def replace():
        os.replace(replacement, first)
        third.hardlink_to(tmp_path / alias_of)

    _collide_once(monkeypatch, first, second, replace)

    with pytest.raises((ValueError, SystemExit), match="independent regular files|2"):
        validate_outputs(tmp_path, [first, second, third])


@pytest.mark.parametrize("existing_candidate", ["alias", "replaced"])
def test_collision_rechecks_another_previously_seen_inode(
    tmp_path, monkeypatch, validate_outputs, descriptors, existing_candidate
):
    earlier, first, second, replacement = (
        tmp_path / name for name in ("earlier.tsv", "first.tsv", "second.tsv", "replacement")
    )
    earlier.write_text("earlier\n")
    first.write_text("old\n")
    second.write_text("independent\n")
    replacement.write_text("new\n")
    real_lstat, real_fstat = os.lstat, os.fstat

    def one_link(info):
        values = list(info)
        values[3] = 1
        return os.stat_result(values)

    monkeypatch.setattr(os, "lstat", lambda *args, **kwargs: one_link(real_lstat(*args, **kwargs)))
    monkeypatch.setattr(os, "fstat", lambda descriptor: one_link(real_fstat(descriptor)))

    def replace():
        if existing_candidate == "alias":
            first.unlink()
            first.hardlink_to(earlier)
        else:
            os.replace(earlier, first)
            os.replace(replacement, earlier)

    _collide_once(monkeypatch, first, second, replace)

    if existing_candidate == "alias":
        with pytest.raises((ValueError, SystemExit), match="independent regular files|2"):
            validate_outputs(tmp_path, [earlier, first, second])
    else:
        validate_outputs(tmp_path, [earlier, first, second])


@pytest.mark.parametrize("changed", ["leaf", "ancestor", "missing"])
def test_collision_rechecks_earlier_paths_that_now_alias_a_colliding_candidate(
    tmp_path, monkeypatch, validate_outputs, descriptors, changed
):
    parent = tmp_path / "earlier"
    parent.mkdir()
    earlier, first, second = parent / "output.tsv", tmp_path / "first.tsv", tmp_path / "second.tsv"
    if changed != "missing":
        earlier.write_text("earlier\n")
    first.write_text("first\n")
    second.write_text("independent\n")
    real_lstat, real_fstat = os.lstat, os.fstat

    def one_link(info):
        values = list(info)
        values[3] = 1
        return os.stat_result(values)

    # Model nlink-one aliases while replacing a path outside the stale collision bucket.
    monkeypatch.setattr(os, "lstat", lambda *args, **kwargs: one_link(real_lstat(*args, **kwargs)))
    monkeypatch.setattr(os, "fstat", lambda descriptor: one_link(real_fstat(descriptor)))

    def change():
        if changed == "ancestor":
            parent.rename(tmp_path / "old-parent")
            parent.mkdir()
        elif changed == "leaf":
            earlier.unlink()
        earlier.hardlink_to(first)

    _collide_once(monkeypatch, first, second, change)

    with pytest.raises((ValueError, SystemExit), match="independent regular files|2"):
        validate_outputs(tmp_path, [earlier, first, second])


def test_collision_refresh_keeps_missing_group_members_and_deduplicates_candidates(
    tmp_path, monkeypatch, validate_outputs, descriptors
):
    first, second, third = (tmp_path / name for name in ("first.tsv", "second.tsv", "third.tsv"))
    first.write_text("moved\n")
    second.write_text("replaced\n")
    third.write_text("independent\n")
    original = first.stat()
    _collide_once(monkeypatch, first, second, lambda: os.replace(first, second))
    real_lstat = os.lstat

    def collide_third(path, *args, **kwargs):
        info = real_lstat(path, *args, **kwargs)
        if Path(path) == third:
            values = list(info)
            values[1:3] = [original.st_ino, original.st_dev]
            return os.stat_result(values)
        return info

    monkeypatch.setattr(os, "lstat", collide_third)

    validate_outputs(tmp_path, [first, second, third])

    assert [name for name, _flags, _descriptor in descriptors] == [second.name, second.name]


@pytest.mark.parametrize("missing", ["first", "second", "root"])
def test_collision_recheck_allows_confirmed_missing(tmp_path, monkeypatch, validate_outputs, descriptors, missing):
    root = tmp_path / "workspace"
    root.mkdir()
    first, second = root / "first.tsv", root / "second.tsv"
    first.write_text("first\n")
    second.write_text("second\n")

    def remove():
        if missing == "root":
            root.rename(tmp_path / "moved-workspace")
        else:
            (first if missing == "first" else second).unlink()

    _collide_once(monkeypatch, first, second, remove)

    validate_outputs(root, [first, second])


@pytest.mark.parametrize("replace_after", ["first.tsv", "second.tsv"])
def test_collision_recheck_reports_unlinked_opened_object_as_uncertain(
    tmp_path, monkeypatch, validate_outputs, descriptors, replace_after
):
    first, second, replacement = (tmp_path / name for name in ("first.tsv", "second.tsv", "replacement"))
    first.write_text("old\n")
    second.write_text("second\n")
    replacement.write_text("new\n")
    _collide_once(monkeypatch, first, second)
    real_open = os.open

    def replace_after_open(path, flags, *args, **kwargs):
        descriptor = real_open(path, flags, *args, **kwargs)
        if Path(path).name == replace_after:
            os.replace(replacement, first)
        return descriptor

    monkeypatch.setattr(os, "open", replace_after_open)

    with pytest.raises(RuntimeError, match="changed|uncertain"):
        validate_outputs(tmp_path, [first, second])


def test_collision_recheck_closes_descriptors_on_open_failure(tmp_path, monkeypatch, validate_outputs, descriptors):
    first, second = tmp_path / "first.tsv", tmp_path / "second.tsv"
    first.write_text("first\n")
    second.write_text("second\n")
    _collide_once(monkeypatch, first, second)
    real_open = os.open

    def deny_second(path, flags, *args, **kwargs):
        if Path(path).name == second.name:
            assert descriptors, "Expected the first leaf to remain pinned before the second open"
            opened = os.fstat(descriptors[0][2])
            assert (opened.st_dev, opened.st_ino) == (first.stat().st_dev, first.stat().st_ino)
            raise PermissionError(errno.EACCES, "injected collision open failure")
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", deny_second)

    with pytest.raises(PermissionError, match="injected collision open failure"):
        validate_outputs(tmp_path, [first, second])


@pytest.mark.parametrize("error_number", [errno.ENOENT, errno.EIO])
def test_collision_fstat_failure_is_not_confirmed_missing(
    tmp_path, monkeypatch, validate_outputs, descriptors, error_number
):
    first, second = tmp_path / "first.tsv", tmp_path / "second.tsv"
    first.write_text("first\n")
    second.write_text("second\n")
    _collide_once(monkeypatch, first, second)
    real_fstat = os.fstat

    def fail_on_leaf(descriptor):
        if descriptor in {opened for _name, _flags, opened in descriptors}:
            raise OSError(error_number, "injected descriptor metadata failure")
        return real_fstat(descriptor)

    monkeypatch.setattr(os, "fstat", fail_on_leaf)

    with pytest.raises(OSError, match="injected descriptor metadata failure"):
        validate_outputs(tmp_path, [first, second])


@pytest.mark.parametrize("unsafe", ["symlink", "hardlink", "fifo", "parent_symlink"])
def test_collision_recheck_does_not_follow_changed_unsafe_paths(
    tmp_path, monkeypatch, validate_outputs, descriptors, unsafe
):
    root = tmp_path / "workspace"
    parent = root / "outputs"
    parent.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    first, second = parent / "first.tsv", parent / "second.tsv"
    first.write_text("first\n")
    second.write_text("second\n")
    sentinel = outside / first.name
    sentinel.write_text("untouched\n")

    def change():
        if unsafe == "parent_symlink":
            parent.rename(root / "original-outputs")
            parent.symlink_to(outside, target_is_directory=True)
        else:
            first.unlink()
            if unsafe == "symlink":
                first.symlink_to(sentinel)
            elif unsafe == "hardlink":
                first.hardlink_to(sentinel)
            else:
                os.mkfifo(first)

    _collide_once(monkeypatch, first, second, change)

    with pytest.raises((ValueError, OSError, SystemExit)):
        validate_outputs(root, [first, second])

    assert sentinel.read_text() == "untouched\n"


def test_missing_collision_peer_does_not_hide_an_unsafe_remaining_target(
    tmp_path, monkeypatch, validate_outputs, descriptors
):
    first, second = tmp_path / "first.tsv", tmp_path / "second.tsv"
    first.write_text("first\n")
    second.write_text("second\n")

    def change():
        first.unlink()
        second.unlink()
        os.mkfifo(second)

    _collide_once(monkeypatch, first, second, change)

    with pytest.raises((ValueError, SystemExit)):
        validate_outputs(tmp_path, [first, second])


@pytest.mark.parametrize("mode", [0o200, 0])
def test_independent_outputs_remain_metadata_only(tmp_path, monkeypatch, validate_outputs, mode):
    first, second = tmp_path / "first.tsv", tmp_path / "second.tsv"
    first.write_text("first\n")
    second.write_text("second\n")
    first.chmod(mode)
    second.chmod(mode)
    monkeypatch.setattr(os, "open", lambda *_args, **_kwargs: pytest.fail("Independent outputs must not be opened"))
    try:
        validate_outputs(tmp_path, [first, second])
    finally:
        first.chmod(0o600)
        second.chmod(0o600)


@pytest.mark.parametrize("missing", ["root", "parent", "leaf"])
def test_missing_outputs_do_not_open_or_create_paths(tmp_path, monkeypatch, validate_outputs, missing):
    root = tmp_path / "workspace"
    parent = root / "outputs"
    if missing != "root":
        root.mkdir()
    if missing == "leaf":
        parent.mkdir()
    monkeypatch.setattr(os, "open", lambda *_args, **_kwargs: pytest.fail("Missing outputs must not be opened"))

    validate_outputs(root, [parent / "first.tsv", parent / "second.tsv"])

    assert not (parent / "first.tsv").exists()
    assert not (parent / "second.tsv").exists()
