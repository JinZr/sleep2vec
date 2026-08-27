from __future__ import annotations

from contextlib import ExitStack, contextmanager
import csv
import ctypes
import errno
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import secrets
import stat
import subprocess  # noqa: F401 -- tests patch experiment_io.subprocess.run (stdlib global)
import time
from typing import Any, Iterator

from . import transport
from .manifests import read_rows, utc_now, validate_managed_header, write_rows, write_text
from .models import json_ready
from .transport import (  # noqa: F401 -- SSH_TIMEOUT_SECONDS re-exported for existing importers/tests
    REMOTE_CONFLICT_RETURN_CODE,
    REMOTE_MISSING_RETURN_CODE,
    SSH_TIMEOUT_SECONDS,
)


@contextmanager
def blocking_file_lock(path: str | Path) -> Iterator[None]:
    lock_path = Path(path)
    for attempt in range(4):
        lock_file = lock_path.open("a+")
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        except OSError as exc:
            lock_file.close()
            if exc.errno != errno.EIO or attempt == 3:
                raise
            # Do not reuse a JuiceFS-backed descriptor after flock reports EIO.
            time.sleep(0.1 * (2**attempt))
        else:
            break
    with lock_file:
        yield


def mkdir_experiment_dirs(root: Path, *, remote: str | None = None) -> None:
    dirs = [root / "reports", root / "wandb" / "history"]
    if remote:
        command = "mkdir -p " + " ".join(transport.sh(path) for path in dirs)
        transport.run_ssh(remote, command, text=True, check=True)
        return
    for path in dirs:
        path.mkdir(parents=True, exist_ok=True)


def remote_dir_nonempty(root: Path, remote: str) -> bool:
    result = transport.run_ssh(
        remote,
        transport.remote_python_program_command("experiment_io.remote_dir_nonempty", str(root)),
        text=True,
    )
    if result.returncode == REMOTE_MISSING_RETURN_CODE:
        return False
    if result.returncode != 0:
        detail = result.stderr.strip() or f"exit code {result.returncode}"
        raise RuntimeError(f"SSH directory probe failed for {root} on {remote}: {detail}")
    return bool(result.stdout.strip())


def path_exists_at(path: str | Path, *, remote: str | None = None) -> bool:
    if not remote:
        target = Path(path)
        return target.exists() or target.is_symlink()
    result = transport.run_ssh(
        remote,
        transport.remote_python_program_command("experiment_io.path_exists", str(path)),
        text=True,
    )
    if result.returncode == REMOTE_MISSING_RETURN_CODE:
        return False
    if result.returncode != 0:
        detail = result.stderr.strip() or f"exit code {result.returncode}"
        raise RuntimeError(f"SSH path probe failed for {path} on {remote}: {detail}")
    return True


def list_managed_subdirectories_at(
    root: str | Path,
    directory: str | Path,
    *,
    remote: str | None = None,
) -> list[str]:
    root = Path(root)
    directory = Path(directory)
    _validate_raw_managed_path(root, directory)
    if remote:
        payload = json.dumps([str(root), str(directory)])
        result = transport.run_ssh(
            remote,
            transport.remote_python_program_command("experiment_io.list_managed_subdirectories", payload),
            text=True,
        )
        if result.returncode == 2:
            raise ValueError(result.stderr.strip() or f"Managed directory is invalid: {directory}")
        if result.returncode != 0:
            detail = result.stderr.strip() or f"exit code {result.returncode}"
            raise RuntimeError(f"SSH directory read failed for {directory} on {remote}: {detail}")
        return json.loads(result.stdout)

    try:
        root_info = os.lstat(root)
    except FileNotFoundError as exc:
        raise ValueError(f"Managed workspace root is missing: {root}") from exc
    if stat.S_ISLNK(root_info.st_mode) or not stat.S_ISDIR(root_info.st_mode):
        raise ValueError(f"Managed workspace root is missing or aliased: {root}")
    relative = directory.relative_to(root)
    current = root
    for part in relative.parts:
        current /= part
        try:
            info = os.lstat(current)
        except FileNotFoundError:
            return []
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise ValueError(f"Managed directory is missing or aliased: {current}")
    names = []
    for entry in os.scandir(directory):
        info = entry.stat(follow_symlinks=False)
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise ValueError(f"Managed directory contains a non-directory entry: {entry.path}")
        names.append(entry.name)
    return sorted(names)


def read_managed_files_at(
    root: str | Path,
    paths: list[str | Path],
    *,
    remote: str | None = None,
    exact_directory_entries: bool = False,
    allow_invalid_utf8: bool = False,
) -> dict[str, dict[str, str | None]]:
    root = Path(root)
    targets = [Path(path) for path in paths]
    for target in targets:
        _validate_raw_managed_path(root, target)
    if len(targets) != len(set(targets)):
        raise ValueError("Managed file paths must be unique.")
    if remote:
        request = json.dumps([str(root), [str(path) for path in targets], exact_directory_entries, allow_invalid_utf8])
        result = transport.run_ssh(
            remote,
            transport.remote_python_program_command("experiment_io.read_managed_files", request),
            text=True,
        )
        if result.returncode == 2:
            raise ValueError(result.stderr.strip() or "Managed control bundle is invalid.")
        if result.returncode != 0:
            detail = result.stderr.strip() or f"exit code {result.returncode}"
            raise RuntimeError(f"SSH managed-file read failed on {remote}: {detail}")
        return json.loads(result.stdout)

    try:
        root_info = os.lstat(root)
    except FileNotFoundError as exc:
        raise ValueError(f"Managed workspace root is missing: {root}") from exc
    if stat.S_ISLNK(root_info.st_mode) or not stat.S_ISDIR(root_info.st_mode):
        raise ValueError(f"Managed workspace root is missing or aliased: {root}")
    seen_inodes = set()
    payload = {}
    for target in targets:
        relative = target.relative_to(root)
        current = root
        for part in relative.parts[:-1]:
            current /= part
            try:
                info = os.lstat(current)
            except FileNotFoundError as exc:
                raise ValueError(f"Managed directory is missing: {current}") from exc
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
                raise ValueError(f"Managed directory is missing or aliased: {current}")
        try:
            before = os.lstat(target)
        except FileNotFoundError as exc:
            raise ValueError(f"Managed file is missing: {target}") from exc
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ValueError(f"Managed file is missing or aliased: {target}")
        inode = (before.st_dev, before.st_ino)
        if inode in seen_inodes:
            raise ValueError(f"Managed files must be independent regular files: {target}")
        seen_inodes.add(inode)
        with target.open("rb") as file_obj:
            opened = os.fstat(file_obj.fileno())
            data = file_obj.read()
            after = os.fstat(file_obj.fileno())
        if (opened.st_dev, opened.st_ino) != inode or (after.st_dev, after.st_ino) != inode:
            raise ValueError(f"Managed file changed while it was read: {target}")
        try:
            text = data.decode("utf-8")
        except UnicodeError as exc:
            if not allow_invalid_utf8:
                raise ValueError(f"Managed file is not valid UTF-8: {target}") from exc
            text = None
        payload[str(target)] = {"text": text, "sha256": hashlib.sha256(data).hexdigest()}
    if exact_directory_entries:
        parents = {target.parent for target in targets}
        if len(parents) != 1:
            raise ValueError("Exact managed control bundle files must share one directory.")
        parent = parents.pop()
        actual_entries = sorted(entry.name for entry in os.scandir(parent))
        expected_entries = sorted(target.name for target in targets)
        if actual_entries != expected_entries:
            raise ValueError(f"Managed control bundle directory entries differ: {parent}")
    return payload


def _validate_raw_managed_path(root: Path, target: Path) -> None:
    if not root.is_absolute() or not target.is_absolute():
        raise ValueError("Managed control paths must be absolute.")
    if ".." in root.parts or ".." in target.parts:
        raise ValueError("Managed control paths must not contain '..' components.")
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Managed control path is outside its workspace: {target}") from exc


def read_rows_at(
    path: str | Path,
    *,
    remote: str | None = None,
    require_managed_identity: bool = False,
    strict: bool = False,
) -> list[dict[str, str]]:
    strict = strict or require_managed_identity
    if not remote and not strict:
        return read_rows(path, require_managed_identity=require_managed_identity)
    if remote:
        text = read_text_at(path, remote=remote)
    else:
        target = Path(path)
        if not target.exists() and not target.is_symlink():
            return []
        text = target.read_text()
    if not text:
        if strict and path_exists_at(path, remote=remote):
            raise ValueError(f"Strict table is empty: {path}")
        return []
    delimiter = "\t" if Path(str(path)).suffix == ".tsv" else ","
    reader = csv.DictReader(io.StringIO(text), delimiter=delimiter, strict=strict)
    if strict:
        fieldnames = reader.fieldnames
        if not fieldnames:
            raise ValueError(f"Strict table has no header: {path}")
        if len(fieldnames) != len(set(fieldnames)):
            raise ValueError(f"Strict table has duplicate header fields: {path}")
    if require_managed_identity:
        validate_managed_header(fieldnames, path)
    try:
        rows = list(reader)
    except csv.Error as exc:
        raise ValueError(f"Strict table is malformed: {path}") from exc
    if strict and any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f"Strict table has a non-rectangular row: {path}")
    return rows


def validate_managed_output_paths(
    root: str | Path,
    paths: list[str | Path],
    *,
    remote: str | None = None,
) -> None:
    if not paths:
        return
    if remote:
        payload = json.dumps([str(root), *(str(path) for path in paths)])
        result = transport.run_ssh(
            remote,
            transport.remote_python_program_command("experiment_io.validate_managed_output_paths", payload),
            text=True,
        )
        if result.returncode == 2:
            raise ValueError(result.stderr.strip() or "Managed output paths must be independent regular files.")
        if result.returncode != 0:
            detail = result.stderr.strip() or f"exit code {result.returncode}"
            raise RuntimeError(f"SSH output path validation failed on {remote}: {detail}")
        return

    root_path = Path(os.path.abspath(root))
    current = Path(root_path.anchor)
    for part in root_path.relative_to(current).parts[:-1]:
        current /= part
        try:
            info = os.lstat(current)
        except FileNotFoundError:
            break
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise ValueError(f"Managed output paths must be independent regular files: {current}")
    try:
        root_info = os.lstat(root_path)
    except FileNotFoundError:
        pass
    else:
        if stat.S_ISLNK(root_info.st_mode) or not stat.S_ISDIR(root_info.st_mode):
            raise ValueError(f"Managed output paths must be independent regular files: {root_path}")
    seen_paths = set()
    seen_inodes = set()
    for raw_target in paths:
        target = Path(os.path.abspath(raw_target))
        try:
            relative = target.relative_to(root_path)
        except ValueError as exc:
            raise ValueError(f"Managed output path is outside its workspace: {target}") from exc
        if target in seen_paths:
            raise ValueError(f"Managed output paths must be independent regular files: {target}")
        seen_paths.add(target)

        ancestors = []
        current = root_path
        for part in relative.parts[:-1]:
            current /= part
            ancestors.append(current)
        missing_ancestor = False
        for ancestor in ancestors:
            try:
                info = os.lstat(ancestor)
            except FileNotFoundError:
                missing_ancestor = True
                break
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
                raise ValueError(f"Managed output paths must be independent regular files: {ancestor}")
        if missing_ancestor:
            continue

        try:
            info = os.lstat(target)
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ValueError(f"Managed output paths must be independent regular files: {target}")
        inode = (info.st_dev, info.st_ino)
        if inode in seen_inodes:
            raise ValueError(f"Managed output paths must be independent regular files: {target}")
        seen_inodes.add(inode)


def read_text_at(path: str | Path, *, remote: str | None = None) -> str:
    if not remote:
        target = Path(path)
        return target.read_bytes().decode() if target.exists() else ""
    result = transport.run_ssh(remote, transport.remote_python_program_command("experiment_io.read_text", str(path)))
    if result.returncode == REMOTE_MISSING_RETURN_CODE:
        return ""
    if result.returncode != 0:
        stderr = result.stderr.decode() if isinstance(result.stderr, bytes) else result.stderr
        detail = stderr.strip() or f"exit code {result.returncode}"
        raise RuntimeError(f"SSH read failed for {path} on {remote}: {detail}")
    return result.stdout.decode() if isinstance(result.stdout, bytes) else result.stdout


def write_rows_at(path: str | Path, rows: list[dict[str, Any]], *, remote: str | None = None) -> None:
    if not remote:
        write_rows(path, rows)
        return
    target = Path(str(path))
    fieldnames = sorted({key for row in rows for key in row}) if rows else ["run_id"]
    delimiter = "\t" if target.suffix == ".tsv" else ","
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, delimiter=delimiter)
    writer.writeheader()
    writer.writerows(rows)
    write_text_at(path, buffer.getvalue(), remote=remote)


def write_text_at(path: str | Path, text: str, *, remote: str | None = None) -> None:
    if not remote:
        write_text(path, text)
        return
    target = Path(str(path))
    transport.run_ssh(
        remote,
        transport.remote_write_command(target),
        input=text,
        text=True,
        check=True,
    )


_DIRECTORY_OPEN_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
_FILE_OPEN_FLAGS = os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC


def _close_descriptor(descriptor: int) -> None:
    try:
        os.close(descriptor)
    except OSError:
        pass


def _open_managed_root(managed_root: Path) -> int:
    current = os.open(managed_root.anchor, _DIRECTORY_OPEN_FLAGS)
    try:
        for part in managed_root.relative_to(managed_root.anchor).parts:
            opened = os.open(part, _DIRECTORY_OPEN_FLAGS, dir_fd=current)
            _close_descriptor(current)
            current = opened
    except BaseException:
        _close_descriptor(current)
        raise
    return current


def _open_managed_parent(root_descriptor: int, relative: Path, *, create: bool) -> tuple[int, str]:
    if not relative.parts:
        raise ValueError("Managed CAS target must name a file below its workspace root.")
    current = os.dup(root_descriptor)
    try:
        for part in relative.parts[:-1]:
            try:
                opened = os.open(part, _DIRECTORY_OPEN_FLAGS, dir_fd=current)
            except FileNotFoundError:
                if not create:
                    raise
                try:
                    os.mkdir(part, 0o755, dir_fd=current)
                except FileExistsError:
                    pass
                opened = os.open(part, _DIRECTORY_OPEN_FLAGS, dir_fd=current)
            _close_descriptor(current)
            current = opened
    except BaseException:
        _close_descriptor(current)
        raise
    return current, relative.name


def _read_regular_file_at(parent_descriptor: int, name: str) -> tuple[bytes, int]:
    descriptor = os.open(name, _FILE_OPEN_FLAGS, dir_fd=parent_descriptor)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ValueError(f"Managed CAS file is missing or aliased: {name}")
        with os.fdopen(descriptor, "rb") as file_obj:
            descriptor = -1
            return file_obj.read(), stat.S_IMODE(info.st_mode)
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _managed_publication_matches(
    root: Path,
    relative: Path,
    root_info: os.stat_result,
    parent_info: os.stat_result,
    published_info: os.stat_result,
    payload: bytes,
) -> bool:
    try:
        with ExitStack() as stack:
            root_descriptor = _open_managed_root(root)
            stack.callback(_close_descriptor, root_descriptor)
            parent_descriptor, target_name = _open_managed_parent(root_descriptor, relative, create=False)
            stack.callback(_close_descriptor, parent_descriptor)
            current_root_info = os.fstat(root_descriptor)
            current_parent_info = os.fstat(parent_descriptor)
            if (current_root_info.st_dev, current_root_info.st_ino) != (
                root_info.st_dev,
                root_info.st_ino,
            ) or (current_parent_info.st_dev, current_parent_info.st_ino) != (
                parent_info.st_dev,
                parent_info.st_ino,
            ):
                return False
            descriptor = os.open(target_name, _FILE_OPEN_FLAGS, dir_fd=parent_descriptor)
            stack.callback(_close_descriptor, descriptor)
            current_info = os.fstat(descriptor)
            if (
                not stat.S_ISREG(current_info.st_mode)
                or current_info.st_nlink != 1
                or (current_info.st_dev, current_info.st_ino) != (published_info.st_dev, published_info.st_ino)
            ):
                return False
            chunks = []
            while chunk := os.read(descriptor, 1024 * 1024):
                chunks.append(chunk)
            return b"".join(chunks) == payload
    except OSError:
        return False


def append_managed_text_at(path: str | Path, text: str, *, managed_root: str | Path) -> None:
    root = Path(str(managed_root))
    target = Path(str(path))
    _validate_raw_managed_path(root, target)
    incoming = text.encode()
    with ExitStack() as stack:
        try:
            root_descriptor = _open_managed_root(root)
        except OSError as exc:
            raise ValueError(f"Managed output paths must be independent regular files: {root}") from exc
        stack.callback(_close_descriptor, root_descriptor)
        root_info = os.fstat(root_descriptor)
        parent_descriptor, target_name = _open_managed_parent(
            root_descriptor,
            target.relative_to(root),
            create=False,
        )
        stack.callback(_close_descriptor, parent_descriptor)
        parent_info = os.fstat(parent_descriptor)
        stack.enter_context(_blocking_file_lock_at(parent_descriptor, f".{target_name}.cas.lock"))
        try:
            current, target_mode = _read_regular_file_at(parent_descriptor, target_name)
            target_exists = True
        except FileNotFoundError:
            current = b""
            target_mode = 0o644
            target_exists = False
        except ValueError as exc:
            raise ValueError(f"Managed output path is missing or aliased: {target}") from exc
        file_descriptor, temporary = _open_temporary_at(parent_descriptor, target_name)
        try:
            replacement = current + incoming
            temporary_info = os.fstat(file_descriptor)
            if not stat.S_ISREG(temporary_info.st_mode) or temporary_info.st_nlink != 1:
                raise ValueError(f"Managed append temporary is aliased: {target}")
            with os.fdopen(file_descriptor, "wb") as file_obj:
                file_descriptor = -1
                file_obj.write(replacement)
                os.fchmod(file_obj.fileno(), target_mode)
                file_obj.flush()
                os.fsync(file_obj.fileno())
                temporary_info = os.fstat(file_obj.fileno())
                if not stat.S_ISREG(temporary_info.st_mode) or temporary_info.st_nlink != 1:
                    raise ValueError(f"Managed append temporary is aliased: {target}")
            try:
                public_root = _open_managed_root(root)
            except OSError as exc:
                raise ValueError(f"Managed output path changed while it was written: {target}") from exc
            try:
                try:
                    public_parent, public_name = _open_managed_parent(
                        public_root,
                        target.relative_to(root),
                        create=False,
                    )
                except OSError as exc:
                    raise ValueError(f"Managed output path changed while it was written: {target}") from exc
                try:
                    public_root_info = os.fstat(public_root)
                    public_parent_info = os.fstat(public_parent)
                    if (public_root_info.st_dev, public_root_info.st_ino) != (
                        root_info.st_dev,
                        root_info.st_ino,
                    ) or (public_parent_info.st_dev, public_parent_info.st_ino) != (
                        parent_info.st_dev,
                        parent_info.st_ino,
                    ):
                        raise ValueError(f"Managed output path changed while it was written: {target}")
                    try:
                        public_current, _public_mode = _read_regular_file_at(public_parent, public_name)
                    except FileNotFoundError:
                        if target_exists:
                            raise ValueError(f"Managed output path changed while it was written: {target}")
                    except ValueError as exc:
                        raise ValueError(f"Managed output path is missing or aliased: {target}") from exc
                    else:
                        if not target_exists or public_current != current:
                            raise ValueError(f"Managed output path changed while it was written: {target}")
                    # Supported event writers share this lock; this binds the namespace for the following rename.
                    if target_exists:
                        os.replace(
                            temporary,
                            public_name,
                            src_dir_fd=public_parent,
                            dst_dir_fd=public_parent,
                        )
                    elif not _rename_noreplace_at(public_parent, temporary, public_name):
                        raise RuntimeError(f"Managed output path changed while it was written: {target}")
                    if not _managed_publication_matches(
                        root,
                        target.relative_to(root),
                        root_info,
                        parent_info,
                        temporary_info,
                        replacement,
                    ):
                        raise RuntimeError(
                            f"Managed publication outcome is unknown because its public path changed: {target}"
                        )
                finally:
                    _close_descriptor(public_parent)
            finally:
                _close_descriptor(public_root)
        except BaseException:
            if file_descriptor >= 0:
                _close_descriptor(file_descriptor)
            try:
                os.unlink(temporary, dir_fd=parent_descriptor)
            except FileNotFoundError:
                pass
            raise


@contextmanager
def _blocking_file_lock_at(parent_descriptor: int, name: str) -> Iterator[None]:
    for attempt in range(4):
        descriptor = -1
        try:
            try:
                descriptor = os.open(
                    name,
                    os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                    0o644,
                    dir_fd=parent_descriptor,
                )
            except FileExistsError:
                descriptor = os.open(
                    name,
                    os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC,
                    dir_fd=parent_descriptor,
                )
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise ValueError(f"Managed CAS lock is missing or aliased: {name}")
            fcntl.flock(descriptor, fcntl.LOCK_EX)
        except FileNotFoundError:
            if descriptor >= 0:
                os.close(descriptor)
            if attempt == 3:
                raise
            time.sleep(0.1 * (2**attempt))
        except OSError as exc:
            if descriptor >= 0:
                os.close(descriptor)
            if exc.errno != errno.EIO or attempt == 3:
                raise
            time.sleep(0.1 * (2**attempt))
        except BaseException:
            if descriptor >= 0:
                os.close(descriptor)
            raise
        else:
            break
    try:
        yield
    finally:
        _close_descriptor(descriptor)


def _open_temporary_at(parent_descriptor: int, target_name: str) -> tuple[int, str]:
    for _attempt in range(100):
        name = f".{target_name}.{secrets.token_hex(8)}"
        try:
            descriptor = os.open(
                name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                0o600,
                dir_fd=parent_descriptor,
            )
        except FileExistsError:
            continue
        return descriptor, name
    raise FileExistsError(f"Could not allocate a temporary file for managed CAS target: {target_name}")


def _rename_noreplace_at(parent_descriptor: int, source_name: str, target_name: str) -> bool:
    libc = ctypes.CDLL(None, use_errno=True)
    source = os.fsencode(source_name)
    destination = os.fsencode(target_name)
    if hasattr(libc, "renameat2"):
        rename = libc.renameat2
        rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        result = rename(parent_descriptor, source, parent_descriptor, destination, 1)
    elif hasattr(libc, "renameatx_np"):
        rename = libc.renameatx_np
        rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        result = rename(parent_descriptor, source, parent_descriptor, destination, 4)
    else:
        raise RuntimeError("Atomic descriptor-relative no-replace rename is unavailable on this platform.")
    if result == 0:
        return True
    error = ctypes.get_errno()
    if error == errno.EEXIST:
        return False
    raise OSError(error, os.strerror(error), target_name)


def conditional_atomic_replace_text_at(
    path: str | Path,
    text: str,
    expected_sha256: str | None,
    *,
    managed_root: str | Path,
    remote: str | None = None,
    dependency_path: str | Path | None = None,
    expected_dependency_sha256: str | None = None,
    guard_path: str | Path | None = None,
    expected_guard_sha256: str | None = None,
) -> bool:
    root = Path(str(managed_root))
    target = Path(str(path))
    _validate_raw_managed_path(root, target)
    if (dependency_path is None) != (expected_dependency_sha256 is None):
        raise ValueError("Dependency path and expected SHA-256 must be provided together.")
    if (guard_path is None) != (expected_guard_sha256 is None):
        raise ValueError("Guard path and expected SHA-256 must be provided together.")
    dependency = Path(str(dependency_path)) if dependency_path is not None else None
    guard = Path(str(guard_path)) if guard_path is not None else None
    if dependency is not None:
        _validate_raw_managed_path(root, dependency)
    if guard is not None:
        _validate_raw_managed_path(root, guard)
    payload = text.encode()
    if not remote:
        with ExitStack() as lock_stack:
            root_descriptor = _open_managed_root(root)
            lock_stack.callback(os.close, root_descriptor)
            root_info = os.fstat(root_descriptor)
            locked_files: set[tuple[int, int, str]] = set()
            target_parent, target_name = _open_managed_parent(
                root_descriptor,
                target.relative_to(root),
                create=True,
            )
            lock_stack.callback(os.close, target_parent)
            if dependency is not None:
                try:
                    dependency_parent, dependency_name = _open_managed_parent(
                        root_descriptor,
                        dependency.relative_to(root),
                        create=False,
                    )
                except FileNotFoundError:
                    return False
                lock_stack.callback(os.close, dependency_parent)
                dependency_lock_name = dependency_name + ".lock"
                dependency_parent_info = os.fstat(dependency_parent)
                dependency_lock_key = (
                    dependency_parent_info.st_dev,
                    dependency_parent_info.st_ino,
                    dependency_lock_name,
                )
                if dependency_lock_key not in locked_files:
                    lock_stack.enter_context(_blocking_file_lock_at(dependency_parent, dependency_lock_name))
                    locked_files.add(dependency_lock_key)
                try:
                    dependency_bytes, _dependency_mode = _read_regular_file_at(dependency_parent, dependency_name)
                except FileNotFoundError:
                    return False
                if hashlib.sha256(dependency_bytes).hexdigest() != expected_dependency_sha256:
                    return False
            target_lock_name = f".{target_name}.cas.lock"
            target_parent_info = os.fstat(target_parent)
            target_lock_key = (target_parent_info.st_dev, target_parent_info.st_ino, target_lock_name)
            if target_lock_key not in locked_files:
                lock_stack.enter_context(_blocking_file_lock_at(target_parent, target_lock_name))
                locked_files.add(target_lock_key)
            if guard is not None:
                try:
                    guard_parent, guard_name = _open_managed_parent(
                        root_descriptor,
                        guard.relative_to(root),
                        create=False,
                    )
                except FileNotFoundError:
                    return False
                lock_stack.callback(os.close, guard_parent)
                try:
                    guard_bytes, _guard_mode = _read_regular_file_at(guard_parent, guard_name)
                except FileNotFoundError:
                    return False
                if hashlib.sha256(guard_bytes).hexdigest() != expected_guard_sha256:
                    return False
            if expected_sha256 is None:
                try:
                    os.stat(target_name, dir_fd=target_parent, follow_symlinks=False)
                except FileNotFoundError:
                    pass
                else:
                    return False
                target_mode = 0o644
            else:
                try:
                    current, target_mode = _read_regular_file_at(target_parent, target_name)
                except FileNotFoundError:
                    return False
                if hashlib.sha256(current).hexdigest() != expected_sha256:
                    return False
            file_descriptor, temporary = _open_temporary_at(target_parent, target_name)
            try:
                with os.fdopen(file_descriptor, "wb") as file_obj:
                    file_obj.write(payload)
                    os.fchmod(file_obj.fileno(), target_mode)
                    file_obj.flush()
                    os.fsync(file_obj.fileno())
                    temporary_info = os.fstat(file_obj.fileno())
                    if not stat.S_ISREG(temporary_info.st_mode) or temporary_info.st_nlink != 1:
                        raise ValueError(f"Managed CAS temporary is aliased: {target}")
                try:
                    public_root = _open_managed_root(root)
                except OSError as exc:
                    raise ValueError(f"Managed CAS path changed during publication: {target}") from exc
                lock_stack.callback(os.close, public_root)
                try:
                    public_parent, public_name = _open_managed_parent(
                        public_root,
                        target.relative_to(root),
                        create=False,
                    )
                except OSError as exc:
                    raise ValueError(f"Managed CAS path changed during publication: {target}") from exc
                lock_stack.callback(os.close, public_parent)
                public_root_info = os.fstat(public_root)
                public_parent_info = os.fstat(public_parent)
                if (public_root_info.st_dev, public_root_info.st_ino) != (
                    root_info.st_dev,
                    root_info.st_ino,
                ) or (public_parent_info.st_dev, public_parent_info.st_ino) != (
                    target_parent_info.st_dev,
                    target_parent_info.st_ino,
                ):
                    raise ValueError(f"Managed CAS path changed during publication: {target}")
                try:
                    public_current, _public_mode = _read_regular_file_at(public_parent, public_name)
                except FileNotFoundError:
                    if expected_sha256 is not None:
                        os.unlink(temporary, dir_fd=public_parent)
                        return False
                except ValueError as exc:
                    raise ValueError(f"Managed output path is missing or aliased: {target}") from exc
                else:
                    if expected_sha256 is None or public_current != current:
                        os.unlink(temporary, dir_fd=public_parent)
                        return False
                # Supported writers share this lock; this binds the namespace for the following rename.
                if expected_sha256 is None:
                    if not _rename_noreplace_at(public_parent, temporary, public_name):
                        os.unlink(temporary, dir_fd=public_parent)
                        return False
                else:
                    os.replace(
                        temporary,
                        public_name,
                        src_dir_fd=public_parent,
                        dst_dir_fd=public_parent,
                    )
                if not _managed_publication_matches(
                    root,
                    target.relative_to(root),
                    root_info,
                    target_parent_info,
                    temporary_info,
                    payload,
                ):
                    raise RuntimeError(
                        f"Managed publication outcome is unknown because its public path changed: {target}"
                    )
            except BaseException:
                try:
                    os.unlink(temporary, dir_fd=target_parent)
                except FileNotFoundError:
                    pass
                raise
        return True

    result = transport.run_ssh(
        remote,
        transport.remote_python_program_command(
            "experiment_io.conditional_atomic_replace_text",
            str(root),
            str(target),
            expected_sha256 or "",
            str(dependency) if dependency is not None else "",
            expected_dependency_sha256 or "",
            str(guard) if guard is not None else "",
            expected_guard_sha256 or "",
        ),
        input=payload,
    )
    if result.returncode == REMOTE_CONFLICT_RETURN_CODE:
        return False
    if result.returncode != 0:
        stderr = result.stderr.decode() if isinstance(result.stderr, bytes) else result.stderr
        detail = stderr.strip() or f"exit code {result.returncode}"
        raise RuntimeError(f"SSH atomic replace failed for {target} on {remote}: {detail}")
    return True


def append_event_at(
    root: Path,
    event_type: str,
    payload: dict[str, Any],
    *,
    remote: str | None = None,
) -> None:
    row = json.dumps({"time": utc_now(), "event_type": event_type, **json_ready(payload)}, sort_keys=True) + "\n"
    path = root / "events.jsonl"
    if not remote:
        append_managed_text_at(path, row, managed_root=root)
        return
    command = transport.remote_python_program_command(
        "experiment_io.conditional_atomic_replace_text",
        str(root),
        str(path),
        "",
        "",
        "",
        "",
        "",
        "append",
    )
    result = transport.run_ssh(
        remote,
        command,
        input=row.encode(),
    )
    if result.returncode != 0:
        stderr = result.stderr.decode() if isinstance(result.stderr, bytes) else result.stderr
        detail = stderr.strip() or f"exit code {result.returncode}"
        raise RuntimeError(f"SSH managed event append failed on {remote}; outcome may be unknown: {detail}")
