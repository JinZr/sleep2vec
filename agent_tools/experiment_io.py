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
import stat
import subprocess  # noqa: F401 -- tests patch experiment_io.subprocess.run (stdlib global)
import tempfile
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
) -> dict[str, dict[str, str]]:
    root = Path(root)
    targets = [Path(path) for path in paths]
    for target in targets:
        _validate_raw_managed_path(root, target)
    if len(targets) != len(set(targets)):
        raise ValueError("Managed file paths must be unique.")
    if remote:
        request = json.dumps([str(root), [str(path) for path in targets], exact_directory_entries])
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
            raise ValueError(f"Managed file is not valid UTF-8: {target}") from exc
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


def conditional_atomic_replace_text_at(
    path: str | Path,
    text: str,
    expected_sha256: str | None,
    *,
    remote: str | None = None,
    dependency_path: str | Path | None = None,
    expected_dependency_sha256: str | None = None,
    guard_path: str | Path | None = None,
    expected_guard_sha256: str | None = None,
) -> bool:
    target = Path(str(path))
    if (dependency_path is None) != (expected_dependency_sha256 is None):
        raise ValueError("Dependency path and expected SHA-256 must be provided together.")
    if (guard_path is None) != (expected_guard_sha256 is None):
        raise ValueError("Guard path and expected SHA-256 must be provided together.")
    dependency = Path(str(dependency_path)) if dependency_path is not None else None
    guard = Path(str(guard_path)) if guard_path is not None else None
    payload = text.encode()
    if not remote:
        target.parent.mkdir(parents=True, exist_ok=True)
        lock_path = target.with_name(f".{target.name}.cas.lock")
        with ExitStack() as lock_stack:
            if dependency is not None:
                dependency_lock = dependency.with_name(dependency.name + ".lock")
                lock_stack.enter_context(blocking_file_lock(dependency_lock))
                try:
                    dependency_bytes = dependency.read_bytes()
                except FileNotFoundError:
                    return False
                if hashlib.sha256(dependency_bytes).hexdigest() != expected_dependency_sha256:
                    return False
            lock_stack.enter_context(blocking_file_lock(lock_path))
            if guard is not None:
                try:
                    guard_bytes = guard.read_bytes()
                except FileNotFoundError:
                    return False
                if hashlib.sha256(guard_bytes).hexdigest() != expected_guard_sha256:
                    return False
            if expected_sha256 is None:
                if os.path.lexists(target):
                    return False
                target_mode = 0o644
            else:
                try:
                    with target.open("rb") as file_obj:
                        current = file_obj.read()
                        target_mode = stat.S_IMODE(os.fstat(file_obj.fileno()).st_mode)
                except FileNotFoundError:
                    return False
                if hashlib.sha256(current).hexdigest() != expected_sha256:
                    return False
            file_descriptor, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
            try:
                with os.fdopen(file_descriptor, "wb") as file_obj:
                    file_obj.write(payload)
                    os.fchmod(file_obj.fileno(), target_mode)
                    file_obj.flush()
                    os.fsync(file_obj.fileno())
                if expected_sha256 is None:
                    libc = ctypes.CDLL(None, use_errno=True)
                    source = os.fsencode(temporary)
                    destination = os.fsencode(target)
                    if hasattr(libc, "renameat2"):
                        rename = libc.renameat2
                        rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
                        rename.restype = ctypes.c_int
                        result = rename(-100, source, -100, destination, 1)
                    elif hasattr(libc, "renamex_np"):
                        rename = libc.renamex_np
                        rename.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
                        rename.restype = ctypes.c_int
                        result = rename(source, destination, 4)
                    else:
                        raise RuntimeError("Atomic no-replace rename is unavailable on this platform.")
                    if result != 0:
                        error = ctypes.get_errno()
                        if error == errno.EEXIST:
                            Path(temporary).unlink(missing_ok=True)
                            return False
                        raise OSError(error, os.strerror(error), str(target))
                else:
                    os.replace(temporary, target)
            except BaseException:
                Path(temporary).unlink(missing_ok=True)
                raise
        return True

    result = transport.run_ssh(
        remote,
        transport.remote_python_program_command(
            "experiment_io.conditional_atomic_replace_text",
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
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as file_obj:
            file_obj.write(row)
        return
    command = transport.remote_append_command(path)
    transport.run_ssh(
        remote,
        command,
        input=row,
        text=True,
        check=True,
    )
