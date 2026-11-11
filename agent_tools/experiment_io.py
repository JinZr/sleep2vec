"""Managed file I/O: locked atomic writes and plain reads for workspace files.

Writers publish through a temporary file in the target's directory, fsync and
``os.replace``, so a reader never sees a half-written file. ``locked_replace``
and ``locked_append`` serialize writers of one file through a sidecar ``flock``;
callers that already hold an owning lock (such as the run lock for
``run_manifest.tsv``) call ``atomic_replace`` directly.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
import csv
import fcntl
import hashlib
import io
import os
from pathlib import Path
import tempfile
from typing import Any, Iterator, TypedDict

from .manifests import read_rows, validate_managed_header, write_rows, write_text


@contextmanager
def blocking_file_lock(path: str | Path) -> Iterator[None]:
    with Path(path).open("a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        yield


def lock_path_for(path: str | Path) -> Path:
    """The sidecar lock that locked_replace and locked_append take for path."""
    target = Path(path)
    return target.with_name(f".{target.name}.lock")


def atomic_replace(path: str | Path, text: str) -> None:
    """Replace path with text through a fsynced temporary file; the caller owns any writer lock."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.", suffix=".tmp")
    try:
        with os.fdopen(descriptor, "wb") as file_obj:
            os.fchmod(file_obj.fileno(), 0o644)
            file_obj.write(text.encode())
            file_obj.flush()
            os.fsync(file_obj.fileno())
        os.replace(temporary, target)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def locked_replace(path: str | Path, text: str, *, expected_sha256: str | None = None) -> None:
    """Atomically replace path under its sidecar lock.

    With expected_sha256, the current bytes must exist and hash to it, or this
    raises RuntimeError without writing: a compare-by-hash guard against lost
    updates by a writer that read the file earlier."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with blocking_file_lock(lock_path_for(target)):
        if expected_sha256 is not None:
            current = target.read_bytes() if target.exists() else None
            if current is None or hashlib.sha256(current).hexdigest() != expected_sha256:
                raise RuntimeError(f"Managed file changed before its replacement: {target}")
        atomic_replace(target, text)


def locked_append(path: str | Path, text: str) -> None:
    """Append text to path under its sidecar lock, publishing the whole file atomically."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with blocking_file_lock(lock_path_for(target)):
        atomic_replace(target, read_text_at(target) + text)


def mkdir_experiment_dirs(root: Path) -> None:
    (root / "reports").mkdir(parents=True, exist_ok=True)


def path_exists_at(path: str | Path) -> bool:
    target = Path(path)
    return target.exists() or target.is_symlink()


def read_text_at(path: str | Path) -> str:
    target = Path(path)
    return target.read_bytes().decode() if target.exists() else ""


def _require_inside(root: Path, target: Path) -> None:
    if not root.is_absolute() or not target.is_absolute():
        raise ValueError("Managed control paths must be absolute.")
    if ".." in root.parts or ".." in target.parts:
        raise ValueError("Managed control paths must not contain '..' components.")
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Managed control path is outside its workspace: {target}") from exc


def validate_managed_output_paths(root: str | Path, paths: Sequence[str | Path]) -> None:
    """Check output file paths once before writing; they need not exist yet.

    The root must not be a symlink, and every path must lie inside it, appear
    once, and not name an existing directory (so two runs cannot share an
    output and a run directory cannot already exist)."""
    if not paths:
        return
    root_path = Path(os.path.abspath(root))
    if root_path.is_symlink():
        raise ValueError(f"Managed workspace root must not be a symlink: {root_path}")
    seen: set[Path] = set()
    for raw_target in paths:
        target = Path(os.path.abspath(raw_target))
        try:
            target.relative_to(root_path)
        except ValueError as exc:
            raise ValueError(f"Managed output path is outside its workspace: {target}") from exc
        if target in seen or target.is_dir():
            raise ValueError(f"Managed output paths must be independent regular files: {target}")
        seen.add(target)


def list_managed_subdirectories_at(root: str | Path, directory: str | Path) -> list[str]:
    root = Path(root)
    directory = Path(directory)
    _require_inside(root, directory)
    if not directory.exists():
        return []
    names = []
    for entry in os.scandir(directory):
        if not entry.is_dir():
            raise ValueError(f"Managed directory contains a non-directory entry: {entry.path}")
        names.append(entry.name)
    return sorted(names)


class ManagedFileSnapshot(TypedDict):
    text: str
    sha256: str


def read_managed_files_at(
    root: str | Path,
    paths: list[str | Path],
    *,
    exact_directory_entries: bool = False,
) -> dict[str, ManagedFileSnapshot]:
    """Read UTF-8 files inside root with their SHA-256; a missing file raises ValueError.

    exact_directory_entries additionally requires the files to share one
    directory that contains nothing else."""
    root = Path(root)
    targets = [Path(path) for path in paths]
    for target in targets:
        _require_inside(root, target)
    if len(targets) != len(set(targets)):
        raise ValueError("Managed file paths must be unique.")
    payload: dict[str, ManagedFileSnapshot] = {}
    for target in targets:
        if not target.is_file():
            raise ValueError(f"Managed file is missing: {target}")
        data = target.read_bytes()
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


def read_managed_output_texts_at(
    root: str | Path,
    paths: Sequence[str | Path],
) -> dict[str, str | None]:
    """Read UTF-8 output files inside root, keyed as requested; a missing file reads as None."""
    root = Path(root)
    path_keys = [str(path) for path in paths]
    targets = [Path(path) for path in path_keys]
    for target in targets:
        _require_inside(root, target)
        if target == root:
            raise ValueError("Managed output target must name a file below its workspace root.")
    if len(targets) != len(set(targets)):
        raise ValueError("Managed output paths must be unique.")
    return {
        key: target.read_bytes().decode("utf-8") if target.exists() else None for key, target in zip(path_keys, targets)
    }


def read_rows_at(
    path: str | Path,
    *,
    require_managed_identity: bool = False,
    strict: bool = False,
) -> list[dict[str, str]]:
    strict = strict or require_managed_identity
    if not strict:
        return read_rows(path, require_managed_identity=require_managed_identity)
    target = Path(path)
    if not target.exists() and not target.is_symlink():
        return []
    text = target.read_text()
    if not text:
        raise ValueError(f"Strict table is empty: {path}")
    delimiter = "\t" if target.suffix == ".tsv" else ","
    reader = csv.DictReader(io.StringIO(text), delimiter=delimiter, strict=True)
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
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f"Strict table has a non-rectangular row: {path}")
    return rows


def write_rows_at(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    write_rows(path, rows)


def write_text_at(path: str | Path, text: str) -> None:
    write_text(path, text)
