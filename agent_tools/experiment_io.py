from __future__ import annotations

import csv
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import subprocess  # noqa: F401 -- tests patch experiment_io.subprocess.run (stdlib global)
import tempfile
from typing import Any

from . import transport
from .manifests import read_rows, utc_now, validate_managed_header, write_rows, write_text
from .models import json_ready
from .transport import (  # noqa: F401 -- SSH_TIMEOUT_SECONDS re-exported for existing importers/tests
    REMOTE_CONFLICT_RETURN_CODE,
    REMOTE_MISSING_RETURN_CODE,
    SSH_TIMEOUT_SECONDS,
)


def mkdir_experiment_dirs(root: Path, *, remote: str | None = None) -> None:
    dirs = [root / "reports", root / "wandb" / "history"]
    if remote:
        command = "mkdir -p " + " ".join(transport.sh(path) for path in dirs)
        transport.run_ssh(remote, command, text=True, check=True)
        return
    for path in dirs:
        path.mkdir(parents=True, exist_ok=True)


def remote_dir_nonempty(root: Path, remote: str) -> bool:
    script = f"""
import os
import sys

path = sys.argv[1]
try:
    os.lstat(path)
except FileNotFoundError:
    raise SystemExit({REMOTE_MISSING_RETURN_CODE})
except OSError as exc:
    print(exc, file=sys.stderr)
    raise SystemExit(1)

try:
    entries = os.listdir(path)
except OSError as exc:
    print(exc, file=sys.stderr)
    raise SystemExit(1)

if entries:
    print("nonempty")
"""
    result = transport.run_ssh(
        remote,
        transport.remote_python_command(script, str(root)),
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
    script = f"""
import os
import sys

try:
    os.lstat(sys.argv[1])
except FileNotFoundError:
    raise SystemExit({REMOTE_MISSING_RETURN_CODE})
except OSError as exc:
    print(exc, file=sys.stderr)
    raise SystemExit(1)
"""
    result = transport.run_ssh(
        remote,
        transport.remote_python_command(script, str(path)),
        text=True,
    )
    if result.returncode == REMOTE_MISSING_RETURN_CODE:
        return False
    if result.returncode != 0:
        detail = result.stderr.strip() or f"exit code {result.returncode}"
        raise RuntimeError(f"SSH path probe failed for {path} on {remote}: {detail}")
    return True


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
        script = """
import json
import os
import stat
import sys

root, *targets = json.loads(sys.argv[1])
root = os.path.abspath(root)
seen_paths = set()
seen_inodes = set()

def reject(path):
    print(f"Managed output paths must be independent regular files: {path}", file=sys.stderr)
    raise SystemExit(2)

try:
    root_info = os.lstat(root)
except FileNotFoundError:
    pass
except OSError as exc:
    print(exc, file=sys.stderr)
    raise SystemExit(1)
else:
    if stat.S_ISLNK(root_info.st_mode) or not stat.S_ISDIR(root_info.st_mode):
        reject(root)

for raw_target in targets:
    target = os.path.abspath(raw_target)
    try:
        if os.path.commonpath([root, target]) != root:
            reject(target)
    except ValueError:
        reject(target)
    if target in seen_paths:
        reject(target)
    seen_paths.add(target)

    relative = os.path.relpath(target, root)
    ancestors = []
    current = root
    for part in relative.split(os.sep)[:-1]:
        current = os.path.join(current, part)
        ancestors.append(current)
    missing_ancestor = False
    for ancestor in ancestors:
        try:
            info = os.lstat(ancestor)
        except FileNotFoundError:
            missing_ancestor = True
            break
        except OSError as exc:
            print(exc, file=sys.stderr)
            raise SystemExit(1)
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            reject(ancestor)
    if missing_ancestor:
        continue

    try:
        info = os.lstat(target)
    except FileNotFoundError:
        continue
    except OSError as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(1)
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        reject(target)
    inode = (info.st_dev, info.st_ino)
    if inode in seen_inodes:
        reject(target)
    seen_inodes.add(inode)
"""
        payload = json.dumps([str(root), *(str(path) for path in paths)])
        result = transport.run_ssh(
            remote,
            transport.remote_python_command(script, payload),
            text=True,
        )
        if result.returncode == 2:
            raise ValueError(result.stderr.strip() or "Managed output paths must be independent regular files.")
        if result.returncode != 0:
            detail = result.stderr.strip() or f"exit code {result.returncode}"
            raise RuntimeError(f"SSH output path validation failed on {remote}: {detail}")
        return

    root_path = Path(os.path.abspath(root))
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
    script = f"""
import os
import sys

path = sys.argv[1]
try:
    os.lstat(path)
except FileNotFoundError:
    raise SystemExit({REMOTE_MISSING_RETURN_CODE})
except OSError as exc:
    print(exc, file=sys.stderr)
    raise SystemExit(1)

try:
    with open(path, encoding="utf-8", newline="") as file_obj:
        sys.stdout.write(file_obj.read())
except (OSError, UnicodeError) as exc:
    print(exc, file=sys.stderr)
    raise SystemExit(1)
"""
    result = transport.run_ssh(remote, transport.remote_python_command(script, str(path)))
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
    expected_sha256: str,
    *,
    remote: str | None = None,
) -> bool:
    target = Path(str(path))
    payload = text.encode()
    if not remote:
        with target.open("rb") as file_obj:
            current = file_obj.read()
            target_mode = stat.S_IMODE(os.fstat(file_obj.fileno()).st_mode)
        if hashlib.sha256(current).hexdigest() != expected_sha256:
            return False
        file_descriptor, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
        try:
            with os.fdopen(file_descriptor, "wb") as file_obj:
                file_obj.write(payload)
                os.fchmod(file_obj.fileno(), target_mode)
                file_obj.flush()
                os.fsync(file_obj.fileno())
            os.replace(temporary, target)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise
        return True

    script = f"""
import fcntl
import hashlib
import os
import stat
import sys
import tempfile

path = sys.argv[1]
expected = sys.argv[2]
lock_path = path + ".lock"
parent = os.path.dirname(path) or "."

with open(lock_path, "a+") as lock_file:
    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
    try:
        with open(path, "rb") as file_obj:
            current = file_obj.read()
            target_mode = stat.S_IMODE(os.fstat(file_obj.fileno()).st_mode)
    except FileNotFoundError:
        raise SystemExit({REMOTE_CONFLICT_RETURN_CODE})
    if hashlib.sha256(current).hexdigest() != expected:
        raise SystemExit({REMOTE_CONFLICT_RETURN_CODE})
    payload = sys.stdin.buffer.read()
    descriptor, temporary = tempfile.mkstemp(prefix="." + os.path.basename(path) + ".", dir=parent)
    try:
        with os.fdopen(descriptor, "wb") as file_obj:
            file_obj.write(payload)
            os.fchmod(file_obj.fileno(), target_mode)
            file_obj.flush()
            os.fsync(file_obj.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
"""
    result = transport.run_ssh(
        remote,
        transport.remote_python_command(script, str(target), expected_sha256),
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
