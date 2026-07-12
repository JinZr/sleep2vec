from __future__ import annotations

import csv
import io
import json
from pathlib import Path
import shlex
import subprocess
from typing import Any

from .manifests import read_rows, utc_now, write_rows, write_text
from .models import json_ready

SSH_TIMEOUT_SECONDS = 10
REMOTE_MISSING_RETURN_CODE = 44


def mkdir_experiment_dirs(root: Path, *, remote: str | None = None) -> None:
    dirs = [root / "reports", root / "wandb" / "history"]
    if remote:
        command = "mkdir -p " + " ".join(shlex.quote(str(path)) for path in dirs)
        subprocess.run(
            ["ssh", remote, command],
            check=True,
            text=True,
            capture_output=True,
            timeout=SSH_TIMEOUT_SECONDS,
        )
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
    result = subprocess.run(
        [
            "ssh",
            remote,
            f"python3 -c {shlex.quote(script)} {shlex.quote(str(root))}",
        ],
        text=True,
        capture_output=True,
        timeout=SSH_TIMEOUT_SECONDS,
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
    result = subprocess.run(
        [
            "ssh",
            remote,
            f"python3 -c {shlex.quote(script)} {shlex.quote(str(path))}",
        ],
        text=True,
        capture_output=True,
        timeout=SSH_TIMEOUT_SECONDS,
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
) -> list[dict[str, str]]:
    if not remote:
        return read_rows(path, require_managed_identity=require_managed_identity)
    text = read_text_at(path, remote=remote)
    if not text:
        if require_managed_identity and path_exists_at(path, remote=remote):
            raise ValueError(f"Managed table is empty: {path}")
        return []
    delimiter = "\t" if Path(str(path)).suffix == ".tsv" else ","
    reader = csv.DictReader(io.StringIO(text), delimiter=delimiter, strict=require_managed_identity)
    if require_managed_identity:
        fieldnames = reader.fieldnames
        if not fieldnames:
            raise ValueError(f"Managed table has no header: {path}")
        if len(fieldnames) != len(set(fieldnames)):
            raise ValueError(f"Managed table has duplicate header fields: {path}")
        if "trial_id" in fieldnames:
            raise ValueError(
                f"Historical managed table fields are read-only; Historical trial_id fields are unsupported: {path}"
            )
        if any(field.startswith("param.") for field in fieldnames):
            raise ValueError(f"Historical parameter fields are read-only: {path}")
        missing = [field for field in ("step_id", "run_id") if field not in fieldnames]
        if missing:
            raise ValueError(
                f"Managed table header must define step_id and run_id; missing {', '.join(missing)}: {path}"
            )
    rows = list(reader)
    if require_managed_identity and any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f"Managed table has a non-rectangular row: {path}")
    return rows


def read_text_at(path: str | Path, *, remote: str | None = None) -> str:
    if not remote:
        target = Path(path)
        return target.read_text() if target.exists() else ""
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
    with open(path, encoding="utf-8") as file_obj:
        sys.stdout.write(file_obj.read())
except (OSError, UnicodeError) as exc:
    print(exc, file=sys.stderr)
    raise SystemExit(1)
"""
    result = subprocess.run(
        [
            "ssh",
            remote,
            f"python3 -c {shlex.quote(script)} {shlex.quote(str(path))}",
        ],
        text=True,
        capture_output=True,
        timeout=SSH_TIMEOUT_SECONDS,
    )
    if result.returncode == REMOTE_MISSING_RETURN_CODE:
        return ""
    if result.returncode != 0:
        detail = result.stderr.strip() or f"exit code {result.returncode}"
        raise RuntimeError(f"SSH read failed for {path} on {remote}: {detail}")
    return result.stdout


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
    command = f"mkdir -p {shlex.quote(str(target.parent))} && cat > {shlex.quote(str(target))}"
    subprocess.run(
        ["ssh", remote, command],
        input=text,
        text=True,
        capture_output=True,
        check=True,
        timeout=SSH_TIMEOUT_SECONDS,
    )


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
    command = f"mkdir -p {shlex.quote(str(path.parent))} && cat >> {shlex.quote(str(path))}"
    subprocess.run(
        ["ssh", remote, command],
        input=row,
        text=True,
        capture_output=True,
        check=True,
        timeout=SSH_TIMEOUT_SECONDS,
    )
