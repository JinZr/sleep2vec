from __future__ import annotations

import csv
import json
from pathlib import Path
import time
from typing import Any

from .models import json_ready


def read_json(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    return json.loads(Path(path).read_text())


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n")


def validate_managed_header(fieldnames: list[str], path: str | Path) -> None:
    if "trial_id" in fieldnames:
        raise ValueError(
            f"Historical managed table fields are read-only; Historical trial_id fields are unsupported: {path}"
        )
    if any(field.startswith("param.") for field in fieldnames):
        raise ValueError(f"Historical parameter fields are read-only: {path}")
    missing = [field for field in ("step_id", "run_id") if field not in fieldnames]
    if missing:
        raise ValueError(f"Managed table header must define step_id and run_id; missing {', '.join(missing)}: {path}")


def read_rows(path: str | Path, *, require_managed_identity: bool = False) -> list[dict[str, str]]:
    table = Path(path)
    if not table.exists() and not table.is_symlink():
        return []
    delimiter = "\t" if table.suffix == ".tsv" else ","
    with table.open(newline="") as file_obj:
        reader = csv.DictReader(file_obj, delimiter=delimiter, strict=require_managed_identity)
        if require_managed_identity:
            fieldnames = reader.fieldnames
            if not fieldnames:
                raise ValueError(f"Managed table has no header: {table}")
            if len(fieldnames) != len(set(fieldnames)):
                raise ValueError(f"Managed table has duplicate header fields: {table}")
            validate_managed_header(fieldnames, table)
        rows = list(reader)
    if require_managed_identity and any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f"Managed table has a non-rectangular row: {table}")
    return rows


def write_rows(path: str | Path, rows: list[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row}) if rows else ["run_id"]
    delimiter = "\t" if target.suffix == ".tsv" else ","
    with target.open("w", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames, delimiter=delimiter)
        writer.writeheader()
        writer.writerows(rows)


def write_text(path: str | Path, text: str, *, executable: bool = False) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text)
    if executable:
        target.chmod(target.stat().st_mode | 0o111)


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
