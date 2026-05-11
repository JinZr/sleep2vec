#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import re
import shutil
import typing as t

import pandas as pd


def parse_args(argv: t.Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fix an index CSV so convert_npz_to_kaldi can generate unique Kaldi sample keys.",
    )
    parser.add_argument("--index", type=Path, required=True, help="Input index CSV.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output fixed index CSV. Defaults to overwriting --index after creating --index.backup.",
    )
    parser.add_argument("--source-field", default="dataset", help="CSV column used as the sample-key source prefix.")
    return parser.parse_args(argv)


def _is_missing(value: t.Any) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _sanitize_key_part(value: t.Any) -> str:
    text = str(value)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")
    return text or "unknown"


def _record_key_from_row(row: pd.Series) -> str:
    session_id = row.get("session_id", None)
    if session_id is not None and not _is_missing(session_id) and str(session_id):
        return _sanitize_key_part(session_id)
    path = Path(str(row["path"]))
    return _sanitize_key_part(f"{path.parent.name}_{path.stem}")


def _source_prefix(row: pd.Series, source_field: str) -> str:
    source_value = row[source_field]
    if _is_missing(source_value) or str(source_value).strip() == "":
        raise ValueError(f"CSV source field {source_field!r} has an empty value.")
    return _sanitize_key_part(source_value)


def _key_prefix(row: pd.Series, *, source_field: str) -> str:
    return f"{_source_prefix(row, source_field)}_{_record_key_from_row(row)}"


def _unique_session_id(
    *,
    row: pd.Series,
    row_number: int,
    source_field: str,
    used_prefixes: set[str],
) -> str:
    source = _source_prefix(row, source_field)
    base = _sanitize_key_part(f"{Path(str(row['path'])).parent.name}_{Path(str(row['path'])).stem}")
    suffix = 1
    while True:
        candidate = f"{base}_row{row_number:06d}" if suffix == 1 else f"{base}_row{row_number:06d}_{suffix}"
        if f"{source}_{candidate}" not in used_prefixes:
            return candidate
        suffix += 1


def fix_index(df: pd.DataFrame, *, source_field: str) -> tuple[pd.DataFrame, int]:
    required = {"path", source_field}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Input index CSV is missing required column(s): {missing}.")

    fixed = df.copy()
    if "session_id" not in fixed.columns:
        fixed["session_id"] = ""

    used_prefixes: set[str] = set()
    changed = 0
    for row_number, (row_index, row) in enumerate(fixed.iterrows(), start=1):
        prefix = _key_prefix(row, source_field=source_field)
        if prefix not in used_prefixes:
            used_prefixes.add(prefix)
            continue

        session_id = _unique_session_id(
            row=row,
            row_number=row_number,
            source_field=source_field,
            used_prefixes=used_prefixes,
        )
        fixed.at[row_index, "session_id"] = session_id
        used_prefixes.add(f"{_source_prefix(row, source_field)}_{session_id}")
        changed += 1

    return fixed, changed


def main(argv: t.Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    index_path = args.index.expanduser()
    output_path = index_path if args.output is None else args.output.expanduser()
    write_back = index_path.resolve() == output_path.resolve()

    df = pd.read_csv(index_path, low_memory=False)
    fixed, changed = fix_index(df, source_field=args.source_field)

    if write_back:
        backup_path = Path(str(index_path) + ".backup")
        if not backup_path.exists():
            shutil.copy2(index_path, backup_path)
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    fixed.to_csv(output_path, index=False)
    print(f"Wrote {output_path} with {len(fixed)} rows; updated {changed} 'session_id' value(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
