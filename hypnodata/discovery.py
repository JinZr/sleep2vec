from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from hypnodata.adapters import call_collect_records
from hypnodata.config import HypnodataConfig
from hypnodata.records import RecordTask, validate_record_id


def discover_records(config: HypnodataConfig, adapter=None) -> list[RecordTask]:
    discovery = config.record_discovery
    if discovery.type == "glob":
        records = _discover_glob(config)
    elif discovery.type == "csv":
        records = _discover_csv(config)
    else:
        records = call_collect_records(adapter, config)
    _validate_unique_record_ids(records)
    return records


def _discover_glob(config: HypnodataConfig) -> list[RecordTask]:
    discovery = config.record_discovery
    assert discovery.root is not None
    paths = sorted(discovery.root.expanduser().glob(discovery.pattern))
    records = []
    for path in paths:
        metadata = dict(discovery.metadata)
        metadata.setdefault("source", config.center)
        metadata.setdefault("subject_id", path.stem)
        metadata.setdefault("session_id", path.stem)
        record_id = _record_id_from_value(path.stem)
        records.append(
            RecordTask(
                record_id=record_id,
                center=config.center,
                files={"edf": path},
                metadata=metadata,
                source_row={"path": str(path), **metadata},
            )
        )
    return records


def _discover_csv(config: HypnodataConfig) -> list[RecordTask]:
    discovery = config.record_discovery
    assert discovery.index is not None
    converters = {discovery.record_id_column: str} if discovery.record_id_column else None
    df = pd.read_csv(discovery.index, low_memory=False, converters=converters)
    file_columns = discovery.file_columns or {"edf": discovery.file_column}
    required = set(file_columns.values())
    if discovery.record_id_column:
        required.add(discovery.record_id_column)
    missing = sorted(column for column in required if column not in df.columns)
    if missing:
        raise ValueError(f"hypnodata discovery CSV missing required column(s): {missing}")

    records = []
    for row_idx, row in df.iterrows():
        row_dict = {column: _json_safe_value(row[column]) for column in df.columns}
        files = {name: Path(str(row[column])).expanduser() for name, column in file_columns.items()}
        if discovery.record_id_column:
            record_id = str(row[discovery.record_id_column])
        else:
            record_id = _record_id_from_value(Path(str(row[discovery.file_column])).stem) + f"__row{int(row_idx)}"
        metadata = dict(discovery.metadata)
        for column in discovery.metadata_columns:
            if column not in row.index:
                raise ValueError(f"metadata column {column!r} is missing from hypnodata discovery CSV.")
            metadata[column] = _json_safe_value(row[column])
        for output_key, column in (
            ("source", discovery.source_column),
            ("split", discovery.split_column),
            ("subject_id", discovery.subject_id_column),
            ("session_id", discovery.session_id_column),
        ):
            if output_key in metadata or column is None or column not in row.index:
                continue
            metadata[output_key] = _json_safe_value(row[column])
        records.append(
            RecordTask(
                record_id=record_id,
                center=config.center,
                files=files,
                metadata=metadata,
                source_row=row_dict,
            )
        )
    return records


def _validate_unique_record_ids(records: list[RecordTask]) -> None:
    seen: set[str] = set()
    duplicates: list[str] = []
    for record in records:
        validate_record_id(record.record_id)
        if record.record_id in seen and record.record_id not in duplicates:
            duplicates.append(record.record_id)
        seen.add(record.record_id)
    if duplicates:
        preview = ", ".join(duplicates[:5])
        suffix = "" if len(duplicates) <= 5 else f", ... ({len(duplicates)} total)"
        raise ValueError(f"hypnodata record_id values must be unique; duplicate record_id(s): {preview}{suffix}.")


def _record_id_from_value(value: Any) -> str:
    if pd.isna(value):
        raw = "na"
    else:
        raw = str(value)
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "-" for ch in raw.strip())
    return cleaned.strip("-") or "na"


def _json_safe_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value
