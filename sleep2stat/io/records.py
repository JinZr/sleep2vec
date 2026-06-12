from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from sleep2stat.config import DataConfig


@dataclass(frozen=True)
class SleepRecord:
    record_id: str
    path: Path
    split: str
    source: str | None
    duration_sec: float
    token_sec: int
    max_tokens: int
    metadata: dict[str, Any] = field(default_factory=dict)


def load_records(
    data_cfg: DataConfig, *, split_override: list[str] | None = None, limit: int | None = None
) -> list[SleepRecord]:
    df = pd.read_csv(data_cfg.index, low_memory=False)
    required = [data_cfg.path_column, data_cfg.duration_column]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"sleep2stat index missing required column(s): {missing}")

    requested_split = list(split_override if split_override is not None else data_cfg.split)
    if requested_split:
        if data_cfg.split_column not in df.columns:
            raise ValueError(f"sleep2stat index missing split column: {data_cfg.split_column!r}")
        df = df[df[data_cfg.split_column].astype(str).isin([str(value) for value in requested_split])]

    records = []
    for row_idx, (_, row) in enumerate(df.iterrows()):
        path = Path(str(row[data_cfg.path_column]))
        split = str(row[data_cfg.split_column]) if data_cfg.split_column in row.index else ""
        source = None
        if data_cfg.source_column and data_cfg.source_column in row.index:
            value = row[data_cfg.source_column]
            source = None if pd.isna(value) else str(value)
        duration_sec = float(row[data_cfg.duration_column])
        metadata = {
            column: _json_safe_value(row[column])
            for column in row.index
            if column not in {data_cfg.path_column, data_cfg.duration_column}
        }
        record_id = _record_id(row, row_idx, path, data_cfg.record_id_columns)
        records.append(
            SleepRecord(
                record_id=record_id,
                path=path,
                split=split,
                source=source,
                duration_sec=duration_sec,
                token_sec=data_cfg.token_sec,
                max_tokens=data_cfg.max_tokens,
                metadata=metadata,
            )
        )
        if limit is not None and len(records) >= limit:
            break
    return records


def records_to_frame(records: list[SleepRecord]) -> pd.DataFrame:
    rows = []
    for record in records:
        rows.append(
            {
                "record_id": record.record_id,
                "path": str(record.path),
                "split": record.split,
                "source": record.source,
                "duration_sec": record.duration_sec,
                "token_sec": record.token_sec,
                "max_tokens": record.max_tokens,
            }
        )
    return pd.DataFrame(rows)


def _record_id(row: pd.Series, row_idx: int, path: Path, columns: list[str]) -> str:
    pieces = []
    for column in columns:
        if column not in row.index:
            raise ValueError(f"record_id column {column!r} is missing from sleep2stat index.")
        value = row[column]
        if pd.isna(value):
            pieces.append("na")
        else:
            pieces.append(str(value))
    if pieces:
        return "__".join(_slug_piece(piece) for piece in pieces)
    return f"{path.stem}__row{row_idx}"


def _slug_piece(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "-" for ch in value.strip())
    return cleaned.strip("-") or "na"


def _json_safe_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value
