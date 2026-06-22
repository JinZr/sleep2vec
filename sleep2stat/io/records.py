from __future__ import annotations

from dataclasses import dataclass, field
import json
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
    raw_path: str | None = None
    resolved_path: Path | None = None
    path_exists: bool = False


def load_records(
    data_cfg: DataConfig, *, split_override: list[str] | None = None, limit: int | None = None
) -> list[SleepRecord]:
    if data_cfg.backend == "kaldi":
        records = _load_kaldi_records(data_cfg, split_override=split_override, limit=limit)
    else:
        records = _load_npz_records(data_cfg, split_override=split_override, limit=limit)
    _validate_unique_record_ids(records)
    return records


def _load_npz_records(
    data_cfg: DataConfig, *, split_override: list[str] | None = None, limit: int | None = None
) -> list[SleepRecord]:
    if data_cfg.index is None:
        raise ValueError("data.index is required for data.backend=npz.")
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
    for _, row in df.iterrows():
        row_idx = int(row.name)
        raw_path = str(row[data_cfg.path_column])
        path = Path(raw_path)
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
        record_id = _record_id(row, row_idx, Path(raw_path), data_cfg.record_id_columns)
        _validate_record_id_segment(record_id, "NPZ")
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
                raw_path=raw_path,
                resolved_path=path,
                path_exists=path.exists(),
            )
        )
        if limit is not None and len(records) >= limit:
            break
    return records


def _load_kaldi_records(
    data_cfg: DataConfig, *, split_override: list[str] | None = None, limit: int | None = None
) -> list[SleepRecord]:
    if data_cfg.kaldi_data_root is None or data_cfg.kaldi_manifest is None:
        raise ValueError("data.backend=kaldi requires data.kaldi_data_root and data.kaldi_manifest.")
    root = data_cfg.kaldi_data_root.expanduser()
    manifest_path = data_cfg.kaldi_manifest.expanduser()
    manifest = json.loads(manifest_path.read_text())
    raw_splits = manifest.get("splits")
    if not isinstance(raw_splits, dict):
        raise ValueError("Kaldi manifest.json must contain a splits mapping.")

    requested_split = list(split_override if split_override is not None else data_cfg.split)
    records = []
    for split_name in requested_split:
        split_spec = raw_splits.get(str(split_name))
        if not isinstance(split_spec, dict):
            raise ValueError(f"Kaldi manifest.json is missing requested split {split_name!r}.")
        split_manifest = root / str(split_spec["manifest"])
        df = pd.read_csv(split_manifest, low_memory=False)
        required = {"sample_key", data_cfg.path_column, "token_start", "token_end"}
        missing = sorted(required - set(df.columns))
        if missing:
            raise ValueError(f"Kaldi split manifest CSV is missing required column(s): {missing}.")
        if data_cfg.split_column in df.columns:
            df = df[df[data_cfg.split_column].astype(str) == str(split_name)]
        for row_idx, (_, row) in enumerate(df.iterrows()):
            token_start = int(row["token_start"])
            token_end = int(row["token_end"])
            duration_sec = float(max(0, token_end - token_start) * data_cfg.token_sec)
            source = None
            if data_cfg.source_column and data_cfg.source_column in row.index:
                value = row[data_cfg.source_column]
                source = None if pd.isna(value) else str(value)
            elif "dataset" in row.index and not pd.isna(row["dataset"]):
                source = str(row["dataset"])
            raw_path = str(row[data_cfg.path_column])
            metadata = {
                column: _json_safe_value(row[column])
                for column in row.index
                if column not in {data_cfg.path_column, data_cfg.duration_column}
            }
            if data_cfg.record_id_columns:
                record_id = _record_id(row, row_idx, Path(raw_path), data_cfg.record_id_columns)
            else:
                sample_key = row["sample_key"]
                record_id = "" if pd.isna(sample_key) else str(sample_key)
            _validate_record_id_segment(record_id, "Kaldi")
            records.append(
                SleepRecord(
                    record_id=record_id,
                    path=Path(raw_path),
                    split=str(split_name),
                    source=source,
                    duration_sec=duration_sec,
                    token_sec=data_cfg.token_sec,
                    max_tokens=data_cfg.max_tokens,
                    metadata=metadata,
                    raw_path=raw_path,
                    resolved_path=Path(raw_path),
                    path_exists=Path(raw_path).exists(),
                )
            )
            if limit is not None and len(records) >= limit:
                return records
    return records


def records_to_frame(records: list[SleepRecord], metadata_columns: list[str] | None = None) -> pd.DataFrame:
    rows = []
    for record in records:
        raw_path = record.raw_path if record.raw_path is not None else str(record.path)
        resolved_path = record.resolved_path if record.resolved_path is not None else record.path
        row = {
            "record_id": record.record_id,
            "path": str(record.path),
            "raw_path": raw_path,
            "resolved_path": str(resolved_path),
            "path_exists": bool(record.path_exists),
            "split": record.split,
            "source": record.source,
            "duration_sec": record.duration_sec,
            "token_sec": record.token_sec,
            "max_tokens": record.max_tokens,
        }
        keys = sorted(record.metadata) if metadata_columns is None else list(metadata_columns)
        for key in keys:
            value = record.metadata.get(key)
            if key not in row and _is_manifest_scalar(value):
                row[key] = value
        rows.append(row)
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


def _validate_record_id_segment(record_id: str, backend: str) -> None:
    if record_id in {"", ".", ".."} or "/" in record_id or "\\" in record_id:
        raise ValueError(
            f"{backend} record_id values must be a single path-safe sleep2stat record_id segment; "
            f"got {record_id!r}."
        )


def _json_safe_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def _is_manifest_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def _validate_unique_record_ids(records: list[SleepRecord]) -> None:
    seen: set[str] = set()
    duplicates: list[str] = []
    for record in records:
        if record.record_id in seen and record.record_id not in duplicates:
            duplicates.append(record.record_id)
        seen.add(record.record_id)
    if duplicates:
        preview = ", ".join(duplicates[:5])
        suffix = "" if len(duplicates) <= 5 else f", ... ({len(duplicates)} total)"
        raise ValueError("sleep2stat record_id values must be unique; duplicate record_id(s): " f"{preview}{suffix}.")
