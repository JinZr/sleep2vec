from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

SURVIVAL_METADATA_KEYS = ("event_time", "is_event", "has_label")


@dataclass(frozen=True)
class SurvivalLabelTable:
    key_column: str
    label_names: list[str]
    event_time: dict[str, np.ndarray]
    is_event: dict[str, np.ndarray]
    has_label: dict[str, np.ndarray]


def load_survival_label_table(config: Any | None, expected_output_dim: int | None = None) -> SurvivalLabelTable | None:
    if config is None:
        return None

    key_column = config.key_column
    disease_columns = _load_disease_columns(config.disease_columns_index)
    event_time = _load_sidecar(config.event_time_index, key_column, disease_columns, "event_time_index")
    is_event = _load_sidecar(config.is_event_index, key_column, disease_columns, "is_event_index")
    has_label = _load_sidecar(config.has_label_index, key_column, disease_columns, "has_label_index")

    if expected_output_dim is not None and int(expected_output_dim) != len(disease_columns):
        raise ValueError(
            f"Survival output_dim ({expected_output_dim}) must match disease column count ({len(disease_columns)})."
        )
    if set(event_time) != set(is_event) or set(event_time) != set(has_label):
        raise ValueError("Survival sidecar key sets must match across event_time, is_event, and has_label.")

    for key in event_time:
        valid = has_label[key] > 0.5
        if not valid.any():
            continue
        if np.isnan(event_time[key][valid]).any() or np.isnan(is_event[key][valid]).any():
            raise ValueError(f"Survival labels for key {key!r} contain missing event_time or is_event values.")

    return SurvivalLabelTable(
        key_column=key_column,
        label_names=disease_columns,
        event_time=event_time,
        is_event=is_event,
        has_label=has_label,
    )


def attach_survival_metadata(metadata: dict[str, Any], key_value: Any, labels: SurvivalLabelTable) -> None:
    key = normalize_survival_key(key_value, labels.key_column)
    if key not in labels.event_time:
        raise ValueError(f"Main index key {key!r} is missing from survival sidecars.")
    metadata[labels.key_column] = key
    metadata["event_time"] = labels.event_time[key].copy()
    metadata["is_event"] = labels.is_event[key].copy()
    metadata["has_label"] = labels.has_label[key].copy()


def stack_survival_metadata(
    samples: list[Any],
    expected_output_dim: int | None = None,
    key_column: str | None = None,
) -> dict[str, Any]:
    expected_length = None if expected_output_dim is None else int(expected_output_dim)
    stacked: dict[str, Any] = {}
    for key in SURVIVAL_METADATA_KEYS:
        values = []
        for sample in samples:
            if key not in sample.metadata:
                raise ValueError(
                    f"Survival preset is missing metadata field {key!r}; regenerate presets with survival sidecars."
                )
            value = np.asarray(sample.metadata[key], dtype=np.float32)
            if value.ndim != 1:
                raise ValueError(f"Survival metadata field {key!r} must be a 1D vector.")
            if expected_length is not None and value.shape[0] != expected_length:
                raise ValueError(
                    f"Survival metadata field {key!r} has length {value.shape[0]}, expected {expected_length}."
                )
            values.append(value)
        stacked[key] = torch.as_tensor(np.stack(values), dtype=torch.float32)
    if key_column is not None:
        keys = []
        for sample in samples:
            if key_column not in sample.metadata:
                raise ValueError(
                    f"Survival preset is missing metadata field {key_column!r}; "
                    "regenerate presets with survival sidecars."
                )
            keys.append(normalize_survival_key(sample.metadata[key_column], key_column))
        stacked[key_column] = keys
    return stacked


def normalize_survival_key(value: Any, key_column: str) -> str:
    if pd.isna(value):
        raise ValueError(f"Survival key column {key_column!r} contains a missing value.")
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            raise ValueError(f"Survival key column {key_column!r} contains a non-finite value.")
        if float(value).is_integer():
            return str(int(value))
    return str(value).strip()


def _load_disease_columns(path: str | Path) -> list[str]:
    columns: list[str] = []
    seen: set[str] = set()
    for line_number, raw_line in enumerate(Path(path).read_text().splitlines(), start=1):
        name = raw_line.strip()
        if not name:
            raise ValueError(f"disease_columns_index contains an empty line at line {line_number}.")
        if name in seen:
            raise ValueError(f"disease_columns_index contains duplicate disease column {name!r}.")
        seen.add(name)
        columns.append(name)
    if not columns:
        raise ValueError("disease_columns_index must contain at least one disease column.")
    return columns


def _load_sidecar(
    path: str | Path,
    key_column: str,
    label_names: list[str],
    field_name: str,
) -> dict[str, np.ndarray]:
    frame = pd.read_csv(path)
    expected_columns = [key_column, *label_names]
    if list(frame.columns) != expected_columns:
        raise ValueError(f"{field_name} columns must exactly match [key_column] + disease_columns_index.")

    values_by_key: dict[str, np.ndarray] = {}
    for _, row in frame.iterrows():
        key = normalize_survival_key(row[key_column], key_column)
        if key in values_by_key:
            raise ValueError(f"{field_name} contains duplicate key {key!r}.")
        values_by_key[key] = pd.to_numeric(row[label_names], errors="coerce").to_numpy(dtype=np.float32)
    return values_by_key
