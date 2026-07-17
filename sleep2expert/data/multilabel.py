from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

MULTILABEL_METADATA_KEYS = ("disease_label", "has_label")


@dataclass(frozen=True)
class MultilabelLabelTable:
    key_column: str
    label_names: list[str]
    disease_label: dict[str, np.ndarray]
    has_label: dict[str, np.ndarray]


def load_multilabel_label_table(
    config: Any | None, expected_output_dim: int | None = None
) -> MultilabelLabelTable | None:
    if config is None:
        return None

    key_column = config.key_column
    disease_columns = load_multilabel_disease_columns(config.disease_columns_index)
    disease_label = _load_sidecar(config.label_index, key_column, disease_columns, "label_index")
    has_label = _load_sidecar(config.has_label_index, key_column, disease_columns, "has_label_index")

    if expected_output_dim is not None and int(expected_output_dim) != len(disease_columns):
        raise ValueError(
            f"Multilabel output_dim ({expected_output_dim}) must match disease column count ({len(disease_columns)})."
        )
    if set(disease_label) != set(has_label):
        raise ValueError("Multilabel sidecar key sets must match across label_index and has_label_index.")

    for key in disease_label:
        mask_values = has_label[key]
        if np.isnan(mask_values).any():
            raise ValueError(f"Multilabel has_label for key {key!r} contains missing values.")
        if not np.isin(mask_values, [0.0, 1.0]).all():
            raise ValueError(f"Multilabel has_label for key {key!r} must be 0 or 1.")
        valid = has_label[key] > 0.5
        if not valid.any():
            continue
        labels = disease_label[key][valid]
        if np.isnan(labels).any():
            raise ValueError(f"Multilabel labels for key {key!r} contain missing values where has_label is true.")
        if not np.isin(labels, [0.0, 1.0]).all():
            raise ValueError(f"Multilabel labels for key {key!r} must be 0 or 1 where has_label is true.")

    return MultilabelLabelTable(
        key_column=key_column,
        label_names=disease_columns,
        disease_label=disease_label,
        has_label=has_label,
    )


def attach_multilabel_metadata(metadata: dict[str, Any], key_value: Any, labels: MultilabelLabelTable) -> None:
    key = normalize_multilabel_key(key_value, labels.key_column)
    if key not in labels.disease_label:
        raise ValueError(f"Main index key {key!r} is missing from multilabel sidecars.")
    metadata[labels.key_column] = key
    metadata["disease_label"] = labels.disease_label[key].copy()
    metadata["has_label"] = labels.has_label[key].copy()


def stack_multilabel_metadata(
    samples: list[Any],
    expected_output_dim: int | None = None,
    key_column: str | None = None,
) -> dict[str, Any]:
    import torch

    expected_length = None if expected_output_dim is None else int(expected_output_dim)
    stacked: dict[str, Any] = {}
    for key in MULTILABEL_METADATA_KEYS:
        values = []
        for sample in samples:
            if key not in sample.metadata:
                raise ValueError(
                    f"Multilabel preset is missing metadata field {key!r}; regenerate presets with multilabel sidecars."
                )
            value = np.asarray(sample.metadata[key], dtype=np.float32)
            if value.ndim != 1:
                raise ValueError(f"Multilabel metadata field {key!r} must be a 1D vector.")
            if expected_length is not None and value.shape[0] != expected_length:
                raise ValueError(
                    f"Multilabel metadata field {key!r} has length {value.shape[0]}, expected {expected_length}."
                )
            values.append(value)
        stacked[key] = torch.as_tensor(np.stack(values), dtype=torch.float32)
    if key_column is not None:
        keys = []
        for sample in samples:
            if key_column not in sample.metadata:
                raise ValueError(
                    f"Multilabel preset is missing metadata field {key_column!r}; "
                    "regenerate presets with multilabel sidecars."
                )
            keys.append(normalize_multilabel_key(sample.metadata[key_column], key_column))
        stacked[key_column] = keys
    return stacked


def normalize_multilabel_key(value: Any, key_column: str) -> str:
    if pd.isna(value):
        raise ValueError(f"Multilabel key column {key_column!r} contains a missing value.")
    key = str(value).strip()
    if not key:
        raise ValueError(f"Multilabel key column {key_column!r} contains an empty value.")
    return key


def load_multilabel_disease_columns(path: str | Path) -> list[str]:
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
    frame = pd.read_csv(path, converters={key_column: str})
    expected_columns = [key_column, *label_names]
    if list(frame.columns) != expected_columns:
        raise ValueError(f"{field_name} columns must exactly match [key_column] + disease_columns_index.")

    values_by_key: dict[str, np.ndarray] = {}
    for _, row in frame.iterrows():
        key = normalize_multilabel_key(row[key_column], key_column)
        if key in values_by_key:
            raise ValueError(f"{field_name} contains duplicate key {key!r}.")
        values_by_key[key] = pd.to_numeric(row[label_names], errors="coerce").to_numpy(dtype=np.float32)
    return values_by_key
