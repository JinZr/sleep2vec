from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import pickle
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from data.metadata import _encode_binary_label
from data.multilabel import load_multilabel_label_table, normalize_multilabel_key
from data.survival import load_survival_label_table, normalize_survival_key

from .config import BaselineConfig


@dataclass(frozen=True)
class BaselineRecord:
    key: str
    features: dict[str, float | int]
    path: str = ""
    token_start: int = 0
    event_time: np.ndarray | None = None
    is_event: np.ndarray | None = None
    disease_label: np.ndarray | None = None
    has_label: np.ndarray | None = None


class SexAgeDataset(Dataset):
    def __init__(self, records: list[BaselineRecord], *, task_type: str, label_names: list[str]) -> None:
        self.records = list(records)
        self.task_type = task_type
        self.label_names = list(label_names)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> BaselineRecord:
        return self.records[index]


def load_split_dataset(
    cfg: BaselineConfig,
    split: str,
    *,
    loaded_splits: list[str] | None = None,
) -> SexAgeDataset:
    frame = _load_metadata_frame(cfg, split=split, loaded_splits=loaded_splits)

    if cfg.finetune.task.type == "survival":
        labels = load_survival_label_table(cfg.finetune.survival, expected_output_dim=cfg.finetune.task.output_dim)
        assert labels is not None
        records = [
            BaselineRecord(
                key=row["_baseline_key"],
                features={name: row[f"_baseline_{name}"] for name in cfg.model.features},
                path=row["_baseline_path"],
                token_start=row["_baseline_token_start"],
                event_time=labels.event_time[_require_label_key(row["_baseline_key"], labels.event_time, split)],
                is_event=labels.is_event[row["_baseline_key"]],
                has_label=labels.has_label[row["_baseline_key"]],
            )
            for _, row in frame.iterrows()
        ]
        return SexAgeDataset(records, task_type=cfg.finetune.task.type, label_names=labels.label_names)

    labels = load_multilabel_label_table(cfg.finetune.multilabel, expected_output_dim=cfg.finetune.task.output_dim)
    assert labels is not None
    records = [
        BaselineRecord(
            key=row["_baseline_key"],
            features={name: row[f"_baseline_{name}"] for name in cfg.model.features},
            path=row["_baseline_path"],
            token_start=row["_baseline_token_start"],
            disease_label=labels.disease_label[_require_label_key(row["_baseline_key"], labels.disease_label, split)],
            has_label=labels.has_label[row["_baseline_key"]],
        )
        for _, row in frame.iterrows()
    ]
    return SexAgeDataset(records, task_type=cfg.finetune.task.type, label_names=labels.label_names)


def make_dataloader(
    dataset: SexAgeDataset,
    *,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    drop_last: bool = False,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        collate_fn=_collate_records,
    )


def _load_metadata_frame(
    cfg: BaselineConfig,
    *,
    split: str,
    loaded_splits: list[str] | None = None,
) -> pd.DataFrame:
    if cfg.data.backend == "npz":
        if cfg.data.finetune_preset_path:
            frame = _load_rows_from_npz_preset(cfg)
        else:
            frame = _load_rows_from_npz_index(cfg)
    elif cfg.data.backend == "kaldi":
        frame = _load_rows_from_kaldi_manifest(cfg)
    else:
        raise ValueError(f"Unsupported sex_age_baseline data backend: {cfg.data.backend}")

    required_columns = {cfg.data.key_column, cfg.data.split_column, *cfg.model.features}
    missing = sorted(required_columns - set(frame.columns))
    if missing:
        raise ValueError(f"Sex/age baseline metadata is missing required columns: {missing}")
    normalize_key = _key_normalizer(cfg)

    if loaded_splits:
        _validate_loaded_split_key_uniqueness(frame, cfg, normalize_key, loaded_splits)

    requested_split = str(split).strip()
    split_values = frame[cfg.data.split_column].map(_raw_split_value)
    frame = frame[split_values == requested_split].copy()

    frame["_baseline_key"] = [normalize_key(value, cfg.data.key_column) for value in frame[cfg.data.key_column]]
    frame["_baseline_split"] = [_parse_split(value, cfg.data.split_column) for value in frame[cfg.data.split_column]]
    invalid = {}
    for name in cfg.model.features:
        values = []
        failures = 0
        for value in frame[name]:
            try:
                values.append(_parse_sex(value) if name == "sex" else _parse_continuous(value, name))
            except (ValueError, TypeError):
                failures += 1
                values.append(None)
        frame[f"_baseline_{name}"] = values
        if failures:
            invalid[name] = failures
    if invalid:
        raise ValueError(f"Invalid selected covariates in split {split!r} ({len(frame)} rows): {invalid}")
    _validate_duplicate_metadata(frame, cfg.model.features)
    if cfg.data.deduplicate_by_key:
        frame = frame.drop_duplicates("_baseline_key", keep="first").copy()
        frame["_baseline_path"] = ""
        frame["_baseline_token_start"] = 0
    else:
        if not {"path", "token_start"}.issubset(frame.columns):
            raise ValueError("Window mode requires explicit path and token_start metadata.")
        if frame["path"].isna().any() or (frame["path"].astype(str).str.strip() == "").any():
            raise ValueError("Window mode requires a non-empty path for every sample.")
        starts = pd.to_numeric(frame["token_start"], errors="raise")
        if not np.isfinite(starts).all() or (starts < 0).any() or (starts % 1 != 0).any():
            raise ValueError("Window token_start must be a finite non-negative integer.")
        frame["_baseline_path"] = frame["path"].astype(str)
        frame["_baseline_token_start"] = starts.astype(int)
    return frame


def _load_rows_from_npz_index(cfg: BaselineConfig) -> pd.DataFrame:
    return pd.read_csv(Path(cfg.data.finetune_data_index), dtype={cfg.data.key_column: "string"})


def _load_rows_from_npz_preset(cfg: BaselineConfig) -> pd.DataFrame:
    with Path(cfg.data.finetune_preset_path).open("rb") as file_obj:
        samples = pickle.load(file_obj)
    rows = []
    for sample in samples:
        metadata = getattr(sample, "metadata", None)
        if not isinstance(metadata, dict):
            raise ValueError("Sex/age baseline preset entries must expose a metadata mapping.")
        rows.append(
            {
                cfg.data.key_column: metadata.get(cfg.data.key_column),
                cfg.data.split_column: metadata.get(cfg.data.split_column),
                **{name: metadata.get(name) for name in cfg.model.features},
                "path": getattr(sample, "path", None),
                "token_start": getattr(sample, "start", None),
            }
        )
    return pd.DataFrame(rows)


def _load_rows_from_kaldi_manifest(cfg: BaselineConfig) -> pd.DataFrame:
    root = Path(cfg.data.kaldi_data_root)
    manifest_path = Path(cfg.data.kaldi_manifest)
    with manifest_path.open() as file_obj:
        manifest = json.load(file_obj)
    splits = manifest.get("splits")
    if not isinstance(splits, dict) or not splits:
        raise ValueError("Kaldi manifest must contain a non-empty 'splits' mapping.")

    frames = []
    for split_name, split_spec in splits.items():
        if not isinstance(split_spec, dict) or not split_spec.get("manifest"):
            raise ValueError(f"Kaldi manifest split {split_name!r} must define a manifest CSV.")
        split_manifest = root / Path(str(split_spec["manifest"]))
        frame = pd.read_csv(split_manifest, dtype={cfg.data.key_column: "string"})
        if cfg.data.split_column not in frame.columns:
            frame[cfg.data.split_column] = split_name
        frames.append(frame)
    return pd.concat(frames, axis=0, ignore_index=True) if frames else pd.DataFrame()


def _key_normalizer(cfg: BaselineConfig) -> Callable[[Any, str], str]:
    if cfg.finetune.task.type == "survival":
        return normalize_survival_key
    return normalize_multilabel_key


def _raw_split_value(value: Any) -> str:
    return "" if pd.isna(value) else str(value).strip()


def _validate_loaded_split_key_uniqueness(
    frame: pd.DataFrame,
    cfg: BaselineConfig,
    normalize_key: Callable[[Any, str], str],
    loaded_splits: list[str],
) -> None:
    loaded = {str(split).strip() for split in loaded_splits}
    key_splits: dict[str, set[str]] = {}
    for _, row in frame.iterrows():
        split = _raw_split_value(row[cfg.data.split_column])
        if split not in loaded:
            continue
        key = normalize_key(row[cfg.data.key_column], cfg.data.key_column)
        key_splits.setdefault(key, set()).add(split)

    for key, splits in key_splits.items():
        if len(splits) > 1:
            split_list = ", ".join(sorted(splits))
            raise ValueError(f"Sex/age baseline key {key!r} appears in multiple loaded splits: {split_list}.")


def _parse_split(value: Any, column: str) -> str:
    if pd.isna(value):
        raise ValueError(f"Index split column {column!r} contains a missing value.")
    split = str(value).strip()
    if not split:
        raise ValueError(f"Index split column {column!r} contains an empty value.")
    return split


def _parse_continuous(value: Any, name: str) -> float:
    if pd.isna(value):
        raise ValueError(f"Index {name} column contains a missing value.")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Index {name} value is not numeric: {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"Index {name} value must be finite: {value!r}")
    return number


def _parse_sex(value: Any) -> int:
    encoded = _encode_binary_label(value)
    if encoded not in (0, 1):
        raise ValueError(f"Index sex value must encode female/0 or male/1: {value!r}")
    return int(encoded)


def _validate_duplicate_metadata(frame: pd.DataFrame, features: list[str]) -> None:
    for key, group in frame.groupby("_baseline_key", sort=False):
        splits = set(group["_baseline_split"].tolist())
        if len(splits) != 1:
            raise ValueError(f"Duplicate key {key!r} has conflicting split values.")
        for name in features:
            values = np.asarray(group[f"_baseline_{name}"].tolist(), dtype=np.float64)
            if not np.allclose(values, values[0], rtol=0.0, atol=1e-6):
                raise ValueError(f"Duplicate key {key!r} has conflicting {name} values.")


def _require_label_key(key: str, labels: dict[str, np.ndarray], split: str) -> str:
    if key not in labels:
        raise ValueError(f"Index key {key!r} from split {split!r} is missing from label sidecars.")
    return key


def _collate_records(records: list[BaselineRecord]) -> dict[str, Any]:
    batch: dict[str, Any] = {
        "key": [record.key for record in records],
        "path": [record.path for record in records],
        "token_start": torch.tensor([record.token_start for record in records], dtype=torch.long),
        "features": {
            name: torch.tensor(
                [record.features[name] for record in records], dtype=torch.long if name == "sex" else torch.float32
            )
            for name in records[0].features
        },
    }
    first = records[0]
    if first.event_time is not None:
        batch["event_time"] = torch.as_tensor(np.stack([record.event_time for record in records]), dtype=torch.float32)
        batch["is_event"] = torch.as_tensor(np.stack([record.is_event for record in records]), dtype=torch.float32)
        batch["has_label"] = torch.as_tensor(np.stack([record.has_label for record in records]), dtype=torch.float32)
    else:
        batch["disease_label"] = torch.as_tensor(
            np.stack([record.disease_label for record in records]), dtype=torch.float32
        )
        batch["has_label"] = torch.as_tensor(np.stack([record.has_label for record in records]), dtype=torch.float32)
    return batch
