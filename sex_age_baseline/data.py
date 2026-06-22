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
    age: float
    sex: int
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


def load_split_dataset(cfg: BaselineConfig, split: str) -> SexAgeDataset:
    frame = _load_metadata_frame(cfg)
    selected = frame[frame["_baseline_split"] == str(split)]

    if cfg.finetune.task.type == "survival":
        labels = load_survival_label_table(cfg.finetune.survival, expected_output_dim=cfg.finetune.task.output_dim)
        assert labels is not None
        records = [
            BaselineRecord(
                key=row["_baseline_key"],
                age=row["_baseline_age"],
                sex=row["_baseline_sex"],
                event_time=labels.event_time[_require_label_key(row["_baseline_key"], labels.event_time, split)],
                is_event=labels.is_event[row["_baseline_key"]],
                has_label=labels.has_label[row["_baseline_key"]],
            )
            for _, row in selected.iterrows()
        ]
        return SexAgeDataset(records, task_type=cfg.finetune.task.type, label_names=labels.label_names)

    labels = load_multilabel_label_table(cfg.finetune.multilabel, expected_output_dim=cfg.finetune.task.output_dim)
    assert labels is not None
    records = [
        BaselineRecord(
            key=row["_baseline_key"],
            age=row["_baseline_age"],
            sex=row["_baseline_sex"],
            disease_label=labels.disease_label[_require_label_key(row["_baseline_key"], labels.disease_label, split)],
            has_label=labels.has_label[row["_baseline_key"]],
        )
        for _, row in selected.iterrows()
    ]
    return SexAgeDataset(records, task_type=cfg.finetune.task.type, label_names=labels.label_names)


def make_dataloader(
    dataset: SexAgeDataset,
    *,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=_collate_records,
    )


def _load_metadata_frame(cfg: BaselineConfig) -> pd.DataFrame:
    if cfg.data.backend == "npz":
        if cfg.data.finetune_preset_path:
            frame = _load_rows_from_npz_preset(cfg)
        else:
            frame = _load_rows_from_npz_index(cfg)
    elif cfg.data.backend == "kaldi":
        frame = _load_rows_from_kaldi_manifest(cfg)
    else:
        raise ValueError(f"Unsupported sex_age_baseline data backend: {cfg.data.backend}")

    required_columns = {cfg.data.key_column, cfg.data.split_column, "age", "sex"}
    missing = sorted(required_columns - set(frame.columns))
    if missing:
        raise ValueError(f"Sex/age baseline metadata is missing required columns: {missing}")
    normalize_key = _key_normalizer(cfg)

    frame = frame.copy()
    frame["_baseline_key"] = [normalize_key(value, cfg.data.key_column) for value in frame[cfg.data.key_column]]
    frame["_baseline_split"] = [_parse_split(value, cfg.data.split_column) for value in frame[cfg.data.split_column]]
    frame["_baseline_age"] = [_parse_age(value) for value in frame["age"]]
    frame["_baseline_sex"] = [_parse_sex(value) for value in frame["sex"]]
    _validate_duplicate_metadata(frame)
    return frame.drop_duplicates("_baseline_key", keep="first")


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
                "age": metadata.get("age"),
                "sex": metadata.get("sex"),
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
        frames.append(pd.read_csv(split_manifest, dtype={cfg.data.key_column: "string"}))
    return pd.concat(frames, axis=0, ignore_index=True) if frames else pd.DataFrame()


def _key_normalizer(cfg: BaselineConfig) -> Callable[[Any, str], str]:
    if cfg.finetune.task.type == "survival":
        return normalize_survival_key
    return normalize_multilabel_key


def _parse_split(value: Any, column: str) -> str:
    if pd.isna(value):
        raise ValueError(f"Index split column {column!r} contains a missing value.")
    split = str(value).strip()
    if not split:
        raise ValueError(f"Index split column {column!r} contains an empty value.")
    return split


def _parse_age(value: Any) -> float:
    if pd.isna(value):
        raise ValueError("Index age column contains a missing value.")
    try:
        age = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Index age value is not numeric: {value!r}") from exc
    if not math.isfinite(age) or age < 0:
        raise ValueError(f"Index age value must be finite and non-negative: {value!r}")
    return age


def _parse_sex(value: Any) -> int:
    encoded = _encode_binary_label(value)
    if encoded not in (0, 1):
        raise ValueError(f"Index sex value must encode female/0 or male/1: {value!r}")
    return int(encoded)


def _validate_duplicate_metadata(frame: pd.DataFrame) -> None:
    for key, group in frame.groupby("_baseline_key", sort=False):
        splits = set(group["_baseline_split"].tolist())
        if len(splits) != 1:
            raise ValueError(f"Duplicate key {key!r} has conflicting split values.")
        sexes = set(int(value) for value in group["_baseline_sex"].tolist())
        if len(sexes) != 1:
            raise ValueError(f"Duplicate key {key!r} has conflicting sex values.")
        ages = np.asarray(group["_baseline_age"].tolist(), dtype=np.float64)
        if not np.allclose(ages, ages[0], rtol=0.0, atol=1e-6):
            raise ValueError(f"Duplicate key {key!r} has conflicting age values.")


def _require_label_key(key: str, labels: dict[str, np.ndarray], split: str) -> str:
    if key not in labels:
        raise ValueError(f"Index key {key!r} from split {split!r} is missing from label sidecars.")
    return key


def _collate_records(records: list[BaselineRecord]) -> dict[str, Any]:
    batch: dict[str, Any] = {
        "key": [record.key for record in records],
        "age": torch.tensor([record.age for record in records], dtype=torch.float32),
        "sex": torch.tensor([record.sex for record in records], dtype=torch.long),
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
