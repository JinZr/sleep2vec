"""Metadata-only loader that gives the covariate baseline the signal model's cohort.

The baseline reads the same index CSV, preset pickle or Kaldi manifest as the matching signal run, applies the same
split, dataset-name and required-covariate filters, and keeps one record per task key. Labels come from the preset when
one is configured and from the task sidecars otherwise, as in the signal loaders. It never opens NPZ or Kaldi
feature files, so duration, NPZ-validity and Kaldi channel drops that the signal loader applies to a raw index are not
reproduced here; a shared preset already carries them.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import logging
import math
from pathlib import Path
import pickle
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from data.kaldi_psg_dataset import _first_present
from data.metadata import process_metadata, required_covariate_is_valid
from data.multilabel import load_multilabel_disease_columns, load_multilabel_label_table, normalize_multilabel_key
from data.survival import load_survival_disease_columns, load_survival_label_table, normalize_survival_key

from .config import BaselineConfig, covariate_task_config

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BaselineRecord:
    key: str
    metadata: dict[str, Any]
    event_time: np.ndarray | None = None
    is_event: np.ndarray | None = None
    disease_label: np.ndarray | None = None
    has_label: np.ndarray | None = None
    sample_index: int = 0


class SexAgeDataset(Dataset):
    def __init__(self, records: list[BaselineRecord], *, task_type: str, label_names: list[str]) -> None:
        self.records = list(records)
        self.task_type = task_type
        self.label_names = list(label_names)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> BaselineRecord:
        return replace(self.records[index], sample_index=index)


def load_split_dataset(
    cfg: BaselineConfig,
    split: str,
    *,
    sources: list[str] | None,
    loaded_splits: list[str] | None = None,
) -> SexAgeDataset:
    """One record per task key in ``split``, restricted to rows whose source contains one of ``sources``."""
    task_cfg = covariate_task_config(cfg)
    key_column = task_cfg.key_column
    rows = _load_metadata_rows(cfg)
    present = set().union(*(row.keys() for row in rows))
    missing = sorted({"split", key_column, *task_cfg.covariates} - present)
    if missing:
        raise ValueError(f"Sex/age baseline metadata is missing required columns: {missing}")
    normalize_key = normalize_survival_key if cfg.finetune.task.type == "survival" else normalize_multilabel_key
    if loaded_splits:
        _validate_loaded_split_key_uniqueness(rows, key_column, normalize_key, loaded_splits)

    rows = _select_rows(rows, split, sources, task_cfg.covariates)
    records = _collapse_by_key(
        [BaselineRecord(key=normalize_key(row[key_column], key_column), metadata=row) for row in rows],
        task_cfg.covariates,
    )

    survival = cfg.finetune.task.type == "survival"
    label_fields = ("event_time", "is_event", "has_label") if survival else ("disease_label", "has_label")
    output_dim = cfg.finetune.task.output_dim
    if cfg.data.backend == "npz" and cfg.data.finetune_preset_path:
        # A preset embeds the labels its signal run trains on; read them as the signal preset path does.
        load_names = load_survival_disease_columns if survival else load_multilabel_disease_columns
        label_names = load_names(task_cfg.disease_columns_index)
        labels = {
            field: {record.key: _preset_label(record, field, output_dim) for record in records}
            for field in label_fields
        }
    else:
        load_table = load_survival_label_table if survival else load_multilabel_label_table
        table = load_table(task_cfg, expected_output_dim=output_dim)
        assert table is not None
        label_names = table.label_names
        labels = {field: getattr(table, field) for field in label_fields}
    records = [
        replace(
            record,
            **{field: labels[field][_require_label_key(record.key, labels[field], split)] for field in label_fields},
        )
        for record in records
    ]
    return SexAgeDataset(records, task_type=cfg.finetune.task.type, label_names=label_names)


def make_dataloader(
    dataset: SexAgeDataset,
    *,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    drop_last: bool = False,
    sampler=None,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        drop_last=drop_last,
        num_workers=num_workers,
        collate_fn=_collate_records,
    )


def _load_metadata_rows(cfg: BaselineConfig) -> list[dict[str, Any]]:
    # The signal runtime's data-source rules (sleep2vec.common.apply_data_backend_args and the preset-or-index read).
    if cfg.data.backend == "kaldi":
        if not cfg.data.kaldi_data_root or not cfg.data.kaldi_manifest:
            raise ValueError("Kaldi backend requires explicit kaldi_data_root and kaldi_manifest.")
        if cfg.data.finetune_preset_path:
            raise ValueError("Kaldi backend uses manifest.json; legacy NPZ preset pickles are unsupported.")
        return _load_rows_from_kaldi_manifest(cfg)
    if cfg.data.finetune_preset_path:
        with Path(cfg.data.finetune_preset_path).open("rb") as file_obj:
            samples = pickle.load(file_obj)
        return [dict(sample.metadata) for sample in samples]
    index = cfg.data.finetune_data_index
    if not index:
        raise ValueError("NPZ backend requires finetune_preset_path or finetune_data_index.")
    key_column = covariate_task_config(cfg).key_column
    # Same parsing as the signal index reader: string keys and paths, and the index path as the default source.
    frame = pd.read_csv(
        index, low_memory=False, converters={"path": str, "source": lambda value: value or None, key_column: str}
    )
    if "source" not in frame.columns:
        frame["source"] = str(index)
    else:
        frame["source"] = frame["source"].where(frame["source"].notna(), str(index))
    return frame.to_dict("records")


def _load_rows_from_kaldi_manifest(cfg: BaselineConfig) -> list[dict[str, Any]]:
    root = Path(cfg.data.kaldi_data_root)
    with Path(cfg.data.kaldi_manifest).open() as file_obj:
        manifest = json.load(file_obj)
    splits = manifest.get("splits")
    if not isinstance(splits, dict) or not splits:
        raise ValueError("Kaldi manifest must contain a non-empty 'splits' mapping.")

    key_column = covariate_task_config(cfg).key_column
    rows = []
    for split_name, split_spec in splits.items():
        if not isinstance(split_spec, dict) or not split_spec.get("manifest"):
            raise ValueError(f"Kaldi manifest split {split_name!r} must define a manifest CSV.")
        frame = pd.read_csv(root / Path(str(split_spec["manifest"])), dtype={key_column: "string"})
        # As KaldiPSGDataset: a split's rows are those its own manifest lists under that split name.
        frame = frame[frame["split"].isin([split_name])]
        for _, row in frame.iterrows():
            metadata = row.to_dict()
            # Same source fallback as KaldiPSGDataset, so dataset-name filters select the same rows.
            metadata["source"] = _first_present(row, ("source", "dataset", "sample_source"), "nan")
            rows.append(metadata)
    return rows


def _select_rows(
    rows: list[dict[str, Any]], split: str, sources: list[str] | None, covariates: list[str]
) -> list[dict[str, Any]]:
    """Rows of ``split`` that pass the signal loader's dataset-name and required-covariate filters."""
    selected = []
    invalid = 0
    for row in rows:
        if _raw_split_value(row.get("split")) != split:
            continue
        source = row.get("source")
        if sources and (source is None or not any(name in str(source) for name in sources)):
            continue
        if not all(required_covariate_is_valid(name, row.get(name)) for name in covariates):
            invalid += 1
            continue
        selected.append(row)
    if invalid:
        logger.info("Dropped %d %s rows with missing or invalid covariates %s.", invalid, split, covariates)
    return selected


def _collapse_by_key(records: list[BaselineRecord], covariates: list[str]) -> list[BaselineRecord]:
    """Keep the first record per key after checking that every record of a key encodes the same covariates."""
    if not records:
        return []
    encoded = process_metadata(records, [])
    first_index: dict[str, int] = {}
    for index, record in enumerate(records):
        first = first_index.setdefault(record.key, index)
        for name in covariates:
            if not math.isclose(float(encoded[name][index]), float(encoded[name][first]), rel_tol=0.0, abs_tol=1e-6):
                raise ValueError(f"Duplicate key {record.key!r} has conflicting {name} values.")
    return [records[index] for index in first_index.values()]


def _raw_split_value(value: Any) -> str:
    return "" if pd.isna(value) else str(value).strip()


def _validate_loaded_split_key_uniqueness(
    rows: list[dict[str, Any]],
    key_column: str,
    normalize_key: Callable[[Any, str], str],
    loaded_splits: list[str],
) -> None:
    loaded = {str(split).strip() for split in loaded_splits}
    key_splits: dict[str, set[str]] = {}
    for row in rows:
        split = _raw_split_value(row.get("split"))
        if split not in loaded:
            continue
        key_splits.setdefault(normalize_key(row[key_column], key_column), set()).add(split)

    for key, splits in key_splits.items():
        if len(splits) > 1:
            split_list = ", ".join(sorted(splits))
            raise ValueError(f"Sex/age baseline key {key!r} appears in multiple loaded splits: {split_list}.")


def _preset_label(record: BaselineRecord, field: str, output_dim: int) -> np.ndarray:
    if field not in record.metadata:
        raise ValueError(f"Preset is missing metadata field {field!r}; regenerate presets with label sidecars.")
    value = np.asarray(record.metadata[field], dtype=np.float32)
    if value.shape != (output_dim,):
        raise ValueError(f"Preset metadata field {field!r} has shape {value.shape}, expected ({output_dim},).")
    return value


def _require_label_key(key: str, labels: dict[str, np.ndarray], split: str) -> str:
    if key not in labels:
        raise ValueError(f"Key {key!r} from split {split!r} is missing from the task labels.")
    return key


def _collate_records(records: list[BaselineRecord]) -> dict[str, Any]:
    batch: dict[str, Any] = {
        "key": [record.key for record in records],
        "sample_index": torch.tensor([record.sample_index for record in records], dtype=torch.long),
        # The signal collate's covariate encoding, so the shared covariate module sees identical inputs.
        "metadata": process_metadata(records, []),
    }
    first = records[0]
    if first.event_time is not None:
        batch["event_time"] = torch.as_tensor(np.stack([record.event_time for record in records]), dtype=torch.float32)
        batch["is_event"] = torch.as_tensor(np.stack([record.is_event for record in records]), dtype=torch.float32)
    else:
        batch["disease_label"] = torch.as_tensor(
            np.stack([record.disease_label for record in records]), dtype=torch.float32
        )
    batch["has_label"] = torch.as_tensor(np.stack([record.has_label for record in records]), dtype=torch.float32)
    return batch
