"""The baseline reads the signal run's index, preset or Kaldi manifest and keeps one record per task key."""

import json
import logging
import pickle
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from data.default_dataset import SampleIndex
from sex_age_baseline.data import _collate_records, load_split_dataset, validate_disjoint_split_keys


def _sidecars(tmp_path, task, keys=("1", "2", "3")):
    columns = tmp_path / "columns.txt"
    columns.write_text("disease\n")
    fields = {"has_label_index": [1] * len(keys)}
    if task == "survival":
        fields.update(
            {"event_time_index": [10 * (i + 1) for i in range(len(keys))], "is_event_index": [1, 0, 1][: len(keys)]}
        )
    else:
        fields["label_index"] = [1, 0, 1][: len(keys)]
    sidecars = {"key_column": "eid", "disease_columns_index": str(columns)}
    for field, values in fields.items():
        target = tmp_path / f"{field}.csv"
        pd.DataFrame({"eid": list(keys), "disease": values}).to_csv(target, index=False)
        sidecars[field] = str(target)
    return sidecars


def _config(tmp_path, *, index=None, preset=None, kaldi=None, task="survival", covariates=("age", "sex"), **data):
    task_cfg = SimpleNamespace(**_sidecars(tmp_path, task), covariates=list(covariates))
    return SimpleNamespace(
        finetune=SimpleNamespace(
            task=SimpleNamespace(type=task, output_dim=1),
            survival=task_cfg if task == "survival" else None,
            multilabel=task_cfg if task != "survival" else None,
        ),
        data=SimpleNamespace(
            backend="kaldi" if kaldi else "npz",
            finetune_data_index=None if index is None else str(index),
            finetune_preset_path=None if preset is None else str(preset),
            kaldi_data_root=None if kaldi is None else str(kaldi.parent),
            kaldi_manifest=None if kaldi is None else str(kaldi),
            **data,
        ),
    )


def _index(tmp_path, rows):
    path = tmp_path / "index.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _row(eid, *, split="train", age=45, sex="female", path=None, **extra):
    return {"path": path or f"{eid}.npz", "split": split, "duration": 600, "eid": eid, "age": age, "sex": sex, **extra}


def test_index_windows_collapse_to_one_record_per_key_without_reading_signals(tmp_path, monkeypatch):
    rows = [_row("2", path="b0.npz"), _row("1"), _row("2", path="b1.npz"), _row("3", split="val")]
    monkeypatch.setattr(np, "load", lambda *args, **kwargs: pytest.fail("signal read"))

    dataset = load_split_dataset(_config(tmp_path, index=_index(tmp_path, rows)), "train", sources=None)

    assert [record.key for record in dataset.records] == ["2", "1"]
    assert dataset.label_names == ["disease"]
    assert [record.event_time.tolist() for record in dataset.records] == [[20.0], [10.0]]


@pytest.mark.parametrize("task", ["survival", "multilabel_classification"])
def test_preset_supplies_rows_and_labels_like_the_signal_preset_path(tmp_path, task):
    labels = (
        {"event_time": [7.0], "is_event": [1.0], "has_label": [1.0]}
        if task == "survival"
        else {"disease_label": [1.0], "has_label": [1.0]}
    )
    samples = [
        SampleIndex(id=i, path="absent.npz", start=0, end=10, metadata={**_row(key), **labels})
        for i, key in enumerate(["9", "9", "8"])
    ]
    preset = tmp_path / "preset.pkl"
    preset.write_bytes(pickle.dumps(samples))
    cfg = _config(tmp_path, preset=preset, task=task)
    # The preset carries the labels; the sidecars do not even contain these keys.
    dataset = load_split_dataset(cfg, "train", sources=None)

    assert [record.key for record in dataset.records] == ["9", "8"]
    field = "event_time" if task == "survival" else "disease_label"
    assert [getattr(record, field).tolist() for record in dataset.records] == [[labels[field][0]]] * 2


def test_preset_without_labels_asks_for_regeneration(tmp_path):
    preset = tmp_path / "preset.pkl"
    preset.write_bytes(pickle.dumps([SampleIndex(id=0, path="x", start=0, end=1, metadata=_row("1"))]))

    with pytest.raises(ValueError, match="regenerate presets"):
        load_split_dataset(_config(tmp_path, preset=preset), "train", sources=None)


def test_dataset_names_are_source_substrings_and_default_to_the_index_path(tmp_path):
    rows = [_row("1", source="shhs1"), _row("2", source="mesa"), _row("3")]
    index = _index(tmp_path, rows)

    selected = load_split_dataset(_config(tmp_path, index=index), "train", sources=["shhs"])
    assert [record.key for record in selected.records] == ["1"]
    # A row without a source falls back to the index path, as the signal index reader does.
    by_path = load_split_dataset(_config(tmp_path, index=index), "train", sources=["index.csv"])
    assert [record.key for record in by_path.records] == ["3"]


def test_kaldi_rows_use_the_signal_source_fallback(tmp_path):
    pd.DataFrame([_row("1", dataset="shhs"), _row("2", dataset="mesa")]).to_csv(tmp_path / "train.csv", index=False)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"splits": {"train": {"manifest": "train.csv"}}}))

    dataset = load_split_dataset(_config(tmp_path, kaldi=manifest), "train", sources=["mesa"])

    assert [record.key for record in dataset.records] == ["2"]


@pytest.mark.parametrize(
    ("values", "kept"),
    [
        ({"bmi": [24.0, 25.0], "bmi_missing": [0, 1]}, ["1", "2"]),
        ({"bmi": [24.0, 25.0], "bmi_missing": [0.5, 2]}, []),
        ({"bmi": [np.inf, np.nan], "bmi_missing": [0, 0]}, []),
        ({"bmi": [24.0, np.nan], "bmi_missing": [0, 1]}, ["1"]),
    ],
)
def test_rows_with_invalid_covariates_are_dropped_like_the_signal_filter(tmp_path, caplog, values, kept):
    rows = [_row(key, bmi=values["bmi"][i], bmi_missing=values["bmi_missing"][i]) for i, key in enumerate(["1", "2"])]
    cfg = _config(tmp_path, index=_index(tmp_path, rows), covariates=("bmi", "bmi_missing"))

    with caplog.at_level(logging.INFO, logger="sex_age_baseline.data"):
        dataset = load_split_dataset(cfg, "train", sources=None)

    assert [record.key for record in dataset.records] == kept
    if len(kept) < 2:
        assert f"Dropped {2 - len(kept)} train rows" in caplog.text


def test_missing_covariate_column_fails(tmp_path):
    cfg = _config(tmp_path, index=_index(tmp_path, [_row("1")]), covariates=("age", "bmi"))

    with pytest.raises(ValueError, match=r"missing required columns: \['bmi'\]"):
        load_split_dataset(cfg, "train", sources=None)


def test_conflicting_duplicates_and_cross_split_keys_fail(tmp_path):
    reused = _config(tmp_path, index=_index(tmp_path, [_row("1"), _row("1", split="test")]))
    datasets = {split: load_split_dataset(reused, split, sources=None) for split in ("train", "test")}
    with pytest.raises(ValueError, match="multiple loaded splits"):
        validate_disjoint_split_keys(datasets)

    conflicting = _index(tmp_path, [_row("1", age=45), _row("1", age=46)])
    with pytest.raises(ValueError, match="conflicting age"):
        load_split_dataset(_config(tmp_path, index=conflicting), "train", sources=None)


def test_index_key_missing_from_sidecars_fails(tmp_path):
    cfg = _config(tmp_path, index=_index(tmp_path, [_row("404")]))

    with pytest.raises(ValueError, match="'404'.*missing from the task labels"):
        load_split_dataset(cfg, "train", sources=None)


def test_collate_uses_the_signal_metadata_encoding(tmp_path):
    rows = [_row("1", age=45.9, sex="male", bmi=23.5, bmi_missing=0), _row("2", age=60, sex=0, bmi=30, bmi_missing=1)]
    cfg = _config(tmp_path, index=_index(tmp_path, rows), covariates=("age", "sex", "bmi", "bmi_missing"))
    dataset = load_split_dataset(cfg, "train", sources=None)

    batch = _collate_records(dataset.records)

    metadata = batch["metadata"]
    # process_metadata truncates age to whole years, exactly as the signal collate does.
    assert metadata["age"].tolist() == [45.0, 60.0]
    assert metadata["sex"].dtype == torch.long and metadata["sex"].tolist() == [1, 0]
    assert metadata["bmi"].tolist() == [23.5, 30.0]
    assert metadata["bmi_missing"].dtype == torch.long and metadata["bmi_missing"].tolist() == [0, 1]
    assert batch["key"] == ["1", "2"]
    assert batch["event_time"].tolist() == [[10.0], [20.0]]


def test_the_data_source_is_required_when_the_rows_are_read(tmp_path):
    with pytest.raises(ValueError, match="requires finetune_preset_path or finetune_data_index"):
        load_split_dataset(_config(tmp_path), "train", sources=None)

    cfg = _config(tmp_path, kaldi=tmp_path / "manifest.json")
    cfg.data.finetune_preset_path = str(tmp_path / "preset.pkl")
    with pytest.raises(ValueError, match="legacy NPZ preset pickles are unsupported"):
        load_split_dataset(cfg, "train", sources=None)
