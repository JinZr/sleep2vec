from itertools import combinations
import json
import pickle
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from data.default_dataset import SampleIndex
from sex_age_baseline.data import BaselineRecord, _collate_records, _load_metadata_frame, load_split_dataset


def config(path, features, deduplicate=True, preset=False):
    return SimpleNamespace(
        model=SimpleNamespace(features=features),
        finetune=SimpleNamespace(task=SimpleNamespace(type="survival")),
        data=SimpleNamespace(
            backend="npz",
            finetune_preset_path=str(path) if preset else None,
            finetune_data_index=None if preset else str(path),
            key_column="eid",
            split_column="split",
            deduplicate_by_key=deduplicate,
        ),
    )


@pytest.mark.parametrize("features", [list(c) for n in (1, 2, 3) for c in combinations(("bmi", "sex", "age"), n)])
def test_selected_features_only(tmp_path, features):
    path = tmp_path / "index.csv"
    pd.DataFrame(
        [{"eid": "1", "split": "train", **{n: {"age": -1, "sex": "female", "bmi": 24}[n] for n in features}}]
    ).to_csv(path, index=False)
    frame = _load_metadata_frame(config(path, features), split="train")
    assert len(frame) == 1
    assert [c.removeprefix("_baseline_") for c in frame if c in {f"_baseline_{n}" for n in features}] == features


def test_windows_keep_order_multiplicity_and_never_read_signals(tmp_path, monkeypatch):
    path = tmp_path / "preset.pickle"
    samples = [
        SampleIndex(
            id=i,
            path="absent-signal.npz",
            start=start,
            end=start + 10,
            metadata={"eid": key, "split": "train", "bmi": 22},
        )
        for i, (key, start) in enumerate([("2", 10), ("1", 0), ("2", 20), ("2", 10)])
    ]
    with path.open("wb") as f:
        pickle.dump(samples, f)
    monkeypatch.setattr(np, "load", lambda *a, **kw: pytest.fail("signal read"))
    frame = _load_metadata_frame(config(path, ["bmi"], False, True), split="train")
    assert frame["_baseline_key"].tolist() == ["2", "1", "2", "2"]
    assert frame["_baseline_token_start"].tolist() == [10, 0, 20, 10]
    participants = _load_metadata_frame(config(path, ["bmi"], True, True), split="train")
    assert participants["_baseline_key"].tolist() == ["2", "1"]


def test_invalid_covariates_report_counts(tmp_path):
    path = tmp_path / "index.csv"
    pd.DataFrame({"eid": ["1", "2"], "split": ["train"] * 2, "bmi": [np.inf, np.nan]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="'bmi': 2"):
        _load_metadata_frame(config(path, ["bmi"]), split="train")


def test_window_identity_required(tmp_path):
    path = tmp_path / "index.csv"
    pd.DataFrame([{"eid": "1", "split": "train", "age": 45}]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="path and token_start"):
        _load_metadata_frame(config(path, ["age"], False), split="train")


def test_duplicate_and_cross_split_consistency(tmp_path):
    path = tmp_path / "index.csv"
    pd.DataFrame({"eid": ["1", "1"], "split": ["train", "test"], "age": [45, 45]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="multiple loaded splits"):
        _load_metadata_frame(config(path, ["age"]), split="train", loaded_splits=["train", "test"])
    pd.DataFrame({"eid": ["1", "1"], "split": ["train"] * 2, "age": [45, 46]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="conflicting age"):
        _load_metadata_frame(config(path, ["age"]), split="train")


def test_collate_features_and_identity():
    records = [
        BaselineRecord(
            key="1",
            features={"sex": 1, "bmi": 23.5},
            path="record",
            token_start=10,
            event_time=np.array([2]),
            is_event=np.array([1]),
            has_label=np.array([1]),
        )
    ]
    batch = _collate_records(records)
    assert list(batch["features"]) == ["sex", "bmi"]
    assert batch["features"]["sex"].dtype.is_floating_point is False
    assert batch["path"] == ["record"]
    assert batch["token_start"].tolist() == [10]


def test_kaldi_manifest_selected_bmi_and_window_identity(tmp_path):
    pd.DataFrame([{"eid": "1", "bmi": 24, "path": "not-opened.ark", "token_start": 3}]).to_csv(
        tmp_path / "train.csv", index=False
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"splits": {"train": {"manifest": "train.csv"}}}))
    cfg = config(manifest, ["bmi"], False)
    cfg.data.backend = "kaldi"
    cfg.data.kaldi_manifest = str(manifest)
    cfg.data.kaldi_data_root = str(tmp_path)
    frame = _load_metadata_frame(cfg, split="train")
    assert frame["_baseline_bmi"].tolist() == [24]
    assert frame["_baseline_token_start"].tolist() == [3]


@pytest.mark.parametrize("task", ["survival", "multilabel_classification"])
def test_window_labels_attach_without_changing_sample_order(tmp_path, task):
    path = tmp_path / "index.csv"
    pd.DataFrame(
        {
            "eid": ["2", "1", "2"],
            "split": ["train"] * 3,
            "age": [50, 45, 50],
            "path": ["b", "a", "b"],
            "token_start": [0, 0, 10],
        }
    ).to_csv(path, index=False)
    cfg = config(path, ["age"], False)
    cfg.finetune.task.type = task
    cfg.finetune.task.output_dim = 1
    columns = tmp_path / "columns.txt"
    columns.write_text("disease\n")
    fields = {"has_label_index": [1, 1]}
    fields.update(
        {"event_time_index": [11, 22], "is_event_index": [1, 0]} if task == "survival" else {"label_index": [1, 0]}
    )
    sidecars = {"key_column": "eid", "disease_columns_index": str(columns)}
    for field, values in fields.items():
        target = tmp_path / (field + ".csv")
        pd.DataFrame({"eid": ["1", "2"], "disease": values}).to_csv(target, index=False)
        sidecars[field] = str(target)
    setattr(cfg.finetune, "survival" if task == "survival" else "multilabel", SimpleNamespace(**sidecars))
    dataset = load_split_dataset(cfg, "train")
    assert [r.key for r in dataset.records] == ["2", "1", "2"]
    batch = _collate_records(dataset.records)
    assert batch["has_label"].tolist() == [[1], [1], [1]]
    assert batch["event_time" if task == "survival" else "disease_label"].tolist() == (
        [[22], [11], [22]] if task == "survival" else [[0], [1], [0]]
    )
