from __future__ import annotations

import importlib
import json
from pathlib import Path
import pickle
import sys

import pandas as pd
import pytest


@pytest.mark.parametrize(
    "module_name",
    [
        "preprocess.merge_dataset_presets",
        "sleep2expert.preprocess.merge_dataset_presets",
        "sleep2vec2.preprocess.merge_dataset_presets",
    ],
)
def test_merge_dataset_presets_writes_progress(tmp_path: Path, monkeypatch, module_name: str):
    module = importlib.import_module(module_name)
    input_a = tmp_path / "a.pkl"
    input_b = tmp_path / "b.pkl"
    output = tmp_path / "merged.pkl"
    input_a.write_bytes(pickle.dumps([1]))
    input_b.write_bytes(pickle.dumps([2]))
    monkeypatch.setattr(
        sys,
        "argv",
        ["merge_dataset_presets.py", "--inputs", str(input_a), str(input_b), "--output", str(output)],
    )

    module.main()

    progress = json.loads((tmp_path / "status" / "progress.json").read_text())
    assert progress["task"] == "merge_dataset_presets"
    assert progress["status"] == "completed"
    assert progress["processed"] == 2


@pytest.mark.parametrize(
    "module_name",
    [
        "preprocess.split_index_by_dataset",
        "sleep2expert.preprocess.split_index_by_dataset",
        "sleep2vec2.preprocess.split_index_by_dataset",
    ],
)
def test_split_index_by_dataset_writes_progress(tmp_path: Path, monkeypatch, module_name: str):
    module = importlib.import_module(module_name)
    input_csv = tmp_path / "input.csv"
    output_csv = tmp_path / "split.csv"
    pd.DataFrame([{"dataset": "a"}, {"dataset": "a"}, {"dataset": "b"}]).to_csv(input_csv, index=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "split_index_by_dataset.py",
            "--input",
            str(input_csv),
            "--output",
            str(output_csv),
            "--n-val",
            "1",
            "--n-test",
            "0",
        ],
    )

    module.main()

    progress = json.loads((tmp_path / "status" / "progress.json").read_text())
    assert progress["task"] == "split_index_by_dataset"
    assert progress["status"] == "completed"
    assert progress["processed"] == 3


@pytest.mark.parametrize(
    "module_name",
    [
        "preprocess.mask_missing_stats",
        "sleep2expert.preprocess.mask_missing_stats",
        "sleep2vec2.preprocess.mask_missing_stats",
    ],
)
def test_mask_missing_stats_writes_progress(tmp_path: Path, monkeypatch, module_name: str):
    module = importlib.import_module(module_name)
    input_csv = tmp_path / "input.csv"
    out_prefix = tmp_path / "stats" / "missing"
    pd.DataFrame(
        [
            {"dataset": "a", "eeg_mask": "true", "ppg_mask": "yes"},
            {"dataset": "b", "eeg_mask": "1.0", "ppg_mask": "0"},
        ]
    ).to_csv(input_csv, index=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mask_missing_stats.py",
            "--csv",
            str(input_csv),
            "--out-prefix",
            str(out_prefix),
            "--chunksize",
            "1",
        ],
    )

    module.main()

    progress = json.loads((out_prefix.parent / "status" / "progress.json").read_text())
    assert progress["task"] == "mask_missing_stats"
    assert progress["status"] == "completed"
    assert progress["processed"] == 2
    overall = pd.read_csv(f"{out_prefix}_overall.csv").set_index("mask_col")
    assert int(overall.loc["eeg_mask", "missing_rows"]) == 0
    assert int(overall.loc["ppg_mask", "missing_rows"]) == 1
