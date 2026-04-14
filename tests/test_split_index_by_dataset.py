from __future__ import annotations

import sys

import pandas as pd

from preprocess.split_index_by_dataset import (
    assign_splits_by_dataset,
    compute_available_channels,
    find_missing_global_pair_coverage,
    get_channel_mask_columns,
    main as split_index_main,
    normalize_mask_frame,
    split_sizes,
)


def _build_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_mask_normalization_and_available_channel_count():
    df = _build_df(
        [
            {"dataset": "demo", "stage_mask": 1, "a_mask": "True", "b_mask": "1.0", "c_mask": "no"},
            {"dataset": "demo", "stage_mask": 0, "a_mask": 0, "b_mask": "yes", "c_mask": "t"},
        ]
    )

    mask_cols = get_channel_mask_columns(df)
    assert mask_cols == ["a_mask", "b_mask", "c_mask"]

    normalized = normalize_mask_frame(df, mask_cols)
    assert normalized.to_dict("list") == {
        "a_mask": [True, False],
        "b_mask": [True, True],
        "c_mask": [False, True],
    }
    assert compute_available_channels(df, mask_cols).tolist() == [2, 2]


def test_split_sizes_match_current_policy():
    assert split_sizes(0) == (0, 0)
    assert split_sizes(5) == (5, 0)
    assert split_sizes(12) == (12, 0)
    assert split_sizes(3000) == (20, 20)
    assert split_sizes(20, n_val=3, n_test=5) == (3, 5)
    assert split_sizes(4, n_val=3, n_test=3) == (3, 1)


def test_assign_splits_by_dataset_preserves_per_dataset_quota():
    rows = [{"dataset": "d1", "a_mask": 1, "b_mask": 1} for _ in range(12)]
    rows += [{"dataset": "d2", "a_mask": 1, "b_mask": 1} for _ in range(45)]
    df = _build_df(rows)

    split, stats = assign_splits_by_dataset(df, seed=0, shuffle=False)

    assert stats == {
        "d1": {"train": 0, "val": 12, "test": 0},
        "d2": {"train": 5, "val": 20, "test": 20},
    }
    counts = pd.crosstab(df["dataset"], split)
    assert int(counts.loc["d1", "val"]) == 12
    assert int(counts.loc["d2", "train"]) == 5
    assert int(counts.loc["d2", "val"]) == 20
    assert int(counts.loc["d2", "test"]) == 20


def test_assign_splits_by_dataset_respects_custom_cli_policy():
    rows = [{"dataset": "d1", "a_mask": 1, "b_mask": 1} for _ in range(20)]
    rows += [{"dataset": "d2", "a_mask": 1, "b_mask": 1} for _ in range(9)]
    df = _build_df(rows)

    split, stats = assign_splits_by_dataset(
        df,
        seed=0,
        shuffle=False,
        n_val=3,
        n_test=2,
    )

    assert stats == {
        "d1": {"train": 15, "val": 3, "test": 2},
        "d2": {"train": 4, "val": 3, "test": 2},
    }
    counts = pd.crosstab(df["dataset"], split)
    assert int(counts.loc["d1", "val"]) == 3
    assert int(counts.loc["d1", "test"]) == 2
    assert int(counts.loc["d2", "val"]) == 3
    assert int(counts.loc["d2", "test"]) == 2


def test_find_missing_global_pair_coverage_reports_uncovered_pairs():
    rows = [
        {"dataset": "demo", "a_mask": 1, "b_mask": 1, "c_mask": 0},  # val -> ab
        {"dataset": "demo", "a_mask": 1, "b_mask": 0, "c_mask": 1},  # test -> ac
    ] + [{"dataset": "demo", "a_mask": 1, "b_mask": 1, "c_mask": 1} for _ in range(10)]
    df = _build_df(rows)
    mask_cols = get_channel_mask_columns(df)
    split, _ = assign_splits_by_dataset(df, seed=0, shuffle=False, n_val=1, n_test=1)

    missing = find_missing_global_pair_coverage(df, split, mask_cols)

    assert missing == {
        "val": ["a__c", "b__c"],
        "test": ["a__b", "b__c"],
    }


def test_find_missing_global_pair_coverage_ignores_globally_impossible_pairs():
    rows = [
        {"dataset": "demo", "eeg_mask": 1, "ecg_mask": 1, "actigraphy_mask": 0},  # val
        {"dataset": "demo", "eeg_mask": 1, "ecg_mask": 1, "actigraphy_mask": 0},  # test
    ] + [{"dataset": "demo", "eeg_mask": 0, "ecg_mask": 0, "actigraphy_mask": 1} for _ in range(10)]
    df = _build_df(rows)
    mask_cols = get_channel_mask_columns(df)
    split, _ = assign_splits_by_dataset(df, seed=0, shuffle=False, n_val=1, n_test=1)

    missing = find_missing_global_pair_coverage(df, split, mask_cols)

    assert missing == {}


def test_split_index_main_keeps_single_modality_rows_by_default(tmp_path, monkeypatch):
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "output.csv"
    pd.DataFrame(
        [
            {"dataset": "demo", "path": "a.npz", "ppg_mask": 1},
            {"dataset": "demo", "path": "b.npz", "ppg_mask": 1},
        ]
    ).to_csv(input_path, index=False)

    monkeypatch.setattr(
        sys,
        "argv",
        ["split_index_by_dataset.py", "--input", str(input_path), "--output", str(output_path), "--no-shuffle"],
    )
    split_index_main()

    output_df = pd.read_csv(output_path)
    assert output_df["path"].tolist() == ["a.npz", "b.npz"]


def test_split_index_main_filters_single_modality_rows_when_min_channels_is_explicit(tmp_path, monkeypatch):
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "output.csv"
    pd.DataFrame(
        [
            {"dataset": "demo", "path": "a.npz", "ppg_mask": 1},
            {"dataset": "demo", "path": "b.npz", "ppg_mask": 1},
        ]
    ).to_csv(input_path, index=False)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "split_index_by_dataset.py",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--min-channels",
            "2",
            "--no-shuffle",
        ],
    )
    split_index_main()

    output_df = pd.read_csv(output_path)
    assert output_df.empty
