from __future__ import annotations

import pandas as pd

from preprocess.split_index_by_dataset import (
    assign_splits_by_dataset,
    compute_available_channels,
    find_missing_global_pair_coverage,
    get_channel_mask_columns,
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
    assert split_sizes(5) == (0, 0)
    assert split_sizes(12) == (1, 1)
    assert split_sizes(3000) == (200, 200)


def test_assign_splits_by_dataset_preserves_per_dataset_quota():
    rows = [{"dataset": "d1", "a_mask": 1, "b_mask": 1} for _ in range(12)]
    rows += [{"dataset": "d2", "a_mask": 1, "b_mask": 1} for _ in range(5)]
    df = _build_df(rows)

    split, stats = assign_splits_by_dataset(df, seed=0, shuffle=False)

    assert stats == {
        "d1": {"train": 10, "val": 1, "test": 1},
        "d2": {"train": 5, "val": 0, "test": 0},
    }
    counts = pd.crosstab(df["dataset"], split)
    assert int(counts.loc["d1", "train"]) == 10
    assert int(counts.loc["d1", "val"]) == 1
    assert int(counts.loc["d1", "test"]) == 1
    assert int(counts.loc["d2", "train"]) == 5


def test_find_missing_global_pair_coverage_reports_uncovered_pairs():
    rows = [
        {"dataset": "demo", "a_mask": 1, "b_mask": 1, "c_mask": 0},  # val -> ab
        {"dataset": "demo", "a_mask": 1, "b_mask": 0, "c_mask": 1},  # test -> ac
    ] + [{"dataset": "demo", "a_mask": 1, "b_mask": 1, "c_mask": 1} for _ in range(10)]
    df = _build_df(rows)
    mask_cols = get_channel_mask_columns(df)
    split, _ = assign_splits_by_dataset(df, seed=0, shuffle=False)

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
    split, _ = assign_splits_by_dataset(df, seed=0, shuffle=False)

    missing = find_missing_global_pair_coverage(df, split, mask_cols)

    assert missing == {}
