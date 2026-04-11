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
