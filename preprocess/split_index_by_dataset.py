#!/usr/bin/env python3
import argparse
from typing import Dict, Tuple

import numpy as np
import pandas as pd


def compute_group_key(dataset_col: pd.Series) -> pd.Series:
    dataset_str = dataset_col.astype("string")
    return dataset_str.fillna("")


def compute_external_mask(dataset_col: pd.Series) -> pd.Series:
    dataset_str = dataset_col.astype("string")
    return dataset_str.str.contains(r"(mros|mesa|shhs|hspS0001)", case=False, na=False)


def get_channel_mask_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if col.endswith("_mask") and col != "stage_mask"]


def compute_available_channels(df: pd.DataFrame, mask_cols: list[str]) -> pd.Series:
    return df[mask_cols].eq(1).sum(axis=1)


def assign_splits(
    df: pd.DataFrame,
    group_key: pd.Series,
    seed: int,
    shuffle: bool,
) -> Tuple[pd.Series, Dict[str, Dict[str, int]]]:
    rng = np.random.default_rng(seed)
    split = pd.Series(index=df.index, dtype="object")
    stats: Dict[str, Dict[str, int]] = {}

    for group, idx in df.groupby(group_key, sort=False).groups.items():
        idx = np.array(list(idx))
        if shuffle:
            rng.shuffle(idx)

        n = len(idx)
        n_val = min(n // 10, 200)
        n_test = min(n // 10, 200)
        if n_val + n_test > n:
            n_test = max(0, n - n_val)

        val_idx = idx[:n_val]
        test_idx = idx[n_val : n_val + n_test]
        train_idx = idx[n_val + n_test :]

        split.loc[val_idx] = "val"
        split.loc[test_idx] = "test"
        split.loc[train_idx] = "train"

        stats[str(group)] = {"train": len(train_idx), "val": len(val_idx), "test": len(test_idx)}

    return split, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split a CSV by dataset groups, with hsp* datasets grouped together, "
            "and write a new CSV with a split column. Rows matching external "
            "datasets are labeled as external."
        )
    )
    parser.add_argument("--input", required=True, help="Path to input CSV file.")
    parser.add_argument("--output", required=True, help="Path to output CSV file.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling.")
    parser.add_argument(
        "--min-channels",
        type=int,
        default=0,
        help="Minimum number of available channels required to keep a row (default: 0).",
    )
    parser.add_argument(
        "--shuffle",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Shuffle rows within each group before splitting (default: True).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.input, low_memory=False)
    if "dataset" not in df.columns:
        raise ValueError("Missing required column: dataset")

    mask_cols = get_channel_mask_columns(df)
    if args.min_channels > 0:
        if not mask_cols:
            raise ValueError("No *_mask columns found (excluding stage_mask); cannot apply --min-channels filter.")
        available_channels = compute_available_channels(df, mask_cols)
        before_count = len(df)
        df = df.loc[available_channels >= args.min_channels].copy()
        filtered_out = before_count - len(df)
        print(f"Filtered out {filtered_out} rows with < {args.min_channels} available channels.")

    external_mask = compute_external_mask(df["dataset"])
    df["split"] = pd.Series(index=df.index, dtype="object")
    df.loc[external_mask, "split"] = "external"

    internal_df = df.loc[~external_mask]
    if not internal_df.empty:
        group_key = compute_group_key(internal_df["dataset"])
        split, stats = assign_splits(internal_df, group_key, seed=args.seed, shuffle=args.shuffle)
        df.loc[internal_df.index, "split"] = split
    else:
        stats = {}

    df.to_csv(args.output, index=False)

    total = df["split"].value_counts().to_dict()
    print("Wrote:", args.output)
    print("Total split counts:", total)
    print("Per-group counts:")
    for group, counts in stats.items():
        print(f"  {group}: {counts}")


if __name__ == "__main__":
    main()
