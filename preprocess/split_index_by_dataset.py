#!/usr/bin/env python3
import argparse
from typing import Dict, Tuple

import numpy as np
import pandas as pd


def compute_group_key(dataset_col: pd.Series) -> pd.Series:
    dataset_str = dataset_col.astype("string")
    is_hsp = dataset_str.str.startswith("hsp", na=False)
    return dataset_str.where(~is_hsp, "hsp").fillna("")


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
        n_val = min(n // 10, 1000)
        n_test = min(n // 10, 1000)
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
            "and write a new CSV with a split column."
        )
    )
    parser.add_argument("--input", required=True, help="Path to input CSV file.")
    parser.add_argument("--output", required=True, help="Path to output CSV file.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling.")
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

    group_key = compute_group_key(df["dataset"])
    split, stats = assign_splits(df, group_key, seed=args.seed, shuffle=args.shuffle)
    df["split"] = split

    df.to_csv(args.output, index=False)

    total = df["split"].value_counts().to_dict()
    print("Wrote:", args.output)
    print("Total split counts:", total)
    print("Per-group counts:")
    for group, counts in stats.items():
        print(f"  {group}: {counts}")


if __name__ == "__main__":
    main()
