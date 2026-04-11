#!/usr/bin/env python3
import argparse
from typing import Dict

import numpy as np
import pandas as pd

TRUTHY_MASK_VALUES = frozenset({"1", "1.0", "true", "t", "yes"})
EXTERNAL_DATASET_PATTERN = r"(?:mros|mesa|shhs|hspS0001)"


def get_channel_mask_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if col.endswith("_mask") and col != "stage_mask"]


def normalize_mask_frame(df: pd.DataFrame, mask_cols: list[str]) -> pd.DataFrame:
    if not mask_cols:
        return pd.DataFrame(index=df.index)

    return (
        df[mask_cols]
        .astype("string")
        .fillna("")
        .apply(lambda col: col.str.strip().str.lower().isin(TRUTHY_MASK_VALUES))
    )


def compute_available_channels(df: pd.DataFrame, mask_cols: list[str]) -> pd.Series:
    return normalize_mask_frame(df, mask_cols).sum(axis=1)


def compute_external_mask(dataset_col: pd.Series) -> pd.Series:
    dataset_str = dataset_col.astype("string")
    return dataset_str.str.contains(EXTERNAL_DATASET_PATTERN, case=False, na=False)


def split_sizes(
    n_rows: int,
    *,
    n_val: int = 20,
    n_test: int = 20,
) -> tuple[int, int]:
    if n_val < 0:
        raise ValueError(f"n_val must be >= 0, got {n_val}")
    n_val = min(n_val, n_rows)

    if n_test < 0:
        raise ValueError(f"n_test must be >= 0, got {n_test}")
    n_test = min(n_test, n_rows)

    if n_val + n_test > n_rows:
        n_test = max(0, n_rows - n_val)
    return n_val, n_test


def assign_splits_by_dataset(
    df: pd.DataFrame,
    seed: int,
    shuffle: bool,
    n_val: int = 20,
    n_test: int = 20,
) -> tuple[pd.Series, Dict[str, Dict[str, int]]]:
    rng = np.random.default_rng(seed)
    split = pd.Series("train", index=df.index, dtype="string")
    stats: Dict[str, Dict[str, int]] = {}

    dataset_key = df["dataset"].astype("string").fillna("")
    for dataset, rows in df.groupby(dataset_key, sort=False).groups.items():
        idx = np.array(list(rows))
        if shuffle:
            rng.shuffle(idx)

        n_val_rows, n_test_rows = split_sizes(
            len(idx),
            n_val=n_val,
            n_test=n_test,
        )
        split.loc[idx[:n_val_rows]] = "val"
        split.loc[idx[n_val_rows : n_val_rows + n_test_rows]] = "test"

        stats[str(dataset)] = {
            "train": int(len(idx) - n_val_rows - n_test_rows),
            "val": int(n_val_rows),
            "test": int(n_test_rows),
        }

    return split, stats


def find_missing_global_pair_coverage(
    df: pd.DataFrame,
    split: pd.Series,
    mask_cols: list[str],
) -> Dict[str, list[str]]:
    if not mask_cols:
        return {}

    mask_frame = normalize_mask_frame(df, mask_cols)
    channels = [col[:-5] for col in mask_cols]
    missing: Dict[str, list[str]] = {}

    for target_split in ("val", "test"):
        target_idx = split.index[split == target_split]
        target_masks = mask_frame.loc[target_idx]
        missing_pairs: list[str] = []

        for i, left in enumerate(channels):
            left_col = f"{left}_mask"
            for right in channels[i + 1 :]:
                right_col = f"{right}_mask"
                feasible = mask_frame[left_col] & mask_frame[right_col]
                if not feasible.any():
                    continue
                covered = target_masks[left_col] & target_masks[right_col]
                if not covered.any():
                    missing_pairs.append(f"{left}__{right}")

        if missing_pairs:
            missing[target_split] = missing_pairs

    return missing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split a CSV by dataset and write a new CSV with a split column. "
            "Rows matching external datasets are labeled as external."
        )
    )
    parser.add_argument("--input", required=True, help="Path to input CSV file.")
    parser.add_argument("--output", required=True, help="Path to output CSV file.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling.")
    parser.add_argument(
        "--n-val",
        type=int,
        default=20,
        help="Number of validation rows per dataset (default: 20).",
    )
    parser.add_argument(
        "--n-test",
        type=int,
        default=20,
        help="Number of test rows per dataset (default: 20).",
    )
    parser.add_argument(
        "--min-channels",
        type=int,
        default=2,
        help="Minimum number of available channels required to keep a row (default: 2).",
    )
    parser.add_argument(
        "--shuffle",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Shuffle rows within each dataset before splitting (default: True).",
    )
    parser.add_argument(
        "--require-pair-coverage",
        action="store_true",
        help="Fail if val or test misses any feasible channel pair.",
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
            raise ValueError("No *_mask columns found (excluding stage_mask); cannot apply --min-channels.")
        available_channels = compute_available_channels(df, mask_cols)
        before_count = len(df)
        df = df.loc[available_channels >= args.min_channels].copy()
        print(f"Filtered out {before_count - len(df)} rows with < {args.min_channels} available channels.")

    df["split"] = pd.Series(index=df.index, dtype="string")
    external_mask = compute_external_mask(df["dataset"])
    df.loc[external_mask, "split"] = "external"

    internal_df = df.loc[~external_mask]
    if internal_df.empty:
        stats: Dict[str, Dict[str, int]] = {}
    else:
        split, stats = assign_splits_by_dataset(
            internal_df,
            seed=args.seed,
            shuffle=args.shuffle,
            n_val=args.n_val,
            n_test=args.n_test,
        )
        df.loc[internal_df.index, "split"] = split

        missing_pairs = find_missing_global_pair_coverage(internal_df, split, mask_cols)
        if missing_pairs:
            details = "; ".join(f"{name}={pairs}" for name, pairs in missing_pairs.items())
            message = f"Missing feasible global channel pairs: {details}"
            if args.require_pair_coverage:
                raise ValueError(message)
            print(f"Warning: {message}")

    df.to_csv(args.output, index=False)

    print("Wrote:", args.output)
    print("Total split counts:", df["split"].value_counts().to_dict())
    print("Per-group counts:")
    for dataset, counts in stats.items():
        print(f"  {dataset}: {counts}")


if __name__ == "__main__":
    main()
