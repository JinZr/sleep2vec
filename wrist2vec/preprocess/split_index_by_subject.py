#!/usr/bin/env python3
"""Split an index CSV by subject and write a new CSV with a split column.

The split unit is ``patient_id``: every row from the same subject receives the same split.

Default policy:
  1. Partition subjects into complete-metadata and incomplete-metadata pools.
  2. Draw test subjects proportionally from both pools so that the
     complete/incomplete ratio is the same in test as in the overall population.
  3. Draw val subjects from the remaining subjects with the same proportional strategy.
  4. Assign all other subjects to train.

Both test and validation draws within each metadata pool are stratified by each
subject's primary device (``dataset`` with the most rows for that subject), so
row-level device ratios stay roughly aligned across train/val/test.
"""

from __future__ import annotations

import argparse
from typing import Dict

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ("patient_id", "dataset")
DEFAULT_COMPLETE_METADATA_COLUMNS = ("age", "sex", "bmi")


def _non_empty_cell_mask(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip().str.lower()
    return series.notna() & ~text.isin(["", "nan", "none", "null", "<na>"])


def complete_metadata_subjects(
    df: pd.DataFrame,
    *,
    subject_col: str,
    metadata_cols: tuple[str, ...],
) -> set[str]:
    missing = [col for col in metadata_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing metadata columns required for complete-metadata test pool: {missing}")

    row_complete = pd.Series(True, index=df.index)
    for col in metadata_cols:
        row_complete &= _non_empty_cell_mask(df[col])

    subjects = df[subject_col].astype("string").fillna("")
    return set(subjects.loc[row_complete & subjects.ne("")].astype(str).tolist())


def primary_dataset_by_subject(
    df: pd.DataFrame,
    *,
    subject_col: str,
    dataset_col: str,
) -> pd.Series:
    subjects = df[subject_col].astype("string").fillna("")
    datasets = df[dataset_col].astype("string").fillna("")
    counts = (
        pd.DataFrame({"subject": subjects, "dataset": datasets})
        .loc[lambda x: x["subject"].ne("")]
        .groupby(["subject", "dataset"], sort=False)
        .size()
        .rename("rows")
        .reset_index()
    )
    counts = counts.sort_values(["subject", "rows", "dataset"], ascending=[True, False, True])
    primary = counts.drop_duplicates("subject", keep="first").set_index("subject")["dataset"]
    return primary.astype(str)


def allocate_stratified_counts(
    group_sizes: pd.Series,
    target_total: int,
) -> Dict[str, int]:
    """Allocate an exact target count across groups by largest remainder."""
    if target_total < 0:
        raise ValueError(f"target_total must be >= 0, got {target_total}")

    sizes = group_sizes.astype(int)
    sizes = sizes.loc[sizes > 0]
    total_available = int(sizes.sum())
    target_total = min(int(target_total), total_available)
    if target_total == 0 or total_available == 0:
        return {str(group): 0 for group in sizes.index}

    exact = sizes * (target_total / total_available)
    base = np.floor(exact).astype(int)
    base = np.minimum(base, sizes)

    remaining = target_total - int(base.sum())
    if remaining > 0:
        remainders = (exact - base).sort_values(ascending=False)
        for group in remainders.index:
            if remaining == 0:
                break
            if base.loc[group] >= sizes.loc[group]:
                continue
            base.loc[group] += 1
            remaining -= 1

    # Defensive pass for very small saturated strata.
    if remaining > 0:
        for group in sizes.sort_values(ascending=False).index:
            if remaining == 0:
                break
            room = int(sizes.loc[group] - base.loc[group])
            if room <= 0:
                continue
            add = min(room, remaining)
            base.loc[group] += add
            remaining -= add

    return {str(group): int(count) for group, count in base.items()}


def draw_subjects_by_primary_dataset(
    subjects: set[str],
    primary_dataset: pd.Series,
    *,
    target_total: int,
    rng: np.random.Generator,
    shuffle: bool,
) -> set[str]:
    pool = sorted(subjects)
    if target_total <= 0 or not pool:
        return set()

    pool_df = pd.DataFrame({"subject": pool})
    pool_df["dataset"] = pool_df["subject"].map(primary_dataset).fillna("")
    group_sizes = pool_df["dataset"].value_counts(sort=False)
    allocations = allocate_stratified_counts(group_sizes, target_total)

    selected: set[str] = set()
    for dataset, n_select in allocations.items():
        if n_select <= 0:
            continue
        candidates = pool_df.loc[pool_df["dataset"] == dataset, "subject"].to_numpy(dtype=str)
        if shuffle:
            rng.shuffle(candidates)
        selected.update(candidates[:n_select].tolist())
    return selected


def _draw_proportional(
    complete: set[str],
    incomplete: set[str],
    primary_dataset: pd.Series,
    *,
    target_total: int,
    rng: np.random.Generator,
    shuffle: bool,
) -> set[str]:
    """Draw *target_total* subjects from complete+incomplete, keeping their ratio intact."""
    n_total = len(complete) + len(incomplete)
    if n_total == 0 or target_total <= 0:
        return set()
    # Allocate proportionally, then clamp to available pool sizes.
    n_complete = min(round(len(complete) / n_total * target_total), len(complete))
    n_incomplete = min(target_total - n_complete, len(incomplete))
    # If one pool is too small, redirect the surplus to the other pool.
    n_complete = min(n_complete + max(0, target_total - n_complete - n_incomplete), len(complete))
    n_incomplete = min(target_total - n_complete, len(incomplete))

    drawn = draw_subjects_by_primary_dataset(complete, primary_dataset, target_total=n_complete, rng=rng, shuffle=shuffle)
    drawn |= draw_subjects_by_primary_dataset(incomplete, primary_dataset, target_total=n_incomplete, rng=rng, shuffle=shuffle)
    return drawn


def assign_splits_by_subject(
    df: pd.DataFrame,
    *,
    seed: int,
    shuffle: bool,
    n_test_subjects: int,
    val_fraction: float,
    subject_col: str,
    dataset_col: str,
    metadata_cols: tuple[str, ...],
) -> tuple[pd.Series, Dict[str, int]]:
    if not 0.0 <= val_fraction <= 1.0:
        raise ValueError(f"val_fraction must be in [0, 1], got {val_fraction}")

    rng = np.random.default_rng(seed)
    subject_series = df[subject_col].astype("string").fillna("")
    all_subjects = set(subject_series.loc[subject_series.ne("")].astype(str).unique().tolist())
    primary_dataset = primary_dataset_by_subject(df, subject_col=subject_col, dataset_col=dataset_col)

    complete_subjects = complete_metadata_subjects(df, subject_col=subject_col, metadata_cols=metadata_cols)
    incomplete_subjects = all_subjects - complete_subjects

    # Test: draw proportionally from complete and incomplete pools.
    test_subjects = _draw_proportional(
        complete_subjects,
        incomplete_subjects,
        primary_dataset,
        target_total=n_test_subjects,
        rng=rng,
        shuffle=shuffle,
    )

    # Val: same proportional strategy applied to the remaining subjects.
    remaining_complete = complete_subjects - test_subjects
    remaining_incomplete = incomplete_subjects - test_subjects
    remaining_subjects = remaining_complete | remaining_incomplete
    n_val_subjects = int(round(len(remaining_subjects) * val_fraction))
    val_subjects = _draw_proportional(
        remaining_complete,
        remaining_incomplete,
        primary_dataset,
        target_total=n_val_subjects,
        rng=rng,
        shuffle=shuffle,
    )

    split = pd.Series("train", index=df.index, dtype="string")
    split.loc[subject_series.isin(val_subjects)] = "val"
    split.loc[subject_series.isin(test_subjects)] = "test"

    stats = {
        "all_subjects": len(all_subjects),
        "complete_metadata_subjects": len(complete_subjects),
        "train_subjects": len(remaining_subjects - val_subjects),
        "val_subjects": len(val_subjects),
        "test_subjects": len(test_subjects),
    }
    return split, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to input CSV file.")
    parser.add_argument("--output", required=True, help="Path to output CSV file.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for subject shuffling.")
    parser.add_argument(
        "--n-test-subjects",
        type=int,
        default=20_000,
        help="Total number of subjects assigned to test (drawn proportionally from complete/incomplete metadata pools; default: 20000).",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.01,
        help="Fraction of non-test subjects assigned to validation (default: 0.01).",
    )
    parser.add_argument(
        "--subject-col",
        default="patient_id",
        help="Subject ID column used for grouped splitting (default: patient_id).",
    )
    parser.add_argument(
        "--dataset-col",
        default="dataset",
        help="Device/dataset column used for stratification (default: dataset).",
    )
    parser.add_argument(
        "--complete-metadata-cols",
        nargs="+",
        default=list(DEFAULT_COMPLETE_METADATA_COLUMNS),
        help="Columns that must be non-empty for a subject to enter the test pool (default: age sex bmi).",
    )
    parser.add_argument(
        "--shuffle",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Shuffle subjects within each primary-device stratum before splitting (default: True).",
    )
    return parser.parse_args()


def print_device_ratios(df: pd.DataFrame, *, dataset_col: str) -> None:
    print("Per-split row counts by device:")
    counts = pd.crosstab(df["split"], df[dataset_col], dropna=False)
    ratios = counts.div(counts.sum(axis=1), axis=0).fillna(0.0)
    for split_name in ("train", "val", "test"):
        if split_name not in counts.index:
            continue
        count_parts = {str(k): int(v) for k, v in counts.loc[split_name].items()}
        ratio_parts = {str(k): round(float(v), 4) for k, v in ratios.loc[split_name].items()}
        print(f"  {split_name}: counts={count_parts}")
        print(f"  {split_name}: ratios={ratio_parts}")


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.input, low_memory=False)
    required = [args.subject_col, args.dataset_col]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    metadata_cols = tuple(str(col) for col in args.complete_metadata_cols)
    split, subject_stats = assign_splits_by_subject(
        df,
        seed=args.seed,
        shuffle=args.shuffle,
        n_test_subjects=args.n_test_subjects,
        val_fraction=args.val_fraction,
        subject_col=args.subject_col,
        dataset_col=args.dataset_col,
        metadata_cols=metadata_cols,
    )

    df["split"] = split
    df.to_csv(args.output, index=False)

    print("Wrote:", args.output)
    print("Total row split counts:", df["split"].value_counts().to_dict())
    print("Subject split stats:", subject_stats)
    print_device_ratios(df, dataset_col=args.dataset_col)


if __name__ == "__main__":
    main()
