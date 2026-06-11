#!/usr/bin/env python3
"""Compute missing-channel statistics from *_mask columns.

Definition:
  - For every column ending with "_mask": value == 1 means the channel is present.
    Any other value (0, NaN, empty, etc.) is treated as missing.

Outputs:
  - <out_prefix>_overall.csv: per-channel missing counts/rates over the whole file
  - <out_prefix>_by_dataset.csv: per-dataset per-channel missing counts/rates
  - <out_prefix>_row_missing_hist_overall.csv: distribution of rows by number of missing channels
  - <out_prefix>_row_missing_hist_by_dataset.csv: same distribution, grouped by dataset

Example:
  python mask_missing_stats.py --csv /path/to/file.csv --dataset-col dataset --out-prefix missing_stats
"""

import argparse
from collections import defaultdict
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

TRUTHY_MASK_VALUES = frozenset({"1", "1.0", "true", "t", "yes"})

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent_tools.progress import write_progress


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="Path to the input CSV")
    p.add_argument("--dataset-col", default="dataset", help="Column used for grouping (default: dataset)")
    p.add_argument("--chunksize", type=int, default=200_000, help="Rows per chunk for streaming (default: 200000)")
    p.add_argument(
        "--out-prefix",
        default="./index/missing_stats",
        help=(
            "Output prefix (default: missing_stats). Writes <prefix>_overall.csv, <prefix>_by_dataset.csv, "
            "<prefix>_row_missing_hist_overall.csv, and <prefix>_row_missing_hist_by_dataset.csv"
        ),
    )
    p.add_argument(
        "--print-per-dataset",
        action="store_true",
        help="Print per-dataset missing summary to stdout (can be long)",
    )
    p.add_argument(
        "--topk",
        type=int,
        default=10,
        help="When printing per-dataset summary, show up to top-k missing channels per dataset (default: 10)",
    )
    return p.parse_args()


def _prefix_path(prefix: str) -> str:
    """Normalize a user-provided prefix like 'foo.csv' -> 'foo'."""
    p = Path(prefix)
    # If user passed something like 'foo/bar.csv', drop the suffix.
    if p.suffix:
        p = p.with_suffix("")
    return p.as_posix()


def main() -> None:
    args = parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise SystemExit(f"ERROR: file not found: {csv_path}")

    header = pd.read_csv(csv_path, nrows=0)
    cols = list(header.columns)

    if args.dataset_col not in cols:
        raise SystemExit(f"ERROR: dataset column '{args.dataset_col}' not found. Available columns: {', '.join(cols)}")

    mask_cols = [c for c in cols if c.endswith("_mask") and c != "stage_mask"]
    if not mask_cols:
        raise SystemExit("ERROR: no columns ending with '_mask' were found (excluding 'stage_mask')")

    num_masks = len(mask_cols)

    print(f"CSV: {csv_path}")
    print(f"Total columns: {len(cols)}")
    print(f"Mask columns ({num_masks}): {', '.join(mask_cols)}")
    progress_dir = Path(_prefix_path(args.out_prefix)).expanduser().parent
    started_at = time.time()
    write_progress(
        progress_dir,
        status="running",
        task="mask_missing_stats",
        processed=0,
        total=None,
        success=0,
        failed=0,
        start_time=started_at,
    )

    overall_total = 0
    overall_missing = pd.Series(0, index=mask_cols, dtype="int64")
    overall_row_missing_hist = np.zeros(num_masks + 1, dtype="int64")

    ds_total = defaultdict(int)
    ds_missing = defaultdict(lambda: pd.Series(0, index=mask_cols, dtype="int64"))
    ds_row_missing_hist = defaultdict(lambda: np.zeros(num_masks + 1, dtype="int64"))

    usecols = [args.dataset_col] + mask_cols

    for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=args.chunksize, low_memory=False):
        # Normalize dataset labels
        ds = chunk[args.dataset_col].astype("string").fillna("<NA>")

        mask_strings = chunk[mask_cols].astype("string").apply(lambda col: col.str.strip().str.lower())
        present = mask_strings.isin(TRUTHY_MASK_VALUES) | chunk[mask_cols].apply(pd.to_numeric, errors="coerce").eq(1)
        missing = ~present

        n = len(chunk)
        overall_total += n
        overall_missing += missing.sum(axis=0).astype("int64")

        # Row-wise missing-channel counts
        missing_count = missing.sum(axis=1).astype("int64")

        # Overall histogram
        overall_row_missing_hist += np.bincount(missing_count.to_numpy(), minlength=num_masks + 1).astype("int64")

        # Per-dataset aggregation within this chunk (column-wise missing)
        grp_missing = missing.groupby(ds).sum()
        grp_total = ds.value_counts()

        for dset, cnt in grp_total.items():
            ds_total[dset] += int(cnt)
            ds_missing[dset] += grp_missing.loc[dset].astype("int64")

        # Per-dataset row-wise histogram
        # Produces a Series with MultiIndex (dataset, missing_count)
        ds_hist_counts = missing_count.groupby(ds).value_counts(sort=False)
        for (dset, k), cnt in ds_hist_counts.items():
            # k is missing-channel count (0..num_masks)
            ds_row_missing_hist[dset][int(k)] += int(cnt)
        write_progress(
            progress_dir,
            status="running",
            task="mask_missing_stats",
            processed=overall_total,
            total=None,
            success=overall_total,
            failed=0,
            start_time=started_at,
        )

    # Overall per-channel table
    overall_df = pd.DataFrame(
        {
            "mask_col": mask_cols,
            "missing_rows": [int(overall_missing[c]) for c in mask_cols],
            "missing_rate": [float(overall_missing[c]) / overall_total for c in mask_cols],
            "total_rows": overall_total,
        }
    ).sort_values(["missing_rate", "missing_rows"], ascending=False, ignore_index=True)

    # Per-dataset wide table (per-channel)
    by_ds_rows = []
    for dset, total in ds_total.items():
        row = {"dataset": dset, "rows": total}
        miss = ds_missing[dset]
        for c in mask_cols:
            row[f"{c}_missing_rows"] = int(miss[c])
            row[f"{c}_missing_rate"] = float(miss[c]) / total if total else np.nan
        by_ds_rows.append(row)

    by_ds_df = pd.DataFrame(by_ds_rows).sort_values("rows", ascending=False, ignore_index=True)

    # Overall row-missing histogram table
    hist_overall_df = pd.DataFrame(
        {
            "missing_channels": list(range(num_masks + 1)),
            "rows": [int(x) for x in overall_row_missing_hist.tolist()],
        }
    )
    hist_overall_df["rate"] = hist_overall_df["rows"] / float(overall_total) if overall_total else np.nan
    hist_overall_df["total_rows"] = overall_total
    hist_overall_df["num_mask_cols"] = num_masks

    # Per-dataset row-missing histogram table (long format)
    hist_by_ds_rows = []
    for dset, total in ds_total.items():
        h = ds_row_missing_hist[dset]
        for k in range(num_masks + 1):
            cnt = int(h[k])
            # Keep all k for completeness.
            hist_by_ds_rows.append(
                {
                    "dataset": dset,
                    "missing_channels": k,
                    "rows": cnt,
                    "rate": (cnt / total) if total else np.nan,
                    "dataset_rows": total,
                    "num_mask_cols": num_masks,
                }
            )

    hist_by_ds_df = pd.DataFrame(hist_by_ds_rows).sort_values(
        ["dataset_rows", "dataset", "missing_channels"], ascending=[False, True, True], ignore_index=True
    )

    # Output paths
    prefix = _prefix_path(args.out_prefix)
    overall_out = prefix + "_overall.csv"
    by_ds_out = prefix + "_by_dataset.csv"
    hist_overall_out = prefix + "_row_missing_hist_overall.csv"
    hist_by_ds_out = prefix + "_row_missing_hist_by_dataset.csv"

    overall_df.to_csv(overall_out, index=False)
    by_ds_df.to_csv(by_ds_out, index=False)
    hist_overall_df.to_csv(hist_overall_out, index=False)
    hist_by_ds_df.to_csv(hist_by_ds_out, index=False)

    print(f"\nTotal rows: {overall_total}")

    print("\nOverall missing by channel (sorted):")
    show = overall_df.copy()
    show["missing_rate"] = (show["missing_rate"] * 100).map(lambda x: f"{x:.4f}%")
    print(show.to_string(index=False))

    # Row-wise distribution summary
    print(f"\nRows by number of missing channels (out of {num_masks}):")
    for k, cnt in enumerate(overall_row_missing_hist.tolist()):
        rate = (cnt / overall_total) if overall_total else 0.0
        # k==0 means all channels are present
        ch_word = "channel" if k == 1 else "channels"
        print(f"There are {cnt} rows missing {k} {ch_word} ({rate*100:.4f}%).")

    if args.print_per_dataset:
        print("\nPer-dataset missing summary (top missing channels per dataset):")
        for _, r in by_ds_df.iterrows():
            dset = r["dataset"]
            total = int(r["rows"])

            # Build (col, missing_rows, missing_rate)
            trip = []
            for c in mask_cols:
                mr = int(r[f"{c}_missing_rows"])
                if mr > 0:
                    trip.append((c, mr, mr / total if total else np.nan))
            trip.sort(key=lambda x: (x[2], x[1]), reverse=True)

            print(f"\n[{dset}] rows={total}")
            if not trip:
                print("  No missing in any *_mask column.")
            else:
                for c, mr, rate in trip[: max(1, args.topk)]:
                    print(f"  {c}: missing_rows={mr} ({rate*100:.4f}%)")

            # Per-dataset row-missing histogram
            h = ds_row_missing_hist[dset]
            print(f"  Rows by missing-channel count (out of {num_masks}):")
            for k in range(num_masks + 1):
                cnt = int(h[k])
                if cnt == 0:
                    continue
                rate = cnt / total if total else 0.0
                ch_word = "channel" if k == 1 else "channels"
                print(f"    There are {cnt} rows missing {k} {ch_word} ({rate*100:.4f}%).")

    print(f"\nWrote: {overall_out}")
    print(f"Wrote: {by_ds_out}")
    print(f"Wrote: {hist_overall_out}")
    print(f"Wrote: {hist_by_ds_out}")
    write_progress(
        progress_dir,
        status="completed",
        task="mask_missing_stats",
        processed=overall_total,
        total=overall_total,
        success=overall_total,
        failed=0,
        start_time=started_at,
        message=f"Wrote {overall_out}",
    )


if __name__ == "__main__":
    main()
