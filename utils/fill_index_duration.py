#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path
import shutil
import sys
import time
import typing as t

import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent_tools.progress import write_progress

TRUTHY_MASK_VALUES = frozenset({"1", "1.0", "true", "t", "yes"})

CHANNEL_SAMPLE_RATES = {
    "heartbeat": 4,
    "breath": 4,
    "eeg_original": 128,
    "ecg_original": 128,
    "eog_original": 128,
    "emg_original": 128,
    "spo2": 4,
    "resp_original": 4,
    "resp_nasal_original": 4,
    "ppg": 100,
    "actigraphy": 32,
}


def parse_args(argv: t.Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fill missing duration values in an index CSV from NPZ channel lengths.",
    )
    parser.add_argument("--index", type=Path, required=True, help="Input index CSV.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output fixed index CSV. Defaults to overwriting --index after creating --index.backup.",
    )
    parser.add_argument(
        "--path-prefix-map",
        action="append",
        default=[],
        metavar="OLD=NEW",
        help="Rewrite NPZ paths by replacing OLD prefix with NEW. May be passed multiple times.",
    )
    parser.add_argument(
        "--duration-column",
        default="duration",
        help="Duration column to fill. Defaults to duration.",
    )
    return parser.parse_args(argv)


def _is_missing(value: t.Any) -> bool:
    try:
        if bool(pd.isna(value)):
            return True
    except (TypeError, ValueError):
        return False
    return str(value).strip() == ""


def _needs_duration_fill(value: t.Any) -> bool:
    if _is_missing(value):
        return True
    try:
        duration = float(value)
    except (TypeError, ValueError):
        return True
    return not math.isfinite(duration) or duration <= 0


def _parse_prefix_maps(raw_maps: t.Sequence[str]) -> list[tuple[str, str]]:
    mappings = []
    for item in raw_maps:
        if "=" not in item:
            raise ValueError(f"--path-prefix-map must use OLD=NEW format, got {item!r}.")
        old, new = item.split("=", 1)
        if not old:
            raise ValueError("--path-prefix-map OLD prefix must not be empty.")
        mappings.append((str(Path(old).expanduser()), str(Path(new).expanduser())))
    return sorted(mappings, key=lambda pair: len(pair[0]), reverse=True)


def _resolve_npz_path(path_value: t.Any, prefix_maps: t.Sequence[tuple[str, str]]) -> Path:
    raw = str(path_value)
    for old, new in prefix_maps:
        old_prefix = old.rstrip("/")
        if raw == old_prefix:
            return Path(new).expanduser()
        if raw.startswith(old_prefix + "/"):
            return Path(new + raw[len(old_prefix) :]).expanduser()
    return Path(raw).expanduser()


def _mask_allows(row: pd.Series, channel: str) -> bool:
    mask_column = "stage_mask" if channel == "stage5" else f"{channel}_mask"
    if mask_column not in row.index:
        return True
    value = row[mask_column]
    if _is_missing(value):
        return False
    return str(value).strip().lower() in TRUTHY_MASK_VALUES


def _channel_length(arr: np.ndarray) -> int:
    if arr.ndim == 0:
        return 0
    return int(max(arr.shape))


def infer_duration(path: Path, row: pd.Series, *, token_sec: int = 30) -> int:
    token_counts = []
    with np.load(path, mmap_mode="r", allow_pickle=False) as npz:
        for channel, sample_rate in CHANNEL_SAMPLE_RATES.items():
            if channel not in npz or not _mask_allows(row, channel):
                continue
            length = _channel_length(np.asarray(npz[channel]))
            if length <= 0:
                continue
            frames_per_token = int(round(sample_rate * token_sec))
            token_counts.append(length // frames_per_token)

    if not token_counts:
        raise ValueError(f"No known duration-bearing channels found in {path}.")
    return int(min(token_counts) * token_sec)


def fill_duration(
    df: pd.DataFrame,
    *,
    duration_column: str,
    prefix_maps: t.Sequence[tuple[str, str]],
    progress_dir: Path | None = None,
) -> tuple[pd.DataFrame, int]:
    required = {"path"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Input index CSV is missing required column(s): {missing}.")

    fixed = df.copy()
    if duration_column not in fixed.columns:
        fixed[duration_column] = pd.NA

    missing_indexes = [row_index for row_index, row in fixed.iterrows() if _needs_duration_fill(row[duration_column])]
    started_at = time.time()
    if progress_dir is not None:
        write_progress(
            progress_dir,
            status="running",
            task="fill_index_duration",
            processed=0,
            total=len(missing_indexes),
            success=0,
            failed=0,
            start_time=started_at,
        )
    processed_count = 0
    try:
        for processed, row_index in enumerate(tqdm(missing_indexes, desc="Filling duration", unit="row"), start=1):
            processed_count = processed
            row = fixed.loc[row_index]
            npz_path = _resolve_npz_path(row["path"], prefix_maps)
            fixed.at[row_index, duration_column] = infer_duration(npz_path, row)
            if progress_dir is not None:
                write_progress(
                    progress_dir,
                    status="running",
                    task="fill_index_duration",
                    processed=processed,
                    total=len(missing_indexes),
                    success=processed,
                    failed=0,
                    start_time=started_at,
                    current_item=str(npz_path),
                )
    except Exception as exc:
        if progress_dir is not None:
            write_progress(
                progress_dir,
                status="failed",
                task="fill_index_duration",
                processed=processed_count,
                total=len(missing_indexes),
                success=processed_count,
                failed=1,
                start_time=started_at,
                message=str(exc),
            )
        raise
    return fixed, len(missing_indexes)


def main(argv: t.Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    index_path = args.index.expanduser()
    output_path = index_path if args.output is None else args.output.expanduser()
    write_back = index_path.resolve() == output_path.resolve()

    df = pd.read_csv(index_path, low_memory=False)
    fixed, changed = fill_duration(
        df,
        duration_column=args.duration_column,
        prefix_maps=_parse_prefix_maps(args.path_prefix_map),
        progress_dir=output_path.parent,
    )

    if write_back:
        backup_path = Path(str(index_path) + ".backup")
        if not backup_path.exists():
            shutil.copy2(index_path, backup_path)
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    fixed.to_csv(output_path, index=False)
    write_progress(
        output_path.parent,
        status="completed",
        task="fill_index_duration",
        processed=changed,
        total=changed,
        success=changed,
        failed=0,
        message=f"Wrote {output_path}",
    )
    print(f"Wrote {output_path} with {len(fixed)} rows; filled {changed} duration value(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
