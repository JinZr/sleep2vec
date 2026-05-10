#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a sleep2wave generative index CSV contract.",
    )
    parser.add_argument("--index", required=True, type=Path, help="Input index CSV.")
    parser.add_argument("--path-col", default="path", help="Index column containing waveform NPZ paths.")
    parser.add_argument("--duration-col", default="duration", help="Index column containing recording duration.")
    parser.add_argument("--split-col", default="split", help="Index column containing split labels.")
    parser.add_argument("--subject-id-col", default="subject_id", help="Index column containing subject ids.")
    parser.add_argument("--night-id-col", default="night_id", help="Index column containing night ids.")
    return parser.parse_args()


def validate_sleep2wave_index(
    index_path: Path,
    *,
    columns=None,
) -> None:
    import numpy as np
    import pandas as pd

    from sleep2wave.data.generative_dataset import (
        IndexColumnConfig,
        prepare_sleep2wave_index_frame,
        resolve_modality_mask_columns,
    )
    from sleep2wave.preprocess.split_index_by_dataset import normalize_mask_frame

    if columns is None:
        columns = IndexColumnConfig()
    df = pd.read_csv(index_path, low_memory=False)
    df, columns = prepare_sleep2wave_index_frame(df, columns=columns)
    if df[columns.path_col].isna().any():
        raise ValueError(f"sleep2wave index contains missing {columns.path_col} values.")
    if df[columns.split_col].isna().any():
        raise ValueError(f"sleep2wave index contains missing {columns.split_col} values.")

    durations = pd.to_numeric(df[columns.duration_col], errors="coerce")
    if durations.isna().any() or (~np.isfinite(durations)).any() or (durations <= 0).any():
        raise ValueError(f"sleep2wave index contains invalid {columns.duration_col} values.")

    mask_columns = resolve_modality_mask_columns(df, require_all=False)
    if mask_columns:
        mask_frame = normalize_mask_frame(df, list(mask_columns.values()))
        empty_rows = mask_frame[list(mask_columns.values())].sum(axis=1) == 0
        if empty_rows.any():
            offenders = empty_rows[empty_rows].index.astype(str).tolist()
            raise ValueError(f"sleep2wave index rows have no available modalities: {offenders}")


def main() -> None:
    args = parse_args()
    from sleep2wave.data.generative_dataset import IndexColumnConfig

    validate_sleep2wave_index(
        args.index,
        columns=IndexColumnConfig(
            path_col=args.path_col,
            duration_col=args.duration_col,
            split_col=args.split_col,
            subject_id_col=args.subject_id_col,
            night_id_col=args.night_id_col,
        ),
    )
    print(f"sleep2wave index is valid: {args.index}")


if __name__ == "__main__":
    main()


__all__ = ["main", "parse_args", "validate_sleep2wave_index"]
