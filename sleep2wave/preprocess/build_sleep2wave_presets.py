#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import pickle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build sleep2wave generative preset pickles from an index CSV.",
    )
    parser.add_argument("--index", required=True, type=Path, help="Input index CSV.")
    parser.add_argument("--output", required=True, type=Path, help="Output preset pickle path.")
    parser.add_argument("--split", nargs="*", default=None, help="Optional split values to keep.")
    parser.add_argument("--context-epochs", type=int, default=15, help="Fixed 30-second context window length.")
    parser.add_argument("--stride-epochs", type=int, default=None, help="Stride in 30-second epochs.")
    parser.add_argument("--path-col", default="path", help="Index column containing waveform NPZ paths.")
    parser.add_argument("--duration-col", default="duration", help="Index column containing recording duration.")
    parser.add_argument("--split-col", default="split", help="Index column containing split labels.")
    parser.add_argument("--subject-id-col", default="subject_id", help="Index column containing subject ids.")
    parser.add_argument("--night-id-col", default="night_id", help="Index column containing night ids.")
    parser.add_argument("--source-col", default="source", help="Index column containing source dataset names.")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Row workers for NPZ inspection during preset build.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate and print count without writing.")
    return parser.parse_args()


def build_sleep2wave_presets(
    *,
    index_path: Path,
    output_path: Path,
    split: list[str] | None,
    context_epochs: int,
    stride_epochs: int | None,
    columns,
    num_workers: int = 1,
    dry_run: bool = False,
) -> list:
    import pandas as pd

    from sleep2wave.data.generative_dataset import IndexColumnConfig, build_sample_indices_from_frame

    if columns is None:
        columns = IndexColumnConfig()
    df = pd.read_csv(index_path, low_memory=False)
    samples = build_sample_indices_from_frame(
        df,
        index_source=str(index_path),
        split=split,
        context_epochs=context_epochs,
        stride_epochs=stride_epochs,
        columns=columns,
        require_all_masks=False,
        num_workers=num_workers,
    )
    if not dry_run:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as f:
            pickle.dump(samples, f)
    return samples


def main() -> None:
    args = parse_args()
    from sleep2wave.data.generative_dataset import IndexColumnConfig

    samples = build_sleep2wave_presets(
        index_path=args.index,
        output_path=args.output,
        split=args.split,
        context_epochs=args.context_epochs,
        stride_epochs=args.stride_epochs,
        columns=IndexColumnConfig(
            path_col=args.path_col,
            duration_col=args.duration_col,
            split_col=args.split_col,
            subject_id_col=args.subject_id_col,
            night_id_col=args.night_id_col,
            source_col=args.source_col,
        ),
        num_workers=args.num_workers,
        dry_run=args.dry_run,
    )
    action = "Would write" if args.dry_run else "Wrote"
    print(f"{action} {len(samples)} sleep2wave generative samples to {args.output}")


if __name__ == "__main__":
    main()


__all__ = ["build_sleep2wave_presets", "main", "parse_args"]
