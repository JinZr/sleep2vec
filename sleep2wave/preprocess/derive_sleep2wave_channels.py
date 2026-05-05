#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plan sleep2wave derived-channel jobs.",
    )
    parser.add_argument("--index", required=True, type=Path, help="Input index CSV.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for derived sidecar NPZ files.")
    parser.add_argument("--derive", nargs="*", choices=["ibi", "resp"], default=[], help="Derived channels to build.")
    parser.add_argument("--path-col", default="path", help="Index column containing waveform NPZ paths.")
    parser.add_argument("--split-col", default="split", help="Index column containing split labels.")
    parser.add_argument("--subject-id-col", default="subject_id", help="Index column containing subject ids.")
    parser.add_argument("--night-id-col", default="night_id", help="Index column containing night ids.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import pandas as pd

    from sleep2wave.data.derivations import plan_derivation_jobs, run_derivation_jobs

    df = pd.read_csv(args.index, low_memory=False)
    jobs = plan_derivation_jobs(
        df,
        output_dir=args.output_dir,
        path_col=args.path_col,
        split_col=args.split_col,
        subject_id_col=args.subject_id_col,
        night_id_col=args.night_id_col,
    )
    written = run_derivation_jobs(jobs, derive=args.derive)
    print(f"Wrote {len(written)} sleep2wave derived-channel sidecars to {args.output_dir}.")


if __name__ == "__main__":
    main()


__all__ = ["main", "parse_args"]
