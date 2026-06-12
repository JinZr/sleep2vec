from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError

from sleep2stat.config import load_config
from sleep2stat.core.pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate sleep2stat alignment/statistics bundles.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Run a sleep2stat analysis bundle.")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--split", nargs="+", default=None)
    run.add_argument("--device", default="cuda")
    run.add_argument("--num-workers", type=int, default=8)
    run.add_argument("--batch-size", type=int, default=None)
    run.add_argument("--limit-records", type=int, default=None)
    run.add_argument("--dry-run", action="store_true")

    validate = subparsers.add_parser("validate-config", help="Validate a sleep2stat YAML config.")
    validate.add_argument("--config", type=Path, required=True)

    summarize = subparsers.add_parser("summarize", help="Summarize a completed sleep2stat run directory.")
    summarize.add_argument("--run-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "validate-config":
        load_config(args.config)
        print(f"sleep2stat config OK: {args.config}")
        return 0
    if args.command == "summarize":
        _summarize(args.run_dir)
        return 0
    config = load_config(args.config)
    run_dir = run_pipeline(config, args)
    print(f"sleep2stat run directory: {run_dir}")
    return 0


def _summarize(run_dir: Path) -> None:
    manifest = run_dir / "run_manifest.json"
    record_manifest = run_dir / "record_manifest.csv"
    night_stats = run_dir / "tables" / "night_stats.csv"
    print(f"run_dir: {run_dir}")
    if manifest.exists():
        print(f"manifest: {manifest}")
    if record_manifest.exists():
        print(f"records: {_csv_row_count(record_manifest)}")
    if night_stats.exists():
        print(f"night_stats_rows: {_csv_row_count(night_stats)}")


def _csv_row_count(path: Path) -> int:
    try:
        return len(pd.read_csv(path))
    except EmptyDataError:
        return 0
