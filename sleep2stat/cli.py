from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError

from sleep2stat.config import load_config
from sleep2stat.core.artifacts import FailureRecord
from sleep2stat.core.pipeline import run_pipeline
from sleep2stat.io.records import SleepRecord
from sleep2stat.io.writers import AnalysisBundleWriter
from sleep2stat.plot import plot_cohort, plot_record


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

    plot = subparsers.add_parser("plot-record", help="Plot one sleep2stat per-record output directory.")
    plot.add_argument("--run-dir", type=Path, required=True)
    plot.add_argument("--record-id", required=True)

    cohort = subparsers.add_parser("plot-cohort", help="Plot sleep2stat cohort-level sleep architecture.")
    cohort.add_argument("--run-dir", type=Path, required=True)
    cohort.add_argument("--group-column", default="source")
    cohort.add_argument("--stage-source", default="auto")
    cohort.add_argument("--adjust-covariates", nargs="*", default=None)
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
    if args.command == "plot-record":
        for path in plot_record(args.run_dir, args.record_id):
            print(path)
        return 0
    if args.command == "plot-cohort":
        for path in plot_cohort(
            args.run_dir,
            group_column=args.group_column,
            stage_source=args.stage_source,
            adjust_covariates=args.adjust_covariates,
        ):
            print(path)
        return 0
    config = load_config(args.config)
    run_dir = run_pipeline(config, args)
    print(f"sleep2stat run directory: {run_dir}")
    return 0


def _summarize(run_dir: Path) -> None:
    config_path = run_dir / "config.yaml"
    if config_path.exists():
        config = load_config(config_path)
        config = replace(config, run=replace(config.run, output_dir=run_dir))
        writer = AnalysisBundleWriter(config)
        records = _read_record_manifest(run_dir / "record_manifest.csv", config.data.token_sec, config.data.max_tokens)
        writer.rebuild_global_tables(records, _read_failures(run_dir / "status" / "failures.csv"))
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


def _read_failures(path: Path) -> list[FailureRecord]:
    if not path.exists():
        return []
    try:
        frame = pd.read_csv(path)
    except EmptyDataError:
        return []
    failures = []
    for _, row in frame.iterrows():
        failures.append(
            FailureRecord(
                record_id=str(row.get("record_id", "")),
                analyzer=str(row.get("analyzer", "")),
                error_type=str(row.get("error_type", "")),
                message=str(row.get("message", "")),
            )
        )
    return failures


def _read_record_manifest(path: Path, token_sec: int, max_tokens: int) -> list[SleepRecord]:
    if not path.exists():
        return []
    try:
        frame = pd.read_csv(path)
    except EmptyDataError:
        return []
    records = []
    for _, row in frame.iterrows():
        records.append(
            SleepRecord(
                record_id=str(row["record_id"]),
                path=Path(str(row["path"])),
                split=str(row.get("split", "")),
                source=None if pd.isna(row.get("source")) else str(row.get("source")),
                duration_sec=float(row.get("duration_sec", 0.0)),
                token_sec=int(row.get("token_sec", token_sec)),
                max_tokens=int(row.get("max_tokens", max_tokens)),
                metadata={},
            )
        )
    return records
