from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.errors import EmptyDataError

from sleep2stat.analyzers.yasa import _finite_float, _sex_to_male
from sleep2stat.config import load_config
from sleep2stat.core.artifacts import FailureRecord
from sleep2stat.core.pipeline import run_pipeline
from sleep2stat.finalize import cohort_finalize
from sleep2stat.io.records import SleepRecord, load_records
from sleep2stat.io.writers import RUN_TERMINAL_STATUSES, AnalysisBundleWriter, _require_terminal_run_manifest
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
    validate.add_argument("--check-records", action="store_true")
    validate.add_argument("--split", nargs="+", default=None)
    validate.add_argument("--limit-records", type=int, default=None)

    summarize = subparsers.add_parser("summarize", help="Summarize a completed sleep2stat run directory.")
    summarize.add_argument("--run-dir", type=Path, required=True)
    summarize.add_argument("--num-workers", type=int, default=8)

    finalize = subparsers.add_parser("cohort-finalize", help="Merge completed sleep2stat cohort run tables.")
    finalize.add_argument("--output-run-dir", type=Path, required=True)
    finalize.add_argument("--input-run-dir", type=Path, action="append", required=True)
    finalize.add_argument("--plot-cohort", action="store_true")
    finalize.add_argument("--group-column", default="source")
    finalize.add_argument("--stage-source", default=None)
    finalize.add_argument("--adjust-covariates", nargs="*", default=None)

    plot = subparsers.add_parser("plot-record", help="Plot one sleep2stat per-record output directory.")
    plot.add_argument("--run-dir", type=Path, required=True)
    plot.add_argument("--record-id", required=True)

    cohort = subparsers.add_parser("plot-cohort", help="Plot sleep2stat cohort-level sleep architecture.")
    cohort.add_argument("--run-dir", type=Path, required=True)
    cohort.add_argument("--group-column", default="source")
    cohort.add_argument("--stage-source", default=None)
    cohort.add_argument("--adjust-covariates", nargs="*", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "validate-config":
        config = load_config(args.config)
        if args.check_records:
            return 0 if _check_records(config, args) else 1
        print(f"sleep2stat config OK: {args.config}")
        return 0
    if args.command == "summarize":
        _summarize(args.run_dir, num_workers=args.num_workers)
        return 0
    if args.command == "cohort-finalize":
        manifest = cohort_finalize(args.output_run_dir, args.input_run_dir)
        print(f"run_dir: {args.output_run_dir}")
        print(f"night_stats_rows: {manifest['night_stats_rows']}")
        print(f"failure_rows: {manifest['failure_rows']}")
        if args.plot_cohort:
            for path in plot_cohort(
                args.output_run_dir,
                group_column=args.group_column,
                stage_source=args.stage_source,
                adjust_covariates=args.adjust_covariates,
            ):
                print(path)
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


def _check_records(config, args: argparse.Namespace) -> bool:
    records = load_records(config.data, split_override=args.split, limit=args.limit_records)
    print(f"sleep2stat records OK: {len(records)}")
    if not any(analyzer.enabled and analyzer.type == "yasa_stage" for analyzer in config.analyzers):
        return True
    sex = _metadata_parse_summary(records, "sex", _sex_to_male)
    age = _metadata_parse_summary(records, "age", _finite_float)
    print(
        "YASA metadata sex: "
        f"present {sex['present']}/{sex['total']}, "
        f"convertible_to_male {sex['convertible']}/{sex['present']}, "
        f"nonconvertible_examples {sex['examples']}"
    )
    print(
        "YASA metadata age: "
        f"present {age['present']}/{age['total']}, "
        f"finite {age['convertible']}/{age['present']}, "
        f"nonconvertible_examples {age['examples']}"
    )
    if sex["total"] > 0 and sex["present"] == sex["total"] and sex["convertible"] == 0:
        print("ERROR: YASA metadata sex is present for all records but 0 are convertible to male.")
        return False
    return True


def _metadata_parse_summary(records: list[SleepRecord], key: str, parser) -> dict[str, Any]:
    present_values = [record.metadata.get(key) for record in records if record.metadata.get(key) is not None]
    convertible = [value for value in present_values if parser(value) is not None]
    examples = []
    for value in present_values:
        if parser(value) is None and repr(value) not in examples:
            examples.append(repr(value))
        if len(examples) >= 5:
            break
    return {
        "total": len(records),
        "present": len(present_values),
        "convertible": len(convertible),
        "examples": examples,
    }


def _summarize(run_dir: Path, *, num_workers: int = 8) -> None:
    _require_terminal_run_manifest(run_dir, RUN_TERMINAL_STATUSES, command="summarize")
    config_path = run_dir / "config.yaml"
    if config_path.exists():
        config = load_config(config_path)
        config = replace(config, run=replace(config.run, output_dir=run_dir))
        writer = AnalysisBundleWriter(config)
        records = _read_record_manifest(run_dir / "record_manifest.csv", config.data.token_sec, config.data.max_tokens)
        writer.rebuild_global_tables(
            records,
            _read_failures(run_dir / "status" / "failures.csv"),
            num_workers=num_workers,
        )
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
