from __future__ import annotations

import argparse
from pathlib import Path

from hypnodata.config import load_config
from hypnodata.pipeline import run_pipeline, validate_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert clinical EDF records to standardized hypnodata NPZ outputs.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Run EDF to NPZ conversion.")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--num-workers", type=int, default=1)
    run.add_argument("--dry-run", action="store_true")

    validate = subparsers.add_parser("validate", help="Run full hypnodata validation without writing NPZ records.")
    validate.add_argument("--config", type=Path, required=True)
    validate.add_argument("--output-dir", type=Path, required=True)
    validate.add_argument("--num-workers", type=int, default=1)

    validate_config = subparsers.add_parser("validate-config", help="Validate a hypnodata YAML config.")
    validate_config.add_argument("--config", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    if args.command == "validate-config":
        print(f"hypnodata config OK: {args.config}")
        return 0
    if args.command == "validate":
        failure_count = validate_pipeline(config, output_dir=args.output_dir, num_workers=args.num_workers)
        print(f"hypnodata validation output directory: {args.output_dir}")
        return 1 if failure_count else 0
    output_dir = run_pipeline(
        config,
        output_dir=args.output_dir,
        num_workers=args.num_workers,
        dry_run=args.dry_run,
    )
    print(f"hypnodata output directory: {output_dir}")
    return 0
