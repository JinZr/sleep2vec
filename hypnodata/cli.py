from __future__ import annotations

import argparse
from pathlib import Path

from hypnodata.config import load_config
from hypnodata.pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert clinical EDF records to standardized hypnodata NPZ outputs.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Run EDF to NPZ conversion.")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--num-workers", type=int, default=1)
    run.add_argument("--limit", type=int, default=None)
    run.add_argument("--overwrite", action="store_true")
    run.add_argument("--resume", action="store_true")
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--crash", action="store_true")
    run.add_argument("--record-id", default=None)

    validate = subparsers.add_parser("validate-config", help="Validate a hypnodata YAML config.")
    validate.add_argument("--config", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    if args.command == "validate-config":
        print(f"hypnodata config OK: {args.config}")
        return 0
    output_dir = run_pipeline(
        config,
        output_dir=args.output_dir,
        num_workers=args.num_workers,
        limit=args.limit,
        overwrite=args.overwrite,
        resume=args.resume,
        dry_run=args.dry_run,
        crash=args.crash,
        record_id=args.record_id,
    )
    print(f"hypnodata output directory: {output_dir}")
    return 0
