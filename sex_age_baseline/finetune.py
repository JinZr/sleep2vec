from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .config import load_config
from .runtime import train_and_save


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a YAML-selected age/sex/BMI covariate baseline.")
    parser.add_argument("--config", type=Path, required=True, help="Sex/age baseline YAML config.")
    parser.add_argument("--label-name", type=str, required=True, help="Downstream label namespace for result files.")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-6, help="AdamW learning rate.")
    parser.add_argument("--warmup-steps", type=int, default=None, help="Linear warmup optimizer steps; defaults to 3%.")
    parser.add_argument("--lr-decay-shape", choices=["cosine", "linear"], default="cosine", help="Decay after warmup.")
    parser.add_argument("--lr-decay-floor", type=float, default=0.1, help="Final to peak learning-rate ratio.")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="AdamW weight decay.")
    parser.add_argument("--batch-size", type=int, default=12, help="Batch size.")
    parser.add_argument("--num-workers", type=int, default=8, help="DataLoader workers.")
    parser.add_argument("--patience", type=int, default=100, help="Early stopping patience.")
    parser.add_argument("--gradient-clip-val", type=float, default=1.0, help="Gradient clipping value.")
    parser.add_argument("--accumulate-grad-batches", type=int, default=1, help="Gradient accumulation batches.")
    parser.add_argument("--precision", type=str, default="bf16-mixed", help="Lightning computation precision.")
    parser.add_argument(
        "--devices", type=int, nargs="+", default=[0], help="GPU IDs; CPU mode uses this many processes."
    )
    parser.add_argument(
        "--device", choices=["cpu", "cuda"], default="cuda", help="Compute device; GPU IDs use --devices."
    )
    parser.add_argument("--ckpt-path", type=str, default=None, help="Optional sex_age_baseline checkpoint to load.")
    parser.add_argument("--version-name", type=str, default=None, help="Run name for logs and checkpoints.")
    parser.add_argument("--results-csv-path", type=Path, required=True, help="Aggregated metrics CSV path.")
    parser.add_argument("--seed", type=int, default=4523, help="Random seed.")
    parser.add_argument(
        "--wandb-mode",
        choices=["online", "offline", "disabled"],
        default=None,
        help="Optional W&B metric logging mode; omitted disables W&B logging.",
    )
    parser.add_argument(
        "--test-after-fit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Run test evaluation after fit by default. Use --no-test-after-fit during validation-based model "
            "selection; evaluate test separately from the selected checkpoint."
        ),
    )
    parser.add_argument(
        "--test-all-checkpoints-after-fit",
        action="store_true",
        help="After fitting, evaluate every saved epoch=*.ckpt on test; requires --test-after-fit.",
    )
    parser.add_argument(
        "--check-val-every-n-epoch",
        type=int,
        default=1,
        help="Run validation every N training epochs.",
    )
    parser.add_argument("--ckpt-every-n-epochs", type=int, default=1, help="Save epoch checkpoints every N epochs.")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    cfg = load_config(args.config, validate_sidecars=True)
    train_and_save(args, cfg)


if __name__ == "__main__":
    main()
