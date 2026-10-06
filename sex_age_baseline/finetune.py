from __future__ import annotations

import argparse
import logging
from pathlib import Path

import wandb

from .config import load_config
from .runtime import train_and_save

# Accepted so the variant CLI matches sleep2vec; the covariate MLP has no backbone or diagnostics pass.
_UNSUPPORTED_OPTIONS = (
    ("--pretrained-backbone-path", "pretrained_backbone_path"),
    ("--print-diagnostics", "print_diagnostics"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a YAML-selected age/sex/BMI covariate baseline.")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="YAML file containing model and loss configuration.",
    )
    parser.add_argument("--epochs", type=int, default=30, help="number of fine-tuning epochs")
    parser.add_argument("--lr", type=float, default=1e-6, help="learning rate for AdamW")
    parser.add_argument(
        "--lr-scheduler",
        choices=["decay", "wsd", "plateau"],
        default="decay",
        help="LR schedule: warmup/decay (default), warmup/stable/decay, or validation-driven plateau.",
    )
    parser.add_argument(
        "--lr-decay-ratio",
        type=float,
        default=None,
        help="WSD only: required fraction of total optimizer steps spent in the final decay phase.",
    )
    parser.add_argument(
        "--lr-plateau-factor",
        type=float,
        default=None,
        help="Plateau only: LR multiplier after a plateau (default: 0.1).",
    )
    parser.add_argument(
        "--lr-plateau-patience",
        type=int,
        default=None,
        help="Plateau only: tolerated validation checks without improvement before reducing LR (default: 10).",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=None,
        help="Override warmup steps for LR schedule (default: 3%% of total steps).",
    )
    parser.add_argument(
        "--lr-decay-floor",
        type=float,
        default=0.1,
        help="Final LR ratio for decay/WSD, or minimum ratio per parameter group for plateau.",
    )
    parser.add_argument(
        "--lr-decay-shape",
        choices=["cosine", "linear"],
        default="cosine",
        help="Post-warmup LR decay shape.",
    )
    parser.add_argument(
        "--weight-decay",
        dest="weight_decay",
        type=float,
        default=1e-5,
        help="weight decay for AdamW",
    )
    parser.add_argument("--batch-size", type=int, default=12, help="batch size for dataloader")
    parser.add_argument("--num-workers", type=int, default=8, help="number of dataloader workers")
    parser.add_argument(
        "--patience",
        type=int,
        default=100,
        help="early stopping patience in validation checks without improvement",
    )
    parser.add_argument("--gradient-clip-val", type=float, default=1.0, help="gradient clipping value")
    parser.add_argument(
        "--accumulate-grad-batches",
        type=int,
        default=1,
        help="Number of batches to accumulate before each optimizer step.",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="bf16",
        choices=[
            "transformer-engine",
            "transformer-engine-float16",
            "16-true",
            "16-mixed",
            "bf16-true",
            "bf16-mixed",
            "32-true",
            "64-true",
            "64",
            "32",
            "16",
            "bf16",
        ],
        help="mixed precision setting passed to Lightning Trainer",
    )
    parser.add_argument(
        "--devices",
        type=int,
        nargs="+",
        default=[0, 1],
        help="GPU device ids for PyTorch Lightning Trainer; with --device cpu, their count sets the CPU processes",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="torch device string; 'cpu' trains on CPU processes, any other value on the GPUs in --devices",
    )
    parser.add_argument(
        "--print-diagnostics",
        action="store_true",
        help="Unsupported by sex_age_baseline; setting it fails.",
    )
    parser.add_argument(
        "--diagnostics-steps",
        type=int,
        default=5,
        help="Inert in sex_age_baseline: it only applies with --print-diagnostics, which fails.",
    )
    parser.add_argument(
        "--label-name",
        type=str,
        required=True,
        help="downstream label name for result files; task semantics come from finetune.task in the YAML config",
    )
    parser.add_argument(
        "--pretrained-backbone-path",
        type=str,
        default=None,
        help="Unsupported by sex_age_baseline (no pretrained backbone); setting it fails.",
    )
    parser.add_argument(
        "--ckpt-path",
        type=str,
        default=None,
        help="optional sex_age_baseline checkpoint (.ckpt) whose weights initialize fine-tuning / testing",
    )
    parser.add_argument(
        "--version-name",
        type=str,
        default=None,
        help=("explicit run name for logging and checkpoint directory; " "if not set, a name will be generated"),
    )
    parser.add_argument(
        "--version-prefix",
        type=str,
        default="psg-finetune",
        help="prefix used when auto-generating version name",
    )
    parser.add_argument(
        "--version-tag",
        type=str,
        default="",
        help="optional suffix appended to auto-generated version name",
    )
    parser.add_argument(
        "--results-csv-path",
        type=Path,
        required=True,
        help="path to the CSV file storing aggregated evaluation metrics",
    )
    parser.add_argument("--wandb-project", type=str, default=None, help="W&B project name.")
    parser.add_argument("--wandb-group", type=str, default=None, help="W&B run group.")
    parser.add_argument("--wandb-mode", type=str, default=None, help="W&B run mode.")
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
        dest="check_val_every_n_epoch",
        type=int,
        default=1,
        help="run validation every N epochs",
    )
    parser.add_argument(
        "--ckpt-every-n-epochs",
        dest="ckpt_every_n_epochs",
        type=int,
        default=1,
        help="save checkpoints every N epochs",
    )
    parser.add_argument(
        "--export-predictions",
        action="store_true",
        help="Save test predictions for every evaluated checkpoint to the run's predictions.csv (default: disabled).",
    )
    args = parser.parse_args()
    unsupported = [option for option, dest in _UNSUPPORTED_OPTIONS if getattr(args, dest) != parser.get_default(dest)]
    if unsupported:
        raise ValueError(f"sex_age_baseline finetuning does not support {', '.join(unsupported)}.")
    return args


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    cfg = load_config(args.config, validate_sidecars=True)
    if args.wandb_mode not in {"offline", "disabled"}:
        wandb.login()
    train_and_save(args, cfg)


if __name__ == "__main__":
    main()
