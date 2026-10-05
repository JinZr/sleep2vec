from __future__ import annotations

import argparse
import logging
from pathlib import Path

from sleep2vec.results import DEFAULT_INFERENCE_RESULTS_ROOT

from .config import load_config
from .runtime import run_inference_and_save


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YAML-selected age/sex/BMI covariate baseline inference.")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="YAML config used for downstream finetuning.",
    )
    parser.add_argument(
        "--ckpt-path",
        type=str,
        required=True,
        help="Checkpoint (.ckpt) path; 'best'/'last' only select the end of a --avg-ckpt-dir average.",
    )
    parser.add_argument(
        "--label-name",
        type=str,
        required=True,
        help="downstream label name for result files; task semantics come from finetune.task in the YAML config",
    )
    parser.add_argument("--batch-size", type=int, default=12, help="Batch size for inference dataloader.")
    parser.add_argument("--num-workers", type=int, default=8, help="Number of dataloader workers.")
    parser.add_argument(
        "--devices",
        type=int,
        nargs="+",
        default=[0],
        help="Device ids passed to Lightning Trainer; with --device cpu, their count sets the CPU processes.",
    )
    parser.add_argument(
        "--accelerator",
        type=str,
        default="gpu",
        choices=["cpu", "gpu", "auto"],
        help="Device accelerator used by Lightning.",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Torch device string passed into models.")
    parser.add_argument("--lr", type=float, default=1e-6, help="Learning rate placeholder used by optimizer init.")
    parser.add_argument(
        "--weight-decay",
        dest="weight_decay",
        type=float,
        default=1e-5,
        help="Weight decay placeholder used by optimizer init.",
    )
    parser.add_argument(
        "--eval-split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate.",
    )
    parser.add_argument(
        "--inference-preset-path",
        type=Path,
        default=None,
        help="Optional preset pickle path for this inference run; overrides data.finetune_preset_path from YAML.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_INFERENCE_RESULTS_ROOT,
        help="Root directory for inference result artifacts.",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="bf16-mixed",
        help="Precision flag forwarded to Lightning Trainer.",
    )
    parser.add_argument(
        "--avg-ckpts",
        type=int,
        default=1,
        help="Average this many checkpoints before inference (1 disables averaging).",
    )
    parser.add_argument(
        "--avg-ckpt-dir",
        type=Path,
        default=None,
        help="Optional checkpoint directory for averaging (defaults to ckpt_path parent).",
    )
    parser.add_argument("--seed", type=int, default=4523, help="Random seed for dataloader shuffling.")
    parser.add_argument(
        "--pretrained-backbone-path",
        type=str,
        default=None,
        help="Unsupported by sex_age_baseline (no pretrained backbone); setting it fails.",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging of inference metrics and result files.",
    )
    parser.add_argument("--wandb-project", type=str, default=None, help="W&B project name.")
    parser.add_argument("--wandb-name", type=str, default=None, help="W&B run name.")
    parser.add_argument("--wandb-entity", type=str, default=None, help="W&B entity/team.")
    parser.add_argument("--wandb-group", type=str, default=None, help="W&B group name.")
    parser.add_argument("--wandb-id", type=str, default=None, help="W&B run id (for resume).")
    parser.add_argument(
        "--wandb-mode",
        type=str,
        default=None,
        choices=["online", "offline", "disabled"],
        help="W&B mode override (online/offline/disabled).",
    )
    parser.add_argument(
        "--no-wandb-artifact",
        dest="wandb_artifact",
        action="store_false",
        default=True,
        help="Log inference metrics to W&B without uploading CSV artifacts.",
    )
    args = parser.parse_args()
    if args.pretrained_backbone_path is not None:
        # Accepted so the variant CLI matches sleep2vec; the covariate MLP has no backbone.
        raise ValueError("sex_age_baseline inference does not support --pretrained-backbone-path.")
    return args


def run_inference(args: argparse.Namespace) -> None:
    if args.accelerator == "cpu" and args.device == "cuda":
        args.device = "cpu"
    if args.ckpt_path not in {"best", "last"}:
        ckpt_path = Path(args.ckpt_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        args.ckpt_path = str(ckpt_path)
    cfg = load_config(args.config, validate_sidecars=True)
    run_inference_and_save(args, cfg)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_inference(parse_args())


if __name__ == "__main__":
    main()
