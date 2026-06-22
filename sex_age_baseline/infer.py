from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .config import load_config
from .runtime import run_inference_and_save


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run age/sex-only baseline inference.")
    parser.add_argument("--config", type=Path, required=True, help="Sex/age baseline YAML config.")
    parser.add_argument("--ckpt-path", type=str, required=True, help="Sex/age baseline checkpoint path.")
    parser.add_argument("--label-name", type=str, required=True, help="Downstream label namespace for result files.")
    parser.add_argument(
        "--eval-split", type=str, default="test", choices=["train", "val", "test"], help="Split to evaluate."
    )
    parser.add_argument("--batch-size", type=int, default=12, help="Batch size.")
    parser.add_argument("--num-workers", type=int, default=8, help="DataLoader workers.")
    parser.add_argument("--devices", type=int, nargs="+", default=[0], help="Accepted for recipe parity; unused.")
    parser.add_argument(
        "--accelerator", type=str, default="gpu", choices=["cpu", "gpu", "auto"], help="Runtime accelerator hint."
    )
    parser.add_argument("--device", type=str, default="cuda", help="Torch device string.")
    parser.add_argument("--precision", type=str, default="bf16-mixed", help="Accepted for recipe parity; unused.")
    parser.add_argument("--lr", type=float, default=1e-6, help="Accepted for result parity; unused.")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Accepted for result parity; unused.")
    parser.add_argument(
        "--avg-ckpts", type=int, default=1, help="Checkpoint averaging is not supported for this baseline."
    )
    parser.add_argument("--avg-ckpt-dir", type=Path, default=None, help="Accepted for recipe parity; unused.")
    parser.add_argument("--seed", type=int, default=4523, help="Random seed.")
    parser.add_argument(
        "--wandb-mode",
        type=str,
        default=None,
        choices=["online", "offline", "disabled"],
        help="Accepted for recipe parity; unused.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    if args.avg_ckpts != 1:
        raise ValueError("sex_age_baseline inference does not support checkpoint averaging.")
    if args.accelerator == "cpu" and args.device == "cuda":
        args.device = "cpu"
    ckpt_path = Path(args.ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    args.ckpt_path = str(ckpt_path)
    cfg = load_config(args.config, validate_sidecars=True)
    run_inference_and_save(args, cfg)


if __name__ == "__main__":
    main()
