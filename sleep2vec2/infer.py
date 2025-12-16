import argparse
import logging
from pathlib import Path
import random
import sys

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
import torch

# Make sure the repository root is importable when running this file directly
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sleep2vec2.common import apply_finetune_config
from sleep2vec2.metrics import save_result_csv
from sleep2vec2.sleep2vec_finetuning import Sleep2vecFinetuning
from sleep2vec2.utils import _build_finetune_loader


def _build_inference_loader(args):
    """Create a single split dataloader for evaluation-only runs."""
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.eval_split == "test":
        sources = args.override_dataset_names or args.test_dataset_names
    else:
        sources = args.override_dataset_names or args.train_dataset_names

    return _build_finetune_loader(
        args,
        split=[args.eval_split],
        sources=sources,
        shuffle=False,
        is_train_set=False,
    )


def run_inference(args):
    _, model_cfg = apply_finetune_config(args)

    trainer_precision = args.precision
    if args.accelerator == "cpu" and isinstance(trainer_precision, str) and "bf16" in trainer_precision:
        trainer_precision = 32

    strategy = (
        DDPStrategy(find_unused_parameters=True) if args.accelerator != "cpu" and len(args.devices) > 1 else "auto"
    )

    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices if args.accelerator != "cpu" else None,
        strategy=strategy,
        logger=False,
        enable_checkpointing=False,
        precision=trainer_precision,
    )

    dataloader = _build_inference_loader(args)
    model = Sleep2vecFinetuning(args, model_cfg)

    logging.info("Running inference on split=%s with %s samples/batch", args.eval_split, args.batch_size)
    test_results = trainer.test(model=model, ckpt_path=args.ckpt_path, dataloaders=dataloader)
    metrics = test_results[0] if test_results else {}
    logging.info("Inference metrics: %s", metrics)

    if args.results_csv_path and metrics:
        save_result_csv(metrics, str(args.results_csv_path), args)


def parse_args():
    parser = argparse.ArgumentParser(description="Run downstream sleep2vec inference from a fine-tuned checkpoint.")
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
        help="Checkpoint (.ckpt) path or Lightning alias such as 'best'/'last'.",
    )
    parser.add_argument("--label-name", type=str, default="age", help="Downstream target name.")
    parser.add_argument("--batch-size", type=int, default=12, help="Batch size for inference dataloader.")
    parser.add_argument("--num-workers", type=int, default=8, help="Number of dataloader workers.")
    parser.add_argument(
        "--devices",
        type=int,
        nargs="+",
        default=[0],
        help="Device ids passed to Lightning Trainer.",
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
        "--override-dataset-names",
        type=str,
        nargs="+",
        default=None,
        help="Optional dataset name list to override YAML train/test lists.",
    )
    parser.add_argument(
        "--results-csv-path",
        type=Path,
        default=None,
        help="Optional CSV path to append aggregated inference metrics.",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="bf16-mixed",
        help="Precision flag forwarded to Lightning Trainer.",
    )
    parser.add_argument("--seed", type=int, default=4523, help="Random seed for dataloader shuffling.")
    parser.add_argument(
        "--pretrained-backbone-path",
        type=str,
        default=None,
        help="Optional backbone checkpoint to load before downstream weights.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    if args.ckpt_path not in {"best", "last"}:
        ckpt_path = Path(args.ckpt_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        args.ckpt_path = str(ckpt_path)
    run_inference(args)
