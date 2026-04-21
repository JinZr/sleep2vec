import argparse
import logging
from pathlib import Path
import random
import sys

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
import torch
import wandb

# Make sure the repository root is importable when running this file directly
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wrist2vec.checkpoints import average_checkpoints, select_checkpoints
from wrist2vec.common import apply_finetune_config
from wrist2vec.distributed import is_rank_zero_process
from wrist2vec.results import save_result_csv
from wrist2vec.wrist2vec_finetuning import Wrist2vecFinetuning
from wrist2vec.utils import _build_finetune_loader


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


def _init_wandb(args):
    if not args.wandb:
        return None
    if not is_rank_zero_process():
        return None

    init_kwargs = {
        "project": args.wandb_project,
        "name": args.wandb_name,
        "entity": args.wandb_entity,
        "group": args.wandb_group,
        "id": args.wandb_id,
        "resume": "allow" if args.wandb_id else None,
        "mode": args.wandb_mode,
        "config": {
            "config": str(args.config),
            "ckpt_path": args.ckpt_path,
            "label_name": args.label_name,
            "eval_split": args.eval_split,
            "batch_size": args.batch_size,
            "devices": args.devices,
            "accelerator": args.accelerator,
            "precision": args.precision,
            "avg_ckpts": args.avg_ckpts,
        },
    }
    init_kwargs = {k: v for k, v in init_kwargs.items() if v is not None}
    return wandb.init(**init_kwargs)


def run_inference(args):
    config_bundle, model_cfg = apply_finetune_config(args)
    if args.label_name == "ahi" and args.avg_ckpts > 1:
        raise ValueError(
            "AHI inference does not support average checkpoints because `ahi_eval_threshold` is checkpoint-specific."
        )

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
    model = Wrist2vecFinetuning(
        args,
        model_cfg,
        finetune_config=config_bundle.finetune,
        averaging_config=config_bundle.averaging,
    )

    ckpt_path = args.ckpt_path
    if args.avg_ckpts > 1:
        ckpt_dir = args.avg_ckpt_dir
        end_ckpt = None if args.ckpt_path in {"best", "last"} else Path(args.ckpt_path)
        if ckpt_dir is None:
            if end_ckpt is None:
                raise ValueError("Use --avg-ckpt-dir when averaging with ckpt-path=best/last.")
            ckpt_dir = end_ckpt.parent
        ckpt_paths = select_checkpoints(Path(ckpt_dir), end_ckpt=end_ckpt, num_ckpts=args.avg_ckpts)
        logging.info("Averaging checkpoints: %s", ", ".join(str(p) for p in ckpt_paths))
        avg_state = average_checkpoints(ckpt_paths, device=torch.device("cpu"))
        missing_keys, unexpected_keys = model.load_state_dict(avg_state, strict=False)
        if missing_keys:
            logging.warning("Missing keys when loading averaged checkpoint: %s", missing_keys)
        if unexpected_keys:
            logging.warning("Unexpected keys when loading averaged checkpoint: %s", unexpected_keys)
        ckpt_path = None

    logging.info("Running inference on split=%s with %s samples/batch", args.eval_split, args.batch_size)
    wandb_run = _init_wandb(args)
    try:
        test_results = trainer.test(model=model, ckpt_path=ckpt_path, dataloaders=dataloader)
        metrics = test_results[0] if test_results else {}
        logging.info("Inference metrics: %s", metrics)
    finally:
        if wandb_run is not None:
            primary_exc_active = sys.exc_info()[0] is not None
            try:
                wandb.finish()
            except BaseException as exc:
                if primary_exc_active:
                    logging.warning("wandb.finish() failed during inference cleanup: %s", exc)
                else:
                    raise

    if args.results_csv_path and metrics:
        save_result_csv(metrics, str(args.results_csv_path), args)


def parse_args():
    parser = argparse.ArgumentParser(description="Run downstream wrist2vec inference from a fine-tuned checkpoint.")
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
    parser.add_argument(
        "--label-name",
        type=str,
        required=True,
        help=(
            "downstream label to predict (built-ins: age, sex, stage3, stage4, stage5, ahi; "
            "custom labels require finetune.task in the YAML config)"
        ),
    )
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
        help="Optional pretrain-model init checkpoint to load before downstream weights.",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging (needed for confusion matrix logging).",
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
