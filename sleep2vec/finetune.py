import argparse
import logging
import os
from pathlib import Path
import shutil
import sys
import typing as t

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies.ddp import DDPStrategy
import wandb

# Make sure the repository root is importable when running this file directly
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sleep2vec.common import apply_finetune_config, dump_cli_args_yaml
from sleep2vec.metrics import save_result_csv
from sleep2vec.sleep2vec_finetuning import Sleep2vecFinetuning
from sleep2vec.utils import get_finetune_dataloaders

# from model.ahi_metric import AHIMetricsCollection


def _build_wandb_logger(*, args):
    if args.wandb_mode == "disabled":
        return False

    settings_kwargs: dict[str, t.Any] = {}
    if getattr(args, "wandb_init_timeout", None):
        settings_kwargs["init_timeout"] = args.wandb_init_timeout
    if getattr(args, "wandb_start_method", None):
        settings_kwargs["start_method"] = args.wandb_start_method

    wandb_settings = None
    if settings_kwargs:
        try:
            wandb_settings = wandb.Settings(**settings_kwargs)
        except TypeError:
            wandb_settings = wandb.Settings(init_timeout=settings_kwargs.get("init_timeout", 90))

    logger_kwargs: dict[str, t.Any] = dict(
        project=args.wandb_project,
        name=f"s2v-finetune-{args.version}",
        save_dir=args.wandb_save_dir,
        log_model=True,
    )

    if args.wandb_mode == "offline":
        logger_kwargs["offline"] = True
    if wandb_settings is not None:
        logger_kwargs["settings"] = wandb_settings

    try:
        return WandbLogger(**logger_kwargs)
    except TypeError:
        logger_kwargs.pop("offline", None)
        logger_kwargs.pop("settings", None)
        return WandbLogger(**logger_kwargs)


def prepare_dataloader(args):
    train_loader, val_loader, test_loader = get_finetune_dataloaders(args)

    logging.info(
        "Prepared dataloaders: train=%d val=%d test=%d",
        len(train_loader),
        len(val_loader),
        len(test_loader),
    )
    return train_loader, val_loader, test_loader


def supervised(args, config_bundle):
    model_config = config_bundle.model

    # Persist YAML alongside experiment artifacts
    exp_root = Path(f"log-finetune/{args.version}/")
    exp_root.mkdir(parents=True, exist_ok=True)
    dest_config = exp_root / "config.yaml"
    try:
        shutil.copy2(args.config, dest_config)
        logging.info(f"Copied config to {dest_config}")
    except Exception as exc:  # pragma: no cover - best-effort
        logging.warning(f"Failed to copy config to {dest_config}: {exc}")

    cli_args_path = exp_root / "cli_args.yaml"
    try:
        dump_cli_args_yaml(args, cli_args_path)
        logging.info(f"Saved CLI args to {cli_args_path}")
    except Exception as exc:  # pragma: no cover - best-effort
        logging.warning(f"Failed to write CLI args YAML to {cli_args_path}: {exc}")

    # get data loaders
    train_loader, val_loader, test_loader = prepare_dataloader(args)

    # define the model/lightning module
    model = Sleep2vecFinetuning(args, model_config)

    # logger and callbacks
    version = args.version
    logger = _build_wandb_logger(args=args)

    early_stop_callback = EarlyStopping(
        monitor=args.monitor,
        patience=args.patience,
        verbose=False,
        mode=args.monitor_mod,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"log-finetune/{version}/checkpoints",  # ← 你想要的目录
        monitor=args.monitor,  # 监控验证集 Cohen κ
        mode=args.monitor_mod,  # 越大越好
        save_top_k=1,  # 只保留最优一个
        filename="{epoch:02d}",
    )

    callbacks = [early_stop_callback, checkpoint_callback]
    enable_checkpointing = True
    trainer_kwargs = dict(
        devices=args.devices,
        accelerator="gpu",
        strategy=DDPStrategy(find_unused_parameters=True),
        # strategy=DeepSpeedStrategy(config="ds_config.json"),  # ← 就这行！
        benchmark=True,
        logger=logger,
        max_epochs=args.epochs,
        log_every_n_steps=args.log_every_n_steps,
        gradient_clip_val=1.0,
        precision="bf16-mixed",  # <---- 开启 BF16
        check_val_every_n_epoch=args.check_val_every_n_epoch,
    )
    if args.print_diagnostics:
        callbacks = []
        enable_checkpointing = False
        trainer_kwargs.update(
            dict(
                enable_progress_bar=False,
                max_steps=args.diagnostics_steps,
                limit_val_batches=0,
                log_every_n_steps=1,
            )
        )

    trainer = pl.Trainer(
        callbacks=callbacks,
        enable_checkpointing=enable_checkpointing,
        **trainer_kwargs,
    )

    if args.epochs > 0:
        # train the model
        trainer.fit(
            model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
            ckpt_path=args.ckpt_path if args.ckpt_path != "" else None,
        )

    if args.print_diagnostics:
        # only collect diagnostics, skip evaluation
        return

    # test the model
    pretrain_result = trainer.test(
        model=model,
        ckpt_path="best" if args.epochs > 0 else args.ckpt_path,
        dataloaders=test_loader,
    )[0]
    logging.info(pretrain_result)
    save_result_csv(pretrain_result, args.results_csv_path, args)


def build_version_name(args) -> str:
    """Return a stable run name based on config and CLI flags."""
    if args.version_name:
        return args.version_name

    chosen_channels = getattr(args, "data_channel_names", None) or getattr(args, "channel_names", None) or []
    if not chosen_channels:
        ch_stub = "mixed"
    elif len(chosen_channels) == len(args.channel_names):
        ch_stub = "full"
    elif len(chosen_channels) == 1:
        ch_stub = chosen_channels[0]
    else:
        ch_stub = "-".join(chosen_channels)

    few_shot = getattr(args, "n_few_shot", None)
    if few_shot is None or (isinstance(few_shot, (int, float)) and few_shot <= 0):
        few_stub = "fullset"
    else:
        few_stub = f"fewshot-{few_shot}"

    pretrain_suffix = "with_pretrain" if args.pretrained_backbone_path else "from_scratch"
    pieces = [
        args.version_prefix,
        args.label_name,
        ch_stub,
        few_stub,
        pretrain_suffix,
    ]
    if args.version_tag:
        pieces.append(args.version_tag)

    return "-".join(pieces)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune sleep2vec downstream models on PSG data.")

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="YAML file containing model and loss configuration.",
    )
    # ---------------- Optimization & training hyper-parameters ----------------
    parser.add_argument("--epochs", type=int, default=200, help="number of fine-tuning epochs")
    parser.add_argument("--lr", type=float, default=1e-6, help="learning rate for AdamW")
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
        help="early stopping patience in epochs (no improvement)",
    )

    # ---------------- Hardware / device configuration ----------------
    parser.add_argument(
        "--devices",
        type=int,
        nargs="+",
        default=[0, 1],
        help="GPU device ids for PyTorch Lightning Trainer",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="torch device string passed to model and dataloaders",
    )
    parser.add_argument(
        "--print-diagnostics",
        action="store_true",
        help="Run a short diagnostic pass (few batches), print tensor stats, and exit (progress bar off).",
    )
    parser.add_argument(
        "--diagnostics-steps",
        type=int,
        default=5,
        help="Number of training steps to gather diagnostics before stopping.",
    )

    # ---------------- Task & data configuration ----------------
    parser.add_argument(
        "--label-name",
        type=str,
        required=True,
        choices=["age", "sex", "stage5"],
        help="downstream label to predict (e.g. age, sex, stage5)",
    )
    # ---------------- Data/configuration now YAML-driven; keep CLI for ckpt paths only ----------------
    parser.add_argument(
        "--pretrained-backbone-path",
        type=str,
        default=None,
        help="optional path to pretrained backbone checkpoint (.ckpt)",
    )
    parser.add_argument(
        "--ckpt-path",
        type=str,
        default=None,
        help="optional checkpoint (.ckpt) to resume fine-tuning / testing",
    )

    # ---------------- Logging / versioning ----------------
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
    parser.add_argument(
        "--check-val-every-n-epoch",
        dest="check_val_every_n_epoch",
        type=int,
        default=1,
        help="run validation every N epochs",
    )
    parser.add_argument(
        "--log-every-n-steps",
        dest="log_every_n_steps",
        type=int,
        default=5,
        help="emit logger metrics every N training steps (W&B/TensorBoard); useful for short runs/diagnostics",
    )
    parser.add_argument(
        "--wandb-mode",
        type=str,
        default=os.environ.get("WANDB_MODE", "online"),
        choices=["online", "offline", "disabled"],
        help="W&B mode: 'online' (default), 'offline' (no network, later `wandb sync`), or 'disabled'",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=os.environ.get("WANDB_PROJECT", "sleep2vec-finetune"),
        help="W&B project name (overrides WANDB_PROJECT)",
    )
    parser.add_argument(
        "--wandb-save-dir",
        type=str,
        default=os.environ.get("WANDB_DIR", "./wandb_logs"),
        help="Local W&B directory for run files (overrides WANDB_DIR).",
    )
    parser.add_argument(
        "--wandb-init-timeout",
        dest="wandb_init_timeout",
        type=int,
        default=int(os.environ.get("WANDB_INIT_TIMEOUT", "90")),
        help="Seconds to wait for `wandb.init()` before timing out (default: 90).",
    )
    parser.add_argument(
        "--wandb-start-method",
        dest="wandb_start_method",
        type=str,
        default=os.environ.get("WANDB_START_METHOD", ""),
        help="Optional W&B start method (e.g., 'thread') for constrained HPC environments.",
    )

    args = parser.parse_args()
    if not getattr(args, "wandb_start_method", ""):
        args.wandb_start_method = None

    config_bundle, _ = apply_finetune_config(args)

    # ---- Build version string used by WandB and checkpoint directory ----
    args.version = build_version_name(args)

    logging.info(args)
    if args.wandb_mode == "online" and int(os.environ.get("RANK", "0")) == 0:
        try:
            wandb.login()
        except Exception as exc:  # pragma: no cover
            logging.warning("wandb.login() failed; relying on wandb.init(): %s", exc)

    # Run fine-tuning
    supervised(args, config_bundle)
