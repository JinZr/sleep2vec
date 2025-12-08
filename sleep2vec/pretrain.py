import argparse
import logging
import os
from pathlib import Path
import shutil
import sys

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy, DeepSpeedStrategy
import wandb

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sleep2vec.common import dump_cli_args_yaml
from sleep2vec.config import load_pretrain_config
from sleep2vec.sleep2vec_modelling import Sleep2vecPretraining
from sleep2vec.utils import get_pretrain_dataloader


def sleep2vec_pretrain(args):

    config_bundle = load_pretrain_config(args.config)
    model_config = config_bundle.model
    loss_config = config_bundle.loss
    averaging_config = config_bundle.averaging
    args.mask_rate = config_bundle.data.mask_rate
    args.max_tokens = config_bundle.data.max_tokens
    args.channel_names = [c.name for c in model_config.channels]
    args.backbone_arch = model_config.backbone.name

    # get data loaders
    train_loader, val_loader_main = get_pretrain_dataloader(args)

    # ========= 目录与 logger =========
    if args.ckpt_path is not None and os.path.isfile(args.ckpt_path):  # NEW
        # 1. 用旧目录继续
        save_path = os.path.dirname(args.ckpt_path)
        run_name = os.path.basename(os.path.dirname(save_path))
        logging.info(f"run_name: {run_name}")
        wandb_id = run_name  # 简单做法，也可手动传 id
    else:
        # 2. 全新训练：创建新目录
        exp_bits = [args.version_name, args.backbone_arch]
        extra_tag = getattr(args, "exp_info", "") or ""
        extra_tag = extra_tag.strip().replace(" ", "_")
        if extra_tag:
            exp_bits.append(extra_tag)
        exp_bits.append("unsupervised")
        run_name = "-".join(filter(None, exp_bits))
        save_path = f"log-pretrain/{run_name}/checkpoints"
        os.makedirs(save_path, exist_ok=True)
        wandb_id = None  # 让 wandb 自动分配
        args.ckpt_path = None  # 防止误传

    # Always stash the YAML used for this run alongside checkpoints.
    exp_dir = Path(save_path).parent
    exp_dir.mkdir(parents=True, exist_ok=True)
    dest_config = exp_dir / "config.yaml"
    try:
        shutil.copy2(args.config, dest_config)
        logging.info(f"Copied config to {dest_config}")
    except Exception as exc:  # pragma: no cover - best-effort
        logging.warning(f"Failed to copy config to {dest_config}: {exc}")

    cli_args_path = exp_dir / "cli_args.yaml"
    try:
        dump_cli_args_yaml(args, cli_args_path)
        logging.info(f"Saved CLI args to {cli_args_path}")
    except Exception as exc:  # pragma: no cover - best-effort
        logging.warning(f"Failed to write CLI args YAML to {cli_args_path}: {exc}")

    model = Sleep2vecPretraining(args, model_config, loss_config, averaging_config=averaging_config)

    logger = WandbLogger(
        project="sleep2vec-pretrain",
        name=f"s2v-pretrain-{run_name}",
        save_dir=os.path.dirname(save_path),
        id=wandb_id,  # NEW：保持同一个 run
        resume="allow" if wandb_id else None,  # NEW：若 id 存在则追加
    )

    monitor = "val_contrastive_acc"
    mode = "max"
    checkpoint_cb = ModelCheckpoint(
        dirpath=save_path,  # 你的 ckpt 目录
        monitor=monitor,  # 监控验证集 Cohen κ
        mode=mode,  # 越大越好
        filename="epoch={epoch}-step={step}",
        save_on_train_epoch_end=True,  # 只在 epoch 末保存
        every_n_epochs=1,  # 每个 epoch 都存
        save_top_k=50,  # -1 全部保留；你也可按需改成 3
    )

    early_stop_cb = EarlyStopping(
        monitor=monitor,  # 必须与 log 名一致
        patience=args.patience,  # 早停容忍 epoch 数
        mode=mode,  # 越小越好
        verbose=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    if args.strategy == "ddp":
        strategy = DDPStrategy(find_unused_parameters=True)
    elif args.strategy == "deepspeed":
        if args.deepspeed_config is None:
            raise ValueError("deepspeed_config must be provided when using DeepSpeed strategy.")
        strategy = DeepSpeedStrategy(
            config=args.deepspeed_config,
        )
    else:
        # fall back to Lightning's default strategy selection
        strategy = "auto"

    trainer = pl.Trainer(
        callbacks=[checkpoint_cb, early_stop_cb, lr_monitor],
        devices=args.devices,
        accelerator="gpu",
        strategy=strategy,
        benchmark=True,
        enable_checkpointing=True,
        logger=logger,
        max_epochs=args.epochs,
        log_every_n_steps=5,
        num_sanity_val_steps=0,
        precision=args.precision,
        gradient_clip_val=args.gradient_clip_val,
    )

    # train the model
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=[val_loader_main],
        ckpt_path=args.ckpt_path,
    )


if __name__ == "__main__":
    wandb.login()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="YAML file containing model and loss configuration.",
    )
    parser.add_argument("--epochs", type=int, default=120, help="number of epochs")
    parser.add_argument("--lr", type=float, default=5e-5, help="learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-2, help="weight decay for AdamW")
    parser.add_argument("--batch-size", type=int, default=320, help="batch size")
    parser.add_argument("--num-workers", type=int, default=8, help="number of dataloader workers")
    parser.add_argument("--devices", type=int, nargs="+", default=[0, 1], help="GPU device ids")
    parser.add_argument(
        "--ckpt-path",
        type=Path,
        default=None,
        help="要继续训练的 checkpoint 路径 (.ckpt)",
    )

    parser.add_argument(
        "--version-name",
        type=str,
        required=True,
        help="version name used for logging and checkpoint directory",
    )
    parser.add_argument("--patience", type=int, default=20, help="early stopping patience in epochs")
    parser.add_argument("--device", type=str, default="cuda", help="torch device used by dataloader")
    parser.add_argument(
        "--pretrain-data-index",
        type=Path,
        default="index/hsp_psg_pretrain.csv",
        help="CSV index file for pretraining data",
    )
    parser.add_argument(
        "--pretrain-preset-path",
        type=Path,
        default="/data/ywx/BIOT/data/5dataset_preset_120.pickle",
        help="path to precomputed preset pickle for PSG dataset",
    )
    parser.add_argument(
        "--exp-info",
        type=str,
        default="",
        help=(
            "Extra tag inserted into log-pretrain/<run_name>; useful for noting "
            "backbone variants or ablation identifiers."
        ),
    )

    parser.add_argument(
        "--precision",
        type=str,
        default="bf16",
        help="mixed precision setting passed to Lightning Trainer",
    )
    parser.add_argument("--gradient-clip-val", type=float, default=1.0, help="gradient clipping value")
    parser.add_argument(
        "--strategy",
        type=str,
        default="ddp",
        choices=["ddp", "deepspeed", "none"],
        help="distributed training strategy",
    )
    parser.add_argument(
        "--deepspeed-config",
        type=str,
        default=None,
        help="DeepSpeed config JSON path when strategy is 'deepspeed'",
    )

    args = parser.parse_args()
    logging.info(args)
    sleep2vec_pretrain(args)
