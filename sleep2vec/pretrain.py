import argparse
import logging
import os
from pathlib import Path
import shutil
import sys
import typing as t

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy, DeepSpeedStrategy
import wandb

# Make sure the repository root is importable when running this file directly
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.samplers import handles_distributed_sharding
from sleep2vec.callbacks.pair_acc_logger import PairAccLoggerCallback
from sleep2vec.common import dump_cli_args_yaml
from sleep2vec.config import load_pretrain_config
from sleep2vec.sleep2vec_modelling import Sleep2vecPretraining
from sleep2vec.utils import get_pretrain_dataloader


def _build_wandb_logger(*, args, run_name: str, save_dir: str, wandb_id: str | None):
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
        name=f"s2v-pretrain-{run_name}",
        save_dir=save_dir,
        id=wandb_id,
        resume="allow" if wandb_id else None,
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
    train_loader, val_loaders = get_pretrain_dataloader(args)
    # Disable Lightning's distributed sampler injection only when our custom
    # batch sampler already shards across ranks.
    train_batch_sampler = getattr(train_loader, "batch_sampler", None)
    main_val_loader = val_loaders[0] if val_loaders else None
    val_batch_sampler = getattr(main_val_loader, "batch_sampler", None)
    use_distributed_sampler = not handles_distributed_sharding(
        train_batch_sampler
    ) and not handles_distributed_sharding(val_batch_sampler)

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

    logger = _build_wandb_logger(
        args=args,
        run_name=run_name,
        save_dir=os.path.dirname(save_path),
        wandb_id=wandb_id,
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

    pair_acc_cb = PairAccLoggerCallback(args.channel_names)
    callbacks = [checkpoint_cb, early_stop_cb, lr_monitor, pair_acc_cb]
    enable_checkpointing = True
    trainer_kwargs = dict(
        devices=args.devices,
        accelerator="gpu",
        strategy=strategy,
        benchmark=True,
        # Custom bucketed batch sampler handles distributed sharding itself.
        # Disable Lightning's distributed sampler injection only in that case.
        use_distributed_sampler=use_distributed_sampler,
        logger=logger,
        max_epochs=args.epochs,
        log_every_n_steps=args.log_every_n_steps,
        num_sanity_val_steps=0,
        precision=args.precision,
        gradient_clip_val=args.gradient_clip_val,
    )
    if args.print_diagnostics:
        # short diagnostic run: disable bar/ckpt/val to keep output clean
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

    # train the model
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loaders if not args.print_diagnostics else None,
        ckpt_path=args.ckpt_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="YAML file containing model and loss configuration.",
    )
    parser.add_argument("--epochs", type=int, default=120, help="number of epochs")
    parser.add_argument("--lr", type=float, default=5e-5, help="learning rate")
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=None,
        help="Override warmup steps for LR schedule (default: 3% of total steps).",
    )
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
        "--allow-missing-channels",
        action="store_true",
        help="Allow samples with missing channels (default: require all configured channels).",
    )
    parser.add_argument(
        "--min-channels",
        type=int,
        default=6,
        help="Minimum available channels required when --allow-missing-channels is enabled.",
    )
    parser.add_argument(
        "--bucket-by-available-channels",
        dest="bucket_by_available_channels",
        action="store_true",
        default=True,
        help="Bucket batches by available-channel signature (default: enabled when allowing missing channels).",
    )
    parser.add_argument(
        "--no-bucket-by-available-channels",
        dest="bucket_by_available_channels",
        action="store_false",
        help="Disable available-channel bucketing even when allowing missing channels.",
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
        "--print-diagnostics",
        action="store_true",
        help="Run a short batch or two, print tensor diagnostics, and exit (disables progress bar).",
    )
    parser.add_argument(
        "--diagnostics-steps",
        type=int,
        default=5,
        help="Number of training steps to accumulate diagnostics before stopping.",
    )
    parser.add_argument("--gradient-clip-val", type=float, default=1.0, help="gradient clipping value")
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
        default=os.environ.get("WANDB_PROJECT", "sleep2vec-pretrain"),
        help="W&B project name (overrides WANDB_PROJECT)",
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
    if not getattr(args, "wandb_start_method", ""):
        args.wandb_start_method = None
    if args.wandb_mode == "online" and int(os.environ.get("RANK", "0")) == 0:
        try:
            wandb.login()
        except Exception as exc:  # pragma: no cover
            logging.warning("wandb.login() failed; relying on wandb.init(): %s", exc)
    sleep2vec_pretrain(args)
