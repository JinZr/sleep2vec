import argparse
import logging
from pathlib import Path
import shutil
import sys

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
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
    averaging_config = config_bundle.averaging

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
    model = Sleep2vecFinetuning(
        args,
        model_config,
        finetune_config=config_bundle.finetune,
        averaging_config=averaging_config,
    )

    # logger and callbacks
    version = args.version
    logger = WandbLogger(
        project="sleep2vec-moe-finetune",  # 相当于 TensorBoard 的 log dir
        name=f"moe-finetune-{version}",  # run 名称
        save_dir="./wandb_logs",  # 本地缓存目录，可选
        log_model=True,  # 训练结束自动保存 ckpt
    )

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
        save_top_k=-1,  # 保留全部 checkpoint
        save_last=True,  # 额外保存 last.ckpt
        every_n_epochs=args.ckpt_every_n_epochs,  # 控制保存频率
        filename="{epoch:02d}",
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks = [early_stop_callback, checkpoint_callback, lr_monitor]
    enable_checkpointing = True
    trainer_kwargs = dict(
        devices=args.devices,
        accelerator="gpu",
        strategy=DDPStrategy(find_unused_parameters=True),
        # strategy=DeepSpeedStrategy(config="ds_config.json"),  # ← 就这行！
        benchmark=True,
        logger=logger,
        max_epochs=args.epochs,
        gradient_clip_val=args.gradient_clip_val,
        precision=args.precision,
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
        # Persist a stable best.ckpt for downstream convenience.
        if enable_checkpointing and trainer.is_global_zero:
            best_path = checkpoint_callback.best_model_path
            if best_path:
                best_dest = Path(checkpoint_callback.dirpath) / "best.ckpt"
                try:
                    if Path(best_path).resolve() != best_dest.resolve():
                        shutil.copy2(best_path, best_dest)
                except Exception as exc:  # pragma: no cover - best-effort
                    logging.warning(f"Failed to copy best checkpoint to {best_dest}: {exc}")

    if args.print_diagnostics:
        # only collect diagnostics, skip evaluation
        return

    # test the model
    if args.epochs > 0:
        ckpt_path = checkpoint_callback.best_model_path or "last"
    else:
        ckpt_path = args.ckpt_path if args.ckpt_path != "" else None
    pretrain_result = trainer.test(
        model=model,
        ckpt_path=ckpt_path,
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
    # Login to WandB only when running as a script
    wandb.login()

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
        "--warmup-steps",
        type=int,
        default=None,
        help="Override warmup steps for LR schedule (default: 3% of total steps).",
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
        help="early stopping patience in epochs (no improvement)",
    )
    parser.add_argument("--gradient-clip-val", type=float, default=1.0, help="gradient clipping value")
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
        help=(
            "downstream label to predict (built-ins: age, sex, stage5; "
            "custom labels require finetune.task in the YAML config)"
        ),
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
        "--ckpt-every-n-epochs",
        dest="ckpt_every_n_epochs",
        type=int,
        default=1,
        help="save checkpoints every N epochs",
    )

    args = parser.parse_args()

    config_bundle, _ = apply_finetune_config(args)

    # ---- Build version string used by WandB and checkpoint directory ----
    args.version = build_version_name(args)

    logging.info(args)

    # Run fine-tuning
    supervised(args, config_bundle)
