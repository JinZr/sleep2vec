import argparse
import logging
import os
from pathlib import Path
import sys

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

from sleep2expert.callbacks.pair_acc_logger import PairAccLoggerCallback
from sleep2expert.checkpoints import load_pretrain_init_weights
from sleep2expert.common import apply_data_backend_args, apply_model_config_args, persist_run_config_and_args
from sleep2expert.config import load_pretrain_config
from sleep2expert.data.samplers import handles_distributed_sharding
from sleep2expert.model_stats import (
    count_total_parameters,
    count_trainable_parameters,
    estimate_active_parameters_per_token,
    estimate_dense_equivalent_ffn_flops,
    estimate_moe_ffn_active_flops,
)
from sleep2expert.sleep2vec_modelling import Sleep2vecPretraining
from sleep2expert.utils import get_pretrain_dataloader


def _optional_path(value):
    if value.lower() in {"null", "none"}:
        return None
    return Path(value)


def sleep2vec_pretrain(args):

    config_bundle = load_pretrain_config(args.config)
    model_config = config_bundle.model
    loss_config = config_bundle.loss
    averaging_config = config_bundle.averaging
    args.mask_rate = config_bundle.data.mask_rate
    args.max_tokens = config_bundle.data.max_tokens
    apply_model_config_args(args, model_config, set_backbone_arch=True)
    apply_data_backend_args(args, config_bundle.data, preset_attr="pretrain_preset_path")

    # get data loaders
    train_loader, val_loader = get_pretrain_dataloader(args)
    # Disable Lightning's distributed sampler injection only when our custom
    # batch sampler already shards across ranks.
    train_batch_sampler = getattr(train_loader, "batch_sampler", None)
    val_batch_sampler = getattr(val_loader, "batch_sampler", None)
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
    persist_run_config_and_args(args, exp_dir)

    model = Sleep2vecPretraining(args, model_config, loss_config, averaging_config=averaging_config)
    moe_cfg = model_config.backbone.moe
    model_stats = {
        "total_params": count_total_parameters(model.model),
        "trainable_params": count_trainable_parameters(model.model),
        "estimated_active_params_per_token": estimate_active_parameters_per_token(model_config),
        "estimated_moe_ffn_active_flops": estimate_moe_ffn_active_flops(model_config, args.max_tokens),
        "estimated_dense_equivalent_ffn_flops": estimate_dense_equivalent_ffn_flops(model_config, args.max_tokens),
        "moe_num_experts": moe_cfg.num_experts if moe_cfg and moe_cfg.enabled else None,
        "moe_top_k": moe_cfg.top_k if moe_cfg and moe_cfg.enabled else None,
        "moe_layers": moe_cfg.layer_indices if moe_cfg and moe_cfg.enabled else None,
        "expert_hidden_size": moe_cfg.expert_hidden_size if moe_cfg and moe_cfg.enabled else None,
    }
    for stat_name, stat_value in model_stats.items():
        logging.info("%s: %s", stat_name, stat_value)
    if args.pretrained_backbone_path and args.ckpt_path is None:
        load_info = load_pretrain_init_weights(model.model, args.pretrained_backbone_path, device="cpu", strict=False)
        logging.info(
            "Loaded pretrain-model init from %s using prefix=%s (%d keys).",
            args.pretrained_backbone_path,
            load_info.used_prefix,
            load_info.loaded_keys,
        )
        if load_info.missing_keys:
            logging.warning("Missing init keys: %s", load_info.missing_keys)
        if load_info.unexpected_keys:
            logging.warning("Unexpected init keys: %s", load_info.unexpected_keys)
        if model.model_averager is not None:
            model.model_averager.sync_from_student()

    logger = WandbLogger(
        project="sleep2expert-pretrain",
        name=f"sleep2expert-pretrain-{run_name}",
        save_dir=os.path.dirname(save_path),
        id=wandb_id,  # NEW：保持同一个 run
        resume="allow" if wandb_id else None,  # NEW：若 id 存在则追加
    )
    logger.log_hyperparams(model_stats)

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

    pair_acc_cb = PairAccLoggerCallback(
        args.channel_names,
        train_pair_monitor_enabled=args.train_pair_monitor_enable,
        train_pair_log_prefix=args.train_pair_monitor_log_prefix,
        train_pair_skew_warn_threshold=args.train_pair_skew_warn_threshold,
        train_pair_min_unique_coverage_warn_threshold=args.train_pair_min_unique_coverage_warn_threshold,
    )
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
        log_every_n_steps=5,
        num_sanity_val_steps=0,
        precision=args.precision,
        gradient_clip_val=args.gradient_clip_val,
        accumulate_grad_batches=args.accumulate_grad_batches,
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
        val_dataloaders=val_loader if not args.print_diagnostics else None,
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
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=None,
        help="Override warmup steps for LR schedule (default: 3%% of total steps).",
    )
    parser.add_argument("--weight-decay", type=float, default=1e-2, help="weight decay for AdamW")
    parser.add_argument("--batch-size", type=int, default=320, help="batch size")
    parser.add_argument("--num-workers", type=int, default=16, help="Training dataloader workers.")
    parser.add_argument(
        "--val-num-workers",
        type=int,
        default=4,
        help="Validation dataloader workers.",
    )
    parser.add_argument("--devices", type=int, nargs="+", default=[0, 1], help="GPU device ids")
    parser.add_argument(
        "--ckpt-path",
        type=Path,
        default=None,
        help="要继续训练的 checkpoint 路径 (.ckpt)",
    )
    parser.add_argument(
        "--pretrained-backbone-path",
        type=Path,
        default=None,
        help="Optional pretrain-model init checkpoint. Loads ema_model. first and falls back to model.",
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
        type=_optional_path,
        default=None,
        help="path to precomputed preset pickle for PSG dataset; use null/none to disable",
    )
    parser.add_argument(
        "--data-backend",
        choices=["npz", "kaldi"],
        default=None,
        help="Data backend for pretraining (default: npz unless YAML data.backend overrides it).",
    )
    parser.add_argument(
        "--kaldi-data-root",
        type=Path,
        default=None,
        help="Kaldi data root when --data-backend kaldi is used.",
    )
    parser.add_argument(
        "--kaldi-manifest",
        type=Path,
        default=None,
        help="Kaldi manifest.json path when --data-backend kaldi is used.",
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
        "--train-pair-track-unique-samples",
        action="store_true",
        help=(
            "Track per-pair unique sampled indices during training monitoring. "
            "Disabled by default to reduce host memory usage."
        ),
    )
    parser.add_argument(
        "--train-pair-skew-warn-threshold",
        type=float,
        default=0.05,
        help="Warn when |actual_pair_ratio - target_pair_ratio| exceeds this threshold in an epoch.",
    )
    parser.add_argument(
        "--train-pair-monitor-enable",
        dest="train_pair_monitor_enable",
        action="store_true",
        default=False,
        help="Enable epoch-level train pair sampling distribution monitoring.",
    )
    parser.add_argument(
        "--no-train-pair-monitor-enable",
        dest="train_pair_monitor_enable",
        action="store_false",
        help="Disable train pair sampling distribution monitoring.",
    )
    parser.add_argument(
        "--train-pair-monitor-log-prefix",
        type=str,
        default="train_pair_sampling",
        help="Metric prefix for train pair sampling logs.",
    )
    parser.add_argument(
        "--train-pair-min-unique-coverage-warn-threshold",
        type=float,
        default=0.1,
        help="Warn when unique sampled indices / pair pool size falls below this threshold in an epoch.",
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
        "--accumulate-grad-batches",
        type=int,
        default=1,
        help="Number of batches to accumulate before each optimizer step.",
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
    sleep2vec_pretrain(args)
