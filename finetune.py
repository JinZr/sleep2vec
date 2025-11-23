import argparse
import logging
from pathlib import Path
import shutil

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies.ddp import DDPStrategy
import wandb

from metrics import save_result_csv
from sleep2vec.config import load_finetune_config
from sleep2vec.sleep2vec_finetuning import Sleep2vecFinetuning
from utils import get_finetune_dataloaders

# from model.ahi_metric import AHIMetricsCollection


def prepare_dataloader(args):
    train_loader, val_loader, test_loader = get_finetune_dataloaders(args)

    logging.info(len(train_loader), len(val_loader), len(test_loader))
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

    # get data loaders
    train_loader, val_loader, test_loader = prepare_dataloader(args)

    # define the model/lightning module
    model = Sleep2vecFinetuning(args, model_config)

    # logger and callbacks
    version = args.version
    logger = WandbLogger(
        project="sleep2vec-finetune",  # 相当于 TensorBoard 的 log dir
        name=f"s2v-finetune-{version}",  # run 名称
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
        save_top_k=1,  # 只保留最优一个
        filename="{epoch:02d}",
    )

    trainer = pl.Trainer(
        devices=args.devices,
        accelerator="gpu",
        strategy=DDPStrategy(find_unused_parameters=True),
        # strategy=DeepSpeedStrategy(config="ds_config.json"),  # ← 就这行！
        benchmark=True,
        enable_checkpointing=True,
        logger=logger,
        max_epochs=args.epochs,
        callbacks=[early_stop_callback, checkpoint_callback],
        gradient_clip_val=1.0,
        precision="bf16-mixed",  # <---- 开启 BF16
        check_val_every_n_epoch=args.check_val_every_n_epoch,
    )

    if args.epochs > 0:
        # train the model
        trainer.fit(
            model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
            ckpt_path=args.ckpt_path if args.ckpt_path != "" else None,
        )

    # test the model
    pretrain_result = trainer.test(
        model=model,
        ckpt_path="best" if args.epochs > 0 else args.ckpt_path,
        dataloaders=test_loader,
    )[0]
    logging.info(pretrain_result)
    save_result_csv(pretrain_result, args.results_csv_path, args)


if __name__ == "__main__":
    # Login to WandB only when running as a script
    wandb.login()

    parser = argparse.ArgumentParser(description="Fine-tune Sleep2Vec downstream models on PSG data.")

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

    # ---------------- Task & data configuration ----------------
    parser.add_argument(
        "--label-name",
        type=str,
        default="age",
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

    args = parser.parse_args()

    config_bundle = load_finetune_config(args.config)
    args.channel_names = [c.name for c in config_bundle.model.channels]
    data_cfg = config_bundle.data
    lora_cfg = config_bundle.lora

    # Data-related overrides from YAML
    args.max_tokens = data_cfg.max_tokens
    args.data_channel_names = data_cfg.data_channel_names or args.channel_names
    args.finetune_data_index = Path(data_cfg.finetune_data_index) if data_cfg.finetune_data_index else None
    args.finetune_preset_path = Path(data_cfg.finetune_preset_path) if data_cfg.finetune_preset_path else None
    args.train_dataset_names = data_cfg.train_dataset_names or []
    args.test_dataset_names = data_cfg.test_dataset_names or []
    args.n_few_shot = data_cfg.n_few_shot

    # LoRA-related toggles from YAML
    args.freeze_backbone_and_insert_lora = lora_cfg.freeze_backbone_and_insert_lora
    args.insert_lora = lora_cfg.insert_lora
    args.separate_adapters = lora_cfg.separate_adapters
    args.head_kwargs = {}

    # ---- Infer task spec from label_name (same spirit as TaskSpec in batch_run_few_shot.py) ----
    if args.label_name == "stage5":
        args.output_dim = 5
        args.is_classification = True
        args.is_seq = True
        args.monitor = "val_accuracy"
        args.monitor_mod = "max"
    elif args.label_name == "sex":
        args.output_dim = 2
        args.is_classification = True
        args.is_seq = False
        args.monitor = "val_accuracy"
        args.monitor_mod = "max"
    else:  # default: regression-style task (e.g. age)
        args.output_dim = 1
        args.is_classification = False
        args.is_seq = False
        args.monitor = "val_mae"
        args.monitor_mod = "min"

    # ---- Build version string used by WandB and checkpoint directory ----
    if args.version_name:
        args.version = args.version_name
    else:
        ch_stub = args.channel_names[0] if args.channel_names else "mixed"
        few_stub = f"fewshot-{args.n_few_shot}"
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
        args.version = "-".join(pieces)

    logging.info(args)

    # Run fine-tuning
    supervised(args, config_bundle)
