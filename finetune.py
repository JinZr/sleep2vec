import argparse
import json
import logging

import pytorch_lightning as pl
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from metrics import save_result_csv
from sleep2vec.downstream.head_registry import available_heads
from sleep2vec.sleep2vec_finetuning import Sleep2vecFinetuning
from utils import get_finetune_dataloaders

# from model.ahi_metric import AHIMetricsCollection


def prepare_dataloader(args):
    train_loader, val_loader, test_loader = get_finetune_dataloaders(args)

    print(len(train_loader), len(val_loader), len(test_loader))
    return train_loader, val_loader, test_loader


def supervised(args):
    # get data loaders

    #################################
    train_loader, val_loader, test_loader = prepare_dataloader(args)

    # define the model/lightning module
    model = Sleep2vecFinetuning(args)

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
        # strategy=DDPStrategy(find_unused_parameters=True),
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
    print(pretrain_result)
    save_result_csv(pretrain_result, "/data/ywx/BIOT/results.csv", args)


if __name__ == "__main__":
    # Login to WandB only when running as a script
    wandb.login()

    parser = argparse.ArgumentParser(
        description="Fine-tune Sleep2Vec downstream models on PSG data."
    )

    # ---------------- Optimization & training hyper-parameters ----------------
    parser.add_argument(
        "--epochs", type=int, default=200, help="number of fine-tuning epochs"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-6, help="learning rate for AdamW"
    )
    parser.add_argument(
        "--weight-decay",
        dest="weight_decay",
        type=float,
        default=1e-5,
        help="weight decay for AdamW",
    )
    parser.add_argument(
        "--batch-size", type=int, default=12, help="batch size for dataloader"
    )
    parser.add_argument(
        "--num-workers", type=int, default=32, help="number of dataloader workers"
    )
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
    parser.add_argument(
        "--channel-names",
        type=str,
        default="eeg_original",
        help="comma-separated backbone channel names",
    )
    parser.add_argument(
        "--data-channel-names",
        type=str,
        default="",
        help="comma-separated dataset channel names; if empty, defaults to --channel-names",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=120,
        help="maximum number of tokens per PSG window",
    )
    parser.add_argument(
        "--finetune-data-index",
        type=str,
        default="index/hsp_psg_pretrain.csv",
        help="CSV index file listing PSG samples for finetuning",
    )
    parser.add_argument(
        "--finetune-preset-path",
        type=str,
        default="/data/ywx/BIOT/data/all_disease_preset_1535_1211.pickle",
        help="path to preset pickle used to accelerate finetuning dataset loading",
    )
    parser.add_argument(
        "--train-dataset-names",
        type=str,
        default="shhs",
        help="comma-separated dataset identifiers used for train/val splits",
    )
    parser.add_argument(
        "--test-dataset-names",
        type=str,
        default="shhs",
        help="comma-separated dataset identifiers used for test split",
    )
    parser.add_argument(
        "--channel-feature-dim",
        type=int,
        default=768,
        help="per-channel feature dimension inside the backbone",
    )
    parser.add_argument(
        "--transformer-hidden-size",
        type=int,
        default=768,
        help="hidden size of Transformer encoder blocks",
    )
    parser.add_argument(
        "--transformer-num-hidden-layers",
        type=int,
        default=12,
        help="number of Transformer encoder layers",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=16,
        help="number of attention heads in each Transformer layer",
    )
    parser.add_argument(
        "--backbone-arch",
        type=str,
        default="roformer",
        choices=["roformer", "hf_bert"],
        help="backbone encoder architecture when building Sleep2vecPretrainModel",
    )
    parser.add_argument(
        "--projection",
        dest="projection",
        action="store_true",
        help="enable projection head on top of the backbone",
    )
    parser.add_argument(
        "--no-projection",
        dest="projection",
        action="store_false",
        help="disable projection head on top of the backbone",
    )
    parser.set_defaults(projection=False)
    parser.add_argument(
        "--n-few-shot",
        type=int,
        default=1280,
        help="number of labeled samples for few-shot setting",
    )
    parser.add_argument(
        "--head-name",
        type=str,
        default="",
        choices=[""] + available_heads(),
        help="registered downstream head name (default auto-selects classification/regression)",
    )
    parser.add_argument(
        "--head-config",
        type=str,
        default="",
        help="JSON string of kwargs forwarded to the downstream head factory",
    )

    # ---------------- Pretrained backbone / LoRA configuration ----------------
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

    parser.add_argument(
        "--freeze-backbone-and-insert-lora",
        dest="freeze_backbone_and_insert_lora",
        action="store_true",
        help="freeze backbone weights and insert LoRA adapters",
    )
    parser.add_argument(
        "--no-freeze-backbone-and-insert-lora",
        dest="freeze_backbone_and_insert_lora",
        action="store_false",
        help="keep backbone trainable and do not insert LoRA adapters",
    )
    parser.set_defaults(freeze_backbone_and_insert_lora=False)

    parser.add_argument(
        "--insert-lora",
        dest="insert_lora",
        action="store_true",
        help="enable insertion of LoRA layers (effective when freezing backbone)",
    )
    parser.add_argument(
        "--no-insert-lora",
        dest="insert_lora",
        action="store_false",
        help="disable insertion of LoRA layers",
    )
    parser.set_defaults(insert_lora=True)

    parser.add_argument(
        "--separate-adapters",
        dest="separate_adapters",
        action="store_true",
        help="use separate LoRA adapters for each task/head",
    )
    parser.add_argument(
        "--no-separate-adapters",
        dest="separate_adapters",
        action="store_false",
        help="share LoRA adapters across tasks",
    )
    parser.set_defaults(separate_adapters=False)

    # ---------------- Logging / versioning ----------------
    parser.add_argument(
        "--version-name",
        type=str,
        default=None,
        help=(
            "explicit run name for logging and checkpoint directory; "
            "if not set, a name will be generated"
        ),
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
        "--check-val-every-n-epoch",
        dest="check_val_every_n_epoch",
        type=int,
        default=1,
        help="run validation every N epochs",
    )

    args = parser.parse_args()

    # ---- Post-process list-like arguments ----
    args.channel_names = [c.strip() for c in args.channel_names.split(",") if c.strip()]
    if args.data_channel_names:
        args.data_channel_names = [
            c.strip() for c in args.data_channel_names.split(",") if c.strip()
        ]
    else:
        args.data_channel_names = args.channel_names
    args.train_dataset_names = [
        name.strip() for name in args.train_dataset_names.split(",") if name.strip()
    ]
    args.test_dataset_names = [
        name.strip() for name in args.test_dataset_names.split(",") if name.strip()
    ]
    if not args.train_dataset_names:
        args.train_dataset_names = ["shhs"]
    if not args.test_dataset_names:
        args.test_dataset_names = ["shhs"]
    if args.head_name == "":
        args.head_name = None
    if args.head_config:
        try:
            args.head_kwargs = json.loads(args.head_config)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Invalid JSON for --head-config: {exc}") from exc
    else:
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
        pretrain_suffix = (
            "with_pretrain" if args.pretrained_backbone_path else "from_scratch"
        )
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
    supervised(args)
