from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sleep2wave.autoencoders.lightning import Sleep2WaveAutoencoderLightning
from sleep2wave.common import persist_run_config_and_args
from sleep2wave.data.generative_dataset import Sleep2WaveGenerativeDataset
from sleep2wave.generative.config import load_sleep2wave_config
from sleep2wave.initialization.sleep2vec2 import load_sleep2vec2_initialization


def _parse_devices(raw: str):
    if raw == "auto":
        return "auto"
    if "," in raw:
        return [int(part) for part in raw.split(",") if part]
    return int(raw)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train sleep2wave modality-specific waveform autoencoders.")
    parser.add_argument("--config", type=Path, required=True, help="Sleep2Wave autoencoder YAML config.")
    parser.add_argument("--version-name", type=str, required=True, help="Version name for logging and outputs.")
    parser.add_argument("--accelerator", type=str, default="auto", help="Lightning accelerator setting.")
    parser.add_argument("--devices", type=str, default="auto", help="Lightning device ids, comma list, or 'auto'.")
    parser.add_argument(
        "--precision",
        type=int,
        default=32,
        help="Mixed precision setting passed to Lightning Trainer.",
    )
    parser.add_argument("--max-steps", type=int, default=None, help="Optional max training steps override.")
    parser.add_argument("--num-workers", type=int, default=0, help="Training dataloader workers.")
    return parser.parse_args(argv)


def build_dataloader(config, *, num_workers: int, split: str = "train"):
    if config.training is None:
        raise ValueError("training block is required for autoencoder training.")
    dataset = Sleep2WaveGenerativeDataset(
        backend=config.data.backend,
        preset_path=config.data.preset_path,
        index=config.data.index,
        kaldi_data_root=config.data.kaldi_data_root,
        kaldi_manifest=config.data.kaldi_manifest,
        split=split,
        context_epochs=config.data.context_epochs,
        task_type="restoration",
    )
    return dataset.dataloader(
        batch_size=config.training.batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
    )


def train_autoencoder(args: argparse.Namespace) -> Path:
    config = load_sleep2wave_config(args.config)
    if config.stage != "autoencoder":
        raise ValueError("train_autoencoder requires stage=autoencoder config.")
    if config.training is None or config.export is None:
        raise ValueError("training and export blocks are required.")

    run_dir = Path(config.export.output_dir) / args.version_name
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    persist_run_config_and_args(args, run_dir)

    train_loader = build_dataloader(config, num_workers=args.num_workers, split="train")
    val_loader = build_dataloader(config, num_workers=args.num_workers, split="val")
    model = Sleep2WaveAutoencoderLightning(config)
    if config.initialization is not None and config.initialization.sleep2vec2_checkpoint is not None:
        report = load_sleep2vec2_initialization(
            model.model,
            config.initialization.sleep2vec2_checkpoint,
            config.initialization,
            target_groups={"autoencoder_encoders"},
            device="cpu",
        )
        logging.info(
            "sleep2wave autoencoder initialization loaded %d keys from %s.",
            len(report.loaded_keys),
            config.initialization.sleep2vec2_checkpoint,
        )
    logger = WandbLogger(
        project="sleep2wave-autoencoder",
        name=f"sleep2wave-autoencoder-{args.version_name}",
        save_dir=str(run_dir),
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="epoch={epoch}-step={step}",
        save_top_k=-1,
        every_n_epochs=1,
        save_on_train_epoch_end=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    validation_cfg = config.training.validation
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=_parse_devices(args.devices),
        precision=args.precision,
        max_epochs=config.training.max_epochs,
        max_steps=args.max_steps if args.max_steps is not None else -1,
        gradient_clip_val=config.training.gradient_clip_val,
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor],
        log_every_n_steps=1,
        num_sanity_val_steps=0,
        val_check_interval=validation_cfg.interval_steps,
        check_val_every_n_epoch=None,
        limit_val_batches=validation_cfg.max_batches_per_modality,
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.save_checkpoint(checkpoint_dir / "last.ckpt")
    return run_dir


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    train_autoencoder(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
