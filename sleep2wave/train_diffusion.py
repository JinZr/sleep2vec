from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sleep2wave.common import persist_run_config_and_args
from sleep2wave.data.generative_dataset import Sleep2WaveGenerativeDataset
from sleep2wave.data.samplers import AvailableChannelsBucketBatchSampler, handles_distributed_sharding
from sleep2wave.diffusion.lightning import Sleep2WaveDiffusionLightning
from sleep2wave.generative.config import load_sleep2wave_config
from sleep2wave.initialization.sleep2vec2 import load_sleep2vec2_initialization
from sleep2wave.training.logging import SLEEP2WAVE_DIFFUSION_PROJECT, build_diffusion_run_name
from sleep2wave.training.phase_schedule import build_phase_schedule


def _parse_devices(raw: str):
    if raw == "auto":
        return "auto"
    if "," in raw:
        return [int(part) for part in raw.split(",") if part]
    return int(raw)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Sleep2Wave latent diffusion model.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--version-name", type=str, required=True)
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--devices", type=str, default="auto")
    parser.add_argument("--precision", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def build_dataloader(config, *, num_workers: int, seed: int):
    if config.training is None:
        raise ValueError("training block is required for diffusion training.")
    if config.diffusion is None:
        raise ValueError("diffusion block is required for diffusion training.")
    if config.data.context_epochs != config.diffusion.context_epochs:
        raise ValueError("data.context_epochs must match diffusion.context_epochs for diffusion training.")
    dataset = Sleep2WaveGenerativeDataset(
        preset_path=config.data.preset_path,
        index=config.data.index,
        split="train",
        context_epochs=config.data.context_epochs,
        task_type="translation",
        corruption_name="gaussian_noise",
        corruption_kwargs={"std": 0.01},
        seed=seed,
    )
    schedule = build_phase_schedule(config.training.phase, config.training.task_mix)
    min_channels = 1
    if schedule.task_mix.get("translation", 0.0) > 0 or schedule.task_mix.get("partial_full", 0.0) > 0:
        min_channels = 2
    if schedule.task_mix.get("two_condition", 0.0) > 0:
        min_channels = 3
    batch_sampler = AvailableChannelsBucketBatchSampler(
        dataset.data,
        batch_size=config.training.batch_size,
        min_channels=min_channels,
        shuffle=True,
        drop_last=False,
        shard_across_ranks=True,
        seed=seed,
    )
    return dataset.dataloader(
        batch_sampler=batch_sampler,
        num_workers=num_workers,
    )


def train_diffusion(args: argparse.Namespace) -> Path:
    config = load_sleep2wave_config(args.config)
    if config.stage != "diffusion":
        raise ValueError("train_diffusion requires stage=diffusion config.")
    if config.training is None or config.diffusion is None or config.export is None:
        raise ValueError("training, diffusion, and export blocks are required.")
    pl.seed_everything(args.seed, workers=True)

    run_dir = Path(config.export.output_dir) / args.version_name
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    persist_run_config_and_args(args, run_dir)

    train_loader = build_dataloader(config, num_workers=args.num_workers, seed=args.seed)
    use_distributed_sampler = not handles_distributed_sharding(getattr(train_loader, "batch_sampler", None))
    model = Sleep2WaveDiffusionLightning(config, seed=args.seed)
    if config.initialization is not None and config.initialization.sleep2vec2_checkpoint is not None:
        report = load_sleep2vec2_initialization(
            model.model,
            config.initialization.sleep2vec2_checkpoint,
            config.initialization,
            target_groups={"diffusion_transformer"},
            device="cpu",
        )
        logging.info(
            "Sleep2Wave diffusion initialization loaded %d keys from %s.",
            len(report.loaded_keys),
            config.initialization.sleep2vec2_checkpoint,
        )
    logger = WandbLogger(
        project=SLEEP2WAVE_DIFFUSION_PROJECT,
        name=build_diffusion_run_name(
            args.version_name,
            phase=config.training.phase,
            context_epochs=config.diffusion.context_epochs,
        ),
        save_dir=str(run_dir),
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="epoch={epoch}-step={step}",
        save_top_k=-1,
        every_n_epochs=1,
        save_on_train_epoch_end=True,
    )
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=_parse_devices(args.devices),
        precision=args.precision,
        max_epochs=config.training.max_epochs,
        max_steps=args.max_steps if args.max_steps is not None else -1,
        gradient_clip_val=config.training.gradient_clip_val,
        logger=logger,
        callbacks=[checkpoint_callback],
        log_every_n_steps=1,
        num_sanity_val_steps=0,
        use_distributed_sampler=use_distributed_sampler,
    )
    trainer.fit(model, train_dataloaders=train_loader)
    trainer.save_checkpoint(checkpoint_dir / "last.ckpt")
    return run_dir


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    train_diffusion(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
