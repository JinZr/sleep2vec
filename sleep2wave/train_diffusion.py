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

from sleep2wave.common import persist_run_config_and_args
from sleep2wave.data.generative_dataset import Sleep2WaveGenerativeDataset
from sleep2wave.data.samplers import AvailableChannelsBucketBatchSampler, handles_distributed_sharding
from sleep2wave.diffusion.latent_cache import Sleep2WaveLatentCacheDataset
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
    parser = argparse.ArgumentParser(description="Train the sleep2wave latent diffusion model.")
    parser.add_argument("--config", type=Path, required=True, help="Sleep2Wave diffusion YAML config.")
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
    parser.add_argument("--seed", type=int, default=0, help="Random seed for data sampling and Lightning.")
    parser.add_argument(
        "--resume-from-checkpoint",
        type=Path,
        default=None,
        help="Optional Lightning checkpoint to resume diffusion training.",
    )
    return parser.parse_args(argv)


def build_dataloader(config, *, num_workers: int, seed: int, split: str = "train"):
    if config.training is None:
        raise ValueError("training block is required for diffusion training.")
    if config.diffusion is None:
        raise ValueError("diffusion block is required for diffusion training.")
    if config.data.context_epochs != config.diffusion.context_epochs:
        raise ValueError("data.context_epochs must match diffusion.context_epochs for diffusion training.")
    is_train = split == "train"
    schedule = build_phase_schedule(
        config.training.phase,
        config.training.task_mix,
        replay_enabled=config.training.replay.enabled,
    )
    min_channels = 1
    if schedule.task_mix.get("translation", 0.0) > 0 or schedule.task_mix.get("partial_full", 0.0) > 0:
        min_channels = 2
    if schedule.task_mix.get("two_condition", 0.0) > 0:
        min_channels = 3
    if config.diffusion.autoencoder_checkpoint is None:
        try:
            dataset = Sleep2WaveLatentCacheDataset(config.diffusion.latent_cache_path, split=split)
        except ValueError as exc:
            if split == "train" or "No sleep2wave latent cache rows are available" not in str(exc):
                raise
            dataset = Sleep2WaveLatentCacheDataset(config.diffusion.latent_cache_path, split="train")
        batch_sampler = AvailableChannelsBucketBatchSampler(
            dataset.data,
            batch_size=config.training.batch_size,
            min_channels=min_channels,
            shuffle=is_train,
            drop_last=False,
            shard_across_ranks=True,
            seed=seed,
        )
        return dataset.dataloader(
            batch_sampler=batch_sampler,
            num_workers=num_workers,
        )
    else:
        dataset = Sleep2WaveGenerativeDataset(
            backend=config.data.backend,
            preset_path=config.data.preset_path,
            index=config.data.index,
            kaldi_data_root=config.data.kaldi_data_root,
            kaldi_manifest=config.data.kaldi_manifest,
            split=split,
            context_epochs=config.data.context_epochs,
            task_type="translation",
            seed=seed,
        )
    batch_sampler = AvailableChannelsBucketBatchSampler(
        dataset.data,
        batch_size=config.training.batch_size,
        min_channels=min_channels,
        shuffle=is_train,
        drop_last=False,
        shard_across_ranks=True,
        seed=seed,
    )
    return dataset.dataloader(
        batch_sampler=batch_sampler,
        num_workers=num_workers,
    )


def _load_phase_checkpoint(model, checkpoint_path: str | Path) -> int:
    import torch

    path = Path(checkpoint_path)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"training.phase_checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state_dict, dict):
        raise ValueError("training.phase_checkpoint must contain a state_dict mapping.")
    target_keys = set(model.state_dict())
    if any(key.startswith("model.") for key in state_dict):
        filtered = {
            key[len("model.") :]: value
            for key, value in state_dict.items()
            if key.startswith("model.") and key[len("model.") :] in target_keys
        }
    else:
        filtered = {key: value for key, value in state_dict.items() if key in target_keys}
    if not filtered:
        raise ValueError("training.phase_checkpoint does not contain sleep2wave diffusion model weights.")
    model.load_state_dict(filtered, strict=True)
    return len(filtered)


def _limit_val_batches(config) -> int:
    validation_cfg = config.training.validation
    schedule = build_phase_schedule(
        config.training.phase,
        config.training.task_mix,
        replay_enabled=config.training.replay.enabled,
    )
    active_task_families = sum(1 for weight in schedule.task_mix.values() if weight > 0)
    return validation_cfg.max_batches_per_modality * len(validation_cfg.examples.modalities) * active_task_families


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

    train_loader = build_dataloader(config, num_workers=args.num_workers, seed=args.seed, split="train")
    val_loader = build_dataloader(config, num_workers=args.num_workers, seed=args.seed, split="val")
    use_distributed_sampler = not handles_distributed_sharding(getattr(train_loader, "batch_sampler", None))
    model = Sleep2WaveDiffusionLightning(config, seed=args.seed)
    if config.training.phase_checkpoint is not None:
        loaded = _load_phase_checkpoint(model.model, config.training.phase_checkpoint)
        logging.info(
            "Loaded %d diffusion model keys from phase checkpoint %s.",
            loaded,
            config.training.phase_checkpoint,
        )
    if config.initialization is not None and config.initialization.sleep2vec2_checkpoint is not None:
        report = load_sleep2vec2_initialization(
            model.model,
            config.initialization.sleep2vec2_checkpoint,
            config.initialization,
            target_groups={"diffusion_transformer"},
            device="cpu",
        )
        logging.info(
            "sleep2wave diffusion initialization loaded %d keys from %s.",
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
    lr_monitor = LearningRateMonitor(logging_interval="step")
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
        use_distributed_sampler=use_distributed_sampler,
        val_check_interval=config.training.validation.interval_steps,
        check_val_every_n_epoch=None,
        limit_val_batches=_limit_val_batches(config),
    )
    resume_from_checkpoint = getattr(args, "resume_from_checkpoint", None)
    if resume_from_checkpoint is not None and not resume_from_checkpoint.is_file():
        raise FileNotFoundError(f"--resume-from-checkpoint not found: {resume_from_checkpoint}")
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=resume_from_checkpoint,
    )
    trainer.save_checkpoint(checkpoint_dir / "last.ckpt")
    return run_dir


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    train_diffusion(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
