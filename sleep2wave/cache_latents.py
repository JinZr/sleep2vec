from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a sleep2wave latent cache from waveform windows.")
    parser.add_argument("--config", type=Path, required=True, help="Sleep2Wave diffusion YAML config.")
    parser.add_argument("--autoencoder-ckpt", type=Path, default=None, help="Optional autoencoder checkpoint override.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output latent-cache directory.")
    parser.add_argument("--batch-size", type=int, default=1, help="Latent-cache dataloader batch size.")
    parser.add_argument("--num-workers", type=int, default=0, help="Latent-cache dataloader workers.")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device used for autoencoder encoding.")
    return parser.parse_args(argv)


def _metadata_rows(batch: dict) -> list[dict]:
    rows: list[dict] = []
    batch_size = batch["epoch_index"].shape[0]
    for idx in range(batch_size):
        row = {key: values[idx] for key, values in batch["metadata"].items()}
        row["start_epoch"] = int(batch["epoch_index"][idx, 0].item())
        row["end_epoch"] = int(batch["epoch_index"][idx, -1].item()) + 1
        rows.append(row)
    return rows


def build_latent_cache(args: argparse.Namespace) -> Path:
    from sleep2wave.autoencoders.checkpoints import load_sleep2wave_autoencoder_checkpoint
    from sleep2wave.data.generative_dataset import Sleep2WaveGenerativeDataset
    from sleep2wave.diffusion.latent_cache import write_latent_cache
    from sleep2wave.generative.config import load_sleep2wave_config

    config = load_sleep2wave_config(args.config)
    if config.diffusion is None or config.data is None:
        raise ValueError("Latent cache building requires a config with data and diffusion blocks.")
    checkpoint = args.autoencoder_ckpt or config.diffusion.autoencoder_checkpoint
    if checkpoint is None:
        raise ValueError("An autoencoder checkpoint is required to build a latent cache.")
    device = torch.device(args.device)
    autoencoder = load_sleep2wave_autoencoder_checkpoint(
        checkpoint,
        latent_dim=config.diffusion.latent_dim,
        modalities=config.modalities.all,
        device=device,
    )
    dataset = Sleep2WaveGenerativeDataset(
        preset_path=config.data.preset_path,
        index=config.data.index,
        split="train",
        context_epochs=config.data.context_epochs,
    )
    loader = dataset.dataloader(batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    latents = {modality: [] for modality in config.modalities.all}
    availability = {modality: [] for modality in config.modalities.all}
    quality = {modality: [] for modality in config.modalities.all}
    epoch_index = []
    night_position = []
    metadata_rows = []
    with torch.no_grad():
        for batch in loader:
            signals = {key: value.to(device) for key, value in batch["clean_signals"].items()}
            encoded = autoencoder(signals).latents
            for modality in config.modalities.all:
                latents[modality].append(encoded[modality].cpu())
                availability[modality].append(batch["availability_mask"][modality].cpu())
                quality[modality].append(batch["quality_mask"][modality].cpu())
            epoch_index.append(batch["epoch_index"].cpu())
            night_position.append(batch["night_position"].cpu())
            metadata_rows.extend(_metadata_rows(batch))
    return write_latent_cache(
        args.output_dir,
        clean_latents={modality: torch.cat(values, dim=0) for modality, values in latents.items()},
        availability_mask={modality: torch.cat(values, dim=0) for modality, values in availability.items()},
        quality_mask={modality: torch.cat(values, dim=0) for modality, values in quality.items()},
        epoch_index=torch.cat(epoch_index, dim=0),
        night_position=torch.cat(night_position, dim=0),
        metadata_rows=metadata_rows,
        modalities=config.modalities.all,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    build_latent_cache(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_latent_cache", "main", "parse_args"]
