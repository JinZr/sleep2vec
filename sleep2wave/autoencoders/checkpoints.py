from __future__ import annotations

from pathlib import Path
import typing as t

import torch

from sleep2wave.autoencoders.model import Sleep2WaveAutoencoder
from sleep2wave.data.modalities import CANONICAL_MODALITIES


def load_sleep2wave_autoencoder_checkpoint(
    checkpoint_path: str | Path,
    *,
    latent_dim: int,
    latent_frames_per_epoch: t.Mapping[str, int],
    modalities: t.Sequence[str] = CANONICAL_MODALITIES,
    device: torch.device | str = "cpu",
) -> Sleep2WaveAutoencoder:
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Autoencoder checkpoint not found: {path}")

    model = Sleep2WaveAutoencoder(
        latent_dim=latent_dim,
        latent_frames_per_epoch=latent_frames_per_epoch,
        modalities=modalities,
    )
    checkpoint = torch.load(path, map_location=torch.device(device), weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state_dict, dict):
        raise ValueError("Autoencoder checkpoint must contain a state_dict mapping.")

    autoencoder_state = {
        key[len("model.") :] if key.startswith("model.") else key: value
        for key, value in state_dict.items()
    }
    model.load_state_dict(autoencoder_state, strict=True)
    model.to(device)
    model.eval()
    model.requires_grad_(False)
    return model


__all__ = ["load_sleep2wave_autoencoder_checkpoint"]
