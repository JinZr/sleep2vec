from __future__ import annotations

from dataclasses import dataclass
import typing as t

import torch
import torch.nn as nn
import torch.nn.functional as F

from sleep2wave.data.modalities import CANONICAL_MODALITIES, MODALITY_SPECS, validate_modality_sequence


@dataclass
class Sleep2WaveAutoencoderOutput:
    latents: dict[str, torch.Tensor]
    reconstructions: dict[str, torch.Tensor]


class _EpochConvEncoder(nn.Module):
    def __init__(self, *, frames_per_epoch: int, latent_dim: int) -> None:
        super().__init__()
        width = 32 if frames_per_epoch >= 1000 else 16
        self.net = nn.Sequential(
            nn.Conv1d(1, width, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv1d(width, width, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(width, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _EpochDecoder(nn.Module):
    def __init__(self, *, frames_per_epoch: int, latent_dim: int) -> None:
        super().__init__()
        self.frames_per_epoch = int(frames_per_epoch)
        width = 32 if frames_per_epoch >= 1000 else 16
        init_length = 16 if frames_per_epoch >= 1000 else 8
        layers: list[nn.Module] = []
        length = init_length
        while length < frames_per_epoch:
            layers.extend(
                [
                    nn.ConvTranspose1d(width, width, kernel_size=4, stride=2, padding=1),
                    nn.ReLU(),
                ]
            )
            length *= 2
        self.proj = nn.Linear(latent_dim, width * init_length)
        self.init_length = init_length
        self.width = width
        self.net = nn.Sequential(*layers, nn.Conv1d(width, 1, kernel_size=3, padding=1))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.proj(z).reshape(z.shape[0], self.width, self.init_length)
        x = self.net(x)
        if x.shape[-1] != self.frames_per_epoch:
            x = F.interpolate(x, size=self.frames_per_epoch, mode="linear", align_corners=False)
        return x


class _ModalityAutoencoder(nn.Module):
    def __init__(self, *, frames_per_epoch: int, latent_dim: int) -> None:
        super().__init__()
        self.frames_per_epoch = int(frames_per_epoch)
        self.latent_dim = int(latent_dim)
        self.encoder = _EpochConvEncoder(frames_per_epoch=frames_per_epoch, latent_dim=latent_dim)
        self.decoder = _EpochDecoder(frames_per_epoch=frames_per_epoch, latent_dim=latent_dim)

    def _prepare_input(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int, int], bool]:
        if x.dim() == 3:
            batch_size, epoch_count, frames = x.shape
            channels = 1
            channel_first = False
            x = x.unsqueeze(2)
        elif x.dim() == 4:
            batch_size, epoch_count, channels, frames = x.shape
            channel_first = True
        else:
            raise ValueError(f"Autoencoder input must be [B, E, S] or [B, E, C, S], got shape {tuple(x.shape)}.")

        if frames != self.frames_per_epoch:
            raise ValueError(f"Expected {self.frames_per_epoch} frames per epoch, got {frames}.")

        return (
            x.reshape(batch_size * epoch_count * channels, 1, frames),
            (batch_size, epoch_count, channels),
            channel_first,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        prepared, (batch_size, epoch_count, channels), channel_first = self._prepare_input(x)
        channel_latent = self.encoder(prepared).reshape(batch_size, epoch_count, channels, -1)
        latent = channel_latent.mean(dim=2)
        reconstruction = self.decode(latent)
        if channels > 1:
            reconstruction = reconstruction.expand(
                batch_size, epoch_count, channels, self.frames_per_epoch
            ).contiguous()
        if not channel_first:
            reconstruction = reconstruction.squeeze(2)
        return latent, reconstruction

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.dim() != 3:
            raise ValueError(f"Latent must have shape [B, E, D], got {tuple(latent.shape)}.")
        batch_size, epoch_count, latent_dim = latent.shape
        if latent_dim != self.latent_dim:
            raise ValueError(f"Expected latent dim {self.latent_dim}, got {latent_dim}.")
        reconstruction = self.decoder(latent.reshape(batch_size * epoch_count, latent_dim))
        return reconstruction.reshape(batch_size, epoch_count, 1, self.frames_per_epoch)


class Sleep2WaveAutoencoder(nn.Module):
    def __init__(
        self,
        *,
        latent_dim: int,
        encoder_type: str = "conv1d_epoch",
        decoder_type: str = "convtranspose1d_epoch",
        modalities: t.Sequence[str] = CANONICAL_MODALITIES,
    ) -> None:
        super().__init__()
        if encoder_type != "conv1d_epoch":
            raise ValueError("sleep2wave autoencoder currently supports encoder_type='conv1d_epoch'.")
        if decoder_type != "convtranspose1d_epoch":
            raise ValueError("sleep2wave autoencoder currently supports decoder_type='convtranspose1d_epoch'.")
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive.")

        self.modalities = validate_modality_sequence(list(modalities), allow_aliases=False)
        self.latent_dim = int(latent_dim)
        self.modality_autoencoders = nn.ModuleDict(
            {
                modality: _ModalityAutoencoder(
                    frames_per_epoch=MODALITY_SPECS[modality].frames_per_epoch,
                    latent_dim=self.latent_dim,
                )
                for modality in self.modalities
            }
        )

    def forward(self, clean_signals: dict[str, torch.Tensor]) -> Sleep2WaveAutoencoderOutput:
        latents: dict[str, torch.Tensor] = {}
        reconstructions: dict[str, torch.Tensor] = {}
        for modality in self.modalities:
            if modality not in clean_signals:
                raise ValueError(f"Missing clean signal for modality '{modality}'.")
            latent, reconstruction = self.modality_autoencoders[modality](clean_signals[modality])
            latents[modality] = latent
            reconstructions[modality] = reconstruction
        return Sleep2WaveAutoencoderOutput(latents=latents, reconstructions=reconstructions)

    def decode_latents(self, latents: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if not latents:
            raise ValueError("latents must be non-empty.")
        reconstructions: dict[str, torch.Tensor] = {}
        for modality, latent in latents.items():
            if modality not in self.modality_autoencoders:
                raise ValueError(f"Unknown autoencoder modality '{modality}'.")
            reconstructions[modality] = self.modality_autoencoders[modality].decode(latent)
        return reconstructions
