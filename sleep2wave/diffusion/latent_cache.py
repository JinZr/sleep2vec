from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import typing as t

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from sleep2wave.data.generative_batch import collate_sleep2wave_generative
from sleep2wave.data.modalities import CANONICAL_MODALITIES, MODALITY_SPECS, validate_modality_sequence

LATENT_CACHE_SCHEMA_VERSION = 2


def _npz_key(family: str, modality: str) -> str:
    return f"{family}/{modality}"


def write_latent_cache(
    output_dir: str | Path,
    *,
    clean_latents: dict[str, torch.Tensor],
    availability_mask: dict[str, torch.Tensor],
    quality_mask: dict[str, torch.Tensor],
    channel_mask: dict[str, torch.Tensor],
    epoch_index: torch.Tensor,
    night_position: torch.Tensor,
    metadata_rows: list[dict[str, t.Any]],
    latent_frames_per_epoch: t.Mapping[str, int],
    patches_per_epoch: int,
    modalities: t.Sequence[str] = CANONICAL_MODALITIES,
) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    modalities = validate_modality_sequence(list(modalities), allow_aliases=False)
    if not clean_latents:
        raise ValueError("clean_latents must be non-empty.")
    if patches_per_epoch <= 0:
        raise ValueError("patches_per_epoch must be positive.")
    first = clean_latents[modalities[0]]
    if first.dim() != 5:
        raise ValueError("Latent tensors must have shape [N, E, C, L, D].")
    num_windows, context_epochs, _channels, _latent_frames, latent_dim = first.shape
    latent_shapes: dict[str, list[int]] = {}
    for modality in modalities:
        if modality not in clean_latents:
            raise ValueError(f"Missing clean latents for modality '{modality}'.")
        latent = clean_latents[modality]
        if latent.dim() != 5:
            raise ValueError(f"Latents for '{modality}' must have shape [N, E, C, L, D].")
        if latent.shape[0] != num_windows or latent.shape[1] != context_epochs or latent.shape[-1] != latent_dim:
            raise ValueError(f"Latents for '{modality}' must share num_windows, context_epochs, and latent_dim.")
        frequency_group = MODALITY_SPECS[modality].frequency_group
        expected_frames = latent_frames_per_epoch[frequency_group]
        if latent.shape[3] != expected_frames:
            raise ValueError(f"Latents for '{modality}' have {latent.shape[3]} frames; expected {expected_frames}.")
        mask = channel_mask[modality]
        if tuple(mask.shape) != tuple(latent.shape[:3]):
            raise ValueError(f"channel_mask['{modality}'] must have shape {tuple(latent.shape[:3])}.")
        latent_shapes[modality] = [int(value) for value in latent.shape]
    manifest = {
        "schema_version": LATENT_CACHE_SCHEMA_VERSION,
        "artifact_type": "sleep2wave_latent_cache",
        "modalities": modalities,
        "num_windows": int(num_windows),
        "context_epochs": int(context_epochs),
        "autoencoder_type": "temporal_conv",
        "latent_dim": int(latent_dim),
        "latent_frames_per_epoch": dict(latent_frames_per_epoch),
        "patches_per_epoch": int(patches_per_epoch),
        "channel_specific": True,
        "latent_shapes": latent_shapes,
    }
    arrays: dict[str, np.ndarray] = {
        "epoch_index": epoch_index.detach().cpu().numpy(),
        "night_position": night_position.detach().cpu().numpy(),
    }
    for modality in modalities:
        arrays[_npz_key("latents", modality)] = clean_latents[modality].detach().cpu().numpy()
        arrays[_npz_key("availability", modality)] = availability_mask[modality].detach().cpu().numpy()
        arrays[_npz_key("quality", modality)] = quality_mask[modality].detach().cpu().numpy()
        arrays[_npz_key("channel_mask", modality)] = channel_mask[modality].detach().cpu().numpy()
    np.savez(output / "latents.npz", **arrays)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    with (output / "metadata.jsonl").open("w") as f:
        for row in metadata_rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    return output


class Sleep2WaveLatentCacheDataset(Dataset):
    def __init__(
        self,
        cache_path: str | Path,
        *,
        split: str | t.Sequence[str] | None = None,
    ) -> None:
        self.cache_path = Path(cache_path)
        manifest_path = self.cache_path / "manifest.json"
        latents_path = self.cache_path / "latents.npz"
        metadata_path = self.cache_path / "metadata.jsonl"
        if not manifest_path.is_file() or not latents_path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(f"Invalid sleep2wave latent cache directory: {self.cache_path}")
        self.manifest = json.loads(manifest_path.read_text())
        if self.manifest.get("schema_version") != LATENT_CACHE_SCHEMA_VERSION:
            raise ValueError("Unsupported sleep2wave latent cache schema_version.")
        self.modalities = tuple(validate_modality_sequence(self.manifest["modalities"], allow_aliases=False))
        self.latent_frames_per_epoch = self.manifest.get("latent_frames_per_epoch", {})
        self.patches_per_epoch = int(self.manifest.get("patches_per_epoch", 0))
        if self.patches_per_epoch <= 0:
            raise ValueError("sleep2wave latent cache manifest must define patches_per_epoch.")
        with np.load(latents_path) as loaded:
            self.arrays = {key: loaded[key] for key in loaded.files}
        self.metadata = [json.loads(line) for line in metadata_path.read_text().splitlines() if line.strip()]
        self._validate_arrays()
        if split is not None:
            split_values = {split} if isinstance(split, str) else set(split)
            self.indices = [idx for idx, row in enumerate(self.metadata) if row.get("split") in split_values]
        else:
            self.indices = list(range(len(self.metadata)))
        if not self.indices:
            raise ValueError("No sleep2wave latent cache rows are available.")
        self.data = [
            SimpleNamespace(
                id=self.metadata[source_idx].get("id", source_idx),
                path=str(self.cache_path),
                payload={"available_channels": self._available_modalities_for_row(source_idx)},
            )
            for source_idx in self.indices
        ]

    def _validate_arrays(self) -> None:
        expected_windows = int(self.manifest["num_windows"])
        expected_epochs = int(self.manifest["context_epochs"])
        expected_dim = int(self.manifest["latent_dim"])
        if len(self.metadata) != expected_windows:
            raise ValueError("sleep2wave latent cache metadata row count does not match manifest.")
        if tuple(self.arrays["epoch_index"].shape) != (expected_windows, expected_epochs):
            raise ValueError("epoch_index must have shape [N, E].")
        if tuple(self.arrays["night_position"].shape) != (expected_windows, expected_epochs):
            raise ValueError("night_position must have shape [N, E].")
        for modality in self.modalities:
            latent = self.arrays[_npz_key("latents", modality)]
            if latent.ndim != 5:
                raise ValueError(f"latents/{modality} must have shape [N, E, C, L, D].")
            if (
                latent.shape[0] != expected_windows
                or latent.shape[1] != expected_epochs
                or latent.shape[-1] != expected_dim
            ):
                raise ValueError(f"latents/{modality} does not match cache window, epoch, or latent-dim metadata.")
            frequency_group = MODALITY_SPECS[modality].frequency_group
            expected_frames = self.latent_frames_per_epoch.get(frequency_group)
            if latent.shape[3] != expected_frames:
                raise ValueError(f"latents/{modality} has incompatible latent-frame count.")
            if tuple(self.arrays[_npz_key("channel_mask", modality)].shape) != tuple(latent.shape[:3]):
                raise ValueError(f"channel_mask/{modality} must have shape [N, E, C].")

    def _available_modalities_for_row(self, source_idx: int) -> list[str]:
        available = []
        for modality in self.modalities:
            values = self.arrays[_npz_key("availability", modality)][source_idx]
            if np.asarray(values).astype(bool).any():
                available.append(modality)
        return available

    def __len__(self) -> int:
        return len(self.indices)

    def dataloader(self, **kwargs: t.Any) -> DataLoader:
        return DataLoader(self, collate_fn=collate_sleep2wave_generative, **kwargs)

    def __getitem__(self, idx: int) -> dict[str, t.Any]:
        source_idx = self.indices[idx]
        clean_latents = {
            modality: torch.as_tensor(self.arrays[_npz_key("latents", modality)][source_idx], dtype=torch.float32)
            for modality in self.modalities
        }
        availability_mask = {
            modality: torch.as_tensor(self.arrays[_npz_key("availability", modality)][source_idx], dtype=torch.bool)
            for modality in self.modalities
        }
        quality_mask = {
            modality: torch.as_tensor(self.arrays[_npz_key("quality", modality)][source_idx], dtype=torch.float32)
            for modality in self.modalities
        }
        channel_mask = {
            modality: torch.as_tensor(self.arrays[_npz_key("channel_mask", modality)][source_idx], dtype=torch.bool)
            for modality in self.modalities
        }
        zero_signals = {
            modality: torch.zeros(
                clean_latents[modality].shape[0],
                clean_latents[modality].shape[1],
                1,
                dtype=torch.float32,
            )
            for modality in self.modalities
        }
        return {
            "clean_signals": zero_signals,
            "observed_signals": zero_signals,
            "clean_latents": clean_latents,
            "observed_latents": clean_latents,
            "availability_mask": availability_mask,
            "quality_mask": quality_mask,
            "corruption_mask": {
                modality: torch.zeros_like(availability_mask[modality], dtype=torch.bool)
                for modality in self.modalities
            },
            "channel_mask": channel_mask,
            "epoch_index": torch.as_tensor(self.arrays["epoch_index"][source_idx], dtype=torch.long),
            "night_position": torch.as_tensor(self.arrays["night_position"][source_idx], dtype=torch.float32),
            "metadata": self.metadata[source_idx],
            "condition_modalities": [],
            "target_modalities": [],
            "task_type": "translation",
        }


__all__ = ["LATENT_CACHE_SCHEMA_VERSION", "Sleep2WaveLatentCacheDataset", "write_latent_cache"]
