from __future__ import annotations

import json
from pathlib import Path
import typing as t

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from sleep2wave.data.generative_batch import collate_sleep2wave_generative
from sleep2wave.data.modalities import CANONICAL_MODALITIES, validate_modality_sequence

LATENT_CACHE_SCHEMA_VERSION = 1


def _npz_key(family: str, modality: str) -> str:
    return f"{family}/{modality}"


def write_latent_cache(
    output_dir: str | Path,
    *,
    clean_latents: dict[str, torch.Tensor],
    availability_mask: dict[str, torch.Tensor],
    quality_mask: dict[str, torch.Tensor],
    epoch_index: torch.Tensor,
    night_position: torch.Tensor,
    metadata_rows: list[dict[str, t.Any]],
    modalities: t.Sequence[str] = CANONICAL_MODALITIES,
) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    modalities = validate_modality_sequence(list(modalities), allow_aliases=False)
    if not clean_latents:
        raise ValueError("clean_latents must be non-empty.")
    first = next(iter(clean_latents.values()))
    if first.dim() != 3:
        raise ValueError("Latent tensors must have shape [N, E, D].")
    manifest = {
        "schema_version": LATENT_CACHE_SCHEMA_VERSION,
        "artifact_type": "sleep2wave_latent_cache",
        "modalities": modalities,
        "num_windows": int(first.shape[0]),
        "context_epochs": int(first.shape[1]),
        "latent_dim": int(first.shape[2]),
    }
    arrays: dict[str, np.ndarray] = {
        "epoch_index": epoch_index.detach().cpu().numpy(),
        "night_position": night_position.detach().cpu().numpy(),
    }
    for modality in modalities:
        arrays[_npz_key("latents", modality)] = clean_latents[modality].detach().cpu().numpy()
        arrays[_npz_key("availability", modality)] = availability_mask[modality].detach().cpu().numpy()
        arrays[_npz_key("quality", modality)] = quality_mask[modality].detach().cpu().numpy()
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
        with np.load(latents_path) as loaded:
            self.arrays = {key: loaded[key] for key in loaded.files}
        self.metadata = [json.loads(line) for line in metadata_path.read_text().splitlines() if line.strip()]
        if split is not None:
            split_values = {split} if isinstance(split, str) else set(split)
            self.indices = [idx for idx, row in enumerate(self.metadata) if row.get("split") in split_values]
        else:
            self.indices = list(range(len(self.metadata)))
        if not self.indices:
            raise ValueError("No sleep2wave latent cache rows are available.")

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
        zero_signals = {
            modality: torch.zeros(
                clean_latents[modality].shape[0],
                1,
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
            "channel_mask": {
                modality: torch.ones((clean_latents[modality].shape[0], 1), dtype=torch.bool)
                for modality in self.modalities
            },
            "epoch_index": torch.as_tensor(self.arrays["epoch_index"][source_idx], dtype=torch.long),
            "night_position": torch.as_tensor(self.arrays["night_position"][source_idx], dtype=torch.float32),
            "metadata": self.metadata[source_idx],
            "condition_modalities": [],
            "target_modalities": [],
            "task_type": "translation",
        }


__all__ = ["LATENT_CACHE_SCHEMA_VERSION", "Sleep2WaveLatentCacheDataset", "write_latent_cache"]
