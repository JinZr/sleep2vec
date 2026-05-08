from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sleep2wave.diffusion.latent_cache import (
    LATENT_CACHE_SCHEMA_VERSION,
    Sleep2WaveLatentCacheDataset,
    write_latent_cache,
)


def _write_cache(tmp_path: Path) -> Path:
    clean_latents = {
        "eeg": torch.randn(2, 2, 2, 60, 4),
        "spo2": torch.randn(2, 2, 1, 30, 4),
    }
    availability = {
        "eeg": torch.ones(2, 2, dtype=torch.bool),
        "spo2": torch.tensor([[True, True], [False, False]]),
    }
    quality = {
        "eeg": torch.ones(2, 2),
        "spo2": torch.ones(2, 2),
    }
    channel_mask = {
        "eeg": torch.tensor([[[True, True], [True, False]], [[True, True], [True, True]]]),
        "spo2": torch.ones(2, 2, 1, dtype=torch.bool),
    }
    return write_latent_cache(
        tmp_path / "cache",
        clean_latents=clean_latents,
        availability_mask=availability,
        quality_mask=quality,
        channel_mask=channel_mask,
        epoch_index=torch.tensor([[0, 1], [2, 3]]),
        night_position=torch.tensor([[0.0, 0.5], [0.6, 1.0]]),
        metadata_rows=[
            {"id": "row-0", "split": "train"},
            {"id": "row-1", "split": "val"},
        ],
        latent_frames_per_epoch={"high_frequency": 60, "low_frequency": 30},
        patches_per_epoch=6,
        modalities=["eeg", "spo2"],
    )


def test_latent_cache_writes_schema_v2_and_preserves_channel_masks(tmp_path: Path):
    cache_path = _write_cache(tmp_path)

    manifest = json.loads((cache_path / "manifest.json").read_text())
    dataset = Sleep2WaveLatentCacheDataset(cache_path)
    item = dataset[0]
    batch = next(iter(dataset.dataloader(batch_size=2)))

    assert manifest["schema_version"] == LATENT_CACHE_SCHEMA_VERSION
    assert manifest["latent_frames_per_epoch"] == {"high_frequency": 60, "low_frequency": 30}
    assert manifest["patches_per_epoch"] == 6
    assert manifest["channel_specific"] is True
    assert item["clean_latents"]["eeg"].shape == (2, 2, 60, 4)
    assert item["channel_mask"]["eeg"].shape == (2, 2)
    assert batch["clean_latents"]["eeg"].shape == (2, 2, 2, 60, 4)
    assert batch["channel_mask"]["eeg"].shape == (2, 2, 2)


def test_latent_cache_exposes_available_channels_for_bucket_sampler(tmp_path: Path):
    cache_path = _write_cache(tmp_path)

    dataset = Sleep2WaveLatentCacheDataset(cache_path)
    train_dataset = Sleep2WaveLatentCacheDataset(cache_path, split="train")

    assert dataset.data[0].payload["available_channels"] == ["eeg", "spo2"]
    assert dataset.data[1].payload["available_channels"] == ["eeg"]
    assert len(train_dataset) == 1
    assert train_dataset.data[0].payload["available_channels"] == ["eeg", "spo2"]


def test_build_latent_cache_pads_channel_chunks_before_write(tmp_path: Path, monkeypatch):
    from sleep2wave.cache_latents import build_latent_cache

    batches = [
        {
            "clean_signals": {"eeg": torch.zeros(1, 2, 1), "spo2": torch.zeros(1, 2, 1)},
            "availability_mask": {
                "eeg": torch.ones(1, 2, dtype=torch.bool),
                "spo2": torch.ones(1, 2, dtype=torch.bool),
            },
            "quality_mask": {"eeg": torch.ones(1, 2), "spo2": torch.ones(1, 2)},
            "channel_mask": {
                "eeg": torch.ones(1, 2, 1, dtype=torch.bool),
                "spo2": torch.ones(1, 2, 1, dtype=torch.bool),
            },
            "epoch_index": torch.tensor([[0, 1]]),
            "night_position": torch.tensor([[0.0, 0.5]]),
            "metadata": {"id": ["row-0"], "split": ["train"]},
        },
        {
            "clean_signals": {"eeg": torch.zeros(1, 2, 1), "spo2": torch.zeros(1, 2, 1)},
            "availability_mask": {
                "eeg": torch.ones(1, 2, dtype=torch.bool),
                "spo2": torch.ones(1, 2, dtype=torch.bool),
            },
            "quality_mask": {"eeg": torch.ones(1, 2), "spo2": torch.ones(1, 2)},
            "channel_mask": {
                "eeg": torch.ones(1, 2, 2, dtype=torch.bool),
                "spo2": torch.ones(1, 2, 1, dtype=torch.bool),
            },
            "epoch_index": torch.tensor([[2, 3]]),
            "night_position": torch.tensor([[0.6, 1.0]]),
            "metadata": {"id": ["row-1"], "split": ["train"]},
        },
    ]

    class FakeDataset:
        def __init__(self, **_kwargs):
            pass

        def dataloader(self, **_kwargs):
            return batches

    class FakeAutoencoder:
        def __init__(self):
            self.calls = 0

        def __call__(self, _signals):
            channels = 1 if self.calls == 0 else 2
            self.calls += 1
            return SimpleNamespace(
                latents={
                    "eeg": torch.ones(1, 2, channels, 60, 4),
                    "spo2": torch.ones(1, 2, 1, 30, 4),
                }
            )

    config = SimpleNamespace(
        data=SimpleNamespace(preset_path="preset.pkl", index=None, context_epochs=2),
        diffusion=SimpleNamespace(
            autoencoder_checkpoint="autoencoder.ckpt",
            latent_dim=4,
            latent_frames_per_epoch={"high_frequency": 60, "low_frequency": 30},
            patches_per_epoch=6,
        ),
        modalities=SimpleNamespace(all=["eeg", "spo2"]),
    )
    monkeypatch.setattr("sleep2wave.generative.config.load_sleep2wave_config", lambda _path: config)
    monkeypatch.setattr(
        "sleep2wave.autoencoders.checkpoints.load_sleep2wave_autoencoder_checkpoint",
        lambda *_args, **_kwargs: FakeAutoencoder(),
    )
    monkeypatch.setattr("sleep2wave.data.generative_dataset.Sleep2WaveGenerativeDataset", FakeDataset)

    cache_path = build_latent_cache(
        SimpleNamespace(
            config=tmp_path / "config.yaml",
            autoencoder_ckpt=None,
            output_dir=tmp_path / "cache",
            batch_size=1,
            num_workers=0,
            device="cpu",
        )
    )
    dataset = Sleep2WaveLatentCacheDataset(cache_path)

    assert dataset.arrays["latents/eeg"].shape == (2, 2, 2, 60, 4)
    assert not dataset.arrays["latents/eeg"][0, :, 1].any()
    assert dataset.arrays["channel_mask/eeg"].shape == (2, 2, 2)
    assert not dataset.arrays["channel_mask/eeg"][0, :, 1].any()
    assert dataset.arrays["channel_mask/eeg"][1, :, 1].all()


def test_latent_cache_rejects_schema_v1_cache(tmp_path: Path):
    cache_path = tmp_path / "cache"
    cache_path.mkdir()
    (cache_path / "manifest.json").write_text(json.dumps({"schema_version": 1}) + "\n")
    np.savez(cache_path / "latents.npz")
    (cache_path / "metadata.jsonl").write_text("")

    with pytest.raises(ValueError, match="Unsupported sleep2wave latent cache schema_version"):
        Sleep2WaveLatentCacheDataset(cache_path)


def test_latent_cache_rejects_3d_latents(tmp_path: Path):
    cache_path = tmp_path / "cache"
    cache_path.mkdir()
    manifest = {
        "schema_version": LATENT_CACHE_SCHEMA_VERSION,
        "artifact_type": "sleep2wave_latent_cache",
        "modalities": ["eeg"],
        "num_windows": 1,
        "context_epochs": 2,
        "autoencoder_type": "temporal_conv",
        "latent_dim": 4,
        "latent_frames_per_epoch": {"high_frequency": 60, "low_frequency": 30},
        "patches_per_epoch": 6,
        "channel_specific": True,
    }
    (cache_path / "manifest.json").write_text(json.dumps(manifest) + "\n")
    np.savez(
        cache_path / "latents.npz",
        **{
            "epoch_index": np.array([[0, 1]]),
            "night_position": np.array([[0.0, 1.0]]),
            "latents/eeg": np.zeros((1, 2, 4), dtype=np.float32),
            "availability/eeg": np.ones((1, 2), dtype=bool),
            "quality/eeg": np.ones((1, 2), dtype=np.float32),
            "channel_mask/eeg": np.ones((1, 2, 1), dtype=bool),
        },
    )
    (cache_path / "metadata.jsonl").write_text(json.dumps({"split": "train"}) + "\n")

    with pytest.raises(ValueError, match=r"latents/eeg must have shape \[N, E, C, L, D\]"):
        Sleep2WaveLatentCacheDataset(cache_path)
