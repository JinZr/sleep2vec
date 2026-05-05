from __future__ import annotations

from pathlib import Path
import pickle

import numpy as np
import pytest
import pytorch_lightning  # noqa: F401
import torch
import wandb  # noqa: F401
import yaml

from sleep2wave.autoencoders.model import Sleep2WaveAutoencoder
from sleep2wave.data.default_dataset import SampleIndex
from sleep2wave.data.modalities import CANONICAL_MODALITIES, MODALITY_SPECS
from sleep2wave.diffusion.lightning import Sleep2WaveDiffusionLightning
from sleep2wave.diffusion.tasks import build_generation_task
from sleep2wave.generative.config import load_sleep2wave_config
from sleep2wave.train_diffusion import build_dataloader, main


def _write_synthetic_preset(tmp_path: Path) -> Path:
    npz_path = tmp_path / "synthetic.npz"
    npz_payload = {}
    for index, modality in enumerate(CANONICAL_MODALITIES):
        frames = 2 * MODALITY_SPECS[modality].frames_per_epoch
        npz_payload[modality] = np.linspace(0.0, 1.0 + index, frames, dtype=np.float32)
    np.savez(npz_path, **npz_payload)

    preset_path = tmp_path / "preset.pkl"
    sample = SampleIndex(
        id="synthetic",
        path=str(npz_path),
        start=0,
        end=2,
        payload={
            "available_channels": list(CANONICAL_MODALITIES),
            "canonical_channel_map": {modality: modality for modality in CANONICAL_MODALITIES},
            "quality_mask_keys": {},
            "availability_mask_keys": {},
            "subject_id": "subject",
            "night_id": "night",
            "night_epoch_count": 2,
        },
        metadata={"split": "train", "source": "synthetic"},
    )
    with preset_path.open("wb") as f:
        pickle.dump([sample], f)
    return preset_path


def _write_autoencoder_checkpoint(tmp_path: Path) -> Path:
    model = Sleep2WaveAutoencoder(latent_dim=8)
    checkpoint_path = tmp_path / "autoencoder.ckpt"
    torch.save({"state_dict": {f"model.{key}": value for key, value in model.state_dict().items()}}, checkpoint_path)
    return checkpoint_path


def _write_config(tmp_path: Path, preset_path: Path, autoencoder_ckpt: Path) -> Path:
    payload = {
        "recipe": "sleep2wave",
        "stage": "diffusion",
        "data": {"preset_path": str(preset_path), "context_epochs": 2},
        "modalities": {
            "epoch_sec": 30,
            "all": list(CANONICAL_MODALITIES),
            "high_frequency": ["eeg", "eog", "emg", "ecg"],
            "low_frequency": ["airflow", "belt", "spo2", "ibi", "resp"],
            "sample_rates": {modality: MODALITY_SPECS[modality].sample_rate_hz for modality in CANONICAL_MODALITIES},
            "frames_per_epoch": {
                modality: MODALITY_SPECS[modality].frames_per_epoch for modality in CANONICAL_MODALITIES
            },
        },
        "diffusion": {
            "latent_dim": 8,
            "autoencoder_checkpoint": str(autoencoder_ckpt),
            "transformer": {"hidden_size": 8, "num_layers": 1, "num_heads": 2, "mlp_ratio": 2},
            "diffusion_steps": 8,
            "beta_schedule": "cosine",
            "prediction_type": "epsilon",
            "context_epochs": 2,
            "embeddings": {
                "diffusion_step": True,
                "modality": True,
                "epoch_position": True,
                "sleep_night_position": True,
                "availability": True,
                "quality": True,
            },
            "task_attention_mask": "directional",
            "auxiliary_restoration_token": True,
            "condition_dropout": 0.15,
        },
        "training": {
            "phase": 1,
            "batch_size": 1,
            "lr": 0.0001,
            "weight_decay": 0.01,
            "max_epochs": 1,
            "gradient_clip_val": 1.0,
            "task_mix": {"restoration": 1.0},
            "condition_counts": [1],
            "replay": {"enabled": False},
        },
        "sampler": {"name": "ddim", "steps": 2, "eta": 0.0, "num_samples": 1},
        "export": {"output_dir": str(tmp_path / "outputs")},
    }
    config_path = tmp_path / "diffusion.yaml"
    config_path.write_text(yaml.safe_dump(payload))
    return config_path


def test_train_diffusion_dataloader_uses_train_split(tmp_path: Path):
    preset_path = _write_synthetic_preset(tmp_path)
    autoencoder_ckpt = _write_autoencoder_checkpoint(tmp_path)
    with preset_path.open("rb") as f:
        train_sample = pickle.load(f)[0]
    test_sample = SampleIndex(
        id="synthetic-test",
        path=train_sample.path,
        start=train_sample.start,
        end=train_sample.end,
        payload=train_sample.payload,
        metadata={**train_sample.metadata, "split": "test"},
    )
    with preset_path.open("wb") as f:
        pickle.dump([train_sample, test_sample], f)
    config_path = _write_config(tmp_path, preset_path, autoencoder_ckpt)

    loader = build_dataloader(load_sleep2wave_config(config_path), num_workers=0, seed=123)

    assert [sample.metadata["split"] for sample in loader.dataset.data] == ["train"]
    assert loader.dataset.seed == 123


def test_train_diffusion_smoke_writes_artifacts(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "offline")
    preset_path = _write_synthetic_preset(tmp_path)
    autoencoder_ckpt = _write_autoencoder_checkpoint(tmp_path)
    config_path = _write_config(tmp_path, preset_path, autoencoder_ckpt)

    main(
        [
            "--config",
            str(config_path),
            "--version-name",
            "smoke",
            "--accelerator",
            "cpu",
            "--devices",
            "1",
            "--max-steps",
            "1",
            "--precision",
            "32",
        ]
    )

    run_dir = tmp_path / "outputs" / "smoke"
    assert (run_dir / "config.yaml").exists()
    assert (run_dir / "cli_args.yaml").exists()
    assert (run_dir / "checkpoints" / "last.ckpt").exists()


def test_condition_dropout_keeps_nonempty_condition_set(tmp_path: Path):
    preset_path = _write_synthetic_preset(tmp_path)
    autoencoder_ckpt = _write_autoencoder_checkpoint(tmp_path)
    config_path = _write_config(tmp_path, preset_path, autoencoder_ckpt)
    payload = yaml.safe_load(config_path.read_text())
    payload["diffusion"]["condition_dropout"] = 1.0
    config_path.write_text(yaml.safe_dump(payload))
    module = Sleep2WaveDiffusionLightning(load_sleep2wave_config(config_path))
    task = build_generation_task(
        "translation",
        condition_modalities=["eeg", "ecg"],
        target_modalities=["spo2"],
    )

    dropped = module._apply_condition_dropout(task)

    assert len(dropped.condition_modalities) == 1
    assert set(dropped.condition_modalities).issubset({"eeg", "ecg"})
