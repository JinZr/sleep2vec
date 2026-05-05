from __future__ import annotations

from pathlib import Path
import pickle

import numpy as np
import pytest
import yaml

pytest.importorskip("torch")
pytest.importorskip("pytorch_lightning")
pytest.importorskip("pandas")
pytest.importorskip("wandb")

from sleep2wave.data.default_dataset import SampleIndex
from sleep2wave.data.modalities import CANONICAL_MODALITIES, MODALITY_SPECS
from sleep2wave.generative.config import load_sleep2wave_config
from sleep2wave.train_autoencoder import build_dataloader, main


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


def _write_config(tmp_path: Path, preset_path: Path) -> Path:
    payload = {
        "recipe": "sleep2wave",
        "stage": "autoencoder",
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
        "autoencoder": {
            "latent_dim": 8,
            "encoder_type": "conv1d_epoch",
            "decoder_type": "convtranspose1d_epoch",
            "one_latent_per_epoch": True,
            "modality_specific": True,
            "losses": {
                "waveform_l1_weight": 1.0,
                "waveform_l2_weight": 0.0,
                "spectral_weight": 0.0,
            },
        },
        "training": {
            "phase": 0,
            "batch_size": 1,
            "lr": 0.0001,
            "weight_decay": 0.01,
            "max_epochs": 1,
            "gradient_clip_val": 1.0,
        },
        "export": {"output_dir": str(tmp_path / "outputs")},
    }
    config_path = tmp_path / "autoencoder.yaml"
    config_path.write_text(yaml.safe_dump(payload))
    return config_path


def test_train_autoencoder_dataloader_uses_train_split(tmp_path: Path):
    preset_path = _write_synthetic_preset(tmp_path)
    with preset_path.open("rb") as f:
        train_sample = pickle.load(f)[0]
    val_sample = SampleIndex(
        id="synthetic-val",
        path=train_sample.path,
        start=train_sample.start,
        end=train_sample.end,
        payload=train_sample.payload,
        metadata={**train_sample.metadata, "split": "val"},
    )
    with preset_path.open("wb") as f:
        pickle.dump([train_sample, val_sample], f)
    config_path = _write_config(tmp_path, preset_path)

    loader = build_dataloader(load_sleep2wave_config(config_path), num_workers=0)

    assert [sample.metadata["split"] for sample in loader.dataset.data] == ["train"]


def test_train_autoencoder_smoke_writes_artifacts(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "offline")
    preset_path = _write_synthetic_preset(tmp_path)
    config_path = _write_config(tmp_path, preset_path)

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
