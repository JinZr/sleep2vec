from __future__ import annotations

from pathlib import Path
import pickle

import numpy as np
import pandas  # noqa: F401
import pytorch_lightning  # noqa: F401
import torch  # noqa: F401
import wandb  # noqa: F401
import yaml

from sleep2wave.autoencoders.lightning import Sleep2WaveAutoencoderLightning
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
    payload = {
        "available_channels": list(CANONICAL_MODALITIES),
        "canonical_channel_map": {modality: modality for modality in CANONICAL_MODALITIES},
        "quality_mask_keys": {},
        "availability_mask_keys": {},
        "subject_id": "subject",
        "night_id": "night",
        "night_epoch_count": 2,
    }
    samples = [
        SampleIndex(
            id=f"synthetic-{split}-{index}",
            path=str(npz_path),
            start=0,
            end=2,
            payload=payload,
            metadata={"split": split, "source": "synthetic"},
        )
        for split, count in (("train", 1), ("val", 2))
        for index in range(count)
    ]
    with preset_path.open("wb") as f:
        pickle.dump(samples, f)
    return preset_path


def _write_config(
    tmp_path: Path,
    preset_path: Path,
    *,
    batch_size: int = 1,
    validation_examples: dict | None = None,
) -> Path:
    autoencoder = {
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
    }
    if validation_examples is not None:
        autoencoder["validation_examples"] = validation_examples
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
        "autoencoder": autoencoder,
        "training": {
            "phase": 0,
            "batch_size": batch_size,
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


def test_autoencoder_validation_step_logs_configured_examples(tmp_path: Path, monkeypatch):
    preset_path = _write_synthetic_preset(tmp_path)
    config_path = _write_config(
        tmp_path,
        preset_path,
        batch_size=2,
        validation_examples={"num_examples": 2, "modalities": ["eeg", "spo2"]},
    )
    config = load_sleep2wave_config(config_path)
    loader = build_dataloader(config, num_workers=0, split="val")
    batch = next(iter(loader))
    batch["availability_mask"]["eeg"][0] = False
    model = Sleep2WaveAutoencoderLightning(config)
    logged_scalars = {}
    logged_payloads = []

    def fake_lightning_log(name, value, **kwargs):
        logged_scalars[name] = (value, kwargs)

    def fake_wandb_log(payload, commit=True):
        logged_payloads.append((payload, commit))

    monkeypatch.setattr(model, "log", fake_lightning_log)
    monkeypatch.setattr(wandb, "run", object())
    monkeypatch.setattr(wandb, "Image", lambda fig: fig)
    monkeypatch.setattr(wandb, "log", fake_wandb_log)

    losses = model.validation_step(batch, 0)

    assert {"loss", "waveform_l1_loss", "waveform_l2_loss", "spectral_loss"} <= set(losses)
    assert {"val_loss", "val_waveform_l1_loss", "val_waveform_l2_loss", "val_spectral_loss"} <= set(logged_scalars)
    assert len(logged_payloads) == 1
    payload, commit = logged_payloads[0]
    assert commit is False
    assert set(payload) == {
        "val_autoencoder_examples/eeg_sample_1",
        "val_autoencoder_examples/spo2_sample_0",
        "val_autoencoder_examples/spo2_sample_1",
    }


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
