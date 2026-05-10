from __future__ import annotations

from pathlib import Path
import pickle

import numpy as np
import pandas  # noqa: F401
import pytest
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
    data_block: dict | None = None,
) -> Path:
    autoencoder = {
        "latent_dim": 8,
        "encoder_type": "temporal_conv",
        "decoder_type": "temporal_conv",
        "latent_frames_per_epoch": {"high_frequency": 60, "low_frequency": 30},
        "channel_specific": True,
        "losses": {
            "waveform_l1_weight": 1.0,
            "waveform_l2_weight": 0.0,
            "spectral_weight": 0.0,
            "derivative_l1_weight": 0.0,
            "mr_stft_weight": 0.0,
        },
    }
    training = {
        "phase": 0,
        "batch_size": batch_size,
        "lr": 0.0001,
        "weight_decay": 0.01,
        "max_epochs": 1,
        "gradient_clip_val": 1.0,
    }
    if validation_examples is not None:
        training["validation"] = {"examples": validation_examples}
    payload = {
        "recipe": "sleep2wave",
        "stage": "autoencoder",
        "data": data_block or {"preset_path": str(preset_path), "context_epochs": 2},
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
        "training": training,
        "export": {"output_dir": str(tmp_path / "outputs")},
    }
    config_path = tmp_path / "autoencoder.yaml"
    config_path.write_text(yaml.safe_dump(payload))
    return config_path


def _write_synthetic_index(tmp_path: Path) -> Path:
    npz_path = tmp_path / "synthetic_kaldi.npz"
    npz_payload = {}
    for index, modality in enumerate(CANONICAL_MODALITIES):
        frames = 2 * MODALITY_SPECS[modality].frames_per_epoch
        npz_payload[modality] = np.linspace(0.0, 1.0 + index, frames, dtype=np.float32)
    np.savez(npz_path, **npz_payload)

    index_path = tmp_path / "index.csv"
    import pandas as pd

    row = {
        "path": str(npz_path),
        "duration": 60,
        "split": "train",
        "subject_id": "subject",
        "night_id": "night",
        "source": "synthetic",
    }
    row.update({f"{modality}_mask": 1 for modality in CANONICAL_MODALITIES})
    pd.DataFrame([row]).to_csv(index_path, index=False)
    return index_path


def _write_synthetic_kaldi_root(tmp_path: Path, config_path: Path) -> tuple[Path, Path]:
    pytest.importorskip("kaldi_native_io")
    from sleep2wave.preprocess.convert_npz_to_kaldi import convert, parse_args

    output_dir = tmp_path / "kaldi"
    index_path = _write_synthetic_index(tmp_path)
    manifest_path, _ = convert(
        parse_args(
            [
                "--index",
                str(index_path),
                "--config",
                str(config_path),
                "--output-dir",
                str(output_dir),
                "--split",
                "train",
                "--stride-epochs",
                "2",
            ]
        )
    )
    return output_dir, manifest_path


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


def test_train_autoencoder_dataloader_uses_kaldi_backend(tmp_path: Path):
    preset_path = _write_synthetic_preset(tmp_path)
    npz_config_path = _write_config(tmp_path, preset_path)
    kaldi_root, kaldi_manifest = _write_synthetic_kaldi_root(tmp_path, npz_config_path)
    config_path = _write_config(
        tmp_path,
        preset_path,
        data_block={
            "backend": "kaldi",
            "kaldi_data_root": str(kaldi_root),
            "kaldi_manifest": str(kaldi_manifest),
            "context_epochs": 2,
        },
    )

    loader = build_dataloader(load_sleep2wave_config(config_path), num_workers=0)

    assert loader.dataset.backend == "kaldi"
    batch = next(iter(loader))
    assert batch["clean_signals"]["eeg"].shape == (1, 2, 1, MODALITY_SPECS["eeg"].frames_per_epoch)


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

    assert {
        "loss",
        "waveform_l1_loss",
        "waveform_l2_loss",
        "spectral_loss",
        "derivative_l1_loss",
        "mr_stft_loss",
    } <= set(losses)
    assert {
        "val_loss",
        "val_waveform_l1_loss",
        "val_waveform_l2_loss",
        "val_spectral_loss",
        "val_derivative_l1_loss",
        "val_mr_stft_loss",
    } <= set(logged_scalars)
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


def test_train_autoencoder_registers_step_lr_monitor(tmp_path: Path, monkeypatch):
    import sleep2wave.train_autoencoder as train_autoencoder

    preset_path = _write_synthetic_preset(tmp_path)
    config_path = _write_config(tmp_path, preset_path)
    trainer_kwargs = {}
    lr_monitor = object()

    class DummyTrainer:
        def __init__(self, **kwargs):
            trainer_kwargs.update(kwargs)

        def fit(self, *args, **kwargs):
            pass

        def save_checkpoint(self, path):
            Path(path).write_text("checkpoint")

    monkeypatch.setattr(train_autoencoder, "WandbLogger", lambda *args, **kwargs: object())
    monkeypatch.setattr(train_autoencoder, "ModelCheckpoint", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        train_autoencoder,
        "LearningRateMonitor",
        lambda *args, **kwargs: lr_monitor if kwargs == {"logging_interval": "step"} else None,
    )
    monkeypatch.setattr(train_autoencoder.pl, "Trainer", DummyTrainer)

    train_autoencoder.main(
        [
            "--config",
            str(config_path),
            "--version-name",
            "lr-monitor",
            "--accelerator",
            "cpu",
            "--devices",
            "1",
            "--max-steps",
            "0",
            "--precision",
            "32",
        ]
    )

    assert lr_monitor in trainer_kwargs["callbacks"]
    assert trainer_kwargs["val_check_interval"] == 1000
    assert trainer_kwargs["check_val_every_n_epoch"] is None
    assert trainer_kwargs["limit_val_batches"] == 1
