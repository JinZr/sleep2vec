from __future__ import annotations

from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pytest
import pytorch_lightning  # noqa: F401
import torch
import wandb  # noqa: F401
import yaml

from sleep2wave.autoencoders.model import Sleep2WaveAutoencoder
from sleep2wave.data.default_dataset import SampleIndex
from sleep2wave.data.modalities import CANONICAL_MODALITIES, MODALITY_SPECS
from sleep2wave.data.samplers import AvailableChannelsBucketBatchSampler
from sleep2wave.diffusion.latent_cache import write_latent_cache
import sleep2wave.diffusion.lightning as diffusion_lightning
from sleep2wave.diffusion.lightning import Sleep2WaveDiffusionLightning
from sleep2wave.diffusion.tasks import build_generation_task
from sleep2wave.generative.config import load_sleep2wave_config
from sleep2wave.train_diffusion import _limit_val_batches, _load_phase_checkpoint, build_dataloader, main
from sleep2wave.training.task_sampler import Sleep2WaveTaskSampler


def _write_synthetic_preset(tmp_path: Path, *, channel_counts: dict[str, int] | None = None) -> Path:
    npz_path = tmp_path / "synthetic.npz"
    npz_payload = {}
    for index, modality in enumerate(CANONICAL_MODALITIES):
        frames = 2 * MODALITY_SPECS[modality].frames_per_epoch
        signal = np.linspace(0.0, 1.0 + index, frames, dtype=np.float32)
        channels = (channel_counts or {}).get(modality, 1)
        if channels == 1:
            npz_payload[modality] = signal
        else:
            npz_payload[modality] = np.stack([signal + channel for channel in range(channels)], axis=0)
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


def _write_autoencoder_checkpoint(tmp_path: Path) -> Path:
    model = Sleep2WaveAutoencoder(latent_dim=8)
    checkpoint_path = tmp_path / "autoencoder.ckpt"
    torch.save({"state_dict": {f"model.{key}": value for key, value in model.state_dict().items()}}, checkpoint_path)
    return checkpoint_path


def _write_latent_cache(tmp_path: Path) -> Path:
    latents = {}
    availability = {}
    quality = {}
    channel_mask = {}
    for modality in CANONICAL_MODALITIES:
        group = MODALITY_SPECS[modality].frequency_group
        latent_frames = 60 if group == "high_frequency" else 30
        latents[modality] = torch.randn(2, 2, 1, latent_frames, 8)
        availability[modality] = torch.ones(2, 2, dtype=torch.bool)
        quality[modality] = torch.ones(2, 2)
        channel_mask[modality] = torch.ones(2, 2, 1, dtype=torch.bool)
    return write_latent_cache(
        tmp_path / "cache",
        clean_latents=latents,
        availability_mask=availability,
        quality_mask=quality,
        channel_mask=channel_mask,
        epoch_index=torch.tensor([[0, 1], [0, 1]]),
        night_position=torch.tensor([[0.0, 1.0], [0.0, 1.0]]),
        metadata_rows=[
            {"id": "cache-train", "split": "train"},
            {"id": "cache-val", "split": "val"},
        ],
        latent_frames_per_epoch={"high_frequency": 60, "low_frequency": 30},
        patches_per_epoch=6,
        modalities=CANONICAL_MODALITIES,
    )


def _write_config(
    tmp_path: Path,
    preset_path: Path,
    autoencoder_ckpt: Path,
    *,
    validation_examples: dict | None = None,
) -> Path:
    diffusion = {
        "latent_dim": 8,
        "latent_frames_per_epoch": {"high_frequency": 60, "low_frequency": 30},
        "patches_per_epoch": 6,
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
            "channel_position": True,
            "patch_position": True,
            "sleep_night_position": True,
            "availability": True,
            "quality": True,
        },
        "task_attention_mask": "directional",
        "auxiliary_restoration_token": True,
        "condition_dropout": 0.15,
    }
    training = {
        "phase": 1,
        "batch_size": 1,
        "lr": 0.0001,
        "weight_decay": 0.01,
        "max_epochs": 1,
        "gradient_clip_val": 1.0,
        "task_mix": {"restoration": 1.0},
        "condition_counts": [1],
        "replay": {"enabled": False},
    }
    if validation_examples is not None:
        training["validation"] = {"examples": validation_examples}
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
        "diffusion": diffusion,
        "training": training,
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
    assert loader.dataset.corruption_name is None


def test_train_diffusion_dataloader_uses_val_split(tmp_path: Path):
    preset_path = _write_synthetic_preset(tmp_path)
    autoencoder_ckpt = _write_autoencoder_checkpoint(tmp_path)
    config_path = _write_config(tmp_path, preset_path, autoencoder_ckpt)

    loader = build_dataloader(load_sleep2wave_config(config_path), num_workers=0, seed=123, split="val")

    assert [sample.metadata["split"] for sample in loader.dataset.data] == ["val", "val"]


def test_train_diffusion_dataloader_buckets_mixed_availability(tmp_path: Path):
    preset_path = _write_synthetic_preset(tmp_path)
    autoencoder_ckpt = _write_autoencoder_checkpoint(tmp_path)
    with preset_path.open("rb") as f:
        base_sample = pickle.load(f)[0]
    samples = []
    for sample_id, available in (("hf", ["eeg", "ecg"]), ("low", ["spo2", "ibi"])):
        payload = dict(base_sample.payload)
        payload["available_channels"] = available
        payload["canonical_channel_map"] = {modality: modality for modality in available}
        samples.append(
            SampleIndex(
                id=sample_id,
                path=base_sample.path,
                start=base_sample.start,
                end=base_sample.end,
                payload=payload,
                metadata={**base_sample.metadata, "split": "train"},
            )
        )
    with preset_path.open("wb") as f:
        pickle.dump(samples, f)
    config_path = _write_config(tmp_path, preset_path, autoencoder_ckpt)
    payload = yaml.safe_load(config_path.read_text())
    payload["training"]["phase"] = 2
    payload["training"]["batch_size"] = 2
    payload["training"]["task_mix"] = {"translation": 1.0}
    config_path.write_text(yaml.safe_dump(payload))

    loader = build_dataloader(load_sleep2wave_config(config_path), num_workers=0, seed=123)

    assert isinstance(loader.batch_sampler, AvailableChannelsBucketBatchSampler)
    task_sampler = Sleep2WaveTaskSampler(phase=2, task_mix={"translation": 1.0}, seed=123)
    for batch in loader:
        common_available = [
            modality
            for modality in CANONICAL_MODALITIES
            if torch.as_tensor(batch["availability_mask"][modality]).any(dim=1).all()
        ]
        assert len(common_available) >= 2
        task_sampler.sample(batch["availability_mask"])


def test_train_diffusion_loads_cache_only_translation_with_bucket_sampler(tmp_path: Path):
    preset_path = _write_synthetic_preset(tmp_path)
    autoencoder_ckpt = _write_autoencoder_checkpoint(tmp_path)
    cache_path = _write_latent_cache(tmp_path)
    config_path = _write_config(tmp_path, preset_path, autoencoder_ckpt)
    payload = yaml.safe_load(config_path.read_text())
    del payload["diffusion"]["autoencoder_checkpoint"]
    payload["diffusion"]["latent_cache_path"] = str(cache_path)
    payload["training"]["phase"] = 2
    payload["training"]["task_mix"] = {"translation": 1.0}
    config_path.write_text(yaml.safe_dump(payload))

    loader = build_dataloader(load_sleep2wave_config(config_path), num_workers=0, seed=123)
    batch = next(iter(loader))

    assert isinstance(loader.batch_sampler, AvailableChannelsBucketBatchSampler)
    assert batch["clean_latents"]["eeg"].shape == (1, 2, 1, 60, 8)
    assert batch["channel_mask"]["eeg"].shape == (1, 2, 1)


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


def test_train_diffusion_registers_step_lr_monitor(tmp_path: Path, monkeypatch):
    import sleep2wave.train_diffusion as train_diffusion

    preset_path = _write_synthetic_preset(tmp_path)
    autoencoder_ckpt = _write_autoencoder_checkpoint(tmp_path)
    config_path = _write_config(tmp_path, preset_path, autoencoder_ckpt)
    trainer_kwargs = {}
    lr_monitor = object()

    class DummyTrainer:
        def __init__(self, **kwargs):
            trainer_kwargs.update(kwargs)

        def fit(self, *args, **kwargs):
            pass

        def save_checkpoint(self, path):
            Path(path).write_text("checkpoint")

    monkeypatch.setattr(train_diffusion, "WandbLogger", lambda *args, **kwargs: object())
    monkeypatch.setattr(train_diffusion, "ModelCheckpoint", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        train_diffusion,
        "LearningRateMonitor",
        lambda *args, **kwargs: lr_monitor if kwargs == {"logging_interval": "step"} else None,
    )
    monkeypatch.setattr(train_diffusion.pl, "Trainer", DummyTrainer)

    train_diffusion.main(
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
    assert trainer_kwargs["limit_val_batches"] == len(CANONICAL_MODALITIES)


def test_diffusion_validation_batch_limit_scales_by_modality_and_task(tmp_path: Path):
    preset_path = _write_synthetic_preset(tmp_path)
    autoencoder_ckpt = _write_autoencoder_checkpoint(tmp_path)
    config_path = _write_config(
        tmp_path,
        preset_path,
        autoencoder_ckpt,
        validation_examples={"num_examples": 1, "modalities": ["eeg", "spo2"]},
    )
    payload = yaml.safe_load(config_path.read_text())
    payload["training"]["phase"] = 3
    payload["training"]["task_mix"] = {"restoration": 1.0, "translation": 1.0, "two_condition": 1.0}
    payload["training"]["validation"]["max_batches_per_modality"] = 3
    config_path.write_text(yaml.safe_dump(payload))

    assert _limit_val_batches(load_sleep2wave_config(config_path)) == 18


def test_diffusion_validation_step_rotates_configured_modalities(tmp_path: Path, monkeypatch):
    preset_path = _write_synthetic_preset(tmp_path)
    autoencoder_ckpt = _write_autoencoder_checkpoint(tmp_path)
    config_path = _write_config(
        tmp_path,
        preset_path,
        autoencoder_ckpt,
        validation_examples={"num_examples": 1, "modalities": ["eeg", "spo2"]},
    )
    payload = yaml.safe_load(config_path.read_text())
    payload["training"]["phase"] = 2
    payload["training"]["task_mix"] = {"translation": 1.0}
    payload["training"]["validation"]["max_batches_per_modality"] = 1
    config_path.write_text(yaml.safe_dump(payload))
    config = load_sleep2wave_config(config_path)
    loader = build_dataloader(config, num_workers=0, seed=123, split="val")
    batch = next(iter(loader))
    model = Sleep2WaveDiffusionLightning(config, seed=123)
    logged_scalars = {}

    def fake_lightning_log(name, value, **kwargs):
        logged_scalars[name] = (value, kwargs)

    monkeypatch.setattr(model, "log", fake_lightning_log)
    monkeypatch.setattr(wandb, "run", None)

    losses = model.validation_step(batch, 0)

    assert losses is not None
    assert {name for name in logged_scalars if name.startswith("val_") and name.endswith("_mse")} == {"val_eeg_mse"}
    logged_scalars.clear()

    losses = model.validation_step(batch, 1)

    assert losses is not None
    assert {name for name in logged_scalars if name.startswith("val_") and name.endswith("_mse")} == {"val_spo2_mse"}


def test_diffusion_step_accepts_padded_two_channel_batch(tmp_path: Path):
    preset_path = _write_synthetic_preset(tmp_path, channel_counts={"eeg": 2})
    autoencoder_ckpt = _write_autoencoder_checkpoint(tmp_path)
    config_path = _write_config(tmp_path, preset_path, autoencoder_ckpt)
    config = load_sleep2wave_config(config_path)
    loader = build_dataloader(config, num_workers=0, seed=123, split="train")
    batch = next(iter(loader))
    assert batch["channel_mask"]["eeg"].shape[-1] == 2
    assert batch["channel_mask"]["belt"].shape[-1] == 1
    batch["channel_mask"]["eeg"][:, :, 1] = False
    for modality in CANONICAL_MODALITIES:
        if modality != "eeg":
            batch["availability_mask"][modality][:] = False
    model = Sleep2WaveDiffusionLightning(config, seed=123)

    losses, _task_family, _task = model._compute_step_losses(batch, apply_condition_dropout=True)

    assert torch.isfinite(losses["loss"])


def test_diffusion_validation_task_selection_rotates_families_and_falls_back(tmp_path: Path):
    preset_path = _write_synthetic_preset(tmp_path)
    autoencoder_ckpt = _write_autoencoder_checkpoint(tmp_path)
    config_path = _write_config(
        tmp_path,
        preset_path,
        autoencoder_ckpt,
        validation_examples={"num_examples": 1, "modalities": ["eeg", "spo2"]},
    )
    payload = yaml.safe_load(config_path.read_text())
    payload["training"]["phase"] = 3
    payload["training"]["task_mix"] = {"restoration": 1.0, "translation": 1.0, "two_condition": 1.0}
    payload["training"]["validation"]["max_batches_per_modality"] = 1
    config_path.write_text(yaml.safe_dump(payload))
    config = load_sleep2wave_config(config_path)
    loader = build_dataloader(config, num_workers=0, seed=123, split="val")
    batch = next(iter(loader))
    model = Sleep2WaveDiffusionLightning(config, seed=123)

    expected = [
        ("restoration", "eeg"),
        ("restoration", "spo2"),
        ("translation", "eeg"),
        ("translation", "spo2"),
        ("two_condition", "eeg"),
        ("two_condition", "spo2"),
    ]
    for batch_idx, (expected_family, expected_target) in enumerate(expected):
        selected = model._validation_task_for_batch(batch, batch_idx)
        assert selected is not None
        task_family, task = selected
        assert task_family == expected_family
        assert expected_target in task.target_modalities

    batch["availability_mask"]["eeg"] = torch.zeros_like(batch["availability_mask"]["eeg"])
    selected = model._validation_task_for_batch(batch, 0)
    assert selected is not None
    _task_family, task = selected
    assert task.target_modalities == ("spo2",)

    for modality in CANONICAL_MODALITIES:
        batch["availability_mask"][modality] = torch.zeros_like(batch["availability_mask"][modality])
    batch["availability_mask"]["ecg"] = torch.ones_like(batch["availability_mask"]["ecg"])

    assert model._validation_task_for_batch(batch, 0) is None


def test_diffusion_validation_step_logs_task_examples(tmp_path: Path, monkeypatch):
    preset_path = _write_synthetic_preset(tmp_path)
    autoencoder_ckpt = _write_autoencoder_checkpoint(tmp_path)
    config_path = _write_config(
        tmp_path,
        preset_path,
        autoencoder_ckpt,
        validation_examples={"num_examples": 1, "modalities": ["eeg"]},
    )
    payload = yaml.safe_load(config_path.read_text())
    payload["training"]["phase"] = 3
    payload["training"]["task_mix"] = {"restoration": 1.0, "translation": 1.0, "two_condition": 1.0}
    config_path.write_text(yaml.safe_dump(payload))
    config = load_sleep2wave_config(config_path)
    loader = build_dataloader(config, num_workers=0, seed=123, split="val")
    batch = next(iter(loader))
    model = Sleep2WaveDiffusionLightning(config, seed=123)
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
    monkeypatch.setattr(diffusion_lightning, "render_waveform_example_plot", lambda *args, **kwargs: plt.figure())

    losses = model.validation_step(batch, 0)

    assert "loss" in losses
    assert "val_loss" in logged_scalars
    assert {name for name in logged_scalars if name.startswith("val_") and name.endswith("_mse")} == {"val_eeg_mse"}
    assert len(logged_payloads) == 1
    payload, commit = logged_payloads[0]
    assert commit is False
    assert set(payload) == {
        "val_diffusion_examples/restoration/eeg_sample_0",
        "val_diffusion_examples/translation/eeg_sample_0",
        "val_diffusion_examples/two_condition/eeg_sample_0",
    }


def test_diffusion_validation_step_supports_cache_only_translation(tmp_path: Path, monkeypatch):
    preset_path = _write_synthetic_preset(tmp_path)
    autoencoder_ckpt = _write_autoencoder_checkpoint(tmp_path)
    cache_path = _write_latent_cache(tmp_path)
    config_path = _write_config(tmp_path, preset_path, autoencoder_ckpt)
    payload = yaml.safe_load(config_path.read_text())
    del payload["diffusion"]["autoencoder_checkpoint"]
    payload["diffusion"]["latent_cache_path"] = str(cache_path)
    payload["training"]["phase"] = 2
    payload["training"]["task_mix"] = {"translation": 1.0}
    config_path.write_text(yaml.safe_dump(payload))
    model = Sleep2WaveDiffusionLightning(load_sleep2wave_config(config_path))
    batch = next(iter(build_dataloader(load_sleep2wave_config(config_path), num_workers=0, seed=123, split="val")))

    losses = model.validation_step(batch, 0)

    assert "loss" in losses


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


def test_condition_dropout_moves_partial_full_dropped_conditions_to_targets(tmp_path: Path):
    preset_path = _write_synthetic_preset(tmp_path)
    autoencoder_ckpt = _write_autoencoder_checkpoint(tmp_path)
    config_path = _write_config(tmp_path, preset_path, autoencoder_ckpt)
    payload = yaml.safe_load(config_path.read_text())
    payload["diffusion"]["condition_dropout"] = 1.0
    config_path.write_text(yaml.safe_dump(payload))
    module = Sleep2WaveDiffusionLightning(load_sleep2wave_config(config_path))
    task = build_generation_task(
        "partial_full",
        condition_modalities=["eeg", "ecg"],
        target_modalities=["spo2"],
    )

    dropped = module._apply_condition_dropout(task)

    assert len(dropped.condition_modalities) == 1
    assert set(dropped.condition_modalities).issubset({"eeg", "ecg"})
    assert set(dropped.condition_modalities).isdisjoint(dropped.target_modalities)
    assert set(dropped.condition_modalities + dropped.target_modalities) == {"eeg", "ecg", "spo2"}


def test_task_corruption_is_applied_inside_diffusion_module(tmp_path: Path):
    preset_path = _write_synthetic_preset(tmp_path)
    autoencoder_ckpt = _write_autoencoder_checkpoint(tmp_path)
    config_path = _write_config(tmp_path, preset_path, autoencoder_ckpt)
    payload = yaml.safe_load(config_path.read_text())
    payload["training"]["corruptions"] = {
        "restoration": {"default": {"name": "gaussian_noise", "kwargs": {"std": 1.0}}},
    }
    config_path.write_text(yaml.safe_dump(payload))
    module = Sleep2WaveDiffusionLightning(load_sleep2wave_config(config_path))
    task = build_generation_task(
        "restoration",
        condition_modalities=["eeg"],
        target_modalities=["eeg"],
        auxiliary_restoration_token=True,
    )
    observed = {"eeg": torch.zeros(1, 2, 1, MODALITY_SPECS["eeg"].frames_per_epoch)}

    corrupted = module._apply_task_corruption(observed, task)

    assert not torch.equal(corrupted["eeg"], observed["eeg"])


def test_task_corruption_samples_weighted_choices_inside_diffusion_module(tmp_path: Path):
    preset_path = _write_synthetic_preset(tmp_path)
    autoencoder_ckpt = _write_autoencoder_checkpoint(tmp_path)
    config_path = _write_config(tmp_path, preset_path, autoencoder_ckpt)
    payload = yaml.safe_load(config_path.read_text())
    payload["training"]["corruptions"] = {
        "restoration": {
            "default": {
                "choices": [
                    {"weight": 0.5, "name": "amplitude_attenuation", "kwargs": {"factor": 1.0}},
                    {"weight": 0.5, "name": "amplitude_attenuation", "kwargs": {"factor": 0.0}},
                ]
            }
        },
    }
    config_path.write_text(yaml.safe_dump(payload))
    module = Sleep2WaveDiffusionLightning(load_sleep2wave_config(config_path))
    spec = module.config_bundle.training.corruptions.restoration.default
    task = build_generation_task(
        "restoration",
        condition_modalities=["eeg"],
        target_modalities=["eeg"],
        auxiliary_restoration_token=True,
    )
    observed = {"eeg": torch.ones(1, 2, 1, MODALITY_SPECS["eeg"].frames_per_epoch)}

    corrupted = module._apply_task_corruption(observed, task)

    assert {spec.select(seed=seed).kwargs["factor"] for seed in range(10)} == {0.0, 1.0}
    assert torch.equal(corrupted["eeg"], torch.zeros_like(observed["eeg"]))


def test_task_corruption_keeps_restoration_auxiliary_conditions_clean(tmp_path: Path):
    preset_path = _write_synthetic_preset(tmp_path)
    autoencoder_ckpt = _write_autoencoder_checkpoint(tmp_path)
    config_path = _write_config(tmp_path, preset_path, autoencoder_ckpt)
    payload = yaml.safe_load(config_path.read_text())
    payload["training"]["corruptions"] = {
        "restoration": {"default": {"name": "gaussian_noise", "kwargs": {"std": 1.0}}},
    }
    config_path.write_text(yaml.safe_dump(payload))
    module = Sleep2WaveDiffusionLightning(load_sleep2wave_config(config_path))
    task = build_generation_task(
        "restoration",
        condition_modalities=["eeg", "ecg"],
        target_modalities=["eeg"],
        auxiliary_restoration_token=True,
    )
    observed = {
        "eeg": torch.zeros(1, 2, 1, MODALITY_SPECS["eeg"].frames_per_epoch),
        "ecg": torch.zeros(1, 2, 1, MODALITY_SPECS["ecg"].frames_per_epoch),
    }

    corrupted = module._apply_task_corruption(observed, task)

    assert not torch.equal(corrupted["eeg"], observed["eeg"])
    assert torch.equal(corrupted["ecg"], observed["ecg"])


def test_phase_checkpoint_loads_diffusion_model_weights(tmp_path: Path):
    preset_path = _write_synthetic_preset(tmp_path)
    autoencoder_ckpt = _write_autoencoder_checkpoint(tmp_path)
    config_path = _write_config(tmp_path, preset_path, autoencoder_ckpt)
    module = Sleep2WaveDiffusionLightning(load_sleep2wave_config(config_path))
    checkpoint_path = tmp_path / "phase.ckpt"
    torch.save(
        {"state_dict": {f"model.{key}": value for key, value in module.model.state_dict().items()}},
        checkpoint_path,
    )

    loaded = _load_phase_checkpoint(module.model, checkpoint_path)

    assert loaded == len(module.model.state_dict())
