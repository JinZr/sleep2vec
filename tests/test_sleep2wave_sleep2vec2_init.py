from __future__ import annotations

from argparse import Namespace
from pathlib import Path
import types

import pytest
import pytorch_lightning  # noqa: F401
import torch
import torch.nn as nn
import wandb  # noqa: F401
import yaml

from sleep2wave.generative.config import InitializationConfig
from sleep2wave.initialization.sleep2vec2 import load_sleep2vec2_initialization


class _TinyAutoencoderBranch(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(2, 2)
        self.decoder = nn.Linear(2, 2)


class _TinyInitTarget(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_projection = nn.Linear(2, 2)
        self.proj_head = nn.Linear(2, 2)
        self.tokenizer_mapping = nn.ModuleDict({"eeg": nn.Linear(2, 2)})
        self.modality_autoencoders = nn.ModuleDict({"eeg": _TinyAutoencoderBranch()})


def _config(
    checkpoint_path: Path,
    *,
    load_groups: dict[str, bool],
    strict_compatible: bool = True,
    require_any_loaded: bool = False,
) -> InitializationConfig:
    return InitializationConfig(
        sleep2vec2_checkpoint=str(checkpoint_path),
        strict_compatible=strict_compatible,
        require_any_loaded=require_any_loaded,
        load_groups=load_groups,
    )


def _save_checkpoint(path: Path, state: dict[str, torch.Tensor]) -> Path:
    torch.save({"state_dict": state}, path)
    return path


def test_sleep2vec2_init_loads_exact_shape_keys(tmp_path: Path):
    target = _TinyInitTarget()
    ckpt_path = _save_checkpoint(
        tmp_path / "init.ckpt",
        {"model.input_projection.weight": torch.full_like(target.input_projection.weight, 3.0)},
    )

    report = load_sleep2vec2_initialization(
        target,
        ckpt_path,
        _config(ckpt_path, load_groups={"diffusion_transformer": True}),
        target_groups={"diffusion_transformer"},
    )

    assert report.used_prefix == "model."
    assert report.loaded_groups == ["diffusion_transformer"]
    assert report.loaded_keys == ["input_projection.weight"]
    assert torch.equal(target.input_projection.weight, torch.full_like(target.input_projection.weight, 3.0))


def test_sleep2vec2_init_prefers_ema_model_prefix(tmp_path: Path):
    target = _TinyInitTarget()
    ckpt_path = _save_checkpoint(
        tmp_path / "init.ckpt",
        {
            "model.input_projection.weight": torch.full_like(target.input_projection.weight, 1.0),
            "ema_model.input_projection.weight": torch.full_like(target.input_projection.weight, 2.0),
        },
    )

    report = load_sleep2vec2_initialization(
        target,
        ckpt_path,
        _config(ckpt_path, load_groups={"diffusion_transformer": True}),
        target_groups={"diffusion_transformer"},
    )

    assert report.used_prefix == "ema_model."
    assert report.loaded_keys == ["input_projection.weight"]
    assert torch.equal(target.input_projection.weight, torch.full_like(target.input_projection.weight, 2.0))


def test_sleep2vec2_init_reports_disabled_group(tmp_path: Path):
    target = _TinyInitTarget()
    ckpt_path = _save_checkpoint(
        tmp_path / "init.ckpt",
        {"model.proj_head.weight": torch.full_like(target.proj_head.weight, 4.0)},
    )

    report = load_sleep2vec2_initialization(
        target,
        ckpt_path,
        _config(ckpt_path, load_groups={"projection": False}),
        target_groups={"projection"},
    )

    assert report.loaded_keys == []
    assert report.skipped_disabled_group == ["proj_head.weight"]


def test_sleep2vec2_init_reports_missing_target(tmp_path: Path):
    target = _TinyInitTarget()
    ckpt_path = _save_checkpoint(
        tmp_path / "init.ckpt",
        {"model.input_projection.extra": torch.ones(2, 2)},
    )

    report = load_sleep2vec2_initialization(
        target,
        ckpt_path,
        _config(ckpt_path, load_groups={"diffusion_transformer": True}),
        target_groups={"diffusion_transformer"},
    )

    assert report.loaded_keys == []
    assert report.skipped_missing_target == ["input_projection.extra"]


def test_sleep2vec2_init_shape_mismatch_raises_when_strict(tmp_path: Path):
    target = _TinyInitTarget()
    ckpt_path = _save_checkpoint(
        tmp_path / "init.ckpt",
        {"model.input_projection.weight": torch.ones(3, 2)},
    )

    with pytest.raises(ValueError, match="shape mismatch"):
        load_sleep2vec2_initialization(
            target,
            ckpt_path,
            _config(ckpt_path, load_groups={"diffusion_transformer": True}, strict_compatible=True),
            target_groups={"diffusion_transformer"},
        )


def test_sleep2vec2_init_shape_mismatch_skips_when_not_strict(tmp_path: Path):
    target = _TinyInitTarget()
    ckpt_path = _save_checkpoint(
        tmp_path / "init.ckpt",
        {"model.input_projection.weight": torch.ones(3, 2)},
    )

    report = load_sleep2vec2_initialization(
        target,
        ckpt_path,
        _config(ckpt_path, load_groups={"diffusion_transformer": True}, strict_compatible=False),
        target_groups={"diffusion_transformer"},
    )

    assert report.loaded_keys == []
    assert report.skipped_shape_mismatch == ["input_projection.weight"]


def test_sleep2vec2_init_require_any_loaded_raises_for_zero_load(tmp_path: Path):
    target = _TinyInitTarget()
    ckpt_path = _save_checkpoint(tmp_path / "init.ckpt", {"model.unknown.weight": torch.ones(2, 2)})

    with pytest.raises(ValueError, match="did not load any compatible"):
        load_sleep2vec2_initialization(
            target,
            ckpt_path,
            _config(
                ckpt_path,
                load_groups={"diffusion_transformer": True},
                require_any_loaded=True,
            ),
            target_groups={"diffusion_transformer"},
        )


def test_sleep2vec2_init_reports_unknown_prefix_and_group(tmp_path: Path):
    target = _TinyInitTarget()
    ckpt_path = _save_checkpoint(
        tmp_path / "init.ckpt",
        {
            "model.unknown.weight": torch.ones(2, 2),
            "student.input_projection.weight": torch.ones(2, 2),
        },
    )

    report = load_sleep2vec2_initialization(
        target,
        ckpt_path,
        _config(ckpt_path, load_groups={"diffusion_transformer": True}),
        target_groups={"diffusion_transformer"},
    )

    assert report.skipped_unknown_group == ["unknown.weight"]
    assert report.skipped_unknown_prefix == ["student.input_projection.weight"]


def _write_autoencoder_config(tmp_path: Path, init_ckpt: Path) -> Path:
    payload = {
        "recipe": "sleep2wave",
        "stage": "autoencoder",
        "data": {"preset_path": str(tmp_path / "preset.pkl"), "context_epochs": 1},
        "modalities": _modalities_block(),
        "autoencoder": {
            "latent_dim": 8,
            "encoder_type": "conv1d_epoch",
            "decoder_type": "convtranspose1d_epoch",
            "one_latent_per_epoch": True,
            "modality_specific": True,
            "losses": {"waveform_l1_weight": 1.0, "waveform_l2_weight": 0.0, "spectral_weight": 0.0},
        },
        "training": {
            "phase": 0,
            "batch_size": 1,
            "lr": 0.0001,
            "weight_decay": 0.01,
            "max_epochs": 1,
            "gradient_clip_val": 1.0,
        },
        "initialization": {
            "sleep2vec2_checkpoint": str(init_ckpt),
            "strict_compatible": True,
            "require_any_loaded": False,
            "load_groups": {"autoencoder_encoders": True},
        },
        "export": {"output_dir": str(tmp_path / "outputs")},
    }
    path = tmp_path / "autoencoder.yaml"
    path.write_text(yaml.safe_dump(payload))
    return path


def _write_diffusion_config(tmp_path: Path, init_ckpt: Path) -> Path:
    payload = {
        "recipe": "sleep2wave",
        "stage": "diffusion",
        "data": {"preset_path": str(tmp_path / "preset.pkl"), "context_epochs": 1},
        "modalities": _modalities_block(),
        "diffusion": {
            "latent_dim": 8,
            "autoencoder_checkpoint": str(tmp_path / "autoencoder.ckpt"),
            "transformer": {"hidden_size": 8, "num_layers": 1, "num_heads": 2, "mlp_ratio": 2},
            "diffusion_steps": 8,
            "beta_schedule": "cosine",
            "prediction_type": "epsilon",
            "context_epochs": 1,
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
            "condition_dropout": 0.0,
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
        "initialization": {
            "sleep2vec2_checkpoint": str(init_ckpt),
            "strict_compatible": True,
            "require_any_loaded": False,
            "load_groups": {"diffusion_transformer": True},
        },
        "export": {"output_dir": str(tmp_path / "outputs")},
    }
    path = tmp_path / "diffusion.yaml"
    path.write_text(yaml.safe_dump(payload))
    return path


def _modalities_block() -> dict:
    from sleep2wave.data.modalities import CANONICAL_MODALITIES, MODALITY_SPECS

    return {
        "epoch_sec": 30,
        "all": list(CANONICAL_MODALITIES),
        "high_frequency": ["eeg", "eog", "emg", "ecg"],
        "low_frequency": ["airflow", "belt", "spo2", "ibi", "resp"],
        "sample_rates": {modality: MODALITY_SPECS[modality].sample_rate_hz for modality in CANONICAL_MODALITIES},
        "frames_per_epoch": {modality: MODALITY_SPECS[modality].frames_per_epoch for modality in CANONICAL_MODALITIES},
    }


class _DummyLogger:
    def __init__(self, *args, **kwargs) -> None:
        pass


class _DummyCheckpoint:
    def __init__(self, *args, **kwargs) -> None:
        pass


class _DummyTrainer:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def fit(self, *args, **kwargs) -> None:
        pass

    def save_checkpoint(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text("checkpoint")


class _DummyLightning:
    def __init__(self, *args, **kwargs) -> None:
        self.model = _TinyInitTarget()


def test_train_autoencoder_calls_initializer_when_configured(tmp_path: Path, monkeypatch):
    import sleep2wave.train_autoencoder as train_autoencoder

    init_ckpt = _save_checkpoint(
        tmp_path / "init.ckpt",
        {"model.modality_autoencoders.eeg.encoder.weight": torch.ones(2, 2)},
    )
    config_path = _write_autoencoder_config(tmp_path, init_ckpt)
    calls = []

    monkeypatch.setattr(train_autoencoder, "build_dataloader", lambda *args, **kwargs: object())
    monkeypatch.setattr(train_autoencoder, "Sleep2WaveAutoencoderLightning", _DummyLightning)
    monkeypatch.setattr(train_autoencoder, "WandbLogger", _DummyLogger)
    monkeypatch.setattr(train_autoencoder, "ModelCheckpoint", _DummyCheckpoint)
    monkeypatch.setattr(train_autoencoder.pl, "Trainer", _DummyTrainer)
    monkeypatch.setattr(
        train_autoencoder,
        "load_sleep2vec2_initialization",
        lambda *args, **kwargs: calls.append((args, kwargs)) or types.SimpleNamespace(loaded_keys=["x"]),
    )

    train_autoencoder.train_autoencoder(
        Namespace(
            config=config_path,
            version_name="init",
            accelerator="cpu",
            devices="1",
            precision=32,
            max_steps=0,
            num_workers=0,
        )
    )

    assert len(calls) == 1
    assert calls[0][1]["target_groups"] == {"autoencoder_encoders"}


def test_train_diffusion_calls_initializer_when_configured(tmp_path: Path, monkeypatch):
    import sleep2wave.train_diffusion as train_diffusion

    init_ckpt = _save_checkpoint(tmp_path / "init.ckpt", {"model.input_projection.weight": torch.ones(2, 2)})
    config_path = _write_diffusion_config(tmp_path, init_ckpt)
    calls = []

    monkeypatch.setattr(train_diffusion, "build_dataloader", lambda *args, **kwargs: object())
    monkeypatch.setattr(train_diffusion, "Sleep2WaveDiffusionLightning", _DummyLightning)
    monkeypatch.setattr(train_diffusion, "WandbLogger", _DummyLogger)
    monkeypatch.setattr(train_diffusion, "ModelCheckpoint", _DummyCheckpoint)
    monkeypatch.setattr(train_diffusion.pl, "Trainer", _DummyTrainer)
    monkeypatch.setattr(
        train_diffusion,
        "load_sleep2vec2_initialization",
        lambda *args, **kwargs: calls.append((args, kwargs)) or types.SimpleNamespace(loaded_keys=["x"]),
    )

    train_diffusion.train_diffusion(
        Namespace(
            config=config_path,
            version_name="init",
            accelerator="cpu",
            devices="1",
            precision=32,
            max_steps=0,
            num_workers=0,
            seed=0,
        )
    )

    assert len(calls) == 1
    assert calls[0][1]["target_groups"] == {"diffusion_transformer"}
