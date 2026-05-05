from __future__ import annotations

from pathlib import Path
import pickle
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

torch = pytest.importorskip("torch")

from sleep2wave.autoencoders.model import Sleep2WaveAutoencoder
from sleep2wave.data.default_dataset import SampleIndex
from sleep2wave.data.modalities import CANONICAL_MODALITIES, MODALITY_SPECS
from sleep2wave.diffusion.model import Sleep2WaveDiffusionTransformer
from sleep2wave.diffusion.tasks import build_generation_task
from sleep2wave.generate import _activate_requested_generation_targets, _resolve_generation_data_source, main


def _write_synthetic_preset(tmp_path: Path) -> Path:
    npz_path = tmp_path / "synthetic.npz"
    payload = {}
    for index, modality in enumerate(CANONICAL_MODALITIES):
        frames = 2 * MODALITY_SPECS[modality].frames_per_epoch
        payload[modality] = np.linspace(0.0, 1.0 + index, frames, dtype=np.float32)
    np.savez(npz_path, **payload)

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
    preset_path = tmp_path / "preset.pkl"
    with preset_path.open("wb") as f:
        pickle.dump([sample], f)
    return preset_path


def _modalities_payload() -> dict:
    return {
        "epoch_sec": 30,
        "all": list(CANONICAL_MODALITIES),
        "high_frequency": ["eeg", "eog", "emg", "ecg"],
        "low_frequency": ["airflow", "belt", "spo2", "ibi", "resp"],
        "sample_rates": {modality: MODALITY_SPECS[modality].sample_rate_hz for modality in CANONICAL_MODALITIES},
        "frames_per_epoch": {modality: MODALITY_SPECS[modality].frames_per_epoch for modality in CANONICAL_MODALITIES},
    }


def _write_checkpoints(tmp_path: Path) -> tuple[Path, Path]:
    autoencoder = Sleep2WaveAutoencoder(latent_dim=8)
    autoencoder_ckpt = tmp_path / "autoencoder.ckpt"
    torch.save(
        {"state_dict": {f"model.{key}": value for key, value in autoencoder.state_dict().items()}},
        autoencoder_ckpt,
    )

    diffusion = Sleep2WaveDiffusionTransformer(
        latent_dim=8,
        hidden_size=8,
        num_layers=1,
        num_heads=2,
        mlp_ratio=2,
        diffusion_steps=8,
        context_epochs=2,
    )
    diffusion_ckpt = tmp_path / "diffusion.ckpt"
    torch.save({"state_dict": {f"model.{key}": value for key, value in diffusion.state_dict().items()}}, diffusion_ckpt)
    return autoencoder_ckpt, diffusion_ckpt


def _write_config(tmp_path: Path, preset_path: Path, autoencoder_ckpt: Path) -> Path:
    payload = {
        "recipe": "sleep2wave",
        "stage": "inference",
        "data": {"preset_path": str(preset_path), "context_epochs": 2},
        "modalities": _modalities_payload(),
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
        "sampler": {"name": "ddim", "steps": 2, "eta": 0.0, "num_samples": 1},
        "export": {"output_dir": str(tmp_path / "generated")},
    }
    config_path = tmp_path / "generate.yaml"
    config_path.write_text(yaml.safe_dump(payload))
    return config_path


def test_generate_cli_rejects_missing_condition_modalities(tmp_path: Path):
    with pytest.raises(SystemExit):
        main(
            [
                "--config",
                str(tmp_path / "config.yaml"),
                "--diffusion-ckpt",
                str(tmp_path / "diffusion.ckpt"),
                "--task",
                "translation",
                "--target-modalities",
                "eeg",
            ]
        )


def test_generate_cli_rejects_unknown_modality(tmp_path: Path):
    preset_path = _write_synthetic_preset(tmp_path)
    autoencoder_ckpt, _diffusion_ckpt = _write_checkpoints(tmp_path)
    config_path = _write_config(tmp_path, preset_path, autoencoder_ckpt)

    with pytest.raises(ValueError, match="canonical modality names"):
        main(
            [
                "--config",
                str(config_path),
                "--diffusion-ckpt",
                str(tmp_path / "missing.ckpt"),
                "--task",
                "translation",
                "--condition-modalities",
                "bad_modality",
                "--target-modalities",
                "eeg",
            ]
        )


def test_generate_cli_rejects_invalid_condition_target_overlap(tmp_path: Path):
    preset_path = _write_synthetic_preset(tmp_path)
    autoencoder_ckpt, _diffusion_ckpt = _write_checkpoints(tmp_path)
    config_path = _write_config(tmp_path, preset_path, autoencoder_ckpt)

    with pytest.raises(ValueError, match="requires disjoint condition and target modalities"):
        main(
            [
                "--config",
                str(config_path),
                "--diffusion-ckpt",
                str(tmp_path / "missing.ckpt"),
                "--task",
                "translation",
                "--condition-modalities",
                "eeg",
                "--target-modalities",
                "eeg",
            ]
        )


def test_generate_cli_data_source_override_replaces_config_source(tmp_path: Path):
    args = SimpleNamespace(preset_path=None, index=tmp_path / "override.csv")
    data_config = SimpleNamespace(preset_path=str(tmp_path / "config.pkl"), index=None)

    preset_path, index = _resolve_generation_data_source(args, data_config)

    assert preset_path is None
    assert index == tmp_path / "override.csv"


def test_generate_activates_unavailable_translation_targets_for_sampling():
    availability = {modality: torch.ones(2, 2, dtype=torch.bool) for modality in CANONICAL_MODALITIES}
    availability["eeg"] = torch.zeros(2, 2, dtype=torch.bool)
    task = build_generation_task("translation", condition_modalities=["ecg"], target_modalities=["eeg"])

    adjusted = _activate_requested_generation_targets(availability, task)

    assert adjusted["eeg"].tolist() == [[True, True], [True, True]]
    assert torch.equal(adjusted["ecg"], availability["ecg"])
    assert availability["eeg"].tolist() == [[False, False], [False, False]]


def test_generate_smoke_writes_required_artifacts(tmp_path: Path):
    preset_path = _write_synthetic_preset(tmp_path)
    autoencoder_ckpt, diffusion_ckpt = _write_checkpoints(tmp_path)
    config_path = _write_config(tmp_path, preset_path, autoencoder_ckpt)
    output_dir = tmp_path / "out"

    main(
        [
            "--config",
            str(config_path),
            "--diffusion-ckpt",
            str(diffusion_ckpt),
            "--task",
            "translation",
            "--condition-modalities",
            "ecg",
            "--target-modalities",
            "eeg",
            "--num-samples",
            "2",
            "--output-dir",
            str(output_dir),
            "--batch-size",
            "1",
            "--device",
            "cpu",
        ]
    )

    generated = np.load(output_dir / "generated.npz")
    uncertainty = np.load(output_dir / "uncertainty.npz")
    masks = np.load(output_dir / "masks.npz")

    assert (output_dir / "manifest.json").exists()
    assert (output_dir / "config.yaml").exists()
    assert (output_dir / "cli_args.yaml").exists()
    assert (output_dir / "metadata.jsonl").exists()
    assert generated["generated/eeg"].shape == (2, 2, 1, 3840)
    assert uncertainty["sample_count/eeg"].tolist() == [2]
    assert masks["condition/ecg"].tolist() == [True, True]
    assert masks["target/eeg"].tolist() == [True, True]
