from __future__ import annotations

from pathlib import Path
import pickle
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from sleep2wave.autoencoders.model import Sleep2WaveAutoencoder
from sleep2wave.data.default_dataset import SampleIndex
from sleep2wave.data.modalities import CANONICAL_MODALITIES, MODALITY_SPECS
from sleep2wave.diffusion.model import Sleep2WaveDiffusionTransformer
from sleep2wave.diffusion.tasks import build_generation_task
from sleep2wave.generate import (
    _activate_requested_generation_targets,
    _collect_generation_windows,
    _decode_generated_latents,
    _resolve_generation_data_source,
    _resolve_inference_corruption_specs,
    main,
)
from sleep2wave.generate_batch import run_batch_generation
from sleep2wave.generative.config import load_sleep2wave_config


def _write_synthetic_preset(tmp_path: Path, *, channel_counts: dict[str, int] | None = None) -> Path:
    npz_path = tmp_path / "synthetic.npz"
    payload = {}
    for index, modality in enumerate(CANONICAL_MODALITIES):
        frames = 2 * MODALITY_SPECS[modality].frames_per_epoch
        signal = np.linspace(0.0, 1.0 + index, frames, dtype=np.float32)
        channels = (channel_counts or {}).get(modality, 1)
        if channels == 1:
            payload[modality] = signal
        else:
            payload[modality] = np.stack([signal + channel for channel in range(channels)], axis=0)
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
        latent_frames_per_epoch={"high_frequency": 60, "low_frequency": 30},
        patches_per_epoch=6,
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
        },
        "inference": {
            "corruptions": {
                "restoration": {"default": {"name": "gaussian_noise", "kwargs": {"std": 0.05}}},
                "imputation": {"default": {"name": "contiguous_window_mask", "kwargs": {"window_frames": 120}}},
            }
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
    data_config = SimpleNamespace(
        backend="kaldi",
        preset_path=None,
        index=None,
        kaldi_data_root=str(tmp_path / "kaldi"),
        kaldi_manifest=str(tmp_path / "kaldi" / "manifest.csv"),
    )

    backend, preset_path, index, kaldi_data_root, kaldi_manifest = _resolve_generation_data_source(args, data_config)

    assert backend == "npz"
    assert preset_path is None
    assert index == tmp_path / "override.csv"
    assert kaldi_data_root is None
    assert kaldi_manifest is None


def test_generate_cli_data_source_uses_configured_kaldi_backend(tmp_path: Path):
    args = SimpleNamespace(preset_path=None, index=None)
    data_config = SimpleNamespace(
        backend="kaldi",
        preset_path=None,
        index=None,
        kaldi_data_root=str(tmp_path / "kaldi"),
        kaldi_manifest=str(tmp_path / "kaldi" / "manifest.csv"),
    )

    backend, preset_path, index, kaldi_data_root, kaldi_manifest = _resolve_generation_data_source(args, data_config)

    assert backend == "kaldi"
    assert preset_path is None
    assert index is None
    assert kaldi_data_root == str(tmp_path / "kaldi")
    assert kaldi_manifest == str(tmp_path / "kaldi" / "manifest.csv")


def test_generate_activates_unavailable_translation_targets_for_sampling():
    availability = {modality: torch.ones(2, 2, dtype=torch.bool) for modality in CANONICAL_MODALITIES}
    availability["eeg"] = torch.zeros(2, 2, dtype=torch.bool)
    task = build_generation_task("translation", condition_modalities=["ecg"], target_modalities=["eeg"])

    adjusted = _activate_requested_generation_targets(availability, task)

    assert adjusted["eeg"].tolist() == [[True, True], [True, True]]
    assert torch.equal(adjusted["ecg"], availability["ecg"])
    assert availability["eeg"].tolist() == [[False, False], [False, False]]


def test_generate_resolves_inference_corruption_from_config(tmp_path: Path):
    preset_path = _write_synthetic_preset(tmp_path)
    autoencoder_ckpt, _diffusion_ckpt = _write_checkpoints(tmp_path)
    config = load_sleep2wave_config(_write_config(tmp_path, preset_path, autoencoder_ckpt))
    task = build_generation_task(
        "imputation",
        condition_modalities=["eeg"],
        target_modalities=["eeg"],
        auxiliary_restoration_token=True,
    )

    specs = _resolve_inference_corruption_specs(
        config=config,
        args=SimpleNamespace(corruption_name=None, corruption_kwargs=None),
        task=task,
    )

    assert specs == {"eeg": ("contiguous_window_mask", {"window_frames": 120})}


def test_generate_imputation_corruption_specs_target_only_with_extra_conditions(tmp_path: Path):
    preset_path = _write_synthetic_preset(tmp_path)
    autoencoder_ckpt, _diffusion_ckpt = _write_checkpoints(tmp_path)
    config = load_sleep2wave_config(_write_config(tmp_path, preset_path, autoencoder_ckpt))
    task = build_generation_task(
        "imputation",
        condition_modalities=["eeg", "ecg"],
        target_modalities=["eeg"],
        auxiliary_restoration_token=True,
    )

    config_specs = _resolve_inference_corruption_specs(
        config=config,
        args=SimpleNamespace(corruption_name=None, corruption_kwargs=None),
        task=task,
    )
    cli_specs = _resolve_inference_corruption_specs(
        config=config,
        args=SimpleNamespace(corruption_name="gaussian_noise", corruption_kwargs='{"std": 0.2}'),
        task=task,
    )

    assert config_specs == {"eeg": ("contiguous_window_mask", {"window_frames": 120})}
    assert cli_specs == {"eeg": ("gaussian_noise", {"std": 0.2})}


def test_generate_resolves_inference_corruption_from_weighted_choices(tmp_path: Path):
    preset_path = _write_synthetic_preset(tmp_path)
    autoencoder_ckpt, _diffusion_ckpt = _write_checkpoints(tmp_path)
    config_path = _write_config(tmp_path, preset_path, autoencoder_ckpt)
    payload = yaml.safe_load(config_path.read_text())
    payload["inference"]["corruptions"]["imputation"]["default"] = {
        "choices": [
            {"weight": 0.5, "name": "contiguous_window_mask", "kwargs": {"window_frames": 120}},
            {"weight": 0.5, "name": "gaussian_noise", "kwargs": {"std": 0.2}},
        ]
    }
    config_path.write_text(yaml.safe_dump(payload))
    config = load_sleep2wave_config(config_path)
    task = build_generation_task(
        "imputation",
        condition_modalities=["eeg"],
        target_modalities=["eeg"],
        auxiliary_restoration_token=True,
    )

    specs = _resolve_inference_corruption_specs(
        config=config,
        args=SimpleNamespace(corruption_name=None, corruption_kwargs=None, seed=0),
        task=task,
    )

    assert specs == {"eeg": ("gaussian_noise", {"std": 0.2})}


def test_generate_cli_corruption_overrides_config(tmp_path: Path):
    preset_path = _write_synthetic_preset(tmp_path)
    autoencoder_ckpt, _diffusion_ckpt = _write_checkpoints(tmp_path)
    config_path = _write_config(tmp_path, preset_path, autoencoder_ckpt)
    payload = yaml.safe_load(config_path.read_text())
    payload["inference"]["corruptions"]["imputation"]["default"] = {
        "choices": [
            {"weight": 1.0, "name": "contiguous_window_mask", "kwargs": {"window_frames": 120}},
        ]
    }
    config_path.write_text(yaml.safe_dump(payload))
    config = load_sleep2wave_config(config_path)
    task = build_generation_task(
        "imputation",
        condition_modalities=["eeg"],
        target_modalities=["eeg"],
        auxiliary_restoration_token=True,
    )

    specs = _resolve_inference_corruption_specs(
        config=config,
        args=SimpleNamespace(corruption_name="gaussian_noise", corruption_kwargs='{"std": 0.2}'),
        task=task,
    )

    assert specs == {"eeg": ("gaussian_noise", {"std": 0.2})}


def test_decode_generated_latents_accepts_temporal_channel_latents():
    autoencoder = Sleep2WaveAutoencoder(latent_dim=8, modalities=["eeg"])
    latents = {"eeg": torch.randn(2, 1, 2, 1, 60, 8)}

    decoded = _decode_generated_latents(autoencoder, latents)

    assert decoded["eeg"].shape == (2, 1, 2, 1, 3840)


def test_generation_passes_patch_condition_availability(monkeypatch):
    captured = {}
    epoch_count = 2
    eeg_frames = MODALITY_SPECS["eeg"].frames_per_epoch
    batch = {
        "observed_signals": {
            modality: torch.zeros(1, epoch_count, 1, spec.frames_per_epoch) for modality, spec in MODALITY_SPECS.items()
        },
        "availability_mask": {
            modality: torch.ones(1, epoch_count, dtype=torch.bool) for modality in CANONICAL_MODALITIES
        },
        "quality_mask": {modality: torch.ones(1, epoch_count) for modality in CANONICAL_MODALITIES},
        "channel_mask": {
            modality: torch.ones(1, epoch_count, 1, dtype=torch.bool) for modality in CANONICAL_MODALITIES
        },
        "corruption_mask": {
            modality: torch.zeros(1, epoch_count, 1, spec.frames_per_epoch, dtype=torch.bool)
            for modality, spec in MODALITY_SPECS.items()
        },
        "epoch_index": torch.tensor([[0, 1]]),
        "night_position": torch.tensor([[0.0, 1.0]]),
        "metadata": {
            "id": ["row-0"],
            "path": ["night.npz"],
            "subject_id": ["subject"],
            "night_id": ["night"],
            "source": ["synthetic"],
            "split": ["train"],
        },
    }
    batch["corruption_mask"]["eeg"][:, 0, :, : eeg_frames // 6] = True

    class FakeDataset:
        def __init__(self, **_kwargs):
            pass

        def dataloader(self, **_kwargs):
            return [batch]

    class FakeAutoencoder:
        def __call__(self, _signals):
            return SimpleNamespace(latents={"eeg": torch.zeros(1, epoch_count, 1, 60, 8)})

        def decode_latents(self, latents):
            return {"eeg": torch.zeros(latents["eeg"].shape[0], epoch_count, 1, eeg_frames)}

    class FakeSampler:
        def sample(self, _model, **kwargs):
            captured["condition_availability_mask"] = kwargs["condition_availability_mask"]
            return SimpleNamespace(generated_latents={"eeg": torch.zeros(1, 1, epoch_count, 1, 60, 8)})

    monkeypatch.setattr("sleep2wave.data.generative_dataset.Sleep2WaveGenerativeDataset", FakeDataset)
    monkeypatch.setattr("sleep2wave.diffusion.samplers.build_sampler", lambda *_args, **_kwargs: FakeSampler())
    task = build_generation_task(
        "imputation",
        condition_modalities=["eeg"],
        target_modalities=["eeg"],
        auxiliary_restoration_token=True,
    )
    config = SimpleNamespace(
        data=SimpleNamespace(
            backend="npz",
            preset_path="preset.pkl",
            index=None,
            kaldi_data_root=None,
            kaldi_manifest=None,
            context_epochs=epoch_count,
        ),
        diffusion=SimpleNamespace(diffusion_steps=8, beta_schedule="cosine", patches_per_epoch=6),
        inference=None,
        modalities=SimpleNamespace(all=list(CANONICAL_MODALITIES)),
    )

    _collect_generation_windows(
        config=config,
        args=SimpleNamespace(
            preset_path=None,
            index=None,
            batch_size=1,
            stride_epochs=1,
            num_workers=0,
            seed=0,
            corruption_name=None,
            corruption_kwargs=None,
            condition_mask_npz=None,
        ),
        model=object(),
        autoencoder=FakeAutoencoder(),
        sampler_config=SimpleNamespace(),
        task=task,
        device=torch.device("cpu"),
    )

    condition_availability = captured["condition_availability_mask"]["eeg"]
    assert condition_availability.shape == (1, epoch_count, 1, 6)
    assert condition_availability[0, 0, 0].tolist() == [False, True, True, True, True, True]
    assert condition_availability[0, 1, 0].tolist() == [True, True, True, True, True, True]


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


def test_generate_smoke_preserves_two_target_channels(tmp_path: Path):
    preset_path = _write_synthetic_preset(tmp_path, channel_counts={"eeg": 2})
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
            "--output-dir",
            str(output_dir),
            "--batch-size",
            "1",
            "--device",
            "cpu",
        ]
    )

    generated = np.load(output_dir / "generated.npz")

    assert generated["generated/eeg"].shape == (1, 2, 2, 3840)


def test_generate_batch_groups_by_subject_night(tmp_path: Path, monkeypatch):
    preset_path = _write_synthetic_preset(tmp_path)
    with preset_path.open("rb") as f:
        first = pickle.load(f)[0]
    second = SampleIndex(
        id="synthetic-2",
        path=first.path,
        start=first.start,
        end=first.end,
        payload={**first.payload, "subject_id": "subject2", "night_id": "night2"},
        metadata={**first.metadata, "subject_id": "subject2", "night_id": "night2"},
    )
    with preset_path.open("wb") as f:
        pickle.dump([first, second], f)
    autoencoder_ckpt, diffusion_ckpt = _write_checkpoints(tmp_path)
    config_path = _write_config(tmp_path, preset_path, autoencoder_ckpt)
    calls = []

    def fake_run_generation(args):
        calls.append(args)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        return args.output_dir

    monkeypatch.setattr("sleep2wave.generate.run_generation", fake_run_generation)

    outputs = run_batch_generation(
        SimpleNamespace(
            config=config_path,
            diffusion_ckpt=diffusion_ckpt,
            autoencoder_ckpt=autoencoder_ckpt,
            preset_path=preset_path,
            index=None,
            task="translation",
            condition_modalities=["ecg"],
            target_modalities=["eeg"],
            corruption_name=None,
            corruption_kwargs=None,
            condition_mask_npz=None,
            num_samples=None,
            output_dir=tmp_path / "batch_out",
            stride_epochs=1,
            overlap_fusion="mean",
            batch_size=1,
            num_workers=0,
            device="cpu",
            seed=0,
        )
    )

    assert len(calls) == 2
    assert len(outputs) == 2
    assert all(call.preset_path.exists() for call in calls)
