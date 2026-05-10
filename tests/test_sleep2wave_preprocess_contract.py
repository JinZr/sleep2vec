from __future__ import annotations

from pathlib import Path
import pickle
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from sleep2wave.data.generative_dataset import IndexColumnConfig
from sleep2wave.data.modalities import CANONICAL_MODALITIES, MODALITY_SPECS
from sleep2wave.preprocess.build_sleep2wave_presets import build_sleep2wave_presets
from sleep2wave.preprocess.convert_npz_to_kaldi import convert, parse_args
from sleep2wave.preprocess.validate_sleep2wave_index import validate_sleep2wave_index


def _write_npz(path: Path, modalities=CANONICAL_MODALITIES, *, epochs: int = 2) -> None:
    arrays = {}
    for modality in modalities:
        spec = MODALITY_SPECS[modality]
        arrays[modality] = np.zeros(epochs * spec.frames_per_epoch, dtype=np.float32)
    np.savez(path, **arrays)


def _index_frame(path: Path) -> pd.DataFrame:
    row = {
        "path": str(path),
        "duration": 60,
        "split": "train",
        "subject_id": "s1",
        "night_id": "n1",
        "source": "synthetic",
    }
    row.update({f"{modality}_mask": 1 for modality in CANONICAL_MODALITIES})
    return pd.DataFrame([row])


def _write_sleep2wave_config(path: Path, *, context_epochs: int = 2) -> Path:
    payload = {
        "recipe": "sleep2wave",
        "stage": "autoencoder",
        "data": {"preset_path": "data/sleep2wave_tiny_preset.pkl", "context_epochs": context_epochs},
        "modalities": {
            "epoch_sec": 30,
            "all": list(CANONICAL_MODALITIES),
            "high_frequency": ["eeg", "eog", "emg", "ecg"],
            "low_frequency": ["airflow", "belt", "spo2", "ibi", "resp"],
            "sample_rates": {
                modality: MODALITY_SPECS[modality].sample_rate_hz for modality in CANONICAL_MODALITIES
            },
            "frames_per_epoch": {
                modality: MODALITY_SPECS[modality].frames_per_epoch for modality in CANONICAL_MODALITIES
            },
        },
        "autoencoder": {
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
        },
        "training": {
            "phase": 0,
            "batch_size": 1,
            "lr": 0.0001,
            "weight_decay": 0.01,
            "max_epochs": 1,
            "gradient_clip_val": 1.0,
        },
        "export": {"output_dir": str(path.parent / "outputs")},
    }
    path.write_text(yaml.safe_dump(payload))
    return path


def test_build_sleep2wave_presets_writes_schema_versioned_sample_indices(tmp_path: Path):
    index_path = tmp_path / "index.csv"
    output_path = tmp_path / "preset.pkl"
    sample_path = tmp_path / "sample.npz"
    _write_npz(sample_path)
    _index_frame(sample_path).to_csv(index_path, index=False)

    samples = build_sleep2wave_presets(
        index_path=index_path,
        output_path=output_path,
        split=["train"],
        context_epochs=2,
        stride_epochs=2,
        columns=None,
    )

    with output_path.open("rb") as f:
        loaded = pickle.load(f)

    assert len(samples) == 1
    assert len(loaded) == 1
    payload = loaded[0].payload
    assert payload["sleep2wave_schema_version"] == 1
    assert payload["available_channels"] == list(CANONICAL_MODALITIES)
    assert payload["sample_rates"]["eeg"] == 128
    assert payload["frames_per_epoch"]["spo2"] == MODALITY_SPECS["spo2"].frames_per_epoch
    assert payload["subject_id"] == "s1"
    assert payload["night_id"] == "n1"


def test_build_sleep2wave_presets_accepts_sleep2vec_style_index_and_implicit_masks(tmp_path: Path):
    index_path = tmp_path / "index.csv"
    sample_path = tmp_path / "sample.npz"
    _write_npz(sample_path, modalities=("eeg", "ecg", "spo2"))
    pd.DataFrame(
        [
            {
                "path": str(sample_path),
                "duration": 60,
                "split": "train",
                "age": 69.0,
                "sex": "Female",
            }
        ]
    ).to_csv(index_path, index=False)

    samples = build_sleep2wave_presets(
        index_path=index_path,
        output_path=tmp_path / "preset.pkl",
        split=["train"],
        context_epochs=2,
        stride_epochs=2,
        columns=None,
        dry_run=True,
    )

    assert len(samples) == 1
    assert samples[0].metadata["subject_id"] == str(sample_path)
    assert samples[0].metadata["night_id"] == str(sample_path)
    assert samples[0].metadata["age"] == 69.0
    assert samples[0].metadata["sex"] == "Female"
    assert samples[0].payload["available_channels"] == ["eeg", "ecg", "spo2"]


def test_build_sleep2wave_presets_accepts_explicit_identity_columns(tmp_path: Path):
    index_path = tmp_path / "index.csv"
    sample_path = tmp_path / "sample.npz"
    _write_npz(sample_path, modalities=("eeg",))
    pd.DataFrame(
        [
            {
                "path": str(sample_path),
                "session_id": "mesa-sleep-0001",
                "duration": 60,
                "split": "train",
            }
        ]
    ).to_csv(index_path, index=False)

    samples = build_sleep2wave_presets(
        index_path=index_path,
        output_path=tmp_path / "preset.pkl",
        split=["train"],
        context_epochs=2,
        stride_epochs=2,
        columns=IndexColumnConfig(subject_id_col="session_id", night_id_col="session_id"),
        dry_run=True,
    )

    assert len(samples) == 1
    assert samples[0].metadata["subject_id"] == "mesa-sleep-0001"
    assert samples[0].metadata["night_id"] == "mesa-sleep-0001"


def test_build_sleep2wave_presets_resolves_alias_keys(tmp_path: Path):
    index_path = tmp_path / "index.csv"
    sample_path = tmp_path / "sample.npz"
    np.savez(
        sample_path,
        eeg_original=np.zeros(2 * MODALITY_SPECS["eeg"].frames_per_epoch, dtype=np.float32),
        resp_nasal_original=np.zeros(2 * MODALITY_SPECS["airflow"].frames_per_epoch, dtype=np.float32),
    )
    pd.DataFrame([{"path": str(sample_path), "duration": 60, "split": "train"}]).to_csv(index_path, index=False)

    samples = build_sleep2wave_presets(
        index_path=index_path,
        output_path=tmp_path / "preset.pkl",
        split=["train"],
        context_epochs=2,
        stride_epochs=2,
        columns=None,
        dry_run=True,
    )

    assert len(samples) == 1
    assert samples[0].payload["available_channels"] == ["eeg", "airflow"]
    assert samples[0].payload["canonical_channel_map"] == {
        "eeg": "eeg_original",
        "airflow": "resp_nasal_original",
    }


def test_convert_npz_to_kaldi_roundtrip_preserves_waveform_batch(tmp_path: Path):
    pytest.importorskip("kaldi_native_io")
    sample_path = tmp_path / "sample.npz"
    eeg = np.arange(2 * MODALITY_SPECS["eeg"].frames_per_epoch, dtype=np.float32)
    ecg_base = np.arange(2 * MODALITY_SPECS["ecg"].frames_per_epoch, dtype=np.float32)
    ecg = np.stack([ecg_base, ecg_base + 1000.0], axis=0)
    np.savez(sample_path, eeg=eeg, ecg=ecg, eeg_quality_mask=np.array([1.0, 0.0], dtype=np.float32))
    index_path = tmp_path / "index.csv"
    frame = _index_frame(sample_path)
    for modality in CANONICAL_MODALITIES:
        frame[f"{modality}_mask"] = 1 if modality in {"eeg", "ecg"} else 0
    frame.to_csv(index_path, index=False)
    config_path = _write_sleep2wave_config(tmp_path / "sleep2wave.yaml")
    output_dir = tmp_path / "kaldi"

    manifest_path, manifest_json_path = convert(
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

    assert manifest_path.exists()
    manifest_json = yaml.safe_load(manifest_json_path.read_text())
    assert manifest_json["backend"] == "kaldi_native_io"

    from sleep2wave.data.generative_dataset import Sleep2WaveGenerativeDataset

    dataset = Sleep2WaveGenerativeDataset(
        backend="kaldi",
        kaldi_data_root=output_dir,
        kaldi_manifest=manifest_path,
        split="train",
        context_epochs=2,
    )
    batch = next(iter(dataset.dataloader(batch_size=1)))

    assert batch["clean_signals"]["eeg"].shape == (1, 2, 1, MODALITY_SPECS["eeg"].frames_per_epoch)
    assert batch["clean_signals"]["ecg"].shape == (1, 2, 2, MODALITY_SPECS["ecg"].frames_per_epoch)
    assert torch.equal(
        batch["clean_signals"]["eeg"][0, :, 0].reshape(-1),
        torch.as_tensor(eeg),
    )
    assert batch["quality_mask"]["eeg"].tolist() == [[1.0, 0.0]]
    assert batch["availability_mask"]["eog"].tolist() == [[False, False]]
    assert batch["metadata"]["subject_id"] == ["s1"]
    assert batch["metadata"]["night_id"] == ["n1"]


def test_convert_npz_to_kaldi_rejects_present_malformed_modality_without_mask(tmp_path: Path):
    pytest.importorskip("kaldi_native_io")
    sample_path = tmp_path / "sample.npz"
    np.savez(sample_path, eeg=np.zeros((2, MODALITY_SPECS["eeg"].frames_per_epoch), dtype=np.float32))
    index_path = tmp_path / "index.csv"
    pd.DataFrame(
        [
            {
                "path": str(sample_path),
                "duration": 60,
                "split": "train",
                "subject_id": "s1",
                "night_id": "n1",
                "source": "synthetic",
            }
        ]
    ).to_csv(index_path, index=False)
    config_path = _write_sleep2wave_config(tmp_path / "sleep2wave.yaml")

    with pytest.raises(ValueError, match="Channel 'eeg' must be"):
        convert(
            parse_args(
                [
                    "--index",
                    str(index_path),
                    "--config",
                    str(config_path),
                    "--output-dir",
                    str(tmp_path / "kaldi"),
                ]
            )
        )


def test_validate_sleep2wave_index_accepts_sleep2vec_style_index_and_implicit_masks(tmp_path: Path):
    index_path = tmp_path / "index.csv"
    pd.DataFrame(
        [
            {
                "path": "/data/ywx/hsp_9_channels/sub-S0001111189075/ses-1.npz",
                "duration": 60,
                "split": "train",
            }
        ]
    ).to_csv(index_path, index=False)

    validate_sleep2wave_index(index_path)


def test_validate_sleep2wave_index_accepts_valid_index(tmp_path: Path):
    index_path = tmp_path / "index.csv"
    _index_frame(tmp_path / "sample.npz").to_csv(index_path, index=False)

    validate_sleep2wave_index(index_path)


def test_validate_sleep2wave_index_accepts_partial_modality_masks(tmp_path: Path):
    index_path = tmp_path / "index.csv"
    frame = _index_frame(tmp_path / "sample.npz").drop(columns=["resp_mask"])
    frame.to_csv(index_path, index=False)

    validate_sleep2wave_index(index_path)


def test_validate_sleep2wave_index_accepts_subjects_in_multiple_splits(tmp_path: Path):
    index_path = tmp_path / "index.csv"
    first = _index_frame(tmp_path / "a.npz").iloc[0].to_dict()
    second = _index_frame(tmp_path / "b.npz").iloc[0].to_dict()
    second["split"] = "test"
    pd.DataFrame([first, second]).to_csv(index_path, index=False)

    validate_sleep2wave_index(index_path)


def test_validate_sleep2wave_index_accepts_missing_subject_id(tmp_path: Path):
    index_path = tmp_path / "index.csv"
    frame = _index_frame(tmp_path / "sample.npz")
    frame.loc[0, "subject_id"] = None
    frame.to_csv(index_path, index=False)

    validate_sleep2wave_index(index_path)


def test_build_sleep2wave_presets_accepts_subjects_in_multiple_splits(tmp_path: Path):
    index_path = tmp_path / "index.csv"
    first_path = tmp_path / "a.npz"
    second_path = tmp_path / "b.npz"
    _write_npz(first_path)
    _write_npz(second_path)
    first = _index_frame(first_path).iloc[0].to_dict()
    second = _index_frame(second_path).iloc[0].to_dict()
    second["split"] = "test"
    pd.DataFrame([first, second]).to_csv(index_path, index=False)

    samples = build_sleep2wave_presets(
        index_path=index_path,
        output_path=tmp_path / "preset.pkl",
        split=["train"],
        context_epochs=2,
        stride_epochs=2,
        columns=None,
    )

    assert len(samples) == 1


def test_build_sleep2wave_presets_supports_num_workers(tmp_path: Path):
    index_path = tmp_path / "index.csv"
    rows = []
    for idx in range(2):
        sample_path = tmp_path / f"sample{idx}.npz"
        _write_npz(sample_path, modalities=("eeg",))
        row = _index_frame(sample_path).iloc[0].to_dict()
        row["subject_id"] = f"s{idx}"
        row["night_id"] = f"n{idx}"
        rows.append(row)
    pd.DataFrame(rows).to_csv(index_path, index=False)

    samples = build_sleep2wave_presets(
        index_path=index_path,
        output_path=tmp_path / "preset.pkl",
        split=["train"],
        context_epochs=2,
        stride_epochs=2,
        columns=None,
        num_workers=2,
        dry_run=True,
    )

    assert [sample.metadata["subject_id"] for sample in samples] == ["s0", "s1"]


def test_build_sleep2wave_presets_accepts_missing_subject_id(tmp_path: Path):
    index_path = tmp_path / "index.csv"
    sample_path = tmp_path / "sample.npz"
    _write_npz(sample_path)
    frame = _index_frame(sample_path)
    frame.loc[0, "subject_id"] = None
    frame.to_csv(index_path, index=False)

    samples = build_sleep2wave_presets(
        index_path=index_path,
        output_path=tmp_path / "preset.pkl",
        split=["train"],
        context_epochs=2,
        stride_epochs=2,
        columns=None,
    )

    assert len(samples) == 1


def test_validate_sleep2wave_index_rejects_zero_available_modality_rows(tmp_path: Path):
    index_path = tmp_path / "index.csv"
    frame = _index_frame(tmp_path / "sample.npz")
    for modality in CANONICAL_MODALITIES:
        frame[f"{modality}_mask"] = 0
    frame.to_csv(index_path, index=False)

    with pytest.raises(ValueError, match="no available modalities"):
        validate_sleep2wave_index(index_path)


@pytest.mark.parametrize(
    "module_name",
    [
        "sleep2wave.preprocess.build_sleep2wave_presets",
        "sleep2wave.preprocess.convert_npz_to_kaldi",
        "sleep2wave.preprocess.validate_sleep2wave_index",
    ],
)
def test_sleep2wave_preprocess_clis_support_help(module_name: str):
    result = subprocess.run(
        [sys.executable, "-m", module_name, "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "usage:" in result.stdout
