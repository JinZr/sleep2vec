from __future__ import annotations

from pathlib import Path
import pickle
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from sleep2wave.data.derivations import derive_record_channels, plan_derivation_jobs
from sleep2wave.data.generative_dataset import IndexColumnConfig
from sleep2wave.data.modalities import CANONICAL_MODALITIES, MODALITY_SPECS
from sleep2wave.preprocess.build_sleep2wave_presets import build_sleep2wave_presets
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


def test_derivation_planning_rejects_subject_split_leakage(tmp_path: Path):
    df = pd.DataFrame(
        [
            {"path": "a.npz", "split": "train", "subject_id": "s1", "night_id": "n1"},
            {"path": "b.npz", "split": "test", "subject_id": "s1", "night_id": "n2"},
        ]
    )

    with pytest.raises(ValueError, match="Subjects appear in multiple splits"):
        plan_derivation_jobs(df, output_dir=tmp_path)


def test_derivation_planning_rejects_missing_subject_id(tmp_path: Path):
    df = pd.DataFrame(
        [
            {"path": "a.npz", "split": "train", "subject_id": None, "night_id": "n1"},
            {"path": "b.npz", "split": "test", "subject_id": "s2", "night_id": "n1"},
        ]
    )

    with pytest.raises(ValueError, match="missing subject_id values"):
        plan_derivation_jobs(df, output_dir=tmp_path)


def test_derivation_planning_is_per_record_and_split_safe(tmp_path: Path):
    df = pd.DataFrame(
        [
            {"path": "a.npz", "split": "train", "subject_id": "s1", "night_id": "n1"},
            {"path": "b.npz", "split": "test", "subject_id": "s2", "night_id": "n1"},
        ]
    )

    jobs = plan_derivation_jobs(df, output_dir=tmp_path)

    assert len(jobs) == 2
    assert jobs[0].subject_id == "s1"
    assert jobs[0].split == "train"
    assert jobs[1].subject_id == "s2"
    assert jobs[1].split == "test"


def test_derive_sleep2wave_channels_writes_ibi_resp_and_quality_masks(tmp_path: Path):
    sample_path = tmp_path / "sample.npz"
    seconds = 60
    ecg_rate = MODALITY_SPECS["ecg"].sample_rate_hz
    resp_rate = MODALITY_SPECS["belt"].sample_rate_hz
    ecg = np.zeros(seconds * ecg_rate, dtype=np.float32)
    ecg[::ecg_rate] = 5.0
    belt = np.sin(np.linspace(0.0, 12.0, seconds * resp_rate, dtype=np.float32))
    np.savez(sample_path, ecg=ecg, belt=belt)
    output_path = tmp_path / "derived.npz"

    derive_record_channels(input_path=sample_path, output_path=output_path, derive=["ibi", "resp"])

    with np.load(output_path) as derived:
        assert derived["ibi"].shape[0] == seconds * MODALITY_SPECS["ibi"].sample_rate_hz
        assert derived["resp"].shape[0] == seconds * MODALITY_SPECS["resp"].sample_rate_hz
        assert derived["ibi_quality_mask"].shape == (seconds // 30,)
        assert derived["resp_quality_mask"].shape == (seconds // 30,)
        assert derived["ibi_quality_mask"].dtype == np.bool_
        assert derived["resp_quality_mask"].dtype == np.bool_


@pytest.mark.parametrize(
    "module_name",
    [
        "sleep2wave.preprocess.build_sleep2wave_presets",
        "sleep2wave.preprocess.derive_sleep2wave_channels",
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
