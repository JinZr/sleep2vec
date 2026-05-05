from __future__ import annotations

from pathlib import Path
import pickle
import subprocess
import sys

import pandas as pd
import pytest

from sleep2wave.data.derivations import plan_derivation_jobs
from sleep2wave.data.modalities import CANONICAL_MODALITIES, MODALITY_SPECS
from sleep2wave.preprocess.build_sleep2wave_presets import build_sleep2wave_presets
from sleep2wave.preprocess.validate_sleep2wave_index import validate_sleep2wave_index


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
    _index_frame(tmp_path / "sample.npz").to_csv(index_path, index=False)

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


def test_validate_sleep2wave_index_accepts_valid_index(tmp_path: Path):
    index_path = tmp_path / "index.csv"
    _index_frame(tmp_path / "sample.npz").to_csv(index_path, index=False)

    validate_sleep2wave_index(index_path)


def test_validate_sleep2wave_index_rejects_missing_modality_mask(tmp_path: Path):
    index_path = tmp_path / "index.csv"
    frame = _index_frame(tmp_path / "sample.npz").drop(columns=["resp_mask"])
    frame.to_csv(index_path, index=False)

    with pytest.raises(ValueError, match="Missing Sleep2Wave modality mask columns"):
        validate_sleep2wave_index(index_path)


def test_validate_sleep2wave_index_rejects_subject_split_leakage(tmp_path: Path):
    index_path = tmp_path / "index.csv"
    first = _index_frame(tmp_path / "a.npz").iloc[0].to_dict()
    second = _index_frame(tmp_path / "b.npz").iloc[0].to_dict()
    second["split"] = "test"
    pd.DataFrame([first, second]).to_csv(index_path, index=False)

    with pytest.raises(ValueError, match="Subjects appear in multiple splits"):
        validate_sleep2wave_index(index_path)


def test_validate_sleep2wave_index_rejects_missing_subject_id(tmp_path: Path):
    index_path = tmp_path / "index.csv"
    frame = _index_frame(tmp_path / "sample.npz")
    frame.loc[0, "subject_id"] = None
    frame.to_csv(index_path, index=False)

    with pytest.raises(ValueError, match="missing subject_id values"):
        validate_sleep2wave_index(index_path)


def test_build_sleep2wave_presets_rejects_subject_split_leakage(tmp_path: Path):
    index_path = tmp_path / "index.csv"
    first = _index_frame(tmp_path / "a.npz").iloc[0].to_dict()
    second = _index_frame(tmp_path / "b.npz").iloc[0].to_dict()
    second["split"] = "test"
    pd.DataFrame([first, second]).to_csv(index_path, index=False)

    with pytest.raises(ValueError, match="Subjects appear in multiple splits"):
        build_sleep2wave_presets(
            index_path=index_path,
            output_path=tmp_path / "preset.pkl",
            split=["train"],
            context_epochs=2,
            stride_epochs=2,
            columns=None,
        )


def test_build_sleep2wave_presets_rejects_missing_subject_id(tmp_path: Path):
    index_path = tmp_path / "index.csv"
    frame = _index_frame(tmp_path / "sample.npz")
    frame.loc[0, "subject_id"] = None
    frame.to_csv(index_path, index=False)

    with pytest.raises(ValueError, match="missing subject_id values"):
        build_sleep2wave_presets(
            index_path=index_path,
            output_path=tmp_path / "preset.pkl",
            split=["train"],
            context_epochs=2,
            stride_epochs=2,
            columns=None,
        )


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
