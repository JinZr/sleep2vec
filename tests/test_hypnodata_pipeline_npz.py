import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from hypnodata.config import load_config
from hypnodata.edf import read_edf_signal
from hypnodata.pipeline import run_pipeline
from preprocess.watchpat_zzp_to_edf import SignalSpec, StudyMetadata, write_edf_manual


def _write_edf(path: Path, *, labels: list[str] | None = None) -> Path:
    labels = labels or ["EEG C3", "SpO2"]
    signals = []
    for label in labels:
        if label == "SpO2":
            samples = np.arange(90, 100, dtype=np.int16)
            sfreq = 1
            unit = "%"
        else:
            samples = np.arange(100, dtype=np.int16)
            sfreq = 10
            unit = "uV"
        signals.append(SignalSpec(label=label, samples=samples, sample_frequency=sfreq, dimension=unit))
    write_edf_manual(
        str(path),
        signals,
        StudyMetadata(source_path=str(path), patient_code="toy"),
    )
    return path


def _write_config(tmp_path: Path, index: Path, *, missing_required: bool = False, include_spo2: bool = True) -> Path:
    eeg_label = "Missing EEG" if missing_required else "EEG C3"
    signals = {
        "eeg": {
            "kind": "eeg",
            "required": True,
            "target_sfreq": 5,
            "target_unit": "uV",
            "candidates": [eeg_label],
            "scale": 2.0,
            "polarity": "invert",
        }
    }
    if include_spo2:
        signals["spo2"] = {
            "kind": "spo2",
            "required": False,
            "target_sfreq": 1,
            "target_unit": "%",
            "candidates": ["SpO2"],
        }
    path = tmp_path / "hypnodata.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "center": "toy",
                "record_discovery": {
                    "type": "csv",
                    "index": str(index),
                    "file_columns": {"edf": "path"},
                    "record_id_column": "record_id",
                    "metadata_columns": ["age"],
                },
                "backend": {"type": "npz"},
                "signals": signals,
            }
        )
    )
    return path


def _write_filter_config(tmp_path: Path, index: Path) -> Path:
    path = tmp_path / "hypnodata_filter.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "center": "toy",
                "record_discovery": {
                    "type": "csv",
                    "index": str(index),
                    "file_columns": {"edf": "path"},
                    "record_id_column": "record_id",
                },
                "backend": {"type": "npz"},
                "signals": {
                    "eeg": {
                        "kind": "eeg",
                        "required": True,
                        "target_sfreq": 5,
                        "target_unit": "uV",
                        "candidates": ["EEG C3"],
                        "preprocess": [
                            {"type": "filter", "method": "bessel", "order": 2, "highcut": 2.0},
                        ],
                    }
                },
            }
        )
    )
    return path


def _write_unit_config(tmp_path: Path, index: Path) -> Path:
    path = tmp_path / "hypnodata_unit.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "center": "toy",
                "record_discovery": {
                    "type": "csv",
                    "index": str(index),
                    "file_columns": {"edf": "path"},
                    "record_id_column": "record_id",
                },
                "backend": {"type": "npz"},
                "signals": {
                    "eeg": {
                        "kind": "eeg",
                        "required": True,
                        "target_sfreq": 1,
                        "target_unit": "uV",
                        "candidates": ["EEG C3"],
                    }
                },
            }
        )
    )
    return path


def _write_index(tmp_path: Path, edf_path: Path) -> Path:
    index = tmp_path / "records.csv"
    pd.DataFrame(
        [
            {
                "record_id": "night1",
                "path": str(edf_path),
                "source": "toy_source",
                "split": "train",
                "subject_id": "sub1",
                "session_id": "ses1",
                "age": 50,
            }
        ]
    ).to_csv(index, index=False)
    return index


def test_pipeline_writes_npz_and_manifests(tmp_path: Path):
    edf_path = _write_edf(tmp_path / "record.edf")
    index = _write_index(tmp_path, edf_path)
    config = load_config(_write_config(tmp_path, index))
    output_dir = tmp_path / "out"

    run_pipeline(config, output_dir=output_dir)

    npz_path = output_dir / "backends" / "npz" / "records" / "night1.npz"
    assert npz_path.exists()
    with np.load(npz_path) as npz:
        assert sorted(npz.files) == ["eeg", "spo2"]
        assert npz["eeg"].shape == (50,)
        assert npz["spo2"].shape == (10,)
        np.testing.assert_array_equal(npz["spo2"], np.arange(90, 100, dtype=np.float32))

    record_manifest = pd.read_csv(output_dir / "manifest" / "record_manifest.csv")
    assert set(
        [
            "record_id",
            "center",
            "source",
            "subject_id",
            "session_id",
            "split",
            "path",
            "duration",
            "backend",
            "qc_status",
            "eeg_mask",
            "spo2_mask",
        ]
    ).issubset(record_manifest.columns)
    assert "duration_sec" not in record_manifest.columns
    assert record_manifest.loc[0, "path"] == str(npz_path)
    assert record_manifest.loc[0, "duration"] == 10.0
    assert record_manifest.loc[0, "eeg_mask"] == 1
    assert record_manifest.loc[0, "spo2_mask"] == 1

    signal_manifest = pd.read_csv(output_dir / "manifest" / "signal_manifest.csv")
    eeg = signal_manifest[signal_manifest["canonical_channel"] == "eeg"].iloc[0]
    assert eeg["raw_label"] == "EEG C3"
    assert eeg["target_sfreq"] == 5
    assert eeg["scale_applied"] == 2.0
    assert eeg["polarity_applied"] == -1
    assert "resample:10->5" in eeg["preprocess_steps"]
    assert eeg["output_key"] == "eeg"


def test_edf_reader_preserves_header_voltage_unit(tmp_path: Path):
    edf_path = _write_edf(tmp_path / "record.edf", labels=["EEG C3"])

    values = read_edf_signal(edf_path, "EEG C3", raw_unit="uV")

    np.testing.assert_allclose(values[:5], np.arange(5, dtype=np.float32))


def test_pipeline_converts_raw_unit_to_target_unit(tmp_path: Path):
    edf_path = tmp_path / "record.edf"
    write_edf_manual(
        str(edf_path),
        [SignalSpec(label="EEG C3", samples=np.arange(1, 11, dtype=np.int16), sample_frequency=1, dimension="mV")],
        StudyMetadata(source_path=str(edf_path), patient_code="toy"),
    )
    index = _write_index(tmp_path, edf_path)
    config = load_config(_write_unit_config(tmp_path, index))
    output_dir = tmp_path / "out"

    run_pipeline(config, output_dir=output_dir)

    with np.load(output_dir / "backends" / "npz" / "records" / "night1.npz") as npz:
        np.testing.assert_allclose(npz["eeg"], np.arange(1, 11, dtype=np.float32) * 1000.0)
    signal_manifest = pd.read_csv(output_dir / "manifest" / "signal_manifest.csv")
    eeg = signal_manifest[signal_manifest["canonical_channel"] == "eeg"].iloc[0]
    assert eeg["raw_unit"] == "mV"
    assert eeg["target_unit"] == "uV"
    assert "unit:mV->uV" in eeg["preprocess_steps"]


def test_edf_reader_uses_native_channel_sample_rate(tmp_path: Path):
    edf_path = _write_edf(tmp_path / "record.edf", labels=["EEG C3", "SpO2"])

    values = read_edf_signal(edf_path, "SpO2", raw_unit="%", raw_index=1)

    assert values.shape == (10,)
    np.testing.assert_array_equal(values, np.arange(90, 100, dtype=np.float32))


def test_edf_reader_uses_raw_index_for_duplicate_labels(tmp_path: Path):
    edf_path = tmp_path / "record.edf"
    write_edf_manual(
        str(edf_path),
        [
            SignalSpec(label="EEG C3", samples=np.arange(100, dtype=np.int16), sample_frequency=10, dimension="uV"),
            SignalSpec(
                label="EEG C3",
                samples=np.arange(100, 200, dtype=np.int16),
                sample_frequency=10,
                dimension="uV",
            ),
        ],
        StudyMetadata(source_path=str(edf_path), patient_code="toy"),
    )

    values = read_edf_signal(edf_path, "EEG C3", raw_unit="uV", raw_index=1)

    np.testing.assert_allclose(values[:5], np.arange(100, 105, dtype=np.float32))


def test_pipeline_passes_selected_raw_index_to_edf_reader(tmp_path: Path, monkeypatch):
    edf_path = _write_edf(tmp_path / "record.edf", labels=["EEG C3"])
    index = _write_index(tmp_path, edf_path)
    config = load_config(_write_config(tmp_path, index, include_spo2=False))
    raw_indices = []

    def fake_read_edf_signal(path, raw_label, raw_unit=None, *, raw_index=None):
        raw_indices.append(raw_index)
        return np.arange(100, dtype=np.float32)

    monkeypatch.setattr("hypnodata.pipeline.read_edf_signal", fake_read_edf_signal)

    run_pipeline(config, output_dir=tmp_path / "out")

    assert raw_indices == [0]


def test_pipeline_manifest_records_structured_preprocess_steps(tmp_path: Path):
    edf_path = _write_edf(tmp_path / "record.edf", labels=["EEG C3"])
    index = _write_index(tmp_path, edf_path)
    config = load_config(_write_filter_config(tmp_path, index))
    output_dir = tmp_path / "out"

    run_pipeline(config, output_dir=output_dir)

    signal_manifest = pd.read_csv(output_dir / "manifest" / "signal_manifest.csv")
    eeg = signal_manifest[signal_manifest["canonical_channel"] == "eeg"].iloc[0]
    assert "filter:bessel:lowpass:2Hz:order=2" in eeg["preprocess_steps"]
    assert "not_implemented" not in eeg["preprocess_steps"]


def test_optional_missing_channel_sets_mask_zero(tmp_path: Path):
    edf_path = _write_edf(tmp_path / "record.edf", labels=["EEG C3"])
    index = _write_index(tmp_path, edf_path)
    config = load_config(_write_config(tmp_path, index))
    output_dir = tmp_path / "out"

    run_pipeline(config, output_dir=output_dir)

    record_manifest = pd.read_csv(output_dir / "manifest" / "record_manifest.csv")
    assert record_manifest.loc[0, "eeg_mask"] == 1
    assert record_manifest.loc[0, "spo2_mask"] == 0
    signal_manifest = pd.read_csv(output_dir / "manifest" / "signal_manifest.csv")
    spo2 = signal_manifest[signal_manifest["canonical_channel"] == "spo2"].iloc[0]
    assert spo2["available"] == 0
    assert spo2["mask_column"] == "spo2_mask"


def test_required_missing_channel_fails_without_terminal_manifests(tmp_path: Path):
    edf_path = _write_edf(tmp_path / "record.edf")
    index = _write_index(tmp_path, edf_path)
    config = load_config(_write_config(tmp_path, index, missing_required=True))
    output_dir = tmp_path / "out"

    with pytest.raises(ValueError, match="Missing required channel"):
        run_pipeline(config, output_dir=output_dir)

    progress = json.loads((output_dir / "status" / "progress.json").read_text())
    assert progress["status"] == "failed"
    assert progress["failed_records"] == 1
    assert not (output_dir / "manifest" / "record_manifest.csv").exists()
    assert not (output_dir / "manifest" / "backend_manifest.json").exists()
    assert not (output_dir / "backends" / "npz" / "records" / "night1.npz").exists()


def test_pipeline_rejects_bdf_without_mne_fallback(tmp_path: Path):
    bdf_path = tmp_path / "record.bdf"
    bdf_path.write_bytes(b"")
    index = _write_index(tmp_path, bdf_path)
    config = load_config(_write_config(tmp_path, index, include_spo2=False))
    output_dir = tmp_path / "out"

    with pytest.raises(ValueError, match="BDF input is not supported"):
        run_pipeline(config, output_dir=output_dir)

    progress = json.loads((output_dir / "status" / "progress.json").read_text())
    assert progress["status"] == "failed"
    assert not (output_dir / "manifest" / "record_manifest.csv").exists()
    assert not (output_dir / "backends" / "npz" / "records" / "night1.npz").exists()


def test_run_raises_on_first_record_failure(tmp_path: Path):
    edf_path = _write_edf(tmp_path / "record.edf")
    index = _write_index(tmp_path, edf_path)
    config = load_config(_write_config(tmp_path, index, missing_required=True))

    with pytest.raises(ValueError, match="Missing required channel"):
        run_pipeline(config, output_dir=tmp_path / "out")
