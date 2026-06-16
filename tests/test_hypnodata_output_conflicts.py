from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hypnodata.config import load_config
from hypnodata.pipeline import run_pipeline
from tests.hypnodata_test_helpers import write_hypnodata_config, write_index, write_tiny_edf


def _one_record_config(tmp_path: Path, *, label: str = "EEG C3", scale: float = 1.0) -> Path:
    edf_path = write_tiny_edf(tmp_path / "night1.edf", duration_sec=10)
    index = write_index(
        tmp_path,
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
        ],
    )
    return write_hypnodata_config(tmp_path, index, label=label, scale=scale, target_sfreq=10)


def _two_record_config(tmp_path: Path, *, scale: float = 1.0) -> Path:
    edf1 = write_tiny_edf(tmp_path / "night1.edf", duration_sec=10)
    edf2 = write_tiny_edf(tmp_path / "night2.edf", duration_sec=10)
    index = write_index(
        tmp_path,
        [
            {
                "record_id": "night1",
                "path": str(edf1),
                "source": "toy_source",
                "split": "train",
                "subject_id": "sub1",
                "session_id": "ses1",
                "age": 50,
            },
            {
                "record_id": "night2",
                "path": str(edf2),
                "source": "toy_source",
                "split": "test",
                "subject_id": "sub2",
                "session_id": "ses2",
                "age": 51,
            },
        ],
    )
    return write_hypnodata_config(tmp_path, index, scale=scale, target_sfreq=10)


def test_hypnodata_refuses_existing_npz_without_rewriting_manifests(tmp_path: Path):
    config = load_config(_one_record_config(tmp_path))
    output_dir = tmp_path / "out"
    run_pipeline(config, output_dir=output_dir)
    manifest_dir = output_dir / "manifest"
    record_manifest = (manifest_dir / "record_manifest.csv").read_text()
    signal_manifest = (manifest_dir / "signal_manifest.csv").read_text()

    with pytest.raises(FileExistsError, match="Output NPZ already exists"):
        run_pipeline(config, output_dir=output_dir)

    assert (manifest_dir / "record_manifest.csv").read_text() == record_manifest
    assert (manifest_dir / "signal_manifest.csv").read_text() == signal_manifest


def test_hypnodata_preflights_existing_npz_before_workers(tmp_path: Path, monkeypatch):
    config = load_config(_two_record_config(tmp_path))
    output_dir = tmp_path / "out"
    existing_npz = output_dir / "backends" / "npz" / "records" / "night2.npz"
    existing_npz.parent.mkdir(parents=True)
    np.savez(existing_npz, eeg=np.ones(100, dtype=np.float32))
    calls = []

    def fail_if_called(*args, **kwargs):
        calls.append(args)
        raise AssertionError("workers should not start")

    monkeypatch.setattr("hypnodata.pipeline._process_record", fail_if_called)

    with pytest.raises(FileExistsError, match="night2"):
        run_pipeline(config, output_dir=output_dir, num_workers=2)

    assert calls == []
    assert not (output_dir / "backends" / "npz" / "records" / "night1.npz").exists()
    assert not (output_dir / "manifest" / "record_manifest.csv").exists()


def test_hypnodata_dry_run_ignores_existing_npz_conflicts(tmp_path: Path):
    output_dir = tmp_path / "out"
    config = load_config(_two_record_config(tmp_path))
    npz_dir = output_dir / "backends" / "npz" / "records"
    npz_dir.mkdir(parents=True)
    np.savez(npz_dir / "night2.npz", eeg=np.ones(100, dtype=np.float32))
    run_pipeline(config, output_dir=output_dir, dry_run=True)

    assert not (npz_dir / "night1.npz").exists()
    assert (npz_dir / "night2.npz").exists()
    assert (output_dir / "manifest" / "discovery_preview.csv").exists()
    record_manifest = pd.read_csv(output_dir / "manifest" / "record_manifest.csv")
    assert sorted(record_manifest["record_id"].tolist()) == ["night1", "night2"]
