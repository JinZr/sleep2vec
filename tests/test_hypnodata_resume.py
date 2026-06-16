import json
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


def test_hypnodata_resume_skips_existing_npz(tmp_path: Path):
    config = load_config(_one_record_config(tmp_path))
    output_dir = tmp_path / "out"
    run_pipeline(config, output_dir=output_dir)
    npz_path = output_dir / "backends" / "npz" / "records" / "night1.npz"
    np.savez(npz_path, eeg=np.full(100, 7, dtype=np.float32))

    run_pipeline(config, output_dir=output_dir, resume=True)

    with np.load(npz_path) as npz:
        assert npz["eeg"].tolist() == [7] * 100
    progress = json.loads((output_dir / "status" / "progress.json").read_text())
    assert progress["processed_records"] == 0
    assert progress["skipped_records"] == 1


def test_hypnodata_resume_retries_failed_record(tmp_path: Path):
    output_dir = tmp_path / "out"
    bad_config = load_config(_one_record_config(tmp_path, label="Missing EEG"))
    run_pipeline(bad_config, output_dir=output_dir)
    assert pd.read_csv(output_dir / "manifest" / "failures.csv")["record_id"].tolist() == ["night1"]

    good_config = load_config(_one_record_config(tmp_path, label="EEG C3"))
    run_pipeline(good_config, output_dir=output_dir, resume=True)

    assert (output_dir / "backends" / "npz" / "records" / "night1.npz").exists()
    assert pd.read_csv(output_dir / "manifest" / "failures.csv").empty
    record_manifest = pd.read_csv(output_dir / "manifest" / "record_manifest.csv")
    assert record_manifest["qc_status"].tolist() == ["ok"]


def test_hypnodata_overwrite_rewrites_existing_npz(tmp_path: Path):
    output_dir = tmp_path / "out"
    run_pipeline(load_config(_one_record_config(tmp_path, scale=1.0)), output_dir=output_dir)
    npz_path = output_dir / "backends" / "npz" / "records" / "night1.npz"
    with np.load(npz_path) as npz:
        before = npz["eeg"][1]

    run_pipeline(load_config(_one_record_config(tmp_path, scale=2.0)), output_dir=output_dir, overwrite=True)

    with np.load(npz_path) as npz:
        after = npz["eeg"][1]
    assert after == before * 2


def test_hypnodata_resume_preserves_manifest_rows(tmp_path: Path):
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
    config = load_config(write_hypnodata_config(tmp_path, index, target_sfreq=10))
    output_dir = tmp_path / "out"
    run_pipeline(config, output_dir=output_dir)

    run_pipeline(config, output_dir=output_dir, resume=True, record_id="night1")

    record_manifest = pd.read_csv(output_dir / "manifest" / "record_manifest.csv")
    assert sorted(record_manifest["record_id"].tolist()) == ["night1", "night2"]
    progress = json.loads((output_dir / "status" / "progress.json").read_text())
    assert progress["total_records"] == 1
    assert progress["skipped_records"] == 1


def test_hypnodata_rejects_resume_and_overwrite_together(tmp_path: Path):
    config = load_config(_one_record_config(tmp_path))

    with pytest.raises(ValueError, match="mutually exclusive"):
        run_pipeline(config, output_dir=tmp_path / "out", resume=True, overwrite=True)
