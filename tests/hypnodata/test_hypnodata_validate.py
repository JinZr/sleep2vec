import json
from pathlib import Path
import subprocess
import sys

import pandas as pd

from hypnodata.config import load_config
from hypnodata.pipeline import validate_pipeline
from tests.hypnodata_test_helpers import write_hypnodata_config, write_index, write_tiny_edf


def _mixed_config(tmp_path: Path) -> Path:
    good_edf = write_tiny_edf(tmp_path / "good.edf", duration_sec=10)
    bad_edf = write_tiny_edf(tmp_path / "bad.edf", label="Other EEG", duration_sec=10)
    index = write_index(
        tmp_path,
        [
            {
                "record_id": "good",
                "path": str(good_edf),
                "source": "toy_source",
                "split": "train",
                "subject_id": "sub1",
                "session_id": "ses1",
                "age": 50,
            },
            {
                "record_id": "bad",
                "path": str(bad_edf),
                "source": "toy_source",
                "split": "train",
                "subject_id": "sub2",
                "session_id": "ses2",
                "age": 51,
            },
        ],
    )
    return write_hypnodata_config(tmp_path, index, target_sfreq=10)


def test_validate_pipeline_collects_failures_and_continues(tmp_path: Path):
    config = load_config(_mixed_config(tmp_path))
    output_dir = tmp_path / "validate"

    failure_count = validate_pipeline(config, output_dir=output_dir, num_workers=2)

    assert failure_count == 1
    record_manifest = pd.read_csv(output_dir / "manifest" / "record_manifest.csv")
    assert record_manifest["record_id"].tolist() == ["good"]
    failures = pd.read_csv(output_dir / "manifest" / "failures.csv")
    assert failures["record_id"].tolist() == ["bad"]
    assert failures.loc[0, "error_type"] == "channel_resolution"
    qc_summary = pd.read_csv(output_dir / "manifest" / "qc_summary.csv")
    assert qc_summary.loc[0, "record_id"] == "bad"
    progress = json.loads((output_dir / "status" / "progress.json").read_text())
    assert progress["status"] == "failed"
    assert progress["processed_records"] == 2
    assert progress["succeeded_records"] == 1
    assert progress["failed_records"] == 1
    assert not (output_dir / "backends" / "npz" / "records" / "good.npz").exists()


def test_validate_pipeline_all_valid_records_returns_zero(tmp_path: Path):
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
    config = load_config(write_hypnodata_config(tmp_path, index, target_sfreq=10))
    output_dir = tmp_path / "validate_ok"

    failure_count = validate_pipeline(config, output_dir=output_dir)

    assert failure_count == 0
    failures = pd.read_csv(output_dir / "manifest" / "failures.csv")
    assert failures.empty
    progress = json.loads((output_dir / "status" / "progress.json").read_text())
    assert progress["status"] == "completed"
    assert not (output_dir / "backends" / "npz" / "records" / "night1.npz").exists()


def test_validate_cli_returns_nonzero_after_writing_report(tmp_path: Path):
    config_path = _mixed_config(tmp_path)
    output_dir = tmp_path / "validate_cli"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "hypnodata",
            "validate",
            "--config",
            str(config_path),
            "--output-dir",
            str(output_dir),
        ],
        cwd=Path(__file__).resolve().parents[2],
        text=True,
        capture_output=True,
    )

    assert result.returncode == 1
    failures = pd.read_csv(output_dir / "manifest" / "failures.csv")
    assert failures["record_id"].tolist() == ["bad"]
