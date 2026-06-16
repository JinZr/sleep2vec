import json
from pathlib import Path
import time

import pytest

from hypnodata.config import BackendConfig, DiscoveryConfig, HypnodataConfig, SignalSpec, load_config
from hypnodata.pipeline import ProcessResult, run_pipeline
from tests.hypnodata_test_helpers import run_tiny_hypnodata, write_hypnodata_config, write_index, write_tiny_edf


def test_hypnodata_progress_uses_record_counts(tmp_path: Path):
    output_dir = run_tiny_hypnodata(tmp_path)

    progress = json.loads((output_dir / "status" / "progress.json").read_text())

    assert progress["task"] == "hypnodata"
    assert progress["status"] == "completed"
    assert progress["total_records"] == 1
    assert progress["processed_records"] == 1
    assert progress["succeeded_records"] == 1
    assert progress["failed_records"] == 0
    assert progress["skipped_records"] == 0
    assert progress["current_record_id"] is None
    assert progress["started_at"]
    assert progress["updated_at"]


def test_hypnodata_multiworker_progress_tracks_completed_records(tmp_path: Path, monkeypatch):
    (tmp_path / "1_slow.edf").write_text("")
    (tmp_path / "2_fast.edf").write_text("")
    config = HypnodataConfig(
        path=tmp_path / "config.yaml",
        center="toy",
        record_discovery=DiscoveryConfig(type="glob", root=tmp_path, pattern="*.edf"),
        backend=BackendConfig(type="npz"),
        signals={
            "eeg": SignalSpec(
                name="eeg",
                kind="eeg",
                required=True,
                target_sfreq=1,
                target_unit="uV",
                candidates=["EEG"],
            )
        },
    )
    progress_calls = []

    def fake_process_record(config, record, **kwargs):
        if record.record_id == "1_slow":
            time.sleep(0.2)
        return ProcessResult(
            record_id=record.record_id,
            record_row={
                "record_id": record.record_id,
                "center": record.center,
                "source": record.center,
                "subject_id": record.record_id,
                "session_id": record.record_id,
                "split": "train",
                "path": str(tmp_path / f"{record.record_id}.npz"),
                "duration": 10.0,
                "backend": "npz",
                "qc_status": "ok",
                "eeg_mask": 1,
            },
        )

    def fake_progress(run_dir, **payload):
        progress_calls.append(payload)
        return Path(run_dir) / "status" / "progress.json"

    monkeypatch.setattr("hypnodata.pipeline._process_record", fake_process_record)
    monkeypatch.setattr("hypnodata.pipeline.write_hypnodata_progress", fake_progress)

    run_pipeline(config, output_dir=tmp_path / "out", num_workers=2)

    running_records = [
        call["current_record_id"]
        for call in progress_calls
        if call["status"] == "running" and call.get("current_record_id")
    ]
    assert running_records[0] == "2_fast"


def test_hypnodata_crash_progress_counts_failed_record_once(tmp_path: Path):
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
    config = load_config(write_hypnodata_config(tmp_path, index, label="Missing EEG"))
    output_dir = tmp_path / "out"

    with pytest.raises(RuntimeError, match="Missing required channel"):
        run_pipeline(config, output_dir=output_dir, crash=True)

    progress = json.loads((output_dir / "status" / "progress.json").read_text())
    assert progress["status"] == "failed"
    assert progress["processed_records"] == 1
    assert progress["failed_records"] == 1
