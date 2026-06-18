from pathlib import Path

import pandas as pd
import pytest
import yaml

from hypnodata.config import load_config
from hypnodata.discovery import discover_records


def _write_config(tmp_path: Path, index: Path, *, record_id_column: str | None = "record_id") -> Path:
    discovery = {
        "type": "csv",
        "index": str(index),
        "file_columns": {"edf": "path"},
    }
    if record_id_column is not None:
        discovery["record_id_column"] = record_id_column
    path = tmp_path / "hypnodata.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "center": "toy",
                "record_discovery": discovery,
                "backend": {"type": "npz"},
                "signals": {
                    "eeg": {
                        "kind": "eeg",
                        "required": True,
                        "target_sfreq": 10,
                        "target_unit": "uV",
                        "candidates": ["EEG C3"],
                    }
                },
            }
        )
    )
    return path


def test_csv_discovery_preserves_configured_record_id(tmp_path: Path):
    index = tmp_path / "records.csv"
    pd.DataFrame([{"record_id": "subject 1", "path": str(tmp_path / "record.edf")}]).to_csv(index, index=False)

    records = discover_records(load_config(_write_config(tmp_path, index)))

    assert records[0].record_id == "subject 1"


def test_csv_discovery_preserves_numeric_and_na_like_configured_record_id(tmp_path: Path):
    index = tmp_path / "records.csv"
    csv_text = f"record_id,path\n001,{tmp_path / 'first.edf'}\nNA,{tmp_path / 'second.edf'}\n"
    index.write_text(csv_text)

    records = discover_records(load_config(_write_config(tmp_path, index)))

    assert [record.record_id for record in records] == ["001", "NA"]


def test_csv_discovery_rejects_invalid_configured_record_id(tmp_path: Path):
    index = tmp_path / "records.csv"
    pd.DataFrame([{"record_id": "a/b", "path": str(tmp_path / "record.edf")}]).to_csv(index, index=False)

    with pytest.raises(ValueError, match="record_id must be a single path-safe segment"):
        discover_records(load_config(_write_config(tmp_path, index)))


def test_csv_discovery_requires_configured_record_id_column(tmp_path: Path):
    index = tmp_path / "records.csv"
    pd.DataFrame([{"path": str(tmp_path / "subject 1.edf")}]).to_csv(index, index=False)

    with pytest.raises(ValueError, match="file_columns requires record_id_column"):
        load_config(_write_config(tmp_path, index, record_id_column=None))
