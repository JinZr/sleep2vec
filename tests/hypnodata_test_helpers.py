from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from hypnodata.config import load_config
from hypnodata.pipeline import run_pipeline
from preprocess.watchpat_zzp_to_edf import SignalSpec, StudyMetadata, write_edf_manual


def write_tiny_edf(path: Path, *, label: str = "EEG C3", duration_sec: int = 60, unit: str = "uV") -> Path:
    sfreq = 10
    samples = np.arange(duration_sec * sfreq, dtype=np.int16)
    write_edf_manual(
        str(path),
        [SignalSpec(label=label, samples=samples, sample_frequency=sfreq, dimension=unit)],
        StudyMetadata(source_path=str(path), patient_code=path.stem),
    )
    return path


def write_index(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "records.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def write_hypnodata_config(
    tmp_path: Path,
    index: Path,
    *,
    label: str = "EEG C3",
    scale: float = 1.0,
    target_sfreq: float = 1.0,
    extra_signals: dict | None = None,
    adapter: str | None = None,
    adapter_options: dict | None = None,
) -> Path:
    record_discovery = {
        "type": "csv",
        "index": str(index),
        "file_column": "path",
        "record_id_column": "record_id",
        "metadata_columns": ["age"],
    }
    if adapter is not None:
        record_discovery = {"type": "custom", "adapter": adapter}
    signals = {
        "eeg": {
            "kind": "eeg",
            "required": True,
            "target_sfreq": target_sfreq,
            "target_unit": "uV",
            "candidates": [{"label": label, "priority": 10}],
            "scale": scale,
            "preprocess": ["finite_check", "truncate_to_common"],
        }
    }
    signals.update(extra_signals or {})
    payload = {
        "center": "toy_center",
        "record_discovery": record_discovery,
        "backend": {"type": "npz"},
        "signals": signals,
    }
    if adapter_options is not None:
        payload["adapter_options"] = adapter_options
    path = tmp_path / "hypnodata.yaml"
    path.write_text(yaml.safe_dump(payload))
    return path


def run_tiny_hypnodata(tmp_path: Path, *, duration_sec: int = 60) -> Path:
    edf_path = write_tiny_edf(tmp_path / "night1.edf", duration_sec=duration_sec)
    index = write_index(
        tmp_path,
        [
            {
                "record_id": "night1",
                "path": str(edf_path),
                "source": "toy_source",
                "split": "test",
                "subject_id": "sub1",
                "session_id": "ses1",
                "age": 55,
            }
        ],
    )
    config = load_config(write_hypnodata_config(tmp_path, index))
    output_dir = tmp_path / "hypnodata_out"
    run_pipeline(config, output_dir=output_dir)
    return output_dir
