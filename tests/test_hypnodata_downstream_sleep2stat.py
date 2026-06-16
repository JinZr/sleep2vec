from pathlib import Path

import yaml

from sleep2stat.config import load_config as load_sleep2stat_config
from sleep2stat.io.records import load_records
from tests.hypnodata_test_helpers import run_tiny_hypnodata


def test_sleep2stat_load_records_consumes_hypnodata_record_manifest(tmp_path: Path):
    output_dir = run_tiny_hypnodata(tmp_path)
    sleep2stat_config = tmp_path / "sleep2stat.yaml"
    sleep2stat_config.write_text(
        yaml.safe_dump(
            {
                "run": {
                    "name": "hypnodata_smoke",
                    "output_dir": str(tmp_path / "sleep2stat_out"),
                    "overwrite": False,
                    "skip_existing": True,
                },
                "data": {
                    "backend": "npz",
                    "index": str(output_dir / "manifest" / "record_manifest.csv"),
                    "split": ["test"],
                    "path_column": "path",
                    "duration_column": "duration",
                    "split_column": "split",
                    "source_column": "source",
                    "record_id_columns": ["record_id"],
                    "metadata_columns": ["center", "subject_id", "session_id", "age", "eeg_mask"],
                    "token_sec": 30,
                    "max_tokens": 2,
                },
                "signals": {
                    "channels": {
                        "eeg": {
                            "source": "eeg",
                            "sfreq": 1,
                            "kind": "eeg",
                            "input_dim": 30,
                            "unit": "uV",
                        }
                    }
                },
                "analyzers": [
                    {
                        "name": "stage_reference",
                        "type": "npz_stage_reference",
                        "enabled": False,
                        "stage_key": "stage5",
                    }
                ],
                "reducers": [],
                "outputs": {
                    "write_global_tables": True,
                    "write_per_record": True,
                    "include_probabilities": True,
                    "include_raw_logits": False,
                    "compression": "gzip",
                },
            }
        )
    )

    config = load_sleep2stat_config(sleep2stat_config)
    records = load_records(config.data)

    assert len(records) == 1
    record = records[0]
    assert record.record_id == "night1"
    assert record.path == output_dir / "backends" / "npz" / "records" / "night1.npz"
    assert record.path_exists is True
    assert record.duration_sec == 60.0
    assert record.source == "toy_source"
    assert record.split == "test"
    assert record.metadata["center"] == "toy_center"
    assert record.metadata["subject_id"] == "sub1"
    assert record.metadata["session_id"] == "ses1"
    assert record.metadata["age"] == 55
    assert record.metadata["eeg_mask"] == 1
