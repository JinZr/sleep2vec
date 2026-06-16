from pathlib import Path

import pytest
import yaml

from hypnodata.config import load_config


def _payload() -> dict:
    return {
        "center": "toy",
        "record_discovery": {
            "type": "csv",
            "index": "records.csv",
            "file_column": "path",
            "record_id_column": "record_id",
        },
        "backend": {"type": "npz"},
        "signals": {
            "eeg": {
                "kind": "eeg",
                "required": True,
                "target_sfreq": 10,
                "target_unit": "uV",
                "candidates": [{"label": "EEG C3", "priority": 5}],
                "preprocess": ["finite_check", "truncate_to_common"],
            }
        },
    }


def _write_yaml(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "hypnodata.yaml"
    path.write_text(yaml.safe_dump(payload))
    return path


def test_load_config_accepts_minimal_yaml(tmp_path: Path):
    config = load_config(_write_yaml(tmp_path, _payload()))

    assert config.center == "toy"
    assert config.record_discovery.type == "csv"
    assert config.backend.type == "npz"
    assert config.signals["eeg"].candidates[0].label == "EEG C3"


def test_load_config_accepts_custom_passthrough_blocks(tmp_path: Path):
    payload = _payload()
    payload["custom"] = {"site": "x"}
    payload["adapter_options"] = {"cohort": "demo"}

    config = load_config(_write_yaml(tmp_path, payload))

    assert config.custom == {"site": "x"}
    assert config.adapter_options == {"cohort": "demo"}


def test_load_config_rejects_schema_version(tmp_path: Path):
    payload = _payload()
    payload["schema_version"] = 1

    with pytest.raises(ValueError, match="schema/version"):
        load_config(_write_yaml(tmp_path, payload))


def test_load_config_rejects_unknown_top_level_key(tmp_path: Path):
    payload = _payload()
    payload["legacy"] = True

    with pytest.raises(ValueError, match="Unknown hypnodata config field"):
        load_config(_write_yaml(tmp_path, payload))


def test_load_config_rejects_legacy_path_column(tmp_path: Path):
    payload = _payload()
    payload["record_discovery"]["path_column"] = "path"

    with pytest.raises(ValueError, match="Unknown hypnodata config field"):
        load_config(_write_yaml(tmp_path, payload))


def test_load_config_requires_record_id_column_for_file_columns(tmp_path: Path):
    payload = _payload()
    del payload["record_discovery"]["record_id_column"]
    payload["record_discovery"]["file_columns"] = {"edf": "edf_path", "annotation": "annotation_path"}

    with pytest.raises(ValueError, match="file_columns requires record_id_column"):
        load_config(_write_yaml(tmp_path, payload))


def test_load_config_requires_core_blocks(tmp_path: Path):
    payload = _payload()
    del payload["signals"]

    with pytest.raises(ValueError, match="Missing required hypnodata config field"):
        load_config(_write_yaml(tmp_path, payload))


def test_load_config_rejects_non_npz_backend(tmp_path: Path):
    payload = _payload()
    payload["backend"]["type"] = "kaldi"

    with pytest.raises(ValueError, match="backend.type must be 'npz'"):
        load_config(_write_yaml(tmp_path, payload))


def test_load_config_requires_exactly_one_candidate_matcher(tmp_path: Path):
    payload = _payload()
    payload["signals"]["eeg"]["candidates"] = [{"label": "EEG", "regex": "EEG"}]

    with pytest.raises(ValueError, match="exactly one"):
        load_config(_write_yaml(tmp_path, payload))
