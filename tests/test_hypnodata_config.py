from pathlib import Path

import pytest
import yaml

from hypnodata.config import FilterStep, NotchStep, load_config


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
                "preprocess": [
                    {"type": "notch", "freq": 50.0, "q": 30.0},
                    {"type": "filter", "method": "bessel", "order": 4, "lowcut": 0.5, "highcut": 45.0},
                ],
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
    assert config.signals["eeg"].preprocess == [
        NotchStep(freq=50.0, q=30.0),
        FilterStep(method="bessel", order=4, lowcut=0.5, highcut=45.0),
    ]


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


def test_load_config_rejects_bare_preprocess_step(tmp_path: Path):
    payload = _payload()
    payload["signals"]["eeg"]["preprocess"] = ["filter"]

    with pytest.raises(ValueError, match=r"signals\.eeg\.preprocess\[0\] must be a mapping"):
        load_config(_write_yaml(tmp_path, payload))


def test_load_config_rejects_unknown_preprocess_step_type(tmp_path: Path):
    payload = _payload()
    payload["signals"]["eeg"]["preprocess"] = [{"type": "smooth", "order": 4}]

    with pytest.raises(ValueError, match=r"preprocess\[0\]\.type must be one of"):
        load_config(_write_yaml(tmp_path, payload))


def test_load_config_rejects_unknown_preprocess_step_field(tmp_path: Path):
    payload = _payload()
    payload["signals"]["eeg"]["preprocess"] = [
        {"type": "filter", "method": "bessel", "order": 4, "lowcut": 0.5, "foo": 1}
    ]

    with pytest.raises(ValueError, match="Unknown hypnodata config field"):
        load_config(_write_yaml(tmp_path, payload))


@pytest.mark.parametrize(
    ("step", "match"),
    [
        ({"type": "filter", "order": 4, "lowcut": 0.5}, r"method is required"),
        ({"type": "filter", "method": "fir", "order": 4, "lowcut": 0.5}, r"method must be one of"),
        ({"type": "filter", "method": "bessel", "lowcut": 0.5}, r"order is required"),
        ({"type": "filter", "method": "bessel", "order": 0, "lowcut": 0.5}, r"positive integer"),
        ({"type": "filter", "method": "bessel", "order": 4}, r"at least one of lowcut or highcut"),
        ({"type": "filter", "method": "bessel", "order": 4, "lowcut": -0.5}, r"positive number"),
        (
            {"type": "filter", "method": "bessel", "order": 4, "lowcut": 45.0, "highcut": 0.5},
            r"lowcut must be smaller",
        ),
    ],
)
def test_load_config_rejects_invalid_filter_step(tmp_path: Path, step: dict, match: str):
    payload = _payload()
    payload["signals"]["eeg"]["preprocess"] = [step]

    with pytest.raises(ValueError, match=match):
        load_config(_write_yaml(tmp_path, payload))


@pytest.mark.parametrize(
    ("step", "match"),
    [
        ({"type": "notch", "q": 30.0}, r"freq is required"),
        ({"type": "notch", "freq": 50.0}, r"q is required"),
        ({"type": "notch", "freq": 0.0, "q": 30.0}, r"positive number"),
        ({"type": "notch", "freq": 50.0, "q": -30.0}, r"positive number"),
    ],
)
def test_load_config_rejects_invalid_notch_step(tmp_path: Path, step: dict, match: str):
    payload = _payload()
    payload["signals"]["eeg"]["preprocess"] = [step]

    with pytest.raises(ValueError, match=match):
        load_config(_write_yaml(tmp_path, payload))
