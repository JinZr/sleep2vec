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
            "file_columns": {"edf": "path"},
            "record_id_column": "record_id",
        },
        "backend": {"type": "npz"},
        "signals": {
            "eeg": {
                "kind": "eeg",
                "required": True,
                "target_sfreq": 10,
                "target_unit": "uV",
                "candidates": ["EEG C3", "C3-A2"],
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
    assert config.signals["eeg"].candidates == ["EEG C3", "C3-A2"]
    assert config.signals["eeg"].preprocess == [
        NotchStep(freq=50.0, q=30.0),
        FilterStep(method="bessel", order=4, lowcut=0.5, highcut=45.0),
    ]


def test_load_config_accepts_adapter_options_passthrough(tmp_path: Path):
    payload = _payload()
    payload["adapter_options"] = {"cohort": "demo"}

    config = load_config(_write_yaml(tmp_path, payload))

    assert config.adapter_options == {"cohort": "demo"}


def test_load_config_rejects_custom_passthrough_block(tmp_path: Path):
    payload = _payload()
    payload["custom"] = {"site": "x"}

    with pytest.raises(ValueError, match="Unknown hypnodata config field"):
        load_config(_write_yaml(tmp_path, payload))


@pytest.mark.parametrize("kind", ["stage", "event_table", "event_dense", "event_anchor"])
def test_load_config_accepts_annotation_only_kinds_without_candidates(tmp_path: Path, kind: str):
    payload = _payload()
    payload["signals"]["events"] = {
        "kind": kind,
        "required": False,
        "candidates": [],
    }
    payload["signals"]["events"].update(
        {
            "stage": {"epoch_sec": 30},
            "event_table": {},
            "event_dense": {"interval_sec": 1},
            "event_anchor": {"window_sec": 10},
        }[kind]
    )

    config = load_config(_write_yaml(tmp_path, payload))

    assert config.signals["events"].kind == kind
    assert config.signals["events"].candidates == []


def test_load_config_accepts_builtin_ahi_signal(tmp_path: Path):
    payload = _payload()
    payload["signals"]["stage5"] = {
        "kind": "stage",
        "required": True,
        "epoch_sec": 30,
        "candidates": [],
    }
    payload["signals"]["ahi"] = {
        "kind": "ahi",
        "required": True,
        "interval_sec": 1,
        "candidates": [],
    }

    config = load_config(_write_yaml(tmp_path, payload))

    assert config.signals["ahi"].kind == "ahi"
    assert config.signals["ahi"].interval_sec == 1


@pytest.mark.parametrize(
    ("signals", "match"),
    [
        (
            {"ahi": {"kind": "ahi", "required": True, "interval_sec": 1, "candidates": []}},
            "requires signals.stage5",
        ),
        (
            {
                "stage5": {"kind": "stage", "required": True, "epoch_sec": 20, "candidates": []},
                "ahi": {"kind": "ahi", "required": True, "interval_sec": 1, "candidates": []},
            },
            "stage5.epoch_sec to be 30",
        ),
        (
            {
                "stage5": {"kind": "stage", "required": True, "epoch_sec": 30, "candidates": []},
                "ahi": {"kind": "ahi", "required": True, "interval_sec": 2, "candidates": []},
            },
            "interval_sec must be 1",
        ),
        (
            {
                "stage5": {"kind": "stage", "required": True, "epoch_sec": 30, "candidates": []},
                "ahi": {"kind": "ahi", "required": True, "interval_sec": 1, "candidates": ["AHI"]},
            },
            "candidates must be empty",
        ),
        (
            {
                "stage5": {"kind": "stage", "required": True, "epoch_sec": 30, "candidates": []},
                "ahi": {"kind": "ahi", "required": True, "interval_sec": 1, "candidates": []},
                "ah_event": {"kind": "event_dense", "required": False, "interval_sec": 1, "candidates": []},
            },
            "cannot be declared with signals.ahi",
        ),
        (
            {
                "stage5": {"kind": "stage", "required": True, "epoch_sec": 30, "candidates": []},
                "ahi_label": {"kind": "ahi", "required": True, "interval_sec": 1, "candidates": []},
            },
            "kind=ahi must be declared as signals.ahi",
        ),
    ],
)
def test_load_config_rejects_invalid_builtin_ahi_contract(tmp_path: Path, signals: dict, match: str):
    payload = _payload()
    payload["signals"].update(signals)

    with pytest.raises(ValueError, match=match):
        load_config(_write_yaml(tmp_path, payload))


def test_load_config_rejects_raw_signal_without_candidates(tmp_path: Path):
    payload = _payload()
    payload["signals"]["eeg"]["candidates"] = []

    with pytest.raises(ValueError, match="candidates must not be empty"):
        load_config(_write_yaml(tmp_path, payload))


def test_load_config_rejects_empty_candidate_label(tmp_path: Path):
    payload = _payload()
    payload["signals"]["eeg"]["candidates"] = [""]

    with pytest.raises(ValueError, match="must be a non-empty string"):
        load_config(_write_yaml(tmp_path, payload))


def test_load_config_rejects_non_boolean_required(tmp_path: Path):
    payload = _payload()
    payload["signals"]["eeg"]["required"] = "false"

    with pytest.raises(ValueError, match=r"signals\.eeg\.required must be a boolean"):
        load_config(_write_yaml(tmp_path, payload))


def test_load_config_rejects_unknown_annotation_only_kind_without_candidates(tmp_path: Path):
    payload = _payload()
    payload["signals"]["mystery"] = {
        "kind": "mystery",
        "required": False,
        "candidates": [],
    }

    with pytest.raises(ValueError, match="candidates must not be empty"):
        load_config(_write_yaml(tmp_path, payload))


def test_load_config_rejects_annotation_source_field(tmp_path: Path):
    payload = _payload()
    payload["signals"]["stage5"] = {
        "kind": "stage",
        "required": False,
        "epoch_sec": 30,
        "candidates": [],
        "annotation": {"type": "csv"},
    }

    with pytest.raises(ValueError, match="Unknown hypnodata config field"):
        load_config(_write_yaml(tmp_path, payload))


@pytest.mark.parametrize(
    ("kind", "fields", "match"),
    [
        ("stage", {}, r"epoch_sec is required"),
        ("event_dense", {}, r"interval_sec is required"),
        ("event_anchor", {}, r"window_sec is required"),
        ("stage", {"target_sfreq": 1.0, "epoch_sec": 30}, r"target_sfreq is not used"),
        ("event_dense", {"window_sec": 30}, r"window_sec is not valid"),
        ("eeg", {"epoch_sec": 30, "candidates": ["EEG C3"]}, r"epoch_sec is only valid"),
    ],
)
def test_load_config_rejects_invalid_annotation_timing(tmp_path: Path, kind: str, fields: dict, match: str):
    payload = _payload()
    signal = {
        "kind": kind,
        "required": False,
        "candidates": [],
    }
    signal.update(fields)
    payload["signals"]["event"] = signal

    with pytest.raises(ValueError, match=match):
        load_config(_write_yaml(tmp_path, payload))


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


def test_load_config_rejects_legacy_file_column(tmp_path: Path):
    payload = _payload()
    payload["record_discovery"]["file_column"] = "path"

    with pytest.raises(ValueError, match="Unknown hypnodata config field"):
        load_config(_write_yaml(tmp_path, payload))


def test_load_config_requires_record_id_column_for_file_columns(tmp_path: Path):
    payload = _payload()
    del payload["record_discovery"]["record_id_column"]

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


@pytest.mark.parametrize(
    "candidate",
    [
        {"label": "EEG"},
        {"regex": "EEG"},
        {"label": "EEG", "priority": 1},
    ],
)
def test_load_config_rejects_candidate_mappings(tmp_path: Path, candidate: dict):
    payload = _payload()
    payload["signals"]["eeg"]["candidates"] = [candidate]

    with pytest.raises(ValueError, match="must be a non-empty string"):
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
