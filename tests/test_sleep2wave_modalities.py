from __future__ import annotations

import pytest

from sleep2wave.data.modalities import (
    CANONICAL_MODALITIES,
    MODALITY_ALIASES,
    MODALITY_SPECS,
    get_modality_spec,
    normalize_modality_name,
    validate_modality_sequence,
)


def test_sleep2wave_modalities_has_expected_order():
    assert CANONICAL_MODALITIES == (
        "eeg",
        "eog",
        "emg",
        "ecg",
        "airflow",
        "belt",
        "spo2",
        "ibi",
        "resp",
    )


def test_sleep2wave_modalities_has_expected_sampling_contract():
    for name in ("eeg", "eog", "emg", "ecg"):
        spec = MODALITY_SPECS[name]
        assert spec.sample_rate_hz == 128
        assert spec.frames_per_epoch == 3840
        assert spec.frequency_group == "high_frequency"

    for name in ("airflow", "belt", "spo2", "ibi", "resp"):
        spec = MODALITY_SPECS[name]
        assert spec.sample_rate_hz == 4
        assert spec.frames_per_epoch == 120
        assert spec.frequency_group == "low_frequency"


def test_sleep2wave_aliases_resolve_to_canonical_names():
    assert MODALITY_ALIASES == {
        "eeg_original": "eeg",
        "eog_original": "eog",
        "emg_original": "emg",
        "ecg_original": "ecg",
        "resp_nasal_original": "airflow",
        "resp_original": "belt",
        "heartbeat": "ibi",
        "breath": "resp",
    }
    assert normalize_modality_name("heartbeat") == "ibi"
    assert get_modality_spec("resp_original").name == "belt"


def test_sleep2wave_validate_modality_sequence_rejects_duplicates():
    with pytest.raises(ValueError, match="Duplicate sleep2wave modality: eeg"):
        validate_modality_sequence(["eeg", "eeg"])


def test_sleep2wave_validate_modality_sequence_rejects_unknown_names():
    with pytest.raises(ValueError, match="canonical modality names"):
        validate_modality_sequence(["eeg_original"])

    with pytest.raises(ValueError, match="Unknown sleep2wave modality"):
        validate_modality_sequence(["unknown"], allow_aliases=True)
