import numpy as np
import pytest

from hypnodata.channels import ChannelSelection
from hypnodata.config import FilterStep, NotchStep, SignalSpec
from hypnodata.preprocess import preprocess_signal


def _selection(sfreq: float, *, raw_unit: str | None = "uV", target_unit: str | None = "uV") -> ChannelSelection:
    return ChannelSelection(
        canonical_channel="eeg",
        kind="eeg",
        available=True,
        required=True,
        raw_file="record.edf",
        raw_label="EEG",
        raw_index=0,
        raw_sfreq=sfreq,
        target_sfreq=None,
        raw_unit=raw_unit,
        target_unit=target_unit,
        raw_n_samples=None,
        selection_reason="label:EEG",
    )


def _spec(*steps, target_unit: str | None = "uV", scale: float = 1.0) -> SignalSpec:
    return SignalSpec(
        name="eeg",
        kind="eeg",
        required=True,
        target_sfreq=None,
        target_unit=target_unit,
        candidates=[],
        scale=scale,
        preprocess=list(steps),
    )


def _fft_amplitude(values: np.ndarray, sfreq: float, freq: float) -> float:
    freqs = np.fft.rfftfreq(values.size, d=1 / sfreq)
    idx = int(np.argmin(np.abs(freqs - freq)))
    return float(np.abs(np.fft.rfft(values)[idx]))


def test_preprocess_signal_applies_neurokit2_bandpass_filter():
    sfreq = 200.0
    time = np.arange(0, 20, 1 / sfreq)
    raw = (
        np.sin(2 * np.pi * 0.2 * time) + np.sin(2 * np.pi * 5.0 * time) + 0.5 * np.sin(2 * np.pi * 40.0 * time)
    ).astype(np.float32)

    processed = preprocess_signal(
        raw,
        _selection(sfreq),
        _spec(FilterStep(method="bessel", order=4, lowcut=1.0, highcut=10.0)),
    )

    assert _fft_amplitude(processed.data, sfreq, 5.0) > _fft_amplitude(raw, sfreq, 5.0) * 0.5
    assert _fft_amplitude(processed.data, sfreq, 0.2) < _fft_amplitude(raw, sfreq, 0.2) * 0.2
    assert _fft_amplitude(processed.data, sfreq, 40.0) < _fft_amplitude(raw, sfreq, 40.0) * 0.2
    assert processed.steps == ["filter:bessel:bandpass:1-10Hz:order=4", "finite_check"]


def test_preprocess_signal_applies_notch_filter():
    sfreq = 200.0
    time = np.arange(0, 20, 1 / sfreq)
    raw = (np.sin(2 * np.pi * 10.0 * time) + 0.5 * np.sin(2 * np.pi * 50.0 * time)).astype(np.float32)

    processed = preprocess_signal(raw, _selection(sfreq), _spec(NotchStep(freq=50.0, q=30.0)))

    assert _fft_amplitude(processed.data, sfreq, 50.0) < _fft_amplitude(raw, sfreq, 50.0) * 0.2
    assert _fft_amplitude(processed.data, sfreq, 10.0) > _fft_amplitude(raw, sfreq, 10.0) * 0.8
    assert processed.steps == ["notch:50Hz:q=30", "finite_check"]


def test_preprocess_signal_converts_voltage_units_before_scale():
    raw = np.asarray([1.0], dtype=np.float32)

    processed = preprocess_signal(raw, _selection(10.0, raw_unit="mV"), _spec(scale=2.0))

    np.testing.assert_allclose(processed.data, np.asarray([2000.0], dtype=np.float32))
    assert processed.unit == "uV"
    assert processed.steps == ["unit:mV->uV", "scale:2", "finite_check"]


def test_preprocess_signal_converts_volts_to_microvolts():
    raw = np.asarray([1.0], dtype=np.float32)

    processed = preprocess_signal(raw, _selection(10.0, raw_unit="V"), _spec())

    np.testing.assert_allclose(processed.data, np.asarray([1_000_000.0], dtype=np.float32))
    assert processed.steps == ["unit:V->uV", "finite_check"]


def test_preprocess_signal_accepts_matching_non_voltage_units():
    raw = np.asarray([95.0], dtype=np.float32)

    processed = preprocess_signal(raw, _selection(1.0, raw_unit="%", target_unit="%"), _spec(target_unit="%"))

    np.testing.assert_allclose(processed.data, raw)
    assert processed.steps == ["finite_check"]


def test_preprocess_signal_rejects_missing_or_incompatible_raw_unit():
    raw = np.asarray([1.0], dtype=np.float32)

    with pytest.raises(ValueError, match="requires a raw unit"):
        preprocess_signal(raw, _selection(10.0, raw_unit=None), _spec())

    with pytest.raises(ValueError, match="cannot convert raw unit"):
        preprocess_signal(raw, _selection(10.0, raw_unit="%", target_unit="uV"), _spec())


def test_preprocess_signal_rejects_runtime_cutoffs_at_or_above_nyquist():
    raw = np.arange(100, dtype=np.float32)

    with pytest.raises(ValueError, match="below Nyquist"):
        preprocess_signal(raw, _selection(100.0), _spec(FilterStep(method="bessel", order=4, highcut=50.0)))

    with pytest.raises(ValueError, match="below Nyquist"):
        preprocess_signal(raw, _selection(100.0), _spec(NotchStep(freq=50.0, q=30.0)))
