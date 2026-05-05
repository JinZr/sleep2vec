from __future__ import annotations

import typing as t

import numpy as np

try:
    from scipy.signal import find_peaks as _scipy_find_peaks
except Exception:  # pragma: no cover - exercised when SciPy is absent.
    _scipy_find_peaks = None


EEG_BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "sigma": (12.0, 16.0),
    "beta": (13.0, 30.0),
}


def _as_float_array(value: t.Any, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.size == 0:
        raise ValueError(f"{name} must be non-empty.")
    return array


def _require_same_shape(reference: np.ndarray, generated: np.ndarray) -> None:
    if reference.shape != generated.shape:
        raise ValueError(f"reference and generated shapes must match: {reference.shape} vs {generated.shape}")


def _flatten_epoch_series(value: np.ndarray) -> np.ndarray:
    return value.reshape(-1, value.shape[-1]) if value.ndim > 1 else value.reshape(1, -1)


def _band_power(value: np.ndarray, *, sample_rate_hz: int, low: float, high: float) -> float:
    series = _flatten_epoch_series(value)
    freqs = np.fft.rfftfreq(series.shape[-1], d=1.0 / sample_rate_hz)
    spectrum = np.square(np.abs(np.fft.rfft(series, axis=-1)))
    band = (freqs >= low) & (freqs < high)
    if not np.any(band):
        return 0.0
    return float(spectrum[:, band].mean())


def eeg_bandpower_error(reference: t.Any, generated: t.Any, *, sample_rate_hz: int = 128) -> dict[str, float]:
    reference_array = _as_float_array(reference, name="reference")
    generated_array = _as_float_array(generated, name="generated")
    _require_same_shape(reference_array, generated_array)
    errors: dict[str, float] = {}
    for name, (low, high) in EEG_BANDS.items():
        reference_power = _band_power(reference_array, sample_rate_hz=sample_rate_hz, low=low, high=high)
        generated_power = _band_power(generated_array, sample_rate_hz=sample_rate_hz, low=low, high=high)
        errors[f"{name}_bandpower_error"] = float(abs(reference_power - generated_power))
    return errors


def emg_tone_error(reference: t.Any, generated: t.Any) -> float:
    reference_array = _as_float_array(reference, name="reference")
    generated_array = _as_float_array(generated, name="generated")
    _require_same_shape(reference_array, generated_array)
    return float(abs(np.sqrt(np.mean(np.square(reference_array))) - np.sqrt(np.mean(np.square(generated_array)))))


def ibi_mae(reference: t.Any, generated: t.Any) -> float:
    reference_array = _as_float_array(reference, name="reference")
    generated_array = _as_float_array(generated, name="generated")
    _require_same_shape(reference_array, generated_array)
    return float(np.mean(np.abs(reference_array - generated_array)))


def spo2_nadir_metrics(reference: t.Any, generated: t.Any) -> dict[str, float]:
    reference_array = _flatten_epoch_series(_as_float_array(reference, name="reference"))
    generated_array = _flatten_epoch_series(_as_float_array(generated, name="generated"))
    _require_same_shape(reference_array, generated_array)
    reference_nadir = reference_array.min(axis=-1)
    generated_nadir = generated_array.min(axis=-1)
    reference_timing = reference_array.argmin(axis=-1)
    generated_timing = generated_array.argmin(axis=-1)
    return {
        "nadir_error": float(np.mean(np.abs(reference_nadir - generated_nadir))),
        "nadir_timing_error": float(np.mean(np.abs(reference_timing - generated_timing))),
    }


def respiratory_amplitude_error(reference: t.Any, generated: t.Any) -> float:
    reference_array = _flatten_epoch_series(_as_float_array(reference, name="reference"))
    generated_array = _flatten_epoch_series(_as_float_array(generated, name="generated"))
    _require_same_shape(reference_array, generated_array)
    reference_amp = np.percentile(reference_array, 95, axis=-1) - np.percentile(reference_array, 5, axis=-1)
    generated_amp = np.percentile(generated_array, 95, axis=-1) - np.percentile(generated_array, 5, axis=-1)
    return float(np.mean(np.abs(reference_amp - generated_amp)))


def ecg_peak_metrics(reference: t.Any, generated: t.Any, *, sample_rate_hz: int = 128) -> dict[str, float]:
    reference_array = _flatten_epoch_series(_as_float_array(reference, name="reference"))
    generated_array = _flatten_epoch_series(_as_float_array(generated, name="generated"))
    _require_same_shape(reference_array, generated_array)
    distance = max(int(0.25 * sample_rate_hz), 1)
    count_errors: list[float] = []
    timing_errors: list[float] = []
    for reference_epoch, generated_epoch in zip(reference_array, generated_array):
        reference_peaks = _find_peaks(reference_epoch, distance=distance)
        generated_peaks = _find_peaks(generated_epoch, distance=distance)
        count_errors.append(abs(float(len(reference_peaks) - len(generated_peaks))))
        if len(reference_peaks) and len(generated_peaks):
            local_errors = [float(np.min(np.abs(generated_peaks - peak))) for peak in reference_peaks]
            timing_errors.append(float(np.mean(local_errors)))
    return {
        "peak_count_error": float(np.mean(count_errors)) if count_errors else 0.0,
        "peak_timing_error": float(np.mean(timing_errors)) if timing_errors else float("nan"),
    }


def _find_peaks(signal: np.ndarray, *, distance: int) -> np.ndarray:
    if _scipy_find_peaks is not None:
        peaks, _ = _scipy_find_peaks(signal, distance=distance)
        return peaks
    candidates = np.flatnonzero((signal[1:-1] > signal[:-2]) & (signal[1:-1] >= signal[2:])) + 1
    if candidates.size <= 1:
        return candidates
    ordered = sorted(candidates.tolist(), key=lambda idx: float(signal[idx]), reverse=True)
    kept: list[int] = []
    for idx in ordered:
        if all(abs(idx - previous) >= distance for previous in kept):
            kept.append(idx)
    return np.asarray(sorted(kept), dtype=np.int64)


def compute_feature_metrics(
    modality: str,
    reference: t.Any,
    generated: t.Any,
    *,
    sample_rate_hz: int,
) -> dict[str, float]:
    if modality == "eeg":
        return eeg_bandpower_error(reference, generated, sample_rate_hz=sample_rate_hz)
    if modality == "emg":
        return {"tone_error": emg_tone_error(reference, generated)}
    if modality == "ibi":
        return {"ibi_mae": ibi_mae(reference, generated)}
    if modality == "spo2":
        return spo2_nadir_metrics(reference, generated)
    if modality in {"airflow", "belt", "resp"}:
        return {"respiratory_amplitude_error": respiratory_amplitude_error(reference, generated)}
    if modality == "ecg":
        return ecg_peak_metrics(reference, generated, sample_rate_hz=sample_rate_hz)
    return {}


__all__ = [
    "compute_feature_metrics",
    "ecg_peak_metrics",
    "eeg_bandpower_error",
    "emg_tone_error",
    "ibi_mae",
    "respiratory_amplitude_error",
    "spo2_nadir_metrics",
]
