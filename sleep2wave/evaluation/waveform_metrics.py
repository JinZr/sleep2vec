from __future__ import annotations

import math
import typing as t

import numpy as np

_EEG_EOG_BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "sigma": (12.0, 16.0),
    "beta": (13.0, 30.0),
}

_ECG_EMG_BANDS = {
    "0_5_5_hz": (0.5, 5.0),
    "5_20_hz": (5.0, 20.0),
    "20_45_hz": (20.0, 45.0),
}

_LOW_FREQUENCY_BANDS = {
    "0_03_0_1_hz": (0.03, 0.1),
    "0_1_0_5_hz": (0.1, 0.5),
    "0_5_1_0_hz": (0.5, 1.0),
}


def _as_float_array(value: t.Any, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.size == 0:
        raise ValueError(f"{name} must be non-empty.")
    return array


def _require_same_shape(reference: np.ndarray, generated: np.ndarray) -> None:
    if reference.shape != generated.shape:
        raise ValueError(f"reference and generated shapes must match: {reference.shape} vs {generated.shape}")


def rmse(reference: t.Any, generated: t.Any) -> float:
    reference_array = _as_float_array(reference, name="reference")
    generated_array = _as_float_array(generated, name="generated")
    _require_same_shape(reference_array, generated_array)
    return float(np.sqrt(np.mean(np.square(reference_array - generated_array))))


def mae(reference: t.Any, generated: t.Any) -> float:
    reference_array = _as_float_array(reference, name="reference")
    generated_array = _as_float_array(generated, name="generated")
    _require_same_shape(reference_array, generated_array)
    return float(np.mean(np.abs(reference_array - generated_array)))


def correlation(reference: t.Any, generated: t.Any) -> float:
    reference_array = _as_float_array(reference, name="reference").reshape(-1)
    generated_array = _as_float_array(generated, name="generated").reshape(-1)
    if reference_array.shape != generated_array.shape:
        raise ValueError(
            f"reference and generated shapes must match: {reference_array.shape} vs {generated_array.shape}"
        )
    if reference_array.size < 2 or np.std(reference_array) == 0.0 or np.std(generated_array) == 0.0:
        return float("nan")
    return float(np.corrcoef(reference_array, generated_array)[0, 1])


def spectral_distance(reference: t.Any, generated: t.Any) -> float:
    reference_array = _as_float_array(reference, name="reference")
    generated_array = _as_float_array(generated, name="generated")
    _require_same_shape(reference_array, generated_array)
    reference_mag = np.abs(np.fft.rfft(reference_array, axis=-1))
    generated_mag = np.abs(np.fft.rfft(generated_array, axis=-1))
    denom = float(np.sqrt(np.mean(np.square(reference_mag)))) + 1e-12
    return float(np.sqrt(np.mean(np.square(reference_mag - generated_mag))) / denom)


def _flatten_series(value: np.ndarray) -> np.ndarray:
    return value.reshape(-1, value.shape[-1])


def _spectral_distance_from_magnitudes(reference_mag: np.ndarray, generated_mag: np.ndarray) -> float:
    denom = float(np.sqrt(np.mean(np.square(reference_mag)))) + 1e-12
    return float(np.sqrt(np.mean(np.square(reference_mag - generated_mag))) / denom)


def _stft_magnitude(series: np.ndarray, *, window_size: int) -> np.ndarray | None:
    frame_count = series.shape[-1]
    if window_size > frame_count:
        return None
    hop = max(window_size // 2, 1)
    starts = list(range(0, frame_count - window_size + 1, hop))
    if starts[-1] != frame_count - window_size:
        starts.append(frame_count - window_size)
    window = np.hanning(window_size)
    windows = np.stack([series[:, start : start + window_size] * window for start in starts], axis=1)
    return np.abs(np.fft.rfft(windows, axis=-1))


def mr_spectral_distance(reference: t.Any, generated: t.Any, *, sample_rate_hz: int) -> float:
    reference_array = _as_float_array(reference, name="reference")
    generated_array = _as_float_array(generated, name="generated")
    _require_same_shape(reference_array, generated_array)
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive.")
    reference_series = _flatten_series(reference_array)
    generated_series = _flatten_series(generated_array)
    distances: list[float] = []
    for window_sec in (0.5, 1.0, 2.0, 4.0):
        window_size = max(int(round(window_sec * sample_rate_hz)), 2)
        reference_mag = _stft_magnitude(reference_series, window_size=window_size)
        generated_mag = _stft_magnitude(generated_series, window_size=window_size)
        if reference_mag is None or generated_mag is None:
            continue
        distances.append(_spectral_distance_from_magnitudes(reference_mag, generated_mag))
    return float(np.mean(distances)) if distances else spectral_distance(reference_array, generated_array)


def _band_ranges_for_modality(modality: str) -> dict[str, tuple[float, float]]:
    if modality in {"eeg", "eog"}:
        return _EEG_EOG_BANDS
    if modality in {"ecg", "emg"}:
        return _ECG_EMG_BANDS
    if modality in {"airflow", "belt", "spo2", "ibi", "resp"}:
        return _LOW_FREQUENCY_BANDS
    return {}


def band_spectral_distances(
    reference: t.Any,
    generated: t.Any,
    *,
    modality: str,
    sample_rate_hz: int,
) -> dict[str, float]:
    reference_array = _as_float_array(reference, name="reference")
    generated_array = _as_float_array(generated, name="generated")
    _require_same_shape(reference_array, generated_array)
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive.")
    reference_series = _flatten_series(reference_array)
    generated_series = _flatten_series(generated_array)
    freqs = np.fft.rfftfreq(reference_series.shape[-1], d=1.0 / sample_rate_hz)
    reference_mag = np.abs(np.fft.rfft(reference_series, axis=-1))
    generated_mag = np.abs(np.fft.rfft(generated_series, axis=-1))
    metrics: dict[str, float] = {}
    for name, (low, high) in _band_ranges_for_modality(modality).items():
        band = (freqs >= low) & (freqs < high)
        key = f"band_spectral_distance_{name}"
        if not np.any(band):
            metrics[key] = 0.0
            continue
        metrics[key] = _spectral_distance_from_magnitudes(reference_mag[:, band], generated_mag[:, band])
    return metrics


def _overlap_for_shift(reference: np.ndarray, generated: np.ndarray, shift: int) -> tuple[np.ndarray, np.ndarray]:
    if shift == 0:
        return reference, generated
    if abs(shift) >= reference.shape[-1]:
        raise ValueError("shift magnitude must be smaller than the last dimension.")
    if shift > 0:
        return reference[..., shift:], generated[..., :-shift]
    left = abs(shift)
    return reference[..., :-left], generated[..., left:]


def _min_shift_metric(reference: t.Any, generated: t.Any, *, max_shift_frames: int, metric: str) -> float:
    reference_array = _as_float_array(reference, name="reference")
    generated_array = _as_float_array(generated, name="generated")
    _require_same_shape(reference_array, generated_array)
    if max_shift_frames < 0:
        raise ValueError("max_shift_frames must be >= 0.")
    max_shift_frames = min(max_shift_frames, reference_array.shape[-1] - 1)
    values: list[float] = []
    for shift in range(-max_shift_frames, max_shift_frames + 1):
        shifted_reference, shifted_generated = _overlap_for_shift(reference_array, generated_array, shift)
        if metric == "rmse":
            values.append(float(np.sqrt(np.mean(np.square(shifted_reference - shifted_generated)))))
        elif metric == "mae":
            values.append(float(np.mean(np.abs(shifted_reference - shifted_generated))))
        else:
            raise ValueError(f"Unsupported min-shift metric: {metric}")
    return min(values)


def min_rmse(reference: t.Any, generated: t.Any, *, max_shift_frames: int) -> float:
    return _min_shift_metric(reference, generated, max_shift_frames=max_shift_frames, metric="rmse")


def min_mae(reference: t.Any, generated: t.Any, *, max_shift_frames: int) -> float:
    return _min_shift_metric(reference, generated, max_shift_frames=max_shift_frames, metric="mae")


def snr_improvement(reference: t.Any, generated: t.Any, baseline: t.Any) -> float:
    reference_array = _as_float_array(reference, name="reference")
    generated_array = _as_float_array(generated, name="generated")
    baseline_array = _as_float_array(baseline, name="baseline")
    _require_same_shape(reference_array, generated_array)
    _require_same_shape(reference_array, baseline_array)
    generated_mse = float(np.mean(np.square(reference_array - generated_array)))
    baseline_mse = float(np.mean(np.square(reference_array - baseline_array)))
    if generated_mse == 0.0:
        return math.inf if baseline_mse > 0.0 else 0.0
    if baseline_mse == 0.0:
        return -math.inf
    return float(10.0 * np.log10(baseline_mse / generated_mse))


def compute_waveform_metrics(
    reference: t.Any,
    generated: t.Any,
    *,
    modality: str,
    sample_rate_hz: int,
    baseline: t.Any | None = None,
    max_shift_frames: int = 0,
) -> dict[str, float]:
    metrics = {
        "rmse": rmse(reference, generated),
        "mae": mae(reference, generated),
        "min_rmse": min_rmse(reference, generated, max_shift_frames=max_shift_frames),
        "min_mae": min_mae(reference, generated, max_shift_frames=max_shift_frames),
        "correlation": correlation(reference, generated),
        "spectral_distance": spectral_distance(reference, generated),
        "mr_spectral_distance": mr_spectral_distance(reference, generated, sample_rate_hz=sample_rate_hz),
    }
    metrics.update(band_spectral_distances(reference, generated, modality=modality, sample_rate_hz=sample_rate_hz))
    if baseline is not None:
        metrics["snr_improvement"] = snr_improvement(reference, generated, baseline)
    return metrics


__all__ = [
    "band_spectral_distances",
    "compute_waveform_metrics",
    "correlation",
    "mae",
    "min_mae",
    "min_rmse",
    "mr_spectral_distance",
    "rmse",
    "snr_improvement",
    "spectral_distance",
]
