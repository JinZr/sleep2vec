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


def _as_epoch_channel_series(value: t.Any, *, name: str) -> np.ndarray:
    array = _as_float_array(value, name=name)
    if array.ndim == 1:
        return array.reshape(1, 1, -1)
    if array.ndim == 2:
        return array[:, None, :]
    if array.ndim == 3:
        return array
    raise ValueError(f"{name} must have shape [frames], [epochs, frames], or [epochs, channels, frames].")


def _flatten_epoch_series(value: np.ndarray) -> np.ndarray:
    return value.reshape(-1, value.shape[-1])


def _finite_mean(values: t.Sequence[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(array)
    return float(np.mean(array[finite])) if finite.any() else float("nan")


def _paired_mae(reference: t.Sequence[float], generated: t.Sequence[float]) -> float:
    reference_array = np.asarray(reference, dtype=np.float64)
    generated_array = np.asarray(generated, dtype=np.float64)
    count = min(reference_array.size, generated_array.size)
    if count == 0:
        return 0.0 if reference_array.size == generated_array.size else float("nan")
    return float(np.mean(np.abs(reference_array[:count] - generated_array[:count])))


def _band_power(value: np.ndarray, *, sample_rate_hz: int, low: float, high: float) -> float:
    series = _flatten_epoch_series(value)
    freqs = np.fft.rfftfreq(series.shape[-1], d=1.0 / sample_rate_hz)
    spectrum = np.square(np.abs(np.fft.rfft(series, axis=-1)))
    band = (freqs >= low) & (freqs < high)
    if not np.any(band):
        return 0.0
    return float(spectrum[:, band].mean())


def eeg_bandpower_error(reference: t.Any, generated: t.Any, *, sample_rate_hz: int = 128) -> dict[str, float]:
    reference_array = _as_epoch_channel_series(reference, name="reference")
    generated_array = _as_epoch_channel_series(generated, name="generated")
    _require_same_shape(reference_array, generated_array)
    errors: dict[str, float] = {}
    for name, (low, high) in EEG_BANDS.items():
        reference_power = _band_power(reference_array, sample_rate_hz=sample_rate_hz, low=low, high=high)
        generated_power = _band_power(generated_array, sample_rate_hz=sample_rate_hz, low=low, high=high)
        errors[f"{name}_bandpower_error"] = float(abs(reference_power - generated_power))
    return errors


def emg_tone_error(reference: t.Any, generated: t.Any) -> float:
    reference_array = _as_epoch_channel_series(reference, name="reference")
    generated_array = _as_epoch_channel_series(generated, name="generated")
    _require_same_shape(reference_array, generated_array)
    return float(abs(np.sqrt(np.mean(np.square(reference_array))) - np.sqrt(np.mean(np.square(generated_array)))))


def ibi_mae(reference: t.Any, generated: t.Any) -> float:
    reference_array = _as_epoch_channel_series(reference, name="reference")
    generated_array = _as_epoch_channel_series(generated, name="generated")
    _require_same_shape(reference_array, generated_array)
    return float(np.mean(np.abs(reference_array - generated_array)))


def _contiguous_segments(mask: np.ndarray) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    start: int | None = None
    for idx, value in enumerate(np.asarray(mask, dtype=bool)):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            segments.append((start, idx))
            start = None
    if start is not None:
        segments.append((start, mask.size))
    return segments


def _desaturation_summaries(
    signal: np.ndarray,
    *,
    sample_rate_hz: int,
    drop: float = 3.0,
    min_duration_sec: float = 2.0,
) -> list[dict[str, float]]:
    summaries: list[dict[str, float]] = []
    min_frames = max(int(round(min_duration_sec * sample_rate_hz)), 1)
    for row in _flatten_epoch_series(signal):
        finite = np.isfinite(row)
        if not finite.any():
            continue
        baseline = float(np.nanmedian(row[finite]))
        for start, end in _contiguous_segments(row <= baseline - drop):
            if end - start < min_frames:
                continue
            segment = row[start:end]
            nadir_offset = int(np.nanargmin(segment))
            nadir = float(segment[nadir_offset])
            depth = baseline - nadir
            time_to_nadir_sec = max(nadir_offset / sample_rate_hz, 1.0 / sample_rate_hz)
            summaries.append(
                {
                    "depth": depth,
                    "duration": (end - start) / sample_rate_hz,
                    "slope": depth / time_to_nadir_sec,
                }
            )
    return summaries


def _summary_mae(reference: list[dict[str, float]], generated: list[dict[str, float]], key: str) -> float:
    if not reference and not generated:
        return 0.0
    return _paired_mae([item[key] for item in reference], [item[key] for item in generated])


def spo2_nadir_metrics(reference: t.Any, generated: t.Any, *, sample_rate_hz: int = 4) -> dict[str, float]:
    reference_series = _as_epoch_channel_series(reference, name="reference")
    generated_series = _as_epoch_channel_series(generated, name="generated")
    _require_same_shape(reference_series, generated_series)
    reference_array = _flatten_epoch_series(reference_series)
    generated_array = _flatten_epoch_series(generated_series)
    _require_same_shape(reference_array, generated_array)
    reference_nadir = reference_array.min(axis=-1)
    generated_nadir = generated_array.min(axis=-1)
    reference_timing = reference_array.argmin(axis=-1)
    generated_timing = generated_array.argmin(axis=-1)
    reference_desats = _desaturation_summaries(reference_series, sample_rate_hz=sample_rate_hz)
    generated_desats = _desaturation_summaries(generated_series, sample_rate_hz=sample_rate_hz)
    return {
        "nadir_error": float(np.mean(np.abs(reference_nadir - generated_nadir))),
        "nadir_timing_error": float(np.mean(np.abs(reference_timing - generated_timing))),
        "desaturation_count_error": abs(float(len(reference_desats) - len(generated_desats))),
        "desaturation_depth_error": _summary_mae(reference_desats, generated_desats, "depth"),
        "desaturation_duration_error": _summary_mae(reference_desats, generated_desats, "duration"),
        "desaturation_slope_error": _summary_mae(reference_desats, generated_desats, "slope"),
    }


def respiratory_amplitude_error(reference: t.Any, generated: t.Any) -> float:
    reference_array = _flatten_epoch_series(_as_epoch_channel_series(reference, name="reference"))
    generated_array = _flatten_epoch_series(_as_epoch_channel_series(generated, name="generated"))
    _require_same_shape(reference_array, generated_array)
    reference_amp = np.percentile(reference_array, 95, axis=-1) - np.percentile(reference_array, 5, axis=-1)
    generated_amp = np.percentile(generated_array, 95, axis=-1) - np.percentile(generated_array, 5, axis=-1)
    return float(np.mean(np.abs(reference_amp - generated_amp)))


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


def _match_peaks(reference_peaks: np.ndarray, generated_peaks: np.ndarray, *, tolerance: int) -> list[tuple[int, int]]:
    matches: list[tuple[int, int]] = []
    used_generated: set[int] = set()
    for reference_peak in reference_peaks:
        if generated_peaks.size == 0:
            continue
        distances = np.abs(generated_peaks - reference_peak)
        for generated_idx in np.argsort(distances):
            if int(generated_idx) in used_generated or distances[generated_idx] > tolerance:
                continue
            used_generated.add(int(generated_idx))
            matches.append((int(reference_peak), int(generated_peaks[generated_idx])))
            break
    return matches


def _peak_slope(signal: np.ndarray, peak: int, *, radius: int) -> float:
    start = max(peak - radius, 0)
    end = min(peak + radius + 1, signal.size)
    segment = signal[start:end]
    if segment.size < 2:
        return float("nan")
    return float(np.max(np.abs(np.diff(segment))))


def ecg_peak_metrics(reference: t.Any, generated: t.Any, *, sample_rate_hz: int = 128) -> dict[str, float]:
    reference_array = _flatten_epoch_series(_as_epoch_channel_series(reference, name="reference"))
    generated_array = _flatten_epoch_series(_as_epoch_channel_series(generated, name="generated"))
    _require_same_shape(reference_array, generated_array)
    distance = max(int(0.25 * sample_rate_hz), 1)
    tolerance = max(int(0.15 * sample_rate_hz), 1)
    slope_radius = max(int(0.04 * sample_rate_hz), 1)
    count_errors: list[float] = []
    timing_errors: list[float] = []
    rr_errors: list[float] = []
    amplitude_errors: list[float] = []
    slope_errors: list[float] = []
    for reference_epoch, generated_epoch in zip(reference_array, generated_array):
        reference_peaks = _find_peaks(reference_epoch, distance=distance)
        generated_peaks = _find_peaks(generated_epoch, distance=distance)
        count_errors.append(abs(float(len(reference_peaks) - len(generated_peaks))))
        if reference_peaks.size > 1 and generated_peaks.size > 1:
            rr_errors.append(
                _paired_mae(
                    np.diff(reference_peaks) / sample_rate_hz,
                    np.diff(generated_peaks) / sample_rate_hz,
                )
            )
        for reference_peak, generated_peak in _match_peaks(
            reference_peaks,
            generated_peaks,
            tolerance=tolerance,
        ):
            timing_errors.append(abs(float(generated_peak - reference_peak)))
            amplitude_errors.append(abs(float(reference_epoch[reference_peak] - generated_epoch[generated_peak])))
            slope_errors.append(
                abs(
                    _peak_slope(reference_epoch, reference_peak, radius=slope_radius)
                    - _peak_slope(generated_epoch, generated_peak, radius=slope_radius)
                )
            )
    return {
        "peak_count_error": float(np.mean(count_errors)) if count_errors else 0.0,
        "peak_timing_error": _finite_mean(timing_errors),
        "rr_interval_mae": _finite_mean(rr_errors),
        "peak_amplitude_error": _finite_mean(amplitude_errors),
        "qrs_slope_error": _finite_mean(slope_errors),
    }


def _cycle_amplitudes(signal: np.ndarray, peaks: np.ndarray, troughs: np.ndarray) -> list[float]:
    amplitudes: list[float] = []
    for peak in peaks:
        nearby_troughs: list[float] = []
        before = troughs[troughs < peak]
        after = troughs[troughs > peak]
        if before.size:
            nearby_troughs.append(float(signal[before[-1]]))
        if after.size:
            nearby_troughs.append(float(signal[after[0]]))
        if nearby_troughs:
            amplitudes.append(float(signal[peak] - min(nearby_troughs)))
    return amplitudes


def _inspiration_expiration_ratios(peaks: np.ndarray, troughs: np.ndarray) -> list[float]:
    ratios: list[float] = []
    for peak in peaks:
        before = troughs[troughs < peak]
        after = troughs[troughs > peak]
        if not before.size or not after.size:
            continue
        inspiration = peak - before[-1]
        expiration = after[0] - peak
        if expiration > 0:
            ratios.append(float(inspiration / expiration))
    return ratios


def respiratory_cycle_metrics(reference: t.Any, generated: t.Any, *, sample_rate_hz: int = 4) -> dict[str, float]:
    reference_array = _flatten_epoch_series(_as_epoch_channel_series(reference, name="reference"))
    generated_array = _flatten_epoch_series(_as_epoch_channel_series(generated, name="generated"))
    _require_same_shape(reference_array, generated_array)
    distance = max(int(1.5 * sample_rate_hz), 1)
    tolerance = max(int(0.5 * sample_rate_hz), 1)
    count_errors: list[float] = []
    peak_timing_errors: list[float] = []
    trough_timing_errors: list[float] = []
    amplitude_errors: list[float] = []
    ratio_errors: list[float] = []
    for reference_epoch, generated_epoch in zip(reference_array, generated_array):
        reference_peaks = _find_peaks(reference_epoch, distance=distance)
        generated_peaks = _find_peaks(generated_epoch, distance=distance)
        reference_troughs = _find_peaks(-reference_epoch, distance=distance)
        generated_troughs = _find_peaks(-generated_epoch, distance=distance)
        count_errors.append(abs(float(len(reference_peaks) - len(generated_peaks))))
        peak_timing_errors.extend(
            abs(float(generated_peak - reference_peak))
            for reference_peak, generated_peak in _match_peaks(
                reference_peaks,
                generated_peaks,
                tolerance=tolerance,
            )
        )
        trough_timing_errors.extend(
            abs(float(generated_trough - reference_trough))
            for reference_trough, generated_trough in _match_peaks(
                reference_troughs,
                generated_troughs,
                tolerance=tolerance,
            )
        )
        amplitude_errors.append(
            _paired_mae(
                _cycle_amplitudes(reference_epoch, reference_peaks, reference_troughs),
                _cycle_amplitudes(generated_epoch, generated_peaks, generated_troughs),
            )
        )
        ratio_errors.append(
            _paired_mae(
                _inspiration_expiration_ratios(reference_peaks, reference_troughs),
                _inspiration_expiration_ratios(generated_peaks, generated_troughs),
            )
        )
    return {
        "cycle_count_error": float(np.mean(count_errors)) if count_errors else 0.0,
        "peak_timing_error": _finite_mean(peak_timing_errors),
        "trough_timing_error": _finite_mean(trough_timing_errors),
        "cycle_amplitude_error": _finite_mean(amplitude_errors),
        "inspiration_expiration_ratio_error": _finite_mean(ratio_errors),
    }


def _safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 2 or np.std(left) == 0.0 or np.std(right) == 0.0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def airflow_belt_coherence_metrics(
    reference_airflow: t.Any,
    generated_airflow: t.Any,
    reference_belt: t.Any,
    generated_belt: t.Any,
) -> dict[str, float]:
    reference_airflow_array = _flatten_epoch_series(
        _as_epoch_channel_series(reference_airflow, name="reference_airflow")
    )
    generated_airflow_array = _flatten_epoch_series(
        _as_epoch_channel_series(generated_airflow, name="generated_airflow")
    )
    reference_belt_array = _flatten_epoch_series(_as_epoch_channel_series(reference_belt, name="reference_belt"))
    generated_belt_array = _flatten_epoch_series(_as_epoch_channel_series(generated_belt, name="generated_belt"))
    _require_same_shape(reference_airflow_array, generated_airflow_array)
    _require_same_shape(reference_airflow_array, reference_belt_array)
    _require_same_shape(reference_airflow_array, generated_belt_array)
    reference_coherence = _finite_mean(
        [_safe_correlation(airflow, belt) for airflow, belt in zip(reference_airflow_array, reference_belt_array)]
    )
    generated_coherence = _finite_mean(
        [_safe_correlation(airflow, belt) for airflow, belt in zip(generated_airflow_array, generated_belt_array)]
    )
    return {
        "reference_coherence": reference_coherence,
        "generated_coherence": generated_coherence,
        "coherence_error": abs(reference_coherence - generated_coherence),
    }


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
        return spo2_nadir_metrics(reference, generated, sample_rate_hz=sample_rate_hz)
    if modality in {"airflow", "belt", "resp"}:
        return {
            "respiratory_amplitude_error": respiratory_amplitude_error(reference, generated),
            **respiratory_cycle_metrics(reference, generated, sample_rate_hz=sample_rate_hz),
        }
    if modality == "ecg":
        return ecg_peak_metrics(reference, generated, sample_rate_hz=sample_rate_hz)
    return {}


__all__ = [
    "airflow_belt_coherence_metrics",
    "compute_feature_metrics",
    "ecg_peak_metrics",
    "eeg_bandpower_error",
    "emg_tone_error",
    "ibi_mae",
    "respiratory_amplitude_error",
    "respiratory_cycle_metrics",
    "spo2_nadir_metrics",
]
