from __future__ import annotations

import math
import typing as t

import numpy as np


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
    }
    if baseline is not None:
        metrics["snr_improvement"] = snr_improvement(reference, generated, baseline)
    return metrics


__all__ = [
    "compute_waveform_metrics",
    "correlation",
    "mae",
    "min_mae",
    "min_rmse",
    "rmse",
    "snr_improvement",
    "spectral_distance",
]
