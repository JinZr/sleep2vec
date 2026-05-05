from __future__ import annotations

import typing as t

import numpy as np

Interval = tuple[float, float]


def _normalize_intervals(intervals: t.Sequence[t.Sequence[float]], *, name: str) -> list[Interval]:
    normalized: list[Interval] = []
    for interval in intervals:
        if len(interval) != 2:
            raise ValueError(f"{name} intervals must contain exactly two values.")
        start = float(interval[0])
        end = float(interval[1])
        if end < start:
            raise ValueError(f"{name} interval end must be >= start.")
        normalized.append((start, end))
    return normalized


def interval_iou(reference: Interval, generated: Interval) -> float:
    left = max(reference[0], generated[0])
    right = min(reference[1], generated[1])
    intersection = max(right - left, 0.0)
    union = max(reference[1], generated[1]) - min(reference[0], generated[0])
    return float(intersection / union) if union > 0.0 else 0.0


def match_intervals(
    reference_intervals: t.Sequence[t.Sequence[float]],
    generated_intervals: t.Sequence[t.Sequence[float]],
    *,
    iou_threshold: float = 0.5,
) -> list[tuple[int, int, float]]:
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be in [0, 1].")
    reference = _normalize_intervals(reference_intervals, name="reference")
    generated = _normalize_intervals(generated_intervals, name="generated")
    candidates: list[tuple[float, int, int]] = []
    for reference_idx, reference_interval in enumerate(reference):
        for generated_idx, generated_interval in enumerate(generated):
            iou = interval_iou(reference_interval, generated_interval)
            if iou >= iou_threshold:
                candidates.append((iou, reference_idx, generated_idx))
    candidates.sort(reverse=True)
    used_reference: set[int] = set()
    used_generated: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for iou, reference_idx, generated_idx in candidates:
        if reference_idx in used_reference or generated_idx in used_generated:
            continue
        used_reference.add(reference_idx)
        used_generated.add(generated_idx)
        matches.append((reference_idx, generated_idx, float(iou)))
    matches.sort()
    return matches


def compute_event_metrics(
    reference_intervals: t.Sequence[t.Sequence[float]],
    generated_intervals: t.Sequence[t.Sequence[float]],
    *,
    iou_threshold: float = 0.5,
) -> dict[str, float]:
    reference = _normalize_intervals(reference_intervals, name="reference")
    generated = _normalize_intervals(generated_intervals, name="generated")
    matches = match_intervals(reference, generated, iou_threshold=iou_threshold)
    tp = float(len(matches))
    fp = float(len(generated) - len(matches))
    fn = float(len(reference) - len(matches))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    if matches:
        ious = np.asarray([match[2] for match in matches], dtype=np.float64)
        onset_errors = np.asarray(
            [abs(reference[ref_idx][0] - generated[gen_idx][0]) for ref_idx, gen_idx, _ in matches],
            dtype=np.float64,
        )
        offset_errors = np.asarray(
            [abs(reference[ref_idx][1] - generated[gen_idx][1]) for ref_idx, gen_idx, _ in matches],
            dtype=np.float64,
        )
        mean_iou = float(np.mean(ious))
        onset_mae = float(np.mean(onset_errors))
        offset_mae = float(np.mean(offset_errors))
    else:
        mean_iou = 0.0
        onset_mae = float("nan")
        offset_mae = float("nan")
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "mean_iou": mean_iou,
        "onset_mae": onset_mae,
        "offset_mae": offset_mae,
    }


def compute_event_metric_groups(
    events: dict[str, dict[str, t.Any]],
    *,
    iou_threshold: float,
) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    for event_name, event_payload in events.items():
        reference = event_payload.get("reference", event_payload.get("truth"))
        generated = event_payload.get("generated", event_payload.get("prediction"))
        if reference is None or generated is None:
            raise ValueError(f"Event '{event_name}' must define reference/generated intervals.")
        metrics[event_name] = compute_event_metrics(reference, generated, iou_threshold=iou_threshold)
    return metrics


def _contiguous_true_intervals(mask: np.ndarray, *, sample_rate_hz: int) -> list[list[float]]:
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    intervals: list[list[float]] = []
    start: int | None = None
    for idx, value in enumerate(mask):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            intervals.append([start / sample_rate_hz, idx / sample_rate_hz])
            start = None
    if start is not None:
        intervals.append([start / sample_rate_hz, mask.size / sample_rate_hz])
    return intervals


def _flatten_signal(signal: t.Any) -> np.ndarray:
    array = np.asarray(signal, dtype=np.float64)
    if array.ndim == 1:
        return array
    if array.ndim == 2:
        return array.reshape(-1)
    if array.ndim >= 3:
        return array[:, 0, :].reshape(-1)
    raise ValueError("Signal must be at least 1D.")


def detect_spo2_desaturation_events(signal: t.Any, *, sample_rate_hz: int, drop: float = 3.0) -> list[list[float]]:
    values = _flatten_signal(signal)
    finite = np.isfinite(values)
    if not finite.any():
        return []
    baseline = float(np.nanmedian(values[finite]))
    return _contiguous_true_intervals(values <= baseline - drop, sample_rate_hz=sample_rate_hz)


def detect_low_amplitude_epoch_events(
    signal: t.Any,
    *,
    sample_rate_hz: int,
    epoch_sec: int = 30,
    fraction: float = 0.5,
) -> list[list[float]]:
    array = np.asarray(signal, dtype=np.float64)
    if array.ndim == 1:
        frames_per_epoch = int(epoch_sec * sample_rate_hz)
        array = array[: array.size // frames_per_epoch * frames_per_epoch].reshape(-1, frames_per_epoch)
    elif array.ndim >= 3:
        array = array[:, 0, :]
    elif array.ndim != 2:
        return []
    if array.size == 0:
        return []
    amplitude = np.nanpercentile(array, 95, axis=-1) - np.nanpercentile(array, 5, axis=-1)
    finite = np.isfinite(amplitude)
    if not finite.any():
        return []
    threshold = float(np.nanmedian(amplitude[finite]) * fraction)
    return [[idx * epoch_sec, (idx + 1) * epoch_sec] for idx, value in enumerate(amplitude) if value <= threshold]


def compute_generated_signal_event_groups(
    reference_by_modality: dict[str, t.Any],
    generated_by_modality: dict[str, t.Any],
    *,
    sample_rates: dict[str, int],
    iou_threshold: float,
) -> dict[str, dict[str, float]]:
    groups: dict[str, dict[str, t.Any]] = {}
    if "spo2" in reference_by_modality and "spo2" in generated_by_modality:
        groups["spo2_desaturation"] = {
            "reference": detect_spo2_desaturation_events(
                reference_by_modality["spo2"],
                sample_rate_hz=sample_rates["spo2"],
            ),
            "generated": detect_spo2_desaturation_events(
                generated_by_modality["spo2"],
                sample_rate_hz=sample_rates["spo2"],
            ),
        }
    for modality in ("airflow", "belt", "resp"):
        if modality in reference_by_modality and modality in generated_by_modality:
            groups[f"{modality}_low_amplitude"] = {
                "reference": detect_low_amplitude_epoch_events(
                    reference_by_modality[modality],
                    sample_rate_hz=sample_rates[modality],
                ),
                "generated": detect_low_amplitude_epoch_events(
                    generated_by_modality[modality],
                    sample_rate_hz=sample_rates[modality],
                ),
            }
    return compute_event_metric_groups(groups, iou_threshold=iou_threshold) if groups else {}


__all__ = [
    "compute_generated_signal_event_groups",
    "compute_event_metric_groups",
    "compute_event_metrics",
    "detect_low_amplitude_epoch_events",
    "detect_spo2_desaturation_events",
    "interval_iou",
    "match_intervals",
]
