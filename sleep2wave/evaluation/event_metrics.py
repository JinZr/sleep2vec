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


__all__ = [
    "compute_event_metric_groups",
    "compute_event_metrics",
    "interval_iou",
    "match_intervals",
]
