from __future__ import annotations

import logging
from typing import Any, Mapping

import numpy as np
from sklearn.metrics import average_precision_score

from sleep2expert.metrics.core import binary_sequence_to_segments, filter_segments_by_duration, vectorized_event_stats

# Source contract: these names and their order describe the source/exporter taxonomy used by this task;
# they are not a universal AASM arousal subtype ontology.
AROUSAL_SUBTYPES = ("RES", "SPONT", "Limb", "PLM")
AROUSAL_SUBTYPE_SLUGS = ("res", "spont", "limb", "plm")
AROUSAL_THRESHOLD_GRID = tuple(round(float(value), 2) for value in np.arange(0.01, 1.0, 0.01))
AROUSAL_MIN_EVENT_DURATION_SECONDS = 3
AROUSAL_MIN_TST_HOURS = 2.0
AROUSAL_SECONDS_PER_TOKEN = 30

AROUSAL_SUBTYPE_INDEX_KEYS = (
    "arousal_res_index_per_hour",
    "arousal_spont_index_per_hour",
    "arousal_limb_index_per_hour",
    "arousal_plm_index_per_hour",
)
AROUSAL_TOTAL_INDEX_KEY = "arousal_index_per_hour"


def validate_arousal_thresholds(thresholds: Mapping[str, float]) -> dict[str, float]:
    if not isinstance(thresholds, Mapping):
        raise ValueError("Arousal thresholds must be a subtype-to-threshold mapping.")
    if set(thresholds) != set(AROUSAL_SUBTYPES):
        raise ValueError(f"Arousal thresholds must contain exactly {list(AROUSAL_SUBTYPES)}.")
    resolved = {name: float(thresholds[name]) for name in AROUSAL_SUBTYPES}
    if any(not np.isfinite(value) or value < 0.0 or value > 1.0 for value in resolved.values()):
        raise ValueError("Arousal thresholds must be finite values in [0, 1].")
    return resolved


def validate_arousal_fallback_subtypes(subtypes) -> list[str]:
    values = [str(value) for value in subtypes]
    if len(values) != len(set(values)) or any(value not in AROUSAL_SUBTYPES for value in values):
        raise ValueError(f"Arousal fallback subtypes must be a unique subset of {list(AROUSAL_SUBTYPES)}.")
    return [name for name in AROUSAL_SUBTYPES if name in values]


def validate_arousal_threshold_protocol(
    thresholds: Mapping[str, float], fallback_subtypes
) -> tuple[dict[str, float], list[str]]:
    resolved_thresholds = validate_arousal_thresholds(thresholds)
    resolved_fallbacks = validate_arousal_fallback_subtypes(fallback_subtypes)
    if any(resolved_thresholds[subtype] != 0.5 for subtype in resolved_fallbacks):
        raise ValueError("Every arousal fallback subtype must have stored threshold 0.5.")
    return resolved_thresholds, resolved_fallbacks


def _validate_arousal_record(record: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    truth_values = np.asarray(record["truth"])
    score = np.asarray(record["score"], dtype=np.float32)
    if truth_values.ndim != 2 or truth_values.shape[1] != len(AROUSAL_SUBTYPES):
        raise ValueError(f"Arousal truth must have shape [T, 4], got {truth_values.shape}.")
    if score.shape != truth_values.shape:
        raise ValueError(f"Arousal truth/score shapes must match, got {truth_values.shape} and {score.shape}.")
    if not np.isfinite(truth_values).all() or not np.isin(truth_values, (0, 1)).all():
        raise ValueError("Arousal truth must contain only finite, exact 0/1 values after padding is removed.")
    if not np.isfinite(score).all():
        raise ValueError("Arousal scores must be finite.")
    truth = truth_values.astype(np.int64, copy=False)

    tst_hours = float(record["tst_hours"])
    if not np.isfinite(tst_hours) or tst_hours <= 0.0:
        raise ValueError(f"Arousal TST must be finite and > 0 hours, got {tst_hours}.")
    true_indices = np.asarray(
        [*[float(record[key]) for key in AROUSAL_SUBTYPE_INDEX_KEYS], float(record[AROUSAL_TOTAL_INDEX_KEY])],
        dtype=np.float64,
    )
    if not np.isfinite(true_indices).all() or (true_indices < 0.0).any():
        raise ValueError("Arousal index-per-hour ground truths must be finite and >= 0.")
    return truth, score, tst_hours, true_indices


def merge_arousal_window_records(records: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    passthrough: list[dict[str, Any]] = []
    for record in records:
        if "path" not in record or "token_start" not in record:
            passthrough.append(dict(record))
            continue
        grouped.setdefault(str(record["path"]), []).append(record)

    merged = passthrough
    scalar_keys = (*AROUSAL_SUBTYPE_INDEX_KEYS, AROUSAL_TOTAL_INDEX_KEY)
    for path, items in grouped.items():
        ordered = sorted(items, key=lambda item: int(item["token_start"]))
        merged_truth: list[np.ndarray] = []
        merged_score: list[np.ndarray] = []
        token_starts: list[int] = []
        scalar_values: dict[str, float] = {}
        tst_hours: float | None = None
        expected_next_start: int | None = None
        previous_token_start: int | None = None
        previous_window: dict[str, Any] | None = None

        for item in ordered:
            token_start = int(item["token_start"])
            truth, score, current_tst, true_indices = _validate_arousal_record(item)
            token_count = int(item.get("n_tokens", 0))
            if token_count <= 0:
                if truth.shape[0] % AROUSAL_SECONDS_PER_TOKEN != 0:
                    raise ValueError(
                        f"Arousal window for path {path} must span whole 30-second tokens, "
                        f"got {truth.shape[0]} seconds."
                    )
                token_count = truth.shape[0] // AROUSAL_SECONDS_PER_TOKEN
            if truth.shape[0] != token_count * AROUSAL_SECONDS_PER_TOKEN:
                raise ValueError(
                    f"Arousal n_tokens does not match target length for path {path}: "
                    f"{token_count} tokens vs {truth.shape[0]} seconds."
                )
            if previous_token_start is not None and token_start == previous_token_start:
                sample_id = item.get("sample_id")
                previous_sample_id = previous_window["sample_id"]
                if sample_id is None or previous_sample_id is None:
                    raise ValueError(
                        f"Duplicate arousal windows for path {path} token_start={token_start} require sample_id."
                    )
                if sample_id != previous_sample_id:
                    raise ValueError(
                        f"Arousal windows for path {path} token_start={token_start} have different sample_id values."
                    )
                if (
                    token_count != previous_window["token_count"]
                    or not np.array_equal(truth, previous_window["truth"])
                    or not np.allclose(score, previous_window["score"])
                    or not np.isclose(current_tst, previous_window["tst_hours"])
                    or not np.allclose(true_indices, previous_window["true_indices"])
                ):
                    raise ValueError(
                        f"Conflicting duplicate arousal window for path {path}, sample_id={sample_id!r}, "
                        f"token_start={token_start}."
                    )
                continue
            if expected_next_start is not None and token_start != expected_next_start:
                raise ValueError(
                    f"Arousal windows for path {path} are not contiguous and non-overlapping: "
                    f"expected token_start={expected_next_start}, got {token_start}."
                )
            if tst_hours is None:
                tst_hours = current_tst
            elif not np.isclose(tst_hours, current_tst):
                raise ValueError(f"Inconsistent scalar 'tst' across arousal windows for path {path}.")
            for key in scalar_keys:
                current = float(item[key])
                if key not in scalar_values:
                    scalar_values[key] = current
                elif not np.isclose(scalar_values[key], current):
                    raise ValueError(f"Inconsistent scalar '{key}' across arousal windows for path {path}.")

            merged_truth.append(truth)
            merged_score.append(score)
            token_starts.append(token_start)
            expected_next_start = token_start + token_count
            previous_token_start = token_start
            previous_window = {
                "sample_id": item.get("sample_id"),
                "token_count": token_count,
                "truth": truth,
                "score": score,
                "tst_hours": current_tst,
                "true_indices": true_indices,
            }

        if not merged_truth:
            continue
        merged.append(
            {
                "path": path,
                "truth": np.concatenate(merged_truth, axis=0),
                "score": np.concatenate(merged_score, axis=0),
                "tst_hours": float(tst_hours),
                "n_windows": len(token_starts),
                "token_starts": token_starts,
                **scalar_values,
            }
        )
    return merged


def _truth_segments(binary: np.ndarray) -> list[list[int]]:
    return binary_sequence_to_segments(np.asarray(binary, dtype=np.int64), interval=1)


def _prediction_segments(binary: np.ndarray) -> list[list[int]]:
    return filter_segments_by_duration(
        binary_sequence_to_segments(np.asarray(binary, dtype=np.int64), interval=1),
        min_duration=AROUSAL_MIN_EVENT_DURATION_SECONDS,
    )


def _segments_to_mask(segments: list[list[int]], length: int) -> np.ndarray:
    mask = np.zeros(length, dtype=np.int64)
    for start, end in segments:
        mask[int(start) : int(end) + 1] = 1
    return mask


def _precision_recall_f1(tp: float, fp: float, fn: float) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return float(precision), float(recall), float(f1)


def _safe_pearson(truth: list[float], prediction: list[float]) -> float:
    if len(truth) < 2 or np.std(truth) == 0.0 or np.std(prediction) == 0.0:
        return float("nan")
    return float(np.corrcoef(np.asarray(truth), np.asarray(prediction))[0, 1])


def evaluate_arousal_record(record: Mapping[str, Any], thresholds: Mapping[str, float]) -> dict[str, Any]:
    thresholds = validate_arousal_thresholds(thresholds)
    truth, score, tst_hours, true_indices = _validate_arousal_record(record)
    prediction = np.zeros_like(truth)
    gt_segments: list[list[list[int]]] = []
    pred_segments: list[list[list[int]]] = []

    # Evaluation contract: threshold each subtype independently, remove predicted runs shorter than 3 seconds,
    # and do not gap-merge. This is a project evaluation rule, not an AASM scoring rule. Methodological
    # context: Brink-Kjaer et al., Clin Neurophysiol. 2020;131(6):1187-1203.
    # doi:10.1016/j.clinph.2020.02.027.
    for subtype_idx, subtype in enumerate(AROUSAL_SUBTYPES):
        subtype_gt_segments = _truth_segments(truth[:, subtype_idx])
        subtype_pred_segments = _prediction_segments(score[:, subtype_idx] > thresholds[subtype])
        prediction[:, subtype_idx] = _segments_to_mask(subtype_pred_segments, truth.shape[0])
        gt_segments.append(subtype_gt_segments)
        pred_segments.append(subtype_pred_segments)

    total_truth = truth.any(axis=1).astype(np.int64)
    total_prediction = prediction.any(axis=1).astype(np.int64)
    total_gt_segments = binary_sequence_to_segments(total_truth, interval=1)
    total_pred_segments = binary_sequence_to_segments(total_prediction, interval=1)
    pred_indices = np.asarray(
        [*[len(value) / tst_hours for value in pred_segments], len(total_pred_segments) / tst_hours],
        dtype=np.float64,
    )
    return {
        "truth": truth,
        "score": score,
        "prediction": prediction,
        "total_truth": total_truth,
        "total_prediction": total_prediction,
        "gt_segments": gt_segments,
        "pred_segments": pred_segments,
        "total_gt_segments": total_gt_segments,
        "total_pred_segments": total_pred_segments,
        "true_indices": true_indices,
        "pred_indices": pred_indices,
        "tst_hours": tst_hours,
    }


def select_arousal_thresholds(
    records: list[Mapping[str, Any]],
    *,
    search_thresholds: tuple[float, ...] = AROUSAL_THRESHOLD_GRID,
) -> tuple[dict[str, float], list[str]]:
    logical_records = merge_arousal_window_records(records)
    if not logical_records:
        raise ValueError("Unable to fit arousal thresholds without validation records.")
    candidates = tuple(float(value) for value in search_thresholds)
    if not candidates or any(not np.isfinite(value) or value < 0.0 or value > 1.0 for value in candidates):
        raise ValueError("Arousal threshold search grid must contain finite values in [0, 1].")

    prepared = []
    for record in logical_records:
        truth, score, _, _ = _validate_arousal_record(record)
        prepared.append((truth, score))

    selected: dict[str, float] = {}
    fallback_subtypes: list[str] = []
    for subtype_idx, subtype in enumerate(AROUSAL_SUBTYPES):
        # Evaluation contract: validation fits each subtype independently using pooled event F1;
        # threshold=0.0 delegates one-to-one any-positive-overlap matching to the shared matcher.
        gt_segments = [_truth_segments(truth[:, subtype_idx]) for truth, _ in prepared]
        support = sum(len(value) for value in gt_segments)
        if support == 0:
            selected[subtype] = 0.5
            fallback_subtypes.append(subtype)
            logging.warning(
                "Arousal validation has no ground-truth %s events; using threshold 0.5 and event F1=0.", subtype
            )
            continue

        best_threshold: float | None = None
        best_f1 = -1.0
        for threshold in candidates:
            tp = fp = fn = 0.0
            for (_, score), record_gt_segments in zip(prepared, gt_segments):
                record_pred_segments = _prediction_segments(score[:, subtype_idx] > threshold)
                record_tp, record_fp, record_fn = vectorized_event_stats(
                    record_gt_segments, record_pred_segments, threshold=0.0
                )
                tp += record_tp
                fp += record_fp
                fn += record_fn
            _, _, event_f1 = _precision_recall_f1(tp, fp, fn)
            if event_f1 > best_f1 or (event_f1 == best_f1 and (best_threshold is None or threshold > best_threshold)):
                best_f1 = event_f1
                best_threshold = threshold
        selected[subtype] = float(best_threshold)
    return validate_arousal_thresholds(selected), fallback_subtypes


def compute_arousal_metrics(
    records: list[Mapping[str, Any]],
    *,
    thresholds: Mapping[str, float] | None = None,
    search_thresholds: tuple[float, ...] = AROUSAL_THRESHOLD_GRID,
) -> tuple[dict[str, float], dict[str, float], list[str]]:
    logical_records = merge_arousal_window_records(records)
    if not logical_records:
        raise ValueError("Unable to compute arousal metrics without records.")
    if thresholds is None:
        thresholds, fallback_subtypes = select_arousal_thresholds(logical_records, search_thresholds=search_thresholds)
    else:
        thresholds = validate_arousal_thresholds(thresholds)
        fallback_subtypes = []

    evaluated = [evaluate_arousal_record(record, thresholds) for record in logical_records]
    metrics: dict[str, float] = {}
    pointwise_values = {"auprc": [], "precision": [], "recall": [], "f1": []}
    event_values = {"precision": [], "recall": [], "f1": []}

    for subtype_idx, (subtype, slug) in enumerate(zip(AROUSAL_SUBTYPES, AROUSAL_SUBTYPE_SLUGS)):
        truth = np.concatenate([item["truth"][:, subtype_idx] for item in evaluated])
        score = np.concatenate([item["score"][:, subtype_idx] for item in evaluated])
        prediction = np.concatenate([item["prediction"][:, subtype_idx] for item in evaluated])
        point_tp = float(((truth == 1) & (prediction == 1)).sum())
        point_fp = float(((truth == 0) & (prediction == 1)).sum())
        point_fn = float(((truth == 1) & (prediction == 0)).sum())
        point_precision, point_recall, point_f1 = _precision_recall_f1(point_tp, point_fp, point_fn)
        point_auprc = float("nan")
        if int((truth == 1).sum()) > 0:
            point_auprc = float(average_precision_score(truth, score))

        event_tp = event_fp = event_fn = 0.0
        event_support = 0
        for item in evaluated:
            gt_segments = item["gt_segments"][subtype_idx]
            pred_segments = item["pred_segments"][subtype_idx]
            tp, fp, fn = vectorized_event_stats(gt_segments, pred_segments, threshold=0.0)
            event_tp += tp
            event_fp += fp
            event_fn += fn
            event_support += len(gt_segments)
        event_precision, event_recall, event_f1 = _precision_recall_f1(event_tp, event_fp, event_fn)

        metrics.update(
            {
                f"arousal_{slug}_pointwise_auprc": point_auprc,
                f"arousal_{slug}_pointwise_precision": point_precision,
                f"arousal_{slug}_pointwise_recall": point_recall,
                f"arousal_{slug}_pointwise_f1": point_f1,
                f"arousal_{slug}_event_precision": event_precision,
                f"arousal_{slug}_event_recall": event_recall,
                f"arousal_{slug}_event_f1": event_f1,
                f"arousal_{slug}_event_support": float(event_support),
                f"arousal_{slug}_opt_threshold": float(thresholds[subtype]),
            }
        )
        pointwise_values["auprc"].append(point_auprc)
        pointwise_values["precision"].append(point_precision)
        pointwise_values["recall"].append(point_recall)
        pointwise_values["f1"].append(point_f1)
        event_values["precision"].append(event_precision)
        event_values["recall"].append(event_recall)
        event_values["f1"].append(event_f1)

    for key, values in pointwise_values.items():
        finite = [value for value in values if np.isfinite(value)]
        metrics[f"arousal_subtype_macro_pointwise_{key}"] = float(np.mean(finite)) if finite else float("nan")
    for key, values in event_values.items():
        metrics[f"arousal_subtype_macro_event_{key}"] = float(np.mean(values))

    total_truth = np.concatenate([item["total_truth"] for item in evaluated])
    total_prediction = np.concatenate([item["total_prediction"] for item in evaluated])
    point_tp = float(((total_truth == 1) & (total_prediction == 1)).sum())
    point_fp = float(((total_truth == 0) & (total_prediction == 1)).sum())
    point_fn = float(((total_truth == 1) & (total_prediction == 0)).sum())
    total_point_precision, total_point_recall, total_point_f1 = _precision_recall_f1(point_tp, point_fp, point_fn)

    event_tp = event_fp = event_fn = 0.0
    total_support = 0
    for item in evaluated:
        tp, fp, fn = vectorized_event_stats(item["total_gt_segments"], item["total_pred_segments"], threshold=0.0)
        event_tp += tp
        event_fp += fp
        event_fn += fn
        total_support += len(item["total_gt_segments"])
    total_event_precision, total_event_recall, total_event_f1 = _precision_recall_f1(event_tp, event_fp, event_fn)
    metrics.update(
        {
            "arousal_total_pointwise_precision": total_point_precision,
            "arousal_total_pointwise_recall": total_point_recall,
            "arousal_total_pointwise_f1": total_point_f1,
            "arousal_total_event_precision": total_event_precision,
            "arousal_total_event_recall": total_event_recall,
            "arousal_total_event_f1": total_event_f1,
            "arousal_total_event_support": float(total_support),
        }
    )

    index_slugs = (*AROUSAL_SUBTYPE_SLUGS, "total")
    for index_idx, slug in enumerate(index_slugs):
        # Evaluation contract: the >=2-hour gate applies only to ArI summary statistics; all valid
        # TST records remain in pointwise and event detection metrics. This is a project evaluation rule.
        true_values = [
            float(item["true_indices"][index_idx]) for item in evaluated if item["tst_hours"] >= AROUSAL_MIN_TST_HOURS
        ]
        pred_values = [
            float(item["pred_indices"][index_idx]) for item in evaluated if item["tst_hours"] >= AROUSAL_MIN_TST_HOURS
        ]
        prefix = "arousal_index_per_hour" if slug == "total" else f"arousal_{slug}_index_per_hour"
        if not true_values:
            metrics[f"{prefix}_mae"] = float("nan")
            metrics[f"{prefix}_bias"] = float("nan")
            metrics[f"{prefix}_pearson"] = float("nan")
            continue
        true_array = np.asarray(true_values)
        pred_array = np.asarray(pred_values)
        metrics[f"{prefix}_mae"] = float(np.mean(np.abs(pred_array - true_array)))
        metrics[f"{prefix}_bias"] = float(np.mean(pred_array - true_array))
        metrics[f"{prefix}_pearson"] = _safe_pearson(true_values, pred_values)

    return metrics, dict(thresholds), fallback_subtypes
