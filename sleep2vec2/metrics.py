from dataclasses import dataclass
import logging
import time
from typing import Any, Mapping

import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

AHI_COARSE_THRESHOLD_GRID = tuple(round(float(x), 2) for x in np.arange(0.1, 1.0, 0.1))
AHI_FINE_THRESHOLD_GRID = tuple(round(float(x), 2) for x in np.arange(0.01, 1.0, 0.01))
AHI_THRESHOLD_GRID = AHI_FINE_THRESHOLD_GRID
AHI_SEVERITY_THRESHOLDS = (5.0, 15.0, 30.0)
AHI_SEGMENT_MERGE_TOLERANCE = 3
AHI_MIN_EVENT_DURATION = 10
AHI_MIN_TST_HOURS = 2.0


def binary_positive_scores_from_two_logits(gts, preds) -> tuple[np.ndarray, np.ndarray]:
    """Return binary labels and positive-class scores from (N, 2) logits or probabilities."""
    y_true = np.asarray(gts)
    y_pred = np.asarray(preds)

    if y_true.ndim == 2 and y_true.shape[1] == 2:
        y_true = y_true.argmax(axis=1)
    y_true = y_true.astype(int).reshape(-1)

    if y_pred.ndim != 2 or y_pred.shape[1] != 2:
        raise ValueError(f"preds must be (N, 2), got {y_pred.shape}")

    row_sum = y_pred.sum(axis=1, keepdims=True)
    looks_like_prob = y_pred.min() >= 0.0 and y_pred.max() <= 1.0 and np.allclose(row_sum, 1.0, atol=1e-4)
    if looks_like_prob:
        y_score = y_pred[:, 1].astype(np.float32)
    else:
        z = y_pred - y_pred.max(axis=1, keepdims=True)
        e = np.exp(z, dtype=np.float64)
        proba = (e / e.sum(axis=1, keepdims=True)).astype(np.float32)
        y_score = proba[:, 1]

    return y_true, y_score


def roc_auc_from_two_logits(gts, preds) -> float:
    """
    计算二分类场景下基于两列 logits/probabilities 的 ROC-AUC。
    """
    y_true, y_score = binary_positive_scores_from_two_logits(gts, preds)

    if np.unique(y_true).size < 2:
        return np.nan

    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return np.nan


def _as_numpy_array(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def compute_survival_c_index_by_disease(
    pred, event_time, is_event, has_label, disease_names=None
) -> list[dict[str, Any]]:
    pred = _as_numpy_array(pred).astype(float)
    event_time = _as_numpy_array(event_time).astype(float)
    is_event = _as_numpy_array(is_event).astype(float)
    has_label = _as_numpy_array(has_label).astype(float)
    if pred.ndim != 2:
        raise ValueError(f"survival predictions must be 2D [N, L], got {pred.shape}")
    if event_time.shape != pred.shape or is_event.shape != pred.shape or has_label.shape != pred.shape:
        raise ValueError("survival pred/event_time/is_event/has_label shapes must match.")
    if disease_names is not None and len(disease_names) != pred.shape[1]:
        raise ValueError("disease_names length must match survival prediction width.")

    from sksurv.metrics import concordance_index_censored

    rows = []
    for disease_idx in range(pred.shape[1]):
        valid = has_label[:, disease_idx] > 0.5
        n_labeled = int(valid.sum())
        events = is_event[valid, disease_idx] > 0.5 if n_labeled else np.asarray([], dtype=bool)
        n_events = int(events.sum())
        c_index = float("nan")
        if n_labeled >= 2 and n_events > 0:
            try:
                c_index = concordance_index_censored(
                    events,
                    event_time[valid, disease_idx],
                    pred[valid, disease_idx],
                )[0]
            except ValueError:
                c_index = float("nan")
        rows.append(
            {
                "disease_idx": disease_idx,
                "disease": disease_names[disease_idx] if disease_names is not None else "",
                "n_labeled": n_labeled,
                "n_events": n_events,
                "c_index": float(c_index),
            }
        )
    return rows


def compute_survival_c_index(pred, event_time, is_event, has_label) -> float:
    values = [row["c_index"] for row in compute_survival_c_index_by_disease(pred, event_time, is_event, has_label)]
    values = [float(value) for value in values if np.isfinite(value)]
    return float(np.mean(values)) if values else float("nan")


def macro_specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """逐类计算 specificity 并宏平均。"""
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    labels = np.unique(np.concatenate([y_true, y_pred], axis=0))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    total = cm.sum()
    specs = []
    for i in range(len(labels)):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = total - tp - fn - fp
        denom = tn + fp
        specs.append((tn / denom) if denom > 0 else 0.0)
    return float(np.mean(specs)) if specs else 0.0


def binary_specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = y_true.astype(int).reshape(-1)
    y_pred = y_pred.astype(int).reshape(-1)
    tn = np.logical_and(y_true == 0, y_pred == 0).sum()
    fp = np.logical_and(y_true == 0, y_pred == 1).sum()
    return float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0


def icc2_two_raters_arrays(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = a.shape[0]
    if n < 2:
        return 0.0

    k = 2
    y = np.concatenate([a, b], axis=0)
    gm = y.mean()
    mean_by_target = (a + b) / 2.0
    mean_a = a.mean()
    mean_b = b.mean()

    sst = np.square(y - gm).sum()
    ssb = k * np.square(mean_by_target - gm).sum()
    ssr = n * ((mean_a - gm) ** 2 + (mean_b - gm) ** 2)
    sse = max(sst - ssb - ssr, 0.0)

    df_subjects = n - 1
    df_raters = k - 1
    df_error = (n - 1) * (k - 1)
    if df_subjects <= 0 or df_raters <= 0 or df_error <= 0:
        return 0.0

    msb = ssb / df_subjects
    msr = ssr / df_raters
    mse = sse / df_error
    denom = msb + (k - 1) * mse + (k * (msr - mse)) / n
    if denom == 0:
        return 0.0
    return float((msb - mse) / denom)


def binary_sequence_to_segments(labels, *, interval: int = 1) -> list[list[int]]:
    cls_interval = np.asarray(labels, dtype=np.int64).reshape(-1)
    padded = np.concatenate(([0], cls_interval, [0]))
    diff = np.diff(padded)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0] - 1
    if starts.size == 0:
        return []
    segments = np.column_stack((starts * interval, ends * interval))
    return segments.tolist()


def merge_intervals(intervals, *, tolerance: int = AHI_SEGMENT_MERGE_TOLERANCE) -> list[list[int]]:
    ordered = [list(interval) for interval in intervals]
    if not ordered:
        return []
    ordered.sort(key=lambda x: x[0])
    merged = [ordered[0]]
    for current in ordered[1:]:
        previous = merged[-1]
        if current[0] <= previous[1] + tolerance:
            previous[1] = max(previous[1], current[1])
        else:
            merged.append(current)
    return merged


def filter_segments_by_stage(intervals, sleep_mask: np.ndarray) -> list[list[int]]:
    filtered: list[list[int]] = []
    mask = np.asarray(sleep_mask, dtype=np.int64).reshape(-1)
    for start, end in intervals:
        left = max(int(round(start)), 0)
        right = min(int(round(end)) + 1, mask.shape[0])
        if right > left and mask[left:right].sum() > 0:
            filtered.append([int(start), int(end)])
    return filtered


def filter_segments_by_duration(intervals, *, min_duration: int = AHI_MIN_EVENT_DURATION) -> list[list[int]]:
    return [list(interval) for interval in intervals if (interval[1] - interval[0] + 1) >= min_duration]


def vectorized_event_stats(gt_segments, pred_segments, *, threshold: float = 0.5) -> tuple[float, float, float]:
    gt_seg = np.asarray(gt_segments, dtype=np.float32)
    pre_seg = np.asarray(pred_segments, dtype=np.float32)

    if gt_seg.size == 0:
        return 0.0, float(len(pred_segments)), 0.0
    if pre_seg.size == 0:
        return 0.0, 0.0, float(len(gt_segments))

    intersect_mask = (gt_seg[:, None, 0] <= pre_seg[None, :, 1]) & (gt_seg[:, None, 1] >= pre_seg[None, :, 0])
    union = (
        np.maximum(gt_seg[:, None, 1], pre_seg[None, :, 1]) - np.minimum(gt_seg[:, None, 0], pre_seg[None, :, 0]) + 1
    )
    gt_lengths = (gt_seg[:, 1] - gt_seg[:, 0] + 1)[:, None]
    pre_lengths = (pre_seg[:, 1] - pre_seg[:, 0] + 1)[None, :]
    overlap = np.where(intersect_mask, gt_lengths + pre_lengths - union, 0.0)
    ratio = np.where(union > 0, overlap / union, 0.0)
    matched = ratio > threshold

    # IoU is computed for every GT/pred pair first, then we enforce a one-to-one
    # assignment so a single predicted event cannot claim multiple GT events and
    # drive FP negative.
    matched_gt_by_pred = np.full(pre_seg.shape[0], -1, dtype=np.int32)

    def _try_match(gt_idx: int, seen_pred: np.ndarray) -> bool:
        for pred_idx in np.flatnonzero(matched[gt_idx]):
            if seen_pred[pred_idx]:
                continue
            seen_pred[pred_idx] = True
            current_gt = matched_gt_by_pred[pred_idx]
            if current_gt == -1 or _try_match(current_gt, seen_pred):
                matched_gt_by_pred[pred_idx] = gt_idx
                return True
        return False

    tp = 0.0
    for gt_idx in range(matched.shape[0]):
        tp += float(_try_match(gt_idx, np.zeros(pre_seg.shape[0], dtype=bool)))
    fp = float(len(pred_segments) - tp)
    fn = float(len(gt_segments) - tp)
    return tp, fp, fn


def _safe_pearson(a: np.ndarray, b: np.ndarray, *, require_min_count: int = 2, nan_if_invalid: bool = False) -> float:
    if len(a) < require_min_count or len(b) < require_min_count:
        return float("nan") if nan_if_invalid else 0.0
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan") if nan_if_invalid else 0.0
    value = float(np.corrcoef(a, b)[0, 1])
    if np.isnan(value):
        return float("nan") if nan_if_invalid else 0.0
    return value


def _format_threshold_suffix(threshold: float) -> str:
    value = int(threshold) if float(threshold).is_integer() else threshold
    return str(value).replace(".", "p")


def compute_binary_label_metrics(gts, preds) -> dict[str, float]:
    y_true = np.asarray(gts, dtype=np.int64).reshape(-1)
    y_score = np.asarray(preds, dtype=np.float32).reshape(-1)
    y_pred = (y_score >= 0.5).astype(np.int64)

    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": binary_specificity(y_true, y_pred),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if np.unique(y_true).size < 2:
        result["roc_auc"] = np.nan
    else:
        try:
            result["roc_auc"] = float(roc_auc_score(y_true, y_score))
        except Exception:
            result["roc_auc"] = np.nan
    return result


def compute_multilabel_classification_metrics(labels, probs, has_label) -> dict[str, float]:
    y_true = np.asarray(labels, dtype=np.float32)
    y_score = np.asarray(probs, dtype=np.float32)
    valid = np.asarray(has_label, dtype=np.float32) > 0.5
    valid_true = y_true[valid].astype(np.int64)
    valid_score = y_score[valid]
    if valid_true.size == 0:
        return {
            "micro_accuracy": np.nan,
            "micro_precision": np.nan,
            "micro_recall": np.nan,
            "micro_f1": np.nan,
            "micro_auroc": np.nan,
            "micro_auprc": np.nan,
            "macro_auroc": np.nan,
            "macro_auprc": np.nan,
        }
    y_pred = (valid_score >= 0.5).astype(np.int64)

    result = {
        "micro_accuracy": float(accuracy_score(valid_true, y_pred)),
        "micro_precision": float(precision_score(valid_true, y_pred, zero_division=0)),
        "micro_recall": float(recall_score(valid_true, y_pred, zero_division=0)),
        "micro_f1": float(f1_score(valid_true, y_pred, zero_division=0)),
    }
    if np.unique(valid_true).size < 2:
        result["micro_auroc"] = np.nan
        result["micro_auprc"] = np.nan
    else:
        result["micro_auroc"] = float(roc_auc_score(valid_true, valid_score))
        result["micro_auprc"] = float(average_precision_score(valid_true, valid_score))

    rows = compute_multilabel_metrics_by_disease(labels, probs, has_label)
    aurocs = [row["auroc"] for row in rows if np.isfinite(row["auroc"])]
    auprcs = [row["auprc"] for row in rows if np.isfinite(row["auprc"])]
    result["macro_auroc"] = float(np.mean(aurocs)) if aurocs else np.nan
    result["macro_auprc"] = float(np.mean(auprcs)) if auprcs else np.nan
    return result


def compute_multilabel_metrics_by_disease(labels, probs, has_label, disease_names: list[str] | None = None):
    y_true = np.asarray(labels, dtype=np.float32)
    y_score = np.asarray(probs, dtype=np.float32)
    valid = np.asarray(has_label, dtype=np.float32) > 0.5
    if y_true.shape != y_score.shape or y_true.shape != valid.shape:
        raise ValueError("Multilabel labels, probabilities, and has_label arrays must have the same shape.")

    rows = []
    if disease_names is not None and len(disease_names) != y_true.shape[1]:
        raise ValueError("disease_names length must match multilabel label width.")
    for disease_idx in range(y_true.shape[1]):
        disease_valid = valid[:, disease_idx]
        labels_d = y_true[disease_valid, disease_idx].astype(np.int64)
        scores_d = y_score[disease_valid, disease_idx]
        n_positive = int((labels_d == 1).sum())
        n_negative = int((labels_d == 0).sum())
        n_labeled = n_positive + n_negative
        if n_positive == 0 or n_negative == 0:
            continue
        rows.append(
            {
                "disease_idx": disease_idx,
                "disease": "" if disease_names is None else str(disease_names[disease_idx]),
                "n_positive": n_positive,
                "n_negative": n_negative,
                "prevalence": float(n_positive / n_labeled) if n_labeled else np.nan,
                "auroc": float(roc_auc_score(labels_d, scores_d)),
                "auprc": float(average_precision_score(labels_d, scores_d)),
            }
        )
    return rows


def compute_ahi_pointwise_metrics(gts, preds) -> dict[str, float]:
    return {f"ahi_pointwise_{key}": value for key, value in compute_binary_label_metrics(gts, preds).items()}


@dataclass(frozen=True)
class PreparedAHIRecord:
    score: np.ndarray
    gt_segments: list[list[int]]
    sleep_mask: np.ndarray
    true_ahi: float
    tst_hours: float
    summary_enabled: bool


def _resolve_ahi_search_mode(search_thresholds: tuple[float, ...]) -> str:
    thresholds = tuple(float(value) for value in search_thresholds)
    if thresholds == AHI_COARSE_THRESHOLD_GRID:
        return "coarse"
    if thresholds == AHI_FINE_THRESHOLD_GRID:
        return "fine"
    return "custom"


def _should_log_ahi_search_progress(index: int, total: int, *, threshold: float, mode: str) -> bool:
    if total <= 10:
        return True
    if mode == "fine":
        return int(round(float(threshold) * 100)) % 10 == 0
    return index == 1 or index == total or index % 10 == 0


def _prepare_ahi_record(record: Mapping[str, np.ndarray]) -> PreparedAHIRecord:
    truth = np.asarray(record["truth"], dtype=np.int64).reshape(-1)
    score = np.asarray(record["score"], dtype=np.float32).reshape(-1)
    true_ahi = float(record["true_ahi"])
    tst_hours = float(record["tst_hours"])

    if truth.shape[0] != score.shape[0]:
        raise ValueError(f"AHI truth/pred length mismatch: {truth.shape[0]} vs {score.shape[0]}")
    if not np.isfinite(true_ahi) or true_ahi < 0:
        raise ValueError(f"AHI summary ground truth must be finite and >= 0, got {true_ahi}")
    if not np.isfinite(tst_hours) or tst_hours <= 0:
        raise ValueError(f"TST hours must be finite and > 0, got {tst_hours}")

    gt_segments = merge_intervals(binary_sequence_to_segments(truth, interval=1))
    gt_segments = filter_segments_by_duration(gt_segments, min_duration=AHI_MIN_EVENT_DURATION)

    stage5 = np.asarray(record["stage5"], dtype=np.int64).reshape(-1)
    second_valid_mask_raw = record.get("second_valid_mask")
    if second_valid_mask_raw is not None:
        second_valid_mask = np.asarray(second_valid_mask_raw, dtype=np.bool_).reshape(-1)
        expected_second_count = stage5.shape[0] * 30
        if second_valid_mask.shape[0] != expected_second_count:
            raise ValueError(
                "AHI second_valid_mask length must match stage5 token count: "
                f"{second_valid_mask.shape[0]} vs {expected_second_count}"
            )
        if int(second_valid_mask.sum()) != truth.shape[0]:
            raise ValueError(
                "AHI second_valid_mask valid-second count must match truth/score length: "
                f"{int(second_valid_mask.sum())} vs {truth.shape[0]}"
            )
        sleep_mask = np.repeat((stage5 > 0).astype(np.int64), 30)[second_valid_mask]
    else:
        if stage5.shape[0] * 30 != truth.shape[0]:
            raise ValueError(
                "AHI stage5 length must match the truth/score token count: "
                f"{stage5.shape[0]} tokens vs {truth.shape[0]} seconds"
            )
        sleep_mask = np.repeat((stage5 > 0).astype(np.int64), 30)

    return PreparedAHIRecord(
        score=score,
        gt_segments=gt_segments,
        sleep_mask=sleep_mask,
        true_ahi=true_ahi,
        tst_hours=tst_hours,
        summary_enabled=tst_hours >= AHI_MIN_TST_HOURS,
    )


def _prepare_ahi_records(records: list[Mapping[str, np.ndarray]]) -> list[PreparedAHIRecord]:
    return [_prepare_ahi_record(record) for record in _merge_ahi_window_records(records)]


def _evaluate_prepared_ahi_record(
    record: PreparedAHIRecord,
    *,
    threshold: float,
) -> tuple[tuple[float, float, float], float | None, float | None]:
    pred_binary = (record.score > threshold).astype(np.int64)
    raw_pred_segments = binary_sequence_to_segments(pred_binary, interval=1)
    pred_segments = merge_intervals(raw_pred_segments)
    summary_pred_segments = filter_segments_by_stage(raw_pred_segments, record.sleep_mask)
    pred_segments = filter_segments_by_stage(pred_segments, record.sleep_mask)
    pred_segments = filter_segments_by_duration(pred_segments, min_duration=AHI_MIN_EVENT_DURATION)
    tp, fp, fn = vectorized_event_stats(record.gt_segments, pred_segments)
    if not record.summary_enabled:
        return (tp, fp, fn), None, None

    pred_ahi = float(len(summary_pred_segments) / record.tst_hours)
    return (tp, fp, fn), pred_ahi, record.true_ahi


def _evaluate_single_ahi_record(
    record: Mapping[str, np.ndarray],
    *,
    threshold: float,
) -> tuple[tuple[float, float, float], float | None, float | None]:
    return _evaluate_prepared_ahi_record(_prepare_ahi_record(record), threshold=threshold)


def _merge_ahi_window_records(records: list[Mapping[str, np.ndarray]]) -> list[dict[str, Any]]:
    """Normalize AHI eval records to one logical record per path.

    Under the current built-in ``ahi`` contract, finetune/infer default to whole-night inputs, so
    this helper is typically a passthrough aside from tolerating duplicate gathered windows. The
    contiguity checks only matter when a caller explicitly feeds multiple windows for the same path.
    """
    grouped: dict[str, list[Mapping[str, np.ndarray]]] = {}
    passthrough: list[dict[str, Any]] = []

    for record in records:
        if "path" not in record or "token_start" not in record:
            passthrough.append(dict(record))
            continue
        grouped.setdefault(str(record["path"]), []).append(record)

    merged: list[dict[str, Any]] = passthrough
    for path, items in grouped.items():
        ordered = sorted(items, key=lambda item: int(item["token_start"]))
        merged_truth: list[np.ndarray] = []
        merged_score: list[np.ndarray] = []
        merged_stage5: list[np.ndarray] = []
        merged_second_valid_mask: list[np.ndarray] = []
        use_second_valid_mask: bool | None = None
        expected_next_start: int | None = None
        true_ahi: float | None = None
        tst_hours: float | None = None
        previous_token_start: int | None = None

        for item in ordered:
            token_start = int(item["token_start"])
            truth = np.asarray(item["truth"], dtype=np.int64).reshape(-1)
            score = np.asarray(item["score"], dtype=np.float32).reshape(-1)
            current_true_ahi = float(item["true_ahi"])
            current_tst_hours = float(item["tst_hours"])
            if truth.shape[0] != score.shape[0]:
                raise ValueError(
                    f"AHI truth/pred length mismatch for path {path}: {truth.shape[0]} vs {score.shape[0]}"
                )
            stage5 = np.asarray(item["stage5"], dtype=np.int64).reshape(-1)
            token_count = stage5.shape[0]
            second_valid_mask_raw = item.get("second_valid_mask")
            second_valid_mask: np.ndarray | None = None
            if second_valid_mask_raw is not None:
                second_valid_mask = np.asarray(second_valid_mask_raw, dtype=np.bool_).reshape(-1)
                expected_second_count = token_count * 30
                if second_valid_mask.shape[0] != expected_second_count:
                    raise ValueError(
                        "AHI second_valid_mask length must match stage5 token count for "
                        f"path {path} token_start={token_start}: "
                        f"{second_valid_mask.shape[0]} vs {expected_second_count}"
                    )
                if int(second_valid_mask.sum()) != truth.shape[0]:
                    raise ValueError(
                        "AHI second_valid_mask valid-second count must match truth/score length for "
                        f"path {path} token_start={token_start}: "
                        f"{int(second_valid_mask.sum())} vs {truth.shape[0]}"
                    )
            elif truth.shape[0] != token_count * 30:
                raise ValueError(
                    "AHI stage5 length must match the truth/score token count for "
                    f"path {path} token_start={token_start}: "
                    f"{token_count} tokens vs {truth.shape[0]} seconds"
                )
            if previous_token_start is not None and token_start == previous_token_start:
                # In distributed evaluation, duplicate windows can be gathered with different
                # padding layouts. Keep the first seen record for a duplicated token_start.
                continue
            has_second_valid_mask = second_valid_mask is not None
            if use_second_valid_mask is None:
                use_second_valid_mask = has_second_valid_mask
            elif use_second_valid_mask != has_second_valid_mask:
                raise ValueError(f"AHI second_valid_mask usage is inconsistent across windows for path {path}")
            if expected_next_start is not None and token_start != expected_next_start:
                raise ValueError(
                    f"AHI windows for path {path} are not contiguous and non-overlapping: "
                    f"expected token_start={expected_next_start}, got {token_start}"
                )
            if true_ahi is None:
                true_ahi = current_true_ahi
            elif not np.isclose(true_ahi, current_true_ahi):
                raise ValueError(
                    f"Inconsistent scalar 'ahi' across windows for path {path}: {true_ahi} vs {current_true_ahi}"
                )
            if tst_hours is None:
                tst_hours = current_tst_hours
            elif not np.isclose(tst_hours, current_tst_hours):
                raise ValueError(
                    f"Inconsistent scalar 'tst' across windows for path {path}: {tst_hours} vs {current_tst_hours}"
                )

            merged_truth.append(truth)
            merged_score.append(score)
            merged_stage5.append(stage5)
            if second_valid_mask is not None:
                merged_second_valid_mask.append(second_valid_mask)
            expected_next_start = token_start + token_count
            previous_token_start = token_start

        merged_record: dict[str, Any] = {
            "path": path,
            "truth": np.concatenate(merged_truth, axis=0),
            "score": np.concatenate(merged_score, axis=0),
            "true_ahi": float(true_ahi) if true_ahi is not None else np.nan,
            "tst_hours": float(tst_hours) if tst_hours is not None else np.nan,
            "stage5": np.concatenate(merged_stage5, axis=0),
        }
        if use_second_valid_mask:
            merged_record["second_valid_mask"] = np.concatenate(merged_second_valid_mask, axis=0)
        merged.append(merged_record)

    return merged


def _aggregate_ahi_records(
    records: list[Mapping[str, np.ndarray]],
    *,
    threshold: float,
) -> dict[str, Any]:
    prepared_records = _prepare_ahi_records(records)
    return _aggregate_prepared_ahi_records(prepared_records, threshold=threshold)


def _aggregate_prepared_ahi_records(
    prepared_records: list[PreparedAHIRecord],
    *,
    threshold: float,
) -> dict[str, Any]:
    tp = fp = fn = 0.0
    pred_ahi: list[float] = []
    true_ahi: list[float] = []

    for record in prepared_records:
        (record_tp, record_fp, record_fn), record_pred_ahi, record_true_ahi = _evaluate_prepared_ahi_record(
            record,
            threshold=threshold,
        )
        tp += record_tp
        fp += record_fp
        fn += record_fn
        if record_pred_ahi is not None and record_true_ahi is not None:
            pred_ahi.append(record_pred_ahi)
            true_ahi.append(record_true_ahi)

    return {
        "event_tp": tp,
        "event_fp": fp,
        "event_fn": fn,
        "pred_ahi": np.asarray(pred_ahi, dtype=np.float32),
        "true_ahi": np.asarray(true_ahi, dtype=np.float32),
    }


def _aggregate_prepared_ahi_summary_records(
    prepared_records: list[PreparedAHIRecord],
    *,
    threshold: float,
) -> dict[str, Any]:
    pred_ahi: list[float] = []
    true_ahi: list[float] = []

    for record in prepared_records:
        if not record.summary_enabled:
            continue
        active = np.asarray(record.score > threshold, dtype=np.int8).reshape(-1)
        padded = np.concatenate(([0], active, [0]))
        diff = np.diff(padded)
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0] - 1
        if starts.size == 0:
            pred_ahi.append(0.0)
        else:
            mask = np.asarray(record.sleep_mask > 0, dtype=np.int64).reshape(-1)
            prefix = np.concatenate(([0], np.cumsum(mask)))
            overlaps = prefix[ends + 1] - prefix[starts]
            pred_ahi.append(float(np.count_nonzero(overlaps > 0) / record.tst_hours))
        true_ahi.append(record.true_ahi)

    return {
        "pred_ahi": np.asarray(pred_ahi, dtype=np.float32),
        "true_ahi": np.asarray(true_ahi, dtype=np.float32),
    }


def extract_ahi_summary_scatter_arrays(
    records: list[Mapping[str, np.ndarray]],
    *,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    prepared_records = _prepare_ahi_records(records)
    aggregate = _aggregate_prepared_ahi_records(prepared_records, threshold=float(threshold))
    return aggregate["true_ahi"], aggregate["pred_ahi"]


def _select_best_ahi_threshold_from_prepared(
    prepared_records: list[PreparedAHIRecord],
    *,
    search_thresholds: tuple[float, ...] = AHI_THRESHOLD_GRID,
) -> tuple[float, dict[str, Any]]:
    thresholds = tuple(float(value) for value in search_thresholds)
    best_threshold: float | None = None
    best_pearson = float("-inf")
    best_mae = float("inf")
    mode = _resolve_ahi_search_mode(thresholds)
    recording_count = len(prepared_records)
    eligible_count = sum(1 for record in prepared_records if record.summary_enabled)
    logging.info(
        "AHI threshold search start: mode=%s thresholds=%d recordings=%d eligible=%d",
        mode,
        len(thresholds),
        recording_count,
        eligible_count,
    )
    started_at = time.perf_counter()

    for index, threshold in enumerate(thresholds, start=1):
        if _should_log_ahi_search_progress(index, len(thresholds), threshold=threshold, mode=mode):
            logging.info(
                "AHI threshold search progress: %d/%d threshold=%.2f",
                index,
                len(thresholds),
                threshold,
            )
        aggregate = _aggregate_prepared_ahi_summary_records(prepared_records, threshold=threshold)
        pred_ahi = aggregate["pred_ahi"]
        true_ahi = aggregate["true_ahi"]
        if pred_ahi.size == 0 or true_ahi.size == 0:
            continue
        pearson = _safe_pearson(true_ahi, pred_ahi, require_min_count=2, nan_if_invalid=False)
        mae = float(np.mean(np.abs(pred_ahi - true_ahi)))
        if pearson > best_pearson or (
            np.isclose(pearson, best_pearson)
            and (
                mae < best_mae or (np.isclose(mae, best_mae) and (best_threshold is None or threshold > best_threshold))
            )
        ):
            best_threshold = float(threshold)
            best_pearson = float(pearson)
            best_mae = float(mae)

    if best_threshold is None:
        raise ValueError(
            "Unable to fit an AHI event threshold from validation records. "
            "Need at least 1 non-skipped sample with TST >= 2h."
        )
    best_aggregate = _aggregate_prepared_ahi_records(prepared_records, threshold=best_threshold)
    logging.info(
        "AHI threshold search done: best=%.2f elapsed=%.2fs",
        best_threshold,
        time.perf_counter() - started_at,
    )
    return best_threshold, best_aggregate


def select_best_ahi_threshold(
    records: list[Mapping[str, np.ndarray]],
    *,
    search_thresholds: tuple[float, ...] = AHI_THRESHOLD_GRID,
) -> tuple[float, dict[str, Any]]:
    prepared_records = _prepare_ahi_records(records)
    return _select_best_ahi_threshold_from_prepared(prepared_records, search_thresholds=search_thresholds)


def _compute_ahi_event_metrics_from_prepared(
    prepared_records: list[PreparedAHIRecord],
    *,
    threshold: float | None = None,
    search_thresholds: tuple[float, ...] = AHI_THRESHOLD_GRID,
    severity_thresholds: tuple[float, ...] = AHI_SEVERITY_THRESHOLDS,
) -> tuple[dict[str, float], float]:
    if threshold is None:
        threshold, aggregate = _select_best_ahi_threshold_from_prepared(
            prepared_records,
            search_thresholds=search_thresholds,
        )
    else:
        threshold = float(threshold)
        aggregate = _aggregate_prepared_ahi_records(prepared_records, threshold=threshold)

    tp = aggregate["event_tp"]
    fp = aggregate["event_fp"]
    fn = aggregate["event_fn"]
    event_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    event_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    event_f1 = (
        2 * event_precision * event_recall / (event_precision + event_recall)
        if (event_precision + event_recall) > 0
        else 0.0
    )

    pred_ahi = aggregate["pred_ahi"]
    true_ahi = aggregate["true_ahi"]
    metrics: dict[str, float] = {
        "ahi_event_precision": float(event_precision),
        "ahi_event_recall": float(event_recall),
        "ahi_event_f1": float(event_f1),
        "ahi_opt_threshold": float(threshold),
    }

    if true_ahi.size == 0 or pred_ahi.size == 0:
        metrics.update(
            {
                "ahi_pearson": np.nan,
                "ahi_mae": np.nan,
                "ahi_icc": np.nan,
                "ahi_acc": np.nan,
                "ahi_macro_f1": np.nan,
                "ahi_weighted_f1": np.nan,
            }
        )
        for severity_threshold in severity_thresholds:
            suffix = _format_threshold_suffix(severity_threshold)
            metrics.update(
                {
                    f"ahi_threshold_{suffix}_precision": np.nan,
                    f"ahi_threshold_{suffix}_recall": np.nan,
                    f"ahi_threshold_{suffix}_f1": np.nan,
                    f"ahi_threshold_{suffix}_specificity": np.nan,
                    f"ahi_threshold_{suffix}_accuracy": np.nan,
                    f"ahi_threshold_{suffix}_auroc": np.nan,
                    f"ahi_threshold_{suffix}_auprc": np.nan,
                }
            )
        return metrics, float(threshold)

    metrics["ahi_mae"] = float(np.mean(np.abs(pred_ahi - true_ahi)))
    metrics["ahi_pearson"] = _safe_pearson(true_ahi, pred_ahi, require_min_count=2, nan_if_invalid=False)
    metrics["ahi_icc"] = icc2_two_raters_arrays(true_ahi, pred_ahi)

    for severity_threshold in severity_thresholds:
        suffix = _format_threshold_suffix(severity_threshold)
        y_true = (true_ahi >= severity_threshold).astype(int)
        y_pred = (pred_ahi >= severity_threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp_bin, fn_bin, tp_bin = (int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1]))
        precision = tp_bin / (tp_bin + fp_bin) if (tp_bin + fp_bin) > 0 else 0.0
        recall = tp_bin / (tp_bin + fn_bin) if (tp_bin + fn_bin) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        specificity = tn / (tn + fp_bin) if (tn + fp_bin) > 0 else 0.0
        accuracy = (tn + tp_bin) / cm.sum() if cm.sum() > 0 else 0.0
        auroc = roc_auc_score(y_true, pred_ahi) if len(np.unique(y_true)) > 1 else 0.0
        auprc = average_precision_score(y_true, pred_ahi) if len(np.unique(y_true)) > 1 else 0.0
        metrics.update(
            {
                f"ahi_threshold_{suffix}_precision": float(precision),
                f"ahi_threshold_{suffix}_recall": float(recall),
                f"ahi_threshold_{suffix}_f1": float(f1),
                f"ahi_threshold_{suffix}_specificity": float(specificity),
                f"ahi_threshold_{suffix}_accuracy": float(accuracy),
                f"ahi_threshold_{suffix}_auroc": float(auroc),
                f"ahi_threshold_{suffix}_auprc": float(auprc),
            }
        )

    bins = [0.0] + list(severity_thresholds) + [np.inf]
    true_cls = np.digitize(true_ahi, bins) - 1
    pred_cls = np.digitize(pred_ahi, bins) - 1
    metrics["ahi_acc"] = float(accuracy_score(true_cls, pred_cls))
    metrics["ahi_macro_f1"] = float(f1_score(true_cls, pred_cls, average="macro"))
    metrics["ahi_weighted_f1"] = float(f1_score(true_cls, pred_cls, average="weighted"))
    return metrics, float(threshold)


def compute_ahi_event_metrics(
    records: list[Mapping[str, np.ndarray]],
    *,
    threshold: float | None = None,
    search_thresholds: tuple[float, ...] = AHI_THRESHOLD_GRID,
    severity_thresholds: tuple[float, ...] = AHI_SEVERITY_THRESHOLDS,
) -> tuple[dict[str, float], float]:
    prepared_records = _prepare_ahi_records(records)
    return _compute_ahi_event_metrics_from_prepared(
        prepared_records,
        threshold=threshold,
        search_thresholds=search_thresholds,
        severity_thresholds=severity_thresholds,
    )


def compute_downstream_metrics(
    gts,
    preds,
    *,
    is_classification: bool,
    is_multilabel: bool = False,
    output_dim: int | None = None,
    stage_names=None,
):
    """统一的下游任务指标计算。"""
    if is_multilabel:
        return compute_binary_label_metrics(gts, preds)

    if is_classification:
        probs = preds.astype(np.float32)
        y_true = gts.astype(np.int64)
        if y_true.ndim == 2:
            y_true = y_true.argmax(axis=1)
        y_true = y_true.reshape(-1)
        y_pred = probs.argmax(axis=1)
        result = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
            "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        }

        if output_dim == 2:
            result["roc_auc"] = roc_auc_from_two_logits(gts, preds)
            result["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
            result["specificity"] = binary_specificity(y_true, y_pred)
        else:
            result["recall"] = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
            result["specificity"] = float(macro_specificity(y_true, y_pred))
        if stage_names is None and output_dim == 5:
            stage_names = ["W", "N1", "N2", "N3", "REM"]
        if stage_names is not None:
            if output_dim is None:
                output_dim = len(stage_names)

            labels = np.arange(output_dim)
            f1_per_class = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)

            assert len(stage_names) == output_dim

            for i, f1 in enumerate(f1_per_class):
                result[f"f1_{stage_names[i]}"] = float(f1)

            # Stage aliases are macro metrics even for two-class stage collapses;
            # binary recall/specificity above keep their class-1-vs-class-0 meaning.
            result["spec"] = float(macro_specificity(y_true, y_pred))
            result["sens"] = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
        return result

    preds = preds.astype(np.float32).reshape(-1)
    gts = gts.astype(np.float32).reshape(-1)
    result = {
        "mse": np.mean((preds - gts) ** 2),
        "mae": np.mean(np.abs(preds - gts)),
    }
    result["pearsonr"], _ = pearsonr(preds, gts)
    return result
