from typing import Any

import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def binary_positive_scores_from_two_logits(
    gts, preds, *, from_logits: bool | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Return binary labels and positive-class scores from (N, 2) logits or probabilities."""
    y_true = np.asarray(gts)
    y_pred = np.asarray(preds)

    if y_true.ndim == 2 and y_true.shape[1] == 2:
        y_true = y_true.argmax(axis=1)
    y_true = y_true.astype(int).reshape(-1)

    if y_pred.ndim != 2 or y_pred.shape[1] != 2:
        raise ValueError(f"preds must be (N, 2), got {y_pred.shape}")

    # Keep heuristic detection for existing ranking-only callers; calibration callers choose explicitly.
    if from_logits is None:
        row_sum = y_pred.sum(axis=1, keepdims=True)
        looks_like_prob = y_pred.min() >= 0.0 and y_pred.max() <= 1.0 and np.allclose(row_sum, 1.0, atol=1e-4)
    else:
        looks_like_prob = not from_logits
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


def compute_binary_probability_metrics(gts, preds, *, from_logits: bool) -> dict[str, float]:
    y_true, y_score = binary_positive_scores_from_two_logits(gts, preds, from_logits=from_logits)

    auprc = float(average_precision_score(y_true, y_score)) if np.unique(y_true).size >= 2 else float("nan")
    brier = float(brier_score_loss(y_true, y_score, pos_label=1))

    bin_edges = np.linspace(0.0, 1.0, 11, dtype=y_score.dtype)
    # Search only internal edges so p=1 remains in the final [0.9, 1] bin.
    bin_indices = np.searchsorted(bin_edges[1:-1], y_score, side="right")
    score_sums = np.bincount(bin_indices, weights=y_score.astype(np.float64), minlength=10)
    label_sums = np.bincount(bin_indices, weights=y_true.astype(np.float64), minlength=10)
    ece = float(np.abs(score_sums - label_sums).sum() / y_true.size)

    return {"auprc": auprc, "brier": brier, "ece": ece}


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


def filter_segments_by_duration(intervals, *, min_duration: int) -> list[list[int]]:
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


def compute_downstream_metrics(
    gts,
    preds,
    *,
    is_classification: bool,
    is_multilabel: bool = False,
    output_dim: int | None = None,
    stage_names=None,
    include_binary_probability_metrics: bool = False,
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
            if include_binary_probability_metrics:
                # Legacy callers may pass logits, so this explicit opt-in requires probabilities.
                result.update(compute_binary_probability_metrics(gts, preds, from_logits=False))
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
