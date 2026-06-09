"""Platt-scale binary downstream prediction CSVs.

This utility calibrates a binary classifier on a validation prediction CSV
and applies the learned two-parameter sigmoid to one or more evaluation CSVs.
It is intended for sleep2vec inference outputs where class probabilities are
stored as ``prob_0`` and ``prob_1``.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

EPS = 1e-6


def _parse_scalar(value: Any, *, column: str) -> float:
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("[") and text.endswith("]"):
            parsed = ast.literal_eval(text)
            arr = np.asarray(parsed, dtype=float).reshape(-1)
            if arr.size != 1:
                raise ValueError(f"{column} must be scalar or length-1 list; got {arr.size} values.")
            return float(arr[0])
        return float(text)

    arr = np.asarray(value)
    if arr.ndim == 0:
        return float(arr.item())
    arr = arr.astype(float).reshape(-1)
    if arr.size != 1:
        raise ValueError(f"{column} must be scalar or length-1 array; got {arr.size} values.")
    return float(arr[0])


def _logit(prob: np.ndarray) -> np.ndarray:
    prob = np.clip(np.asarray(prob, dtype=np.float64), EPS, 1.0 - EPS)
    return np.log(prob / (1.0 - prob))


def _sigmoid(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    out = np.empty_like(value, dtype=np.float64)
    positive = value >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    exp_value = np.exp(value[~positive])
    out[~positive] = exp_value / (1.0 + exp_value)
    return out


def load_binary_predictions(
    csv_path: Path,
    *,
    label_column: str = "groundtruth",
    prob_column: str = "prob_1",
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    missing = [column for column in (label_column, prob_column) if column not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {missing}")

    y_true = np.asarray([_parse_scalar(value, column=label_column) for value in df[label_column]], dtype=np.int64)
    prob = np.asarray([_parse_scalar(value, column=prob_column) for value in df[prob_column]], dtype=np.float64)

    valid = np.isfinite(y_true) & np.isfinite(prob)
    valid &= np.isin(y_true, [0, 1])
    valid &= (prob >= 0.0) & (prob <= 1.0)
    if int(valid.sum()) != len(df):
        bad = len(df) - int(valid.sum())
        raise ValueError(f"{csv_path} contains {bad} invalid binary labels or probabilities.")
    if np.unique(y_true).size < 2:
        raise ValueError(f"{csv_path} must contain both classes for calibration/evaluation.")
    return df, y_true, prob


def fit_platt_scaler(
    y_true: np.ndarray,
    prob: np.ndarray,
    *,
    l2: float = 1e-6,
    max_iter: int = 100,
    tol: float = 1e-9,
) -> tuple[float, float]:
    """Fit ``sigmoid(coef * logit(prob) + intercept)`` by logistic NLL.

    The two-parameter Newton solver avoids depending on scikit-learn in the
    lightweight analysis environment. A tiny ridge term keeps perfectly
    separated validation sets finite.
    """
    y = y_true.astype(np.float64).reshape(-1)
    x = _logit(prob).reshape(-1)
    x_mean = float(x.mean())
    x_scale = float(x.std())
    if not np.isfinite(x_scale) or x_scale < EPS:
        x_scale = 1.0
    x_scaled = (x - x_mean) / x_scale
    design = np.column_stack([x_scaled, np.ones_like(x_scaled)])
    theta = np.zeros(2, dtype=np.float64)
    penalty = np.diag([float(l2), 0.0])

    for _ in range(max_iter):
        z = design @ theta
        pred = _sigmoid(z)
        grad = design.T @ (pred - y) + penalty @ theta
        weight = pred * (1.0 - pred)
        hessian = design.T @ (design * weight[:, None]) + penalty
        try:
            step = np.linalg.solve(hessian, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(hessian) @ grad
        theta -= step
        if float(np.linalg.norm(step)) < tol:
            break

    coef_scaled, intercept_scaled = float(theta[0]), float(theta[1])
    coef = coef_scaled / x_scale
    intercept = intercept_scaled - (coef_scaled * x_mean / x_scale)
    return float(coef), float(intercept)


def apply_platt_scaler(prob: np.ndarray, *, coef: float, intercept: float) -> np.ndarray:
    raw_score = _logit(prob)
    return _sigmoid(coef * raw_score + intercept)


def _confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int]:
    y_true = y_true.astype(np.int64).reshape(-1)
    y_pred = y_pred.astype(np.int64).reshape(-1)
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    return tn, fp, fn, tp


def _roc_auc(y_true: np.ndarray, prob: np.ndarray) -> float:
    y_true = y_true.astype(np.int64).reshape(-1)
    prob = np.asarray(prob, dtype=np.float64).reshape(-1)
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(prob, kind="mergesort")
    sorted_prob = prob[order]
    ranks = np.empty_like(sorted_prob, dtype=np.float64)
    start = 0
    while start < sorted_prob.size:
        end = start + 1
        while end < sorted_prob.size and sorted_prob[end] == sorted_prob[start]:
            end += 1
        ranks[start:end] = (start + 1 + end) / 2.0
        start = end

    original_ranks = np.empty_like(ranks)
    original_ranks[order] = ranks
    pos_rank_sum = float(original_ranks[y_true == 1].sum())
    return float((pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _balanced_accuracy_from_pred(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tn, fp, fn, tp = _confusion_counts(y_true, y_pred)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return float((sensitivity + specificity) / 2.0)


def compute_binary_metrics(y_true: np.ndarray, prob: np.ndarray, *, threshold: float = 0.5) -> dict[str, float]:
    pred = (prob >= float(threshold)).astype(np.int64)
    tn, fp, fn, tp = _confusion_counts(y_true, pred)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    metrics = {
        "n": int(y_true.size),
        "n_positive": int((y_true == 1).sum()),
        "n_negative": int((y_true == 0).sum()),
        "pred_positive": int((pred == 1).sum()),
        "pred_negative": int((pred == 0).sum()),
        "positive_rate": float((y_true == 1).mean()),
        "pred_positive_rate": float((pred == 1).mean()),
        "threshold": float(threshold),
        "accuracy": float((tp + tn) / y_true.size),
        "balanced_accuracy": float((recall + specificity) / 2.0),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }
    metrics["auroc"] = _roc_auc(y_true, prob)
    return metrics


def best_threshold_by_balanced_accuracy(y_true: np.ndarray, prob: np.ndarray) -> tuple[float, float]:
    thresholds = np.unique(np.clip(prob, 0.0, 1.0))
    candidates = np.unique(np.concatenate(([0.0, 0.5, 1.0], thresholds)))
    best_threshold = 0.5
    best_score = -np.inf
    for threshold in candidates:
        pred = (prob >= float(threshold)).astype(np.int64)
        score = _balanced_accuracy_from_pred(y_true, pred)
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold, best_score


def _write_confusion_matrix(path: Path, y_true: np.ndarray, prob: np.ndarray, *, threshold: float = 0.5) -> None:
    pred = (prob >= float(threshold)).astype(np.int64)
    tn, fp, fn, tp = _confusion_counts(y_true, pred)
    df = pd.DataFrame([[tn, fp], [fn, tp]], index=["true_0", "true_1"], columns=["pred_0", "pred_1"])
    df.to_csv(path)


def calibrate_predictions(
    calibration_csv: Path,
    eval_csvs: list[Path],
    *,
    output_dir: Path,
    calibration_name: str,
    eval_names: list[str] | None = None,
    label_column: str = "groundtruth",
    prob_column: str = "prob_1",
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cal_df, y_cal, p_cal_raw = load_binary_predictions(
        calibration_csv,
        label_column=label_column,
        prob_column=prob_column,
    )
    coef, intercept = fit_platt_scaler(y_cal, p_cal_raw)

    if eval_names is None:
        eval_names = [path.stem for path in eval_csvs]
    if len(eval_names) != len(eval_csvs):
        raise ValueError("--eval-name count must match --eval-predictions count.")

    all_sets = [(calibration_name, "calibration", calibration_csv, cal_df, y_cal, p_cal_raw)]
    for name, csv_path in zip(eval_names, eval_csvs):
        df, y_true, prob_raw = load_binary_predictions(csv_path, label_column=label_column, prob_column=prob_column)
        all_sets.append((name, "evaluation", csv_path, df, y_true, prob_raw))

    metrics_rows: list[dict[str, Any]] = []
    for name, role, csv_path, df, y_true, prob_raw in all_sets:
        prob_calibrated = apply_platt_scaler(prob_raw, coef=coef, intercept=intercept)
        raw_best_threshold, raw_best_bal_acc = best_threshold_by_balanced_accuracy(y_true, prob_raw)
        cal_best_threshold, cal_best_bal_acc = best_threshold_by_balanced_accuracy(y_true, prob_calibrated)

        for score_name, prob in (("raw", prob_raw), ("platt", prob_calibrated)):
            row = {
                "dataset_name": name,
                "role": role,
                "prediction_csv": str(csv_path),
                "score": score_name,
                **compute_binary_metrics(y_true, prob, threshold=0.5),
            }
            if score_name == "raw":
                row["best_balanced_accuracy_threshold"] = raw_best_threshold
                row["best_balanced_accuracy"] = raw_best_bal_acc
            else:
                row["best_balanced_accuracy_threshold"] = cal_best_threshold
                row["best_balanced_accuracy"] = cal_best_bal_acc
            metrics_rows.append(row)

        out_predictions = df.copy()
        out_predictions["raw_prob_1"] = prob_raw
        out_predictions["raw_logit_1"] = _logit(prob_raw)
        out_predictions["platt_prob_1"] = prob_calibrated
        out_predictions["platt_prediction"] = (prob_calibrated >= 0.5).astype(np.int64)
        out_predictions.to_csv(output_dir / f"predictions__{name}__platt.csv", index=False)

        _write_confusion_matrix(output_dir / f"confusion_matrix__{name}__raw.csv", y_true, prob_raw)
        _write_confusion_matrix(output_dir / f"confusion_matrix__{name}__platt.csv", y_true, prob_calibrated)

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(output_dir / "metrics.csv", index=False)

    manifest = {
        "method": "platt_scaling",
        "label_column": label_column,
        "prob_column": prob_column,
        "calibration_csv": str(calibration_csv),
        "calibration_name": calibration_name,
        "eval_csvs": [str(path) for path in eval_csvs],
        "eval_names": eval_names,
        "coef": coef,
        "intercept": intercept,
        "formula": "platt_prob_1 = sigmoid(coef * logit(raw_prob_1) + intercept)",
        "metrics_csv": str(output_dir / "metrics.csv"),
    }
    (output_dir / "calibrator.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-predictions", type=Path, required=True, help="GZ/validation prediction CSV.")
    parser.add_argument(
        "--eval-predictions",
        type=Path,
        nargs="*",
        default=[],
        help="Optional evaluation prediction CSVs, for example MGH.",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for calibrated outputs.")
    parser.add_argument("--calibration-name", type=str, default="GZ", help="Name used for calibration outputs.")
    parser.add_argument(
        "--eval-name",
        type=str,
        nargs="+",
        default=None,
        help="Names for evaluation outputs; must match --eval-predictions count.",
    )
    parser.add_argument("--label-column", type=str, default="groundtruth")
    parser.add_argument("--prob-column", type=str, default="prob_1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = calibrate_predictions(
        args.calibration_predictions,
        args.eval_predictions,
        output_dir=args.output_dir,
        calibration_name=args.calibration_name,
        eval_names=args.eval_name,
        label_column=args.label_column,
        prob_column=args.prob_column,
    )
    print(f"Wrote Platt calibration outputs to {args.output_dir}")
    print(f"coef={manifest['coef']:.6g} intercept={manifest['intercept']:.6g}")
    metrics_path = args.output_dir / "metrics.csv"
    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
        summary_columns = [
            "dataset_name",
            "role",
            "score",
            "n",
            "n_positive",
            "n_negative",
            "pred_positive",
            "pred_negative",
            "auroc",
            "balanced_accuracy",
            "recall",
            "specificity",
            "f1",
            "tn",
            "fp",
            "fn",
            "tp",
        ]
        summary_columns = [column for column in summary_columns if column in metrics.columns]
        print(metrics.loc[:, summary_columns].to_string(index=False))


if __name__ == "__main__":
    main()
