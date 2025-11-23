import copy
import os
from typing import Any, Mapping

import numpy as np
import pandas as pd
from pyhealth.metrics import multiclass_metrics_fn
from scipy.stats import pearsonr
from sklearn.metrics import confusion_matrix, f1_score, recall_score, roc_auc_score


def roc_auc_from_two_logits(gts, preds) -> float:
    """
    计算二分类场景下基于两列 logits/probabilities 的 ROC-AUC。
    """
    y_true = np.asarray(gts)
    y_pred = np.asarray(preds)

    if y_true.ndim == 2 and y_true.shape[1] == 2:
        y_true = y_true.argmax(axis=1)
    y_true = y_true.astype(int).reshape(-1)

    if y_pred.ndim != 2 or y_pred.shape[1] != 2:
        raise ValueError(f"preds 必须是 (N,2)，当前 {y_pred.shape}")

    row_sum = y_pred.sum(axis=1, keepdims=True)
    looks_like_prob = y_pred.min() >= 0.0 and y_pred.max() <= 1.0 and np.allclose(row_sum, 1.0, atol=1e-4)
    if looks_like_prob:
        y_score = y_pred[:, 1].astype(np.float32)
    else:
        z = y_pred - y_pred.max(axis=1, keepdims=True)
        e = np.exp(z, dtype=np.float64)
        proba = (e / e.sum(axis=1, keepdims=True)).astype(np.float32)
        y_score = proba[:, 1]

    if np.unique(y_true).size < 2:
        return np.nan

    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return np.nan


def save_result_csv(pretrain_result: Mapping[str, float], csv_path: str, args: Any | None = None):
    """
    将实验结果写入/追加到 CSV 文件中。
    """
    new_row: dict[str, Any] = dict(copy.deepcopy(pretrain_result))

    if args is not None:
        new_row["ckpt_path"] = getattr(args, "ckpt_path", None)
        new_row["lr"] = getattr(args, "lr", None)
        new_row["batch_size"] = getattr(args, "batch_size", None)
        new_row["n_few_shot"] = getattr(args, "n_few_shot", None)
        new_row["label_name"] = getattr(args, "label_name", None)
        channel_names = getattr(args, "channel_names", None)
        if isinstance(channel_names, (list, tuple)):
            new_row["channel_names"] = ",".join(str(name) for name in channel_names)
        elif isinstance(channel_names, str):
            new_row["channel_names"] = channel_names
        else:
            new_row["channel_names"] = ""

    df_new = pd.DataFrame([new_row])

    if os.path.exists(csv_path):
        df_old = pd.read_csv(csv_path)
        df_merged = pd.concat([df_old, df_new], axis=0, join="outer", ignore_index=True)
    else:
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        df_merged = df_new

    df_merged.to_csv(csv_path, index=False)
    print(f"Results written to {csv_path}")


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


def compute_downstream_metrics(
    gts,
    preds,
    *,
    is_classification: bool,
    output_dim: int | None = None,
    stage_names=None,
):
    """统一的下游任务指标计算。"""
    if is_classification:
        result = multiclass_metrics_fn(
            gts,
            preds,
            metrics=["accuracy", "cohen_kappa", "f1_weighted", "f1_macro"],
        )
        if output_dim == 2:
            result["roc_auc"] = roc_auc_from_two_logits(gts, preds)
        if output_dim == 5:
            probs = preds.astype(np.float32)
            y_true = gts.astype(np.int64)
            y_pred = probs.argmax(axis=1)

            labels = np.arange(output_dim)
            f1_per_class = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)

            stage_names = stage_names or ["W", "N1", "N2", "N3", "REM"]
            assert len(stage_names) == output_dim

            for i, f1 in enumerate(f1_per_class):
                result[f"f1_{stage_names[i]}"] = float(f1)

            sens = recall_score(y_true, y_pred, average="macro", zero_division=0)
            result["spec"] = float(macro_specificity(y_true, y_pred))
            result["sens"] = float(sens)
        return result

    preds = preds.astype(np.float32).reshape(-1)
    gts = gts.astype(np.float32).reshape(-1)
    result = {
        "mse": np.mean((preds - gts) ** 2),
        "mae": np.mean(np.abs(preds - gts)),
    }
    result["pearsonr"], _ = pearsonr(preds, gts)
    return result
