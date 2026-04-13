from __future__ import annotations

import typing as t

import numpy as np
from sklearn.metrics import auc, confusion_matrix, roc_curve

from sleep2vec.metrics import binary_positive_scores_from_two_logits
from sleep2vec.visualization.curves import render_binary_roc_curve
from sleep2vec.visualization.heatmaps import render_matrix_heatmap
from sleep2vec.visualization.scatter import render_prediction_scatter
from sleep2vec.visualization.theme import _OPENAI_BLUE_CMAP, use_openai_like_theme


def render_confusion_matrix_heatmap(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_labels: t.Sequence[str],
    *,
    title: str = "Confusion Matrix",
):
    labels = np.arange(len(class_labels), dtype=np.int64)
    cm = confusion_matrix(y_true.astype(np.int64), y_pred.astype(np.int64), labels=labels)
    n_classes = len(class_labels)

    fig_width = max(6.0, 1.05 * n_classes + 3.1)
    fig_height = max(5.2, 0.95 * n_classes + 2.2)
    return render_matrix_heatmap(
        cm,
        class_labels,
        class_labels,
        title=title,
        xlabel="Predicted Label",
        ylabel="True Label",
        cmap=_OPENAI_BLUE_CMAP,
        vmin=0.0,
        vmax=max(1, int(cm.max())),
        figsize=(fig_width, fig_height),
        annotation_formatter=lambda value: f"{int(round(value))}",
        colorbar_title="Count",
        integer_colorbar_ticks=True,
        subplots_adjust={"left": 0.14, "right": 0.88, "bottom": 0.14, "top": 0.88},
    )


def render_regression_scatter_plot(
    targets: np.ndarray,
    preds: np.ndarray,
    *,
    title: str = "Prediction vs Target",
):
    return render_prediction_scatter(
        targets,
        preds,
        title=title,
        xlabel="True Target",
        ylabel="Predicted Target",
        figsize=(7.1, 6.7),
        subplots_adjust={"left": 0.15, "right": 0.97, "bottom": 0.14, "top": 0.90},
    )


def render_binary_roc_curve_plot(
    targets: np.ndarray,
    preds: np.ndarray,
    *,
    title: str = "ROC Curve",
):
    y_true, y_score = binary_positive_scores_from_two_logits(targets, preds)
    if y_true.size == 0 or np.unique(y_true).size < 2:
        return None

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = float(auc(fpr, tpr))
    return render_binary_roc_curve(
        fpr,
        tpr,
        roc_auc=roc_auc,
        title=title,
        figsize=(7.2, 6.7),
        subplots_adjust={"left": 0.16, "right": 0.97, "bottom": 0.15, "top": 0.90},
    )


__all__ = [
    "use_openai_like_theme",
    "render_binary_roc_curve_plot",
    "render_confusion_matrix_heatmap",
    "render_regression_scatter_plot",
]
