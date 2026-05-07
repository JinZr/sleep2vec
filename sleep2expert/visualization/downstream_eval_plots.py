from __future__ import annotations

import typing as t

from matplotlib.ticker import FuncFormatter
import numpy as np
from sklearn.metrics import auc, confusion_matrix, roc_curve

from sleep2expert.metrics import binary_positive_scores_from_two_logits
from sleep2expert.visualization.curves import render_binary_roc_curve
from sleep2expert.visualization.heatmaps import add_axis_title_boxes, render_matrix_heatmap
from sleep2expert.visualization.scatter import render_prediction_scatter
from sleep2expert.visualization.theme import _OPENAI_BLUE_CMAP, use_openai_like_theme


def _format_percentage(value: float) -> str:
    return f"{value:.1f}".rstrip("0").rstrip(".") + "%"


def render_confusion_matrix_heatmap(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_labels: t.Sequence[str],
    *,
    title: str = "Confusion Matrix",
    normalize_rows: bool = False,
    show_raw_counts: bool = False,
):
    labels = np.arange(len(class_labels), dtype=np.int64)
    cm = confusion_matrix(y_true.astype(np.int64), y_pred.astype(np.int64), labels=labels)
    n_classes = len(class_labels)

    fig_width = max(6.0, 1.05 * n_classes + 3.1)
    fig_height = max(5.2, 0.95 * n_classes + 2.2)

    if normalize_rows:
        row_totals = cm.sum(axis=1, keepdims=True).astype(np.float32)
        cm_percent = np.divide(
            cm.astype(np.float32) * 100.0,
            row_totals,
            out=np.zeros_like(cm, dtype=np.float32),
            where=row_totals > 0,
        )
        if show_raw_counts:
            annotation_texts = np.empty(cm.shape, dtype=object)
            for row_idx in range(cm.shape[0]):
                for col_idx in range(cm.shape[1]):
                    annotation_texts[row_idx, col_idx] = (
                        f"{_format_percentage(cm_percent[row_idx, col_idx])}\n({cm[row_idx, col_idx]})"
                    )
        else:
            annotation_texts = np.vectorize(_format_percentage, otypes=[object])(cm_percent)
        fig = render_matrix_heatmap(
            cm_percent,
            class_labels,
            class_labels,
            title=title,
            xlabel="Predicted Label",
            ylabel="True Label",
            cmap=_OPENAI_BLUE_CMAP,
            vmin=0.0,
            vmax=100.0,
            figsize=(fig_width, fig_height),
            annotation_formatter=_format_percentage,
            annotation_texts=annotation_texts,
            annotation_fontweight="normal",
            colorbar_title="Percent",
            integer_colorbar_ticks=False,
            subplots_adjust={"left": 0.20, "right": 0.86, "bottom": 0.30, "top": 0.88},
        )
    else:
        fig = render_matrix_heatmap(
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
            annotation_fontweight="normal",
            colorbar_title="Count",
            integer_colorbar_ticks=True,
            subplots_adjust={"left": 0.20, "right": 0.86, "bottom": 0.30, "top": 0.88},
        )

    ax = fig.axes[0]
    for label in ax.get_xticklabels():
        label.set_rotation(90)
        label.set_ha("right")
        label.set_va("center")
        label.set_rotation_mode("anchor")
    add_axis_title_boxes(fig, ax, xlabel="Predicted Label", ylabel="True Label")
    if normalize_rows:
        fig.axes[1].yaxis.set_major_formatter(
            FuncFormatter(lambda value, _: "0" if abs(value) < 1e-6 else f"{value:.0f}%")
        )
    return fig


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
