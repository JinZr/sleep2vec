from __future__ import annotations

import math
import typing as t

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, confusion_matrix, roc_curve

from sleep2wave.metrics import binary_positive_scores_from_two_logits
from sleep2wave.visualization.curves import render_binary_roc_curve
from sleep2wave.visualization.heatmaps import render_matrix_heatmap
from sleep2wave.visualization.scatter import render_prediction_scatter
from sleep2wave.visualization.theme import (
    _FIGURE_BG,
    _OPENAI_BLUE_CMAP,
    _PRIMARY,
    _PRIMARY_DARK,
    _PRIMARY_LIGHT,
    _PRIMARY_MID,
    add_openai_legend,
    apply_plot_layout,
    style_axes,
    style_plot_text,
    use_openai_like_theme,
)

_MAX_RENDER_POINTS = 2000


def _flatten_waveform(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float32).reshape(-1)


def _downsample_for_display(*arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    length = max((array.size for array in arrays), default=0)
    if length <= _MAX_RENDER_POINTS:
        return arrays
    stride = int(math.ceil(length / _MAX_RENDER_POINTS))
    return tuple(array[::stride] for array in arrays)


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
        return render_matrix_heatmap(
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
            colorbar_title="Percent",
            integer_colorbar_ticks=False,
            subplots_adjust={"left": 0.14, "right": 0.88, "bottom": 0.14, "top": 0.88},
        )

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


def render_waveform_example_plot(
    clean: np.ndarray,
    generated: np.ndarray,
    *,
    sample_rate_hz: int,
    title: str,
    generated_label: str,
    observed: np.ndarray | None = None,
) -> plt.Figure:
    use_openai_like_theme()

    clean_values = _flatten_waveform(clean)
    generated_values = _flatten_waveform(generated)
    time_sec = np.arange(clean_values.size, dtype=np.float32) / float(sample_rate_hz)
    if observed is None:
        time_sec, clean_values, generated_values = _downsample_for_display(
            time_sec,
            clean_values,
            generated_values,
        )
        observed_values = None
    else:
        observed_values = _flatten_waveform(observed)
        time_sec, clean_values, observed_values, generated_values = _downsample_for_display(
            time_sec,
            clean_values,
            observed_values,
            generated_values,
        )

    fig, ax = plt.subplots(figsize=(10.5, 4.4), facecolor=_FIGURE_BG)
    style_axes(ax, show_grid=True)
    ax.plot(time_sec, clean_values, color=_PRIMARY_DARK, linewidth=1.25, linestyle="solid", label="Clean")
    if observed_values is not None:
        ax.plot(
            time_sec,
            observed_values,
            color=_PRIMARY_LIGHT,
            linewidth=1.0,
            linestyle="solid",
            alpha=0.9,
            label="Observed",
        )
    generated_color = _PRIMARY if generated_label == "Reconstruction" else _PRIMARY_MID
    ax.plot(
        time_sec,
        generated_values,
        color=generated_color,
        linewidth=1.1,
        linestyle="solid",
        alpha=0.9,
        label=generated_label,
    )
    style_plot_text(ax, title=title, xlabel="Time (s)", ylabel="Amplitude")
    add_openai_legend(
        ax,
        loc="upper right",
        bbox_to_anchor=(0.98, 0.98),
    )
    apply_plot_layout(fig, defaults={"left": 0.08, "right": 0.98, "bottom": 0.16, "top": 0.86})
    return fig


__all__ = [
    "use_openai_like_theme",
    "render_binary_roc_curve_plot",
    "render_confusion_matrix_heatmap",
    "render_regression_scatter_plot",
    "render_waveform_example_plot",
]
