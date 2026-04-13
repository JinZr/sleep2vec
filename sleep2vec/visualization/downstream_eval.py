from __future__ import annotations

import typing as t

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
import wandb

from sleep2vec.config import EvalVisualizationsConfig


def render_confusion_matrix_heatmap(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_labels: t.Sequence[str],
    *,
    title: str = "Confusion Matrix",
) -> plt.Figure:
    labels = np.arange(len(class_labels), dtype=np.int64)
    cm = confusion_matrix(y_true.astype(np.int64), y_pred.astype(np.int64), labels=labels)

    fig_width = max(6.0, 0.8 * len(class_labels) + 2.0)
    fig_height = max(5.0, 0.7 * len(class_labels) + 1.5)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=list(class_labels),
        yticklabels=list(class_labels),
        cbar=True,
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    plt.tight_layout()
    return fig


def render_regression_scatter_plot(
    targets: np.ndarray,
    preds: np.ndarray,
    *,
    title: str = "Prediction vs Target",
) -> plt.Figure:
    targets = np.asarray(targets, dtype=np.float32).reshape(-1)
    preds = np.asarray(preds, dtype=np.float32).reshape(-1)

    bounds = np.concatenate([targets, preds], axis=0)
    lower = float(bounds.min())
    upper = float(bounds.max())
    if lower == upper:
        lower -= 1.0
        upper += 1.0

    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    ax.scatter(targets, preds, alpha=0.35, s=16, edgecolors="none")
    ax.plot([lower, upper], [lower, upper], linestyle="--", color="black", linewidth=1.0)
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_title(title)
    ax.set_xlabel("True Target")
    ax.set_ylabel("Predicted Target")
    plt.tight_layout()
    return fig


class DownstreamEvalVisualizer:
    def __init__(self, config: EvalVisualizationsConfig | None) -> None:
        self._config = config

    def enabled_for_stage(self, stage: str) -> bool:
        return bool(self._config is not None and self._config.enabled and stage in self._config.stages)

    def log(
        self,
        *,
        stage: str,
        preds: np.ndarray,
        targets: np.ndarray,
        is_classification: bool,
        output_dim: int | None,
        label_name: str,
        current_epoch: int,
        class_labels: t.Sequence[str] | None = None,
    ) -> None:
        if not self.enabled_for_stage(stage):
            return
        if getattr(wandb, "run", None) is None:
            return

        payload: dict[str, t.Any] = {}

        if is_classification and self._config.confusion_matrix.enabled:
            pred_labels = np.asarray(preds, dtype=np.float32).argmax(axis=1)
            resolved_labels = self._resolve_class_labels(class_labels, output_dim, pred_labels)
            fig = render_confusion_matrix_heatmap(
                np.asarray(targets, dtype=np.int64).reshape(-1),
                pred_labels.reshape(-1),
                resolved_labels,
                title=f"{stage.title()} Confusion Matrix ({label_name}, epoch {current_epoch})",
            )
            payload[f"{stage}_eval/confusion_matrix"] = wandb.Image(fig)
            plt.close(fig)

        if (not is_classification) and output_dim == 1 and self._config.regression_scatter.enabled:
            fig = render_regression_scatter_plot(
                np.asarray(targets, dtype=np.float32).reshape(-1),
                np.asarray(preds, dtype=np.float32).reshape(-1),
                title=f"{stage.title()} Prediction Scatter ({label_name}, epoch {current_epoch})",
            )
            payload[f"{stage}_eval/regression_scatter"] = wandb.Image(fig)
            plt.close(fig)

        if payload:
            wandb.log(payload, commit=False)

    @staticmethod
    def _resolve_class_labels(
        class_labels: t.Sequence[str] | None,
        output_dim: int | None,
        pred_labels: np.ndarray,
    ) -> list[str]:
        if class_labels is not None:
            labels = [str(label) for label in class_labels]
            if output_dim is None or len(labels) == output_dim:
                return labels

        resolved_dim = (
            int(output_dim) if output_dim is not None else (int(pred_labels.max()) + 1 if pred_labels.size else 0)
        )
        return [str(idx) for idx in range(max(resolved_dim, 0))]


__all__ = [
    "DownstreamEvalVisualizer",
    "render_confusion_matrix_heatmap",
    "render_regression_scatter_plot",
]
