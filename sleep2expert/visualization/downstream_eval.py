from __future__ import annotations

import typing as t

import matplotlib.pyplot as plt
import numpy as np
import wandb

from sleep2expert.config import EvalVisualizationsConfig
from sleep2expert.visualization.downstream_eval_plots import (
    render_binary_roc_curve_plot,
    render_confusion_matrix_heatmap,
    render_regression_scatter_plot,
)


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
                normalize_rows=True,
                show_raw_counts=self._config.confusion_matrix.show_raw_counts,
            )
            payload[f"{stage}_eval/confusion_matrix"] = wandb.Image(fig)
            plt.close(fig)

        if is_classification and output_dim == 2 and self._config.roc_curve.enabled:
            fig = render_binary_roc_curve_plot(
                np.asarray(targets, dtype=np.int64).reshape(-1),
                np.asarray(preds, dtype=np.float32),
                title=f"{stage.title()} ROC Curve ({label_name}, epoch {current_epoch})",
            )
            if fig is not None:
                payload[f"{stage}_eval/roc_curve"] = wandb.Image(fig)
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

    def log_ahi_summary_scatter(
        self,
        *,
        stage: str,
        preds: np.ndarray,
        targets: np.ndarray,
        label_name: str,
        current_epoch: int,
    ) -> None:
        if not self.enabled_for_stage(stage):
            return
        if self._config is None or not self._config.regression_scatter.enabled:
            return
        if getattr(wandb, "run", None) is None:
            return

        preds = np.asarray(preds, dtype=np.float32).reshape(-1)
        targets = np.asarray(targets, dtype=np.float32).reshape(-1)
        if preds.size == 0 or targets.size == 0:
            return

        fig = render_regression_scatter_plot(
            targets,
            preds,
            title=f"{stage.title()} Prediction Scatter ({label_name}, epoch {current_epoch})",
        )
        wandb.log({f"{stage}_eval/regression_scatter": wandb.Image(fig)}, commit=False)
        plt.close(fig)

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
    "render_binary_roc_curve_plot",
    "render_confusion_matrix_heatmap",
    "render_regression_scatter_plot",
]
