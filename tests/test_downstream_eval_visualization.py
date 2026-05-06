from __future__ import annotations

from matplotlib.patches import FancyBboxPatch
import matplotlib.pyplot as plt
import numpy as np

from sleep2vec.config import ConfusionMatrixVisualizationConfig, EvalVisualizationPlotConfig, EvalVisualizationsConfig
import sleep2vec.visualization.downstream_eval as downstream_eval
from sleep2vec.visualization.downstream_eval import (
    DownstreamEvalVisualizer,
    render_binary_roc_curve_plot,
    render_confusion_matrix_heatmap,
    render_regression_scatter_plot,
)


def test_render_confusion_matrix_heatmap_uses_stage_labels():
    fig = render_confusion_matrix_heatmap(
        np.array([0, 1, 2], dtype=np.int64),
        np.array([0, 1, 2], dtype=np.int64),
        ["W", "NREM", "REM"],
        title="demo",
    )

    ax = fig.axes[0]
    assert [tick.get_text() for tick in ax.get_xticklabels()] == ["W", "NREM", "REM"]
    assert [tick.get_text() for tick in ax.get_yticklabels()] == ["W", "NREM", "REM"]
    plt.close(fig)


def test_render_confusion_matrix_heatmap_uses_sex_labels():
    fig = render_confusion_matrix_heatmap(
        np.array([0, 1], dtype=np.int64),
        np.array([0, 1], dtype=np.int64),
        ["female", "male"],
        title="demo",
    )

    ax = fig.axes[0]
    assert [tick.get_text() for tick in ax.get_xticklabels()] == ["female", "male"]
    assert [tick.get_text() for tick in ax.get_yticklabels()] == ["female", "male"]
    plt.close(fig)


def test_render_confusion_matrix_heatmap_normalizes_binary_rows():
    fig = render_confusion_matrix_heatmap(
        np.array([0, 0, 0, 1, 1], dtype=np.int64),
        np.array([0, 1, 1, 1, 0], dtype=np.int64),
        ["female", "male"],
        title="demo",
        normalize_rows=True,
    )

    ax = fig.axes[0]
    heatmap = np.asarray(ax.images[0].get_array(), dtype=np.float32)
    assert np.allclose(
        heatmap,
        np.array(
            [
                [33.333332, 66.666664],
                [50.0, 50.0],
            ],
            dtype=np.float32,
        ),
        atol=1e-4,
    )
    assert {text.get_text() for text in ax.texts} >= {"33%", "67%", "50%"}
    assert {tick.get_rotation() for tick in ax.get_xticklabels()} == {90.0}
    assert {tick.get_ha() for tick in ax.get_xticklabels()} == {"right"}
    axis_box = ax.get_position()
    title_boxes = [patch for patch in fig.patches if isinstance(patch, FancyBboxPatch)]
    assert len(title_boxes) == 2
    assert any(np.isclose(box.get_width(), axis_box.width) for box in title_boxes)
    assert any(np.isclose(box.get_height(), axis_box.height) for box in title_boxes)
    x_title_box = next(box for box in title_boxes if np.isclose(box.get_width(), axis_box.width))
    y_title_box = next(box for box in title_boxes if np.isclose(box.get_height(), axis_box.height))
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    x_tick_bottom = min(
        label.get_window_extent(renderer=renderer).transformed(fig.transFigure.inverted()).y0
        for label in ax.get_xticklabels()
    )
    assert np.isclose(x_title_box.get_y() + x_title_box.get_height(), x_tick_bottom - 0.018)
    assert x_title_box.get_y() >= 0.018
    y_tick_left = min(
        label.get_window_extent(renderer=renderer).transformed(fig.transFigure.inverted()).x0
        for label in ax.get_yticklabels()
    )
    assert np.isclose(y_title_box.get_x() + y_title_box.get_width(), y_tick_left - 0.018)
    assert y_title_box.get_x() >= 0.018
    assert {text.get_text() for text in fig.texts} >= {"Predicted Label", "True Label"}
    assert fig.axes[1].get_title(loc="left") == "Percent"
    plt.close(fig)


def test_render_confusion_matrix_heatmap_keeps_axis_boxes_clear_of_long_labels():
    fig = render_confusion_matrix_heatmap(
        np.array([0, 0, 1, 1], dtype=np.int64),
        np.array([0, 1, 1, 0], dtype=np.int64),
        ["very-long-wake-label", "very-long-nrem-label"],
        title="demo",
        normalize_rows=True,
    )

    ax = fig.axes[0]
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axis_box = ax.get_position()
    title_boxes = [patch for patch in fig.patches if isinstance(patch, FancyBboxPatch)]
    x_title_box = next(box for box in title_boxes if np.isclose(box.get_width(), axis_box.width))
    y_title_box = next(box for box in title_boxes if np.isclose(box.get_height(), axis_box.height))
    x_tick_bottom = min(
        label.get_window_extent(renderer=renderer).transformed(fig.transFigure.inverted()).y0
        for label in ax.get_xticklabels()
    )
    y_tick_left = min(
        label.get_window_extent(renderer=renderer).transformed(fig.transFigure.inverted()).x0
        for label in ax.get_yticklabels()
    )

    assert x_title_box.get_y() + x_title_box.get_height() <= x_tick_bottom - 0.017
    assert y_title_box.get_x() + y_title_box.get_width() <= y_tick_left - 0.017
    assert x_title_box.get_y() >= 0.018
    assert y_title_box.get_x() >= 0.018
    plt.close(fig)


def test_render_confusion_matrix_heatmap_normalizes_stage_rows_and_can_show_counts():
    fig = render_confusion_matrix_heatmap(
        np.array([0, 0, 0, 1, 1, 2], dtype=np.int64),
        np.array([0, 1, 1, 1, 0, 2], dtype=np.int64),
        ["W", "NREM", "REM"],
        title="demo",
        normalize_rows=True,
        show_raw_counts=True,
    )

    ax = fig.axes[0]
    heatmap = np.asarray(ax.images[0].get_array(), dtype=np.float32)
    assert np.allclose(
        heatmap,
        np.array(
            [
                [33.333332, 66.666664, 0.0],
                [50.0, 50.0, 0.0],
                [0.0, 0.0, 100.0],
            ],
            dtype=np.float32,
        ),
        atol=1e-4,
    )
    assert {text.get_text() for text in ax.texts} >= {"33%\n(1)", "67%\n(2)", "100%\n(1)"}
    assert fig.axes[1].get_title(loc="left") == "Percent"
    plt.close(fig)


def test_render_regression_scatter_plot_draws_scatter_and_diagonal():
    fig = render_regression_scatter_plot(
        np.array([20.0, 30.0, 40.0], dtype=np.float32),
        np.array([21.0, 29.0, 41.0], dtype=np.float32),
        title="demo",
    )

    ax = fig.axes[0]
    assert ax.get_xlabel() == "True Target"
    assert ax.get_ylabel() == "Predicted Target"
    assert len(ax.collections) == 1
    assert len(ax.lines) == 1
    plt.close(fig)


def test_render_binary_roc_curve_plot_draws_roc_and_chance_lines():
    fig = render_binary_roc_curve_plot(
        np.array([0, 0, 1, 1], dtype=np.int64),
        np.array(
            [
                [0.95, 0.05],
                [0.80, 0.20],
                [0.30, 0.70],
                [0.05, 0.95],
            ],
            dtype=np.float32,
        ),
        title="demo",
    )

    assert fig is not None
    ax = fig.axes[0]
    assert ax.get_xlabel() == "False Positive Rate"
    assert ax.get_ylabel() == "True Positive Rate"
    assert len(ax.lines) == 2
    legend = ax.get_legend()
    assert legend is not None
    assert legend.get_frame().get_visible() is True
    plt.close(fig)


def test_downstream_eval_visualizer_respects_stage_gating_and_logs_confusion_matrix(monkeypatch):
    logged = []
    monkeypatch.setattr(downstream_eval.wandb, "run", object(), raising=False)
    monkeypatch.setattr(downstream_eval.wandb, "Image", lambda fig: fig)
    monkeypatch.setattr(downstream_eval.wandb, "log", lambda payload, commit=False: logged.append((payload, commit)))

    visualizer = DownstreamEvalVisualizer(
        EvalVisualizationsConfig(
            enabled=True,
            stages=["val"],
            confusion_matrix=ConfusionMatrixVisualizationConfig(enabled=True),
            roc_curve=EvalVisualizationPlotConfig(enabled=True),
            regression_scatter=EvalVisualizationPlotConfig(enabled=True),
        )
    )

    visualizer.log(
        stage="test",
        preds=np.eye(2, dtype=np.float32),
        targets=np.array([0, 1], dtype=np.int64),
        is_classification=True,
        output_dim=2,
        label_name="sex",
        current_epoch=3,
        class_labels=["female", "male"],
    )
    assert logged == []

    visualizer.log(
        stage="val",
        preds=np.eye(2, dtype=np.float32),
        targets=np.array([0, 1], dtype=np.int64),
        is_classification=True,
        output_dim=2,
        label_name="sex",
        current_epoch=3,
        class_labels=["female", "male"],
    )

    assert len(logged) == 1
    payload, commit = logged[0]
    assert list(payload) == ["val_eval/confusion_matrix", "val_eval/roc_curve"]
    assert commit is False


def test_downstream_eval_visualizer_normalizes_confusion_matrix_and_threads_raw_count_flag(monkeypatch):
    logged = []
    captured = {}
    monkeypatch.setattr(downstream_eval.wandb, "run", object(), raising=False)
    monkeypatch.setattr(downstream_eval.wandb, "Image", lambda fig: fig)
    monkeypatch.setattr(downstream_eval.wandb, "log", lambda payload, commit=False: logged.append((payload, commit)))

    def _fake_confusion_matrix(*args, **kwargs):
        captured.update(kwargs)
        return plt.figure()

    monkeypatch.setattr(downstream_eval, "render_confusion_matrix_heatmap", _fake_confusion_matrix)

    visualizer = DownstreamEvalVisualizer(
        EvalVisualizationsConfig(
            enabled=True,
            stages=["test"],
            confusion_matrix=ConfusionMatrixVisualizationConfig(enabled=True, show_raw_counts=True),
            roc_curve=EvalVisualizationPlotConfig(enabled=False),
            regression_scatter=EvalVisualizationPlotConfig(enabled=False),
        )
    )

    visualizer.log(
        stage="test",
        preds=np.eye(2, dtype=np.float32),
        targets=np.array([0, 1], dtype=np.int64),
        is_classification=True,
        output_dim=2,
        label_name="sex",
        current_epoch=2,
        class_labels=["female", "male"],
    )

    assert captured["normalize_rows"] is True
    assert captured["show_raw_counts"] is True
    assert len(logged) == 1


def test_downstream_eval_visualizer_logs_regression_scatter(monkeypatch):
    logged = []
    monkeypatch.setattr(downstream_eval.wandb, "run", object(), raising=False)
    monkeypatch.setattr(downstream_eval.wandb, "Image", lambda fig: fig)
    monkeypatch.setattr(downstream_eval.wandb, "log", lambda payload, commit=False: logged.append((payload, commit)))

    visualizer = DownstreamEvalVisualizer(
        EvalVisualizationsConfig(
            enabled=True,
            stages=["val", "test"],
            confusion_matrix=ConfusionMatrixVisualizationConfig(enabled=False),
            roc_curve=EvalVisualizationPlotConfig(enabled=False),
            regression_scatter=EvalVisualizationPlotConfig(enabled=True),
        )
    )

    visualizer.log(
        stage="test",
        preds=np.array([20.0, 40.0], dtype=np.float32),
        targets=np.array([18.0, 41.0], dtype=np.float32),
        is_classification=False,
        output_dim=1,
        label_name="age",
        current_epoch=1,
    )

    assert len(logged) == 1
    payload, commit = logged[0]
    assert list(payload) == ["test_eval/regression_scatter"]
    assert commit is False


def test_downstream_eval_visualizer_logs_ahi_summary_scatter(monkeypatch):
    logged = []
    monkeypatch.setattr(downstream_eval.wandb, "run", object(), raising=False)
    monkeypatch.setattr(downstream_eval.wandb, "Image", lambda fig: fig)
    monkeypatch.setattr(downstream_eval.wandb, "log", lambda payload, commit=False: logged.append((payload, commit)))

    visualizer = DownstreamEvalVisualizer(
        EvalVisualizationsConfig(
            enabled=True,
            stages=["val", "test"],
            confusion_matrix=ConfusionMatrixVisualizationConfig(enabled=False),
            roc_curve=EvalVisualizationPlotConfig(enabled=False),
            regression_scatter=EvalVisualizationPlotConfig(enabled=True),
        )
    )

    visualizer.log_ahi_summary_scatter(
        stage="val",
        preds=np.array([12.0, 18.0], dtype=np.float32),
        targets=np.array([10.0, 20.0], dtype=np.float32),
        label_name="ahi",
        current_epoch=4,
    )

    assert len(logged) == 1
    payload, commit = logged[0]
    assert list(payload) == ["val_eval/regression_scatter"]
    assert commit is False


def test_downstream_eval_visualizer_skips_roc_curve_when_targets_have_one_class(monkeypatch):
    logged = []
    monkeypatch.setattr(downstream_eval.wandb, "run", object(), raising=False)
    monkeypatch.setattr(downstream_eval.wandb, "Image", lambda fig: fig)
    monkeypatch.setattr(downstream_eval.wandb, "log", lambda payload, commit=False: logged.append((payload, commit)))

    visualizer = DownstreamEvalVisualizer(
        EvalVisualizationsConfig(
            enabled=True,
            stages=["test"],
            confusion_matrix=ConfusionMatrixVisualizationConfig(enabled=False),
            roc_curve=EvalVisualizationPlotConfig(enabled=True),
            regression_scatter=EvalVisualizationPlotConfig(enabled=False),
        )
    )

    visualizer.log(
        stage="test",
        preds=np.array([[0.9, 0.1], [0.8, 0.2]], dtype=np.float32),
        targets=np.array([0, 0], dtype=np.int64),
        is_classification=True,
        output_dim=2,
        label_name="sex",
        current_epoch=1,
        class_labels=["female", "male"],
    )

    assert logged == []
