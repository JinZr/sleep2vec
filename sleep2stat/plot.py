from __future__ import annotations

from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError


def plot_record(run_dir: Path, record_id: str) -> list[Path]:
    record_dir = run_dir / "per_record" / record_id
    if not record_dir.exists():
        raise FileNotFoundError(f"sleep2stat per-record directory not found: {record_dir}")
    plots_dir = record_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    epoch = _read_table(record_dir, "epoch_alignment")
    second = _read_table(record_dir, "second_alignment")
    events = _read_table(record_dir, "event_alignment")
    if events.empty:
        events = _read_table(record_dir, "events")
    outputs = [
        _plot_hypnogram_overlay(epoch, plots_dir / "hypnogram_overlay.png"),
        _plot_ahi_spo2_trace(second, events, plots_dir / "ahi_spo2_trace.png"),
    ]
    return outputs


def _read_table(record_dir: Path, stem: str) -> pd.DataFrame:
    for suffix in (".csv.gz", ".csv"):
        path = record_dir / f"{stem}{suffix}"
        if not path.exists():
            continue
        try:
            return pd.read_csv(path)
        except EmptyDataError:
            return pd.DataFrame()
    return pd.DataFrame()


def _plot_hypnogram_overlay(epoch: pd.DataFrame, path: Path) -> Path:
    import matplotlib.pyplot as plt
    from sleep2vec.visualization.theme import (
        _FIGURE_BG,
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

    use_openai_like_theme()
    fig, ax = plt.subplots(figsize=(10.0, 3.8), facecolor=_FIGURE_BG)
    style_axes(ax, show_grid=True)
    pred_columns = [
        column for column in epoch.columns if column.endswith("_pred") and pd.api.types.is_numeric_dtype(epoch[column])
    ]
    x = epoch["start_sec"] / 3600.0 if "start_sec" in epoch.columns else epoch.index
    palette = [_PRIMARY_DARK, _PRIMARY, _PRIMARY_MID, _PRIMARY_LIGHT]
    for idx, column in enumerate(pred_columns[:4]):
        ax.step(
            x,
            epoch[column],
            where="post",
            label=column.removesuffix("_pred"),
            color=palette[idx % len(palette)],
            linewidth=2.0,
            zorder=3,
        )
    if not pred_columns:
        ax.text(0.5, 0.5, "No hypnogram columns", ha="center", va="center", transform=ax.transAxes, zorder=20)
    style_plot_text(ax, title="Hypnogram Overlay", xlabel="Hours", ylabel="Stage")
    ax.set_yticks([0, 1, 2, 3, 4])
    ax.set_yticklabels(["W", "N1", "N2", "N3", "REM"])
    if pred_columns:
        legend = add_openai_legend(ax, loc="upper right", fontsize=8, title_fontsize=8)
        legend.set_zorder(20)
    apply_plot_layout(fig, defaults={"left": 0.10, "right": 0.98, "bottom": 0.22, "top": 0.84})
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_ahi_spo2_trace(second: pd.DataFrame, events: pd.DataFrame, path: Path) -> Path:
    import matplotlib.pyplot as plt
    from sleep2vec.visualization.theme import (
        _FIGURE_BG,
        _PRIMARY,
        _PRIMARY_DARK,
        _PRIMARY_LIGHT,
        _PRIMARY_MID,
        _PRIMARY_PALE,
        add_openai_legend,
        apply_plot_layout,
        style_axes,
        style_plot_text,
        use_openai_like_theme,
    )

    use_openai_like_theme()
    fig, ax = plt.subplots(figsize=(10.0, 3.8), facecolor=_FIGURE_BG)
    style_axes(ax, show_grid=True)
    x = second["start_sec"] / 3600.0 if "start_sec" in second.columns else second.index
    plotted = False
    palette = [_PRIMARY_DARK, _PRIMARY, _PRIMARY_MID, _PRIMARY_LIGHT]
    line_idx = 0
    for column in second.columns:
        if column.endswith("_prob") or column.endswith("_pred") or column.lower().startswith("spo2"):
            if pd.api.types.is_numeric_dtype(second[column]):
                ax.plot(
                    x,
                    second[column],
                    label=column,
                    color=palette[line_idx % len(palette)],
                    linewidth=1.8,
                    zorder=3,
                )
                line_idx += 1
                plotted = True
    if not events.empty and {"onset_sec", "offset_sec"}.issubset(events.columns):
        for _, row in events.iterrows():
            ax.axvspan(
                float(row["onset_sec"]) / 3600.0,
                float(row["offset_sec"]) / 3600.0,
                color=_PRIMARY_PALE,
                alpha=0.45,
                zorder=1,
            )
        plotted = True
    if not plotted:
        ax.text(0.5, 0.5, "No AHI/SpO2 trace columns", ha="center", va="center", transform=ax.transAxes, zorder=20)
    style_plot_text(ax, title="AHI and SpO2 Trace", xlabel="Hours", ylabel="Value")
    if plotted:
        legend = add_openai_legend(ax, loc="upper right", fontsize=8, title_fontsize=8)
        legend.set_zorder(20)
    apply_plot_layout(fig, defaults={"left": 0.10, "right": 0.98, "bottom": 0.22, "top": 0.84})
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path
