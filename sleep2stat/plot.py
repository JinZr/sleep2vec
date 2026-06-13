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

    fig, ax = plt.subplots(figsize=(10, 3))
    pred_columns = [
        column for column in epoch.columns if column.endswith("_pred") and pd.api.types.is_numeric_dtype(epoch[column])
    ]
    x = epoch["start_sec"] / 3600.0 if "start_sec" in epoch.columns else epoch.index
    for column in pred_columns[:4]:
        ax.step(x, epoch[column], where="post", label=column.removesuffix("_pred"))
    if not pred_columns:
        ax.text(0.5, 0.5, "No hypnogram columns", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("Hours")
    ax.set_ylabel("Stage")
    ax.set_yticks([0, 1, 2, 3, 4])
    ax.set_yticklabels(["W", "N1", "N2", "N3", "REM"])
    if pred_columns:
        ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_ahi_spo2_trace(second: pd.DataFrame, events: pd.DataFrame, path: Path) -> Path:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 3))
    x = second["start_sec"] / 3600.0 if "start_sec" in second.columns else second.index
    plotted = False
    for column in second.columns:
        if column.endswith("_prob") or column.endswith("_pred") or column.lower().startswith("spo2"):
            if pd.api.types.is_numeric_dtype(second[column]):
                ax.plot(x, second[column], label=column, linewidth=1.0)
                plotted = True
    if not events.empty and {"onset_sec", "offset_sec"}.issubset(events.columns):
        for _, row in events.iterrows():
            ax.axvspan(float(row["onset_sec"]) / 3600.0, float(row["offset_sec"]) / 3600.0, color="tab:red", alpha=0.15)
        plotted = True
    if not plotted:
        ax.text(0.5, 0.5, "No AHI/SpO2 trace columns", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("Hours")
    ax.set_ylabel("Value")
    if plotted:
        ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
