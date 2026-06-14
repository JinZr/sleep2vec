from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError

STAGE_LABELS = ("N1", "N2", "N3", "REM")


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


def plot_cohort(
    run_dir: Path,
    *,
    group_column: str = "source",
    stage_source: str | None = None,
    adjust_covariates: list[str] | None = None,
) -> list[Path]:
    frame = _load_cohort_frame(run_dir, group_column=group_column)
    stage_prefix = _select_stage_source(frame, stage_source)
    plots_dir = run_dir / "plots" / "cohort"
    plots_dir.mkdir(parents=True, exist_ok=True)

    outputs = []
    sleep_metrics = []
    if stage_prefix is not None:
        sleep_metrics = _sleep_metric_specs(frame, stage_prefix)
        outputs.extend(
            [
                _plot_cohort_stage_composition(
                    frame, stage_prefix, group_column, plots_dir / "cohort_stage_composition.png"
                ),
                _plot_cohort_sleep_metrics(
                    frame,
                    sleep_metrics,
                    group_column,
                    plots_dir / "cohort_sleep_metrics.png",
                ),
                _plot_cohort_stage_ratio_distribution(
                    frame,
                    stage_prefix,
                    group_column,
                    plots_dir / "cohort_stage_ratio_distribution.png",
                ),
            ]
        )

    if stage_prefix is not None and frame[group_column].nunique(dropna=True) > 1:
        outputs.append(
            _plot_harmonization_diagnostics(
                frame,
                sleep_metrics,
                group_column,
                adjust_covariates or [],
                plots_dir / "cohort_harmonization_diagnostics.png",
            )
        )

    respiratory = _respiratory_metric_specs(frame)
    if respiratory:
        outputs.append(
            _plot_metric_panel(
                frame,
                respiratory,
                group_column,
                "Respiratory and Hypoxemia Risk",
                plots_dir / "cohort_respiratory_risk.png",
            )
        )

    microstructure = _microstructure_metric_specs(frame)
    if microstructure:
        outputs.append(
            _plot_metric_panel(
                frame,
                microstructure,
                group_column,
                "Sleep Microstructure and Autonomic Signals",
                plots_dir / "cohort_microstructure_autonomic.png",
            )
        )

    if not outputs:
        raise ValueError("No usable sleep2stat cohort metrics found.")
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


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame()


def _load_cohort_frame(run_dir: Path, *, group_column: str) -> pd.DataFrame:
    night_path = run_dir / "tables" / "night_stats.csv"
    if not night_path.exists():
        raise FileNotFoundError(f"sleep2stat night_stats.csv not found: {night_path}")
    night = _read_csv(night_path)
    if night.empty:
        raise ValueError(f"sleep2stat night_stats.csv is empty: {night_path}")
    if "record_id" not in night.columns:
        raise ValueError("sleep2stat night_stats.csv must contain record_id.")

    manifest_path = run_dir / "record_manifest.csv"
    if manifest_path.exists():
        manifest = _read_csv(manifest_path)
        if not manifest.empty and "record_id" in manifest.columns:
            extra = [column for column in manifest.columns if column == "record_id" or column not in night.columns]
            night = night.merge(manifest[extra], on="record_id", how="left")

    if group_column not in night.columns:
        if group_column != "source":
            raise ValueError(f"group column {group_column!r} was not found in night_stats.csv or record_manifest.csv.")
        night[group_column] = "cohort"
    night[group_column] = night[group_column].fillna("unknown").astype(str)
    return night


def _select_stage_source(frame: pd.DataFrame, requested: str | None) -> str | None:
    if requested in (None, ""):
        return None
    if all(_stage_ratio_column(frame, requested, stage) is not None for stage in STAGE_LABELS):
        return requested
    raise ValueError(f"stage source {requested!r} does not have complete N1/N2/N3/REM ratio columns.")


def _stage_ratio_column(frame: pd.DataFrame, prefix: str, stage: str) -> str | None:
    canonical = f"{prefix}_{stage}_ratio_TST"
    return canonical if canonical in frame.columns else None


def _sleep_metric_specs(frame: pd.DataFrame, prefix: str) -> list[tuple[str, str, float]]:
    candidates = [
        ("TST hours", [f"{prefix}_TST_min"], 1.0 / 60.0),
        ("Sleep efficiency %", [f"{prefix}_SE_pct"], 1.0),
        ("WASO SPT min", [f"{prefix}_WASO_SPT_min"], 1.0),
        ("SOL min", [f"{prefix}_SOL_min"], 1.0),
        ("REM latency min", [f"{prefix}_REM_latency_min"], 1.0),
        ("Sleep-to-wake index", [f"{prefix}_sleep_to_wake_transition_index"], 1.0),
        ("Stage shift / sleep hr", [f"{prefix}_stage_shift_rate_per_sleep_hour"], 1.0),
    ]
    selected = []
    for label, columns, scale in candidates:
        column = next((item for item in columns if item in frame.columns), None)
        if column is not None:
            selected.append((label, column, scale))
    return selected


def _respiratory_metric_specs(frame: pd.DataFrame) -> list[tuple[str, str, float]]:
    suffixes = [
        ("Pred AHI", "_pred_ahi", 1.0),
        ("Pred REM AHI", "_pred_REM_AHI_onset_stage", 1.0),
        ("Pred NREM AHI", "_pred_NREM_AHI_onset_stage", 1.0),
        ("ODI3", "ODI3_per_recording_hour", 1.0),
        ("ODI4", "ODI4_per_recording_hour", 1.0),
        ("T90 % recording", "spo2_t90_pct_recording", 1.0),
        ("SpO2 nadir", "spo2_nadir", 1.0),
        ("Desaturation area/hr", "desaturation_area_burden_pctmin_per_recording_hour", 1.0),
        ("Hypoxic burden/hr", "resp_event_hypoxic_burden_pctmin_per_recording_hour", 1.0),
    ]
    selected = []
    used: set[str] = set()
    used_labels: set[str] = set()
    for label, suffix, scale in suffixes:
        if label in used_labels:
            continue
        matches = [
            column
            for column in frame.columns
            if column.lower().endswith(suffix.lower()) and column not in used and _has_numeric_values(frame[column])
        ]
        if matches:
            column = matches[0]
            selected.append((label, column, scale))
            used.add(column)
            used_labels.add(label)
    return selected


def _microstructure_metric_specs(frame: pd.DataFrame) -> list[tuple[str, str, float]]:
    patterns = [
        ("N3 delta", "_N3_delta_mean", 1.0),
        ("REM alpha", "_REM_alpha_mean", 1.0),
        ("Sigma", "_sigma_rel_mean", 1.0),
        ("Delta", "_delta_rel_mean", 1.0),
        ("Spindle density", "spindle_density_per_min_N2N3", 1.0),
        ("Slow-wave density", "slowwave_density_per_min_NREM", 1.0),
        ("REM density", "rapid_eye_movement_density_per_min_REM", 1.0),
        ("Spindle density", "spindles_event_density_per_hour", 1.0),
        ("Slow-wave density", "slowwaves_event_density_per_hour", 1.0),
        ("HRV", "hrv", 1.0),
    ]
    return _metric_specs_from_patterns(frame, patterns)


def _metric_specs_from_patterns(
    frame: pd.DataFrame, patterns: list[tuple[str, str, float]]
) -> list[tuple[str, str, float]]:
    selected = []
    used: set[str] = set()
    for label, pattern, scale in patterns:
        matches = [
            column
            for column in frame.columns
            if pattern.lower() in column.lower() and column not in used and _has_numeric_values(frame[column])
        ]
        if matches:
            column = matches[0]
            selected.append((label, column, scale))
            used.add(column)
    return selected


def _has_numeric_values(series: pd.Series) -> bool:
    return pd.to_numeric(series, errors="coerce").notna().any()


def _metric_values(frame: pd.DataFrame, column: str, scale: float) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce") * float(scale)


def _group_order(frame: pd.DataFrame, group_column: str) -> list[str]:
    counts = frame[group_column].fillna("unknown").astype(str).value_counts()
    return counts.index.tolist()


def _plot_cohort_stage_composition(frame: pd.DataFrame, prefix: str, group_column: str, path: Path) -> Path:
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
    tst_column = f"{prefix}_TST_min"
    tib_column = f"{prefix}_TIB_min"
    use_tib = (
        tst_column in frame.columns
        and tib_column in frame.columns
        and _has_numeric_values(frame[tst_column])
        and _has_numeric_values(frame[tib_column])
    )
    labels = ["Wake", *STAGE_LABELS] if use_tib else list(STAGE_LABELS)
    colors = (
        [_PRIMARY_PALE, _PRIMARY_LIGHT, _PRIMARY, _PRIMARY_DARK, _PRIMARY_MID]
        if use_tib
        else [
            _PRIMARY_LIGHT,
            _PRIMARY,
            _PRIMARY_DARK,
            _PRIMARY_MID,
        ]
    )
    groups = ["Overall"] + _group_order(frame, group_column)
    rows = []
    for group in groups:
        subset = frame if group == "Overall" else frame[frame[group_column] == group]
        if use_tib:
            tib = _metric_values(subset, tib_column, 1.0)
            tst = _metric_values(subset, tst_column, 1.0)
            tib_denominator = tib.where(tib > 0, np.nan)
            tst_ratio_tib = (tst / tib_denominator).clip(lower=0.0, upper=1.0)
            wake_ratio_tib = ((tib - tst) / tib_denominator).clip(lower=0.0, upper=1.0)
            rows.append(
                [wake_ratio_tib.mean()]
                + [
                    (_metric_values(subset, _stage_ratio_column(frame, prefix, stage), 1.0) * tst_ratio_tib).mean()
                    for stage in STAGE_LABELS
                ]
            )
        else:
            rows.append(
                [
                    _metric_values(subset, _stage_ratio_column(frame, prefix, stage), 1.0).mean()
                    for stage in STAGE_LABELS
                ]
            )
    data = np.nan_to_num(np.asarray(rows, dtype=np.float64), nan=0.0)

    fig_width = max(8.2, 0.65 * len(groups) + 4.0)
    fig, ax = plt.subplots(figsize=(fig_width, 5.8), facecolor=_FIGURE_BG)
    style_axes(ax, show_grid=True)
    x = np.arange(len(groups))
    bottom = np.zeros(len(groups), dtype=np.float64)
    for idx, label in enumerate(labels):
        ax.bar(x, data[:, idx], bottom=bottom, color=colors[idx], edgecolor=_FIGURE_BG, linewidth=0.8, label=label)
        bottom += data[:, idx]
    ax.set_xticks(x, labels=groups)
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_ha("right")
    ax.set_ylim(0.0, max(1.0, float(np.nanmax(bottom)) if bottom.size else 1.0))
    ylabel = "Ratio of TIB" if use_tib else "Ratio of TST"
    style_plot_text(ax, title="Cohort Stage Composition", xlabel="Center", ylabel=ylabel)
    legend = add_openai_legend(ax, loc="upper right", fontsize=9, title_fontsize=9)
    legend.set_zorder(20)
    apply_plot_layout(fig, defaults={"left": 0.10, "right": 0.98, "bottom": 0.28, "top": 0.86})
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_cohort_sleep_metrics(
    frame: pd.DataFrame,
    metrics: list[tuple[str, str, float]],
    group_column: str,
    path: Path,
) -> Path:
    if not metrics:
        raise ValueError("No sleep metric columns were found for cohort_sleep_metrics.")
    return _plot_metric_panel(frame, metrics, group_column, "Cohort Sleep Metrics", path)


def _plot_cohort_stage_ratio_distribution(
    frame: pd.DataFrame,
    prefix: str,
    group_column: str,
    path: Path,
) -> Path:
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
    fig, ax = plt.subplots(figsize=(8.8, 5.6), facecolor=_FIGURE_BG)
    style_axes(ax, show_grid=True)
    stage_values = [
        _metric_values(frame, _stage_ratio_column(frame, prefix, stage), 1.0).dropna().to_numpy()
        for stage in STAGE_LABELS
    ]
    box = ax.boxplot(stage_values, positions=np.arange(len(STAGE_LABELS)), patch_artist=True, widths=0.48)
    palette = [_PRIMARY_LIGHT, _PRIMARY, _PRIMARY_DARK, _PRIMARY_MID]
    for idx, patch in enumerate(box["boxes"]):
        patch.set_facecolor(palette[idx % len(palette)])
        patch.set_edgecolor(_PRIMARY_DARK)
        patch.set_linewidth(1.2)
        patch.set_alpha(0.82)
    for item in box["medians"]:
        item.set_color(_PRIMARY_DARK)
        item.set_linewidth(1.5)

    groups = _group_order(frame, group_column)
    if len(groups) > 1:
        offsets = np.linspace(-0.24, 0.24, num=len(groups))
        for group_idx, group in enumerate(groups):
            subset = frame[frame[group_column] == group]
            means = [
                _metric_values(subset, _stage_ratio_column(frame, prefix, stage), 1.0).mean() for stage in STAGE_LABELS
            ]
            ax.scatter(
                np.arange(len(STAGE_LABELS)) + offsets[group_idx],
                means,
                s=44,
                color=palette[group_idx % len(palette)],
                edgecolor=_PRIMARY_DARK,
                linewidth=0.8,
                label=group,
                zorder=4,
            )
        legend = add_openai_legend(ax, loc="upper right", fontsize=8, title_fontsize=8)
        legend.set_zorder(20)

    ax.set_xticks(np.arange(len(STAGE_LABELS)), labels=STAGE_LABELS)
    ax.set_ylim(0.0, 1.0)
    style_plot_text(ax, title="Stage Ratio Distribution", xlabel="Stage", ylabel="Ratio of TST")
    apply_plot_layout(fig, defaults={"left": 0.12, "right": 0.98, "bottom": 0.16, "top": 0.86})
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_metric_panel(
    frame: pd.DataFrame,
    metrics: list[tuple[str, str, float]],
    group_column: str,
    title: str,
    path: Path,
) -> Path:
    import matplotlib.pyplot as plt

    from sleep2vec.visualization.theme import (
        _FIGURE_BG,
        _PRIMARY,
        _PRIMARY_DARK,
        _PRIMARY_LIGHT,
        _PRIMARY_MID,
        apply_plot_layout,
        style_axes,
        use_openai_like_theme,
    )

    use_openai_like_theme()
    metrics = metrics[:8]
    groups = _group_order(frame, group_column)
    ncols = min(4, max(1, len(metrics)))
    nrows = int(np.ceil(len(metrics) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(max(10.0, 3.4 * ncols), max(6.4, 3.6 * nrows)),
        facecolor=_FIGURE_BG,
        squeeze=False,
    )
    palette = [_PRIMARY_LIGHT, _PRIMARY, _PRIMARY_MID, _PRIMARY_DARK]
    for idx, (label, column, scale) in enumerate(metrics):
        ax = axes[idx // ncols][idx % ncols]
        style_axes(ax, show_grid=True)
        values = [
            _metric_values(frame[frame[group_column] == group], column, scale).dropna().to_numpy() for group in groups
        ]
        values = [item if item.size else np.asarray([np.nan]) for item in values]
        box = ax.boxplot(values, patch_artist=True, widths=0.55, showfliers=False)
        for patch_idx, patch in enumerate(box["boxes"]):
            patch.set_facecolor(palette[patch_idx % len(palette)])
            patch.set_edgecolor(_PRIMARY_DARK)
            patch.set_alpha(0.82)
            patch.set_linewidth(1.1)
        for median in box["medians"]:
            median.set_color(_PRIMARY_DARK)
            median.set_linewidth(1.4)
        ax.set_xticks(np.arange(1, len(groups) + 1), labels=groups)
        for tick in ax.get_xticklabels():
            tick.set_rotation(45)
            tick.set_ha("right")
            tick.set_fontsize(9)
        ax.set_title(label, fontsize=13, pad=8)
    for idx in range(len(metrics), nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")
    fig.suptitle(title, fontsize=18, fontweight=700)
    apply_plot_layout(
        fig,
        defaults={"left": 0.07, "right": 0.98, "bottom": 0.18, "top": 0.82, "hspace": 0.72, "wspace": 0.30},
    )
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_harmonization_diagnostics(
    frame: pd.DataFrame,
    metrics: list[tuple[str, str, float]],
    group_column: str,
    adjust_covariates: list[str],
    path: Path,
) -> Path:
    import matplotlib.pyplot as plt

    from sleep2vec.visualization.theme import (
        _FIGURE_BG,
        _OPENAI_BLUE_CMAP,
        _PRIMARY,
        _PRIMARY_DARK,
        apply_plot_layout,
        style_axes,
        style_plot_text,
        use_openai_like_theme,
    )

    use_openai_like_theme()
    metrics = metrics[:8]
    labels = [label for label, _, _ in metrics]
    columns = [column for _, column, _ in metrics]
    groups = _group_order(frame, group_column)
    adjusted = _residualized_frame(frame, metrics, adjust_covariates)
    ncols = 4 if adjusted is not None else 3
    fig, axes = plt.subplots(1, ncols, figsize=(4.8 * ncols, 6.4), facecolor=_FIGURE_BG, squeeze=False)
    axes = axes[0]

    counts = frame[group_column].value_counts().reindex(groups).fillna(0)
    style_axes(axes[0], show_grid=True)
    axes[0].bar(np.arange(len(groups)), counts.to_numpy(), color=_PRIMARY, edgecolor=_PRIMARY_DARK, linewidth=1.0)
    axes[0].set_xticks(np.arange(len(groups)), labels=groups)
    for tick in axes[0].get_xticklabels():
        tick.set_rotation(45)
        tick.set_ha("right")
    style_plot_text(axes[0], title="Records", xlabel="Center", ylabel="Count")

    missing = _group_metric_matrix(frame, columns, group_column, groups, mode="missing")
    _plot_matrix(
        axes[1],
        missing,
        groups,
        labels,
        "Missingness",
        "% missing",
        cmap=_OPENAI_BLUE_CMAP,
        vmin=0,
        vmax=100,
    )

    raw_shift = _group_metric_matrix(frame, columns, group_column, groups, mode="shift")
    _plot_matrix(axes[2], raw_shift, groups, labels, "Raw Center Shift", "z-score", cmap="coolwarm", vmin=-2, vmax=2)

    if adjusted is not None:
        adjusted_shift = _group_metric_matrix(adjusted, columns, group_column, groups, mode="shift")
        _plot_matrix(
            axes[3],
            adjusted_shift,
            groups,
            labels,
            "Adjusted Center Shift",
            "residual z-score",
            cmap="coolwarm",
            vmin=-2,
            vmax=2,
        )

    fig.suptitle("Cohort Harmonization Diagnostics", fontsize=18, fontweight=700)
    apply_plot_layout(fig, defaults={"left": 0.06, "right": 0.98, "bottom": 0.34, "top": 0.80, "wspace": 0.52})
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _group_metric_matrix(
    frame: pd.DataFrame,
    columns: list[str],
    group_column: str,
    groups: list[str],
    *,
    mode: str,
) -> np.ndarray:
    matrix = np.full((len(groups), len(columns)), np.nan, dtype=np.float64)
    numeric = {column: pd.to_numeric(frame[column], errors="coerce") for column in columns}
    for col_idx, column in enumerate(columns):
        values = numeric[column]
        if mode == "shift":
            std = float(values.std(skipna=True))
            mean = float(values.mean(skipna=True))
            z = (values - mean) / std if std > 0 else values * np.nan
        for group_idx, group in enumerate(groups):
            mask = frame[group_column] == group
            if mode == "missing":
                matrix[group_idx, col_idx] = float(values[mask].isna().mean() * 100.0) if mask.any() else np.nan
            else:
                matrix[group_idx, col_idx] = float(z[mask].mean()) if mask.any() else np.nan
    return matrix


def _plot_matrix(
    ax,
    matrix: np.ndarray,
    row_labels: list[str],
    column_labels: list[str],
    title: str,
    colorbar_label: str,
    *,
    cmap,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    image = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, zorder=2)
    ax.set_title(title, fontsize=13, pad=8)
    ax.set_xticks(np.arange(len(column_labels)), labels=column_labels)
    ax.set_yticks(np.arange(len(row_labels)), labels=row_labels)
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)
        tick.set_ha("right")
        tick.set_fontsize(8)
    for tick in ax.get_yticklabels():
        tick.set_fontsize(9)
    ax.grid(False)
    colorbar = ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label(colorbar_label)
    colorbar.ax.tick_params(labelsize=8)


def _residualized_frame(
    frame: pd.DataFrame,
    metrics: list[tuple[str, str, float]],
    covariates: list[str],
) -> pd.DataFrame | None:
    available = [column for column in covariates if column in frame.columns]
    if not available:
        return None
    design = _covariate_design(frame, available)
    if design is None:
        return None
    output = frame.copy()
    for _, column, _ in metrics:
        y = pd.to_numeric(frame[column], errors="coerce")
        valid = y.notna() & np.isfinite(design).all(axis=1)
        if int(valid.sum()) <= design.shape[1]:
            continue
        x = design[valid.to_numpy()]
        target = y[valid].to_numpy(dtype=np.float64)
        beta, *_ = np.linalg.lstsq(x, target, rcond=None)
        residual = pd.Series(np.nan, index=frame.index, dtype=float)
        residual.loc[valid] = target - x @ beta
        output[column] = residual
    return output


def _covariate_design(frame: pd.DataFrame, columns: list[str]) -> np.ndarray | None:
    values = [np.ones(len(frame), dtype=np.float64)]
    for column in columns:
        series = frame[column]
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().any():
            values.append(numeric.to_numpy(dtype=np.float64))
            continue
        codes, _ = pd.factorize(series.fillna("unknown").astype(str), sort=True)
        if np.unique(codes).size > 1:
            values.append(codes.astype(np.float64))
    if len(values) == 1:
        return None
    return np.column_stack(values)


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
