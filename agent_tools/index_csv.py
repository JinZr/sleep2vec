from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .configs import config_summary
from .models import repo_relative, resolve_repo_path


def index_summary(
    index_paths: list[str | Path],
    *,
    config: str | Path | None = None,
    label_name: str | None = None,
    sample_path_check: int = 0,
    sample_npz_check: int = 0,
) -> dict[str, Any]:
    resolved_paths = [resolve_repo_path(path) for path in index_paths]
    paths = [path for path in resolved_paths if path is not None]
    missing_inputs = [str(path) for path in paths if not path.exists()]
    frames = [pd.read_csv(path, low_memory=False) for path in paths if path.exists()]
    df = pd.concat(frames, axis=0, ignore_index=True) if frames else pd.DataFrame()
    required_columns = {name: name in df.columns for name in ("path", "split", "duration")}
    duration = {}
    if "duration" in df.columns and not df.empty:
        duration_series = pd.to_numeric(df["duration"], errors="coerce").dropna()
        if not duration_series.empty:
            duration = {
                "min": float(duration_series.min()),
                "median": float(duration_series.median()),
                "max": float(duration_series.max()),
            }
    labels = ["age", "sex", "ahi", "stage3", "stage4", "stage5"]
    if label_name and label_name not in labels:
        labels.append(label_name)
    label_presence = {
        label: {"exists": label in df.columns, "non_null": int(df[label].notna().sum()) if label in df.columns else 0}
        for label in labels
    }
    mask_columns = {}
    for column in df.columns:
        if column.endswith("_mask") or column in {"stage_mask", "ah_event_mask"}:
            values = pd.to_numeric(df[column], errors="coerce").fillna(0)
            mask_columns[column] = {
                "exists": True,
                "true_count": int((values == 1).sum()),
                "false_count": int((values != 1).sum()),
            }
    channel_coverage = {}
    if config:
        cfg = config_summary(config)
        for channel in cfg["data"]["data_channel_names"]:
            if channel == "stage5":
                mask_column = "stage_mask"
            elif channel == "ahi":
                mask_column = "ah_event_mask"
            else:
                mask_column = f"{channel}_mask"
            available = mask_columns.get(mask_column, {}).get("true_count", len(df) if not df.empty else 0)
            channel_coverage[channel] = {"mask_column": mask_column, "available_rows": int(available)}
    source_col = _first_existing(df, ["source", "dataset", "sample_source", "original_dataset"])
    label_cols = _label_columns(df, label_name=label_name)
    split_source_label_counts = {}
    if "split" in df.columns and source_col:
        for label in label_cols:
            counts = (
                df.groupby(["split", source_col, label], dropna=False)
                .size()
                .reset_index(name="rows")
                .to_dict(orient="records")
            )
            split_source_label_counts[label] = counts

    channel_mask_coverage_by_split_source = {}
    if "split" in df.columns and source_col:
        for column in mask_columns:
            values = pd.to_numeric(df[column], errors="coerce")
            tmp = df[["split", source_col]].copy()
            tmp["available_fraction"] = (values == 1).astype(float)
            channel_mask_coverage_by_split_source[column] = (
                tmp.groupby(["split", source_col], dropna=False)["available_fraction"]
                .agg(["count", "mean"])
                .reset_index()
                .rename(columns={"count": "rows"})
                .to_dict(orient="records")
            )

    numeric_shift_metrics = _numeric_shift_metrics(df)

    path_check = {"checked": 0, "existing": 0, "missing_examples": []}
    if sample_path_check and "path" in df.columns:
        examples = [Path(str(path)) for path in df["path"].dropna().head(sample_path_check)]
        path_check = {
            "checked": len(examples),
            "existing": sum(path.exists() for path in examples),
            "missing_examples": [str(path) for path in examples if not path.exists()][:5],
        }

    warnings: list[str] = []
    blocking_issues = [f"Index CSV not found: {path}" for path in missing_inputs]
    for column, exists in required_columns.items():
        if not exists:
            blocking_issues.append(f"Index CSV missing required column: {column}")
    if sample_npz_check:
        warnings.append("--sample-npz-check is accepted but only path existence is checked by this lightweight tool.")

    return {
        "index_paths": [repo_relative(path) for path in paths],
        "rows": int(len(df)),
        "columns": list(df.columns),
        "required_columns": required_columns,
        "split_counts": df["split"].value_counts(dropna=False).to_dict() if "split" in df.columns else {},
        "source_counts": (
            df["source"].value_counts(dropna=False).to_dict()
            if "source" in df.columns
            else df["dataset"].value_counts(dropna=False).to_dict() if "dataset" in df.columns else {}
        ),
        "duration": duration,
        "label_presence": label_presence,
        "mask_columns": mask_columns,
        "channel_coverage_from_config": channel_coverage,
        "split_source_label_counts": split_source_label_counts,
        "channel_mask_coverage_by_split_source": channel_mask_coverage_by_split_source,
        "numeric_shift_metrics": numeric_shift_metrics,
        "sample_path_check": path_check,
        "warnings": warnings,
        "blocking_issues": blocking_issues,
    }


def _first_existing(df: pd.DataFrame, names: list[str]) -> str | None:
    for name in names:
        if name in df.columns:
            return name
    return None


def _label_columns(df: pd.DataFrame, *, label_name: str | None = None) -> list[str]:
    candidates = ["label", "target", "groundtruth", "sex", "age", "ahi"]
    if label_name:
        candidates.insert(0, label_name)
    candidates = list(dict.fromkeys(candidates))
    labels: list[str] = []
    for column in candidates:
        if column not in df.columns:
            continue
        unique_count = df[column].nunique(dropna=True)
        if unique_count <= 20:
            labels.append(column)
    return labels


def _numeric_shift_metrics(df: pd.DataFrame) -> dict[str, Any]:
    if "split" not in df.columns or df.empty:
        return {}
    candidates = [
        "duration_hours",
        "duration",
        "wake_fraction",
        "wake_frac",
        "sleep_hours",
        "num_tokens",
        "token_count",
    ]
    out: dict[str, Any] = {}
    train_like = df[df["split"].isin(["train", "val"])]
    test = df[df["split"] == "test"]
    if train_like.empty or test.empty:
        return out
    for column in candidates:
        if column not in df.columns:
            continue
        left = pd.to_numeric(train_like[column], errors="coerce").dropna()
        right = pd.to_numeric(test[column], errors="coerce").dropna()
        if left.empty or right.empty:
            continue
        pooled = ((left.var(ddof=1) + right.var(ddof=1)) / 2) ** 0.5
        smd = float((left.mean() - right.mean()) / pooled) if pooled else 0.0
        out[column] = {
            "train_val_mean": float(left.mean()),
            "test_mean": float(right.mean()),
            "train_val_median": float(left.median()),
            "test_median": float(right.median()),
            "standardized_mean_difference": smd,
        }
    return out
