"""Dataset index CSV summaries: coverage, labels, task covariates, and sampled path checks.

Domain module, but a config-summary *consumer* rather than a leaf: it imports
``configs`` and ``configs`` never imports it back, so the edge stays one-way and
the guard tolerates it. Removing it would mean taking the config summary as an
argument instead.

That import is also why this module must not be aggregated in
``domain/__init__``; see the package docstring.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypedDict

import pandas as pd

from ..configs import config_summary
from ..models import ConfigSummaryInput, repo_relative, resolve_repo_path


class LabelPresence(TypedDict):
    exists: bool
    non_null: int


class MaskCoverage(TypedDict):
    exists: bool
    true_count: int
    false_count: int


class ChannelCoverage(TypedDict):
    mask_column: str
    available_rows: int


class CovariateSummary(TypedDict):
    exists: bool
    non_null_rows: int
    missing_rows: int


class KeySummary(CovariateSummary):
    key_column: str
    unique_keys: int
    sidecar_key_count: int | None
    missing_from_sidecars: int | None
    missing_from_sidecars_examples: list[str]


class NumericShiftMetrics(TypedDict):
    train_val_mean: float
    test_mean: float
    train_val_median: float
    test_median: float
    standardized_mean_difference: float


class SamplePathCheck(TypedDict):
    checked: int
    existing: int
    missing_examples: list[str]


class IndexStatistics(TypedDict):
    duration: dict[str, float]
    label_presence: dict[str, LabelPresence]
    mask_columns: dict[str, MaskCoverage]
    channel_coverage_from_config: dict[str, ChannelCoverage]
    # Grouping columns and their values come from the input dataframe.
    split_source_label_counts: dict[str, list[dict[str, Any]]]
    channel_mask_coverage_by_split_source: dict[str, list[dict[str, Any]]]
    numeric_shift_metrics: dict[str, NumericShiftMetrics]


class IndexSummary(IndexStatistics):
    index_paths: list[str]
    rows: int
    columns: list[str]
    required_columns: dict[str, bool]
    split_counts: dict[Any, int]
    source_counts: dict[Any, int]
    survival_key: KeySummary | None
    multilabel_key: KeySummary | None
    covariates: dict[str, CovariateSummary]
    sample_path_check: SamplePathCheck
    warnings: list[str]
    blocking_issues: list[str]


def _index_statistics(
    df: pd.DataFrame,
    cfg: ConfigSummaryInput | None,
    *,
    label_name: str | None,
    split_column: str,
) -> IndexStatistics:
    duration: dict[str, float] = {}
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
    label_presence: dict[str, LabelPresence] = {
        label: {"exists": label in df.columns, "non_null": int(df[label].notna().sum()) if label in df.columns else 0}
        for label in labels
    }
    mask_columns: dict[str, MaskCoverage] = {}
    for column in df.columns:
        if column.endswith("_mask") or column in {"stage_mask", "ah_event_mask"}:
            values = pd.to_numeric(df[column], errors="coerce").fillna(0)
            mask_columns[column] = {
                "exists": True,
                "true_count": int((values == 1).sum()),
                "false_count": int((values != 1).sum()),
            }
    channel_coverage: dict[str, ChannelCoverage] = {}
    if cfg:
        data: Any = cfg.get("data") or {}
        for channel in data.get("data_channel_names", []):
            if channel == "stage5":
                mask_column = "stage_mask"
            elif channel == "ahi":
                mask_column = "ah_event_mask"
            else:
                mask_column = f"{channel}_mask"
            coverage: MaskCoverage | dict[str, int] = mask_columns.get(mask_column, {})
            available = coverage.get("true_count", len(df) if not df.empty else 0)
            channel_coverage[channel] = {"mask_column": mask_column, "available_rows": int(available)}
    source_col = _first_existing(df, ["source", "dataset", "sample_source", "original_dataset"])
    label_cols = _label_columns(df, label_name=label_name)
    split_source_label_counts = {}
    if split_column in df.columns and source_col:
        for label in label_cols:
            counts = (
                df.groupby([split_column, source_col, label], dropna=False)
                .size()
                .reset_index(name="rows")
                .to_dict(orient="records")
            )
            split_source_label_counts[label] = counts

    channel_mask_coverage_by_split_source = {}
    if split_column in df.columns and source_col:
        for column in mask_columns:
            values = pd.to_numeric(df[column], errors="coerce")
            tmp = df[[split_column, source_col]].copy()
            tmp["available_fraction"] = (values == 1).astype(float)
            channel_mask_coverage_by_split_source[column] = (
                tmp.groupby([split_column, source_col], dropna=False)["available_fraction"]
                .agg(["count", "mean"])
                .reset_index()
                .rename(columns={"count": "rows"})
                .to_dict(orient="records")
            )

    numeric_shift_metrics = _numeric_shift_metrics(df)

    return {
        "duration": duration,
        "label_presence": label_presence,
        "mask_columns": mask_columns,
        "channel_coverage_from_config": channel_coverage,
        "split_source_label_counts": split_source_label_counts,
        "channel_mask_coverage_by_split_source": channel_mask_coverage_by_split_source,
        "numeric_shift_metrics": numeric_shift_metrics,
    }


def index_summary(
    index_paths: list[str | Path],
    *,
    config: str | Path | None = None,
    config_bytes: bytes | None = None,
    local_path_base: str | Path | None = None,
    label_name: str | None = None,
    split_values: list[str] | None = None,
    sample_path_check: int = 0,
    sample_npz_check: int = 0,
    validated_summary: tuple[ConfigSummaryInput, dict[str, set[str]]] | None = None,
) -> IndexSummary:
    cfg: ConfigSummaryInput | None
    resolved_paths = [resolve_repo_path(path, relative_to=local_path_base) for path in index_paths]
    paths = [path for path in resolved_paths if path is not None]
    if validated_summary is None:
        cfg = config_summary(config, config_bytes=config_bytes, local_path_base=local_path_base) if config else None
        survival_sidecar_keys = _survival_sidecar_keys(cfg, local_path_base=local_path_base)
        multilabel_sidecar_keys = _multilabel_sidecar_keys(cfg, local_path_base=local_path_base)
    else:
        cfg, validated_sidecar_keys = validated_summary
        survival_sidecar_keys = validated_sidecar_keys.get("survival")
        multilabel_sidecar_keys = validated_sidecar_keys.get("multilabel")
    survival_key_column = _survival_key_column(cfg)
    covariate_names = _task_covariates(cfg)
    multilabel_key_column = _multilabel_key_column(cfg)
    split_column = "split"
    read_csv_kwargs: dict[str, Any] = {"low_memory": False}
    converters = {key_column: str for key_column in (survival_key_column, multilabel_key_column) if key_column}
    if converters:
        read_csv_kwargs["converters"] = converters
    missing_inputs = [str(path) for path in paths if not path.exists()]
    frames = [pd.read_csv(path, **read_csv_kwargs) for path in paths if path.exists()]
    df = pd.concat(frames, axis=0, ignore_index=True) if frames else pd.DataFrame()
    df = _filter_splits(df, split_values, split_column=split_column)
    required_names = ("path", "split", "duration")
    required_columns = {name: name in df.columns for name in required_names}
    statistics = _index_statistics(df, cfg, label_name=label_name, split_column=split_column)

    path_check: SamplePathCheck = {"checked": 0, "existing": 0, "missing_examples": []}
    if sample_path_check and "path" in df.columns:
        example_paths = [Path(str(path)) for path in df["path"].dropna().head(sample_path_check)]
        path_check = {
            "checked": len(example_paths),
            "existing": sum(path.exists() for path in example_paths),
            "missing_examples": [str(path) for path in example_paths if not path.exists()][:5],
        }

    warnings: list[str] = []
    blocking_issues = [f"Index CSV not found: {path}" for path in missing_inputs]
    for column, exists in required_columns.items():
        if not exists:
            blocking_issues.append(f"Index CSV missing required column: {column}")
    survival_key = _key_summary(df, survival_key_column, sidecar_keys=survival_sidecar_keys)
    if survival_key is not None and not survival_key["exists"]:
        blocking_issues.append(f"Index CSV missing required survival key column: {survival_key_column}")
    if survival_key is not None and survival_key["missing_rows"]:
        blocking_issues.append(f"Index CSV contains empty survival key values in column: {survival_key_column}")
    if survival_key is not None and survival_key["missing_from_sidecars"]:
        examples = ", ".join(survival_key["missing_from_sidecars_examples"])
        blocking_issues.append(
            f"Index CSV contains survival key values missing from sidecars in column {survival_key_column}: "
            f"{survival_key['missing_from_sidecars']} missing (examples: {examples})"
        )
    multilabel_key = _key_summary(df, multilabel_key_column, sidecar_keys=multilabel_sidecar_keys)
    if multilabel_key is not None and not multilabel_key["exists"]:
        blocking_issues.append(f"Index CSV missing required multilabel key column: {multilabel_key_column}")
    if multilabel_key is not None and multilabel_key["missing_rows"]:
        blocking_issues.append(f"Index CSV contains empty multilabel key values in column: {multilabel_key_column}")
    if multilabel_key is not None and multilabel_key["missing_from_sidecars"]:
        examples = ", ".join(multilabel_key["missing_from_sidecars_examples"])
        blocking_issues.append(
            f"Index CSV contains multilabel key values missing from sidecars in column {multilabel_key_column}: "
            f"{multilabel_key['missing_from_sidecars']} missing (examples: {examples})"
        )
    covariates = _covariate_summary(df, covariate_names)
    for covariate, details in covariates.items():
        if not details["exists"]:
            blocking_issues.append(f"Index CSV missing required covariate column: {covariate}")
        elif details["missing_rows"]:
            blocking_issues.append(f"Index CSV contains empty covariate values in column: {covariate}")
    if sample_npz_check:
        warnings.append("--sample-npz-check is accepted but only path existence is checked by this lightweight tool.")

    return {
        "index_paths": [repo_relative(path) for path in paths],
        "rows": int(len(df)),
        "columns": list(df.columns),
        "required_columns": required_columns,
        "split_counts": df[split_column].value_counts(dropna=False).to_dict() if split_column in df.columns else {},
        "source_counts": (
            df["source"].value_counts(dropna=False).to_dict()
            if "source" in df.columns
            else df["dataset"].value_counts(dropna=False).to_dict() if "dataset" in df.columns else {}
        ),
        "duration": statistics["duration"],
        "label_presence": statistics["label_presence"],
        "mask_columns": statistics["mask_columns"],
        "channel_coverage_from_config": statistics["channel_coverage_from_config"],
        "survival_key": survival_key,
        "multilabel_key": multilabel_key,
        "covariates": covariates,
        "split_source_label_counts": statistics["split_source_label_counts"],
        "channel_mask_coverage_by_split_source": statistics["channel_mask_coverage_by_split_source"],
        "numeric_shift_metrics": statistics["numeric_shift_metrics"],
        "sample_path_check": path_check,
        "warnings": warnings,
        "blocking_issues": blocking_issues,
    }


def _survival_key_column(cfg: ConfigSummaryInput | None) -> str | None:
    if not cfg:
        return None
    finetune_value: Any = cfg.get("finetune") or {}
    task = finetune_value.get("task") or {}
    finetune_value = cfg.get("finetune") or {}
    survival = finetune_value.get("survival") or {}
    key_column = survival.get("key_column")
    if task.get("type") != "survival" or key_column in (None, ""):
        return None
    return str(key_column)


def _task_covariates(cfg: ConfigSummaryInput | None) -> list[str]:
    if not cfg:
        return []
    finetune_value: Any = cfg.get("finetune") or {}
    task_type = (finetune_value.get("task") or {}).get("type")
    if task_type == "survival":
        task_block = finetune_value.get("survival") or {}
    elif task_type == "multilabel_classification":
        task_block = finetune_value.get("multilabel") or {}
    else:
        return []
    covariates = task_block.get("covariates")
    if not isinstance(covariates, list):
        return []
    return [item for item in covariates if isinstance(item, str) and item]


def _multilabel_key_column(cfg: ConfigSummaryInput | None) -> str | None:
    if not cfg:
        return None
    finetune_value: Any = cfg.get("finetune") or {}
    task = finetune_value.get("task") or {}
    finetune_value = cfg.get("finetune") or {}
    multilabel = finetune_value.get("multilabel") or {}
    key_column = multilabel.get("key_column")
    if task.get("type") != "multilabel_classification" or key_column in (None, ""):
        return None
    return str(key_column)


def _survival_sidecar_keys(
    cfg: ConfigSummaryInput | None, *, local_path_base: str | Path | None = None
) -> set[str] | None:
    if not cfg:
        return None
    finetune_value: Any = cfg.get("finetune") or {}
    survival = finetune_value.get("survival") or {}
    if not survival.get("valid"):
        return None
    key_column = survival.get("key_column")
    event_time_path = resolve_repo_path(survival.get("event_time_index"), relative_to=local_path_base)
    if not key_column or event_time_path is None:
        return None

    from data.survival import normalize_survival_key

    frame = pd.read_csv(event_time_path, converters={str(key_column): str})
    return {normalize_survival_key(value, str(key_column)) for value in frame[str(key_column)]}


def _multilabel_sidecar_keys(
    cfg: ConfigSummaryInput | None, *, local_path_base: str | Path | None = None
) -> set[str] | None:
    if not cfg:
        return None
    finetune_value: Any = cfg.get("finetune") or {}
    multilabel = finetune_value.get("multilabel") or {}
    if not multilabel.get("valid"):
        return None
    key_column = multilabel.get("key_column")
    label_path = resolve_repo_path(multilabel.get("label_index"), relative_to=local_path_base)
    if not key_column or label_path is None:
        return None

    from data.multilabel import normalize_multilabel_key

    frame = pd.read_csv(label_path, converters={str(key_column): str})
    return {normalize_multilabel_key(value, str(key_column)) for value in frame[str(key_column)]}


def _filter_splits(df: pd.DataFrame, split_values: list[str] | None, *, split_column: str) -> pd.DataFrame:
    splits = [
        _normalized_split_value(value)
        for value in split_values or []
        if _normalized_split_value(value) not in ("", "ASK_USER")
    ]
    if not splits or split_column not in df.columns:
        return df
    normalized = df[split_column].map(_normalized_split_value)
    filtered = df[normalized.isin(splits)].copy()
    filtered[split_column] = filtered[split_column].map(_normalized_split_value)
    return filtered


def _normalized_split_value(value: Any) -> str:
    return "" if pd.isna(value) else str(value).strip()


def _covariate_summary(df: pd.DataFrame, covariates: list[str]) -> dict[str, CovariateSummary]:
    summary: dict[str, CovariateSummary] = {}
    for covariate in covariates:
        if covariate not in df.columns:
            summary[covariate] = {
                "exists": False,
                "non_null_rows": 0,
                "missing_rows": int(len(df)),
            }
            continue
        missing = df[covariate].isna() | df[covariate].astype(str).str.strip().eq("")
        summary[covariate] = {
            "exists": True,
            "non_null_rows": int((~missing).sum()),
            "missing_rows": int(missing.sum()),
        }
    return summary


def _key_summary(
    df: pd.DataFrame,
    key_column: str | None,
    *,
    sidecar_keys: set[str] | None = None,
) -> KeySummary | None:
    if not key_column:
        return None
    if key_column not in df.columns:
        return {
            "key_column": key_column,
            "exists": False,
            "non_null_rows": 0,
            "missing_rows": int(len(df)),
            "unique_keys": 0,
            "sidecar_key_count": len(sidecar_keys) if sidecar_keys is not None else None,
            "missing_from_sidecars": None,
            "missing_from_sidecars_examples": [],
        }

    keys = df[key_column]
    missing = keys.isna() | keys.astype(str).str.strip().eq("")
    valid_keys = keys[~missing].astype(str).str.strip()
    missing_from_sidecars = sorted(set(valid_keys) - sidecar_keys) if sidecar_keys is not None else []
    return {
        "key_column": key_column,
        "exists": True,
        "non_null_rows": int((~missing).sum()),
        "missing_rows": int(missing.sum()),
        "unique_keys": int(valid_keys.nunique()),
        "sidecar_key_count": len(sidecar_keys) if sidecar_keys is not None else None,
        "missing_from_sidecars": len(missing_from_sidecars) if sidecar_keys is not None else None,
        "missing_from_sidecars_examples": missing_from_sidecars[:5],
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


def _numeric_shift_metrics(df: pd.DataFrame) -> dict[str, NumericShiftMetrics]:
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
    out: dict[str, NumericShiftMetrics] = {}
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
