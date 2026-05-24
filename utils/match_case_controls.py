#!/usr/bin/env python3
"""MatchIt-style case-control matching from a CSV file."""

from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Match case rows to control rows with optional exact-match strata, hard calipers, "
            "and standardized-mean-difference balance diagnostics."
        )
    )
    parser.add_argument("--input", required=True, type=Path, help="Input CSV path")
    parser.add_argument("--output", required=True, type=Path, help="Matched output CSV path")
    parser.add_argument("--case-col", required=True, help="Column that marks cases")
    parser.add_argument("--case-value", required=True, help="Value in --case-col that identifies cases")
    parser.add_argument("--id-col", required=True, help="Stable subject/sample ID column")
    parser.add_argument("--covariates", nargs="+", required=True, help="Covariates used for matching diagnostics")
    parser.add_argument(
        "--exact-cols",
        nargs="*",
        default=[],
        help="Columns that must match exactly. Default: none.",
    )
    parser.add_argument(
        "--caliper-cols",
        nargs="*",
        default=None,
        help="Columns with hard absolute-distance calipers. Defaults to columns named by --caliper.",
    )
    parser.add_argument(
        "--caliper",
        nargs="*",
        default=[],
        help="Caliper limits as COL=VALUE entries. Default: none.",
    )
    parser.add_argument("--ratio", type=int, default=1, help="Maximum controls per case")
    parser.add_argument("--min-controls-per-case", type=int, default=1, help="Minimum acceptable controls per case")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for genetic-style weight search")
    parser.add_argument(
        "--genetic-maxiter",
        type=int,
        default=100,
        help="Differential-evolution iterations for genetic-style weight search. Use 0 to disable.",
    )
    parser.add_argument(
        "--genetic-popsize",
        type=int,
        default=15,
        help="Population size multiplier for genetic-style weight search.",
    )
    parser.add_argument(
        "--require-full-ratio",
        action="store_true",
        help="Fail if any case receives fewer controls than --ratio",
    )
    parser.add_argument("--dedupe-cols", nargs="*", default=[], help="Optional columns used to drop duplicate rows")
    parser.add_argument("--unmatched-cases-output", type=Path, default=None, help="Unmatched cases CSV path")
    parser.add_argument("--excluded-output", type=Path, default=None, help="Excluded rows CSV path")
    parser.add_argument("--case-match-counts-output", type=Path, default=None, help="Case-level match count CSV path")
    parser.add_argument("--balance-output", type=Path, default=None, help="Balance diagnostics CSV path")
    parser.add_argument(
        "--max-smd",
        type=float,
        default=None,
        help="Maximum allowed absolute post-match SMD. Default: report only.",
    )
    parser.add_argument(
        "--fail-on-unmatched-cases",
        action="store_true",
        help="Exit nonzero when any case has fewer than --min-controls-per-case controls",
    )
    args = parser.parse_args(argv)

    if args.ratio < 1:
        parser.error("--ratio must be >= 1")
    if args.min_controls_per_case < 1:
        parser.error("--min-controls-per-case must be >= 1")
    if args.min_controls_per_case > args.ratio:
        parser.error("--min-controls-per-case cannot exceed --ratio")
    if args.genetic_maxiter < 0:
        parser.error("--genetic-maxiter must be >= 0")
    if args.genetic_popsize < 1:
        parser.error("--genetic-popsize must be >= 1")
    if args.max_smd is not None and args.max_smd < 0:
        parser.error("--max-smd must be >= 0")

    return args


def parse_calipers(caliper_items: list[str], caliper_cols: list[str] | None) -> dict[str, float]:
    parsed = {}
    for item in caliper_items:
        if "=" not in item:
            raise SystemExit(f"Invalid --caliper entry {item!r}; expected COL=VALUE")
        col, value = item.split("=", 1)
        col = col.strip()
        if not col:
            raise SystemExit(f"Invalid --caliper entry {item!r}; column name is empty")
        try:
            parsed[col] = float(value)
        except ValueError as exc:
            raise SystemExit(f"Invalid --caliper value for {col}: {value!r}") from exc

    if caliper_cols is None:
        return parsed
    if not caliper_cols:
        return {}

    missing = [col for col in caliper_cols if col not in parsed]
    if missing:
        raise SystemExit(f"Missing --caliper values for: {', '.join(missing)}")
    return {col: parsed[col] for col in caliper_cols}


def default_output_path(output: Path, suffix: str) -> Path:
    return output.with_name(f"{output.stem}_{suffix}{output.suffix}")


def resolve_output_paths(args: argparse.Namespace) -> None:
    args.unmatched_cases_output = args.unmatched_cases_output or default_output_path(args.output, "unmatched_cases")
    args.excluded_output = args.excluded_output or default_output_path(args.output, "excluded_rows")
    args.case_match_counts_output = args.case_match_counts_output or default_output_path(
        args.output, "case_match_counts"
    )
    args.balance_output = args.balance_output or default_output_path(args.output, "balance")


def require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise SystemExit(f"Missing required columns: {', '.join(missing)}")


def is_missing_value(series: pd.Series) -> pd.Series:
    as_string = series.astype("string")
    return series.isna() | as_string.str.strip().eq("")


def case_mask(series: pd.Series, case_value: str) -> pd.Series:
    return series.astype("string").str.strip().eq(str(case_value).strip()).fillna(False)


def prepare_rows(
    df: pd.DataFrame,
    *,
    dedupe_cols: list[str],
    required_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    excluded_parts = []
    kept = df.copy()

    missing_mask = pd.Series(False, index=kept.index)
    missing_reasons = pd.Series("", index=kept.index, dtype="object")
    for col in required_cols:
        col_missing = is_missing_value(kept[col])
        missing_mask = missing_mask | col_missing
        missing_reasons.loc[col_missing] = missing_reasons.loc[col_missing].map(
            lambda current, column=col: f"{current};missing:{column}" if current else f"missing:{column}"
        )

    if missing_mask.any():
        missing = kept.loc[missing_mask].copy()
        missing["exclude_reason"] = missing_reasons.loc[missing_mask].values
        excluded_parts.append(missing)
        kept = kept.loc[~missing_mask].copy()

    if dedupe_cols:
        duplicate_mask = kept.duplicated(subset=dedupe_cols, keep="first")
        if duplicate_mask.any():
            duplicates = kept.loc[duplicate_mask].copy()
            duplicates["exclude_reason"] = f"duplicate_by:{','.join(dedupe_cols)}"
            excluded_parts.append(duplicates)
            kept = kept.loc[~duplicate_mask].copy()

    if excluded_parts:
        excluded = pd.concat(excluded_parts, ignore_index=True)
    else:
        excluded = pd.DataFrame(columns=list(df.columns) + ["exclude_reason"])

    return kept.reset_index(drop=True), excluded


def quote_patsy_column(column: str) -> str:
    return f"Q({column!r})"


def build_patsy_formula(covariates: list[str]) -> str:
    return "_treat ~ " + " + ".join(quote_patsy_column(col) for col in covariates)


def design_metadata(columns: list[str], covariates: list[str]) -> list[tuple[str, str, str]]:
    quoted = {quote_patsy_column(col): col for col in covariates}
    metadata = []
    for encoded_col in columns:
        covariate = encoded_col
        level = ""
        for quoted_col, original_col in quoted.items():
            if encoded_col == quoted_col:
                covariate = original_col
                break
            prefix = f"{quoted_col}[T."
            if encoded_col.startswith(prefix) and encoded_col.endswith("]"):
                covariate = original_col
                level = encoded_col[len(prefix) : -1]
                break
        metadata.append((covariate, level, encoded_col))
    return metadata


def build_design_matrices(
    df: pd.DataFrame, is_case: pd.Series, covariates: list[str]
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame, list[tuple[str, str, str]]]:
    design_df = df.copy()
    design_df["_treat"] = is_case.astype(int).to_numpy()
    try:
        y, x = patsy.dmatrices(
            build_patsy_formula(covariates),
            data=design_df,
            return_type="dataframe",
            NA_action="raise",
        )
    except Exception as exc:
        raise SystemExit(f"Failed to build Patsy design matrix: {exc}") from exc

    x = x.astype(float)
    feature_columns = [col for col in x.columns if col != "Intercept"]
    features = x.loc[:, feature_columns].copy()
    metadata = design_metadata(feature_columns, covariates)
    return y.iloc[:, 0].astype(float), x, features, metadata


def scale_features(features: pd.DataFrame) -> np.ndarray:
    matrix = features.to_numpy(dtype=float)
    if matrix.size == 0:
        return matrix
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0, ddof=1) if len(matrix) > 1 else np.ones(matrix.shape[1])
    std[(std == 0) | np.isnan(std)] = 1.0
    return (matrix - mean) / std


def estimate_propensity_scores(y: pd.Series, x: pd.DataFrame) -> pd.Series:
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = sm.GLM(y.to_numpy(dtype=float), x.to_numpy(dtype=float), family=sm.families.Binomial()).fit()
    except Exception as exc:
        raise SystemExit(f"Failed to estimate GLM propensity scores: {exc}") from exc
    if any("PerfectSeparation" in warning.category.__name__ for warning in caught):
        raise SystemExit("Failed to estimate GLM propensity scores: perfect separation detected.")
    if not getattr(result, "converged", True):
        raise SystemExit("Failed to estimate GLM propensity scores: model did not converge.")

    scores = pd.Series(np.asarray(result.fittedvalues, dtype=float), index=x.index)
    if not np.isfinite(scores).all():
        raise SystemExit("Failed to estimate GLM propensity scores: non-finite fitted values.")
    return scores


def build_caliper_distance_features(
    df: pd.DataFrame, calipers: dict[str, float], covariates: list[str]
) -> pd.DataFrame:
    extras = pd.DataFrame(index=df.index)
    for col in calipers:
        if col in covariates:
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        if values.isna().any():
            raise SystemExit(f"Caliper column has non-numeric values after filtering: {col}")
        extras[f"caliper:{col}"] = values.astype(float)
    return extras


def build_matching_features(
    features: pd.DataFrame, propensity: pd.Series, caliper_features: pd.DataFrame | None = None
) -> pd.DataFrame:
    features = features.copy()
    if propensity.notna().any():
        features["propensity_score"] = propensity.loc[features.index].to_numpy(dtype=float)
    if caliper_features is not None and not caliper_features.empty:
        features = pd.concat([features, caliper_features.loc[features.index]], axis=1)
    return features


def matching_feature_space(features: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    scaled = pd.DataFrame(scale_features(features), index=features.index, columns=features.columns)
    if scaled.shape[1] == 0:
        return scaled, np.empty((0, 0))
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.corrcoef(scaled.to_numpy(dtype=float), rowvar=False)
    corr = np.atleast_2d(corr).astype(float)
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 1.0)
    return scaled, np.linalg.pinv(corr)


def weighted_matrix(base_matrix: np.ndarray, weights: np.ndarray) -> np.ndarray:
    if base_matrix.size == 0:
        return base_matrix
    scale = np.diag(np.sqrt(weights))
    return scale @ base_matrix @ scale


def matching_distance(case_features: pd.Series, control_features: pd.Series, weight_matrix: np.ndarray) -> float:
    if weight_matrix.size == 0:
        return 0.0
    diff = case_features.to_numpy(dtype=float) - control_features.to_numpy(dtype=float)
    distance = float(diff @ weight_matrix @ diff.T)
    return float(np.sqrt(max(distance, 0.0)))


def order_cases_like_matchit(case_indices: list[int], propensity: pd.Series) -> list[int]:
    if not propensity.loc[case_indices].notna().any():
        return case_indices
    return sorted(
        case_indices,
        key=lambda idx: (
            -float(propensity.loc[idx]) if pd.notna(propensity.loc[idx]) else np.inf,
            idx,
        ),
    )


def passes_calipers(case_row: pd.Series, control_row: pd.Series, calipers: dict[str, float]) -> bool:
    for col, limit in calipers.items():
        case_value = pd.to_numeric(pd.Series([case_row[col]]), errors="coerce").iloc[0]
        control_value = pd.to_numeric(pd.Series([control_row[col]]), errors="coerce").iloc[0]
        if pd.isna(case_value) or pd.isna(control_value):
            return False
        distance = abs(float(case_value) - float(control_value))
        if limit >= 0 and distance > limit:
            return False
        if limit < 0 and distance <= abs(limit):
            return False
    return True


def passes_exact(case_row: pd.Series, control_row: pd.Series, exact_cols: list[str]) -> bool:
    for col in exact_cols:
        if case_row[col] != control_row[col]:
            return False
    return True


def match_cases(
    df: pd.DataFrame,
    *,
    is_case: pd.Series,
    id_col: str,
    exact_cols: list[str],
    calipers: dict[str, float],
    ratio: int,
    min_controls_per_case: int,
    propensity: pd.Series,
    distance_features: pd.DataFrame,
    weight_matrix: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[int], list[str]]:
    used_controls = set()
    matched_parts = []
    matched_indices = []
    matched_roles = []
    count_rows = []
    unmatched_parts = []
    match_group = 0
    case_indices = order_cases_like_matchit([idx for idx in df.index if bool(is_case.loc[idx])], propensity)
    control_indices = [idx for idx in df.index if not bool(is_case.loc[idx])]

    for case_idx in case_indices:
        case_row = df.loc[case_idx]
        candidates = []
        for control_idx in control_indices:
            if control_idx in used_controls:
                continue
            control_row = df.loc[control_idx]
            if not passes_exact(case_row, control_row, exact_cols):
                continue
            if not passes_calipers(case_row, control_row, calipers):
                continue
            distance = matching_distance(
                distance_features.loc[case_idx],
                distance_features.loc[control_idx],
                weight_matrix,
            )
            candidates.append((distance, control_idx))

        candidates.sort(key=lambda item: (item[0], item[1]))
        chosen = candidates[:ratio]
        chosen_indices = [idx for _, idx in chosen]
        matched_count = len(chosen_indices)

        if matched_count >= ratio:
            status = "full"
        elif matched_count >= min_controls_per_case:
            status = "partial"
        else:
            status = "unmatched"

        count_row = {
            "case_id": case_row[id_col],
            "requested_ratio": ratio,
            "matched_control_count": matched_count,
            "min_controls_per_case": min_controls_per_case,
            "status": status,
        }
        for col in exact_cols:
            count_row[col] = case_row[col]
        count_rows.append(count_row)

        if status == "unmatched":
            unmatched = case_row.to_frame().T.copy()
            unmatched["matched_control_count"] = matched_count
            unmatched["min_controls_per_case"] = min_controls_per_case
            unmatched["requested_ratio"] = ratio
            unmatched["unmatched_reason"] = f"fewer_than_{min_controls_per_case}_controls"
            unmatched_parts.append(unmatched)
            continue

        match_group += 1
        group_indices = [case_idx] + chosen_indices
        group = df.loc[group_indices].copy()
        group["match_group"] = match_group
        group["matched_role"] = ["case"] + ["control"] * len(chosen_indices)
        group["case_id"] = case_row[id_col]
        group["propensity_score"] = propensity.loc[group_indices].values
        group["match_distance"] = [0.0] + [distance for distance, _ in chosen]
        matched_parts.append(group)
        matched_indices.extend(group_indices)
        matched_roles.extend(group["matched_role"].tolist())
        used_controls.update(chosen_indices)

    if matched_parts:
        matched = pd.concat(matched_parts, ignore_index=True)
    else:
        matched = pd.DataFrame(
            columns=list(df.columns) + ["match_group", "matched_role", "case_id", "propensity_score", "match_distance"]
        )

    if unmatched_parts:
        unmatched_cases = pd.concat(unmatched_parts, ignore_index=True)
    else:
        unmatched_cases = pd.DataFrame(
            columns=list(df.columns)
            + ["matched_control_count", "min_controls_per_case", "requested_ratio", "unmatched_reason"]
        )

    match_counts = pd.DataFrame(count_rows)
    return matched, unmatched_cases, match_counts, matched_indices, matched_roles


def smd_denominator(case_values: np.ndarray, control_values: np.ndarray) -> float:
    if len(case_values) == 0 or len(control_values) == 0:
        return np.nan
    case_var = float(np.var(case_values, ddof=1)) if len(case_values) > 1 else 0.0
    control_var = float(np.var(control_values, ddof=1)) if len(control_values) > 1 else 0.0
    return float(np.sqrt((case_var + control_var) / 2.0))


def smd(case_values: np.ndarray, control_values: np.ndarray, denominator: float) -> float:
    if len(case_values) == 0 or len(control_values) == 0:
        return np.nan
    diff = float(np.mean(case_values) - np.mean(control_values))
    if denominator == 0:
        return 0.0 if diff == 0 else np.inf
    return diff / denominator


def compute_balance(
    encoded: pd.DataFrame,
    matched_indices: list[int],
    matched_roles: list[str],
    *,
    before_is_case: pd.Series,
    metadata: list[tuple[str, str, str]],
    max_smd: float | None,
) -> pd.DataFrame:
    columns = ["covariate", "level", "before_smd", "after_smd", "abs_after_smd", "passes_max_smd"]
    before_case_mask = before_is_case.loc[encoded.index]
    after_encoded = encoded.loc[matched_indices] if matched_indices else encoded.iloc[0:0]
    after_case_mask = pd.Series(matched_roles, index=after_encoded.index).eq("case")

    rows = []
    for covariate, level, encoded_col in metadata:
        before_case_values = encoded.loc[before_case_mask, encoded_col].to_numpy(dtype=float)
        before_control_values = encoded.loc[~before_case_mask, encoded_col].to_numpy(dtype=float)
        denominator = smd_denominator(before_case_values, before_control_values)
        before_value = smd(
            before_case_values,
            before_control_values,
            denominator,
        )
        after_value = smd(
            after_encoded.loc[after_case_mask, encoded_col].to_numpy(dtype=float),
            after_encoded.loc[~after_case_mask, encoded_col].to_numpy(dtype=float),
            denominator,
        )
        abs_after = abs(after_value) if not pd.isna(after_value) else np.nan
        rows.append(
            {
                "covariate": covariate,
                "level": level,
                "before_smd": before_value,
                "after_smd": after_value,
                "abs_after_smd": abs_after,
                "passes_max_smd": True if max_smd is None else bool(np.isfinite(abs_after) and abs_after <= max_smd),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def is_binary_values(values: np.ndarray) -> bool:
    finite = values[np.isfinite(values)]
    return len(np.unique(finite)) <= 2


def paired_balance_values(
    encoded: pd.DataFrame, matched_roles: list[str], encoded_col: str
) -> tuple[np.ndarray, np.ndarray]:
    case_values = []
    control_values = []
    current_case_value = None
    for idx, role in zip(encoded.index, matched_roles):
        value = float(encoded.loc[idx, encoded_col])
        if role == "case":
            current_case_value = value
        elif current_case_value is not None:
            case_values.append(current_case_value)
            control_values.append(value)
    return np.asarray(case_values, dtype=float), np.asarray(control_values, dtype=float)


def ttest_pvalue(case_values: np.ndarray, control_values: np.ndarray) -> float:
    from scipy import stats

    if len(case_values) == 0 or len(control_values) == 0:
        return 0.0
    case_mean = float(np.mean(case_values))
    control_mean = float(np.mean(control_values))
    if np.var(case_values) == 0 and np.var(control_values) == 0:
        return 1.0 if case_mean == control_mean else 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = stats.ttest_rel(case_values, control_values, nan_policy="omit")
    if np.isfinite(result.pvalue):
        return float(result.pvalue)
    return 1.0 if case_mean == control_mean else 0.0


def genetic_balance_loss(
    encoded: pd.DataFrame,
    matched_indices: list[int],
    matched_roles: list[str],
    metadata: list[tuple[str, str, str]],
) -> float:
    from scipy import stats

    if not matched_indices:
        return 1.0
    after_encoded = encoded.loc[matched_indices]
    pvalues = []
    for _, _, encoded_col in metadata:
        case_values, control_values = paired_balance_values(after_encoded, matched_roles, encoded_col)
        pvalues.append(ttest_pvalue(case_values, control_values))
        combined = np.concatenate([case_values, control_values])
        if len(case_values) > 0 and len(control_values) > 0 and not is_binary_values(combined):
            ks_pvalue = stats.ks_2samp(case_values, control_values).pvalue
            pvalues.append(float(ks_pvalue) if np.isfinite(ks_pvalue) else 0.0)

    if not pvalues:
        return 1.0
    return 1.0 - min(pvalues)


def matching_objective(
    *,
    encoded: pd.DataFrame,
    metadata: list[tuple[str, str, str]],
    before_is_case: pd.Series,
    matched_indices: list[int],
    matched_roles: list[str],
    unmatched_cases: pd.DataFrame,
    match_counts: pd.DataFrame,
    ratio: int,
) -> float:
    if not matched_indices:
        return 1_000_000.0
    balance_loss = genetic_balance_loss(
        encoded,
        matched_indices,
        matched_roles,
        metadata=metadata,
    )

    unmatched_penalty = 1000.0 * len(unmatched_cases)
    shortfall = 0
    if not match_counts.empty:
        shortfall = int((ratio - match_counts["matched_control_count"]).clip(lower=0).sum())
    return balance_loss + unmatched_penalty + 100.0 * shortfall


def scipy_differential_evolution(*args, **kwargs):
    from scipy.optimize import differential_evolution

    return differential_evolution(*args, **kwargs)


def optimize_weight_matrix(
    df: pd.DataFrame,
    *,
    is_case: pd.Series,
    id_col: str,
    exact_cols: list[str],
    calipers: dict[str, float],
    ratio: int,
    min_controls_per_case: int,
    propensity: pd.Series,
    distance_features: pd.DataFrame,
    base_matrix: np.ndarray,
    encoded_balance: pd.DataFrame,
    metadata: list[tuple[str, str, str]],
    seed: int,
    genetic_maxiter: int,
    genetic_popsize: int,
) -> np.ndarray:
    if distance_features.shape[1] == 0:
        return base_matrix
    if genetic_maxiter == 0:
        return weighted_matrix(base_matrix, np.ones(distance_features.shape[1]))

    def objective(log_weights: np.ndarray) -> float:
        weights = np.exp(log_weights)
        candidate_matrix = weighted_matrix(base_matrix, weights)
        _, unmatched_cases, match_counts, matched_indices, matched_roles = match_cases(
            df,
            is_case=is_case,
            id_col=id_col,
            exact_cols=exact_cols,
            calipers=calipers,
            ratio=ratio,
            min_controls_per_case=min_controls_per_case,
            propensity=propensity,
            distance_features=distance_features,
            weight_matrix=candidate_matrix,
        )
        return matching_objective(
            encoded=encoded_balance,
            metadata=metadata,
            before_is_case=is_case,
            matched_indices=matched_indices,
            matched_roles=matched_roles,
            unmatched_cases=unmatched_cases,
            match_counts=match_counts,
            ratio=ratio,
        )

    result = scipy_differential_evolution(
        objective,
        bounds=[(np.log(1e-6), np.log(1000.0))] * distance_features.shape[1],
        maxiter=genetic_maxiter,
        popsize=genetic_popsize,
        seed=seed,
        x0=np.zeros(distance_features.shape[1]),
        polish=False,
    )
    return weighted_matrix(base_matrix, np.exp(result.x))


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main(argv=None) -> None:
    args = parse_args(argv)
    resolve_output_paths(args)
    calipers = parse_calipers(args.caliper, args.caliper_cols)

    string_cols = list(dict.fromkeys([args.case_col, args.id_col] + args.exact_cols + args.dedupe_cols))
    header = pd.read_csv(args.input, nrows=0)
    df = pd.read_csv(args.input, converters={col: str for col in string_cols if col in header.columns})
    required_cols = list(
        dict.fromkeys([args.case_col, args.id_col] + args.covariates + args.exact_cols + list(calipers))
    )
    require_columns(df, required_cols + args.dedupe_cols)

    clean_df, excluded = prepare_rows(df, dedupe_cols=args.dedupe_cols, required_cols=required_cols)
    if clean_df.empty:
        write_csv(excluded, args.excluded_output)
        raise SystemExit("No rows remain after de-duplication and missing-value filtering.")
    for col in calipers:
        if pd.to_numeric(clean_df[col], errors="coerce").isna().any():
            raise SystemExit(f"Caliper column has non-numeric values after filtering: {col}")

    is_case = case_mask(clean_df[args.case_col], args.case_value)
    if not is_case.any():
        write_csv(excluded, args.excluded_output)
        raise SystemExit("No case rows found after filtering.")
    if is_case.all():
        write_csv(excluded, args.excluded_output)
        raise SystemExit("No control rows found after filtering.")
    y, glm_features, encoded_features, metadata = build_design_matrices(clean_df, is_case, args.covariates)
    propensity = estimate_propensity_scores(y, glm_features)
    caliper_features = build_caliper_distance_features(clean_df, calipers, args.covariates)
    distance_features, base_matrix = matching_feature_space(
        build_matching_features(encoded_features, propensity, caliper_features)
    )
    weight_matrix = optimize_weight_matrix(
        clean_df,
        is_case=is_case,
        id_col=args.id_col,
        exact_cols=args.exact_cols,
        calipers=calipers,
        ratio=args.ratio,
        min_controls_per_case=args.min_controls_per_case,
        propensity=propensity,
        distance_features=distance_features,
        base_matrix=base_matrix,
        encoded_balance=encoded_features,
        metadata=metadata,
        seed=args.seed,
        genetic_maxiter=args.genetic_maxiter,
        genetic_popsize=args.genetic_popsize,
    )
    matched, unmatched_cases, match_counts, matched_indices, matched_roles = match_cases(
        clean_df,
        is_case=is_case,
        id_col=args.id_col,
        exact_cols=args.exact_cols,
        calipers=calipers,
        ratio=args.ratio,
        min_controls_per_case=args.min_controls_per_case,
        propensity=propensity,
        distance_features=distance_features,
        weight_matrix=weight_matrix,
    )
    balance = compute_balance(
        encoded_features,
        matched_indices,
        matched_roles,
        before_is_case=is_case,
        metadata=metadata,
        max_smd=args.max_smd,
    )

    write_csv(matched, args.output)
    write_csv(excluded, args.excluded_output)
    write_csv(unmatched_cases, args.unmatched_cases_output)
    write_csv(match_counts, args.case_match_counts_output)
    write_csv(balance, args.balance_output)

    failures = []
    if matched.empty:
        failures.append("No matched groups were produced.")
    if args.fail_on_unmatched_cases and not unmatched_cases.empty:
        failures.append(f"{len(unmatched_cases)} case rows received fewer than {args.min_controls_per_case} controls.")
    if args.require_full_ratio and not match_counts.empty:
        shortfall = match_counts[match_counts["matched_control_count"] < args.ratio]
        if not shortfall.empty:
            failures.append(f"{len(shortfall)} case rows received fewer than the requested ratio {args.ratio}.")
    if args.max_smd is not None:
        failed_balance = balance[~balance["passes_max_smd"]]
        if not failed_balance.empty:
            failures.append(f"{len(failed_balance)} balance rows exceeded --max-smd {args.max_smd}.")

    if failures:
        raise SystemExit("Matching completed with failed criteria: " + " ".join(failures))

    print(f"Wrote {len(matched)} matched rows to {args.output}")
    print(f"Wrote balance diagnostics to {args.balance_output}")


if __name__ == "__main__":
    main()
