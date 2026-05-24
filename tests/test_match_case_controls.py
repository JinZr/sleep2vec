from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from utils import match_case_controls as matcher


def run_match(tmp_path: Path, rows: list[dict], extra_args: list[str] | None = None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "matched.csv"
    unmatched_path = tmp_path / "unmatched_cases.csv"
    excluded_path = tmp_path / "excluded_rows.csv"
    counts_path = tmp_path / "case_match_counts.csv"
    balance_path = tmp_path / "balance.csv"
    pd.DataFrame(rows).to_csv(input_path, index=False)

    argv = [
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--case-col",
        "MSA",
        "--case-value",
        "1",
        "--id-col",
        "ID",
        "--covariates",
        "Age",
        "Sex",
        "--exact-cols",
        "Sex",
        "--caliper-cols",
        "Age",
        "--caliper",
        "Age=5",
        "--ratio",
        "2",
        "--min-controls-per-case",
        "1",
        "--seed",
        "1",
        "--genetic-maxiter",
        "0",
        "--dedupe-cols",
        "Cohort",
        "ID",
        "--unmatched-cases-output",
        str(unmatched_path),
        "--excluded-output",
        str(excluded_path),
        "--case-match-counts-output",
        str(counts_path),
        "--balance-output",
        str(balance_path),
    ]
    if extra_args:
        argv.extend(extra_args)

    matcher.main(argv)
    return output_path, unmatched_path, excluded_path, counts_path, balance_path


def test_msa_like_run_writes_matched_unmatched_excluded_counts_and_balance(tmp_path):
    rows = [
        {"Cohort": "RJ-MSA", "ID": "C1", "MSA": 1, "Age": 60, "Sex": "F"},
        {"Cohort": "Control", "ID": "T1", "MSA": 0, "Age": 58, "Sex": "F"},
        {"Cohort": "Control", "ID": "T2", "MSA": 0, "Age": 62, "Sex": "F"},
        {"Cohort": "RJ-MSA", "ID": "C2", "MSA": 1, "Age": 70, "Sex": "M"},
        {"Cohort": "Control", "ID": "T3", "MSA": 0, "Age": 80, "Sex": "M"},
        {"Cohort": "Control", "ID": "T4", "MSA": 0, "Age": 82, "Sex": "M"},
        {"Cohort": "Control", "ID": "T2", "MSA": 0, "Age": 62, "Sex": "F"},
        {"Cohort": "Control", "ID": "MISSING", "MSA": 0, "Age": "", "Sex": "F"},
    ]

    output_path, unmatched_path, excluded_path, counts_path, balance_path = run_match(tmp_path, rows)

    matched = pd.read_csv(output_path)
    unmatched = pd.read_csv(unmatched_path)
    excluded = pd.read_csv(excluded_path)
    counts = pd.read_csv(counts_path)
    balance = pd.read_csv(balance_path)

    assert matched["ID"].tolist() == ["C1", "T1", "T2"]
    assert matched["matched_role"].tolist() == ["case", "control", "control"]
    assert matched["case_id"].tolist() == ["C1", "C1", "C1"]
    assert unmatched["ID"].tolist() == ["C2"]
    assert sorted(excluded["exclude_reason"].tolist()) == ["duplicate_by:Cohort,ID", "missing:Age"]
    assert counts.set_index("case_id").loc["C1", "status"] == "full"
    assert counts.set_index("case_id").loc["C2", "status"] == "unmatched"
    assert {"Age", "Sex"} <= set(balance["covariate"])
    assert balance["passes_max_smd"].all()


def test_missing_rows_are_filtered_before_duplicate_suppression(tmp_path):
    rows = [
        {"Cohort": "RJ-MSA", "ID": "C1", "MSA": 1, "Age": 60, "Sex": "F"},
        {"Cohort": "Control", "ID": "T1", "MSA": 0, "Age": "", "Sex": "F"},
        {"Cohort": "Control", "ID": "T1", "MSA": 0, "Age": 59, "Sex": "F"},
        {"Cohort": "Control", "ID": "T2", "MSA": 0, "Age": 62, "Sex": "F"},
    ]

    output_path, _, excluded_path, counts_path, _ = run_match(tmp_path, rows)

    assert pd.read_csv(output_path)["ID"].tolist() == ["C1", "T1", "T2"]
    assert pd.read_csv(excluded_path)["exclude_reason"].tolist() == ["missing:Age"]
    counts = pd.read_csv(counts_path).set_index("case_id")
    assert counts.loc["C1", "status"] == "full"


def test_identifier_columns_are_read_as_strings(tmp_path):
    rows = [
        {"Cohort": "RJ-MSA", "ID": "010", "MSA": "1", "Age": 60, "Sex": "F"},
        {"Cohort": "Control", "ID": "001", "MSA": "0", "Age": 59, "Sex": "F"},
        {"Cohort": "Control", "ID": "002", "MSA": "0", "Age": 62, "Sex": "F"},
    ]

    output_path, _, _, counts_path, _ = run_match(tmp_path, rows, ["--ratio", "1"])

    matched = pd.read_csv(output_path, dtype={"ID": "string", "case_id": "string"})
    counts = pd.read_csv(counts_path, dtype={"case_id": "string"})
    assert matched["ID"].tolist() == ["010", "001"]
    assert matched["case_id"].tolist() == ["010", "010"]
    assert counts["case_id"].tolist() == ["010"]


def test_case_mask_uses_exact_string_codes():
    mask = matcher.case_mask(pd.Series(["01", "1", "001", "0"]), "01")

    assert mask.tolist() == [True, False, False, False]


def test_exact_matching_prevents_cross_sex_controls(tmp_path):
    rows = [
        {"Cohort": "RJ-MSA", "ID": "C1", "MSA": 1, "Age": 60, "Sex": "F"},
        {"Cohort": "RJ-MSA", "ID": "C2", "MSA": 1, "Age": 70, "Sex": "M"},
        {"Cohort": "Control", "ID": "T1", "MSA": 0, "Age": 68, "Sex": "M"},
        {"Cohort": "Control", "ID": "T2", "MSA": 0, "Age": 72, "Sex": "M"},
    ]

    output_path, unmatched_path, _, _, _ = run_match(tmp_path, rows, ["--ratio", "1", "--caliper", "Age=100"])

    matched = pd.read_csv(output_path)
    assert pd.read_csv(unmatched_path)["ID"].tolist() == ["C1"]
    assert matched[matched["matched_role"].eq("control")]["Sex"].tolist() == ["M"]


def test_age_caliper_prevents_distant_same_sex_controls(tmp_path):
    rows = [
        {"Cohort": "RJ-MSA", "ID": "C1", "MSA": 1, "Age": 60, "Sex": "F"},
        {"Cohort": "Control", "ID": "T1", "MSA": 0, "Age": 70, "Sex": "F"},
        {"Cohort": "RJ-MSA", "ID": "C2", "MSA": 1, "Age": 70, "Sex": "M"},
        {"Cohort": "Control", "ID": "T2", "MSA": 0, "Age": 70, "Sex": "M"},
    ]

    output_path, unmatched_path, _, _, _ = run_match(tmp_path, rows, ["--ratio", "1"])

    assert pd.read_csv(unmatched_path)["ID"].tolist() == ["C1"]
    assert pd.read_csv(output_path)["ID"].tolist() == ["C2", "T2"]


def test_controls_are_not_reused_across_cases(tmp_path):
    rows = [
        {"Cohort": "RJ-MSA", "ID": "C1", "MSA": 1, "Age": 60, "Sex": "F"},
        {"Cohort": "RJ-MSA", "ID": "C2", "MSA": 1, "Age": 60, "Sex": "F"},
        {"Cohort": "Control", "ID": "T1", "MSA": 0, "Age": 60, "Sex": "F"},
    ]

    output_path, unmatched_path, _, counts_path, _ = run_match(tmp_path, rows, ["--ratio", "1"])

    matched = pd.read_csv(output_path)
    controls = matched[matched["matched_role"].eq("control")]
    counts = pd.read_csv(counts_path).set_index("case_id")

    assert controls["ID"].tolist() == ["T1"]
    assert pd.read_csv(unmatched_path)["ID"].tolist() == ["C2"]
    assert counts.loc["C1", "status"] == "full"
    assert counts.loc["C2", "status"] == "unmatched"


def test_ratio_is_maximum_and_allows_partial_matches(tmp_path):
    rows = [
        {"Cohort": "RJ-MSA", "ID": "C1", "MSA": 1, "Age": 60, "Sex": "F"},
        {"Cohort": "Control", "ID": "T1", "MSA": 0, "Age": 59, "Sex": "F"},
        {"Cohort": "Control", "ID": "T2", "MSA": 0, "Age": 61, "Sex": "F"},
    ]

    output_path, unmatched_path, _, counts_path, _ = run_match(tmp_path, rows, ["--ratio", "3"])

    assert pd.read_csv(output_path)["ID"].tolist() == ["C1", "T1", "T2"]
    assert pd.read_csv(unmatched_path).empty
    counts = pd.read_csv(counts_path).set_index("case_id")
    assert counts.loc["C1", "matched_control_count"] == 2
    assert counts.loc["C1", "status"] == "partial"


def test_require_full_ratio_fails_after_writing_outputs(tmp_path):
    rows = [
        {"Cohort": "RJ-MSA", "ID": "C1", "MSA": 1, "Age": 60, "Sex": "F"},
        {"Cohort": "Control", "ID": "T1", "MSA": 0, "Age": 59, "Sex": "F"},
        {"Cohort": "Control", "ID": "T2", "MSA": 0, "Age": 80, "Sex": "F"},
    ]

    with pytest.raises(SystemExit, match="fewer than the requested ratio"):
        run_match(tmp_path, rows, ["--ratio", "2", "--require-full-ratio"])

    assert pd.read_csv(tmp_path / "matched.csv")["ID"].tolist() == ["C1", "T1"]
    assert pd.read_csv(tmp_path / "case_match_counts.csv").loc[0, "status"] == "partial"


def test_balance_uses_pre_match_denominator_when_matched_variance_collapses(tmp_path):
    rows = [
        {"Cohort": "RJ-MSA", "ID": "C1", "MSA": 1, "Age": 60, "Sex": "F"},
        {"Cohort": "Control", "ID": "T1", "MSA": 0, "Age": 55, "Sex": "F"},
        {"Cohort": "Control", "ID": "T2", "MSA": 0, "Age": 65, "Sex": "F"},
    ]

    output_path, _, _, _, balance_path = run_match(tmp_path, rows, ["--ratio", "1", "--max-smd", "2"])

    assert len(pd.read_csv(output_path)) == 2
    age_balance = pd.read_csv(balance_path).set_index("covariate").loc["Age"]
    assert age_balance["abs_after_smd"] == 1.0
    assert age_balance["passes_max_smd"]


def test_max_smd_failure_happens_after_outputs_are_written(tmp_path):
    rows = [
        {"Cohort": "RJ-MSA", "ID": "C1", "MSA": 1, "Age": 60, "Sex": "F"},
        {"Cohort": "Control", "ID": "T1", "MSA": 0, "Age": 65, "Sex": "F"},
        {"Cohort": "RJ-MSA", "ID": "C2", "MSA": 1, "Age": 80, "Sex": "F"},
        {"Cohort": "Control", "ID": "T2", "MSA": 0, "Age": 85, "Sex": "F"},
    ]

    with pytest.raises(SystemExit, match="exceeded --max-smd"):
        run_match(tmp_path, rows, ["--ratio", "1", "--caliper", "Age=10", "--max-smd", "0.1"])

    assert len(pd.read_csv(tmp_path / "matched.csv")) == 4
    balance = pd.read_csv(tmp_path / "balance.csv")
    assert not balance["passes_max_smd"].all()


def test_matchit_defaults_do_not_apply_exact_or_age_caliper(tmp_path):
    args = matcher.parse_args(
        [
            "--input",
            str(tmp_path / "input.csv"),
            "--output",
            str(tmp_path / "matched.csv"),
            "--case-col",
            "MSA",
            "--case-value",
            "1",
            "--id-col",
            "ID",
            "--covariates",
            "Age",
            "Sex",
        ]
    )

    assert args.exact_cols == []
    assert args.ratio == 1
    assert matcher.parse_calipers(args.caliper, args.caliper_cols) == {}


def test_caliper_columns_not_in_formula_are_added_to_matching_features():
    df = pd.DataFrame({"Age": [60, 62], "VisitDay": [10, 30]})
    encoded_features = pd.DataFrame({"Q('Age')": [0.0, 1.0]}, index=df.index)
    propensity = pd.Series([0.8, 0.2], index=df.index)

    caliper_features = matcher.build_caliper_distance_features(df, {"VisitDay": 7}, ["Age"])
    matching_features = matcher.build_matching_features(encoded_features, propensity, caliper_features)

    assert matching_features.columns.tolist() == ["Q('Age')", "propensity_score", "caliper:VisitDay"]
    assert matching_features["caliper:VisitDay"].tolist() == [10.0, 30.0]


def test_negative_caliper_rejects_too_close_matches_like_matchit_restrict():
    case_row = pd.Series({"Age": 60})
    close_control = pd.Series({"Age": 62})
    distant_control = pd.Series({"Age": 70})

    assert not matcher.passes_calipers(case_row, close_control, {"Age": -5})
    assert matcher.passes_calipers(case_row, distant_control, {"Age": -5})


def test_matchit_default_order_matches_largest_propensity_first():
    df = pd.DataFrame(
        [
            {"Cohort": "RJ-MSA", "ID": "C70", "MSA": 1, "Age": 70, "Sex": "F"},
            {"Cohort": "RJ-MSA", "ID": "C80", "MSA": 1, "Age": 80, "Sex": "F"},
            {"Cohort": "Control", "ID": "T50", "MSA": 0, "Age": 50, "Sex": "F"},
            {"Cohort": "Control", "ID": "T60", "MSA": 0, "Age": 60, "Sex": "F"},
        ]
    )
    is_case = matcher.case_mask(df["MSA"], "1")
    propensity = pd.Series([0.7, 0.9, 0.1, 0.2], index=df.index)
    distance_features = pd.DataFrame({"Age": [70, 80, 50, 60]}, index=df.index, dtype=float)

    matched, _, _, _, _ = matcher.match_cases(
        df,
        is_case=is_case,
        id_col="ID",
        exact_cols=[],
        calipers={},
        ratio=1,
        min_controls_per_case=1,
        propensity=propensity,
        distance_features=distance_features,
        weight_matrix=np.eye(1),
    )

    assert matched["ID"].tolist() == ["C80", "T60", "C70", "T50"]


def test_patsy_treatment_coding_avoids_duplicate_categorical_levels():
    df = pd.DataFrame(
        [
            {"ID": "C1", "MSA": 1, "Age": 60, "Sex": "F"},
            {"ID": "T1", "MSA": 0, "Age": 62, "Sex": "M"},
            {"ID": "T2", "MSA": 0, "Age": 58, "Sex": "F"},
        ]
    )
    is_case = matcher.case_mask(df["MSA"], "1")

    _, glm_features, encoded_features, metadata = matcher.build_design_matrices(df, is_case, ["Age", "Sex"])

    assert "Intercept" in glm_features.columns
    assert encoded_features.columns.tolist() == ["Q('Sex')[T.M]", "Q('Age')"]
    assert metadata == [("Sex", "M", "Q('Sex')[T.M]"), ("Age", "", "Q('Age')")]


def test_genetic_maxiter_zero_uses_identity_weights():
    df = pd.DataFrame(
        [
            {"ID": "C1", "MSA": 1, "Age": 60, "Sex": "F"},
            {"ID": "T1", "MSA": 0, "Age": 62, "Sex": "F"},
        ]
    )
    is_case = matcher.case_mask(df["MSA"], "1")
    distance_features = pd.DataFrame({"Age": [0.0, 1.0], "propensity_score": [0.8, 0.2]}, index=df.index)

    matrix = matcher.optimize_weight_matrix(
        df,
        is_case=is_case,
        id_col="ID",
        exact_cols=[],
        calipers={},
        ratio=1,
        min_controls_per_case=1,
        propensity=pd.Series([0.8, 0.2], index=df.index),
        distance_features=distance_features,
        base_matrix=np.eye(2),
        encoded_balance=distance_features[["Age"]],
        metadata=[("Age", "", "Age")],
        seed=1,
        genetic_maxiter=0,
        genetic_popsize=15,
    )

    assert np.allclose(matrix, np.eye(2))


def test_genetic_weight_result_changes_selected_match(monkeypatch):
    df = pd.DataFrame(
        [
            {"ID": "C1", "MSA": 1},
            {"ID": "T_x_close", "MSA": 0},
            {"ID": "T_y_close", "MSA": 0},
        ]
    )
    is_case = matcher.case_mask(df["MSA"], "1")
    propensity = pd.Series([0.5, 0.5, 0.5], index=df.index)
    distance_features = pd.DataFrame({"x": [0.0, 0.0, 10.0], "y": [0.0, 10.0, 0.0]}, index=df.index)

    def fake_differential_evolution(*args, **kwargs):
        return SimpleNamespace(x=np.log([0.01, 100.0]))

    monkeypatch.setattr(matcher, "scipy_differential_evolution", fake_differential_evolution)

    matrix = matcher.optimize_weight_matrix(
        df,
        is_case=is_case,
        id_col="ID",
        exact_cols=[],
        calipers={},
        ratio=1,
        min_controls_per_case=1,
        propensity=propensity,
        distance_features=distance_features,
        base_matrix=np.eye(2),
        encoded_balance=distance_features,
        metadata=[("x", "", "x"), ("y", "", "y")],
        seed=1,
        genetic_maxiter=1,
        genetic_popsize=2,
    )
    matched, _, _, _, _ = matcher.match_cases(
        df,
        is_case=is_case,
        id_col="ID",
        exact_cols=[],
        calipers={},
        ratio=1,
        min_controls_per_case=1,
        propensity=propensity,
        distance_features=distance_features,
        weight_matrix=matrix,
    )

    assert matched["ID"].tolist() == ["C1", "T_y_close"]


def test_genetic_objective_penalizes_unmatched_and_ratio_shortfall():
    encoded = pd.DataFrame({"Age": [60.0, 62.0, 70.0]})
    is_case = pd.Series([True, False, False], index=encoded.index)
    unmatched = pd.DataFrame([{"ID": "C1"}])
    counts = pd.DataFrame([{"matched_control_count": 1}])

    score = matcher.matching_objective(
        encoded=encoded,
        metadata=[("Age", "", "Age")],
        before_is_case=is_case,
        matched_indices=[0, 1],
        matched_roles=["case", "control"],
        unmatched_cases=unmatched,
        match_counts=counts,
        ratio=2,
    )

    assert score > 1100


def test_seed_is_passed_to_genetic_optimizer(monkeypatch):
    df = pd.DataFrame(
        [
            {"ID": "C1", "MSA": 1},
            {"ID": "T1", "MSA": 0},
        ]
    )
    is_case = matcher.case_mask(df["MSA"], "1")
    distance_features = pd.DataFrame({"Age": [0.0, 1.0]}, index=df.index)
    seen = []

    def fake_differential_evolution(*args, **kwargs):
        seen.append(kwargs["seed"])
        return SimpleNamespace(x=np.zeros(1))

    monkeypatch.setattr(matcher, "scipy_differential_evolution", fake_differential_evolution)

    matcher.optimize_weight_matrix(
        df,
        is_case=is_case,
        id_col="ID",
        exact_cols=[],
        calipers={},
        ratio=1,
        min_controls_per_case=1,
        propensity=pd.Series([0.8, 0.2], index=df.index),
        distance_features=distance_features,
        base_matrix=np.eye(1),
        encoded_balance=distance_features,
        metadata=[("Age", "", "Age")],
        seed=123,
        genetic_maxiter=1,
        genetic_popsize=2,
    )

    assert seen == [123]


def test_genetic_search_with_same_seed_is_stable(monkeypatch, tmp_path):
    rows = [
        {"Cohort": "RJ-MSA", "ID": "C60", "MSA": 1, "Age": 60, "Sex": "F"},
        {"Cohort": "Control", "ID": "T50", "MSA": 0, "Age": 50, "Sex": "F"},
        {"Cohort": "Control", "ID": "T70", "MSA": 0, "Age": 70, "Sex": "F"},
    ]

    def fake_differential_evolution(*args, **kwargs):
        return SimpleNamespace(x=np.zeros(len(kwargs["bounds"])))

    monkeypatch.setattr(matcher, "scipy_differential_evolution", fake_differential_evolution)

    first_output, _, _, _, _ = run_match(tmp_path / "first", rows, ["--ratio", "1", "--genetic-maxiter", "1"])
    second_output, _, _, _, _ = run_match(tmp_path / "second", rows, ["--ratio", "1", "--genetic-maxiter", "1"])

    assert pd.read_csv(first_output).equals(pd.read_csv(second_output))
