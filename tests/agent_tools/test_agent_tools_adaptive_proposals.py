from __future__ import annotations

import copy
import json
import math

import pytest

from agent_tools.adaptive_proposals import (
    build_proposal_input,
    canonical_sha256,
    load_strict_json,
    proposal_request_id,
    validate_parameter_envelopes,
    validate_proposal,
    validate_proposal_input,
)


def _input_payload(
    *,
    parameters: dict | None = None,
    bounds: dict | None = None,
    runs: int = 8,
    round_size: int = 6,
) -> dict:
    parameters = parameters or {
        "runtime.lr": [0.0001, 0.0003],
        "runtime.batch_size": [8, 16],
        "yaml:/model/head/name": ["linear", "mlp"],
    }
    return {
        "source_round": 0,
        "target_round": 1,
        "objective": {"metric": "val_auroc", "mode": "max"},
        "remaining_budget": {"rounds": 3, "runs": runs, "round_size": round_size},
        "digest_rows": [
            {
                "run_id": "run-001",
                "status": "completed",
                "val_auroc": 0.72,
                "artifact_path": "/tmp/run-001",
            },
            {"run_id": "run-002", "status": "failed", "error": "out of memory"},
        ],
        "parameter_envelopes": validate_parameter_envelopes(parameters, bounds),
        "resolved_recipe_sha256": "a" * 64,
        "source_config_sha256": "c" * 64,
        "execution_identity": {"target": "local", "runtime_commit": "b" * 40},
    }


def _snapshot(**kwargs) -> dict:
    input_payload = _input_payload(**kwargs)
    request_id = proposal_request_id(input_payload)
    return build_proposal_input(
        input_payload,
        expected_proposal_path=f"adaptive/proposal_submissions/round_001--{request_id[7:19]}.json",
    )


def _proposal(snapshot: dict, *, parameters: dict | None = None) -> dict:
    return {
        "schema_version": 1,
        "request_id": snapshot["request_id"],
        "target_round": 1,
        "parameters": parameters
        or {
            "runtime.lr": [0.0001],
            "runtime.batch_size": [8],
            "yaml:/model/head/name": ["linear"],
        },
        "evidence_run_ids": ["run-001"],
        "rationale": "The completed run supports this bounded follow-up.",
        "proposer": {"agent": "codex", "model": "gpt-5"},
    }


def test_parameter_envelopes_default_to_source_range_and_preserve_categories():
    envelopes = validate_parameter_envelopes(
        {
            "runtime.lr": [0.0001, 0.0003],
            "runtime.batch_size": [8, 16],
            "runtime.precision": ["16-mixed", "32"],
            "runtime.flag": [False, True],
        }
    )

    assert envelopes == {
        "runtime.lr": {"kind": "number", "min": 0.0001, "max": 0.0003},
        "runtime.batch_size": {"kind": "integer", "min": 8, "max": 16},
        "runtime.precision": {"kind": "categorical", "choices": ["16-mixed", "32"]},
        "runtime.flag": {"kind": "categorical", "choices": [False, True]},
    }


def test_parameter_envelopes_allow_explicit_numeric_expansion_and_narrowing():
    envelopes = validate_parameter_envelopes(
        {"runtime.lr": [0.0001, 0.0003], "runtime.batch_size": [8, 16]},
        {"runtime.lr": [0.00005, 0.001], "runtime.batch_size": [10, 12]},
    )

    assert envelopes["runtime.lr"] == {"kind": "number", "min": 0.00005, "max": 0.001}
    assert envelopes["runtime.batch_size"] == {"kind": "integer", "min": 10, "max": 12}


@pytest.mark.parametrize(
    ("parameters", "bounds", "message"),
    [
        ({"runtime.lr": [0.1]}, {"runtime.unknown": [0.0, 1.0]}, "unknown parameter"),
        ({"runtime.lr": [0.1]}, {"runtime.lr": [0.0]}, "two-element list"),
        ({"runtime.lr": [0.1]}, {"runtime.lr": [1.0, 0.0]}, "min <= max"),
        ({"runtime.batch_size": [8]}, {"runtime.batch_size": [False, 16]}, "must be an integer"),
        ({"runtime.lr": [0.1]}, {"runtime.lr": [0.0, math.inf]}, "finite number"),
        ({"runtime.name": ["linear"]}, {"runtime.name": [0, 1]}, "categorical parameter"),
        ({"runtime.mixed": [1, "one"]}, None, "mixed or composite"),
        ({"runtime.composite": [[1], [2]]}, None, "mixed or composite"),
    ],
)
def test_parameter_envelopes_reject_invalid_agent_boundaries(parameters, bounds, message):
    with pytest.raises(ValueError, match=message):
        validate_parameter_envelopes(parameters, bounds)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ('{"outer":{"value":1,"value":2}}', "duplicate field"),
        ('{"value":NaN}', "non-finite"),
        ('{"value":Infinity}', "non-finite"),
        ("[1, 2]", "JSON object"),
        ('{"value":', "Malformed"),
    ],
)
def test_load_strict_json_rejects_ambiguous_or_nonstandard_json(text, message):
    with pytest.raises(ValueError, match=message):
        load_strict_json(text, source="proposal")


def test_request_id_is_canonical_and_does_not_depend_on_submission_path():
    input_payload = _input_payload()
    reversed_input = dict(reversed(list(input_payload.items())))

    first = build_proposal_input(input_payload, expected_proposal_path="first.json")
    second = build_proposal_input(reversed_input, expected_proposal_path="second.json")

    assert first["request_id"] == second["request_id"]
    assert first["request_id"] == f"sha256:{canonical_sha256(first['input'])}"
    assert validate_proposal_input(first) == first


def test_request_id_binds_source_config_hash():
    original = _input_payload()
    modified = copy.deepcopy(original)
    modified["source_config_sha256"] = "d" * 64

    assert proposal_request_id(original) != proposal_request_id(modified)


@pytest.mark.parametrize("source_config_sha256", ["C" * 64, "c" * 63, "g" * 64, 1])
def test_validate_proposal_input_requires_lowercase_source_config_sha256(source_config_sha256):
    input_payload = _input_payload()
    input_payload["source_config_sha256"] = source_config_sha256

    with pytest.raises(ValueError, match="source_config_sha256.*lowercase 64-hex"):
        build_proposal_input(input_payload, expected_proposal_path="proposal.json")


def test_build_proposal_input_requires_source_config_sha256():
    input_payload = _input_payload()
    del input_payload["source_config_sha256"]

    with pytest.raises(ValueError, match="missing field.*source_config_sha256"):
        build_proposal_input(input_payload, expected_proposal_path="proposal.json")


def test_validate_proposal_input_rejects_v1_snapshot():
    snapshot = _snapshot()
    snapshot["schema_version"] = 1

    with pytest.raises(ValueError, match="schema_version must be 2"):
        validate_proposal_input(snapshot)


def test_v2_input_accepts_v1_proposal_submission():
    snapshot = _snapshot()
    proposal = _proposal(snapshot)

    assert snapshot["schema_version"] == 2
    assert proposal["schema_version"] == 1
    assert validate_proposal(proposal, snapshot)["request_id"] == snapshot["request_id"]


def test_validate_proposal_input_rejects_tampered_snapshot():
    snapshot = _snapshot()
    snapshot["input"]["digest_rows"][0]["val_auroc"] = 0.99

    with pytest.raises(ValueError, match="does not match"):
        validate_proposal_input(snapshot)


def test_validate_proposal_accepts_authorized_expansion_and_returns_product():
    snapshot = _snapshot(bounds={"runtime.lr": [0.00005, 0.001], "runtime.batch_size": [4, 64]})
    proposal = _proposal(
        snapshot,
        parameters={
            "runtime.lr": [0.00005, 0.0001],
            "runtime.batch_size": [8, 32],
            "yaml:/model/head/name": ["linear"],
        },
    )

    validated = validate_proposal(proposal, snapshot)

    assert validated["parameters"] == proposal["parameters"]
    assert validated["max_runs"] == 4
    assert validated["evidence_run_ids"] == ["run-001"]


def test_validate_proposal_rejects_value_outside_default_source_range():
    snapshot = _snapshot()
    proposal = _proposal(snapshot)
    proposal["parameters"]["runtime.lr"] = [0.00005]

    with pytest.raises(ValueError, match="must be within"):
        validate_proposal(proposal, snapshot)


@pytest.mark.parametrize("value", [True, 8.0])
def test_validate_proposal_preserves_integer_type(value):
    snapshot = _snapshot()
    proposal = _proposal(snapshot)
    proposal["parameters"]["runtime.batch_size"] = [value]

    with pytest.raises(ValueError, match="must be an integer"):
        validate_proposal(proposal, snapshot)


def test_validate_proposal_allows_integer_for_number_parameter():
    snapshot = _snapshot(bounds={"runtime.lr": [0, 1]})
    proposal = _proposal(snapshot)
    proposal["parameters"]["runtime.lr"] = [0, 0.5]

    assert validate_proposal(proposal, snapshot)["max_runs"] == 2


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda proposal: proposal["parameters"].pop("runtime.lr"), "exactly match"),
        (lambda proposal: proposal["parameters"].update({"runtime.epochs": [5]}), "exactly match"),
        (lambda proposal: proposal["parameters"].update({"runtime.lr": [0.0001, 0.0001]}), "duplicate"),
        (
            lambda proposal: proposal["parameters"].update({"yaml:/model/head/name": ["unknown"]}),
            "authorized categorical",
        ),
        (lambda proposal: proposal.update({"evidence_run_ids": ["run-999"]}), "unknown evidence"),
        (lambda proposal: proposal.update({"evidence_run_ids": ["run-001", "run-001"]}), "must be unique"),
        (lambda proposal: proposal.update({"rationale": "  "}), "non-empty string"),
        (lambda proposal: proposal.update({"unexpected": True}), "unknown field"),
        (
            lambda proposal: proposal["proposer"].update({"temperature": 0}),
            "unknown field",
        ),
        (lambda proposal: proposal.update({"proposer": None}), "must be a mapping"),
    ],
)
def test_validate_proposal_rejects_closed_schema_and_evidence_errors(mutation, message):
    snapshot = _snapshot()
    proposal = _proposal(snapshot)
    mutation(proposal)

    with pytest.raises(ValueError, match=message):
        validate_proposal(proposal, snapshot)


def test_validate_proposal_rejects_round_size_before_remaining_run_budget():
    snapshot = _snapshot(round_size=3, runs=8)
    proposal = _proposal(snapshot)
    proposal["parameters"] = {
        "runtime.lr": [0.0001, 0.0003],
        "runtime.batch_size": [8, 16],
        "yaml:/model/head/name": ["linear"],
    }

    with pytest.raises(ValueError, match="round_size 3"):
        validate_proposal(proposal, snapshot)


def test_validate_proposal_rejects_remaining_run_budget():
    snapshot = _snapshot(round_size=8, runs=3)
    proposal = _proposal(snapshot)
    proposal["parameters"] = {
        "runtime.lr": [0.0001, 0.0003],
        "runtime.batch_size": [8, 16],
        "yaml:/model/head/name": ["linear"],
    }

    with pytest.raises(ValueError, match="remaining run budget 3"):
        validate_proposal(proposal, snapshot)


def test_validate_proposal_binds_request_and_target_round():
    snapshot = _snapshot()
    proposal = _proposal(snapshot)
    wrong_request = copy.deepcopy(proposal)
    wrong_request["request_id"] = "sha256:" + "f" * 64
    wrong_round = copy.deepcopy(proposal)
    wrong_round["target_round"] = 2

    with pytest.raises(ValueError, match="request_id"):
        validate_proposal(wrong_request, snapshot)
    with pytest.raises(ValueError, match="target_round"):
        validate_proposal(wrong_round, snapshot)


def test_strict_json_round_trip_validates_submission():
    snapshot = _snapshot()
    proposal = load_strict_json(json.dumps(_proposal(snapshot)), source="proposal")

    assert validate_proposal(proposal, snapshot)["max_runs"] == 1


def _configuration_proposal(snapshot: dict, *, configurations: list | None = None) -> dict:
    proposal = _proposal(snapshot)
    del proposal["parameters"]
    proposal["configurations"] = configurations or [
        {"runtime.lr": 0.0001, "runtime.batch_size": 8, "yaml:/model/head/name": "linear"},
        {"runtime.lr": 0.0003, "runtime.batch_size": 16, "yaml:/model/head/name": "mlp"},
    ]
    return proposal


def test_validate_proposal_accepts_configuration_points_and_counts_points():
    snapshot = _snapshot()
    proposal = _configuration_proposal(snapshot)

    validated = validate_proposal(proposal, snapshot)

    assert validated["max_runs"] == 2  # two points, not the 2x2x2 product
    assert validated["configurations"] == proposal["configurations"]
    assert "parameters" not in validated


def test_validate_proposal_requires_exactly_one_of_parameters_or_configurations():
    snapshot = _snapshot()
    both = _proposal(snapshot)
    both["configurations"] = [{"runtime.lr": 0.0001, "runtime.batch_size": 8, "yaml:/model/head/name": "linear"}]
    neither = _proposal(snapshot)
    del neither["parameters"]

    with pytest.raises(ValueError, match="exactly one of parameters or configurations"):
        validate_proposal(both, snapshot)
    with pytest.raises(ValueError, match="exactly one of parameters or configurations"):
        validate_proposal(neither, snapshot)


def test_validate_proposal_rejects_configuration_key_mismatch_with_point_index():
    snapshot = _snapshot()
    missing_key = _configuration_proposal(
        snapshot,
        configurations=[{"runtime.lr": 0.0001, "runtime.batch_size": 8}],
    )
    extra_key = _configuration_proposal(
        snapshot,
        configurations=[
            {
                "runtime.lr": 0.0001,
                "runtime.batch_size": 8,
                "yaml:/model/head/name": "linear",
                "runtime.epochs": 3,
            }
        ],
    )

    with pytest.raises(ValueError, match=r"configurations\[0\].*missing: yaml:/model/head/name"):
        validate_proposal(missing_key, snapshot)
    with pytest.raises(ValueError, match=r"configurations\[0\].*unknown: runtime.epochs"):
        validate_proposal(extra_key, snapshot)


def test_validate_proposal_rejects_configuration_value_outside_envelope():
    snapshot = _snapshot()
    out_of_range = _configuration_proposal(
        snapshot,
        configurations=[{"runtime.lr": 0.5, "runtime.batch_size": 8, "yaml:/model/head/name": "linear"}],
    )
    bad_category = _configuration_proposal(
        snapshot,
        configurations=[{"runtime.lr": 0.0001, "runtime.batch_size": 8, "yaml:/model/head/name": "attention"}],
    )
    bool_as_int = _configuration_proposal(
        snapshot,
        configurations=[{"runtime.lr": 0.0001, "runtime.batch_size": True, "yaml:/model/head/name": "linear"}],
    )

    with pytest.raises(ValueError, match=r"configurations\[0\].runtime.lr must be within"):
        validate_proposal(out_of_range, snapshot)
    with pytest.raises(ValueError, match=r"configurations\[0\].yaml:/model/head/name is not one of"):
        validate_proposal(bad_category, snapshot)
    with pytest.raises(ValueError, match=r"configurations\[0\].runtime.batch_size must be an integer"):
        validate_proposal(bool_as_int, snapshot)


def test_validate_proposal_rejects_duplicate_configuration_points_type_sensitively():
    snapshot = _snapshot()
    exact_duplicate = _configuration_proposal(
        snapshot,
        configurations=[
            {"runtime.lr": 0.0001, "runtime.batch_size": 8, "yaml:/model/head/name": "linear"},
            {"runtime.lr": 0.0001, "runtime.batch_size": 8, "yaml:/model/head/name": "linear"},
        ],
    )
    partial_overlap = _configuration_proposal(
        snapshot,
        configurations=[
            {"runtime.lr": 0.0001, "runtime.batch_size": 8, "yaml:/model/head/name": "linear"},
            {"runtime.lr": 0.0001, "runtime.batch_size": 16, "yaml:/model/head/name": "linear"},
        ],
    )

    with pytest.raises(ValueError, match="duplicate configuration points"):
        validate_proposal(exact_duplicate, snapshot)
    assert validate_proposal(partial_overlap, snapshot)["max_runs"] == 2


def test_validate_proposal_configuration_count_budget_checks_round_size_first():
    snapshot = _snapshot(round_size=2, runs=2)
    three_points = _configuration_proposal(
        snapshot,
        configurations=[
            {"runtime.lr": 0.0001, "runtime.batch_size": 8, "yaml:/model/head/name": "linear"},
            {"runtime.lr": 0.0002, "runtime.batch_size": 8, "yaml:/model/head/name": "linear"},
            {"runtime.lr": 0.0003, "runtime.batch_size": 8, "yaml:/model/head/name": "linear"},
        ],
    )

    with pytest.raises(ValueError, match="configuration count 3 exceeds adaptive round_size 2"):
        validate_proposal(three_points, snapshot)

    snapshot_runs = _snapshot(round_size=6, runs=2)
    three_points_runs = _configuration_proposal(
        snapshot_runs,
        configurations=three_points["configurations"],
    )
    three_points_runs["request_id"] = snapshot_runs["request_id"]
    with pytest.raises(ValueError, match="configuration count 3 exceeds remaining run budget 2"):
        validate_proposal(three_points_runs, snapshot_runs)


def test_parameters_snapshot_accepts_configuration_submission_without_new_request():
    # Input snapshots do not encode the expansion mode, so a snapshot generated
    # before the point-list capability accepts a configurations submission.
    snapshot = _snapshot()
    parameters_validated = validate_proposal(_proposal(snapshot), snapshot)
    configurations_validated = validate_proposal(_configuration_proposal(snapshot), snapshot)

    assert parameters_validated["request_id"] == configurations_validated["request_id"]
