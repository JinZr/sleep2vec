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
