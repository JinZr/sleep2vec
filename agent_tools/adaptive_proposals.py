from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import math
import re
from typing import Any

_INPUT_FIELDS = {
    "source_round",
    "target_round",
    "objective",
    "remaining_budget",
    "digest_rows",
    "parameter_envelopes",
    "resolved_recipe_sha256",
    "source_config_sha256",
    "execution_identity",
}
_PROPOSAL_INPUT_FIELDS = {"schema_version", "request_id", "input", "expected_proposal_path"}
_PROPOSAL_FIELDS = {
    "schema_version",
    "request_id",
    "target_round",
    "parameters",
    "configurations",
    "evidence_run_ids",
    "rationale",
    "proposer",
}
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_REQUEST_ID_RE = re.compile(r"sha256:[0-9a-f]{64}")


def validate_parameter_envelopes(
    parameters: Mapping[str, Any], bounds: Mapping[str, Any] | None = None
) -> dict[str, dict[str, Any]]:
    if not isinstance(parameters, Mapping) or not parameters:
        raise ValueError("search.parameters must be a non-empty mapping for agent proposals.")
    if bounds is None:
        bounds = {}
    if not isinstance(bounds, Mapping):
        raise ValueError("adaptive.suggest.bounds must be a mapping.")
    invalid_bound_keys = [key for key in bounds if not isinstance(key, str) or key not in parameters]
    if invalid_bound_keys:
        names = ", ".join(sorted(repr(key) for key in invalid_bound_keys))
        raise ValueError(f"adaptive.suggest.bounds contains unknown parameter(s): {names}.")

    envelopes: dict[str, dict[str, Any]] = {}
    for key, values in parameters.items():
        if not isinstance(key, str) or not key:
            raise ValueError("Agent proposal parameter names must be non-empty strings.")
        kind = _parameter_kind(key, values)
        if kind == "categorical":
            if key in bounds:
                raise ValueError(f"adaptive.suggest.bounds cannot constrain categorical parameter {key}.")
            envelopes[key] = {"kind": kind, "choices": list(values)}
            continue

        interval = bounds.get(key, [min(values), max(values)])
        if not isinstance(interval, list) or len(interval) != 2:
            raise ValueError(f"adaptive.suggest.bounds.{key} must be a two-element list.")
        lower, upper = interval
        _validate_numeric_value(lower, kind, f"adaptive.suggest.bounds.{key}[0]")
        _validate_numeric_value(upper, kind, f"adaptive.suggest.bounds.{key}[1]")
        if lower > upper:
            raise ValueError(f"adaptive.suggest.bounds.{key} must satisfy min <= max.")
        envelopes[key] = {"kind": kind, "min": lower, "max": upper}
    return envelopes


def proposal_request_id(input_payload: Mapping[str, Any]) -> str:
    normalized = _validate_input_payload(input_payload)
    return f"sha256:{canonical_sha256(normalized)}"


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def build_proposal_input(input_payload: Mapping[str, Any], *, expected_proposal_path: str) -> dict[str, Any]:
    normalized = _validate_input_payload(input_payload)
    if not isinstance(expected_proposal_path, str) or not expected_proposal_path.strip():
        raise ValueError("expected_proposal_path must be a non-empty string.")
    return {
        "schema_version": 2,
        "request_id": proposal_request_id(normalized),
        "input": normalized,
        "expected_proposal_path": expected_proposal_path,
    }


def load_strict_json(text: str, *, source: str = "JSON") -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"{source} contains a non-finite number: {value}.")

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{source} contains a duplicate field: {key}.")
            result[key] = value
        return result

    try:
        payload = json.loads(text, object_pairs_hook=reject_duplicate_keys, parse_constant=reject_constant)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed {source}: {exc.msg}.") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{source} must contain a JSON object.")
    _validate_json_value(payload, source)
    return payload


def validate_proposal_input(document: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(document, Mapping):
        raise ValueError("Proposal input must be a mapping.")
    _validate_closed_fields(document, _PROPOSAL_INPUT_FIELDS, _PROPOSAL_INPUT_FIELDS, "Proposal input")
    if type(document["schema_version"]) is not int or document["schema_version"] != 2:
        raise ValueError("Proposal input schema_version must be 2.")
    request_id = document["request_id"]
    if not isinstance(request_id, str) or _REQUEST_ID_RE.fullmatch(request_id) is None:
        raise ValueError("Proposal input request_id must be a sha256:<64-hex> value.")
    expected_path = document["expected_proposal_path"]
    if not isinstance(expected_path, str) or not expected_path.strip():
        raise ValueError("Proposal input expected_proposal_path must be a non-empty string.")
    input_payload = _validate_input_payload(document["input"])
    expected_request_id = proposal_request_id(input_payload)
    if request_id != expected_request_id:
        raise ValueError("Proposal input request_id does not match its canonical input snapshot.")
    normalized = {
        "schema_version": 2,
        "request_id": request_id,
        "input": input_payload,
        "expected_proposal_path": expected_path,
    }
    _validate_json_value(normalized, "Proposal input")
    return normalized


def validate_proposal(proposal: Mapping[str, Any], proposal_input: Mapping[str, Any]) -> dict[str, Any]:
    snapshot = validate_proposal_input(proposal_input)
    if not isinstance(proposal, Mapping):
        raise ValueError("Proposal must be a mapping.")
    required = _PROPOSAL_FIELDS - {"proposer", "parameters", "configurations"}
    _validate_closed_fields(proposal, required, _PROPOSAL_FIELDS, "Proposal")
    if ("parameters" in proposal) == ("configurations" in proposal):
        raise ValueError("Proposal must contain exactly one of parameters or configurations.")
    _validate_json_value(proposal, "Proposal")
    if type(proposal["schema_version"]) is not int or proposal["schema_version"] != 1:
        raise ValueError("Proposal schema_version must be 1.")
    if proposal["request_id"] != snapshot["request_id"]:
        raise ValueError("Proposal request_id does not match the bound input snapshot.")
    input_payload = snapshot["input"]
    if type(proposal["target_round"]) is not int or proposal["target_round"] != input_payload["target_round"]:
        raise ValueError("Proposal target_round does not match the bound input snapshot.")

    envelopes = input_payload["parameter_envelopes"]
    if "parameters" in proposal:
        normalized_search = {"parameters": _validate_proposal_parameters(proposal["parameters"], envelopes)}
        max_runs = 1
        for values in normalized_search["parameters"].values():
            max_runs *= len(values)
        budget_noun = "Cartesian product"
    else:
        points = _validate_proposal_configurations(proposal["configurations"], envelopes)
        normalized_search = {"configurations": points}
        max_runs = len(points)
        budget_noun = "configuration count"

    budget = input_payload["remaining_budget"]
    if max_runs > budget["round_size"]:
        raise ValueError(f"Proposal {budget_noun} {max_runs} exceeds adaptive round_size {budget['round_size']}.")
    if max_runs > budget["runs"]:
        raise ValueError(f"Proposal {budget_noun} {max_runs} exceeds remaining run budget {budget['runs']}.")

    evidence_run_ids = proposal["evidence_run_ids"]
    if not isinstance(evidence_run_ids, list) or not evidence_run_ids:
        raise ValueError("Proposal evidence_run_ids must be a non-empty list.")
    if any(not isinstance(run_id, str) or not run_id.strip() for run_id in evidence_run_ids):
        raise ValueError("Proposal evidence_run_ids must contain non-empty strings.")
    if len(evidence_run_ids) != len(set(evidence_run_ids)):
        raise ValueError("Proposal evidence_run_ids must be unique.")
    available_run_ids = {row["run_id"] for row in input_payload["digest_rows"]}
    unknown_evidence = sorted(set(evidence_run_ids) - available_run_ids)
    if unknown_evidence:
        raise ValueError(f"Proposal references unknown evidence run(s): {', '.join(unknown_evidence)}.")

    rationale = proposal["rationale"]
    if not isinstance(rationale, str) or not rationale.strip():
        raise ValueError("Proposal rationale must be a non-empty string.")
    proposer = None
    if "proposer" in proposal:
        proposer = proposal["proposer"]
        if not isinstance(proposer, Mapping):
            raise ValueError("Proposal proposer must be a mapping.")
        _validate_closed_fields(proposer, {"agent", "model"}, {"agent", "model"}, "Proposal proposer")
        if any(not isinstance(proposer[field], str) or not proposer[field].strip() for field in ("agent", "model")):
            raise ValueError("Proposal proposer agent and model must be non-empty strings.")
        proposer = dict(proposer)

    return {
        "request_id": proposal["request_id"],
        "target_round": proposal["target_round"],
        **normalized_search,
        "evidence_run_ids": list(evidence_run_ids),
        "rationale": rationale.strip(),
        "proposer": proposer,
        "max_runs": max_runs,
    }


def _validate_proposal_parameters(parameters: Any, envelopes: Mapping[str, Any]) -> dict[str, list[Any]]:
    if not isinstance(parameters, Mapping):
        raise ValueError("Proposal parameters must be a mapping.")
    missing = sorted(set(envelopes) - set(parameters))
    unknown = sorted(set(parameters) - set(envelopes))
    if missing or unknown:
        detail = []
        if missing:
            detail.append(f"missing: {', '.join(missing)}")
        if unknown:
            detail.append(f"unknown: {', '.join(unknown)}")
        raise ValueError(f"Proposal parameter keys must exactly match the input snapshot ({'; '.join(detail)}).")

    normalized_parameters: dict[str, list[Any]] = {}
    for key, envelope in envelopes.items():
        values = parameters[key]
        if not isinstance(values, list) or not values:
            raise ValueError(f"Proposal parameter {key} must be a non-empty list.")
        for index, value in enumerate(values):
            _validate_proposal_value(f"Proposal parameters.{key}[{index}]", value, envelope)
        if _has_duplicates(values, envelope["kind"]):
            raise ValueError(f"Proposal parameter {key} contains duplicate values.")
        normalized_parameters[key] = list(values)
    return normalized_parameters


def _validate_proposal_configurations(configurations: Any, envelopes: Mapping[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(configurations, list) or not configurations:
        raise ValueError("Proposal configurations must be a non-empty list.")
    points: list[dict[str, Any]] = []
    for index, point in enumerate(configurations):
        if not isinstance(point, Mapping):
            raise ValueError(f"Proposal configurations[{index}] must be a mapping.")
        missing = sorted(set(envelopes) - set(point))
        unknown = sorted(set(point) - set(envelopes))
        if missing or unknown:
            detail = []
            if missing:
                detail.append(f"missing: {', '.join(missing)}")
            if unknown:
                detail.append(f"unknown: {', '.join(unknown)}")
            raise ValueError(
                f"Proposal configurations[{index}] keys must exactly match the input snapshot ({'; '.join(detail)})."
            )
        for key, envelope in envelopes.items():
            _validate_proposal_value(f"Proposal configurations[{index}].{key}", point[key], envelope)
        points.append(dict(point))
    if _has_duplicate_points(points, envelopes):
        raise ValueError("Proposal configurations contains duplicate configuration points.")
    return points


def _parameter_kind(key: str, values: Any) -> str:
    if not isinstance(values, list) or not values:
        raise ValueError(f"Search parameter {key} must have a non-empty list of values.")
    if all(isinstance(value, bool) for value in values):
        return "categorical"
    if all(type(value) is int for value in values):
        return "integer"
    if all(_is_number(value) for value in values):
        for value in values:
            if not _is_finite_number(value):
                raise ValueError(f"Search parameter {key} contains a non-finite number.")
        return "number"
    if all(isinstance(value, str) for value in values):
        return "categorical"
    raise ValueError(f"Search parameter {key} has unsupported mixed or composite values for agent proposals.")


def _validate_input_payload(input_payload: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(input_payload, Mapping):
        raise ValueError("Proposal input snapshot must be a mapping.")
    _validate_closed_fields(input_payload, _INPUT_FIELDS, _INPUT_FIELDS, "Proposal input snapshot")
    source_round = input_payload["source_round"]
    target_round = input_payload["target_round"]
    if type(source_round) is not int or source_round < 0:
        raise ValueError("Proposal input source_round must be a non-negative integer.")
    if type(target_round) is not int or target_round <= source_round:
        raise ValueError("Proposal input target_round must be greater than source_round.")

    objective = input_payload["objective"]
    if not isinstance(objective, Mapping):
        raise ValueError("Proposal input objective must be a mapping.")
    _validate_closed_fields(objective, {"metric", "mode"}, {"metric", "mode"}, "Proposal input objective")
    if not isinstance(objective["metric"], str) or not objective["metric"].strip():
        raise ValueError("Proposal input objective.metric must be a non-empty string.")
    if not isinstance(objective["mode"], str) or objective["mode"] not in {"min", "max"}:
        raise ValueError("Proposal input objective.mode must be min or max.")

    budget = input_payload["remaining_budget"]
    if not isinstance(budget, Mapping):
        raise ValueError("Proposal input remaining_budget must be a mapping.")
    _validate_closed_fields(
        budget, {"rounds", "runs", "round_size"}, {"rounds", "runs", "round_size"}, "Proposal input remaining_budget"
    )
    for field in ("rounds", "runs", "round_size"):
        if type(budget[field]) is not int or budget[field] <= 0:
            raise ValueError(f"Proposal input remaining_budget.{field} must be a positive integer.")

    digest_rows = input_payload["digest_rows"]
    if not isinstance(digest_rows, list) or not digest_rows:
        raise ValueError("Proposal input digest_rows must be a non-empty list.")
    run_ids: list[str] = []
    for index, row in enumerate(digest_rows):
        if not isinstance(row, dict):
            raise ValueError(f"Proposal input digest_rows[{index}] must be a mapping.")
        for field in ("run_id", "status"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"Proposal input digest_rows[{index}].{field} must be a non-empty string.")
        run_ids.append(row["run_id"])
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("Proposal input digest_rows must have unique run_id values.")

    _validate_envelope_document(input_payload["parameter_envelopes"])
    recipe_hash = input_payload["resolved_recipe_sha256"]
    if not isinstance(recipe_hash, str) or _SHA256_RE.fullmatch(recipe_hash) is None:
        raise ValueError("Proposal input resolved_recipe_sha256 must be a 64-hex SHA-256 value.")
    config_hash = input_payload["source_config_sha256"]
    if not isinstance(config_hash, str) or _SHA256_RE.fullmatch(config_hash) is None:
        raise ValueError("Proposal input source_config_sha256 must be a lowercase 64-hex SHA-256 value.")
    if not isinstance(input_payload["execution_identity"], Mapping) or not input_payload["execution_identity"]:
        raise ValueError("Proposal input execution_identity must be a non-empty mapping.")

    normalized = json.loads(_canonical_json(dict(input_payload)))
    return normalized


def _validate_envelope_document(envelopes: Any) -> None:
    if not isinstance(envelopes, Mapping) or not envelopes:
        raise ValueError("Proposal input parameter_envelopes must be a non-empty mapping.")
    for key, envelope in envelopes.items():
        if not isinstance(key, str) or not key:
            raise ValueError("Proposal input parameter envelope names must be non-empty strings.")
        if not isinstance(envelope, Mapping):
            raise ValueError(f"Proposal input parameter_envelopes.{key} must be a mapping.")
        kind = envelope.get("kind")
        if isinstance(kind, str) and kind in {"integer", "number"}:
            _validate_closed_fields(envelope, {"kind", "min", "max"}, {"kind", "min", "max"}, f"Envelope {key}")
            _validate_numeric_value(envelope["min"], kind, f"Envelope {key}.min")
            _validate_numeric_value(envelope["max"], kind, f"Envelope {key}.max")
            if envelope["min"] > envelope["max"]:
                raise ValueError(f"Envelope {key} must satisfy min <= max.")
        elif kind == "categorical":
            _validate_closed_fields(envelope, {"kind", "choices"}, {"kind", "choices"}, f"Envelope {key}")
            if _parameter_kind(key, envelope["choices"]) != "categorical":
                raise ValueError(f"Envelope {key}.choices must contain only booleans or only strings.")
        else:
            raise ValueError(f"Envelope {key}.kind must be integer, number, or categorical.")


def _validate_proposal_value(location: str, value: Any, envelope: Mapping[str, Any]) -> None:
    kind = envelope["kind"]
    if kind in {"integer", "number"}:
        _validate_numeric_value(value, kind, location)
        if value < envelope["min"] or value > envelope["max"]:
            raise ValueError(f"{location} must be within [{envelope['min']}, {envelope['max']}].")
        return
    if not any(type(value) is type(choice) and value == choice for choice in envelope["choices"]):
        raise ValueError(f"{location} is not one of the authorized categorical choices.")


def _validate_numeric_value(value: Any, kind: str, location: str) -> None:
    if kind == "integer":
        if type(value) is not int:
            raise ValueError(f"{location} must be an integer and must not be boolean.")
        return
    if not _is_finite_number(value):
        raise ValueError(f"{location} must be a finite number and must not be boolean.")


def _has_duplicates(values: list[Any], kind: str) -> bool:
    for index, value in enumerate(values):
        for other in values[:index]:
            if kind in {"integer", "number"} and value == other:
                return True
            if kind == "categorical" and type(value) is type(other) and value == other:
                return True
    return False


def _has_duplicate_points(points: list[Mapping[str, Any]], envelopes: Mapping[str, Any]) -> bool:
    def values_equal(a: Any, b: Any, kind: str) -> bool:
        if kind == "categorical":
            return type(a) is type(b) and a == b
        return a == b

    for index, point in enumerate(points):
        for other in points[:index]:
            if all(values_equal(point[key], other[key], envelopes[key]["kind"]) for key in envelopes):
                return True
    return False


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_finite_number(value: Any) -> bool:
    return _is_number(value) and (type(value) is int or math.isfinite(value))


def _validate_closed_fields(value: Mapping[str, Any], required: set[str], allowed: set[str], label: str) -> None:
    missing = sorted(required - set(value), key=str)
    unknown = sorted(set(value) - allowed, key=str)
    if missing:
        raise ValueError(f"{label} is missing field(s): {', '.join(missing)}.")
    if unknown:
        raise ValueError(f"{label} contains unknown field(s): {', '.join(str(field) for field in unknown)}.")


def _canonical_json(value: Any) -> str:
    _validate_json_value(value, "Canonical input")
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _validate_json_value(value: Any, location: str) -> None:
    if value is None or isinstance(value, (str, bool)) or type(value) is int:
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{location} contains a non-finite number.")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, f"{location}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{location} contains a non-string object key.")
            _validate_json_value(item, f"{location}.{key}")
        return
    raise ValueError(f"{location} contains a non-JSON value of type {type(value).__name__}.")
