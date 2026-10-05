"""Adaptive agent-proposal handshake: request input, submission binding, and replay.

Domain-free kernel step between ``adaptive_hparam`` and an external proposer.
Writes the immutable proposal input for the next round from canonical digest
rows and binds it with one ``agent_proposal_requested`` event; loads a
submission, validates it against ``adaptive_proposals`` and binds it to the
exact request, input bytes, recipe and source-config hashes it answers; and
replays an already accepted proposal on re-execution. It publishes no
suggestion, receipt, or round: ``adaptive_hparam`` owns that publication, and
``adaptive_state`` the workflow state read here. See the proposal handshake in
``doc/agent_contracts/hparam_workflow.md``.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from . import adaptive_evidence, adaptive_proposals, adaptive_state, experiment_io as exp_io, run_artifacts as artifacts
from .experiment_workspace import (
    AdaptiveProposalRequestBinding,
    AdaptiveProposalRequestedEvent,
    append_event as _write_experiment_event,
    experiment_root,
    file_sha256,
    managed_run_key,
    read_experiment_events,
    read_run_manifest,
    validate_managed_run_rows,
    validated_run_key,
)
from .manifests import read_rows
from .models import resolve_repo_path
from .recipes import strip_internal_recipe_keys


class AcceptedProposalPayload(adaptive_proposals.ValidatedProposal):
    schema_version: Literal[1]
    input_path: str
    input_sha256: str
    proposal_path: str
    proposal_sha256: str


def proposal_digest_rows(root: Path, workspace: Path) -> list[dict[str, Any]]:
    registry = read_rows(root / "adaptive" / "run_registry.tsv", require_managed_identity=True)
    events = read_experiment_events(workspace)
    rows: list[dict[str, Any]] = []
    for round_index in sorted(adaptive_state.committed_round_indexes(root)):
        round_dir = adaptive_state.round_path(root, round_index)
        plan = artifacts.read_hparam_plan(round_dir)
        adaptive_state.validate_round_registry(root, round_index, plan, registry)
        if not adaptive_state.round_is_terminal(round_dir, workspace):
            raise ValueError(f"Agent proposal history round {round_index:03d} is not terminal.")
        recipe = plan["recipe"]
        objective = adaptive_state.workflow_objective(root, recipe)
        round_rows = adaptive_evidence.digest_rows(
            round_dir, round_index, workspace, objective, read_run_manifest=read_run_manifest
        )
        if round_index:
            context = _proposal_round_context(root, workspace, round_index, events)
            for row in round_rows:
                row.update(context)
        rows.extend(round_rows)
    successful = [row for row in rows if row["status"] in {"completed", "finished"}]
    ranked = adaptive_evidence.rank_rows(successful, objective)
    for row in rows:
        row["is_incumbent"] = bool(ranked) and row is ranked[0]
    fieldnames = sorted({key for row in rows for key in row})
    normalized = [{key: "" if row.get(key) is None else str(row.get(key)) for key in fieldnames} for row in rows]
    for row in normalized:
        for field in ("checkpoint_test_results", "training_history"):
            if row.get(field):
                row[field] = json.loads(row[field])
    return normalized


def _proposal_round_context(
    root: Path, workspace: Path, round_index: int, events: list[dict[str, Any]]
) -> dict[str, str]:
    binding_directories = {
        "proposal_path": root / "adaptive" / "proposal_submissions",
        "suggestion": root / "adaptive" / "suggestions",
        "round_dir": root / "adaptive" / "rounds",
    }
    events = [
        event
        for event in events
        if any(
            isinstance(event.get(field), str) and Path(event[field]).parent == directory
            for field, directory in binding_directories.items()
        )
    ]
    accepted_event = adaptive_state.agent_proposal_accepted_event(events, round_index)
    proposal_file, input_path, expected_path, proposal, proposal_sha256 = _load_agent_proposal_binding(
        root, workspace, accepted_event["proposal_path"]
    )
    proposal_input, _ = _load_agent_proposal_input(workspace, input_path, expected_path)
    validated = adaptive_proposals.validate_proposal(proposal, proposal_input)
    if (
        proposal_file != expected_path
        or proposal_sha256 != accepted_event["proposal_sha256"]
        or validated["target_round"] != round_index
        or validated["request_id"] != accepted_event["request_id"]
    ):
        raise ValueError(f"Agent proposal history differs from accepted round {round_index:03d}.")
    adaptive_state.validate_agent_proposal_execute_events(
        events, accepted_event, adaptive_state.round_path(root, round_index)
    )
    return {
        "proposal_path": str(proposal_file),
        "proposal_sha256": proposal_sha256,
        "proposal_rationale": validated["rationale"],
    }


def source_config_sha256(recipe: dict[str, Any]) -> str:
    source_config = (recipe.get("inputs") or {}).get("config")
    config_path = resolve_repo_path(source_config)
    if config_path is None or not config_path.exists():
        raise ValueError(f"Cannot hash missing source config: {source_config}")
    return file_sha256(config_path)


def bound_source_config_bytes(recipe: dict[str, Any], expected_sha256: str) -> bytes:
    source_config = (recipe.get("inputs") or {}).get("config")
    config_path = resolve_repo_path(source_config)
    if config_path is None or not config_path.exists():
        raise ValueError(f"Cannot read missing source config: {source_config}")
    config_bytes = config_path.read_bytes()
    if hashlib.sha256(config_bytes).hexdigest() != expected_sha256:
        raise ValueError("Adaptive source config changed after the proposal input was created.")
    return config_bytes


def _agent_proposal_input_payload(
    root: Path,
    workflow: dict[str, Any],
    recipe: dict[str, Any],
    rows: list[dict[str, Any]],
) -> adaptive_proposals.ProposalInputSnapshot:
    source_round = adaptive_state.latest_round_index(root)
    target_round = adaptive_state.next_round_index(root)
    adaptive = adaptive_state.adaptive_settings(recipe)
    committed_rounds = adaptive_state.committed_round_indexes(root)
    expected_keys = {
        (str(round_index), validated_run_key(run))
        for round_index in committed_rounds
        for run in artifacts.read_hparam_plan(adaptive_state.round_path(root, round_index))["runs"]
    }
    digest_keys = [(str(row.get("round")), managed_run_key(row)) for row in rows]
    if len(digest_keys) != len(set(digest_keys)) or set(digest_keys) != expected_keys:
        raise ValueError("Agent proposal digest rows do not match the committed round plans.")
    registered = read_rows(root / "adaptive" / "run_registry.tsv", require_managed_identity=True)
    remaining_rounds = int(adaptive.get("max_rounds") or 1) - len(committed_rounds)
    remaining_runs = int(adaptive.get("max_runs_total") or 10**9) - len(registered)
    if remaining_rounds <= 0 or remaining_runs <= 0:
        raise ValueError("Adaptive budget is exhausted; no agent proposal can be requested.")
    suggest_value = adaptive.get("suggest")
    suggest = suggest_value if isinstance(suggest_value, dict) else {}
    parameters = (recipe.get("search") or {}).get("parameters") or {}
    return {
        "source_round": source_round,
        "target_round": target_round,
        "objective": adaptive_state.workflow_objective(root, recipe),
        "remaining_budget": {
            "rounds": remaining_rounds,
            "runs": remaining_runs,
            "round_size": int(adaptive.get("round_size") or 1),
        },
        "digest_rows": rows,
        "parameter_envelopes": adaptive_proposals.validate_parameter_envelopes(parameters, suggest.get("bounds")),
        "resolved_recipe_sha256": _proposal_recipe_sha256(recipe, workflow),
        "source_config_sha256": source_config_sha256(recipe),
        "execution_identity": copy.deepcopy(workflow["execution_identity"]),
    }


def write_agent_proposal_input(
    root: Path,
    workflow: dict[str, Any],
    recipe: dict[str, Any],
    digest: Path,
    rows: list[dict[str, Any]],
) -> Path:
    input_payload = _agent_proposal_input_payload(root, workflow, recipe, rows)
    source_round = input_payload["source_round"]
    target_round = input_payload["target_round"]
    request_id = adaptive_proposals.proposal_request_id(input_payload)
    id12 = request_id.removeprefix("sha256:")[:12]
    proposal_path = root / "adaptive" / "proposal_submissions" / f"round_{target_round:03d}--{id12}.json"
    document = adaptive_proposals.build_proposal_input(
        input_payload,
        expected_proposal_path=str(proposal_path),
    )
    input_path = root / "adaptive" / "proposal_inputs" / f"round_{target_round:03d}--{id12}.json"
    workspace = experiment_root(recipe)
    if workspace is None:
        raise ValueError("Adaptive workflow is not bound to an experiment workspace.")
    events_lock = workspace / "events.jsonl.lock"
    exp_io.validate_managed_output_paths(
        workspace,
        [input_path, proposal_path, workspace / "events.jsonl", events_lock],
    )
    text = json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
    input_sha256 = hashlib.sha256(text.encode()).hexdigest()
    events_lock.parent.mkdir(parents=True, exist_ok=True)
    with exp_io.blocking_file_lock(events_lock):
        if not input_path.exists():
            input_path.parent.mkdir(parents=True, exist_ok=True)
            input_path.write_text(text)
        elif input_path.read_text() != text:
            raise ValueError(f"Agent proposal input snapshot was modified: {input_path}")
        events = _proposal_request_events(workspace)
        expected = _proposal_request_event_fields(document, input_path, input_sha256, proposal_path)
        related = _related_proposal_request_events(events, expected)
        if related:
            _validate_proposal_request_event(
                workspace,
                document,
                input_path,
                input_sha256,
                proposal_path,
                events=events,
            )
        else:
            # Only a completely unbound snapshot is the recoverable crash gap between write and issuance.
            requested_event: AdaptiveProposalRequestedEvent = {
                "source_round": source_round,
                "target_round": target_round,
                "digest": str(digest),
                "request_id": request_id,
                "input_path": str(input_path),
                "input_sha256": input_sha256,
                "proposal_path": str(proposal_path),
            }
            _write_experiment_event(workspace, "agent_proposal_requested", requested_event)
    return input_path


def _proposal_request_events(workspace: Path) -> list[dict[str, Any]]:
    return [
        event for event in read_experiment_events(workspace) if event.get("event_type") == "agent_proposal_requested"
    ]


def _proposal_request_event_fields(
    proposal_input: adaptive_proposals.ProposalInputDocument,
    input_path: Path,
    input_sha256: str,
    proposal_path: Path,
) -> AdaptiveProposalRequestBinding:
    snapshot = proposal_input["input"]
    return {
        "source_round": snapshot["source_round"],
        "target_round": snapshot["target_round"],
        "request_id": proposal_input["request_id"],
        "input_path": str(input_path),
        "input_sha256": input_sha256,
        "proposal_path": str(proposal_path),
    }


def _related_proposal_request_events(
    events: list[dict[str, Any]], expected: AdaptiveProposalRequestBinding
) -> list[dict[str, Any]]:
    direct_binding_fields: tuple[Literal["request_id", "input_path", "proposal_path"], ...] = (
        "request_id",
        "input_path",
        "proposal_path",
    )
    path_fields: tuple[Literal["input_path", "proposal_path"], ...] = ("input_path", "proposal_path")
    expected_parents = {field: Path(expected[field]).parent for field in path_fields}
    related = []
    for event in events:
        if any(event.get(field) == expected[field] for field in direct_binding_fields):
            related.append(event)
            continue
        same_workflow = any(
            isinstance(event.get(field), str) and Path(event[field]).parent == expected_parents[field]
            for field in path_fields
        )
        if same_workflow and event.get("target_round") == expected["target_round"]:
            related.append(event)
    return related


def _validate_proposal_request_event(
    workspace: Path,
    proposal_input: adaptive_proposals.ProposalInputDocument,
    input_path: Path,
    input_sha256: str,
    proposal_path: Path,
    *,
    events: list[dict[str, Any]] | None = None,
) -> None:
    events_path = workspace / "events.jsonl"
    if not events_path.exists():
        raise ValueError("Agent proposal input was not issued by phase one.")
    expected = _proposal_request_event_fields(proposal_input, input_path, input_sha256, proposal_path)
    all_events = events if events is not None else _proposal_request_events(workspace)
    related = _related_proposal_request_events(all_events, expected)
    if len(related) != 1:
        raise ValueError("Agent proposal input has no unique phase-one issuance record.")
    event = related[0]
    if any(event.get(field) != value for field, value in expected.items()):
        raise ValueError("Agent proposal input differs from its phase-one issuance record.")


def _load_agent_proposal_input(
    workspace: Path,
    input_path: Path,
    proposal_path: Path,
    *,
    expected_sha256: str | None = None,
) -> tuple[adaptive_proposals.ProposalInputDocument, str]:
    input_bytes = input_path.read_bytes()
    input_sha256 = hashlib.sha256(input_bytes).hexdigest()
    if expected_sha256 is not None and input_sha256 != expected_sha256:
        raise ValueError("Agent proposal input snapshot changed during validation.")
    raw_input = adaptive_proposals.load_strict_json(input_bytes.decode(), source=str(input_path))
    proposal_input = adaptive_proposals.validate_proposal_input(raw_input)
    if Path(proposal_input["expected_proposal_path"]) != proposal_path:
        raise ValueError("Proposal input expected path does not match its request id and target round.")
    _validate_proposal_request_event(workspace, proposal_input, input_path, input_sha256, proposal_path)
    return proposal_input, input_sha256


def validated_agent_proposal_input(
    root: Path,
    workflow: dict[str, Any],
    recipe: dict[str, Any],
    workspace: Path,
    input_path: Path,
    proposal_path: Path,
    *,
    expected_sha256: str | None = None,
) -> tuple[adaptive_proposals.ProposalInputDocument, str]:
    proposal_input, input_sha256 = _load_agent_proposal_input(
        workspace,
        input_path,
        proposal_path,
        expected_sha256=expected_sha256,
    )
    snapshot = proposal_input["input"]
    recipe_hash = _proposal_recipe_sha256(recipe, workflow)
    if snapshot["resolved_recipe_sha256"] != recipe_hash:
        raise ValueError("Adaptive source recipe changed after the proposal input was created.")
    if snapshot["source_config_sha256"] != source_config_sha256(recipe):
        raise ValueError("Adaptive source config changed after the proposal input was created.")
    if snapshot["execution_identity"] != workflow["execution_identity"]:
        raise ValueError("Adaptive execution identity changed after the proposal input was created.")
    if snapshot["source_round"] != adaptive_state.latest_round_index(root) or snapshot[
        "target_round"
    ] != adaptive_state.next_round_index(root):
        raise ValueError("Agent proposal round binding is stale.")
    if not adaptive_state.round_is_terminal(adaptive_state.round_path(root, snapshot["source_round"]), workspace):
        raise ValueError("Agent proposal source round is no longer terminal.")
    authoritative = _agent_proposal_input_payload(root, workflow, recipe, proposal_digest_rows(root, workspace))
    if snapshot != authoritative:
        raise ValueError("Agent proposal input does not match the current authoritative snapshot.")
    return proposal_input, input_sha256


def _load_agent_proposal_binding(
    root: Path,
    workspace: Path,
    proposal_path: str | Path,
) -> tuple[Path, Path, Path, dict[str, Any], str]:
    raw_path = Path(proposal_path).expanduser()
    proposal_file = raw_path if raw_path.is_absolute() else (Path.cwd() / raw_path).absolute()
    exp_io.validate_managed_output_paths(workspace, [proposal_file])
    proposal_bytes = proposal_file.read_bytes()
    proposal_sha256 = hashlib.sha256(proposal_bytes).hexdigest()
    proposal = adaptive_proposals.load_strict_json(proposal_bytes.decode(), source=str(proposal_file))
    request_id = proposal.get("request_id")
    target_round = proposal.get("target_round")
    if (
        not isinstance(request_id, str)
        or not request_id.startswith("sha256:")
        or len(request_id) != 71
        or any(char not in "0123456789abcdef" for char in request_id[7:])
    ):
        raise ValueError("Proposal request_id must be a sha256:<64-hex> value.")
    if type(target_round) is not int or target_round < 1:
        raise ValueError("Proposal target_round must be a positive integer.")
    id12 = request_id.removeprefix("sha256:")[:12]
    input_path = root / "adaptive" / "proposal_inputs" / f"round_{target_round:03d}--{id12}.json"
    expected_proposal_path = root / "adaptive" / "proposal_submissions" / f"round_{target_round:03d}--{id12}.json"
    exp_io.validate_managed_output_paths(workspace, [proposal_file, input_path])
    return proposal_file, input_path, expected_proposal_path, proposal, proposal_sha256


def load_agent_proposal(
    root: Path,
    workflow: dict[str, Any],
    recipe: dict[str, Any],
    workspace: Path,
    proposal_path: str | Path,
) -> tuple[Path, Path, adaptive_proposals.ValidatedProposal, str, str]:
    proposal_file, input_path, expected_proposal_path, proposal, proposal_sha256 = _load_agent_proposal_binding(
        root, workspace, proposal_path
    )
    proposal_input, input_sha256 = validated_agent_proposal_input(
        root, workflow, recipe, workspace, input_path, expected_proposal_path
    )
    if proposal_file != expected_proposal_path:
        raise ValueError("Proposal path does not match the bound input snapshot.")
    validated = adaptive_proposals.validate_proposal(proposal, proposal_input)
    if adaptive_state.budget_exhausted(root, recipe, prospective_runs=validated["max_runs"]):
        raise ValueError("Agent proposal no longer fits the remaining adaptive budget.")
    return proposal_file, input_path, validated, proposal_sha256, input_sha256


def applied_agent_proposal(root: Path, workspace: Path, proposal_path: str | Path) -> Path | None:
    initial_plan = artifacts.read_hparam_plan(adaptive_state.round_path(root, 0))
    initial_recipe_value = initial_plan.get("recipe")
    initial_recipe = initial_recipe_value if isinstance(initial_recipe_value, dict) else {}
    if adaptive_state.suggest_strategy(initial_recipe) != "agent_proposal":
        return None

    adaptive_state.read_workflow(root)
    proposal_file, input_path, expected_proposal_path, proposal, proposal_sha256 = _load_agent_proposal_binding(
        root, workspace, proposal_path
    )
    proposal_input, input_sha256 = _load_agent_proposal_input(workspace, input_path, expected_proposal_path)
    if proposal_file != expected_proposal_path:
        raise ValueError("Proposal path does not match the bound input snapshot.")
    validated = adaptive_proposals.validate_proposal(proposal, proposal_input)
    target_round = validated["target_round"]
    committed_rounds = adaptive_state.committed_round_indexes(root)
    if target_round not in committed_rounds:
        return None

    # A successful replay is proven from frozen artifacts because its live round binding is stale by design.
    round_dir = adaptive_state.round_path(root, target_round)
    round_plan = artifacts.read_hparam_plan(round_dir)
    registry_path = root / "adaptive" / "run_registry.tsv"
    registry_rows = read_rows(registry_path, require_managed_identity=True)
    validate_managed_run_rows(registry_rows, source=str(registry_path), cardinality="one_per_run")
    adaptive_state.validate_round_registry(root, target_round, round_plan, registry_rows)
    unresolved, abandoned = adaptive_state.uncommitted_launch_attempts(root, workspace)
    if unresolved or abandoned:
        attempts = list(unresolved)
        attempts.extend((round_index, str(row["run_id"])) for round_index, row in abandoned)
        detail = ", ".join(f"round {round_index:03d} {run_id}" for round_index, run_id in attempts)
        raise RuntimeError(
            f"Uncommitted adaptive launch evidence remains for {detail}; exact committed replay is blocked."
        )
    accepted_path = root / "adaptive" / "proposals" / f"round_{target_round:03d}.json"
    suggestion_path = root / "adaptive" / "suggestions" / f"round_{target_round:03d}.yaml"
    accepted_payload: AcceptedProposalPayload = {
        "schema_version": 1,
        **validated,
        "input_path": str(input_path),
        "input_sha256": input_sha256,
        "proposal_path": str(proposal_file),
        "proposal_sha256": proposal_sha256,
    }
    accepted_text = json.dumps(accepted_payload, indent=2, sort_keys=True) + "\n"
    snapshots = exp_io.read_managed_files_at(
        workspace,
        [input_path, proposal_file, accepted_path, suggestion_path],
    )
    if (
        snapshots[str(input_path)]["sha256"] != input_sha256
        or snapshots[str(proposal_file)]["sha256"] != proposal_sha256
    ):
        raise ValueError("Agent proposal submission or input snapshot changed during replay validation.")
    if snapshots[str(accepted_path)]["text"] != accepted_text:
        raise ValueError(f"Existing adaptive projection differs from the accepted proposal: {accepted_path}")
    suggestion_sha256 = snapshots[str(suggestion_path)]["sha256"]
    accepted_event = {
        "round": target_round,
        "request_id": validated["request_id"],
        "proposal_path": str(proposal_file),
        "proposal_sha256": proposal_sha256,
        "suggestion": str(suggestion_path),
        "suggestion_sha256": suggestion_sha256,
    }
    events = read_experiment_events(workspace)
    adaptive_state.validate_agent_proposal_execute_events(events, accepted_event, round_dir)
    canonical_by_key = {managed_run_key(row): row for row in read_run_manifest(workspace)}
    for later_round in sorted(round_index for round_index in committed_rounds if round_index > target_round):
        later_dir = adaptive_state.round_path(root, later_round)
        later_plan = artifacts.read_hparam_plan(later_dir)
        adaptive_state.validate_round_registry(root, later_round, later_plan, registry_rows)
        later_event = adaptive_state.agent_proposal_accepted_event(events, later_round)
        adaptive_state.validate_agent_proposal_execute_events(events, later_event, later_dir)
        later_keys = {managed_run_key(row) for row in registry_rows if str(row.get("round") or "") == str(later_round)}
        if any(canonical_by_key[key].get("status") == "launch_failed" for key in later_keys):
            raise ValueError(f"Later committed agent proposal has canonical launch failures: round {later_round:03d}")
    return suggestion_path


def _proposal_recipe_sha256(recipe: dict[str, Any], workflow: dict[str, Any]) -> str:
    payload = strip_internal_recipe_keys(copy.deepcopy(recipe))
    execution = payload.get("execution") if isinstance(payload.get("execution"), dict) else None
    baseline_value = workflow.get("execution_identity")
    baseline = baseline_value if isinstance(baseline_value, dict) else {}
    if execution is not None and baseline.get("runtime_commit") not in (None, ""):
        execution["runtime_commit"] = baseline["runtime_commit"]
    return adaptive_proposals.canonical_sha256(payload)
