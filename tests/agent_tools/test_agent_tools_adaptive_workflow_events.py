from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_tools import adaptive_hparam
from tests.agent_tools import (
    adaptive_hparam_test_support as test_support,
    test_agent_tools_adaptive_history as history_tests,
)

_stub_execution_snapshot_preflight = test_support._stub_execution_snapshot_preflight
completed_history = history_tests.completed_history


@pytest.fixture
def history_with_foreign_events(tmp_path: Path, completed_history):
    workflow, _initial_run, _second_run, proposal, _launch_calls = completed_history
    events_path = tmp_path / "events.jsonl"
    events = [json.loads(line) for line in events_path.read_text().splitlines()]
    round_events = {
        event["event_type"]: event
        for event in events
        if event.get("round") == 1
        and event.get("event_type") in {"agent_proposal_accepted", "launch_round", "agent_proposal_execute_completed"}
    }
    for event in round_events.values():
        foreign = dict(event)
        for field in ("proposal_path", "suggestion", "round_dir"):
            if field in foreign:
                foreign[field] = str(tmp_path / "foreign-workflow" / Path(foreign[field]).relative_to(workflow))
        events.append(foreign)
    events_path.write_text("".join(json.dumps(event) + "\n" for event in events))
    return workflow, proposal, events_path, round_events


def test_proposal_history_scopes_same_round_events_to_its_workflow(history_with_foreign_events):
    workflow, proposal, _events_path, _round_events = history_with_foreign_events

    input_path = adaptive_hparam.adaptive_step(workflow)

    assert input_path is not None
    rows = json.loads(input_path.read_text())["input"]["digest_rows"]
    assert rows[-1]["proposal_path"] == str(proposal)
    assert rows[-1]["proposal_rationale"] == json.loads(proposal.read_text())["rationale"]


@pytest.mark.parametrize(
    ("event_type", "changed_path", "message"),
    [
        ("agent_proposal_accepted", False, "lacks one exact acceptance event"),
        ("agent_proposal_accepted", True, "lacks one exact acceptance event"),
        ("launch_round", False, "event history conflicts"),
        ("agent_proposal_execute_completed", False, "event history conflicts"),
        ("agent_proposal_execute_completed", True, "event history conflicts"),
    ],
)
def test_foreign_workflow_does_not_hide_local_duplicate_or_conflicting_events(
    tmp_path: Path, history_with_foreign_events, event_type: str, changed_path: bool, message: str
):
    workflow, _proposal, events_path, round_events = history_with_foreign_events
    conflicting = dict(round_events[event_type])
    if changed_path:
        conflicting["proposal_path"] = str(tmp_path / "foreign-workflow" / "conflicting-proposal.json")
    with events_path.open("a") as output:
        output.write(json.dumps(conflicting) + "\n")

    with pytest.raises(ValueError, match=message):
        adaptive_hparam.adaptive_step(workflow)
    assert not list((workflow / "adaptive" / "proposal_inputs").glob("round_002--*.json"))
